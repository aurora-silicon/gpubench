#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Dispatch immutable GPU batch jobs, collect verified rows, and publish one PR line."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
import tempfile
import time

from fleet_row import BOARDS, RESULT_CODES, digest, read

REPOS = {"mesa": ("aurora-silicon/mesa", "aurora-wip"), "kernel": ("aurora-silicon/linux", "main")}
CASES = {"gl/render.postfx", "vk/render.postfx", "vk/compute.fp16", "vk/render.bindless_ubo"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def request(args):
    def git(*cmd):
        return subprocess.check_output(["git", "-C", args.tree, *cmd], text=True).strip()
    def commit(ref):
        return git("rev-parse", "--verify", "--end-of-options", ref+"^{commit}")
    base, candidate = commit(args.base), commit(args.candidate)
    boards = args.boards
    require(len(set(boards)) == len(boards) >= 2 and "j700" in boards, "select J700 and another board")
    require(base != candidate, "empty batch")
    git("merge-base", "--is-ancestor", base, candidate)
    members = []
    for pair in args.member:
        num, sha = pair.split(":", 1)
        require(num.isdecimal() and int(num) > 0, "invalid PR number")
        head = commit(sha)
        git("merge-base", "--is-ancestor", head, candidate)
        members.append(dict(number=int(num), head=head))
    require(len({m["number"] for m in members}) == len(members), "duplicate PR")
    prefixes = git("rev-list", "--first-parent", "--reverse", base+".."+candidate).splitlines()
    require(prefixes and git("rev-parse",prefixes[0]+"^1") == base, "base outside first-parent train")
    if args.kind == "mesa":
        require(len(prefixes) == len(members), "expected one train merge per Mesa PR")
        for prefix, member in zip(prefixes, members):
            require(git("rev-parse",prefix+"^2") == member["head"], "PR order/head differs from train")
    req = dict(kind=args.kind, base=base, candidate=candidate, tree=git("rev-parse",candidate+"^{tree}"),
               boards=sorted(boards), members=members)
    if args.kind == "kernel":
        require(re.fullmatch("[a-f0-9]{40}", args.mesa or ""), "pin full Mesa SHA for kernel row")
        req["mesa"] = args.mesa
    payload = (json.dumps(req, sort_keys=True, indent=2)+"\n").encode()
    job = args.spool.resolve() / hashlib.sha256(payload).hexdigest()
    job.mkdir(parents=True, exist_ok=True)
    # link(2) makes the request visible only after its contents have been written.
    dest = job/"request.json"
    with tempfile.NamedTemporaryFile(dir=job) as tmp:
        tmp.write(payload)
        tmp.flush()
        try:
            os.link(tmp.name, dest)
        except FileExistsError:
            require(dest.read_bytes() == payload, "existing request differs")
    print(job, flush=True)
    return job


def summary(directory, meta, expected, visual):
    path = directory/"run/summary.json"
    require(digest(path) == meta["summary_sha256"], "row summary changed")
    s = read(path)
    require(s["requested_sha"] == expected and s["visual_only"] is visual, "tested arms differ")
    require(s["scope"] == "gpubench + captures", "missing capture scope")
    if s["verdict"] == "PASS":
        require(not s["findings"] and s["kernel"] == meta["release"], "inconsistent PASS")
        require(s["kernel_faults"] == [], "GPU faults present")
        require(set(s["crcs"]) == CASES and all(isinstance(c["A"], str) and
                re.fullmatch("[0-9a-fA-F]{8}",c["A"]) and c["A"] == c["B"] for c in s["crcs"].values()),
                "incomplete CRC result")
        shots = s["replays"]
        require(meta["capture_frames"] >= 1 and
                all(sum(r["arm"] == side for r in shots) == meta["capture_frames"] for side in "AB"),
                "incomplete capture evidence")
        require(all(0 <= r["bad_fraction"] <= 0.001 and
                    (r["region_bad_fraction"] is None or 0 <= r["region_bad_fraction"] <= 0.001)
                    for r in shots), "image differs")
        require((not s["perf"]) if visual else (set(s["perf"]) == CASES and
                all(t["verdict"] == "PASS" for t in s["perf"].values())), "missing clean timings")
    require(s["verdict"] in RESULT_CODES, "invalid measurement verdict")
    return s


def cross_boot(a, b, meta, calibration, restored=None):
    """Compare base boot a with candidate boot b.

    With a measured restored boot (the base build booted again after the
    candidate), the boots form an A, B, A crossover: the candidate is compared
    with the mean of both base boots, which cancels a steady drift, and the
    base-to-restored change is the floor. Otherwise a reviewed calibration of
    the same kernel and Mesa supplies the floor.
    """
    require(a["executors_sha256"] == b["executors_sha256"], "executor changed between boots")
    if a["verdict"] != "PASS":
        return "FLAG", "baseline inconclusive"
    if b["verdict"] != "PASS":
        # On this boot BOTH runner arms use the candidate kernel. Timing-only
        # findings are A/A noise. The stock boot already passed the same oracles.
        failures = [msg for _, msg in b["findings"] if re.match(r"^[AB] ", msg)]
        patterns = ("no valid JSON", "replayer exited", "missing or ambiguous screenshot",
                    "output oracle failed", "differs from golden", "new GPU fault in kernel log")
        if any(any(p in msg for p in patterns) or
               re.search(r"(?:--info exited -?[1-9][0-9]*|\(exit -?[1-9][0-9]*\))$", msg)
               for msg in failures):
            return "FAIL", "candidate workload or GPU fault"
        return "FLAG", "candidate inconclusive"
    require(a["identity"] == b["identity"], "Mesa identity changed between boots")
    if a["crcs"] != b["crcs"]:
        return "FLAG", "cross-boot CRC change"
    if a["visual_only"]:
        return "PASS", "visual only"
    centre = lambda run, case: median(run["perf"][case]["median_us"].values())
    if restored is not None:
        if restored["verdict"] != "PASS":
            return "FLAG", "restored boot inconclusive"
        require(restored["identity"] == a["identity"] and
                restored["executors_sha256"] == a["executors_sha256"], "Mesa identity changed between boots")
        if restored["crcs"] != a["crcs"]:
            return "FLAG", "cross-boot CRC change"
        reference = {case: (centre(a, case) + centre(restored, case)) / 2 for case in a["perf"]}
        floors = {case: abs(centre(restored, case) / centre(a, case) - 1) for case in a["perf"]}
        noise = {case: max(run["perf"][case]["noise"] for run in (a, b, restored)) for case in a["perf"]}
    else:
        require(calibration.is_file(), "missing cross-boot calibration")
        c = read(calibration)
        require(c["board"] == "apple," + meta["board"] and len(set(c["boot_ids"])) == 2 and
                c["input_verdicts"] == ["PASS", "PASS"] and re.fullmatch("[0-9a-f]{7,40}", c["mesa"]) and
                a["requested_sha"]["A"].startswith(c["mesa"]) and
                c["kernel"] == a["kernel"] and c["kernel_notes_sha256"] == meta["kernel_notes_sha256"] and
                set(c["cases"]) == CASES, "inapplicable calibration")
        reference = {case: centre(a, case) for case in a["perf"]}
        floors = {case: abs(c["cases"][case]["change"]) for case in a["perf"]}
        noise = {case: max(a["perf"][case]["noise"], b["perf"][case]["noise"]) for case in a["perf"]}
    for case in a["perf"]:
        change = centre(b, case) / reference[case] - 1
        require(math.isfinite(floors[case]) and math.isfinite(change), "nonfinite calibration or measurement")
        if floors[case] > 0.05 or change > max(0.05, 3*noise[case]):
            return "FLAG", f"{case} needs boot crossover"
    return "PASS", "three-boot crossover" if restored is not None else "two-boot comparison"


def board_result(job, req, board):
    root = job/board
    result = {}
    metadata = {}
    faults = {}
    mesa = req["kind"] == "mesa"
    health = read(root/"health.json")
    token = digest(job/"request.json")
    require(health["request_sha256"] == token and health["restored"] is True, "restore or health unverified")
    for arm in (["pair"] if mesa else ["base","candidate","restored"]):
        row = root/arm
        m = read(row/"row.json")
        metadata[arm] = m
        require(m["board"] == board and m["arm"] == arm and m["request_sha256"] == token,
                "mismatched hardware receipt")
        if not mesa:
            require(m["kernel_commit"] == req["candidate" if arm=="candidate" else "base"], "kernel SHA changed")
        h = health["boots"][m["boot_id"]]
        require(type(h["gpu_faults"]) is int and h["gpu_faults"] >= 0, "missing fault count")
        faults[arm] = h["gpu_faults"]
        for key in ("new_kernel_errors","failed_units"):
            require(type(h[key]) is int and h[key] == 0, f"{arm} health: {key}")
        if arm != "restored" or "summary_sha256" in m:
            expect = dict(A=req["base"], B=req["candidate"]) if mesa else dict(A=req["mesa"], B=req["mesa"])
            result[arm] = summary(row, m, expect, board == "j516s")
    if mesa:
        s = result["pair"]
        if faults["pair"] and s["verdict"] != "FAIL":
            return "FLAG", "GPU fault outside attributed candidate failure"
        why = f"{metadata['pair']['release']}, {s['elapsed_s']:.0f}s"
        if s["verdict"] == "PASS":
            why += f", {len(s['replays'])} images, {len(s['crcs'])} CRC pairs, clean health"
        else:
            why += ", " + ", ".join(m for _, m in s["findings"] if "timing" in m)
            why = why.rstrip(", ")
        return s["verdict"], why
    base, candidate, restored = (metadata[key] for key in ("base","candidate","restored"))
    require(len({m["boot_id"] for m in metadata.values()}) == 3 and
            base["build_id"] == restored["build_id"] != candidate["build_id"], "invalid one-shot/restore sequence")
    require(base["manifest_sha256"] == candidate["manifest_sha256"], "corpus changed across boots")
    require(base["mesa_libs_sha256"] == candidate["mesa_libs_sha256"] ==
            restored.get("mesa_libs_sha256", base["mesa_libs_sha256"]), "Mesa library bytes changed")
    if faults["base"] or faults["restored"]:
        return "FLAG", "GPU fault on stock boot"
    if faults["candidate"] and result["base"]["verdict"] == "PASS":
        return "FAIL", "GPU fault on candidate boot"
    return cross_boot(result["base"], result["candidate"], base, root/"calibration.json", result.get("restored"))


def publish(job, req, line):
    repo, branch = REPOS[req["kind"]]
    def gh(route, *args, body=None):
        raw = subprocess.check_output(["gh","api",route,*args], text=True,
                                      input=json.dumps(body) if body is not None else None)
        return json.loads(raw)
    last = req["members"][-1]["number"]
    # Re-check batch membership at publication time, including a rebase of the live tip.
    marker = "<!-- gpubench-fleet-gate -->"
    with (job.parent/f".{req['kind']}-{last}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ref = gh(f"repos/{repo}/git/ref/heads/{branch}")
        require(ref["object"]["sha"] == req["base"], "live branch moved; summary retained locally")
        for member in req["members"]:
            pr = gh(f"repos/{repo}/pulls/{member['number']}")
            require(pr["state"] == "open" and pr["head"]["sha"] == member["head"] and
                    pr["base"]["ref"] == branch, "PR moved; summary retained locally")
        login = gh("user")["login"]
        endpoint = f"repos/{repo}/issues/{last}/comments"
        pages = gh(endpoint, "--paginate", "--slurp")
        existing = [c for page in pages for c in page if c["user"]["login"] == login and marker in c["body"]]
        require(len(existing) <= 1, "multiple existing gate comments")
        body = line + " " + marker
        if not existing:
            gh(endpoint, "-X", "POST", "--input", "-", body={"body":body})
        elif existing[0]["body"] != body:
            gh(f"repos/{repo}/issues/comments/{existing[0]['id']}", "-X", "PATCH", "--input", "-", body={"body":body})


def finish(job, post, errors=None):
    req = read(job/"request.json")
    statuses = []
    for board in req["boards"]:
        try:
            if errors and board in errors:
                status, why = "FLAG", errors[board]
            else:
                status, why = board_result(job,req,board)
        except FileNotFoundError:
            status, why = "FLAG", "row pending"
        except (OSError, KeyError, TypeError, ValueError, ArithmeticError) as exc:
            print(f"{board}: {exc}", file=sys.stderr)
            status, why = "FLAG", "row evidence incomplete"
        statuses.append((board,status,why))
    verdict = max((x[1] for x in statuses), key=["PASS","FLAG","FAIL"].index)
    line = (f"GPU batch {verdict} · `{req['base'][:12]}` → `{req['candidate'][:12]}` "
            f"(tree `{req['tree'][:12]}`, PR head `{req['members'][-1]['head'][:12]}`) · "
            + " · ".join(f"{board.upper()} {status} ({why})" for board,status,why in statuses))
    print(line, flush=True)
    if post:
        publish(job,req,line)
    return RESULT_CODES[verdict]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="op", required=True)
    for name in ("dispatch","enqueue"):
        d = sub.add_parser(name)
        d.add_argument("--kind", required=True, choices=REPOS)
        d.add_argument("--tree", type=Path, required=True)
        d.add_argument("--base", required=True)
        d.add_argument("--candidate", required=True)
        d.add_argument("--member", action="append", required=True, metavar="PR:SHA")
        d.add_argument("--boards", nargs="+", choices=sorted(BOARDS), required=True)
        d.add_argument("--spool", type=Path, required=True)
        d.add_argument("--mesa", help="fixed Mesa SHA for kernel requests")
        if name == "dispatch":
            d.add_argument("--workers", required=True, type=Path, help="trusted board to command argv JSON")
            d.add_argument("--no-post", action="store_true")
            d.add_argument("--wait-seconds", type=int, default=900)
    f = sub.add_parser("finish")
    f.add_argument("job", type=Path)
    f.add_argument("--post", action="store_true")
    f.add_argument("--wait-seconds", type=int, default=0)
    args = p.parse_args()
    if args.op == "finish":
        deadline = time.monotonic() + args.wait_seconds
        boards = read(args.job/"request.json")["boards"]
        while time.monotonic() < deadline and not all((args.job/b/"health.json").exists() for b in boards):
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        return finish(args.job, args.post)
    job = request(args)
    if args.op == "enqueue":
        return 0
    workers = read(args.workers)
    for board in args.boards:
        cmd = workers[board]
        require(isinstance(cmd,list) and cmd and all(isinstance(x,str) for x in cmd), "invalid worker argv")
    require(args.wait_seconds > 0, "wait budget must be positive")
    # Submit each immutable job once. Workers survive coordinator timeout so a lead
    # can finish its one-shot restore. A later finish reads the same evidence.
    processes, errors = {}, {}
    with (job/"dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for board in args.boards:
            sent = job/(board+".submitted")
            if sent.exists():
                continue
            try:
                with (job/(board+".worker.log")).open("ab") as log:
                    proc = subprocess.Popen([*workers[board], str(job), board],
                                            stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                sent.touch()
                processes[board] = proc
            except OSError as exc:
                print(f"{board}: {exc}", file=sys.stderr)
                errors[board] = "worker unavailable"
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        for board, proc in list(processes.items()):
            code = proc.poll()
            if code is not None:
                del processes[board]
                if code and not (job/board/"health.json").exists():
                    errors[board] = "worker returned without evidence"
        if all(b in errors or (job/b/"health.json").exists() for b in args.boards):
            break
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    return finish(job, not args.no_post, errors)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError, TypeError, subprocess.SubprocessError) as ex:
        print(f"FLAG: {ex}", file=sys.stderr)
        sys.exit(2)
