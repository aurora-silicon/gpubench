#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Compare two staged Mesa builds on one Apple GPU. See batch-gate.md."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
from statistics import median
import subprocess
import sys
import time

# Each has a readback oracle and reports untimed wall time per submission.
ROWS = [
    ("gl", "render.postfx", 2, ["res=1080p", "msaa=1"]),
    ("vk", "render.postfx", 1, ["res=1080p", "msaa=4"]),
    ("vk", "compute.fp16", 4, ["mode=gemv"]),
    ("vk", "render.bindless_ubo", 1, ["mode=ubo_vec4", "draws=512"]),
]
ORDERS = ["ABBA", "BAAB"] * 2
LEVELS = ["PASS", "FLAG", "FAIL"]


class Flag(Exception):
    """Missing evidence, unstable measurement or invalid baseline."""


class BudgetExpired(Flag):
    """The total gate budget elapsed."""


class Regression(Exception):
    """Candidate failed an oracle that passed on the baseline."""


def arm_error(arm, message):
    return Regression(message) if arm == "B" else Flag(message)


# Kernel MESSAGE only (no hostname). Ignore firmware inventory at boot.
GPU_SOURCE = re.compile(r"\.gpu: |^(?:aurora|asahi)(?:-g1[3-8]\w*)?[: ]|^G1[3-8]\w*: ", re.I)
GPU_EVENT = re.compile(r"GPU work .*failed|ETIMEDOUT|lost device|device lost|client failed|"
                       r"timed? ?out|\bfault|pagefault|unhealthy|channel.?error|firmware (?:crash|panic)|"
                       r"recover(?:y|ed)|guilty|\bhang|restart storm|setup failed|service failed|"
                       r"failed to (?:flush|send)|retains outstanding|capacity was lost", re.I)


def gpu_fault(entry):
    msg = entry["MESSAGE"]
    if not isinstance(msg, str):
        return False
    if re.search(r"G17: (?:crash logs from the firmware data segment|no fw-data region from the boot loader)", msg):
        return False
    return bool(GPU_SOURCE.search(msg) and (GPU_EVENT.search(msg) or int(entry.get("PRIORITY", 6)) <= 4))


class KernelJournal:
    def __init__(self, evidence):
        self.evidence = evidence
        self.command = ["journalctl"]
        self.last_arm, self.last_step = "A", "preflight"
        try:
            self.last_cursor = self.cursor()
        except (Flag, OSError, subprocess.CalledProcessError):
            self.command = ["sudo", "-n", "journalctl"]
            self.last_cursor = self.cursor()

    def read(self, *args):
        return subprocess.check_output([*self.command, "-q", "-k", "-b", "--no-pager", *args],
                                       text=True, timeout=5)

    def cursor(self):
        data = self.read("-n", "1", "--show-cursor", "-o", "cat")
        cursors = [line.removeprefix("-- cursor: ") for line in data.splitlines()
                   if line.startswith("-- cursor: ")]
        if not cursors or not cursors[-1]:
            raise Flag("cannot read current kernel journal cursor")
        return cursors[-1]

    def drain(self):
        # Advance only past entries we actually inspected. An empty read leaves
        # the cursor untouched, including when journald delivers teardown events late.
        arm, step, before = self.last_arm, self.last_step, self.last_cursor
        pending = sys.exc_info()[1]
        try:
            lines = self.read("--after-cursor", before, "-o", "json")
            entries = [json.loads(line) for line in lines.splitlines()]
            faults = []
            for entry in entries:
                if gpu_fault(entry):
                    faults.append(dict(message=entry["MESSAGE"], cursor=entry["__CURSOR"],
                                       monotonic_us=entry.get("_SOURCE_MONOTONIC_TIMESTAMP",
                                                              entry["__MONOTONIC_TIMESTAMP"]),
                                       realtime_us=entry["__REALTIME_TIMESTAMP"]))
            if entries:
                self.last_cursor = entries[-1]["__CURSOR"]
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            self.evidence.append(dict(arm=arm, step=step, error=str(exc)))
            if not isinstance(pending, Regression):
                raise Flag(f"{arm} {step}: cannot read kernel log") from exc
            return
        if faults:
            self.evidence.append(dict(arm=arm, step=step, cursor=before, lines=faults))
            if arm == "B" or not isinstance(pending, Regression):
                raise arm_error(arm, f"{arm} {step}: new GPU fault in kernel log")

    @contextmanager
    def observe(self, arm, step):
        self.drain()  # Any inter-step events still belong to the previous GPU process.
        self.last_arm, self.last_step = arm, step
        try:
            yield
        finally:
            self.drain()


class GpuClients:
    """GPU time used by other DRM clients, read from /proc fdinfo between steps.

    One client is one open of a DRM device (drm-client-id). Every process that
    inherited the descriptor shows the same client, so it is counted once. The
    drm-engine-<name> keys hold the client's accumulated time on each engine.
    Processes of other users cannot be read and are not counted. A driver
    without engine keys shows no time, so the check cannot flag there.
    """

    # One percent of the GPU can move a timing case by about 1 %, a fifth of
    # the 5 % floor.
    LIMIT = 0.01
    clock = staticmethod(time.monotonic)

    def __init__(self, summary):
        self.findings, self.flagged = summary["findings"], False
        scan = time.monotonic()
        self.known, self.since = self.snapshot(), self.clock()
        self.record = summary["gpu_clients"] = dict(
            at_start=[dict(client=cid, **c) for cid, c in sorted(self.known.items())], intervals=[],
            checks=1, scan_s=round(time.monotonic() - scan, 3))

    @staticmethod
    def snapshot(root=Path("/proc")):
        own = os.getpid()
        clients = {}
        # In PID order, so a shared client is named after its lowest process: usually the opener.
        for proc in sorted((p for p in root.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
            if int(proc.name) == own:
                continue
            try:
                for fd in (proc / "fd").iterdir():
                    if not os.readlink(fd).startswith("/dev/dri/"):
                        continue
                    keys = dict(line.split(":", 1) for line in
                                (proc / "fdinfo" / fd.name).read_text().splitlines() if ":" in line)
                    cid = keys.get("drm-client-id", "").strip()
                    if not cid or cid in clients:
                        continue
                    busy = sum(int(value.split()[0]) for key, value in keys.items()
                               if re.fullmatch(r"drm-engine-[a-z0-9_]+", key) and value.split()[1:] == ["ns"])
                    clients[cid] = dict(comm=(proc / "comm").read_text().strip(), pid=int(proc.name), busy_ns=busy)
            except (OSError, ValueError):  # Exited, or owned by another user.
                continue
        return clients

    def check(self, arm, step):
        scan = time.monotonic()
        now, at = self.snapshot(), self.clock()
        self.record["checks"] += 1
        self.record["scan_s"] = round(self.record["scan_s"] + time.monotonic() - scan, 3)
        wall = at - self.since
        used = {cid: c["busy_ns"] - self.known.get(cid, {"busy_ns": 0})["busy_ns"] for cid, c in now.items()}
        used = {cid: ns for cid, ns in used.items() if ns > 0}
        started = sorted(set(now) - set(self.known))
        exited = sorted(set(self.known) - set(now))
        if used or started or exited:
            name = lambda cid, table: f"{table[cid]['comm']}[{table[cid]['pid']}]"
            self.record["intervals"].append(dict(
                arm=arm, step=step, wall_s=round(wall, 3),
                used_ms={name(cid, now): round(ns / 1e6, 3) for cid, ns in used.items()},
                started=[name(cid, now) for cid in started], exited=[name(cid, self.known) for cid in exited]))
            total = sum(used.values()) / 1e9
            # A gap shorter than a second is judged against one second, so a
            # compositor frame in a short gap between steps cannot flag.
            if total > self.LIMIT * max(wall, 1.0) and not self.flagged:  # Later ones stay in the record.
                self.flagged = True
                self.findings.append(("FLAG", f"other GPU clients used {total * 1e3:.1f} ms of GPU time "
                                              f"in {wall:.1f} s by the end of {arm} {step} "
                                              f"({', '.join(name(cid, now) for cid in used)})"))
        self.known, self.since = now, at


def digest(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def environment(path):
    env = dict(os.environ, **read_json(Path(path)))
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET"):
        env.pop(key, None)
    env.update(EGL_PLATFORM="surfaceless", WAFFLE_PLATFORM="surfaceless_egl")
    return env


def execute(cmd, env, log, *, timeout=40):
    with log.open("a") as stderr:
        print(cmd, file=stderr, flush=True)
        result = subprocess.run(cmd, env=env, stderr=stderr,
                                stdout=subprocess.PIPE, text=True, timeout=timeout)
        stderr.write(f"exit={result.returncode}\n{result.stdout}\n")
    return result


def info(exe, api, env, sha, log, arm="A"):
    result = execute([exe, "--info"], env, log)
    if result.returncode:
        raise arm_error(arm, f"{arm} {api} --info exited {result.returncode}")
    try:
        data = json.loads(result.stdout)
    except ValueError as exc:
        raise arm_error(arm, f"{arm} {api} --info produced no valid JSON") from exc
    device = data["renderer" if api == "gl" else "device_name"]
    version = data["version" if api == "gl" else "driver_info"]
    revision = re.search(r"\bgit-([a-fA-F0-9]{7,40})\b", version)
    if ("Apple" not in device or not revision or
            not re.fullmatch(r"[a-fA-F0-9]{7,40}", sha) or
            not (sha.startswith(revision[1]) or revision[1].startswith(sha))):
        raise Flag(f"{device}: cannot verify requested build {sha} in {version}")
    if api == "vk" and data.get("driver_name") != "Borealis":
        raise Flag(f"unexpected Vulkan driver: {data.get('driver_name')}")
    if api == "gl" and "zink" in device.lower():
        raise Flag(f"expected native GL: {device}")
    return data


def measure(row, arm, exe, env, out, stage):
    api, name, depth, params = row
    warmup, runs, window = (250, 5, 450) if stage == "perf" else (50, 1, 100)
    cmd = [exe, "--test", name, "--depths", str(depth), "--warmup-ms", str(warmup),
           "--runs", str(runs), "--run-ms", str(window), "--timestamps", "off"]
    for param in params:
        cmd.extend(["--param", param])
    if api == "gl" and stage == "verify":
        cmd.append("--verify")
    result = execute(cmd, env, out / "executors.log")
    try:
        record = json.loads(result.stdout)
    except ValueError as exc:
        raise arm_error(arm, f"{arm} {api} {name}: no valid JSON (exit {result.returncode})") from exc
    with (out / "rows.jsonl").open("a") as f:
        print(json.dumps(dict(arm=arm, api=api, stage=stage, record=record)), file=f)
    if result.returncode or record.get("status") != "ok":
        raise arm_error(arm, f"{arm} {api} {name}: {record.get('error') or record.get('status')} (exit {result.returncode})")
    if (record["schema"] != "gpubench-result/1" or record["test"] != name or
            record["depth"] != depth or record["mode"] != "queue" or
            any(record["params"].get(k) != v for k, v in (p.split("=", 1) for p in params))):
        raise Flag(f"{arm} {name}: unexpected benchmark identity")
    check = record.get("check", {})
    if check.get("valid") is False:
        issue = Regression if arm == "B" else Flag
        raise issue(f"{arm} {api} {name}: output oracle failed")
    if stage == "verify":
        if check.get("valid") is not True or not re.fullmatch("[0-9a-fA-F]{8}", check.get("crc", "")):
            raise Flag(f"{arm} {api} {name}: missing verified CRC")
        return check["crc"]
    metric = record["metrics"]["us_per_submit"]
    value = metric["median"]
    if ((record["warmup_ms"], record["runs"], record["run_ms"]) != (warmup, runs, window)
            or metric["n"] != runs or metric["unit"] != "us"
            or not math.isfinite(value) or value <= 0):
        raise Flag(f"{arm} {api} {name}: invalid timing")
    return value


def third_difference(order, block):
    """Process-to-process noise left in one block, relative to its median.

    The weights -1, 3, -3, 1 over the four processes in run order sum to zero
    for each arm and cancel any linear or quadratic trend, so neither the arm
    difference nor a drift both arms share (such as the GPU warming up) enters
    it. With independent noise sigma per process it has variance 20 sigma^2.
    """
    arms = {arm: iter(block[arm]) for arm in "AB"}
    x = [next(arms[arm]) for arm in order]
    return abs(-x[0] + 3 * x[1] - 3 * x[2] + x[3]) / (math.sqrt(20) * median(x))


def classify(blocks):
    samples = {arm: [x for b in blocks for x in b[arm]] for arm in "AB"}
    # Mean absolute deviation of a normal variable is sigma * sqrt(2 / pi).
    residuals = [third_difference(order, b) for order, b in zip(ORDERS, blocks, strict=True)]
    noise = sum(residuals) / len(residuals) / math.sqrt(2 / math.pi)
    changes = [median(b["B"]) / median(b["A"]) - 1 for b in blocks]
    limit = max(0.05, 3 * noise)
    if min(changes) > limit:
        verdict = "FAIL"
    elif median(changes) > 0.05 or noise > 0.05:
        verdict = "FLAG"
    else:
        verdict = "PASS"
    return dict(verdict=verdict, median_us={k: median(v) for k, v in samples.items()},
                block_deltas=changes, noise=noise, threshold=limit)


def channel_difference(actual, reference):
    from PIL import ImageChops
    red, green, blue = ImageChops.difference(actual.convert("RGB"), reference.convert("RGB")).split()
    return ImageChops.lighter(ImageChops.lighter(red, green), blue)


def bad_fraction(diff):
    return sum(diff.histogram()[9:]) / (diff.width * diff.height)


def replayer_identity(binary, env):
    result = subprocess.run([binary, "--version"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, check=True, timeout=5)
    version = " ".join(result.stdout.split())
    if not version:
        raise Flag(f"cannot identify {binary}")
    return dict(sha256=digest(Path(binary)), version=version)


def replay(manifest, device, envs, out, summary, observe):
    # One invocation per capture/arm preserves history between selected frames.
    from PIL import Image
    spec = read_json(manifest)
    if spec["device"] != device or not spec["captures"]:
        raise Flag("capture manifest lacks cases for this GPU")
    for index, case in enumerate(spec["captures"]):
        trace = (manifest.parent / case["file"]).resolve()
        if digest(trace) != case["sha256"]:
            raise Flag(f"capture digest mismatch: {trace.name}")
        frames = case["frames"]
        if not frames or not all(re.fullmatch("0|[1-9][0-9]*", n) for n in frames):
            raise Flag("expected numbered frames or GL calls")
        selected = ",".join(sorted(frames, key=int))
        for arm in "AB":
            dest = out / f"replay-{index}-{arm}"
            dest.mkdir()
            if trace.suffix == ".gfxr":
                cmd = ["gfxrecon-replay", "--gpu", "0", "--wsi", "headless", "--screenshots", selected,
                       "--screenshot-format", "bmp", "--screenshot-dir", dest, trace]
            elif trace.suffix == ".trace":
                cmd = ["eglretrace", "--headless", "-s", str(dest / "frame"), "-S", selected, trace]
            else:
                raise Flag(f"unsupported capture: {trace.name}")
            binary = shutil.which(cmd[0], path=envs[arm].get("PATH"))
            if not binary:
                raise Flag(f"missing replayer: {cmd[0]}")
            tool = replayer_identity(binary, envs[arm])
            if tool != case["replayer"]:
                raise Flag(f"{arm} {trace.name}: replayer version or binary differs from manifest")
            with observe(arm, trace.name):
                proc = execute(cmd, envs[arm], out / "executors.log", timeout=60)
            if proc.returncode:
                raise arm_error(arm, f"{arm} {trace.name}: replayer exited {proc.returncode}")
            for n, golden in frames.items():
                path = (manifest.parent / golden["file"]).resolve()
                if digest(path) != golden["sha256"]:
                    raise Flag(f"golden digest mismatch: {path.name}")
                suffix = "*.bmp" if trace.suffix == ".gfxr" else "*.png"
                matches = [p for p in dest.glob(suffix)
                           if re.search(rf"(?<!\d)0*{int(n)}$", p.stem)]
                if len(matches) != 1:
                    raise arm_error(arm, f"{arm}: missing or ambiguous screenshot for {trace.name} {n}")
                with Image.open(matches[0]) as shot, Image.open(path) as ref:
                    mismatch = shot.size != ref.size
                    fraction = region_fraction = None
                    if not mismatch:
                        diff = channel_difference(shot, ref)
                        fraction = bad_fraction(diff)
                        mismatch = fraction > 0.001
                        if "roi" in golden:
                            x0, y0, x1, y1 = golden["roi"]
                            if not (0 <= x0 < x1 <= shot.width and 0 <= y0 < y1 <= shot.height):
                                raise Flag(f"invalid region in {path.name}")
                            region_fraction = bad_fraction(diff.crop((x0, y0, x1, y1)))
                            mismatch |= region_fraction > 0.001
                summary.append(dict(trace=trace.name, arm=arm, frame=n, bad_fraction=fraction,
                                    region_bad_fraction=region_fraction, golden_sha256=golden["sha256"],
                                    capture_sha256=case["sha256"],
                                    replayer_sha256=tool["sha256"], replayer_version=tool["version"]))
                if mismatch:
                    issue = Regression if arm == "B" else Flag
                    raise issue(f"{arm} {trace.name} frame/call {n} differs from golden")


def gate(args, summary):
    out = args.out.resolve()
    envs = {k: environment(pair[1]) for k, pair in (("A", args.base), ("B", args.candidate))}
    bins = {api: args.bin.resolve() / f"gpubench-{api}" for api in ("gl", "vk")}
    summary["executors_sha256"] = {api: digest(exe) for api, exe in bins.items()}
    journal = KernelJournal(summary["kernel_faults"])
    clients = GpuClients(summary)

    @contextmanager
    def observe(arm, step):
        # Between steps only other workloads can hold the GPU open.
        clients.check(journal.last_arm, journal.last_step)
        with journal.observe(arm, step):
            yield

    try:
        verdict = compare(args, summary, out, envs, bins, observe)
    finally:
        journal.drain()  # Include teardown after the last arm and before the verdict.
        clients.check(journal.last_arm, journal.last_step)
    return max([verdict, *(status for status, _ in summary["findings"])], key=LEVELS.index)


def compare(args, summary, out, envs, bins, observe):
    def run(row, arm, stage):
        api, name, _, _ = row
        with observe(arm, f"{api}/{name}/{stage}"):
            return measure(row, arm, bins[api], envs[arm], out, stage)

    identities = summary["identity"] = {}
    for arm, (sha, _) in (("A", args.base), ("B", args.candidate)):
        identities[arm] = {}
        for api, exe in bins.items():
            with observe(arm, f"{api}/info"):
                identities[arm][api] = info(exe, api, envs[arm], sha, out / "executors.log", arm)
    for api, field in (("gl", "renderer"), ("vk", "device_name")):
        if identities["A"][api][field] != identities["B"][api][field]:
            raise Flag(f"{api}: selected device changed")
    device = identities["A"]["vk"]["device_name"]
    findings = summary["findings"]
    for row in ROWS:
        api, name, _, _ = row
        crc = {arm: run(row, arm, "verify") for arm in "AB"}
        summary["crcs"][f"{api}/{name}"] = crc
        if crc["A"] != crc["B"]:
            findings.append(("FLAG", f"{api}/{name}: CRC changed within oracle tolerance"))
    if args.traces:
        replay(args.traces, device, envs, out, summary["replays"], observe)
    if not args.visual_only:
        for row in ROWS:
            api, name, _, _ = row
            blocks = []
            for order in ORDERS:
                block = {"A": [], "B": []}
                for arm in order:
                    block[arm].append(run(row, arm, "perf"))
                blocks.append(block)
            stats = classify(blocks)
            summary["perf"][f"{api}/{name}"] = stats
            if stats["verdict"] != "PASS":
                findings.append((stats["verdict"], f"{api}/{name}: timing"))
    return max((status for status, _ in findings), key=LEVELS.index, default="PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("--base", "--candidate"):
        p.add_argument(option, nargs=2, metavar=("MESA_SHA", "ENV_JSON"), required=True)
    p.add_argument("--bin", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new evidence directory")
    p.add_argument("--traces", type=Path, help="generation-specific replay manifest")
    p.add_argument("--visual-only", action="store_true")
    args = p.parse_args()
    args.out.mkdir()
    start = time.monotonic()
    summary = dict(kernel=platform.release(), requested_sha={"A": args.base[0], "B": args.candidate[0]},
                   crcs={}, perf={}, replays=[], findings=[], kernel_faults=[],
                   scope="gpubench + captures" if args.traces else "gpubench only",
                   visual_only=args.visual_only)
    def deadline(signum, frame):
        raise BudgetExpired("batch hit its 15-minute runtime budget")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        verdict = gate(args, summary)
    except Exception as exc:  # An infrastructure error must never signal a regression to git bisect.
        verdict = "FAIL" if isinstance(exc, Regression) else "FLAG"
        summary["findings"].append((verdict, str(exc)))
        if any(level == "FAIL" for level, _ in summary["findings"]):
            verdict = "FAIL"
    finally:
        signal.alarm(0)
    summary.update(verdict=verdict, elapsed_s=round(time.monotonic() - start, 2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{verdict} ({summary['scope']}, {summary['elapsed_s']} s)", flush=True)
    for level, msg in summary["findings"]:
        print(f"{level} {msg}")
    return {"PASS": 0, "FAIL": 1, "FLAG": 2}[verdict]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as exc:
        # Includes creation of the evidence directory and the final summary write.
        print(f"FLAG: cannot store gate evidence: {exc}", file=sys.stderr)
        sys.exit(2)
