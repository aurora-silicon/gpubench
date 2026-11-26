#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Compare two staged Mesa builds on one Apple GPU. See batch-gate.md."""
import argparse
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


class Flag(Exception):
    """Missing evidence, unstable measurement or invalid baseline."""


class Regression(Exception):
    """Candidate failed an oracle that passed on the baseline."""


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


def info(exe, api, env, sha, log):
    result = execute([exe, "--info"], env, log)
    result.check_returncode()
    data = json.loads(result.stdout)
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
        raise Flag(f"{arm} {api} {name}: no valid JSON (exit {result.returncode})") from exc
    with (out / "rows.jsonl").open("a") as f:
        print(json.dumps(dict(arm=arm, api=api, stage=stage, record=record)), file=f)
    if (record["schema"] != "gpubench-result/1" or record["test"] != name or
            record["depth"] != depth or record["mode"] != "queue" or
            any(record["params"].get(k) != v for k, v in (p.split("=", 1) for p in params))):
        raise Flag(f"{arm} {name}: unexpected benchmark identity")
    check = record.get("check", {})
    if check.get("valid") is False:
        issue = Regression if arm == "B" else Flag
        raise issue(f"{arm} {api} {name}: output oracle failed")
    if result.returncode or record["status"] != "ok":
        raise Flag(f"{arm} {api} {name}: {record['status']} (exit {result.returncode})")
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


def classify(blocks):
    samples = {arm: [x for b in blocks for x in b[arm]] for arm in "AB"}
    # Robust within-arm dispersion, expressed as a fraction of its median.
    def spread(values):
        center = median(values)
        return 1.4826 * median(abs(v / center - 1) for v in values)
    noise = max(spread(values) for values in samples.values())
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


def bad_fraction(luma):
    return sum(luma.histogram()[9:]) / (luma.width * luma.height)


def replay(manifest, device, envs, out, summary):
    # One invocation per capture/arm preserves history between selected frames.
    from PIL import Image, ImageChops
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
            tool_sha = digest(Path(binary))
            execute(cmd, envs[arm], out / "executors.log", timeout=60).check_returncode()
            for n, golden in frames.items():
                path = (manifest.parent / golden["file"]).resolve()
                if digest(path) != golden["sha256"]:
                    raise Flag(f"golden digest mismatch: {path.name}")
                suffix = "*.bmp" if trace.suffix == ".gfxr" else "*.png"
                matches = [p for p in dest.glob(suffix)
                           if re.search(rf"(?<!\d)0*{int(n)}$", p.stem)]
                if len(matches) != 1:
                    raise Flag(f"missing or ambiguous screenshot for {trace.name} {n}")
                with Image.open(matches[0]) as shot, Image.open(path) as ref:
                    mismatch = shot.size != ref.size
                    fraction = region_fraction = None
                    if not mismatch:
                        # Absolute channel differences avoid cancellation at equal luminance.
                        diff = ImageChops.difference(shot.convert("RGB"), ref.convert("RGB"))
                        luma = diff.convert("L")
                        fraction = bad_fraction(luma)
                        mismatch = fraction > 0.001
                        if "roi" in golden:
                            x0, y0, x1, y1 = golden["roi"]
                            if not (0 <= x0 < x1 <= shot.width and 0 <= y0 < y1 <= shot.height):
                                raise Flag(f"invalid region in {path.name}")
                            region_fraction = bad_fraction(luma.crop((x0, y0, x1, y1)))
                            mismatch |= region_fraction > 0.001
                summary.append(dict(trace=trace.name, arm=arm, frame=n, bad_fraction=fraction,
                                    region_bad_fraction=region_fraction, golden_sha256=golden["sha256"],
                                    capture_sha256=case["sha256"],
                                    replayer_sha256=tool_sha))
                if mismatch:
                    issue = Regression if arm == "B" else Flag
                    raise issue(f"{arm} {trace.name} frame/call {n} differs from golden")


def gate(args, summary):
    out = args.out.resolve()
    envs = {k: environment(pair[1]) for k, pair in (("A", args.base), ("B", args.candidate))}
    bins = {api: args.bin.resolve() / f"gpubench-{api}" for api in ("gl", "vk")}
    summary["executors_sha256"] = {api: digest(exe) for api, exe in bins.items()}
    identities = summary["identity"] = {}
    for arm, (sha, _) in (("A", args.base), ("B", args.candidate)):
        identities[arm] = {api: info(exe, api, envs[arm], sha, out / "executors.log")
                           for api, exe in bins.items()}
    for api, field in (("gl", "renderer"), ("vk", "device_name")):
        if identities["A"][api][field] != identities["B"][api][field]:
            raise Flag(f"{api}: selected device changed")
    device = identities["A"]["vk"]["device_name"]
    findings = summary["findings"]
    for row in ROWS:
        api, name, _, _ = row
        crc = {arm: measure(row, arm, bins[api], envs[arm], out, "verify") for arm in "AB"}
        summary["crcs"][f"{api}/{name}"] = crc
        if crc["A"] != crc["B"]:
            findings.append(("FLAG", f"{api}/{name}: CRC changed within oracle tolerance"))
    if args.traces:
        replay(args.traces, device, envs, out, summary["replays"])
    if not args.visual_only:
        for row in ROWS:
            api, name, _, _ = row
            blocks = []
            for order in ORDERS:
                block = {"A": [], "B": []}
                for arm in order:
                    block[arm].append(measure(row, arm, bins[api], envs[arm], out, "perf"))
                blocks.append(block)
            stats = classify(blocks)
            summary["perf"][f"{api}/{name}"] = stats
            if stats["verdict"] != "PASS":
                findings.append((stats["verdict"], f"{api}/{name}: timing"))
    return max((status for status, _ in findings), key=["PASS", "FLAG", "FAIL"].index, default="PASS")


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
                   crcs={}, perf={}, replays=[], findings=[],
                   scope="gpubench + captures" if args.traces else "gpubench only",
                   visual_only=args.visual_only)
    def deadline(signum, frame):
        raise Flag("batch hit its 15-minute runtime budget")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        verdict = gate(args, summary)
    except (Flag, Regression, OSError, ValueError, KeyError, TypeError, ImportError,
            subprocess.SubprocessError) as exc:
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
