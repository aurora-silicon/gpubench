#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Run from a lead-owned, quiet hardware slot. All paths except kit assets are caller-relative."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from PIL import Image, ImageChops, ImageFilter

kit = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("batch_gate", kit / "batch-gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def snapshot():
    src = Path("/sys")
    values = {}
    for pattern in ("class/power_supply/*/online", "class/thermal/thermal_zone*/temp",
                    "class/devfreq/*/cur_freq", "devices/system/cpu/cpufreq/policy*/scaling_cur_freq",
                    "module/asahi/srcversion", "module/aurora/srcversion"):
        for path in src.glob(pattern):
            try:
                values[str(path)] = path.read_text().strip()
            except OSError:
                pass
    return dict(boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                kernel=platform.release(), kernel_notes_sha256=gate.digest(src / "kernel/notes"),
                compatible=Path("/proc/device-tree/compatible").read_bytes().split(b"\0")[0].decode(),
                sensors=values)


def golden(shot, reference, gl):
    with Image.open(shot) as a, Image.open(reference) as b:
        actual, ref = a.convert("RGB"), b.convert("RGB")
    if actual.size != ref.size:
        raise ValueError(f"{shot}: dimensions differ")
    delta = gate.channel_difference(actual, ref)
    fraction = gate.bad_fraction(delta)
    if gl:
        bg = Image.new("RGB", ref.size, ref.getpixel((0, 0)))
        mask = ImageChops.difference(ref, bg).convert("L").point(lambda v: 255 if v > 1 else 0)
        edge = ImageChops.difference(mask.filter(ImageFilter.MaxFilter(3)), mask.filter(ImageFilter.MinFilter(3)))
        bad = delta.point(lambda v: 255 if v > 8 else 0)
        ok = ImageChops.subtract(bad, edge).getbbox() is None and fraction < 0.004
    else:
        ok = fraction <= 0.001
    print(f"{shot.name} independent reference changed fraction {fraction:.6f}", flush=True)
    if not ok:
        raise ValueError(f"{shot}: independent reference mismatch")
    return actual


def enroll(args, env, device):
    corpus = args.corpus.resolve()
    corpus.mkdir()  # Never replace previous goldens.
    dest = corpus / "goldens" / args.board
    dest.mkdir(parents=True)
    cases = []
    scratch = args.out / "shots"
    scratch.mkdir()
    for trace, frames in [("triangle.gfxr", {"1":"screenshot_frame_1.bmp","2":"screenshot_frame_2.bmp"}),
                          ("tri_glsl.trace", {"6935":"glframe0000006935.png"})]:
        file = corpus / trace
        shutil.copyfile(kit / trace, file)
        gl = trace.endswith(".trace")
        cmd = (["eglretrace", "--headless", "-s", str(scratch/"glframe"), "-S", "6935"] if gl else
               ["gfxrecon-replay", "--gpu", "0", "--wsi", "headless", "--screenshots", "1,2",
                "--screenshot-format", "bmp", "--screenshot-dir", scratch])
        gate.execute([*cmd, file], env, args.out / "enroll.log", timeout=60).check_returncode()
        refs = {}
        for n, name in frames.items():
            im = golden(scratch/name, kit/"ref"/(name if gl else "vulkan.png"), gl)
            path = dest / (Path(name).stem + ".png")
            im.save(path)
            refs[n] = dict(file=str(path.relative_to(corpus)), sha256=gate.digest(path))
        exe = shutil.which(cmd[0], path=env["PATH"])
        cases.append(dict(file=trace, sha256=gate.digest(file), frames=refs,
                          replayer=gate.replayer_identity(exe, env)))
    write(corpus / (args.board + ".json"), dict(device=device, captures=cases))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--board", required=True, choices=["j700","j714s","j773g","j516s","j316s","j413"])
    p.add_argument("--sha", required=True)
    p.add_argument("--env", type=Path, help="Mesa loader environment; default is installed Mesa")
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--enroll", action="store_true")
    p.add_argument("--visual-only", action="store_true")
    args = p.parse_args()
    if args.board == "j516s" and not args.visual_only:
        p.error("J516s enrollment currently requires --visual-only")
    args.out = args.out.resolve()
    args.out.mkdir()
    before = snapshot()
    if before["compatible"] != "apple," + args.board:
        raise ValueError(f"unexpected board: {before['compatible']}")
    write(args.out/"before.json", before)
    config = gate.read_json(args.env) if args.env else {}
    libs = config.get("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
    config["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [libs, str(kit/"replaylibs")]))
    config["PATH"] = str(kit/"bin") + os.pathsep + os.environ["PATH"]
    config["MESA_SHADER_CACHE_DIR"] = str(args.out/"cache")
    config.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "asahi")
    arm = args.out/"env.json"
    write(arm, config)
    env = gate.environment(arm)
    for api in ("gl","vk"):
        data = gate.info(kit/"bin"/("gpubench-"+api),api,env,args.sha,args.out/"identity.log")
        if api == "vk":
            device = data["device_name"]
    if args.enroll:
        enroll(args,env,device)
    cmd = [sys.executable,kit/"batch-gate.py","--bin",kit/"bin","--out",args.out/"run",
           "--base",args.sha,arm,"--candidate",args.sha,arm,
           "--traces",args.corpus/(args.board+".json")]
    if args.visual_only:
        cmd.append("--visual-only")
    rc = subprocess.run(cmd).returncode
    after = snapshot()
    write(args.out/"after.json", after)
    return rc if after["boot"] == before["boot"] else 2


if __name__ == "__main__":
    sys.exit(main())
