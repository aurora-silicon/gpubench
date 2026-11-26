#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Collect a fleet request inside a machine lead's existing hardware slot."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys


BOARDS = {"j700", "j773g", "j714s", "j516s"}
RESULT_CODES = {"PASS": 0, "FLAG": 2, "FAIL": 1}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def kernel_build_id(notes):
    pos = 0
    while pos + 12 <= len(notes):
        namesz, descsz, kind = struct.unpack_from("=III", notes, pos)
        pos += 12
        name = notes[pos:pos + namesz]
        pos += (namesz + 3) & ~3
        desc = notes[pos:pos + descsz]
        pos += (descsz + 3) & ~3
        if kind == 3 and name == b"GNU\0" and len(desc) == descsz and desc:
            return desc.hex()
    raise ValueError("running kernel lacks GNU build ID")


def collect(args):
    req = read(args.request)
    if args.board not in req["boards"]:
        raise ValueError("board outside batch")
    board = Path("/proc/device-tree/compatible").read_bytes().split(b"\0")[0].decode()
    if board != "apple," + args.board:
        raise ValueError(f"requested {args.board}, found {board}")
    boot_file = Path("/proc/sys/kernel/random/boot_id")
    boot = boot_file.read_text().strip()
    notes = Path("/sys/kernel/notes").read_bytes()
    kid = kernel_build_id(notes)
    if req["kind"] == "kernel":
        if args.arm not in ("base", "candidate", "restored") or not args.kernel_identity:
            raise ValueError("kernel row needs arm and build manifest from the lead")
        # The lead obtains this ID from the built vmlinux before staging its one-shot.
        manifest = read(args.kernel_identity)
        want = req["candidate" if args.arm == "candidate" else "base"]
        if manifest["commit"] != want or manifest["build_id"].lower() != kid:
            raise ValueError("running kernel does not match the requested artifact")
        a = b = req["mesa"]
        pinned_kernel = want
    else:
        if args.arm != "pair":
            raise ValueError("Mesa measurements use --arm pair")
        a, b = req["base"], req["candidate"]
        pinned_kernel = None
    args.out.mkdir()
    meta = dict(board=args.board, arm=args.arm, boot_id=boot, release=platform.release(),
                kernel_commit=pinned_kernel, build_id=kid, kernel_notes_sha256=hashlib.sha256(notes).hexdigest(),
                request_sha256=digest(args.request))
    # The restored boot measures too when given a bench, so base, candidate and
    # restored form an A, B, A crossover of the base build.
    measured = args.arm != "restored" or args.bin is not None
    if measured:
        if not all((args.bin, args.traces, args.base_env, args.candidate_env or req["kind"] == "kernel")):
            raise ValueError("measurement needs benchmark binaries, corpus and staged environments")
        manifest = read(args.traces)
        if not manifest["captures"]:
            raise ValueError("empty replay corpus")
        if req["kind"] == "kernel":
            if not args.mesa_libs:
                raise ValueError("kernel measurements require pinned GL and Vulkan library paths")
            meta["mesa_libs_sha256"] = dict(zip(("gl","vk"), map(digest, args.mesa_libs)))
        meta["manifest_sha256"] = digest(args.traces)
        meta["capture_frames"] = sum(len(c["frames"]) for c in manifest["captures"])
        cmd = [sys.executable, Path(__file__).with_name("batch-gate.py"), "--out", args.out/"run",
               "--bin", args.bin, "--traces", args.traces, "--base", a, args.base_env,
               "--candidate", b, args.base_env if req["kind"] == "kernel" else args.candidate_env]
        if args.board == "j516s":
            cmd.append("--visual-only")
        rc = subprocess.run(cmd).returncode
        summary = args.out/"run/summary.json"
        if rc != RESULT_CODES[read(summary)["verdict"]]:
            raise ValueError("runner exit disagrees with evidence")
        meta["summary_sha256"] = digest(summary)
        if req["kind"] == "kernel" and list(meta["mesa_libs_sha256"].values()) != list(map(digest,args.mesa_libs)):
            raise ValueError("Mesa libraries changed inside row")
    if boot_file.read_text().strip() != boot:
        raise ValueError("boot changed inside row")
    (args.out/"row.json").write_text(json.dumps(meta, indent=2)+"\n")
    return rc if measured else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("request", type=Path)
    p.add_argument("--board", choices=sorted(BOARDS), required=True)
    p.add_argument("--arm", choices=["pair","base","candidate","restored"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mesa-libs", type=Path, nargs=2, metavar=("GL", "VK"))
    for option in ("--kernel-identity", "--bin", "--traces", "--base-env", "--candidate-env"):
        p.add_argument(option, type=Path)
    try:
        sys.exit(collect(p.parse_args()))
    except (OSError, KeyError, ValueError, TypeError) as e:
        print(f"FLAG: {e}", file=sys.stderr)
        sys.exit(2)
