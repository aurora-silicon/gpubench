#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Build a reproducible fleet kit for the given boards.

The kit holds fleet_row.py and batch-gate.py from this checkout, each board's
replay manifest and goldens, the pinned captures, and the executors and
replayers the caller supplies. Every capture and replayer must match what the
manifests pin. The same inputs always give a byte-identical tarball.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

TOOLS = Path(__file__).resolve().parent
CORPUS = TOOLS / "batch-gate"
EXECUTORS = ("gpubench-gl", "gpubench-vk")
REPLAYERS = {".gfxr": "gfxrecon-replay", ".trace": "eglretrace"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def collect(boards, bin_dir, captures, libs):
    """Return {kit path: bytes, executable?}, refusing anything a manifest does not pin."""
    files = {}

    def add(path, data, executable=False):
        if path in files and files[path][0] != data:
            raise ValueError(f"{path}: two boards need different contents")
        files[path] = (data, executable)

    add("fleet/fleet_row.py", (TOOLS / "fleet_row.py").read_bytes(), True)
    add("fleet/batch-gate.py", (TOOLS / "batch-gate.py").read_bytes(), True)
    for name in EXECUTORS:
        add(f"fleet/bin/{name}", (bin_dir / name).read_bytes(), True)
    for board in boards:
        raw = (CORPUS / f"{board}.json").read_bytes()
        add(f"fleet/corpus/{board}.json", raw)
        for case in json.loads(raw)["captures"]:
            data = (captures / case["file"]).read_bytes()
            if sha256(data) != case["sha256"]:
                raise ValueError(f"{case['file']}: not the capture {board} pins")
            add(f"fleet/corpus/{case['file']}", data)
            tool = REPLAYERS[Path(case["file"]).suffix]
            binary = (bin_dir / tool).read_bytes()
            if sha256(binary) != case["replayer"]["sha256"]:
                raise ValueError(f"{tool}: not the replayer {board} pins; build one kit per replayer set")
            add(f"fleet/bin/{tool}", binary, True)
            for frame in case["frames"].values():
                golden = (CORPUS / frame["file"]).read_bytes()
                if sha256(golden) != frame["sha256"]:
                    raise ValueError(f"{frame['file']}: golden differs from {board}.json")
                add(f"fleet/corpus/{frame['file']}", golden)
    if libs:
        for lib in sorted(libs.iterdir()):
            add(f"fleet/replaylibs/{lib.name}", lib.read_bytes())
    sums = "".join(f"{sha256(data)}  ./{path.removeprefix('fleet/')}\n" for path, (data, _) in sorted(files.items()))
    add("fleet/SHA256SUMS", sums.encode())
    return files


def write(files, out):
    """Deterministic tar.gz: sorted names, fixed owner and times, no gzip timestamp."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        dirs = sorted({str(p) for path in files for p in Path(path).parents if str(p) != "."})
        for d in dirs:
            info = tarfile.TarInfo(d)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(info)
        for path, (data, executable) in sorted(files.items()):
            info = tarfile.TarInfo(path)
            info.size, info.mode = len(data), 0o755 if executable else 0o644
            tar.addfile(info, io.BytesIO(data))
    with open(out, "xb") as dest:  # Never replace a kit that may have been handed over.
        with gzip.GzipFile(filename="", mode="wb", fileobj=dest, mtime=0) as gz:
            gz.write(raw.getvalue())
    return sha256(Path(out).read_bytes())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boards", nargs="+", required=True)
    p.add_argument("--bin", type=Path, required=True,
                   help="directory with gpubench-gl, gpubench-vk, gfxrecon-replay and eglretrace")
    p.add_argument("--captures", type=Path, required=True, help="directory with the pinned capture files")
    p.add_argument("--libs", type=Path, help="extra libraries the replayers need, shipped as fleet/replaylibs")
    p.add_argument("--out", type=Path, required=True, help="new tarball; an existing file is never replaced")
    args = p.parse_args(argv)
    files = collect(args.boards, args.bin, args.captures, args.libs)
    print(f"{write(files, args.out)}  {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError) as exc:
        print(f"fleet-kit: {exc}", file=sys.stderr)
        sys.exit(1)
