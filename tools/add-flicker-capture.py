#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Assemble a private multi-frame replay manifest from validated good/bad shots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from PIL import Image, ImageChops


def checked_file(base, item):
    path = (base / item["file"]).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"]:
        raise ValueError(f"digest mismatch: {path.name}")
    return path


def assemble(corpus, receipt):
    result = json.loads(corpus.read_text())
    case = json.loads(receipt.read_text())
    for sha in ("good_mesa", "bad_mesa"):
        if not re.fullmatch(r"[0-9a-f]{40}", case[sha]):
            raise ValueError(f"expected full SHA for {sha}")
    if case["good_mesa"] == case["bad_mesa"]:
        raise ValueError("expected different good and bad revisions")
    base = receipt.parent
    trace = checked_file(base, case["trace"])
    if trace.suffix != ".gfxr":
        raise ValueError("expected a Vulkan capture")
    selected = sorted(map(int, case["frames"]))
    if len(selected) < 2 or selected[0] < 1 or selected != list(range(selected[0], selected[-1]+1)):
        raise ValueError("select at least two consecutive replay frames")
    if any(old["sha256"] == case["trace"]["sha256"] for old in result["captures"]):
        raise ValueError("capture already enrolled")
    refs = {}
    reproduced = False
    for n in selected:
        frame = case["frames"][str(n)]
        good = checked_file(base, frame["good"])
        bad = checked_file(base, frame["bad"])
        local = frame["good"] if case["source_device"] == result["device"] else frame["target_good"]
        if local == frame.get("target_good") and case.get("target_device") != result["device"]:
            raise ValueError("target golden device mismatch")
        golden = checked_file(base, local)
        x0, y0, x1, y1 = frame["roi"]
        with Image.open(good) as a, Image.open(bad) as b, Image.open(golden) as ref:
            if (a.size != b.size or a.size != ref.size or
                    not (0 <= x0 < x1 <= a.width and 0 <= y0 < y1 <= a.height)):
                raise ValueError(f"invalid geometry for frame {n}")
            red, green, blue = ImageChops.difference(a.convert("RGB"), b.convert("RGB")).split()
            delta = ImageChops.lighter(ImageChops.lighter(red,green),blue)
            roi = delta.crop((x0, y0, x1, y1))
            reproduced |= sum(roi.histogram()[9:]) > roi.width * roi.height * 0.001
        refs[str(n)] = {**local, "file": str(golden), "roi": frame["roi"]}
    if not reproduced:
        raise ValueError("good and bad source replays do not distinguish the flicker in the selected region")
    for old in result["captures"]:
        old["file"] = str((corpus.parent / old["file"]).resolve())
        for golden in old["frames"].values():
            golden["file"] = str((corpus.parent / golden["file"]).resolve())
    pins = [entry["replayer"] for entry in result["captures"] if entry["file"].endswith(".gfxr")]
    if not pins or any(pin != pins[0] for pin in pins):
        raise ValueError("expected one pinned Vulkan replayer identity in target corpus")
    result["captures"].append({"file":str(trace), "sha256":case["trace"]["sha256"],
                               "replayer":pins[0], "frames":refs})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "receipt", "out"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    try:
        result = assemble(args.corpus, args.receipt)
        with os.fdopen(os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    except (KeyError, OSError, TypeError, ValueError) as e:
        sys.exit(f"Capture not enrolled: {e}")
