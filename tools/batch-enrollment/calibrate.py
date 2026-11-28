#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Compare two stock-boot enrollment records with identical builds and fixtures."""
import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def measure(first, second):
    metadata = []
    runs = []
    for path in (first, second):
        start = read(path / "before.json")
        end = read(path / "after.json")
        for key in ("boot", "kernel", "kernel_notes_sha256", "compatible"):
            if not start[key] or start[key] != end[key]:
                raise ValueError(f"{path}: {key} changed")
        metadata.append(start)
        run = read(path / "run/summary.json")
        if run["visual_only"] or run["requested_sha"]["A"] != run["requested_sha"]["B"]:
            raise ValueError("requires timed same-Mesa A/A rows")
        if run["scope"] != "gpubench + captures" or not run["replays"]:
            raise ValueError("missing replay evidence")
        runs.append(run)
    a, b = metadata
    if a["boot"] == b["boot"]:
        raise ValueError("same boot twice")
    for field in ("kernel", "kernel_notes_sha256", "compatible"):
        if a[field] != b[field]:
            raise ValueError(f"{field} differs across boots")
    for field in ("identity", "requested_sha", "executors_sha256", "crcs"):
        if runs[0][field] != runs[1][field]:
            raise ValueError(f"{field} differs across boots")
    def fingerprints(run):
        return [(r["arm"], r["frame"], r["capture_sha256"], r["golden_sha256"], r["replayer_sha256"])
                for r in run["replays"]]
    if fingerprints(runs[0]) != fingerprints(runs[1]):
        raise ValueError("capture, golden or replayer changed")
    perf = [r["perf"] for r in runs]
    if not perf[0] or perf[0].keys() != perf[1].keys():
        raise ValueError("missing timing case")
    stats = {}
    for case in perf[0]:
        centers = [median([p[case]["median_us"][side] for side in "AB"]) for p in perf]
        if not all(math.isfinite(c) and c > 0 for c in centers):
            raise ValueError(f"{case}: invalid timing")
        stats[case] = {"boot1_us": centers[0], "boot2_us": centers[1],
                       "change": centers[1]/centers[0]-1}
    return dict(board=a["compatible"], kernel=a["kernel"], kernel_notes_sha256=a["kernel_notes_sha256"],
                mesa=runs[0]["requested_sha"]["A"], boot_ids=[r["boot"] for r in metadata],
                input_verdicts=[r["verdict"] for r in runs], cases=stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("boot1", type=Path)
    parser.add_argument("boot2", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.boot1, args.boot2)
        print(json.dumps(result, indent=2, allow_nan=False))
        sys.exit(0 if result["input_verdicts"] == ["PASS", "PASS"] else 2)
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(f"FLAG: {err}", file=sys.stderr)
        sys.exit(2)
