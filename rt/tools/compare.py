#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""compare.py <linux.jsonl|record-dir> <macos.jsonl|record-dir> [--markdown] [--test rt.scene]

Joins two sets of rt results on (test, params, mode, depth) (records/SCHEMA.md) and prints, per
configuration, the primary metric on both sides with the ratio, and for rt.scene the per-pass GPU
times, so the pass where Linux falls behind most stands out.
"""
import argparse
import json
import os
import sys

PASSES = ["gpu_update_us", "gpu_as_us", "gpu_raster_us", "gpu_rt_us", "gpu_composite_us", "gpu_us"]


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.jsonl")
    out = {}
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("schema") != "gpubench-result/1" or d.get("lane") != "rt":
            continue
        key = (d["test"], json.dumps(d.get("params", {}), sort_keys=True), d.get("mode"), d.get("depth"))
        out[key] = d
    return out


def med(d, m):
    x = d.get("metrics", {}).get(m)
    return x["median"] if x and x.get("median") is not None else None


def label(key):
    test, params, mode, depth = key
    p = json.loads(params)
    short = {k: v for k, v in p.items() if k in ("preset", "kind", "tris", "rays", "build", "trace")}
    extra = ""
    return f"{test} {' '.join(f'{k}={v}' for k, v in short.items())} d{depth}{extra}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("linux")
    ap.add_argument("macos")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--test")
    a = ap.parse_args()
    L, M = load(a.linux), load(a.macos)
    keys = [k for k in L if k in M and (not a.test or k[0] == a.test)]
    keys.sort(key=lambda k: (k[0], k[1], k[3]))
    md = a.markdown
    # primary metric table
    hdr = ["config", "metric", "Linux", "macOS", "Linux/macOS"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) if md else "  ".join(hdr))
    for k in keys:
        l, m = L[k], M[k]
        if l.get("status") != "ok" or m.get("status") != "ok":
            print(f"| {label(k)} | - | {l.get('status')} | {m.get('status')} | |" if md else f"{label(k)}: {l.get('status')} / {m.get('status')}")
            continue
        prim = l.get("primary") or m.get("primary")
        lv, mv = med(l, prim), med(m, prim)
        ratio = (lv / mv) if lv and mv else None
        cells = [label(k), prim, f"{lv:.4g}" if lv else "-", f"{mv:.4g}" if mv else "-", f"{ratio:.2f}" if ratio else "-"]
        print(("| " + " | ".join(cells) + " |") if md else "  ".join(cells))
    # per-pass table for the scene
    sk = [k for k in keys if k[0] == "rt.scene" and L[k].get("status") == "ok" and M[k].get("status") == "ok"]
    if sk:
        print("\nPer-pass GPU time, ms (Linux / macOS = ratio)\n")
        hdr = ["config"] + [p.replace("gpu_", "").replace("_us", "") for p in PASSES] + ["worst pass"]
        print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) if md else "  ".join(hdr))
        for k in sk:
            l, m = L[k], M[k]
            cells, worst = [label(k)], (None, 0)
            for p in PASSES:
                lv, mv = med(l, p), med(m, p)
                if lv is None or mv is None:
                    cells.append("-")
                    continue
                cells.append(f"{lv/1e3:.2f} / {mv/1e3:.2f} = {lv/mv if mv else 0:.2f}")
                if p != "gpu_us" and lv - mv > worst[1]:
                    worst = (p.replace("gpu_", "").replace("_us", ""), lv - mv)
            cells.append(f"{worst[0]} (+{worst[1]/1e3:.2f} ms)" if worst[0] else "-")
            print(("| " + " | ".join(cells) + " |") if md else "  ".join(cells))
    # census cross-check: frame 0 rendered identically on both sides
    if sk:
        print("\nCensus (frame 0) cross-check: image CRC, rays per frame, reflective pixel fraction\n")
        hdr = ["config", "CRC Linux", "CRC macOS", "rays Linux", "rays macOS", "rays diff", "refl Linux", "refl macOS", "verdict"]
        print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) if md else "  ".join(hdr))
        for k in sk:
            li, mi = L[k].get("info", {}), M[k].get("info", {})
            lr, mr = li.get("rays", {}), mi.get("rays", {})
            lp, mp = lr.get("per_frame") or 0, mr.get("per_frame") or 0
            diff = (lp - mp) / mp if mp else 0.0
            lf, mf = lr.get("reflective_pixel_frac") or 0, mr.get("reflective_pixel_frac") or 0
            same_crc = li.get("image_crc32") == mi.get("image_crc32")
            ok = abs(diff) < 0.005 and abs(lf - mf) < 0.002
            verdict = "identical" if same_crc else ("same work" if ok else "DIFFERENT")
            cells = [label(k), str(li.get("image_crc32")), str(mi.get("image_crc32")), str(lp), str(mp), f"{100 * diff:+.2f}%",
                     f"{lf:.4f}", f"{mf:.4f}", verdict]
            print(("| " + " | ".join(cells) + " |") if md else "  ".join(cells))
    missing = [k for k in L if k not in M]
    if missing:
        print(f"\n{len(missing)} Linux configurations have no macOS counterpart", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
