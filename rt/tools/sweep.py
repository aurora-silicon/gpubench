#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Parameter sweeps for rt.scene: vary one parameter at a time from a base preset and find where
performance falls off.

  sweep.py plan [--base medium] [--only res,rpp] [--quick]   > sweep-plan.txt
      prints plan lines ("rt-sweep-<param> rt rt.scene ...") for tools/run-plan.sh or
      tools/run-record.sh; one chunk per parameter.
  sweep.py analyze results.jsonl [...] [--base medium] [--markdown]
      per parameter: FPS, frame time and per-pass GPU time at each value, the value at which FPS
      drops under 60 and 30, the steepest step, and which pass grew the most there.

The results of several runs (e.g. Linux and macOS) can be analysed separately; compare.py joins
them.
"""
import argparse
import json
import sys

# Parameter -> values swept (from the base preset). Extra fixed params per sweep go in FIXED.
SWEEPS = {
    "res": ["1280x720", "1920x1080", "2560x1440", "3840x2160"],
    "reflective": ["0", "0.25", "0.5", "0.75", "1"],
    "rpp": ["1", "2", "4", "8"],
    "bounces": ["1", "2", "3", "4", "6"],
    "shadows": ["0", "1"],
    "tmax": ["25", "100", "400", "1600"],
    "alpha": ["0", "1"],
    "mesh-tris": ["5000", "20000", "80000", "160000"],
    "instances": ["64", "256", "1024", "4096"],
    "water-tris": ["8192", "32768", "131072", "524288"],
    "dyn-patches": ["1", "4", "8", "16"],
    "blas": ["static", "refit", "rebuild"],
    "tlas": ["static", "refit", "rebuild"],
    "moving": ["0.1", "0.25", "0.5", "1"],
}
FIXED = {
    "dyn-patches": ["blas=refit"],
    "blas": ["dyn-patches=16"],
    "moving": ["tlas=rebuild"],
}
PASSES = ["gpu_update_us", "gpu_as_us", "gpu_raster_us", "gpu_rt_us", "gpu_composite_us"]


def cmd_plan(a):
    only = set(a.only.split(",")) if a.only else None
    win = "--warmup-ms 800 --runs 3 --run-ms 700" if a.quick else ""
    print(f"# rt.scene sweeps from preset {a.base}: one parameter at a time (sweep.py plan)")
    for k, vals in SWEEPS.items():
        if only and k not in only:
            continue
        chunk = "rt-sweep-" + k
        for v in vals:
            extra = " ".join("--param " + f for f in FIXED.get(k, []))
            print(f"{chunk:22s} rt rt.scene --tag sweep:{k} --param preset={a.base} {extra} --param {k}={v} {win}".rstrip())


def load(paths):
    out = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("schema") == "gpubench-result/1" and d.get("test") == "rt.scene":
                out.append(d)
    return out


def med(d, m):
    x = d.get("metrics", {}).get(m)
    return x["median"] if x and x.get("median") is not None else None


def numeric(v):
    try:
        if "x" in str(v):
            w, h = str(v).split("x")
            return int(w) * int(h)
        return float(v)
    except ValueError:
        return {"static": 0, "refit": 1, "rebuild": 2}.get(str(v), 0)


def series(results, base):
    """{param: {value: result}} from the results tagged "sweep:<param>" by `sweep.py plan`."""
    out = {}
    for r in results:
        tag = r.get("tag", "")
        if r.get("status") != "ok" or not tag.startswith("sweep:") or r["params"].get("preset") != base:
            continue
        k = tag[6:]
        out.setdefault(k, {})[str(r["params"].get(k))] = r
    return out


def cmd_analyze(a):
    res = load(a.results)
    ser = series(res, a.base)
    md = a.markdown
    for k in SWEEPS:
        pts = ser.get(k)
        if not pts or len(pts) < 2:
            continue
        vals = sorted(pts, key=numeric)
        rows = []
        for v in vals:
            r = pts[v]
            row = {"value": v, "fps": med(r, "fps"), "low1": med(r, "fps_1pct_low"), "frame_ms": med(r, "frame_ms"),
                   "gpu_ms": (med(r, "gpu_us") or 0) / 1e3, "mrays": med(r, "mrays_s"),
                   "rays_frame": r.get("info", {}).get("rays", {}).get("per_frame")}
            for p in PASSES:
                row[p] = (med(r, p) or 0) / 1e3
            rows.append(row)
        print(f"\n## {k} (base {a.base})\n" if md else f"\n== {k} (base {a.base})")
        hdr = ["value", "fps", "1% low", "frame ms", "gpu ms", "update", "as", "raster", "rt", "composite", "Mray/s", "rays/frame"]
        if md:
            print("| " + " | ".join(hdr) + " |")
            print("|" + "---|" * len(hdr))
        else:
            print("  ".join(f"{h:>10s}" for h in hdr))
        for row in rows:
            cells = [str(row["value"]), f"{row['fps'] or 0:.1f}", f"{row['low1'] or 0:.1f}", f"{row['frame_ms'] or 0:.2f}",
                     f"{row['gpu_ms']:.2f}"] + [f"{row[p]:.2f}" for p in PASSES] + \
                    [f"{row['mrays'] or 0:.1f}", str(row["rays_frame"] or "")]
            print(("| " + " | ".join(cells) + " |") if md else "  ".join(f"{c:>10s}" for c in cells))
        # where it falls off
        notes = []
        for thr in (60, 30):
            below = [r for r in rows if r["fps"] is not None and r["fps"] < thr]
            if below and rows[0]["fps"] and rows[0]["fps"] >= thr:
                notes.append(f"FPS < {thr} at {k}={below[0]['value']}")
        worst, wi = None, None
        for i in range(1, len(rows)):
            a0, a1 = rows[i - 1]["frame_ms"], rows[i]["frame_ms"]
            if a0 and a1:
                f = a1 / a0
                if worst is None or f > worst:
                    worst, wi = f, i
        if wi is not None:
            grow = {p: rows[wi][p] - rows[wi - 1][p] for p in PASSES}
            top = max(grow, key=grow.get)
            notes.append(f"steepest step {rows[wi-1]['value']} -> {rows[wi]['value']}: frame time x{worst:.2f}, "
                         f"mostly {top.replace('gpu_', '').replace('_us', '')} (+{grow[top]:.2f} ms)")
        for n in notes:
            print(("- " if md else "   ") + n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--base", default="medium")
    p.add_argument("--only")
    p.add_argument("--quick", action="store_true", help="shorter windows (0.8 s warm-up, 3 x 0.7 s)")
    p.set_defaults(fn=cmd_plan)
    p = sp.add_parser("analyze")
    p.add_argument("results", nargs="+")
    p.add_argument("--base", default="medium")
    p.add_argument("--markdown", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
