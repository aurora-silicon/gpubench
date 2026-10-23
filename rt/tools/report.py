#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""report.py --linux <record|results.jsonl>... [--macos <record|results.jsonl>...] [--out FILE]

Markdown summary of rt records: the preset ladder (FPS, 1% low, per-pass GPU ms, rays/s) per OS
and the Linux/macOS ratio per pass, the AS build table, rays/s against queue depth, and the sweep
falloff points (sweep.py). Several files per side are merged (later results win).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # noqa: E402

PASSES = ["gpu_update_us", "gpu_as_us", "gpu_raster_us", "gpu_rt_us", "gpu_composite_us"]
PRESETS = ["raster", "light", "medium", "solarbay", "heavy", "extreme"]


def load(paths):
    out = {}
    for p in paths or []:
        if os.path.isdir(p):
            p = os.path.join(p, "results.jsonl")
        for line in open(p):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("schema") != "gpubench-result/1" or d.get("lane") != "rt":
                continue
            key = (d["test"], json.dumps(d.get("params", {}), sort_keys=True), d.get("mode"), d.get("depth"),
                   d.get("tag", ""))
            out[key] = d
    return out


def med(d, m, scale=1.0):
    x = (d or {}).get("metrics", {}).get(m)
    return x["median"] * scale if x and x.get("median") is not None else None


def f(v, fmt="{:.2f}"):
    return "-" if v is None else fmt.format(v)


def scene_rows(res):
    rows = {}
    for (test, params, mode, depth, tag), d in res.items():
        p = json.loads(params)
        if test != "rt.scene" or tag or d.get("status") != "ok":
            continue
        if p.get("trace") != "query":
            continue
        rows[(p["preset"], depth)] = d
    return rows


def table(hdr, rows):
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--linux", nargs="+", required=True)
    ap.add_argument("--macos", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    L, M = load(a.linux), load(a.macos)
    out = []
    sides = [("Linux", L)] + ([("macOS", M)] if M else [])
    out.append("## rt.scene presets (default depth 2 unless noted)\n")
    for name, res in sides:
        rows = scene_rows(res)
        tr = []
        for pr in PRESETS:
            for depth in sorted({dp for (p, dp) in rows if p == pr}):
                d = rows[(pr, depth)]
                info = d.get("info", {})
                tr.append([f"{pr} d{depth}", f(med(d, "fps"), "{:.1f}"), f(med(d, "fps_1pct_low"), "{:.1f}"),
                           f(med(d, "frame_ms")), f(med(d, "gpu_us", 1e-3))] +
                          [f(med(d, p, 1e-3)) for p in PASSES] +
                          [f(med(d, "mrays_s"), "{:.0f}"), str(info.get("rays", {}).get("per_frame", "-")),
                           f(med(d, "fps_timed"), "{:.1f}")])
        out.append(f"### {name}\n")
        out.append(table(["preset", "fps", "1% low", "frame ms", "gpu ms", "update", "as", "raster", "rt", "composite",
                          "Mray/s", "rays/frame", "fps (timed)"], tr))
        out.append("")
    if M:
        rl, rm = scene_rows(L), scene_rows(M)
        tr = []
        for k in sorted(set(rl) & set(rm), key=lambda k: (PRESETS.index(k[0]) if k[0] in PRESETS else 99, k[1])):
            l, m = rl[k], rm[k]
            cells = [f"{k[0]} d{k[1]}", f(med(l, "fps"), "{:.1f}") + " / " + f(med(m, "fps"), "{:.1f}")]
            worst, wv = None, 0
            for p in PASSES + ["gpu_us"]:
                lv, mv = med(l, p, 1e-3), med(m, p, 1e-3)
                cells.append(f"{lv / mv:.2f}" if lv and mv else "-")
                if p != "gpu_us" and lv is not None and mv is not None and lv - mv > wv:
                    worst, wv = p.replace("gpu_", "").replace("_us", ""), lv - mv
            cells.append(f"{worst} (+{wv:.2f} ms)" if worst else "-")
            tr.append(cells)
        out.append("### Linux / macOS (per-pass GPU time ratio; > 1 = Linux slower)\n")
        out.append(table(["preset", "fps L / M", "update", "as", "raster", "rt", "composite", "frame", "largest gap"], tr))
        out.append("")
    # AS build
    out.append("## rt.asbuild\n")
    for name, res in sides:
        tr = []
        for (test, params, mode, depth, tag), d in sorted(res.items(), key=lambda kv: (kv[0][1])):
            if test != "rt.asbuild" or d.get("status") != "ok":
                continue
            p = json.loads(params)
            info = d.get("info", {})
            prim = d.get("primary")
            tr.append([p["kind"] + (" compact" if p.get("compact") else ""), p["build"], str(p["tris"]),
                       f(med(d, "gpu_us", 1e-3), "{:.3f}"), f(med(d, prim), "{:.1f}") + " " + (prim or ""),
                       f(med(d, "us_per_submit", 1e-3), "{:.3f}"),
                       f(info.get("as_bytes", 0) / max(1, p["tris"]), "{:.0f}"),
                       str(info.get("compacted_bytes", "-"))])
        tr.sort(key=lambda r: (r[0], r[1], int(r[2])))
        out.append(f"### {name}\n")
        out.append(table(["kind", "build", "tris/inst", "gpu ms", "rate", "wall ms/submit", "AS B/prim", "compacted B"], tr))
        out.append("")
    # rays
    out.append("## rt.rays (Mray/s wall clock / GPU)\n")
    for name, res in sides:
        tr = []
        for (test, params, mode, depth, tag), d in res.items():
            if test != "rt.rays" or d.get("status") != "ok":
                continue
            p = json.loads(params)
            tr.append([p["kind"], str(depth), f(med(d, "mrays_s"), "{:.0f}"), f(med(d, "mrays_s_gpu"), "{:.0f}"),
                       f(med(d, "us_per_submit"), "{:.0f}"), f(med(d, "gpu_us"), "{:.0f}"), f(med(d, "gpu_gap_us"), "{:.1f}")])
        tr.sort(key=lambda r: (r[0], int(r[1])))
        out.append(f"### {name}\n")
        out.append(table(["kind", "depth", "Mray/s", "Mray/s (GPU)", "us/submit", "gpu us", "gap us"], tr))
        out.append("")
    # sweeps
    out.append("## Sweeps (one parameter at a time from medium)\n")
    for name, res in sides:
        results = [d for d in res.values() if d.get("test") == "rt.scene"]
        ser = sweep.series(results, "medium")
        if not ser:
            continue
        out.append(f"### {name}\n")
        tr = []
        for k in sweep.SWEEPS:
            pts = ser.get(k)
            if not pts:
                continue
            vals = sorted(pts, key=sweep.numeric)
            fps = [med(pts[v], "fps") for v in vals]
            tr.append([k, " / ".join(f"{v}: {f(x, '{:.0f}')}" for v, x in zip(vals, fps))])
        out.append(table(["parameter", "value: fps"], tr))
        out.append("")
    text = "\n".join(out)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
