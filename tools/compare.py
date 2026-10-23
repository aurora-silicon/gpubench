#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""compare.py: Linux vs macOS (or any two records) as Markdown tables.

    tools/compare.py RECORD_A RECORD_B [--labels Linux,macOS] [--metric NAME]
                     [--top 15] [--out FILE]

A record is a directory with meta.json and results.jsonl (records/SCHEMA.md).
Results are joined on (lane, test, params, mode, depth). The ratio is always
"A relative to B, >1 means A is worse": A/B for lower-is-better metrics and
B/A for higher-is-better ones. With A = Linux and B = macOS, 1.30 means Linux
takes 30% longer (or delivers 1/1.30 of the throughput).
"""
import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

DETAIL_METRICS = ["us_per_submit", "latency_us", "submit_cpu_us", "gpu_us", "gpu_gap_us",
                  "us_per_submit_timed"]


def hint(test, metric, depth, mode):
    """Which part of the stack a gap in this metric points at."""
    if metric == "submit_cpu_us":
        return "submission: submit call / ioctl CPU cost"
    if metric == "record_cpu_us":
        return "userspace: command recording / encoding"
    if metric == "gpu_gap_us":
        return "scheduling: GPU idle between queued jobs (firmware/kernel dispatch)"
    if metric == "gpu_us":
        return "GPU execution of the job itself (clocks, codegen, firmware job setup)"
    if mode == "procs":
        return "scheduling across processes: per-context rings, firmware queue arbitration"
    if mode == "queues":
        return "scheduling across queues of one process: firmware queues, render pairs"
    if test == "submit.empty":
        return "submission + fence signal path with no GPU work"
    if test in ("compute.light", "render.light"):
        if depth == 1 or metric == "latency_us":
            return "round trip: submit + firmware kick + completion IRQ + fence wake-up"
        path = "render-pass / tiler" if test == "render.light" else "compute dispatch"
        return f"per-submission {path} overhead in kernel/firmware at depth"
    if test in ("compute.heavy", "render.heavy"):
        return "GPU throughput under load; at depth: keeping the GPU fed (scheduling gaps)"
    if test == "render.tvb":
        return "tiler memory: TVB overflow / partial renders / heap growth" + (" (cold)" if metric == "first_us" else "")
    if test == "sync.signal":
        return "semaphore paths: wait-before-signal, cross-queue dependency signaling"
    if test.startswith("mem."):
        return "memory management: GEM/VM bind, mmap, page faults, first GPU use"
    if test == "copy.bw":
        return "copy/blit path (driver meta kernels) and memory bandwidth"
    if test == "concurrent.cr":
        return "compute/render concurrency in firmware; render<->compute switch cost"
    if test == "query":
        return "query/timestamp implementation and readback"
    if test == "idle.wake":
        return "GPU power management: idle power-down and wake-up latency (firmware/kernel)"
    return ""


def load(rec):
    rec = Path(rec)
    meta = json.loads((rec / "meta.json").read_text()) if (rec / "meta.json").exists() else {}
    res = OrderedDict()
    for line in (rec / "results.jsonl").read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("schema") != "gpubench-result/1":
            continue
        res[key(r)] = r
    return meta, res


def key(r):
    params = tuple(sorted((r.get("params") or {}).items()))
    return (r.get("lane", "suite"), r["test"], params, r.get("mode", "queue"), int(r.get("depth", 1)))


def pstr(params):
    return ",".join(f"{k}={v}" for k, v in params) or "-"


def ratio(m, a, b):
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return a / b if m.get("better", "lower") == "lower" else b / a


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    av = abs(v)
    if av >= 1000:
        return f"{v:,.0f}"
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fratio(r, bold=1.25):
    if r is None:
        return "-"
    s = f"{r:.2f}"
    return f"**{s}**" if r >= bold or r <= 1 / bold else s


def describe(meta, label):
    if not meta:
        return f"- {label}: (no meta.json)"
    d = meta.get("driver") or {}
    bits = [meta.get("device", "?"), meta.get("os", "?"), meta.get("os_version", "")]
    if meta.get("os") == "linux":
        km = meta.get("kernel_module") or {}
        bits += [f"kernel {meta.get('kernel')}", f"asahi.ko {str(km.get('build_id') or '?')[:12]}",
                 f"Mesa {meta.get('mesa_sha')}"]
    else:
        bits += [f"build {meta.get('os_build')}"]
    bits += [d.get("device_name", ""), f"suite {meta.get('suite_sha')}", meta.get("date_utc", "")]
    return f"- {label}: " + ", ".join(b for b in bits if b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", default="Linux,macOS")
    ap.add_argument("--metric", default=None, help="compare this metric instead of each result's primary")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    la, lb = args.labels.split(",", 1)
    ma, ra = load(args.a)
    mb, rb = load(args.b)
    out = []
    w = out.append
    w(f"# {la} vs {lb}\n")
    w(describe(ma, la))
    w(describe(mb, lb))
    w("")
    w(f"Ratio = {la} relative to {lb}; >1 means {la} is slower (bold when off by 25% or more).\n")

    keys = [k for k in ra if k in rb]
    only_a = [k for k in ra if k not in rb]
    only_b = [k for k in rb if k not in ra]

    # ---- per-test summary: one row per (lane, test, params, mode), one column per depth
    groups = OrderedDict()
    for k in keys:
        groups.setdefault(k[:4], []).append(k)
    depths = sorted({k[4] for k in keys})
    w("## Summary (primary metric ratio per depth)\n")
    w("| test | params | mode | metric | " + " | ".join(f"d{d}" for d in depths) + " |")
    w("|---|---|---|---|" + "---|" * len(depths))
    gaps = []
    for g, ks in groups.items():
        lane, test, params, mode = g
        cells = {}
        metric = None
        for k in ks:
            A, B = ra[k], rb[k]
            if A.get("status") != "ok" or B.get("status") != "ok":
                cells[k[4]] = f"{A.get('status')}/{B.get('status')}"
                continue
            mname = args.metric or A.get("primary") or "us_per_submit"
            metric = mname
            m_a = (A.get("metrics") or {}).get(mname)
            m_b = (B.get("metrics") or {}).get(mname)
            if not m_a or not m_b:
                cells[k[4]] = "-"
                continue
            r = ratio(m_a, m_a.get("median"), m_b.get("median"))
            cells[k[4]] = fratio(r)
            if r is not None:
                gaps.append((r, k, mname, m_a.get("median"), m_b.get("median"), m_a.get("unit", "")))
        name = test if lane == "suite" else f"{lane}:{test}"
        w(f"| {name} | {pstr(params)} | {mode} | {metric or '-'} | "
          + " | ".join(cells.get(d, "") for d in depths) + " |")
    w("")

    # ---- biggest gaps
    gaps.sort(key=lambda x: -abs(math.log(x[0])))
    w(f"## Biggest gaps (top {args.top})\n")
    w(f"| test | params | mode | depth | metric | {la} | {lb} | ratio | points at |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r, k, mname, va, vb, unit in gaps[: args.top]:
        lane, test, params, mode, depth = k
        w(f"| {test} | {pstr(params)} | {mode} | {depth} | {mname} ({unit}) | {fmt(va)} | {fmt(vb)} | {fratio(r)} "
          f"| {hint(test, mname, depth, mode)} |")
    w("")

    # ---- details: every metric of every joined result
    w("## Details (medians)\n")
    for g, ks in groups.items():
        lane, test, params, mode = g
        names = []
        for k in ks:
            for side in (ra[k], rb[k]):
                for n in (side.get("metrics") or {}):
                    if n not in names:
                        names.append(n)
        names.sort(key=lambda n: (DETAIL_METRICS.index(n) if n in DETAIL_METRICS else 99, n))
        w(f"### {test} {pstr(params)} ({mode})\n")
        w("| depth | metric | unit | " + f"{la} | {lb} | ratio |")
        w("|---|---|---|---|---|---|")
        for k in ks:
            A, B = ra[k], rb[k]
            if A.get("status") != "ok" or B.get("status") != "ok":
                why = A.get("reason") or A.get("error") or B.get("reason") or B.get("error") or ""
                w(f"| {k[4]} | status | | {A.get('status')} | {B.get('status')} | {why} |")
                continue
            for n in names:
                m_a = (A.get("metrics") or {}).get(n)
                m_b = (B.get("metrics") or {}).get(n)
                # Records before 2026-10-23 carry metal_sync (a string) inside metrics.
                m_a, m_b = (m if isinstance(m, dict) else None for m in (m_a, m_b))
                if not m_a and not m_b:
                    continue
                m = m_a or m_b
                va = m_a.get("median") if m_a else None
                vb = m_b.get("median") if m_b else None
                w(f"| {k[4]} | {n} | {m.get('unit', '')} | {fmt(va)} | {fmt(vb)} | {fratio(ratio(m, va, vb))} |")
        w("")
    if only_a or only_b:
        w("## Not joined\n")
        for k in only_a:
            w(f"- only {la}: {k[1]} {pstr(k[2])} {k[3]} d{k[4]} ({ra[k].get('status')})")
        for k in only_b:
            w(f"- only {lb}: {k[1]} {pstr(k[2])} {k[3]} d{k[4]} ({rb[k].get('status')})")
        w("")
    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
