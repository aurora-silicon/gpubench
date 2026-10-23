#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""import-record.py <combined-record-dir> <dest-root> [--fairness DIR]

Copies the rt part of a record written by the suite runner (tools/run-macos.sh or run-linux.sh,
which run suite and rt plans into one record) into <dest-root>/<record-name>/:
  results.jsonl  only the "lane": "rt" lines
  log.txt        only the executor stderr sections of rt plan lines
  meta.json      the record's meta with "lane": "rt" and the source path in "notes"
  powermetrics.txt, if present
With --fairness, also copies rt-archive* from the window's fairness directory.
"""
import argparse
import json
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record")
    ap.add_argument("dest_root")
    ap.add_argument("--fairness")
    a = ap.parse_args()
    src = a.record.rstrip("/")
    dst = os.path.join(a.dest_root, os.path.basename(src))
    os.makedirs(dst, exist_ok=True)
    n = 0
    with open(os.path.join(src, "results.jsonl")) as fi, open(os.path.join(dst, "results.jsonl"), "w") as fo:
        for line in fi:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("lane") == "rt":
                fo.write(line if line.endswith("\n") else line + "\n")
                n += 1
    log = os.path.join(src, "log.txt")
    if os.path.exists(log):
        keep = False
        with open(log, errors="replace") as fi, open(os.path.join(dst, "log.txt"), "w") as fo:
            for line in fi:
                if line.startswith("### "):
                    keep = " rt " in line[:40]
                if keep:
                    fo.write(line)
    meta_p = os.path.join(src, "meta.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        meta["lane"] = "rt"
        meta["notes"] = (meta.get("notes") or "") + f" [rt lines imported from {src}]"
        json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)
    pm = os.path.join(src, "powermetrics.txt")
    if os.path.exists(pm):
        shutil.copy(pm, dst)
    if a.fairness:
        for fn in os.listdir(a.fairness):
            if fn.startswith("rt-archive"):
                shutil.copy(os.path.join(a.fairness, fn), dst)
    print(f"{dst}: {n} rt results")


if __name__ == "__main__":
    sys.exit(main())
