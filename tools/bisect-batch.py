#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Run a batch row. On failure, isolate the first failing first-parent prefix."""
import argparse
import subprocess
import sys


def diagnose(commits, run):
    """run(base, candidate) returns a batch-gate exit code."""
    def probe(i):
        result = run(commits[0], commits[i])
        print(f"prefix {i} {commits[i]}: exit {result}", flush=True)
        return result

    last = len(commits) - 1
    rc = probe(last)
    if rc != 1:
        return (0, "PASS") if rc == 0 else (2, "FLAG: batch row inconclusive")
    if probe(last) != 1 or probe(0) != 0:
        return 2, "FLAG: could not confirm both endpoints"

    good, bad = 0, last
    while good + 1 < bad:
        mid = (good + bad) // 2
        rc = probe(mid)
        if rc == 0:
            good = mid
        elif rc == 1:
            bad = mid
        else:
            return 2, f"FLAG: inconclusive midpoint in {commits[good]}..{commits[bad]}"
    if probe(good) != 0 or probe(bad) != 1:
        return 2, "FLAG: adjacent prefixes did not reproduce"
    return 1, f"FAIL: {commits[good]} passes, next prefix {commits[bad]} fails"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter-timeout", type=int, default=900, metavar="SECONDS")
    p.add_argument("repo", help="repository containing the train commits")
    p.add_argument("base")
    p.add_argument("tip")
    p.add_argument("command", nargs="+", help="row adapter, called with BASE_SHA PREFIX_SHA appended")
    args = p.parse_args()
    if args.adapter_timeout < 1:
        p.error("adapter timeout must be positive")

    def git(*cmd):
        return subprocess.check_output(["git", "-C", args.repo, *cmd], text=True).strip()
    try:
        base, tip = (git("rev-parse", "--verify", "--end-of-options", ref + "^{commit}")
                     for ref in (args.base, args.tip))
        prefixes = git("rev-list", "--first-parent", "--reverse", base + ".." + tip).splitlines()
        if not prefixes or git("rev-parse", prefixes[0] + "^1") != base:
            p.error("base must precede tip on the same first-parent train")
        def run(a, b):
            return subprocess.run([*args.command, a, b], timeout=args.adapter_timeout).returncode
        rc, msg = diagnose([base, *prefixes], run)
    except (OSError, subprocess.SubprocessError) as exc:
        rc, msg = 2, f"FLAG: {exc}"
    print(msg, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
