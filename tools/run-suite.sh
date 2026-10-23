#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only
# run-suite.sh: run plan lines and append result lines to a record directory.
# Portable (bash 3.2 on stock macOS, no python).
#
#   run-suite.sh --api vk|mtl --bin DIR --out RECORD_DIR [--chunks a,b] [--plans "p1 p2"]
#                [--tests t1,t2] [--dry-run]
#
# Each plan line is "<chunk> <lane> <test> <args...>". lane "suite" runs
# gpubench-<api>, any other lane X runs gpubench-X-<api>. Results go to
# RECORD_DIR/results.jsonl, executor stderr to RECORD_DIR/log.txt, and one
# progress line per plan line to stdout. Each plan line is killed after
# GB_LINE_TIMEOUT seconds (default 300); a killed line is recorded as a fail.
set -u
LIMIT=${GB_LINE_TIMEOUT:-300}
API=""; BIN=""; OUT=""; CHUNKS=""; PLANS=""; TESTS=""; DRY=""
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
while [ $# -gt 0 ]; do
  case "$1" in
    --api) API=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --chunks) CHUNKS=$2; shift 2;;
    --plans) PLANS=$2; shift 2;;
    --tests) TESTS=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "run-suite.sh: unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$API" ] && [ -n "$BIN" ] && [ -n "$OUT" ] || { echo "usage: run-suite.sh --api vk|mtl --bin DIR --out DIR" >&2; exit 2; }
if [ -z "$PLANS" ]; then
  PLANS="$ROOT/suite/plan.txt"
  [ -f "$ROOT/rt/plan.txt" ] && PLANS="$PLANS $ROOT/rt/plan.txt"
fi
mkdir -p "$OUT"
in_list() { # in_list item "a,b,c" (empty list = everything)
  [ -z "$2" ] && return 0
  case ",$2," in *",$1,"*) return 0;; esac
  return 1
}
n=0
for plan in $PLANS; do
  [ -f "$plan" ] || continue
  while read -r chunk lane test args; do
    case "$chunk" in ''|'#'*) continue;; esac
    in_list "$chunk" "$CHUNKS" || continue
    in_list "$test" "$TESTS" || continue
    if [ "$lane" = suite ]; then exe="$BIN/gpubench-$API"; else exe="$BIN/gpubench-$lane-$API"; fi
    n=$((n + 1))
    if [ ! -x "$exe" ]; then
      echo "$(date +%T) [$chunk] $lane $test $args: no executor $exe, skipped"
      continue
    fi
    # Run-control overrides (smoke passes) go to every executor as the
    # standard options of suite/README.md.
    ctl=""
    [ -n "${GB_WARMUP_MS:-}" ] && ctl="$ctl --warmup-ms $GB_WARMUP_MS"
    [ -n "${GB_RUNS:-}" ] && ctl="$ctl --runs $GB_RUNS"
    [ -n "${GB_RUN_MS:-}" ] && ctl="$ctl --run-ms $GB_RUN_MS"
    if [ -n "$DRY" ]; then echo "$exe --test $test $args$ctl"; continue; fi
    t0=$(date +%s)
    echo "### $(date +%T) $lane $test $args" >> "$OUT/log.txt"
    # shellcheck disable=SC2086
    "$exe" --test "$test" $args $ctl >> "$OUT/results.jsonl" 2>> "$OUT/log.txt" < /dev/null &
    pid=$!
    ( trap 'kill $sp 2>/dev/null; exit 0' TERM; sleep "$LIMIT" & sp=$!; wait $sp; kill -9 "$pid" 2>/dev/null ) &
    wd=$!
    wait "$pid"; rc=$?
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    if [ "$rc" -ge 128 ]; then
      printf '{"schema":"gpubench-result/1","lane":"%s","test":"%s","params":{},"mode":"queue","depth":0,"status":"fail","error":"executor killed or crashed (rc=%s) running: %s"}\n' \
        "$lane" "$test" "$rc" "$args" >> "$OUT/results.jsonl"
    fi
    echo "$(date +%T) [$chunk] $lane $test $args: rc=$rc $(( $(date +%s) - t0 ))s"
  done < "$plan"
done
[ "$n" -gt 0 ] || echo "run-suite.sh: no plan lines selected" >&2
exit 0
