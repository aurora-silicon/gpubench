#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
# run-plan.sh <exe> <plan> <out.jsonl> [chunk-regex] [extra executor args...]
# Runs every "<chunk> rt <test> <args>" line of <plan> whose chunk matches <chunk-regex>
# (default: every chunk), appending result lines to <out.jsonl> and executor stderr to
# <out.jsonl>.log. POSIX sh only, so it runs on a stock macOS install.
# RT_LOCK=<file>: wrap each chunk in `flock <file>` (Neo Linux: /var/tmp/neo-gpu.lock).
# RT_MEMMAX=<size>: run each chunk in a systemd user scope with MemoryMax (Linux).
# RT_COOL=<celsius>: before each chunk (inside the lock), wait until every thermal zone reads below
#   this, at most RT_COOL_MAX seconds (default 120); the wait is logged (Linux sysfs only).
set -u
EXE=$1; PLAN=$2; OUT=$3; RE=${4:-.}
shift 3; [ $# -gt 0 ] && shift
LOG=$OUT.log
chunks=$(grep -v '^[[:space:]]*#' "$PLAN" | awk 'NF >= 3 && $2 == "rt" { print $1 }' | uniq | grep -E "$RE")
for c in $chunks; do
   echo "$(date -u +%H:%M:%S) chunk $c" | tee -a "$LOG" >&2
   tmp=$OUT.chunk.sh
   : > "$tmp"
   if [ -n "${RT_COOL:-}" ] && [ -d /sys/class/thermal ]; then
      cat >> "$tmp" <<COOL
i=0; m=0
while [ \$i -lt ${RT_COOL_MAX:-120} ]; do
   m=\$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
   [ "\$m" -lt $((RT_COOL * 1000)) ] && break
   sleep 1; i=\$((i + 1))
done
echo "cool-down: max zone \$((m / 1000)) C after \${i} s" >> "$LOG"
COOL
   fi
   grep -v '^[[:space:]]*#' "$PLAN" | awk -v c="$c" 'NF >= 3 && $1 == c && $2 == "rt"' | while read -r ch lane test args; do
      echo "\"$EXE\" --test $test $args $* >> \"$OUT\" 2>> \"$LOG\" || echo \"exit \$? for $test $args\" >> \"$LOG\"" >> "$tmp"
   done
   run="sh $tmp"
   [ -n "${RT_MEMMAX:-}" ] && run="systemd-run --user --scope -q -p MemoryMax=$RT_MEMMAX $run"
   if [ -n "${RT_LOCK:-}" ]; then flock "$RT_LOCK" $run; else $run; fi
   rm -f "$tmp"
done
