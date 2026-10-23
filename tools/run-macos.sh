#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only
# run-macos.sh: run the plan on this Mac and write one record (stock macOS:
# bash 3.2, no python, no developer tools needed).
#
#   run-macos.sh [--device NAME] [--chunks a,b] [--tests t1,t2] [--out-root DIR]
#                [--bin DIR] [--note TEXT]
#
# When `sudo -n` works, powermetrics samples GPU clocks and thermal state once
# per second for the whole run (powermetrics.txt in the record).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
. "$HERE/lib-meta.sh"
DEVICE=$(hostname -s 2>/dev/null || echo mac); CHUNKS=""; TESTS=""; OUTROOT="$ROOT/records"
BIN="$ROOT/bin/macos"; NOTE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE=$2; shift 2;;
    --chunks) CHUNKS=$2; shift 2;;
    --tests) TESTS=$2; shift 2;;
    --out-root) OUTROOT=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    --note) NOTE=$2; shift 2;;
    *) echo "run-macos.sh: unknown argument $1" >&2; exit 2;;
  esac
done
EXE="$BIN/gpubench-mtl"
[ -x "$EXE" ] || { echo "no $EXE" >&2; exit 2; }
INFO=$("$EXE" --info 2>/dev/null) || { echo "gpubench-mtl --info failed" >&2; exit 3; }
BUILD=$(sw_vers -buildVersion)
DATE=$(date -u +%Y%m%d)
REC=$(pick_record_dir "$OUTROOT/$DEVICE/macos/$DATE-$BUILD")
mkdir -p "$REC"
T0=$(date +%s)
START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

state_json() {
  local pm=""
  if sudo -n true 2>/dev/null; then
    pm=$(sudo -n powermetrics -n 1 -i 500 --samplers gpu_power,thermal 2>/dev/null |
         grep -E 'GPU HW active|GPU idle|GPU SW|GPU Power|pressure level' | head -20)
  fi
  jobj therm "$(jstr "$(pmset -g therm 2>/dev/null)")" powermetrics "$(jstr_or_null "$pm")" \
    loadavg "$(jstr "$(sysctl -n vm.loadavg 2>/dev/null)")"
}

STATE_BEFORE=$(state_json)
PM_PID=""
if sudo -n true 2>/dev/null; then
  sudo -n powermetrics -i 1000 --samplers gpu_power,thermal -o "$REC/powermetrics.txt" >/dev/null 2>&1 &
  PM_PID=$!
fi
echo "record: $REC"
PLANS="$ROOT/suite/plan.txt"; [ -f "$ROOT/rt/plan.txt" ] && PLANS="$PLANS $ROOT/rt/plan.txt"
ALL=$(cat $PLANS | awk '$1 !~ /^#/ && NF >= 3 {print $1}' | awk '!seen[$0]++')
if [ -n "$CHUNKS" ]; then SEL=$(echo "$CHUNKS" | tr ',' ' '); else SEL=$ALL; fi
for c in $SEL; do
  echo "$(date +%T) chunk $c"
  "$HERE/run-suite.sh" --api mtl --bin "$BIN" --out "$REC" --chunks "$c" ${TESTS:+--tests "$TESTS"} --plans "$PLANS"
done
if [ -n "$PM_PID" ]; then sudo -n kill "$PM_PID" 2>/dev/null; fi
STATE_AFTER=$(state_json)
GPU_NAME=$(printf '%s' "$INFO" | sed -n 's/.*"device_name":"\([^"]*\)".*/\1/p')
SUITE_SHA=$(cat "$BIN/SUITE_SHA" 2>/dev/null || echo unknown)
jobj schema '"gpubench-record/1"' device "$(jstr "$DEVICE")" os '"macos"' \
  os_version "$(jstr "$(sw_vers -productVersion)")" os_build "$(jstr "$BUILD")" kernel "$(jstr "$(uname -v)")" \
  kernel_module null driver "$INFO" mesa_sha null gpu "$(jobj name "$(jstr "$GPU_NAME")")" firmware null \
  date_utc "$(jstr "$START_ISO")" host "$(jstr "$(hostname)")" suite_sha "$(jstr "$SUITE_SHA")" \
  runner "$(jobj chunks "$(jstr "$(echo $SEL | tr ' ' ',')")" env "$(env_json)" duration_s "$(( $(date +%s) - T0 ))")" \
  state_before "$STATE_BEFORE" state_after "$STATE_AFTER" health null notes "$(jstr "$NOTE")" > "$REC/meta.json"
echo "$(date +%T) done: $(wc -l < "$REC/results.jsonl" | tr -d ' ') results in $REC"
