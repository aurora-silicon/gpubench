#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only
# run-linux.sh: run the plan on this Linux machine and write one record.
#
#   run-linux.sh [--device NAME] [--chunks a,b] [--tests t1,t2] [--out-root DIR]
#                [--bin DIR] [--lock FILE] [--note TEXT]
#
# --lock wraps every chunk in `flock FILE` (for example /var/tmp/gpu.lock), so
# the lock is held for one chunk at a time. After every chunk the GPU kernel
# log and the desktop are checked; a GPU fault stops the run.
# The record goes to <out-root>/<device>/linux/<date>-<kernel>-<mesa>/.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
. "$HERE/lib-meta.sh"
DEVICE=$(uname -n 2>/dev/null || echo linux); CHUNKS=""; TESTS=""; OUTROOT="$ROOT/records"
BIN="$ROOT/build/bin"; LOCK=""; NOTE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE=$2; shift 2;;
    --chunks) CHUNKS=$2; shift 2;;
    --tests) TESTS=$2; shift 2;;
    --out-root) OUTROOT=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    --lock) LOCK=$2; shift 2;;
    --note) NOTE=$2; shift 2;;
    *) echo "run-linux.sh: unknown argument $1" >&2; exit 2;;
  esac
done
EXE="$BIN/gpubench-vk"
[ -x "$EXE" ] || { echo "no $EXE (build first)" >&2; exit 2; }
INFO=$("$EXE" --info 2>/dev/null) || { echo "gpubench-vk --info failed" >&2; exit 3; }
KREL=$(uname -r)
MESA=$(printf '%s' "$INFO" | grep -o 'git-[0-9a-f]*' | head -1 | cut -c5-11)
[ -n "$MESA" ] || MESA=$(printf '%s' "$INFO" | grep -o 'Mesa [0-9][^ ,)"]*' | head -1 | cut -d' ' -f2)
DATE=$(date -u +%Y%m%d)
REC=$(pick_record_dir "$OUTROOT/$DEVICE/linux/$DATE-$KREL-${MESA:-unknown}")
mkdir -p "$REC"
T0=$(date +%s)
START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

state_json() {
  local first=1 z t v p
  printf '{"thermal":{'
  for z in /sys/class/thermal/thermal_zone*; do
    [ -r "$z/temp" ] || continue
    t=$(cat "$z/type" 2>/dev/null); v=$(cat "$z/temp" 2>/dev/null) || continue
    [ $first = 1 ] || printf ','; first=0
    printf '"%s":%s' "$(basename "$z"):$t" "${v:-null}"
  done
  printf '},"cpufreq_khz":{'; first=1
  for p in /sys/devices/system/cpu/cpufreq/policy*; do
    [ -r "$p/scaling_cur_freq" ] || continue
    [ $first = 1 ] || printf ','; first=0
    printf '"%s":%s' "$(basename "$p")" "$(cat "$p/scaling_cur_freq")"
  done
  printf '},"devfreq":{'; first=1
  for p in /sys/class/devfreq/*; do
    [ -r "$p/cur_freq" ] || continue
    [ $first = 1 ] || printf ','; first=0
    printf '"%s":%s' "$(basename "$p")" "$(jstr "$(cat "$p/cur_freq") $(cat "$p/governor" 2>/dev/null)")"
  done
  printf '},"loadavg":%s}' "$(jstr "$(cat /proc/loadavg)")"
}

module_json() { # the GPU kernel module: aurora, or asahi on kernels before the rename
  local m=/sys/module/aurora f bid="" first=1
  [ -d "$m" ] || m=/sys/module/asahi
  [ -d "$m" ] || { printf null; return; }
  f=$m/notes/.note.gnu.build-id
  bid=$( (cat "$f" 2>/dev/null || sudo -n cat "$f" 2>/dev/null) | od -An -tx1 -v | tr -d ' \n' | tail -c 40)
  printf '{"name":"%s","build_id":%s,"srcversion":%s,"params":{' "${m##*/}" "$(jstr_or_null "$bid")" \
    "$(jstr_or_null "$(cat $m/srcversion 2>/dev/null)")"
  for f in $m/parameters/*; do
    [ -r "$f" ] || continue
    [ $first = 1 ] || printf ','; first=0
    printf '"%s":%s' "$(basename "$f")" "$(jstr "$(cat "$f" 2>/dev/null)")"
  done
  printf '}}'
}

gpu_log() { # kernel log lines since the run started that point at GPU trouble
  journalctl -k --since "@$T0" --no-pager -q 2>/dev/null |
    grep -iE 'aurora|asahi|agx|drm' | grep -iE 'GPU fault|page fault|crashed|runtime stopped|client failed|timed out|timeout|still draining|queued-failure|hang|recover' || true
}

gpu_notes() { # other GPU driver lines worth keeping with the record (do not stop the run)
  journalctl -k --since "@$T0" --no-pager -q 2>/dev/null | grep -iE 'aurora|asahi|agx' |
    grep -viE 'GPU fault|page fault|crashed|runtime stopped|client failed|timed out|timeout|still draining|queued-failure|hang|recover' |
    sed -E 's/^.*(aurora|asahi|agx)/\1/I; s/[0-9]{4,}/N/g' | sort | uniq -c | sort -rn | head -20 || true
}

desktop_alive() {
  if pgrep -x Hyprland >/dev/null 2>&1; then echo true; elif [ -n "${WAYLAND_DISPLAY:-}" ]; then echo false; else echo null; fi
}

STATE_BEFORE=$(state_json)
echo "record: $REC"
# chunk order follows the plans
PLANS="$ROOT/suite/plan.txt"; [ -f "$ROOT/rt/plan.txt" ] && PLANS="$PLANS $ROOT/rt/plan.txt"
ALL=$(cat $PLANS | awk '$1 !~ /^#/ && NF >= 3 {print $1}' | awk '!seen[$0]++')
if [ -n "$CHUNKS" ]; then SEL=$(echo "$CHUNKS" | tr ',' ' '); else SEL=$ALL; fi
STOPPED=""
DESK0=$(desktop_alive)
OK_PREV=-1
for c in $SEL; do
  # Optional cool-down before each chunk. A chunk that recorded no ok rows
  # added no heat, so the chunk after it starts without one.
  OK=$(grep -c '"status":"ok"' "$REC/results.jsonl" 2>/dev/null || true)
  if [ -n "${GB_COOLDOWN_CMD:-}" ] && [ "$OK" != "$OK_PREV" ]; then
    sh "$GB_COOLDOWN_CMD" "$REC" "$c"
  fi
  OK_PREV=$OK
  echo "$(date +%T) chunk $c"
  if [ -n "$LOCK" ]; then
    flock "$LOCK" "$HERE/run-suite.sh" --api vk --bin "$BIN" --out "$REC" --chunks "$c" ${TESTS:+--tests "$TESTS"} --plans "$PLANS"
  else
    "$HERE/run-suite.sh" --api vk --bin "$BIN" --out "$REC" --chunks "$c" ${TESTS:+--tests "$TESTS"} --plans "$PLANS"
  fi
  bad=$(gpu_log)
  desk=$(desktop_alive)
  if [ -n "$bad" ] || { [ "$DESK0" = true ] && [ "$desk" != true ]; }; then
    echo "STOP after chunk $c: desktop=$desk"; printf '%s\n' "$bad" | tail -20
    STOPPED="after chunk $c"
    break
  fi
done
STATE_AFTER=$(state_json)
FW=$(journalctl -k -b --no-pager -q 2>/dev/null | grep -iE '(aurora|asahi).*(firmware|fw).*version|firmware version' | tail -1 | sed 's/.*\(irmware\|fw\)[^0-9]*//I' | cut -c1-40)
OSV=$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-linux}")
GPU_NAME=$(printf '%s' "$INFO" | sed -n 's/.*"device_name":"\([^"]*\)".*/\1/p')
SUITE_SHA=$(cat "$BIN/SUITE_SHA" 2>/dev/null || git -C "$ROOT" describe --always --dirty=+dirty --abbrev=7 2>/dev/null || echo unknown)
HEALTH=$(jobj gpu_log_lines "$(jstr "$(gpu_log)")" gpu_other_lines "$(jstr "$(gpu_notes)")" desktop_alive "$(desktop_alive)" stopped "$(jstr_or_null "$STOPPED")")
jobj schema '"gpubench-record/1"' device "$(jstr "$DEVICE")" os '"linux"' os_version "$(jstr "$OSV")" \
  os_build null kernel "$(jstr "$KREL")" kernel_module "$(module_json)" driver "$INFO" \
  mesa_sha "$(jstr_or_null "$MESA")" gpu "$(jobj name "$(jstr "$GPU_NAME")")" firmware "$(jstr_or_null "$FW")" \
  date_utc "$(jstr "$START_ISO")" host "$(jstr "$(uname -n)")" suite_sha "$(jstr "$SUITE_SHA")" \
  runner "$(jobj chunks "$(jstr "$(echo $SEL | tr ' ' ',')")" env "$(env_json)" duration_s "$(( $(date +%s) - T0 ))")" \
  state_before "$STATE_BEFORE" state_after "$STATE_AFTER" health "$HEALTH" notes "$(jstr "$NOTE")" > "$REC/meta.json"
echo "$(date +%T) done: $(wc -l < "$REC/results.jsonl" 2>/dev/null || echo 0) results in $REC ${STOPPED:+(STOPPED $STOPPED)}"
