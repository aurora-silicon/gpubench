#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
# run-record.sh <exe> <device> <records-root> [chunk-regex] [plan]
# Runs the rt plan (default rt/plan.txt next to this script, or ./plan.txt) with <exe> and writes one
# record directory (records/SCHEMA.md):
#   <records-root>/<YYYYMMDD>-<id>/meta.json, results.jsonl, log.txt
# <id> is "<uname -r>-<mesa short sha>" on Linux and the macOS build on macOS.
# POSIX sh; works on Linux (on a shared test device, set for example RT_LOCK=/var/tmp/gpu.lock RT_MEMMAX=5G) and stock macOS.
set -u
EXE=$1; DEV=$2; ROOT=$3; RE=${4:-.}
H=$(cd "$(dirname "$0")" && pwd)
PLAN=${5:-}
[ -z "$PLAN" ] && { [ -f "$H/../plan.txt" ] && PLAN=$H/../plan.txt || PLAN=$H/plan.txt; }
OS=$(uname -s | tr 'A-Z' 'a-z'); [ "$OS" = darwin ] && OS=macos
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
INFO=$("$EXE" --info 2>/dev/null)
jget() { printf '%s' "$INFO" | sed -n "s/.*\"$1\":\"\([^\"]*\)\".*/\1/p" | head -1; }
if [ "$OS" = macos ]; then
   ID=$(sw_vers -buildVersion)
   OSV="macOS $(sw_vers -productVersion)"; OSB=$ID; KERN=$(uname -v)
else
   SHA=$(jget mesa_sha); ID="$(uname -r)-${SHA:-unknown}"
   OSV=$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-Linux}"); OSB=""; KERN=$(uname -r)
fi
D=$ROOT/$(date -u +%Y%m%d)-$ID
n=0; base=$D
while [ -e "$D" ]; do n=$((n + 1)); D=$base-$(printf "\\$(printf %o $((97 + n)))"); done
mkdir -p "$D"

esc() { sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | awk 'BEGIN { ORS = "" } { if (NR > 1) print "\\n"; print }'; }

state() { # JSON object with the thermal and clock state
   if [ "$OS" = macos ]; then
      th=$(pmset -g therm 2>/dev/null | esc)
      pm=$(sudo -n powermetrics --samplers gpu_power,thermal -i 500 -n 1 2>/dev/null | grep -E 'GPU|frequency|residency|pressure|Thermal' | head -20 | esc)
      printf '{"pmset_therm":"%s","powermetrics":"%s","thermal_state":"%s"}' "$th" "$pm" "$(pmset -g therm 2>/dev/null | grep -i -o 'CPU_Speed_Limit[^,]*' | head -1)"
   else
      printf '{"thermal":{'
      first=1
      for z in /sys/class/thermal/thermal_zone*; do
         [ -r "$z/temp" ] || continue
         t=$(cat "$z/temp" 2>/dev/null) || continue; ty=$(cat "$z/type" 2>/dev/null)
         [ $first = 1 ] || printf ','; first=0
         printf '"%s:%s":%s' "$(basename "$z")" "$ty" "$(awk -v t="$t" 'BEGIN { printf "%.1f", t / 1000 }')"
      done
      printf '},"cpufreq_khz":{'
      first=1
      for p in /sys/devices/system/cpu/cpufreq/policy*; do
         [ -r "$p/scaling_cur_freq" ] || continue
         [ $first = 1 ] || printf ','; first=0
         printf '"%s":%s' "$(basename "$p")" "$(cat "$p/scaling_cur_freq")"
      done
      printf '},"devfreq":{'
      first=1
      for p in /sys/class/devfreq/*; do
         [ -r "$p/cur_freq" ] || continue
         [ $first = 1 ] || printf ','; first=0
         printf '"%s":%s' "$(basename "$p")" "$(cat "$p/cur_freq")"
      done
      printf '}}'
   fi
}

S0=$(state)
[ "$OS" = macos ] && { sudo -n powermetrics --samplers gpu_power,thermal -i 1000 > "$D/powermetrics.txt" 2>/dev/null & PM=$!; }
T0=$(date +%s)
"$H/run-plan.sh" "$EXE" "$PLAN" "$D/results.jsonl" "$RE"
mv "$D/results.jsonl.log" "$D/log.txt" 2>/dev/null
[ -n "${PM:-}" ] && sudo -n kill "$PM" 2>/dev/null
S1=$(state)
CHUNKS=$(grep -v '^[[:space:]]*#' "$PLAN" | awk 'NF >= 3 && $2 == "rt" { print $1 }' | uniq | grep -E "$RE" | awk 'BEGIN { ORS = "" } { if (NR > 1) print ","; print "\"" $1 "\"" }')
ENVJ=$(env | grep -E '^(VK_DRIVER_FILES|VK_ICD_FILENAMES|MESA_|HK_|AGX_|ASAHI_)' | sed -e 's/"/\\"/g' -e 's/^\([^=]*\)=\(.*\)$/"\1":"\2"/' | awk 'BEGIN { ORS = "" } { if (NR > 1) print ","; print }')
if [ "$OS" = macos ]; then
   KMOD=null; HEALTH=null; FW=null
else
   bid=$(modinfo -F srcversion asahi 2>/dev/null); par=""
   for f in /sys/module/asahi/parameters/*; do [ -r "$f" ] && par="$par${par:+,}\"$(basename "$f")\":\"$(cat "$f" 2>/dev/null)\""; done
   bidf=$(readelf -n "$(modinfo -n asahi 2>/dev/null)" 2>/dev/null | awk '/Build ID/ { print $3; exit }')
   KMOD="{\"name\":\"asahi\",\"build_id\":\"$bidf\",\"srcversion\":\"$bid\",\"params\":{$par}}"
   KL=$(journalctl -k -b --since "@$T0" 2>/dev/null | grep -i -E 'asahi|agx|gpu|drm|fault|timeout' | tail -40 | esc)
   alive=false; pgrep -x Hyprland > /dev/null && alive=true
   HEALTH="{\"kernel_log\":\"$KL\",\"desktop_alive\":$alive}"
   FW=null
fi
# a job directory carries SUITE_SHA; never call git on stock macOS (it prompts for the tools)
if [ -f "$H/SUITE_SHA" ]; then SUITE=$(cat "$H/SUITE_SHA")
elif [ "$OS" != macos ]; then
   SUITE=$(cd "$H" && git rev-parse --short HEAD 2>/dev/null)
   [ -n "$SUITE" ] && ! (cd "$H" && git diff --quiet HEAD -- . 2>/dev/null) && SUITE="$SUITE+dirty"
fi
[ -z "${SUITE:-}" ] && SUITE=unknown
cat > "$D/meta.json" <<JSON
{"schema":"gpubench-record/1","device":"$DEV","os":"$OS","os_version":"$OSV","os_build":$( [ -n "$OSB" ] && echo "\"$OSB\"" || echo null ),
 "kernel":"$KERN","kernel_module":$KMOD,"driver":$(printf '%s' "$INFO" | sed -n 's/.*"driver":\({[^}]*}\).*/\1/p'),
 "gpu":$(printf '%s' "$INFO" | sed -n 's/.*"gpu":\({[^}]*}\).*/\1/p'),"firmware":$FW,"date_utc":"$START","host":"$(hostname)",
 "suite_sha":"$SUITE","lane":"rt","runner":{"chunks":[$CHUNKS],"plan":"$(basename "$PLAN")","env":{$ENVJ}},
 "state_before":$S0,"state_after":$S1,"health":$HEALTH,"notes":"${RT_NOTES:-}"}
JSON
echo "$D"
