# SPDX-License-Identifier: GPL-2.0-only
# lib-meta.sh: JSON helpers for the runners (bash 3.2 compatible, no python).
jesc() { # escape stdin as the inside of a JSON string (newlines -> \n)
  awk 'BEGIN{ORS=""} {gsub(/\\/,"\\\\"); gsub(/"/,"\\\""); gsub(/\t/," "); gsub(/\r/,""); if (NR>1) print "\\n"; print}'
}
jstr() { printf '"%s"' "$(printf '%s' "$1" | jesc)"; }
jstr_or_null() { if [ -n "$1" ]; then jstr "$1"; else printf null; fi; }
# jobj k1 v1 k2 v2 ...: values are raw JSON
jobj() {
  local first=1; printf '{'
  while [ $# -ge 2 ]; do
    [ $first = 1 ] || printf ','; first=0
    printf '"%s":%s' "$1" "$2"; shift 2
  done
  printf '}'
}
env_json() { # driver-relevant environment
  local first=1 k
  printf '{'
  for k in $(env | grep -E '^(VK_DRIVER_FILES|VK_ICD_FILENAMES|MESA_[A-Z0-9_]*|HK_[A-Z0-9_]*|AGX_[A-Z0-9_]*|ASAHI_[A-Z0-9_]*|GB_[A-Z0-9_]*|MTL_[A-Z0-9_]*)=' | cut -d= -f1 | sort -u); do
    [ $first = 1 ] || printf ','; first=0
    printf '"%s":%s' "$k" "$(jstr "$(printenv "$k")")"
  done
  printf '}'
}
pick_record_dir() { # pick_record_dir base -> base, base-b, base-c, ...
  local d=$1 s
  if [ ! -e "$d" ]; then echo "$d"; return; fi
  for s in b c d e f g h i j k l m n o p q r s t u v w x y z; do
    [ -e "$d-$s" ] || { echo "$d-$s"; return; }
  done
  echo "$d-$(date +%H%M%S)"
}
