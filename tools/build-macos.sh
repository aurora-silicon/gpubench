#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only
# build-macos.sh [OUT_DIR]: build the Metal executors on a Mac (needs the
# command-line tools). Output: OUT_DIR (default bin/macos)/gpubench-mtl, plus
# rt/metal/build.sh's executors when that script exists.
# GB_MACOS_MIN sets the deployment target (default 14.0, below Neo's macOS).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$R/bin/macos}
MIN=${GB_MACOS_MIN:-14.0}
mkdir -p "$OUT"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
{
  echo '/* Generated from the suite shader sources. */'
  echo 'static const char gb_msl_source[] ='
  for f in "$R"/suite/shaders/*.metal; do
    sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$/\\n"/' "$f"
  done
  echo ';'
} > "$T/msl_sources.h"
SHA=$(cat "$R/.suite_sha" 2>/dev/null || git -C "$R" describe --always --dirty=+dirty --abbrev=7 2>/dev/null || echo unknown)
clang -O2 -Wall -Wno-unused-parameter -Wno-missing-field-initializers -fobjc-arc -arch arm64 \
  -mmacosx-version-min="$MIN" -I"$R/suite/vk/src" -I"$T" \
  "$R/suite/metal/main.m" "$R/suite/vk/src/gb_common.c" \
  -framework Metal -framework Foundation -o "$OUT/gpubench-mtl"
echo "$SHA" > "$OUT/SUITE_SHA"
if [ -f "$R/rt/metal/build.sh" ]; then
  sh "$R/rt/metal/build.sh" "$OUT" || echo "WARNING: rt/metal/build.sh failed; the suite executor is still built" >&2
fi
ls -l "$OUT"
