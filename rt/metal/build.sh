#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
# rt/metal/build.sh [out-dir]: builds gpubench-rt-mtl (arm64 macOS, Metal) on a Mac with the
# command-line tools. tools/build-macos.sh calls it with its output directory. The MSL source
# (rt/shaders/rt.metal) and the preset table are embedded; the binary compiles the MSL at startup
# unless <exe>.metallib exists next to it (RT_METALLIB=1 builds one with the Metal toolchain).
set -eu
H=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$H/../build/macos}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
G=$(mktemp -d)
trap 'rm -rf "$G"' EXIT
MIN=${RT_MACOS_MIN:-14.0}
embed() { # embed <src> <symbol>
   (cat "$1"; printf '\000') > "$G/$2"
   (cd "$G" && xxd -i "$2" | sed -e 's/^unsigned char/const char/' -e '/^unsigned int/d') > "$G/$2.c"
}
embed "$H/shaders/rt.metal" rt_metal_src
embed "$H/presets/presets.ini" rt_presets_ini
xcrun -sdk macosx clang -std=gnu11 -O2 -arch arm64 -mmacosx-version-min="$MIN" -fobjc-arc \
   -Wall -Wno-unused-parameter -Wno-missing-field-initializers \
   -o "$OUT/gpubench-rt-mtl" "$H/metal/main.m" "$H/common/params.c" "$H/common/scene.c" "$H/common/json.c" \
   "$H/common/util.c" "$H/common/result.c" "$H/common/trav.c" "$G/rt_metal_src.c" "$G/rt_presets_ini.c" \
   -framework Metal -framework Foundation
if [ "${RT_METALLIB:-0}" = 1 ]; then
   xcrun -sdk macosx metal -std=metal3.0 -O3 -mmacosx-version-min="$MIN" -c "$H/shaders/rt.metal" -o "$G/rt.air"
   xcrun -sdk macosx metallib "$G/rt.air" -o "$OUT/gpubench-rt-mtl.metallib"
fi
git -C "$H" rev-parse --short HEAD > "$OUT/RT_SHA" 2>/dev/null || cat "$H/../.suite_sha" > "$OUT/RT_SHA" 2>/dev/null || true
echo "built $OUT/gpubench-rt-mtl"
