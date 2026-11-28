#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only
# Assemble a self-contained directory using the host's installed replay tools.
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: prepare.sh NEW_KIT GPUBENCH_BIN}
bin=${2:?supply directory containing gpubench-gl and gpubench-vk}
mkdir "$out"
mkdir "$out/bin" "$out/ref"
cp "$src"/{../batch-gate.py,run.py} "$out/"
cp "$bin"/gpubench-{gl,vk} "$out/bin/"
for tool in gfxrecon-replay eglretrace; do cp "$(command -v "$tool")" "$out/bin/"; done
cp "$src/ref/glframe0000006935.png" "$out/ref/"
cp "$src/ref/vulkan.png" "$out/ref/"
cd "$out"
curl --fail --location --silent --show-error https://raw.githubusercontent.com/LunarG/gfxreconstruct/4401228c791b11fb7ea15b93795cbd1e938307c0/test/known_good/triangle.gfxr -o triangle.gfxr
curl --fail --location --silent --show-error https://raw.githubusercontent.com/apitrace/apitrace-tests/7b1b37ed02c3c1de9a7979129f7b14d041847d18/traces/tri_glsl.trace -o tri_glsl.trace
sha256sum -c <<'SUMS'
c52c469ca52c9f5f7e01500eefc20e92ee962bc2364d34fdd85b831dfcaabcb0  triangle.gfxr
ae070bbfd74f3d988e057ff74ae13f51fa1896b4b2e99e51e764ea4b6eb370d3  tri_glsl.trace
SUMS
