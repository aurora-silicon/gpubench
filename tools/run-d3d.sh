#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
# Run matched gpubench-d3d11 (DXVK), gpubench-d3d12 (vkd3d-proton) and optional
# gpubench-vk baseline into separate subdirectories suitable for tools/compare.py.
#
# Usage:
#   tools/run-d3d.sh NEW_OUTPUT_DIRECTORY [d3d11|d3d12|vk|all]
#
# Environment:
#   GB_D3D11_BIN       path to gpubench-d3d11 (default: build/bin/gpubench-d3d11)
#   GB_D3D12_BIN       path to gpubench-d3d12 (default: build/bin/gpubench-d3d12)
#   GB_VK_BIN          path to gpubench-vk    (default: build/bin/gpubench-vk)
#   GB_D3D_LIB_DIR     directory with libdxvk_*.so / libvkd3d-proton-*.so (prepended to LD_LIBRARY_PATH)
#   VK_DRIVER_FILES    optional Vulkan ICD manifest(s) for isolated Mesa overlays
#   GB_WARMUP_MS       warmup ms (default: 300)
#   GB_RUNS            untimed windows (default: 5)
#   GB_RUN_MS          window length ms (default: 250)
#   GB_TIMED_RUNS      timed windows (default: 3)
#   GB_VERIFY          1 to pass --verify on every D3D test (default: 1)
set -eu

out=${1:?usage: tools/run-d3d.sh NEW_OUTPUT_DIRECTORY [d3d11|d3d12|vk|all]}
which_api=${2:-all}

d11_bin=${GB_D3D11_BIN:-build/bin/gpubench-d3d11}
d12_bin=${GB_D3D12_BIN:-build/bin/gpubench-d3d12}
vk_bin=${GB_VK_BIN:-build/bin/gpubench-vk}
warm=${GB_WARMUP_MS:-300}
runs=${GB_RUNS:-5}
run_ms=${GB_RUN_MS:-250}
timed=${GB_TIMED_RUNS:-3}
verify_flag=
if [ "${GB_VERIFY:-1}" = 1 ]; then verify_flag=--verify; fi

if [ -n "${GB_D3D_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="${GB_D3D_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$out"

run_case() {
    exe=$1
    dir=$2
    is_vk=$3
    shift 3
    if [ "$is_vk" = 1 ]; then
        "$exe" "$@" --warmup-ms "$warm" --runs "$runs" --run-ms "$run_ms" \
            --timed-runs "$timed" --timestamps on >> "$dir/results.jsonl" 2>> "$dir/log.txt"
    else
        "$exe" $verify_flag "$@" --warmup-ms "$warm" --runs "$runs" --run-ms "$run_ms" \
            --timed-runs "$timed" --timestamps on >> "$dir/results.jsonl" 2>> "$dir/log.txt"
    fi
}

run_suite_for() {
    api=$1
    exe=$2
    is_vk=0
    if [ "$api" = vk ]; then is_vk=1; fi
    dir=$out/$api
    mkdir -p "$dir"
    "$exe" --info > "$dir/info.json" 2> "$dir/log.txt"
    : > "$dir/results.jsonl"

    run_case "$exe" "$dir" "$is_vk" --test render.light --param kind=tri --depths 1,8,32
    run_case "$exe" "$dir" "$is_vk" --test render.heavy --param kind=opaque --param res=1080p --depths 1,4
    run_case "$exe" "$dir" "$is_vk" --test render.heavy --param kind=fill --param res=1080p --depths 1,4
    if [ "$is_vk" = 0 ]; then
        run_case "$exe" "$dir" "$is_vk" --test render.heavy --param kind=fill --param res=1080p --param msaa=4 --depths 1,4
        run_case "$exe" "$dir" "$is_vk" --test render.draws --param mode=direct --param draws=2048 --depths 1,4
        run_case "$exe" "$dir" "$is_vk" --test render.draws --param mode=instanced --param draws=2048 --depths 1,4
        run_case "$exe" "$dir" "$is_vk" --test render.draws --param mode=indirect --param draws=2048 --depths 1,4
    fi
    run_case "$exe" "$dir" "$is_vk" --test render.deferred --param mode=dram --param res=1080p --param msaa=1 --depths 1,2
    run_case "$exe" "$dir" "$is_vk" --test render.deferred --param mode=dram --param res=1080p --param msaa=4 --depths 1,2
    if [ "$api" = d3d12 ]; then
        run_case "$exe" "$dir" "$is_vk" --test render.deferred --param mode=split --param res=1080p --param msaa=1 --depths 1,2
    fi
    run_case "$exe" "$dir" "$is_vk" --test render.shadow --param mode=depth_only --param res=2048 --depths 1,2
    run_case "$exe" "$dir" "$is_vk" --test render.shadow --param mode=alpha_test --param res=2048 --depths 1,2
    run_case "$exe" "$dir" "$is_vk" --test render.postfx --param res=1440p --param msaa=1 --depths 1,2
    run_case "$exe" "$dir" "$is_vk" --test render.postfx --param res=1440p --param msaa=4 --depths 1,2
    run_case "$exe" "$dir" "$is_vk" --test compute.light --depths 1,8,32
    run_case "$exe" "$dir" "$is_vk" --test compute.heavy --param kind=alu --depths 1,4
    run_case "$exe" "$dir" "$is_vk" --test compute.heavy --param kind=bw --depths 1,4
    run_case "$exe" "$dir" "$is_vk" --test compute.conv2d --param kind=gaussian --param res=1080p --depths 1,4
    run_case "$exe" "$dir" "$is_vk" --test transfer.ops --param kind=clear_4m --depths 1,4
    run_case "$exe" "$dir" "$is_vk" --test transfer.ops --param kind=copy_b2i --depths 1,4
    run_case "$exe" "$dir" "$is_vk" --test transfer.ops --param kind=mipchain --depths 1,4
    MESA_SHADER_CACHE_DISABLE=true DXVK_STATE_CACHE=0 VKD3D_SHADER_CACHE_PATH=0 \
        run_case "$exe" "$dir" "$is_vk" --test compile.pso --param kind=compute --param count=32
    MESA_SHADER_CACHE_DISABLE=true DXVK_STATE_CACHE=0 VKD3D_SHADER_CACHE_PATH=0 \
        run_case "$exe" "$dir" "$is_vk" --test compile.pso --param kind=graphics --param count=32
    run_case "$exe" "$dir" "$is_vk" --test sync.signal --param kind=host
}

case "$which_api" in
    d3d11) run_suite_for d3d11 "$d11_bin" ;;
    d3d12) run_suite_for d3d12 "$d12_bin" ;;
    vk)    run_suite_for vk "$vk_bin" ;;
    all)
        [ -x "$vk_bin" ] && run_suite_for vk "$vk_bin"
        [ -x "$d11_bin" ] && run_suite_for d3d11 "$d11_bin"
        [ -x "$d12_bin" ] && run_suite_for d3d12 "$d12_bin"
        ;;
    *) echo "unknown target: $which_api (expected d3d11|d3d12|vk|all)" >&2; exit 2 ;;
esac
