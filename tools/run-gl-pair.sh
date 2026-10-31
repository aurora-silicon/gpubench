#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
# Headless matched native-GL / Zink run. Use a new output directory each time.
set -eu
out=${1:?usage: tools/run-gl-pair.sh NEW_OUTPUT_DIRECTORY}
exe=${GB_GL_BIN:-build/bin/gpubench-gl}
mkdir "$out"
gles=
if [ "${GB_GL_GLES:-0}" = 1 ]; then gles=--gles; fi
for driver in asahi zink; do
    dir=$out/$driver
    mkdir "$dir"
    export MESA_LOADER_DRIVER_OVERRIDE=$driver
    # gles is either empty or one fixed flag.
    "$exe" $gles --info > "$dir/info.json" 2> "$dir/log.txt"
    : > "$dir/results.jsonl"
    for spec in render.light:tri render.light:clear render.heavy:fill \
                render.heavy:opaque render.heavy:texture render.heavy:geom \
                compute.light:none compute.heavy:alu compute.heavy:bw; do
        test=${spec%:*}
        kind=${spec#*:}
        set -- --test "$test" --depths 1,4,16 --warmup-ms 300 --runs 5 \
               --run-ms 250 --timed-runs 3 --timestamps on
        if [ "$kind" != none ]; then set -- "$@" --param "kind=$kind"; fi
        if [ "$test" = render.heavy ]; then set -- "$@" --param res=1080p; fi
        "$exe" $gles "$@" >> "$dir/results.jsonl" 2>> "$dir/log.txt"
    done
done
