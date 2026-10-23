# gpubench

Vulkan and Metal benchmarks for GPU driver development. The standard suite measures submission, scheduling, synchronization, memory management, and GPU execution. The ray-tracing extension adds procedural scenes, acceleration-structure builds, and ray-query workloads.

Both executors use matching workloads, dimensions, queue depths, warm-up periods, and timing windows. Untimed throughput and GPU-timestamp measurements are separate passes. See [the timing method](suite/README.md) and [comparison limits](suite/FAIRNESS.md).

## Published records

| Device | OS | Configurations | Status | Coverage |
| --- | --- | ---: | --- | --- |
| [Neo / A18 Pro](records/neo/macos/20261023-25G83/) | macOS 26.6.2 (25G83) | 209 | 207 passed, 2 unsupported | 111 standard, 98 ray tracing |
| [M5 Pro](records/m5-pro/macos/20261023-26B5086k/) | macOS 27.2 (26B5086k) | 82 | 82 passed | Standard core, heavy work, concurrency, and driver features |

The Neo skips are Vulkan-only ray-tracing pipeline modes. M5 Pro ray-tracing, memory-allocation, and tiler-overflow sweeps were not run in this record. Results cover different test sets; compare matching configurations. Linux measurement records will be added after qualification.

[Record documentation](records/README.md) describes provenance, coverage, thermal data, and historical source revisions. Raw measurement values are retained unchanged.

## Build and run

### macOS

Requires an arm64 Mac and Xcode command-line tools.

```sh
tools/build-macos.sh bin/macos
tools/run-macos.sh --device my-device --chunks core-a,core-b,multi,feat
```

### Linux

Requires CMake, Ninja, a C11 compiler, the Vulkan loader, and `glslangValidator`. The included Vulkan headers are a fallback when system headers are unavailable.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
tools/run-linux.sh --device my-device --chunks core-a,core-b,multi,feat
```

Set `VK_DRIVER_FILES` to select a specific driver. The Linux runner accepts `--lock FILE` to coordinate GPU use with other jobs. Use `--chunks core-a` for the submission tests, or omit `--chunks` for the complete standard and ray-tracing plans.

For a short functional run, set `GB_WARMUP_MS=20 GB_RUNS=1 GB_RUN_MS=40 GB_TIMED_RUNS=1`. Keep those results separate from measured runs.

## Layout

- [suite/](suite/README.md): standard compute, render, synchronization, and memory benchmarks.
- [rt/](rt/README.md): ray-tracing scenes and microbenchmarks.
- [records/](records/README.md): per-device records and JSON schema.
- `tools/`: portable builders, runners, and record comparison.

## License

Project code is licensed under [GNU GPL version 2 only](LICENSE), identified as `GPL-2.0-only`. Vendored Vulkan headers retain their upstream licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).
