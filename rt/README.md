# gpubench rt: ray-tracing benchmarks

A ray-tracing benchmark modelled on 3DMark Solar Bay, plus RT micro-benchmarks, with a Vulkan
executor (`gpubench-rt-vk`, Linux) and a Metal executor (`gpubench-rt-mtl`, macOS) that render the
same scene with the same parameters, data layouts, passes and timing method.

## The scene (`rt.scene`)

A procedural bay, generated identically by both executors from the parameters and a seed
(`common/scene.c`):

- **Water:** a 4x4 grid of patch meshes, one BLAS each. `dyn-patches` of them (the centre first)
  are deformed every frame by a compute shader (`deform`), and their BLASes are refit or rebuilt
  (`blas=refit|rebuild`).
- **Objects:** `instances` instances of `meshes` unique meshes (one BLAS each): spheres, towers, tori,
  displaced rocks and sails, each about `mesh-tris` triangles. A fraction `moving` of the instances
  spins and bobs; with `tlas=refit|rebuild` the TLAS is refit or rebuilt every frame.
- **Sails** are two-sided and, with `alpha=1`, alpha-tested: non-opaque geometry whose holes are
  evaluated per candidate hit (the any-hit path) in the ray queries and with `discard` in the
  raster pass.
- **Materials:** a fraction `reflective` of the surfaces (instances and water patches) reflects,
  with per-instance reflectivity and roughness. Rough surfaces jitter their reflection rays.

Per frame, five passes, each timed with GPU timestamps:

| Pass | Work |
| --- | --- |
| `update` | water deformation of the animated patches (compute) |
| `as` | BLAS refit/rebuild of the animated patches, then the TLAS build or refit |
| `raster` | G-buffer: albedo + roughness (RGBA8), normal + reflectivity (RGBA16F), depth (D32F, reversed Z) |
| `rt` | one invocation per pixel (8x8 groups): a shadow ray to the sun (`shadows=1`), and on reflective pixels `rpp` reflection rays with up to `bounces` bounces, each hit shaded with its own shadow ray. Inline ray queries (`VK_KHR_ray_query` / `metal::raytracing::intersection_query`), the Solar Bay approach. `trace=pipeline` (Vulkan only) does the same with a ray-tracing pipeline. |
| `composite` | direct light x shadow + reflection by reflectivity, sky, tone map; RGBA8 out |

Animation runs on a fixed time step (frame k is drawn at k/60 s): the water and the moving objects
animate per frame, while the camera stays at its frame-0 pose by default, so the work per frame
does not depend on how many frames a device renders in a window (`camera=orbit` makes it orbit).
After the timed windows a census frame, always frame 0, counts the rays (atomics, not in the timed
frames), measures the reflective pixel fraction and the image CRC, and can be written as PNG
(`--png`); the census is identical on every device.

### Presets (`presets/presets.ini`)

| Preset | Resolution | Unique / instanced tris | Instances | Dynamics | Rays |
| --- | --- | --- | --- | --- | --- |
| `raster` | 1080p | 232K / 5.1M | 256 | none | none (no reflection, no shadow rays) |
| `light` | 720p | 32K / 262K | 64 | none | 30% reflective, 1 rpp, 1 bounce, no shadows, no alpha |
| `medium` (default) | 1080p | 232K / 5.1M | 256 | 4 patches refit, TLAS rebuild, 10% moving | 50% reflective, 1 rpp, 1 bounce, shadows, alpha |
| `solarbay` | 1440p | 349K / 9.2M | 384 | 8 patches refit, TLAS rebuild, 20% moving | 60% reflective, 1 rpp, 1 bounce, no shadows, alpha |
| `heavy` | 1080p | 609K / 15.4M | 512 | 16 patches refit, TLAS rebuild, 25% moving | 70% reflective, 2 rpp, 2 bounces, shadows, alpha |
| `extreme` | 1440p | 1.69M / 30.7M | 512 | 16 patches rebuilt, TLAS rebuild, 50% moving | 100% reflective, 4 rpp, 3 bounces, shadows, alpha |

(Triangle counts are what the generator produces for the preset; the record's `info.scene` has
the exact numbers.)

## Micro-benchmarks

- `rt.asbuild`: one acceleration-structure build per submission. `kind=blas` builds a
  `tris`-triangle displaced-sphere BLAS (`build=fasttrace|fastbuild`), `kind=refit` updates it,
  `kind=tlas` builds a TLAS of `tris` instances of a 1K-triangle BLAS. Metrics: `mtri_s` /
  `minst_s` from the GPU time, `gpu_us`, `us_per_submit`, plus AS and scratch sizes.
- `rt.rays`: one dispatch of `rays` (default 256K) closest-hit ray queries per submission against
  the static scene of the preset, opaque traversal. `kind=coherent` = pinhole camera rays,
  `kind=incoherent` = random origins and directions. Depths 1..64 (ring of N submissions in
  flight). Metrics: `mrays_s` (wall clock), `mrays_s_gpu`, `us_per_submit`, `gpu_us`,
  `gpu_gap_us`.

## Executor interface

Both executors implement the suite's interface (`suite/README.md`, "Executor interface") and print
one `gpubench-result/1` line per configuration on stdout (`records/SCHEMA.md`); everything else
goes to stderr.

```
gpubench-rt-vk --info | --list
gpubench-rt-vk --test rt.scene --param preset=medium [--param rpp=4 ...] [--depths 1,2,3]
gpubench-rt-vk --test rt.asbuild --param kind=blas --param tris=262144
gpubench-rt-vk --test rt.rays --param kind=incoherent --depths 1,2,4,8,16,32,64
   [--warmup-ms N] [--runs N] [--run-ms N] [--timestamps on|off] [--device N]
```

Convenience spellings: `--preset X` is `--param preset=X`, and `--rpp 4` is `--param rpp=4` for
every parameter. Run controls that do not change the work: `--png PATH`, `--shader-stats`
(per-pipeline compiler statistics from `VK_KHR_pipeline_executable_properties`), `--frames N`
(fixed number of frames per window, for short checks), `--tag T`, `--preset-file`,
`--list-presets`.

Scene parameters: `res` (WxH or 720p/1080p/1440p/2160p), `mesh-tris`, `meshes`, `instances`,
`water-tris`, `blas` (static|refit|rebuild), `dyn-patches`, `tlas` (static|refit|rebuild),
`moving`, `reflective`, `rpp`, `bounces`, `shadows`, `tmax`, `alpha`, `trace` (query|pipeline),
`seed`.

### Timing method (both executors)

- Depth N = frames (submissions) in flight on one queue: before recording frame k the CPU waits
  for frame k-N (Vulkan: timeline semaphore value k-N; Metal: `waitUntilCompleted` on command
  buffer k-N). Every frame is recorded fresh. The scene's frames depend on each other (shared
  vertex buffer, TLAS and G-buffer): Vulkan puts a full memory barrier at the top of each frame,
  Metal relies on its default hazard tracking, which orders the same accesses.
- Warm-up (default 1.5 s), then 5 untimed windows of 1 s (a window also lasts until it has seen
  two completions, so frames slower than the window still give a rate): `fps` per window, `frame_ms` (observed
  completion to completion), `fps_1pct_low` (mean of the slowest 1% of frame intervals, per
  window), `latency_us`, `record_cpu_us`, `submit_cpu_us`.
- Then 3 timed windows with GPU timestamps around every pass: `gpu_update_us`, `gpu_as_us`,
  `gpu_raster_us`, `gpu_rt_us`, `gpu_composite_us`, `gpu_us` (whole frame), `fps_timed`, and
  `mrays_s` = rays per frame (census) / RT pass time. Vulkan: `vkCmdWriteTimestamp2` between the
  passes; pass i = stamp(i+1) - stamp(i). Metal: counter sample buffers at encoder boundaries
  (`MTLCounterSamplingPointAtStageBoundary`), with the same boundary arithmetic; Metal also
  reports `gpu_cb_us` (`GPUEndTime - GPUStartTime`).
- AS build flags map one to one: `static` = PREFER_FAST_TRACE / `MTLAccelerationStructureUsageNone`,
  `refit` = ALLOW_UPDATE / `Refit`, `rebuild` = PREFER_FAST_BUILD / `PreferFastBuild`. Opaque
  geometry is marked opaque on both sides; the sails are non-opaque with no duplicate any-hit.

## Building

Linux (from the repository root, or standalone):

```
cmake -S rt -B build/rt -G Ninja && ninja -C build/rt       # -> build/rt/bin/gpubench-rt-vk
```

Needs a C11 compiler, `glslangValidator` and the Vulkan loader; the Vulkan headers are vendored in
`rt/third_party` (override with `-DGB_VULKAN_INCLUDE=...`). The SPIR-V and the preset table are
embedded in the binary.

macOS: `rt/metal/build.sh [out-dir]` (arm64, deployment target 14.0, Xcode command-line tools).
The MSL (`shaders/rt.metal`) is embedded and compiled at startup; `RT_METALLIB=1` also builds
`gpubench-rt-mtl.metallib`, which the binary prefers when it sits next to it.

## Running

```sh
# One Vulkan configuration, with a census image.
build/bin/gpubench-rt-vk --preset solarbay --png /tmp/scene.png

# RT chunks through the shared record runners.
tools/run-linux.sh --device my-device --chunks rt-scene-a,rt-asbuild,rt-rays
tools/run-macos.sh --device my-device --chunks rt-scene-a,rt-asbuild,rt-rays

# Analyse a completed parameter sweep.
rt/tools/sweep.py analyze RECORD/results.jsonl --markdown
```

The standalone plan and record helpers are also available under `rt/tools/`. Set `RTB_NO_RT=1` only for a raster/composite functional check on a device without ray queries; this changes the workload and is excluded from RT performance comparisons.

## Notes on the implementations

- hk (Mesa tip `bfb73507d0`) on G17P lowers ray queries to the native Gen1 traversal
  (`agx_nir_lower_gen1_ray_query`, SCS-staged hardware traversal) and builds native-layout
  acceleration structures (`bvh/agx3_*.comp`); see `FAIRNESS.md`.
- Metal has no ray-tracing pipeline, so `trace=pipeline` is Vulkan only (Metal prints a skip).
- `--window` (Vulkan only) shows the scene in a Wayland window for tweaking; it produces no
  record.

## Files

- `common/`: parameters and presets, scene generator, JSON/record helpers, PNG writer (shared by
  both executors).
- `shaders/`: GLSL (`*.comp`, `*.vert`, `*.frag`, `rtpipe.*`, includes `*.glsl`) and the MSL twin
  `rt.metal` (one function per GLSL shader, same algorithm and layouts).
- `vulkan/`: the Vulkan executor. `metal/`: the Metal executor and its build script.
- `presets/presets.ini`, `plan.txt`: presets and the standard rt plan.
- `tools/`: plan/record runners, parameter sweeps, record import, and comparison/report helpers.
- `FAIRNESS.md`: workload equivalence and comparison limits.
