# Headless GL/GLES executor

`gpubench-gl` measures the standard suite's render and compute workloads through
EGL. Use the same binary with `MESA_LOADER_DRIVER_OVERRIDE=asahi` and `zink` to
compare native GL with GL over Vulkan. The requested override and the actual
GL renderer, vendor, version, capabilities, and suite revision are included in
`--info` and every result. Check the renderer before accepting a measurement;
an environment variable alone does not prove which driver loaded.

## Build and smoke

Install the standard Vulkan build dependencies plus EGL, libepoxy development
files, pkg-config, and Python 3. On Debian/Ubuntu the extra packages are
`libepoxy-dev libegl1-mesa-dev libgl1-mesa-dri mesa-vulkan-drivers`.

```sh
cmake -S . -B build -G Ninja -DGB_BUILD_GL=ON -DBUILD_TESTING=ON
cmake --build build --target gpubench-gl gpubench-vk test-gl-work -j8
ctest --test-dir build --output-on-failure

EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
  MESA_LOADER_DRIVER_OVERRIDE=llvmpipe GALLIUM_DRIVER=llvmpipe \
  build/bin/gpubench-gl --test render.heavy --param kind=texture --param res=720p \
  --depths 1 --warmup-ms 20 --runs 2 --run-ms 100 --timed-runs 1 --verify
```

`GB_BUILD_GL` defaults to `OFF`, preserving existing Vulkan-only builds.
The CPU smoke runs desktop GL and GLES on llvmpipe, checks output readback,
checks Vulkan configuration keys and derived work counts, and tests fallback
contexts and unsupported configurations. CI retains the test log.

## Matched work

| Test | Work per submission |
| --- | --- |
| `render.light kind=tri` (default) / `clear` | 64x64 RGBA8, clear to (0.1, 0.2, 0.3, 1); optional full-screen triangle |
| `render.heavy kind=fill` (default) | 16 full-screen triangle instances, RGBA8, source-alpha blending; color and alpha factors match Vulkan |
| `render.heavy kind=opaque` | Same 16 instances, blending disabled |
| `render.heavy kind=texture` | Same 16 instances and source-alpha blending as `fill`, one nearest-filtered repeating level-zero lookup per fragment from a 256x256 RGBA8 pattern |
| `render.heavy kind=geom` | 262144 grid triangles, blending disabled |
| `compute.light` | 1 group of 32 threads, one uint store per thread |
| `compute.heavy kind=alu` (default) | 1024 groups of 256 threads, 8 independent FMA chains, 4 steps per iteration, 128 iterations, one float output per thread |
| `compute.heavy kind=bw` | Copy 64 MiB of float4 values; 64 MiB read plus 64 MiB written |

Render resolutions are `720p`, `1080p` (default), `1440p`, and `2160p`.
`fill` retains the existing Vulkan workload. `opaque` and `texture` are also
implemented in Vulkan; Metal does not implement these two new variants.
Texture bytes are `(x, y, x xor y, 255)`. Fragment coordinates divided by 256
supply UVs; the sampled value is multiplied by the same per-submission color
used by the flat-color workload. The GL adapter changes only shader version,
vertex ID, descriptor, and uniform syntax in the shared GLSL sources.

Desktop GL needs 3.3 for rendering and 4.3 for compute. `--gles` selects GLES
3.0 rendering / 3.1 compute. The FMA ALU shader additionally needs GLES 3.2
or GLES 3.1 with [`EXT_gpu_shader5`](https://registry.khronos.org/OpenGL/extensions/EXT/EXT_gpu_shader5.txt).
Missing compute, FMA, or matched local-size support produces `status=skip`.
One EGL default device and one context are supported. `--mode queues` and
`procs`, and depths outside 1..64, produce explicit skips. Nonzero `--device`
is rejected; use the platform's EGL device selection environment when needed.
`--verify` checks deterministic render/compute readback before timing.

## Timing and comparison limits

The executor uses the standard CLI defaults (300 ms warm-up, five 250 ms
windows) and shared statistics and result serializer. `us_per_submit` counts
completions observed in each window on a continuous stream, without draining
between windows. Warm-up completions and the final drain are excluded. Depth
is a ring of N GL fences; each job is flushed, and its fence must complete
before the slot is reused. The default depths match Vulkan: 1,2,4,8,16,32,64.

The optional timed pass has three windows and at most 100 ms warm-up, matching
Vulkan. Desktop GL uses timestamp queries; GLES uses
`EXT_disjoint_timer_query` when available. Unsupported timers omit the timed
pass with `timed_pass_skip`; a disjoint timer fails the configuration. The
untimed pass has no queries. Compare `us_per_submit` between drivers, keeping
instrumented metrics separate.

GL issues work immediately: `record_cpu_us` covers GL draw/dispatch and state
calls, while `submit_cpu_us` covers fence insertion plus flush. Vulkan records
command buffers and then submits them. These CPU metrics describe different
API scopes. GL also imposes its own ordering and resource hazard handling;
source equivalence does not establish equal native shader code, identical
render-pass boundaries, or identical scheduling. The fill workload is already
blended, so use `opaque` versus `fill` to examine blend cost. The `opaque` rate counts requested layers; hidden-surface removal may eliminate
covered fragments. `texture` retains blending so all layers contribute. The minimal
texture test measures one cached lookup, not a texture bandwidth sweep.

## Paired native-versus-Zink commands

Run from the same checkout and Mesa installation. Select the intended Vulkan
ICD for Zink with `VK_DRIVER_FILES` when several ICDs are installed. Leave
`LIBGL_ALWAYS_SOFTWARE`, `GALLIUM_DRIVER`, and Mesa version overrides unset
for hardware measurements. Keep all API, size, depth, and timing arguments
identical between the two driver runs.

```sh
# Writes new directories asahi/ and zink/ under a new output directory.
tools/run-gl-pair.sh gl-pair
python3 tools/compare.py gl-pair/asahi gl-pair/zink \
  --labels native-GL,Zink --metric us_per_submit --out gl-pair/comparison.md
```

The runner records driver identity and runs the nine workloads above, including
all four heavy-render kinds at 1080p, at depths 1,4,16. It uses explicit standard
warm-up and timing settings. Results and stderr are retained even on failure.
For GLES, use `GB_GL_GLES=1 tools/run-gl-pair.sh gl-pair-gles` and compare only
with other GLES records. Functional CPU results are smoke evidence, not GPU
performance measurements.
