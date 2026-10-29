# gpubench standard suite

A small, fixed set of GPU benchmarks for the Linux driver stack (Mesa hk, the
drm/asahi kernel driver and the GPU firmware interface), each with a Vulkan
executor (Linux) and a Metal executor (macOS) that do the same work with the
same dimensions and the same timing method.

It is not a conformance suite and not a shader compiler test. Shaders are
deliberately trivial and are checked to compile to near-identical G17P code on
both sides (see `FAIRNESS.md`). What differs between the two columns of a
comparison is the submission, completion, scheduling and memory-management
path: that is what the suite measures.

## Tests

| Test | What it does per submission | What it isolates |
| --- | --- | --- |
| `submit.empty` | one empty command buffer | submit ioctl, fence/timeline signal, completion IRQ path, CPU wake-up |
| `compute.light` | 1 workgroup x 32 threads, one store each | per-dispatch kernel/firmware overhead, round-trip latency |
| `compute.heavy` | `kind=alu`: 1024 x 256 threads, 8 FMA chains x 512 iterations (2.1 GFLOP); `kind=bw`: 64 MiB read + 64 MiB write (float4) | scheduling and GPU feed under load; throughput vs depth |
| `render.light` | 64x64 RGBA8 pass, load-op clear, `kind=tri` 1 triangle or `kind=clear` none | render-pass / tiler / firmware render path, render pairs |
| `render.heavy` | `kind=fill`: 16 blended full-screen layers at 720p/1080p/2160p; `kind=geom`: 262144 small opaque triangles at 1080p | fill and geometry throughput, render scheduling at depth |
| `render.tvb` | 1080p pass with N long thin triangles (8K/32K/128K) that cross many tiles; `dim=N` renders into an N x N target instead (Vulkan) | tiler memory (TVB) overflow, partial renders, heap growth; cold first-pass cost |
| `sync.signal` | `kind=host`: CPU signals a semaphore a parked submission waits on; `kind=queue`: two queues ping-pong a semaphore (`sema=timeline` or `binary`) | wait-before-signal path, cross-queue dependency latency |
| `mem.buffer` | create+allocate+bind, map, touch every 16 KiB page, unmap, free; plus a full alloc -> GPU fill -> wait -> free cycle; 4 KiB .. 64 MiB | GEM/VM bind, mmap, page-fault and free path |
| `mem.image` | same for 2D RGBA8 images 256^2 .. 4096^2, cycle = alloc -> clear pass -> wait -> free | image allocation, layout and first-use path |
| `copy.bw` | 64 MiB `kind=buf2buf`, `buf2img` (4096x4096 RGBA8) or `fill` | copy/blit bandwidth |
| `diag.fill` | `vkCmdFillBuffer` / `fillBuffer` of `size_mib` (1..1024, default 64) MiB; `init=1` fills the buffer once before timing, `storage=private\|shared` picks device-local or host-visible memory | fill bandwidth against first-touch page cost and memory placement (diagnostic, not in the plan) |
| `concurrent.cr` | `mode=queues`: compute.heavy and render.heavy on two queues, alone and together; `mode=interleave`: compute.light and render.light alternating on one queue | compute/render overlap; render<->compute switch cost |
| `idle.wake` | one compute.light submission after `gap_ms` (2, 20, 200) of GPU idle, then one immediately after | GPU power-down / wake-up path in firmware and kernel (`wake_cost_us`) |
| `query` | compute.light with start/end timestamps (`kind=timestamp`) or render.light with an occlusion query (`kind=occlusion`), plus result readback; base and query windows alternate in ABBA order, each after its own warm-up | query/timestamp overhead and readback |

Present/swapchain is not covered yet (headless first).

## Depth

"Depth" is submission concurrency: how many submissions are in flight at
once. It is always the *total* number in flight, so every mode is comparable
at the same N.

- `mode=queue` (the default axis, N = 1, 2, 4, 8, 16, 32, 64): one queue, a
  ring of N slots. Vulkan: one `VkQueue`, N command pools/buffers, one
  timeline semaphore; submission k signals value k, and before submitting k
  the CPU waits for value k-N. Metal: one `MTLCommandQueue`; before committing
  command buffer k the CPU calls `waitUntilCompleted` on command buffer k-N.
- `mode=queues` (N = 2, 4, 8): N queues in one process, one thread per queue,
  each with 1 submission in flight. Vulkan: N `VkQueue`s of the family (hk
  gives each its own kernel queue and firmware context); Metal: N
  `MTLCommandQueue`s. Skipped when the device has fewer queues than N.
- `mode=procs` (N = 2, 4, 8, 16): N processes, each with one queue and 1
  submission in flight. Each process has its own DRM file, VM and firmware
  contexts. Children are started one at a time (each finishes its device setup
  and warm-up before the next starts), then all start the measured phase at
  the same instant.

Submissions are independent: no barriers or semaphores between them (Vulkan),
and every resource is created with untracked hazard tracking (Metal), so
neither driver adds dependencies the other does not.

Each submission is recorded fresh on both sides (Vulkan: reset the slot's
command pool, record, submit; Metal: new command buffer with unretained
references, encode, commit), because Metal has no reusable command buffers.

## Timing method (both executors)

- **Warm-up:** `--warmup-ms` (default 300) of the same stream, discarded.
- **Runs:** `--runs` (default 5) consecutive windows of `--run-ms` (default
  250) each, measured on a continuous stream (no drain between windows).
  Completions are attributed to the window in which the CPU observes them.
- **Wall-clock metrics** (untimed pass: no timestamps at all):
  - `us_per_submit`: the time from the window's first to its last
    completion, divided by the completions after the first. A window with a
    single completion uses window length / 1. The primary metric of every
    submission test. Its inverse is the submission rate. (Records made before
    this method divided the window length by its whole number of completions,
    which quantizes slow rows; compare only records made with the same
    method.)
  - `latency_us`: per submission, from just before `vkQueueSubmit`/`commit`
    to the moment the CPU wait for it returns. At depth 1 this is the round
    trip. At depth N it includes queueing (about N x `us_per_submit`).
  - `submit_cpu_us`: time inside `vkQueueSubmit` / `commit`.
  - `record_cpu_us`: time to record/encode the command buffer.
- **GPU metrics** (separate timed pass, `--timestamps on`, default on; 3
  windows): Vulkan writes a timestamp at the start and the end of each command
  buffer (host query reset, no in-stream reset commands); Metal reads
  `GPUStartTime`/`GPUEndTime` of each command buffer.
  - `gpu_us`: end - start per submission, where start is the later of the
    submission's own start stamp and the previous submission's end. Vulkan
    lets a driver write the top-of-pipe stamp as soon as the queue reaches it,
    so at depth >= 2 the raw start can precede earlier work finishing.
  - `gpu_gap_us`: start of submission k - end of k-1 on the same queue (GPU
    idle between back-to-back submissions; meaningful at depth >= 2). A
    negative value means the start stamp was written before k-1 finished.
  - `us_per_submit_timed`: wall-clock rate during the timed pass. On hk
    timestamps ride on command retirement, so the difference from
    `us_per_submit` is itself a measured cost.
- Percentiles: `median`, `p5` and `p95` are taken over the per-window values
  for rate metrics and over all individual samples for per-submission
  metrics; `n` is the number of values.
- All CPU times use the monotonic clock (`CLOCK_MONOTONIC` /
  `mach_absolute_time`).

## API equivalents and reading notes

- **Semaphores:** Vulkan timeline and binary semaphores have no exact Metal
  counterpart. `sync.signal` uses `MTLSharedEvent` for host signals and
  `MTLEvent` for queue-to-queue signals, for both `sema` values (the result
  carries `metal_sync`).
- **Timestamps:**
  - The `query kind=timestamp` test uses `vkCmdWriteTimestamp2` on Vulkan and
    stage-boundary counter sampling (`MTLCounterSampleBuffer`) on Metal.
  - The timed pass of every other test uses command-buffer start/end
    timestamps on Vulkan, and `GPUStartTime`/`GPUEndTime` on Metal.
  - Stock Mesa 26.2 hk on G14 does not bracket a compute dispatch with
    start/end timestamps (`gpu_us` stays around 24 us whatever the work);
    the tip tree does. Check that `gpu_us` tracks the work before trusting it.
- **Occlusion:** a non-precise occlusion query on Vulkan; the boolean
  visibility result mode on Metal.
- **Fills:** both backends write repeated `0x3f` bytes. Vulkan uses the
  32-bit pattern `0x3f3f3f3f` to match Metal's byte-valued `fillBuffer`.
- **Copies and fills:** both are driver-internal operations (hk uses meta
  compute kernels), so `copy.bw` measures the driver's selected implementation,
  including internal shader codegen.
- **Empty submissions:** `submit.empty` measures what the API does with an
  empty command buffer. hk completes it in userspace without a kernel job
  (about 1 us round trip on the host). Use `compute.light` for the cheapest
  submission that reaches the kernel and firmware.
- **Latency at depth N:** `latency_us` at depth N includes about N - 1
  queued submissions ahead of each one. Compare round-trip latency at depth 1,
  and throughput (`us_per_submit`) at depth.

## Layout

- `shaders/`: GLSL (compiled to SPIR-V at build time) and the MSL with the
  same algorithm. One file pair per shader.
- `vk/`: Vulkan executor `gpubench-vk` (C11, CMake, Vulkan loader + headers,
  glslang). Static libraries `gb_common` (CLI, stats, JSON, timing, child
  processes) and `gb_vk` (instance/device/queue, buffers, images, pipelines,
  the submission-stream engine) are exported for other lanes.
- `metal/`: Metal executor `gpubench-mtl` (Objective-C, one binary), built on
  a Mac with `tools/build-macos.sh`.
- `plan.txt`: the standard run plan (see below).
- `FAIRNESS.md`: compiled-code comparison per shader.

## Executor interface (common to `suite/` and `rt/`)

Every executor is a command-line binary. The Vulkan one is named
`gpubench-<lane>-vk` and the Metal one `gpubench-<lane>-mtl`, except the
standard suite's, which are `gpubench-vk` and `gpubench-mtl`.

```
<exe> --info                       print one JSON object: device and driver identity
<exe> --list                       print the test names, one per line
<exe> --test NAME [options]        run one test, print one result line per configuration
```

Options every executor accepts (unknown `--param` keys are an error):

| Option | Meaning |
| --- | --- |
| `--depths 1,2,4` | depth list; one result per depth (default: the test's own) |
| `--mode queue\|queues\|procs` | concurrency mode (default `queue`) |
| `--param key=value` | test parameter, repeatable (e.g. `kind=alu`, `res=1080p`) |
| `--warmup-ms N` | warm-up per configuration |
| `--runs N` | measured windows |
| `--run-ms N` | window length |
| `--timestamps on\|off` | run the timed pass |
| `--device N` | device index (Vulkan) |

Output: each configuration prints exactly one line on stdout that is a JSON
object with `"schema": "gpubench-result/1"` (see `records/SCHEMA.md`).
Anything else goes to stderr. A configuration the device cannot run prints a
result with `"status": "skip"` and a `"reason"`; a configuration that fails
prints `"status": "fail"` and an `"error"`. The exit status is non-zero only
when the executor could not run at all.

## Plans and the runner

`suite/plan.txt` (and `rt/plan.txt` when it exists) list one executor
invocation per line:

```
# chunk   lane    test            arguments
core-a    suite   compute.light   --depths 1,2,4,8,16,32,64
```

`lane` selects the executor (`suite` -> `gpubench-vk`/`gpubench-mtl`, `rt`
-> `gpubench-rt-vk`/`gpubench-rt-mtl`). `chunk` groups lines into runs of a
couple of minutes so that a locked GPU run on Neo stays short.
`tools/run-suite.sh` (used by `tools/run-linux.sh` and `tools/run-macos.sh`)
reads every plan, runs the selected chunks and writes one record directory.
It needs only a POSIX shell, so it runs on a stock macOS install.

## Building

Linux (Vulkan):

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      [-DGB_VULKAN_INCLUDE=/path/to/Vulkan-Headers/include]
ninja -C build            # -> build/bin/gpubench-vk (and rt executors)
```

macOS (Metal), on a Mac with the command-line tools:

```
tools/build-macos.sh      # -> bin/macos/gpubench-mtl (and rt executors)
```

See `records/README.md` for running the suite and adding a record.
