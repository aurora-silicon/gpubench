# gpubench stress suite

`gpubench-stress-vk` runs GPU driver stress tests with a pass/fail verdict each. It targets the
failure classes that GPU kernel drivers, firmware interfaces and Vulkan drivers actually hit under
load: lost or reordered cross-queue work, renders that finish but never complete, tiler-buffer
growth, watchdog kills of legitimate long jobs, teardown races, clients that take innocent clients
down, allocation limits, dense synchronisation and power transitions. It is not a conformance
suite and does not replace the Vulkan CTS; it is a short, repeatable gate that can run on a
desktop.

Every test checks its results against a CPU model (checksums, exact pixel values, exact ray hits),
not only "it did not crash". With `--kernel-log`, new GPU error lines in the kernel log during a
test also fail it.

## Build

The stress lane is part of the top-level CMake build (Linux only):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
build/bin/gpubench-stress-vk --list
```

## Run

```sh
gpubench-stress-vk                         # every safe test, about 5 minutes
gpubench-stress-vk --duration 120          # the same in about 2 minutes
gpubench-stress-vk --test xq.order,xq.chain --duration 60
gpubench-stress-vk --kernel-log            # also fail on new GPU kernel-log errors
VK_DRIVER_FILES=/path/to/icd.json gpubench-stress-vk
```

| Option | Meaning |
| --- | --- |
| `--all` | every test except the dangerous ones (the default) |
| `--test NAME[,NAME]` | selected tests (repeatable) |
| `--dangerous` | allow dangerous tests; with `--all`, include them |
| `--duration S` | total time budget, split by test weight. Default: 20 s per unit of weight, at most 300 s (the default set is 300 s). Each test gets at least 4 s |
| `--kernel-log` | check the kernel log around every test (Linux) |
| `--kernel-log-pattern RE` | add an error pattern (POSIX extended regex, case-insensitive; repeatable) |
| `--kernel-log-ignore RE` | ignore matching lines (repeatable) |
| `--kernel-log-no-defaults` | use only the patterns given with `--kernel-log-pattern` |
| `--device N` | Vulkan physical device index (default 0, or `GB_DEVICE`) |
| `--seed N` | random seed; the seed of every run is printed so a failure can be replayed |
| `--verbose` | progress details on stderr |
| `--no-fork` | run tests in the runner process, without the watchdog (for debuggers) |

Each test runs in its own process with its own Vulkan instances and devices, so a crash or a lost
device in one test does not affect the next. A test that does not finish within its budget plus a
setup allowance (the budget again, at least 60 s) is killed and fails as a hang. Helper processes of
a test are in the test's process group and are killed and reaped after it.

Tests use a few hundred MiB of GPU memory, except `mem.limit`, which holds memory up to the
smaller of 60% of the device-local heap and what leaves at least 25% of system memory (and at
least 1.5 GiB) available.

### Output and exit status

stdout carries JSON lines: one `gpubench-stress-run/1` header (device, driver, features, seed),
one `gpubench-stress/1` line per test (`status` pass/fail/skip, `reason`, `iterations`, `checks`,
`errors`, test metrics such as worst latencies, and the kernel-log lines when checked), and one
`gpubench-stress-summary/1` line. stderr carries a readable log.

| Exit | Meaning |
| --- | --- |
| 0 | every selected test passed or was skipped |
| 1 | at least one test failed, including new kernel-log errors with `--kernel-log` |
| 2 | usage error, no usable Vulkan device, or the kernel log could not be read (at start, or for any test when no test failed; such tests carry `kernel_log.read: false` and the summary counts them in `kernel_log_unread`) |
| 3 | a test process could not be killed after its deadline (the GPU or driver is probably wedged); the run was aborted |

A test skips when the device lacks what it needs (two queues, buffer device address, ray query,
sparse residency).

## Tests

| Test | What it does | Protects against |
| --- | --- | --- |
| `xq.order` | 16 producer/consumer pairs per batch on two queues. Both take tickets from one atomic counter and the consumer copies the producer's marker. Modes rotate: timeline semaphores, binary semaphores, wait-before-signal (every consumer submitted before any producer, producers in reverse order) and host signals. Producers spin for a random time. | Cross-queue dependencies that are ignored, satisfied early or lose memory visibility. |
| `xq.chain` | Chains of 2-16 links alternate a render queue and a compute queue. A render link reads the previous result in a fragment shader and writes a new value per pixel. A compute link transforms the buffer in place. 16K values are checked against a CPU model after every chain. | Chained data between render and compute queues. |
| `xq.exit` | Child processes submit a slow producer, a dependent consumer on another queue and a render. They then exit at once, exit right after the wait returns, exit cleanly, or are killed a few milliseconds in. Each must exit within 30 s; a fresh client then checks that the GPU is healthy. | Teardown with cross-queue work in flight. |
| `cr.concurrent` | Render rings (1024x1024 and 256x256, 8 and 16 deep) on two queues beside compute rings on two other queues. Every frame and buffer is checked. | Concurrent render and compute work. |
| `render.first` | Each fresh device's first submission is a heavy 1080p render: 131K triangles plus 8 additive layers, every pixel exact. Light and heavy renders then follow after idle gaps of 0-600 ms. A render slower than 5 s fails as a late or lost completion. | Renders that finish on the GPU while their completion is reported late or never. Cold first-frame paths. |
| `render.tiler` | Each fresh device renders a light frame, then a cold 128K-quad frame, then random sizes, sometimes two frames in flight. Every quad covers one full-width row, so it is binned into every tile of that row. Additive blending makes every pixel equal its row's exact quad count. | Tiler-buffer growth on the first big frame, and fragment work lost after tiling. |
| `compute.long` | Two solo dispatches of about 1.5 s each must complete with the right result. Then 400 ms dispatches run back to back beside a render ring and a compute ring in the same process, with a ray-query client in another process. | Watchdogs that kill legitimate long jobs, or that time out work waiting behind them. |
| `life.churn` | Two lanes of short-lived processes exit after a wait, exit cleanly, exit with work in flight, or are killed mid-work. A third lane creates and destroys devices with 1-4 queues in a loop. Every process does checked work. | Context teardown races and process-exit paths. |
| `life.fault` | One client page-faults the GPU on purpose (buffer-device-address stores to unmapped addresses), several times a second. Another process meanwhile runs checked render and compute rings. The healthy client must never see an error. The faulting one must get a lost device or, where the driver drops such stores, complete; it must never hang. `faults_device_lost` and `faults_ignored` count the outcomes. | A failing client that takes innocent clients down, for example when their work is retired without running. |
| `queue.exhaust` | Opens devices that use every queue, one at a time, until the driver refuses. After each new device, every older one must still complete checked work. A fresh client must work once everything is released. | Queue or context exhaustion that hangs or breaks other contexts instead of refusing cleanly. |
| `mem.limit` | Asks for twice the heap and for more than `maxMemoryAllocationSize`; both must fail promptly with an out-of-memory error. Then fills memory chunk by chunk with GPU writes up to the safe cap and runs checked work while holding it, repeatedly. | Allocation-limit handling, and submissions that fail under memory pressure. |
| `mem.churn` | Allocates hundreds to 2048 small host-visible buffers, each with its own allocation. The CPU writes them, half are unmapped, the GPU copies from all of them and fills part of each, and everything is checked from both sides. Buffers are freed in random order. Ends with 2000 map/unmap cycles. | Buffer-object, mapping and VM churn. |
| `sync.dense` | Random dependency graphs of 384 submissions over four queues, with timeline and binary semaphores and one fence per submission. Each queue's stream is in order, but the streams are merged at random, so many waits are submitted before their signals. Semaphores are recreated regularly. Each node computes its graph depth, so a broken edge shows. Child processes also exit with 64 submissions blocked on a semaphore that is never signalled. | Dense synchronisation, and cancellation of pending work at teardown. |
| `idle.wake` | Checked render and compute bursts after 2, 20 or 200 ms of idle, in random order. A burst must finish within 2 s of waking. | Power-down and wake transitions that lose, delay or corrupt work. |
| `queue.depth` | 128 compute and 64 render submissions in flight on one queue. Then three devices at low, medium and high global priority (when supported) keep every queue busy, with relative priorities 0, 0.5 and 1. | Queue-depth extremes, priority handling and starvation. |
| `sparse.churn` | Binds and unbinds random pages of a 64 MiB sparse buffer to random pages of a memory pool. The bound pages are filled through the sparse buffer and read back both through it and through a plain buffer that aliases the pool. Sizes stay small on purpose. | Sparse binding churn and page-table updates. |
| `rq.stress` | A BLAS of 256 flat quads at moving heights is refit three times and rebuilt the fourth, under a TLAS of 4096 moving instances rebuilt every iteration. 128K rays per dispatch, and every hit's instance, primitive and distance is checked. | Ray-query traversal and acceleration-structure updates. |

### Dangerous tests

These have hung machines or crashed GPU firmware during driver development. `--all` never runs
them; each needs `--dangerous`. Run them only on a machine you can power-cycle remotely.

| Test | What it does | Why it is dangerous |
| --- | --- | --- |
| `cr.pairs` | 4 and then 8 processes are released at the same instant. Each creates its device and renders at once, 20 checked iterations per process. | Simultaneous construction of render contexts in several processes has frozen a whole machine. |
| `compute.starve` | One process issues 1.5 s dispatches back to back. Another keeps four compute queues and a render queue busy. A third starts clients and kills them a few milliseconds into their work. | Starvation behind long dispatches, combined with kills, has crashed GPU firmware: the device is lost until reboot. |
| `mix.wedge` | Five concurrent lanes: process churn, tiler-heavy renders, concurrent render and compute, 1.5 s dispatches with short work, and ray query. | This mix has wedged GPUs and hung machines. |
| `mem.exhaust` | Allocates GPU-filled memory until the driver refuses, then checks a fresh client. | Reaching the real limit can invoke the kernel OOM killer on other processes, such as the desktop. |

## Kernel-log check

`--kernel-log` reads `/dev/kmsg` from its end, or `journalctl -k` after a cursor when
`/dev/kmsg` is not readable. The lines logged while each test runs are collected; a line that
matches an error pattern and no ignore pattern fails that test. If neither source can be read,
the run exits with status 2. The default patterns are generic for Linux DRM drivers:

```
\b(gpu|drm)\b.*(hang|hung|time ?out|timed out|reset|recover|lost|crash|fault|wedged)
\[drm(:[^]]*)?\] \*error\*
\b(amdgpu|radeon|i915|xe|nouveau|msm|freedreno|panfrost|panthor|lima|v3d|vc4|etnaviv|powervr|aurora|asahi)\b.*(error|fail|fault|time ?out|timed out|hang|reset|lost|crash)
firmware.*(crash|hang|time ?out|timed out)
\b(etimedout|device lost)\b
\b(oops|kernel bug|bug:|unable to handle kernel|general protection fault|use-after-free|kasan)
warning: .*(drm|gpu)
```

Add driver-specific patterns with `--kernel-log-pattern`, or replace the defaults with
`--kernel-log-no-defaults`. `life.fault` causes GPU faults on purpose. Its fault-report lines are
counted as `expected_errors` and do not fail it. Recovery, reset, crash, timeout and lost-device
lines still do.
