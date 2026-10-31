# Workload equivalence and comparison limits

The Vulkan and Metal executors implement the same algorithms and dimensions. GLSL and MSL source are included under `shaders/`. Submission depths have the same total number of jobs in flight, and the standard submission tests use independent work without additional inter-submission hazards.

The standard suite reports an untimed pass and a separate GPU-timestamp pass. Instrumentation can change scheduling, query-result publication, and throughput. Compare `us_per_submit` with the same untimed metric on the other system; `us_per_submit_timed` measures the instrumented path. A timestamped GPU gap is not an uninstrumented utilization measurement.

API operations also differ in scope. Vulkan timeline/binary semaphores use Metal events as documented in [README.md](README.md). Empty Vulkan submissions can complete in userspace, so `compute.light` is the minimum useful kernel/firmware submission test. Public API timings do not directly measure kernel CPU time or display scanout.

For a driver comparison, verify matching shader work, compiler output, resource formats, concurrency, and operating conditions. The Vulkan executor exposes pipeline statistics with `--shader-stats`; the Metal executor can retain its own pipeline archives with `--dump-archive DIR`. Compiler differences, register use, and fixed-function setup can affect execution even when source algorithms match.

The current publication contains macOS measurement records. Linux performance comparisons will be published separately after qualification. See [record provenance](../records/README.md).

The optional GL/GLES executor shares render/compute shader algorithms and workload dimensions with Vulkan. GL fence/flush submission and implicit hazard handling have different API scopes from Vulkan command buffers and explicit synchronization. No compiled-code parity is claimed for GL or Zink. See [the GL timing notes](gl/README.md#timing-and-comparison-limits).
