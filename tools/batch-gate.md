# Compare two Mesa builds on one Apple GPU

Use the same `gpubench-gl` and `gpubench-vk` binaries for both arms. Stage the baseline and candidate side by side. Give each its own EGL vendor JSON, Vulkan ICD JSON (both with absolute library paths), DRI library directory and shader cache. Each environment file is a JSON object of string variables, for example:

```json
{
  "LD_LIBRARY_PATH": "/stage/a/usr/lib64",
  "LIBGL_DRIVERS_PATH": "/stage/a/usr/lib64/dri",
  "GBM_BACKENDS_PATH": "/stage/a/usr/lib64/gbm",
  "__EGL_VENDOR_LIBRARY_FILENAMES": "/stage/a/egl.json",
  "VK_DRIVER_FILES": "/stage/a/icd.json",
  "MESA_LOADER_DRIVER_OVERRIDE": "asahi",
  "MESA_SHADER_CACHE_DIR": "/stage/a/cache"
}
```

```sh
python3 tools/batch-gate.py --bin /opt/gpubench/bin --out run-001 \
    --base 25f15c435b61 a.json --candidate 8f515f859e15 b.json --traces corpus/j700.json
```

The runner checks the Mesa Git revision reported by both APIs and records device identity, executable hashes and kernel release. Use a fresh output directory. The 15-minute runtime limit returns FLAG on expiry. It retains the executor logs, raw JSON rows, screenshots, CRCs, block deltas and elapsed seconds in the run directory. Omit `--traces` for an explicitly gpubench-only row. Use `--visual-only` on a frequency-constrained machine. Record power and temperature during the run. The runner follows one kernel-journal cursor through all GPU invocations and drains it after the last step. It retains fault messages from the GPU device or driver with kernel timestamps, arm and step. A fault delivered between invocations is attributed to the preceding invocation. A confirmed candidate GPU fault takes precedence over a workload timeout. It needs journal read access directly or through noninteractive sudo. An unreadable journal makes the result inconclusive. Before each GPU step and after the last, the runner reads the DRM clients of other processes from `/proc/<pid>/fdinfo`. It counts each `drm-client-id` once, however many processes inherited it, and sums its `drm-engine-*` time since that client was last seen. If other clients used more than 1% of the wall time between two checks, the result flags. A gap shorter than a second is judged against one second. A benchmark that outlived its hardware lock is caught this way. Some processes cannot be read as the runner's user: other users' processes, and a compositor with file capabilities such as `kwin_wayland`. The runner reads those through noninteractive sudo (`batch-gate.py --gpu-clients`, read-only) at the start and the end, and holds their total use over the run to the same limit. Without sudo, or if the helper's output does not parse, the result flags, as it does for an unreadable journal. If such a client is gone at the end and its process has ended, the result flags, since its use after the first read is unknown. A client closed by a process that is still running, judged by its start time in `/proc/<pid>/stat`, is only recorded. An unparsable engine time flags. `summary.json` lists the clients at start, every interval in which another client used the GPU, started or exited, and the time spent scanning. A driver without engine keys shows no time, so it cannot flag, and a job that starts and ends between two checks is not seen.

## Capture manifest

Install Pillow and stage the appropriate replayer. Vulkan uses `gfxrecon-replay --wsi headless` with BMP readback. GL uses a Waffle-enabled `eglretrace --headless` with `WAFFLE_PLATFORM=surfaceless_egl`. Both run with desktop display variables removed. No game or audio process is launched.

The manifest binds goldens to the device name from `gpubench-vk --info`. Paths are relative to the manifest. Use the exact file hashes and frame numbers from the accepted corpus. For Vulkan, index replay frames starting at 1. For GL, index calls, typically the swap call. Capture consecutive frames from the same replay for temporal defects. To score a small flickering region in addition to the full image, add `"roi": [left, top, right, bottom]` to the frame entry in pixel coordinates (right and bottom exclusive).

```json
{
  "device": "Apple A18 Pro (G17P B1)",
  "captures": [{
    "file": "scene.gfxr",
    "sha256": "<capture SHA-256>",
    "replayer": {"version": "<whitespace-normalized --version output>", "sha256": "<binary SHA-256>"},
    "frames": {
      "2": {"file": "g17p/scene-2.png", "sha256": "<image SHA-256>"}
    }
  }]
}
```

The supplied J700 manifest selects Vulkan frames 1 and 2 and GL call 6935. Copy the goldens and fetch the pinned upstream captures once. The runner verifies their hashes before replay.

```sh
cp -R tools/batch-gate corpus
curl -fL https://raw.githubusercontent.com/LunarG/gfxreconstruct/4401228c791b11fb7ea15b93795cbd1e938307c0/test/known_good/triangle.gfxr -o corpus/triangle.gfxr
curl -fL https://raw.githubusercontent.com/apitrace/apitrace-tests/7b1b37ed02c3c1de9a7979129f7b14d041847d18/traces/tri_glsl.trace -o corpus/tri_glsl.trace
```

The J700 goldens were enrolled on Mesa `25f15c435b61` and checked against software replay. The Vulkan images matched within tolerance. The GL reference differed above tolerance on 224 of 62,500 pixels, all along the two sloped triangle edges. The J700 baseline reproduced its own reference. Keep independent goldens for other GPUs and review changes to an existing golden separately.

## Interpreting a result

| Exit | Meaning |
| --- | --- |
| 0 PASS | All checks in the requested scope passed. |
| 1 FAIL | The candidate crashes after its base control passed, produces invalid output, logs a new GPU fault, or exceeds the slowdown threshold in every timing block. Repeat the case to confirm. |
| 2 FLAG | A required measurement is missing, the baseline is invalid, the tolerant readback changed CRC, another process used the GPU during the run, or timing is inconclusive. |

GL postfx uses `--verify` at 1080p with one sample, the configuration that reproduced the G13/G14 multi-pass regression. The Vulkan postfx case covers 4× MSAA. Vulkan checks its output internally. Image comparison requires identical dimensions and allows at most 0.1% of pixels with a difference above 8 in any RGB channel on the 8-bit scale. PNG and BMP files are compared as decoded RGB. A baseline that misses its golden flags the run.

For each case the runner alternates ABBA and BAAB twice, giving eight process medians per arm. Each process measures five 450 ms untimed submission windows after 250 ms of warm-up. Positive deltas mean the candidate took longer. Noise is the relative standard deviation of one process, estimated from the third difference of each block's four processes in run order (weights −1, 3, −3, 1). That difference cancels the arm offset and any linear or quadratic drift shared by both arms, such as the GPU warming up. On J700, GL postfx slows by about 25% over its first 40 s in every run. Its pooled spread read 2–13% while the balanced block changes stayed within 2.2%. The threshold is the larger of 5% and three times the noise. A median block slowdown above 5% or noise above 5% flags the case when a failure is inconclusive. A disturbance that repeats every seven to nine processes lines up with the arm pattern and reads as a steady slowdown with little noise. It can fail an A/A run, and no estimator built from these blocks can tell it from a real slowdown. `bisect-batch.py` repeats the tip and checks A/A before bisecting. Calibrate on repeated A/A runs before making the pilot a required merge check.

## Isolating a failed batch

```sh
python3 tools/bisect-batch.py /src/mesa "$base" "$batch_tip" -- /path/to/stage-and-run
```

The caller supplies a machine-specific adapter taking two Git SHAs (baseline and candidate prefix). It stages or boots the builds, invokes the row in a fresh evidence directory and returns 0 for pass, 1 for a reproduced regression, 2 for an inconclusive row. Map build, deployment and infrastructure failures to 2. Keep the oracle and its fingerprint fixed during bisection so different failures cannot be confused.

The wrapper runs the tip once on the successful path. On failure it repeats the tip, verifies A/A, bisects the immutable first-parent history and repeats the adjacent passing and failing prefixes. It invokes the adapter without modifying the checkout. The default adapter timeout is 900 seconds per probe. Expiry stops the adapter process group, including its child workloads. An inconclusive midpoint preserves the bracket. As with Git bisect, interpret the result as a transition under the tested ordering. Interacting or intermittent failures need separate attribution.
