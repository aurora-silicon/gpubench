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
    --base 0539df01ea a.json --candidate 1eae3d92d5 b.json --traces j700.json
```

The runner checks the Mesa Git revision reported by both APIs and records device identity, executable hashes and kernel release. Use a fresh output directory. The 15-minute runtime limit returns FLAG on expiry. It retains the executor logs, raw JSON rows, screenshots, CRCs, block deltas and elapsed seconds in the run directory. Omit `--traces` for an explicitly gpubench-only row. Use `--visual-only` on a frequency-constrained machine. Record power, temperature and GPU health outside the runner.

## Capture manifest

Install Pillow and stage the appropriate replayer. Vulkan uses `gfxrecon-replay --wsi headless` with BMP readback. GL uses a Waffle-enabled `eglretrace --headless` with `WAFFLE_PLATFORM=surfaceless_egl`. Both run with desktop display variables removed. No game or audio process is launched.

The manifest binds goldens to the device name from `gpubench-vk --info`. Paths are relative to the manifest. Use the exact file hashes and frame numbers from the accepted corpus. For Vulkan, index replay frames starting at 1. For GL, index calls, typically the swap call. Capture consecutive frames from the same replay for temporal defects. To score a small flickering region in addition to the full image, add `"roi": [left, top, right, bottom]` to the frame entry in pixel coordinates (right and bottom exclusive).

```json
{
  "device": "Apple A18 Pro (G17P B1)",
  "captures": [{
    "file": "scene.gfxr",
    "sha256": "<capture SHA-256>",
    "frames": {
      "2": {"file": "g17p/scene-2.png", "sha256": "<image SHA-256>"}
    }
  }]
}
```

Small public fixtures for initial enrollment are [GFXReconstruct's triangle](https://github.com/LunarG/gfxreconstruct/blob/4401228c791b11fb7ea15b93795cbd1e938307c0/test/known_good/triangle.gfxr) (frames 1 and 2) and [apitrace's GLSL triangle](https://github.com/apitrace/apitrace-tests/blob/7b1b37ed02c3c1de9a7979129f7b14d041847d18/traces/tri_glsl.trace) (call 6935). Produce and inspect goldens on a known-good baseline for each GPU. Keep the capture, selected calls, independent reference and approved golden hashes together.

## Interpreting a result

| Exit | Meaning |
| --- | --- |
| 0 PASS | All checks in the requested scope passed. |
| 1 FAIL | A candidate readback or image differs outside tolerance, or every timing block exceeds the slowdown threshold. Repeat the case to confirm. |
| 2 FLAG | A required measurement is missing, the baseline is invalid, the tolerant readback changed CRC, or timing is inconclusive. |

GL postfx uses `--verify` at 1080p with one sample, the configuration that reproduced the G13/G14 multi-pass regression. The Vulkan postfx case covers 4× MSAA. Vulkan checks its output internally. Image comparison requires identical dimensions and allows at most 0.1% of pixels with a luma-weighted absolute RGB difference above 8 on the 8-bit scale. PNG and BMP files are compared as decoded RGB. A baseline that misses its golden flags the run.

For each case the runner alternates ABBA and BAAB twice, giving eight process medians per arm. Each process measures five 450 ms untimed submission windows after 250 ms of warm-up. Positive deltas mean the candidate took longer. The threshold is the larger of 5% and three times the higher arm's scaled median absolute deviation divided by its median. A median block slowdown above 5% or within-arm noise above 5% flags the case when a failure is inconclusive. Calibrate on repeated A/A runs before making the pilot a required merge check.

## Isolating a failed batch

```sh
python3 tools/bisect-batch.py /src/mesa "$base" "$batch_tip" -- /path/to/stage-and-run
```

The caller supplies a machine-specific adapter taking two Git SHAs (baseline and candidate prefix). It stages or boots the builds, invokes the row in a fresh evidence directory and returns 0 for pass, 1 for a reproduced regression, 2 for an inconclusive row. Map build, deployment and infrastructure failures to 2. Keep the oracle and its fingerprint fixed during bisection so different failures cannot be confused.

The wrapper runs the tip once on the successful path. On failure it repeats the tip, verifies A/A, bisects the immutable first-parent history and repeats the adjacent passing and failing prefixes. It invokes the adapter without modifying the checkout. An inconclusive midpoint preserves the bracket. As with Git bisect, interpret the result as a transition under the tested ordering. Interacting or intermittent failures need separate attribution.
