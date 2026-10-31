# Record schema

A record is one run of one or more plan chunks on one device, OS and driver
build. It is a directory:

```
records/<device>/<os>/<date>-<id>/
  meta.json        identity of the device, OS, kernel/driver and suite build
  results.jsonl    one result object per line (gpubench-result/1)
  log.txt          executor stderr (optional, trimmed)
```

- `<device>`: short board name, e.g. `neo`, `m5`, `m2`.
- `<os>`: `linux` or `macos`.
- `<date>`: UTC date `YYYYMMDD`.
- `<id>`: Linux `<kernel release>-<mesa short sha>`, e.g.
  `7.3.0-1-tip-ARCH-bfb7350`; macOS the build, e.g. `25A123`. A suffix
  (`-b`, `-c`) is added when there are several runs that day.

## meta.json (`gpubench-record/1`)

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | `"gpubench-record/1"` |
| `device` | string | `neo`, ... |
| `os` | string | `linux` or `macos` |
| `os_version` | string | `/etc/os-release` PRETTY_NAME, or `sw_vers` product version |
| `os_build` | string | macOS build (`sw_vers -buildVersion`), `null` on Linux |
| `kernel` | string | `uname -r` (Linux), `uname -v` (macOS) |
| `kernel_module` | object | Linux: `{"name": "aurora", "build_id": "...", "srcversion": "...", "params": {...}}` (`asahi` on kernels before the rename) |
| `driver` | object | from the executor `--info`: `name`, `info` (driver info string), `version`, `api_version`, `mesa_sha` (Linux) |
| `gpu` | object | `name`, `cores` when known, `family` |
| `firmware` | string | GPU firmware version when visible, else `null` |
| `date_utc` | string | ISO 8601 start time |
| `host` | string | hostname |
| `suite_sha` | string | `git rev-parse --short HEAD` of this repo at build time, `+dirty` when modified |
| `runner` | object | `chunks`, `runs`, `run_ms`, `warmup_ms`, `env` (driver-relevant environment: `VK_DRIVER_FILES`, `MESA_*`, `HK_*`, `AGX_*`) |
| `state_before` / `state_after` | object | thermal and clock snapshot: Linux `thermal` zones, `cpufreq`, GPU devfreq if any; macOS `pmset -g therm`, `powermetrics` GPU residency when `sudo -n` works |
| `health` | object | Linux: GPU-related kernel log lines during the run, desktop alive; macOS: `null` |
| `notes` | string | free text |

## results.jsonl (`gpubench-result/1`)

One object per configuration (test x parameters x mode x depth):

```json
{
  "schema": "gpubench-result/1",
  "lane": "suite",
  "test": "compute.light",
  "params": {"kind": "default"},
  "mode": "queue",
  "depth": 4,
  "status": "ok",
  "warmup_ms": 300, "runs": 5, "run_ms": 250,
  "primary": "us_per_submit",
  "metrics": {
    "us_per_submit": {"unit": "us", "better": "lower",
                      "median": 21.4, "p5": 21.1, "p95": 22.0, "n": 5,
                      "samples": [21.3, 21.4, 21.1, 22.0, 21.6]},
    "latency_us":    {"unit": "us", "better": "lower",
                      "median": 85.0, "p5": 80.2, "p95": 97.3, "n": 58210},
    "gpu_us":        {"unit": "us", "better": "lower", "median": 3.1, "p5": 2.9, "p95": 3.6, "n": 31000}
  },
  "counts": {"submits": 58210}
}
```

Rules:

- `params` holds only parameters that change the work (never run controls
  such as `runs`). Two results describe the same benchmark exactly when
  `lane`, `test`, `params`, `mode` and `depth` are equal; `tools/compare.py`
  joins Linux and macOS on that key.
- `depth` is the total number of submissions in flight (see
  `suite/README.md`); tests without a depth axis use `1`.
- Every metric has `unit`, `better` (`lower` or `higher`), `median`, `p5`,
  `p95` and `n`. Rate metrics (one value per window) also carry `samples`.
  Per-submission metrics are pooled over all submissions of all windows and
  carry no `samples`.
- `primary` names the metric used for headline comparisons.
- `status` is `ok`, `skip` (with `reason`) or `fail` (with `error`).
- Extra fields are allowed (`counts`, `info`, test-specific detail); readers
  ignore what they do not know.

Standard metric names (use these where they apply, so tables line up):

| Metric | Unit | Meaning |
| --- | --- | --- |
| `us_per_submit` | us | wall-clock time per completed submission (untimed pass) |
| `latency_us` | us | submit -> CPU-observed completion, per submission |
| `submit_cpu_us` | us | time inside the submit/commit call |
| `record_cpu_us` | us | time to record/encode one submission |
| `gpu_us` | us | GPU start -> end per submission (timed pass) |
| `gpu_gap_us` | us | GPU idle between consecutive submissions on a queue |
| `us_per_submit_timed` | us | `us_per_submit` measured during the timed pass |
| `gbps` | GB/s | bytes per second, 1 GB = 1e9 bytes |
| `gflops` | GFLOP/s | arithmetic rate |
| `gpix_s` | Gpix/s | shaded/blended pixels per second |
| `mtri_s` | Mtri/s | triangles per second |
| `*_us` | us | named per-operation CPU times (`alloc_us`, `map_us`, `free_us`, ...) |
