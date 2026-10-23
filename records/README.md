# Benchmark records

Each record contains `meta.json` and `results.jsonl`, following [SCHEMA.md](SCHEMA.md). A result identifies the workload, parameters, concurrency mode, depth, run controls, status, and metrics. The primary statistic is the median across measurement windows where applicable. Unsupported configurations remain in the record with a reason.

## macOS records

- [Neo, macOS 26.6.2 / 25G83](neo/macos/20261023-25G83/): 111 standard configurations and 98 ray-tracing configurations. Two RT pipeline configurations are unsupported by the Metal executor. The power/thermal trace spans the full combined run.
- [M5 Pro, macOS 27.2 / 26B5086k](m5-pro/macos/20261023-26B5086k/): 82 standard configurations from `core-a`, `core-b`, `multi`, and `feat`. No RT sweep was performed. Privileged power sampling was unavailable; the metadata retains the available thermal state and records that limitation.

These are macOS measurements. Linux records are intentionally absent from this initial publication.

## Provenance

The measurements predate this public repository. [SOURCES.json](SOURCES.json) records the development revisions and original source-file hashes. Standard-suite algorithms are taken from revision `311a5e6`; the RT implementation is taken from `01efdc2`, the source revision recorded for the Neo run. The change from Neo's standard revision `3a13953` to `311a5e6` moves the `metal_sync` label outside the metrics object; it does not change the measured workload.

Publication adds licensing, portable build/runner cleanup, and documentation. Hostnames are replaced with device labels. Raw `results.jsonl` values are byte-identical to the retained originals; original metadata hashes and published record hashes are in SOURCES.json. Historical records may contain the `metal_sync` string inside `metrics`; comparison tools handle this representation. The `gpu_family` field reports a supported Metal family, not necessarily the highest family supported by the device.

## Comparing records

```sh
python3 tools/compare.py RECORD_A RECORD_B --labels A,B --out comparison.md
```

Only matching `(lane, test, parameters, mode, depth)` configurations join. Do not treat a missing configuration as zero. Differences in hardware, OS version, clocks, thermal state, and compiler output must be considered before attributing a gap to a driver path. GPU timestamp passes can change scheduling and throughput; use the untimed primary metric for submission throughput and inspect timed results separately.
