# GPU dashboard

Production is [gpu.aurorasilicon.org](https://gpu.aurorasilicon.org), hosted on Cloudflare Pages. Site source and the public benchmark feed live on the `gpu-dashboard` branch of this repository. The dashboard follows the visitor's system appearance and uses the official [Aurora Silicon mark](https://aurorasilicon.org/brand/aurora-silicon-mark.svg).

## Update benchmark results

Start with the latest completed run in the M5 Pro, Neo or M4 mini development thread. Use its saved gpubench `meta.json` and `results.jsonl` as evidence. Preserve the measured parameters, kernel, driver and suite revisions. For a focused run, record its scope with `--note` and leave unknown revisions unset. Keep internal thread links in the workspace. Use `--source` to attach a public run record or measurement receipt.

```sh
python3 dashboard/import-record.py /path/to/run
python3 dashboard/import-record.py /path/to/gl-run --api gl-native
python3 dashboard/import-record.py /path/to/zink-run --api gl-zink
git add dashboard/site/data/benchmarks.json
git commit -s -m "dashboard: publish GPU benchmark results"
git push origin gpu-dashboard
```

The importer replaces the entry for that device, API and UTC timestamp. It projects medians, percentiles, sample counts and record identifiers from the inputs. The feed omits hostnames, environment variables, system logs and raw samples. Focused runs extracted from mixed experiment logs retain the captured result rows and use the artifact capture time if the start time is unavailable. The SHA-256 covers the imported `results.jsonl`, including when it was extracted from an experiment log.

The site reads the public GitHub feed on page load, so data-only commits appear after GitHub's cache refresh without a Pages upload. A copy of the feed ships with each deployment. Future imports require the saved artifacts from a device thread.

The dashboard has separate Vulkan, OpenGL, Direct3D 11 and Direct3D 12 views. OpenGL shows one backend at a time. Direct3D measurements are collected on Linux through DXVK or vkd3d-proton and compare with native Metal only where the device, workload parameters, depth, mode, and primary metric match. Each API has its own scoreboard. The dashboard groups results by workload. Each section charts selected cases and includes a table of all results for that group. Average speed is the geometric mean across all matched cases in the current chip, API and search scope. A coverage chart appears when the selected results have no valid Metal ratios. Filters, open tables, and the workload inspector are stored in the URL so shared links restore the view.

The comparison table selects the newest run with a recorded, clean suite revision per chip, API and configuration. It counts each paired depth once for each Linux API. The comparison uses the primary metric. Duplicate keys, failures and skipped results go to Unmatched. The workload chart selects one run per API per workload. Readers can select an older run from the Run Records section. Metal-relative values require matching device, lane, test, params, mode, depth, metric, unit and improvement direction. Runs with an unknown or dirty suite revision remain visible in measured mode. The table retains duplicate keys, skipped cases and failures. The chart omits those measurements.

## Deploy the site

Bump the version query on changed CSS and JavaScript URLs before deploying UI changes. This lets browsers pick up the new files on the custom domain. From this repository's `gpu-dashboard` branch, run:

```sh
cd dashboard
npx --yes wrangler@4.166.0 pages deploy --branch gpu-dashboard
```

Run the matching and ratio checks with `node --test dashboard/comparison.test.mjs`.

To serve a local copy:

```sh
python3 -m http.server 8891 --bind 127.0.0.1 --directory dashboard/site
```

The local copy also reads the live feed when reachable. Start offline to inspect the bundled snapshot on its own.
