# Gate a Mesa or drm/asahi batch

The coordinator runs on the shared host. Hardware slots, builds, health checks, one-shot boots and restoration belong to each machine lead. Use a spool shared with both leads. The submitting account needs write access to the final PR only when publishing. One invocation addresses the ordered batch, including its exact synthetic merge SHA.

```sh
python3 tools/train-gate.py dispatch --kind mesa --tree "$MESA_REPO" \
  --spool "$SPOOL" --base "$LIVE_TIP" --candidate "$TRAIN_COMMIT" \
  --member "$PR1:$HEAD1" --member "$LAST_PR:$LAST_HEAD" \
  --boards j700 j773g --workers "$LEAD_COMMANDS"
```

The trusted local workers JSON maps each board to a command argv, e.g. `{"j700":["/opt/rows/queue-neo"],"j773g":["/opt/rows/queue-mini"]}`. These are site-owned commands. Each receives two additional arguments, the absolute job directory and board. Dispatch starts both commands, waits up to 900 seconds by default, and posts or edits one line on the last PR. A delayed owner can finish its restore after the wait expires. Run `finish JOB --post` to update the same comment. Use `--no-post` to rehearse. For leads using a message-based queue, replace `dispatch` with `enqueue`, omit `--workers`, send the printed job path to both leads, then call `finish`. The job ID binds the base, candidate, expected tree, ordered PR heads and board set. Publication rechecks the live baseline and all PR heads.

## Mesa lead

Under the existing lease, stage both native Mesa builds and the enrolled corpus. Copy the job's unmodified `request.json` to the Mac and use two overlay environment JSON files as in the batch-gate instructions. Write into a fresh result directory.

```sh
python3 tools/fleet_row.py request.json --board "$BOARD" --arm pair --out pair \
  --base-env a.json --candidate-env b.json --bin "$BENCH_BIN" --traces "$MANIFEST"
```

Copy the completed `pair/` evidence into `JOB/BOARD/`. Keep any existing targeted CTS rows alongside this gate. J516s automatically runs visual only.

## Kernel lead

Dispatch with `--kind kernel --tree "$LINUX_REPO" --base "$BASE_KERNEL" --candidate "$FLEET_CANDIDATE" --mesa "$FIXED_MESA_FULL_SHA"`. Supply the ordered kernel PRs as for Mesa. Use the lead's normal guarded one-shot path. Each build manifest is produced **before boot** from the exact Git SHA and the GNU build ID of its vmlinux (`readelf -n vmlinux`): `{"commit":"<40 hex>","build_id":"<hex>"}`. The collector checks the ID against the running kernel's ELF notes.

1. Boot the base build. It is a one-shot boot unless the Mac's stock kernel is exactly that build. Run `fleet_row.py request.json --board BOARD --arm base --kernel-identity base.json --base-env mesa.json --mesa-libs GALLIUM_SO BOREALIS_SO --bin BIN --traces MANIFEST --out base`.
2. Have the lead stage and boot the candidate with its existing dead-man and saved stock entry. On that boot repeat the command with `--arm candidate --kernel-identity candidate.json --out candidate`. Use the exact same Mesa libraries and corpus.
3. Boot the base build again and repeat the command with `--arm restored --kernel-identity base.json --out restored`, with the same measurement arguments. Base, candidate and restored then form an A, B, A crossover. The candidate is compared with the mean of the two base boots, which cancels a steady drift, and the base-to-restored change is the floor; no calibration file is needed. Without the measurement arguments the restored arm only checks identity. The board's reviewed `calibration.json` then supplies the floor, from two same-kernel stock enrollment boots: `python3 tools/batch-enrollment/calibrate.py boot1 boot2 > calibration.json`. Reviewed calibrations live in `tools/batch-gate/calibration/BOARD.json`, next to the board's replay manifest. Copy the three directories (and any calibration) into `JOB/BOARD/`, then return the Mac to its stock entry. Retain power/clock and library evidence from all boots.

The collector records the three boot IDs and verifies stock was restored. A candidate-only output mismatch can fail the batch. A cross-boot timing slowdown requires a repeat crossover under the lead. Missing or noisy calibration leaves perf pending.

## Health and publication

On the shared host, after restoring the machine and copying complete evidence, the owner writes `JOB/BOARD/health.json` **last**, by atomic rename. Use the digest of the exact `request.json`. Populate a record for each measured and restored boot ID from the lead's actual before/after health evidence:

```json
{"request_sha256":"<sha256>", "restored":true, "boots":{
  "<boot UUID>":{"gpu_faults":0,"new_kernel_errors":0,"failed_units":0}
}}
```

Missing, inconsistent, or unhealthy evidence yields FLAG. A confirmed measurement regression yields FAIL. The one-line summary includes board verdicts, tested SHAs and the candidate tree. Publishing updates the caller's own marked comment. It leaves moved PRs, changed branch tips and unrelated comments alone.
