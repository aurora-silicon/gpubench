# Full Neo performance comparison

`performance-gate.py` compares one supplied Linux record against the exact
recorded Neo macOS run, retaining all 209 benchmark identities. It does not
combine historical and current results or inspect live hardware.

```
python3 tools/performance-gate.py /path/to/current-linux-record \
  --out /path/to/results/current-vs-macos
```

This writes `current-vs-macos.json` and `current-vs-macos.md`. The reference is
`records/neo/macos/20261023-25G83`, pinned to results SHA-256
`cf60e2cb2810e63f53764c2f9bd3066b86e3907866763790ddb26e1c722b1270`.
`--macos` may point to a byte-identical retained copy; it cannot silently select
a smaller or different reference.

Every one of the 207 `ok` reference rows must strictly beat its macOS primary
median. Equal values, gaps, missing/non-ok Linux targets, incompatible metrics
or sampling controls block the comparison. The two macOS skips are explicitly
listed as noncomparable: the medium and solarbay `rt.scene` configurations
using `trace=pipeline`, which Metal cannot run. They do not become wins or
silently disappear. A reported Linux failure remains a record eligibility
failure even if the matching macOS case is noncomparable.

The identity is lane, test, canonical deep parameters, mode, depth, tag and
effective stamps. Missing/null/default RT stamps means `passes`; explicit
`frame` remains distinct. Other lanes default to `default`. Tags preserve
repeated RT work in different sweeps. The old descriptive `compare.py` key
omits these fields and can collapse 16 legitimate records to 193; it is not
the full qualification gate. An exact repeated identity is an error, never a
last-row overwrite. Runs, run_ms and warmup_ms must match outside the identity,
so a short smoke run cannot qualify a full sampling configuration.

Primary names, units and higher/lower direction must match. Comparison uses raw
finite medians without rounding or ratio thresholds. Signed values, including
the three negative macOS query `overhead_us` medians, remain targets. Ratios
are unavailable for nonpositive medians; that does not remove the comparison.
Malformed JSON, duplicate JSON members, malformed metric shapes, nonfinite
numbers and unexpected Linux identities fail closed.

`comparison_passed` reports the complete numeric comparison.
`qualification_eligible` separately requires identifiable, consistent module,
kernel, Mesa source/library and benchmark-assets metadata, a completed run,
live desktop, zero new faults and no recorded executor failure. An embedded
assets manifest must bind the module build ID, source commits, Mesa library
SHA-256 and hashes of both plans, both executors and the full-run driver.
If a separate `assets-manifest.json` exists it must agree. Retained command and
fault evidence is hashed, and known failures contradicting metadata block
eligibility. Missing health evidence, including legacy null desktop state,
cannot be treated as a healthy current record.

A record from `tools/run-linux.sh` alone carries the comparable results but
not this provenance, so it can show `comparison_passed` without being eligible.

Overall `passed` requires both fields. This establishes eligibility of the
supplied full benchmark record; it is not live-stack attestation, statistical
significance analysis or sustained-stress qualification. The ledger records
source paths, raw input/metadata hashes, embedded/retained manifest hashes,
retained evidence hashes and generator hash so the actual scope is reviewable.

Exit status is 0 for a passing eligible record, 1 for valid-input comparison
gaps/missing/equal rows, and 2 for malformed input or ineligible record evidence.
Every reference row appears in both output ledgers, including missing and
noncomparable rows; additional Linux identities are explicitly listed.

Host-only verification:

```
python3 tools/tests/test_performance_gate.py
```

The tests use temporary synthetic data and audit the retained reference.
Synthetic wins are test fixtures, never hardware measurements.
