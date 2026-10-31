#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Host-only data/provenance tests; synthetic wins are never benchmark results."""
from pathlib import Path
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
TOOL = Path(__file__).resolve().parents[1] / 'performance-gate.py'
spec = importlib.util.spec_from_file_location('performance_gate', TOOL)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def result(value=10, better='lower', metric='us_per_submit', **fields):
    row = dict(schema='gpubench-result/1', lane='suite', test='compute.light', params={},
               mode='queue', depth=1, status='ok', warmup_ms=300, runs=5, run_ms=250,
               primary=metric, metrics={metric: dict(unit='us', better=better, median=value,
                                                    p5=value, p95=value, n=5)})
    row.update(fields)
    return row


def metadata(os):
    meta = dict(schema='gpubench-record/1', device='neo', os=os, suite_sha='3a13953')
    if os == 'linux':
        assets = dict(module_build_id='a' * 40, kernel_source='b' * 40,
                      mesa_source='c' * 40, mesa_sha256='d' * 64,
                      suite_source='3a13953', rt_source='5275db1',
                      files={name: 'e' * 64 for name in ('suite-plan.txt', 'rt-plan.txt',
                              'gpubench-vk', 'gpubench-rt-vk', 'full-performance-baseline.py')})
        meta.update(date_utc='2026-10-24T00:00:00Z', kernel='test-kernel',
                    kernel_module=dict(name='asahi', build_id=assets['module_build_id']),
                    mesa_sha=assets['mesa_source'], mesa_sha256=assets['mesa_sha256'],
                    rt_source=assets['rt_source'], assets_manifest=assets,
                    health=dict(completed=True, desktop_alive=True, new_fault_lines=0),
                    runner=dict(failures=[], duration_s=1))
    return meta


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gpubench-gate-tests-')
        self.root = Path(self.temp.name)
        self.serial = 0

    def tearDown(self):
        self.temp.cleanup()

    def record(self, rows, os='linux', raw=None, meta=None):
        self.serial += 1
        path = self.root / str(self.serial)
        path.mkdir()
        (path / 'results.jsonl').write_text(raw if raw is not None else
                                           ''.join(json.dumps(row) + '\n' for row in rows))
        (path / 'meta.json').write_text(json.dumps(meta or metadata(os)))
        return path

    def compare(self, linux, macos):
        a = gate.load_record(self.record(linux), 'linux')
        b = gate.load_record(self.record(macos, 'macos'), 'macos')
        return gate.compare(a, b, pinned=False)

    def test_missing_and_equal_block(self):
        missing = self.compare([], [result()])
        self.assertFalse(missing['passed'])
        self.assertEqual(missing['rows'][0]['outcome'], 'missing')
        equal = self.compare([result()], [result()])
        self.assertFalse(equal['passed'])
        self.assertEqual(equal['rows'][0]['outcome'], 'equal')

    def test_both_directions_use_raw_values(self):
        for better, linux, outcome in [('lower', 9, 'win'), ('lower', 11, 'gap'),
                                        ('higher', 11, 'win'), ('higher', 9, 'gap')]:
            with self.subTest(better=better, linux=linux):
                ledger = self.compare([result(linux, better)], [result(10, better)])
                self.assertEqual(ledger['rows'][0]['outcome'], outcome)
                self.assertEqual(ledger['passed'], outcome == 'win')
        ledger = self.compare([result(1.00000009)], [result(1.0000001)])
        self.assertTrue(ledger['passed'])

    def test_signed_overhead_compares_without_ratio(self):
        for linux, outcome in [(-0.5, 'win'), (-0.465, 'equal'), (-0.4, 'gap'), (0, 'gap')]:
            with self.subTest(linux=linux):
                ledger = self.compare([result(linux, metric='overhead_us')],
                                      [result(-0.465, metric='overhead_us')])
                self.assertEqual(ledger['rows'][0]['outcome'], outcome)
                self.assertIsNone(ledger['rows'][0]['linux_relative_ratio'])
                self.assertEqual(ledger['counts']['comparable_targets'], 1)
        self.assertEqual(self.compare([result(1)], [result(0)])['rows'][0]['outcome'], 'gap')

    def test_exact_duplicate_is_not_silently_overwritten(self):
        ledger = self.compare([result(9), result(8)], [result(10)])
        self.assertFalse(ledger['passed'])
        self.assertTrue(any('duplicate' in error for error in ledger['errors']))
        self.assertEqual(ledger['rows'][0]['outcome'], 'invalid')

    def test_legitimate_tags_and_stamps_remain_distinct(self):
        reference = [result(tag='sweep:a', lane='rt'), result(tag='sweep:b', lane='rt'),
                     result(tag='stamps-frame', lane='rt', stamps='frame'),
                     result(tag='stamps-frame', lane='rt')]
        linux = copy.deepcopy(reference)
        for row in linux:
            row['metrics']['us_per_submit']['median'] = 9
        ledger = self.compare(linux, reference)
        self.assertTrue(ledger['passed'])
        self.assertEqual(ledger['counts']['wins'], 4)
        explicit = result(lane='rt', stamps='passes')
        implicit = result(lane='rt')
        self.assertEqual(gate.key(explicit), gate.key(implicit))
        duplicate = self.compare([explicit, implicit], [explicit])
        self.assertFalse(duplicate['passed'])

    def test_deep_params_are_canonical_and_bool_is_not_one(self):
        a = result(params={'a': [1.0, {'x': 2}], 'b': True})
        b = result(params={'b': True, 'a': [1, {'x': 2.0}]})
        self.assertEqual(gate.key(a), gate.key(b))
        b['params']['b'] = 1
        self.assertNotEqual(gate.key(a), gate.key(b))

    def test_linux_skip_or_fail_never_qualifies_reference_ok(self):
        for status in ('skip', 'fail'):
            with self.subTest(status=status):
                ledger = self.compare([result(status=status, reason='unsupported', error='failure')], [result()])
                self.assertFalse(ledger['passed'])
                self.assertEqual(ledger['rows'][0]['outcome'], 'linux_not_ok')

    def test_unit_direction_primary_and_controls_mismatch_block(self):
        for field, value in [('unit', 'ms'), ('better', 'higher')]:
            candidate = result(9)
            candidate['metrics']['us_per_submit'][field] = value
            ledger = self.compare([candidate], [result()])
            self.assertEqual(ledger['rows'][0]['outcome'], 'incompatible')
        for candidate in (result(9, metric='other'), result(9, runs=3), result(9, run_ms=50),
                          result(9, warmup_ms=0)):
            self.assertFalse(self.compare([candidate], [result()])['passed'])

    def test_malformed_nonfinite_and_duplicate_json_members_fail_closed(self):
        good = json.dumps(result(9))
        broken = ['not json\n' + good + '\n', good.replace('"median": 9', '"median": NaN'),
                  good.replace('"median": 9', '"median": Infinity'),
                  good.replace('"median": 9', '"median": 1e999'),
                  good.replace('"depth": 1', '"depth": 1, "depth": 2'),
                  good.replace('"median": 9', '"median": true')]
        for raw in broken:
            with self.subTest(raw=raw[:60]):
                a = gate.load_record(self.record([], raw=raw), 'linux')
                b = gate.load_record(self.record([result()], 'macos'), 'macos')
                ledger = gate.compare(a, b, pinned=False)
                self.assertFalse(ledger['passed'])
                self.assertTrue(ledger['errors'])

    def test_extra_rows_and_wrong_platform_metadata_block(self):
        extra = self.compare([result(9), result(9, test='unexpected')], [result()])
        self.assertFalse(extra['passed'])
        self.assertEqual(extra['counts']['outcomes']['unexpected_linux'], 1)
        wrong = gate.load_record(self.record([result(9)], os='macos'), 'linux')
        self.assertTrue(wrong.metadata_errors)

    def test_numeric_comparison_does_not_claim_unknown_or_failed_stack_eligible(self):
        for mutate in (lambda meta: meta.update(health=None),
                       lambda meta: meta['health'].update(completed=False),
                       lambda meta: meta['health'].update(desktop_alive=None),
                       lambda meta: meta['health'].update(new_fault_lines=1),
                       lambda meta: meta['health'].update(new_fault_lines=False),
                       lambda meta: meta['kernel_module'].update(build_id='unknown'),
                       lambda meta: meta.update(mesa_sha256='unknown'),
                       lambda meta: meta['assets_manifest'].update(mesa_source='f' * 40),
                       lambda meta: meta['assets_manifest']['files'].pop('gpubench-rt-vk'),
                       lambda meta: meta['runner'].update(failures=[{'rc': 1}])):
            meta = metadata('linux')
            mutate(meta)
            candidate = gate.load_record(self.record([result(9)], meta=meta), 'linux')
            reference = gate.load_record(self.record([result()], 'macos'), 'macos')
            ledger = gate.compare(candidate, reference, pinned=False)
            self.assertTrue(ledger['comparison_passed'])
            self.assertFalse(ledger['qualification_eligible'])
            self.assertFalse(ledger['passed'])

    def test_both_gpu_module_names_identify_the_stack(self):
        for name, eligible in (('aurora', True), ('asahi', True), ('panfrost', False)):
            with self.subTest(name=name):
                meta = metadata('linux')
                meta['kernel_module'].update(name=name)
                candidate = gate.load_record(self.record([result(9)], meta=meta), 'linux')
                errors = gate.qualification_errors(candidate)
                self.assertEqual(not any('GPU kernel module' in e for e in errors), eligible)

    def test_retained_evidence_contradictions_block_eligibility(self):
        for name, data in [('assets-manifest.json', '{}'), ('new-faults.txt', 'GPU fault'),
                           ('commands.jsonl', '{"rc":1}\n')]:
            with self.subTest(name=name):
                path = self.record([result(9)])
                (path / name).write_text(data)
                reference = gate.load_record(self.record([result()], 'macos'), 'macos')
                ledger = gate.compare(gate.load_record(path, 'linux'), reference, pinned=False)
                self.assertTrue(ledger['comparison_passed'])
                self.assertFalse(ledger['qualification_eligible'])
                self.assertFalse(ledger['passed'])
        path = self.record([result(9)])
        meta = json.loads((path / 'meta.json').read_text())
        (path / 'assets-manifest.json').write_text(json.dumps(meta['assets_manifest']))
        candidate = gate.load_record(path, 'linux')
        self.assertFalse(gate.qualification_errors(candidate))
        self.assertIn('assets_manifest_file_sha256', candidate.provenance)

    def test_noncomparable_is_explicit_and_does_not_hide_targets(self):
        skipped = result(status='skip', test='rt.scene', lane='rt', reason='Metal pipeline unsupported')
        ledger = self.compare([result(9)], [result(), skipped])
        self.assertTrue(ledger['passed'])
        self.assertEqual(ledger['counts']['noncomparable'], 1)
        self.assertEqual(ledger['rows'][1]['outcome'], 'noncomparable')
        self.assertEqual(ledger['rows'][1]['linux_status'], 'missing')

    def test_exact_reference_preserves_all_209_and_three_negative_primaries(self):
        macos = gate.load_record(gate.REFERENCE, 'macos')
        self.assertFalse(macos.errors)
        self.assertFalse(gate.reference_errors(macos))
        self.assertEqual(len(macos.rows), 209)
        negatives = [entry for entry in macos.rows.values() if entry.row.get('status') == 'ok'
                     and entry.row['metrics'][entry.row['primary']]['median'] < 0]
        self.assertEqual(len(negatives), 3)

    def test_complete_synthetic_win_and_cli_missing_fail(self):
        macos = gate.load_record(gate.REFERENCE, 'macos')
        synthetic = []
        for entry in macos.rows.values():
            row = copy.deepcopy(entry.row)
            if row['status'] == 'ok':
                metric = row['metrics'][row['primary']]
                change = max(abs(metric['median']) * 0.1, 0.001)
                metric['median'] += -change if metric['better'] == 'lower' else change
            synthetic.append(row)
        linux = gate.load_record(self.record(synthetic), 'linux')
        ledger = gate.compare(linux, macos)
        self.assertTrue(ledger['passed'])
        self.assertEqual(ledger['counts']['wins'], 207)
        self.assertEqual(len(ledger['rows']), 209)
        self.assertEqual(ledger['counts']['noncomparable'], 2)
        # This is a host-generated test fixture, never written as a GPU record.
        missing = self.record(synthetic[:19])
        out = self.root / 'ledger'
        with redirect_stdout(io.StringIO()):
            code = gate.main([str(missing), '--out', str(out)])
        self.assertEqual(code, 1)
        emitted = json.loads(out.with_suffix('.json').read_text())
        self.assertFalse(emitted['passed'])
        self.assertEqual(len(emitted['rows']), 209)
        self.assertIn('missing', emitted['counts']['outcomes'])
        self.assertIn('BLOCKED', out.with_suffix('.md').read_text())
        self.assertEqual(emitted['provenance']['macos']['results_sha256'], gate.REFERENCE_SHA256)


if __name__ == '__main__':
    unittest.main()
