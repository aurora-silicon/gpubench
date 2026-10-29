#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Strict full-record Linux performance gate against Neo's recorded macOS run.

Never combines records. All 207 comparable reference rows must strictly win;
equal, missing, failed, incompatible or malformed results block qualification.
The two reference skips remain explicit noncomparable ledger rows.
"""
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
import argparse
import hashlib
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'records/neo/macos/20261023-25G83'
REFERENCE_SHA256 = 'cf60e2cb2810e63f53764c2f9bd3066b86e3907866763790ddb26e1c722b1270'
CONTROLS = ('warmup_ms', 'runs', 'run_ms')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite numeric value')
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON member {key!r}')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'nonfinite JSON constant {value}')

    result = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    finite_tree(result)
    return result


def canonical(value):
    # Numeric params written as 1 and 1.0 describe the same work; booleans
    # remain distinct from numbers. Arrays preserve order and maps sort deeply.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in sorted(value.items())}
    return value


def identity(row):
    for name in ('lane', 'test', 'mode'):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f'{name} must be a nonempty string')
    if not isinstance(row.get('params'), dict):
        raise ValueError('params must be an object')
    depth = row.get('depth')
    if type(depth) is not int or depth < 0 or (row.get('status') == 'ok' and depth == 0):
        raise ValueError('depth must be a nonnegative integer, positive for ok rows')
    tag = row.get('tag')
    if tag is None:
        tag = ''
    if not isinstance(tag, str):
        raise ValueError('tag must be a string or null')
    stamps = row.get('stamps')
    default = 'passes' if row['lane'] == 'rt' else 'default'
    if stamps is None or stamps == 'default':
        stamps = default
    if stamps not in ('passes', 'frame', 'off', 'default'):
        raise ValueError('unrecognized stamps mode')
    return dict(lane=row['lane'], test=row['test'], params=canonical(row['params']),
                mode=row['mode'], depth=depth, tag=tag, stamps=stamps)


def key(row):
    return json.dumps(identity(row), sort_keys=True, separators=(',', ':'), allow_nan=False)


def number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def row_errors(row):
    errors = []
    if row.get('schema') != 'gpubench-result/1':
        errors.append('unsupported result schema')
    if row.get('status') not in ('ok', 'skip', 'fail'):
        errors.append('unrecognized status')
    for control in CONTROLS:
        value = row.get(control)
        if value is not None and (type(value) is not int or value < (0 if control == 'warmup_ms' else 1)):
            errors.append(f'invalid {control}')
    metrics = row.get('metrics', {})
    if not isinstance(metrics, dict):
        return errors + ['metrics must be an object']
    for name, metric in metrics.items():
        # This is a documented legacy descriptive field in the exact reference.
        if name == 'metal_sync' and isinstance(metric, str):
            continue
        if not isinstance(metric, dict):
            errors.append(f'metric {name!r} must be an object')
            continue
        if not isinstance(metric.get('unit'), str) or not metric['unit']:
            errors.append(f'metric {name!r} lacks a unit')
        if metric.get('better') not in ('lower', 'higher'):
            errors.append(f'metric {name!r} lacks a valid direction')
        if not number(metric.get('median')):
            errors.append(f'metric {name!r} lacks a finite numeric median')
        if not number(metric.get('p5')) or not number(metric.get('p95')):
            errors.append(f'metric {name!r} lacks finite percentiles')
        if type(metric.get('n')) is not int or metric['n'] <= 0:
            errors.append(f'metric {name!r} lacks a positive integer sample count')
        if 'samples' in metric and (not isinstance(metric['samples'], list) or
                                    not all(number(value) for value in metric['samples']) or
                                    len(metric['samples']) != metric.get('n')):
            errors.append(f'metric {name!r} has malformed samples')
    if row.get('status') == 'ok':
        primary = row.get('primary')
        if not isinstance(primary, str) or not isinstance(metrics.get(primary), dict):
            errors.append('ok row lacks its named primary metric')
    return errors


@dataclass
class Entry:
    row: dict
    line: int
    errors: list = field(default_factory=list)


@dataclass
class Record:
    path: Path
    rows: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    metadata_errors: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    line_count: int = 0


def load_record(path, expected_os):
    path = Path(path).resolve()
    source = path / 'results.jsonl' if path.is_dir() else path
    record = Record(source)
    record.provenance = dict(results_path=str(source), metadata_path=str(source.parent / 'meta.json'))
    try:
        data = source.read_bytes()
        record.provenance['results_sha256'] = sha(data)
        text = data.decode('utf-8')
    except (OSError, UnicodeError) as error:
        record.errors.append(f'cannot read results: {error}')
        return record
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        record.line_count += 1
        try:
            row = strict_json(line)
            if not isinstance(row, dict):
                raise ValueError('result must be an object')
            row_key = key(row)
        except (ValueError, TypeError) as error:
            record.errors.append(f'line {line_no}: {error}')
            continue
        errors = row_errors(row)
        entry = Entry(row, line_no, errors)
        if row_key in record.rows:
            message = f'exact duplicate benchmark identity at lines {record.rows[row_key].line} and {line_no}'
            record.rows[row_key].errors.append(message)
            record.errors.append(message)
        else:
            record.rows[row_key] = entry
        record.errors.extend(f'line {line_no}: {error}' for error in errors)
    try:
        data = (source.parent / 'meta.json').read_bytes()
        record.provenance['metadata_sha256'] = sha(data)
        meta = strict_json(data)
        if not isinstance(meta, dict) or meta.get('schema') != 'gpubench-record/1':
            raise ValueError('unsupported record metadata schema')
        if meta.get('device') != 'neo' or meta.get('os') != expected_os:
            raise ValueError(f'metadata must identify Neo/{expected_os}')
        if not isinstance(meta.get('suite_sha'), str) or not meta['suite_sha']:
            raise ValueError('metadata lacks suite_sha')
        record.provenance['metadata'] = meta
    except (OSError, ValueError, TypeError) as error:
        record.metadata_errors.append(f'invalid record metadata: {error}')
    record.provenance['nonempty_result_lines'] = record.line_count
    record.provenance['unique_benchmark_identities'] = len(record.rows)
    return record


def reference_errors(record):
    errors = []
    if record.provenance.get('results_sha256') != REFERENCE_SHA256:
        errors.append('reference does not match the pinned Neo macOS results SHA-256')
    statuses = Counter(entry.row.get('status') for entry in record.rows.values())
    if record.line_count != 209 or len(record.rows) != 209 or statuses != {'ok': 207, 'skip': 2}:
        errors.append('reference must preserve exactly 209 identities: 207 ok and 2 skips')
    return errors


def qualification_errors(record):
    """Record eligibility only; this is not live hardware or stress qualification."""
    errors = list(record.metadata_errors)
    meta = record.provenance.get('metadata', {})
    health = meta.get('health')
    if not isinstance(health, dict) or health.get('completed') is not True or health.get('desktop_alive') is not True:
        errors.append('Linux run lacks affirmative completed and desktop-alive evidence')
    if not isinstance(health, dict) or type(health.get('new_fault_lines')) is not int or health['new_fault_lines'] != 0 or health.get('error'):
        errors.append('Linux run lacks zero-new-fault evidence or reports a health error')
    runner = meta.get('runner')
    if not isinstance(runner, dict) or runner.get('failures') != []:
        errors.append('Linux runner failures are missing or nonempty')
    if not isinstance(meta.get('kernel'), str) or not meta['kernel']:
        errors.append('Linux kernel release is missing')
    if not isinstance(meta.get('date_utc'), str) or not meta['date_utc']:
        errors.append('Linux record date is missing')
    module = meta.get('kernel_module')
    module_id = module.get('build_id') if isinstance(module, dict) else None
    commit = lambda value: isinstance(value, str) and re.fullmatch(r'[0-9a-f]{7,40}', value)
    digest = lambda value: isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)
    if not isinstance(module_id, str) or not re.fullmatch(r'[0-9a-f]{40}', module_id):
        errors.append('Linux module build ID is missing or malformed')
    if not isinstance(module, dict) or module.get('name') != 'asahi':
        errors.append('Linux stack does not identify the asahi module')
    if not commit(meta.get('mesa_sha')) or not digest(meta.get('mesa_sha256')):
        errors.append('Linux Mesa source/library identity is missing or malformed')
    assets = meta.get('assets_manifest')
    if not isinstance(assets, dict):
        errors.append('embedded assets manifest is missing')
    else:
        record.provenance['assets_manifest_embedded_sha256'] = sha(
            json.dumps(assets, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        matches = dict(module_build_id=module_id, mesa_source=meta.get('mesa_sha'),
                       mesa_sha256=meta.get('mesa_sha256'), suite_source=meta.get('suite_sha'),
                       rt_source=meta.get('rt_source'))
        for name, expected in matches.items():
            if expected is None or assets.get(name) != expected:
                errors.append(f'assets manifest identity mismatch: {name}')
        for name in ('kernel_source', 'mesa_source', 'suite_source', 'rt_source'):
            if not commit(assets.get(name)):
                errors.append(f'assets manifest source is missing or malformed: {name}')
        files = assets.get('files')
        required = {'suite-plan.txt', 'rt-plan.txt', 'gpubench-vk', 'gpubench-rt-vk', 'full-performance-baseline.py'}
        if not isinstance(files, dict) or not required <= files.keys() or any(
                not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or
                not digest(value) for name, value in files.items()):
            errors.append('assets manifest lacks valid plan/executor/runner SHA-256 evidence')
        sidecar = record.path.parent / 'assets-manifest.json'
        if sidecar.exists():
            try:
                data = sidecar.read_bytes()
                record.provenance['assets_manifest_file_sha256'] = sha(data)
                if strict_json(data) != assets:
                    errors.append('assets-manifest.json disagrees with embedded manifest')
            except (OSError, ValueError, TypeError) as error:
                errors.append(f'invalid assets-manifest.json: {error}')
    for name in ('commands.jsonl', 'new-faults.txt', 'kernel-delta.txt', 'suite-info.json', 'rt-info.json'):
        path = record.path.parent / name
        if not path.exists():
            continue
        try:
            data = path.read_bytes()
            record.provenance.setdefault('retained_evidence_sha256', {})[name] = sha(data)
            if name == 'new-faults.txt' and data.strip():
                errors.append('retained new-faults.txt is nonempty')
            if name == 'commands.jsonl':
                for line_no, line in enumerate(data.splitlines(), 1):
                    if line.strip():
                        command = strict_json(line)
                        if not isinstance(command, dict) or type(command.get('rc')) is not int or command['rc'] != 0:
                            errors.append(f'command ledger line {line_no} reports missing/failed return code')
        except (OSError, ValueError, TypeError) as error:
            errors.append(f'invalid retained {name}: {error}')
    if any(entry.row.get('status') == 'fail' for entry in record.rows.values()):
        errors.append('Linux record contains a failed benchmark, including any noncomparable case')
    return errors


def compare(linux, macos, pinned=True):
    errors = [f'linux: {error}' for error in linux.errors] + [f'macos: {error}' for error in macos.errors]
    if pinned:
        errors += reference_errors(macos)
    rows = []
    for row_key, reference in macos.rows.items():
        candidate = linux.rows.get(row_key)
        row = dict(identity=identity(reference.row), macos_line=reference.line,
                   linux_line=candidate.line if candidate else None,
                   macos_status=reference.row.get('status'), linux_status=candidate.row.get('status') if candidate else 'missing',
                   metric=reference.row.get('primary'), linux=None, macos=None, unit=None,
                   better=None, linux_relative_ratio=None, linux_minus_macos=None, reasons=[])
        if reference.row.get('status') == 'skip' and not reference.errors:
            row.update(outcome='noncomparable', reasons=[reference.row.get('reason', 'reference skipped')])
            if candidate:
                row['linux_reason'] = candidate.row.get('reason') or candidate.row.get('error')
            rows.append(row)
            continue
        if reference.errors or reference.row.get('status') != 'ok':
            row.update(outcome='invalid', reasons=reference.errors or ['reference is not ok'])
        elif candidate is None:
            row.update(outcome='missing', reasons=['no result in the supplied Linux record'])
        elif candidate.errors:
            row.update(outcome='invalid', reasons=candidate.errors)
        elif candidate.row.get('status') != 'ok':
            row.update(outcome='linux_not_ok', reasons=[candidate.row.get('error') or candidate.row.get('reason') or 'Linux status is not ok'])
        else:
            name = reference.row['primary']
            a = candidate.row.get('metrics', {}).get(name)
            b = reference.row['metrics'][name]
            if candidate.row.get('primary') != name:
                row['reasons'].append('primary metric names differ')
            if not isinstance(a, dict):
                row['reasons'].append('Linux lacks the reference primary metric')
            elif (a['unit'], a['better']) != (b['unit'], b['better']):
                row['reasons'].append('primary metric unit/direction mismatch')
            control_a = {name: candidate.row.get(name) for name in CONTROLS}
            control_b = {name: reference.row.get(name) for name in CONTROLS}
            row['run_controls'] = dict(linux=control_a, macos=control_b)
            if control_a != control_b:
                row['reasons'].append('run-control mismatch')
            if row['reasons']:
                row['outcome'] = 'incompatible'
            else:
                av, bv = a['median'], b['median']
                row.update(linux=av, macos=bv, unit=b['unit'], better=b['better'])
                # Never drop signed differences or classify through a ratio.
                row['outcome'] = ('equal' if av == bv else 'win' if
                                  (av < bv if b['better'] == 'lower' else av > bv) else 'gap')
                delta = av - bv
                row['linux_minus_macos'] = delta if number(delta) else None
                if av > 0 and bv > 0:
                    ratio = av / bv if b['better'] == 'lower' else bv / av
                    row['linux_relative_ratio'] = ratio if number(ratio) else None
        rows.append(row)
    for row_key, candidate in linux.rows.items():
        if row_key not in macos.rows:
            rows.append(dict(identity=identity(candidate.row), outcome='unexpected_linux',
                             linux_line=candidate.line, macos_line=None,
                             linux_status=candidate.row.get('status'), macos_status='missing',
                             reasons=['Linux identity is absent from the exact reference']))
    counts = Counter(row['outcome'] for row in rows)
    comparable = sum(entry.row.get('status') == 'ok' for entry in macos.rows.values())
    blocked = sum(count for outcome, count in counts.items() if outcome not in ('win', 'noncomparable'))
    comparison_passed = not errors and comparable > 0 and counts['win'] == comparable and blocked == 0
    eligibility = qualification_errors(linux) + [f'macos: {error}' for error in macos.metadata_errors]
    eligible = not eligibility
    return dict(schema='gpubench-performance-gate/1', passed=comparison_passed and eligible,
                comparison_passed=comparison_passed, qualification_eligible=eligible,
                qualification_scope='Eligibility of this supplied benchmark record; no live hardware or sustained-stress qualification.',
                generator=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__).read_bytes())),
                requirement='Every comparable reference primary median must be strictly better; equality blocks.',
                identity_fields=['lane', 'test', 'params', 'mode', 'depth', 'tag', 'stamps'],
                stamps_defaults=dict(rt='passes', other='default'),
                sampling_controls=list(CONTROLS),
                scope='One supplied Linux record only; no historical/current merging or live hardware identity verification.',
                counts=dict(reference_rows=macos.line_count, linux_rows=linux.line_count,
                            comparable_targets=comparable, noncomparable=counts['noncomparable'],
                            wins=counts['win'], blocked_rows=blocked, errors=len(errors),
                            eligibility_errors=len(eligibility), outcomes=dict(counts)),
                provenance=dict(linux=linux.provenance, macos=macos.provenance,
                                pinned_reference_sha256=REFERENCE_SHA256 if pinned else None),
                errors=errors, eligibility_errors=eligibility, rows=rows)


def cell(value):
    if value is None:
        return '-'
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return str(value).replace('|', '\\|').replace('\n', ' ')


def markdown(ledger):
    count = ledger['counts']
    lines = ['# Neo performance gate', '', '**' + ('PASS' if ledger['passed'] else 'BLOCKED') + '**', '',
             ledger['requirement'], '', ledger['scope'], '',
             f"Raw comparison passed: {ledger['comparison_passed']}; record eligible: {ledger['qualification_eligible']}.", '',
             ledger['qualification_scope'], '',
             f"Reference rows: {count['reference_rows']}; comparable: {count['comparable_targets']}; "
             f"wins: {count['wins']}; blocked rows: {count['blocked_rows']}; "
             f"noncomparable: {count['noncomparable']}; input errors: {count['errors']}; "
             f"record eligibility errors: {count['eligibility_errors']}.", '',
             'Comparison uses raw medians. Nonpositive medians remain comparable; their ratio is unavailable.', '',
             '## Provenance', '']
    for name, source in ledger['provenance'].items():
        if isinstance(source, dict):
            lines += [f"- {name}: {cell(source.get('results_path'))}",
                      f"  results SHA-256: {cell(source.get('results_sha256'))}; metadata SHA-256: {cell(source.get('metadata_sha256'))}."]
    if ledger['errors']:
        lines += ['', '## Input errors', ''] + ['- ' + cell(error) for error in ledger['errors']]
    if ledger['eligibility_errors']:
        lines += ['', '## Record eligibility errors', ''] + ['- ' + cell(error) for error in ledger['eligibility_errors']]
    lines += ['', '## Every reference row and any unmatched Linux rows', '',
              '| outcome | lane/test | params | mode/depth | tag | stamps | primary | Linux | macOS | unit/direction | ratio | reason |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in ledger['rows']:
        identity = row['identity']
        values = [row['outcome'], identity['lane'] + '/' + identity['test'], identity['params'],
                  f"{identity['mode']}/d{identity['depth']}", identity['tag'], identity['stamps'],
                  row.get('metric'), row.get('linux'), row.get('macos'),
                  f"{row.get('unit') or '-'}/{row.get('better') or '-'}", row.get('linux_relative_ratio'),
                  '; '.join(row.get('reasons', []))]
        lines.append('| ' + ' | '.join(cell(value) for value in values) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('linux', type=Path, help='One current Linux record directory or results.jsonl')
    parser.add_argument('--macos', type=Path, default=REFERENCE, help='Exact pinned macOS record or a byte-identical copy')
    parser.add_argument('--out', type=Path, required=True, help='Output prefix for .json and .md ledgers')
    args = parser.parse_args(argv)
    ledger = compare(load_record(args.linux, 'linux'), load_record(args.macos, 'macos'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(args.out) + '.json')
    md_path = Path(str(args.out) + '.md')
    inputs = {Path(source[key]).resolve() for source in ledger['provenance'].values()
              if isinstance(source, dict) for key in ('results_path', 'metadata_path')}
    if json_path.resolve() in inputs or md_path.resolve() in inputs:
        parser.error('output must not overwrite an input record')
    json_path.write_text(json.dumps(ledger, indent=2, allow_nan=False) + '\n')
    md_path.write_text(markdown(ledger))
    print(('PASS' if ledger['passed'] else 'BLOCKED') + ' ' + json.dumps(ledger['counts'], sort_keys=True))
    return 0 if ledger['passed'] else 2 if ledger['errors'] or ledger['eligibility_errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
