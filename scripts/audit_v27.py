"""Read-only V27 provenance, guarded stream selection and task-quality audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_v26 import (read_json, read_lines, sha256, check_close,
                               minimum, feasible, bootstrap, independent_folds)

QPS = (30, 35, 40, 45, 50)
ARMS = ('anchor', 'adaptive', 'controls', 'learned_guarded', 'learned_raw',
        'static_adaptive', 'group_static_adaptive', 'v25_bank_oracle', 'v26_bank_oracle', 'bank_oracle')


def validate_evaluation_records(manifest, cfg, records):
    """Bind every score and rate to a complete, immutable DEV source grid."""
    from adaptive_vcm.data import partition, fingerprint
    ids = manifest['ids']
    assert manifest['split'] == 'dev'
    assert manifest['rate_denominator'] == 'original pre-transform T*H*W pixels'
    assert len(ids) == len(set(ids)) == manifest['count']
    assert manifest['ids_sha256'] == fingerprint(ids)
    assert all(partition(identifier) == 'dev' for identifier in ids)
    assert tuple(manifest['codecs']) == ('h264', 'h265')
    expected = {(identifier, codec, qp, arm) for identifier in ids for codec in manifest['codecs']
                for qp in QPS for arm in ARMS}
    actual = [(r['id'], r['codec'], r['qp'], r['arm']) for r in records]
    assert len(actual) == len(expected) and set(actual) == expected
    denominator = cfg['frames'] * cfg['ar_size'] ** 2
    source_hashes = {identifier: set() for identifier in ids}
    for row in records:
        assert type(row['coded_bytes']) is int and row['coded_bytes'] > 0
        assert row['bpp'] == 8 * row['coded_bytes'] / denominator
        assert set(row['correct']) == set(cfg['ar_evaluators'])
        assert all(value in (0, 1) for value in row['correct'].values())
        assert len(row['source_sha256']) == 64
        source_hashes[row['id']].add(row['source_sha256'])
    assert all(len(hashes) == 1 for hashes in source_hashes.values())


def audit_training(train, *, code_root=ROOT, expected_commit=None):
    """Validate measured labels and reproduce every registered TRAIN CV recipe.

    Labels are recomputed independently from integer bytes/guards. CV fitting
    replay checks consistency with the registered algorithm, not a second model.
    """
    from adaptive_vcm.utility_ranking import load_record_directory
    from adaptive_vcm.portfolio_ranking import load_portfolio_preprocessor, fit_portfolio_model
    bundle = load_record_directory(train)
    manifest = bundle['manifest']
    assert manifest['device'] in ('cuda', 'cpu')
    if expected_commit:
        assert manifest['code']['commit'] == expected_commit
    for name, digest in manifest['code']['files_sha256'].items():
        assert hashlib.sha256((code_root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest
    state = torch.load(train / 'preprocessor_last.pth', map_location='cpu', weights_only=True)
    assert state['training_config'] == manifest['config']
    assert state['measurements'] == manifest['measurements']
    assert state['train_ids_sha256'] == manifest['train_ids_sha256']
    assert state['train_records_sha256'] == manifest['train_records_sha256'] == sha256(train / 'train_records.npz')
    assert state['measurements_sha256'] == manifest['measurements_sha256'] == sha256(train / 'measurements.jsonl')
    model = load_portfolio_preprocessor(state)
    records = read_lines(train / 'measurements.jsonl')
    x = bundle['context']
    utility = np.zeros((len(records), len(model.action_names) - 1))
    for index, row in enumerate(records):
        baseline = row['controls_coded_bytes']
        size = row['actions'][0]['coded_bytes']
        assert minimum(row['controls'])['coded_bytes'] == baseline
        for action, values in enumerate(row['actions'][1:]):
            if feasible(values, size) and values['coded_bytes'] < baseline:
                utility[index, action] = (baseline - values['coded_bytes']) / size
    np.testing.assert_array_equal(model.memory_context.numpy(), x)
    np.testing.assert_array_equal(model.memory_utility.numpy(), utility)
    assert model.source_ids == bundle['source_ids']
    fit = read_json(train / 'fit_diagnostics.json')
    check_close(fit, manifest['fit_diagnostics'])
    check_close(fit, model.fit_diagnostics)
    assert fit['fold_assignment'] == independent_folds(bundle['source_ids'], fit['folds']).tolist()
    refit, refit_report = fit_portfolio_model(x, bundle['safety'], bundle['log_rate'], bundle['source_ids'],
        bundle['action_names'], anchor_bytes=bundle['anchor_bytes'], action_bytes=bundle['action_bytes'],
        control_bytes=bundle['control_bytes'], baseline_log_rate=bundle['baseline_log_rate'])
    check_close(fit, refit_report)
    assert refit.recipes == model.recipes
    return model, manifest, records


def audit(run, *, code_root=ROOT, expected_commit=None, baseline_v26=None):
    from adaptive_vcm.anchor_bank import ACTION_NAMES
    from adaptive_vcm.task_bank import ACTION_NAMES as V25_NAMES
    from adaptive_vcm.stabilized_bank import ACTION_NAMES as V26_NAMES
    from adaptive_vcm.data import partition
    from adaptive_vcm.metrics import curve_summary, bd_rate
    model, training, measured = audit_training(run / 'train', code_root=code_root, expected_commit=expected_commit)
    evaluation = run / 'eval'
    manifest, summary = read_json(evaluation / 'manifest.json'), read_json(evaluation / 'summary.json')
    cfg = training['config']
    assert manifest['config'] == cfg and manifest['code'] == training['code']
    assert manifest['checkpoint_sha256'] == sha256(run / 'train/preprocessor_last.pth')
    assert tuple(model.action_names) == ACTION_NAMES and manifest['task'] == 'ar'
    assert tuple(manifest['codecs']) == ('h264', 'h265')
    records = [row for codec in manifest['codecs'] for file in (f'{codec}_rows.jsonl', f'{codec}_components.jsonl')
               for row in read_lines(evaluation / file)]
    validate_evaluation_records(manifest, cfg, records)
    indexed = {(r['id'], r['codec'], r['qp'], r['arm']): r for r in records}
    points = manifest['count'] * 10
    assert len(indexed) == len(records) == points * len(ARMS)
    assert {r['arm'] for r in records} == set(ARMS)
    assert all(partition(r['id']) == manifest['split'] for r in records)
    assert not {r['source_sha256'] for r in records} & {r['source_sha256'] for r in measured}
    audits = read_lines(evaluation / 'selection_audit.jsonl')
    assert len(audits) == len({(r['id'], r['codec'], r['qp']) for r in audits}) == points
    coverage, selections, groups = Counter(), Counter(), {}
    for entry in audits:
        key = entry['id'], entry['codec'], entry['qp']
        candidates = entry['candidates']
        controls = [c for c in candidates if c['action_index'] is None]
        bank = {c['action_index']: c for c in candidates if c['action_index'] is not None}
        assert [c['name'] for c in controls] == cfg['ar_candidates']
        assert list(bank) == list(range(1, len(ACTION_NAMES)))
        assert [c['profile'] for c in bank.values()] == list(ACTION_NAMES[1:])
        context = np.asarray(candidates[0]['ranking_context'], np.float64)
        assert hashlib.sha256(context.tobytes()).hexdigest() == candidates[0]['ranking_context_sha256']
        assert round(context[0] * 51) == entry['qp'] and round(context[1]) == int(entry['codec'] == 'h265')
        size = candidates[0]['coded_bytes']
        assert np.isclose(context[-1], np.log1p(8 * size / (cfg['frames'] * cfg['ar_size'] ** 2)), atol=1e-12, rtol=0)
        learned, static = model.rank(context, 3), model.static_action_order[:3]
        group = model.group_static_action_order(context, 3)
        details = model.proposal_details(context, 3)
        check_close(candidates[0]['proposal_details'], details)
        assert candidates[0]['learned_order'] == learned and candidates[0]['global_static_order'] == static
        assert candidates[0]['group_static_order'] == group
        prefix = 'trained_prior__' if details['prior_only'] else 'learned_rank__'
        assert all(c['name'] == prefix + c['profile'] for c in bank.values())
        for field, indices in (('proposed_by_learned', learned), ('proposed_by_static', static), ('proposed_by_group_static', group)):
            assert len(indices) == len(set(indices)) == 3
            assert {c['action_index'] for c in candidates if c[field]} == set(indices)
        subsets = dict(anchor=[candidates[0]], controls=controls, adaptive=controls + [bank[i] for i in learned],
            learned_guarded=[candidates[0]] + [bank[i] for i in learned],
            static_adaptive=controls + [bank[i] for i in static], group_static_adaptive=controls + [bank[i] for i in group],
            v25_bank_oracle=controls + [bank[i] for i in range(1, len(V25_NAMES))],
            v26_bank_oracle=controls + [bank[i] for i in range(1, len(V26_NAMES))], bank_oracle=candidates)
        assert indexed[(*key, 'adaptive')]['candidate'] == entry['selected']
        for arm, pool in subsets.items():
            winner, row = minimum(pool), indexed[(*key, arm)]
            assert (row['candidate'], row['coded_bytes'], row['stream_sha256']) == (winner['name'], winner['coded_bytes'], winner['stream_sha256'])
        raw = indexed[(*key, 'learned_raw')]
        assert raw['stream_sha256'] == bank[learned[0]]['stream_sha256'] and raw['candidate'] == bank[learned[0]]['name']
        assert raw['coded_bytes'] == bank[learned[0]]['coded_bytes']
        baseline = minimum(controls)['coded_bytes']
        useful = [c for c in bank.values() if feasible(c, size) and c['coded_bytes'] < baseline]
        retrieved = [c for c in useful if c['action_index'] in learned]
        slot = groups.setdefault(f'{entry["codec"]}/{entry["qp"]}', Counter())
        for counter in (coverage, slot):
            counter['points'] += 1
            counter['bank_marginal_points'] += bool(useful)
            counter['retrieved_marginal_points'] += bool(retrieved)
            counter['conditional_proposal_points'] += not details['prior_only']
            counter['marginal_saved_bytes'] += baseline - indexed[(*key, 'adaptive')]['coded_bytes']
            counter['new_bank_capacity_bytes'] += indexed[(*key, 'v26_bank_oracle')]['coded_bytes'] - indexed[(*key, 'bank_oracle')]['coded_bytes']
        selections[entry['selected']] += 1
    metrics = {}
    for codec in manifest['codecs']:
        metrics[codec] = {}
        for name in cfg['ar_evaluators']:
            paired = [r for r in records if r['codec'] == codec and r['arm'] in ('anchor', 'adaptive')]
            curves = {arm: dict(bpp=[float(np.mean([r['bpp'] for r in paired if r['arm'] == arm and r['qp'] == q])) for q in QPS],
                quality=[float(np.mean([r['correct'][name] for r in paired if r['arm'] == arm and r['qp'] == q])) for q in QPS]) for arm in ('anchor', 'adaptive')}
            ci = bootstrap(paired, name, QPS, manifest['bootstrap_draws'], cfg['seed'], bd_rate)
            result = {'curves': curves, **curve_summary(curves['anchor']['bpp'], curves['anchor']['quality'], curves['adaptive']['bpp'], curves['adaptive']['quality'], ci=ci)}
            check_close(result, summary['results'][codec][name])
            metrics[codec][name] = result
    assert summary['target_confirmed'] is False
    pairing = None
    if baseline_v26:
        previous = [r for codec in manifest['codecs'] for filename in (f'{codec}_rows.jsonl', f'{codec}_components.jsonl')
                    for r in read_lines(baseline_v26 / filename)]
        old = {(r['id'], r['codec'], r['qp'], r['arm']): r for r in previous}
        comparable = [r for r in records if r['arm'] in ('anchor', 'controls') and (r['id'], r['codec'], r['qp'], r['arm']) in old]
        matching = sum(all(r[k] == old[(r['id'], r['codec'], r['qp'], r['arm'])][k]
                          for k in ('stream_sha256', 'coded_bytes', 'source_sha256')) for r in comparable)
        pairing = dict(comparable_streams=len(comparable), exact_stream_matches=matching, full_pairing=len(comparable) == 2 * points)
    return dict(passes=True, points=points, coverage=dict(coverage), by_codec_qp={k: dict(v) for k, v in groups.items()},
        selected_candidates=dict(selections), metrics=metrics, baseline_v26_pairing=pairing,
        limitations='No teacher/video re-inference. Persisted source/anchor context and measured guard labels are checked; TRAIN CV replay is not independent DEV.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, default=ROOT)
    parser.add_argument('--expected-commit')
    parser.add_argument('--baseline-v26', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root))
    result = audit(args.run, code_root=args.code_root, expected_commit=args.expected_commit, baseline_v26=args.baseline_v26)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passes=result['passes'], points=result['points'], coverage=result['coverage'])))


if __name__ == '__main__':
    main()
