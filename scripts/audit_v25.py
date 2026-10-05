"""Audit downloaded V25 records, actual selection subsets and policy bytes."""
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
sys.path.insert(0, str(ROOT))
from adaptive_vcm.data import fingerprint, partition
from adaptive_vcm.metrics import curve_summary, paired_ar_bootstrap
from adaptive_vcm.ranking import load_rank_preprocessor, measurement_targets, static_action_order
from adaptive_vcm.selection import Observation, select
from adaptive_vcm.task_bank import ACTION_NAMES


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def assert_ci_matches(actual, declared):
    """Allow only numerical roundoff in interval endpoints across libm builds."""
    assert actual.keys() == declared.keys()
    for name in actual:
        if name in ('lo', 'hi') and actual[name] is not None and declared[name] is not None:
            assert np.isclose(actual[name], declared[name], atol=1e-12, rtol=0)
        else:
            assert actual[name] == declared[name]


def audit(run: Path):
    train, evaluation = run / 'train', run / 'eval'
    training, manifest = read_json(train / 'training_manifest.json'), read_json(evaluation / 'manifest.json')
    cfg = manifest['config']
    assert training['config'] == cfg
    assert training['code'] == manifest['code']
    assert tuple(training['action_names']) == ACTION_NAMES
    assert cfg['ar_require_anchor_decision'] and cfg['ar_kl_slack'] == .1 and cfg['min_savings'] == .01
    assert cfg['rank_top_k'] == 3 and cfg['qps'] == [30, 35, 40, 45, 50]
    assert manifest['ar_guard_rule'] == 'anchor_relative_v2'
    assert manifest['rate_denominator'] == 'original pre-transform T*H*W pixels'
    assert manifest['ids_sha256'] == fingerprint(manifest['ids'])
    assert training['train_ids_sha256'] == fingerprint(training['train_ids'])
    assert all(partition(id_) == 'train' for id_ in training['train_ids'])
    assert all(partition(id_) == manifest['split'] for id_ in manifest['ids'])
    assert not set(training['train_ids']) & set(manifest['ids'])
    state = torch.load(train / 'preprocessor_last.pth', map_location='cpu', weights_only=True)
    load_rank_preprocessor(state)
    assert state['training_config'] == cfg and state['steps'] == training['steps']
    assert state['train_ids_sha256'] == training['train_ids_sha256']
    assert hashlib.sha256((train / 'preprocessor_last.pth').read_bytes()).hexdigest() == manifest['checkpoint_sha256']
    assert hashlib.sha256((train / 'train_records.npz').read_bytes()).hexdigest() == state['train_records_sha256']
    assert state['train_records_sha256'] == training['train_records_sha256']
    assert hashlib.sha256((train / 'measurements.jsonl').read_bytes()).hexdigest() == training['measurements_sha256']
    measured = read_lines(train / 'measurements.jsonl')
    assert len(measured) == training['measurements']
    arrays = np.load(train / 'train_records.npz')
    training_stats = {}
    for index, record in enumerate(measured):
        assert record['source_id'] in training['train_ids']
        assert tuple(a['name'] for a in record['actions']) == ACTION_NAMES
        safety, log_rate, eligible = measurement_targets(record['actions'], slack=.1, min_savings=.01)
        for name, value in [('context', record['context']), ('safety', safety), ('log_rate', log_rate), ('eligible', eligible)]:
            np.testing.assert_allclose(arrays[name][index], value, rtol=1e-6, atol=1e-7)
        assert record['feasible_actions'] == int(eligible.sum())
        key = f"{record['codec']}/{record['qp']}"
        slot = training_stats.setdefault(key, {'records': 0, 'feasible_records': 0, 'feasible_actions': 0,
                                               'oracle_saving_pct_sum': 0.})
        slot['records'] += 1
        slot['feasible_records'] += int(eligible.any())
        slot['feasible_actions'] += int(eligible.sum())
        slot['oracle_saving_pct_sum'] += record['oracle_saving_pct']
    for slot in training_stats.values():
        slot['oracle_mean_saving_pct'] = slot.pop('oracle_saving_pct_sum') / slot['records']
    group_orders = {}
    for group in training_stats:
        indices = [i for i, r in enumerate(measured) if f"{r['codec']}/{r['qp']}" == group]
        group_orders[group] = static_action_order(arrays['safety'][indices], arrays['log_rate'][indices], .01)[:3]
    fit_logs = read_lines(train / 'train.jsonl')
    assert len(fit_logs) == state['steps'] and fit_logs[-1]['step'] == state['steps']
    assert any(row['gradient_norm'] > 0 for row in fit_logs)
    records = []
    for codec in manifest['codecs']:
        records += read_lines(evaluation / f'{codec}_rows.jsonl')
        records += read_lines(evaluation / f'{codec}_components.jsonl')
    indexed = {(row['id'], row['codec'], row['qp'], row['arm']): row for row in records}
    assert len(indexed) == len(records)
    audits = read_lines(evaluation / 'selection_audit.jsonl')
    assert len(audits) == len(manifest['ids']) * len(manifest['codecs']) * len(cfg['qps'])
    assert len({(a['id'], a['codec'], a['qp']) for a in audits}) == len(audits)
    arm_count = Counter()
    group_static_rows = []
    for entry in audits:
        candidates = entry['candidates']
        assert candidates[0]['name'] == 'identity'
        controls = [i for i, a in enumerate(candidates) if a['action_index'] is None]
        learned = [i for i, a in enumerate(candidates) if a['proposed_by_learned']]
        static = [i for i, a in enumerate(candidates) if a['proposed_by_static']]
        assert len(learned) == len(static) == 3
        assert [a['profile'] for a in candidates if a['action_index'] is not None] == list(ACTION_NAMES[1:])
        observations = [Observation(a['name'], a['coded_bytes'],
                                   tuple(float('inf') if d is None else d for d in a['relative_task_distance']),
                                   tuple(a['preserves_decision'])) for a in candidates]
        assert observations[0].distances == (0., 0.) and all(observations[0].preserves_decision)
        for c in candidates:
            if c['stream_sha256'] == candidates[0]['stream_sha256']:
                assert c['coded_bytes'] == candidates[0]['coded_bytes'] and all(c['preserves_decision'])
        arms = {'anchor': [0], 'controls': controls, 'adaptive': controls + learned,
                'static_adaptive': controls + static, 'learned_guarded': [0] + learned,
                'bank_oracle': list(range(len(candidates)))}
        anchor_row = indexed[(entry['id'], entry['codec'], entry['qp'], 'anchor')]
        for arm, pool in arms.items():
            row = indexed[(entry['id'], entry['codec'], entry['qp'], arm)]
            winner = candidates[pool[select([observations[i] for i in pool], .1, .01)]]
            # Rank ordering can break equal-byte action ties; verify membership
            # and exact minimum admissible bytes, not a fabricated rank order.
            matches = [candidates[i] for i in pool if candidates[i]['name'] == row['candidate']
                       and candidates[i]['stream_sha256'] == row['stream_sha256']]
            assert len(matches) == 1 and row['coded_bytes'] == winner['coded_bytes'] == matches[0]['coded_bytes']
            if row['candidate'] != 'identity':
                assert all(matches[0]['preserves_decision'])
                assert all(d is not None and d <= .1 for d in matches[0]['relative_task_distance'])
            assert row['source_sha256'] == anchor_row['source_sha256']
            assert np.isclose(row['bpp'], 8 * row['coded_bytes'] / np.prod([cfg['frames'], cfg['ar_size'], cfg['ar_size']]))
            arm_count[arm] += 1
        raw = indexed[(entry['id'], entry['codec'], entry['qp'], 'learned_raw')]
        assert any(raw['stream_sha256'] == candidates[i]['stream_sha256'] and raw['coded_bytes'] == candidates[i]['coded_bytes'] for i in learned)
        assert indexed[(entry['id'], entry['codec'], entry['qp'], 'adaptive')]['candidate'] == entry['selected']
        group_order = group_orders[f"{entry['codec']}/{entry['qp']}"]
        group_pool = controls + [i for i, c in enumerate(candidates) if c['action_index'] in group_order]
        group_winner = candidates[group_pool[select([observations[i] for i in group_pool], .1, .01)]]
        # The original run did not score all bank actions with the evaluator.
        # Recover quality only when this exact stream was scored in an arm.
        same_stream = next((r for r in records if r['id'] == entry['id'] and r['codec'] == entry['codec']
                            and r['qp'] == entry['qp'] and r['stream_sha256'] == group_winner['stream_sha256']), None)
        group_static_rows.append({'id': entry['id'], 'codec': entry['codec'], 'qp': entry['qp'],
                                  'coded_bytes': group_winner['coded_bytes'], 'candidate': group_winner['name'],
                                  'correct': same_stream['correct'] if same_stream else None})
    dev_hashes = {r['source_sha256'] for r in records}
    assert not dev_hashes & {r['source_sha256'] for r in measured}
    table = {}
    summary = read_json(evaluation / 'summary.json')
    recomputed = {}
    for codec in manifest['codecs']:
        table[codec] = {}
        for qp in cfg['qps']:
            subsets = {arm: [row for row in records if row['codec'] == codec and row['qp'] == qp and row['arm'] == arm]
                       for arm in ['anchor', 'adaptive', 'controls', 'static_adaptive', 'learned_guarded', 'bank_oracle']}
            assert all(len(rows) == manifest['count'] for rows in subsets.values())
            totals = {arm: sum(row['coded_bytes'] for row in rows) for arm, rows in subsets.items()}
            anchor = {r['id']: r for r in subsets['anchor']}
            table[codec][str(qp)] = {
                'total_bytes': totals,
                'saving_pct': {arm: 100 * (1 - total / totals['anchor']) for arm, total in totals.items() if arm != 'anchor'},
                'learned_incremental_pct_of_anchor': 100 * (totals['controls'] - totals['adaptive']) / totals['anchor'],
                'policy_incremental_over_static_pct_of_anchor': 100 * (totals['static_adaptive'] - totals['adaptive']) / totals['anchor'],
                'selected_learned_points': sum(r['candidate'].startswith('learned_rank__') for r in subsets['adaptive']),
                'top1_gap_pp': {name: 100 * np.mean([r['correct'][name] - anchor[r['id']]['correct'][name] for r in subsets['adaptive']])
                                for name in cfg['ar_evaluators']}}
            group_rows = [r for r in group_static_rows if r['codec'] == codec and r['qp'] == qp]
            group_total = sum(r['coded_bytes'] for r in group_rows)
            table[codec][str(qp)]['group_static_rate_only'] = {
                'train_top3': [ACTION_NAMES[i] for i in group_orders[f'{codec}/{qp}']],
                'coded_bytes': group_total,
                'saving_pct': 100 * (1 - group_total / totals['anchor']),
                'policy_incremental_pct_of_anchor': 100 * (group_total - totals['adaptive']) / totals['anchor'],
                'quality_scored_points': sum(r['correct'] is not None for r in group_rows),
                'quality_unscored_points': sum(r['correct'] is None for r in group_rows),
                'scope': 'TRAIN codec/QP table, same top3 proposal budget and teacher guard; primary evaluator quality incomplete'}
            capacity = totals['controls'] - totals['bank_oracle']
            table[codec][str(qp)]['incremental_bank_capacity_capture_fraction'] = (
                (totals['controls'] - totals['adaptive']) / capacity if capacity else None)
            if qp >= 40:
                declared = summary['high_qp_diagnostics'][codec][str(qp)]['adaptive']
                assert np.isclose(declared['rate_change_pct'], -table[codec][str(qp)]['saving_pct']['adaptive'])
                assert declared['top1_gap_pp'] == table[codec][str(qp)]['top1_gap_pp']
            declared = summary['policy_contribution'][codec][str(qp)]
            assert declared['adaptive_bytes'] == totals['adaptive'] and declared['anchor_bytes'] == totals['anchor']
            assert declared['selected_learned_points'] == table[codec][str(qp)]['selected_learned_points']
        paired = [row for row in records if row['codec'] == codec and row['arm'] in ('anchor', 'adaptive')]
        recomputed[codec] = {}
        for name in cfg['ar_evaluators']:
            curves = {}
            for arm in ('anchor', 'adaptive'):
                groups = [[r for r in paired if r['arm'] == arm and r['qp'] == q] for q in cfg['qps']]
                curves[arm] = {'bpp': [float(np.mean([r['bpp'] for r in group])) for group in groups],
                               'quality': [float(np.mean([r['correct'][name] for r in group])) for group in groups]}
            ci = paired_ar_bootstrap(paired, name, cfg['qps'], manifest['bootstrap_draws'], cfg['seed'])
            metrics = curve_summary(curves['anchor']['bpp'], curves['anchor']['quality'],
                                    curves['adaptive']['bpp'], curves['adaptive']['quality'], ci=ci)
            declared = summary['results'][codec][name]
            assert declared['curves'] == curves
            assert_ci_matches(ci, declared['ci'])
            for key in ('bd_rate_pct', 'pchip_bd_rate_pct', 'bd_quality_pp'):
                assert (metrics[key] is None and declared[key] is None) or np.isclose(metrics[key], declared[key], atol=1e-8)
            assert metrics['guards'] == declared['guards'] and metrics['screen_passes'] == declared['screen_passes']
            recomputed[codec][name] = metrics
    return {'scope': f"Completed source-separated {manifest['split']} evidence; audit does not promote the candidate",
            'commit': manifest['code']['commit'], 'split': manifest['split'], 'count': manifest['count'],
            'operating_points': len(audits), 'selection_checks': dict(arm_count),
            'train_codec_qp': training_stats, 'measured_train_sources': len({r['source_id'] for r in measured}),
            'train_dev_id_and_pixel_overlap': 0, 'table': table, 'results': recomputed, 'target_confirmed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'audited': str(args.out), 'points': result['operating_points'],
                      'count': result['count'], 'target_confirmed': False}))
