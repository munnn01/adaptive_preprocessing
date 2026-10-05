"""Audit completed V26 DEV artifacts, TRAIN CV, actual bytes and prior provenance.

This reads evidence only. It never infers teachers, selects a DEV checkpoint,
changes an experiment, or treats a small zero-bootstrap pilot as confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
QPS = (30, 35, 40, 45, 50)
GRID = tuple((a, m, p) for a in (10., 100., 1000.) for m in (0., .25, .5, 1.) for p in (4., 16.))
ARMS = ('anchor', 'adaptive', 'controls', 'learned_guarded', 'learned_raw',
        'static_adaptive', 'group_static_adaptive', 'v25_bank_oracle', 'bank_oracle')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_close(actual, expected, *, atol=1e-8):
    """Compare structured metrics without accepting missing keys or NaNs."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected)
        for key in expected:
            check_close(actual[key], expected[key], atol=atol)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            check_close(a, b, atol=atol)
    elif expected is None or isinstance(expected, (str, bool)):
        assert actual == expected
    else:
        assert math.isfinite(float(actual)) and math.isfinite(float(expected))
        assert np.isclose(actual, expected, rtol=1e-8, atol=atol), (actual, expected)


def fields(row):
    return (row['relative_task_distance'], row['preserves_decision']) if 'relative_task_distance' in row else (row['distances'], row['decisions'])


def safe(row):
    distances, decisions = fields(row)
    return (len(distances) == len(decisions) == 2 and all(decisions)
            and all(value is not None and math.isfinite(value) and value <= .1 for value in distances))


def feasible(row, anchor_bytes):
    assert type(row['coded_bytes']) is int and row['coded_bytes'] > 0
    return safe(row) and row['coded_bytes'] <= anchor_bytes * .99


def minimum(pool):
    assert pool and pool[0]['name'] == 'identity' and safe(pool[0])
    anchor_bytes = pool[0]['coded_bytes']
    assert fields(pool[0])[0] == [0., 0.]
    eligible = [pool[0]] + [row for row in pool[1:] if feasible(row, anchor_bytes)]
    return min(eligible, key=lambda row: row['coded_bytes'])


def targets(actions):
    assert actions[0]['name'] == 'identity'
    size = actions[0]['coded_bytes']
    return (np.asarray([safe(row) for row in actions[1:]], np.float32),
            np.asarray([math.log(row['coded_bytes'] / size) for row in actions[1:]], np.float32),
            np.asarray([feasible(row, size) for row in actions[1:]], np.float32))


def independent_folds(ids, folds=4):
    unique = sorted(set(ids), key=lambda value: hashlib.sha256(('utility-train-cv-v1:' + value).encode()).digest())
    assert len(unique) >= folds
    assignment = {value: index % folds for index, value in enumerate(unique)}
    return np.asarray([assignment[value] for value in ids], np.int64)


def retrieval(scores, gains, sizes):
    order = np.argsort(-scores, axis=1, kind='stable')[:, :3]
    selected = np.take_along_axis(gains, order, axis=1).max(1)
    saved = np.rint(selected * sizes).astype(np.int64)
    return {'mean_guarded_saving_pct': float(100 * selected.mean()),
            'total_byte_saving_pct': float(100 * saved.sum() / sizes.sum()),
            'feasible_records': int((selected > 0).sum()), 'records': len(gains),
            'saved_bytes': int(saved.sum()),
            'top1_counts': dict(Counter(str(int(i + 1)) for i in order[:, 0]))}


def independent_cv(x, safety, rate, baseline, ids, sizes, fit, checkpoint, *, action_bytes, control_bytes):
    """Recompute every source-blocked recipe from archived TRAIN labels only."""
    assert x.shape == (len(ids), 41) and x.dtype == np.float64
    group = np.rint(x[:, 1]).astype(int) * 5 + np.asarray([QPS.index(int(round(q * 51))) for q in x[:, 0]])
    assignment = independent_folds(ids, 4)
    assert fit['folds'] == 4 and fit['source_ids'] == ids
    assert fit['fold_assignment'] == assignment.tolist()
    assert fit['utility_target_scope'] == checkpoint['utility_target_scope'] == 'marginal_saving_beyond_guarded_controls'
    assert fit['label_precision'] == 'actual_integer_bytes_v1'
    assert np.all(action_bytes == np.rint(action_bytes)) and np.all(control_bytes == np.rint(control_bytes))
    np.testing.assert_allclose(rate, np.log(action_bytes / sizes[:, None]), atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(baseline, np.log(control_bytes / sizes), atol=1e-12, rtol=1e-12)
    eligible = (safety == 1.) & (action_bytes <= sizes[:, None] * .99)
    gain = np.where(eligible, np.maximum(0, control_bytes[:, None] - action_bytes) / sizes[:, None], 0.)

    def prior(xx, yy, gg, pseudo):
        global_mean = yy.mean(0)
        means = np.empty((10, yy.shape[1]))
        counts = np.zeros(10)
        for index in range(10):
            slot = yy[gg == index]
            counts[index] = len(slot)
            means[index] = ((slot.sum(0) + pseudo * global_mean) / (len(slot) + pseudo)
                            if len(slot) + pseudo else global_mean)
        return means, counts, global_mean

    def fit_state(xx, yy, gg, alpha, mix, pseudo):
        means, counts, global_mean = prior(xx, yy, gg, pseudo)
        center, scale = xx.mean(0), xx.std(0)
        scale = np.where(scale < 1e-6, 1., scale)
        z = (xx - center) / scale
        residual = yy - means[gg]
        intercept = residual.mean(0)
        coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(41), z.T @ (residual - intercept))
        return {'prior': means, 'counts': counts, 'global_mean': global_mean,
                'group_static_prior': prior(xx, yy, gg, 0.)[0], 'mean': center, 'scale': scale,
                'coefficients': coefficients, 'intercept': intercept, 'mix': mix}

    def predict(state, xx, gg):
        residual = ((xx - state['mean']) / state['scale']) @ state['coefficients'] + state['intercept']
        return np.maximum(0., state['prior'][gg] + state['mix'] * residual)

    all_scores, cv = [], []
    global_scores, group_scores = np.zeros_like(gain), np.zeros_like(gain)
    for fold in range(4):
        train, valid = assignment != fold, assignment == fold
        assert not set(np.asarray(ids)[train]) & set(np.asarray(ids)[valid])
        global_scores[valid] = gain[train].mean(0)
        group_scores[valid] = prior(x[train], gain[train], group[train], 0.)[0][group[valid]]
    for alpha, mix, pseudo in GRID:
        scores = np.zeros_like(gain)
        for fold in range(4):
            train, valid = assignment != fold, assignment == fold
            state = fit_state(x[train], gain[train], group[train], alpha, mix, pseudo)
            scores[valid] = predict(state, x[valid], group[valid])
        all_scores.append(scores)
        cv.append({'alpha': alpha, 'mix': mix, 'prior_pseudo_count': pseudo, **retrieval(scores, gain, sizes)})
    check_close(cv, fit['cv_grid'])
    best = min(range(len(cv)), key=lambda i: (-cv[i]['saved_bytes'], cv[i]['mix'], -cv[i]['alpha'], -cv[i]['prior_pseudo_count']))
    chosen = cv[best]
    check_close({key: chosen[key] for key in ('alpha', 'mix', 'prior_pseudo_count')}, fit['selected_recipe'])
    check_close({'alpha': chosen['alpha'], 'mix': chosen['mix'], 'pseudo': chosen['prior_pseudo_count']}, checkpoint['recipe'])
    assert fit['prior_only'] == (chosen['mix'] == 0.)
    state = fit_state(x, gain, group, chosen['alpha'], chosen['mix'], chosen['prior_pseudo_count'])
    for name, expected in state.items():
        if name != 'mix':
            np.testing.assert_allclose(checkpoint['model'][name].numpy(), expected, rtol=1e-8, atol=1e-9)
            assert checkpoint['model'][name].dtype == torch.float64
    order = [int(i + 1) for i in np.argsort(-state['global_mean'], kind='stable')]
    assert checkpoint['static_action_order'] == order
    reports = {'selected_oof': retrieval(all_scores[best], gain, sizes),
               'global_static_oof': retrieval(global_scores, gain, sizes),
               'group_static_oof': retrieval(group_scores, gain, sizes),
               'bank_oracle': retrieval(gain, gain, sizes),
               'final_train_resubstitution': retrieval(predict(state, x, group), gain, sizes)}
    for name, report in reports.items():
        check_close(report, fit[name])
    for number in sorted(set(group)):
        slot = group == number
        name = f"{'h265' if number >= 5 else 'h264'}/{QPS[number % 5]}"
        expected = {key: retrieval(scores[slot], gain[slot], sizes[slot]) for key, scores in
                    (('selected_oof', all_scores[best]), ('global_static_oof', global_scores),
                     ('group_static_oof', group_scores), ('bank_oracle', gain))}
        check_close(expected, fit['by_codec_qp'][name])
    for fold in range(4):
        slot = assignment == fold
        expected = {'fold': fold, **{key: retrieval(scores[slot], gain[slot], sizes[slot]) for key, scores in
                    (('selected_oof', all_scores[best]), ('global_static_oof', global_scores), ('group_static_oof', group_scores))}}
        check_close(expected, fit['oof_folds'][fold])
    return {'folds': 4, 'source_leakage': 0, 'recipes_recomputed': len(GRID),
            'selected_recipe': fit['selected_recipe'], 'prior_only': fit['prior_only'], **reports}


def capacity(records, old_names):
    output = {}
    for codec, qp in sorted({(r['codec'], r['qp']) for r in records}):
        rows = [r for r in records if (r['codec'], r['qp']) == (codec, qp)]
        old_flags, new_flags, old_gains, new_gains = [], [], [], []
        new_trials, pure_newly_feasible, pure_flags = 0, 0, []
        for record in rows:
            size = record['actions'][0]['coded_bytes']
            gains = [1 - row['coded_bytes'] / size if feasible(row, size) else 0. for row in record['actions'][1:]]
            split = len(old_names) - 1
            old, new = any(gains[:split]), any(gains)
            old_flags.append(old)
            new_flags.append(new)
            old_gains.append(max(gains[:split]))
            new_gains.append(max(gains))
            new_trials += sum(gain > 0 for gain in gains[split:])
            pure = [gain for row, gain in zip(record['actions'][1:], gains) if row['shape'] == record['source_shape']]
            pure_old = [gain for row, gain in zip(record['actions'][1:split + 1], gains[:split]) if row['shape'] == record['source_shape']]
            pure_flags.append(any(pure))
            pure_newly_feasible += int(any(pure) and not any(pure_old))
        output[f'{codec}/{qp}'] = {'records': len(rows), 'v25_feasible_records': sum(old_flags),
            'expanded_feasible_records': sum(new_flags),
            'newly_feasible_records': sum(new and not old for old, new in zip(old_flags, new_flags)),
            'new_action_feasible_trials': new_trials,
            'v25_oracle_mean_saving_pct': 100 * float(np.mean(old_gains)),
            'expanded_oracle_mean_saving_pct': 100 * float(np.mean(new_gains)),
            'new_nonidentity_trials': len(rows) * (len(records[0]['actions']) - len(old_names)),
            'pure_filter_feasible_records': sum(pure_flags),
            'newly_pure_filter_feasible_records': pure_newly_feasible}
    return output


def bootstrap(rows, model, qps, draws, seed, bd_rate):
    ids = sorted({row['id'] for row in rows})
    lookup = {(r['id'], r['qp'], r['arm']): r for r in rows}
    assert len(lookup) == len(rows) == len(ids) * len(qps) * 2
    arrays = {arm: (np.asarray([[lookup[(id_, q, arm)]['bpp'] for q in qps] for id_ in ids]),
                     np.asarray([[lookup[(id_, q, arm)]['correct'][model] for q in qps] for id_ in ids]))
              for arm in ('anchor', 'adaptive')}
    rng, values = np.random.default_rng(seed), []
    for _ in range(draws):
        sample = rng.integers(0, len(ids), len(ids))
        ar, aq = (array[sample].mean(0) for array in arrays['anchor'])
        br, bq = (array[sample].mean(0) for array in arrays['adaptive'])
        result = bd_rate(ar, aq, br, bq)
        if math.isfinite(result):
            values.append(result)
    return {'lo': float(np.percentile(values, 2.5)) if values else None,
            'hi': float(np.percentile(values, 97.5)) if values else None,
            'draws': draws, 'finite_draws': len(values), 'finite_fraction': len(values) / draws if draws else 0.,
            'method': 'paired_source_video_bootstrap'}


def audit(run: Path, *, code_root: Path = ROOT, expected_commit=None, baseline_v25=None):
    train, evaluation = run / 'train', run / 'eval'
    training, manifest, summary = read_json(train / 'training_manifest.json'), read_json(evaluation / 'manifest.json'), read_json(evaluation / 'summary.json')
    cfg = manifest['config']
    assert manifest['task'] == 'ar' and manifest['split'] == 'dev'
    assert training['config'] == cfg and training['code'] == manifest['code']
    assert cfg['ar_training'] == 'source_validated_utility'
    assert cfg['ar_require_anchor_decision'] and cfg['ar_kl_slack'] == .1 and cfg['min_savings'] == .01
    assert cfg['rank_top_k'] == 3 and cfg['qps'] == list(QPS)
    assert (cfg['frames'], cfg['ar_size'], cfg['temporal_stride'], cfg['preset'], cfg['fps']) == (16, 128, 2, 'medium', 25)
    assert cfg['ar_teachers'] == ['r3d_18', 'mc3_18'] and cfg['ar_evaluators'] == ['r2plus1d_18', 'r3d_18']
    assert manifest['ar_guard_rule'] == 'anchor_relative_v2'
    assert manifest['rate_denominator'] == 'original pre-transform T*H*W pixels'
    assert manifest['codecs'] == ['h264', 'h265'] and manifest['component_evaluation']
    if expected_commit:
        assert manifest['code']['commit'] == expected_commit
    for filename, expected in manifest['code']['files_sha256'].items():
        assert hashlib.sha256((code_root / filename).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
    sys.path.insert(0, str(code_root))
    from adaptive_vcm.data import fingerprint, partition
    from adaptive_vcm.metrics import bd_rate, curve_summary
    from adaptive_vcm.task_bank import ACTION_NAMES as OLD_NAMES
    from adaptive_vcm.stabilized_bank import ACTION_NAMES
    from adaptive_vcm.utility_ranking import load_utility_preprocessor
    assert len(ACTION_NAMES) == 34 and tuple(ACTION_NAMES[:len(OLD_NAMES)]) == OLD_NAMES
    assert tuple(training['action_names']) == ACTION_NAMES
    assert manifest['ids_sha256'] == fingerprint(manifest['ids'])
    assert training['train_ids_sha256'] == fingerprint(training['train_ids'])
    assert all(partition(id_) == 'train' for id_ in training['train_ids'])
    assert all(partition(id_) == 'dev' for id_ in manifest['ids'])
    assert not set(training['train_ids']) & set(manifest['ids'])
    checkpoint = torch.load(train / 'preprocessor_last.pth', map_location='cpu', weights_only=True)
    model = load_utility_preprocessor(checkpoint, action_names=ACTION_NAMES)
    assert checkpoint['fit_method'] == 'sourceblocked_train_cv_ridge' and 'steps' not in checkpoint
    assert checkpoint['training_config'] == cfg
    assert checkpoint['context_dim'] == training['context_dim'] == 41
    assert checkpoint['context_schema'] == training['context_schema'] == 'teacher-scalars-source-statistics-anchor-bpp-v1'
    assert checkpoint['train_ids_sha256'] == training['train_ids_sha256']
    assert sha256(train / 'preprocessor_last.pth') == manifest['checkpoint_sha256']
    assert sha256(train / 'train_records.npz') == checkpoint['train_records_sha256'] == training['train_records_sha256']
    assert sha256(train / 'measurements.jsonl') == training['measurements_sha256']
    config_bytes = (code_root / 'configs/v26_screen.json').read_bytes()
    assert read_json(code_root / 'configs/v26_screen.json') == cfg
    assert checkpoint['config_sha256'] in {hashlib.sha256(config_bytes).hexdigest(), hashlib.sha256(config_bytes.replace(b'\r\n', b'\n')).hexdigest()}
    measured = read_lines(train / 'measurements.jsonl')
    assert len(measured) == training['measurements'] == checkpoint['measurements']
    arrays = np.load(train / 'train_records.npz')
    assert set(arrays.files) == {'context', 'safety', 'log_rate', 'eligible', 'baseline_log_rate'}
    assert arrays['context'].dtype == arrays['baseline_log_rate'].dtype == np.float64
    sizes, source_ids = [], []
    for index, record in enumerate(measured):
        assert record['measurement'] == index + 1 and record['source_id'] in training['train_ids']
        assert tuple(a['name'] for a in record['actions']) == ACTION_NAMES
        assert tuple(a['name'] for a in record['controls']) == tuple(cfg['ar_candidates'])
        anchor = record['actions'][0]
        assert record['controls'][0]['stream_sha256'] == anchor['stream_sha256']
        assert record['controls'][0]['coded_bytes'] == anchor['coded_bytes']
        winner = minimum(record['controls'])
        assert winner['name'] == record['controls_selected'] and winner['coded_bytes'] == record['controls_coded_bytes']
        expected_baseline = math.log(winner['coded_bytes'] / anchor['coded_bytes'])
        assert record['baseline_log_rate'] == expected_baseline
        assert arrays['baseline_log_rate'][index] == expected_baseline
        safe_, rate_, eligible_ = targets(record['actions'])
        for name, expected in [('context', record['context']), ('safety', safe_), ('log_rate', rate_), ('eligible', eligible_)]:
            np.testing.assert_allclose(arrays[name][index], expected, atol=1e-8, rtol=1e-8)
        assert record['teacher_safe_actions'] == int(safe_.sum()) and record['feasible_actions'] == int(eligible_.sum())
        shape = record['source_shape']
        assert shape == [16, 128, 128, 3] and len(record['source_sha256']) == 64
        bpp = 8 * anchor['coded_bytes'] / np.prod(shape[:3])
        assert bpp == record['actual_anchor_bpp']
        context = np.asarray(record['context'], np.float64)
        assert context.shape == (41,) and np.isfinite(context).all()
        assert int(round(context[0] * 51)) == record['qp'] and int(round(context[1])) == int(record['codec'] == 'h265')
        assert math.isclose(context[-1], math.log1p(bpp), rel_tol=0., abs_tol=1e-12)
        sizes.append(anchor['coded_bytes'])
        source_ids.append(record['source_id'])
    fit = read_json(train / 'fit_diagnostics.json')
    check_close(fit, training['fit_diagnostics'])
    cv = independent_cv(arrays['context'], arrays['safety'], arrays['log_rate'], arrays['baseline_log_rate'],
                        source_ids, np.asarray(sizes, np.float64), fit, checkpoint,
                        action_bytes=np.asarray([[a['coded_bytes'] for a in r['actions'][1:]] for r in measured]),
                        control_bytes=np.asarray([r['controls_coded_bytes'] for r in measured]))
    training_capacity = capacity(measured, OLD_NAMES)
    declared_capacity = read_json(train / 'capacity_diagnostics.json')
    check_close(declared_capacity, training['capacity_diagnostics'])
    for key, declared in declared_capacity.items():
        check_close({name: training_capacity[key][name] for name in declared}, declared, atol=1e-5)

    records = []
    for codec in manifest['codecs']:
        records += read_lines(evaluation / f'{codec}_rows.jsonl')
        records += read_lines(evaluation / f'{codec}_components.jsonl')
    indexed = {(r['id'], r['codec'], r['qp'], r['arm']): r for r in records}
    points = manifest['count'] * len(manifest['codecs']) * len(QPS)
    assert len(indexed) == len(records) == points * len(ARMS)
    assert {r['arm'] for r in records} == set(ARMS)
    audits = read_lines(evaluation / 'selection_audit.jsonl')
    assert len(audits) == points and len({(a['id'], a['codec'], a['qp']) for a in audits}) == points
    assert not {r['source_sha256'] for r in records} & {r['source_sha256'] for r in measured}
    selection_count, proposal_count = Counter(), Counter()
    prefix = 'trained_prior__' if model.learned_mix == 0 else 'learned_rank__'
    for entry in audits:
        candidates = entry['candidates']
        assert candidates[0]['name'] == 'identity'
        controls = [a for a in candidates if a['action_index'] is None]
        bank = {a['action_index']: a for a in candidates if a['action_index'] is not None}
        assert [a['name'] for a in controls] == cfg['ar_candidates']
        assert list(bank) == list(range(1, len(ACTION_NAMES)))
        assert [a['profile'] for a in bank.values()] == list(ACTION_NAMES[1:])
        assert all(a['name'] == prefix + a['profile'] for a in bank.values())
        context = np.asarray(candidates[0]['ranking_context'], np.float64)
        assert context.shape == (41,) and np.isfinite(context).all()
        assert hashlib.sha256(context.tobytes()).hexdigest() == candidates[0]['ranking_context_sha256']
        size = candidates[0]['coded_bytes']
        assert math.isclose(context[-1], math.log1p(8 * size / (16 * 128 * 128)), rel_tol=0., abs_tol=1e-12)
        assert int(round(context[0] * 51)) == entry['qp'] and int(round(context[1])) == int(entry['codec'] == 'h265')
        learned = model.rank(context, top_k=3)
        static = model.static_action_order[:3]
        group_static = model.group_static_action_order(context, top_k=3)
        assert candidates[0]['learned_order'] == learned and candidates[0]['global_static_order'] == static
        assert candidates[0]['group_static_order'] == group_static
        check_close(candidates[0]['proposal_details'], model.proposal_details(context, 3))
        for field, indices in [('proposed_by_learned', learned), ('proposed_by_static', static), ('proposed_by_group_static', group_static)]:
            assert len(indices) == len(set(indices)) == 3
            assert {a['action_index'] for a in candidates if a[field]} == set(indices)
        subsets = {'anchor': [candidates[0]], 'controls': controls,
                   'adaptive': controls + [bank[i] for i in learned],
                   'learned_guarded': [candidates[0]] + [bank[i] for i in learned],
                   'static_adaptive': controls + [bank[i] for i in static],
                   'group_static_adaptive': controls + [bank[i] for i in group_static],
                   'v25_bank_oracle': controls + [bank[i] for i in range(1, len(OLD_NAMES))],
                   'bank_oracle': candidates}
        key = (entry['id'], entry['codec'], entry['qp'])
        assert indexed[(*key, 'adaptive')]['candidate'] == entry['selected']
        for arm, pool in subsets.items():
            winner = minimum(pool)
            row = indexed[(*key, arm)]
            assert row['candidate'] == winner['name'] and row['coded_bytes'] == winner['coded_bytes']
            assert row['stream_sha256'] == winner['stream_sha256']
            selection_count[arm] += 1
        raw = indexed[(*key, 'learned_raw')]
        assert raw['candidate'] == bank[learned[0]]['name'] and raw['stream_sha256'] == bank[learned[0]]['stream_sha256']
        for candidate in candidates:
            if candidate['stream_sha256'] == candidates[0]['stream_sha256']:
                assert candidate['coded_bytes'] == size and safe(candidate)
        for arm in ARMS:
            row = indexed[(*key, arm)]
            assert row['source_sha256'] == indexed[(*key, 'anchor')]['source_sha256']
            assert row['bpp'] == 8 * row['coded_bytes'] / (16 * 128 * 128)
            assert set(row['correct']) == set(cfg['ar_evaluators']) and all(v in (0, 1) for v in row['correct'].values())
        proposal_count['points'] += 1
        proposal_count['learned_points'] += int(model.learned_mix > 0 and entry['selected'].startswith(prefix))
        proposal_count['trained_prior_points'] += int(model.learned_mix == 0 and entry['selected'].startswith(prefix))

    metrics, table = {}, {}
    for codec in manifest['codecs']:
        metrics[codec], table[codec] = {}, {}
        for arm in ARMS[1:]:
            metrics[codec][arm] = {}
            paired = [r for r in records if r['codec'] == codec and r['arm'] in ('anchor', arm)]
            paired = [{**r, 'arm': 'adaptive' if r['arm'] == arm else 'anchor'} for r in paired]
            for name in cfg['ar_evaluators']:
                curves = {a: {'bpp': [float(np.mean([r['bpp'] for r in paired if r['arm'] == a and r['qp'] == q])) for q in QPS],
                              'quality': [float(np.mean([r['correct'][name] for r in paired if r['arm'] == a and r['qp'] == q])) for q in QPS]}
                          for a in ('anchor', 'adaptive')}
                ci = bootstrap(paired, name, QPS, manifest['bootstrap_draws'] if arm == 'adaptive' else 0, cfg['seed'], bd_rate)
                result = {'curves': curves, **curve_summary(curves['anchor']['bpp'], curves['anchor']['quality'],
                          curves['adaptive']['bpp'], curves['adaptive']['quality'], ci=ci)}
                declared = summary['results'][codec][name] if arm == 'adaptive' else summary['component_results'][codec][arm][name]
                check_close(result, declared)
                metrics[codec][arm][name] = result
        for qp in QPS:
            rows = {arm: [r for r in records if r['codec'] == codec and r['qp'] == qp and r['arm'] == arm] for arm in ARMS}
            assert all(len(slot) == manifest['count'] for slot in rows.values())
            totals = {arm: sum(r['coded_bytes'] for r in slot) for arm, slot in rows.items()}
            count = Counter(r['candidate'] for r in rows['adaptive'])
            chosen_families, incremental_families = Counter(), Counter()
            controls = {r['id']: r for r in rows['controls']}
            for row in rows['adaptive']:
                if row['candidate'].startswith(prefix):
                    family = 'resample' if row['candidate'].split('__', 1)[1].startswith('resample') else 'pure_filter'
                    chosen_families[family] += 1
                    incremental_families[family] += controls[row['id']]['coded_bytes'] - row['coded_bytes']
            anchor = {r['id']: r for r in rows['anchor']}
            gaps = {arm: {name: 100 * float(np.mean([r['correct'][name] - anchor[r['id']]['correct'][name] for r in slot]))
                          for name in cfg['ar_evaluators']} for arm, slot in rows.items()}
            contribution = summary['policy_contribution'][codec][str(qp)]
            assert contribution['adaptive_bytes'] == totals['adaptive'] and contribution['anchor_bytes'] == totals['anchor']
            assert contribution['selected_learned_points'] == sum(r['candidate'].startswith('learned_rank__') for r in rows['adaptive'])
            assert contribution['selected_trained_prior_points'] == sum(r['candidate'].startswith('trained_prior__') for r in rows['adaptive'])
            for arm in ('controls', 'static_adaptive', 'group_static_adaptive', 'v25_bank_oracle', 'bank_oracle'):
                assert contribution[arm]['coded_bytes'] == totals[arm]
                assert np.isclose(contribution[arm]['incremental_saving_pct_of_anchor'], 100 * (totals[arm] - totals['adaptive']) / totals['anchor'])
            if qp >= 40:
                for arm in ARMS[1:]:
                    check_close(summary['high_qp_diagnostics'][codec][str(qp)][arm],
                                {'rate_change_pct': 100 * (totals[arm] / totals['anchor'] - 1),
                                 'candidate_counts': dict(Counter(r['candidate'] for r in rows[arm])), 'top1_gap_pp': gaps[arm]})
            table[codec][str(qp)] = {'total_bytes': totals,
                'saving_pct': {arm: 100 * (1 - total / totals['anchor']) for arm, total in totals.items()},
                'top1_gap_pp': gaps, 'selected_candidate_counts': dict(count),
                'selected_policy_by_family': dict(chosen_families), 'incremental_bytes_by_family': dict(incremental_families),
                'expanded_bank_extra_bytes_over_v25_bank': totals['v25_bank_oracle'] - totals['bank_oracle'],
                'policy_incremental_over_controls_pct': 100 * (totals['controls'] - totals['adaptive']) / totals['anchor'],
                'policy_incremental_over_group_static_pct': 100 * (totals['group_static_adaptive'] - totals['adaptive']) / totals['anchor']}
    assert summary['policy_fit']['prior_only'] == (model.learned_mix == 0)
    assert summary['policy_fit']['learned_mix'] == model.learned_mix
    assert summary['proposal_budget']['learned'] == summary['proposal_budget']['static'] == summary['proposal_budget']['group_static'] == 3
    assert summary['proposal_budget']['bank_oracle'] == len(ACTION_NAMES) - 1
    assert summary['target_confirmed'] is False
    pairing = None
    if baseline_v25:
        previous = []
        old_manifest = read_json(Path(baseline_v25) / 'manifest.json')
        assert old_manifest['code']['files_sha256']['adaptive_vcm/task_bank.py'] == manifest['code']['files_sha256']['adaptive_vcm/task_bank.py']
        for codec in manifest['codecs']:
            previous += read_lines(Path(baseline_v25) / f'{codec}_rows.jsonl')
            previous += read_lines(Path(baseline_v25) / f'{codec}_components.jsonl')
        previous = {(r['id'], r['codec'], r['qp'], r['arm']): r for r in previous}
        comparable = [r for r in records if r['arm'] == 'anchor' and (r['id'], r['codec'], r['qp'], 'anchor') in previous]
        matched = sum(all(r[key] == previous[(r['id'], r['codec'], r['qp'], 'anchor')][key]
                          for key in ('coded_bytes', 'stream_sha256', 'source_sha256')) for r in comparable)
        pairing = {'comparable_points': len(comparable), 'exact_anchor_matches': matched,
                   'all_comparable_anchors_match': matched == len(comparable),
                   'covers_whole_current_dev': len(comparable) == points,
                   'exact_control_stream_matches': sum(indexed[(r['id'], r['codec'], r['qp'], 'controls')]['stream_sha256'] ==
                                                       previous[(r['id'], r['codec'], r['qp'], 'controls')]['stream_sha256'] for r in comparable)}
    return {'scope': 'Completed DEV evidence; no independent TEST confirmation', 'commit': manifest['code']['commit'],
            'count': manifest['count'], 'operating_points': points, 'rows_checked': len(records),
            'guarded_selection_checks': dict(selection_count), 'proposal_counts': dict(proposal_count),
            'train_dev_id_and_pixel_overlap': 0, 'context_dim': 41, 'checkpoint_cv': cv,
            'training_capacity': training_capacity, 'table': table, 'results': metrics,
            'v25_anchor_pairing': pairing, 'target_confirmed': False,
            'limits': 'Teacher probabilities and source videos are not re-inferred; persisted compact scalars are schema/hash checked. Small zero-bootstrap pilots do not confirm BD-rate.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, default=ROOT)
    parser.add_argument('--expected-commit')
    parser.add_argument('--baseline-v25', type=Path)
    args = parser.parse_args()
    result = audit(args.run, code_root=args.code_root, expected_commit=args.expected_commit, baseline_v25=args.baseline_v25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'audited': str(args.out), 'points': result['operating_points'], 'rows': result['rows_checked'], 'target_confirmed': False}))
