"""Frozen teacher-only guards; CAL labels are used only by fit_policy.

The selector, statics and full-bank oracle all call the same row adapter.
Primary evaluator observations are deliberately stripped before calibration.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ..coco_metrics import coco_box, coco_map
from ..selection import relative_guard
from .measure_models import role_view
from .protocol import CODECS, QPS, canonical_hash, validate_config


VERSION = 'v31-cal-regret-1'
AR_GRID = (0., .01, .03, .05, .10)
OD_GRID = (0., .01, .02, .03)


def fit_temperature(logits, labels):
    x, y = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    if x.ndim != 2 or x.shape[1] < 2 or not len(x) or not np.isfinite(x).all() or y.shape != (len(x),) or y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= x.shape[1]):
        raise ValueError('invalid temperature logits/labels')
    def nll(log_t):
        z = x / math.exp(float(log_t))
        return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))
    before, temperature, fallback = nll(0.), 1., None
    try:
        result = minimize_scalar(nll, bounds=(math.log(.25), math.log(8.)), method='bounded',
                                 options={'xatol': 1e-10})
        if not result.success or not math.isfinite(float(result.fun)):
            raise ValueError('bounded NLL minimization failed')
        candidates = [(before, 1.), (nll(math.log(.25)), .25), (nll(math.log(8.)), 8.),
                      (float(result.fun), math.exp(float(result.x)))]
        _, temperature = min(candidates, key=lambda item: (item[0], abs(math.log(item[1])), item[1]))
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        fallback = type(exc).__name__ + ': ' + str(exc)[:300]
    return {'temperature': temperature, 'nll_before': before, 'nll_after': nll(math.log(temperature)),
            'examples': len(y), 'fallback': fallback}


def _probability(value, temperature=1.):
    if isinstance(value, dict):
        p = np.asarray(value['probabilities'], dtype=np.float64)
        x = np.asarray(value['logits'], dtype=np.float64)
        if x.shape != p.shape or not np.isfinite(x).all():
            raise ValueError('invalid logits')
        original = np.exp(x - logsumexp(x))
        if not np.allclose(p, original, atol=1e-6, rtol=1e-5):
            raise ValueError('probabilities disagree with logits')
    else:
        p = np.asarray(value, dtype=np.float64)
        x = np.log(np.maximum(p, 1e-12))
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1., atol=1e-5) or not math.isfinite(temperature) or not .25 <= temperature <= 8:
        raise ValueError('invalid calibrated probabilities')
    if temperature == 1.:
        return p
    return np.exp(x / temperature - logsumexp(x / temperature))


def _soft_regret(source, anchor, trial):
    return float(-np.sum(source * (np.log(np.maximum(trial, 1e-12)) - np.log(np.maximum(anchor, 1e-12)))))


def guard_features(task, source_teacher, anchor_teacher, candidate_teacher, policy):
    """No ground truth/evaluator access; malformed observations fail closed."""
    try:
        names = policy['teacher_names']
        if task != policy['task'] or not names or not (len(source_teacher) == len(anchor_teacher) == len(candidate_teacher) == len(names)):
            raise ValueError('teacher count/task mismatch')
        if task == 'od':
            s, a, c = [[p.get('canonical', p) for p in group] for group in
                       (source_teacher, anchor_teacher, candidate_teacher)]
            d, decisions = relative_guard(task, s, a, c, policy['strict_config'])
            return {'valid': all(math.isfinite(v) for v in d), 'distances': list(d) if all(math.isfinite(v) for v in d) else [],
                    'decisions': list(decisions), 'anchor_identical': False}
        temperatures = [policy['temperatures'].get(name, {}).get('temperature', 1.) if policy['arm'] == 'c' else 1. for name in names]
        s, a, c = [[_probability(p, t) for p, t in zip(group, temperatures)] for group in
                   (source_teacher, anchor_teacher, candidate_teacher)]
        if len({p.shape for group in (s, a, c) for p in group}) != 1:
            raise ValueError('inconsistent teacher probability shapes')
        if policy['arm'] != 'c':
            d, decisions = relative_guard(task, s, a, c, policy['strict_config'])
            return {'valid': all(math.isfinite(v) for v in d), 'distances': list(d),
                    'decisions': list(decisions), 'anchor_identical': False}
        regrets, protected, decisions, changes = [], [], [], []
        for ps, pa, pc in zip(s, a, c):
            agreed = int(ps.argmax()) == int(pa.argmax())
            protect = agreed and ps.max() >= .6 and pa.max() >= .6
            cls = int(pa.argmax())
            regret = float(math.log(max(pa[cls], 1e-12) / max(pc[cls], 1e-12))) if agreed else _soft_regret(ps, pa, pc)
            regrets.append(regret)
            protected.append(bool(protect))
            decisions.append(bool(not protect or int(pc.argmax()) == cls))
            changes.append(int(pc.argmax()) != cls)
        ensemble = _soft_regret(np.mean(s, axis=0), np.mean(a, axis=0), np.mean(c, axis=0))
        return {'valid': math.isfinite(ensemble) and all(math.isfinite(v) for v in regrets),
                'ensemble_regret': ensemble, 'teacher_regrets': regrets, 'hard_protected': protected,
                'decisions': decisions, 'class_changes': changes, 'anchor_identical': False}
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, FloatingPointError):
        return {'valid': False, 'anchor_identical': False}


def is_feasible(observation, features, policy):
    if not observation.get('available') or type(observation.get('total_bytes')) is not int or observation['total_bytes'] <= 0:
        return False
    if observation.get('descriptor', {}).get('name') == 'identity':
        return True
    # An exact decoded AR identity has exactly the anchor task output.
    if features.get('anchor_identical') and policy['task'] == 'ar':
        return True
    group = policy['groups'].get(f"{observation.get('codec')}:{observation.get('qp')}")
    if not features.get('valid') or not group or not group['enabled']:
        return False
    decisions = features.get('decisions', [])
    if len(decisions) != len(policy['teacher_names']) or not all(decisions):
        return False
    if policy['arm'] == 'c' and policy['task'] == 'ar':
        regrets = features.get('teacher_regrets', [])
        ensemble = features.get('ensemble_regret', math.inf)
        return (len(regrets) == len(decisions) and math.isfinite(ensemble) and ensemble <= group['ensemble_threshold']
                and all(math.isfinite(v) and v <= group['teacher_threshold'] for v in regrets))
    distances = features.get('distances', [])
    slack = group['distance_threshold']
    return len(distances) == len(decisions) and all(math.isfinite(v) and v <= slack for v in distances)


def annotate_row(row, policy):
    names = policy['teacher_names']
    source = [row['source_predictions']['teachers'][name] for name in names]
    anchor = [row['anchor_predictions']['teachers'][name] for name in names]
    observations = []
    anchor_hash = row['actions'][0].get('decoded_sha256')
    for item in row['actions']:
        observation = copy.deepcopy(item)
        observation.update(codec=row['codec'], qp=row['qp'])
        if item.get('available'):
            candidate = [item['predictions']['teachers'][name] for name in names]
            features = guard_features(row['task'], source, anchor, candidate, policy)
            features['anchor_identical'] = bool(anchor_hash and item.get('decoded_sha256') == anchor_hash)
        else:
            features = {'valid': False, 'anchor_identical': False}
        observation['guard_features'] = features
        observations.append(observation)
    return observations


def choose_feasible(observations, policy):
    if not observations or observations[0].get('descriptor', {}).get('name') != 'identity' or not is_feasible(observations[0], observations[0].get('guard_features', {}), policy):
        raise ValueError('valid identity must be first')
    anchor = observations[0]['total_bytes']
    feasible = [0]
    for i, observation in enumerate(observations[1:], 1):
        if is_feasible(observation, observation.get('guard_features', {}), policy) and observation['total_bytes'] <= anchor * (1 - policy['min_savings']):
            feasible.append(i)
    return min(feasible, key=lambda i: (observations[i]['total_bytes'], i))


def _ensemble(prediction, policy):
    return np.mean([_probability(prediction['teachers'][name], policy['temperatures'][name]['temperature'])
                    for name in policy['teacher_names']], axis=0)


def _quality(rows, choices, policy):
    if policy['task'] == 'ar':
        p = np.stack([_ensemble(row['actions'][index]['predictions'], policy) for row, index in zip(rows, choices)])
        y = np.asarray([row['ground_truth']['label'] for row in rows])
        if y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= p.shape[1]):
            raise ValueError('invalid CAL labels')
        return {'top1': float(np.mean(p.argmax(axis=1) == y)),
                'nll': float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())}
    ids, gt, results = [], {}, []
    categories = rows[0]['ground_truth']['categories']
    for row, index in zip(rows, choices):
        ground = row['ground_truth']
        image_id = ground['image_id']
        if image_id in ids or ground['categories'] != categories:
            raise ValueError('inconsistent CAL COCO identity/categories')
        ids.append(image_id)
        gt[image_id] = ground['annotations']
        pred = row['actions'][index]['predictions']['teachers'][policy['teacher_names'][0]]['original']
        results.extend({'image_id': image_id, 'category_id': int(label), 'bbox': coco_box(box), 'score': float(score)}
                       for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']))
    ap, ap50 = coco_map(results, gt, ids, {'categories': categories})
    if not math.isfinite(ap) or ap < 0:
        raise ValueError('CAL COCO mAP is undefined')
    return {'map': ap, 'map50': ap50}


def fit_policy(cal_rows, task, cfg):
    cfg = validate_config(cfg)
    if task not in ('ar', 'od'):
        raise ValueError('invalid guard task')
    rows = role_view(cal_rows, 'cal')
    names = cfg['ar_teachers'] if task == 'ar' else [cfg['od_teacher']]
    identities = {canonical_hash({'models': row['model_hashes'], 'code': row.get('code_manifest_hash'),
                                  'config': row.get('config_hash')}) for row in rows}
    if len(identities) > 1:
        raise ValueError('CAL model/code/config identity changed')
    policy = {'version': VERSION, 'task': task, 'arm': cfg['v31_arm'], 'teacher_names': names,
              'config_hash': canonical_hash(cfg), 'cal_hash': canonical_hash(rows),
              'cal_model_hashes': rows[0]['model_hashes']['teachers'] if rows else {},
              'cal_code_manifest_hash': rows[0].get('code_manifest_hash') if rows else None,
              'cal_measurement_config_hash': rows[0].get('config_hash') if rows else None,
              'cal_source_ids': sorted({row['source_id'] for row in rows}),
              'cal_source_pixels_sha256': {row['source_id']: row['source']['source_sha256'] for row in rows
                                          if row.get('source', {}).get('source_sha256')},
              'cal_pixels_sha256': sorted({row.get('source', {}).get('source_sha256') for row in rows
                                          if row.get('source', {}).get('source_sha256')}),
              'strict_config': {key: cfg[key] for key in ('ar_confidence', 'ar_require_anchor_decision', 'od_score_threshold')},
              'min_savings': cfg['min_savings'], 'temperatures': {}, 'groups': {}}
    cells, sources = set(), {}
    for row in rows:
        if row['task'] != task or row['codec'] not in CODECS or row['qp'] not in QPS:
            raise ValueError('invalid CAL condition')
        cell = (row['source_id'], row['codec'], row['qp'])
        if cell in cells:
            raise ValueError('duplicate CAL condition')
        cells.add(cell)
        source_identity = canonical_hash({'prediction': row['source_predictions'], 'ground_truth': row['ground_truth'],
                                          'models': row['model_hashes']})
        if row['source_id'] in sources and sources[row['source_id']][0] != source_identity:
            raise ValueError('CAL source identity changed across conditions')
        sources[row['source_id']] = (source_identity, row)
    if task == 'ar':
        for name in names:
            if cfg['v31_arm'] == 'c' and rows:
                examples = [item[1] for item in sources.values()]
                logits = [row['source_predictions']['teachers'][name]['logits'] for row in examples]
                labels = [row['ground_truth']['label'] for row in examples]
                logits.extend(row['anchor_predictions']['teachers'][name]['logits'] for row in rows)
                labels.extend(row['ground_truth']['label'] for row in rows)
                policy['temperatures'][name] = fit_temperature(logits, np.asarray(labels))
            else:
                policy['temperatures'][name] = {'temperature': 1., 'examples': 0, 'fallback': 'empty CAL' if not rows else None}
    for codec, qp in itertools.product(CODECS, QPS):
        key = f'{codec}:{qp}'
        subset = [row for row in rows if row['codec'] == codec and row['qp'] == qp]
        group = {'enabled': cfg['v31_arm'] != 'c', 'distance_threshold': cfg['ar_kl_slack'] if task == 'ar' else cfg['od_distance_slack'],
                 'ensemble_threshold': 0., 'teacher_threshold': 0., 'cal_sources': len(subset),
                 'reason': 'strict historical rule' if cfg['v31_arm'] != 'c' else 'no complete CAL group'}
        policy['groups'][key] = group
        if cfg['v31_arm'] != 'c' or not subset or {row['source_id'] for row in subset} != set(sources):
            continue
        baseline = _quality(subset, [0]*len(subset), policy)
        anchor_bytes = sum(row['actions'][0]['total_bytes'] for row in subset)
        group.update(anchor_quality=baseline, selected_quality=baseline, saved_bytes=0, trials=[])
        trials = itertools.product(AR_GRID, AR_GRID) if task == 'ar' else [(v,) for v in OD_GRID]
        best = None
        for thresholds in trials:
            group['enabled'] = True
            if task == 'ar':
                group.update(ensemble_threshold=thresholds[0], teacher_threshold=thresholds[1])
            else:
                group['distance_threshold'] = thresholds[0]
            selections = [choose_feasible(annotate_row(row, policy), policy) for row in subset]
            quality = _quality(subset, selections, policy)
            saving = anchor_bytes - sum(row['actions'][index]['total_bytes'] for row, index in zip(subset, selections))
            valid = (quality['top1'] >= baseline['top1'] and quality['nll'] <= baseline['nll'] + 1e-12) if task == 'ar' else quality['map'] >= baseline['map'] - 1e-12
            group['trials'].append({'thresholds': list(thresholds), 'saved_bytes': saving, 'quality': quality, 'valid': bool(valid)})
            if valid and saving > 0 and (best is None or (-saving, thresholds) < (-best[0], best[1])):
                best = (saving, thresholds, quality)
        group['enabled'] = best is not None
        if best:
            group.update(saved_bytes=best[0], selected_quality=best[2], reason='CAL quality preserved')
            if task == 'ar':
                group.update(ensemble_threshold=best[1][0], teacher_threshold=best[1][1])
            else:
                group['distance_threshold'] = best[1][0]
        else:
            group.update(reason='no valid non-identity CAL policy', ensemble_threshold=0., teacher_threshold=0., distance_threshold=0.)
    policy['policy_hash'] = canonical_hash(policy)
    return policy
