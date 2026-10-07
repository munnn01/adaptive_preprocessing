"""Literal guard decisions and full CAL selections, with real COCO scoring."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_vcm.selection import relative_guard
from adaptive_vcm.v31.protocol import CODECS, QPS


def mod():
    assert importlib.util.find_spec('adaptive_vcm.v31.guard') is not None, 'CAL guard missing'
    return importlib.import_module('adaptive_vcm.v31.guard')


def cfg(arm='c'):
    return json.loads((Path(__file__).parents[1] / f'configs/v31_{arm}.json').read_text())


def ar(p):
    return {'probabilities': list(p), 'logits': np.log(p).tolist()}


def det(score=.9, box=(0, 0, 10, 10)):
    value = {'boxes': [list(box)], 'scores': [score], 'labels': [1]}
    return {'canonical': copy.deepcopy(value), 'original': copy.deepcopy(value)}


def predictions(task, p):
    names = ['r3d_18', 'mc3_18'] if task == 'ar' else ['mobilenet']
    return {'teachers': {name: copy.deepcopy(p) for name in names},
            'evaluators': {'independent': {'never': 'calibrate'}}}


def rows(task='ar', anchor=None, candidate=None):
    anchor = anchor or (ar([.8, .2]) if task == 'ar' else det())
    candidate = candidate or (ar([.85, .15]) if task == 'ar' else det())
    result = []
    for codec in CODECS:
        for qp in QPS:
            actions = []
            for name, value, size in [('identity', anchor, 1000), ('small', candidate, 500)]:
                actions.append({'descriptor': {'name': name}, 'available': True,
                                'total_bytes': size, 'decoded_sha256': name * 8,
                                'predictions': predictions(task, value)})
            ground = {'label': 0} if task == 'ar' else {
                'image_id': 1, 'categories': [{'id': 1, 'name': 'x'}],
                'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                                 'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0}]}
            result.append({'task': task, 'source_id': '1', 'split': 'cal',
                           'codec': codec, 'qp': qp, 'ground_truth': ground,
                           'config_hash': 'measurement-b', 'code_manifest_hash': 'fixed-code',
                           'model_hashes': {'teachers': {n: n for n in predictions(task, anchor)['teachers']}},
                           'source_predictions': predictions(task, anchor),
                           'anchor_predictions': predictions(task, anchor), 'actions': actions})
    return result


@pytest.mark.parametrize('arm', ['a', 'b'])
def test_strict_rules_are_existing_relative_guard(arm):
    g = mod()
    policy = g.fit_policy(rows(), 'ar', cfg(arm))
    s, a, c = [np.array([.8, .2])]*2, [np.array([.45, .55])]*2, [np.array([.55, .45])]*2
    expected = relative_guard('ar', s, a, c, cfg(arm))
    features = g.guard_features('ar', s, a, c, policy)
    np.testing.assert_allclose(features['distances'], expected[0], atol=0, rtol=0)
    assert tuple(features['decisions']) == expected[1] == (False, False)


def test_temperature_bounded_nll_improvement_and_invalid_labels():
    g = mod()
    result = g.fit_temperature(np.array([[5., 0.], [5., 0.]]), np.array([0, 1]))
    assert 1 < result['temperature'] <= 8
    assert result['nll_after'] < result['nll_before']
    with pytest.raises(ValueError):
        g.fit_temperature(np.array([[1., 0.]]), np.array([2]))


def test_continuous_regrets_match_literal_values_and_thresholds():
    g = mod()
    policy = g.fit_policy(rows(), 'ar', cfg())
    policy['temperatures'] = {name: {'temperature': 1.} for name in cfg()['ar_teachers']}
    f = g.guard_features('ar', [ar([.8, .2])]*2, [ar([.75, .25])]*2, [ar([.7, .3])]*2, policy)
    expected = -.8*np.log(.7/.75) - .2*np.log(.3/.25)
    assert f['ensemble_regret'] == pytest.approx(expected)
    assert f['teacher_regrets'] == pytest.approx([np.log(.75/.7)]*2)
    assert f['hard_protected'] == [True, True]
    obs = {'available': True, 'total_bytes': 500, 'codec': 'h264', 'qp': 30}
    policy['groups']['h264:30'].update(enabled=True, ensemble_threshold=.03, teacher_threshold=.1)
    assert g.is_feasible(obs, f, policy)
    policy['groups']['h264:30']['teacher_threshold'] = .05
    assert not g.is_feasible(obs, f, policy)


def test_disagreement_does_not_protect_conflicting_classes():
    g = mod()
    policy = g.fit_policy(rows(), 'ar', cfg())
    policy['temperatures'] = {name: {'temperature': 1.} for name in cfg()['ar_teachers']}
    f = g.guard_features('ar', [ar([.8, .2])]*2, [ar([.45, .55])]*2, [ar([.55, .45])]*2, policy)
    assert f['hard_protected'] == [False, False]
    assert f['decisions'] == [True, True]
    assert all(r < 0 for r in f['teacher_regrets'])
    agreed = g.guard_features('ar', [ar([.8, .2])]*2, [ar([.75, .25])]*2, [ar([.4, .6])]*2, policy)
    assert agreed['hard_protected'] == [True, True] and agreed['decisions'] == [False, False]


def test_source_temperature_examples_are_not_counted_eight_times():
    g = mod()
    policy = g.fit_policy(rows(), 'ar', cfg())
    assert {p['examples'] for p in policy['temperatures'].values()} == {9}


def test_policy_ignores_noncal_labels_and_all_independent_outputs():
    g = mod()
    original = rows()
    extra = copy.deepcopy(original)
    for row in extra:
        row.update(split='dev', source_id='dev', ground_truth={'label': 999})
    first = g.fit_policy(original + extra, 'ar', cfg())
    changed = copy.deepcopy(original + extra)
    for row in changed:
        row['source_predictions']['evaluators'] = {'bad': float('nan')}
        row['anchor_predictions']['evaluators'] = {'bad': float('nan')}
        for action in row['actions']:
            action['predictions']['evaluators'] = {'bad': float('nan')}
        if row['split'] == 'dev':
            row['ground_truth'] = {'label': -200}
    assert g.fit_policy(changed, 'ar', cfg()) == first


def test_missing_cal_condition_disables_only_incomplete_group():
    g = mod()
    policy = g.fit_policy(rows()[1:], 'ar', cfg())
    assert not policy['groups']['h264:30']['enabled']
    assert policy['groups']['h264:35']['enabled']
    empty = g.fit_policy([], 'ar', cfg())
    assert not any(group['enabled'] for group in empty['groups'].values())


def test_all_unsafe_and_anchor_identical_fallback():
    g = mod()
    measured = rows(candidate=ar([.1, .9]))
    policy = g.fit_policy(measured, 'ar', cfg('b'))
    observations = g.annotate_row(measured[0], policy)
    assert g.choose_feasible(observations, policy) == 0
    measured[0]['actions'][1]['decoded_sha256'] = measured[0]['actions'][0]['decoded_sha256']
    observations = g.annotate_row(measured[0], policy)
    assert g.choose_feasible(observations, policy) == 1
    observations[1]['available'] = False
    assert g.choose_feasible(observations, policy) == 0


def test_nonfinite_guard_fails_closed_and_identity_remains_valid():
    g = mod()
    policy = g.fit_policy(rows(), 'ar', cfg('b'))
    f = g.guard_features('ar', [[float('nan'), 0]]*2, [[.8,.2]]*2, [[.8,.2]]*2, policy)
    assert not f['valid']
    assert not g.is_feasible({'available': True, 'total_bytes': 500, 'codec':'h264','qp':30}, f, policy)
    observations = g.annotate_row(rows()[0], policy)
    observations[1]['total_bytes'] = 995
    assert g.choose_feasible(observations, policy) == 0  # minimum saving uses transmitted bytes


def test_od_true_coco_calibration_rejects_low_ap_despite_saving():
    g = mod()
    measured = rows('od', candidate=det(box=(20,20,30,30)))
    policy = g.fit_policy(measured, 'od', cfg())
    assert not any(group['enabled'] for group in policy['groups'].values())
    assert policy['groups']['h264:30']['anchor_quality']['map'] == pytest.approx(1)
    good = rows('od')
    good[0]['actions'][1]['total_bytes'] = 600
    fitted = g.fit_policy(good, 'od', cfg())
    assert fitted['groups']['h264:30']['enabled']
    assert g.choose_feasible(g.annotate_row(good[0], fitted), fitted) == 1


def test_od_no_reliable_object_forces_anchor():
    g = mod()
    measured = rows('od', anchor=det(score=.2), candidate=det(score=.2))
    policy = g.fit_policy(measured, 'od', cfg('b'))
    assert g.choose_feasible(g.annotate_row(measured[0], policy), policy) == 0


def test_duplicate_cal_cell_and_changed_source_prediction_rejected():
    g = mod()
    measured = rows()
    with pytest.raises(ValueError, match='duplicate'):
        g.fit_policy(measured + [measured[0]], 'ar', cfg())
    measured[1]['source_predictions'] = predictions('ar', ar([.7,.3]))
    with pytest.raises(ValueError, match='source'):
        g.fit_policy(measured, 'ar', cfg())
