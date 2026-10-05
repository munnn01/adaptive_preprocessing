"""Auditor gates reject unsafe byte wins and source leakage independently."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_v26 import capacity, independent_cv, independent_folds, minimum, targets


def observation(name, size, *, distances=None, decisions=None, shape=None):
    return {'name': name, 'coded_bytes': size, 'distances': distances or [0., 0.],
            'decisions': decisions or [True, True], 'shape': shape or [16, 128, 128, 3]}


def test_minimum_rejects_unsafe_wins_and_preserves_threshold_ties():
    anchor = observation('identity', 1000)
    rows = [anchor, observation('unsafe_class', 400, decisions=[True, False]),
            observation('unsafe_kl', 500, distances=[0., .100001]),
            observation('too_small_gain', 991), observation('exact_one_percent', 990),
            observation('equal_bytes_later', 990)]
    assert minimum(rows)['name'] == 'exact_one_percent'
    safety, rate, eligible = targets(rows)
    np.testing.assert_array_equal(eligible, [0., 0., 0., 1., 1.])
    assert safety[2] == 1 and np.isclose(rate[3], math.log(.99))


def test_capacity_counts_new_sources_not_merely_more_action_trials():
    old = ('identity', 'old_a', 'old_b')
    records = [{'codec': 'h265', 'qp': 50, 'source_shape': [16, 128, 128, 3],
                'actions': [observation('identity', 1000), observation('old_a', 995),
                            observation('old_b', 900, decisions=[False, True]),
                            observation('new_filter', 980), observation('new_filter_weaker', 990)]},
               {'codec': 'h265', 'qp': 50, 'source_shape': [16, 128, 128, 3],
                'actions': [observation('identity', 1000), observation('old_a', 980),
                            observation('old_b', 990), observation('new_filter', 975),
                            observation('new_filter_weaker', 985)]}]
    result = capacity(records, old)['h265/50']
    assert result['newly_feasible_records'] == result['newly_pure_filter_feasible_records'] == 1
    assert result['new_action_feasible_trials'] == 4
    assert result['v25_feasible_records'] == 1 and result['expanded_feasible_records'] == 2


def test_source_folds_keep_every_codec_qp_record_of_source_together():
    ids = ['a'] * 10 + ['b'] * 10 + ['c'] * 10 + ['d'] * 10
    assignments = independent_folds(ids)
    for id_ in set(ids):
        assert len(set(assignments[np.asarray(ids) == id_])) == 1
    assert set(assignments) == {0, 1, 2, 3}


def utility_module():
    path = ROOT / 'adaptive_vcm/utility_ranking.py'
    if not path.exists():
        path = Path('D:/STUDY/LAB/bao_1/adaptive_v25_build/adaptive_vcm/utility_ranking.py')
    spec = importlib.util.spec_from_file_location('adaptive_vcm.utility_ranking', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cv_recomputes_all_recipes_and_rejects_corrupted_source_assignment():
    from adaptive_vcm.data import partition
    utility = utility_module()
    ids = [f'source/{i}.mp4' for i in range(100) if partition(f'source/{i}.mp4') == 'train'][:12]
    ids = [value for value in ids for _ in range(2)]
    rng = np.random.default_rng(417)
    x = rng.random((len(ids), 41))
    x[:, 0] = np.resize(np.asarray([30, 35, 40, 45, 50]) / 51., len(x))
    x[:, 1] = np.arange(len(x)) % 2
    x[:, 2:4], x[:, 4] = .7, .5
    sizes = np.arange(len(x)) * 31 + 1000
    x[:, -1] = np.log1p(8 * sizes / (16 * 128 * 128))
    safety = (rng.random((len(x), 4)) > .3).astype(np.float32)
    rate = np.log(rng.uniform(.78, 1.08, size=safety.shape)).astype(np.float32)
    baseline = np.log(np.full(len(x), .95))
    names = ('identity', 'a', 'b', 'c', 'd')
    model, fit = utility.fit_utility_model(x, safety, rate, ids, names,
                                          anchor_bytes=sizes, baseline_log_rate=baseline)
    result = independent_cv(x, safety, rate, baseline, ids, sizes, fit, model.checkpoint_state())
    assert result['recipes_recomputed'] == 24 and result['source_leakage'] == 0
    broken = copy.deepcopy(fit)
    broken['fold_assignment'][0] = (broken['fold_assignment'][0] + 1) % 4
    with pytest.raises(AssertionError):
        independent_cv(x, safety, rate, baseline, ids, sizes, broken, model.checkpoint_state())
