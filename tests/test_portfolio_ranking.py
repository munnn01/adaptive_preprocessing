"""Regressions for complementary retrieval, byte credit and TRAIN isolation."""
import io

import numpy as np
import pytest
import torch

from adaptive_vcm.data import partition
from adaptive_vcm.utility_ranking import CONTEXT_DIM, source_folds


def train_ids(count):
    return [f'portfolio-fixture/{i}.mp4' for i in range(count * 4)
            if partition(f'portfolio-fixture/{i}.mp4') == 'train'][:count]


def contexts(count, qp=50):
    x = np.zeros((count, CONTEXT_DIM), np.float64)
    x[:, 0], x[:, 2:4], x[:, 4], x[:, -1] = qp / 51, .6, .5, np.log1p(.1)
    return x


def fit(x, ids, actions, *, names=None, control=None):
    from adaptive_vcm.portfolio_ranking import fit_portfolio_model
    anchors = np.full(len(x), 1000)
    names = names or ('identity', 'a', 'b', 'c', 'd')
    kwargs = dict(anchor_bytes=anchors, action_bytes=actions)
    if control is not None:
        kwargs.update(control_bytes=control, baseline_log_rate=np.log(control / anchors))
    return fit_portfolio_model(x, np.ones_like(actions),
                              np.log(actions / anchors[:, None]).astype(np.float32),
                              ids, names, **kwargs)


def test_greedy_portfolio_covers_complementary_actions_instead_of_three_duplicates():
    # Selecting independent means chooses a,b,c and misses half the sources.
    # One redundant proposal must give way to d to recover the other half.
    ids = train_ids(24)
    x = contexts(24)
    actions = np.full((24, 4), 1000)
    actions[:12, :3] = 900
    actions[12:, 3] = 920
    model, report = fit(x, ids, actions)
    assert model.rank(x[0]) == [1, 4, 2]
    assert report['selected_oof']['saved_bytes'] == 2160
    assert report['selected_oof']['feasible_records'] == 24
    assert report['global_static_oof']['saved_bytes'] == 2160
    assert report['group_static_oof']['saved_bytes'] == 2160
    assert model.proposal_details(x[0])['prior_only']
    assert report['selected_recipes']['high']['mix'] == 0.


def test_equal_integer_bytes_are_zero_credit_and_one_saved_byte_survives():
    ids = train_ids(12)
    x = contexts(12)
    actions = np.full((12, 4), 990)
    control = np.full(12, 990)
    _, report = fit(x, ids, actions, control=control)
    assert report['bank_oracle']['saved_bytes'] == 0
    assert report['selected_oof']['feasible_records'] == 0
    actions[:, 0] = 989
    _, report = fit(x, ids, actions, control=control)
    assert report['selected_oof']['saved_bytes'] == 12
    assert report['selected_oof']['feasible_records'] == 12


def test_every_qp_of_source_stays_held_out_from_its_fold_memory_and_scaler():
    ids = train_ids(4)
    source_assignment = source_folds(ids)
    special = ids[int(np.flatnonzero(source_assignment == 0)[0])]
    repeated = [source for source in ids for _ in range(10)]
    x = contexts(40)
    x[:, 5] = [400. if source == special else 0. for source in repeated]
    actions = np.full((40, 4), 990)
    _, report = fit(x, repeated, actions)
    assert report['fold_assignment'] == np.repeat(source_assignment, 10).tolist()
    fold = report['oof_folds'][0]
    assert fold['train_context_mean'][5] == 0.
    assert fold['valid_source_ids'] == [special]
    assert special not in fold['train_source_ids']
    assert len(fold['train_source_ids']) == 3
    assert report['records'] == 40


def test_checkpoint_replays_portfolio_and_rejects_corrupt_training_memory():
    from adaptive_vcm.portfolio_ranking import load_portfolio_preprocessor
    ids = train_ids(12)
    x = contexts(12)
    x[:, 5] = np.linspace(0., 1., 12)
    actions = np.full((12, 4), 990)
    model, _ = fit(x, ids, actions)
    state = model.checkpoint_state()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored = load_portfolio_preprocessor(torch.load(buffer, weights_only=True),
                                           action_names=model.action_names)
    assert restored.rank(x[-1]) == model.rank(x[-1])
    assert restored.proposal_details(x[-1]) == model.proposal_details(x[-1])
    with pytest.raises(ValueError):
        load_portfolio_preprocessor(state, action_names=('identity', 'a', 'b', 'd', 'c'))
    state['model']['memory_utility'][0, 0] = torch.nan
    with pytest.raises(ValueError):
        load_portfolio_preprocessor(state, action_names=model.action_names)


def test_nontrain_ids_and_invalid_proposal_count_fail_closed():
    ids = train_ids(12)
    x = contexts(12)
    actions = np.full((12, 4), 990)
    model, _ = fit(x, ids, actions)
    with pytest.raises(ValueError):
        model.rank(x[0], top_k=2)
    invalid = next(f'portfolio-fixture/{i}.mp4' for i in range(100)
                   if partition(f'portfolio-fixture/{i}.mp4') != 'train')
    with pytest.raises(ValueError):
        fit(x, [invalid] * 12, actions)


def test_kernel_recovers_source_conditioned_actions_with_separate_qp_origins():
    # Four disjoint source conditions each have one feasible action. A static
    # K3 can cover only three; nearest-context proposals must cover all four.
    ids = train_ids(64)
    x = contexts(128)
    x[::2, 0] = 30 / 51
    x[1::2, 5] = np.repeat([0., 10., 20., 30.], 16)
    actions = np.full((128, 4), 1000)
    actions[::2, 0] = 900
    for index in range(64):
        actions[2 * index + 1, index // 16] = 900
    model, report = fit(x, [source for source in ids for _ in range(2)], actions)
    assert report['selected_oof']['saved_bytes'] == 12800
    assert report['selected_oof']['feasible_records'] == 128
    assert report['global_static_oof']['saved_bytes'] < 12800
    assert report['selected_recipes']['low']['mix'] == 0.
    assert report['selected_recipes']['high']['mix'] > 0.
    assert model.proposal_details(x[0])['prior_only']
    assert not model.proposal_details(x[-1])['prior_only']
    assert model.rank(x[-1])[0] == 4


def test_missing_group_falls_back_to_prior_without_false_learned_origin():
    from adaptive_vcm.portfolio_ranking import load_portfolio_preprocessor
    ids = train_ids(12)
    x = contexts(12)
    model, _ = fit(x, ids, np.full((12, 4), 990))
    state = model.checkpoint_state()
    state['recipes']['high']['mix'] = 1.
    restored = load_portfolio_preprocessor(state, action_names=model.action_names)
    query = contexts(1, qp=40)[0]
    details = restored.proposal_details(query)
    assert details['missing_group_fallback']
    assert details['prior_only'] and details['selected_mix'] == 0.
    assert details['action_indices'] == [1, 2, 3]


@pytest.mark.parametrize('buffer', ['mean', 'scale'])
def test_checkpoint_scaler_must_be_fitted_only_from_saved_train_memory(buffer):
    from adaptive_vcm.portfolio_ranking import load_portfolio_preprocessor
    ids = train_ids(12)
    x = contexts(12)
    x[:, 5] = np.arange(12)
    model, _ = fit(x, ids, np.full((12, 4), 990))
    state = model.checkpoint_state()
    state['model'][buffer][5] += 10.
    with pytest.raises(ValueError):
        load_portfolio_preprocessor(state, action_names=model.action_names)
