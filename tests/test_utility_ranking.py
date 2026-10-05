import numpy as np
import pytest
import torch

from adaptive_vcm.data import partition
from adaptive_vcm.utility_ranking import (CONTEXT_DIM, UtilityRankPreprocessor,
    build_utility_context, fit_utility_model, load_utility_preprocessor, source_folds, _utility)


def train_ids(count):
    return [f"utility-fixture/{index}.mp4" for index in range(count * 3)
            if partition(f"utility-fixture/{index}.mp4") == "train"][:count]


def test_source_folds_block_every_codec_qp_of_one_source():
    ids = train_ids(12)
    records = [source for source in ids for _ in range(10)]
    fold = source_folds(records)
    for source in ids:
        assert len(set(fold[np.asarray(records) == source])) == 1
    shuffled = np.random.default_rng(26).permutation(len(records))
    np.testing.assert_array_equal(source_folds([records[index] for index in shuffled]), fold[shuffled])
    assert set(fold) == {0, 1, 2, 3}


def test_compact_context_omits_class_identity_and_uses_actual_anchor_bpp():
    clip = np.random.default_rng(26).integers(0, 256, (4, 16, 24, 3), np.uint8)
    mask = np.zeros((16, 24), np.float32)
    source, anchor = np.full(400, .1 / 399), np.full(400, .2 / 399)
    source[2], anchor[7] = .9, .8
    kwargs = dict(clip=clip, qp=50, codec="h265", protection=mask,
                  source_predictions=[source, source], anchor_predictions=[anchor, anchor], anchor_bpp=.123)
    context = build_utility_context(**kwargs)
    assert context.shape == (CONTEXT_DIM,)
    np.testing.assert_allclose(np.expm1(context[-1]), .123)
    permutation = np.random.default_rng(26).permutation(400)
    kwargs.update(source_predictions=[source[permutation]] * 2, anchor_predictions=[anchor[permutation]] * 2)
    np.testing.assert_allclose(build_utility_context(**kwargs), context, atol=1e-7, rtol=1e-7)
    kwargs["anchor_bpp"] = 0.
    with pytest.raises(ValueError):
        build_utility_context(**kwargs)


def fixture():
    ids = train_ids(12)
    context = np.zeros((len(ids) * 2, CONTEXT_DIM))
    context[:, 0], context[:, 2:4], context[:, 4] = 50 / 51, .6, .5
    context[1::2, 1] = 1.
    context[:, -1] = np.log1p(.1)
    context[:, 5] = np.repeat(np.linspace(0, 1, len(ids)), 2)
    safety = np.ones((len(context), 4))
    safety[:, 3] = 0.
    rates = np.tile(np.log([.95, .96, .97, .1]), (len(context), 1))
    return context, safety, rates, [source for source in ids for _ in range(2)],


def test_source_cv_fit_loader_and_proposal_origin_are_auditable():
    context, safety, rates, ids = fixture()
    names = ("identity", "a", "b", "c", "unsafe")
    model, report = fit_utility_model(context, safety, rates, ids, names)
    assert report["scope"].startswith("source-blocked TRAIN")
    assert report["selected_oof"]["feasible_records"] == len(context)
    assert len(report["by_codec_qp"]) == 2
    assert len(report["oof_folds"]) == 4
    # All source conditions have identical utility, so the fixed tie rule selects
    # prior-only and the origin must not be credited to learned residuals.
    assert model.learned_mix == 0.
    details = model.proposal_details(context[0])
    assert details["prior_only"] and details["origin"] == "TRAIN_group_prior"
    assert model.rank(context[0]) == [1, 2, 3]
    assert model.group_static_action_order(context[0]) == [1, 2, 3]
    state = model.checkpoint_state()
    restored = load_utility_preprocessor(state, action_names=names).to("cpu").eval()
    assert restored.rank(context[0]) == model.rank(context[0])
    state["schema"] = "adaptive-vcm-ranking-v4"
    with pytest.raises(ValueError):
        load_utility_preprocessor(state, action_names=names)
    with pytest.raises(ValueError):
        fit_utility_model(context, safety, rates, ["nontrain"] * len(ids), names)


def test_nan_coefficients_and_wrong_action_bank_fail_closed():
    model = UtilityRankPreprocessor(("identity", "a", "b", "c"))
    state = model.checkpoint_state()
    with pytest.raises(ValueError):
        load_utility_preprocessor(state, action_names=("identity", "a", "c", "b"))
    state["model"]["scale"][0] = torch.nan
    with pytest.raises(ValueError):
        load_utility_preprocessor(state, action_names=model.action_names)


def test_marginal_target_rewards_only_saving_beyond_actual_guarded_controls():
    safety = np.array([[1., 1., 0., 1.]])
    rate = np.log(np.array([[.95, .90, .10, .995]]))
    baseline = np.log(np.array([.94]))
    np.testing.assert_allclose(_utility(safety, rate, .01, baseline), [[0., .04, 0., 0.]])
    context, safe, rates, ids = fixture()
    model, report = fit_utility_model(context, safe, rates, ids,
        ("identity", "a", "b", "c", "unsafe"), baseline_log_rate=np.log(np.full(len(context), .94)))
    assert report["utility_target_scope"] == "marginal_saving_beyond_guarded_controls"
    assert model.checkpoint_state()["utility_target_scope"] == report["utility_target_scope"]
    assert report["bank_oracle"]["saved_bytes"] == 0.
