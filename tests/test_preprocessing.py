import numpy as np
import pytest

from adaptive_vcm.preprocessing import (action_protection, boxes_to_mask, feather,
                                         make_candidates, suppress, validate_clip)


def clip(seed=17, t=4, h=48, w=64):
    return np.random.default_rng(seed).integers(0, 256, (t, h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("task", ["ar", "od"])
@pytest.mark.parametrize("qp", [30, 50])
def test_candidates_are_bounded_deterministic_and_source_is_unchanged(task, qp):
    source = clip()
    original = source.copy()
    mask = np.ones(source.shape[1:3], np.float32)
    first = make_candidates(source, mask, task, qp)
    second = make_candidates(source, mask, task, qp)
    assert first[0].name == "identity"
    assert len({c.name for c in first}) == len(first)
    for a, b in zip(first, second):
        assert a.clip.dtype == np.uint8
        assert a.clip.shape[0] == len(source)
        np.testing.assert_array_equal(a.clip, b.clip)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("temporal", [0, .5, 1])
def test_exact_semantic_core_is_preserved(temporal):
    source = clip()
    mask = np.zeros(source.shape[1:3], np.float32)
    mask[8:32, 16:48] = 1
    result = suppress(source, mask, sigma=4, strength=1, temporal=temporal)
    np.testing.assert_array_equal(result[:, 8:32, 16:48], source[:, 8:32, 16:48])
    assert np.any(result[:, :8] != source[:, :8])


def test_static_motion_has_no_spurious_protection_and_real_motion_is_protected():
    source = np.repeat(clip(t=1), 3, axis=0)
    np.testing.assert_array_equal(action_protection(source), 0)
    source[1, 10:25, 20:35] = 0
    protection = action_protection(source)
    assert protection[15, 25] == 1


def test_cut_resets_temporal_reuse():
    source = np.stack([np.zeros((48, 64, 3), np.uint8), np.full((48, 64, 3), 255, np.uint8)])
    result = suppress(source, np.zeros((48, 64)), sigma=2, strength=1, temporal=1)
    np.testing.assert_array_equal(result, source)


def test_temporal_filter_is_causal_and_reduces_static_noise():
    source = np.clip(128 + np.random.default_rng(1).normal(0, 2, (4, 48, 64, 3)), 0, 255).astype(np.uint8)
    mask = np.zeros((48, 64))
    temporal = suppress(source, mask, sigma=1, strength=.3, temporal=.5)
    spatial = suppress(source, mask, sigma=1, strength=.3)
    assert np.var(np.diff(temporal.astype(float), axis=0)) < np.var(np.diff(spatial.astype(float), axis=0))
    changed = source.copy()
    changed[3] = 255
    np.testing.assert_array_equal(suppress(changed, mask, sigma=1, strength=.3, temporal=.5)[:3], temporal[:3])


def test_mask_outward_alignment_halo_and_feather():
    mask = boxes_to_mask(48, 64, np.array([[17, 17, 31, 31]]), halo=0, grid=16, radius=4)
    np.testing.assert_array_equal(mask[16:32, 16:32], 1)
    assert 0 < mask[15, 20] < 1
    assert mask[5, 5] == 0
    assert boxes_to_mask(48, 64, np.empty((0, 4))).sum() == 0
    assert feather(np.ones((5, 5), bool)).min() == 1


@pytest.mark.parametrize("bad", [np.zeros((0, 8, 8, 3), np.uint8), np.zeros((2, 8, 8, 3)), np.zeros((8, 8, 3), np.uint8)])
def test_invalid_clip_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_clip(bad)


@pytest.mark.parametrize("setting", [{"strength": float("nan")}, {"temporal": -1}, {"sigma": 0}])
def test_invalid_filter_is_rejected(setting):
    options = {"strength": .5, "temporal": .1, "sigma": 1}
    options.update(setting)
    with pytest.raises(ValueError):
        suppress(clip(), np.zeros((48, 64)), **options)
