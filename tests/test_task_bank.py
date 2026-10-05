import numpy as np
import pytest

from adaptive_vcm.codec import StandardCodec, locate_ffmpeg, reference_bpp
from adaptive_vcm.preprocessing import make_candidates
from adaptive_vcm.task_bank import ACTION_NAMES, CORE_ACTIONS, build_task_bank


def fixture():
    rng = np.random.default_rng(25)
    base = rng.integers(40, 210, (64, 96, 3), dtype=np.uint8)
    clip = np.stack([np.clip(base.astype(float) + rng.normal(0, 2, base.shape), 0, 255).round().astype(np.uint8)
                     for _ in range(8)])
    clip[:, 24:40, 40:56] = [232, 48, 16]
    mask = np.zeros((64, 96), np.float32)
    mask[20:44, 36:60] = 1
    return clip, mask


def test_fixed_bank_is_deterministic_and_does_not_mutate_inputs():
    clip, mask = fixture()
    source, protection = clip.copy(), mask.copy()
    first, second = build_task_bank(clip, mask, 50), build_task_bank(clip, mask, 50)
    assert tuple(c.name for c in first) == ACTION_NAMES
    assert len(first) == len(set(ACTION_NAMES)) == 18
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.clip, b.clip)
        assert a.clip.dtype == np.uint8 and a.clip.shape[0] == len(clip)
        assert a.clip.shape[-1] == 3 and min(a.clip.shape[1:3]) >= 2
        assert not a.clip.shape[1] % 2 and not a.clip.shape[2] % 2
        assert not np.shares_memory(a.clip, clip)
        assert not np.shares_memory(a.clip, b.clip)
    np.testing.assert_array_equal(first[0].clip, source)
    np.testing.assert_array_equal(clip, source)
    np.testing.assert_array_equal(mask, protection)


def test_core_actions_are_source_exact_and_full_protection_yields_identity():
    clip, mask = fixture()
    for candidate in build_task_bank(clip, mask, 50):
        if candidate.name in CORE_ACTIONS:
            assert candidate.clip.shape == clip.shape
            np.testing.assert_array_equal(candidate.clip[:, mask == 1], clip[:, mask == 1])
    bank = build_task_bank(clip, np.ones_like(mask), 50)
    for candidate in bank:
        if candidate.name in CORE_ACTIONS:
            np.testing.assert_array_equal(candidate.clip, clip)


def test_temporal_actions_reset_and_do_not_carry_recursive_state_across_cuts():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 100, (20, 30, 3), dtype=np.uint8)
    b = rng.integers(150, 250, (20, 30, 3), dtype=np.uint8)
    clip = np.stack([a, a, b, b])
    mask = np.zeros(clip.shape[1:3])
    whole = build_task_bank(clip, mask, 50)
    after = build_task_bank(clip[2:], mask, 50)
    for full, new in zip(whole, after):
        np.testing.assert_array_equal(full.clip[2:], new.clip)


def test_bank_is_genuinely_distinct_from_existing_analytic_controls():
    rng = np.random.default_rng(30)
    clip = rng.integers(0, 256, (3, 128, 128, 3), np.uint8)
    mask = np.zeros((128, 128), np.float32)
    bank = build_task_bank(clip, mask, 45)
    controls = make_candidates(clip, mask, 'ar', 45)
    for candidate in bank[1:]:
        assert not any(candidate.clip.shape == c.clip.shape and np.array_equal(candidate.clip, c.clip)
                       for c in controls)
    # Different fixed actions can coincide for a particular source (e.g., no
    # temporal reuse at cuts or a zero protection mask). They are not required
    # to create an edit when no corresponding signal is present.


def test_soft_residual_has_bounded_per_channel_movement_and_qp_response():
    clip, mask = fixture()
    bank30 = {c.name: c.clip for c in build_task_bank(clip, mask, 30)}
    bank50 = {c.name: c.clip for c in build_task_bank(clip, mask, 50)}
    for name, limit in [('uniform_detail_soft', 8), ('uniform_detail_medium', 16)]:
        delta = np.abs(bank50[name].astype(int) - clip.astype(int))
        assert delta.max() <= limit
        assert delta.sum() > np.abs(bank30[name].astype(int) - clip.astype(int)).sum()


@pytest.mark.parametrize('qp', [True, 40.5, -1, 52])
def test_invalid_qp_is_rejected(qp):
    clip, mask = fixture()
    with pytest.raises(ValueError, match='QP'):
        build_task_bank(clip, mask, qp)


@pytest.mark.parametrize('kind', ['shape', 'nan', 'negative', 'above_one'])
def test_invalid_protection_is_rejected(kind):
    clip, mask = fixture()
    if kind == 'shape':
        mask = mask[:10]
    else:
        mask[0, 0] = {'nan': np.nan, 'negative': -.01, 'above_one': 1.01}[kind]
    with pytest.raises(ValueError, match='protection'):
        build_task_bank(clip, mask, 40)


@pytest.mark.codec
@pytest.mark.parametrize('codec', ['h264', 'h265'])
@pytest.mark.parametrize('qp', [40, 45, 50])
def test_actual_high_qp_byte_opportunity_and_original_denominator(codec, qp):
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    clip, mask = fixture()
    # Make block texture resolvable at QP50; the tiny 64x96 fixture is header
    # dominated in H.265 and has no guaranteed >=1% preprocessing opportunity.
    clip = np.repeat(np.repeat(clip, 2, axis=1), 2, axis=2)
    mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    encoder = StandardCodec(codec, qp)
    bank = build_task_bank(clip, mask, qp)
    raw = encoder.roundtrip(bank[0].clip)
    # Synthetic encoder check only. Feasibility for frozen action teachers is
    # tested in the actual TRAIN/DEV experiment; this is no Top-1 assertion.
    results = [encoder.roundtrip(c.clip) for c in bank[1:6]]
    assert any(r.coded_bytes <= .99 * raw.coded_bytes for r in results)
    for candidate, result in zip(bank[1:6], results):
        assert result.decoded.shape == candidate.clip.shape
        assert reference_bpp(result.coded_bytes, clip.shape) == 8 * result.coded_bytes / np.prod(clip.shape[:3])
