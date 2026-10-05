import importlib
import importlib.util

import numpy as np
import pytest

from adaptive_vcm.codec import StandardCodec, locate_ffmpeg, reference_bpp
from adaptive_vcm.stabilized_bank import ACTION_NAMES as V26_ACTION_NAMES, build_stabilized_bank


def anchor_bank():
    # Feature absence is an assertion failure, not a collection/import error.
    assert importlib.util.find_spec("adaptive_vcm.anchor_bank") is not None, "anchor reconstruction bank is missing"
    return importlib.import_module("adaptive_vcm.anchor_bank")


def fixture():
    source = np.full((4, 16, 24, 3), 100, np.uint8)
    anchor = np.full_like(source, 140)
    mask = np.zeros(source.shape[1:3], np.float32)
    mask[4:8, 8:16] = 1
    mask[0, 0] = .5
    return source, anchor, mask


def test_registered_blends_shrink_source_to_anchor_residual_by_known_amounts():
    module = anchor_bank()
    source, anchor, mask = fixture()
    actions = module.build_anchor_actions(source, mask, 50, anchor)
    assert tuple(c.name for c in actions) == (
        "uniform_anchor_25", "uniform_anchor_50", "uniform_anchor_75", "uniform_anchor_100",
        "core_anchor_25", "core_anchor_50", "core_anchor_75", "core_anchor_100",
    )
    for candidate, expected in zip(actions[:4], [110, 120, 130, 140]):
        np.testing.assert_array_equal(candidate.clip, np.full_like(source, expected))
    # A half-protected pixel attenuates the blend; a fully protected pixel stays source-exact.
    for candidate, expected in zip(actions[4:], [105, 110, 115, 120]):
        assert np.all(candidate.clip[:, 0, 0] == expected)
        np.testing.assert_array_equal(candidate.clip[:, mask == 1], source[:, mask == 1])


@pytest.mark.parametrize("qp", [0, 30, 40, 45, 50, 51])
def test_old_registered_prefix_preserves_order_and_pixels(qp):
    module = anchor_bank()
    source, anchor, mask = fixture()
    old = build_stabilized_bank(source, mask, qp)
    new = module.build_anchor_bank(source, mask, qp, anchor)
    assert len(new) == 42
    assert tuple(c.name for c in new) == module.ACTION_NAMES
    assert module.ACTION_NAMES[:34] == V26_ACTION_NAMES
    for original, extended in zip(old, new):
        assert original.name == extended.name
        np.testing.assert_array_equal(original.clip, extended.clip)


def test_exact_anchor_full_blend_retains_color_geometry_and_owns_output_memory():
    module = anchor_bank()
    rng = np.random.default_rng(27)
    source = rng.integers(0, 256, (3, 16, 24, 3), dtype=np.uint8)
    anchor = rng.integers(0, 256, source.shape, dtype=np.uint8)
    mask = np.zeros(source.shape[1:3], np.float32)
    actions = module.build_anchor_actions(source, mask, 45, anchor)
    np.testing.assert_array_equal(actions[3].clip, anchor)
    for candidate in actions:
        assert candidate.clip.shape == source.shape
        assert candidate.clip.dtype == np.uint8
        assert not np.shares_memory(candidate.clip, source)
        assert not np.shares_memory(candidate.clip, anchor)
        # Every channel remains inside the source/anchor convex hull, including 0/255.
        assert np.all(candidate.clip >= np.minimum(source, anchor))
        assert np.all(candidate.clip <= np.maximum(source, anchor))


def test_input_buffers_are_unchanged_and_core_full_mask_is_exact_identity():
    module = anchor_bank()
    source, anchor, mask = fixture()
    before = source.copy(), anchor.copy(), mask.copy()
    first = module.build_anchor_actions(source, mask, 50, anchor)
    second = module.build_anchor_actions(source, mask, 50, anchor)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.clip, right.clip)
    for original, current in zip(before, [source, anchor, mask]):
        np.testing.assert_array_equal(original, current)
    for candidate in module.build_anchor_actions(source, np.ones_like(mask), 50, anchor)[4:]:
        np.testing.assert_array_equal(candidate.clip, source)


@pytest.mark.parametrize("builder", ["build_anchor_actions", "build_anchor_bank"])
@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype", "layout"])
def test_missing_or_invalid_anchor_fails_closed_for_both_builders(builder, invalid):
    module = anchor_bank()
    source, anchor, mask = fixture()
    bad = {"missing": None, "shape": anchor[:2], "dtype": anchor.astype(np.float32),
           "layout": anchor[..., :1]}[invalid]
    with pytest.raises(ValueError, match="anchor"):
        getattr(module, builder)(source, mask, 50, bad)


@pytest.mark.parametrize("qp", [True, -1, 52, 45.5])
def test_invalid_qp_cannot_render_anchor_actions(qp):
    module = anchor_bank()
    source, anchor, mask = fixture()
    with pytest.raises(ValueError, match="QP"):
        module.build_anchor_actions(source, mask, qp, anchor)


@pytest.mark.parametrize("invalid", ["shape", "nan", "negative", "large"])
def test_invalid_protection_cannot_render_anchor_actions(invalid):
    module = anchor_bank()
    source, anchor, mask = fixture()
    if invalid == "shape":
        mask = mask[:2]
    else:
        mask[0, 0] = {"nan": np.nan, "negative": -.01, "large": 1.01}[invalid]
    with pytest.raises(ValueError, match="protection"):
        module.build_anchor_actions(source, mask, 50, anchor)


@pytest.mark.parametrize("invalid", ["missing", "dtype", "empty", "layout"])
def test_invalid_source_fails_before_blending(invalid):
    module = anchor_bank()
    source, anchor, mask = fixture()
    source = {"missing": None, "dtype": source.astype(np.float32), "empty": source[:0],
              "layout": source[..., :1]}[invalid]
    with pytest.raises(ValueError, match="source|uint8|empty"):
        module.build_anchor_actions(source, mask, 50, anchor)


@pytest.mark.codec
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_codec_roundtrip_accounts_for_real_stream_headers(codec):
    module = anchor_bank()
    if locate_ffmpeg() is None:
        pytest.skip("FFmpeg unavailable")
    source, _, mask = fixture()
    original = StandardCodec(codec, 45).roundtrip(source)
    candidate = module.build_anchor_actions(source, mask, 45, original.decoded)[5]
    encoded = StandardCodec(codec, 45).roundtrip(candidate.clip)
    assert encoded.decoded.shape == source.shape
    assert encoded.coded_bytes == len(encoded.data) > 0
    assert reference_bpp(encoded.coded_bytes, source.shape) == 8 * len(encoded.data) / (4 * 16 * 24)


def test_rate_probe_does_not_credit_subthreshold_byte_changes_as_guarded_capacity():
    from scripts import anchor_probe
    summarize = getattr(anchor_probe, "summarize", None)
    assert callable(summarize), "probe must distinguish raw changes from 1% guarded rate feasibility"
    results = [
        dict(same_geometry=True, source_edit_mae=9., coded_bytes=999),
        dict(same_geometry=True, source_edit_mae=10., coded_bytes=995),
        dict(same_geometry=False, source_edit_mae=0., coded_bytes=100),
        dict(same_geometry=True, source_edit_mae=11., coded_bytes=100),
    ]
    assert summarize(1000, results) == dict(mild_saving_actions=0, best_mild_bytes=995,
                                            best_guarded_mild_bytes=1000)
    results.append(dict(same_geometry=True, source_edit_mae=10., coded_bytes=990))
    assert summarize(1000, results) == dict(mild_saving_actions=1, best_mild_bytes=990,
                                            best_guarded_mild_bytes=990)
