"""Full-resolution renderer behavior and real elementary-stream measurements."""
import importlib

import numpy as np
import pytest
import torch

from adaptive_vcm.codec import StandardCodec, locate_ffmpeg


def module():
    return importlib.import_module("adaptive_vcm.motion_learned")


def support(clip, core=True, cuts=None):
    mask = np.zeros(clip.shape[:3], np.float32)
    if core:
        mask[:, :2, :3] = 1
    return {"protection": mask, "motion": np.zeros_like(mask),
            "cuts": np.array([True] + [False] * (len(clip) - 1)) if cuts is None else np.array(cuts, bool),
            "metadata": {}}


@pytest.mark.parametrize("shape,task", [((1, 3, 3, 17, 21), "ar"), ((2, 3, 1, 1, 7), "od")])
def test_neural_renderer_preserves_geometry_bounds_exact_cores_and_real_gradients(shape, task):
    # Detaching maps or bypassing the network would lose parameter gradients.
    torch.manual_seed(61)
    torch.set_num_threads(2)
    model = module().MotionAwarePreprocessor(8, task)
    source = torch.rand(shape, requires_grad=True)
    mask = torch.zeros(shape[0], 1, *shape[2:])
    mask[..., :1, :2] = 1
    rendered, aux = model(source, torch.full((shape[0],), 45), torch.zeros(shape[0]), mask, return_aux=True)
    assert rendered.shape == shape
    assert rendered.min() >= 0 and rendered.max() <= 1
    torch.testing.assert_close(rendered[..., :1, :2], source[..., :1, :2], rtol=0, atol=0)
    assert aux["alpha"].shape == mask.shape
    assert aux["mixture"].shape == (shape[0], 4, *shape[2:])
    torch.testing.assert_close(aux["mixture"].sum(1), torch.ones(shape[0], *shape[2:]))
    (rendered - source).square().mean().backward()
    assert torch.isfinite(source.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_alpha_smoothing_is_causal_and_resets_exactly_at_cut():
    # Reusing alpha across cuts makes the third frame depend on the old scene.
    torch.manual_seed(7)
    model = module().MotionAwarePreprocessor(8)
    source = torch.rand(1, 3, 3, 11, 13)
    mask = torch.zeros(1, 1, 3, 11, 13)
    motion = torch.zeros_like(mask)
    cuts = torch.tensor([[True, False, True]])
    _, aux = model(source, torch.tensor([40]), torch.tensor([0]), mask, motion, cuts, return_aux=True)
    torch.testing.assert_close(aux["alpha"][:, :, 1], .5 * (aux["raw_alpha"][:, :, 0] + aux["raw_alpha"][:, :, 1]))
    torch.testing.assert_close(aux["alpha"][:, :, 2], aux["raw_alpha"][:, :, 2], rtol=0, atol=0)
    changed = source.clone()
    changed[:, :, 2] = 0
    _, changed_aux = model(changed, torch.tensor([40]), torch.tensor([0]), mask, motion, cuts, return_aux=True)
    torch.testing.assert_close(aux["alpha"][:, :, :2], changed_aux["alpha"][:, :, :2], rtol=0, atol=0)


def test_strength_zero_and_full_protection_are_exact_identity():
    # An unmasked DC expert or nonzero residual at scale zero edits identity.
    model = module().MotionAwarePreprocessor(8)
    source = torch.rand(1, 3, 2, 9, 15)
    mask = torch.ones(1, 1, 2, 9, 15)
    torch.testing.assert_close(model(source, torch.tensor([45]), torch.tensor([1]), mask), source, rtol=0, atol=0)
    torch.testing.assert_close(model(source, torch.tensor([45]), torch.tensor([1]), mask * 0, strength_scale=0), source, rtol=0, atol=0)


def test_qp_and_codec_conditioning_change_neural_pixel_maps():
    # Dropping either condition channel prevents conditional learned proposals.
    torch.manual_seed(9)
    model = module().MotionAwarePreprocessor(8)
    source = torch.rand(1, 3, 2, 9, 15)
    mask = torch.zeros(1, 1, 2, 9, 15)
    a = model(source, torch.tensor([30]), torch.tensor([0]), mask)
    b = model(source, torch.tensor([50]), torch.tensor([0]), mask)
    c = model(source, torch.tensor([30]), torch.tensor([1]), mask)
    assert not torch.equal(a, b)
    assert not torch.equal(a, c)


def test_twelve_profiles_preserve_odd_geometry_and_exact_semantic_cores():
    # A crop, downsample or omitted protection changes coordinates/core pixels.
    clip = np.random.default_rng(8).integers(0, 256, (3, 17, 21, 3), dtype=np.uint8)
    candidates = module().profile_candidates(clip, support(clip), "ar", 45)
    assert len(candidates) == len(module().PROFILE_NAMES) == 12
    assert tuple(candidate.name for candidate in candidates) == module().PROFILE_NAMES
    assert len(set(candidate.name for candidate in candidates)) == 12
    for candidate in candidates:
        assert candidate.clip.shape == clip.shape
        assert candidate.clip.dtype == np.uint8
        np.testing.assert_array_equal(candidate.clip[:, :2, :3], clip[:, :2, :3])
        assert not np.array_equal(candidate.clip[:, 4:, 4:], clip[:, 4:, 4:])


def test_background_dc_excludes_cores_and_stops_at_scene_cuts():
    # Pooling over protected pixels or the entire clip gives the wrong DC.
    clip = np.concatenate([np.full((2, 5, 7, 3), 10, np.uint8), np.full((2, 5, 7, 3), 210, np.uint8)])
    clip[:2, :2, :3] = 250
    clip[2:, :2, :3] = 0
    maps = support(clip, cuts=[True, False, True, False])
    dc = {c.name: c.clip for c in module().profile_candidates(clip, maps, "ar", 50)}["motion_background_dc_100"]
    np.testing.assert_array_equal(dc[:2, 2:, :], np.full((2, 3, 7, 3), 10, np.uint8))
    np.testing.assert_array_equal(dc[2:, 2:, :], np.full((2, 3, 7, 3), 210, np.uint8))
    np.testing.assert_array_equal(dc[:, :2, :3], clip[:, :2, :3])


def test_block_lowpass_keeps_eight_pixel_cells_at_odd_boundaries():
    # Resizing the pooled 3x3 grid directly to 17x23 stretches block geometry.
    yy, xx = np.mgrid[:17, :23]
    clip = np.repeat((8 * yy + 2 * xx)[None, ..., None], 3, axis=-1).astype(np.uint8)
    candidate = {c.name: c.clip for c in module().profile_candidates(clip, support(clip, core=False), "ar", 45)}["motion_block_lowpass_100"]
    # Rightmost x-cell contains 16..22,22: mean x=19.375.
    expected = np.array([[35] * 8 + [51] * 8 + [67] * 7] * 8
                        + [[99] * 8 + [115] * 8 + [131] * 7] * 8
                        + [[135] * 8 + [151] * 8 + [167] * 7], np.uint8)
    np.testing.assert_array_equal(candidate[0, ..., 0], expected)


def test_od_gaussian_expert_suppresses_more_background_detail_than_ar():
    # Sharing AR's mild blur with OD removes the intended stronger OD expert.
    clip = np.random.default_rng(39).integers(0, 256, (1, 33, 47, 3), dtype=np.uint8)
    maps = support(clip, core=False)
    ar = {c.name: c.clip for c in module().profile_candidates(clip, maps, "ar", 50)}["motion_strong_gaussian_100"]
    od = {c.name: c.clip for c in module().profile_candidates(clip, maps, "od", 50)}["motion_strong_gaussian_100"]
    assert od[:, 8:-8, 8:-8].astype(np.float32).var() < .6 * ar[:, 8:-8, 8:-8].astype(np.float32).var()


def test_all_protected_profiles_and_single_frame_od_are_source_exact():
    clip = np.random.default_rng(81).integers(0, 256, (1, 13, 19, 3), dtype=np.uint8)
    maps = support(clip)
    maps["protection"][:] = 1
    for candidate in module().profile_candidates(clip, maps, "od", 45):
        np.testing.assert_array_equal(candidate.clip, clip)


def test_od_profiles_keep_every_small_box_without_a_selected_roi():
    # A single density/crop rectangle would miss one separated detector core.
    clip = np.random.default_rng(53).integers(0, 256, (1, 19, 27, 3), dtype=np.uint8)
    maps = support(clip, core=False)
    maps["protection"][:, 1:3, 2:4] = 1
    maps["protection"][:, 15:17, 23:26] = 1
    for candidate in module().profile_candidates(clip, maps, "od", 50):
        np.testing.assert_array_equal(candidate.clip[:, 1:3, 2:4], clip[:, 1:3, 2:4])
        np.testing.assert_array_equal(candidate.clip[:, 15:17, 23:26], clip[:, 15:17, 23:26])


@pytest.mark.parametrize("cuts", [np.array([False, False]), np.array([1., 0.]), np.array([True])])
def test_invalid_cuts_do_not_silently_pool_segments(cuts):
    # Coercing malformed cuts silently drops a segment boundary.
    clip = np.zeros((2, 7, 9, 3), np.uint8)
    maps = support(clip, cuts=cuts)
    maps["cuts"] = cuts
    with pytest.raises(ValueError):
        module().profile_candidates(clip, maps, "ar", 40)
    with pytest.raises(ValueError):
        module().MotionAwarePreprocessor(8)(torch.rand(1, 3, 2, 7, 9), torch.tensor([40]), torch.tensor([0]),
                                           torch.zeros(1, 1, 2, 7, 9), cuts=torch.from_numpy(cuts)[None])


@pytest.mark.parametrize("field,value", [("protection", float("nan")), ("motion", float("inf")), ("motion", -1.)])
def test_nonfinite_or_out_of_range_support_is_rejected(field, value):
    clip = np.zeros((1, 8, 10, 3), np.uint8)
    maps = support(clip)
    maps[field][:] = value
    with pytest.raises(ValueError):
        module().profile_candidates(clip, maps, "od", 40)
    source = torch.zeros(1, 3, 1, 8, 10)
    protection = torch.from_numpy(maps["protection"])[None, None]
    motion = torch.from_numpy(maps["motion"])[None, None]
    with pytest.raises(ValueError):
        module().MotionAwarePreprocessor(8, "od")(source, torch.tensor([40]), torch.tensor([0]), protection, motion)


@pytest.mark.parametrize("qp,codec,scale", [(float("nan"), 0, 1), (40, 2, 1), (40, 0, float("inf")), (40, 0, -1)])
def test_invalid_neural_conditions_are_rejected(qp, codec, scale):
    with pytest.raises(ValueError):
        module().MotionAwarePreprocessor(8)(torch.rand(1, 3, 1, 5, 7), torch.tensor([qp]), torch.tensor([codec]),
                                           torch.zeros(1, 1, 1, 5, 7), strength_scale=scale)


@pytest.mark.codec
@pytest.mark.parametrize("codec_name,qp", [("h264", 45), ("h264", 50), ("h265", 45), ("h265", 50)])
def test_real_high_qp_stream_bytes_save_on_editable_noise_with_foreground_proxy(codec_name, qp):
    # An identity bank or fake rate proxy cannot pass these actual stream bytes.
    if locate_ffmpeg() is None:
        pytest.skip("FFmpeg not installed")
    clip = np.random.default_rng(777).integers(20, 236, (6, 64, 80, 3), dtype=np.uint8)
    clip[:, 20:44, 24:52] = np.array([160, 45, 75], np.uint8)
    maps = support(clip, core=False)
    maps["protection"][:, 20:44, 24:52] = 1
    candidate = {c.name: c for c in module().profile_candidates(clip, maps, "ar", qp)}["motion_background_dc_100"]
    np.testing.assert_array_equal(candidate.clip[:, 20:44, 24:52], clip[:, 20:44, 24:52])
    codec = StandardCodec(codec_name, qp, preset="ultrafast", threads=1)
    original = codec.roundtrip(clip)
    edited = codec.roundtrip(candidate.clip)
    assert edited.coded_bytes == len(edited.data) > 0
    assert edited.decoded.shape == clip.shape
    assert edited.coded_bytes < original.coded_bytes
    core = (slice(None), slice(24, 40), slice(28, 48))
    source_core = clip[core].astype(np.float32)
    original_mse = np.square(original.decoded[core].astype(np.float32) - source_core).mean()
    edited_mse = np.square(edited.decoded[core].astype(np.float32) - source_core).mean()
    # A source-exact core does not imply an identical lossy decode: use a
    # declared, simple foreground-color proxy and report the actual MSE too.
    decoded_color = edited.decoded[core].astype(np.float32).mean((0, 1, 2))
    assert decoded_color[0] > decoded_color[1] + 40
    assert decoded_color[0] > decoded_color[2] + 40
    print(f"{codec_name} QP{qp}: bytes={original.coded_bytes}->{edited.coded_bytes}; foreground_mse={original_mse:.3f}->{edited_mse:.3f}")
