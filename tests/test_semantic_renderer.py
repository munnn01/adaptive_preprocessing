"""V29 renderer contracts: fixed/neural parity and task-specific background edits."""
import math

import numpy as np
import pytest
import torch

from adaptive_vcm.motion_learned import MotionAwarePreprocessor, profile_candidates


@pytest.fixture(autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def support(clip, motion=0, cuts=None):
    protection = np.zeros(clip.shape[:3], np.float32)
    return {"protection": protection,
            "motion": np.full_like(protection, motion),
            "cuts": np.array(cuts if cuts is not None else [True] + [False] * (len(clip) - 1), bool)}


def profiles(clip, maps, task="ar", variant="a"):
    return {c.name: c.clip for c in profile_candidates(clip, maps, task, 45, variant=variant)}


def tensors(clip, maps):
    source = torch.from_numpy(clip.astype(np.float32) / 255).permute(3, 0, 1, 2)[None]
    protection = torch.from_numpy(maps["protection"])[None, None]
    motion = torch.from_numpy(maps["motion"])[None, None]
    cuts = torch.from_numpy(maps["cuts"])[None]
    return source, protection, motion, cuts


@pytest.mark.parametrize("variant", ["a", "b", "c"])
def test_motion_attenuation_is_shared_by_fixed_and_neural_rendering(variant):
    # Omitting motion from fixed profiles makes learned/profile supervision disagree.
    clip = np.repeat(np.array([0, 200] * 4, np.uint8)[None, None, :, None], 3, axis=-1)
    maps = support(clip, motion=1)
    if variant == "b":
        name = "motion_block_lowpass_100"
    else:
        name = "motion_background_dc_100"
    result = profiles(clip, maps, variant=variant)[name]
    np.testing.assert_array_equal(result[0, 0, :, 0], [25, 175] * 4)
    model = MotionAwarePreprocessor(4, "ar", variant=variant)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias[0] = 0
    source, protection, motion, cuts = tensors(clip, maps)
    _, aux = model(source, torch.tensor([45]), torch.tensor([0]), protection, motion, cuts, return_aux=True)
    torch.testing.assert_close(aux["alpha"], torch.full_like(protection, .125))


def test_legacy_default_retains_unattenuated_profiles_and_checkpoint_schema():
    # Accidentally enabling V29 by default changes saved V28 candidate pixels.
    clip = np.repeat(np.array([[[[0], [200]]]], np.uint8), 3, axis=-1)
    legacy = {c.name: c.clip for c in profile_candidates(clip, support(clip, 1), "ar", 45)}
    np.testing.assert_array_equal(legacy["motion_background_dc_100"][0, 0, :, 0], [100, 100])
    assert MotionAwarePreprocessor(4).schema == "adaptive-vcm-motion-v7"


def test_b_ar_dc_uses_stationary_background_and_keeps_dynamic_pixels():
    # A plain segment mean would pollute stationary background with moving colors.
    clip = np.zeros((4, 1, 3, 3), np.uint8)
    clip[:2, :, 0] = 20
    clip[2:, :, 0] = 100
    clip[:, :, 1] = 200  # Dynamic editable content.
    clip[:, :, 2] = 250  # Protected source content.
    maps = support(clip, cuts=[True, False, True, False])
    maps["motion"][:, :, 1] = 1
    maps["protection"][:, :, 2] = 1
    dc = profiles(clip, maps, variant="b")["motion_background_dc_100"]
    np.testing.assert_array_equal(dc, clip)


def test_b_ar_stationary_dc_stops_at_scene_cuts():
    # Cross-cut pooling replaces each scene by an average of unrelated scenes.
    clip = np.repeat(np.array([10, 30, 150, 170], np.uint8)[:, None, None, None], 3, axis=-1)
    result = profiles(clip, support(clip, cuts=[True, False, True, False]), variant="b")
    np.testing.assert_array_equal(result["motion_background_dc_100"][:, 0, 0, 0], [20, 20, 160, 160])


def test_b_ar_stationary_weights_limit_partially_moving_color_leakage():
    # Uniform or linear weights let moving colors dominate the stationary mean.
    clip = np.repeat(np.array([[[[20], [200]]]], np.uint8), 3, axis=-1)
    maps = support(clip)
    maps["motion"][0, 0, 1] = .5
    dc = profiles(clip, maps, variant="b")["motion_background_dc_100"]
    # Mean=(20+.25*200)/1.25=56; second expert pixel=164, alpha=.625.
    np.testing.assert_array_equal(dc[0, 0, :, 0], [56, 178])


def test_b_od_dc_retains_half_luma_detail_with_shared_editable_chroma():
    # Flat RGB DC erases all luma detail; per-pixel chroma leaves color noise intact.
    clip = np.array([[[[40, 40, 40], [200, 200, 200], [255, 0, 0]]]], np.uint8)
    maps = support(clip)
    maps["protection"][0, 0, 2] = 1
    result = profiles(clip, maps, "od", "b")["motion_background_dc_100"]
    np.testing.assert_array_equal(result[0, 0], [[80, 80, 80], [160, 160, 160], [255, 0, 0]])
    # The RGB channel differences are the editable mean's chroma.
    color_clip = np.array([[[[60, 20, 20], [100, 140, 140]]]], np.uint8)
    color = profiles(color_clip, support(color_clip), "od", "b")["motion_background_dc_100"]
    np.testing.assert_allclose(color[0, 0], [[56, 56, 56], [104, 104, 104]], atol=1)


def test_b_expert_logits_smooth_causally_and_reset_at_cut():
    # Independent frame logits flicker; using old-scene state contaminates a cut.
    torch.manual_seed(17)
    model = MotionAwarePreprocessor(4, "ar", variant="b")
    source = torch.rand(1, 3, 3, 5, 7)
    mask = torch.zeros(1, 1, 3, 5, 7)
    motion = torch.full_like(mask, .4)
    args = (torch.tensor([45]), torch.tensor([0]))
    _, aux = model(source, *args, mask, motion, torch.tensor([[True, False, True]]), return_aux=True)
    _, independent = model(source, *args, mask, motion, torch.ones(1, 3, dtype=torch.bool), return_aux=True)
    raw = independent["expert_logits"]
    torch.testing.assert_close(aux["expert_logits"][:, :, 1], .7 * raw[:, :, 1] + .3 * raw[:, :, 0])
    torch.testing.assert_close(aux["expert_logits"][:, :, 2], raw[:, :, 2], rtol=0, atol=0)
    torch.testing.assert_close(aux["mixture"], aux["expert_logits"].softmax(1))
    changed = source.clone()
    changed[:, :, 2] = 0
    _, changed_aux = model(changed, *args, mask, motion, torch.tensor([[True, False, True]]), return_aux=True)
    torch.testing.assert_close(aux["mixture"][:, :, :2], changed_aux["mixture"][:, :, :2], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["a", "b", "c"])
@pytest.mark.parametrize("task", ["ar", "od"])
def test_neural_pure_experts_reach_every_registered_profile(variant, task):
    # Different motion gates, DC operators or strength semantics break teacher reachability.
    clip = np.random.default_rng(38).integers(0, 256, (2 if task == "ar" else 1, 9, 11, 3), np.uint8)
    maps = support(clip, motion=.35)
    maps["motion"][:, 2:5, 3:7] = 1
    maps["protection"][:, :2, :3] = 1
    source, protection, motion, cuts = tensors(clip, maps)
    model = MotionAwarePreprocessor(4, task, variant=variant)
    assert model.schema == "adaptive-vcm-semantic-v8"
    fixed = profile_candidates(clip, maps, task, 45, variant=variant)
    assert len(fixed) == 12
    for index, candidate in enumerate(fixed):
        strength = (.4, .75, 1.)[index % 3]
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(-80)
            model.head.bias[0] = math.log(strength / (1 - strength)) if strength < 1 else 80
            model.head.bias[1 + index // 3] = 80
            rendered = model(source, torch.tensor([45]), torch.tensor([0]), protection, motion, cuts)
        pixels = (rendered[0].permute(1, 2, 3, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        np.testing.assert_allclose(pixels.astype(float), candidate.clip.astype(float), atol=1)
        np.testing.assert_array_equal(pixels[:, :2, :3], clip[:, :2, :3])


@pytest.mark.parametrize("variant", ["a", "b", "c"])
def test_v29_auxiliary_losses_backpropagate_to_strength_and_experts(variant):
    # Detached mixture/alpha tensors silently defeat new curriculum supervision.
    torch.manual_seed(9)
    model = MotionAwarePreprocessor(4, "ar", variant=variant)
    source = torch.rand(1, 3, 2, 7, 9, requires_grad=True)
    mask = torch.zeros(1, 1, 2, 7, 9)
    mask[..., :2, :2] = 1
    result, aux = model(source, torch.tensor([45]), torch.tensor([1]), mask,
                        cuts=torch.tensor([[True, False]]), return_aux=True)
    loss = (aux["raw_alpha"] - .75).square().mean() - aux["mixture"][:, 3].log().mean()
    loss.backward()
    assert model.head.bias.grad[0].abs() > 0
    assert model.head.bias.grad[1:].abs().sum() > 0
    assert torch.isfinite(source.grad).all()
    torch.testing.assert_close(result[..., :2, :2], source[..., :2, :2], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["a", "b", "c"])
def test_unknown_full_protection_stays_source_exact_for_all_v29_candidates(variant):
    # Editing the fallback mask would alter unknown foreground despite the hard constraint.
    clip = np.random.default_rng(29).integers(0, 256, (1, 5, 7, 3), np.uint8)
    maps = support(clip)
    maps["protection"][:] = 1
    for candidate in profile_candidates(clip, maps, "od", 45, variant=variant):
        np.testing.assert_array_equal(candidate.clip, clip)
    source, protection, motion, cuts = tensors(clip, maps)
    model = MotionAwarePreprocessor(4, "od", variant=variant)
    torch.testing.assert_close(model(source, torch.tensor([45]), torch.tensor([1]), protection, motion, cuts),
                               source, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["", "d", 1])
def test_unknown_variants_are_rejected_instead_of_silently_using_v28(variant):
    # A typo in a registered branch must not evaluate a different renderer.
    with pytest.raises(ValueError, match="variant"):
        MotionAwarePreprocessor(4, variant=variant)
    clip = np.zeros((1, 2, 2, 3), np.uint8)
    with pytest.raises(ValueError, match="variant"):
        profile_candidates(clip, support(clip), "od", 45, variant=variant)
