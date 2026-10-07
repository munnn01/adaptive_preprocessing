"""V30 source-level decisions, canonical profile pixels, and context experts."""
import numpy as np
import pytest
import torch

from adaptive_vcm.conditional_learned import (
    CONDITIONAL_SCHEMA, ConditionalPreprocessor, canonical_target,
    context_expert, profile_candidates, profile_registry,
)
from adaptive_vcm.motion_learned import PROFILE_NAMES, profile_candidates as baseline_profiles


@pytest.fixture(autouse=True)
def bounded_threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


def maps(clip, protection=None, motion=0, cuts=None):
    shape = clip.shape[:3]
    return {"protection": np.zeros(shape, np.float32) if protection is None else protection,
            "motion": np.full(shape, motion, np.float32),
            "cuts": np.array(cuts if cuts is not None else [True] + [False] * (len(clip) - 1), bool)}


def model_inputs(clip, support):
    x = torch.from_numpy(clip.astype(np.float32) / 255).permute(3, 0, 1, 2)[None]
    p = torch.from_numpy(support["protection"])[None, None]
    m = torch.from_numpy(support["motion"])[None, None]
    cuts = torch.from_numpy(support["cuts"])[None]
    return x, p, m, cuts


def test_registry_has_canonical_pure_and_mixed_targets():
    # A missing mixture or malformed weights silently trains the wrong action.
    assert CONDITIONAL_SCHEMA == "adaptive-vcm-conditional-v9"
    a, b, c = (profile_registry(v) for v in "abc")
    assert [r["name"] for r in a] == list(PROFILE_NAMES)
    assert [r["name"] for r in c] == list(PROFILE_NAMES)
    assert len(b) == 20
    assert [r["name"] for r in b[:12]] == list(PROFILE_NAMES)
    assert sorted(round(r["strength"], 2) for r in b[12:]) == [.2] * 4 + [.75] * 2 + [1.] * 2
    for variant, registry in zip("abc", (a, b, c)):
        assert len({r["name"] for r in registry}) == len(registry)
        for row in registry:
            assert sum(row["expert_weights"]) == pytest.approx(1)
            assert all(np.isfinite(row["expert_weights"]))
            assert canonical_target(row["name"], variant) == {"admission": True, "strength": row["strength"],
                                                                "expert_weights": row["expert_weights"]}
    assert canonical_target("identity", "b") == {"admission": False, "strength": 0., "expert_weights": [0., 0., 0., 0.]}
    with pytest.raises(ValueError):
        canonical_target("not_a_profile", "b")


@pytest.mark.parametrize("variant", ["a", "b", "c"])
def test_profiles_have_exact_geometry_cores_and_empty_support_identity(variant):
    # A softened core or phantom edit on fully protected input breaks pixel integrity.
    clip = np.random.default_rng(32).integers(0, 256, (2, 15, 19, 3), dtype=np.uint8)
    protection = np.zeros(clip.shape[:3], np.float32)
    protection[:, :3, :4] = 1
    candidates = profile_candidates(clip, maps(clip, protection), "ar", 45, variant)
    assert [x.name for x in candidates] == [r["name"] for r in profile_registry(variant)]
    for candidate in candidates:
        assert candidate.clip.shape == clip.shape and candidate.clip.dtype == np.uint8
        np.testing.assert_array_equal(candidate.clip[:, :3, :4], clip[:, :3, :4])
    for candidate in profile_candidates(clip, maps(clip, np.ones_like(protection)), "ar", 45, variant):
        np.testing.assert_array_equal(candidate.clip, clip)


def test_a_reference_profiles_match_baseline_pixels():
    # Changing expert algebra while keeping old labels makes distillation inconsistent.
    clip = np.random.default_rng(51).integers(0, 256, (3, 17, 23, 3), dtype=np.uint8)
    support = maps(clip, motion=.3, cuts=[True, False, True])
    support["protection"][:, 2:5, 4:8] = 1
    expected = baseline_profiles(clip, support, "ar", 40, variant="a")
    actual = profile_candidates(clip, support, "ar", 40, variant="a")
    for old, new in zip(expected, actual):
        assert old.name == new.name
        np.testing.assert_array_equal(new.clip, old.clip)


def test_inference_hard_gate_and_training_soft_gate_have_source_level_shapes_and_gradients():
    # A detached or dense head loses gradients or makes inconsistent local decisions.
    torch.manual_seed(19)
    model = ConditionalPreprocessor(12, "ar", "a")
    clip = np.random.default_rng(12).integers(0, 256, (2, 15, 17, 3), dtype=np.uint8)
    x, p, motion, cuts = model_inputs(clip, maps(clip))
    model.train()
    output, aux = model(x, torch.tensor([40]), torch.tensor([1]), p, motion, cuts, return_aux=True)
    assert output.shape == x.shape
    assert aux["gate_probability"].shape == aux["gate_logit"].shape == aux["strength"].shape == (1, 1, 1, 1, 1)
    assert aux["expert_weights"].shape == (1, 4, 1, 1, 1)
    assert aux["alpha"].shape == aux["raw_alpha"].shape == p.shape
    assert aux["mixture"].shape == aux["expert_logits"].shape == (1, 4, 2, 15, 17)
    assert aux["expert_weights"].sum().item() == pytest.approx(1)
    ((output - x).square().mean() + aux["gate_logit"].square().mean()).backward()
    for name, head in (("gate", model.gate_head), ("expert", model.expert_head), ("strength", model.strength_head)):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0, name
    model.eval()
    with torch.no_grad():
        model.gate_head.weight.zero_()
        model.gate_head.bias.fill_(-20)
    identity = model(x, torch.tensor([40]), torch.tensor([1]), p, motion, cuts)
    torch.testing.assert_close(identity, x, atol=0, rtol=0)
    with torch.no_grad():
        model.gate_head.bias.fill_(20)
    edited = model(x, torch.tensor([40]), torch.tensor([1]), p, motion, cuts)
    assert not torch.equal(edited, x)


def test_context_ar_resets_at_cut_and_suppresses_motion():
    # Carrying temporal state across a scene cut contaminates the new scene.
    levels = [0., .4, .4, 1.]
    x = torch.tensor(levels).view(1, 1, 4, 1, 1).expand(1, 3, 4, 2, 2).clone()
    p = torch.zeros(1, 1, 4, 2, 2)
    motion = torch.zeros_like(p)
    cuts = torch.tensor([[True, False, True, False]])
    expert = context_expert(x, p, motion, cuts, "ar")
    torch.testing.assert_close(expert[0, 0, :, 0, 0], torch.tensor([0., .2, .4, .7]))
    motion[:, :, 1] = 1
    held = context_expert(x, p, motion, cuts, "ar")
    torch.testing.assert_close(held[:, :, 1], x[:, :, 1], atol=0, rtol=0)


def test_context_od_preserves_float_luma_and_gamut():
    # Clipping a projected RGB shift changes weighted luma near gamut edges.
    x = torch.rand(1, 3, 1, 19, 23)
    x[:, :, :, 0, 0] = torch.tensor([0., 1., 0.])[:, None]
    p = torch.zeros(1, 1, 1, 19, 23)
    expert = context_expert(x, p, torch.zeros_like(p), torch.tensor([[True]]), "od")
    weights = x.new_tensor([.299, .587, .114])[None, :, None, None, None]
    assert expert.min() >= 0 and expert.max() <= 1
    torch.testing.assert_close((expert * weights).sum(1), (x * weights).sum(1), atol=1e-6, rtol=0)
    rounded = (expert * 255).round() / 255
    assert ((rounded - expert) * weights).sum(1).abs().max() <= .5 / 255 + 1e-6


def test_half_probability_admits_and_three_scales_affect_only_strength():
    # Using > instead of >= drops the specified boundary; scales must not alter admission.
    clip = np.random.default_rng(88).integers(0, 256, (1, 13, 15, 3), dtype=np.uint8)
    x, p, motion, cuts = model_inputs(clip, maps(clip))
    model = ConditionalPreprocessor(8, "od", "b").eval()
    with torch.no_grad():
        model.gate_head.weight.zero_()
        model.gate_head.bias.zero_()
        model.strength_head.weight.zero_()
        model.strength_head.bias.zero_()
        model.expert_head.weight.zero_()
        model.expert_head.bias.copy_(torch.tensor([20., -20., -20., -20.]))
    outputs = [model(x, torch.tensor([45]), torch.tensor([0]), p, motion, cuts,
                     strength_scale=s) for s in (.5, 1., 1.5)]
    assert not torch.equal(outputs[0], x)
    # At strength .5, 1, and 1.5, alpha is .25, .5, and .75.
    first = outputs[0] - x
    torch.testing.assert_close(outputs[1] - x, 2 * first, atol=2e-7, rtol=0)
    torch.testing.assert_close(outputs[2] - x, 3 * first, atol=2e-7, rtol=0)


def test_training_with_no_editable_support_is_exact_identity():
    # A soft admission cannot leak a change through a fully protected source.
    clip = np.random.default_rng(4).integers(0, 256, (2, 9, 11, 3), dtype=np.uint8)
    x, p, motion, cuts = model_inputs(clip, maps(clip, np.ones(clip.shape[:3], np.float32)))
    model = ConditionalPreprocessor(8, "ar", "c").train()
    result, aux = model(x, torch.tensor([30]), torch.tensor([1]), p, motion, cuts, return_aux=True)
    torch.testing.assert_close(result, x, atol=0, rtol=0)
    torch.testing.assert_close(aux["alpha"], torch.zeros_like(p), atol=0, rtol=0)
    assert torch.isfinite(aux["gate_probability"]).all()


def test_c_od_profile_rounding_preserves_source_luma_within_half_code():
    # An RGB clamp after the chroma shift can move OD luma by several codes.
    clip = np.random.default_rng(72).integers(0, 256, (1, 25, 27, 3), dtype=np.uint8)
    profile = {c.name: c.clip for c in profile_candidates(clip, maps(clip), "od", 45, "c")}
    result = profile["motion_block_lowpass_100"]
    luma_weights = np.array([.299, .587, .114])
    delta = (result.astype(float) - clip.astype(float)) @ luma_weights
    assert np.abs(delta).max() <= .50001


def test_b_mixed_profile_is_half_of_two_experts_at_full_strength():
    # Averaging profile metadata without averaging rendered experts corrupts labels.
    clip = np.random.default_rng(71).integers(0, 256, (1, 19, 21, 3), dtype=np.uint8)
    bank = {c.name: c.clip for c in profile_candidates(clip, maps(clip), "od", 45, "b")}
    mixed = bank["motion_mild_gaussian+block_lowpass_100"].astype(np.int16)
    first = bank["motion_mild_gaussian_100"].astype(np.int16)
    second = bank["motion_block_lowpass_100"].astype(np.int16)
    assert np.abs(mixed - .5 * (first + second)).max() <= 1


@pytest.mark.parametrize("variant,task,name,weights", [
    ("a", "ar", "motion_mild_gaussian_075", [20., -20., -20., -20.]),
    ("b", "od", "motion_mild_gaussian+block_lowpass_075", [20., -20., 20., -20.]),
    ("c", "ar", "motion_block_lowpass_075", [-20., -20., 20., -20.]),
])
def test_forced_neural_action_matches_canonical_profile_after_uint8_rounding(variant, task, name, weights):
    # Divergent profile/neural blend equations would poison RGB distillation.
    count = 1 if task == "od" else 2
    clip = np.random.default_rng(103).integers(0, 256, (count, 13, 17, 3), dtype=np.uint8)
    support = maps(clip, motion=.3)
    support["protection"][:, :2, :3] = 1
    x, p, motion, cuts = model_inputs(clip, support)
    model = ConditionalPreprocessor(8, task, variant).eval()
    with torch.no_grad():
        for head in (model.gate_head, model.strength_head, model.expert_head):
            head.weight.zero_()
        model.gate_head.bias.fill_(20)
        model.strength_head.bias.fill_(np.log(3))  # sigmoid(log(3)) = .75
        model.expert_head.bias.copy_(torch.tensor(weights))
        rendered = model(x, torch.tensor([45]), torch.tensor([1]), p, motion, cuts)
    pixels = np.clip(np.rint(rendered[0].permute(1, 2, 3, 0).numpy() * 255), 0, 255).astype(np.uint8)
    target = {c.name: c.clip for c in profile_candidates(clip, support, task, 45, variant)}[name]
    np.testing.assert_array_equal(pixels, target)
