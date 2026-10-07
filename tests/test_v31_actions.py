"""Frozen actions, source immutability and independent OD coordinate evidence."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import importlib
import importlib.util

import cv2
import numpy as np
import pytest
import torch

from adaptive_vcm.conditional_learned import profile_candidates as conditional_profiles
from adaptive_vcm.motion_learned import PROFILE_NAMES, profile_candidates
from adaptive_vcm.preprocessing import make_candidates
from tests.v31_fixture import action_source


def module():
    assert importlib.util.find_spec("adaptive_vcm.v31.actions") is not None, "V31 actions are missing"
    return importlib.import_module("adaptive_vcm.v31.actions")


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def named(task, name, arm="b"):
    return next(a for a in module().action_registry(task, arm) if a.name == name)


@pytest.mark.parametrize("task,counts", [("ar", (21, 27)), ("od", (20, 23))])
def test_registry_preserves_aliases_subsets_order_and_frozen_descriptors(task, counts):
    mod = module()
    a, b, c = (mod.action_registry(task, arm) for arm in "abc")
    assert isinstance(a, tuple) and (len(a), len(b)) == counts
    assert b[:len(a)] == a and c == b
    assert len({row.name for row in b}) == len(b)
    assert [row.profile for row in a if row.kind == "profile"] == list(PROFILE_NAMES)
    assert all(row.task == task for row in b)
    with pytest.raises(FrozenInstanceError):
        a[0].size = 2
    for bad_task, bad_arm in (("audio", "a"), (task, "d")):
        with pytest.raises(ValueError):
            mod.action_registry(bad_task, bad_arm)


@pytest.mark.parametrize("task", ["ar", "od"])
def test_existing_controls_and_v29_a_profiles_keep_exact_rgb(task):
    mod = module()
    source, support = action_source(task)
    expected = {c.name: c.clip for c in make_candidates(
        source["rgb"], source["control_protection"], task, 35)}
    expected.update({c.name: c.clip for c in profile_candidates(source["rgb"], support, task, 35, variant="a")})
    for name, pixels in expected.items():
        result = mod.execute_action(source, named(task, name), 35, support)
        assert result["available"] and result["reason"] is None
        np.testing.assert_array_equal(result["rgb"], pixels)
        if name.startswith(("protected", "background", "motion_")):
            core = support["protection"] == 1 if name.startswith("motion_") else np.broadcast_to(source["control_protection"] == 1, source["rgb"].shape[:3])
            np.testing.assert_array_equal(result["rgb"][core], source["rgb"][core])


def test_drop2_transmits_even_indices_preserves_duration_and_coded_geometry():
    mod = module()
    source, support = action_source()
    for size in (128, 112, 96):
        result = mod.execute_action(source, named("ar", f"drop2_{size}"), 30, support)
        assert result["sample_indices"] == (0, 2, 4, 6, 8, 10, 12, 14)
        assert result["rgb"].shape == (8, size, size, 3)
        expected = source["rgb"][::2] if size == 128 else np.stack([
            cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in source["rgb"][::2]])
        np.testing.assert_array_equal(result["rgb"], expected)
        recipe = result["recipe"]
        assert (recipe.coded_frames, recipe.analyzer_frames, recipe.repeat_factor) == (8, 16, 2)
        assert recipe.duration == Fraction(16016, 15000)
        assert recipe.fps == Fraction(7500, 1001)
        assert result["source_shape"] == (128, 128) and result["coded_shape"] == (size, size)


@pytest.mark.parametrize("changes,reason", [({"padded": True}, "padded_source"),
    ({"source_fps": None}, "unknown_source_fps"), ({"source_fps": float("nan")}, "unknown_source_fps"),
    ({"duration": None}, "unknown_source_duration")])
def test_unavailable_temporal_action_never_has_encodable_rgb_or_recipe(changes, reason):
    source, support = action_source()
    source.update(changes)
    result = module().execute_action(source, named("ar", "drop2_128"), 35, support)
    assert not result["available"] and result["reason"] == reason
    assert result["rgb"] is None and result["recipe"] is None


def test_missing_ar_timing_never_becomes_valid_primary_packet():
    source, support = action_source()
    source["duration"] = None
    result = module().execute_action(source, named("ar", "identity"), 35, support)
    assert not result["available"] and result["recipe"] is None


def test_temporal_uses_v30_c_full_strength_then_resize_and_resets_at_cut():
    source, support = action_source()
    source["rgb"][:8] = 40
    source["rgb"][8:] = 200
    source["rgb"][1] = 80
    support["protection"][:] = 0
    support["motion"][:] = 0
    support["cuts"][:] = False
    support["cuts"][[0, 8]] = True
    expected = {c.name: c.clip for c in conditional_profiles(source["rgb"], support, "ar", 40, variant="c")}["motion_block_lowpass_100"]
    for size in (128, 112, 96):
        result = module().execute_action(source, named("ar", f"temporal_{size}"), 40, support)
        pixels = expected if size == 128 else np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in expected])
        np.testing.assert_array_equal(result["rgb"], pixels)
        assert result["recipe"].repeat_factor == 1
    full = module().execute_action(source, named("ar", "temporal_128"), 40, support)["rgb"]
    assert full[1, 40, 40, 0] == 60
    assert full[8, 40, 40, 0] == 200
    assert full[9, 40, 40, 0] == 200


def test_od_direct_resize_and_background8_composition_and_transport_duration():
    source, support = action_source("od")
    source["duration"] = None
    background = {c.name: c.clip for c in make_candidates(source["rgb"], source["control_protection"], "od", 45)}["background8"]
    for size in (256, 224, 192):
        for name, before in ((f"area{size}", source["rgb"]), (f"background8_area{size}", background)):
            result = module().execute_action(source, named("od", name), 45, support)
            np.testing.assert_array_equal(result["rgb"], np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in before]))
            assert result["recipe"].duration == Fraction(1, 25)
            assert result["recipe"].analyzer_frames == 1


@pytest.mark.parametrize("task,count", [("ar", 8), ("od", 2)])
def test_primary_action_rejects_noncanonical_counts(task, count):
    source, support = action_source(task)
    source["rgb"] = np.repeat(source["rgb"][:1], count, axis=0)
    with pytest.raises(ValueError, match="frame|sample|count"):
        module().execute_action(source, named(task, "identity"), 30, support)


@pytest.mark.parametrize("task", ["ar", "od"])
def test_primary_actions_reject_spatial_geometry_drift(task):
    source, support = action_source(task)
    source["rgb"] = source["rgb"][:, :96, :96]
    with pytest.raises(ValueError, match="geometry"):
        module().execute_action(source, named(task, "identity"), 30, support)


def test_control_and_profile_protection_remain_distinct_for_od_without_objects():
    source, support = action_source("od")
    source["control_protection"][:] = 0
    support["protection"][:] = 1
    mod = module()
    control = mod.execute_action(source, named("od", "background8"), 35, support)["rgb"]
    expected = {c.name: c.clip for c in make_candidates(source["rgb"], source["control_protection"], "od", 35)}["background8"]
    np.testing.assert_array_equal(control, expected)
    assert not np.array_equal(control, source["rgb"])
    profile = mod.execute_action(source, named("od", "motion_strong_gaussian_100"), 35, support)["rgb"]
    np.testing.assert_array_equal(profile, source["rgb"])
    for invalid in (None, np.ones((2, 2)), np.full((320, 320), np.nan)):
        with pytest.raises(ValueError):
            mod.execute_action({**source, "control_protection": invalid}, named("od", "background8"), 35, support)


def test_od_rejects_temporal_descriptor_and_invalid_condition_or_support():
    mod = module()
    source, support = action_source("od")
    with pytest.raises(ValueError):
        mod.execute_action(source, replace(named("ar", "drop2_128"), task="od"), 30, support)
    for changes in ({"codec": "av1"}, {"task": "ar"}):
        with pytest.raises(ValueError):
            mod.execute_action({**source, **changes}, named("od", "identity"), 30, support)
    with pytest.raises(ValueError):
        mod.execute_action(source, named("od", "identity"), 29, support)
    with pytest.raises(ValueError):
        mod.execute_action(source, named("od", "identity"), 30, {**support, "cuts": np.array([False])})


@pytest.mark.parametrize("task", ["ar", "od"])
@pytest.mark.filterwarnings("error:The given NumPy array is not writable")
def test_source_and_support_are_immutable_across_every_action(task):
    source, support = action_source(task)
    originals = {key: value.copy() for key, value in support.items() if isinstance(value, np.ndarray)}
    pixels = source["rgb"].copy()
    control = source["control_protection"].copy()
    source["rgb"].setflags(write=False)
    source["control_protection"].setflags(write=False)
    for value in originals:
        support[value].setflags(write=False)
    for action in module().action_registry(task, "b"):
        result = module().execute_action(source, action, 35, support)
        assert not np.shares_memory(result["rgb"], source["rgb"])
        result["rgb"][:] = 0
    np.testing.assert_array_equal(source["rgb"], pixels)
    np.testing.assert_array_equal(source["control_protection"], control)
    for key, value in originals.items():
        np.testing.assert_array_equal(support[key], value)


def test_mapping_uses_effective_rounded_scales_explicit_canvas_and_clips_copies():
    # Original 101x200 -> rounded content 161x320 within 320-square canvas.
    # Top=79, bottom=80: inferring canvas from twice the top gives wrong 319.
    prediction = {"boxes": np.array([[0, 47.4, 192, 144], [-6, -3, 204, 198]], np.float32),
                  "scores": np.array([.8, .4], np.float32), "labels": np.array([3, 7])}
    result = module().map_detections(prediction, (192, 192), (1.6, 161 / 101, 0, 79, 320, 320), (101, 200))
    np.testing.assert_allclose(result["canonical"]["boxes"], [[0, 79, 320, 240], [0, 0, 320, 320]], atol=1e-5)
    np.testing.assert_allclose(result["original"]["boxes"], [[0, 0, 200, 101], [0, 0, 200, 101]], atol=1e-5)
    for coordinate in ("canonical", "original"):
        for key in ("boxes", "scores", "labels"):
            assert not np.shares_memory(result[coordinate][key], prediction[key])
    result["canonical"]["scores"][:] = 0
    assert result["original"]["scores"][0] == prediction["scores"][0]


def test_mapping_handles_nonsquare_canvas_and_horizontal_rounded_letterbox():
    prediction = {"boxes": np.array([[46.2, 0, 146.4, 144]], np.float32),
                  "scores": np.array([1.]), "labels": np.array([1])}
    result = module().map_detections(prediction, (144, 192), (167 / 101, 1.2, 77, 0, 240, 320), (200, 101))
    np.testing.assert_allclose(result["canonical"]["boxes"], [[77, 0, 244, 240]], atol=2e-5)
    np.testing.assert_allclose(result["original"]["boxes"], [[0, 0, 101, 200]], atol=2e-5)


def test_mapping_rejects_missing_canvas_nonfinite_boxes_and_bad_shapes():
    prediction = {"boxes": np.array([[0, 0, 1, 1.]]), "scores": np.array([1.]), "labels": np.array([1])}
    for transform in ((1, 1, 0, 0), (0, 1, 0, 0, 320, 320)):
        with pytest.raises(ValueError):
            module().map_detections(prediction, (192, 192), transform, (101, 200))
    with pytest.raises(ValueError):
        module().map_detections({**prediction, "boxes": np.array([[0, 0, np.nan, 1]])}, (192, 192), (1, 1, 0, 0, 320, 320), (101, 200))
