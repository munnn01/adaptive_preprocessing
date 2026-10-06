"""Source-only motion support: camera drift must not become foreground."""
import importlib
import json

import cv2
import numpy as np
import pytest

from adaptive_vcm.preprocessing import boxes_to_mask


def build(clip, protection, task="ar"):
    return importlib.import_module("adaptive_vcm.motion_support").build_motion_support(clip, protection, task)


def pan_clip(local_motion=False):
    rng = np.random.default_rng(91)
    base = cv2.GaussianBlur(rng.integers(30, 210, (72, 96, 3), dtype=np.uint8), (3, 3), .7)
    patch = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    frames = []
    for index in range(5):
        frame = np.roll(base, 2 * index, axis=1).copy()
        if local_motion:
            frame[25:45, 18 + 5 * index:38 + 5 * index] = patch
        frames.append(frame)
    return np.stack(frames)


def test_camera_translation_does_not_protect_scene_wide_texture():
    # Omitting median global-flow subtraction would protect the entire pan.
    clip = pan_clip()
    result = build(clip, np.zeros(clip.shape[1:3], np.float32))
    assert result["motion"].mean() < .03
    assert np.count_nonzero(result["protection"] == 1) < .02 * result["protection"].size
    assert not result["cuts"][1:].any()
    assert result["metadata"]["flow_source"] == "rgb_farneback_forward_backward"
    assert result["metadata"]["camera_compensation"] == "median_translation"


def test_local_motion_survives_camera_compensation_and_semantic_cores():
    # Replacing residual density with the compensated zero map loses this patch.
    clip = pan_clip(local_motion=True)
    semantic = np.zeros(clip.shape[1:3], np.float32)
    semantic[3:7, 3:8] = 1
    result = build(clip, semantic)
    assert result["motion"][:, 23:47, 15:62].mean() > .08
    assert result["motion"][:, 3:15, 65:85].mean() < .04
    assert (result["protection"][:, 3:7, 3:8] == 1).all()
    assert result["protection"][:, 23:47, 15:62].mean() > .2
    repeated = build(clip, semantic)
    np.testing.assert_array_equal(result["protection"], repeated["protection"])
    assert any(segment["mode"] == "motion_density" for segment in result["metadata"]["segments"])


def test_scene_cut_removes_old_motion_tube():
    # A clip-wide max instead of segment aggregation leaks the moving ROI.
    first = pan_clip(local_motion=True)[:3]
    second = np.full((3, 72, 96, 3), 235, np.uint8)
    semantic = np.zeros((72, 96), np.float32)
    semantic[2:5, 2:5] = 1
    result = build(np.concatenate([first, second]), semantic)
    assert result["cuts"].tolist() == [True, False, False, True, False, False]
    np.testing.assert_array_equal(result["motion"][3:], np.zeros((3, 72, 96), np.float32))
    np.testing.assert_array_equal(result["protection"][3:], np.broadcast_to(semantic, (3, 72, 96)))


def test_same_histogram_scene_cut_resets_segment_experts():
    # Histogram-only cut detection pools unrelated scenes with equal colors.
    first = np.random.default_rng(51).integers(0, 256, (72, 96, 3), dtype=np.uint8)
    second = first[::-1, ::-1].copy()
    clip = np.stack([first, first, second, second])
    result = build(clip, np.zeros((72, 96), np.float32))
    assert result["cuts"].tolist() == [True, False, True, False]
    assert len(result["metadata"]["segments"]) == 2


def test_incoherent_noisy_frames_do_not_invent_a_motion_region():
    # Accepting dense inconsistent flow as foreground hides the fallback.
    clip = np.random.default_rng(135).integers(0, 256, (3, 60, 80, 3), dtype=np.uint8)
    mask = np.zeros((60, 80), np.float32)
    mask[5:8, 6:9] = 1
    result = build(clip, mask)
    np.testing.assert_array_equal(result["protection"], np.broadcast_to(mask, (3, 60, 80)))
    assert not result["motion"].any()
    assert all(segment["mode"] == "semantic_fallback" for segment in result["metadata"]["segments"])


def test_od_single_frame_keeps_multiple_tiny_boxes_and_bypasses_flow():
    # Retaining only one selected rectangle would erase a distant small box.
    clip = np.random.default_rng(8).integers(0, 256, (1, 31, 47, 3), dtype=np.uint8)
    mask = boxes_to_mask(31, 47, np.array([[1, 2, 4, 5], [40, 25, 44, 29]]), halo=0, grid=1, radius=2)
    result = build(clip, mask, "od")
    np.testing.assert_array_equal(result["protection"][0], mask)
    assert result["cuts"].tolist() == [True]
    assert not result["motion"].any()
    assert result["metadata"]["flow_source"] is None
    assert result["metadata"]["fallback_reason"] == "od_single_frame"


def test_untextured_clip_has_explicit_semantic_fallback():
    # Trusting arbitrary flow on uniform fields creates invented foreground.
    clip = np.stack([np.full((19, 25, 3), value, np.uint8) for value in (70, 72, 74)])
    mask = np.zeros((19, 25), np.float32)
    mask[4:7, 8:11] = 1
    result = build(clip, mask)
    np.testing.assert_array_equal(result["protection"], np.broadcast_to(mask, (3, 19, 25)))
    assert not result["motion"].any()
    assert result["metadata"]["segments"][0]["mode"] == "semantic_fallback"
    assert result["metadata"]["segments"][0]["reason"] == "unreliable_flow"


@pytest.mark.parametrize("shape", [(1, 1, 5, 3), (3, 7, 9, 3)])
def test_support_preserves_odd_and_small_geometry_with_json_metadata(shape):
    # Resizing for flow or returning NaNs breaks the source geometry contract.
    clip = np.full(shape, 113, np.uint8)
    result = build(clip, np.zeros(shape[1:3], np.float32))
    for name in ("protection", "motion"):
        assert result[name].shape == shape[:3]
        assert result[name].dtype == np.float32
        assert np.isfinite(result[name]).all()
    assert result["cuts"].dtype == np.bool_
    assert result["cuts"][0]
    json.dumps(result["metadata"], allow_nan=False)


@pytest.mark.parametrize("bad", [np.full((8, 10), np.nan), np.full((8, 10), np.inf),
                                  np.full((8, 10), 1.1), np.zeros((10, 8))])
def test_invalid_semantic_protection_is_rejected(bad):
    with pytest.raises(ValueError):
        build(np.zeros((2, 8, 10, 3), np.uint8), bad)


def test_invalid_clip_task_and_multiframe_od_are_rejected():
    for clip, task in [(np.zeros((1, 8, 10, 3)), "ar"),
                       (np.zeros((2, 8, 10, 3), np.uint8), "od"),
                       (np.zeros((1, 8, 10, 3), np.uint8), "unknown")]:
        with pytest.raises(ValueError):
            build(clip, np.zeros((8, 10), np.float32), task)
