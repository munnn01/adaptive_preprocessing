import numpy as np
import torch
import pytest
from adaptive_vcm.v34.preprocessing import region_maps, filter_clip
from adaptive_vcm.v34.model import Controller


def test_identity_preserves_q8_and_filter_really_edits_background():
    rng = np.random.default_rng(4)
    rgb = rng.integers(0, 256, (3, 24, 24, 3), dtype=np.uint8)
    maps = region_maps(rgb, [{"boxes": [], "scores": []}]*3)
    assert np.array_equal(filter_clip(rgb, maps, [0]*6), rgb)
    changed = filter_clip(rgb, maps, [0, 0, 0, 0, -.7, -.3])
    assert changed.dtype == np.uint8 and changed.shape == rgb.shape
    assert np.count_nonzero(changed != rgb) > rgb.size // 3


def test_maps_cannot_use_future_detection_and_reset_at_scene_cut():
    rgb = np.zeros((3, 24, 24, 3), np.uint8)
    rgb[2] = 255
    predictions = [{"boxes": [[2, 2, 8, 8]], "scores": [.9]},
                   {"boxes": [], "scores": []}, {"boxes": [], "scores": []}]
    maps = region_maps(rgb, predictions)
    assert maps[1, 4, 4, 0] > .5
    assert maps[2, 4, 4, 0] == 0
    future = predictions[:2] + [{"boxes": [[15, 15, 22, 22]], "scores": [.9]}]
    assert np.array_equal(region_maps(rgb, future)[:2], maps[:2])
    assert np.allclose(maps.sum(-1), 1)
    assert np.isfinite(maps).all()


def test_invalid_coefficients_and_maps_do_not_silently_clamp():
    rgb = np.zeros((1, 8, 8, 3), np.uint8)
    maps = region_maps(rgb, [{"boxes": [], "scores": []}])
    with pytest.raises(ValueError): filter_clip(rgb, maps, [float('nan')]*6)
    with pytest.raises(ValueError): filter_clip(rgb, maps, [-2]*6)
    with pytest.raises(ValueError): filter_clip(rgb, maps*0, [0]*6)


def test_controller_backprop_updates_only_controller_and_handles_one_frame():
    model = Controller(7)
    for t in (1, 4):
        x = torch.rand(2, t, 3, 16, 16)
        maps = torch.zeros_like(x); maps[:, :, 2] = 1
        out = model(x, maps, torch.zeros(2, 6))
        assert out['logits'].shape == (2, 7)
        assert out['risk'].shape == (2, 7, 3)
        assert out['rate'].shape == (2, 7)
        loss = out['logits'].square().mean()+out['risk'].square().mean()+out['rate'].mean()
        model.zero_grad(); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
