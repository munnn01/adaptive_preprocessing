import pytest
import torch

from adaptive_vcm.learned import AdaptiveBlendPreprocessor
from adaptive_vcm.train import rate_prior


@pytest.mark.parametrize("shape", [(1, 3, 4, 16, 24), (2, 3, 1, 1, 5), (1, 3, 2, 5, 1)])
def test_blend_bounds_protection_and_gradients(shape):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    model = AdaptiveBlendPreprocessor(8)
    source = torch.rand(shape, requires_grad=True)
    mask = torch.zeros(shape[0], 1, *shape[2:])
    mask[..., :1, :1] = 1
    result, aux = model(source, torch.full((shape[0],), 40), torch.zeros(shape[0]), mask, return_aux=True)
    assert result.shape == source.shape
    assert result.min() >= 0 and result.max() <= 1
    torch.testing.assert_close(result[..., :1, :1], source[..., :1, :1], rtol=0, atol=0)
    assert (aux["alpha"] >= 0).all() and (aux["alpha"] <= 1).all()
    rate_prior(result).backward()
    assert torch.isfinite(source.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_learned_temporal_gate_has_no_future_information():
    torch.manual_seed(8)
    model = AdaptiveBlendPreprocessor(8)
    with torch.no_grad():
        model.head.weight.normal_(0, .1)
    source = torch.rand(1, 3, 4, 16, 16)
    first = model(source, torch.tensor([40]), torch.tensor([0]))
    changed = source.clone()
    changed[:, :, 3] = 0
    second = model(changed, torch.tensor([40]), torch.tensor([0]))
    torch.testing.assert_close(first[:, :, :3], second[:, :, :3], rtol=0, atol=0)


def test_codec_qp_conditioning_is_live_after_nonzero_head():
    torch.manual_seed(8)
    model = AdaptiveBlendPreprocessor(8)
    with torch.no_grad():
        model.head.weight.normal_(0, .5)
    source = torch.rand(1, 3, 2, 16, 16)
    a = model(source, torch.tensor([30]), torch.tensor([0]))
    b = model(source, torch.tensor([50]), torch.tensor([1]))
    assert not torch.equal(a, b)


@pytest.mark.parametrize("qp,codec", [(float("nan"), 0), (-1, 0), (40, 2)])
def test_bad_condition_is_rejected(qp, codec):
    with pytest.raises(ValueError):
        AdaptiveBlendPreprocessor(8)(torch.rand(1, 3, 1, 4, 4), torch.tensor([qp]), torch.tensor([codec]))
