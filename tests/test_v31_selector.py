import importlib
import importlib.util

import numpy as np
import pytest
import torch

from tests.v31_fixture import action_source
from tests.test_v31_guard import ar,det


def mod():
    assert importlib.util.find_spec('adaptive_vcm.v31.selector') is not None, 'V31 selector missing'
    return importlib.import_module('adaptive_vcm.v31.selector')


@pytest.mark.parametrize('task',['ar','od'])
def test_ordered_context_is_label_free_and_48_values(task):
    g = mod()
    sample,support = action_source(task)
    teachers = [ar([.8,.2])]*2 if task=='ar' else [det()]
    context = g.build_context(sample,teachers,teachers,{'total_bytes':1000},task,40,'h265',support)
    assert context.shape == (48,) and context.dtype == np.float32
    assert len(g.CONTEXT_FIELDS) == len(set(g.CONTEXT_FIELDS)) == 48
    assert np.isfinite(context).all()
    if task=='od':
        np.testing.assert_array_equal(context[36:],0)
    for forbidden in ['ground_truth','label','evaluators','choices','candidate_predictions']:
        with pytest.raises(ValueError,match='runtime'):
            g.build_context({**sample,forbidden:{}},teachers,teachers,{'total_bytes':1000},task,40,'h265',support)


def test_rank_stable_ties_availability_and_bias():
    g = mod()
    model = g.ActionSelector('ar',('identity','a','b','c','d'))
    assert torch.equal(model.rate_head.bias,torch.full((4,),-.03))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    assert model.rank(np.zeros(48,np.float32),3) == [1,2,3]
    assert model.rank(np.zeros(48,np.float32),3,np.array([True,False,True,True,False])) == [2,3]
    with pytest.raises(ValueError):
        model.rank(np.zeros(48),3,np.array([True,False]))
    logits = torch.tensor([[1.,0.,0.,0.]])
    with torch.no_grad():
        model.safety_log_weight.copy_(torch.tensor([1.,0.,0.,0.]))
    scores = model.scores(logits,torch.full((1,4),np.log(.8)))
    assert scores[0,0] == pytest.approx(float(scores[0,1]))


def test_loss_masks_unsafe_rates_but_preserves_safety_gradients():
    g = mod()
    model = g.ActionSelector('od',('identity','safe','unsafe','unavailable'))
    context = torch.zeros((4,48))
    safety = torch.tensor([[1.,0.,0.]]*4)
    rates = torch.tensor([[np.log(.8),np.log(.01),0.]]*4,dtype=torch.float32)
    valid = torch.tensor([[True,True,False]]*4)
    loss,parts = g.selector_loss(model,context,safety,rates,valid,torch.ones(3))
    other = rates.clone(); other[:,1] = 9
    _,changed = g.selector_loss(model,context,safety,other,valid,torch.ones(3))
    assert parts['rate_loss'] == changed['rate_loss']
    loss.backward()
    assert model.safety_head.bias.grad[1] > 0
    assert torch.isfinite(model.rate_head.bias.grad).all()
