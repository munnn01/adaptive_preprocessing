import copy
import importlib
import importlib.util
import inspect
import json

import numpy as np
import pytest
import torch

from adaptive_vcm.v31.actions import action_registry
from adaptive_vcm.v31.guard import fit_policy
from tests.test_v31_guard import cfg,rows
from tests.test_v31_measure import Model
from tests.v31_fixture import action_source


def mod():
    assert importlib.util.find_spec('adaptive_vcm.v31.select') is not None, 'matched-budget V31 runtime missing'
    return importlib.import_module('adaptive_vcm.v31.select')


class Proposer:
    def __init__(self,task='ar',indices=(1,2,3),arm='b'):
        self.task,self.indices = task,list(indices)
        self.action_names = tuple(a.name for a in action_registry(task,arm))
    def rank(self,context,top_k=3,available=None):
        assert context.shape==(48,) and top_k==3
        return [i for i in self.indices if available is None or available[i]]


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def teachers(task):
    return {name:Model(name,task) for name in (('r3d_18','mc3_18') if task=='ar' else ('mobilenet',))}


def bind_policy(policy,nets):
    from adaptive_vcm.v31.protocol import canonical_hash
    policy['cal_model_hashes'] = {name:net.model_hash for name,net in nets.items()}
    policy['policy_hash'] = canonical_hash({k:v for k,v in policy.items() if k!='policy_hash'})
    return policy


def test_primary_encodes_only_anchor_plus_three_and_uses_actual_bytes(monkeypatch):
    g = mod()
    sample,_ = action_source('ar')
    policy = fit_policy(rows(),'ar',cfg('b'))
    calls = []
    original = g.V31Codec.roundtrip
    def counted(self,rgb,recipe):
        calls.append((rgb.shape,recipe))
        return original(self,rgb,recipe)
    monkeypatch.setattr(g.V31Codec,'roundtrip',counted)
    nets = teachers('ar'); bind_policy(policy,nets)
    result = g.choose_stream(sample,Proposer(),policy,('h264',35),nets,cfg('b'))
    assert result['stats']['slots']==4 and len(calls)==result['stats']['distinct_encodes']==4
    assert result['proposals']==[1,2,3]
    assert result['packet'].total_bytes==result['observations'][result['selection_index']]['total_bytes']
    assert set(inspect.signature(g.choose_stream).parameters)=={'sample','selector','policy','codec','teachers','cfg'}
    assert 'ground_truth' not in result['teacher_only_row']
    assert all(set(a['predictions'])=={'teachers'} for a in result['observations'] if a['available'])


def test_pixel_alias_slots_deduplicate_encoding_and_teachers():
    g = mod()
    sample,_ = action_source('od')
    sample['rgb'] = np.zeros_like(sample['rgb'])
    policy = fit_policy(rows('od'),'od',cfg('b'))
    nets = teachers('od')
    bind_policy(policy,nets)
    result = g.choose_stream(sample,Proposer('od',(1,2,3)),policy,('h265',40),nets,cfg('b'))
    assert result['stats']['slots']==4 and result['stats']['distinct_encodes']==1
    # H.265 reconstructs some RGB values as1 here. Source and decoded input
    # each need one observation; the three pixel aliases add none.
    assert result['stats']['teacher_observations']==2
    assert result['action_name']=='identity'  # exact bytes tie with anchor; minimum saving applies
    assert result['stats']['aliases']==3


def test_drop2_restored_before_teacher_scoring_and_recipe_not_aliased():
    g = mod()
    sample,_ = action_source('ar')
    sample['rgb'] = np.zeros_like(sample['rgb'])
    policy = fit_policy(rows(),'ar',cfg('b'))
    registry = action_registry('ar','b')
    drop = next(i for i,a in enumerate(registry) if a.name=='drop2_128')
    nets = teachers('ar')
    bind_policy(policy,nets)
    result = g.choose_stream(sample,Proposer(indices=(drop,)),policy,('h264',35),nets,cfg('b'))
    assert result['stats']['slots']==2 and result['stats']['distinct_encodes']==2
    assert all(shape[0]==16 for net in nets.values() for shape in net.calls)
    assert result['observations'][1]['recipe']['coded_frames']==8
    assert result['observations'][1]['recipe']['repeat_factor']==2


def test_runtime_fields_rejected_before_teacher_inference():
    g = mod()
    sample,_ = action_source('ar')
    policy = fit_policy(rows(),'ar',cfg('b'))
    for extra in ['ground_truth','label','evaluators','choices']:
        nets = teachers('ar')
        with pytest.raises(ValueError,match='runtime'):
            g.choose_stream({**sample,extra:{}},Proposer(),policy,('h264',35),nets,cfg('b'))
        assert not any(net.calls for net in nets.values())
    with pytest.raises(ValueError,match='teacher'):
        g.choose_stream(sample,Proposer(),policy,('h264',35),{**teachers('ar'),'r2plus1d_18':Model('r2plus1d_18','ar')},cfg('b'))


def test_bad_proposals_and_diagnostic_union_budget():
    g = mod()
    sample,_ = action_source('od')
    policy = fit_policy(rows('od'),'od',cfg('b'))
    with pytest.raises(ValueError,match='proposal'):
        nets = teachers('od'); bind_policy(policy,nets)
        g.choose_stream(sample,Proposer('od',(0,1,2)),policy,('h264',35),nets,cfg('b'))
    nets = teachers('od'); bind_policy(policy,nets)
    result = g.choose_stream(sample,Proposer('od',(5,6,7)),policy,('h264',35),nets,
                             {**cfg('b'),'selection_mode':'diagnostic_union_controls'})
    assert result['scope']=='diagnostic_union_controls'
    assert result['stats']['slots']==8
