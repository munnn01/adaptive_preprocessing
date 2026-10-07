import copy
import importlib
import importlib.util

import numpy as np
import pytest
import torch

from adaptive_vcm.v31.protocol import canonical_hash
from tests.test_v31_guard import cfg,rows


def mod():
    assert importlib.util.find_spec('adaptive_vcm.v31.train') is not None, 'V31 FIT training missing'
    return importlib.import_module('adaptive_vcm.v31.train')


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_failed_gate_prevents_optimizer_and_artifacts(tmp_path,monkeypatch):
    g = mod()
    called = []
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError,match='gate'):
        g.fit_selector([],{'eligible':False},{},cfg('b'),tmp_path/'train')
    assert not called and not (tmp_path/'train').exists()


@pytest.mark.parametrize('split',['cal','tune','dev'])
def test_nonfit_rows_rejected_before_optimizer(split,tmp_path,monkeypatch):
    g = mod()
    from adaptive_vcm.v31.oracle import assess_headroom
    from tests.test_v31_oracle import headroom_fixture
    gate = {'version':'v31-oracle-gate-1','eligible':True,'task':'ar','arm':'b',
            'bindings':{key:'a'*64 for key in ('config_hash','policy_hash','registry_hash','fit_rows_hash','tune_rows_hash','code_manifest_hash')},
            'integrity':{'passed':True,'reasons':[],'fit_sources':96,'cal_sources':32,'tune_sources':128,
                         'fit_conditions':768,'tune_conditions':1024},
            'results':headroom_fixture(),'headroom':assess_headroom(headroom_fixture())}
    gate['gate_hash'] = canonical_hash(gate)
    called = []
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError,match='only FIT'):
        g.fit_selector([{'split':split}],gate,{},cfg('b'),tmp_path/'train')
    assert not called and not (tmp_path/'train').exists()


def test_targets_keep_unsafe_unavailable_and_identity_only_rows():
    g = mod()
    from adaptive_vcm.v31.guard import fit_policy
    measured = rows(candidate={'probabilities':[.1,.9],'logits':np.log([.1,.9]).tolist()})
    policy = fit_policy(measured,'ar',cfg('b'))
    safety,log_rate,valid = g.measured_targets(measured[0],policy)
    assert safety.tolist() == [0] and valid.tolist() == [True]
    assert log_rate[0] == pytest.approx(np.log(.5))
    measured[0]['actions'][1].update(available=False,total_bytes=None,predictions=None)
    safety,log_rate,valid = g.measured_targets(measured[0],policy)
    assert safety.tolist() == [0] and valid.tolist() == [False]


def test_eight_epochs_backprop_improves_retrieval_and_final_last():
    g = mod()
    from adaptive_vcm.v31.selector import ActionSelector
    x = np.zeros((512,48),np.float32)
    x[:256,0] = 2; x[256:,0] = -2
    safety = np.zeros((512,3),np.float32)
    safety[:256,1] = 1; safety[256:,2] = 1
    rates = np.zeros_like(safety); rates[safety==1] = np.log(.5)
    valid = np.ones_like(safety,bool)
    torch.manual_seed(303101)
    model = ActionSelector('ar',('identity','unsafe','left','right'))
    initial = {name:value.clone() for name,value in model.state_dict().items()}
    before = sum(model.rank(row,1)[0]==(2 if i<256 else 3) for i,row in enumerate(x)) / len(x)
    trained,history = g.train_arrays(model,x,safety,rates,valid,[str(i//8) for i in range(512)],cfg('b'))
    after = sum(trained.rank(row,1)[0]==(2 if i<256 else 3) for i,row in enumerate(x)) / len(x)
    assert len(history) == 8 and history[-1]['checkpoint_role'] == 'LAST'
    assert after > before and after >= .95
    assert any(not torch.equal(value,initial[name]) for name,value in trained.state_dict().items())
    assert all(epoch['conditions'] == 512 and epoch['sources'] == 64 for epoch in history)


def test_checkpoint_strict_sha_schema_registry_and_expected_bindings():
    g = mod()
    from adaptive_vcm.v31.selector import ActionSelector,CONTEXT_FIELDS
    from dataclasses import asdict
    from adaptive_vcm.v31.actions import action_registry
    registry = action_registry('ar','b')
    model = ActionSelector('ar',tuple(a.name for a in registry))
    metadata = {'schema':g.CHECKPOINT_SCHEMA,'task':'ar','arm':'b','width':64,
                'action_names':list(model.action_names),'context_fields':list(CONTEXT_FIELDS),
                'policy_hash':'a'*64,'registry_hash':canonical_hash([asdict(a) for a in registry]),
                'checkpoint_role':'LAST','gate_hash':'a'*64,'config_hash':'a'*64,'fit_rows_hash':'a'*64,
                'code_manifest_hash':'a'*64,'epochs':8,'seed':303101,
                'training_source_ids':[str(i) for i in range(96)],'training_conditions':768,
                'history':[{'epoch':i+1,'conditions':768,'sources':96,'checkpoint_role':'LAST' if i==7 else 'training'} for i in range(8)]}
    checkpoint = g.make_checkpoint(model,metadata)
    assert g.load_selector(checkpoint,{'task':'ar','policy_hash':'a'*64}).action_names == model.action_names
    for expected in [{'task':'od'},{'registry_hash':'c'*64},{'policy_hash':'c'*64}]:
        with pytest.raises(ValueError):
            g.load_selector(checkpoint,expected)
    changed = copy.deepcopy(checkpoint)
    changed['model']['rate_head.bias'][0] += 1
    with pytest.raises(ValueError,match='SHA'):
        g.load_selector(changed,{})
    changed = copy.deepcopy(checkpoint)
    changed['metadata']['schema'] = 'adaptive-vcm-ranking-v4'
    with pytest.raises(ValueError):
        g.load_selector(changed,{})
    incomplete = copy.deepcopy(checkpoint)
    incomplete['metadata'].pop('gate_hash')
    incomplete['metadata_hash'] = canonical_hash(incomplete['metadata'])
    with pytest.raises(ValueError,match='provenance'):
        g.load_selector(incomplete,{})
