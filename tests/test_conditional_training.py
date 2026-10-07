"""Catch gate/strength conflation and unverified mixed target supervision."""
import argparse
import json
import math

import numpy as np

import pytest
import torch

from adaptive_vcm import train_motion as training


def objective():
    result = getattr(training, 'conditional_loss', None)
    assert callable(result), 'V30 conditional loss is not implemented'
    return result


def fixture(profile='identity', *, strength=.6, probability=.4):
    output = torch.zeros(1, 3, 1, 2, 2, requires_grad=True)
    alpha = torch.full((1, 1, 1, 1, 1), strength, requires_grad=True)
    logits = torch.tensor([[math.log(probability/(1-probability))]], requires_grad=True)
    weights = torch.tensor([.5, .5, 0., 0.]).view(1, 4, 1, 1, 1).requires_grad_()
    aux = dict(strength=alpha, gate_logit=logits, gate_probability=logits.sigmoid(),
               expert_weights=weights, alpha=alpha.expand(1, 1, 1, 2, 2))
    row = dict(target_profile=profile, controls_coded_bytes=1000,
               marginal_saved_bytes=0 if profile == 'identity' else 50)
    return output, torch.zeros_like(output), aux, torch.zeros(1, 1, 1, 2, 2), row


def test_identity_direct_supervision_trains_gate_without_zero_strength_target():
    output, target, aux, mask, row = fixture()
    row['target_parameters'] = dict(admission=False, strength=0., expert_weights=[0., 0., 0., 0.])
    loss, parts = objective()(output, target, aux, mask, row, 1., 'a')
    loss.backward()
    assert parts['alpha_loss'].item() == 0
    assert parts['expert_loss'].item() == 0
    assert parts['gate_loss'].item() == pytest.approx(-math.log(.6))
    assert aux['gate_logit'].grad.item() > 0
    assert aux['strength'].grad is None or aux['strength'].grad.item() == 0


def test_positive_gate_strength_and_expert_losses_use_measured_parameters():
    output, target, aux, mask, row = fixture('motion_mild_gaussian_040')
    row['target_parameters'] = dict(admission=True, strength=.4, expert_weights=[1., 0., 0., 0.])
    _, parts = objective()(output, target, aux, mask, row, 1., 'a')
    assert parts['alpha_loss'].item() == pytest.approx(.04)
    assert parts['expert_loss'].item() == pytest.approx(math.log(2))
    assert parts['gate_loss'].item() == pytest.approx(-math.log(.4))
    assert parts['utility_weight'].item() == 1.5


def test_canonical_target_metadata_cannot_override_profile_or_be_missing():
    values = fixture('motion_mild_gaussian_040')
    with pytest.raises(ValueError, match='target'):
        objective()(*values, 1., 'a')
    values[4]['target_parameters'] = dict(admission=True, strength=.4, expert_weights=[0., 1., 0., 0.])
    with pytest.raises(ValueError, match='target'):
        objective()(*values, 1., 'a')


def test_bank_summary_separates_baseline_and_extra_guarded_margin():
    function = getattr(training, 'conditional_bank_summary', None)
    assert callable(function), 'V30 bank headroom summary is not implemented'
    action = lambda name, size, decision=True: dict(name=name, coded_bytes=size,
        distances=[0.], decisions=[decision])
    row = dict(task='od', codec='h265', qp=50, anchor_coded_bytes=1000,
        controls_coded_bytes=900, slack=.03, min_savings=.01,
        profiles=[action('motion_mild_gaussian_040', 920), action('new_weak', 880),
                  action('unsafe_extra', 800, False)])
    summary = function([row], ['motion_mild_gaussian_040'])['h265/50']
    assert summary['points'] == 1
    assert summary['baseline_positive_targets'] == 0
    assert summary['positive_targets'] == 1
    assert summary['marginal_saved_bytes'] == summary['extra_bank_saved_bytes'] == 20


def test_v30_fitter_keeps_all_conditions_and_emits_gate_diagnostics(tmp_path):
    from adaptive_vcm.conditional_learned import canonical_target
    torch.set_num_threads(2)
    source = np.random.default_rng(7).integers(30, 180, (1, 8, 8, 3), dtype=np.uint8)
    np.savez_compressed(tmp_path/'source.npz', source=source,
        protection=np.zeros(source.shape[:3], np.float32),
        motion=np.zeros(source.shape[:3], np.float32), cuts=np.array([True]))
    np.savez_compressed(tmp_path/'target.npz', target=np.full_like(source, 100))
    rows = [dict(source_id='fixture', target_profile=name, codec='h264', qp=50,
        source_cache='source.npz', target_cache=None if name=='identity' else 'target.npz',
        controls_coded_bytes=1000, marginal_saved_bytes=0 if name=='identity' else 50,
        target_parameters=canonical_target(name, 'a'))
        for name in ['identity', 'motion_background_dc_100']]
    args = argparse.Namespace(out=tmp_path, task='od', width=4, seed=41,
                              epochs=2, variant=None, conditional_variant='a')
    model, fitting = training._fit(args, rows, 'cpu')
    assert model.schema == 'adaptive-vcm-conditional-v9'
    assert fitting['steps'] == 4
    logs = [json.loads(line) for line in (tmp_path/'train.jsonl').read_text().splitlines()]
    assert [sum(row['target_profile']=='identity' for row in logs if row['epoch']==e)
            for e in [1,2]] == [1,1]
    assert all('gate_loss' in row and 'gate_gradient_norm' in row for row in logs)
    assert all(row['alpha_loss']==0 and row['expert_loss']==0
               for row in logs if row['target_profile']=='identity')
    assert any(row['gate_gradient_norm'] > 0 for row in logs)


@pytest.mark.parametrize('variant', ['a','b','c'])
def test_complete_v30_collection_exports_loadable_last_checkpoint(tmp_path, monkeypatch, variant):
    """Replace external dataset/models/codec; keep support, labels, fit and loader real."""
    import hashlib
    from adaptive_vcm.codec import Encoded
    from adaptive_vcm.data import partition
    from adaptive_vcm.rateaware import load_preprocessor
    from adaptive_vcm.conditional_learned import canonical_target
    source_id = next(str(i) for i in range(100) if partition(f'coco2017/{i}')=='train')
    pixels = np.random.default_rng(6).integers(30,180,(1,32,32,3),dtype=np.uint8)
    class Teacher:
        def __init__(self,*args): pass
        def predict(self,clip):
            return dict(boxes=np.array([[0.,0.,1.,1.]]),scores=np.array([.9]),labels=np.array([1]))
    class Codec:
        def __init__(self,*args): pass
        def roundtrip(self,clip):
            size = 320 + int(clip.var())
            digest = hashlib.sha256(clip.tobytes()).digest()
            data = (digest*((size+31)//32))[:size]
            return Encoded(clip.copy(),data,.1)
    monkeypatch.setattr(training,'DetectionAnalyzer',Teacher)
    monkeypatch.setattr(training,'StandardCodec',Codec)
    monkeypatch.setattr(training,'od_plan',lambda *args:([dict(id=source_id)],{}))
    monkeypatch.setattr(training,'read_image',lambda *args:(pixels.copy(),{}))
    args=argparse.Namespace(task='od',root=tmp_path,annotations=tmp_path/'annotations.json',
        config=training.ROOT/f'configs/v30_{variant}_screen.json',count=1,epochs=1,width=4,
        seed=302901,out=tmp_path/'output')
    manifest=training.train(args)
    checkpoint=torch.load(args.out/'preprocessor_last.pth',weights_only=True)
    model=load_preprocessor(checkpoint,'od')
    assert model.variant==variant and model.schema=='adaptive-vcm-conditional-v9'
    assert manifest['steps']==manifest['measurements']==10
    rows=[json.loads(line) for line in (args.out/'measurements.jsonl').read_text().splitlines()]
    assert {(r['codec'],r['qp']) for r in rows}=={(c,q) for c in ['h264','h265'] for q in [30,35,40,45,50]}
    assert all(len(r['profiles'])==(20 if variant=='b' else 12) for r in rows)
    assert all(len(r['baseline_profiles'])==12 and r['target_parameters']==canonical_target(r['target_profile'],variant)
               for r in rows)
    assert (args.out/'bank_headroom.json').exists()
    altered=[dict(r) for r in rows]
    altered[0]['target_parameters']=dict(admission=False,strength=.2,expert_weights=[0.,0.,0.,0.])
    with pytest.raises(ValueError,match='target'):
        training.validate_training_records(altered,[source_id],'od',checkpoint['profile_names'],variant)
