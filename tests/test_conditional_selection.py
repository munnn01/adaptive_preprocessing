"""Reject V30 provenance drift and keep baseline audit filters out of primary."""
import copy
import hashlib
import json
import importlib.util

import numpy as np
import pytest
import torch

from adaptive_vcm.data import fingerprint, partition
from adaptive_vcm.evaluate import ROOT, choose_stream
from adaptive_vcm.rateaware import load_preprocessor


def conditional():
    assert importlib.util.find_spec('adaptive_vcm.conditional_learned'), 'V30 renderer missing'
    from adaptive_vcm import conditional_learned
    return conditional_learned


def state(variant='a', task='od'):
    module = conditional()
    config = json.loads((ROOT/f'configs/v30_{variant}_screen.json').read_text())
    ids = [str(i) for i in range(100) if partition(f'coco2017/{i}' if task=='od' else str(i))=='train'][:2]
    model = module.ConditionalPreprocessor(4, task, variant)
    registry = module.profile_registry(variant)
    names = [r['name'] for r in registry]
    groups = [f'{c}/{q}' for c in ['h264','h265'] for q in [30,35,40,45,50]]
    from adaptive_vcm.motion_learned import PROFILE_NAMES
    from adaptive_vcm.train_motion import _code_manifest,_json_bytes
    code=_code_manifest()
    return dict(schema=model.schema, task=task, width=4, model=model.state_dict(),
        epochs=1, steps=20, measurements=20, train_count=2, train_ids=ids,
        train_ids_sha256=fingerprint(ids), fit_ids=ids, fit_ids_sha256=fingerprint(ids),
        calibration_ids=[], variant=variant, training_config=config,
        code=code,code_sha256=hashlib.sha256(_json_bytes(code)).hexdigest(),
        config_sha256=hashlib.sha256((ROOT/f'configs/v30_{variant}_screen.json').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        measurements_sha256='a'*64, profile_names=names, profile_registry=registry,
        train_source_sha256={i:'a'*64 for i in ids},
        static_orders={g:names[:3] for g in groups},
        baseline_static_orders={g:list(PROFILE_NAMES[:3]) for g in groups})


@pytest.mark.parametrize('variant', ['a','b','c'])
def test_v30_checkpoint_preserves_variant_registry_and_baseline_portfolio(variant):
    value = state(variant)
    model = load_preprocessor(value, 'od')
    assert model.schema == 'adaptive-vcm-conditional-v9' and model.variant==variant
    assert model.baseline_static_orders == value['baseline_static_orders']
    assert model.profile_registry == value['profile_registry']
    assert model.training is False


def test_linux_checkpoint_config_digest_survives_windows_newlines(monkeypatch):
    from pathlib import Path
    value=state('a')
    config_path=ROOT/'configs/v30_a_screen.json'
    read_bytes=Path.read_bytes
    windows=read_bytes(config_path).replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
    monkeypatch.setattr(Path,'read_bytes',lambda path:windows if path==config_path else read_bytes(path))
    assert load_preprocessor(value,'od').variant=='a'


@pytest.mark.parametrize('fault', ['variant','registry','baseline','steps','calibration',
                                  'code_missing','code_hash','renderer_drift','config_hash'])
def test_v30_checkpoint_fails_closed_on_contract_drift(fault):
    value = state('b')
    if fault=='variant': value['variant']='c'
    elif fault=='registry': value['profile_registry'][0]['strength']=.2
    elif fault=='baseline': value['baseline_static_orders']['h265/50']=['unknown']*3
    elif fault=='steps': value['steps']=10
    elif fault=='calibration': value['calibration_ids']=[value['train_ids'][0]]
    elif fault=='code_missing': value.pop('code')
    elif fault=='code_hash': value['code_sha256']='0'*64
    elif fault=='config_hash': value['config_sha256']='0'*64
    else:
        from adaptive_vcm.train_motion import _json_bytes
        value['code']['files_sha256']['adaptive_vcm/conditional_learned.py']='0'*64
        value['code_sha256']=hashlib.sha256(_json_bytes(value['code'])).hexdigest()
    with pytest.raises(ValueError): load_preprocessor(value, 'od')


def test_v30_primary_streams_are_codec_measured_and_never_include_audit_profiles():
    from adaptive_vcm.codec import StandardCodec
    from adaptive_vcm.motion_support import build_motion_support
    from test_motion_selection import ConstantTeacher
    value = state('b', 'ar')
    model = load_preprocessor(value, 'ar').eval()
    source = np.random.default_rng(12).integers(30, 200, (2, 16, 24, 3), dtype=np.uint8)
    protection = np.zeros(source.shape[1:3], np.float32)
    support = build_motion_support(source, protection, 'ar')
    teacher = ConstantTeacher()
    teachers = [teacher,teacher]
    _, selected, name, audit, arms = choose_stream(source,protection,'ar',StandardCodec('h264',50),
        value['training_config'],teachers,[teacher.probabilities(source)]*2,model,
        learned_mask=support,components=True)
    primary = [row for row in audit if row['primary_pool']]
    learned = [row for row in primary if row['name'].startswith('learned_motion_')]
    assert len(learned)==3
    assert all(row['coded_bytes'] > 0 and len(row['stream_sha256'])==64 for row in primary)
    assert name in [row['name'] for row in primary]
    assert {'static_baseline','profile_baseline_oracle','static_adaptive','profile_oracle'} <= arms.keys()
    assert selected.coded_bytes > 0
    anchor_pixels = next(row['pixel_sha256'] for row in primary if row['name']=='identity')
    for row in learned:
        action=row['conditional_action']
        assert 0<=action['gate_probability']<=1 and 0<=action['strength']<=1
        assert sum(action['expert_weights'])==pytest.approx(1.)
        if not action['admitted']:
            assert row['pixel_sha256']==anchor_pixels


def test_high_qp_failure_diagnostics_partition_points_and_count_actual_probes():
    from adaptive_vcm import motion_selection as selection
    function=getattr(selection,'conditional_selection_diagnostics',None)
    assert callable(function), 'V30 aggregate selection diagnostics are missing'
    cfg=json.loads((ROOT/'configs/v30_a_screen.json').read_text())
    from adaptive_vcm.motion_learned import PROFILE_NAMES
    def candidate(name,size,guard=True,primary=True):
        row=dict(name=name,coded_bytes=size,relative_task_distance=[0.],
            preserves_decision=[guard],primary_pool=primary,codec_seconds=.1,
            distinct_codec_encode=True,distinct_teacher_evaluation=True)
        if name.startswith('learned_'):
            row['conditional_action']=dict(gate_probability=.6,admitted=True,strength=.4,
                expert_weights=[1.,0.,0.,0.],output_edit_fraction=.25)
        return row
    records=[]
    for i,(size,guard) in enumerate([(999,True),(850,False),(920,True),(850,True)]):
        records.append(dict(id=str(i),codec='h265',qp=50,selected='control',candidates=[
            candidate('identity',1000),candidate('control',900),
            candidate('learned_motion_s100',size,guard),
            *[candidate(name,880 if name=='motion_mild_gaussian_040' else 1000,primary=False)
              for name in PROFILE_NAMES]]))
    report=function(records,cfg,'od')['h265/50']
    assert report['failure_categories']==dict(byte_threshold=1,teacher_guard=1,
        control_dominance=1,learned_extra_win=1,no_learned=0)
    assert report['points']==4 and report['decoded_teacher_model_calls']==60
    assert report['distinct_codec_encodes']==60 and report['actual_probe_codec_seconds']==pytest.approx(6.)
    assert report['gate_confusion_vs_reference']==dict(tp=4,tn=0,fp=0,fn=0)
    assert report['positive_reference_strength_mse']==0
    assert report['positive_reference_expert_kl']==0


def test_direct_evaluation_without_reference_audit_does_not_invent_negative_labels():
    from adaptive_vcm.motion_selection import conditional_selection_diagnostics
    cfg=json.loads((ROOT/'configs/v30_a_screen.json').read_text())
    candidates=[dict(name=name,coded_bytes=size,relative_task_distance=[0.],
        preserves_decision=[True],primary_pool=True,codec_seconds=.1)
        for name,size in [('identity',1000),('control',900),('learned_motion_s100',1000)]]
    candidates[-1]['conditional_action']=dict(gate_probability=.4,admitted=False,
        strength=.7,expert_weights=[.25]*4,output_edit_fraction=0.)
    report=conditional_selection_diagnostics([dict(codec='h265',qp=50,candidates=candidates)],cfg,'od')['h265/50']
    assert report['gate_reference_points']==0
    assert report['gate_accuracy_vs_reference'] is None
    assert report['identity_reference_points']==0
    assert report['identity_reference_output_edit_fraction'] is None
    assert report['failure_categories']['byte_threshold']==1
