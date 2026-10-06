import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from adaptive_vcm.codec import Encoded, StandardCodec, locate_ffmpeg
from adaptive_vcm.data import fingerprint, partition
from adaptive_vcm.evaluate import ROOT, choose_stream
from adaptive_vcm.preprocessing import Candidate
from adaptive_vcm.rateaware import load_preprocessor


def cfg():
    return json.loads((ROOT / 'configs/v28_screen.json').read_text())


def state(task='ar'):
    from adaptive_vcm.motion_learned import MotionAwarePreprocessor, PROFILE_NAMES
    ids = [str(i) for i in range(20, 80) if partition(f'coco2017/{i}' if task == 'od' else str(i)) == 'train'][:2]
    model = MotionAwarePreprocessor(4, task)
    return {'schema': model.schema, 'task': task, 'width': 4, 'model': model.state_dict(),
            'epochs': 1, 'steps': 20, 'measurements': 20, 'train_count': 2,
            'train_ids': ids, 'train_ids_sha256': fingerprint(ids),
            'measurements_sha256': 'a' * 64, 'training_config': cfg(),
            'profile_names': list(PROFILE_NAMES), 'train_source_sha256':{i:('a' if n==0 else 'b')*64 for n,i in enumerate(ids)},
            'static_orders': {f'{c}/{q}': list(PROFILE_NAMES[:3]) for c in ('h264','h265') for q in (30,35,40,45,50)}}


@pytest.mark.parametrize('task', ['ar','od'])
def test_motion_checkpoint_loads_actual_neural_model_and_rejects_cross_task(task):
    from adaptive_vcm.motion_learned import MotionAwarePreprocessor
    assert isinstance(load_preprocessor(state(task), task), MotionAwarePreprocessor)
    with pytest.raises(ValueError):
        load_preprocessor(state(task), 'od' if task == 'ar' else 'ar')


@pytest.mark.parametrize('fault', ['incomplete','duplicate','nontrain','hash','static','nan','steps','source_hash'])
def test_motion_checkpoint_fails_closed_on_unverifiable_training(fault):
    value = state()
    if fault == 'incomplete': value['measurements'] -= 1
    elif fault == 'duplicate': value['train_ids'][1] = value['train_ids'][0]
    elif fault == 'nontrain': value['train_ids'][0] = next(str(i) for i in range(100) if partition(str(i)) == 'dev')
    elif fault == 'hash': value['measurements_sha256'] = 'unverified'
    elif fault == 'static': value['static_orders']['h264/50'] = ['not-registered'] * 3
    elif fault == 'nan': next(iter(value['model'].values())).fill_(float('nan'))
    elif fault=='source_hash': value.pop('train_source_sha256')
    else: value['steps'] -= 1
    with pytest.raises(ValueError): load_preprocessor(value, 'ar')


class ConstantTeacher:
    def probabilities(self, clip): return np.array([.8,.1,.1])
    def predict(self, clip):
        return {'boxes': np.empty((0,4)), 'scores': np.empty(0), 'labels': np.empty(0,dtype=int)}


class ProbeCodec:
    qp, codec = 50, 'h264'
    def roundtrip(self, clip):
        size = 100 + int(clip.mean()) * 10
        return Encoded(clip.copy(), bytes([int(clip.mean())]) * size, 0.)


class ProbeModel(nn.Module):
    schema, task = 'adaptive-vcm-motion-v7', 'ar'
    def __init__(self, names):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.static_orders = {'h264/50': list(names[:3])}
    def forward(self, video, qp, codec, protection, *, motion=None,cuts=None,strength_scale=1.,**kwargs):
        return video * (.5 - .1 * strength_scale)


def test_static_and_oracle_trials_never_enter_primary_selection(monkeypatch):
    import adaptive_vcm.motion_selection as selection
    from adaptive_vcm.motion_learned import PROFILE_NAMES
    source = np.full((2,16,24,3),200,np.uint8)
    config = cfg(); config['ar_candidates'] = ['identity']
    monkeypatch.setattr(selection, 'profile_candidates', lambda *args:
                        [Candidate(n, np.full_like(source,50 if i < 3 else 10)) for i,n in enumerate(PROFILE_NAMES)])
    model = ProbeModel(PROFILE_NAMES)
    bundle = choose_stream(source,np.zeros((16,24),np.float32),'ar',ProbeCodec(),config,
                           [ConstantTeacher(),ConstantTeacher()],[np.array([.8,.1,.1])]*2,model,components=True)
    assert bundle[2].startswith('learned_motion_')
    assert bundle[1].coded_bytes > bundle[4]['static_adaptive'][0].coded_bytes
    assert bundle[4]['static_adaptive'][0].coded_bytes > bundle[4]['profile_oracle'][0].coded_bytes
    assert bundle[4]['learned_raw'][1] == 'learned_motion_s100'
    assert sum(a['name'].startswith('learned_motion_') for a in bundle[3]) == 3


def test_unknown_od_foreground_retains_identity_without_guard_nan():
    teacher = ConstantTeacher()
    source = np.full((1,16,24,3),200,np.uint8)
    value=state('od'); model=load_preprocessor(value,'od')
    bundle=choose_stream(source,np.zeros((16,24),np.float32),'od',ProbeCodec(),cfg(),[teacher],
                         [teacher.predict(source)],model,components=True)
    assert bundle[2] == 'identity' and bundle[1].data == bundle[0].data
    assert len(bundle[3]) == 1
    json.dumps(bundle[3],allow_nan=False)
    assert all(s.data == bundle[0].data for s,n in bundle[4].values())


@pytest.mark.codec
@pytest.mark.parametrize('task',['ar','od'])
def test_motion_selection_retains_guard_and_paired_real_codec_geometry(task):
    if locate_ffmpeg() is None: pytest.skip('FFmpeg unavailable')
    from test_pipeline import TinyAction
    from test_od_pipeline import TinyDetector
    teacher_types = [TinyAction,TinyAction] if task == 'ar' else [TinyDetector]
    teachers = [t(str(i),'cpu') for i,t in enumerate(teacher_types)]
    source=np.random.default_rng(128).integers(80,160,(3 if task=='ar' else 1,32,48,3),np.uint8)
    mask=np.zeros((32,48),np.float32); mask[8:20,8:20]=1
    predictions=[t.probabilities(source) if task=='ar' else t.predict(source) for t in teachers]
    model=load_preprocessor(state(task),task)
    config=cfg(); config[f'{task}_candidates']=['identity']
    for codec in ('h264','h265'):
        bundle=choose_stream(source,mask,task,StandardCodec(codec,50),config,teachers,predictions,model,components=True)
        assert bundle[0].decoded.shape == source.shape == bundle[1].decoded.shape
        assert bundle[1].coded_bytes <= bundle[0].coded_bytes
        audit = next(a for a in bundle[3] if a['name']==bundle[2])
        assert all(audit['preserves_decision'])
        assert all(d is not None and d <= config['ar_kl_slack' if task=='ar' else 'od_distance_slack'] for d in audit['relative_task_distance'])
        assert len(bundle[3]) == 16


def test_evaluation_rejects_train_dev_pixel_overlap_before_selection(tmp_path,monkeypatch):
    import adaptive_vcm.evaluate as evaluation
    import hashlib
    source=np.full((2,32,48,3),100,np.uint8)
    value=state(); value['train_source_sha256'][value['train_ids'][0]]=hashlib.sha256(source.tobytes()).hexdigest()
    checkpoint=tmp_path/'model.pth'; torch.save(value,checkpoint)
    config=tmp_path/'cfg.json'; config.write_text(json.dumps(cfg()))
    dev_id=next(str(i) for i in range(100) if partition(str(i))=='dev')
    from test_pipeline import TinyAction
    monkeypatch.setattr(evaluation,'ar_plan',lambda *a:([{'id':dev_id,'path':'fixture','label':0}],{}))
    monkeypatch.setattr(evaluation,'read_video',lambda *a:source.copy())
    monkeypatch.setattr(evaluation,'ActionAnalyzer',TinyAction)
    def forbidden_selection(*args,**kwargs):
        raise RuntimeError('selection started before source-overlap rejection')
    monkeypatch.setattr(evaluation,'choose_stream',forbidden_selection)
    args=SimpleNamespace(config=config,task='ar',root=tmp_path,annotations=None,count=1,split='dev',
                         codecs=['h264','h265'],bootstrap=0,checkpoint=checkpoint,ablate_learned=True,
                         save_streams=False,out=tmp_path/'eval')
    with pytest.raises(ValueError,match='overlap'):
        evaluation.run(args)
