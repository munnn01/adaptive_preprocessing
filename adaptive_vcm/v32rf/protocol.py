"""Locked paired recipe and explicitly executed standard-codec conditions."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from ..data import partition
from ..v32.protocol import ROOT
from ..v32.codec import V32Codec


def settings(path=None):
    cfg=json.loads((Path(path) if path else ROOT/'configs/v32rf.json').read_text())
    expected=dict(schema='adaptive-vcm-v32rf-1',seed=303231,width=12,epochs=2,lr=1e-4,
        rate_weight=.01,distortion_weight=10.,feature_weight=.1,step_scales=[.5,1.,2.],
        checkpoint_role='LAST',warm_origin='28cc97db6708c87346c025107a8a7ea485c21163',
        primary_preset='medium',transfer_preset='fast')
    if cfg!=expected: raise ValueError('V32-RF recipe drift')
    return cfg


def split_cal(ids,task):
    if task not in ('ar','od') or len(ids)!=16 or len(set(ids))!=16:
        raise ValueError('sixteen distinct original CAL sources required')
    if any(partition('coco2017/'+i if task=='od' else i)!='train' for i in ids):
        raise ValueError('held-out source cannot enter proxy calibration/validation')
    ranked=sorted(ids,key=lambda i:hashlib.sha256(f'303231/{i}'.encode()).hexdigest())
    return ranked[4:],ranked[:4]


def execution(task,preset='medium'):
    if task not in ('ar','od') or preset not in ('medium','fast','ultrafast'):
        raise ValueError('invalid RF codec execution conditions')
    return dict(preset=preset,gop_settings=dict(gop=16 if task=='ar' else 1,
        bframes=0,scenecut=False,closed_gop=True))


def codec_factory(task,preset=None):
    def create(codec,qp,base_preset='medium',fps=Fraction(25)):
        conditions=execution(task,preset or base_preset)
        return V32Codec(codec,qp,conditions['preset'],fps,gop_settings=conditions['gop_settings'])
    return create
