"""Locked shared-model recipe, source namespaces and actual codec settings."""
import hashlib
import json
from pathlib import Path
import re
from fractions import Fraction
from ..data import partition
from ..v32.protocol import ROOT
from ..v32rf.protocol import codec_factory as rf_codec_factory
from . import SCHEMA


def settings(path=None):
    cfg = json.loads((Path(path) if path else ROOT/'configs/v33ap.json').read_text())
    expected = dict(schema=SCHEMA, seed=303251, width=12, epochs=2, lr=1e-4, train_count=64,
        feature_weight=.1, distortion_weight=10., rate_weight=.01, temporal_weight=.1,
        step_scales=[.5,1.,2.], qps=[30,35,40,45,50], codecs=['h264','h265'],
        preset='medium', fps=25, frames=16, temporal_stride=2, ar_size=128, od_size=320,
        ar_confidence=.6, ar_require_anchor_decision=True, ar_kl_slack=.1,
        od_distance_slack=.03, od_score_threshold=.25, min_savings=.01,
        pseudo_threshold=.5, pseudo_topk=20, pseudo_weight=.25, ar_kl_weight=.25,
        checkpoint_role='LAST', roi=dict(margin=.05, factors=[2,4]))
    if cfg != expected: raise ValueError('V33-AP recipe drift')
    return cfg


def arm_weights(arm):
    if arm not in ('joint','ar_only','od_only'): raise ValueError('invalid V33-AP arm')
    return {'joint':(1.,1.), 'ar_only':(1.,0.), 'od_only':(0.,1.)}[arm]


def source_id(task, raw_id):
    if (task not in ('ar','od') or not isinstance(raw_id,str) or not raw_id
        or '\\' in raw_id or ':' in raw_id or raw_id.startswith('/')
        or any(p in ('','..','.') for p in raw_id.split('/'))
        or task=='od' and re.fullmatch(r'[0-9]+',raw_id) is None):
        raise ValueError('invalid source ID/task')
    return ('od/coco2017/' if task=='od' else 'ar/')+raw_id


def split_train(ids, task):
    if task not in ('ar','od') or len(ids)!=64 or len(set(ids))!=64:
        raise ValueError('64 distinct TRAIN sources required')
    prefix = 'od/coco2017/' if task=='od' else 'ar/'
    if any(not isinstance(i,str) or not i.startswith(prefix) or partition(i[3:])!='train' for i in ids):
        raise ValueError('source outside matching TRAIN domain')
    ranked = sorted(ids,key=lambda i:hashlib.sha256(('303251/'+i).encode()).hexdigest())
    return ranked[16:], ranked[4:16], ranked[:4]


def codec_factory(task, preset='medium'):
    if task not in ('ar','od') or preset not in ('medium','fast','ultrafast'):
        raise ValueError('invalid codec domain/preset')
    from .codec import MeasuredCodec
    from ..v32rf.protocol import execution
    def create(codec,qp,base_preset='medium',fps=Fraction(25)):
        e=execution(task,preset)
        return MeasuredCodec(codec,qp,e['preset'],fps,gop_settings=e['gop_settings'])
    return create
