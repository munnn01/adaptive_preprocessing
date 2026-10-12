"""One shared LAST checkpoint; strict recipe, runtime and source provenance."""
import hashlib
from pathlib import Path
import re
import numpy as np
import torch
from ..v32.protocol import canonical_hash,provenance
from ..v32vc.checkpoint import state_hash,LearnedPreprocessor
from .model import JointPreprocessor
from .protocol import settings,split_train,arm_weights
from .sources import validate_held_out


def native_metadata(value):
    """Keep checkpoint metadata independent of NumPy pickle globals."""
    if isinstance(value,np.generic): return native_metadata(value.item())
    if isinstance(value,dict):
        if any(type(k) is not str for k in value): raise TypeError('metadata keys must be strings')
        return {k:native_metadata(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [native_metadata(v) for v in value]
    if value is None or type(value) in (str,bool,int,float): return value
    raise TypeError('unsupported checkpoint metadata type: '+type(value).__name__)


def make_checkpoint(model,metadata):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    m=native_metadata(dict(metadata,schema=settings()['schema'],checkpoint_role='LAST',state_sha256=state_hash(state),code=provenance(settings())))
    return dict(metadata=m,metadata_sha256=canonical_hash(m),model=state)


class LearnedJointPreprocessor(LearnedPreprocessor):
    pass


def is_hash(value): return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def load_checkpoint(path,device='cpu'):
    path=Path(path); v=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(v,dict) or set(v)!={'metadata','metadata_sha256','model'}: raise ValueError('invalid shared checkpoint')
    m=v['metadata']; cfg=settings()
    if (m.get('schema')!=cfg['schema'] or m.get('checkpoint_role')!='LAST' or m.get('settings')!=cfg
        or canonical_hash(m)!=v['metadata_sha256'] or state_hash(v['model'])!=m.get('state_sha256')): raise ValueError('shared checkpoint state/recipe mismatch')
    arm_weights(m.get('arm'))
    splits=m.get('splits',{}); ids=[]
    if set(splits)!={'ar','od'}: raise ValueError('both domains required')
    for task in ('ar','od'):
        s=splits[task]
        if set(s)!={'fit','cal','val'}: raise ValueError('invalid split schema')
        domain=s['fit']+s['cal']+s['val']
        if (s['fit'],s['cal'],s['val'])!=split_train(domain,task): raise ValueError('invalid split ordering')
        ids+=domain
    hashes=m.get('source_pixel_sha256',{})
    if set(hashes)!=set(ids) or len(set(hashes.values()))!=128 or not all(is_hash(h) for h in hashes.values()): raise ValueError('invalid source hashes')
    validate_held_out(m.get('held_out',{}),hashes)
    if not all(is_hash(m.get(k)) for k in ('annotations_sha256','supervision_sha256')): raise ValueError('missing target provenance')
    models=m.get('task_models',{})
    if set(models)!={'ar','od'} or any(not is_hash(h.get('state_sha256')) or not h.get('name') for h in models.values()): raise ValueError('missing task model provenance')
    calibration=m.get('calibration',{}); keys={f'{c}/{q}' for c in cfg['codecs'] for q in cfg['qps']}
    if set(calibration)!={'ar','od'}: raise ValueError('both domain calibrations required')
    for entries in calibration.values():
        if set(entries)!=keys: raise ValueError('incomplete codec grid calibration')
        for e in entries.values():
            if any(type(e.get(k)) not in (float,int) or not np.isfinite(e[k]) for k in ('step','rate_scale','rate_offset')) or min(e['step'],e['rate_scale'])<=0 or e['rate_offset']<0: raise ValueError('invalid calibration')
    history=m.get('history',[])
    if len(history)!=2 or [r.get('epoch') for r in history]!=[1,2]: raise ValueError('incomplete LAST history')
    for r in history:
        if r.get('steps')!=960 or any(type(r.get(k)) not in (int,float) or not np.isfinite(r[k]) or r[k]<0 for k in ('loss','seconds','actual_bpp','estimated_bpp','gradient_norm','actual_codec_seconds')) or r['actual_bpp']<=0: raise ValueError('invalid LAST measurements')
    if m.get('code',{}).get('code_files_sha256')!=provenance(cfg)['code_files_sha256']: raise ValueError('runtime source mismatch')
    model=JointPreprocessor(cfg['width']).to(device); model.load_state_dict(v['model'],strict=True)
    return LearnedJointPreprocessor(model.eval().requires_grad_(False),m,hashlib.sha256(path.read_bytes()).hexdigest())
