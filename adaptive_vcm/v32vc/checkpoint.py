"""Strict task/config/state provenance and FIT+CAL overlap protection."""
import hashlib
from pathlib import Path
import re
import numpy as np
import torch
from . import SCHEMA
from .model import ResidualPreprocessor
from ..data import partition
from ..v32.protocol import canonical_hash, provenance, load_config


def state_hash(state):
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError('invalid/nonfinite model state')
        value = value.detach().cpu().contiguous()
        digest.update(canonical_hash([name,list(value.shape),str(value.dtype)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_checkpoint(model,metadata):
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    metadata = dict(metadata,schema=SCHEMA,checkpoint_role='LAST',
                    state_sha256=state_hash(state),protocol=load_config(),code=provenance(load_config()))
    return dict(metadata=metadata,metadata_sha256=canonical_hash(metadata),model=state)


class LearnedPreprocessor:
    def __init__(self,model,metadata,path_hash):
        self.model,self.metadata = model,metadata
        self.manifest = dict(checkpoint_sha256=path_hash,**metadata)

    def assert_source(self,identifier,rgb):
        hashes = self.metadata['source_pixel_sha256']
        if identifier in hashes or hashlib.sha256(rgb.tobytes()).hexdigest() in hashes.values():
            raise ValueError('held-out source overlaps V32-VC FIT/CAL')

    @torch.no_grad()
    def pixels(self,rgb,support,codec):
        device = next(self.model.parameters()).device
        x = torch.from_numpy(rgb.copy()).to(device).float().permute(3,0,1,2)[None]/255
        p,m = [torch.from_numpy(np.asarray(support[k]).copy()).to(x)[None,None] for k in ('protection','motion')]
        c = torch.from_numpy(support['cuts'].copy()).to(device)[None]
        y = self.model(x,codec.qp,int(codec.codec=='h265'),p,m,c)
        return y[0].permute(1,2,3,0).mul(255).round().clamp(0,255).byte().cpu().numpy()


def load_checkpoint(path,task,device='cpu'):
    path = Path(path)
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict): raise ValueError('invalid checkpoint')
    meta = payload.get('metadata',{})
    if (meta.get('schema')!=SCHEMA or meta.get('checkpoint_role')!='LAST' or meta.get('task')!=task
            or meta.get('arm') not in ('plain','joint') or meta.get('width')!=12
            or meta.get('seed')!=303221 or meta.get('epochs')!=4
            or canonical_hash(meta)!=payload.get('metadata_sha256')
            or state_hash(payload.get('model',{}))!=meta.get('state_sha256')):
        raise ValueError('checkpoint task/schema/state/metadata mismatch')
    fit,cal = meta.get('fit_ids',[]),meta.get('cal_ids',[])
    ids = fit+cal; hashes = meta.get('source_pixel_sha256',{})
    if (not fit or not cal or any(type(i)!=str or not i for i in ids)
            or len(ids)!=len(set(ids)) or set(hashes)!=set(ids)
            or len(set(hashes.values()))!=len(hashes)
            or any(re.fullmatch('[0-9a-f]{64}',str(h)) is None for h in hashes.values())
            or any(partition('coco2017/'+i if task=='od' else i)!='train' for i in ids)):
        raise ValueError('invalid FIT/CAL source provenance')
    from .train import settings, split_train
    if (meta.get('settings')!=settings() or len(fit)!=48 or len(cal)!=16
            or (fit,cal)!=split_train(ids,task,303221)):
        raise ValueError('checkpoint differs from locked FIT48/CAL16 recipe')
    if (re.fullmatch('[0-9a-f]{64}',str(meta.get('supervision_sha256',''))) is None
            or task=='od' and re.fullmatch('[0-9a-f]{64}',str(meta.get('annotations_sha256',''))) is None):
        raise ValueError('supervised target provenance missing')
    expected = {f'{c}/{q}' for c in ('h264','h265') for q in (30,35,40,45,50)}
    if set(meta.get('calibration',{}))!=expected or meta.get('protocol')!=load_config():
        raise ValueError('incomplete calibration/protocol mismatch')
    for entry in meta['calibration'].values():
        if (not isinstance(entry,dict) or any(k not in entry for k in ('step','rate_scale','rate_offset'))
                or any(not np.isfinite(entry[k]) for k in ('step','rate_scale','rate_offset'))
                or min(entry['step'],entry['rate_scale'])<=0 or entry['rate_offset']<0):
            raise ValueError('invalid virtual calibration')
    history = meta.get('history',[])
    if len(history)!=4 or [r.get('epoch') for r in history]!=[1,2,3,4]:
        raise ValueError('incomplete LAST history')
    for row in history:
        if row.get('steps')!=480 or any(type(row.get(k)) not in (int,float) or not np.isfinite(row[k])
            or row[k]<0 for k in ('seconds','loss','task_loss','distortion','estimated_bpp','gradient_norm')):
            raise ValueError('incomplete/nonfinite locked epoch history')
    if meta.get('code',{}).get('code_files_sha256')!=provenance(load_config())['code_files_sha256']:
        raise ValueError('checkpoint runtime source mismatch')
    model = ResidualPreprocessor(task,meta['width']).to(device)
    model.load_state_dict(payload['model'],strict=True)
    return LearnedPreprocessor(model.eval(),meta,hashlib.sha256(path.read_bytes()).hexdigest())


def validate_pair(models):
    if set(models)!={'plain','joint'} or any(m.metadata.get('arm')!=a for a,m in models.items()):
        raise ValueError('plain/joint checkpoint arms required')
    keys = ('task','width','seed','epochs','fit_ids','cal_ids','source_pixel_sha256','calibration',
            'settings','protocol','code','supervision_sha256','annotations_sha256')
    if any(models['plain'].metadata.get(k)!=models['joint'].metadata.get(k) for k in keys):
        raise ValueError('plain/joint training/calibration provenance differs')
