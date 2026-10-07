"""Verified oracle gate precedes every empirical FIT optimizer construction."""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import hashlib
import io
from pathlib import Path
import re

import numpy as np
import torch

from .actions import action_registry
from .guard import annotate_row,is_feasible
from .measure_store import atomic_bytes,atomic_json,load_source_artifact,sha
from .oracle import project_rows,validate_gate
from .protocol import SOURCE_COUNTS,canonical_hash,code_manifest_v31,validate_config
from .selector import (ActionSelector,CONTEXT_FIELDS,CONTEXT_SCHEMA,RUNTIME_FIELDS,TEACHER_FIELDS,
                       build_context,selector_loss)


CHECKPOINT_SCHEMA = 'adaptive-vcm-v31-selector-1'


def portable_rows(rows):
    return [{key:value for key,value in row.items() if key!='artifact_store'} for row in rows]


def measured_targets(row,policy):
    observations = annotate_row(row,policy)
    anchor = observations[0]['total_bytes']
    safety,rates,valid = [],[],[]
    for obs in observations[1:]:
        available = obs['available'] and type(obs['total_bytes']) is int and obs['total_bytes']>0
        valid.append(bool(available))
        safety.append(float(is_feasible(obs,obs['guard_features'],policy)))
        rates.append(float(np.log(obs['total_bytes']/anchor)) if available else 0.)
    return np.asarray(safety,np.float32),np.asarray(rates,np.float32),np.asarray(valid,bool)


def context_from_row(row,task):
    artifact = load_source_artifact(row['artifact_store'],row)
    sample = {key:value for key,value in row['source'].items() if key in RUNTIME_FIELDS}
    sample.update(task=task,rgb=artifact['rgb'],control_protection=artifact['control_protection'])
    for key in ('duration','source_fps'):
        sample[key] = Fraction(*sample[key]) if sample.get(key) is not None else None
    source = row['source_predictions']['teachers']
    anchor = row['anchor_predictions']['teachers']
    names = ['r3d_18','mc3_18'] if task=='ar' else ['mobilenet']
    return build_context(sample,[source[name] for name in names],[anchor[name] for name in names],
                         row['actions'][0],task,row['qp'],row['codec'],artifact['support'])


def train_arrays(model,context,safety,log_rate,valid,source_ids,cfg):
    """Numerical training kernel; empirical callers must use fit_selector."""
    cfg = validate_config(cfg)
    x,y,r,v = np.asarray(context,np.float32),np.asarray(safety,np.float32),np.asarray(log_rate,np.float32),np.asarray(valid,bool)
    if x.ndim!=2 or x.shape[1]!=48 or not len(x) or y.shape!=r.shape or y.shape!=v.shape or y.shape!=(len(x),len(model.action_names)-1) or len(source_ids)!=len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(r).all() or np.any((y!=0)&(y!=1)):
        raise ValueError('invalid FIT numerical training arrays')
    sources = sorted(set(source_ids))
    blocks = {source:np.flatnonzero(np.asarray(source_ids)==source) for source in sources}
    if any(len(block)!=8 for block in blocks.values()):
        raise ValueError('FIT requires full eight-condition source blocks')
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    x,y,r,v = torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(r),torch.from_numpy(v)
    mean,std = x.mean(dim=0),x.std(dim=0,unbiased=False)
    std = torch.where(std<1e-6,torch.ones_like(std),std)
    weight = ((len(y)-y.sum(dim=0))/y.sum(dim=0).clamp(min=1)).clamp(.25,20)
    with torch.no_grad():
        model.context_mean.copy_(mean); model.context_std.copy_(std); model.safety_log_weight.copy_(weight.log())
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg['optimizer']['lr'])
    history = []
    for epoch in range(cfg['epochs']):
        order = np.concatenate([blocks[sources[i]] for i in rng.permutation(len(sources))])
        totals = {'loss':0.,'safety_loss':0.,'rate_loss':0.,'pairwise_loss':0.}
        model.train()
        for start in range(0,len(order),cfg['optimizer']['batch_size']):
            indices = order[start:start+cfg['optimizer']['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss,parts = selector_loss(model,x[indices],y[indices],r[indices],v[indices],weight,cfg['min_savings'])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite FIT training loss')
            loss.backward(); optimizer.step()
            totals['loss'] += float(loss.detach())*len(indices)
            for key,value in parts.items(): totals[key]+=value*len(indices)
        model.eval()
        with torch.no_grad():
            logits,predicted = model(x)
            scores = model.scores(logits,predicted).cpu().numpy()
            proposals = np.argsort(-scores,axis=1,kind='stable')[:,:cfg['proposal_k']]
            utility = np.where((y.numpy()>.5)&v.numpy(),np.maximum(1-np.exp(r.numpy()),0),0)
            best = utility.max(axis=1)
            retrieved = np.take_along_axis(utility,proposals,axis=1).max(axis=1)
            helpful = best>=cfg['min_savings']
            retrieval = float((retrieved[helpful]>=best[helpful]-1e-7).mean()) if helpful.any() else None
        history.append({'epoch':epoch+1,**{key:value/len(x) for key,value in totals.items()},
                        'conditions':len(x),'sources':len(sources),'oracle_retrieval_at_k':retrieval,
                        'identity_only_conditions':int((~helpful).sum()),
                        'checkpoint_role':'LAST' if epoch+1==cfg['epochs'] else 'training'})
    return model.eval(),history


def state_sha256(state):
    digest = hashlib.sha256()
    for name,tensor in sorted(state.items()):
        if not isinstance(tensor,torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError('invalid selector checkpoint tensor')
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical_hash({'name':name,'shape':list(value.shape),'dtype':str(value.dtype)}).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_checkpoint(model,metadata):
    state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    metadata = dict(metadata,state_sha256=state_sha256(state),context_schema=CONTEXT_SCHEMA,
                    teacher_fields=list(TEACHER_FIELDS[model.task]),
                    normalization_hash=canonical_hash({'mean':state['context_mean'].tolist(),'std':state['context_std'].tolist()}))
    return {'metadata':metadata,'metadata_hash':canonical_hash(metadata),'model':state}


def load_selector(checkpoint,expected):
    metadata = checkpoint.get('metadata',{})
    if (metadata.get('schema')!=CHECKPOINT_SCHEMA or metadata.get('context_schema')!=CONTEXT_SCHEMA or
            metadata.get('context_fields')!=list(CONTEXT_FIELDS) or metadata.get('checkpoint_role')!='LAST' or
            checkpoint.get('metadata_hash')!=canonical_hash(metadata)):
        raise ValueError('incompatible selector checkpoint/schema/metadata')
    if state_sha256(checkpoint.get('model',{}))!=metadata.get('state_sha256'):
        raise ValueError('selector tensor SHA mismatch')
    task,arm = metadata.get('task'),metadata.get('arm')
    required = ('gate_hash','policy_hash','registry_hash','config_hash','fit_rows_hash','code_manifest_hash')
    if task not in SOURCE_COUNTS or arm not in ('a','b','c') or any(re.fullmatch('[0-9a-f]{64}',str(metadata.get(key,''))) is None for key in required):
        raise ValueError('selector oracle provenance missing')
    registry = action_registry(task,arm)
    if (metadata.get('action_names')!=[a.name for a in registry] or
            metadata['registry_hash']!=canonical_hash([asdict(a) for a in registry])):
        raise ValueError('selector frozen action registry mismatch')
    sources,history = metadata.get('training_source_ids',[]),metadata.get('history',[])
    count = SOURCE_COUNTS[task]['fit']
    if (len(sources)!=count or len(set(sources))!=count or any(type(s) is not str or not s for s in sources) or
            metadata.get('training_conditions')!=count*8 or metadata.get('epochs')!=8 or metadata.get('seed')!=303101 or
            len(history)!=8 or any(item.get('epoch')!=i+1 or item.get('conditions')!=count*8 or item.get('sources')!=count
                                  for i,item in enumerate(history)) or history[-1].get('checkpoint_role')!='LAST'):
        raise ValueError('selector FIT history/source provenance incomplete')
    for key,value in expected.items():
        if metadata.get(key)!=value:
            raise ValueError('selector expected identity mismatch: '+key)
    model = ActionSelector(metadata['task'],metadata['action_names'],metadata['width'])
    if metadata.get('teacher_fields')!=list(TEACHER_FIELDS[model.task]):
        raise ValueError('selector teacher context schema mismatch')
    model.load_state_dict(checkpoint['model'],strict=True)
    norm = {'mean':model.context_mean.tolist(),'std':model.context_std.tolist()}
    if torch.any(model.context_std<=0) or canonical_hash(norm)!=metadata.get('normalization_hash'):
        raise ValueError('selector normalization identity mismatch')
    model.metadata = metadata
    return model.eval()


def load_checkpoint(path,expected_sha,expected):
    data = Path(path).read_bytes()
    if sha(data)!=expected_sha:
        raise ValueError('checkpoint file SHA mismatch')
    payload = torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    return load_selector(payload,expected)


def fit_selector(fit_rows,gate,policy,cfg,out):
    # First operation: even corrupt/missing gates fail before optimizer/output.
    validate_gate(gate,{})
    cfg = validate_config(cfg)
    task,arm = gate['task'],cfg['v31_arm']
    registry = action_registry(task,arm)
    if any(row['split']!='fit' for row in fit_rows):
        raise ValueError('only FIT rows may supervise selector')
    rows = project_rows(fit_rows,registry)
    expected = {'task':task,'arm':arm,'config_hash':canonical_hash(cfg),'policy_hash':policy['policy_hash'],
                'registry_hash':canonical_hash([asdict(a) for a in registry]),
                'fit_rows_hash':canonical_hash(portable_rows(rows)),
                'code_manifest_hash':code_manifest_v31(Path(__file__).resolve().parents[2])['manifest_hash']}
    validate_gate(gate,expected)
    if policy['policy_hash']!=canonical_hash({k:v for k,v in policy.items() if k!='policy_hash'}):
        raise ValueError('frozen policy hash mismatch')
    contexts = np.stack([context_from_row(row,task) for row in rows])
    targets = [measured_targets(row,policy) for row in rows]
    safety,rates,valid = (np.stack([item[i] for item in targets]) for i in range(3))
    torch.manual_seed(cfg['seed'])
    model = ActionSelector(task,tuple(a.name for a in registry),cfg['width'])
    model,history = train_arrays(model,contexts,safety,rates,valid,[row['source_id'] for row in rows],cfg)
    metadata = {'schema':CHECKPOINT_SCHEMA,**expected,'action_names':list(model.action_names),'width':cfg['width'],
                'gate_hash':gate['gate_hash'],'context_fields':list(CONTEXT_FIELDS),'checkpoint_role':'LAST',
                'epochs':cfg['epochs'],'training_source_ids':sorted({row['source_id'] for row in rows}),
                'training_conditions':len(rows),'seed':cfg['seed'],'optimizer':cfg['optimizer'],
                'history':history,'loss':{'safety_bce':1.,'feasible_smooth_l1':2.,'beta':.1,'pairwise_utility':.25}}
    metadata.update(source_partitions=gate['integrity']['source_partitions'],
                    source_pixels_sha256=gate['integrity']['source_pixels_sha256'])
    payload = make_checkpoint(model,metadata)
    buffer = io.BytesIO(); torch.save(payload,buffer)
    data = buffer.getvalue()
    out = Path(out)
    atomic_bytes(out/'selector_last.pt',data)
    receipt = {'checkpoint':'selector_last.pt','checkpoint_sha256':sha(data),'metadata':payload['metadata'],
               'metadata_hash':payload['metadata_hash'],'history':history}
    atomic_json(out/'training.json',receipt)
    return receipt
