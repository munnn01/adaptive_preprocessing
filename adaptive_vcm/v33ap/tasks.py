"""Frozen task weights, differentiable input losses and detached source caches."""
from contextlib import contextmanager
import hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..v32vc.train import tensor
from .protocol import arm_weights


def frame_indices(t):
    if type(t) is not int or t<1: raise ValueError('positive frame count required')
    return tuple(sorted({0,t//2,t-1}))


@contextmanager
def mode(model, training=False):
    states=[(m,m.training) for m in model.modules()]
    try:
        model.train(training)
        for m in model.modules():
            if isinstance(m,(nn.modules.batchnorm._BatchNorm,nn.modules.dropout._DropoutNd)): m.eval()
        yield
    finally:
        for m,s in states: m.training=s


def ar_forward(action,x):
    captured=[]
    hook=action.model.features[1].register_forward_hook(lambda m,a,o:captured.append(o))
    try:
        with mode(action.model): logits=action.logits(x)
        if len(captured)!=1 or captured[0].ndim!=5: raise ValueError('Swin feature hook unavailable')
        feature=F.adaptive_avg_pool3d(captured[0].permute(0,4,1,2,3),(2,4,4)).flatten(1)
        return logits,feature
    finally: hook.remove()


def od_forward(detector,x,indices,targets=None):
    captured=[]
    hook=detector.model.backbone.register_forward_hook(lambda m,a,o:captured.append(o))
    try:
        with mode(detector.model, targets is not None):
            result=detector.model([x[0,:,i] for i in indices],targets) if targets is not None else detector.model([x[0,:,i] for i in indices])
        if len(captured)!=1: raise ValueError('OD FPN feature hook unavailable')
        maps=captured[0]; maps={'0':maps} if isinstance(maps,torch.Tensor) else maps
        feature=torch.cat([F.adaptive_avg_pool2d(v,(4,4)).flatten(1) for _,v in sorted(maps.items())],1).mean(0,keepdim=True)
        return result,feature
    finally: hook.remove()


def prediction(value):
    return {k:v.detach().cpu().numpy().copy() for k,v in value.items() if k in ('boxes','labels','scores')}


def pseudo(value,threshold=.5,topk=20):
    keep=torch.where(value['scores']>=threshold)[0]
    keep=keep[torch.argsort(value['scores'][keep],descending=True)[:topk]]
    return {k:value[k][keep].detach().cpu().clone() for k in ('boxes','labels')}


class TaskHeads:
    def __init__(self,device,*,action=None,detector=None):
        if action is None:
            from ..v32rf.swin import SwinAnalyzer
            action=SwinAnalyzer(device)
        if detector is None:
            from ..analyzers import DetectionAnalyzer
            detector=DetectionAnalyzer('mobilenet',device)
        self.device,self.action,self.detector=device,action,detector
        for h in (action,detector): h.model.to(device).eval().requires_grad_(False)

    def source_targets(self,record):
        x=tensor(record.rgb,self.device); features,preds={},{}
        with torch.no_grad():
            if record.task=='ar':
                logits,f=ar_forward(self.action,x)
                preds['ar']=logits.softmax(-1)[0].cpu().numpy().copy(); features['ar']=f.detach().cpu()
            indices=frame_indices(len(record.rgb))
            boxes,f=od_forward(self.detector,x,indices)
            preds['od']={i:prediction(v) for i,v in zip(indices,boxes)}; features['od']=f.detach().cpu()
            if record.task=='ar': record.targets['pseudo']={i:pseudo(v) for i,v in zip(indices,boxes)}
        record.source_features,record.source_predictions=features,preds
        return preds

    def hashes(self):
        from ..v32vc.checkpoint import state_hash
        return {k:dict(name=h.name,state_sha256=state_hash(h.model.state_dict())) for k,h in (('ar',self.action),('od',self.detector))}


def distance(current,target):
    target=target.detach().to(current)
    if current.shape!=target.shape: raise ValueError('source feature shape mismatch')
    return (1-F.cosine_similarity(current,target,dim=1,eps=1e-8)).clamp(0,2).mean()


def losses(heads,reconstruction,record,arm):
    wa,wo=arm_weights(arm); zero=reconstruction.sum()*0
    result=dict(ar=zero,od=zero,feature=zero,active=[]); features=[]
    if wa and record.task=='ar':
        if 'label' not in record.item: raise ValueError('FIT AR label missing')
        logits,f=ar_forward(heads.action,reconstruction)
        label=torch.tensor([record.item['label']],device=logits.device,dtype=torch.long)
        target=torch.as_tensor(record.source_predictions['ar'],device=logits.device,dtype=logits.dtype)[None].detach()
        result['ar']=F.cross_entropy(logits,label)+.25*F.kl_div(logits.log_softmax(-1),target,reduction='batchmean')
        features.append(distance(f,record.source_features['ar'])); result['active'].append('ar')
    if wo:
        indices=frame_indices(reconstruction.shape[2])
        targets=([record.targets['pseudo'][i] for i in indices] if record.task=='ar' else [record.targets])
        targets=[{k:v.detach().to(reconstruction.device) for k,v in t.items()} for t in targets]
        terms,f=od_forward(heads.detector,reconstruction,indices,targets)
        if not isinstance(terms,dict) or not terms: raise ValueError('detector loss unavailable')
        result['od']=sum(terms.values())*(.25 if record.task=='ar' else 1.)
        features.append(distance(f,record.source_features['od'])); result['active'].append('od')
    if features: result['feature']=torch.stack(features).mean()
    if any(not torch.isfinite(result[k]) for k in ('ar','od','feature')): raise ValueError('nonfinite task loss')
    return result
