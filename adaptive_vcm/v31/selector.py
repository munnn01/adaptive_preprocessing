"""Label-free 48-scalar context and supervised nonidentity action retrieval."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..motion_learned import validate_support
from ..selection import detection_distance
from .guard import _probability
from .protocol import CODECS, QPS


CONTEXT_SCHEMA = 'v31-source-anchor-summary-48-v1'
COMMON_FIELDS = ('qp_div51','codec_h265','task_od','height_log2_div10','width_log2_div10','frames_div32',
                 'red_mean','green_mean','blue_mean','red_std','green_std','blue_std',
                 'motion_mean','motion_p95','gradient_y','gradient_x','protection_mean','protected_core_fraction',
                 'support_motion_mean','cut_fraction','anchor_log1p_bpp','anchor_log1p_rate_div10',
                 'duration_log1p','source_fps_div60')
CONTEXT_FIELDS = COMMON_FIELDS + tuple(f'teacher{teacher}_summary{i:02d}' for teacher in (1,2) for i in range(12))
TEACHER_FIELDS = {
    'ar':('source_max','anchor_max','source_margin','anchor_margin','source_entropy','anchor_entropy',
          'source_anchor_kl','anchor_source_kl','jensen_shannon','argmax_agreement','anchor_at_source_class','source_at_anchor_class'),
    'od':('source_count_div100','anchor_count_div100','source_score_mean','anchor_score_mean','source_score_max','anchor_score_max',
          'source_box_area_mean','anchor_box_area_mean','source_reliable_fraction','anchor_reliable_fraction','distance','has_reliable_source')}
RUNTIME_FIELDS = frozenset(('id','task','rgb','codec','padded','padding','duration','source_fps','source_transform',
                           'original_shape','control_protection','timing_status','timing_error','raw_sample_indices',
                           'sample_indices','source_sha256'))


def _ar_summary(source,anchor):
    for item in (source,anchor):
        if isinstance(item,dict) and set(item) != {'probabilities','logits'}:
            raise ValueError('runtime teacher contains unauthorized fields')
    s,a = _probability(source),_probability(anchor)
    if s.shape != a.shape:
        raise ValueError('inconsistent AR teacher summaries')
    tops,topa = np.sort(s)[-2:],np.sort(a)[-2:]
    logs,loga = np.log(np.maximum(s,1e-12)),np.log(np.maximum(a,1e-12))
    middle = np.log(np.maximum((s+a)/2,1e-12))
    return [tops[-1],topa[-1],tops[-1]-tops[-2],topa[-1]-topa[-2],
            -np.sum(s*logs)/math.log(len(s)),-np.sum(a*loga)/math.log(len(a)),
            np.clip(np.sum(s*(logs-loga))/20,0,1),np.clip(np.sum(a*(loga-logs))/20,0,1),
            .5*(np.sum(s*(logs-middle))+np.sum(a*(loga-middle)))/math.log(2),
            float(s.argmax()==a.argmax()),a[s.argmax()],s[a.argmax()]]


def _od_summary(source,anchor,height,width):
    values = []
    for item in (source,anchor):
        if set(item) != {'canonical','original'} or any(set(item[key]) != {'boxes','scores','labels'} for key in item):
            raise ValueError('runtime detection contains unauthorized fields')
        pred = item['canonical']
        detection_distance(pred,pred,.25)  # validates lengths/finite boxes/scores
        scores = np.asarray(pred['scores'],float)
        boxes = np.asarray(pred['boxes'],float).reshape(-1,4)
        if np.any(scores < 0) or np.any(scores > 1):
            raise ValueError('invalid runtime detection scores')
        area = np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(axis=1)/(height*width)
        values.append([len(scores)/100,float(scores.mean()) if len(scores) else 0.,
                       float(scores.max()) if len(scores) else 0.,float(area.mean()) if len(area) else 0.,
                       float((scores>=.25).mean()) if len(scores) else 0.])
    s,a = values
    distance = detection_distance(source['canonical'],anchor['canonical'],.25)
    return [s[0],a[0],s[1],a[1],s[2],a[2],s[3],a[3],s[4],a[4],
            distance if math.isfinite(distance) else 1.,float(math.isfinite(distance))]


def build_context(sample,source_teacher,anchor_teacher,anchor_packet,task,qp,codec,support):
    if not isinstance(sample,dict) or set(sample)-RUNTIME_FIELDS or task != sample.get('task'):
        raise ValueError('runtime sample contains unauthorized fields/task')
    if task not in ('ar','od') or type(qp) is not int or qp not in QPS or codec not in CODECS:
        raise ValueError('invalid V31 context operating point')
    rgb = sample['rgb']
    protection,motion_support,cuts = validate_support(rgb,support,task)
    if rgb.shape[:3] != ((16,128,128) if task=='ar' else (1,320,320)):
        raise ValueError('invalid primary context geometry')
    size = anchor_packet.get('total_bytes') if isinstance(anchor_packet,dict) else anchor_packet.total_bytes
    if type(size) is not int or size <= 32:
        raise ValueError('invalid transmitted anchor bytes')
    x = rgb.astype(np.float32)/255.
    temporal = np.abs(np.diff(x,axis=0)).mean(axis=-1) if len(x)>1 else np.zeros((1,*x.shape[1:3]))
    duration = sample.get('duration')
    fps = sample.get('source_fps')
    if task=='ar' and (not isinstance(duration,Fraction) or duration <= 0 or not isinstance(fps,Fraction) or fps<=0):
        raise ValueError('context needs verified exact AR timing')
    if task=='od':
        duration,fps = Fraction(1,25),None
    original = sample['original_shape']
    if len(original)!=2 or any(type(v) is not int or v<=0 for v in original):
        raise ValueError('invalid runtime original geometry')
    rate = size*8/float(duration) if task=='ar' else size*8/math.prod(original)
    values = [qp/51,float(codec=='h265'),float(task=='od'),math.log2(x.shape[1])/10,
              math.log2(x.shape[2])/10,len(x)/32,*x.mean(axis=(0,1,2)),*x.std(axis=(0,1,2)),
              float(temporal.mean()),float(np.percentile(temporal,95)),
              float(np.abs(np.diff(x,axis=1)).mean()),float(np.abs(np.diff(x,axis=2)).mean()),
              float(protection.mean()),float((protection>=.95).mean()),float(motion_support.mean()),float(cuts.mean()),
              math.log1p(size*8/np.prod(rgb.shape[:3])),math.log1p(rate)/10,math.log1p(float(duration)),float(fps)/60 if fps else 0.]
    count = 2 if task=='ar' else 1
    if len(source_teacher)!=count or len(anchor_teacher)!=count:
        raise ValueError('invalid runtime context teacher count')
    for source,anchor in zip(source_teacher,anchor_teacher):
        values.extend(_ar_summary(source,anchor) if task=='ar' else _od_summary(source,anchor,*rgb.shape[1:3]))
    if task=='od':
        values.extend([0.]*12)
    context = np.asarray(values,np.float32)
    if context.shape != (48,) or not np.isfinite(context).all():
        raise ValueError('invalid runtime context scalars')
    return context


class ActionSelector(nn.Module):
    def __init__(self,task,action_names,width=64):
        super().__init__()
        names = tuple(action_names)
        if task not in ('ar','od') or type(width) is not int or width!=64 or len(names)<2 or names[0]!='identity' or len(set(names))!=len(names) or any(type(n) is not str or not n for n in names):
            raise ValueError('invalid frozen V31 selector architecture/registry')
        self.task,self.action_names,self.width = task,names,width
        self.encoder = nn.Sequential(nn.Linear(48,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.safety_head,self.rate_head = nn.Linear(width,len(names)-1),nn.Linear(width,len(names)-1)
        nn.init.zeros_(self.rate_head.weight); nn.init.constant_(self.rate_head.bias,-.03)
        self.register_buffer('safety_log_weight',torch.zeros(len(names)-1))
        self.register_buffer('context_mean',torch.zeros(48))
        self.register_buffer('context_std',torch.ones(48))

    def forward(self,context):
        x = torch.as_tensor(context,dtype=self.rate_head.weight.dtype,device=self.rate_head.weight.device)
        if x.ndim==1: x=x[None]
        if x.ndim!=2 or x.shape[1]!=48 or not torch.isfinite(x).all():
            raise ValueError('invalid V31 selector context')
        hidden = self.encoder((x-self.context_mean)/self.context_std)
        return self.safety_head(hidden),self.rate_head(hidden)

    def scores(self,logits,log_rate):
        return (logits-self.safety_log_weight).sigmoid()*(1-log_rate.clamp(-10,10).exp()).clamp(min=0.)

    @torch.no_grad()
    def rank(self,context,top_k=3,available=None):
        if type(top_k) is not int or top_k<1:
            raise ValueError('invalid proposal budget')
        logits,rates = self(context)
        if len(logits)!=1:
            raise ValueError('rank expects one operating point')
        scores = self.scores(logits,rates)[0].cpu().numpy()
        if available is None:
            valid = np.ones(len(self.action_names),bool)
        else:
            valid = np.asarray(available)
            if valid.shape!=(len(self.action_names),) or valid.dtype!=np.bool_:
                raise ValueError('invalid action availability')
        return [int(i+1) for i in np.argsort(-scores,kind='stable') if valid[i+1]][:top_k]


def selector_loss(model,context,safety,log_rate,valid,positive_weight,min_savings=.01):
    logits,predicted = model(context)
    if safety.shape!=logits.shape or log_rate.shape!=logits.shape or valid.shape!=logits.shape:
        raise ValueError('inconsistent all-action supervision')
    safety_loss = F.binary_cross_entropy_with_logits(logits,safety,pos_weight=positive_weight)
    mask = valid & (safety>.5)
    rate_loss = F.smooth_l1_loss(predicted[mask],log_rate[mask],beta=.1) if mask.any() else predicted.sum()*0
    saving = (1-log_rate.exp()).clamp(min=0)
    utility = torch.where(mask & (saving>=min_savings),saving,torch.zeros_like(saving))
    pairs = utility[:,:,None]-utility[:,None,:]>=min_savings
    preference = F.logsigmoid(logits-model.safety_log_weight)-predicted
    margins = preference[:,:,None]-preference[:,None,:]
    pair_loss = F.softplus(-margins[pairs]).mean() if pairs.any() else logits.sum()*0
    return safety_loss+2*rate_loss+.25*pair_loss,{'safety_loss':float(safety_loss.detach()),
               'rate_loss':float(rate_loss.detach()),'pairwise_loss':float(pair_loss.detach())}
