"""V28 geometry-preserving neural proposals with measured teacher guards.

The primary pool contains old controls and exactly three neural renderings.
TRAIN static profiles and the full reference-profile oracle are audit arms.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np
import torch

from .data import fingerprint, partition
from .motion_learned import MotionAwarePreprocessor, PROFILE_NAMES, profile_candidates
from .motion_support import build_motion_support
from .preprocessing import Candidate, make_candidates
from .selection import Observation, relative_guard, select

SCHEMA = 'adaptive-vcm-motion-v7'
SEMANTIC_SCHEMA = 'adaptive-vcm-semantic-v8'
PROPOSALS = (('learned_motion_s050', .5), ('learned_motion_s100', 1.), ('learned_motion_s150', 1.5))


def validate_motion_config(cfg):
    if 'v29_variant' in cfg and cfg['v29_variant'] not in ('a', 'b', 'c'):
        raise ValueError('invalid V29 variant')
    if (cfg.get('qps') != [30,35,40,45,50] or cfg.get('ar_mode') != 'motion_spatial'
            or cfg.get('od_mode') != 'motion_spatial' or cfg.get('motion_static_k') != 3
            or cfg.get('motion_proposal_scales') != [.5,1.,1.5]
            or cfg.get('ar_require_anchor_decision') is not True
            or cfg.get('ar_kl_slack') != .1 or cfg.get('od_distance_slack') != .03
            or cfg.get('min_savings') != .01):
        raise ValueError('V28 requires registered complete grid, three proposals and strict guards')


def load_motion_preprocessor(state, task):
    """Reject unverifiable/incomplete training; no implicit untrained fallback."""
    semantic = state.get('schema') == SEMANTIC_SCHEMA
    if state.get('schema') not in (SCHEMA, SEMANTIC_SCHEMA) or task not in ('ar','od') or state.get('task') != task:
        raise ValueError('motion checkpoint task/schema mismatch')
    cfg = state.get('training_config', {})
    validate_motion_config(cfg)
    variant = cfg.get('v29_variant')
    if semantic != (variant is not None) or semantic and state.get('variant') != variant:
        raise ValueError('semantic checkpoint/configuration variant mismatch')
    ids=state.get('train_ids', [])
    count, epochs = state.get('train_count'), state.get('epochs')
    if (type(count) is not int or count < 1 or type(epochs) is not int or epochs < 1
            or not isinstance(ids,list) or len(ids)!=count or any(not isinstance(i,str) or not i for i in ids)):
        raise ValueError('invalid motion TRAIN provenance')
    if fingerprint(ids) != state.get('train_ids_sha256'):
        raise ValueError('motion TRAIN identity hash mismatch')
    if any(partition(f'coco2017/{i}' if task=='od' else i) != 'train' for i in ids):
        raise ValueError('motion checkpoint contains non-TRAIN sources')
    fit_ids = ids
    if semantic:
        fit_ids = state.get('fit_ids', [])
        calibration_ids = state.get('calibration_ids', [])
        if (not isinstance(fit_ids, list) or not fit_ids or not isinstance(calibration_ids, list)
                or any(type(i) is not str for i in fit_ids + calibration_ids)
                or len(set(fit_ids)) != len(fit_ids) or len(set(calibration_ids)) != len(calibration_ids)
                or set(fit_ids) & set(calibration_ids) or set(fit_ids + calibration_ids) != set(ids)):
            raise ValueError('invalid disjoint semantic TRAIN partition')
        if variant == 'c':
            if (len(calibration_ids) != max(1, count // 4)
                    or type(state.get('calibration_measurements')) is not int
                    or state['calibration_measurements'] != 10 * len(calibration_ids)
                    or state.get('fit_ids_sha256') != fingerprint(fit_ids)
                    or state.get('calibration_ids_sha256') != fingerprint(calibration_ids)
                    or not re.fullmatch(r'[0-9a-f]{64}', str(state.get('calibration_measurements_sha256', '')))):
                raise ValueError('missing semantic calibration provenance')
            policy = state.get('admission_policy', {})
            groups = {f'{c}/{q}' for c in ('h264', 'h265') for q in cfg['qps']}
            if not isinstance(policy, dict) or set(policy) != groups:
                raise ValueError('incomplete semantic calibration policy')
            for entry in policy.values():
                if not isinstance(entry, dict):
                    raise ValueError('malformed semantic calibration policy')
                threshold, enabled = entry.get('threshold'), entry.get('enabled')
                points, support = entry.get('calibration_points'), entry.get('n_nonworsening')
                if (type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold > 0
                        or type(enabled) is not bool or type(points) is not int or points != len(calibration_ids)
                        or type(support) is not int or not 0 <= support <= 3 * points
                        or enabled != (support > 0)):
                    raise ValueError('invalid non-worsening semantic calibration policy')
            try:
                policy_hash = hashlib.sha256(json.dumps(
                    policy, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
            except (TypeError, ValueError) as error:
                raise ValueError('malformed semantic calibration policy') from error
            if state.get('admission_policy_sha256') != policy_hash:
                raise ValueError('semantic calibration policy hash mismatch')
        elif calibration_ids or fit_ids != ids or state.get('admission_policy'):
            raise ValueError('unexpected calibration in uncalibrated semantic variant')
    source_hashes=state.get('train_source_sha256',{})
    if (not isinstance(source_hashes,dict) or set(source_hashes)!=set(ids)
            or any(not isinstance(v,str) or not re.fullmatch(r'[0-9a-f]{64}',v) for v in source_hashes.values())):
        raise ValueError('missing motion TRAIN source-pixel hashes')
    if (type(state.get('measurements')) is not int or state['measurements'] != count*10
            or type(state.get('steps')) is not int or state['steps'] != epochs*len(fit_ids)*10
            or not re.fullmatch(r'[0-9a-f]{64}',str(state.get('measurements_sha256','')))):
        raise ValueError('incomplete motion grid/optimization or missing measurement hash')
    if state.get('profile_names') != list(PROFILE_NAMES):
        raise ValueError('motion profile registry mismatch')
    orders=state.get('static_orders',{})
    groups={f'{c}/{q}' for c in ('h264','h265') for q in cfg['qps']}
    if not isinstance(orders,dict) or set(orders)!=groups:
        raise ValueError('incomplete motion static groups')
    for names in orders.values():
        if (not isinstance(names,list) or len(names)!=3 or any(not isinstance(n,str) for n in names)
                or len(set(names))!=3 or set(names)-set(PROFILE_NAMES)):
            raise ValueError('invalid motion static profile order')
    if type(state.get('width')) is not int or state['width'] < 4:
        raise ValueError('invalid motion model width')
    weights=state.get('model',{})
    if (not isinstance(weights,dict) or not weights
            or any(not isinstance(v,torch.Tensor) or not v.is_floating_point()
                   or not torch.isfinite(v).all() for v in weights.values())):
        raise ValueError('nonfinite or malformed motion model state')
    model=(MotionAwarePreprocessor(state['width'],task,variant=variant) if semantic
           else MotionAwarePreprocessor(state['width'],task))
    try:
        model.load_state_dict(weights,strict=True)
    except (RuntimeError,TypeError) as error:
        raise ValueError('incompatible motion model state') from error
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('nonfinite motion model after dtype conversion')
    model.static_orders={group:list(names) for group,names in orders.items()}
    model.training_measurements_sha256=state['measurements_sha256']
    model.train_source_sha256=dict(source_hashes)
    if variant == 'c':
        model.admission_policy={group:dict(entry) for group,entry in state['admission_policy'].items()}
    return model


def support_hash(support):
    digest=hashlib.sha256()
    for key in ('protection','motion','cuts'):
        digest.update(np.ascontiguousarray(support[key]).tobytes())
    digest.update(json.dumps(support['metadata'],sort_keys=True,allow_nan=False).encode())
    return digest.hexdigest()


def neural_candidates(clip,support,codec,model):
    device=next(model.parameters()).device
    source=torch.from_numpy(clip.copy()).to(device).float().permute(3,0,1,2)[None]/255
    protection=torch.from_numpy(support['protection'].copy()).to(source)[None,None]
    motion=torch.from_numpy(support['motion'].copy()).to(source)[None,None]
    cuts=torch.from_numpy(support['cuts'].copy()).to(device)[None]
    candidates=[]
    with torch.no_grad():
        for name,scale in PROPOSALS:
            output=model(source,source.new_tensor([codec.qp]),source.new_tensor([int(codec.codec=='h265')]),
                         protection,motion=motion,cuts=cuts,strength_scale=scale)
            if not torch.isfinite(output).all():
                raise ValueError(f'nonfinite neural proposal: {name}')
            pixels=output[0].permute(1,2,3,0).mul(255).round().clamp(0,255).byte().cpu().numpy()
            candidates.append(Candidate(name,pixels))
    return candidates


def choose_motion_stream(clip,protection,task,codec,cfg,teachers,source_predictions,model,
                         *, learned_mask=None,components=False):
    validate_motion_config(cfg)
    if getattr(model,'task',None)!=task:
        raise ValueError('motion model/selection task mismatch')
    variant = cfg.get('v29_variant')
    if variant != getattr(model, 'variant', None):
        raise ValueError('motion model/selection variant mismatch')
    known=task=='ar' or np.any(source_predictions[0]['scores']>=cfg['od_score_threshold'])
    controls=make_candidates(clip,protection,task,codec.qp,cfg[f'{task}_candidates'])
    support=None
    if known:
        support=(learned_mask if isinstance(learned_mask,dict) else build_motion_support(
            clip,protection if learned_mask is None else learned_mask,task))
        learned=neural_candidates(clip,support,codec,model)
    else:
        controls=controls[:1]; learned=[]
    candidates=controls+learned
    primary_count=len(candidates)
    static_names=model.static_orders[f'{codec.codec}/{codec.qp}'] if known else []
    if components and known:
        candidates+=(profile_candidates(clip,support,task,codec.qp,variant=variant) if variant is not None
                     else profile_candidates(clip,support,task,codec.qp))
    # All proposals are fixed before a nonidentity outcome is observed.
    encoded,predictions,observations,original_observations,audit=[],[],[],[],[]
    slack=cfg['ar_kl_slack' if task=='ar' else 'od_distance_slack']
    for index,candidate in enumerate(candidates):
        stream=codec.roundtrip(candidate.clip)
        same=bool(encoded and stream.data==encoded[0].data)
        trial=(predictions[0] if same else [t.probabilities(stream.decoded) if task=='ar'
                                         else t.predict(stream.decoded) for t in teachers])
        distances,decisions=((tuple(0. for _ in teachers),tuple(True for _ in teachers))
                             if index==0 or same else relative_guard(task,source_predictions,predictions[0],trial,cfg))
        encoded.append(stream); predictions.append(trial)
        original = Observation(candidate.name,stream.coded_bytes,distances,decisions)
        original_observations.append(original)
        admission = None
        if variant == 'c' and candidate.name.startswith('learned_motion_'):
            policy = model.admission_policy[f'{codec.codec}/{codec.qp}']
            admission = bool(policy['enabled'] and all(np.isfinite(d) and d <= policy['threshold'] for d in distances))
        observations.append(original if admission is not False else
                            Observation(candidate.name, stream.coded_bytes, distances, tuple(False for _ in teachers)))
        audit.append({'name':candidate.name,'coded_bytes':stream.coded_bytes,
                      'relative_task_distance':[float(d) if np.isfinite(d) else None for d in distances],
                      'preserves_decision':list(decisions),'codec_seconds':stream.seconds,
                      'stream_sha256':hashlib.sha256(stream.data).hexdigest(),
                      'pixel_sha256':hashlib.sha256(candidate.clip.tobytes()).hexdigest(),
                      'coded_shape':list(candidate.clip.shape),'identity_stream':same or index==0,
                      'primary_pool':index<primary_count,
                      'proposal_origin':('learned_spatial' if candidate.name.startswith('learned_motion_') else
                                         'fixed_control' if index<len(controls) else 'fixed_profile_audit')})
        if admission is not None:
            audit[-1].update(policy_admitted=admission, admission_threshold=policy['threshold'],
                             policy_origin='disjoint_TRAIN_calibration')
    audit[0]['motion_support']=None if support is None else support['metadata']
    audit[0]['motion_support_sha256']=None if support is None else support_hash(support)
    audit[0]['learned_proposal_order']=[c.name for c in learned]
    audit[0]['train_static_order']=static_names
    def winner(indices):
        return indices[select([observations[i] for i in indices],slack,cfg['min_savings'])]
    primary=winner(list(range(primary_count)))
    if not components:
        return encoded[0],encoded[primary],candidates[primary].name,audit
    control_ids=list(range(len(controls)))
    learned_ids=list(range(len(controls),primary_count))
    profile_ids=list(range(primary_count,len(candidates)))
    static_ids=[i for i in profile_ids if candidates[i].name in static_names]
    # Comparator ties obey their saved TRAIN order, independent of registry order.
    static_ids.sort(key=lambda i:static_names.index(candidates[i].name))
    raw=next((i for i in learned_ids if candidates[i].name=='learned_motion_s100'),0)
    indices={'controls':winner(control_ids),'learned_guarded':winner([0,*learned_ids]),
             'learned_raw':raw,'static_adaptive':winner([*control_ids,*static_ids]),
             'profile_oracle':winner([*control_ids,*profile_ids])}
    if variant == 'c':
        indices['policy_unrestricted']=select(original_observations[:primary_count],slack,cfg['min_savings'])
    alternatives={arm:(encoded[i],candidates[i].name) for arm,i in indices.items()}
    return encoded[0],encoded[primary],candidates[primary].name,audit,alternatives


def motion_contribution(rows,components):
    """Actual marginal bytes; raw prediction/selection counts never imply gains."""
    output={}
    for qp in sorted({r['qp'] for r in rows}):
        anchor={r['id']:r for r in rows if r['arm']=='anchor' and r['qp']==qp}
        adaptive={r['id']:r for r in rows if r['arm']=='adaptive' and r['qp']==qp}
        controls={r['id']:r for r in components if r['arm']=='controls' and r['qp']==qp}
        static={r['id']:r for r in components if r['arm']=='static_adaptive' and r['qp']==qp}
        if not controls: continue
        if set(anchor)!=set(adaptive) or set(anchor)!=set(controls) or set(anchor)!=set(static):
            raise ValueError('incomplete paired motion contribution rows')
        saved=sum(controls[i]['coded_bytes']-adaptive[i]['coded_bytes'] for i in anchor)
        output[str(qp)]={'points':len(anchor),
                         'learned_selected':sum(r['candidate'].startswith('learned_motion_') for r in adaptive.values()),
                         'learned_extra_bytes_vs_controls':saved,
                         'learned_extra_anchor_rate_pp':100*saved/sum(r['coded_bytes'] for r in anchor.values()),
                         'adaptive_extra_bytes_vs_static':sum(static[i]['coded_bytes']-adaptive[i]['coded_bytes'] for i in anchor)}
    return output
