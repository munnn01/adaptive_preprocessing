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
CONDITIONAL_SCHEMA = 'adaptive-vcm-conditional-v9'
PROPOSALS = (('learned_motion_s050', .5), ('learned_motion_s100', 1.), ('learned_motion_s150', 1.5))


def validate_motion_config(cfg):
    if 'v29_variant' in cfg and cfg['v29_variant'] not in ('a', 'b', 'c'):
        raise ValueError('invalid V29 variant')
    if 'v30_variant' in cfg:
        from .train_motion import _validate_config
        _validate_config(cfg)
        if cfg['v30_variant'] not in ('a','b','c') or 'v29_variant' in cfg:
            raise ValueError('invalid V30 variant')
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
    conditional = state.get('schema') == CONDITIONAL_SCHEMA
    if state.get('schema') not in (SCHEMA, SEMANTIC_SCHEMA,CONDITIONAL_SCHEMA) or task not in ('ar','od') or state.get('task') != task:
        raise ValueError('motion checkpoint task/schema mismatch')
    cfg = state.get('training_config', {})
    validate_motion_config(cfg)
    variant = cfg.get('v30_variant') if conditional else cfg.get('v29_variant')
    if (semantic != (cfg.get('v29_variant') is not None)
            or conditional != (cfg.get('v30_variant') is not None)
            or (semantic or conditional) and state.get('variant') != variant):
        raise ValueError('semantic checkpoint/configuration variant mismatch')
    if conditional:
        from .train_motion import ROOT,_code_manifest,_json_bytes
        code=state.get('code')
        if (code!=_code_manifest() or
                state.get('code_sha256')!=hashlib.sha256(_json_bytes(code)).hexdigest()):
            raise ValueError('conditional checkpoint code provenance mismatch')
        config_bytes=(ROOT/f'configs/v30_{variant}_screen.json').read_bytes().replace(b'\r\n',b'\n')
        if (state.get('config_sha256')!=hashlib.sha256(config_bytes).hexdigest()
                or cfg!=json.loads(config_bytes)):
            raise ValueError('conditional checkpoint configuration provenance mismatch')
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
    if semantic or conditional:
        fit_ids = state.get('fit_ids', [])
        calibration_ids = state.get('calibration_ids', [])
        if (not isinstance(fit_ids, list) or not fit_ids or not isinstance(calibration_ids, list)
                or any(type(i) is not str for i in fit_ids + calibration_ids)
                or len(set(fit_ids)) != len(fit_ids) or len(set(calibration_ids)) != len(calibration_ids)
                or set(fit_ids) & set(calibration_ids) or set(fit_ids + calibration_ids) != set(ids)):
            raise ValueError('invalid disjoint semantic TRAIN partition')
        if variant == 'c' and semantic:
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
        if conditional and state.get('fit_ids_sha256') != fingerprint(fit_ids):
            raise ValueError('conditional TRAIN fit identity hash mismatch')
    source_hashes=state.get('train_source_sha256',{})
    if (not isinstance(source_hashes,dict) or set(source_hashes)!=set(ids)
            or any(not isinstance(v,str) or not re.fullmatch(r'[0-9a-f]{64}',v) for v in source_hashes.values())):
        raise ValueError('missing motion TRAIN source-pixel hashes')
    if (type(state.get('measurements')) is not int or state['measurements'] != count*10
            or type(state.get('steps')) is not int or state['steps'] != epochs*len(fit_ids)*10
            or not re.fullmatch(r'[0-9a-f]{64}',str(state.get('measurements_sha256','')))):
        raise ValueError('incomplete motion grid/optimization or missing measurement hash')
    registry = None
    names = list(PROFILE_NAMES)
    if conditional:
        from .conditional_learned import profile_registry,ConditionalPreprocessor
        registry = profile_registry(variant)
        names = [p['name'] for p in registry]
        if state.get('profile_registry') != registry:
            raise ValueError('conditional profile metadata mismatch')
    if state.get('profile_names') != names:
        raise ValueError('motion profile registry mismatch')
    orders=state.get('static_orders',{})
    groups={f'{c}/{q}' for c in ('h264','h265') for q in cfg['qps']}
    if not isinstance(orders,dict) or set(orders)!=groups:
        raise ValueError('incomplete motion static groups')
    for names in orders.values():
        if (not isinstance(names,list) or len(names)!=3 or any(not isinstance(n,str) for n in names)
                or len(set(names))!=3 or set(names)-set(state['profile_names'])):
            raise ValueError('invalid motion static profile order')
    baseline_orders = None
    if conditional:
        baseline_orders = state.get('baseline_static_orders',{})
        if not isinstance(baseline_orders,dict) or set(baseline_orders)!=groups:
            raise ValueError('incomplete conditional baseline static groups')
        for baseline_order in baseline_orders.values():
            if (not isinstance(baseline_order,list) or len(baseline_order)!=3
                    or any(type(n) is not str for n in baseline_order)
                    or len(set(baseline_order))!=3 or set(baseline_order)-set(PROFILE_NAMES)):
                raise ValueError('invalid conditional baseline static profile order')
    if type(state.get('width')) is not int or state['width'] < 4:
        raise ValueError('invalid motion model width')
    weights=state.get('model',{})
    if (not isinstance(weights,dict) or not weights
            or any(not isinstance(v,torch.Tensor) or not v.is_floating_point()
                   or not torch.isfinite(v).all() for v in weights.values())):
        raise ValueError('nonfinite or malformed motion model state')
    model=(ConditionalPreprocessor(state['width'],task,variant=variant) if conditional else
           MotionAwarePreprocessor(state['width'],task,variant=variant) if semantic else
           MotionAwarePreprocessor(state['width'],task))
    try:
        model.load_state_dict(weights,strict=True)
    except (RuntimeError,TypeError) as error:
        raise ValueError('incompatible motion model state') from error
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('nonfinite motion model after dtype conversion')
    model.static_orders={group:list(names) for group,names in orders.items()}
    model.training_measurements_sha256=state['measurements_sha256']
    model.train_source_sha256=dict(source_hashes)
    if conditional:
        model.profile_registry=registry
        model.baseline_static_orders={g:list(n) for g,n in baseline_orders.items()}
        model.eval()
    if variant == 'c' and semantic:
        model.admission_policy={group:dict(entry) for group,entry in state['admission_policy'].items()}
    return model


def support_hash(support):
    digest=hashlib.sha256()
    for key in ('protection','motion','cuts'):
        digest.update(np.ascontiguousarray(support[key]).tobytes())
    digest.update(json.dumps(support['metadata'],sort_keys=True,allow_nan=False).encode())
    return digest.hexdigest()


def neural_candidates(clip,support,codec,model,*,diagnostics=None):
    conditional=getattr(model,'schema',None)==CONDITIONAL_SCHEMA
    if conditional and model.training:
        raise ValueError('conditional inference requires eval mode for the hard gate')
    device=next(model.parameters()).device
    source=torch.from_numpy(clip.copy()).to(device).float().permute(3,0,1,2)[None]/255
    protection=torch.from_numpy(support['protection'].copy()).to(source)[None,None]
    motion=torch.from_numpy(support['motion'].copy()).to(source)[None,None]
    cuts=torch.from_numpy(support['cuts'].copy()).to(device)[None]
    candidates=[]
    with torch.no_grad():
        for name,scale in PROPOSALS:
            values=model(source,source.new_tensor([codec.qp]),source.new_tensor([int(codec.codec=='h265')]),
                protection,motion=motion,cuts=cuts,strength_scale=scale,
                **({'return_aux':True} if diagnostics is not None else {}))
            if diagnostics is not None:
                output,aux=values
                probability=float(aux['gate_probability'].flatten()[0])
                diagnostics[name]=dict(gate_probability=probability,admitted=probability>=.5,
                    strength=float(aux['strength'].flatten()[0]),strength_scale=scale,
                    expert_weights=aux['expert_weights'].flatten().tolist())
            else:
                output=values
            if not torch.isfinite(output).all():
                raise ValueError(f'nonfinite neural proposal: {name}')
            pixels=output[0].permute(1,2,3,0).mul(255).round().clamp(0,255).byte().cpu().numpy()
            if diagnostics is not None:
                diagnostics[name]['output_edit_fraction']=float(np.any(pixels!=clip,axis=-1).mean())
            candidates.append(Candidate(name,pixels))
    return candidates


def choose_motion_stream(clip,protection,task,codec,cfg,teachers,source_predictions,model,
                         *, learned_mask=None,components=False):
    validate_motion_config(cfg)
    if getattr(model,'task',None)!=task:
        raise ValueError('motion model/selection task mismatch')
    conditional = getattr(model,'schema',None)==CONDITIONAL_SCHEMA
    variant = cfg.get('v30_variant') if conditional else cfg.get('v29_variant')
    if variant != getattr(model, 'variant', None):
        raise ValueError('motion model/selection variant mismatch')
    known=task=='ar' or np.any(source_predictions[0]['scores']>=cfg['od_score_threshold'])
    controls=make_candidates(clip,protection,task,codec.qp,cfg[f'{task}_candidates'])
    support=None
    conditional_diagnostics={} if conditional else None
    if known:
        support=(learned_mask if isinstance(learned_mask,dict) else build_motion_support(
            clip,protection if learned_mask is None else learned_mask,task))
        learned=neural_candidates(clip,support,codec,model,
            **({'diagnostics':conditional_diagnostics} if conditional else {}))
    else:
        controls=controls[:1]; learned=[]
    candidates=controls+learned
    primary_count=len(candidates)
    static_names=model.static_orders[f'{codec.codec}/{codec.qp}'] if known else []
    if components and known:
        if conditional:
            from .conditional_learned import profile_candidates as conditional_profiles
            candidates += conditional_profiles(clip,support,task,codec.qp,variant=variant)
            if variant=='c':
                candidates += [Candidate('baseline_'+c.name,c.clip) for c in
                    profile_candidates(clip,support,task,codec.qp,variant='a')]
        else:
            candidates+=(profile_candidates(clip,support,task,codec.qp,variant=variant) if variant is not None
                         else profile_candidates(clip,support,task,codec.qp))
    # All proposals are fixed before a nonidentity outcome is observed.
    encoded,predictions,observations,original_observations,audit=[],[],[],[],[]
    slack=cfg['ar_kl_slack' if task=='ar' else 'od_distance_slack']
    pixel_cache,prediction_cache = {},{}
    for index,candidate in enumerate(candidates):
        pixel_hash=hashlib.sha256(candidate.clip.tobytes()).hexdigest()
        pixel_key=(candidate.clip.shape,pixel_hash)
        new_encode = not conditional or pixel_key not in pixel_cache
        stream=codec.roundtrip(candidate.clip) if new_encode else pixel_cache[pixel_key]
        if conditional:
            pixel_cache[pixel_key]=stream
        stream_hash=hashlib.sha256(stream.data).hexdigest()
        same=bool(encoded and stream.data==encoded[0].data)
        new_prediction=not same and (not conditional or stream_hash not in prediction_cache)
        trial=(predictions[0] if same else prediction_cache[stream_hash] if not new_prediction else
               [t.probabilities(stream.decoded) if task=='ar' else t.predict(stream.decoded) for t in teachers])
        if conditional:
            prediction_cache[stream_hash]=trial
        distances,decisions=((tuple(0. for _ in teachers),tuple(True for _ in teachers))
                             if index==0 or same else relative_guard(task,source_predictions,predictions[0],trial,cfg))
        encoded.append(stream); predictions.append(trial)
        original = Observation(candidate.name,stream.coded_bytes,distances,decisions)
        original_observations.append(original)
        admission = None
        if variant == 'c' and not conditional and candidate.name.startswith('learned_motion_'):
            policy = model.admission_policy[f'{codec.codec}/{codec.qp}']
            admission = bool(policy['enabled'] and all(np.isfinite(d) and d <= policy['threshold'] for d in distances))
        observations.append(original if admission is not False else
                            Observation(candidate.name, stream.coded_bytes, distances, tuple(False for _ in teachers)))
        audit.append({'name':candidate.name,'coded_bytes':stream.coded_bytes,
                      'relative_task_distance':[float(d) if np.isfinite(d) else None for d in distances],
                      'preserves_decision':list(decisions),'codec_seconds':stream.seconds if new_encode else 0.,
                      'stream_sha256':stream_hash,
                      'pixel_sha256':pixel_hash,
                      'coded_shape':list(candidate.clip.shape),'identity_stream':same or index==0,
                      'primary_pool':index<primary_count,
                      'proposal_origin':('learned_spatial' if candidate.name.startswith('learned_motion_') else
                                         'fixed_control' if index<len(controls) else 'fixed_profile_audit')})
        if conditional:
            audit[-1].update(distinct_codec_encode=new_encode,distinct_teacher_evaluation=new_prediction,
                reference_family='baseline_v29_a' if candidate.name.startswith('baseline_') else
                    f'conditional_v30_{variant}' if index>=primary_count else None)
            if candidate.name in conditional_diagnostics:
                audit[-1]['conditional_action']=conditional_diagnostics[candidate.name]
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
    if conditional:
        current_ids=[i for i in profile_ids if not candidates[i].name.startswith('baseline_')]
        prefix='baseline_' if variant=='c' else ''
        baseline_names=[prefix+n for n in model.baseline_static_orders[f'{codec.codec}/{codec.qp}']] if known else []
        baseline_ids=[i for i in profile_ids if candidates[i].name in [prefix+n for n in PROFILE_NAMES]]
        baseline_static_ids=[i for i in baseline_ids if candidates[i].name in baseline_names]
        baseline_static_ids.sort(key=lambda i:baseline_names.index(candidates[i].name))
        indices.update(profile_oracle=winner([*control_ids,*current_ids]),
            static_baseline=winner([*control_ids,*baseline_static_ids]),
            profile_baseline_oracle=winner([*control_ids,*baseline_ids]))
    if variant == 'c' and not conditional:
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


def conditional_selection_diagnostics(records,cfg,task):
    """Exclusive measured failure stages and frozen-bank gate/action diagnostics.

    A reference-bank target is descriptive, not independent task truth; a neural
    mixture can beat that bank. This function never changes an admission rule.
    """
    from .conditional_learned import canonical_target,profile_registry
    from .train_motion import _eligible
    teacher_count=2 if task=='ar' else 1
    slack=cfg['ar_kl_slack' if task=='ar' else 'od_distance_slack']
    names={p['name'] for p in profile_registry(cfg['v30_variant'])}
    output={}
    def observation(row):
        return Observation(row['name'],row['coded_bytes'],
            tuple(math.inf if d is None else d for d in row['relative_task_distance']),
            tuple(row['preserves_decision']))
    def eligible(row,anchor):
        return _eligible(dict(coded_bytes=row['coded_bytes'],
            distances=row['relative_task_distance'],decisions=row['preserves_decision']),
            anchor,slack,cfg['min_savings'],teacher_count)
    for codec,qp in sorted({(r['codec'],r['qp']) for r in records}):
        points=[r for r in records if (r['codec'],r['qp'])==(codec,qp)]
        failures=dict(byte_threshold=0,teacher_guard=0,control_dominance=0,learned_extra_win=0,no_learned=0)
        confusion=dict(tp=0,tn=0,fp=0,fn=0)
        identity_edits,strength_errors,expert_errors=[],[],[]
        encodes=calls=0
        seconds=0.
        for point in points:
            candidates=point['candidates']
            anchor=candidates[0]['coded_bytes']
            controls=[r for r in candidates if r['primary_pool'] and not r['name'].startswith('learned_motion_')]
            observations=[observation(r) for r in controls]
            control=controls[select(observations,slack,cfg['min_savings'])]['coded_bytes']
            learned=[r for r in candidates if r['primary_pool'] and r['name'].startswith('learned_motion_')]
            byte_ok=[r for r in learned if r['coded_bytes']<=(1-cfg['min_savings'])*anchor]
            guarded=[r for r in byte_ok if eligible(r,anchor)]
            category=('no_learned' if not learned else 'byte_threshold' if not byte_ok else
                      'teacher_guard' if not guarded else 'control_dominance' if
                      min(r['coded_bytes'] for r in guarded)>=control else 'learned_extra_win')
            failures[category]+=1
            encodes+=sum(bool(r.get('distinct_codec_encode')) for r in candidates)
            calls+=teacher_count*sum(bool(r.get('distinct_teacher_evaluation')) for r in candidates)
            seconds+=sum(r['codec_seconds'] for r in candidates)
            raw=next((r for r in learned if r['name']=='learned_motion_s100'),None)
            if raw is None:
                continue
            action=raw['conditional_action']
            bank=[r for r in candidates if not r['primary_pool'] and r['name'] in names]
            if {r['name'] for r in bank}!=names or len(bank)!=len(names):
                continue  # No full audit bank means no reference target was measured.
            references=[r for r in bank if eligible(r,anchor) and r['coded_bytes']<control]
            target=min(references,key=lambda r:r['coded_bytes']) if references else None
            positive=target is not None
            confusion['tp' if positive and action['admitted'] else 'fn' if positive else
                      'fp' if action['admitted'] else 'tn']+=1
            if positive:
                desired=canonical_target(target['name'],cfg['v30_variant'])
                strength_errors.append((action['strength']-desired['strength'])**2)
                weights=np.asarray(desired['expert_weights'])
                predicted=np.asarray(action['expert_weights'])
                expert_errors.append(float(np.sum(weights*(np.log(np.maximum(weights,1e-8))-
                                                          np.log(np.maximum(predicted,1e-8))))))
            else:
                identity_edits.append(action['output_edit_fraction'])
        gate_points=sum(confusion.values())
        output[f'{codec}/{qp}']=dict(points=len(points),failure_categories=failures,
            failure_precedence='no learned; byte threshold; teacher guard; control dominance; extra win',
            gate_reference_points=gate_points,gate_confusion_vs_reference=confusion,
            gate_accuracy_vs_reference=(confusion['tp']+confusion['tn'])/gate_points if gate_points else None,
            reference_scope='fresh current-profile bank with original guards and extra control bytes; exploratory diagnostic, not task truth',
            identity_reference_points=len(identity_edits),
            identity_reference_output_edit_fraction=float(np.mean(identity_edits)) if identity_edits else None,
            positive_reference_points=len(strength_errors),
            positive_reference_strength_mse=float(np.mean(strength_errors)) if strength_errors else None,
            positive_reference_expert_kl=float(np.mean(expert_errors)) if expert_errors else None,
            distinct_codec_encodes=encodes,decoded_teacher_model_calls=calls,
            actual_probe_codec_seconds=seconds)
    return output
