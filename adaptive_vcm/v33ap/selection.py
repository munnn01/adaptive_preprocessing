"""Encoder-only dual preservation guards, with independent forced-CNN evidence."""
import hashlib
import time
import numpy as np
import torch
from ..selection import ar_guard,detection_distance
from ..v32.roi import build_grids,retarget as apply_retarget
from ..v32vc.train import tensor
from .tasks import frame_indices,prediction,mode
from .receiver import finite_detection,timed,sync


def head_predictions(heads,rgb,task):
    key=(rgb.shape,hashlib.sha256(rgb.tobytes()).hexdigest(),task)
    cache=getattr(heads,'_prediction_cache',{})
    if key in cache: return cache[key]
    x=tensor(rgb,heads.device); p={}
    with torch.no_grad():
        if task=='ar':
            with mode(heads.action.model): p['ar']=heads.action.logits(x).softmax(-1)[0].cpu().numpy()
        with mode(heads.detector.model): boxes=heads.detector.model([x[0,:,i] for i in frame_indices(len(rgb))])
        p['od']={i:prediction(d) for i,d in zip(frame_indices(len(rgb)),boxes)}
    # Bounded cache, shared source/identity and repeated reconstructions only.
    if len(cache)>=16: cache={}
    cache[key]=p; heads._prediction_cache=cache
    return p


def guard(source,anchor,trial,task,cfg):
    audit=dict(passed=True,ar=None,od={},omitted_frames=[],reason=None)
    try:
        if task=='ar':
            d,decision=ar_guard(source['ar'],anchor['ar'],trial['ar'],cfg['ar_confidence'],require_anchor_decision=cfg['ar_require_anchor_decision'])
            audit['ar']=dict(relative_kl=d,decision_preserved=decision,slack=cfg['ar_kl_slack'])
            audit['passed'] &= bool(np.isfinite(d) and d<=cfg['ar_kl_slack'] and decision)
        if set(source['od'])!=set(anchor['od']) or set(source['od'])!=set(trial['od']): raise ValueError('frame evidence scope differs')
        for i,s in source['od'].items():
            a,t=anchor['od'][i],trial['od'][i]
            if not all(finite_detection(p) for p in (s,a,t)): raise ValueError('nonfinite detections')
            if not np.any(s['scores']>=cfg['od_score_threshold']): audit['omitted_frames'].append(i); continue
            d=detection_distance(s,t,cfg['od_score_threshold'])-detection_distance(s,a,cfg['od_score_threshold'])
            audit['od'][i]=dict(relative_distance=d,slack=cfg['od_distance_slack'])
            audit['passed'] &= bool(np.isfinite(d) and d<=cfg['od_distance_slack'])
        if not audit['od']: audit.update(passed=False,reason='no reliable source OD evidence')
    except (ValueError,KeyError,TypeError): audit.update(passed=False,reason='invalid predictions')
    return audit


def choose_stream(rgb,support,model,codec,heads,cfg,*,task,retarget=False):
    if task not in ('ar','od') or task=='od' and len(rgb)!=1 or retarget and task!='od': raise ValueError('invalid task/retarget geometry')
    sync(); before=time.perf_counter(); audits=[]; costs={}
    source,costs['source_prediction_seconds']=timed(lambda:head_predictions(heads,rgb,task))
    edited,costs['preprocessing_seconds']=timed(lambda:model.pixels(rgb,support,codec))
    packets={}; predictions={}; wire_cache={}
    def candidate(name,pixels,grids=()):
        # Avoid re-encoding a bit-identical input/geometry recipe.
        key=(pixels.shape,pixels.tobytes(),tuple(grids))
        reused=key in wire_cache
        if reused: packet=wire_cache[key]; stages={k:0. for k in codec.last_times}
        else:
            packet=codec.roundtrip(pixels,source_shape=rgb.shape,task=task,grids=grids); wire_cache[key]=packet
            stages=codec.last_times.copy()
        p,secs=timed(lambda:head_predictions(heads,packet.decoded,task))
        packets[name]=packet; predictions[name]=p
        audits.append(dict(name=name,packet_sha256=packet.packet_sha256,coded_bytes=packet.coded_bytes,
            elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,coded_shape=list(packet.coded_shape),
            distinct_encode=not reused,codec_seconds=0. if reused else packet.seconds,guard_prediction_seconds=secs,
            codec_stages=stages,encode_command=getattr(codec,'last_encode_command',None)))
    candidate('identity',rgb); candidate('cnn',edited)
    checks=guard(source,predictions['identity'],predictions['cnn'],task,cfg)
    saving=packets['cnn'].coded_bytes<=packets['identity'].coded_bytes*(1-cfg['min_savings'])
    chosen='cnn' if checks['passed'] and saving else 'identity'; audits[1].update(guards=checks,savings_passed=bool(saving))
    arms=dict(anchor=(packets['identity'],'identity'),forced=(packets['cnn'],'cnn'),guarded=(packets[chosen],chosen))
    if retarget:
        for factor in cfg['roi']['factors']:
            grids,secs=timed(lambda:build_grids(rgb.shape,support,factor,cfg['roi']['margin'])); costs[f'grid{factor}_seconds']=secs
            for prefix,px in (('raw',rgb),('cnn',edited)):
                name=f'{prefix}_retarget{factor}'
                coded,secs=timed(lambda:apply_retarget(px,grids) if grids else px); costs[name+'_resampling_seconds']=secs
                candidate(name,coded,grids or ()); arms[name]=(packets[name],name if grids else 'identity_geometry')
    sync(); costs['total_selection_seconds']=time.perf_counter()-before
    costs['codec_combined_seconds']=sum(a['codec_seconds'] for a in audits)
    costs.update({k:sum(a['codec_stages'][k] for a in audits) for k in ('encode_seconds','decode_seconds','inverse_seconds')})
    costs['codec_timing_scope']='standard encode, independent packet decode and received-grid inverse measured separately'
    return dict(arms=arms,selected_packet=packets[chosen],forced_packet=packets['cnn'],selected_name=chosen,audit=audits,costs=costs)
