"""Separate matched 64-frame AR GOP audit. Never qualifies a primary oracle."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from ..codec import locate_ffmpeg
from .codec import V31Codec
from .measure import _video_timing
from .measure_models import build_models,model_hashes
from .measure_store import atomic_json,sha,write_packet
from .metrics import score_ar
from .protocol import canonical_hash,code_manifest_v31,validate_partitions
from .transport import Recipe

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='v31-gop-audit-1'
WINDOW_STARTS=(0,16,32)


def gop_settings(codec,keyint):
    if codec not in ('h264','h265') or type(keyint) is not int or keyint not in (8,16,32):
        raise ValueError('separate AR GOP audit supports only H264/H265 GOP8/16/32')
    return {'gop':keyint,'scenecut':False,'closed_gop':True,'bframes':0}


def analyzer_windows(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[0]!=64 or rgb.shape[-1]!=3:
        raise ValueError('fixed analyzer windows require 64 contiguous RGB samples')
    return [rgb[start:start+32:2].copy() for start in WINDOW_STARTS]


def encode_audit_packet(rgb,codec,qp,fps,keyint,preset,out):
    analyzer_windows(rgb)
    if not isinstance(fps,Fraction) or fps<=0: raise ValueError('GOP audit requires exact source FPS')
    duration=64/fps
    recipe=Recipe('ar',codec,rgb.shape[2],rgb.shape[1],64,64,duration.numerator,duration.denominator,1)
    packet=V31Codec(codec,qp,preset,fps,gop_settings(codec,keyint)).roundtrip(rgb,recipe)
    artifact=write_packet(out,packet)
    executable=Path(locate_ffmpeg())
    probe=shutil.which('ffprobe') or str(executable.with_name('ffprobe'+executable.suffix))
    value=json.loads(subprocess.check_output([probe,'-v','error','-select_streams','v:0','-show_entries',
        'frame=key_frame,pict_type','-of','json',str(Path(out)/artifact['stream_path'])],stderr=subprocess.PIPE))
    frames=value['frames']; keys=[i for i,frame in enumerate(frames) if frame['key_frame']==1]
    types=[frame['pict_type'] for frame in frames]
    if len(frames)!=64 or keys!=list(range(0,64,keyint)) or 'B' in types:
        raise ValueError('actual encoded GOP/keyframe/B-frame contract differs from request')
    return packet,{'frames':len(frames),'keyframes':keys,'picture_types':types,'probe':'actual ffprobe frames'}


def _config(cfg):
    plain={k:v for k,v in cfg.items() if k!='models'}
    if plain.get('schema')!=SCHEMA or plain.get('task')!='ar' or plain.get('frames')!=64:
        raise ValueError('separate GOP audit requires AR64; OD excluded')
    if (plain.get('preset')!='medium' or plain.get('window_starts')!=list(WINDOW_STARTS) or plain.get('window_stride')!=2 or
            plain.get('ar_size')!=128 or plain.get('scenecut') is not False or plain.get('closed_gop') is not True or plain.get('bframes')!=0):
        raise ValueError('invalid matched GOP/analyzer/latency config')
    for key,allowed in [('qps',{30,35,40,45}),('codecs',{'h264','h265'}),('gops',{8,16,32}),('sizes',{112,96})]:
        values=plain.get(key)
        if not isinstance(values,list) or not values or len(values)!=len(set(values)) or any(v not in allowed or isinstance(v,bool) for v in values):
            raise ValueError('invalid GOP coverage: '+key)
    canonical_hash(plain)
    return plain


def _sample(record):
    cap=cv2.VideoCapture(record['path'])
    if not cap.isOpened(): raise ValueError('GOP source video unreadable')
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); start=max(0,total-64)//2
    if total<64:
        cap.release(); raise ValueError('GOP audit needs at least64 real contiguous frames; no padding')
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    frames=[]
    try:
        for _ in range(64):
            ok,bgr=cap.read()
            if not ok: raise ValueError('short contiguous GOP source; no padding')
            frames.append(cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(128,128),interpolation=cv2.INTER_AREA))
    finally: cap.release()
    fps,status,error=_video_timing(record['path'],start,start+64)
    if status!='known_constant' or fps is None: raise ValueError('GOP source timing unavailable: '+str(error))
    return np.stack(frames),fps,list(range(start,start+64))


def gop_audit_plan(source_records,cfg):
    plain=_config(cfg)
    if not source_records: raise ValueError('AR GOP audit needs source records')
    records=[]; pixels={}
    for record in source_records:
        if not isinstance(record.get('id'),str) or type(record.get('label')) is not int or not 0<=record['label']<400 or record.get('split')=='test' or 'image_id' in record:
            raise ValueError('AR source records required; OD/TEST forbidden')
        rgb,fps,indices=_sample(record)
        value={**record,'source_sha256':sha(rgb.tobytes()),'fps':[fps.numerator,fps.denominator],
               'duration':[(64/fps).numerator,(64/fps).denominator],'raw_indices':indices}
        records.append(value); pixels[record['id']]=value['source_sha256']
    validate_partitions({'dev':records},pixels)
    plan={'schema':SCHEMA,'task':'ar','config_hash':canonical_hash(plain),'sources':records,
          'primary_oracle_eligible':False,'scope':'separate contiguous64 AR diagnostic; never primary16 headroom'}
    return {**plan,'plan_hash':canonical_hash(plan)}


def _predictions(rgb,evaluators):
    result={}
    for name,model in evaluators.items():
        probabilities=[]
        for window in analyzer_windows(rgb):
            observation=model.observe(window.copy())
            p=np.asarray(observation['probabilities'],np.float64)
            if p.ndim!=1 or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):
                raise ValueError('invalid GOP analyzer probabilities')
            probabilities.append(p)
        mean=np.mean(probabilities,axis=0)
        result[name]={'probabilities':mean.tolist(),'top1':int(np.argmax(mean)),
                      'window_probabilities':[p.tolist() for p in probabilities],'aggregation':'arithmetic mean probabilities once per source'}
    return result


def run_gop_audit(plan,cfg,out):
    plain=_config(cfg)
    if plan.get('schema')!=SCHEMA or plan.get('primary_oracle_eligible') is not False or plan['config_hash']!=canonical_hash(plain) or plan['plan_hash']!=canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'}):
        raise ValueError('separate GOP plan identity mismatch')
    primary=json.loads((ROOT/'configs/v31_b.json').read_text()); primary['task']='ar'
    models=cfg.get('models') or build_models('ar',primary,'cuda')
    evaluators=models['ar']['evaluators']
    if set(evaluators)!={'r2plus1d_18','r3d_18'}: raise ValueError('fixed independent AR evaluators required')
    hashes=model_hashes(models,'ar',primary); rows=[]
    out=Path(out)
    for record in plan['sources']:
        rgb,fps,indices=_sample(record)
        if sha(rgb.tobytes())!=record['source_sha256'] or [fps.numerator,fps.denominator]!=record['fps'] or indices!=record['raw_indices']:
            raise ValueError('GOP source content/timing changed after plan freeze')
        for codec in plain['codecs']:
            for qp in plain['qps']:
                for keyint in plain['gops']:
                    for size in [128]+plain['sizes']:
                        coded=rgb.copy() if size==128 else np.stack([cv2.resize(frame,(size,size),interpolation=cv2.INTER_AREA) for frame in rgb])
                        packet,probe=encode_audit_packet(coded,codec,qp,fps,keyint,plain['preset'],out/'packets')
                        rows.append({'source_id':record['id'],'label':record['label'],'method':'identity' if size==128 else 'area'+str(size),
                            'codec':codec,'qp':qp,'gop':keyint,'settings':gop_settings(codec,keyint),'preset':plain['preset'],
                            'fps':record['fps'],'duration':record['duration'],'raw_indices':indices,
                            'window_indices':[list(range(s,s+32,2)) for s in WINDOW_STARTS],
                            'packet':write_packet(out/'packets',packet),'keyframe_probe':probe,
                            'total_bytes':packet.total_bytes,'rate':8*packet.total_bytes/float(packet.recipe.duration),
                            'random_access_max_seconds':float(keyint/fps),'reorder_delay_frames':0,
                            'aggregate_predictions':_predictions(packet.decoded,evaluators)})
    comparisons=[]
    for codec in plain['codecs']:
        for qp in plain['qps']:
            for keyint in plain['gops']:
                anchor=[r for r in rows if (r['codec'],r['qp'],r['gop'],r['method'])==(codec,qp,keyint,'identity')]
                for size in plain['sizes']:
                    selected=[r for r in rows if (r['codec'],r['qp'],r['gop'],r['method'])==(codec,qp,keyint,'area'+str(size))]
                    labels={r['source_id']:r['label'] for r in selected}
                    quality=score_ar([dict(r['aggregate_predictions']['r2plus1d_18'],source_id=r['source_id']) for r in selected],labels)
                    baseline=score_ar([dict(r['aggregate_predictions']['r2plus1d_18'],source_id=r['source_id']) for r in anchor],labels)
                    comparisons.append({'codec':codec,'qp':qp,'gop':keyint,'reference':'identity','method':'area'+str(size),
                        'byte_saving_pct':100*(1-sum(r['total_bytes'] for r in selected)/sum(r['total_bytes'] for r in anchor)),
                        'top1_gap_pp':quality['top1_pct']-baseline['top1_pct'],'quality':quality,'anchor_quality':baseline,
                        'aggregation_unit':'source; three fixed windows pooled before Top1'})
    report={'schema':SCHEMA,'primary_oracle_eligible':False,'primary_metric':'source-averaged r2plus1d_18 Top1',
            'sources':len(plan['sources']),'plan_hash':plan['plan_hash'],'config_hash':canonical_hash(plain),
            'code_provenance':code_manifest_v31(ROOT),'model_hashes':hashes,'rows':rows,'comparisons':comparisons,
            'latency':{'same_gop_matched':True,'cross_gop_relaxation_labelled':True,
                       'statement':'compare preprocessing only at equal GOP; increasing GOP changes maximum random-access delay'},
            'scope':'separate GOP diagnostic; paired statistical unit is source; no primary adaptive claim or oracle eligibility'}
    report['report_hash']=canonical_hash(report); atomic_json(out/'gop_report.json',report)
    return report
