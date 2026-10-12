"""Real-codec candidate cache, controller training, and shared packet evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from . import SCHEMA
from .protocol import ACTIONS,settings,digest,codec_factory
from .preprocessing import region_maps,filter_clip
from .model import Controller
from .learning import train_controller,calibrate,select_action
from .checkpoint import save_checkpoint,load_checkpoint

def write_json(path,value):
    from ..v33ap.sources import serial
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')

def input_pair(rgb,maps):
    def prepare(a):
        t=torch.from_numpy(a.copy()).permute(0,3,1,2).float()
        return F.interpolate(t,(48,48),mode='bilinear',align_corners=False)
    return prepare(rgb)/255,prepare(maps)

def condition(codec,qp,mask,cfg,domain=None):
    value=[float(codec=='h265'),qp/51,*map(float,mask),cfg['budget_ratio']]
    if cfg.get('domain_condition'):
        if domain not in ('ar','od','ot'): raise ValueError('source domain required')
        value += [float(domain==d) for d in ('ar','od','ot')]
    return value

def measure_candidates(rgb,maps,actions,codec,task,score):
    rows=[]; costs=[]; sizes=[]
    for action in actions:
        start=time.perf_counter(); y=filter_clip(rgb,maps,action); preprocess=time.perf_counter()-start
        p=codec.roundtrip(y,source_shape=rgb.shape,task=task)
        values=score(p.decoded); costs.append(values); sizes.append(p.coded_bytes)
        rows.append(dict(bytes=p.coded_bytes,elementary_bytes=p.elementary_bytes,metadata_bytes=p.metadata_bytes,
            sha256=p.packet_sha256,decoded_sha256=hashlib.sha256(p.decoded.tobytes()).hexdigest(),
            q8_edits=int(np.count_nonzero(y!=rgb)),preprocess_seconds=preprocess,**codec.last_times))
    damage=[[None if v is None else float(v-costs[0][k]) for k,v in enumerate(row)] for row in costs]
    return dict(rate=[s/sizes[0] for s in sizes],damage=damage,cost=costs,packets=rows)

def validate_cache(cache):
    if cache.get('schema')!=SCHEMA or cache.get('experiment') not in ('ar_od','ar_ot'):
        raise ValueError('cache schema/experiment mismatch')
    if digest({k:v for k,v in cache.items() if k!='sha256'})!=cache.get('sha256'): raise ValueError('cache content mismatch')
    for r in cache['rows']:
        if r['split'] not in ('train','cal') or r['id'] not in cache['sources'] or cache['sources'][r['id']]['split']!=r['split']:
            raise ValueError('held-out supervision or source mismatch')
        if len(r['rate'])!=len(r['damage']) or not r['mask'][0] and not r['mask'][1] and not r['mask'][2]: raise ValueError('invalid measured mask')
    return cache

@torch.no_grad()
def infer(model,pair,c):
    device=next(model.parameters()).device
    result=model(pair[0][None].to(device),pair[1][None].to(device),torch.tensor([c],device=device))
    return {k:v[0].cpu().numpy() for k,v in result.items()}

def train_models(cache,inputs,cfg,out,device,task_models):
    validate_cache(cache); fit=[r for r in cache['rows'] if r['split']=='train']; cal=[r for r in cache['rows'] if r['split']=='cal']; models=[]
    from .learning import targets
    choices=[targets(r['rate'],np.array(r['damage'],float),r['mask'],cfg['damage_tolerance'],cfg['budget_ratio']) for r in fit]
    write_json(out/'train_oracle.json',dict(scope='TRAIN constrained measured oracle; never deployed TEST GT selection',
        count=len(choices),action_counts={str(a):choices.count(a) for a in range(len(ACTIONS))},
        records=[dict(id=r['id'],codec=r.get('codec'),qp=r.get('qp'),action=a) for r,a in zip(fit,choices)]))
    for seed in cfg['seeds']:
        torch.manual_seed(seed); np.random.seed(seed)
        model=Controller(len(ACTIONS),conditions=9 if cfg.get('domain_condition') else 6); history=train_controller(model,fit,inputs,cfg,device)
        pred=[infer(model,inputs[r['id']],r['condition']) for r in cal]
        actual=np.array([r['damage'] for r in cal],float)
        if cfg.get('domain_condition'):
            from .learning import calibrate_domains
            calibration=calibrate_domains(cal,pred,cfg['cal_coverage'])
        else:
            calibration=calibrate(np.array([p['risk'] for p in pred]),actual,np.array([p['rate'] for p in pred]),
                np.array([r['rate'] for r in cal]),np.array([r['mask'] for r in cal],bool),cfg['cal_coverage'])
        meta=dict(seed=seed,settings=cfg,actions=[list(a) for a in ACTIONS],experiment=cache['experiment'],
            source_commit=cache['origin']['origin_commit'],source_archive_sha256=cache['origin']['bundle_sha256'],
            cache_sha256=cache['sha256'],sources=cache['sources'],task_models=task_models,calibration=calibration,
            checkpoint_role='LAST',training_history=history,codec_backpropagation=False)
        path=out/f'controller_{seed}.pth'; save_checkpoint(path,model,meta)
        models.append(load_checkpoint(path,device)); write_json(out/f'training_{seed}.json',meta)
    return models

def run(args):
    from .sources import cohorts
    from .tasks import Heads
    from .evaluation import evaluate
    cfg=settings(getattr(args,'config',None)); out=Path(args.out)
    if cfg.get('revision')=='arot-r1' and args.experiment!='ar_ot': raise ValueError('AR+OT repair recipe only')
    if out.exists() and any(out.iterdir()): raise ValueError('immutable empty output required')
    out.mkdir(parents=True,exist_ok=True); device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    origin=json.loads(Path(args.origin).read_text()); write_json(out/'origin.json',origin)
    groups,meta=cohorts(args.ar_root,args.od_root,args.annotations,cfg,args.experiment)
    if args.experiment=='ar_ot':
        from .tracking import tracking_sources
        ot_groups,ot_plan=tracking_sources(Path(args.mot_root),cfg)
        for split in groups: groups[split]+=ot_groups[split]
        meta['ot']=ot_plan
    heads=Heads(device,args.experiment,cfg); task_models=heads.hashes()
    sources={}; hashes=set()
    for split,records in groups.items():
        for r in records:
            sha=hashlib.sha256(r.rgb.tobytes()).hexdigest()
            if r.id in sources or sha in hashes: raise ValueError('duplicate pixels/ID across cohorts')
            sources[r.id]=dict(split=split,pixel_sha256=sha,frame_recipe=r.frame_recipe); hashes.add(sha)
    write_json(out/'cohorts.json',sources)
    cache=dict(schema=SCHEMA,experiment=args.experiment,origin=origin,settings=cfg,actions=[list(a) for a in ACTIONS],
        sources={k:v for k,v in sources.items() if v['split'] in ('train','cal')},task_models=task_models,rows=[])
    inputs={}
    for split in ('train','cal'):
        for index,r in enumerate(groups[split]):
            heads.source(r); maps=region_maps(r.rgb,r.source_predictions['od']); inputs[r.id]=input_pair(r.rgb,maps)
            for codec in cfg['codecs']:
                for qp in cfg['qps']:
                    c=codec_factory(r.task,codec,qp,cfg)
                    table=measure_candidates(r.rgb,maps,ACTIONS,c,r.task,lambda rgb:heads.costs(rgb,r))
                    cache['rows'].append(dict(id=r.id,split=split,domain=r.task,codec=codec,qp=qp,mask=r.mask,
                        provenance=r.provenance,condition=condition(codec,qp,r.mask,cfg,domain=r.task),**table))
            write_json(out/'measurements.partial.json',cache)
            print(f'V34 {args.experiment} cache {split} {index+1}/{len(groups[split])}',flush=True)
    cache['sha256']=digest(cache); write_json(out/'measurements.json',cache)
    # Inputs are a separate immutable tensor archive; only native tensors.
    torch.save(inputs,out/'controller_inputs.pth')
    models=train_models(cache,inputs,cfg,out,device,task_models)
    evaluate(groups,heads,models,cfg,meta,out,Path(args.trackeval) if args.trackeval else None)
    write_json(out/'complete.json',dict(status='complete',experiment=args.experiment,cache_sha256=cache['sha256'],
        checkpoint_sha256=[m.checkpoint_sha256 for m in models],scope='exploratory historical domain pilot; MOT20 has no AR ground truth'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',choices=['ar_od','ar_ot'],required=True)
    p.add_argument('--config')
    for key in ('ar_root','od_root','annotations','mot_root','trackeval','origin','out'): p.add_argument('--'+key.replace('_','-'),required=key in ('ar_root','origin','out'))
    run(p.parse_args())

if __name__=='__main__': main()
