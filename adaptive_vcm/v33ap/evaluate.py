"""Paired exploratory AR/OD evaluation of one shared preprocessing checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch
from ..analyzers import ActionAnalyzer,DetectionAnalyzer
from ..data import ar_plan,od_plan,partition
from ..evaluate import write_json,_od_predictions
from ..coco_metrics import coco_map
from ..metrics import curve_summary
from ..v32.evaluate import empty_output
from ..v32vc.checkpoint import state_hash
from .protocol import settings,source_id,codec_factory
from .sources import SourceRecord,read_source,label_free_support,pixel_audit,serial,assert_held_out
from .tasks import TaskHeads
from .checkpoint import load_checkpoint
from .receiver import SharedReceiver,timed
from .selection import choose_stream


def inverse_boxes(d,item,g):
    sx,sy,left,top=g; b=np.asarray(d['boxes']).copy()
    b[:,[0,2]]=np.clip((b[:,[0,2]]-left)/sx,0,item['width'])
    b[:,[1,3]]=np.clip((b[:,[1,3]]-top)/sy,0,item['height'])
    return {**d,'boxes':b}


def safe_curve(a,b):
    try:
        result=curve_summary(a['bpp'],a['quality'],b['bpp'],b['quality'])
        if min(len(set(a['quality'])),len(set(b['quality'])))<3 or max(min(a['quality']),min(b['quality']))>=min(max(a['quality']),max(b['quality'])):
            result.update(bd_rate_pct=None,pchip_bd_rate_pct=None,bd_quality_pp=None,invalid_reason='flat/insufficient quality overlap',point_passes=False,screen_passes=False)
        return result
    except ValueError: return dict(bd_rate_pct=None,pchip_bd_rate_pct=None,invalid_reason='invalid curve')


def summarize(rows,ids,task,cfg,meta=None,gt=None):
    arms=['anchor','forced','guarded']+([] if task=='ar' else [f'{p}_retarget{f}' for f in cfg['roi']['factors'] for p in ('raw','cnn')])
    expected={(i,c,q,a) for i in ids for c in cfg['codecs'] for q in cfg['qps'] for a in arms}
    keys=[(r['id'],r['codec'],r['qp'],r['arm']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('incomplete paired evaluation grid')
    results={}
    for c in cfg['codecs']:
        curves={}; ap50={}
        names=list(rows[0]['correct']) if task=='ar' else ['resnet50']
        for arm in arms:
            curves[arm]={n:dict(bpp=[],quality=[]) for n in names}; ap50[arm]=[]
            for q in cfg['qps']:
                subset=[r for r in rows if r['codec']==c and r['qp']==q and r['arm']==arm]; rate=float(np.mean([r['bpp'] for r in subset]))
                if task=='od':
                    ap,ap5=coco_map([p for r in subset for p in r['predictions']],gt,sorted(gt),meta); ap50[arm].append(ap5)
                for n in names:
                    curves[arm][n]['bpp'].append(rate); curves[arm][n]['quality'].append(float(np.mean([r['correct'][n] for r in subset])) if task=='ar' else ap)
        results[c]={a:{n:dict(curves=dict(anchor=curves['anchor'][n],adaptive=curves[a][n]),**safe_curve(curves['anchor'][n],curves[a][n])) for n in names} for a in arms if a!='anchor'}
        if task=='od': results[c]['ap50_curves']=ap50
    return dict(task=task,results=results,ar_accuracy_scope='whole-clip K400' if task=='ar' else None,
        od_map_scope='COCO mAP@[.5:.95]' if task=='od' else None,od_map=None if task=='ar' else 'see curves',
        ar_accuracy=None if task=='od' else 'see curves',scope='exploratory historical partitions',target_confirmed=False,complete_paired_grid=True,
        rate_scope='one paid packet for both heads, including stream/header/serialized inverse grids',bootstrap_draws=0)


def run(args,*,heads=None,receiver=None,model=None,plan_override=None,cfg=None):
    cfg=cfg or settings(); out=empty_output(args.out)
    if args.task not in ('ar','od') or args.split not in ('dev','test'): raise ValueError('held-out ar/od required')
    expected_count=8 if args.transfer else 16 if args.split=='dev' else 32
    if plan_override is None and (args.count!=expected_count or args.transfer and args.split!='test'): raise ValueError('locked evaluation count differs')
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(2)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    model=model or load_checkpoint(args.checkpoint,device); heads=heads or TaskHeads(device)
    if heads.hashes()!=model.metadata['task_models']: raise ValueError('task model weights differ from supervision')
    if receiver is None:
        actions={'swin3d_t':heads.action,'r3d_18':ActionAnalyzer('r3d_18',device),'r2plus1d_18':ActionAnalyzer('r2plus1d_18',device)} if args.task=='ar' else heads.action
        receiver=SharedReceiver(actions,DetectionAnalyzer('resnet50',device))
    plan,meta=plan_override if plan_override else ar_plan(args.root,args.split,args.count) if args.task=='ar' else od_plan(args.root,args.annotations,args.split,args.count)
    if args.task=='ar' and meta.get('unmapped_folders'): raise ValueError('unknown K400 labels')
    if plan_override is None:
        expected=model.metadata['held_out']['splits'][args.task][args.split][:args.count]
        if [source_id(args.task,item['id']) for item in plan]!=expected: raise ValueError('evaluation differs from frozen held-out plan')
    manifest=dict(task=args.task,split=args.split,count=len(plan),preset='fast' if args.transfer else cfg['preset'],settings=cfg,
        checkpoint_sha256=model.manifest['checkpoint_sha256'],arm=model.metadata['arm'],source_pixel_sha256={},frame_recipes={},status='running',
        device=device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,python=platform.python_version(),torch=str(torch.__version__),
        evaluators={k:state_hash(v.model.state_dict()) for k,v in receiver.actions.items()} if args.task=='ar' else {},
        od_evaluator_state_sha256=state_hash(receiver.detector.model.state_dict()),scope='exploratory historical sources',
        codec_timing_scope='encode/independent decode/inverse measured separately; selection cost includes guards/search and preprocessing')
    write_json(out/'manifest.json',manifest); rows=[]; audit_records=[]
    for index,item in enumerate(plan):
        identifier=source_id(args.task,item['id'])
        if partition(identifier[3:])!=args.split: raise ValueError('source partition mismatch')
        rgb,g,recipe=read_source(args.task,item,cfg); model.assert_source(identifier,rgb)
        if 'held_out' in model.metadata: assert_held_out(model.metadata['held_out'],identifier,rgb,args.task,args.split)
        r=SourceRecord(identifier,args.task,{k:v for k,v in item.items() if k!='label'},rgb,g,recipe)
        def prepare(): heads.source_targets(r); r.support=label_free_support(r,heads,cfg)
        _,preparation=timed(prepare); audit_records.append(r); hashes=pixel_audit(audit_records)
        manifest['source_pixel_sha256']=hashes; manifest['frame_recipes'][identifier]=recipe
        for c in cfg['codecs']:
            for q in cfg['qps']:
                codec=codec_factory(args.task,'fast' if args.transfer else cfg['preset'])(c,q)
                bundle=choose_stream(rgb,r.support,model,codec,heads,cfg,task=args.task,retarget=args.task=='od')
                for arm,(packet,name) in bundle['arms'].items():
                    received=receiver.predict(packet,tasks=('ar','od') if args.task=='ar' else ('od',))
                    row=dict(id=identifier,image_id=item.get('image_id'),codec=c,qp=q,arm=arm,candidate=name,
                        coded_bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,
                        bpp=8*packet.coded_bytes/np.prod(rgb.shape[:3]),packet_sha256=packet.packet_sha256,coded_shape=list(packet.coded_shape),
                        received=serial(received),source_preparation_seconds=preparation,costs=bundle['costs'])
                    if args.task=='ar': row['correct']={n:int(np.argmax(p)==item['label']) for n,p in received['ar_backbones'].items()}
                    else: row['predictions']=_od_predictions(inverse_boxes(received['od'][0],item,g),item['image_id'])
                    path=out/'packets'/c/f'{index:04d}-q{q}-{arm}.v33'; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(packet.data)
                    row['packet_path']=path.relative_to(out).as_posix(); rows.append(row)
                    with (out/'rows.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
                with (out/'selection_audit.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(serial(dict(id=identifier,codec=c,qp=q,audit=bundle['audit'],costs=bundle['costs'],support=r.support['metadata'])),allow_nan=False)+'\n')
        write_json(out/'manifest.json',manifest); print('EVAL '+identifier,flush=True)
    gt={i['image_id']:[a for a in meta['annotations'] if a['image_id']==i['image_id']] for i in plan} if args.task=='od' else None
    result=summarize(rows,list(manifest['source_pixel_sha256']),args.task,cfg,meta,gt)
    write_json(out/'summary.json',result); manifest['status']='complete'; write_json(out/'manifest.json',manifest); return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--task',choices=['ar','od'],required=True); p.add_argument('--split',choices=['dev','test'],required=True)
    for name in ('root','checkpoint','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--annotations',type=Path); p.add_argument('--count',type=int,required=True); p.add_argument('--transfer',action='store_true'); p.add_argument('--device'); run(p.parse_args())


if __name__=='__main__': main()
