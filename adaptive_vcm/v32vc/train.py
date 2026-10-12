"""TRAIN-only scalar codec calibration and two supervised residual CNN fits."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.optimize import nnls
import torch
import torch.nn.functional as F
from torch import nn

from ..analyzers import ActionAnalyzer, DetectionAnalyzer
from ..data import ar_plan, od_plan, partition, read_video, read_image
from ..evaluate import write_json
from ..v32.codec import V32Codec
from ..v32.evaluate import empty_output, prepare_source, SourceAudit
from ..v32.protocol import ROOT, load_config, canonical_hash, provenance
from ..v32.roi import build_grids, retarget
from ..v32.transport import pack_packet
from .checkpoint import make_checkpoint
from .geometry import transform
from .model import ResidualPreprocessor
from .virtual import VirtualCodec


def settings(path=None):
    cfg = json.loads((Path(path) if path else ROOT/'configs/v32vc.json').read_text())
    expected = dict(schema='adaptive-vcm-v32vc-1',seed=303221,width=12,train_count=64,epochs=4,lr=1e-4,
                    rate_weight=.01,distortion_weight=10.,step_scales=[.5,1.,2.],checkpoint_role='LAST')
    if cfg!=expected: raise ValueError('V32-VC locked training settings differ')
    return cfg


def split_train(ids,task,seed):
    if task not in ('ar','od') or len(ids)<4 or len(ids)!=len(set(ids)):
        raise ValueError('at least four unique TRAIN sources required')
    if any(partition('coco2017/'+i if task=='od' else i)!='train' for i in ids):
        raise ValueError('held-out source enters fitting')
    ranked = sorted(ids,key=lambda i:hashlib.sha256(f'{seed}/{i}'.encode()).hexdigest())
    count = max(1,len(ids)//4)
    return ranked[count:],ranked[:count]


def od_targets(meta,item,geometry,device):
    sx,sy,left,top = geometry
    boxes,labels = [],[]
    for ann in meta['annotations']:
        if ann['image_id']!=item['image_id'] or ann.get('iscrowd',0): continue
        x,y,w,h = ann['bbox']
        if w<=0 or h<=0: continue
        boxes.append([x*sx+left,y*sy+top,(x+w)*sx+left,(y+h)*sy+top]); labels.append(ann['category_id'])
    return dict(boxes=torch.tensor(boxes,device=device,dtype=torch.float32).reshape(-1,4),
                labels=torch.tensor(labels,device=device,dtype=torch.int64))


def task_loss(task,teachers,reconstruction,target):
    if task=='ar':
        label = torch.as_tensor(target,device=reconstruction.device,dtype=torch.long).reshape(-1)
        return torch.stack([F.cross_entropy(t.logits(reconstruction),label) for t in teachers]).mean()
    if task!='od' or len(teachers)!=1: raise ValueError('invalid task teachers')
    model = teachers[0].model
    states = [(module,module.training) for module in model.modules()]
    try:
        # Detection loss requires train-mode RPN/ROI heads; frozen BN/dropout stay fixed.
        model.train()
        for module in model.modules():
            if isinstance(module,(nn.modules.batchnorm._BatchNorm,nn.modules.dropout._DropoutNd)):
                module.eval()
        terms = model([reconstruction[0,:,0]],[target])
        if not isinstance(terms,dict) or not terms: raise ValueError('OD training losses unavailable')
        return sum(terms.values())
    finally:
        for module,state in states: module.training = state


def fit_rate(estimated,measured):
    x,y = np.asarray(estimated,float),np.asarray(measured,float)
    if x.shape!=y.shape or x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or (x<0).any() or (y<=0).any():
        raise ValueError('finite positive real rate calibration required')
    fitted,_ = nnls(np.column_stack((x,np.ones_like(x))),y)
    return max(1e-6,float(fitted[0])),float(fitted[1])


def tensor(rgb,device):
    return torch.from_numpy(rgb.copy()).to(device).float().permute(3,0,1,2)[None]/255


def support_tensors(support,reference):
    p,m = [torch.from_numpy(support[k].copy()).to(reference)[None,None] for k in ('protection','motion')]
    c = torch.from_numpy(support['cuts'].copy()).to(reference.device)[None]
    return p,m,c


def calibrate(records,cal_ids,cfg,vcfg,task,device,out,*,codec_factory=V32Codec):
    rows = []
    for index,identifier in enumerate(cal_ids):
        record = records[identifier]; rgb,support = record['rgb'],record['support']
        low = np.stack([cv2.GaussianBlur(frame,(0,0),1.5) for frame in rgb]).astype(np.float32)
        alpha = ((1-support['protection'])*(1-.75*support['motion']))[...,None]
        variants = [('identity',rgb,()),('bounded_blur',np.clip(np.rint(rgb+.1*alpha*(low-rgb)),0,255).astype(np.uint8),())]
        grid = build_grids(rgb.shape,support,2,.05)
        if grid: variants.append(('retarget2',retarget(rgb,grid),grid))
        for codec in cfg['codecs']:
            for qp in cfg['qps']:
                actual = codec_factory(codec,qp,cfg['preset'],Fraction(cfg['fps']))
                for name,pixels,grids in variants:
                    packet = actual.roundtrip(pixels,source_shape=rgb.shape,task=task,grids=grids)
                    coded = tensor(pixels,device); cuts = torch.from_numpy(support['cuts'].copy()).to(device)[None]
                    observations = []
                    for scale in vcfg['step_scales']:
                        step = .015*2**((qp-30)/6)*scale
                        virtual = VirtualCodec({f'{codec}/{qp}':dict(step=step,rate_scale=1.,rate_offset=0.)}).to(device).eval()
                        with torch.no_grad():
                            reconstructed,bpp = virtual(coded,qp,int(codec=='h265'),cuts)
                            if grids: reconstructed = transform(reconstructed,grids,inverse=True)
                            mse = float(F.mse_loss(reconstructed,tensor(packet.decoded,device)))
                        observations.append(dict(step=step,bpp=float(bpp),mse=mse))
                    rows.append(dict(source_id=identifier,codec=codec,qp=qp,variant=name,
                        elementary_bpp=8*packet.elementary_bytes/np.prod(pixels.shape[:3]),
                        total_bytes=packet.coded_bytes,metadata_bytes=packet.metadata_bytes,observations=observations))
        print(f'CAL {index+1}/{len(cal_ids)} complete',flush=True)
    calibration = {}
    for codec in cfg['codecs']:
        for qp in cfg['qps']:
            group = [r for r in rows if r['codec']==codec and r['qp']==qp]
            chosen = min(range(3),key=lambda i:sum(r['observations'][i]['mse'] for r in group))
            predicted = np.array([r['observations'][chosen]['bpp'] for r in group])
            measured = np.array([r['elementary_bpp'] for r in group])
            scale,offset = fit_rate(predicted,measured)
            deltas = []
            for identifier in cal_ids:
                pair = [r for r in group if r['source_id']==identifier and r['variant'] in ('identity','bounded_blur')]
                if len(pair)==2:
                    a,b = pair
                    deltas.append(bool(np.sign(b['observations'][chosen]['bpp']-a['observations'][chosen]['bpp'])==np.sign(b['elementary_bpp']-a['elementary_bpp'])))
            calibration[f'{codec}/{qp}'] = dict(step=group[0]['observations'][chosen]['step'],rate_scale=scale,rate_offset=offset,
                n=len(group),pixel_mse=float(np.mean([r['observations'][chosen]['mse'] for r in group])),
                rate_relative_mae=float(np.mean(np.abs(scale*predicted+offset-measured)/measured)),
                edit_rate_sign_agreement=float(np.mean(deltas)))
    write_json(out/'calibration_rows.json',rows); write_json(out/'calibration.json',calibration)
    return calibration


def run(args):
    out = empty_output(args.out); cfg,vcfg = load_config(),settings(args.config)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2); torch.manual_seed(vcfg['seed']); np.random.seed(vcfg['seed'])
    if args.task=='ar':
        plan,meta = ar_plan(args.root,'train',vcfg['train_count'])
        teachers = [ActionAnalyzer(n,device) for n in cfg['ar_teachers']]
    else:
        if not args.annotations: raise ValueError('COCO annotations required')
        plan,meta = od_plan(args.root,args.annotations,'train',vcfg['train_count'])
        teachers = [DetectionAnalyzer(cfg['od_teacher'],device)]
    fit_ids,cal_ids = split_train([r['id'] for r in plan],args.task,vcfg['seed'])
    records,audit = {},SourceAudit()
    for index,item in enumerate(plan):
        if args.task=='ar': rgb = read_video(item['path'],cfg['frames'],cfg['ar_size'],cfg['temporal_stride']); geometry = None
        else: rgb,geometry = read_image(item,cfg['od_size'])
        audit.add(item['id'],rgb)
        _,support,_ = prepare_source(rgb,args.task,cfg,teachers)
        records[item['id']] = dict(rgb=rgb,support=support,item=item,geometry=geometry)
        print(f'TRAIN source preparation {index+1}/{len(plan)}',flush=True)
    manifest = dict(task=args.task,settings=vcfg,fit_ids=fit_ids,cal_ids=cal_ids,source_pixel_sha256=audit.hashes,
                    code=provenance(cfg),protocol=cfg,status='calibration',device=device)
    supervised = {}
    for identifier in fit_ids:
        record = records[identifier]
        if args.task=='ar': supervised[identifier] = record['item']['label']
        else:
            target = od_targets(meta,record['item'],record['geometry'],'cpu')
            supervised[identifier] = {k:v.tolist() for k,v in target.items()}
    manifest.update(supervision_sha256=canonical_hash(supervised),
                    annotations_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest() if args.task=='od' else None)
    write_json(out/'fit_supervision.json',supervised)
    write_json(out/'manifest.json',manifest)
    calibration = calibrate(records,cal_ids,cfg,vcfg,args.task,device,out)
    groups = [(c,q) for c in cfg['codecs'] for q in cfg['qps']]
    for arm in ('plain','joint'):
        torch.manual_seed(vcfg['seed'])
        rng = np.random.default_rng(vcfg['seed'])
        model = ResidualPreprocessor(args.task,vcfg['width']).to(device).train()
        virtual = VirtualCodec(calibration).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(),lr=vcfg['lr'])
        history = []; step = 0
        for epoch in range(vcfg['epochs']):
            totals = dict(loss=0.,task_loss=0.,distortion=0.,estimated_bpp=0.,gradient_norm=0.)
            start = time.perf_counter()
            for index in rng.permutation(len(fit_ids)):
                identifier = fit_ids[int(index)]; record = records[identifier]
                x = tensor(record['rgb'],device); p,m,cuts = support_tensors(record['support'],x)
                target = record['item']['label'] if args.task=='ar' else od_targets(meta,record['item'],record['geometry'],device)
                grids = {factor:build_grids(record['rgb'].shape,record['support'],factor,.05) for factor in (2,4)}
                for group_index in rng.permutation(len(groups)):
                    codec,qp = groups[int(group_index)]
                    factor = (None,2,4)[step%3] if arm=='joint' else None
                    grid = grids.get(factor)
                    y = model(x,qp,int(codec=='h265'),p,m,cuts)
                    coded = transform(y,grid) if grid else y
                    decoded,estimated = virtual(coded,qp,int(codec=='h265'),cuts)
                    if grid: decoded = transform(decoded,grid,inverse=True)
                    source_shape = record['rgb'].shape
                    coded_shape = (source_shape[0],coded.shape[-2],coded.shape[-1],3)
                    metadata = len(pack_packet(b'x',task=args.task,codec=codec,source_shape=source_shape,
                        coded_shape=coded_shape,fps=Fraction(cfg['fps']),grids=grid or ()))-1
                    estimated = estimated*(coded.shape[-2]*coded.shape[-1])/(x.shape[-2]*x.shape[-1])+8*metadata/np.prod(source_shape[:3])
                    task = task_loss(args.task,teachers,decoded,target)
                    distortion = F.mse_loss(decoded,x)
                    loss = task+vcfg['rate_weight']*estimated+vcfg['distortion_weight']*distortion
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite supervised objective')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
                    row = dict(step=step+1,epoch=epoch+1,arm=arm,source_id=identifier,codec=codec,qp=qp,
                               factor=factor if grid else None,loss=float(loss.detach()),task_loss=float(task.detach()),
                               distortion=float(distortion.detach()),estimated_bpp=float(estimated.detach()),gradient_norm=float(norm))
                    with (out/f'{arm}_training.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(row,allow_nan=False)+'\n')
                    for k in totals: totals[k] += row[k]
                    step += 1
                    if step%50==0: print(json.dumps(row),flush=True)
            count = len(fit_ids)*len(groups)
            history.append(dict(epoch=epoch+1,steps=count,seconds=time.perf_counter()-start,**{k:v/count for k,v in totals.items()}))
            print(json.dumps(dict(arm=arm,**history[-1])),flush=True)
        metadata = dict(task=args.task,arm=arm,width=vcfg['width'],seed=vcfg['seed'],epochs=vcfg['epochs'],settings=vcfg,
                        fit_ids=fit_ids,cal_ids=cal_ids,source_pixel_sha256=audit.hashes,calibration=calibration,history=history,
                        supervision_sha256=manifest['supervision_sha256'],annotations_sha256=manifest['annotations_sha256'])
        torch.save(make_checkpoint(model,metadata),out/f'{arm}_last.pth')
    manifest.update(status='complete',calibration=calibration)
    write_json(out/'manifest.json',manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=['ar','od']); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--annotations',type=Path); parser.add_argument('--config',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda']); parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__': main()
