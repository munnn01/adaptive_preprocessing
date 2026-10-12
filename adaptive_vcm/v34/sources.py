"""Explicit historical pilot cohorts; CAL has no optimizer updates."""
import hashlib
from pathlib import Path
from ..data import ar_plan,od_plan,partition
from ..v33ap.sources import SourceRecord,read_source
from ..v32vc.train import od_targets

def identifier(task,item):
    return ('od/coco2017/' if task=='od' else 'ar/')+item['id']

def split_training(items,task,cfg):
    if (task not in ('ar','od') or len(items)!=cfg['train_count']+cfg['cal_count']
        or len({i['id'] for i in items})!=len(items)
        or any(partition(('coco2017/' if task=='od' else '')+i['id'])!='train' for i in items)):
        raise ValueError('distinct TRAIN-domain sources required')
    ranked=sorted(items,key=lambda i:hashlib.sha256(('v34/'+identifier(task,i)).encode()).hexdigest())
    return ranked[cfg['cal_count']:],ranked[:cfg['cal_count']]

def cohorts(ar_root,od_root,annotations,cfg,experiment):
    groups={s:[] for s in ('train','cal','dev','test')}; metadata={}
    for task,root in [('ar',ar_root)]+([('od',od_root)] if experiment=='ar_od' else []):
        plan_fn=lambda s,n: ar_plan(Path(root),s,n) if task=='ar' else od_plan(Path(root),Path(annotations),s,n)
        training,meta=plan_fn('train',cfg['train_count']+cfg['cal_count'])
        if task=='ar' and meta['unmapped_folders']: raise ValueError('unmapped K400 folders')
        fit,cal=split_training(training,task,cfg)
        plans={'train':fit,'cal':cal}
        for split in ('dev','test'): plans[split],_=plan_fn(split,cfg[split+'_count'])
        metadata[task]=meta
        for split,items in plans.items():
            for item in items:
                rgb,g,recipe=read_source(task,item,cfg)
                r=SourceRecord(identifier(task,item),task,item.copy(),rgb,g,recipe)
                if task=='od': r.targets=od_targets(meta,item,g,'cpu')
                r.split=split; groups[split].append(r)
    return groups,metadata
