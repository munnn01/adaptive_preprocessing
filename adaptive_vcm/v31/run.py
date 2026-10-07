"""Oracle-first CLI with immutable state identity and verified stage resume."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import math
import shutil
import time

from ..data import ar_plan,od_plan
from .guard import fit_policy
from .measure import collect
from .measure_models import build_models
from .measure_store import atomic_json,load_measurements,read_json,sha
from .metrics import paired_comparisons
from .oracle import oracle_report,validate_gate
from .protocol import SOURCE_COUNTS,allocate_sources,canonical_hash,code_manifest_v31,validate_config
from .select import evaluate_dev
from .train import fit_selector,load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
STAGES = ('plan','startup','measure','calibrate','oracle','train','dev','all')


def _configs(args):
    primary = 'b' if args.arm=='all' else args.arm
    path = Path(args.config) if getattr(args,'config',None) else ROOT/'configs'/f'v31_{primary}.json'
    base = validate_config(read_json(path))
    if base['v31_arm']!=primary:
        raise ValueError('config arm differs from requested stage')
    arms = tuple('abc') if args.arm=='all' else (args.arm,)
    return {arm:validate_config(dict(base,task=args.task,v31_arm=arm,experiment=f'v31-{arm}')) for arm in arms},validate_config(dict(base,task=args.task,v31_arm='b',experiment='v31-b'))


def build_plan(task,root,annotations,seed):
    root = Path(root)
    if task=='ar':
        train,_ = ar_plan(root,'train',256)
        dev,_ = ar_plan(root,'dev',128)
    else:
        train,meta = od_plan(root,Path(annotations),'train',200)
        dev,_ = od_plan(root,Path(annotations),'dev',100)
        by_id = defaultdict(list)
        chosen = {row['image_id'] for row in train+dev}
        for annotation in meta['annotations']:
            if annotation['image_id'] in chosen:
                by_id[annotation['image_id']].append(annotation)
        for row in train+dev:
            row.update(annotations=by_id[row['image_id']],categories=meta['categories'])
    return allocate_sources(train,dev,task,seed)


def portable_plan(plans):
    return {split:[{key:value for key,value in row.items() if key!='path'} for row in records]
            for split,records in plans.items()}


def _check_plan(plans,task):
    if not isinstance(plans,dict) or set(plans)!={'fit','cal','tune','dev'}:
        raise ValueError('primary V31 plan requires FIT/CAL/TUNE/DEV partitions; TEST forbidden')
    ids = []
    for records in plans.values():
        if not isinstance(records,list) or not records:
            raise ValueError('empty source partition')
        for row in records:
            identifier = row.get('id')
            if type(identifier) is not str or not identifier or identifier!=identifier.strip() or not isinstance(row.get('path'),str):
                raise ValueError('invalid source plan identity/path')
            if task=='od' and (type(row.get('image_id')) is not int or str(row['image_id'])!=identifier):
                raise ValueError('OD source plan image identity mismatch')
            ids.append(identifier)
    if len(ids)!=len(set(ids)):
        raise ValueError('source plan partitions overlap')


def import_resume(directory,out):
    directory,out = Path(directory).resolve(),Path(out).resolve()
    if not directory.is_dir():
        raise ValueError('resume requires a verified extracted directory, not an unverified archive')
    manifest = read_json(directory/'resume_manifest.json')
    if manifest.get('version')!='v31-resume-1' or not isinstance(manifest.get('files'),dict) or not manifest['files']:
        raise ValueError('resume artifact manifest missing')
    entries = []
    for name,digest in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or not name or directory==out:
            raise ValueError('unsafe resume artifact path')
        source = directory/relative
        destination = out/relative
        if (not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(directory) or
                not destination.resolve().is_relative_to(out) or destination.is_symlink() or sha(source.read_bytes())!=digest):
            raise ValueError('resume artifact checksum/path mismatch')
        if destination.exists() and name not in ('run_state.json','plan.json') and sha(destination.read_bytes())!=digest:
            raise ValueError('resume conflicts with existing immutable artifact')
        entries.append((source,destination))
    for source,destination in entries:
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)


def _signed(value,field):
    if value.get(field)!=canonical_hash({key:item for key,item in value.items() if key!=field}):
        raise ValueError('immutable artifact hash mismatch: '+field)


def execute_qualified(gate,expected,train_callback,dev_callback):
    _signed(gate,'gate_hash')
    if gate.get('eligible') is not True:
        return {'status':'ORACLE_BLOCKED','gate_hash':gate['gate_hash'],
                'reasons':gate.get('integrity',{}).get('reasons',[]) or gate.get('headroom')}
    validate_gate(gate,expected)
    train_callback()
    dev_callback()
    return {'status':'DEV_COMPLETE','gate_hash':gate['gate_hash']}


def _load_rows(store,cfg,plans):
    expected = read_json(store/'expected.json')
    if expected['config_hash']!=canonical_hash(cfg) or expected['code_provenance']!=code_manifest_v31(ROOT):
        raise ValueError('measurement config/code identity differs from release')
    ids = [row['id'] for split in ('fit','cal','tune') for row in plans[split]]
    if expected['source_ids']!=ids or expected['source_splits']!={row['id']:split for split in ('fit','cal','tune') for row in plans[split]}:
        raise ValueError('incomplete or mismatched measurement source plan')
    rows = load_measurements(store,expected)
    return rows,expected


def estimate_budget(measurement_seconds,metric_seconds,task):
    counts = SOURCE_COUNTS[task]
    source_count = sum(counts[split] for split in ('fit','cal','tune'))
    measurement = measurement_seconds*source_count/4*1.35
    n = counts['tune']
    # Matching is cached; the remaining global score sort is n log n,
    # rather than quadratic rematching. Include a 2x uncertainty factor.
    scale = n/4*(math.log2(n+1)/math.log2(5) if task=='od' else 1.)
    scoring = metric_seconds*(2000/20)*(3*4/2)*scale*2
    return {'full_measurement_estimate_seconds':measurement,
            'full_scoring_conservative_estimate_seconds':scoring,
            'total_conservative_estimate_seconds':measurement+scoring+1800,
            'metric_scaling':'TUNE source n log n; three arms and four primary curves',
            'estimate_caveat':'four-source projection; category/detection density and runtime must be observed, not a guaranteed deadline'}


def _startup(plans,cfg,out,models):
    records = [row for split in ('fit','cal','tune') for row in plans[split]][:4]
    if len(records)!=4:
        raise ValueError('empirical startup requires four whole source IDs')
    # Startup is a bounded cost audit, never an eligibility experiment.
    start = time.perf_counter()
    measured = collect({'fit':records},cfg,out/'startup'/'measurements',models)
    seconds = time.perf_counter()-start
    rows = [dict(row,choices={'probe':1}) for row in measured['rows']]
    # A 20-draw exact metric probe estimates CPU scoring cost. No performance
    # result from these four sources enters the full oracle gate.
    metric_start = time.perf_counter()
    paired_comparisons(rows,cfg['task'],[('anchor','probe')],20,cfg['seed'])
    metric_seconds = time.perf_counter()-metric_start
    result = {'version':'v31-startup-1','task':cfg['task'],'source_ids':[r['id'] for r in records],
              'conditions':32,'measurement_seconds':seconds,'metric_probe_seconds':metric_seconds,
              'metric_probe_draws':20,**estimate_budget(seconds,metric_seconds,cfg['task']),
              'measurement_expected_hash':canonical_hash(measured['expected']),
              'scope':'four-source cost audit only; cannot qualify an oracle or reduce full counts'}
    result['status'] = 'STARTUP_READY' if result['total_conservative_estimate_seconds']<10.5*3600 else 'SHARD_REQUIRED'
    atomic_json(out/'startup'/'startup.json',result)
    return result


def run_stage(args):
    if args.stage not in STAGES or args.task not in ('ar','od') or args.arm not in ('a','b','c','all'):
        raise ValueError('invalid V31 stage/task/arm')
    configs,measurement_cfg = _configs(args)
    plans = read_json(args.plan_file) if getattr(args,'plan_file',None) else build_plan(args.task,args.root,args.annotations,measurement_cfg['seed'])
    _check_plan(plans,args.task)
    out = Path(args.out).resolve()
    if getattr(args,'resume_from',None): import_resume(args.resume_from,out)
    identity = {'version':'v31-stage-state-1','task':args.task,
                'config_hashes':{arm:canonical_hash(cfg) for arm,cfg in configs.items()},
                'measurement_config_hash':canonical_hash(measurement_cfg),
                'code_provenance':code_manifest_v31(ROOT),'plan_hash':canonical_hash(portable_plan(plans))}
    state_path = out/'run_state.json'
    state = read_json(state_path) if state_path.exists() else {'identity':identity,'stages':{}}
    if state.get('identity')!=identity:
        raise ValueError('resume plan/config/code identity mismatch')
    counts = {split:len(records) for split,records in plans.items()}
    atomic_json(out/'plan.json',{'version':'v31-source-plan-1','task':args.task,'plans':plans,
                               'counts':counts,'plan_hash':identity['plan_hash'],
                               'full_protocol':counts==SOURCE_COUNTS[args.task]})
    def save(stage,result):
        state['stages'][stage] = result
        atomic_json(state_path,state)
        return result
    if args.stage=='plan': return save('plan',{'status':'PLANNED','counts':counts,'plan_hash':identity['plan_hash']})
    def models():
        return getattr(args,'models',None) or build_models(args.task,measurement_cfg,getattr(args,'device','cpu'))
    if args.stage=='startup': return save('startup',_startup(plans,measurement_cfg,out,models()))
    store = out/'measurements'
    if args.stage in ('measure','all'):
        selected = {split:plans[split] for split in ('fit','cal','tune')}
        shard_path = getattr(args,'shard_manifest',None)
        if shard_path:
            if args.stage!='measure': raise ValueError('source shards measure only; centralized calibration/oracle requires complete merge')
            shard = read_json(shard_path)
            ids = shard.get('source_ids',[])
            available = {r['id'] for records in selected.values() for r in records}
            if shard.get('parent_plan_hash')!=identity['plan_hash'] or not ids or len(ids)!=len(set(ids)) or not set(ids)<=available:
                raise ValueError('invalid whole-source shard manifest')
            selected = {split:[r for r in records if r['id'] in set(ids)] for split,records in selected.items()}
            selected = {split:records for split,records in selected.items() if records}
        measured = collect(selected,measurement_cfg,store,models())
        result = {'status':'SHARD_MEASURED' if shard_path else 'MEASURED','conditions':len(measured['rows']),
                  'sources':len(measured['expected']['source_ids']),'expected_hash':canonical_hash(measured['expected'])}
        save('measure',result)
        if args.stage=='measure': return result
    rows,expected = _load_rows(store,measurement_cfg,plans)
    fit = [r for r in rows if r['split']=='fit']
    tune = [r for r in rows if r['split']=='tune']
    arms = {}
    for arm,cfg in configs.items():
        directory = out/'arms'/arm
        policy = fit_policy(rows,args.task,cfg)
        policy_path = directory/'policy.json'
        if policy_path.exists() and read_json(policy_path)!=policy:
            raise ValueError('immutable calibration policy differs from CAL/config')
        atomic_json(policy_path,policy)
        if args.stage=='calibrate':
            arms[arm] = {'status':'CALIBRATED','policy_hash':policy['policy_hash']}
            continue
        from .actions import action_registry
        gate_path = directory/'oracle_gate.json'
        if gate_path.exists():
            gate = read_json(gate_path); _signed(gate,'gate_hash')
            bindings = gate['bindings']
            from .oracle import project_rows
            expected_bindings = {'config_hash':canonical_hash(cfg),'policy_hash':policy['policy_hash'],
                                 'fit_rows_hash':canonical_hash(project_rows(fit,action_registry(args.task,arm))),
                                 'tune_rows_hash':canonical_hash(project_rows(tune,action_registry(args.task,arm))),
                                 'code_manifest_hash':identity['code_provenance']['manifest_hash']}
            if any(bindings.get(key)!=value for key,value in expected_bindings.items()):
                raise ValueError('immutable oracle gate input identity mismatch')
        else:
            gate = oracle_report(fit,tune,policy,action_registry(args.task,arm),args.task,cfg)
            atomic_json(gate_path,gate)
        if args.stage=='oracle' or not gate['eligible']:
            arms[arm] = {'status':'ORACLE_ELIGIBLE' if gate['eligible'] else 'ORACLE_BLOCKED',
                         'gate_hash':gate['gate_hash'],'integrity':gate['integrity'],'headroom':gate['headroom']}
            continue
        validate_gate(gate,{'task':args.task,'arm':arm,'config_hash':canonical_hash(cfg),'policy_hash':policy['policy_hash']})
        training_path = directory/'train'/'training.json'
        if args.stage!='dev' and not training_path.exists():
            fit_selector([dict(row,artifact_store=str(store)) for row in fit],gate,policy,cfg,directory/'train')
        receipt = read_json(training_path)
        selector = load_checkpoint(directory/'train'/receipt['checkpoint'],receipt['checkpoint_sha256'],
                                   {'task':args.task,'arm':arm,'gate_hash':gate['gate_hash'],'policy_hash':policy['policy_hash'],
                                    'config_hash':canonical_hash(cfg),'code_manifest_hash':identity['code_provenance']['manifest_hash']})
        if args.stage=='train':
            arms[arm] = {'status':'TRAINED','checkpoint_sha256':receipt['checkpoint_sha256']}
            continue
        report_path = directory/'dev'/'dev_report.json'
        if report_path.exists():
            report = read_json(report_path); _signed(report,'report_hash')
            if report['checkpoint_state_sha256']!=selector.metadata['state_sha256'] or report['policy_hash']!=policy['policy_hash']:
                raise ValueError('frozen DEV report identity mismatch')
        else:
            dev_store = out/'dev_measurements'
            if not (dev_store/'complete.json').exists():
                collect({'dev':plans['dev']},measurement_cfg,dev_store,models())
            report = evaluate_dev({'measurement_store':str(dev_store)},selector,policy,gate['static'],cfg,directory/'dev',models())
        arms[arm] = {'status':'DEV_COMPLETE','report_hash':report['report_hash'],'claims':report['claims']}
    status = 'ORACLE_BLOCKED' if all(value['status']=='ORACLE_BLOCKED' for value in arms.values()) else 'STAGE_COMPLETE'
    return save(args.stage,{'status':status,'task':args.task,'arms':arms,'counts':counts})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage',choices=STAGES,required=True)
    result.add_argument('--task',choices=('ar','od'),required=True)
    result.add_argument('--arm',choices=('a','b','c','all'),default='all')
    result.add_argument('--config')
    result.add_argument('--root')
    result.add_argument('--annotations')
    result.add_argument('--plan-file')
    result.add_argument('--out',required=True)
    result.add_argument('--resume-from')
    result.add_argument('--shard-manifest')
    result.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    return result


def main():
    import json
    result = run_stage(parser().parse_args())
    print('[v31] '+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__': main()
