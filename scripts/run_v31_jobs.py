"""Durable V31 intents, bounded account preflight and verified artifact collection."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile

from scripts.kaggle_runner import pool_environment
from adaptive_vcm.v31.protocol import canonical_hash,SOURCE_COUNTS,validate_partitions
from adaptive_vcm.v31.measure_store import atomic_json,load_measurements,read_json,sha,condition_path,assert_complete,read_condition
from adaptive_vcm.v31.oracle import validate_gate

TERMINAL={'COMPLETE','ERROR','CANCEL_ACKNOWLEDGED','CANCELLED'}


def now(): return datetime.now(timezone.utc).isoformat()


class KaggleClient:
    def __init__(self,pool):
        self.pool=Path(pool)
        values=read_json_pool(self.pool).values()
        self.secrets=sorted({value if isinstance(value,str) else value['key'] for value in values},key=len,reverse=True)
    def safe(self,value):
        if isinstance(value,bytes): value=value.decode('utf-8',errors='replace')
        for secret in self.secrets:
            if secret: value=value.replace(secret,'[REDACTED]')
        return value
    def call(self,account,argv,timeout=45):
        environment,_=pool_environment(self.pool,account)
        environment.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
        try:
            result=subprocess.run([sys.executable,'-m','kaggle',*argv],env=environment,capture_output=True,timeout=timeout)
            code,stdout,stderr=result.returncode,result.stdout,result.stderr
        except subprocess.TimeoutExpired as error:
            code,stdout,stderr='TIMEOUT',error.stdout or b'',error.stderr or b''
        return {'exit_code':code,'stdout':self.safe(stdout)[-50000:],'stderr':self.safe(stderr)[-4000:],'checked_utc':now()}


def read_json_pool(path):
    value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value,dict) or not value: raise ValueError('empty account pool')
    return value


def select_account(pool,probe):
    for account in read_json_pool(pool):
        if probe(account): return account
    return None


def actual_status(evidence):
    match=re.search(r'KernelWorkerStatus\.([A-Z_]+)',evidence['stdout'])
    return match.group(1) if evidence['exit_code']==0 and match else 'UNKNOWN'


def preflight_account(client,account,handle,min_hours=12):
    reasons=[]
    quota=client.call(account,['quota','--format','json'])
    inventory=client.call(account,['kernels','list','--mine','--page-size','100','--format','json'])
    states=[]; hours=None
    try:
        if quota['exit_code']!=0 or inventory['exit_code']!=0: raise ValueError('quota/inventory call failed')
        gpu=next(row for row in json.loads(quota['stdout']) if row['resource']=='GPU')
        hours=float(gpu['remaining'].removesuffix('h'))
        if not math.isfinite(hours) or hours<0:
            hours=None
            raise ValueError('invalid verified GPU quota')
        if hours<min_hours: reasons.append('insufficient verified GPU quota')
        if re.search('next page token',inventory['stdout']+inventory['stderr'],re.I):
            raise ValueError('incomplete owner inventory; pagination requires follow-up')
        entries=json.loads(inventory['stdout']); refs=[row['ref'] for row in entries]
        if any(not ref.startswith(account+'/') for ref in refs) or len(refs)!=len(set(refs)):
            raise ValueError('invalid owner inventory')
        if handle in refs: reasons.append('exact slug already exists; duplicate prevented')
        for ref in refs:
            evidence=client.call(account,['kernels','status',ref])
            state=actual_status(evidence); states.append({'handle':ref,'status':state,'evidence':evidence})
            if state not in TERMINAL: reasons.append('active or unknown owner job: '+ref)
    except (ValueError,KeyError,TypeError,StopIteration) as error:
        reasons.append(str(error))
    return {'eligible':not reasons,'reasons':reasons,'checked_utc':now(),'gpu_remaining_hours':hours,
            'quota':quota,'inventory':inventory,'owner_status_checks':states}


def extract_archive(archive,destination):
    destination=Path(destination).resolve()
    if destination.exists(): raise ValueError('archive destination already exists; immutable extraction')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers(); seen=set(); total=0
        for member in members:
            path=PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.name or ':' in member.name or
                    not path.parts or path.parts[0]!='outputs' or member.name in seen or
                    not (member.isdir() or member.isfile())):
                raise ValueError('unsafe archive path/type')
            target=(destination/Path(*path.parts)).resolve()
            if not target.is_relative_to(destination): raise ValueError('unsafe archive escape')
            seen.add(member.name); total+=member.size
        if len(members)>200000 or total>80*1024**3: raise ValueError('archive exceeds bounded artifact budget')
        destination.mkdir(parents=True)
        for member in members:
            target=destination/Path(*PurePosixPath(member.name).parts)
            if member.isdir(): target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,target.open('xb') as output: shutil.copyfileobj(source,output)
    return destination


def verify_archive(directory,commit):
    directory=Path(directory)
    manifest=read_json(directory/'archive_manifest.json')
    if manifest.get('version')!='v31-archive-1' or manifest.get('commit')!=commit:
        raise ValueError('archive release identity mismatch')
    files={p.relative_to(directory).as_posix():sha(p.read_bytes()) for p in directory.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
    if manifest['files']!=files: raise ValueError('archive content checksum/membership mismatch')
    return manifest


def audit_artifacts(directory,commit,require_gate=False,arm='all'):
    directory=Path(directory); plan=read_json(directory/'plan.json'); state=read_json(directory/'run_state.json')
    identity=state['identity']; task=identity['task']
    if identity['code_provenance']['commit']!=commit or plan['plan_hash']!=identity['plan_hash']:
        raise ValueError('artifact plan/release mismatch')
    if plan['counts']!=SOURCE_COUNTS[task] or not plan['full_protocol']:
        raise ValueError('incomplete full source plan')
    store=directory/'measurements'; expected=read_json(store/'expected.json')
    rows=load_measurements(store,expected)
    ids=[r['id'] for split in ('fit','cal','tune') for r in plan['plans'][split]]
    if expected['source_ids']!=ids or expected['code_provenance']!=identity['code_provenance'] or expected['config_hash']!=identity['measurement_config_hash']:
        raise ValueError('incomplete or changed full measurement membership')
    validate_partitions({s:plan['plans'][s] for s in ('fit','cal','tune')},expected['source_pixels_sha256'])
    gates={}; eligible=[]
    for variant in ('abc' if arm=='all' else arm):
        gate=read_json(directory/'arms'/variant/'oracle_gate.json')
        if gate.get('gate_hash')!=canonical_hash({k:v for k,v in gate.items() if k!='gate_hash'}):
            raise ValueError('oracle gate checksum mismatch')
        if gate['task']!=task or gate['arm']!=variant or not gate['integrity']['passed']:
            raise ValueError('oracle audit integrity incomplete')
        policy=read_json(directory/'arms'/variant/'policy.json')
        if policy['policy_hash']!=canonical_hash({k:v for k,v in policy.items() if k!='policy_hash'}): raise ValueError('policy checksum mismatch')
        from adaptive_vcm.v31.actions import action_registry
        from adaptive_vcm.v31.oracle import project_rows,assess_headroom
        bindings={'task':task,'arm':variant,'config_hash':identity['config_hashes'][variant],
                  'policy_hash':policy['policy_hash'],'code_manifest_hash':identity['code_provenance']['manifest_hash'],
                  'fit_rows_hash':canonical_hash(project_rows([r for r in rows if r['split']=='fit'],action_registry(task,variant))),
                  'tune_rows_hash':canonical_hash(project_rows([r for r in rows if r['split']=='tune'],action_registry(task,variant)))}
        if any(gate['bindings'].get(k)!=v for k,v in bindings.items() if k not in ('task','arm')): raise ValueError('gate measurement/policy bindings differ')
        if gate['headroom']!=assess_headroom(gate['results']): raise ValueError('gate headroom differs from measured curves')
        if gate['eligible']:
            validate_gate(gate,bindings); eligible.append(variant)
        gates[variant]={'gate_hash':gate['gate_hash'],'eligible':gate['eligible'],'headroom':gate['headroom']}
    if require_gate and not eligible: raise ValueError('no eligible validated oracle gate for train-dev resume')
    return {'version':'v31-collected-audit-1','commit':commit,'task':task,'plan_hash':plan['plan_hash'],
            'sources':len(ids),'conditions':len(rows),'expected_hash':canonical_hash(expected),'eligible_arms':eligible,'gates':gates}


def audit_partial_artifacts(directory,commit):
    """Verify existing cells without promoting incomplete grids to eligibility."""
    directory=Path(directory); plan=read_json(directory/'plan.json'); state=read_json(directory/'run_state.json')
    identity=state['identity']; task=identity['task']; plans=plan['plans']
    from adaptive_vcm.v31.run import portable_plan,_check_plan
    _check_plan(plans,task)
    counts={split:len(records) for split,records in plans.items()}
    if (counts!=SOURCE_COUNTS[task] or plan['counts']!=counts or not plan['full_protocol'] or
            plan['plan_hash']!=canonical_hash(portable_plan(plans)) or plan['plan_hash']!=identity['plan_hash'] or
            identity['code_provenance']['commit']!=commit):
        raise ValueError('partial checkpoint parent plan/release/count mismatch')
    provenance=identity['code_provenance']
    if provenance['manifest_hash']!=canonical_hash({k:v for k,v in provenance.items() if k!='manifest_hash'}):
        raise ValueError('partial checkpoint code provenance checksum mismatch')
    store=directory/'measurements'; expected=read_json(store/'expected.json')
    if expected['task']!=task or expected['config_hash']!=identity['measurement_config_hash'] or expected['code_provenance']!=provenance:
        raise ValueError('partial measurement code/config identity mismatch')
    if expected.get('tracked_config_sha256')!=provenance['files_sha256'].get('configs/v31_b.json'):
        raise ValueError('partial tracked config digest differs from release')
    from dataclasses import asdict
    from adaptive_vcm.v31.actions import action_registry
    if expected['registry']!=[asdict(action) for action in action_registry(task,'b')]:
        raise ValueError('partial measurement registry differs from frozen B-union bank')
    names=({'teachers':{'r3d_18','mc3_18'},'evaluators':{'r2plus1d_18','r3d_18'}} if task=='ar'
           else {'teachers':{'mobilenet'},'evaluators':{'resnet50'}})
    if (set(expected['model_hashes'])!=set(names) or
            any(set(expected['model_hashes'][role])!=group for role,group in names.items()) or
            any(not isinstance(digest,str) or re.fullmatch('[0-9a-f]{64}',digest) is None
                for group in expected['model_hashes'].values() for digest in group.values())):
        raise ValueError('partial frozen named model identity mismatch')
    parent_ids=[r['id'] for split in ('fit','cal','tune') for r in plans[split]]
    shard_path=directory/'shard.json'; shard=read_json(shard_path) if shard_path.exists() else None
    ids=parent_ids
    if shard:
        chosen=shard.get('source_ids',[])
        if shard.get('parent_plan_hash')!=plan['plan_hash'] or not chosen or len(chosen)!=len(set(chosen)) or not set(chosen)<=set(parent_ids):
            raise ValueError('partial source shard parent identity mismatch')
        ids=[identifier for identifier in parent_ids if identifier in set(chosen)]
    if expected['source_ids']!=ids:
        raise ValueError('partial measurement source membership differs from parent/shard')
    membership={r['id']:split for split in ('fit','cal','tune') for r in plans[split] if r['id'] in set(ids)}
    if expected['source_splits']!=membership: raise ValueError('partial measurement source partition mismatch')
    for key in ('source_pixels_sha256','source_metadata','ground_truth_hashes'):
        if set(expected[key])!=set(ids): raise ValueError('partial source metadata membership mismatch')
    for records in plans.values():
        for record in records:
            identifier=record['id']
            if identifier not in membership: continue
            truth={key:record[key] for key in ('label','image_id','annotations','categories') if key in record}
            if (expected['ground_truth_hashes'][identifier]!=canonical_hash(truth) or
                    expected['source_metadata'][identifier]['source_sha256']!=expected['source_pixels_sha256'][identifier]):
                raise ValueError('partial parent ground truth/source metadata mismatch')
    validate_partitions({split:[r for r in plans[split] if r['id'] in set(ids)] for split in ('fit','cal','tune')},expected['source_pixels_sha256'])
    rows=[]; seen=set()
    for path in sorted((store/'conditions').glob('*.json')):
        envelope=read_json(path); row=read_condition(path,envelope['identity'],store)
        identifier=row['source_id']; key=(identifier,row['codec'],row['qp'])
        if key in seen or identifier not in membership or path!=condition_path(store,*key):
            raise ValueError('duplicate/unexpected partial measurement condition')
        seen.add(key)
        if (row['task']!=task or row['config_hash']!=expected['config_hash'] or row['model_hashes']!=expected['model_hashes'] or
                row['code_manifest_hash']!=provenance['manifest_hash'] or row['split']!=membership[identifier] or
                row['source']!=expected['source_metadata'][identifier] or
                row['source']['source_sha256']!=expected['source_pixels_sha256'][identifier] or
                canonical_hash(row['ground_truth'])!=expected['ground_truth_hashes'][identifier] or
                [a['descriptor'] for a in row['actions']]!=expected['registry']):
            raise ValueError('partial condition does not match expected parent observation identity')
        rows.append(row)
    complete=(store/'complete.json').exists()
    if complete: load_measurements(store,expected)
    return {'version':'v31-partial-measurement-audit-1','commit':commit,'task':task,'plan_hash':plan['plan_hash'],
            'sources':len(ids),'conditions':len(rows),'expected_conditions':len(ids)*8,'complete':complete,
            'primary_complete':complete and shard is None,'eligible':False,'expected_hash':canonical_hash(expected),'shard':shard,
            'scope':'verified measurement checkpoint only; never authorizes optimizer or oracle completeness'}


def package_resume(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists() or destination==source or destination.is_relative_to(source): raise ValueError('resume destination must be new and outside source')
    entries={}
    for path in sorted(source.rglob('*')):
        if path.is_symlink(): raise ValueError('unsafe resume symlink')
        if (path.is_file() and path.name not in ('archive_manifest.json','resume_manifest.json') and
                path.relative_to(source).as_posix()!='runtime.json'):
            entries[path.relative_to(source).as_posix()]=sha(path.read_bytes())
    if not entries: raise ValueError('empty resume')
    for name in entries:
        path=destination/name; path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source/name,path)
    manifest={'version':'v31-resume-1','files':entries}
    atomic_json(destination/'resume_manifest.json',manifest)
    return manifest


def merge_shards(directories,plan_directory,out):
    """Merge complete disjoint source grids; calibration remains centralized."""
    plan_directory,out=Path(plan_directory),Path(out)
    plan=read_json(plan_directory/'plan.json'); state=read_json(plan_directory/'run_state.json')
    ids=[r['id'] for split in ('fit','cal','tune') for r in plan['plans'][split]]
    expected=None; by_id={}; rows=[]; entries=[]
    variable=('source_ids','source_pixels_sha256','source_splits','source_metadata','ground_truth_hashes')
    stores=[]
    for directory in directories:
        directory=Path(directory); shard_state=read_json(directory/'run_state.json')
        if shard_state['identity']!=state['identity']: raise ValueError('shard parent identity mismatch')
        store=directory/'measurements'; exp=read_json(store/'expected.json'); actual=load_measurements(store,exp)
        if expected is None: expected={**exp,'source_ids':ids,**{key:{} for key in variable[1:]}}
        if any(exp[k]!=expected[k] for k in exp if k not in variable): raise ValueError('shard model/config/code mismatch')
        if set(exp['source_ids'])&set(by_id) or not set(exp['source_ids'])<=set(ids): raise ValueError('duplicate/unexpected shard source')
        for identifier in exp['source_ids']:
            by_id[identifier]=store
            for key in variable[1:]: expected[key][identifier]=exp[key][identifier]
        rows.extend(actual); stores.append(store)
    if set(by_id)!=set(ids): raise ValueError('incomplete source shards')
    validate_partitions({s:plan['plans'][s] for s in ('fit','cal','tune')},expected['source_pixels_sha256'])
    assert_complete(rows,expected)
    out.mkdir(parents=True,exist_ok=True); store=out/'measurements'
    for src in stores:
        for path in src.rglob('*'):
            if not path.is_file() or path.name in ('expected.json','complete.json'): continue
            target=store/path.relative_to(src); target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=path.read_bytes(): raise ValueError('shard artifact collision')
            if not target.exists(): shutil.copyfile(path,target)
    for identifier in ids:
        for codec in ('h264','h265'):
            for qp in (30,35,40,45):
                path=condition_path(store,identifier,codec,qp)
                entries.append({'path':path.relative_to(store).as_posix(),'sha256':sha(path.read_bytes())})
    atomic_json(store/'expected.json',expected)
    atomic_json(store/'complete.json',{'schema':expected['schema'],'expected':expected,'expected_hash':canonical_hash(expected),'conditions':entries})
    atomic_json(out/'plan.json',plan); atomic_json(out/'run_state.json',{'identity':state['identity'],'stages':{'measure':{'status':'MEASURED','sources':len(ids),'conditions':len(entries)}}})
    load_measurements(store,expected)
    return {'sources':len(ids),'conditions':len(entries),'expected_hash':canonical_hash(expected)}


class Registry:
    def __init__(self,path,client): self.path,self.client=Path(path),client
    def read(self): return read_json(self.path) if self.path.exists() else {'version':'v31-run-registry-1','jobs':{}}
    def save(self,value): value['updated_utc']=now(); atomic_json(self.path,value)
    def prepare(self,directory):
        directory=Path(directory).resolve(); job=read_json(directory/'job.json')
        identifier=job['account']+'/'+job['slug']; value=self.read()
        digests={name:sha((directory/name).read_bytes()) for name in ('job.json','notebook.ipynb','kernel-metadata.json')}
        if identifier in value['jobs']:
            if value['jobs'][identifier]['payload_sha256']!=digests: raise ValueError('duplicate job with changed immutable payload')
            return identifier
        value['jobs'][identifier]={'job':job,'directory':str(directory),'payload_sha256':digests,'status':'PREPARED','history':[{'action':'prepare','utc':now()}]}
        self.save(value); return identifier
    def preflight(self,identifier):
        value=self.read(); entry=value['jobs'][identifier]; job=entry['job']
        evidence=preflight_account(self.client,job['account'],identifier)
        if job['requires_ar_audit'] and not any(e.get('audit',{}).get('task')=='ar' and e['audit']['commit']==job['commit'] for e in value['jobs'].values()):
            evidence['eligible']=False; evidence['reasons'].append('OD waits for audited AR oracle from same release')
        entry['preflight']=evidence; entry['history'].append({'action':'preflight','utc':now(),'eligible':evidence['eligible']})
        self.save(value); return evidence
    def submit(self,identifier):
        value=self.read(); entry=value['jobs'][identifier]
        if entry.get('submit_intent'): raise ValueError('duplicate submit intent prevented; verify existing job/version')
        evidence=entry.get('preflight',{})
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(evidence.get('checked_utc','1970-01-01T00:00:00+00:00'))).total_seconds()
        if not evidence.get('eligible') or not 0<=age<600: raise ValueError('fresh eligible preflight required')
        directory=Path(entry['directory'])
        if any(sha((directory/name).read_bytes())!=digest for name,digest in entry['payload_sha256'].items()): raise ValueError('immutable payload changed')
        intent={'utc':now(),'payload_sha256':entry['payload_sha256']}
        intent_path=self.path.parent/('submit-intent-'+canonical_hash([identifier])+'.json')
        intent_path.parent.mkdir(parents=True,exist_ok=True)
        try:
            with intent_path.open('x',encoding='utf-8') as handle:
                json.dump(intent,handle,sort_keys=True)
        except FileExistsError as error:
            raise ValueError('duplicate durable submit intent prevented') from error
        entry['submit_intent']=intent; entry['status']='SUBMIT_INTENT'; self.save(value)
        receipt=self.client.call(entry['job']['account'],['kernels','push','--path',str(directory),'--accelerator','NvidiaTeslaT4'],timeout=90)
        match=re.search(r'version\s+(\d+)',receipt['stdout'],re.I)
        entry['receipt']=receipt; entry['version']=int(match.group(1)) if match else None
        entry['status']='SUBMITTED_UNVERIFIED' if receipt['exit_code']==0 else 'SUBMISSION_FAILED'
        entry['history'].append({'action':'submit','utc':now(),'status':entry['status'],'version':entry['version']}); self.save(value)
        return {'status':entry['status'],'version':entry['version']}
    def status(self,identifier):
        value=self.read(); entry=value['jobs'][identifier]; account=entry['job']['account']
        evidence=self.client.call(account,['kernels','status',identifier]); entry['status']=actual_status(evidence); entry['status_evidence']=evidence
        logs=self.client.call(account,['kernels','logs',identifier]); entry['logs']=logs
        for line in logs['stdout'].splitlines():
            marker=line.find('[v31-runtime] ')
            if marker>=0:
                try: entry['runtime']=json.loads(line[marker+len('[v31-runtime] '):])
                except ValueError: pass
            marker=line.find('[v31-stage] ')
            if marker>=0:
                try:
                    stage=json.loads(line[marker+len('[v31-stage] '):]); entry.setdefault('stage_evidence',{})[stage['stage']]=stage['result']
                except (ValueError,KeyError): pass
        entry['history'].append({'action':'status','utc':now(),'status':entry['status']})
        self.save(value); return {'status':entry['status'],'runtime':entry.get('runtime'),'stages':entry.get('stage_evidence',{})}
    def collect(self,identifier):
        value=self.read(); entry=value['jobs'][identifier]; job=entry['job']
        if entry['status'] not in TERMINAL: raise ValueError('collect requires verified terminal status')
        base=self.path.parent/'archives'/job['slug']; base.mkdir(parents=True,exist_ok=True)
        receipt=self.client.call(job['account'],['kernels','output',identifier,'--path',str(base),'--file-pattern',r'.*\.tgz$','--page-size','100'],timeout=120)
        entry['collection_receipt']=receipt; self.save(value)
        if receipt['exit_code']!=0: raise ValueError('archive download failed')
        archives=list(base.glob('*.tgz'))
        if len(archives)!=1: raise ValueError('exactly one archive expected')
        destination=base/'extracted'
        if not destination.exists(): extract_archive(archives[0],destination)
        directory=destination/'outputs'/job['slug']; manifest=verify_archive(directory,job['commit'])
        entry['archive']={'path':str(archives[0]),'sha256':sha(archives[0].read_bytes()),'directory':str(directory),'exit_code':manifest['exit_code']}
        if manifest['exit_code']==0 and not job.get('shard'):
            try: entry['audit']=audit_artifacts(directory,job['commit'],arm=job['arm'])
            except ValueError as error: entry['audit_pending']=str(error)
        elif job.get('shard'):
            expected=read_json(directory/'measurements'/'expected.json'); rows=load_measurements(directory/'measurements',expected)
            if set(expected['source_ids'])!=set(job['shard']['source_ids']): raise ValueError('incomplete shard source membership')
            entry['shard_audit']={'sources':len(expected['source_ids']),'conditions':len(rows),'expected_hash':canonical_hash(expected)}
        entry['history'].append({'action':'collect','utc':now(),'archive_sha256':entry['archive']['sha256'],'audited':bool(entry.get('audit'))})
        self.save(value); return {'archive':entry['archive'],'audit':entry.get('audit'),'audit_pending':entry.get('audit_pending'),'shard_audit':entry.get('shard_audit')}
    def resume(self,identifier,destination):
        value=self.read(); entry=value['jobs'][identifier]
        if not entry.get('archive'): raise ValueError('resume requires collected verified measurement artifacts')
        directory=Path(entry['archive']['directory'])
        if (directory/'archive_manifest.json').exists(): verify_archive(directory,entry['job']['commit'])
        audit=audit_partial_artifacts(directory,entry['job']['commit'])
        allowed=['oracle']
        if entry.get('audit',{}).get('eligible_arms'):
            audit_artifacts(directory,entry['job']['commit'],require_gate=True,arm=entry['job']['arm']); allowed.append('train-dev')
        manifest=package_resume(directory,destination)
        entry['resume']={'directory':str(Path(destination).resolve()),'manifest_hash':canonical_hash(manifest),
                         'audit':audit,'allowed_stages':allowed,'utc':now()}
        entry['history'].append({'action':'resume','utc':now(),'allowed_stages':allowed})
        self.save(value); return entry['resume']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','preflight','submit','status','collect','resume','merge'))
    parser.add_argument('--registry',required=True); parser.add_argument('--pool',required=True)
    parser.add_argument('--job'); parser.add_argument('--directory'); parser.add_argument('--shards',nargs='+'); parser.add_argument('--plan-directory')
    args=parser.parse_args(); registry=Registry(args.registry,KaggleClient(args.pool))
    if args.operation=='prepare': result={'job':registry.prepare(args.directory)}
    elif args.operation=='resume': result=registry.resume(args.job,args.directory)
    elif args.operation=='merge': result=merge_shards(args.shards,args.plan_directory,args.directory)
    else: result=getattr(registry,args.operation)(args.job)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
