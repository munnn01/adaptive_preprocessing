"""Dedicated immutable four-QP V31 notebooks; legacy runners are unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from adaptive_vcm.v31.protocol import SCHEMA,SOURCE_COUNTS,canonical_hash
from adaptive_vcm.v31.measure_store import atomic_json,read_json

ROOT = Path(__file__).resolve().parents[1]


def release_manifest(commit):
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):
        raise ValueError('an immutable full commit SHA is required')
    actual = subprocess.check_output(['git','rev-parse',commit],cwd=ROOT,text=True).strip()
    if actual!=commit: raise ValueError('commit identity mismatch')
    names = subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT,text=True).splitlines()
    names = [name for name in names if (name.startswith('adaptive_vcm/') and name.endswith('.py')) or
             name in [f'configs/v31_{arm}.json' for arm in 'abc'] or
             (name.startswith('scripts/') and 'v31' in name and name.endswith('.py'))]
    hashes = {name:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT).replace(b'\r\n',b'\n')).hexdigest() for name in sorted(names)}
    value = {'schema':SCHEMA,'commit':commit,'files_sha256':hashes}
    return {**value,'manifest_hash':canonical_hash(value)}


def notebook_source(job):
    # Data is JSON, never shell interpolation. Each subprocess receives argv.
    source = '''import argparse, hashlib, json, os, pathlib, subprocess, sys, tarfile, time
JOB = json.loads(__JOB__)
WORK = pathlib.Path('/kaggle/working')
INPUT = pathlib.Path('/kaggle/input')
REPO = WORK/'adaptive_preprocessing'
OUT = WORK/'outputs'/JOB['slug']
OUT.mkdir(parents=True,exist_ok=True)
os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
def command(argv):
    subprocess.run(argv,check=True)
def emit(tag,value):
    print(tag+' '+json.dumps(value,sort_keys=True,allow_nan=False),flush=True)
def main():
    command(['git','clone','-q','https://github.com/munnn01/adaptive_preprocessing.git',str(REPO)])
    command(['git','-C',str(REPO),'checkout','-q',JOB['commit']])
    command([sys.executable,'-m','pip','install','-q','pycocotools'])
    sys.path.insert(0,str(REPO))
    import torch
    from adaptive_vcm.v31.protocol import code_manifest_v31
    from adaptive_vcm.v31.run import run_stage
    assert code_manifest_v31(REPO)==JOB['manifest'], 'immutable release manifest mismatch'
    assert torch.cuda.is_available(), 'actual GPU required'
    runtime={'commit':JOB['commit'],'gpu':torch.cuda.get_device_name(0),'qps':JOB['qps'],'started_unix':time.time()}
    (OUT/'runtime.json').write_text(json.dumps(runtime))
    emit('[v31-runtime]',runtime)
    if JOB['task']=='ar':
        roots=list(INPUT.rglob('kineticscleaned'))
        assert roots, 'Kinetics mount missing'
        root=str(roots[0]); annotations=None
    else:
        annotations=next(INPUT.rglob('instances_val2017.json'),None)
        images=next((p for p in INPUT.rglob('val2017') if p.is_dir()),None)
        assert annotations and images, 'COCO mount missing'
        root=str(images); annotations=str(annotations)
    resume=None
    if JOB.get('resume'):
        matches=list(INPUT.rglob('resume_manifest.json'))
        matches=[p for p in matches if hashlib.sha256(p.read_bytes()).hexdigest()==JOB['resume']['manifest_sha256']]
        assert len(matches)==1, 'pinned resume mount missing or ambiguous'
        resume=str(matches[0].parent)
    shard=None
    if JOB.get('shard'):
        shard=str(OUT/'shard.json'); pathlib.Path(shard).write_text(json.dumps(JOB['shard']))
    base=dict(task=JOB['task'],arm=JOB['arm'],config=str(REPO/'configs'/('v31_'+('b' if JOB['arm']=='all' else JOB['arm'])+'.json')),
        root=root,annotations=annotations,plan_file=None,out=str(OUT),resume_from=resume,shard_manifest=None,device='cuda')
    for stage in JOB['stages']:
        args=argparse.Namespace(**base,stage=stage)
        if stage=='measure': args.shard_manifest=shard
        result=run_stage(args)
        base['resume_from']=None
        emit('[v31-stage]',{'stage':stage,'result':result})
        if stage=='startup' and result['status']=='SHARD_REQUIRED' and not shard:
            emit('[v31-sharding]',{'status':'SHARD_REQUIRED','full_job_started':False})
            return
status=1
try:
    main(); status=0
finally:
    files={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='archive_manifest.json'}
    (OUT/'archive_manifest.json').write_text(json.dumps({'version':'v31-archive-1','commit':JOB['commit'],'exit_code':status,'files':files},sort_keys=True))
    with tarfile.open(WORK/(JOB['slug']+'.tgz'),'w:gz') as archive:
        archive.add(OUT,arcname='outputs/'+JOB['slug'])
    emit('[v31-exit]',{'exit_code':status,'archive':JOB['slug']+'.tgz'})
'''
    return source.replace('__JOB__',repr(json.dumps(job,sort_keys=True,allow_nan=False)))


def prepare_v31(args):
    if args.task not in ('ar','od') or args.stage not in ('oracle','train-dev') or args.arm not in ('all','a','b','c'):
        raise ValueError('invalid V31 task/stage/arm')
    if not re.fullmatch('[a-zA-Z0-9_-]+',args.account or '') or not re.fullmatch('[a-z0-9][a-z0-9-]{5,70}',args.slug or ''):
        raise ValueError('invalid Kaggle handle/slug')
    manifest = release_manifest(args.commit)
    job = {'version':'v31-kaggle-1','task':args.task,'stage':args.stage,'arm':args.arm,
           'commit':args.commit,'manifest':manifest,'account':args.account,'slug':args.slug,
           'qps':[30,35,40,45],'source_counts':SOURCE_COUNTS[args.task],
           'startup_sources':4,'requires_ar_audit':args.task=='od',
           'stages':['startup','measure','calibrate','oracle'],'resume':None,'shard':None}
    datasets = ['qktttttttttt/kineticscleaned' if args.task=='ar' else 'awsaf49/coco-2017-dataset']
    if getattr(args,'shard_manifest',None):
        if args.stage!='oracle': raise ValueError('shards are measurement only')
        job['shard']=read_json(args.shard_manifest)
        job['stages']=['startup','measure']
    if args.stage=='train-dev':
        dataset = getattr(args,'resume_dataset',None)
        directory = getattr(args,'resume_directory',None)
        if not directory or not re.fullmatch('[a-zA-Z0-9_-]+/[a-z0-9-]+/[1-9][0-9]*',dataset or ''):
            raise ValueError('train-dev requires verified gate/measurement resume and explicit dataset version')
        from scripts.run_v31_jobs import audit_artifacts
        audit = audit_artifacts(directory,args.commit,require_gate=True,arm=args.arm)
        resume = Path(directory)/'resume_manifest.json'
        if not resume.is_file(): raise ValueError('verified resume manifest missing')
        # Validate all checksums without copying; same rules as import_resume.
        value = read_json(resume)
        for name,digest in value['files'].items():
            path=(Path(directory)/name).resolve()
            if not path.is_relative_to(Path(directory).resolve()) or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('resume checksum/path mismatch')
        job['resume']={'dataset_version':dataset,'manifest_sha256':hashlib.sha256(resume.read_bytes()).hexdigest(),'audit':audit}
        job['stages']=['train','dev']; datasets.append(dataset)
    directory = Path(args.directory)
    notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
                'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':notebook_source(job).splitlines(True)}]}
    metadata = {'id':args.account+'/'+args.slug,'title':args.slug,'code_file':'notebook.ipynb','language':'python',
                'kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
                'dataset_sources':datasets,'competition_sources':[],'kernel_sources':[]}
    for name,value in [('job.json',job),('notebook.ipynb',notebook),('kernel-metadata.json',metadata)]:
        path=directory/name
        if path.exists() and read_json(path)!=value: raise ValueError('immutable prepared payload differs')
        atomic_json(path,value)
    return job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['prepare'])
    for key in ('task','stage','arm','commit','account','slug','directory'): parser.add_argument('--'+key,required=True)
    for key in ('resume-dataset','resume-directory','shard-manifest'): parser.add_argument('--'+key)
    job=prepare_v31(parser.parse_args())
    print(json.dumps({key:job[key] for key in ('task','stage','arm','commit','account','slug','source_counts','qps')}))


if __name__=='__main__': main()
