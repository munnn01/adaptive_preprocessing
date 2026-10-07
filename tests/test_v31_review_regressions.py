"""Reproduce the four material findings from the single Native branch review."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def one_thread():
    import torch
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_empty_od_measure_store_coco_and_live_identity_fallback(tmp_path):
    from tests.test_v31_measure import cfg,models,image_record
    from adaptive_vcm.v31.measure import collect
    from adaptive_vcm.v31.measure_store import load_measurements
    from adaptive_vcm.v31.metrics import score_od
    from adaptive_vcm.v31.select import choose_stream
    from adaptive_vcm.v31.guard import fit_policy
    from tests.test_v31_guard import rows,cfg as policy_cfg
    from tests.test_v31_select import Proposer,bind_policy
    from tests.v31_fixture import action_source
    nets=models()
    for group in nets['od'].values():
        for name,model in group.items():
            model.model_hash=hashlib.sha256((name+'empty-detector-fixture').encode()).hexdigest()
            model.observe=lambda rgb:{'boxes':np.empty((0,4)), 'scores':np.empty(0), 'labels':np.empty(0,np.int64)}
    record=image_record(tmp_path)
    gt={'image_id':1,'categories':[{'id':1,'name':'x'}],
        'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[0,0,10,10],'area':100,'iscrowd':0}]}
    record.update(gt)
    result=collect({'fit':[record]},cfg(),tmp_path/'store',nets)
    stored=load_measurements(tmp_path/'store',result['expected'])
    assert len(stored)==8
    prediction=stored[0]['actions'][0]['predictions']['evaluators']['resnet50']['original']
    assert prediction=={'boxes':[],'scores':[],'labels':[]}
    assert score_od([dict(prediction,source_id='1',image_id=1)],
                    {'gt_by_id':{1:gt['annotations']},'categories':gt['categories']})['map_pct']==0
    teachers=nets['od']['teachers']; policy=bind_policy(fit_policy(rows('od'),'od',policy_cfg('b')),teachers)
    sample,_=action_source('od'); sample['rgb']=np.zeros_like(sample['rgb'])
    selected=choose_stream(sample,Proposer('od'),policy,('h264',45),teachers,policy_cfg('b'))
    assert selected['action_name']=='identity' and selected['packet'].total_bytes>32


def test_parent_all_arms_can_continue_one_exact_arm(tmp_path):
    from tests.test_v31_run import fixture_args
    from adaptive_vcm.v31.run import run_stage
    args,_=fixture_args(tmp_path,stage='plan')
    run_stage(args)
    parent=json.loads((Path(args.out)/'run_state.json').read_text())['identity']
    args.arm='b'; assert run_stage(args)['status']=='PLANNED'
    assert json.loads((Path(args.out)/'run_state.json').read_text())['identity']==parent
    config=tmp_path/'changed.json'
    from tests.test_v31_guard import cfg
    config.write_text(json.dumps(dict(cfg('b'),changed_identity='different')))
    args.config=str(config)
    with pytest.raises(ValueError,match='identity|config'): run_stage(args)


def test_real_packaged_runtime_does_not_conflict_with_new_notebook(tmp_path,monkeypatch):
    from scripts.kaggle_v31 import prepare_v31,ROOT
    import scripts.run_v31_jobs as jobs
    import adaptive_vcm.v31.run as runner
    import adaptive_vcm.v31.protocol as protocol
    import torch
    from tests.test_v31_run import fixture_args
    from tests.test_kaggle_v31 import args as payload_args
    initial,plans=fixture_args(tmp_path,stage='plan'); runner.run_stage(initial)
    (Path(initial.out)/'runtime.json').write_text(json.dumps({'gpu':'previous GPU','started_unix':1}))
    resume=tmp_path/'resume'; jobs.package_resume(initial.out,resume)
    monkeypatch.setattr(jobs,'audit_artifacts',lambda *a,**k:{'eligible_arms':['b']})
    a=payload_args(tmp_path,task='od'); a.stage='train-dev'; a.resume_directory=resume; a.resume_dataset='fixtureowner/v31-resume/3'
    job=prepare_v31(a)
    calls=[]; real=runner.run_stage
    provenance=json.loads((Path(initial.out)/'run_state.json').read_text())['identity']['code_provenance']
    # The clone is simulated, while actual import_resume and identity checks run.
    # Keep the already frozen fixture provenance instead of invoking mocked git.
    monkeypatch.setattr(runner,'code_manifest_v31',lambda _:provenance)
    monkeypatch.setattr(runner,'build_plan',lambda *a:plans)
    def plan_handoff(a):
        calls.append(a.stage); a.stage='plan'; return real(a)
    monkeypatch.setattr(runner,'run_stage',plan_handoff)
    def process(command,**kwargs):
        if command[:2]==['git','clone']:
            repo=Path(command[-1]); (repo/'configs').mkdir(parents=True)
            for arm in 'abc': (repo/'configs'/f'v31_{arm}.json').write_bytes((ROOT/'configs'/f'v31_{arm}.json').read_bytes())
        return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr(subprocess,'run',process)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'new fixture GPU')
    monkeypatch.setattr(protocol,'code_manifest_v31',lambda _:job['manifest'])
    working=tmp_path/'working'; working.mkdir(); inputs=tmp_path/'inputs'; (inputs/'val2017').mkdir(parents=True)
    (inputs/'instances_val2017.json').write_text('{}')
    mount=inputs/'resume'; import shutil; shutil.copytree(resume,mount)
    source=''.join(json.loads((a.directory/'notebook.ipynb').read_text())['cells'][0]['source'])
    source=source.replace('/kaggle/working',working.as_posix()).replace('/kaggle/input',inputs.as_posix())
    exec(compile(source,'continuation handoff','exec'),{'__name__':'__main__'})
    assert calls==['train','dev']
    runtime=json.loads((working/'outputs'/a.slug/'runtime.json').read_text())
    assert runtime['gpu']=='new fixture GPU'
    assert 'runtime.json' not in json.loads((resume/'resume_manifest.json').read_text())['files']


def partial_fixture(tmp_path):
    from tests.test_v31_measure import models,cfg,image_record
    from adaptive_vcm.v31.measure import source_sample,measure_source
    from adaptive_vcm.v31.actions import action_registry
    from adaptive_vcm.v31.protocol import canonical_hash,code_manifest_v31
    from adaptive_vcm.v31.measure_store import atomic_json
    root=tmp_path/'partial'; store=root/'measurements'; record=image_record(tmp_path)
    record.update(categories=[{'id':1,'name':'x'}],annotations=[])
    sample=source_sample(record,'od',cfg()); sample['split']='fit'
    row=measure_source(sample,action_registry('od','b'),'h264',30,models(),cfg(),store)
    plans={split:[dict(record,id=str(i),image_id=i,path='/fixture/'+str(i)+'.png') for i in indices]
           for split,indices in [('fit',range(1,76)),('cal',range(76,101)),('tune',range(101,201)),('dev',range(201,301))]}
    ids=[r['id'] for split in ('fit','cal','tune') for r in plans[split]]
    pixels={identifier:hashlib.sha256(identifier.encode()).hexdigest() for identifier in ids}; pixels['1']=row['source']['source_sha256']
    metadata={identifier:dict(row['source'],source_sha256=pixels[identifier]) for identifier in ids}
    expected={'schema':row['schema'],'task':'od','source_ids':ids,'source_pixels_sha256':pixels,
              'source_splits':{r['id']:split for split in ('fit','cal','tune') for r in plans[split]},'source_metadata':metadata,
              'ground_truth_hashes':{identifier:canonical_hash(dict(row['ground_truth'],image_id=int(identifier))) for identifier in ids},
              'config_hash':row['config_hash'],'registry':[a['descriptor'] for a in row['actions']],
              'model_hashes':row['model_hashes'],'code_provenance':code_manifest_v31(Path(__file__).parents[1])}
    expected['tracked_config_sha256']=expected['code_provenance']['files_sha256']['configs/v31_b.json']
    expected['ground_truth_hashes']['1']=canonical_hash(row['ground_truth'])
    parent_hash=canonical_hash({s:[{k:v for k,v in r.items() if k!='path'} for r in records] for s,records in plans.items()})
    identity={'version':'v31-stage-state-1','task':'od','config_hashes':{'b':canonical_hash(cfg())},
              'measurement_config_hash':row['config_hash'],'code_provenance':expected['code_provenance'],'plan_hash':parent_hash}
    atomic_json(root/'plan.json',{'version':'v31-source-plan-1','task':'od','plans':plans,'counts':{'fit':75,'cal':25,'tune':100,'dev':100},
                               'plan_hash':parent_hash,'full_protocol':True})
    atomic_json(root/'run_state.json',{'identity':identity,'stages':{}}); atomic_json(store/'expected.json',expected)
    return root,expected,identity


def test_partial_measurements_resume_without_completeness_or_gate(tmp_path):
    import scripts.run_v31_jobs as jobs
    root,expected,identity=partial_fixture(tmp_path)
    audit=jobs.audit_partial_artifacts(root,identity['code_provenance']['commit'])
    assert audit['conditions']==1 and audit['expected_conditions']==1600
    assert audit['complete'] is False and audit['eligible'] is False
    client=object(); registry=jobs.Registry(tmp_path/'registry.json',client)
    registry.save({'version':'v31-run-registry-1','jobs':{'owner/partial':{'job':{'commit':identity['code_provenance']['commit'],'arm':'b'},
                   'archive':{'directory':str(root)},'history':[]}}})
    receipt=registry.resume('owner/partial',tmp_path/'resume')
    assert receipt['allowed_stages']==['oracle']
    from adaptive_vcm.v31.run import import_resume
    import_resume(tmp_path/'resume',tmp_path/'restored')
    assert not (tmp_path/'restored'/'measurements'/'complete.json').exists()
    condition=next((root/'measurements'/'conditions').glob('*.json')); condition.write_text('corrupt')
    with pytest.raises(ValueError): jobs.audit_partial_artifacts(root,identity['code_provenance']['commit'])


def test_oracle_payload_explicitly_mounts_partial_checkpoint(tmp_path,monkeypatch):
    import scripts.run_v31_jobs as jobs
    from scripts.kaggle_v31 import prepare_v31
    from tests.test_kaggle_v31 import args
    root,expected,identity=partial_fixture(tmp_path)
    resume=tmp_path/'resume'; jobs.package_resume(root,resume)
    a=args(tmp_path,task='od'); a.arm='b'; a.resume_directory=resume; a.resume_dataset='fixtureowner/partial-observations/2'
    job=prepare_v31(a)
    assert job['resume']['dataset_version']=='fixtureowner/partial-observations/2'
    assert job['stages']==['startup','measure','calibrate','oracle']
    assert job['resume']['audit']['eligible'] is False
    assert job['resume']['audit']['conditions']==1


def test_complete_shard_can_resume_without_missing_siblings_or_gate(tmp_path):
    import scripts.run_v31_jobs as jobs
    from adaptive_vcm.v31.measure_store import read_json,atomic_json
    from adaptive_vcm.v31.measure import collect
    from tests.test_v31_measure import image_record,cfg,models
    from scripts.kaggle_v31 import prepare_v31
    from tests.test_kaggle_v31 import args
    parent,_,identity=partial_fixture(tmp_path)
    shard=tmp_path/'source-shard'
    atomic_json(shard/'plan.json',read_json(parent/'plan.json'))
    atomic_json(shard/'run_state.json',read_json(parent/'run_state.json'))
    manifest={'parent_plan_hash':identity['plan_hash'],'source_ids':['1']}
    atomic_json(shard/'shard.json',manifest)
    record=image_record(tmp_path); record.update(categories=[{'id':1,'name':'x'}],annotations=[])
    collect({'fit':[record]},cfg(),shard/'measurements',models())
    audit=jobs.audit_partial_artifacts(shard,identity['code_provenance']['commit'])
    assert audit['complete'] and audit['conditions']==8 and audit['expected_conditions']==8
    assert audit['primary_complete'] is False and audit['eligible'] is False
    resume=tmp_path/'resume'; jobs.package_resume(shard,resume)
    a=args(tmp_path,task='od'); a.arm='b'; a.resume_directory=resume; a.resume_dataset='fixtureowner/one-shard/1'
    job=prepare_v31(a)
    assert job['shard']==manifest and job['stages']==['startup','measure']
    assert not (resume/'arms'/'b'/'oracle_gate.json').exists()


def test_partial_checkpoint_must_match_tracked_config_digest(tmp_path):
    import scripts.run_v31_jobs as jobs
    from adaptive_vcm.v31.measure_store import atomic_json
    root,expected,identity=partial_fixture(tmp_path)
    expected['tracked_config_sha256']='f'*64
    atomic_json(root/'measurements'/'expected.json',expected)
    with pytest.raises(ValueError,match='config'): jobs.audit_partial_artifacts(root,identity['code_provenance']['commit'])
