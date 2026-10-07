import argparse
import importlib
import json
import subprocess
from pathlib import Path

import pytest


def mod():
    return importlib.import_module('scripts.kaggle_v31')


def args(tmp_path, task='ar', **extra):
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    return argparse.Namespace(task=task,stage='oracle',arm='all',commit=commit,
        account='fixtureowner',slug='v31-fixture-oracle',directory=tmp_path/'payload',
        resume_dataset=None,resume_directory=None,shard_manifest=None,**extra)


@pytest.mark.parametrize('task,dataset,counts',[
    ('ar','qktttttttttt/kineticscleaned',{'fit':96,'cal':32,'tune':128,'dev':128}),
    ('od','awsaf49/coco-2017-dataset',{'fit':75,'cal':25,'tune':100,'dev':100})])
def test_pinned_oracle_has_full_counts_and_no_train(tmp_path,task,dataset,counts):
    g = mod(); a = args(tmp_path,task); result = g.prepare_v31(a)
    meta = json.loads((a.directory/'kernel-metadata.json').read_text())
    notebook = json.loads((a.directory/'notebook.ipynb').read_text())
    source = ''.join(notebook['cells'][0]['source'])
    compile(source,'notebook','exec')
    assert meta['dataset_sources']==[dataset]
    assert meta['is_private'] and meta['enable_gpu']
    assert result['qps']==[30,35,40,45] and result['source_counts']==counts
    assert result['stages']==['startup','measure','calibrate','oracle']
    assert result['commit']==a.commit and a.commit in source
    assert result['manifest']['commit']==a.commit
    assert 'fit_selector' not in source and "'all'" not in result['stages']
    assert 'KAGGLE_KEY' not in source and 'pool.json' not in source
    assert result['requires_ar_audit']==(task=='od')


def test_notebook_executes_startup_then_oracle_under_safe_harness(tmp_path,monkeypatch):
    g = mod(); a = args(tmp_path); job = g.prepare_v31(a)
    import torch
    import adaptive_vcm.v31.protocol as protocol
    import adaptive_vcm.v31.run as runner
    calls = []
    def fake_process(command,**kwargs):
        if command[:2]==['git','clone']:
            repo = Path(command[-1]); (repo/'configs').mkdir(parents=True)
            for arm in 'abc':
                (repo/'configs'/f'v31_{arm}.json').write_bytes((g.ROOT/'configs'/f'v31_{arm}.json').read_bytes())
        return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr(subprocess,'run',fake_process)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'fixture GPU')
    monkeypatch.setattr(protocol,'code_manifest_v31',lambda _:job['manifest'])
    def fake_stage(a):
        calls.append(a.stage)
        return {'status':'STARTUP_READY','conditions':32} if a.stage=='startup' else {'status':'MEASURED'}
    monkeypatch.setattr(runner,'run_stage',fake_stage)
    working = tmp_path/'working'; working.mkdir()
    inputs = tmp_path/'inputs'/'kineticscleaned'; inputs.mkdir(parents=True)
    source = ''.join(json.loads((a.directory/'notebook.ipynb').read_text())['cells'][0]['source'])
    source = source.replace('/kaggle/working',working.as_posix()).replace('/kaggle/input',inputs.parent.as_posix())
    exec(compile(source,'safe notebook','exec'),{'__name__':'__main__'})
    assert calls==['startup','measure','calibrate','oracle']
    assert (working/(a.slug+'.tgz')).is_file()


def test_startup_sharding_stops_before_full_measurement(tmp_path,monkeypatch):
    g = mod(); a = args(tmp_path); job = g.prepare_v31(a)
    assert job['startup_sources']==4
    assert "SHARD_REQUIRED" in ''.join(json.loads((a.directory/'notebook.ipynb').read_text())['cells'][0]['source'])
    a.stage='train-dev'
    with pytest.raises(ValueError,match='resume|gate'):
        g.prepare_v31(a)


def test_mutable_commit_and_unsafe_handles_rejected(tmp_path):
    g = mod(); a = args(tmp_path); a.commit='main'
    with pytest.raises(ValueError,match='SHA|commit'): g.prepare_v31(a)
    a = args(tmp_path); a.slug='../escape'
    with pytest.raises(ValueError,match='handle|slug'): g.prepare_v31(a)


def test_resume_dataset_uses_installed_cli_three_part_version_syntax(tmp_path,monkeypatch):
    g=mod()
    import scripts.run_v31_jobs as jobs
    source=tmp_path/'source'; source.mkdir(); (source/'one.json').write_text('{}')
    resume=tmp_path/'resume'; jobs.package_resume(source,resume)
    monkeypatch.setattr(jobs,'audit_artifacts',lambda *args,**kwargs:{'eligible_arms':['b']})
    a=args(tmp_path); a.stage='train-dev'; a.resume_dataset='fixtureowner/v31-resume/3'; a.resume_directory=resume
    result=g.prepare_v31(a)
    meta=json.loads((a.directory/'kernel-metadata.json').read_text())
    assert meta['dataset_sources'][-1]=='fixtureowner/v31-resume/3'
    assert result['resume']['dataset_version']=='fixtureowner/v31-resume/3'
    assert result['stages']==['train','dev']
