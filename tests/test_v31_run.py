import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from adaptive_vcm.codec import locate_ffmpeg
from adaptive_vcm.v31.protocol import canonical_hash
from tests.test_v31_guard import cfg
from tests.test_v31_measure import models
from tests.v31_fixture import action_source


def mod():
    assert importlib.util.find_spec('adaptive_vcm.v31.run') is not None, 'V31 stage CLI missing'
    return importlib.import_module('adaptive_vcm.v31.run')


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture_args(tmp_path,task='od',stage='all'):
    plans = {}
    for i,split in enumerate(('fit','cal','tune','dev')):
        if task=='od':
            image_id = i+1
            path = tmp_path/f'{image_id}.png'
            Image.fromarray(np.full((100,201,3),i*48,np.uint8)).save(path)
            record = {'id':str(image_id),'image_id':image_id,'path':str(path),'width':201,'height':100,
                      'categories':[{'id':1,'name':'x'}],
                      'annotations':[{'id':image_id,'image_id':image_id,'category_id':1,
                                      'bbox':[0,0,10,10],'area':100,'iscrowd':0}]}
        else:
            path = tmp_path/f'source{i}.mp4'
            rgb,_ = action_source('ar')
            frames = np.tile(rgb['rgb'],(3,1,1,1)).copy()
            frames[:,:,:,i%3] = (frames[:,:,:,i%3].astype(int)+i*23)%256
            subprocess.run([locate_ffmpeg(),'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
                            '-s','128x128','-r','30000/1001','-i','pipe:0','-an','-c:v','libx264',
                            '-preset','ultrafast','-qp','0',str(path)],input=frames.tobytes(),check=True)
            record = {'id':f'source{i}','path':str(path),'label':0}
        plans[split] = [record]
    plan_file = tmp_path/'fixture_plan.json'
    plan_file.write_text(json.dumps(plans))
    return SimpleNamespace(stage=stage,task=task,arm='all',config=None,root=None,annotations=None,
                           out=str(tmp_path/'run'),plan_file=str(plan_file),resume_from=None,
                           device='cpu',models=models(task),shard_manifest=None),plans


@pytest.mark.parametrize('task',['ar','od'])
def test_real_codec_both_tasks_oracle_blocked_no_train_dev_and_valid_resume(task,tmp_path,monkeypatch):
    g = mod()
    args,plans = fixture_args(tmp_path,task)
    optimizer_calls = []
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:optimizer_calls.append(True))
    result = g.run_stage(args)
    assert result['status']=='ORACLE_BLOCKED'
    assert set(result['arms'])=={'a','b','c'}
    assert all(value['status']=='ORACLE_BLOCKED' for value in result['arms'].values())
    assert not optimizer_calls
    assert not list(Path(args.out).rglob('selector_last.pt'))
    assert not list(Path(args.out).rglob('dev_report.json'))
    before = sum(len(net.calls) for role in args.models[task].values() for net in role.values())
    args.stage='measure'
    repeated = g.run_stage(args)
    assert repeated['status']=='MEASURED' and repeated['conditions']==24
    after = sum(len(net.calls) for role in args.models[task].values() for net in role.values())
    assert after==before
    assert json.loads((Path(args.out)/'plan.json').read_text())['plans']==plans
    # The fitted policy cannot be replaced with a looser hand-edited guard.
    policy_path = Path(args.out)/'arms'/'b'/'policy.json'
    policy = json.loads(policy_path.read_text())
    changed = copy.deepcopy(policy)
    changed['groups']['h264:30']['distance_threshold']=.9
    changed['policy_hash']=canonical_hash({k:v for k,v in changed.items() if k!='policy_hash'})
    policy_path.write_text(json.dumps(changed))
    args.stage='oracle'
    with pytest.raises(ValueError,match='calibration'):
        g.run_stage(args)
    policy_path.write_text(json.dumps(policy))
    # A condition edited without its immutable content hash is rejected.
    condition = next((Path(args.out)/'measurements'/'conditions').glob('*.json'))
    corrupted = json.loads(condition.read_text())
    corrupted['row']['actions'][0]['total_bytes']+=1
    condition.write_text(json.dumps(corrupted))
    args.stage='measure'
    with pytest.raises(ValueError,match='hash|integrity'):
        g.run_stage(args)


def test_changed_plan_and_config_reject_resume_before_models(tmp_path):
    g = mod()
    args,plans = fixture_args(tmp_path,stage='plan')
    assert g.run_stage(args)['status']=='PLANNED'
    plans['fit'][0]['width']=202
    Path(args.plan_file).write_text(json.dumps(plans))
    with pytest.raises(ValueError,match='plan'):
        g.run_stage(args)
    plans['fit'][0]['width']=201
    Path(args.plan_file).write_text(json.dumps(plans))
    path = tmp_path/'config.json'; path.write_text(json.dumps({**cfg('b'),'additional_identity':'different'}))
    args.config=str(path)
    with pytest.raises(ValueError,match='config|identity'):
        g.run_stage(args)


def test_test_partition_and_unverified_archive_are_rejected(tmp_path):
    g = mod()
    args,plans = fixture_args(tmp_path,stage='plan')
    plans['test']=[]
    Path(args.plan_file).write_text(json.dumps(plans))
    with pytest.raises(ValueError,match='TEST|partition'):
        g.run_stage(args)
    plans.pop('test'); Path(args.plan_file).write_text(json.dumps(plans))
    archive = tmp_path/'corrupt.tgz'; archive.write_bytes(b'not an archive')
    args.resume_from=str(archive)
    with pytest.raises(ValueError,match='resume'):
        g.run_stage(args)


def test_qualified_callbacks_run_train_then_dev_and_failure_isolated():
    g = mod()
    from adaptive_vcm.v31.oracle import assess_headroom
    from tests.test_v31_oracle import headroom_fixture
    gate = {'version':'v31-oracle-gate-1','eligible':True,'task':'ar','arm':'b',
            'bindings':{key:'a'*64 for key in ('config_hash','policy_hash','registry_hash','fit_rows_hash','tune_rows_hash','code_manifest_hash')},
            'integrity':{'passed':True,'reasons':[],'fit_sources':96,'cal_sources':32,'tune_sources':128,
                         'fit_conditions':768,'tune_conditions':1024},
            'results':headroom_fixture(),'headroom':assess_headroom(headroom_fixture())}
    gate['gate_hash']=canonical_hash(gate)
    calls = []
    assert g.execute_qualified(gate,{},lambda:calls.append('train'),lambda:calls.append('dev'))['status']=='DEV_COMPLETE'
    assert calls==['train','dev']
    blocked = copy.deepcopy(gate); blocked['eligible']=False
    blocked['gate_hash']=canonical_hash({k:v for k,v in blocked.items() if k!='gate_hash'})
    assert g.execute_qualified(blocked,{},lambda:calls.append('bad'),lambda:calls.append('bad'))['status']=='ORACLE_BLOCKED'
    assert calls==['train','dev']


def test_startup_budget_scales_primary_tune_scoring_without_quadratic_guess():
    g = mod()
    estimate = g.estimate_budget(100.,.1,'od')
    assert estimate['full_measurement_estimate_seconds']==pytest.approx(100*200/4*1.35)
    assert estimate['metric_scaling']=='TUNE source n log n; three arms and four primary curves'
    assert 0 < estimate['full_scoring_conservative_estimate_seconds'] < 12*3600
