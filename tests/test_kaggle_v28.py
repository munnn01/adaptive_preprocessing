import argparse
import json
import pytest
from scripts.kaggle_runner import prepare


def args(tmp_path,task='ar'):
    return argparse.Namespace(commit='a'*40,slug=f'v28-motion-{task}-pilot',account='baoancut',
                              task=task,mode='learned',count=4,steps=1,epochs=2,width=8,
                              bootstrap=0,recipe='v28',measurements=40,train_count=4,
                              seed=302301,directory=tmp_path/'payload')


@pytest.mark.parametrize('task',['ar','od'])
def test_v28_payload_binds_task_training_grid_and_final_checkpoint(tmp_path,task):
    directory=prepare(args(tmp_path,task))
    job=json.loads((directory/'job.json').read_text())
    notebook=json.loads((directory/'notebook.ipynb').read_text())
    cell=''.join(notebook['cells'][0]['source'])
    assert job['task']==task and job['recipe']=='v28'
    assert job['train_count']==4 and job['measurements']==40 and job['epochs']==2
    assert job['steps'] is None and job['width']==8
    # Check the generated external command boundary and artifact contract.
    assert 'adaptive_vcm.train_motion' in cell and '--epochs 2 --width 8' in cell
    assert '--steps' not in cell and '--measurements' not in cell
    assert 'configs/v28_screen.json' in cell and '--split dev' in cell
    assert '--ablate-learned' in cell and 'preprocessor_last.pth' in cell
    metadata=json.loads((directory/'kernel-metadata.json').read_text())
    assert metadata['dataset_sources']==(['qktttttttttt/kineticscleaned'] if task=='ar' else ['awsaf49/coco-2017-dataset'])
    assert metadata['is_private'] and metadata['enable_gpu']


@pytest.mark.parametrize('field,value',[('measurements',39),('epochs',0),('width',3),('mode','analytic')])
def test_v28_payload_rejects_incomplete_or_untrained_recipe(tmp_path,field,value):
    value_args=args(tmp_path);setattr(value_args,field,value)
    with pytest.raises(ValueError):prepare(value_args)


def test_v28_joint_payload_runs_two_tasks_sequentially_without_repo_collision(tmp_path):
    directory=prepare(args(tmp_path,'both'))
    notebook=json.loads((directory/'notebook.ipynb').read_text())
    assert len(notebook['cells'])==2
    sources=[''.join(c['source']) for c in notebook['cells']]
    assert '--task ar' in sources[0] and '--task od' in sources[1]
    assert 'REPO=/kaggle/working/adaptive_preprocessing_ar' in sources[0]
    assert 'REPO=/kaggle/working/adaptive_preprocessing_od' in sources[1]
    metadata=json.loads((directory/'kernel-metadata.json').read_text())
    assert metadata['id']=='baoancut/v28-motion-both-pilot'
    assert metadata['dataset_sources']==['qktttttttttt/kineticscleaned','awsaf49/coco-2017-dataset']
    job=json.loads((directory/'job.json').read_text())
    assert job['task']=='both' and job['sequential_tasks']==['ar','od']
    assert job['measurements_per_task']==40 and job['commit']=='a'*40
