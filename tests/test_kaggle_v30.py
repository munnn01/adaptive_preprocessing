"""V30 Kaggle payload contract for the exploratory AR/OD screen."""

import argparse
import ast
import json
import shlex
import subprocess
import sys

import pytest

from scripts.kaggle_runner import ROOT, prepare


def screen_args(tmp_path, variant='a', task='both'):
    return argparse.Namespace(
        commit='d' * 40, slug=f'v30-{variant}-dev32-test', account='example',
        recipe='v30', variant=variant, task=task, mode='learned',
        count=32, train_count=32, measurements=320, epochs=8, width=12,
        steps=1000, bootstrap=2000, seed=302901, directory=tmp_path,
    )


def commands(cell):
    code = ''.join(cell['source'])
    compile(code, '<generated V30 notebook>', 'exec')
    tree = ast.parse(code)
    script = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'command'
                          for target in node.targets))
    calls = [shlex.split(line) for line in script.splitlines()
             if line.startswith('python -m adaptive_vcm.')]
    return script, calls


def test_v30_cli_defaults_create_train32_dev32_payload(tmp_path):
    command = [sys.executable, str(ROOT / 'scripts/kaggle_runner.py'), 'prepare',
               '--recipe', 'v30', '--variant', 'a', '--task', 'both',
               '--account', 'example', '--slug', 'v30-a-dev32-default',
               '--commit', 'd' * 40, '--directory', str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    job = json.loads((tmp_path / 'job.json').read_text())
    assert (job['count'], job['train_count'], job['measurements_per_task'],
            job['epochs'], job['width'], job['bootstrap'], job['seed']) == (
                32, 32, 320, 8, 12, 2000, 302901)
    assert job['recipe'] == 'v30' and job['variant'] == 'a'


@pytest.mark.parametrize('variant', ['a', 'b', 'c'])
def test_v30_joint_payload_pins_commit_and_runs_ar_then_od(variant, tmp_path):
    prepare(screen_args(tmp_path, variant))
    notebook = json.loads((tmp_path / 'notebook.ipynb').read_text())
    metadata = json.loads((tmp_path / 'kernel-metadata.json').read_text())
    job = json.loads((tmp_path / 'job.json').read_text())
    assert len(notebook['cells']) == 2
    assert metadata['is_private'] is True and metadata['enable_gpu'] is True
    assert metadata['dataset_sources'] == [
        'qktttttttttt/kineticscleaned', 'awsaf49/coco-2017-dataset']
    assert job['task'] == 'both' and job['sequential_tasks'] == ['ar', 'od']
    assert job['measurements_per_task'] == 320
    for task, cell in zip(('ar', 'od'), notebook['cells']):
        script, calls = commands(cell)
        assert len(calls) == 2
        training, evaluation = calls
        assert training[2] == 'adaptive_vcm.train_motion'
        assert evaluation[2] == 'adaptive_vcm.evaluate'
        assert training[training.index('--task') + 1] == task
        assert training[training.index('--config') + 1] == f'configs/v30_{variant}_screen.json'
        assert training[training.index('--count') + 1] == '32'
        assert training[training.index('--epochs') + 1] == '8'
        assert training[training.index('--width') + 1] == '12'
        assert evaluation[evaluation.index('--count') + 1] == '32'
        assert evaluation[evaluation.index('--split') + 1] == 'dev'
        assert evaluation[evaluation.index('--bootstrap') + 1] == '2000'
        assert '--ablate-learned' in evaluation
        assert '--checkpoint' in evaluation and 'preprocessor_last.pth' in evaluation[evaluation.index('--checkpoint') + 1]
        assert '--steps' not in training and '--measurements' not in training
        assert 'git -C "$REPO" checkout -q ' + 'd' * 40 in script
        assert 'test "$(git -C "$REPO" rev-parse HEAD)" = ' + 'd' * 40 in script
        assert f'REPO=/kaggle/working/adaptive_preprocessing_{task}' in script
        assert 'pool.json' not in script and 'KAGGLE_API_TOKEN' not in script


@pytest.mark.parametrize('field,value', [
    ('variant', None), ('variant', 'd'), ('measurements', 319),
    ('train_count', 1), ('epochs', 0), ('width', 3), ('mode', 'analytic'),
])
def test_v30_rejects_undeclared_or_incomplete_screen(field, value, tmp_path):
    args = screen_args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        prepare(args)


@pytest.mark.parametrize('variant', ['a', 'b', 'c'])
def test_v30_configs_keep_original_guards_and_three_primary_proposals(variant):
    config = json.loads((ROOT / f'configs/v30_{variant}_screen.json').read_text())
    baseline = json.loads((ROOT / 'configs/v29_a_screen.json').read_text())
    fixed = ('qps', 'preset', 'fps', 'frames', 'temporal_stride', 'ar_size', 'od_size',
             'ar_teachers', 'ar_evaluators', 'od_teacher', 'od_evaluator',
             'ar_kl_slack', 'ar_confidence', 'od_distance_slack',
             'od_score_threshold', 'min_savings', 'ar_candidates', 'od_candidates',
             'ar_require_anchor_decision', 'ar_guard_rule', 'ar_mode', 'od_mode',
             'motion_static_k', 'motion_proposal_scales')
    assert {key: config[key] for key in fixed} == {key: baseline[key] for key in fixed}
    assert config['experiment'] == f'v30-{variant}'
    assert config['v30_variant'] == variant and 'v29_variant' not in config
    assert config['motion_proposal_scales'] == [0.5, 1.0, 1.5]
    assert config['bootstrap_draws'] == 2000


@pytest.mark.parametrize('variant', ['a', 'b', 'c'])
def test_v30_notebook_config_is_accepted_by_train_and_eval_consumers(variant):
    from adaptive_vcm.motion_selection import validate_motion_config
    from adaptive_vcm.train_motion import _validate_config

    config = json.loads((ROOT / f'configs/v30_{variant}_screen.json').read_text())
    _validate_config(config)
    validate_motion_config(config)
