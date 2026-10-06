"""Catches a V29 payload using the wrong variant, counts, commit or task order."""
import argparse
import ast
import json
import shlex
import subprocess
import sys

import pytest

from test_kaggle_runner import runner


def test_v29_cli_defaults_prepare_the_declared_screen_without_budget_overrides(tmp_path):
    command = [sys.executable, str(runner.ROOT/'scripts/kaggle_runner.py'), 'prepare',
               '--recipe', 'v29', '--variant', 'a', '--task', 'both',
               '--account', 'example', '--slug', 'v29-a-dev32-default',
               '--commit', 'd'*40, '--directory', str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    job = json.loads((tmp_path/'job.json').read_text())
    assert (job['count'],job['train_count'],job['epochs'],job['bootstrap'],job['seed']) == (32,32,8,2000,302901)


@pytest.mark.parametrize('variant', ['a', 'b', 'c'])
def test_declared_notebook_config_is_accepted_by_the_actual_trainer(variant):
    from adaptive_vcm.train_motion import _validate_config
    from adaptive_vcm.motion_selection import validate_motion_config
    config = json.loads((runner.ROOT / f'configs/v29_{variant}_screen.json').read_text())
    _validate_config(config)
    validate_motion_config(config)
    assert config['v29_variant'] == variant


@pytest.mark.parametrize('variant', ['a', 'b', 'c'])
def test_joint_screen_emits_two_complete_task_commands_for_the_same_variant(tmp_path, variant):
    args = argparse.Namespace(commit='d' * 40, slug=f'v29-{variant}-dev32-test', account='example',
                              recipe='v29', variant=variant, task='both', mode='learned',
                              count=32, train_count=32, measurements=320, epochs=8, width=12,
                              steps=1000, bootstrap=2000, seed=302901, directory=tmp_path)
    runner.prepare(args)
    notebook = json.loads((tmp_path / 'notebook.ipynb').read_text())
    metadata = json.loads((tmp_path / 'kernel-metadata.json').read_text())
    assert len(notebook['cells']) == 2
    commands = []
    for cell in notebook['cells']:
        code = ''.join(cell['source'])
        compile(code, '<generated notebook>', 'exec')
        tree = ast.parse(code)
        command = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'command' for t in node.targets))
        calls = [shlex.split(line) for line in command.splitlines() if line.startswith('python -m adaptive_vcm.')]
        assert len(calls) == 2
        training, evaluation = calls
        assert training[2] == 'adaptive_vcm.train_motion'
        assert training[training.index('--config') + 1] == f'configs/v29_{variant}_screen.json'
        assert training[training.index('--count') + 1] == '32'
        assert training[training.index('--epochs') + 1] == '8'
        assert evaluation[evaluation.index('--count') + 1] == '32'
        assert evaluation[evaluation.index('--split') + 1] == 'dev'
        assert '--ablate-learned' in evaluation
        assert args.commit in command
        assert 'pool.json' not in command and 'KAGGLE_API_TOKEN' not in command
        commands.append(training[training.index('--task') + 1])
    assert commands == ['ar', 'od']
    assert metadata['is_private'] and metadata['enable_gpu']
    assert set(metadata['dataset_sources']) == {'qktttttttttt/kineticscleaned', 'awsaf49/coco-2017-dataset'}
    job = json.loads((tmp_path / 'job.json').read_text())
    assert job['variant'] == variant and job['count'] == job['train_count'] == 32
