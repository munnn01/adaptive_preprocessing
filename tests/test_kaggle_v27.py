import argparse
import json
import subprocess
import sys

import pytest

from scripts.kaggle_runner import prepare
from scripts.kaggle_runner import ROOT


def args(tmp_path, measurements=320):
    return argparse.Namespace(commit='a' * 40, slug='v27-portfolio-ar-pilot',
        account='baoancut', task='ar', mode='learned', count=16, steps=1,
        bootstrap=0, recipe='v27', measurements=measurements, train_count=32,
        seed=302201, directory=tmp_path / 'payload')


def test_v27_kaggle_payload_runs_complete_train_grid_then_separate_dev(tmp_path):
    directory = prepare(args(tmp_path))
    notebook = json.loads((directory / 'notebook.ipynb').read_text())
    cell = ''.join(notebook['cells'][0]['source'])
    assert 'adaptive_vcm.train_portfolio' in cell
    assert 'configs/v27_screen.json' in cell
    assert '--count 32 --measurements 320' in cell
    assert '--count 16 --split dev' in cell
    assert '--steps' not in cell
    assert 'stdout=subprocess.PIPE' in cell
    assert ' --ablate-learned ' in cell
    assert 'a' * 40 in cell
    job = json.loads((directory / 'job.json').read_text())
    assert job['measurements'] == 320 and job['train_count'] == 32
    assert job['steps'] is None


def test_v27_kaggle_rejects_incomplete_train_codec_qp_grid(tmp_path):
    with pytest.raises(ValueError, match='complete.*codec/QP'):
        prepare(args(tmp_path, 32))


def test_v27_auditor_starts_as_a_direct_cli():
    result = subprocess.run([sys.executable, 'scripts/audit_v27.py', '--help'],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--expected-commit' in result.stdout and '--baseline-v26' in result.stdout
