"""Regression cases for false marginal credit and mixed TRAIN metadata."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from adaptive_vcm.data import fingerprint, partition
from adaptive_vcm.stabilized_bank import ACTION_NAMES
from adaptive_vcm.utility_ranking import (CONTEXT_DIM, CONTEXT_SCHEMA,
    build_utility_context, fit_utility_model, load_record_directory)

ROOT = Path(__file__).resolve().parents[1]


def test_equal_bytes_never_create_cv_credit_but_one_real_byte_does():
    ids = [f'byte-fixture/{i}' for i in range(100)
           if partition(f'byte-fixture/{i}') == 'train'][:12]
    x = np.zeros((12, CONTEXT_DIM))
    x[:, 0], x[:, 2:4], x[:, 4], x[:, -1] = 50 / 51, .6, .5, np.log1p(.1)
    sizes = np.full(12, 1000)
    control = np.full(12, 990)
    actions = np.full((12, 3), 990)
    rates = np.log(actions / sizes[:, None]).astype(np.float32)
    kwargs = dict(anchor_bytes=sizes, baseline_log_rate=np.log(control / sizes),
                  action_bytes=actions, control_bytes=control)
    _, report = fit_utility_model(x, np.ones_like(actions), rates, ids,
                                 ('identity', 'a', 'b', 'c'), **kwargs)
    assert report['bank_oracle']['feasible_records'] == 0
    assert report['selected_oof']['saved_bytes'] == 0
    actions[:, 0] = 989
    rates = np.log(actions / sizes[:, None]).astype(np.float32)
    _, report = fit_utility_model(x, np.ones_like(actions), rates, ids,
                                 ('identity', 'a', 'b', 'c'), **kwargs)
    assert report['bank_oracle']['feasible_records'] == 12
    assert report['selected_oof']['saved_bytes'] == 12


@pytest.fixture
def archive(tmp_path):
    cfg = json.loads((ROOT / 'configs/v26_screen.json').read_text())
    source = next(f'byte-fixture/{i}' for i in range(100)
                  if partition(f'byte-fixture/{i}') == 'train')
    shape = [4, 64, 64, 3]
    p = np.zeros(400)
    p[0] = 1
    x = build_utility_context(np.zeros(shape, np.uint8), 30, 'h264',
        np.zeros(shape[1:3], np.float32), [p, p], [p, p], anchor_bpp=1000 / 2048)
    def observation(name):
        return dict(name=name, coded_bytes=1000, distances=[0., 0.],
                    decisions=[True, True], stream_sha256='a' * 64, shape=shape)
    row = dict(source_id=source, codec='h264', qp=30, source_sha256='b' * 64,
               source_shape=shape, context=x.tolist(), actual_anchor_bpp=1000 / 2048,
               actions=[observation(n) for n in ACTION_NAMES],
               controls=[observation(n) for n in cfg['ar_candidates']],
               controls_selected='identity', controls_coded_bytes=1000, baseline_log_rate=0.)
    files = {}
    for name in ('task_bank', 'stabilized_bank'):
        files[f'adaptive_vcm/{name}.py'] = hashlib.sha256(
            (ROOT / f'adaptive_vcm/{name}.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    manifest = dict(task='ar', config=cfg, measurements=1, train_ids=[source],
        train_ids_sha256=fingerprint([source]), code=dict(files_sha256=files),
        action_names=list(ACTION_NAMES), context_dim=CONTEXT_DIM, context_schema=CONTEXT_SCHEMA)
    def write():
        measured = tmp_path / 'measurements.jsonl'
        measured.write_text(json.dumps(row) + '\n')
        manifest['measurements_sha256'] = hashlib.sha256(measured.read_bytes()).hexdigest()
        (tmp_path / 'training_manifest.json').write_text(json.dumps(manifest))
    write()
    assert load_record_directory(tmp_path)['anchor_bytes'].tolist() == [1000]
    return tmp_path, row, write


@pytest.mark.parametrize('corruption', ['codec', 'qp', 'geometry', 'missing_controls',
                                       'anchor_stream', 'winner_name', 'winner_bytes'])
def test_replay_rejects_hash_valid_but_inconsistent_records(archive, corruption):
    path, row, write = archive
    if corruption == 'codec':
        row['context'][1] = 1.
    elif corruption == 'qp':
        row['context'][0] = 50 / 51
    elif corruption == 'geometry':
        row['context'][4] = 16 / 32
    elif corruption == 'missing_controls':
        row['controls'] = row['controls'][:1]
    elif corruption == 'anchor_stream':
        row['controls'][0]['stream_sha256'] = 'c' * 64
    elif corruption == 'winner_name':
        row['controls_selected'] = row['controls'][1]['name']
    elif corruption == 'winner_bytes':
        row['controls_coded_bytes'] = 999
    write()  # Rehash to exercise metadata validation, rather than checksum rejection.
    with pytest.raises(ValueError):
        load_record_directory(path)


def test_legacy_mlp_audit_rejects_compact_context_before_fitting(archive):
    path, _, _ = archive
    result = subprocess.run([sys.executable, '-m', 'scripts.audit_utility_cv',
        '--records', str(path), '--out', str(path / 'result.json')],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'legacy V25 measurement context' in result.stderr
