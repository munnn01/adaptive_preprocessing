"""Frozen V32 settings and complete source/codec/QP manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from . import SCHEMA

ROOT = Path(__file__).resolve().parents[2]


def validate_config(cfg):
    required = dict(schema=1, v32_schema=SCHEMA, qps=[30, 35, 40, 45, 50],
        codecs=['h264', 'h265'], ar_teachers=['r3d_18', 'mc3_18'],
        ar_evaluators=['r2plus1d_18', 'r3d_18'], od_teacher='mobilenet', od_evaluator='resnet50',
        ar_kl_slack=.1, od_distance_slack=.03, ar_confidence=.6,
        od_score_threshold=.25, ar_require_anchor_decision=True, min_savings=.01,
        ar_size=128, od_size=320, frames=16, temporal_stride=2, fps=25,
        roi={'margin': .05, 'strengths': [.5, 1.], 'factors': [2, 4]})
    if not isinstance(cfg, dict) or any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('V32 config grid/guard/geometry/ROI drift')
    if type(cfg.get('seed')) is not int or type(cfg.get('bootstrap_draws')) is not int or cfg['bootstrap_draws'] < 0:
        raise ValueError('invalid V32 seed/bootstrap settings')
    if cfg.get('preset') not in ('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'):
        raise ValueError('unsupported V32 codec preset')
    return cfg


def load_config(path=None):
    path = Path(path) if path else ROOT / 'configs/v32_roi.json'
    return validate_config(json.loads(path.read_text(encoding='utf-8')))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def provenance(cfg):
    files = sorted((ROOT / 'adaptive_vcm').rglob('*.py'))
    code = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return dict(schema=SCHEMA, config_sha256=canonical_hash(cfg), code_files_sha256=code, commit=commit)
