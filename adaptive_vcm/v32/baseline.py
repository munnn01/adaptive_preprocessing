"""Load V30-B through its immutable original runtime, never by weakening provenance."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

import torch


@dataclass
class FrozenV30Baseline:
    model: object
    neural_candidates: object
    config: dict
    manifest: dict

    @classmethod
    def load(cls, checkpoint, runtime_root, task, device='cpu'):
        root = Path(runtime_root).resolve()
        path = Path(checkpoint).resolve()
        init = root / 'adaptive_vcm/__init__.py'
        if not init.is_file() or not path.is_file() or task not in ('ar', 'od'):
            raise ValueError('V30 baseline requires an existing explicit runtime/checkpoint/task')
        alias = '_v32_original_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        if alias not in sys.modules:
            spec = importlib.util.spec_from_file_location(alias, init, submodule_search_locations=[str(init.parent)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[alias] = package
            spec.loader.exec_module(package)
        upstream = importlib.import_module(alias + '.motion_selection')
        state = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(state, dict):
            raise ValueError('invalid baseline checkpoint schema')
        model = upstream.load_motion_preprocessor(state, task).to(device).eval()
        cfg = state['training_config']
        if cfg.get('v30_variant') != 'b':
            raise ValueError('V32 frozen baseline requires V30-B')
        manifest = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            runtime_root=str(root), code=state['code'], code_sha256=state['code_sha256'],
            config_sha256=state['config_sha256'], train_source_sha256=state['train_source_sha256'],
            train_ids_sha256=state['train_ids_sha256'])
        return cls(model, upstream.neural_candidates, cfg, manifest)

    def assert_source(self, identifier, rgb):
        hashes = self.manifest['train_source_sha256']
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if identifier in hashes or digest in hashes.values():
            raise ValueError('V32 source overlaps frozen V30-B TRAIN')

    def validate_config(self, cfg):
        keys = ('qps', 'preset', 'fps', 'frames', 'temporal_stride', 'ar_size', 'od_size',
                'ar_teachers', 'ar_evaluators', 'od_teacher', 'od_evaluator', 'ar_kl_slack',
                'od_distance_slack', 'ar_confidence', 'od_score_threshold', 'min_savings',
                'ar_candidates', 'od_candidates', 'ar_require_anchor_decision')
        if any(self.config.get(k) != cfg.get(k) for k in keys):
            raise ValueError('V32 geometry/codec/guard differs from frozen V30-B baseline')

    def proposals(self, rgb, support, codec):
        return self.neural_candidates(rgb, support, codec, self.model)
