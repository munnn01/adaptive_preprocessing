"""Frozen named analyzer adapters and role-limited observation views.

Injected adapters implement name/model_hash/observe(owned RGB), plus saliency
for AR teachers. Their 64-hex hash must cover actual weights and preprocessing;
build_models supplies that contract for the production named analyzers.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
from importlib.metadata import version
import re

import numpy as np
import torch

from ..analyzers import ActionAnalyzer, DetectionAnalyzer
from .actions import map_detections
from .protocol import canonical_hash, validate_config


ADAPTER_VERSION = 'v31-frozen-analyzer-1'


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


class AnalyzerAdapter:
    """State and preprocessing identity; observation never borrows packet memory."""
    def __init__(self, analyzer, task):
        self.analyzer, self.task, self.name = analyzer, task, analyzer.name
        analyzer.model.eval().requires_grad_(False)
        digest = hashlib.sha256()
        for name, tensor in sorted(analyzer.model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(canonical_hash({'name': name, 'dtype': str(value.dtype), 'shape': list(value.shape)}).encode())
            digest.update(value.numpy().tobytes())
        settings = {'version': ADAPTER_VERSION, 'task': task, 'name': self.name,
                    'analyzer_code': hashlib.sha256(inspect.getsource(type(analyzer)).encode()).hexdigest(),
                    'torch_version': str(torch.__version__), 'torchvision_version': version('torchvision'),
                    'device': str(getattr(analyzer, 'device', 'cpu')),
                    'model_class': type(analyzer.model).__module__ + '.' + type(analyzer.model).__qualname__,
                    'model_repr': repr(analyzer.model),
                    'mean': json_value(analyzer.mean.detach().cpu().numpy()) if task == 'ar' else None,
                    'std': json_value(analyzer.std.detach().cpu().numpy()) if task == 'ar' else None}
        # Detection resize/normalization and score settings reside in model_repr.
        if task == 'od':
            settings['transform'] = {key: json_value(getattr(analyzer.model.transform, key))
                for key in ('min_size', 'max_size', 'image_mean', 'image_std', 'size_divisible')}
            settings['detection_settings'] = {key: json_value(getattr(analyzer.model, key, None))
                for key in ('score_thresh', 'nms_thresh')}
            settings['roi_settings'] = {key: getattr(analyzer.model.roi_heads, key) for key in
                                       ('score_thresh', 'nms_thresh', 'detections_per_img')}
        self.identity = {**settings, 'state_sha256': digest.hexdigest()}
        self.model_hash = canonical_hash(self.identity)

    def observe(self, rgb):
        owned = np.array(rgb, copy=True, order='C')
        if self.task == 'od':
            return self.analyzer.predict(owned)
        with torch.no_grad():
            logits = self.analyzer.logits(self.analyzer.tensor(owned))[0]
            return {'logits': logits.cpu().numpy(), 'probabilities': logits.softmax(-1).cpu().numpy()}

    def saliency(self, rgb):
        return self.analyzer.saliency(np.array(rgb, copy=True, order='C'))


def build_models(task, cfg, device='cpu'):
    cfg = validate_config(cfg)
    if task not in ('ar', 'od') or cfg.get('task', task) != task:
        raise ValueError('invalid frozen model task')
    cls = ActionAnalyzer if task == 'ar' else DetectionAnalyzer
    names = {'teachers': cfg['ar_teachers'] if task == 'ar' else [cfg['od_teacher']],
             'evaluators': cfg['ar_evaluators'] if task == 'ar' else [cfg['od_evaluator']]}
    adapters = {name: AnalyzerAdapter(cls(name, device), task) for name in dict.fromkeys(sum(names.values(), []))}
    return {task: {role: {name: adapters[name] for name in group} for role, group in names.items()}}


def model_hashes(models, task, cfg):
    expected = {'teachers': cfg['ar_teachers'] if task == 'ar' else [cfg['od_teacher']],
                'evaluators': cfg['ar_evaluators'] if task == 'ar' else [cfg['od_evaluator']]}
    if task not in models or set(models[task]) != set(expected):
        raise ValueError('frozen model role identity mismatch')
    result = {}
    for role, names in expected.items():
        if set(models[task][role]) != set(names):
            raise ValueError('frozen model names mismatch')
        result[role] = {}
        for name in names:
            adapter = models[task][role][name]
            digest = getattr(adapter, 'model_hash', '')
            if getattr(adapter, 'name', None) != name or not callable(getattr(adapter, 'observe', None)) or re.fullmatch('[0-9a-f]{64}', digest) is None:
                raise ValueError('model identity requires actual weights/settings SHA256 hash and named observation adapter')
            if task == 'ar' and role == 'teachers' and not callable(getattr(adapter, 'saliency', None)):
                raise ValueError('AR teacher adapter requires source saliency')
            result[role][name] = digest
    return result


class ObservationCache(dict):
    """Scoped to one collect/condition invocation; exact RGB/shape/model keys."""
    def saliency(self, model, rgb):
        key = (model.model_hash, 'saliency', rgb.shape, hashlib.sha256(rgb.tobytes()).hexdigest())
        if key not in self:
            value = np.array(model.saliency(np.array(rgb, copy=True, order='C')), dtype=np.float32, copy=True)
            if value.shape != rgb.shape[1:3] or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError('invalid source teacher saliency')
            value.setflags(write=False)
            self[key] = value
        return self[key].copy()

    def observe(self, model, rgb, task):
        key = (model.model_hash, task, rgb.shape, hashlib.sha256(rgb.tobytes()).hexdigest())
        if key not in self:
            result = json_value(model.observe(np.array(rgb, copy=True, order='C')))
            if task == 'ar':
                logits, probabilities = (np.asarray(result.get(k)) for k in ('logits', 'probabilities'))
                if logits.ndim != 1 or logits.size == 0 or probabilities.shape != logits.shape or not np.isfinite(logits).all() or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
                    raise ValueError('invalid AR logits/probabilities observation')
                exponential = np.exp(logits.astype(np.float64) - float(logits.max()))
                if not np.allclose(probabilities, exponential / exponential.sum(), atol=1e-6, rtol=1e-5):
                    raise ValueError('AR probabilities do not agree with frozen logits')
            else:
                map_detections(result, rgb.shape[1:3], (1., 1., 0, 0, *rgb.shape[1:3]), rgb.shape[1:3])
            canonical_hash(result)
            self[key] = result
        return copy.deepcopy(self[key])


def predict(models, hashes, rgb, task, geometry, cache):
    result = {}
    for role, names in hashes.items():
        result[role] = {}
        for name in names:
            observation = cache.observe(models[task][role][name], rgb, task)
            if task == 'od':
                observation = json_value(map_detections(observation, rgb.shape[1:3],
                                                       geometry['source_transform'], geometry['original_shape']))
            result[role][name] = observation
    return result


def role_view(rows, role):
    """CAL labels retained for calibration; runtime receives teacher/context only."""
    if role not in ('cal', 'runtime', 'fit', 'offline'):
        raise ValueError('unknown measurement observation role')
    result = copy.deepcopy([row for row in rows if role not in ('cal', 'fit') or row['split'] == role])
    if role == 'offline':
        return result
    for row in result:
        for key in ('source_predictions', 'anchor_predictions'):
            row[key] = {'teachers': row[key]['teachers']}
        for action in row['actions']:
            if action['predictions'] is not None:
                action['predictions'] = {'teachers': action['predictions']['teachers']}
        row['model_hashes'] = {'teachers': row['model_hashes']['teachers']}
        if role in ('runtime', 'fit'):
            row.pop('ground_truth', None)
    return result
