"""Policy-independent real-codec observations for immutable complete action grids.

collect accepts only explicitly supplied stage partitions and never discovers
another split. Effective config includes task; arm views must keep this exact
measurement identity and separately record their policy configuration identity.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
import copy
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from ..codec import locate_ffmpeg
from ..data import read_image
from ..motion_support import build_motion_support
from ..preprocessing import action_protection, boxes_to_mask, normalize_map
from ..rateaware import semantic_protection
from .actions import action_registry, execute_actions
from .codec import V31Codec
from .measure_models import (ObservationCache, build_models, json_value,
                             model_hashes, predict, role_view)
from .measure_store import (assert_complete, atomic_json, condition_path,
                            load_measurements, load_source_artifact, read_condition, read_json, sha,
                            write_packet, write_source_artifact)
from .protocol import (CODECS, QPS, SCHEMA, canonical_hash, code_manifest_v31,
                       validate_config, validate_partitions)
from .transport import pack_recipe


class CandidateEncodingError(RuntimeError):
    """Positively identified action-local rejection, never generic FFmpeg failure.

    The identity anchor is mandatory even for this typed error. Ordinary backend
    subprocess, timeout, I/O and geometry exceptions propagate and stop the stage.
    """


def _video_timing(path, start, stop):
    """Prefer exact probed rationals; timing ambiguity disables qualification."""
    executable = locate_ffmpeg()
    if executable is None:
        return None, 'unknown_fps', 'FFmpeg timing probe unavailable; no timing fallback is allowed'
    binary = Path(executable)
    probe = shutil.which('ffprobe') or str(binary.with_name('ffprobe' + binary.suffix))
    try:
        raw = subprocess.check_output([probe, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=avg_frame_rate,r_frame_rate:format=format_name:frame=best_effort_timestamp_time,pkt_duration_time',
            '-of', 'json', str(path)], stderr=subprocess.PIPE)
        result = json.loads(raw)
        if result.get('format', {}).get('format_name') in ('h264', 'hevc'):
            return None, 'unknown_fps', 'elementary source has ambiguous demux/field rates and no authoritative container timestamps'
        stream = result['streams'][0]
        average = Fraction(stream.get('avg_frame_rate', '0/1'))
        nominal = Fraction(stream.get('r_frame_rate', '0/1'))
        selected = result.get('frames', [])[start:stop]
        timestamps = [float(frame['best_effort_timestamp_time']) for frame in selected
                      if 'best_effort_timestamp_time' in frame]
        durations = [float(frame['pkt_duration_time']) for frame in selected
                     if 'pkt_duration_time' in frame and float(frame['pkt_duration_time']) > 0]
        cadence = np.diff(timestamps) if len(timestamps) > 1 else np.asarray(durations)
        fps = None
        variable = False
        if len(cadence):
            period = float(np.median(cadence))
            variable = period <= 0 or not np.allclose(cadence, period, atol=2e-5, rtol=2e-4)
            for candidate in (average, nominal):
                if candidate > 0 and np.allclose(cadence, float(1 / candidate), atol=2e-5, rtol=2e-4):
                    fps = candidate
                    break
            if fps is None and not variable:
                # A regular but metadata-unrepresented cadence is diagnostic only:
                # rounded decimal timestamps cannot establish an exact source FPS.
                return None, 'unknown_fps', 'sampled cadence lacks an exact consistent source FPS rational'
        else:
            fps = average if average > 0 else nominal if nominal > 0 else None
            variable = average > 0 and nominal > 0 and average != nominal
        if variable:
            return fps, 'variable_timing', 'variable/contradictory source timing cannot qualify the AR stage'
        if fps is None:
            return None, 'unknown_fps', 'source FPS is missing; no timing fallback is allowed'
        return fps, 'known_constant', None
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return None, 'unknown_fps', 'source timing probe failed; no timing fallback is allowed'


def source_sample(record, task, cfg):
    """Decode the historical centered stride recipe and retain source evidence."""
    effective = validate_config(cfg)
    if task not in ('ar', 'od') or effective.get('task', task) != task or not isinstance(record.get('id'), str) or not record['id']:
        raise ValueError('invalid source task/identifier')
    if task == 'od':
        rgb, transform = read_image(record, effective['od_size'])
        original = (record['height'], record['width'])
        transform = (*transform, *rgb.shape[1:3])
        fps, duration, timing_status, timing_error = None, None, 'image', None
        raw_indices, indices, padding = [0], [0], 0
    else:
        frames, stride, size = effective['frames'], effective['temporal_stride'], effective['ar_size']
        cap = cv2.VideoCapture(record['path'])
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open source video: {record['path']}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = max(0, total - frames * stride) // 2
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        picked, raw_indices, original = [], [], None
        try:
            while len(picked) < frames:
                ok, frame = cap.read()
                if not ok:
                    break
                geometry = frame.shape[:2]
                if original is not None and geometry != original:
                    raise ValueError('source video geometry changed during sampling')
                original = geometry
                raw_indices.append(start + len(picked) * stride)
                picked.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA))
                for _ in range(stride - 1):
                    if not cap.grab():
                        break
        finally:
            cap.release()
        if not picked:
            raise RuntimeError(f"source decoded zero frames: {record['path']}")
        fps, timing_status, timing_error = _video_timing(record['path'], start, start + frames * stride)
        padding = frames - len(picked)
        indices = raw_indices + [raw_indices[-1]] * padding
        while len(picked) < frames:
            picked.append(picked[-1].copy())
        rgb = np.stack(picked)
        # Historical square resize has separate effective x/y scales.
        transform = (size / original[1], size / original[0], 0, 0, size, size)
        duration = Fraction(frames * stride, 1) / fps if fps is not None and timing_status == 'known_constant' else None
    rgb.setflags(write=False)
    ground_truth = {key: copy.deepcopy(record[key]) for key in ('label', 'image_id', 'annotations', 'categories') if key in record}
    return {'id': record['id'], 'task': task, 'rgb': rgb, 'source_sha256': sha(rgb.tobytes()),
            'source_fps': fps, 'duration': duration, 'timing_status': timing_status, 'timing_error': timing_error,
            'raw_sample_indices': raw_indices, 'sample_indices': indices, 'padding': padding,
            'padded': bool(padding), 'original_shape': original, 'source_transform': transform,
            'ground_truth': ground_truth}


class _ScopedModels(dict):
    def __init__(self, models):
        super().__init__(models)
        self.observations = ObservationCache()
        self.rendered = {}
        self.provenance = _provenance()


def _provenance():
    return code_manifest_v31(Path(__file__).resolve().parents[2])


def _metadata(sample):
    result = {key: json_value(sample.get(key)) for key in
              ('source_sha256', 'original_shape', 'source_transform', 'raw_sample_indices',
               'sample_indices', 'padding', 'padded', 'timing_status', 'timing_error')}
    for key in ('source_fps', 'duration'):
        value = sample.get(key)
        result[key] = [value.numerator, value.denominator] if isinstance(value, Fraction) else None
    result['canonical_shape'] = list(sample['rgb'].shape)
    return result


def measure_source(sample, registry, codec_name, qp, models, cfg, store):
    """One complete condition; resume only exact source/config/model identity.

    Packet.decoded already contains restored analyzer samples. Model calls own
    writable copies; coordinate remapping takes place after exact-input caching.
    """
    effective = validate_config(cfg)
    task = effective.get('task')
    if task != sample.get('task') or tuple(registry) != action_registry(task, effective['v31_arm']):
        raise ValueError('measurement task/frozen registry identity mismatch')
    if codec_name not in CODECS or type(qp) is not int or qp not in QPS:
        raise ValueError('unsupported measurement codec/QP')
    rgb = sample['rgb']
    if sample.get('source_sha256') != sha(rgb.tobytes()):
        raise ValueError('source pixel hash identity mismatch')
    hashes = model_hashes(models, task, effective)
    provenance = models.provenance if isinstance(models, _ScopedModels) else _provenance()
    source_meta = _metadata(sample)
    identity = {'schema': SCHEMA, 'source_id': sample['id'], 'source': source_meta,
                'ground_truth_hash': canonical_hash(sample.get('ground_truth', {})),
                'split': sample.get('split', 'fit'), 'task': task, 'codec': codec_name, 'qp': qp,
                'config_hash': canonical_hash(effective), 'registry': [asdict(a) for a in registry],
                'model_hashes': hashes, 'code_manifest_hash': provenance['manifest_hash']}
    store = Path(store)
    path = condition_path(store, sample['id'], codec_name, qp)
    if path.exists():
        return read_condition(path, identity, store)
    cache = models.observations if isinstance(models, _ScopedModels) else ObservationCache()
    source_predictions = predict(models, hashes, rgb, task, sample, cache)
    if task == 'ar':
        if not isinstance(sample.get('source_fps'), Fraction) or sample['source_fps'] <= 0 or not isinstance(sample.get('duration'), Fraction) or sample['duration'] <= 0:
            atomic_json(store / 'diagnostics' / (canonical_hash([sample['id']]) + '.json'),
                {'schema': SCHEMA, 'task': task, 'source_id': sample['id'], 'source': source_meta,
                 'source_predictions': source_predictions, 'model_hashes': hashes,
                 'config_hash': identity['config_hash'], 'qualification_error': sample.get('timing_error') or 'unknown_source_duration'})
            raise ValueError('AR stage timing unavailable: known constant rational source FPS/duration required; source diagnostic only')
        semantic = np.maximum.reduce([normalize_map(cache.saliency(model, rgb))
                                     for model in models[task]['teachers'].values()])
        controls = action_protection(rgb, semantic)
        learned = semantic_protection(semantic)
    else:
        teacher = source_predictions['teachers'][effective['od_teacher']]['canonical']
        scores, boxes = np.asarray(teacher['scores']), np.asarray(teacher['boxes']).reshape(-1, 4)
        selected = boxes[scores >= effective['od_score_threshold']]
        controls = boxes_to_mask(*rgb.shape[1:3], selected)
        learned = controls if len(selected) else np.ones(rgb.shape[1:3], np.float32)
    support = build_motion_support(rgb, learned, task)
    support_hash = canonical_hash({key: sha(np.asarray(support[key]).tobytes()) for key in ('protection', 'motion', 'cuts')})
    source_artifact = write_source_artifact(store, rgb, controls, support)
    conditioned = {**sample, 'codec': codec_name, 'control_protection': controls}
    # RGB rendering is codec-independent, while its wire recipe is not. Keep
    # this cache inside collect and clear it after each complete source grid.
    render_key = (source_artifact['sha256'], canonical_hash(source_meta), qp, identity['config_hash'])
    rendered = models.rendered if isinstance(models, _ScopedModels) else {}
    if render_key not in rendered:
        templates = execute_actions(conditioned, tuple(registry), qp, support)
        for template in templates:
            if template['rgb'] is not None:
                template['rgb'].setflags(write=False)
        rendered[render_key] = templates
    executions = []
    for template in rendered[render_key]:
        execution = dict(template)
        if execution['available']:
            execution['rgb'] = execution['rgb'].copy()
            execution['recipe'] = replace(execution['recipe'], codec=codec_name)
        executions.append(execution)
    actions, packets = [], {}
    for descriptor, execution in zip(registry, executions):
        observation = {'descriptor': asdict(descriptor), 'available': execution['available'], 'reason': execution['reason'],
                       'sample_indices': list(execution['sample_indices']), 'coded_shape': json_value(execution['coded_shape']),
                       'rgb_sha256': execution['rgb_sha256'], 'recipe': None, 'stream_path': None, 'recipe_path': None,
                       'stream_sha256': None, 'recipe_sha256': None, 'packet_hash': None, 'decoded_sha256': None,
                       'elementary_bytes': None, 'total_bytes': None, 'seconds': None, 'predictions': None, 'rate': None,
                       'error': None}
        if execution['available']:
            recipe = execution['recipe']
            # Recipe-distinct inputs cannot share packets even if RGB is identical.
            packet_key = (execution['rgb_sha256'], pack_recipe(recipe), codec_name, qp, identity['config_hash'])
            if packet_key not in packets:
                codec = V31Codec(codec_name, qp, effective['preset'], recipe.fps)
                try:
                    packets[packet_key] = codec.roundtrip(execution['rgb'], recipe)
                except CandidateEncodingError as error:
                    if descriptor.name == 'identity':
                        raise
                    packets[packet_key] = str(error)[:1000]
            packet = packets[packet_key]
            if isinstance(packet, str):
                observation.update(available=False, reason='candidate_encoding_failure', error=packet)
            else:
                observation.update(write_packet(store, packet))
                observation['predictions'] = predict(models, hashes, packet.decoded, task, sample, cache)
                denominator = float(sample['duration']) if task == 'ar' else float(np.prod(sample['original_shape']))
                observation['rate'] = packet.total_bytes * 8 / denominator
        actions.append(observation)
    row = {'schema': SCHEMA, 'task': task, 'source_id': sample['id'], 'split': sample.get('split', 'fit'),
           'codec': codec_name, 'qp': qp, 'config_hash': identity['config_hash'], 'model_hashes': hashes,
           'code_manifest_hash': provenance['manifest_hash'],
           'source': source_meta, 'source_predictions': source_predictions,
           'anchor_predictions': copy.deepcopy(actions[0]['predictions']), 'actions': actions,
           'support_sha256': support_hash, 'control_protection_sha256': sha(controls.tobytes()),
           'source_artifact': source_artifact,
           'ground_truth': json_value(sample.get('ground_truth', {})),
           'counts': {'registered': len(actions), 'available': sum(a['available'] for a in actions),
                      'unavailable': sum(not a['available'] for a in actions),
                      'unavailable_reasons': {reason: sum(a['reason'] == reason for a in actions)
                         for reason in sorted({a['reason'] for a in actions if not a['available']})}}}
    atomic_json(path, {'identity': identity, 'row_hash': canonical_hash(row), 'row': row})
    return row


def collect(plans, cfg, store, models):
    """Explicit partition subset -> eight cells/source and final complete manifest.

    A interrupted store resumes valid condition files. Corrupt files fail closed;
    manual repair/new store is explicit. No missing source is silently replaced.
    """
    effective = validate_config(cfg)
    task = effective.get('task')
    if task not in ('ar', 'od') or not plans or set(plans) - {'fit', 'cal', 'tune', 'dev'}:
        raise ValueError('collect needs explicit task and stage partition subset')
    hashes = model_hashes(models, task, effective)
    registry = action_registry(task, effective['v31_arm'])
    samples = []
    for split, records in plans.items():
        for record in records:
            if split == 'cal':
                if task == 'ar' and (type(record.get('label')) is not int or not 0 <= record['label'] < 400):
                    raise ValueError('CAL calibration requires a valid Kinetics400 label')
                if task == 'od':
                    if type(record.get('image_id')) is not int or not isinstance(record.get('annotations'), list) or not isinstance(record.get('categories'), list) or not record['categories']:
                        raise ValueError('CAL calibration requires explicit COCO image_id, annotations and categories')
                    category_ids = {category['id'] for category in record['categories']}
                    for annotation in record['annotations']:
                        if annotation.get('image_id') != record['image_id'] or annotation.get('category_id') not in category_ids or not {'id', 'bbox', 'area', 'iscrowd'} <= annotation.keys():
                            raise ValueError('CAL calibration annotation does not match source/category or lacks COCO fields')
            sample = source_sample(record, task, effective)
            sample['split'] = split
            samples.append(sample)
    if not samples:
        raise ValueError('measurement stage has no sources')
    pixels = {sample['id']: sample['source_sha256'] for sample in samples}
    validate_partitions(plans, pixels)
    expected = {'schema': SCHEMA, 'task': task, 'source_ids': [s['id'] for s in samples],
                'source_pixels_sha256': pixels, 'source_splits': {s['id']: s['split'] for s in samples},
                'source_metadata': {s['id']: _metadata(s) for s in samples},
                'ground_truth_hashes': {s['id']: canonical_hash(s['ground_truth']) for s in samples},
                'config_hash': canonical_hash(effective), 'registry': [asdict(a) for a in registry], 'model_hashes': hashes,
                'code_provenance': _provenance(),
                'tracked_config_sha256': sha((Path(__file__).resolve().parents[2] / 'configs' /
                    f"v31_{effective['v31_arm']}.json").read_bytes().replace(b'\r\n', b'\n'))}
    store = Path(store)
    if (store / 'expected.json').exists():
        if read_json(store / 'expected.json') != expected:
            raise ValueError('immutable measurement store expected identity/configuration mismatch')
    else:
        atomic_json(store / 'expected.json', expected)
    if (store / 'complete.json').exists():
        rows = load_measurements(store, expected)
        return {'expected': expected, 'rows': rows, 'manifest': read_json(store / 'complete.json')}
    scoped_models = _ScopedModels(models)
    rows = []
    for sample in samples:
        for codec in CODECS:
            for qp in QPS:
                rows.append(measure_source(sample, registry, codec, qp, scoped_models, effective, store))
        scoped_models.observations.clear()
        scoped_models.rendered.clear()
    assert_complete(rows, expected)
    entries = []
    for row in rows:
        path = condition_path(store, row['source_id'], row['codec'], row['qp'])
        entries.append({'path': path.relative_to(store).as_posix(), 'sha256': sha(path.read_bytes())})
    manifest = {'schema': SCHEMA, 'expected': expected, 'expected_hash': canonical_hash(expected), 'conditions': entries}
    atomic_json(store / 'complete.json', manifest)
    return {'expected': expected, 'rows': rows, 'manifest': manifest}
