"""Atomic immutable condition artifacts and complete-grid integrity checks."""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import hashlib
import io
import json
import os
import re
from pathlib import Path
import uuid

import numpy as np

from .codec import Packet
from .protocol import CODECS, QPS, SCHEMA, canonical_hash, expected_conditions
from .transport import Recipe, pack_recipe
from ..motion_learned import validate_support


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    canonical_hash(value)
    atomic_bytes(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode('utf-8'))


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        canonical_hash(value)
        return value
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f'corrupt measurement JSON: {path}') from error


def condition_path(store, source_id, codec, qp):
    return Path(store) / 'conditions' / (canonical_hash([source_id, codec, qp]) + '.json')


def artifact(store, relative):
    if not isinstance(relative, str):
        raise ValueError('invalid artifact path')
    root = Path(store).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('artifact path escapes store')
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f'missing measurement artifact: {relative}') from error


def write_packet(store, packet):
    stream = f'streams/{packet.packet_hash}.{packet.recipe.codec}'
    recipe_path = f'recipes/{packet.packet_hash}.bin'
    for relative, data in ((stream, packet.encoded), (recipe_path, pack_recipe(packet.recipe))):
        path = Path(store) / relative
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('immutable packet artifact integrity mismatch')
        else:
            atomic_bytes(path, data)
    return {'stream_path': stream, 'recipe_path': recipe_path, 'stream_sha256': sha(packet.encoded),
            'recipe_sha256': sha(pack_recipe(packet.recipe)), 'packet_hash': packet.packet_hash,
            'elementary_bytes': packet.elementary_bytes, 'total_bytes': packet.total_bytes,
            'decoded_sha256': sha(packet.decoded.tobytes()), 'recipe': asdict(packet.recipe),
            'seconds': packet.seconds}


def write_source_artifact(store, rgb, controls, support):
    """One compressed numeric source/control/support artifact per exact content."""
    support_hash = canonical_hash({key: sha(support[key].tobytes()) for key in ('protection', 'motion', 'cuts')})
    metadata = json.dumps(support['metadata'], sort_keys=True, allow_nan=False).encode()
    identity = {'schema': SCHEMA, 'rgb': sha(rgb.tobytes()), 'controls': sha(controls.tobytes()),
                'support': support_hash, 'metadata': canonical_hash(support['metadata'])}
    relative = f"sources/{canonical_hash(identity)}.npz"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, rgb=rgb, control_protection=controls,
        protection=support['protection'], motion=support['motion'], cuts=support['cuts'],
        metadata_utf8=np.frombuffer(metadata, np.uint8))
    data = buffer.getvalue()
    path = Path(store) / relative
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('immutable source artifact hash integrity mismatch')
    else:
        atomic_bytes(path, data)
    return {'path': relative, 'sha256': sha(data), 'bytes': len(data), 'format': 'npz_numeric_v1',
            'support_metadata_sha256': identity['metadata']}


def load_source_artifact(store, row):
    """Return owned read-only rgb, control_protection, support; never pickle."""
    try:
        info = row['source_artifact']
        data = artifact(store, info['path'])
        if info['format'] != 'npz_numeric_v1' or sha(data) != info['sha256'] or len(data) != info['bytes']:
            raise ValueError('source artifact hash/size integrity mismatch')
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            keys = {'rgb', 'control_protection', 'protection', 'motion', 'cuts', 'metadata_utf8'}
            if set(archive.files) != keys:
                raise ValueError('source artifact fields mismatch')
            arrays = {key: np.array(archive[key], copy=True) for key in keys}
        rgb, control = arrays['rgb'], arrays['control_protection']
        if rgb.dtype != np.uint8 or list(rgb.shape) != row['source']['canonical_shape'] or sha(rgb.tobytes()) != row['source']['source_sha256']:
            raise ValueError('source artifact pixel dtype/shape/hash mismatch')
        if control.dtype != np.float32 or control.shape != rgb.shape[1:3] or not np.isfinite(control).all() or np.any((control < 0) | (control > 1)) or sha(control.tobytes()) != row['control_protection_sha256']:
            raise ValueError('source artifact control dtype/shape/hash mismatch')
        if any(arrays[key].dtype != np.float32 for key in ('protection', 'motion')) or arrays['cuts'].dtype != np.bool_:
            raise ValueError('source artifact support dtype mismatch')
        support = {key: arrays[key] for key in ('protection', 'motion', 'cuts')}
        validate_support(rgb, support, row['task'])
        if canonical_hash({key: sha(support[key].tobytes()) for key in support}) != row['support_sha256']:
            raise ValueError('source artifact support hash mismatch')
        metadata = arrays['metadata_utf8']
        if metadata.dtype != np.uint8 or metadata.ndim != 1:
            raise ValueError('source artifact metadata dtype/shape mismatch')
        support['metadata'] = json.loads(metadata.tobytes())
        if canonical_hash(support['metadata']) != info['support_metadata_sha256'] or support['metadata']['task'] != row['task']:
            raise ValueError('source artifact metadata hash mismatch')
        for value in (rgb, control, *[support[key] for key in ('protection', 'motion', 'cuts')]):
            value.setflags(write=False)
        return {'rgb': rgb, 'control_protection': control, 'support': support}
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f'corrupt source artifact: {error}') from error


def validate_row(row, store=None):
    try:
        if row['schema'] != SCHEMA or row['task'] not in ('ar', 'od'):
            raise ValueError('invalid row schema/task')
        if store is not None:
            load_source_artifact(store, row)
        if row['codec'] not in CODECS or type(row['qp']) is not int or row['qp'] not in QPS:
            raise ValueError('invalid condition codec/QP')
        for predictions in [row['source_predictions'], row['anchor_predictions']] + [a['predictions'] for a in row['actions'] if a['available']]:
            if not isinstance(predictions, dict) or set(predictions) != set(row['model_hashes']) or any(
                    set(predictions[role]) != set(names) for role, names in row['model_hashes'].items()):
                raise ValueError('missing named model predictions')
        if len({a['descriptor']['name'] for a in row['actions']}) != len(row['actions']):
            raise ValueError('repeated action observations')
        actual_counts = {'registered': len(row['actions']), 'available': sum(a['available'] for a in row['actions']),
                         'unavailable': sum(not a['available'] for a in row['actions']),
                         'unavailable_reasons': {reason: sum(a['reason'] == reason for a in row['actions'])
                            for reason in sorted({a['reason'] for a in row['actions'] if not a['available']})}}
        if row['counts'] != actual_counts:
            raise ValueError('unavailable/registered observation counts mismatch')
        if row['anchor_predictions'] != row['actions'][0]['predictions'] or row['actions'][0]['descriptor']['name'] != 'identity':
            raise ValueError('missing identity anchor observation')
        for action in row['actions']:
            if not action['available']:
                if action['reason'] not in ('padded_source', 'unknown_source_fps', 'unknown_source_duration', 'candidate_encoding_failure') or any(action[k] is not None for k in
                    ('recipe', 'stream_path', 'recipe_path', 'total_bytes', 'elementary_bytes', 'predictions', 'rate')):
                    raise ValueError('fabricated unavailable action observation')
                if action['reason'] == 'candidate_encoding_failure' and (not isinstance(action['error'], str) or len(action['error']) > 1000):
                    raise ValueError('invalid bounded candidate encoding failure audit')
                continue
            recipe = Recipe(**action['recipe'])
            if action['total_bytes'] != action['elementary_bytes'] + 32 or action['total_bytes'] <= 32 or action['reason'] is not None or action['predictions'] is None:
                raise ValueError('invalid successful packet accounting')
            if recipe.task != row['task'] or recipe.codec != row['codec']:
                raise ValueError('packet recipe/condition mismatch')
            if recipe.analyzer_frames != (16 if row['task'] == 'ar' else 1) or [recipe.height, recipe.width] != action['coded_shape']:
                raise ValueError('packet recipe geometry/count mismatch')
            duration = Fraction(*row['source']['duration']) if row['task'] == 'ar' else Fraction(1, 25)
            expected_size = action['descriptor']['size']
            expected_shape = ([expected_size, expected_size] if expected_size is not None
                              else row['source']['canonical_shape'][1:3])
            if recipe.duration != duration:
                raise ValueError('packet recipe duration differs from source contract')
            if action['coded_shape'] != expected_shape or recipe.repeat_factor != action['descriptor']['repeat_factor']:
                raise ValueError('packet recipe geometry/sampling differs from action contract')
            denominator = (row['source']['duration'][0] / row['source']['duration'][1] if row['task'] == 'ar'
                           else np.prod(row['source']['original_shape']))
            if denominator <= 0 or action['rate'] != action['total_bytes'] * 8 / denominator:
                raise ValueError('invalid primary rate accounting')
            if any(re.fullmatch('[0-9a-f]{64}', action[key]) is None for key in
                   ('rgb_sha256', 'stream_sha256', 'recipe_sha256', 'packet_hash', 'decoded_sha256')):
                raise ValueError('invalid packet hash observation')
            if store is not None:
                encoded, header = artifact(store, action['stream_path']), artifact(store, action['recipe_path'])
                if sha(encoded) != action['stream_sha256'] or len(encoded) != action['elementary_bytes'] or sha(header) != action['recipe_sha256'] or header != pack_recipe(recipe):
                    raise ValueError('packet artifact hash integrity mismatch')
                packet = Packet(encoded, recipe, np.zeros((recipe.analyzer_frames, recipe.height, recipe.width, 3), np.uint8), 0)
                if packet.packet_hash != action['packet_hash']:
                    raise ValueError('packet identity hash integrity mismatch')
        canonical_hash(row)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('malformed measurement row') from error


def read_condition(path, identity, store):
    value = read_json(path)
    if value.get('identity') != identity:
        raise ValueError('immutable condition identity/configuration mismatch')
    row = value.get('row')
    if not isinstance(row, dict) or value.get('row_hash') != canonical_hash(row):
        raise ValueError('condition row hash integrity mismatch')
    validate_row(row, store)
    fields = ('schema', 'source_id', 'source', 'split', 'task', 'codec', 'qp',
              'config_hash', 'model_hashes', 'code_manifest_hash')
    if any(row.get(key) != identity.get(key) for key in fields):
        raise ValueError('condition row does not match expected identity envelope')
    if ([a['descriptor'] for a in row['actions']] != identity.get('registry') or
            canonical_hash(row['ground_truth']) != identity.get('ground_truth_hash')):
        raise ValueError('condition registry/ground truth does not match identity envelope')
    return row


def assert_complete(rows, expected):
    try:
        conditions = expected_conditions(expected['source_ids'])
        found = set()
        for row in rows:
            validate_row(row)
            key = (row['source_id'], row['codec'], row['qp'])
            if key in found:
                raise ValueError('duplicate/repeated measurement condition')
            found.add(key)
            if row['schema'] != expected['schema'] or row['task'] != expected['task'] or row['config_hash'] != expected['config_hash'] or row['model_hashes'] != expected['model_hashes']:
                raise ValueError('measurement identity mismatch')
            if row['source']['source_sha256'] != expected['source_pixels_sha256'][row['source_id']] or [a['descriptor'] for a in row['actions']] != expected['registry']:
                raise ValueError('source/registry measurement identity mismatch')
            if row['split'] != expected['source_splits'][row['source_id']]:
                raise ValueError('source split measurement identity mismatch')
            if row['source'] != expected['source_metadata'][row['source_id']] or canonical_hash(row['ground_truth']) != expected['ground_truth_hashes'][row['source_id']]:
                raise ValueError('source metadata/ground truth measurement identity mismatch')
            if row['code_manifest_hash'] != expected['code_provenance']['manifest_hash']:
                raise ValueError('code provenance measurement identity mismatch')
        if found != conditions:
            raise ValueError('incomplete measurement grid: missing or unexpected conditions')
    except (KeyError, TypeError) as error:
        raise ValueError('malformed expected measurement contract') from error


def load_measurements(store, expected):
    store = Path(store)
    if not (store / 'complete.json').is_file():
        raise ValueError('incomplete measurements: complete manifest missing')
    manifest = read_json(store / 'complete.json')
    if manifest.get('expected') != expected or manifest.get('expected_hash') != canonical_hash(expected):
        raise ValueError('complete manifest expected identity mismatch')
    rows = []
    for entry in manifest.get('conditions', []):
        data = artifact(store, entry['path'])
        if sha(data) != entry['sha256']:
            raise ValueError('condition manifest hash integrity mismatch')
        value = json.loads(data)
        rows.append(read_condition(store / entry['path'], value['identity'], store))
    assert_complete(rows, expected)
    return rows
