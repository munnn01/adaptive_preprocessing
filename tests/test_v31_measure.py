"""Real packets/store integrity with deterministic models at the model boundary."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from adaptive_vcm.v31.actions import action_registry, execute_action
from tests.v31_fixture import action_source


def module():
    assert importlib.util.find_spec('adaptive_vcm.v31.measure') is not None, 'V31 measurements missing'
    return importlib.import_module('adaptive_vcm.v31.measure')


def cfg(task='od'):
    result = json.loads((Path(__file__).parents[1] / 'configs/v31_b.json').read_text())
    return {**result, 'task': task}


class Model:
    def __init__(self, name, task, revision='1'):
        self.name, self.task, self.calls = name, task, []
        self.model_hash = hashlib.sha256((name + revision + 'real-observation-v1').encode()).hexdigest()

    def observe(self, rgb):
        assert rgb.flags.writeable and rgb.flags.owndata
        self.calls.append(rgb.shape)
        value = float(rgb.mean())
        rgb[:] = 0  # Model ownership must protect source and restored packet.
        if self.task == 'ar':
            logits = np.array([value / 255, 1.])
            exponential = np.exp(logits - logits.max())
            return {'logits': logits, 'probabilities': exponential / exponential.sum()}
        return {'boxes': np.array([[16., 80., 160., 240.]]),
                'scores': np.array([.8]), 'labels': np.array([1])}

    def saliency(self, rgb):
        assert rgb.flags.writeable
        return np.zeros(rgb.shape[1:3], np.float32)


def models(task='od'):
    teachers = ('mobilenet',) if task == 'od' else ('r3d_18', 'mc3_18')
    evaluators = ('resnet50',) if task == 'od' else ('r2plus1d_18', 'r3d_18')
    return {task: {role: {name: Model(name, task) for name in names}
                   for role, names in [('teachers', teachers), ('evaluators', evaluators)]}}


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def image_record(tmp_path, identifier='1'):
    yy, xx = np.indices((100, 201))
    rgb = np.zeros((100, 201, 3), np.uint8)
    path = tmp_path / (identifier + '.png')
    Image.fromarray(rgb).save(path)
    return {'id': identifier, 'image_id': int(identifier), 'path': str(path), 'width': 201, 'height': 100}


def test_batch_renders_each_authoritative_bank_once_and_matches_single_actions(monkeypatch):
    import adaptive_vcm.v31.actions as mod
    assert hasattr(mod, 'execute_actions'), 'validated batch action rendering missing'
    source, support = action_source('ar')
    original = source['rgb'].copy()
    expected = [execute_action(source, a, 35, support) for a in action_registry('ar', 'b')]
    calls = {'controls': 0, 'profiles': 0, 'temporal': 0}
    for attr, key in [('make_candidates', 'controls'), ('profile_candidates', 'profiles'), ('conditional_profiles', 'temporal')]:
        old = getattr(mod, attr)
        def counted(*args, _old=old, _key=key, **kwargs):
            calls[_key] += 1
            return _old(*args, **kwargs)
        monkeypatch.setattr(mod, attr, counted)
    actual = mod.execute_actions(source, action_registry('ar', 'b'), 35, support)
    assert calls == {'controls': 1, 'profiles': 1, 'temporal': 1}
    for left, right in zip(expected, actual):
        assert left['recipe'] == right['recipe']
        np.testing.assert_array_equal(left['rgb'], right['rgb'])
    actual[0]['rgb'][:] = 0
    np.testing.assert_array_equal(source['rgb'], original)
    with pytest.raises(ValueError):
        mod.execute_actions({**source, 'rgb': original[:, :96]}, action_registry('ar', 'b'), 35, support)


def test_complete_real_od_grid_resume_geometry_and_role_views(tmp_path):
    mod = module()
    record = image_record(tmp_path)
    nets = models()
    store = tmp_path / 'measurements'
    result = mod.collect({'fit': [record]}, cfg(), store, nets)
    rows = mod.load_measurements(store, result['expected'])
    assert len(rows) == 8
    mod.assert_complete(rows, result['expected'])
    row = rows[0]
    assert len(row['actions']) == 23
    anchor = row['actions'][0]
    assert row['anchor_predictions'] == anchor['predictions']
    assert anchor['total_bytes'] == anchor['elementary_bytes'] + 32
    assert anchor['stream_sha256'] == hashlib.sha256((store / anchor['stream_path']).read_bytes()).hexdigest()
    assert row['source']['original_shape'] == [100, 201]
    assert row['source']['source_transform'] == [320 / 201, 159 / 100, 0, 80, 320, 320]
    pred = anchor['predictions']['teachers']['mobilenet']
    np.testing.assert_allclose(pred['original']['boxes'][0], [16 * 201 / 320, 0, 160 * 201 / 320, 100])
    assert anchor['rate'] == anchor['total_bytes'] * 8 / (100 * 201)
    calls = len(nets['od']['teachers']['mobilenet'].calls)
    restarted = models()
    assert mod.collect({'fit': [record]}, cfg(), store, restarted)['rows'] == rows
    assert restarted['od']['teachers']['mobilenet'].calls == []
    assert calls > 1
    runtime = mod.role_view(rows, 'runtime')
    assert 'evaluators' not in json.dumps(runtime) and 'ground_truth' not in json.dumps(runtime)
    calibration = copy.deepcopy(rows[:1])
    calibration[0]['split'] = 'cal'
    view = mod.role_view(rows + calibration, 'cal')
    assert len(view) == 1 and view[0]['ground_truth']['image_id'] == 1
    assert 'evaluators' not in json.dumps(view)
    with pytest.raises(ValueError, match='duplicate|repeated'):
        mod.assert_complete(rows + rows[:1], result['expected'])
    with pytest.raises(ValueError, match='complete|missing'):
        mod.assert_complete(rows[:-1], result['expected'])
    (store / anchor['stream_path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity|hash|corrupt'):
        mod.load_measurements(store, result['expected'])
    with pytest.raises(ValueError, match='integrity|hash|corrupt'):
        mod.collect({'fit': [record]}, cfg(), store, models())


def test_interrupted_conditions_resume_but_incomplete_manifest_cannot_load(tmp_path, monkeypatch):
    mod = module()
    record = image_record(tmp_path)
    store = tmp_path / 'store'
    old = mod.measure_source
    count = [0]
    def interrupt(*args, **kwargs):
        count[0] += 1
        if count[0] == 3:
            raise RuntimeError('interrupted test')
        return old(*args, **kwargs)
    monkeypatch.setattr(mod, 'measure_source', interrupt)
    with pytest.raises(RuntimeError, match='interrupted'):
        mod.collect({'fit': [record]}, cfg(), store, models())
    assert len(list(store.glob('conditions/*.json'))) == 2
    with pytest.raises(ValueError, match='manifest|incomplete'):
        mod.load_measurements(store, {})
    monkeypatch.setattr(mod, 'measure_source', old)
    result = mod.collect({'fit': [record]}, cfg(), store, models())
    assert len(mod.load_measurements(store, result['expected'])) == 8


def test_drop_restore_is_once_exact_prediction_cache_and_unavailable_actions(tmp_path):
    mod = module()
    sample, _ = action_source('ar')
    sample.update(id='clip', ground_truth={'label': 1}, source_sha256=hashlib.sha256(sample['rgb'].tobytes()).hexdigest(),
                  sample_indices=list(range(0, 32, 2)), raw_sample_indices=list(range(0, 32, 2)), padding=0)
    original = sample['rgb'].copy()
    nets = models('ar')
    row = mod.measure_source(sample, action_registry('ar', 'b'), 'h264', 35, nets, cfg('ar'), tmp_path)
    drop = next(a for a in row['actions'] if a['descriptor']['name'] == 'drop2_128')
    assert drop['recipe']['analyzer_frames'] == 16 and drop['recipe']['coded_frames'] == 8
    assert all(shape[0] == 16 for net in nets['ar']['teachers'].values() for shape in net.calls)
    np.testing.assert_array_equal(sample['rgb'], original)
    assert drop['rate'] == drop['total_bytes'] * 8 / float(sample['duration'])
    padded = {**sample, 'id': 'padded', 'padded': True, 'padding': 2}
    row2 = mod.measure_source(padded, action_registry('ar', 'b'), 'h264', 35, models('ar'), cfg('ar'), tmp_path)
    unavailable = [a for a in row2['actions'] if a['descriptor']['kind'] == 'drop2']
    assert all(a['reason'] == 'padded_source' and a['total_bytes'] is None and a['stream_path'] is None for a in unavailable)
    unknown = {**sample, 'id': 'unknown', 'duration': None, 'source_fps': None}
    with pytest.raises(ValueError, match='timing|duration|FPS'):
        mod.measure_source(unknown, action_registry('ar', 'b'), 'h264', 35, models('ar'), cfg('ar'), tmp_path)
    diagnostic = list((tmp_path / 'diagnostics').glob('*.json'))
    assert diagnostic and json.loads(diagnostic[0].read_text())['qualification_error']
    changed = models('ar')
    changed['ar']['teachers']['r3d_18'] = Model('r3d_18', 'ar', 'changed')
    with pytest.raises(ValueError, match='identity|configuration|immutable'):
        mod.measure_source(sample, action_registry('ar', 'b'), 'h264', 35, changed, cfg('ar'), tmp_path)


def test_source_sampling_matches_historical_decoder_and_rational_timing(tmp_path):
    mod = module()
    from adaptive_vcm.data import read_video
    from adaptive_vcm.codec import locate_ffmpeg
    from fractions import Fraction
    rgb = np.stack([np.full((128, 128, 3), t * 4, np.uint8) for t in range(40)])
    path = tmp_path / 'source.mp4'
    subprocess.run([locate_ffmpeg(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '128x128', '-r', '30000/1001', '-i', 'pipe:0', '-c:v', 'libx264', '-bf', '0',
        '-threads', '2', '-pix_fmt', 'yuv420p', str(path)], input=rgb.tobytes(), check=True)
    sample = mod.source_sample({'id': 'source', 'path': str(path), 'label': 2}, 'ar', cfg('ar'))
    np.testing.assert_array_equal(sample['rgb'], read_video(str(path), 16, 128, 2))
    assert sample['source_fps'] == Fraction(30000, 1001)
    assert sample['duration'] == Fraction(32, 1) / sample['source_fps']
    assert len(sample['raw_sample_indices']) == 16 and sample['padding'] == 0
    assert sample['rgb'].flags.writeable is False


def test_duplicate_source_pixels_and_malformed_model_identity_fail(tmp_path):
    mod = module()
    one, two = image_record(tmp_path, '1'), image_record(tmp_path, '2')
    two.update(annotations=[], categories=[{'id': 1, 'name': 'person'}])
    with pytest.raises(ValueError, match='duplicate source pixel'):
        mod.collect({'fit': [one], 'cal': [two]}, cfg(), tmp_path / 'store', models())
    nets = models()
    nets['od']['teachers']['mobilenet'].model_hash = 'mobilenet'
    with pytest.raises(ValueError, match='model.*hash|identity'):
        mod.collect({'fit': [one]}, cfg(), tmp_path / 'bad', nets)


def test_exact_restored_cache_preserves_all_recipe_distinct_aliases(tmp_path, monkeypatch):
    mod = module()
    source, _ = action_source('ar')
    source['rgb'][:] = 0
    source.update(id='constant', source_sha256=hashlib.sha256(source['rgb'].tobytes()).hexdigest())
    nets = models('ar')
    old = mod.V31Codec.roundtrip
    recipes = []
    def counted(self, rgb, recipe):
        recipes.append(recipe)
        return old(self, rgb, recipe)
    monkeypatch.setattr(mod.V31Codec, 'roundtrip', counted)
    row = mod.measure_source(source, action_registry('ar', 'b'), 'h264', 35, nets, cfg('ar'), tmp_path)
    assert len(row['actions']) == 27 and len(recipes) == 6
    assert len(nets['ar']['teachers']['r3d_18'].calls) == 3
    anchor, drop = row['actions'][0], next(a for a in row['actions'] if a['descriptor']['name'] == 'drop2_128')
    assert anchor['decoded_sha256'] == drop['decoded_sha256']
    assert anchor['recipe'] != drop['recipe'] and anchor['packet_hash'] != drop['packet_hash']
    assert anchor['stream_path'] != drop['stream_path']
    # Recipes/model identities survive per-descriptor aliases, even at equal RGB.
    assert len({a['descriptor']['name'] for a in row['actions']}) == 27


def test_selected_window_timing_does_not_consult_unrelated_vfr_section(monkeypatch):
    mod = module()
    frames = [{'best_effort_timestamp_time': str(t / 25)} for t in range(40)]
    frames[39]['best_effort_timestamp_time'] = '9'
    payload = json.dumps({'streams': [{'avg_frame_rate': '20/1', 'r_frame_rate': '25/1'}], 'frames': frames}).encode()
    monkeypatch.setattr(mod.subprocess, 'check_output', lambda *args, **kwargs: payload)
    fps, status, error = mod._video_timing('unused', 0, 32)
    assert str(fps) == '25' and status == 'known_constant' and error is None
    fps, status, error = mod._video_timing('unused', 8, 40)
    assert status == 'variable_timing' and error


def test_completeness_rejects_malformed_rate_and_missing_predictions(tmp_path):
    mod = module()
    source, _ = action_source('od')
    source.update(id='x', source_sha256=hashlib.sha256(source['rgb'].tobytes()).hexdigest())
    row = mod.measure_source(source, action_registry('od', 'b'), 'h264', 35, models(), cfg(), tmp_path)
    import adaptive_vcm.v31.measure_store as store
    malformed = copy.deepcopy(row)
    malformed['qp'] = 35.0
    with pytest.raises(ValueError, match='condition|QP'):
        store.validate_row(malformed)
    malformed = copy.deepcopy(row)
    malformed['actions'][0]['rate'] *= 2
    with pytest.raises(ValueError, match='rate'):
        store.validate_row(malformed)
    malformed = copy.deepcopy(row)
    malformed['source_predictions'] = {}
    with pytest.raises(ValueError, match='prediction|model'):
        store.validate_row(malformed)


def test_calibration_requires_explicit_gt_and_keeps_only_cal_teacher_observations(tmp_path):
    mod = module()
    record = image_record(tmp_path)
    with pytest.raises(ValueError, match='CAL.*annotations|calibration'):
        mod.collect({'cal': [record]}, cfg(), tmp_path / 'missing', models())
    record.update(annotations=[], categories=[{'id': 1, 'name': 'person'}])
    result = mod.collect({'cal': [record]}, cfg(), tmp_path / 'valid', models())
    view = mod.role_view(result['rows'], 'cal')
    assert len(view) == 8 and view[0]['ground_truth'] == {'image_id': 1, 'annotations': [], 'categories': record['categories']}
    assert 'evaluators' not in json.dumps(view)
    assert 'ground_truth' not in json.dumps(mod.role_view(result['rows'], 'runtime'))
    record['annotations'] = [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 20, 20], 'area': 400, 'iscrowd': 0}]
    with pytest.raises(ValueError, match='immutable|identity'):
        mod.collect({'cal': [record]}, cfg(), tmp_path / 'valid', models())


def test_short_source_records_padding_and_unknown_fps_diagnostic_without_guess(tmp_path, monkeypatch):
    mod = module()
    from adaptive_vcm.codec import StandardCodec
    from adaptive_vcm.data import read_video
    from fractions import Fraction
    rgb = np.stack([np.full((128, 128, 3), t * 30, np.uint8) for t in range(5)])
    path = tmp_path / 'short.h264'
    path.write_bytes(StandardCodec('h264', 30, fps=Fraction(25)).roundtrip(rgb).data)
    monkeypatch.setattr(mod, '_video_timing', lambda *args: (None, 'unknown_fps', 'missing exact FPS'))
    sample = mod.source_sample({'id': 'short', 'path': str(path)}, 'ar', cfg('ar'))
    np.testing.assert_array_equal(sample['rgb'], read_video(str(path), 16, 128, 2))
    assert sample['source_fps'] is None and sample['duration'] is None
    assert sample['raw_sample_indices'] == [0, 2, 4] and sample['padding'] == 13
    assert sample['sample_indices'] == [0, 2, 4] + [4] * 13 and sample['padded']
    with pytest.raises(ValueError, match='timing'):
        mod.measure_source(sample, action_registry('ar', 'b'), 'h264', 30, models('ar'), cfg('ar'), tmp_path / 'bad')


def test_elementary_source_demuxer_default_is_not_qualified_source_timing(monkeypatch):
    mod = module()
    payload = json.dumps({'format': {'format_name': 'h264'},
        'streams': [{'avg_frame_rate': '25/1', 'r_frame_rate': '30000/1001'}],
        'frames': [{'pkt_duration_time': '.04'}] * 32}).encode()
    monkeypatch.setattr(mod.subprocess, 'check_output', lambda *args, **kwargs: payload)
    fps, status, error = mod._video_timing('raw.h264', 0, 32)
    assert status == 'unknown_fps' and error and fps is None


class TinyAnalyzer:
    name = 'r3d_18'
    def __init__(self):
        self.model = torch.nn.Linear(1, 2)
        self.mean, self.std = torch.zeros(3), torch.ones(3)
    def tensor(self, rgb):
        assert rgb.flags.writeable
        return torch.tensor([[float(rgb.mean())]])
    def logits(self, tensor):
        return self.model(tensor)
    def saliency(self, rgb):
        assert rgb.flags.writeable
        return np.zeros(rgb.shape[1:3], np.float32)


def test_production_adapter_hashes_weights_and_preprocess_and_owns_readonly_input():
    module()
    from adaptive_vcm.v31.measure_models import AnalyzerAdapter
    analyzer = TinyAnalyzer()
    adapter = AnalyzerAdapter(analyzer, 'ar')
    first = adapter.model_hash
    assert not analyzer.model.training and all(not p.requires_grad for p in analyzer.model.parameters())
    rgb = np.ones((16, 128, 128, 3), np.uint8)
    rgb.setflags(write=False)
    result = adapter.observe(rgb)
    assert len(result['logits']) == 2 and np.isclose(result['probabilities'].sum(), 1)
    with torch.no_grad():
        analyzer.model.weight.add_(1)
    second = AnalyzerAdapter(analyzer, 'ar').model_hash
    assert second != first
    analyzer.mean.add_(.5)
    assert AnalyzerAdapter(analyzer, 'ar').model_hash != second


def test_real_elementary_ffmpeg_source_does_not_guess_timing(tmp_path):
    mod = module()
    from adaptive_vcm.codec import StandardCodec
    from fractions import Fraction
    rgb = np.zeros((16, 128, 128, 3), np.uint8)
    path = tmp_path / 'ambiguous.h264'
    path.write_bytes(StandardCodec('h264', 30, fps=Fraction(30000, 1001)).roundtrip(rgb).data)
    sample = mod.source_sample({'id': 'ambiguous', 'path': str(path)}, 'ar', cfg('ar'))
    assert sample['source_fps'] is None and sample['duration'] is None and sample['timing_error']
    with pytest.raises(ValueError, match='timing'):
        mod.measure_source(sample, action_registry('ar', 'b'), 'h264', 30, models('ar'), cfg('ar'), tmp_path / 'store')
    assert not (tmp_path / 'store' / 'complete.json').exists()
    assert list((tmp_path / 'store' / 'diagnostics').glob('*.json'))


def test_typed_candidate_failure_is_audited_but_backend_or_anchor_failure_stops(tmp_path, monkeypatch):
    mod = module()
    assert hasattr(mod, 'CandidateEncodingError'), 'positive candidate failure classification missing'
    source, _ = action_source('ar')
    source['rgb'][:] = 0
    source.update(id='unit', source_sha256=hashlib.sha256(source['rgb'].tobytes()).hexdigest())
    old = mod.V31Codec.roundtrip
    def unit_failure(self, rgb, recipe):
        if recipe.width == 96:
            raise mod.CandidateEncodingError('positively identified unit failure ' + 'x' * 1500)
        return old(self, rgb, recipe)
    monkeypatch.setattr(mod.V31Codec, 'roundtrip', unit_failure)
    row = mod.measure_source(source, action_registry('ar', 'b'), 'h264', 35, models('ar'), cfg('ar'), tmp_path / 'unit')
    unavailable = [a for a in row['actions'] if not a['available']]
    assert len(unavailable) == 3 and row['counts']['unavailable'] == 3
    assert all(a['reason'] == 'candidate_encoding_failure' and len(a['error']) <= 1000 and
               a['total_bytes'] is None and a['recipe'] is None and a['stream_path'] is None for a in unavailable)
    import adaptive_vcm.v31.measure_store as store
    store.validate_row(row, tmp_path / 'unit')
    def backend_failure(*args):
        raise subprocess.CalledProcessError(1, ['ffmpeg'])
    monkeypatch.setattr(mod.V31Codec, 'roundtrip', backend_failure)
    with pytest.raises(subprocess.CalledProcessError):
        mod.measure_source({**source, 'id': 'backend'}, action_registry('ar', 'b'), 'h264', 35, models('ar'), cfg('ar'), tmp_path / 'backend')
    assert not list((tmp_path / 'backend').glob('conditions/*.json'))
    def anchor_failure(*args):
        raise mod.CandidateEncodingError('identity unavailable')
    monkeypatch.setattr(mod.V31Codec, 'roundtrip', anchor_failure)
    with pytest.raises(mod.CandidateEncodingError):
        mod.measure_source({**source, 'id': 'anchor'}, action_registry('ar', 'b'), 'h264', 35, models('ar'), cfg('ar'), tmp_path / 'anchor')


def test_immutable_source_support_artifact_rebuilds_owned_readonly_features(tmp_path):
    mod = module()
    assert hasattr(mod, 'load_source_artifact'), 'immutable source/support loader missing'
    source, _ = action_source('od')
    source['rgb'][:] = 0
    source.update(id='source-artifact', source_sha256=hashlib.sha256(source['rgb'].tobytes()).hexdigest())
    row = mod.measure_source(source, action_registry('od', 'b'), 'h264', 35, models(), cfg(), tmp_path)
    loaded = mod.load_source_artifact(tmp_path, row)
    assert all(not value.flags.writeable and value.flags.owndata for value in
               [loaded['rgb'], loaded['control_protection'], *[loaded['support'][key] for key in ('protection', 'motion', 'cuts')]])
    np.testing.assert_array_equal(loaded['rgb'], source['rgb'])
    assert loaded['control_protection'].shape == (320, 320)
    assert loaded['support']['protection'].shape == (1, 320, 320)
    assert row['source_artifact']['bytes'] == (tmp_path / row['source_artifact']['path']).stat().st_size
    before = (tmp_path / row['source_artifact']['path']).read_bytes()
    assert mod.measure_source(source, action_registry('od', 'b'), 'h264', 35, models(), cfg(), tmp_path) == row
    assert (tmp_path / row['source_artifact']['path']).read_bytes() == before
    with pytest.raises(ValueError):
        loaded['rgb'][:] = 5
    with (tmp_path / row['source_artifact']['path']).open('wb') as handle:
        np.savez_compressed(handle, rgb=source['rgb'].astype(np.uint16))
    tampered = copy.deepcopy(row)
    tampered['source_artifact']['sha256'] = hashlib.sha256((tmp_path / row['source_artifact']['path']).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='source|artifact|shape|dtype'):
        mod.load_source_artifact(tmp_path, tampered)


def test_collect_shares_exact_pixel_rendering_across_codec_conditions(tmp_path, monkeypatch):
    """Removing source/QP render reuse doubles authoritative CPU work."""
    import adaptive_vcm.v31.actions as actions
    from adaptive_vcm.codec import locate_ffmpeg
    calls = {'profile': 0, 'temporal': 0}
    for name, key in [('profile_candidates', 'profile'), ('conditional_profiles', 'temporal')]:
        original = getattr(actions, name)
        def counted(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(actions, name, counted)
    path = tmp_path / 'constant.mp4'
    subprocess.run([locate_ffmpeg(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '128x128', '-r', '25', '-i', 'pipe:0', '-c:v', 'libx264', '-bf', '0',
        '-threads', '2', '-pix_fmt', 'yuv420p', str(path)],
        input=np.zeros((40, 128, 128, 3), np.uint8).tobytes(), check=True)
    result = module().collect({'fit': [{'id': 'shared-render', 'path': str(path), 'label': 0}]},
                              cfg('ar'), tmp_path / 'store', models('ar'))
    assert len(result['rows']) == 8 and calls == {'profile': 4, 'temporal': 4}
    for row in result['rows']:
        assert all(a['recipe']['codec'] == row['codec'] for a in row['actions'] if a['available'])


def test_model_observation_rejects_inconsistent_logits_and_probabilities():
    from adaptive_vcm.v31.measure_models import ObservationCache
    model = Model('r3d_18', 'ar')
    model.observe = lambda rgb: {'logits': [0., 0.], 'probabilities': [.9, .1]}
    with pytest.raises(ValueError, match='logits|probabilities'):
        ObservationCache().observe(model, np.zeros((16, 128, 128, 3), np.uint8), 'ar')


def test_resumed_condition_binds_row_to_expected_identity_envelope(tmp_path):
    from adaptive_vcm.v31.measure_store import atomic_json, condition_path
    from adaptive_vcm.v31.protocol import canonical_hash
    mod = module()
    source, _ = action_source('od')
    source['rgb'][:] = 0
    source.update(id='bound-envelope', source_sha256=hashlib.sha256(source['rgb'].tobytes()).hexdigest())
    row = mod.measure_source(source, action_registry('od', 'b'), 'h264', 35, models(), cfg(), tmp_path)
    path = condition_path(tmp_path, source['id'], 'h264', 35)
    envelope = json.loads(path.read_text())
    envelope['row']['source_id'] = 'different-source'
    envelope['row_hash'] = canonical_hash(envelope['row'])
    atomic_json(path, envelope)
    with pytest.raises(ValueError, match='identity|envelope'):
        mod.measure_source(source, action_registry('od', 'b'), 'h264', 35, models(), cfg(), tmp_path)


@pytest.mark.parametrize('drift', ['duration', 'geometry'])
def test_measurement_recipe_cannot_drift_from_source_or_action_contract(tmp_path, drift):
    from adaptive_vcm.v31.measure_store import validate_row
    source, _ = action_source('od')
    source['rgb'][:] = 0
    source.update(id='recipe-bound', source_sha256=hashlib.sha256(source['rgb'].tobytes()).hexdigest())
    row = module().measure_source(source, action_registry('od', 'b'), 'h264', 35, models(), cfg(), tmp_path)
    if drift == 'duration':
        row['actions'][0]['recipe']['duration_num'] = 2
    else:
        row['actions'][0]['recipe'].update(width=192, height=192)
        row['actions'][0]['coded_shape'] = [192, 192]
    with pytest.raises(ValueError, match='recipe|duration|geometry|contract'):
        validate_row(row)
