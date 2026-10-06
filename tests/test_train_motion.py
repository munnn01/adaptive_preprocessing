"""Breaks caught: omitted grid points, unsafe targets, unsafe static credit,
held-out provenance, source recaching, and a fitter that never updates pixels.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json

import numpy as np
import pytest
import torch


def training_module():
    assert importlib.util.find_spec('adaptive_vcm.train_motion') is not None, 'V28 TRAIN collector is missing'
    import adaptive_vcm.train_motion as training
    return training


def observation(name, size, distance=0., decision=True):
    return dict(name=name, coded_bytes=size, distances=[distance], decisions=[decision])


def test_complete_schedule_caches_each_source_across_all_ten_groups():
    training = training_module()
    plan = [dict(id=f'clip-{i}') for i in range(3)]
    schedule = training.dense_schedule(plan, [30, 35, 40, 45, 50], 41)
    assert len(schedule) == 30
    visits = Counter(schedule)
    assert set(visits) == {(i, c, q) for i in range(3) for c in ('h264', 'h265') for q in (30, 35, 40, 45, 50)}
    assert set(visits.values()) == {1}
    assert all(len({i for i, _, _ in schedule[j:j + 10]}) == 1 for j in (0, 10, 20))
    assert schedule == training.dense_schedule(plan, [30, 35, 40, 45, 50], 41)
    with pytest.raises(ValueError, match='complete.*codec/QP'):
        training.dense_schedule(plan, [30, 40, 50], 41)


@pytest.mark.parametrize('profiles,want', [
    ([observation('extra', 940)], 'extra'),
    ([observation('tie', 950)], 'identity'),
    ([observation('unsafe', 500, .11)], 'identity'),
    ([observation('decision_flip', 500, decision=False)], 'identity'),
    ([observation('unknown', 500, None)], 'identity'),
    ([observation('tiny_saving', 995)], 'identity'),
    ([observation('float_bytes', 900.5)], 'identity'),
])
def test_targets_require_guarded_additional_actual_integer_byte_savings(profiles, want):
    training = training_module()
    controls = [observation('identity', 1000), observation('control', 950)]
    assert training.choose_training_target(controls, profiles, slack=.1, min_savings=.01) == want


def test_static_three_profiles_cover_complementary_actual_guarded_bytes():
    training = training_module()
    # A wins 300+300; B repeats A; C adds 200 on an uncovered source.
    # Unsafe D's apparent 900-byte win must earn no credit.
    names = ('A', 'B', 'C', 'D')
    records = []
    for source, sizes in enumerate(((700, 750, 1000, 100), (700, 750, 1000, 100), (1000, 1000, 800, 100))):
        records.append(dict(source_id=str(source), codec='h264', qp=40, anchor_coded_bytes=1000,
                            controls_coded_bytes=1000, slack=.1, min_savings=.01,
                            profiles=[observation(n, b, .2 if n == 'D' else 0.) for n, b in zip(names, sizes)]))
    assert training.fit_static_orders(records, names, k=3) == {'h264/40': ['A', 'C', 'B']}
    # Distinct codec/QP portfolios; byte utility is not normalized per source.
    records += [dict(source_id='other', codec='h265', qp=50, anchor_coded_bytes=1000,
                     controls_coded_bytes=950, slack=.1, min_savings=.01,
                     profiles=[observation('A', 950), observation('B', 945), observation('C', 900), observation('D', 700)])]
    assert training.fit_static_orders(records, names, k=3)['h265/50'] == ['D', 'A', 'B']


def train_ids(task, count):
    from adaptive_vcm.data import partition
    ids = [str(i) if task == 'od' else f'motion-fixture-{i}' for i in range(200)]
    return [i for i in ids if partition(f'coco2017/{i}' if task == 'od' else i) == 'train'][:count]


def test_grid_audit_rejects_duplicate_points_and_uses_coco_partition_prefix():
    training = training_module()
    names = ('A', 'B', 'C')
    ids = train_ids('od', 1)
    rows = [dict(task='od', source_id=ids[0], codec=c, qp=q, profiles=[observation(n, 1000) for n in names],
                 controls=[observation('identity', 1000)], controls_selected='identity',
                 anchor_coded_bytes=1000, controls_coded_bytes=1000, target_profile='identity',
                 target_coded_bytes=1000, marginal_saved_bytes=0, slack=.03, min_savings=.01)
            for c in ('h264', 'h265') for q in (30, 35, 40, 45, 50)]
    training.validate_training_records(rows, ids, 'od', names)
    with pytest.raises(ValueError, match='complete.*codec/QP'):
        training.validate_training_records([rows[0], rows[0], *rows[2:]], ids, 'od', names)
    from adaptive_vcm.data import partition
    heldout = next(str(i) for i in range(200) if partition(f'coco2017/{i}') != 'train')
    with pytest.raises(ValueError, match='TRAIN'):
        training.validate_training_records(rows, [heldout], 'od', names)


def test_static_order_refuses_partial_ar_teacher_guards():
    training = training_module()
    row = dict(task='ar', source_id='fixture', codec='h264', qp=40, anchor_coded_bytes=1000,
               controls_coded_bytes=1000, slack=.1, min_savings=.01,
               profiles=[observation('missing_teacher', 100),
                         dict(name='safe', coded_bytes=900, distances=[0., 0.], decisions=[True, True]),
                         dict(name='less_saving', coded_bytes=950, distances=[0., 0.], decisions=[True, True])])
    assert training.fit_static_orders([row], ('missing_teacher', 'safe', 'less_saving'))['h264/40'][0] == 'safe'


def test_grid_audit_rejects_relabelled_targets_and_inconsistent_control_winner():
    training = training_module()
    ids = train_ids('od', 1)
    names = ('A', 'B', 'C')
    rows = [dict(task='od', source_id=ids[0], codec=c, qp=q,
                 profiles=[observation('A', 800), observation('B', 900), observation('C', 1000)],
                 controls=[observation('identity', 1000), observation('control', 950)],
                 controls_selected='control', anchor_coded_bytes=1000, controls_coded_bytes=950,
                 target_profile='A', target_coded_bytes=800, marginal_saved_bytes=150, slack=.03, min_savings=.01)
            for c in ('h264', 'h265') for q in (30, 35, 40, 45, 50)]
    training.validate_training_records(rows, ids, 'od', names)
    with pytest.raises(ValueError, match='target'):
        training.validate_training_records([{**rows[0], 'target_profile': 'B'}, *rows[1:]], ids, 'od', names)
    with pytest.raises(ValueError, match='control'):
        training.validate_training_records([{**rows[0], 'controls_coded_bytes': 900}, *rows[1:]], ids, 'od', names)


def test_incomplete_grid_is_rejected_before_output_or_teacher_initialization(tmp_path):
    training = training_module()
    config = json.loads((training.ROOT / 'configs/v27_screen.json').read_text())
    config['qps'] = [30, 40, 50]
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps(config))
    args = argparse.Namespace(task='ar', config=path, root=tmp_path, count=1, epochs=1,
                              width=6, seed=41, out=tmp_path / 'out')
    with pytest.raises(ValueError, match='complete.*codec/QP'):
        training.train(args)
    assert not args.out.exists()


def test_invalid_renderer_width_fails_before_any_collection(tmp_path, monkeypatch):
    training = training_module()
    def forbidden_plan(*args):
        raise RuntimeError('source planning must not start for invalid model width')
    monkeypatch.setattr(training, 'ar_plan', forbidden_plan)
    args = argparse.Namespace(task='ar', config=training.ROOT / 'configs/v28_screen.json',
                              root=tmp_path, count=1, epochs=1, width=3, seed=41, out=tmp_path / 'out')
    with pytest.raises(ValueError, match='budget|width'):
        training.train(args)
    assert not args.out.exists()


def test_non_v28_config_fails_before_source_collection(tmp_path, monkeypatch):
    training = training_module()
    def forbidden_plan(*args):
        raise RuntimeError('source planning must not start for incompatible recipe')
    monkeypatch.setattr(training, 'ar_plan', forbidden_plan)
    args = argparse.Namespace(task='ar', config=training.ROOT / 'configs/v27_screen.json',
                              root=tmp_path, count=1, epochs=1, width=6, seed=41, out=tmp_path / 'out')
    with pytest.raises(ValueError, match='V28'):
        training.train(args)
    assert not args.out.exists()


@pytest.mark.codec
@pytest.mark.parametrize('task', ['ar', 'od'])
def test_real_codec_train_caches_source_and_fits_final_spatial_pixels(tmp_path, monkeypatch, capsys, task):
    training = training_module()
    from adaptive_vcm.codec import locate_ffmpeg
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    from adaptive_vcm.motion_learned import MotionAwarePreprocessor, PROFILE_NAMES
    ids = train_ids(task, 1)
    rng = np.random.default_rng(41)
    clip = rng.integers(20, 220, (4 if task == 'ar' else 1, 48, 64, 3), dtype=np.uint8)
    reads = Counter()
    def read(*args):
        reads['source'] += 1
        return clip.copy() if task == 'ar' else (clip.copy(), (1., 1., 0, 0))
    monkeypatch.setattr(training, 'read_video' if task == 'ar' else 'read_image', read)
    monkeypatch.setattr(training, 'ar_plan' if task == 'ar' else 'od_plan',
                        lambda *args: ([dict(id=ids[0], path='fixture', image_id=int(ids[0]) if task == 'od' else 0)], {}))
    calls = Counter()
    class FrozenAction:
        def __init__(self, name, device):
            assert name in ('r3d_18', 'mc3_18')
        def probabilities(self, pixels):
            calls['predictions'] += 1
            return np.array([.8, .2])
        def saliency(self, pixels):
            calls['saliency'] += 1
            return np.zeros(pixels.shape[1:3], dtype=np.float32)
    class FrozenDetector:
        def __init__(self, name, device):
            assert name == 'mobilenet'
        def predict(self, pixels):
            calls['predictions'] += 1
            return dict(boxes=np.array([[17., 17., 23., 23.]]), scores=np.array([.9]), labels=np.array([1]))
    monkeypatch.setattr(training, 'ActionAnalyzer', FrozenAction)
    monkeypatch.setattr(training, 'DetectionAnalyzer', FrozenDetector)
    config = json.loads((training.ROOT / 'configs/v28_screen.json').read_text())
    config.update(preset='ultrafast', frames=len(clip), ar_size=64, od_size=64)
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps(config))
    args = argparse.Namespace(task=task, config=path, root=tmp_path, annotations=path,
                              count=1, epochs=1, width=6, seed=41, out=tmp_path / 'out')
    torch.manual_seed(41)
    fresh = MotionAwarePreprocessor(6, task).state_dict()
    manifest = training.train(args)
    capsys.readouterr()
    rows = [json.loads(line) for line in (args.out / 'measurements.jsonl').read_text().splitlines()]
    state = torch.load(args.out / 'preprocessor_last.pth', weights_only=True)
    assert reads['source'] == 1
    assert calls['saliency'] == (2 if task == 'ar' else 0)
    assert len(rows) == manifest['measurements'] == state['measurements'] == 10
    assert state['epochs'] == 1 and state['steps'] == 10 and state['train_count'] == 1
    assert state['train_ids'] == ids and state['training_config'] == config
    assert state['code'] == manifest['code']
    assert state['train_source_sha256'] == manifest['train_source_sha256'] == {ids[0]: hashlib.sha256(clip.tobytes()).hexdigest()}
    assert state['measurements_sha256'] == hashlib.sha256((args.out / 'measurements.jsonl').read_bytes()).hexdigest()
    assert state['profile_names'] == list(PROFILE_NAMES)
    assert len(state['static_orders']) == 10 and all(len(v) == len(set(v)) == 3 for v in state['static_orders'].values())
    assert len(list((args.out / 'cache/sources').glob('*.npz'))) == 1
    with np.load(args.out / rows[0]['source_cache'], allow_pickle=False) as cached:
        np.testing.assert_array_equal(cached['source'], clip)
        assert cached['protection'].shape == cached['motion'].shape == clip.shape[:3]
        assert cached['cuts'].dtype == np.bool_
    assert len({r['source_sha256'] for r in rows}) == 1
    assert all(len(r['profiles']) == 12 and len(r['controls']) == (9 if task == 'ar' else 5) for r in rows)
    assert all(type(a['coded_bytes']) is int and len(a['stream_sha256']) == 64 for r in rows for a in r['profiles'] + r['controls'])
    assert all(r['target_profile'] == 'identity' or (r['target_coded_bytes'] < r['controls_coded_bytes']
               and r['target_coded_bytes'] <= r['anchor_coded_bytes'] * .99) for r in rows)
    assert any(not torch.equal(value, fresh[key]) for key, value in state['model'].items())
    logs = [json.loads(line) for line in (args.out / 'train.jsonl').read_text().splitlines()]
    assert len(logs) == 10 and any(r['gradient_norm'] > 0 for r in logs)
    assert {r['qp_weight'] for r in logs if r['qp'] < 40} == {1.}
    assert {r['qp_weight'] for r in logs if r['qp'] >= 40} == {2.}
    assert all(r['nonzero_edit_pixels'] >= 0 for r in logs)


def test_fitter_reports_each_epoch_loss_gradient_and_identity_target_edits(tmp_path):
    training = training_module()
    torch.set_num_threads(2)
    source = np.random.default_rng(3).integers(30, 200, (1, 16, 24, 3), dtype=np.uint8)
    cache = tmp_path / 'source.npz'
    np.savez_compressed(cache, source=source, protection=np.zeros(source.shape[:3], np.float32),
                        motion=np.zeros(source.shape[:3], np.float32), cuts=np.array([True]))
    args = argparse.Namespace(out=tmp_path, task='od', width=6, seed=41, epochs=2)
    records = [dict(source_cache='source.npz', target_cache=None, source_id='fixture', codec='h264', qp=45,
                    target_profile='identity')]
    _, diagnostics = training._fit(args, records, 'cpu')
    assert len(diagnostics['epoch_logs']) == 2
    for epoch in diagnostics['epoch_logs']:
        assert epoch['mean_loss'] > 0 and epoch['mean_gradient_norm'] > 0
        assert epoch['positive_targets'] == 0 and epoch['identity_targets'] == 1
        assert 0 < epoch['output_edit_fraction'] <= 1 and epoch['target_edit_fraction'] == 0
    logs = [json.loads(line) for line in (tmp_path / 'train.jsonl').read_text().splitlines()]
    assert all(0 < r['output_edit_fraction'] <= 1 and r['target_edit_fraction'] == 0 for r in logs)


@pytest.mark.codec
def test_unknown_od_all_profile_targets_are_identity_with_finite_same_stream_guards(tmp_path, monkeypatch, capsys):
    training = training_module()
    from adaptive_vcm.codec import locate_ffmpeg
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    ids = train_ids('od', 1)
    monkeypatch.setattr(training, 'od_plan', lambda *a: ([dict(id=ids[0], path='fixture')], {}))
    clip = np.random.default_rng(91).integers(20, 220, (1, 32, 40, 3), dtype=np.uint8)
    monkeypatch.setattr(training, 'read_image', lambda *a: (clip.copy(), (1., 1., 0, 0)))
    class UnknownDetector:
        def __init__(self, *args):
            pass
        def predict(self, pixels):
            return dict(boxes=np.empty((0, 4)), scores=np.empty(0), labels=np.empty(0, dtype=np.int64))
    monkeypatch.setattr(training, 'DetectionAnalyzer', UnknownDetector)
    cfg = json.loads((training.ROOT / 'configs/v28_screen.json').read_text())
    cfg['preset'] = 'ultrafast'
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps(cfg))
    args = argparse.Namespace(task='od', config=path, root=tmp_path, annotations=path,
                              count=1, epochs=1, width=6, seed=41, out=tmp_path / 'out')
    manifest = training.train(args)
    capsys.readouterr()
    rows = [json.loads(line) for line in (args.out / 'measurements.jsonl').read_text().splitlines()]
    assert manifest['positive_targets'] == 0 and manifest['identity_targets'] == 10
    assert not list((args.out / 'cache/targets').glob('*.npz'))
    for row in rows:
        assert row['target_profile'] == 'identity' and row['target_cache'] is None
        assert row['controls_selected'] == 'identity'
        assert all(a['identity_stream'] and a['distances'] == [0.] and a['decisions'] == [True]
                   for a in row['profiles'] + row['controls'])
