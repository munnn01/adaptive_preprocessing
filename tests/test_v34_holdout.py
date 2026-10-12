"""Catch held-out contamination, changing samples across shards, and unpaired CIs."""
import importlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('adaptive_vcm.v34.holdout') is not None, 'independent holdout evaluator missing'
    return importlib.import_module('adaptive_vcm.v34.holdout')


def fixture(n=20):
    records = [dict(youtube_id=f'video{i:06d}', path=f'/data/video{i:06d}') for i in range(n)]
    annotations = {r['youtube_id']: dict(label=i % 2, time_start=0, time_end=10) for i, r in enumerate(records)}
    return records, annotations


def test_official_validation_excludes_train_video_even_if_also_in_validation():
    h = module()
    rows = [dict(youtube_id='abcdefghij0', label='class A', time_start='0', time_end='10'),
            dict(youtube_id='abcdefghij1', label='class B', time_start='1', time_end='11')]
    assert h.validation_annotations(rows, {'abcdefghij0'}, ['class A', 'class B']) == {
        'abcdefghij1': dict(label=1, time_start=1, time_end=11)}


def test_selection_excludes_seen_youtube_ids_and_unknown_videos():
    h = module(); records, annotations = fixture()
    records.append(dict(youtube_id='unknown0000', path='/data/unknown'))
    selected, audit = h.select_records(records, annotations, {'youtube_ids': ['video000000', 'video000001']},
                                       count=10, seed=42, shards=1, shard=0)
    assert len(selected) == 10
    assert not {'video000000', 'video000001', 'unknown0000'} & {r['youtube_id'] for r in selected}
    assert audit['eligible_count'] == 18


def test_shards_are_same_global_sample_disjoint_and_order_independent():
    h = module(); records, annotations = fixture()
    whole, manifest = h.select_records(records, annotations, {}, count=12, seed=23, shards=1, shard=0)
    parts = []
    for shard in range(3):
        part, audit = h.select_records(records[::-1], annotations, {}, count=12, seed=23, shards=3, shard=shard)
        assert len(part) == 4
        assert audit['global_ids'] == manifest['global_ids']
        parts.extend(r['youtube_id'] for r in part)
    assert len(parts) == len(set(parts)) == 12
    assert set(parts) == {r['youtube_id'] for r in whole}


def test_insufficient_eligible_sources_fails_instead_of_shrinking_sample():
    h = module(); records, annotations = fixture(3)
    with pytest.raises(ValueError, match='insufficient'):
        h.select_records(records, annotations, {}, count=4, seed=42, shards=1, shard=0)


def test_identical_pixels_to_history_are_rejected_before_evaluation():
    h = module()
    rgb = np.zeros((16, 128, 128, 3), dtype=np.uint8)
    import hashlib
    sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    with pytest.raises(ValueError, match='historical pixels'):
        h.audit_pixels(rgb, {'pixel_sha256': [sha]}, set())


def test_pairing_mismatch_is_not_silently_bootstrapped():
    h = module()
    rows = [dict(id='a', codec='h264', qp=30, arm='anchor', bpp=1, correct={'swin3d_t': 1}),
            dict(id='b', codec='h264', qp=30, arm='selected_303401', bpp=.9, correct={'swin3d_t': 1})]
    with pytest.raises(ValueError, match='paired'):
        h.paired_arrays(rows, 'h264', 'selected_303401', 'swin3d_t', [30])


def test_annotation_hash_mismatch_fails_before_source_or_checkpoint_loading(tmp_path):
    h = module()
    import json
    from types import SimpleNamespace
    (tmp_path/'protocol.json').write_text(json.dumps(dict(settings={}, design={}, annotation_csv_sha256='a'*64, exclusions_sha256='b'*64)))
    (tmp_path/'val.csv').write_text('youtube_id,label,time_start,time_end\nvideo000000,0,0,10\n')
    (tmp_path/'exclusions.json').write_text('{}')
    args = SimpleNamespace(protocol=str(tmp_path/'protocol.json'), annotations=str(tmp_path/'val.csv'), exclusions=str(tmp_path/'exclusions.json'))
    with pytest.raises(ValueError, match='annotation'):
        h.run(args)


def test_raw_frame_reader_preserves_sixteen_distinct_center_stride_frames(tmp_path):
    h = module()
    from PIL import Image
    for i in range(40):
        Image.fromarray(np.full((8, 8, 3), i, np.uint8)).save(tmp_path / f'img_{i+1:05d}.jpg')
    rgb, recipe = h.read_frames(tmp_path, frames=16, stride=2, size=128)
    assert rgb.shape == (16, 128, 128, 3)
    assert recipe['frame_indices'] == list(range(4, 36, 2))
    assert len(np.unique(rgb[:, 0, 0, 0])) == 16


def test_short_frame_folder_is_not_padded_to_satisfy_holdout(tmp_path):
    h = module()
    from PIL import Image
    for i in range(2): Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / f'img_{i+1:05d}.jpg')
    with pytest.raises(ValueError, match='short'):
        h.read_frames(tmp_path, frames=16, stride=2, size=128)


def test_inventory_excludes_short_provider_sources_before_hash_selection(tmp_path):
    h = module()
    annotations = {i: dict(label=0, time_start=0, time_end=10)
                   for i in ['video000000', 'video000001', 'video000002']}
    for yt, count in [('video000000', 31), ('video000001', 32), ('video000002', 40), ('unknown0000', 40)]:
        folder = tmp_path/'validation'/yt; folder.mkdir(parents=True)
        for j in range(count): (folder/f'img_{j+1:05d}.jpg').write_bytes(b'count-only fixture')
    records, audit = h.frame_inventory(tmp_path, annotations, minimum_frames=32)
    selected, plan = h.select_records(records, annotations, {}, count=2, seed=42, shards=1, shard=0)
    assert {r['youtube_id'] for r in selected} == {'video000001', 'video000002'}
    assert audit['official_available'] == 3 and audit['eligible_sources'] == 2
    assert audit['short_sources'] == {'video000000': 31}
    assert audit['minimum_frames'] == 32 and plan['eligible_count'] == 2
    with pytest.raises(ValueError, match='insufficient'):
        h.select_records(records, annotations, {}, count=3, seed=42, shards=1, shard=0)


def merge_fixture(tmp_path):
    import json
    h = module()
    ids = [f'video{i:06d}' for i in range(4)]
    cfg = dict(seeds=[303401, 303402, 303403], codecs=['h264', 'h265'], qps=[30, 35, 40, 45, 50])
    protocol = dict(settings=cfg, design=dict(count=4, shards=2, selection_seed=42, bootstrap_seed=43,
        bootstrap_draws=5000, primary_seed=303401), checkpoints={str(s): dict(sha256=str(s), state_sha256='state'+str(s)) for s in cfg['seeds']})
    frozen = tmp_path/'frozen.json'; frozen.write_text(json.dumps(protocol))
    paths = []
    arms = ['anchor', 'fixed']+[f'{a}_{s}' for s in cfg['seeds'] for a in ['forced', 'calibrated', 'selected']]
    for shard in range(2):
        path = tmp_path/f'shard{shard}'; path.mkdir(); paths.append(path)
        plan = dict(global_ids=ids, global_ids_sha256=h.digest(ids), available_ids_sha256='available',
            count=4, seed=42, shards=2, shard=shard)
        cohorts = {'ar/official-k400-val/'+i: dict(youtube_id=i, pixel_sha256='pixels'+i) for i in ids[shard::2]}
        status = dict(status='complete', parameter_updates=False, training_performed=False,
            checkpoint_sha256=[str(s) for s in cfg['seeds']], shard=shard, sources=2,
            global_ids_sha256=h.digest(ids), controller_state_before=['state'+str(s) for s in cfg['seeds']],
            controller_state_after=['state'+str(s) for s in cfg['seeds']])
        rows = [dict(id=i, task='ar', split='test', codec=c, qp=q, arm=a, bpp=1.,
                     correct={'swin3d_t': 1, 'r3d_18': 1})
                for i in cohorts for c in cfg['codecs'] for q in cfg['qps'] for a in arms]
        for name, value in [('protocol.json', protocol), ('selection.json', plan), ('complete.json', status),
                            ('cohorts.json', cohorts), ('test_rows.json', rows)]:
            (path/name).write_text(json.dumps(value))
    return paths, frozen


@pytest.mark.parametrize('tamper', ['rows', 'cohort', 'status_shard', 'status_count', 'status_selection', 'state'])
def test_merge_rejects_unselected_rows_wrong_shard_or_unfrozen_state(tmp_path, tamper):
    import json
    h = module(); paths, frozen = merge_fixture(tmp_path)
    if tamper == 'rows':
        target = paths[0]/'test_rows.json'; data = json.loads(target.read_text())
        for row in data: row['id'] = row['id'].replace('video', 'other')
    elif tamper == 'cohort':
        target = paths[0]/'cohorts.json'; data = json.loads(target.read_text())
        data = {k.replace('000000', '000001'): {**v, 'youtube_id': v['youtube_id'].replace('000000', '000001')} for k,v in data.items()}
    else:
        target = paths[0]/'complete.json'; data = json.loads(target.read_text())
        if tamper == 'status_shard': data['shard'] = 1
        if tamper == 'status_count': data['sources'] = 3
        if tamper == 'status_selection': data['global_ids_sha256'] = 'wrong'
        if tamper == 'state': data['controller_state_before'] = data['controller_state_after'] = ['other']*3
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='membership|shard|state'):
        h.merge(paths, tmp_path/'merged', protocol_path=frozen)


def test_merge_rejects_changing_preregistered_bootstrap_count(tmp_path):
    h = module(); paths, frozen = merge_fixture(tmp_path)
    with pytest.raises(ValueError, match='bootstrap'):
        h.merge(paths, tmp_path/'merged', protocol_path=frozen, draws=1)


def test_complete_frozen_shards_bind_every_row_to_selected_video(tmp_path):
    import json
    h = module(); paths, frozen = merge_fixture(tmp_path)
    rows = h.validate_shards(paths[::-1], json.loads(frozen.read_text()))
    assert len(rows) == 4*2*5*11
    ids = {r['id'] for r in rows}
    assert ids == {'ar/official-k400-val/video000000', 'ar/official-k400-val/video000001',
                   'ar/official-k400-val/video000002', 'ar/official-k400-val/video000003'}


def test_paired_bootstrap_real_five_qp_binary_curves(tmp_path):
    h = module()
    # Paired equal accuracy at every QP; uniform 20% rate saving has known BD.
    rates = np.tile([1., .8, .6, .4, .2], (40, 1))
    correct = np.arange(40)[:, None] < np.array([35, 30, 25, 20, 15])
    ci = h.paired_bootstrap([(rates, correct), (rates*.8, correct)], draws=200, seed=43, alpha=.0125)
    assert ci['finite_draws'] >= 198
    assert ci['confidence'] == .975
    assert ci['lo'] == pytest.approx(-20, abs=1e-6) and ci['hi'] == pytest.approx(-20, abs=1e-6)


@pytest.mark.codec
def test_frozen_holdout_runner_real_codecs_never_exposes_label_to_guard(tmp_path, monkeypatch):
    h = module()
    import json
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from torchvision.models.video import Swin3D_T_Weights
    from adaptive_vcm.v34.protocol import settings, digest, ACTIONS
    from adaptive_vcm.v34.model import Controller
    from adaptive_vcm.v34.checkpoint import save_checkpoint
    from adaptive_vcm.v32vc.checkpoint import state_hash
    from adaptive_vcm.v34 import tasks, evaluation

    class Analyzer:
        def __init__(self, name):
            self.name = name; self.model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        def probabilities(self, rgb): return np.array([1., 0.])

    class Heads:
        def __init__(self, device, experiment, cfg):
            self.device = device; self.experiment = experiment; self.action = Analyzer('swin3d_t')
        def hashes(self): return {'external_test_heads': 'fixed'}
        def detections(self, rgb):
            return [dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), labels=np.zeros(0, int)) for _ in rgb]
        def source(self, r):
            from adaptive_vcm.v34.tracking import track_frames
            assert 'label' not in r.item
            r.source_predictions = {'od': self.detections(r.rgb), 'ar': np.array([1., 0.])}
            r.source_predictions['ot'] = track_frames(r.source_predictions['od'])
            r.mask = [True, False, True]; r.source_detector_seconds = 0.; r.diagnostic_teacher_seconds = 0.
        def costs(self, rgb, r, source_only=False):
            assert source_only and 'label' not in r.item
            return [0., None, 0.]

    monkeypatch.setattr(tasks, 'Heads', Heads)
    monkeypatch.setattr(evaluation, 'ActionAnalyzer', lambda name, device: Analyzer(name))
    frames = tmp_path/'frames'/'video000000'; frames.mkdir(parents=True)
    for i in range(40):
        Image.fromarray(np.full((8, 8, 3), i, np.uint8)).save(frames/f'img_{i+1:05d}.jpg')
    cfg = settings(Path(__file__).resolve().parents[1]/'configs/v34_arot_r1.json')
    cfg['qps'] = [35]
    checkpoints = tmp_path/'weights'; checkpoints.mkdir()
    checks = {}
    for seed in cfg['seeds']:
        model = Controller(len(ACTIONS), conditions=9)
        path = checkpoints/f'controller_{seed}.pth'
        save_checkpoint(path, model, dict(seed=seed, settings=cfg, source_commit='a'*40,
            task_models={'external_test_heads': 'fixed'}, calibration=dict(risk=[[0]*3]*len(ACTIONS), rate=[0]*len(ACTIONS))))
        import hashlib
        checks[str(seed)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), state_sha256=state_hash(model.state_dict()))
    protocol = dict(settings=cfg, design=dict(count=1, shards=1, selection_seed=42, minimum_source_frames=32), checkpoints=checks,
                    controller_source_commit='a'*40, categories_sha256=digest(Swin3D_T_Weights.KINETICS400_V1.meta['categories']))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'exclusions.json').write_text('{}')
    (tmp_path/'val.csv').write_text('youtube_id,label,time_start,time_end\nvideo000000,0,0,10\n')
    protocol['annotation_csv_sha256'] = hashlib.sha256((tmp_path/'val.csv').read_bytes()).hexdigest()
    protocol['exclusions_sha256'] = hashlib.sha256((tmp_path/'exclusions.json').read_bytes()).hexdigest()
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    args = SimpleNamespace(root=str(tmp_path/'frames'), protocol=str(tmp_path/'protocol.json'),
        annotations=str(tmp_path/'val.csv'), exclusions=str(tmp_path/'exclusions.json'),
        checkpoints=str(checkpoints), out=str(tmp_path/'results'), shard=0)
    h.run(args)
    complete = json.loads((tmp_path/'results'/'complete.json').read_text())
    assert complete['parameter_updates'] is False and complete['training_performed'] is False
    assert complete['controller_state_before'] == complete['controller_state_after']
    rows = json.loads((tmp_path/'results'/'test_rows.json').read_text())
    assert len(rows) == 22 and {r['codec'] for r in rows} == {'h264', 'h265'}
    assert all(r['bytes'] > r['elementary_bytes'] > 0 for r in rows)
