import argparse
import importlib.util
import json
import hashlib
from collections import Counter

import pytest
import numpy as np
import torch


def module():
    assert importlib.util.find_spec('adaptive_vcm.train_portfolio') is not None, 'dense TRAIN collector is missing'
    import adaptive_vcm.train_portfolio as training
    return training


def test_dense_schedule_measures_every_codec_qp_for_every_source_once():
    training = module()
    plan = [{'id': f'clip-{i}', 'path': f'clip-{i}'} for i in range(8)]
    qps = [30, 35, 40, 45, 50]
    schedule = training.dense_schedule(plan, qps, 302201)
    assert len(schedule) == 80
    assert schedule == training.dense_schedule(plan, qps, 302201)
    assert schedule != training.dense_schedule(plan, qps, 302202)
    visits = Counter((i, codec, qp) for i, codec, qp in schedule)
    expected = {(i, c, q) for i in range(8) for c in ('h264', 'h265') for q in qps}
    assert set(visits) == expected
    assert set(visits.values()) == {1}
    # Keep a source's ten measurements together for teacher/source caching.
    assert all(len({i for i, _, _ in schedule[j:j + 10]}) == 1 for j in range(0, 80, 10))


def test_incomplete_dense_budget_rejected_before_teacher_collection(tmp_path):
    training = module()
    cfg = json.loads((training.ROOT / 'configs/v27_screen.json').read_text())
    config = tmp_path / 'cfg.json'
    config.write_text(json.dumps(cfg))
    args = argparse.Namespace(config=config, root=tmp_path, count=128,
                              measurements=512, seed=302201, out=tmp_path / 'train')
    with pytest.raises(ValueError, match='complete.*codec/QP'):
        training.train(args)
    assert not args.out.exists()


def test_dense_train_measures_distinct_groups_caches_source_teachers_and_restores_policy(tmp_path, monkeypatch, capsys):
    training = module()
    from adaptive_vcm.codec import Encoded
    from adaptive_vcm.data import partition
    from adaptive_vcm.rateaware import load_preprocessor
    from adaptive_vcm.utility_ranking import load_record_directory
    ids = [f'dense-fixture-{i}' for i in range(100) if partition(f'dense-fixture-{i}') == 'train'][:4]
    rng = np.random.default_rng(27)
    clips = {i: rng.integers(40, 180, (4, 24, 32, 3), np.uint8) for i in ids}
    monkeypatch.setattr(training, 'ar_plan', lambda *a: ([dict(id=i, path=i) for i in ids], {}))
    read_counts = Counter()
    def read(path, *args):
        read_counts[path] += 1
        return clips[path].copy()
    monkeypatch.setattr(training, 'read_video', read)
    saliency_calls = []
    class Teacher:
        def __init__(self, *a):
            pass
        def probabilities(self, clip):
            p = np.zeros(400)
            p[:2] = [.8, .2]
            return p
        def saliency(self, clip):
            saliency_calls.append(1)
            return np.ones(clip.shape[1:3], np.float32)
    class Codec:
        def __init__(self, codec, qp, *a):
            self.codec, self.qp = codec, qp
        def roundtrip(self, clip):
            size = 1000 + int(clip.size / 50) + int(clip.std()) + self.qp
            raw = clip.tobytes() + self.codec.encode() + bytes([self.qp])
            digest = hashlib.sha256(raw).digest()
            decoded = np.maximum(clip.astype(int) - 4, 0).astype(np.uint8)
            return Encoded(decoded, digest.ljust(size, b'x'), 0.)
    monkeypatch.setattr(training, 'ActionAnalyzer', Teacher)
    monkeypatch.setattr(training, 'StandardCodec', Codec)
    config = training.ROOT / 'configs/v27_screen.json'
    args = argparse.Namespace(config=config, root=tmp_path, count=4, measurements=40,
                              seed=302201, out=tmp_path / 'train')
    manifest = training.train(args)
    capsys.readouterr()
    rows = [json.loads(line) for line in (args.out / 'measurements.jsonl').read_text().splitlines()]
    assert len(rows) == 40 and read_counts == Counter({i: 1 for i in ids})
    assert len(saliency_calls) == 8
    assert len({(r['source_id'], r['codec'], r['qp']) for r in rows}) == 40
    assert all(len(r['actions']) == 42 for r in rows)
    assert all(len({r['source_sha256'] for r in rows if r['source_id'] == i}) == 1 for i in ids)
    assert len({r['actions'][0]['stream_sha256'] for r in rows}) == 40
    state = torch.load(args.out / 'preprocessor_last.pth', weights_only=True)
    policy = load_preprocessor(state, 'ar')
    assert len(policy.rank(np.asarray(rows[0]['context']), top_k=3)) == 3
    assert manifest['measurements'] == state['measurements'] == 40
    assert state['training_config'] == manifest['config']
    assert state['measurements_sha256'] == manifest['measurements_sha256']
    assert not manifest.get('steps')
    bundle = load_record_directory(args.out)
    assert bundle['context'].shape == (40, 41)
    assert tuple(bundle['source_ids']) == tuple(r['source_id'] for r in rows)
    from scripts.audit_v27 import audit_training
    audited, _, _ = audit_training(args.out)
    assert audited.rank(bundle['context'][0]) == policy.rank(bundle['context'][0])
    # Saved model utility must agree with actual byte subtraction and guard,
    # even when shapes and scaler provenance are otherwise valid.
    original_state = torch.load(args.out / 'preprocessor_last.pth', weights_only=True)
    changed_state = dict(original_state)
    changed_state['model'] = {k: v.clone() for k, v in original_state['model'].items()}
    changed_state['model']['memory_utility'][0, 0] += .00001
    torch.save(changed_state, args.out / 'preprocessor_last.pth')
    with pytest.raises(AssertionError):
        audit_training(args.out)
    torch.save(original_state, args.out / 'preprocessor_last.pth')
    # A digest alone does not prove a complete grid: rehashed duplicate groups
    # must still fail before a replay can fit on the incomplete measurements.
    changed = rows.copy()
    changed[1] = {**rows[0], 'measurement': 2}
    path = args.out / 'measurements.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in changed), encoding='utf-8')
    manifest['measurements_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out / 'training_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='complete.*codec/QP'):
        load_record_directory(args.out)
