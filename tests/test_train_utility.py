import argparse
import hashlib
import json

import numpy as np
import torch


def test_train_collects_guarded_controls_and_fits_only_sourceblocked_train(tmp_path, monkeypatch):
    import adaptive_vcm.train_utility as training
    from adaptive_vcm.codec import Encoded
    from adaptive_vcm.data import partition
    from adaptive_vcm.selection import Observation, select
    from adaptive_vcm.stabilized_bank import ACTION_NAMES
    from adaptive_vcm.utility_ranking import load_record_directory, load_utility_preprocessor
    from adaptive_vcm.rateaware import load_preprocessor
    ids = [f'fixture-{i}' for i in range(30) if partition(f'fixture-{i}') == 'train'][:6]
    rng = np.random.default_rng(26)
    clips = {identifier: rng.integers(40, 180, (4, 24, 32, 3), np.uint8) for identifier in ids}
    monkeypatch.setattr(training, 'ar_plan', lambda *a: ([{'id': i, 'path': i} for i in ids], {}))
    monkeypatch.setattr(training, 'read_video', lambda path, *a: clips[path].copy())
    class Teacher:
        def __init__(self, *a):
            pass
        def probabilities(self, clip):
            p = np.zeros(400)
            p[:2] = [.8, .2]
            return p
        def saliency(self, clip):
            return np.ones(clip.shape[1:3], np.float32)
    class Codec:
        def __init__(self, codec, qp, *a):
            self.codec, self.qp = codec, qp
        def roundtrip(self, clip):
            size = 1000 + int(clip.size / 50) + int(clip.std())
            digest = hashlib.sha256(clip.tobytes()).digest()
            return Encoded(clip.copy(), digest.ljust(size, b'x'), 0.)
    monkeypatch.setattr(training, 'ActionAnalyzer', Teacher)
    monkeypatch.setattr(training, 'StandardCodec', Codec)
    cfg = json.loads((training.ROOT / 'configs/v26_screen.json').read_text())
    cfg['qps'] = [30, 40, 50]
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(cfg))
    args = argparse.Namespace(config=config, root=tmp_path, count=6, measurements=6,
                              seed=302101, out=tmp_path / 'train')
    manifest = training.train(args)
    rows = list(map(json.loads, (args.out / 'measurements.jsonl').read_text().splitlines()))
    arrays = np.load(args.out / 'train_records.npz')
    assert arrays['context'].shape == (6, 41)
    assert manifest['fit_diagnostics']['utility_target_scope'] == 'marginal_saving_beyond_guarded_controls'
    assert len({(r['codec'], r['qp']) for r in rows}) == 6
    assert all(tuple(a['name'] for a in r['actions']) == ACTION_NAMES for r in rows)
    assert len({r['source_id'] for r in rows}) == 6
    for row in rows:
        controls = row['controls']
        winner = select([Observation(c['name'], c['coded_bytes'], tuple(c['distances']), tuple(c['decisions']))
                         for c in controls], cfg['ar_kl_slack'], cfg['min_savings'])
        assert controls[winner]['coded_bytes'] == row['controls_coded_bytes']
        assert np.isclose(row['baseline_log_rate'], np.log(row['controls_coded_bytes']/row['actions'][0]['coded_bytes']))
    state = torch.load(args.out / 'preprocessor_last.pth', weights_only=True)
    model = load_utility_preprocessor(state)
    assert load_preprocessor(state, 'ar').action_names == ACTION_NAMES
    replay = load_record_directory(args.out)
    np.testing.assert_allclose(replay['context'], arrays['context'])
    np.testing.assert_allclose(replay['baseline_log_rate'], arrays['baseline_log_rate'])
    assert len(model.rank(arrays['context'][0])) == 3
    assert state['training_config'] == cfg
    assert not manifest.get('steps')  # closed-form fitting has no SGD-step claim
    # Debug repairs must reuse exactly the measured TRAIN streams, without
    # starting a new teacher/codec collection or consulting DEV.
    def unexpected_collection(*a):
        raise AssertionError('cached fitting must not construct an AR teacher')
    monkeypatch.setattr(training, 'ActionAnalyzer', unexpected_collection)
    cached = args.out
    args.reuse_records, args.out = cached, tmp_path / 'replay'
    replay_manifest = training.train(args)
    assert (args.out / 'measurements.jsonl').read_bytes() == (cached / 'measurements.jsonl').read_bytes()
    assert replay_manifest['measurements_sha256'] == manifest['measurements_sha256']
    assert replay_manifest['fit_diagnostics']['label_precision'] == 'actual_integer_bytes_v1'
    restored = load_preprocessor(torch.load(args.out / 'preprocessor_last.pth', weights_only=True), 'ar')
    assert restored.rank(arrays['context'][0]) == model.rank(arrays['context'][0])
