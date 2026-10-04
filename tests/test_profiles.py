import argparse
import json

import numpy as np
import pytest
import torch

from adaptive_vcm.codec import StandardCodec, locate_ffmpeg
from adaptive_vcm.evaluate import ROOT
from adaptive_vcm.profiles import ProfilePreprocessor, feasible_profile_target
from adaptive_vcm.rateaware import load_preprocessor
from adaptive_vcm.selection import ar_guard, relative_guard
from adaptive_vcm.train_rateaware import pixels


@pytest.mark.parametrize('anchor', [[.1, .9], [.49, .51]])
@pytest.mark.parametrize('strict', [True, False])
def test_anchor_is_admissible_despite_confident_source_disagreement(anchor, strict):
    source, anchor = np.array([.95, .05]), np.array(anchor)
    cfg = {'ar_confidence': .6, 'ar_require_anchor_decision': strict}
    assert relative_guard('ar', [source], [anchor], [anchor.copy()], cfg) == ((0.,), (True,))
    assert ar_guard(source, anchor, anchor.copy()) == (0., True)
    if strict:
        assert relative_guard('ar', [source], [anchor], [source], cfg)[1] == (False,)


def test_anchor_reference_retains_class_protection_and_relative_kl_limit():
    source, anchor = np.array([.9, .1]), np.array([.1, .9])
    cfg = {'ar_confidence': .6, 'ar_require_anchor_decision': True}
    better = np.array([.2, .8])
    distance, decision = relative_guard('ar', [source], [anchor], [better], cfg)
    assert decision == (True,) and distance[0] < 0
    worse = np.array([.001, .999])
    distance, decision = relative_guard('ar', [source], [anchor], [worse], cfg)
    assert decision == (True,) and distance[0] > .1


def test_profile_target_never_trades_task_violation_for_bits():
    measurements = [dict(profile=n, coded_bytes=b, distances=[d], decisions=[ok])
                    for n, b, d, ok in [('identity', 1000, 0, True), ('unsafe', 1, .2, True),
                                       ('flip', 10, 0, False), ('safe', 700, .01, True),
                                       ('tie', 700, .01, True)]]
    assert feasible_profile_target(measurements, slack=.1) == 3
    assert feasible_profile_target(measurements[:3], slack=.1) == 0


def test_profile_identity_and_semantic_core_are_exact_after_rounding():
    torch.set_num_threads(2)
    x = torch.rand(1, 3, 4, 17, 23)
    mask = torch.zeros(1, 1, 4, 17, 23)
    mask[..., 5:9, 7:11] = 1
    model = ProfilePreprocessor(8)
    q, c = torch.tensor([50]), torch.tensor([1])
    bank = model.filter_bank(x, q)
    for index in range(len(model.profiles)):
        output = model.render_profile(x, q, c, mask, [index], bank=bank)
        assert output.min() >= 0 and output.max() <= 1
        torch.testing.assert_close(output[..., 5:9, 7:11], x[..., 5:9, 7:11], atol=0, rtol=0)
        if index == 0:
            torch.testing.assert_close(output, x, atol=0, rtol=0)
        else:
            assert np.any(pixels(output) != pixels(x))
    state = dict(schema=model.schema, task='ar', steps=1, width=8, train_ids_sha256='fixture', model=model.state_dict())
    assert isinstance(load_preprocessor(state, 'ar'), ProfilePreprocessor)
    with pytest.raises(ValueError):
        load_preprocessor(state, 'od')


def test_temporal_profiles_reset_at_scene_cuts():
    model = ProfilePreprocessor(8)
    x = torch.zeros(1, 3, 2, 8, 8)
    x[:, :, 1] = 1
    for i in (7, 8):
        output, aux = model.render_profile(x, torch.tensor([50]), torch.tensor([0]), None, [i], return_aux=True)
        assert aux['reuse'][:, :, 1].count_nonzero() == 0
        torch.testing.assert_close(output, x, rtol=0, atol=1e-6)


@pytest.mark.codec
@pytest.mark.parametrize('codec', ['h264', 'h265'])
@pytest.mark.parametrize('qp', [40, 45, 50])
def test_ar_profile_has_real_high_qp_byte_leverage(codec, qp):
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    # Compressibility fixture only; no claim about real action accuracy.
    frame = np.random.default_rng(24).integers(0, 256, (64, 96, 3), np.uint8)
    frame[24:40, 40:56] = [240, 32, 16]
    clip = np.repeat(frame[None], 8, axis=0)
    x = torch.from_numpy(clip.copy()).float().permute(3, 0, 1, 2)[None] / 255
    mask = torch.zeros(1, 1, 8, 64, 96)
    mask[..., 20:44, 36:60] = 1
    model = ProfilePreprocessor(8)
    encoder = StandardCodec(codec, qp)
    edited = pixels(model.render_profile(x, torch.tensor([qp]), torch.tensor([int(codec == 'h265')]), mask, [6]))
    np.testing.assert_array_equal(edited[:, 20:44, 36:60], clip[:, 20:44, 36:60])
    assert encoder.roundtrip(edited).coded_bytes <= .99 * encoder.roundtrip(clip).coded_bytes


@pytest.mark.codec
def test_profile_training_and_component_evaluation_with_real_codecs(tmp_path, monkeypatch):
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    import adaptive_vcm.evaluate as evaluation
    import adaptive_vcm.train_profiles as training
    from test_pipeline import TinyAction
    torch.set_num_threads(2)
    frame = np.random.default_rng(24).integers(60, 190, (32, 48, 3), np.uint8)
    clip = np.repeat(frame[None], 4, axis=0)
    class DisagreeingAction(TinyAction):
        # Deliberately reproduce the V23 conflict: confident source class 0,
        # but compressed anchor class 1. Exact-anchor candidates must pass.
        def probabilities(self, frames):
            return np.array([.9, .05, .05] if np.array_equal(frames, clip) else [.05, .9, .05])
    plan = [{'id': str(i), 'path': 'fixture', 'label': 0} for i in (1, 2)]
    cfg = json.loads((ROOT / 'configs/v24_screen.json').read_text())
    cfg['qps'] = [30, 40, 50]
    cfg['ar_candidates'] = ['identity', 'protected_mild']
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(cfg))
    for module in (evaluation, training):
        monkeypatch.setattr(module, 'ar_plan', lambda *a: (plan, {}))
        monkeypatch.setattr(module, 'read_video', lambda *a: clip.copy())
        monkeypatch.setattr(module, 'ActionAnalyzer', DisagreeingAction)
    args = argparse.Namespace(task='ar', root=tmp_path, config=config, count=2, steps=3,
                              width=8, seed=24, lr=.001, out=tmp_path / 'train')
    training.train(args)
    logs = [json.loads(s) for s in (args.out / 'train.jsonl').read_text().splitlines()]
    assert any(r['gradient_norm'] > 0 for r in logs)
    assert any(r['target_profile'] != 'identity' for r in logs)
    for r in logs:
        assert r['profiles'][0]['distances'] == [0., 0.]
        assert r['profiles'][0]['decisions'] == [True, True]
    args = argparse.Namespace(task='ar', root=tmp_path, config=config, annotations=None, count=2,
                              split='dev', codecs=['h264', 'h265'], bootstrap=2,
                              checkpoint=args.out / 'preprocessor_last.pth', save_streams=False,
                              ablate_learned=True, out=tmp_path / 'eval')
    summary = evaluation.run(args)
    assert not summary['target_confirmed']
    for row in map(json.loads, (args.out / 'selection_audit.jsonl').read_text().splitlines()):
        c = next(c for c in row['candidates'] if c['name'] == 'learned_profile')
        assert c['profile'] in {p[0] for p in ProfilePreprocessor.profiles}
        assert c['preserves_decision'] == [True, True]
