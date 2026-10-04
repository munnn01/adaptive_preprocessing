import argparse
import itertools
import json

import numpy as np
import pytest
import torch

from adaptive_vcm.codec import locate_ffmpeg, StandardCodec
from adaptive_vcm.evaluate import ROOT
from adaptive_vcm.rateaware import RateAwarePreprocessor, semantic_protection, load_preprocessor
from adaptive_vcm.selection import relative_guard
from adaptive_vcm.train_rateaware import measured_objective, spsa_gradient


@pytest.mark.parametrize('task', ['ar', 'od'])
def test_rateaware_active_high_qp_edit_protected_core_and_gradients(task):
    torch.manual_seed(24)
    torch.set_num_threads(2)
    model = RateAwarePreprocessor(8, task)
    source = torch.rand(2, 3, 2, 17, 23)
    mask = torch.zeros(2, 1, 2, 17, 23)
    mask[..., 5:9, 6:10] = 1
    output, aux = model(source, torch.tensor([30, 50]), torch.tensor([0, 1]), mask, return_aux=True)
    assert output.shape == source.shape and output.min() >= 0 and output.max() <= 1
    torch.testing.assert_close(output[..., 5:9, 6:10], source[..., 5:9, 6:10], atol=0, rtol=0)
    assert aux['strength'][1] > aux['strength'][0] > .3
    assert (output[1] - source[1]).abs().mean() > 1 / 255
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_semantic_core_is_exact_without_all_time_motion_tube():
    score = np.array([[0., .5, .96, 1.]], np.float32)
    np.testing.assert_allclose(semantic_protection(score), [[0., .25, 1., 1.]])
    model = RateAwarePreprocessor(8, 'ar')
    source = torch.rand(1, 3, 2, 1, 4)
    mask = torch.ones(1, 1, 2, 1, 4)
    torch.testing.assert_close(model(source, torch.tensor([50]), torch.tensor([0]), mask), source, atol=0, rtol=0)


def test_temporal_reuse_resets_on_scene_cut():
    model = RateAwarePreprocessor(8, 'ar')
    source = torch.zeros(1, 3, 2, 8, 8)
    source[:, :, 1] = 1
    result, aux = model(source, torch.tensor([50]), torch.tensor([1]), return_aux=True)
    assert aux['reuse'][:, :, 1].count_nonzero() == 0
    torch.testing.assert_close(result, source, atol=1e-6, rtol=0)


def test_spsa_direction_average_recovers_quadratic_gradient():
    x = torch.tensor([[.2, -.3]])
    matrix = torch.tensor([[3., 0.], [0., 5.]])
    def objective(value):
        return float((value @ matrix * value).sum())
    gradients = []
    for direction in itertools.product((-1., 1.), repeat=2):
        d = torch.tensor([direction])
        gradients.append(spsa_gradient(objective(x + .1*d), objective(x - .1*d), d, .1))
    torch.testing.assert_close(torch.stack(gradients).mean(0), 2 * x @ matrix)


def test_measured_relative_rate_signal_survives_high_qp_small_streams():
    high, _ = measured_objective(800, 1000, (.01,), (True,), .03, 2.)
    low, _ = measured_objective(80000, 100000, (.01,), (True,), .03, 2.)
    assert high == pytest.approx(low) and high < 0
    failed, violation = measured_objective(800, 1000, (.01,), (False,), .03, 2.)
    assert failed > 0 and violation == 1
    # Very large byte savings must never outweigh a violated task constraint.
    unsafe, violation = measured_objective(1, 100000, (.031,), (True,), .03, 2.)
    assert unsafe > 0 and violation > 0


def test_v23_anchor_decision_guard_checks_low_confidence_without_labels():
    cfg = json.loads((ROOT / 'configs/v23_screen.json').read_text())
    source, anchor, trial = [np.array(v) for v in ([.45, .4, .15], [.45, .4, .15], [.4, .45, .15])]
    _, decisions = relative_guard('ar', [source], [anchor], [trial], cfg)
    assert decisions == (False,)


def test_checkpoint_task_mismatch_fails_closed():
    model = RateAwarePreprocessor(8, 'od')
    state = {'schema': model.schema, 'width': 8, 'task': 'od', 'model': model.state_dict(),
             'steps': 3, 'train_ids_sha256': 'fixture'}
    assert isinstance(load_preprocessor(state, 'od'), RateAwarePreprocessor)
    with pytest.raises(ValueError, match='mismatch'):
        load_preprocessor(state, 'ar')


@pytest.mark.codec
@pytest.mark.parametrize('codec', ['h264', 'h265'])
@pytest.mark.parametrize('qp', [40, 45, 50])
def test_high_qp_learned_initialization_reduces_actual_background_bytes(codec, qp):
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    torch.set_num_threads(2)
    source = np.random.default_rng(23).integers(0, 256, (1, 64, 96, 3), np.uint8)
    source[:, 24:40, 40:56] = [240, 32, 16]
    x = torch.from_numpy(source.copy()).float().permute(3, 0, 1, 2)[None] / 255
    mask = torch.zeros(1, 1, 1, 64, 96)
    mask[..., 20:44, 36:60] = 1
    model = RateAwarePreprocessor(8, 'od')
    with torch.no_grad():
        filtered = model(x, torch.tensor([qp]), torch.tensor([int(codec == 'h265')]), mask)
    pixels = filtered[0].permute(1, 2, 3, 0).mul(255).round().byte().numpy()
    np.testing.assert_array_equal(pixels[:, 20:44, 36:60], source[:, 20:44, 36:60])
    encoder = StandardCodec(codec, qp)
    anchor, edited = encoder.roundtrip(source), encoder.roundtrip(pixels)
    assert edited.coded_bytes <= anchor.coded_bytes * .99


@pytest.mark.codec
@pytest.mark.parametrize('task', ['ar', 'od'])
def test_actual_byte_training_and_isolated_component_evaluation(task, tmp_path, monkeypatch):
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    import adaptive_vcm.evaluate as evaluation
    import adaptive_vcm.train_rateaware as training
    from test_pipeline import TinyAction
    from test_od_pipeline import TinyDetector

    source = np.random.default_rng(23).integers(50, 200, (4 if task == 'ar' else 1, 32, 48, 3), np.uint8)
    cfg = json.loads((ROOT / 'configs/v23_screen.json').read_text())
    cfg['qps'] = [30, 40, 50]
    cfg['ar_candidates'] = ['identity', 'protected_mild']
    cfg['od_candidates'] = ['identity', 'background4']
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(cfg))
    plan = [{'id': str(i), 'path': 'fixture', 'label': 0, 'image_id': i} for i in (1, 2)]
    meta = {'images': [{'id': i} for i in (1, 2)], 'categories': [{'id': 1, 'name': 'object'}],
            'annotations': [{'id': i, 'image_id': i, 'category_id': 1, 'bbox': [8, 8, 8, 8],
                             'area': 64, 'iscrowd': 0} for i in (1, 2)]}
    ann = tmp_path / 'annotations.json'
    ann.write_text(json.dumps(meta))
    for module in (evaluation, training):
        monkeypatch.setattr(module, 'ar_plan', lambda *args: (plan, {}))
        monkeypatch.setattr(module, 'od_plan', lambda *args: (plan, meta))
        monkeypatch.setattr(module, 'read_video', lambda *args: source.copy())
        monkeypatch.setattr(module, 'read_image', lambda *args: (source.copy(), (1, 1, 0, 0)))
        monkeypatch.setattr(module, 'ActionAnalyzer', TinyAction)
        monkeypatch.setattr(module, 'DetectionAnalyzer', TinyDetector)
    args = argparse.Namespace(config=config, task=task, root=tmp_path, annotations=ann,
                              count=2, steps=3, width=8, seed=23, lr=.001, epsilon=.5,
                              dual_lr=.05, probe_every=2, out=tmp_path / 'train')
    manifest = training.train(args)
    assert 'actual_bytes' in manifest['objective'] and len(manifest['qp_sampling']) == 6
    logs = [json.loads(s) for s in (args.out / 'train.jsonl').read_text().splitlines()]
    assert any(r['changed_pixel_fraction'] > 0 for r in logs)
    assert any(r['gradient_norm'] > 0 for r in logs)
    args = argparse.Namespace(config=config, task=task, root=tmp_path, annotations=ann,
                              count=2, split='dev', codecs=['h264', 'h265'], bootstrap=2,
                              checkpoint=args.out / 'preprocessor_last.pth', save_streams=False,
                              ablate_learned=True, out=tmp_path / 'eval')
    summary = evaluation.run(args)
    assert not summary['target_confirmed']
    for codec in args.codecs:
        components = [json.loads(s) for s in (args.out / f'{codec}_components.jsonl').read_text().splitlines()]
        assert len(components) == 2 * 3 * 3
        assert set(summary['component_results'][codec]) == {'controls', 'learned_guarded', 'learned_raw'}
        assert set(summary['high_qp_diagnostics'][codec]) == {'40', '50'}
        assert any(r['candidate'] == 'learned_rateaware' for r in components if r['arm'] == 'learned_raw')

