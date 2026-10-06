"""Catch diluted targets, unbalanced groups, fit/calibration leakage and unsafe admission."""
import argparse
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_vcm import train_motion as training


def row(profile='identity', codec='h264', qp=40, marginal=0, control=1000):
    return dict(source_id='fixture', target_profile=profile, codec=codec, qp=qp,
                marginal_saved_bytes=marginal, controls_coded_bytes=control)


def test_group_balance_keeps_every_codec_qp_and_equal_positive_identity_mass():
    # Removing subgroup balancing should give the three identities triple mass.
    rows = [row(), row(), row(), row('motion_background_dc_100', marginal=50),
            row(codec='h265', qp=50)]
    weights = training.semantic_group_weights(rows)
    assert weights == pytest.approx([5/12, 5/12, 5/12, 5/4, 5/2])
    assert sum(weights) == pytest.approx(5)


def test_direct_strength_and_expert_supervision_rejects_wrong_positive_expert():
    # RGB-identical experts must still train the measured expert, not tie at MSE zero.
    output = torch.zeros(1, 3, 1, 2, 2, requires_grad=True)
    protection = torch.zeros(1, 1, 1, 2, 2)
    raw_alpha = torch.full_like(protection, .75, requires_grad=True)
    good = torch.tensor([.1, .7, .1, .1]).view(1, 4, 1, 1, 1).expand(1, 4, 1, 2, 2)
    bad = torch.tensor([.7, .1, .1, .1]).view(1, 4, 1, 1, 1).expand_as(good)
    target_row = row('motion_strong_gaussian_075', marginal=50)
    good_loss, good_parts = training.semantic_loss(output, output.detach(),
        {'raw_alpha': raw_alpha, 'alpha': raw_alpha, 'mixture': good}, protection, target_row, 1.)
    bad_loss, _ = training.semantic_loss(output, output.detach(),
        {'raw_alpha': raw_alpha, 'alpha': raw_alpha, 'mixture': bad}, protection, target_row, 1.)
    assert good_parts['rgb_loss'].item() == 0
    assert good_parts['alpha_loss'].item() == 0
    assert good_parts['expert_loss'].item() == pytest.approx(-math.log(.7))
    assert bad_loss > good_loss
    assert good_parts['utility_weight'].item() == pytest.approx(1.5)


def test_identity_gate_learns_zero_without_imposing_an_expert_choice():
    output = torch.zeros(1, 3, 1, 1, 1, requires_grad=True)
    raw = torch.full((1, 1, 1, 1, 1), .5, requires_grad=True)
    aux = {'raw_alpha': raw, 'alpha': raw, 'mixture': torch.full((1, 4, 1, 1, 1), .25)}
    loss, parts = training.semantic_loss(output, output.detach(), aux, torch.zeros_like(raw), row(), 1.)
    loss.backward()
    assert parts['expert_loss'].item() == 0
    assert parts['alpha_loss'].item() == pytest.approx(.25)
    assert raw.grad.item() > 0


def test_editable_normalization_ignores_protected_error_and_caps_actual_byte_utility():
    # Adding protected pixels must neither dilute editable error nor reward protected changes.
    target = torch.zeros(1, 3, 1, 1, 2)
    output = torch.tensor([.2, 1.]).view(1, 1, 1, 1, 2).expand_as(target).clone().requires_grad_()
    protection = torch.tensor([0., 1.]).view(1, 1, 1, 1, 2)
    raw = torch.full_like(protection, .4, requires_grad=True)
    aux = {'raw_alpha': raw, 'alpha': raw, 'mixture': torch.full((1, 4, 1, 1, 2), .25)}
    _, parts = training.semantic_loss(output, target, aux, protection,
                                     row('motion_mild_gaussian_040', marginal=900), 1.)
    assert parts['rgb_loss'].item() == pytest.approx(.04)
    assert parts['utility_weight'].item() == 2.
    all_protected_loss, _ = training.semantic_loss(output, target, aux, torch.ones_like(protection), row(), 1.)
    assert all_protected_loss.item() == 0
    all_protected_loss.backward()
    assert torch.isfinite(output.grad).all()


def test_train_calibration_split_is_deterministic_disjoint_and_source_level():
    ids = [f'source-{i}' for i in range(32)]
    fit, calibration = training.calibration_split(ids, 41)
    fit_again, calibration_again = training.calibration_split(list(reversed(ids)), 41)
    assert len(fit) == 24 and len(calibration) == 8
    assert set(fit).isdisjoint(calibration) and set(fit) | set(calibration) == set(ids)
    assert set(fit_again) == set(fit) and set(calibration_again) == set(calibration)
    with pytest.raises(ValueError, match='calibration'):
        training.calibration_split(['single-source'], 41)


def calibration_action(name, size, distance, decision=True):
    return dict(name=name, coded_bytes=size, distances=[distance], decisions=[decision])


def test_calibration_uses_nonworsening_extra_bytes_and_disables_empty_groups():
    # Unsafe, worsening, control ties and <1% anchor saving must not fit the threshold.
    rows = [dict(codec=c, qp=q, anchor_coded_bytes=1000, controls_coded_bytes=950,
                 source_id='calibration', proposals=[])
            for c in ('h264', 'h265') for q in (30, 35, 40, 45, 50)]
    rows[0]['proposals'] = [calibration_action('learned_motion_s050', 900, -.04),
        calibration_action('learned_motion_s100', 850, -.02),
        calibration_action('learned_motion_s150', 800, -.01)]
    rows[1]['proposals'] = [calibration_action('learned_motion_s050', 950, -.5),
        calibration_action('learned_motion_s100', 900, .001),
        calibration_action('learned_motion_s150', 800, -.5, decision=False)]
    policy = training.fit_admission_policy(rows, 'od')
    assert len(policy) == 10
    assert policy['h264/30'] == dict(threshold=pytest.approx(-.015), enabled=True,
                                   calibration_points=1, n_nonworsening=3)
    assert policy['h264/35'] == dict(threshold=0., enabled=False,
                                   calibration_points=1, n_nonworsening=0)


def test_calibration_refuses_partial_ar_teachers_and_incomplete_codec_qp_grid():
    rows = [dict(codec=c, qp=q, anchor_coded_bytes=1000, controls_coded_bytes=950,
                 source_id='calibration', proposals=[])
            for c in ('h264', 'h265') for q in (30, 35, 40, 45, 50)]
    rows[0]['proposals'] = [calibration_action('learned_motion_s050', 800, -.2)]
    assert not training.fit_admission_policy(rows, 'ar')['h264/30']['enabled']
    with pytest.raises(ValueError, match='complete.*calibration'):
        training.fit_admission_policy(rows[:-1], 'ar')


def test_v29_variant_mismatch_fails_before_collection():
    cfg = json.loads((training.ROOT/'configs/v28_screen.json').read_text())
    cfg.update(v29_variant='a', experiment='v29-c')
    with pytest.raises(ValueError, match='variant'):
        training._validate_config(cfg)


def test_v29_fitter_emits_positive_identity_head_diagnostics_and_keeps_last(tmp_path):
    torch.set_num_threads(2)
    source = np.random.default_rng(3).integers(30, 200, (1, 8, 8, 3), dtype=np.uint8)
    np.savez_compressed(tmp_path/'source.npz', source=source,
                        protection=np.zeros(source.shape[:3], np.float32),
                        motion=np.zeros(source.shape[:3], np.float32), cuts=np.array([True]))
    np.savez_compressed(tmp_path/'target.npz', target=np.full_like(source, 80))
    rows = [{**row(), 'source_cache':'source.npz', 'target_cache':None},
            {**row('motion_background_dc_100', marginal=50), 'source_cache':'source.npz',
             'target_cache':'target.npz'}]
    args = argparse.Namespace(out=tmp_path, task='od', width=6, seed=41, epochs=2, variant='a')
    model, diagnostics = training._fit(args, rows, 'cpu')
    assert model.schema == 'adaptive-vcm-semantic-v8'
    assert diagnostics['steps'] == 4
    logs = [json.loads(s) for s in (tmp_path/'train.jsonl').read_text().splitlines()]
    assert len(logs) == 4
    assert all({'rgb_loss','alpha_loss','expert_loss','alpha_mean','expert_entropy',
                'expert_margin','group_weight','utility_weight'} <= set(r) for r in logs)
    assert all(r['expert_loss'] == 0 for r in logs if r['target_profile'] == 'identity')
    assert all(r['expert_loss'] > 0 for r in logs if r['target_profile'] != 'identity')
    assert any(r['gradient_norm'] > 0 for r in logs)
    assert all({'positive_loss','identity_loss'} <= set(r) for r in diagnostics['epoch_logs'])


def test_c_collection_is_complete_but_optimizer_never_sees_calibration_sources(tmp_path, monkeypatch):
    # Fit/calibration overlap or fitting all collected records breaks checkpoint and log provenance.
    from adaptive_vcm.data import partition
    ids = [str(i) for i in range(200) if partition(f'coco2017/{i}') == 'train'][:4]
    monkeypatch.setattr(training, 'od_plan', lambda *a: ([dict(id=i, path='fixture') for i in ids], {}))
    source = np.random.default_rng(2).integers(20, 220, (1, 8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(training, 'read_image', lambda *a: (source.copy(), (1., 1., 0, 0)))
    # External teacher and encoder boundaries are replaced; collector/fitter/calibrator remain real.
    class UnknownDetector:
        def __init__(self, name, device):
            assert name == 'mobilenet'
        def predict(self, pixels):
            return dict(boxes=np.empty((0, 4)), scores=np.empty(0), labels=np.empty(0, np.int64))
    class Codec:
        def __init__(self, codec, qp, preset, fps):
            self.codec, self.qp = codec, qp
        def roundtrip(self, pixels):
            data = hashlib.sha256(pixels.tobytes()).digest()
            return SimpleNamespace(data=data, coded_bytes=len(data), decoded=pixels.copy(), seconds=0.)
    monkeypatch.setattr(training, 'DetectionAnalyzer', UnknownDetector)
    monkeypatch.setattr(training, 'StandardCodec', Codec)
    cfg = json.loads((training.ROOT/'configs/v28_screen.json').read_text())
    cfg.update(v29_variant='c', experiment='v29-c')
    config = tmp_path/'cfg.json'; config.write_text(json.dumps(cfg))
    args = argparse.Namespace(task='od', config=config, root=tmp_path, annotations=config,
                              count=4, epochs=1, width=6, seed=41, out=tmp_path/'out')
    manifest = training.train(args)
    checkpoint = torch.load(args.out/'preprocessor_last.pth', weights_only=True)
    fitted_ids = {json.loads(s)['source_id'] for s in (args.out/'train.jsonl').read_text().splitlines()}
    calibration = [json.loads(s) for s in (args.out/'calibration_measurements.jsonl').read_text().splitlines()]
    assert manifest['measurements'] == checkpoint['measurements'] == 40
    assert manifest['steps'] == checkpoint['steps'] == 30
    assert len(checkpoint['fit_ids']) == 3 and len(checkpoint['calibration_ids']) == 1
    assert fitted_ids == set(checkpoint['fit_ids'])
    assert fitted_ids.isdisjoint(checkpoint['calibration_ids'])
    assert {r['source_id'] for r in calibration} == set(checkpoint['calibration_ids'])
    assert len(calibration) == 10
    assert all([a['name'] for a in r['proposals']] ==
               ['learned_motion_s050', 'learned_motion_s100', 'learned_motion_s150'] for r in calibration)
    assert all(not policy['enabled'] for policy in checkpoint['admission_policy'].values())
    assert checkpoint['calibration_measurements_sha256'] == hashlib.sha256(
        (args.out/'calibration_measurements.jsonl').read_bytes()).hexdigest()
    from adaptive_vcm.rateaware import load_preprocessor
    loaded = load_preprocessor(checkpoint, 'od')
    assert loaded.schema == 'adaptive-vcm-semantic-v8'
    assert loaded.admission_policy == checkpoint['admission_policy']
