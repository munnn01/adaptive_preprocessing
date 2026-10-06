"""Catches missing V29 provenance and calibration bypass without changing controls."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from test_motion_selection import state, ProbeCodec, ProbeModel, ConstantTeacher
from adaptive_vcm.data import fingerprint
from adaptive_vcm.evaluate import choose_stream
from adaptive_vcm.rateaware import load_preprocessor


def semantic_state(variant='a'):
    value = state()
    value['schema'] = 'adaptive-vcm-semantic-v8'
    value['variant'] = variant
    value['training_config']['v29_variant'] = variant
    value['fit_ids'] = list(value['train_ids'])
    value['calibration_ids'] = []
    if variant == 'c':
        value['fit_ids'] = value['train_ids'][:1]
        value['calibration_ids'] = value['train_ids'][1:]
        value['steps'] = 10
        value['fit_ids_sha256'] = fingerprint(value['fit_ids'])
        value['calibration_ids_sha256'] = fingerprint(value['calibration_ids'])
        value['calibration_measurements_sha256'] = 'c' * 64
        value['admission_policy'] = {f'{codec}/{qp}': {
            'threshold': -.005, 'enabled': True, 'calibration_points': 1,
            'n_nonworsening': 1}
            for codec in ('h264', 'h265') for qp in (30, 35, 40, 45, 50)}
        value['calibration_measurements'] = 10
        value['admission_policy_sha256'] = hashlib.sha256(json.dumps(
            value['admission_policy'], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return value


@pytest.mark.parametrize('variant', ['a', 'b', 'c'])
def test_semantic_checkpoint_reconstructs_the_declared_renderer(variant):
    model = load_preprocessor(semantic_state(variant), 'ar')
    assert model.schema == 'adaptive-vcm-semantic-v8'
    assert model.variant == variant


@pytest.mark.parametrize('fault', ['overlap', 'missing_group', 'positive_slack', 'unverified', 'wrong_steps'])
def test_calibration_cannot_be_loaded_with_leakage_or_relaxed_guards(fault):
    value = semantic_state('c')
    if fault == 'overlap':
        value['calibration_ids'] = value['fit_ids']
    elif fault == 'missing_group':
        value['admission_policy'].pop('h265/50')
    elif fault == 'positive_slack':
        value['admission_policy']['h264/50']['threshold'] = .1
    elif fault == 'unverified':
        value.pop('calibration_measurements_sha256')
    else:
        value['steps'] = 20
    with pytest.raises(ValueError):
        load_preprocessor(value, 'ar')


@pytest.mark.parametrize('fault', ['policy_tampering', 'partial_calibration', 'wrong_fraction'])
def test_c_checkpoint_rejects_changed_policy_or_incomplete_calibration_provenance(fault):
    value = semantic_state('c')
    if fault == 'policy_tampering':
        value['admission_policy']['h264/50']['threshold'] = -.001
    elif fault == 'partial_calibration':
        value['calibration_measurements'] = 9
    else:
        # All source IDs remain valid, unique and disjoint, but one of eight
        # calibration sources violates the preregistered quarter.
        from adaptive_vcm.data import partition
        ids = [str(i) for i in range(100) if partition(str(i)) == 'train'][:8]
        value.update(train_ids=ids, train_ids_sha256=fingerprint(ids), train_count=8,
                     measurements=80, fit_ids=ids[:7], fit_ids_sha256=fingerprint(ids[:7]),
                     calibration_ids=ids[7:], calibration_ids_sha256=fingerprint(ids[7:]),
                     steps=70, train_source_sha256={i:'a'*64 for i in ids})
    with pytest.raises(ValueError, match='calibration|policy'):
        load_preprocessor(value, 'ar')


def test_calibration_rejects_a_learned_stream_but_keeps_the_original_control(monkeypatch):
    import adaptive_vcm.motion_selection as selection
    from adaptive_vcm.motion_learned import PROFILE_NAMES
    from adaptive_vcm.preprocessing import Candidate
    source = np.full((2, 16, 24, 3), 200, np.uint8)
    config = semantic_state('c')['training_config']
    config['ar_candidates'] = ['identity', 'blur20']
    monkeypatch.setattr(selection, 'make_candidates', lambda *args:
                        [Candidate('identity', source), Candidate('blur20', np.full_like(source, 120))])
    model = ProbeModel(PROFILE_NAMES)
    model.schema, model.variant = 'adaptive-vcm-semantic-v8', 'c'
    model.admission_policy = {'h264/50': {'threshold': -.005, 'enabled': True}}
    teachers = [ConstantTeacher(), ConstantTeacher()]
    bundle = choose_stream(source, np.zeros((16, 24), np.float32), 'ar', ProbeCodec(), config,
                           teachers, [teacher.probabilities(source) for teacher in teachers], model)
    assert bundle[2] == 'blur20'
    assert bundle[1].coded_bytes == 1300
    learned = [row for row in bundle[3] if row['name'].startswith('learned_motion_')]
    assert len(learned) == 3
    assert all(row['policy_admitted'] is False for row in learned)
    assert all(row['preserves_decision'] == [True, True] for row in learned)
