import hashlib

import numpy as np
import pytest
import torch

from adaptive_vcm.codec import StandardCodec, locate_ffmpeg
from adaptive_vcm.evaluate import choose_stream
from adaptive_vcm.rateaware import load_preprocessor
from adaptive_vcm.stabilized_bank import ACTION_NAMES
from adaptive_vcm.utility_ranking import UtilityRankPreprocessor


def test_actual_high_qp_utility_streams_keep_proposals_and_group_comparators_auditable(tmp_path):
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    torch.set_num_threads(2)
    rng = np.random.default_rng(26)
    frame = rng.integers(40, 160, (24, 32, 3), np.uint8)
    clip = np.stack([np.clip(frame.astype(int) + shift, 0, 255).astype(np.uint8)
                     for shift in (0, 6, -6, 0)])
    mask = np.full((24, 32), .5, np.float32)
    class Teacher:
        def probabilities(self, frames):
            p = np.zeros(400)
            p[:2] = [.8, .2]
            return p
    teachers = [Teacher(), Teacher()]
    source = [t.probabilities(clip) for t in teachers]
    model = UtilityRankPreprocessor(ACTION_NAMES)
    model.mix = .5
    model.prior[:, [17, 20, 23]] = .02
    checkpoint = tmp_path / 'utility.pth'
    state = model.checkpoint_state()
    state.update(fit_method='sourceblocked_train_cv_ridge', measurements=4,
                 train_ids_sha256='fixture')
    torch.save(state, checkpoint)
    fitted = load_preprocessor(torch.load(checkpoint, weights_only=True), 'ar')
    with pytest.raises(ValueError, match='provenance'):
        load_preprocessor(model.checkpoint_state(), 'ar')
    cfg = dict(ar_candidates=['identity', 'area96', 'protected_mild'], rank_top_k=3,
               ar_confidence=.6, ar_require_anchor_decision=True, ar_kl_slack=.1, min_savings=.01)
    for codec in ('h264', 'h265'):
        bundle = choose_stream(clip, mask, 'ar', StandardCodec(codec, 50), cfg, teachers,
                               source, fitted, learned_mask=mask, components=True)
        anchor, primary, name, audit, arms = bundle
        proposals = audit[0]['learned_order']
        assert proposals == [18, 21, 24]
        context = np.asarray(audit[0]['ranking_context'], np.float64)
        assert hashlib.sha256(context.tobytes()).hexdigest() == audit[0]['ranking_context_sha256']
        assert np.isclose(np.expm1(context[-1]), 8*anchor.coded_bytes/(4*24*32))
        assert sum(row['proposed_by_learned'] for row in audit) == 3
        assert sum(row['proposed_by_group_static'] for row in audit) == 3
        assert primary.coded_bytes <= arms['controls'][0].coded_bytes
        assert arms['bank_oracle'][0].coded_bytes <= primary.coded_bytes
        assert arms['bank_oracle'][0].coded_bytes <= arms['v25_bank_oracle'][0].coded_bytes
        assert 'group_static_adaptive' in arms
        if name.startswith('learned_rank__'):
            assert next(r['action_index'] for r in audit if r['name'] == name) in proposals
