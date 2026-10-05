import hashlib

import numpy as np
import pytest
import torch

from adaptive_vcm.codec import Encoded
from adaptive_vcm.evaluate import choose_stream
from adaptive_vcm.stabilized_bank import ACTION_NAMES
from adaptive_vcm.task_bank import ACTION_NAMES as V25_NAMES


@pytest.mark.parametrize('names', [ACTION_NAMES, V25_NAMES])
def test_portfolio_dispatch_keeps_anchor_only_input_and_per_point_prior_credit(names):
    calls = []
    proposed = [18, 21, 24] if names == ACTION_NAMES else [1, 2, 3]
    class Codec:
        codec, qp = 'h265', 50
        def roundtrip(self, clip):
            calls.append(clip.copy())
            digest = hashlib.sha256(clip.tobytes()).digest()
            return Encoded(clip.copy(), digest.ljust(2000 + int(clip.std()), b'x'), 0.)
    class Teacher:
        def probabilities(self, clip):
            p = np.zeros(400)
            p[:2] = [.8, .2]
            return p
    class Portfolio(torch.nn.Module):
        schema = 'adaptive-vcm-portfolio-v6'
        action_names = names
        learned_mix = 1.
        static_action_order = [1, 2, 3]
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
        def forward(self, *a, **kw):
            pytest.fail('portfolio checkpoints must use the ranked stream selector')
        def rank(self, context, top_k):
            assert len(calls) == 1, 'no nonidentity encode may inform proposal'
            assert top_k == 3 and context.shape == (41,)
            return proposed
        def group_static_action_order(self, context, top_k):
            return [6, 8, 10]
        def proposal_details(self, context, top_k):
            return dict(prior_only=True, action_indices=proposed)
    clip = np.random.default_rng(27).integers(40, 180, (4, 24, 32, 3), np.uint8)
    mask = np.full((24, 32), .5, np.float32)
    teachers = [Teacher(), Teacher()]
    source = [t.probabilities(clip) for t in teachers]
    cfg = dict(ar_candidates=['identity', 'area96', 'protected_mild'], rank_top_k=3,
        ar_confidence=.6, ar_require_anchor_decision=True, ar_kl_slack=.1, min_savings=.01)
    _, _, _, audit, arms = choose_stream(clip, mask, 'ar', Codec(), cfg, teachers,
        source, Portfolio(), learned_mask=mask, components=True)
    assert audit[0]['learned_order'] == proposed
    assert audit[0]['proposal_details']['prior_only'] is True
    assert all(r['name'].startswith('trained_prior__') for r in audit if r['action_index'] is not None)
    assert sum(r['proposed_by_learned'] for r in audit) == 3
    assert 'group_static_adaptive' in arms
    assert ('v26_bank_oracle' in arms) == (names == ACTION_NAMES)


def test_actual_codecs_v6_anchor_bank_and_checkpoint_keep_primary_subset_auditable(tmp_path):
    from adaptive_vcm.anchor_bank import ACTION_NAMES as ANCHOR_NAMES
    from adaptive_vcm.codec import StandardCodec, locate_ffmpeg, reference_bpp
    from adaptive_vcm.portfolio_ranking import PortfolioRankPreprocessor, _fit_memory
    from adaptive_vcm.utility_ranking import build_utility_context
    from adaptive_vcm.data import partition
    from adaptive_vcm.rateaware import load_preprocessor
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    torch.set_num_threads(2)
    frame = np.random.default_rng(27).integers(40, 160, (24, 32, 3), np.uint8)
    clip = np.stack([np.clip(frame.astype(int) + i, 0, 255).astype(np.uint8) for i in (0, 6, -6, 0)])
    mask = np.full((24, 32), .5, np.float32)
    class Teacher:
        def probabilities(self, frames):
            p = np.zeros(400)
            p[:2] = [.8, .2]
            return p
    teachers = [Teacher(), Teacher()]
    source = [t.probabilities(clip) for t in teachers]
    ids = [f'anchor-policy-fixture-{i}' for i in range(100)
           if partition(f'anchor-policy-fixture-{i}') == 'train'][:4]
    contexts = []
    for codec in ('h264', 'h265'):
        anchor = StandardCodec(codec, 50).roundtrip(clip)
        context = build_utility_context(clip, 50, codec, mask, source,
            [t.probabilities(anchor.decoded) for t in teachers],
            anchor_bpp=reference_bpp(anchor.coded_bytes, clip.shape))
        contexts.extend([context, context.copy()])
    utility = np.zeros((4, 41))
    utility[[0, 2], 33] = .02
    utility[[1, 3], 36] = .03
    model = PortfolioRankPreprocessor(ANCHOR_NAMES, _fit_memory(np.stack(contexts), utility),
        dict(low=dict(neighbors=32, mix=0.), high=dict(neighbors=8, mix=.5)), ids)
    state = model.checkpoint_state()
    state.update(fit_method='sourceblocked_train_cv_portfolio', measurements=40,
                 train_ids_sha256='fixture')
    checkpoint = tmp_path / 'portfolio.pth'
    torch.save(state, checkpoint)
    policy = load_preprocessor(torch.load(checkpoint, weights_only=True), 'ar')
    cfg = dict(ar_candidates=['identity', 'area96', 'protected_mild'], rank_top_k=3,
        ar_confidence=.6, ar_require_anchor_decision=True, ar_kl_slack=.1, min_savings=.01)
    for codec in ('h264', 'h265'):
        anchor, primary, name, audit, arms = choose_stream(clip, mask, 'ar', StandardCodec(codec, 50),
            cfg, teachers, source, policy, learned_mask=mask, components=True)
        proposed = audit[0]['learned_order']
        assert any(i >= 34 for i in proposed)
        assert len(proposed) == len(set(proposed)) == 3
        assert len([r for r in audit if r['action_index'] is not None]) == 41
        assert audit[0]['anchor_decoded_sha256'] == hashlib.sha256(anchor.decoded.tobytes()).hexdigest()
        assert primary.coded_bytes <= arms['controls'][0].coded_bytes
        assert arms['bank_oracle'][0].coded_bytes <= arms['v26_bank_oracle'][0].coded_bytes
        assert arms['bank_oracle'][0].coded_bytes <= primary.coded_bytes
        if name.startswith('learned_rank__'):
            assert next(r['action_index'] for r in audit if r['name'] == name) in proposed
