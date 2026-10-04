"""Content/QP/codec-conditioned AR policy over measurable preprocessing profiles.

The policy learns profile selection, not filter coefficients. A finite action
bank prevents continuous logits from shrinking every edit below pixel rounding.
Identity remains an explicit action; this does not force edits or guarantee rate.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .rateaware import RateAwarePreprocessor


class ProfilePreprocessor(RateAwarePreprocessor):
    schema = 'adaptive-vcm-profile-v3'
    candidate_name = 'learned_profile'
    # name, spectral expert, absolute spatial blend, temporal logit.
    # Expert 4 is block low-pass below QP45 and frame DC at QP45/50.
    profiles = (
        ('identity', 0, 0., -20.),
        ('detail_mild', 0, .25, -20.),
        ('detail_medium', 1, .45, -20.),
        ('coarse', 2, .65, -20.),
        ('coarse_strong', 3, .85, -20.),
        ('dc_medium', 4, .60, -20.),
        ('dc_strong', 4, .90, -20.),
        ('temporal_coarse', 2, .50, 1.5),
        ('temporal_dc', 4, .75, 1.5),
    )

    def __init__(self, width=24, task='ar'):
        if task != 'ar':
            raise ValueError('profile policy is registered for AR only')
        super().__init__(width, task)
        self.head = nn.Linear(width, len(self.profiles))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def profile_logits(self, video, qp, codec, protection=None):
        return super().policy(video, qp, codec, protection)

    def render_profile(self, video, qp, codec, protection, indices, *, bank=None, return_aux=False):
        qp, codec, protection = self._validate(video, qp, codec, protection)
        indices = torch.as_tensor(indices, device=video.device, dtype=torch.long).reshape(-1)
        if indices.numel() != video.shape[0] or (indices < 0).any() or (indices >= len(self.profiles)).any():
            raise ValueError('one valid profile index per source required')
        q = ((qp - 30) / 20).clamp(0, 1)
        prior = -.5 + 1.5 * q
        mixture_prior = torch.stack([.8 - 1.5*q, .5 - .5*q, q, -.8 + 2*q, -.7 + 1.5*q], 1)
        controls = video.new_zeros(video.shape[0], 7)
        for i, index in enumerate(indices.tolist()):
            _, expert, strength, temporal = self.profiles[index]
            controls[i, 0] = math.log(max(strength, 1e-8) / max(1 - strength, 1e-8)) - prior[i]
            controls[i, 1:6] = -8 - mixture_prior[i]
            controls[i, 1 + expert] = 8 - mixture_prior[i, expert]
            controls[i, 6] = temporal
        output, aux = super().forward(video, qp, codec, protection, controls=controls, bank=bank, return_aux=True)
        active = (indices != 0)[:, None, None, None, None]
        output = torch.where(active, output, video)
        aux['alpha'] = aux['alpha'] * active
        aux['reuse'] = aux['reuse'] * active
        aux['profile_indices'] = indices
        return (output, aux) if return_aux else output

    def forward(self, video, qp, codec, protection=None, *, return_aux=False):
        indices = self.profile_logits(video, qp, codec, protection).argmax(1)
        return self.render_profile(video, qp, codec, protection, indices, return_aux=return_aux)


def feasible_profile_target(measurements, *, slack, min_savings=.01):
    """Oracle TRAIN label: minimum actual bytes among feasible profiles.

    Only encoder-teacher observations are accepted. Ties prefer identity/lower
    profile index. No ground-truth label or evaluation-model result is an input.
    """
    from .selection import Observation, select
    if not measurements or measurements[0]['profile'] != 'identity':
        raise ValueError('identity measurement must be first')
    observations = [Observation(m['profile'], m['coded_bytes'], tuple(m['distances']), tuple(m['decisions']))
                    for m in measurements]
    return select(observations, slack, min_savings)
