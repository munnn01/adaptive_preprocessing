"""V23 low-dimensional learned policy for measured-codec optimization.

The policy sees full-clip context. Rendering uses protected convex low-pass
mixtures and causal stationary-background reuse, with no decoder side channel.
V22 checkpoints continue to load through the explicit schema factory.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .learned import AdaptiveBlendPreprocessor, lowpass


def semantic_protection(score):
    """Retain salient core; do not duplicate the all-time motion tube in V23."""
    import numpy as np
    score = np.asarray(score, np.float32)
    if score.ndim != 2 or not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
        raise ValueError("normalized 2D semantic map required")
    return np.where(score >= .95, 1., score ** 2).astype(np.float32)


class RateAwarePreprocessor(nn.Module):
    schema = "adaptive-vcm-rateaware-v2"
    candidate_name = "learned_rateaware"
    control_count = 7  # strength, five spectral experts, temporal reuse

    def __init__(self, width=24, task="ar"):
        super().__init__()
        if width < 4 or task not in ("ar", "od"):
            raise ValueError("invalid width/task")
        self.width, self.task = width, task
        self.scales = (.7, 1.5, 3., 6.) if task == "ar" else (2., 4., 8., 12.)
        self.trunk = nn.Sequential(nn.Conv2d(7, width, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.head = nn.Linear(width, self.control_count)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _validate(self, video, qp, codec, protection):
        if video.ndim != 5 or video.shape[1] != 3 or min(video.shape[2:]) < 1:
            raise ValueError("expected [B,3,T,H,W]")
        if not torch.isfinite(video).all() or video.min() < 0 or video.max() > 1:
            raise ValueError("source must be finite in [0,1]")
        b, _, t, h, w = video.shape
        qp = torch.as_tensor(qp, device=video.device, dtype=video.dtype).reshape(-1)
        codec = torch.as_tensor(codec, device=video.device, dtype=video.dtype).reshape(-1)
        if qp.numel() != b or codec.numel() != b or not torch.isfinite(qp).all() or (qp < 0).any() or (qp > 51).any():
            raise ValueError("one valid QP/codec per sample required")
        if not ((codec == 0) | (codec == 1)).all():
            raise ValueError("invalid codec")
        if protection is None:
            protection = video.new_zeros(b, 1, t, h, w)
        protection = protection.to(video)
        if protection.shape != (b, 1, t, h, w) or not torch.isfinite(protection).all() or (protection < 0).any() or (protection > 1).any():
            raise ValueError("invalid protection")
        return qp, codec, protection

    @staticmethod
    def _motion(video):
        previous = torch.cat([video[:, :, :1], video[:, :, :-1]], 2)
        difference = (video - previous).abs().mean(1, keepdim=True)
        return (difference / (12 / 255)).clamp(0, 1), difference.mean((1, 3, 4), keepdim=True) >= (32 / 255)

    def policy(self, video, qp, codec, protection=None):
        qp, codec, protection = self._validate(video, qp, codec, protection)
        b, c, t, h, w = video.shape
        motion, _ = self._motion(video)
        extra = torch.cat([motion, protection,
                           (qp / 51)[:, None, None, None, None].expand(b, 1, t, h, w),
                           codec[:, None, None, None, None].expand(b, 1, t, h, w)], 1)
        features = torch.cat([video, extra], 1).permute(0, 2, 1, 3, 4).reshape(b * t, 7, h, w)
        context = self.trunk(features).mean((2, 3)).reshape(b, t, self.width).mean(1)
        return self.head(context)

    def filter_bank(self, video, qp):
        b, c, t, h, w = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        experts = [lowpass(frames, sigma) for sigma in self.scales]
        blocks = []
        for index in range(b):
            q = float(qp.reshape(-1)[index])
            group = frames[index * t:(index + 1) * t]
            if q >= 45:
                # A DC expert offers real extra rate leverage when coefficients
                # already quantize away. Semantic cores remain source-exact.
                blocks.append(group.mean((2, 3), keepdim=True).expand_as(group))
                continue
            grid = 4 if q < 40 else 8
            padded = F.pad(group, (0, (-w) % grid, 0, (-h) % grid), mode="replicate")
            small = F.avg_pool2d(padded, grid, stride=grid)
            # Restore the padded geometry before cropping; no odd-size shift.
            coarse = F.interpolate(small, padded.shape[-2:], mode="bilinear", align_corners=False)
            blocks.append(coarse[:, :, :h, :w])
        experts.append(torch.cat(blocks))
        return torch.stack(experts, 1)

    def forward(self, video, qp, codec, protection=None, *, controls=None, bank=None, return_aux=False):
        qp, codec, protection = self._validate(video, qp, codec, protection)
        b, c, t, h, w = video.shape
        controls = self.policy(video, qp, codec, protection) if controls is None else controls
        if controls.shape != (b, self.control_count) or not torch.isfinite(controls).all():
            raise ValueError("invalid policy controls")
        q = ((qp - 30) / 20).clamp(0, 1)
        prior = (-.5 + 1.5 * q) if self.task == "ar" else (.5 + 2 * q)
        strength = torch.sigmoid(controls[:, 0] + prior)
        if self.task == "ar":
            mixture_prior = torch.stack([.8 - 1.5*q, .5 - .5*q, q, -.8 + 2*q, -.7 + 1.5*q], 1)
        else:
            mixture_prior = torch.stack([-1 - q, -.5*q, .8 + q, 1 + 1.5*q, -.8 + 2*q], 1)
        weights = (controls[:, 1:6] + mixture_prior).softmax(1)
        bank = self.filter_bank(video, qp) if bank is None else bank
        if bank.shape != (b * t, 5, c, h, w):
            raise ValueError("invalid filter bank")
        low = (bank * weights[:, None, :, None, None, None].expand(b, t, 5, 1, 1, 1).reshape(b*t, 5, 1, 1, 1)).sum(1)
        low = low.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        motion, cuts = self._motion(video)
        motion_attenuation = (.65 - .30 * q)[:, None, None, None, None]
        alpha = strength[:, None, None, None, None] * (1 - protection) * (1 - motion_attenuation * motion)
        result = video + alpha * (low - video)
        temporal = torch.sigmoid(controls[:, 6] - 1) * (.15 + .45 * q)
        previous_low = torch.cat([low[:, :, :1], low[:, :, :-1]], 2)
        reuse = temporal[:, None, None, None, None] * (1 - protection) * (1 - motion) * (~cuts)
        reuse = reuse * (torch.arange(t, device=video.device)[None, None, :, None, None] > 0)
        result = (result + reuse * (previous_low - result)).clamp(0, 1)
        result = torch.where(protection == 1, video, result)
        if return_aux:
            return result, {"alpha": alpha, "strength": strength, "scale_weights": weights,
                            "temporal_strength": temporal, "reuse": reuse, "controls": controls}
        return result


def load_preprocessor(state, task):
    from .profiles import ProfilePreprocessor
    if state.get("task") != task or state.get("steps", 0) < 1 or not state.get("train_ids_sha256"):
        raise ValueError("checkpoint task/training provenance mismatch")
    if state.get("schema") == "adaptive-vcm-blend-v1":
        model = AdaptiveBlendPreprocessor(state["width"])
    elif state.get("schema") == RateAwarePreprocessor.schema:
        model = RateAwarePreprocessor(state["width"], task)
    elif state.get("schema") == ProfilePreprocessor.schema:
        model = ProfilePreprocessor(state["width"], task)
    else:
        raise ValueError("unsupported preprocessor checkpoint schema")
    model.load_state_dict(state["model"], strict=True)
    return model
