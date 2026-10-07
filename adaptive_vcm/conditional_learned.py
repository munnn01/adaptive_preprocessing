"""V30 source-level admission, expert mixture, and strength for RGB preprocessing.

Reference profiles and the neural renderer share one blend equation. A/B use
the original V29-A four experts; C changes only the third expert slot.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .learned import lowpass
from .motion_learned import EXPERT_NAMES, PROFILE_NAMES, PROFILE_STRENGTHS, _experts, validate_support
from .preprocessing import Candidate


CONDITIONAL_SCHEMA = "adaptive-vcm-conditional-v9"


def _check_variant(variant: str) -> None:
    if variant not in ("a", "b", "c"):
        raise ValueError("variant must be a, b, or c")


def profile_registry(variant: str = "a") -> list[dict]:
    """Return ordered, canonical profile parameters for one experiment arm."""
    _check_variant(variant)
    registry = []
    for index, name in enumerate(PROFILE_NAMES):
        weights = [0.] * 4
        weights[index // len(PROFILE_STRENGTHS)] = 1.
        registry.append({"name": name, "strength": float(PROFILE_STRENGTHS[index % 3]),
                         "expert_weights": weights})
    if variant == "b":
        for index, expert in enumerate(EXPERT_NAMES):
            weights = [0.] * 4
            weights[index] = 1.
            registry.append({"name": f"motion_{expert}_020", "strength": .2,
                             "expert_weights": weights})
        for label, weights in (("mild_gaussian+block_lowpass", [.5, 0., .5, 0.]),
                               ("strong_gaussian+background_dc", [0., .5, 0., .5])):
            for strength in (.75, 1.):
                registry.append({"name": f"motion_{label}_{int(strength * 100):03d}",
                                 "strength": strength, "expert_weights": weights.copy()})
    return registry


def canonical_target(profile_name: str, variant: str = "a") -> dict:
    """Resolve a measured profile name into an admission and blend label."""
    if profile_name == "identity":
        _check_variant(variant)
        return {"admission": False, "strength": 0., "expert_weights": [0.] * 4}
    for row in profile_registry(variant):
        if row["name"] == profile_name:
            return {"admission": True, "strength": row["strength"],
                    "expert_weights": row["expert_weights"].copy()}
    raise ValueError(f"unknown conditional profile: {profile_name}")


def context_expert(video: torch.Tensor, protection: torch.Tensor, motion: torch.Tensor,
                   cuts: torch.Tensor, task: str) -> torch.Tensor:
    """C's replacement expert in B3THW, before the common support blend."""
    if task == "ar":
        states = []
        state = video[:, :, 0]
        for index in range(video.shape[2]):
            current = video[:, :, index]
            if index:
                continued = .5 * current + .5 * state
                state = torch.where(cuts[:, index, None, None, None], current, continued)
            states.append(current + (1 - motion[:, :, index]).square() * (state - current))
        return torch.stack(states, dim=2)
    if task != "od" or video.shape[2] != 1:
        raise ValueError("context expert expects AR video or one-frame OD")
    source = video[:, :, 0]
    delta = lowpass(source, 8.) - source
    weights = video.new_tensor([.299, .587, .114])[None, :, None, None]
    perpendicular = delta - (delta * weights).sum(1, keepdim=True)
    positive = perpendicular > 0
    negative = perpendicular < 0
    upper = torch.where(positive, (1 - source) / perpendicular.clamp_min(1e-12),
                        torch.where(negative, source / (-perpendicular).clamp_min(1e-12),
                                    torch.ones_like(source)))
    step = upper.amin(1, keepdim=True).clamp(0, 1)
    return (source + step * perpendicular)[:, :, None]


def _expert_stack(video: torch.Tensor, protection: torch.Tensor, motion: torch.Tensor,
                  cuts: torch.Tensor, task: str, variant: str) -> torch.Tensor:
    """Return BT4CHW experts in the canonical slot order."""
    result = _experts(video, protection, cuts, task, motion, variant="a")
    if variant == "c":
        b, _, t, h, w = video.shape
        replacement = context_expert(video, protection, motion, cuts, task)
        replacement = replacement.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
        result = torch.stack((result[:, 0], result[:, 1], replacement, result[:, 3]), dim=1)
    return result


def _render(video: torch.Tensor, protection: torch.Tensor, motion: torch.Tensor,
            cuts: torch.Tensor, task: str, variant: str, alpha: torch.Tensor,
            weights: torch.Tensor, experts: torch.Tensor | None = None) -> torch.Tensor:
    b, _, t, h, w = video.shape
    if experts is None:
        experts = _expert_stack(video, protection, motion, cuts, task, variant)
    source_frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
    blend_weights = weights.expand(b, 4, t, h, w).permute(0, 2, 1, 3, 4).reshape(b * t, 4, 1, h, w)
    chosen = (experts * blend_weights).sum(1)
    frame_alpha = alpha.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
    result = (source_frames + frame_alpha * (chosen - source_frames)).clamp(0, 1)
    result = result.reshape(b, t, 3, h, w).permute(0, 2, 1, 3, 4)
    return torch.where(protection == 1, video, result)


class ConditionalPreprocessor(nn.Module):
    schema = CONDITIONAL_SCHEMA
    candidate_name = "learned_conditional"

    def __init__(self, width: int = 12, task: str = "ar", variant: str = "a"):
        super().__init__()
        _check_variant(variant)
        if isinstance(width, bool) or not isinstance(width, int) or width < 4 or task not in ("ar", "od"):
            raise ValueError("invalid width or task")
        self.width, self.task, self.variant = width, task, variant
        self.trunk = nn.Sequential(nn.Conv2d(7, width, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.gate_head = nn.Linear(3 * width + 2, 1)
        self.expert_head = nn.Linear(3 * width + 2, 4)
        self.strength_head = nn.Linear(3 * width + 2, 1)
        nn.init.constant_(self.gate_head.bias, -1.5)

    def forward(self, video: torch.Tensor, qp: torch.Tensor, codec: torch.Tensor,
                protection: torch.Tensor, motion: torch.Tensor | None = None,
                cuts: torch.Tensor | None = None, strength_scale: float = 1.,
                return_aux: bool = False):
        if video.ndim != 5 or video.shape[1] != 3 or min(video.shape[0:1] + video.shape[2:]) < 1 or not video.is_floating_point():
            raise ValueError("expected floating B3THW video")
        if not torch.isfinite(video).all() or (video < 0).any() or (video > 1).any():
            raise ValueError("video must be finite in [0,1]")
        b, _, t, h, w = video.shape
        if self.task == "od" and t != 1:
            raise ValueError("OD requires one frame")
        if not isinstance(strength_scale, (int, float)) or not math.isfinite(strength_scale) or strength_scale < 0:
            raise ValueError("strength scale must be finite and nonnegative")
        qp = torch.as_tensor(qp, device=video.device, dtype=video.dtype).reshape(-1)
        codec = torch.as_tensor(codec, device=video.device, dtype=video.dtype).reshape(-1)
        if qp.numel() != b or codec.numel() != b or not torch.isfinite(qp).all() or (qp < 0).any() or (qp > 51).any():
            raise ValueError("one finite QP/codec per sample required")
        if not ((codec == 0) | (codec == 1)).all():
            raise ValueError("codec must be 0 (H264) or 1 (H265)")
        if motion is None:
            motion = video.new_zeros(b, 1, t, h, w)
        for name, value in (("protection", protection), ("motion", motion)):
            if value.shape != (b, 1, t, h, w) or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
                raise ValueError(f"invalid {name}")
        protection, motion = protection.to(video), motion.to(video)
        if cuts is None:
            cuts = torch.zeros((b, t), device=video.device, dtype=torch.bool)
            cuts[:, 0] = True
            if t > 1:
                cuts[:, 1:] = (video[:, :, 1:] - video[:, :, :-1]).abs().mean((1, 3, 4)) >= 32 / 255
        else:
            cuts = torch.as_tensor(cuts, device=video.device)
            if cuts.shape != (b, t) or cuts.dtype != torch.bool or not cuts[:, 0].all():
                raise ValueError("cuts must be boolean BT with first frame a segment start")
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
        conditions = torch.cat((protection, motion,
                                (qp / 51)[:, None, None, None, None].expand(b, 1, t, h, w),
                                codec[:, None, None, None, None].expand(b, 1, t, h, w)), 1)
        features = torch.cat((frames, conditions.permute(0, 2, 1, 3, 4).reshape(b * t, 4, h, w)), 1)
        trunk = self.trunk(features).reshape(b, t, self.width, h, w).permute(0, 2, 1, 3, 4)
        def weighted_pool(weight):
            total = weight.sum((2, 3, 4)).clamp_min(1e-8)
            return (trunk * weight).sum((2, 3, 4)) / total
        pooled = torch.cat((trunk.mean((2, 3, 4)), weighted_pool(protection),
                            weighted_pool(1 - protection), (qp / 51)[:, None], codec[:, None]), dim=1)
        gate_logit = self.gate_head(pooled).reshape(b, 1, 1, 1, 1)
        gate_probability = gate_logit.sigmoid()
        gate = gate_probability if self.training else (gate_probability >= .5).to(video)
        expert_logits = self.expert_head(pooled).reshape(b, 4, 1, 1, 1)
        expert_weights = expert_logits.softmax(1)
        strength = self.strength_head(pooled).sigmoid().reshape(b, 1, 1, 1, 1)
        raw_alpha = strength.expand(b, 1, t, h, w)
        alpha = gate * (strength * strength_scale).clamp(0, 1) * (1 - protection) * (1 - .75 * motion)
        result = _render(video, protection, motion, cuts, self.task, self.variant, alpha, expert_weights)
        if return_aux:
            return result, {"alpha": alpha, "raw_alpha": raw_alpha,
                            "mixture": expert_weights.expand(b, 4, t, h, w),
                            "expert_logits": expert_logits.expand(b, 4, t, h, w),
                            "gate_probability": gate_probability, "gate_logit": gate_logit,
                            "strength": strength, "expert_weights": expert_weights}
        return result


def profile_candidates(clip: np.ndarray, support: dict, task: str, qp: int,
                       variant: str = "a") -> list[Candidate]:
    """Render the complete measured reference bank from canonical parameters."""
    _check_variant(variant)
    protection, motion, cuts = validate_support(clip, support, task)
    if isinstance(qp, bool) or not isinstance(qp, (int, np.integer)) or not 0 <= qp <= 51:
        raise ValueError("invalid QP")
    video = torch.from_numpy(clip.astype(np.float32) / 255.).permute(3, 0, 1, 2)[None]
    p = torch.from_numpy(protection)[None, None]
    m = torch.from_numpy(motion)[None, None]
    cut_tensor = torch.from_numpy(cuts)[None]
    gain = (1 - p) * (1 - .75 * m)
    candidates = []
    with torch.no_grad():
        experts = _expert_stack(video, p, m, cut_tensor, task, variant)
        for row in profile_registry(variant):
            weights = video.new_tensor(row["expert_weights"]).reshape(1, 4, 1, 1, 1)
            rendered = _render(video, p, m, cut_tensor, task, variant, row["strength"] * gain, weights, experts)
            pixels = np.clip(np.rint(rendered[0].permute(1, 2, 3, 0).numpy() * 255.), 0, 255).astype(np.uint8)
            pixels[protection == 1] = clip[protection == 1]
            candidates.append(Candidate(row["name"], pixels))
    return candidates
