"""V28/V29 spatial neural blend and twelve full-geometry profiles.

All four experts act in place. Segment DC uses editable source pixels, with no
cross-cut pooling. The learned alpha is causal; the encoder may inspect the
source segment to construct its DC expert. This module does not load weights.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .learned import lowpass
from .preprocessing import Candidate, validate_clip


EXPERT_NAMES = ("mild_gaussian", "strong_gaussian", "block_lowpass", "background_dc")
PROFILE_STRENGTHS = (.4, .75, 1.)
PROFILE_NAMES = tuple(f"motion_{expert}_{int(strength * 100):03d}"
                      for expert in EXPERT_NAMES for strength in PROFILE_STRENGTHS)


def validate_support(clip: np.ndarray, support: dict, task: str):
    """Validate cached/source support without changing its geometry or values."""
    validate_clip(clip)
    if task not in ("ar", "od") or (task == "od" and len(clip) != 1):
        raise ValueError("unsupported task or multi-frame OD input")
    if not isinstance(support, dict) or not {"protection", "motion", "cuts"} <= support.keys():
        raise ValueError("missing motion support")
    arrays = [np.asarray(support[key], dtype=np.float32) for key in ("protection", "motion")]
    for value in arrays:
        if value.shape != clip.shape[:3] or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("expected finite [0,1] THW support")
    cuts = np.asarray(support["cuts"])
    if cuts.shape != (len(clip),) or cuts.dtype != np.bool_ or not cuts[0]:
        raise ValueError("cuts must be boolean T with first frame a segment start")
    return *arrays, cuts


def _experts(video: torch.Tensor, protection: torch.Tensor, cuts: torch.Tensor, task: str,
             motion: torch.Tensor | None = None, variant: str | None = None):
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    mild_sigma, strong_sigma = (.8, 2.5) if task == "ar" else (2., 8.)
    mild, strong = lowpass(frames, mild_sigma), lowpass(frames, strong_sigma)
    padded = F.pad(frames, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
    block = F.interpolate(F.avg_pool2d(padded, 8, stride=8), size=padded.shape[-2:], mode="nearest")
    block = block[..., :h, :w]
    dc_batches = []
    for batch in range(b):
        starts = torch.nonzero(cuts[batch], as_tuple=False).flatten().tolist() + [t]
        dc_segments = []
        for start, end in zip(starts[:-1], starts[1:]):
            editable = 1 - protection[batch, :, start:end]
            segment = video[batch, :, start:end]
            if variant == "b" and task == "ar":
                stationary = (1 - motion[batch, :, start:end]).square()
                editable = editable * stationary
            denominator = editable.sum().clamp_min(1e-8)
            average = (segment * editable).sum((1, 2, 3)) / denominator
            constant = average[:, None, None, None].expand(c, end - start, h, w)
            if variant == "b" and task == "ar":
                constant = segment + stationary * (constant - segment)
            elif variant == "b" and task == "od":
                # Preserve half the local luma variation; background chroma is
                # shared from editable RGB, never from protected foreground.
                coefficients = video.new_tensor([.299, .587, .114])
                luma = (segment * coefficients[:, None, None, None]).sum(0, keepdim=True)
                mean_luma = (average * coefficients).sum()
                constant = .5 * luma + .5 * mean_luma + (average - mean_luma)[:, None, None, None]
            dc_segments.append(constant)
        dc_batches.append(torch.cat(dc_segments, 1))
    dc = torch.stack(dc_batches).permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return torch.stack([mild, strong, block, dc], 1)


class MotionAwarePreprocessor(nn.Module):
    schema = "adaptive-vcm-motion-v7"
    candidate_name = "learned_motion"

    def __init__(self, width: int = 12, task: str = "ar", variant: str | None = None):
        super().__init__()
        if variant is not None and variant not in ("a", "b", "c"):
            raise ValueError("invalid variant")
        if isinstance(width, bool) or not isinstance(width, int) or width < 4 or task not in ("ar", "od"):
            raise ValueError("invalid width or task")
        self.width, self.task, self.variant = width, task, variant
        if variant is not None:
            self.schema = "adaptive-vcm-semantic-v8"
        self.trunk = nn.Sequential(nn.Conv2d(7, width, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.head = nn.Conv2d(width, 5, 1)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = -1.5

    def forward(self, video: torch.Tensor, qp: torch.Tensor, codec: torch.Tensor,
                protection: torch.Tensor, motion: torch.Tensor | None = None,
                cuts: torch.Tensor | None = None, strength_scale: float = 1., return_aux: bool = False):
        if video.ndim != 5 or video.shape[1] != 3 or min(video.shape[0:1] + video.shape[2:]) < 1 or not video.is_floating_point():
            raise ValueError("expected floating B3THW video")
        if not torch.isfinite(video).all() or (video < 0).any() or (video > 1).any():
            raise ValueError("video must be finite in [0,1]")
        b, c, t, h, w = video.shape
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
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        conditioning = torch.cat([protection, motion,
                                  (qp / 51)[:, None, None, None, None].expand(b, 1, t, h, w),
                                  codec[:, None, None, None, None].expand(b, 1, t, h, w)], 1)
        features = torch.cat([frames, conditioning.permute(0, 2, 1, 3, 4).reshape(b * t, 4, h, w)], 1)
        maps = self.head(self.trunk(features))
        maps = F.avg_pool2d(F.pad(maps, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        raw_alpha = maps[:, :1].sigmoid().reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4)
        states = []
        state = raw_alpha[:, :, 0]
        for index in range(t):
            current = raw_alpha[:, :, index]
            if index:
                keep = .5 * (~cuts[:, index])[:, None, None, None].to(video) * (1 - motion[:, :, index])
                state = current * (1 - keep) + state * keep
            states.append(state)
        alpha = (torch.stack(states, 2) * strength_scale).clamp(0, 1) * (1 - protection) * (1 - .75 * motion)
        expert_maps = maps[:, 1:]
        if self.variant == "b":
            logits = expert_maps.reshape(b, t, 4, h, w).permute(0, 2, 1, 3, 4)
            logit_states = []
            logit_state = logits[:, :, 0]
            for index in range(t):
                current = logits[:, :, index]
                if index:
                    keep = .5 * (~cuts[:, index])[:, None, None, None].to(video) * (1 - motion[:, :, index])
                    logit_state = current * (1 - keep) + logit_state * keep
                logit_states.append(logit_state)
            logits = torch.stack(logit_states, 2)
            expert_maps = logits.permute(0, 2, 1, 3, 4).reshape(b * t, 4, h, w)
        mixture_frames = expert_maps.softmax(1)
        low = (_experts(video, protection, cuts, self.task, motion, self.variant) * mixture_frames[:, :, None]).sum(1)
        alpha_frames = alpha.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        result = (frames + alpha_frames * (low - frames)).clamp(0, 1)
        result = result.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        result = torch.where(protection == 1, video, result)
        if return_aux:
            mixture = mixture_frames.reshape(b, t, 4, h, w).permute(0, 2, 1, 3, 4)
            aux = {"alpha": alpha, "raw_alpha": raw_alpha, "mixture": mixture}
            if self.variant is not None:
                aux["expert_logits"] = expert_maps.reshape(b, t, 4, h, w).permute(0, 2, 1, 3, 4)
            return result, aux
        return result


def profile_candidates(clip: np.ndarray, support: dict, task: str, qp: int,
                       variant: str | None = None) -> list[Candidate]:
    """Return four experts x three strengths in ``PROFILE_NAMES`` order.

    QP validates the registered measurement condition; these reference pixel
    transforms have fixed strengths at every QP. All bytes/guards are measured
    by the caller. There is no identity profile or inference portfolio here.
    """
    if variant is not None and variant not in ("a", "b", "c"):
        raise ValueError("invalid variant")
    protection, motion, cuts = validate_support(clip, support, task)
    if isinstance(qp, bool) or not isinstance(qp, (int, np.integer)) or not 0 <= qp <= 51:
        raise ValueError("invalid QP")
    video = torch.from_numpy(clip.astype(np.float32) / 255.).permute(3, 0, 1, 2)[None]
    mask = torch.from_numpy(protection)[None, None]
    motion_tensor = torch.from_numpy(motion)[None, None]
    with torch.no_grad():
        experts = _experts(video, mask, torch.from_numpy(cuts)[None], task, motion_tensor, variant)
        source = video.permute(0, 2, 1, 3, 4).reshape(len(clip), 3, *clip.shape[1:3])
        alpha = 1 - mask
        if variant is not None:
            alpha = alpha * (1 - .75 * motion_tensor)
        alpha = alpha.permute(0, 2, 1, 3, 4).reshape(len(clip), 1, *clip.shape[1:3])
        candidates = []
        for expert_index, expert_name in enumerate(EXPERT_NAMES):
            for strength in PROFILE_STRENGTHS:
                rendered = source + strength * alpha * (experts[:, expert_index] - source)
                pixels = np.clip(np.rint(rendered.permute(0, 2, 3, 1).numpy() * 255.), 0, 255).astype(np.uint8)
                pixels[protection == 1] = clip[protection == 1]
                candidates.append(Candidate(f"motion_{expert_name}_{int(strength * 100):03d}", pixels))
    return candidates
