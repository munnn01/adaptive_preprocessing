"""Multiscale source-blend preprocessor extending the v21 bounded editor.

An optional small model learns spatial blur strengths and a mixture of fixed
low-pass scales. Encoder-provided semantic protection is a hard constraint.
Inference has causal motion/cut gating and takes codec identity and QP. It has
no learned RGB residual and no decoder network. An untrained model is never
silently used in evaluation.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def lowpass(frames: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = math.ceil(3 * sigma)
    coordinates = torch.arange(-radius, radius + 1, device=frames.device, dtype=frames.dtype)
    kernel = torch.exp(-.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    channels = frames.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    result = F.conv2d(F.pad(frames, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels)
    return F.conv2d(F.pad(result, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)


class AdaptiveBlendPreprocessor(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        if width < 4:
            raise ValueError("width must be at least four")
        self.width = width
        self.scales = (.7, 1.5, 3.)
        self.trunk = nn.Sequential(nn.Conv2d(7, width, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.head = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = -3.

    def forward(self, video: torch.Tensor, qp: torch.Tensor, codec: torch.Tensor,
                protection: torch.Tensor | None = None, *, return_aux: bool = False):
        if video.ndim != 5 or video.shape[1] != 3 or min(video.shape[2:]) < 1:
            raise ValueError("expected [B,3,T,H,W]")
        if not torch.isfinite(video).all() or video.min() < 0 or video.max() > 1:
            raise ValueError("source must be finite in [0,1]")
        b, c, t, h, w = video.shape
        qp = torch.as_tensor(qp, device=video.device, dtype=video.dtype).reshape(-1)
        codec = torch.as_tensor(codec, device=video.device, dtype=video.dtype).reshape(-1)
        if qp.numel() != b or codec.numel() != b or not torch.isfinite(qp).all() or (qp < 0).any() or (qp > 51).any():
            raise ValueError("one valid QP and codec per batch sample required")
        if not ((codec == 0) | (codec == 1)).all():
            raise ValueError("codec ID must be 0 (H.264) or 1 (H.265)")
        if protection is None:
            protection = video.new_zeros(b, 1, t, h, w)
        if protection.shape != (b, 1, t, h, w) or not torch.isfinite(protection).all() or (protection < 0).any() or (protection > 1).any():
            raise ValueError("invalid protection")
        protection = protection.to(video)
        previous = torch.cat([video[:, :, :1], video[:, :, :-1]], 2)
        difference = (video - previous).abs().mean(1, keepdim=True)
        motion = (difference / (12 / 255)).clamp(0, 1)
        cut = difference.mean((1, 3, 4), keepdim=True) >= (32 / 255)
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        extra = torch.cat([motion, protection,
                           (qp / 51)[:, None, None, None, None].expand(b, 1, t, h, w),
                           codec[:, None, None, None, None].expand(b, 1, t, h, w)], 1)
        features = torch.cat([frames, extra.permute(0, 2, 1, 3, 4).reshape(b * t, 4, h, w)], 1)
        maps = self.head(self.trunk(features))
        # Replicate borders preserve a spatially constant gate at the borders.
        maps = F.avg_pool2d(F.pad(maps, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        raw_alpha = torch.sigmoid(maps[:, :1]).reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4)
        smooth = []
        state = raw_alpha[:, :, 0]
        for index in range(t):
            current = raw_alpha[:, :, index]
            if index:
                keep = (~cut[:, :, index]).to(video.dtype) * (1 - motion[:, :, index])
                state = current * (1 - .5 * keep) + state * (.5 * keep)
            else:
                state = current
            smooth.append(state)
        alpha = torch.stack(smooth, 2) * (1 - protection) * (1 - .75 * motion)
        weights = maps[:, 1:].softmax(1)
        filtered = torch.stack([lowpass(frames, sigma) for sigma in self.scales], 1)
        low = (filtered * weights[:, :, None]).sum(1)
        alpha_frames = alpha.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        result = frames + alpha_frames * (low - frames)
        result = result.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        if return_aux:
            return result, {"alpha": alpha, "scale_weights": weights}
        return result
