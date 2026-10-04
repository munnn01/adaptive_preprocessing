"""Source suppression with semantic protection and motion-aware temporal blending.

Transforms see source pixels and encoder predictions only. No labels, decoder
postprocessor, ROI side channel, or codec parameter modification is required.
Spatially varying convex blending is bounded but does not guarantee fewer bits.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


def validate_clip(clip: np.ndarray) -> None:
    if clip.dtype != np.uint8 or clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError("expected uint8 RGB [T,H,W,3]")
    if min(clip.shape[:3]) < 1:
        raise ValueError("empty clip")


def gaussian(clip: np.ndarray, sigma: float) -> np.ndarray:
    validate_clip(clip)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    return np.stack([cv2.GaussianBlur(f.astype(np.float32), (0, 0), sigma,
                                    borderType=cv2.BORDER_REPLICATE) for f in clip])


def normalize_map(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    if not np.isfinite(score).all() or np.any(score < 0):
        raise ValueError("importance must be finite and nonnegative")
    top = float(np.percentile(score, 95))
    return np.clip(score / max(top, 1e-8), 0, 1)


def feather(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    """Only feather outside exact protected pixels; never weaken the ROI core."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or radius < 0:
        raise ValueError("expected 2D mask and nonnegative radius")
    if not mask.any() or radius == 0:
        return mask.astype(np.float32)
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return np.maximum(mask, np.clip(1 - distance / radius, 0, 1)).astype(np.float32)


def boxes_to_mask(height: int, width: int, boxes: np.ndarray, *,
                  halo: float = .15, grid: int = 16, radius: int = 8) -> np.ndarray:
    """Outward block-aligned context halo for frozen source detector boxes."""
    if height < 1 or width < 1 or not math.isfinite(halo) or halo < 0 or grid < 1:
        raise ValueError("invalid mask geometry")
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not np.isfinite(boxes).all():
        raise ValueError("nonfinite boxes")
    mask = np.zeros((height, width), bool)
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        dx, dy = (x2 - x1) * halo, (y2 - y1) * halo
        left = max(0, min(width, math.floor((x1 - dx) / grid) * grid))
        top = max(0, min(height, math.floor((y1 - dy) / grid) * grid))
        right = max(0, min(width, math.ceil((x2 + dx) / grid) * grid))
        bottom = max(0, min(height, math.ceil((y2 + dy) / grid) * grid))
        mask[top:bottom, left:right] = True
    return feather(mask, radius)


def action_protection(clip: np.ndarray, saliency: np.ndarray | None = None,
                      motion_threshold: float = 12.) -> np.ndarray:
    """Stable semantic/motion tube; motion is protected without per-clip min/max."""
    validate_clip(clip)
    if motion_threshold <= 0 or not math.isfinite(motion_threshold):
        raise ValueError("invalid motion threshold")
    motion = np.zeros(clip.shape[1:3], np.float32)
    if len(clip) > 1:
        delta = np.abs(np.diff(clip.astype(np.float32), axis=0)).mean(axis=3)
        motion = np.clip(delta.max(axis=0) / motion_threshold, 0, 1)
        motion = cv2.dilate(motion, np.ones((5, 5), np.uint8))
    if saliency is None:
        return motion
    if np.shape(saliency) != motion.shape:
        raise ValueError("saliency dimensions do not match source")
    semantic = normalize_map(saliency)
    semantic = cv2.GaussianBlur(semantic, (0, 0), 1.5, borderType=cv2.BORDER_REPLICATE)
    return np.maximum(motion, semantic)


def suppress(clip: np.ndarray, protection: np.ndarray, *, sigma: float,
             strength: float, temporal: float = 0., motion_threshold: float = 12.,
             scene_cut_threshold: float = 32.) -> np.ndarray:
    """QP-conditioned source blend; causal reuse only in stationary background.

    The temporal reference is the previous SOURCE frame, so cuts cannot leak
    stale recursive state. Source motion determines gating before spatial blur.
    """
    validate_clip(clip)
    if not all(math.isfinite(v) for v in (strength, temporal, motion_threshold, scene_cut_threshold)):
        raise ValueError("nonfinite suppression settings")
    if not 0 <= strength <= 1 or not 0 <= temporal <= 1 or motion_threshold <= 0 or scene_cut_threshold <= 0:
        raise ValueError("invalid suppression settings")
    protection = np.asarray(protection, dtype=np.float32)
    if protection.shape != clip.shape[1:3] or not np.isfinite(protection).all():
        raise ValueError("invalid protection map")
    if np.any((protection < 0) | (protection > 1)):
        raise ValueError("protection must lie in [0,1]")
    source = clip.astype(np.float32)
    low = gaussian(clip, sigma)
    alpha = strength * (1 - protection)[None, ..., None]
    result = source + alpha * (low - source)
    for t in range(1, len(clip)):
        motion = np.abs(source[t] - source[t - 1]).mean(axis=-1)
        if float(motion.mean()) >= scene_cut_threshold:
            continue
        stationary = np.clip(1 - motion / motion_threshold, 0, 1)
        reuse = temporal * (1 - protection) * stationary
        result[t] += reuse[..., None] * (low[t - 1] - result[t])
    # Exact source pixels wherever protection is 1, including after rounding.
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class Candidate:
    name: str
    clip: np.ndarray


def _area(clip: np.ndarray, ratio: float) -> np.ndarray:
    h, w = clip.shape[1:3]
    shape = (max(2, int(w * ratio) // 2 * 2), max(2, int(h * ratio) // 2 * 2))
    return np.stack([cv2.resize(f, shape, interpolation=cv2.INTER_AREA) for f in clip])


def make_candidates(clip: np.ndarray, protection: np.ndarray, task: str, qp: int,
                    enabled: tuple[str, ...] | list[str] | None = None) -> list[Candidate]:
    """Preregistered transforms; no outcome-dependent strength/threshold tuning."""
    validate_clip(clip)
    if task not in ("ar", "od") or not 0 <= qp <= 51:
        raise ValueError("unsupported task/QP")
    q = np.clip((qp - 25) / 25, 0, 1)
    candidates = [Candidate("identity", clip.copy())]
    if task == "ar":
        area112 = _area(clip, .875)
        candidates.extend([Candidate("area112", area112), Candidate("area96", _area(clip, .75)),
                           Candidate("area112_up", np.stack([cv2.resize(f, (clip.shape[2], clip.shape[1]),
                                                interpolation=cv2.INTER_LINEAR) for f in area112]))])
        # Historical source-blur baselines, plus new semantic/motion-aware arms.
        for amount in (.2, .4):
            candidates.append(Candidate(f"blur{int(amount * 100)}", suppress(
                clip, np.zeros(clip.shape[1:3]), sigma=1.5, strength=amount)))
        for name, sigma, strength, temporal in (
            ("protected_mild", 1.5, .25 + .25 * q, 0),
            ("protected_strong", 2.5, .45 + .35 * q, 0),
            ("protected_temporal", 2.5, .45 + .35 * q, .35)):
            candidates.append(Candidate(name, suppress(clip, protection, sigma=sigma,
                                                      strength=float(strength), temporal=temporal)))
    else:
        for sigma in (2., 4., 8., 12.):
            candidates.append(Candidate(f"background{int(sigma)}", suppress(
                clip, protection, sigma=sigma, strength=1.)))
    if enabled is not None:
        valid = {c.name for c in candidates}
        if "identity" not in enabled or set(enabled) - valid or len(set(enabled)) != len(enabled):
            raise ValueError("invalid candidate allowlist; identity is required")
        candidates = [c for c in candidates if c.name in enabled]
    return candidates
