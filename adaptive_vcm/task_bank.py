"""Fixed AR preprocessing actions for real-codec, encoder-teacher supervision.

This bank has no access to labels, decoded task results or codec controls.  Its
uniform actions are deliberately separate from semantic-core-preserving actions:
both are subject to the same downstream teacher guard.  Spatial resampling changes
pixel dimensions but never frame count; rates must use original-source pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import Candidate, validate_clip


ACTION_NAMES = (
    "identity",
    "resample120_denoise", "resample104_denoise", "resample88_denoise",
    "resample104_detailq", "resample88_detailq",
    "uniform_detail_soft", "uniform_detail_medium",
    "uniform_temporal_soft", "uniform_temporal_medium", "uniform_chroma_soft",
    "core_spatial_soft", "core_spatial_medium",
    "core_temporal_soft", "core_temporal_medium",
    "core_residual_soft", "core_residual_medium", "core_joint",
)

# Coarse/DC filtering is absent: preserve low-frequency appearance and remove
# only a bounded amount of fine detail. These definitions are fixed before DEV.
CORE_ACTIONS = ACTION_NAMES[11:]


def _blur(source: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([cv2.GaussianBlur(f, (0, 0), sigma,
                                    borderType=cv2.BORDER_REPLICATE) for f in source])


def _detail(source: np.ndarray, sigma: float, threshold: float) -> np.ndarray:
    """Soft dead zone for residual magnitude, preserving larger edges.

    The output moves each channel by at most ``threshold`` and remains between
    the source and its local average. This is pixel preprocessing, not changing
    the standard codec's quantizer or counting estimated transform bits.
    """
    low = _blur(source, sigma)
    residual = source - low
    retained = np.sign(residual) * np.maximum(np.abs(residual) - threshold, 0.)
    return low + retained


def _temporal(source: np.ndarray, filtered: np.ndarray, amount: float) -> np.ndarray:
    """Causal, motion-gated reuse of previous SOURCE, reset on scene cuts.

    References never recursively accumulate. Local movement gates reuse down
    to zero at a 12-level RGB difference; 32-level mean change resets a cut.
    """
    out = filtered.copy()
    for t in range(1, len(source)):
        delta = np.abs(source[t] - source[t - 1]).mean(axis=-1)
        if float(delta.mean()) >= 32.:
            continue
        stationary = np.clip(1. - delta / 12., 0., 1.)
        reuse = amount * stationary[..., None]
        out[t] = (1. - reuse) * filtered[t] + reuse * source[t - 1]
    return out


def _pixels(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def _resample(value: np.ndarray, ratio: float) -> np.ndarray:
    h, w = value.shape[1:3]
    # Even dimensions are required by unchanged yuv420p encoders.  Small clips
    # are supported as fixtures; no original-dimension assumption is hidden.
    shape = (max(2, int(w * ratio) // 2 * 2), max(2, int(h * ratio) // 2 * 2))
    return _pixels(np.stack([cv2.resize(f, shape, interpolation=cv2.INTER_AREA)
                            for f in value]))


def build_task_bank(clip: np.ndarray, protection: np.ndarray, qp: int) -> list[Candidate]:
    """Return identity plus 17 fixed AR actions in ``ACTION_NAMES`` order.

    Inputs are source uint8 RGB [T,H,W,3], encoder semantic protection [H,W] in
    [0,1], and standard codec QP. Uniform actions may edit semantic pixels; core
    actions preserve protection==1 exactly. No action guarantees task feasibility
    or byte savings: training/evaluation must measure and apply the shared guard.
    """
    validate_clip(clip)
    if type(qp) is not int or not 0 <= qp <= 51:
        raise ValueError("invalid QP")
    mask = np.asarray(protection, dtype=np.float32)
    if mask.shape != clip.shape[1:3] or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError("invalid protection map")
    source = clip.astype(np.float32)
    q = float(np.clip((qp - 30) / 20., 0., 1.))
    soft = _detail(source, .8, 2. + 6. * q)
    medium = _detail(source, 1.2, 4. + 12. * q)
    spatial_soft = source + (.15 + .20 * q) * (_blur(source, .8) - source)
    spatial_medium = source + (.30 + .25 * q) * (_blur(source, 1.2) - source)
    temporal_soft = _temporal(source, soft, .20 + .15 * q)
    temporal_medium = _temporal(source, medium, .35 + .20 * q)

    # Chroma-only filtering offers a different rate lever while retaining luma
    # structure, rather than flattening whole frames to their average color.
    ycc = np.stack([cv2.cvtColor(f / 255., cv2.COLOR_RGB2YCrCb) for f in source])
    chroma_low = _blur(ycc, 1. + q)
    ycc[..., 1:] += (.30 + .30 * q) * (chroma_low[..., 1:] - ycc[..., 1:])
    chroma = np.stack([cv2.cvtColor(f, cv2.COLOR_YCrCb2RGB) * 255. for f in ycc])
    core_alpha = (1. - mask)[None, ..., None]

    def core(value):
        result = _pixels(source + core_alpha * (value - source))
        result[:, mask == 1.] = clip[:, mask == 1.]
        return result

    # Resampling is combined with mild stationary denoising or detail shrinkage
    # at distinct resolutions from the historical area112/area96 controls.
    denoised = _temporal(source, spatial_soft, .20 + .15 * q)
    actions = [clip.copy(),
               _resample(denoised, .9375), _resample(denoised, .8125), _resample(denoised, .6875),
               _resample(soft, .8125), _resample(medium, .6875),
               _pixels(soft), _pixels(medium),
               _pixels(temporal_soft), _pixels(temporal_medium), _pixels(chroma),
               core(spatial_soft), core(spatial_medium),
               core(temporal_soft), core(temporal_medium),
               core(soft), core(medium), core(_temporal(source, spatial_medium, .35 + .20 * q))]
    return [Candidate(name, pixels) for name, pixels in zip(ACTION_NAMES, actions)]
