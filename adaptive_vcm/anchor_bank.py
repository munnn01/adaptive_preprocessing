"""Registered same-geometry source-to-anchor reconstruction residual blends.

The encoder's identity roundtrip provides the reference. Callers must pass its
decoded RGB clip from the same codec, QP and geometry as the current source.
There is no re-encoding inside this bank, no codec setting change and no task
label access. Pixel blending is a hypothesis: measured bytes and the existing
teacher guard, rather than the blend coefficient, determine feasibility.
"""
from __future__ import annotations

import numpy as np

from .preprocessing import Candidate, validate_clip
from .stabilized_bank import ACTION_NAMES as V26_ACTION_NAMES, build_stabilized_bank


ANCHOR_ACTION_NAMES = (
    "uniform_anchor_25", "uniform_anchor_50", "uniform_anchor_75", "uniform_anchor_100",
    "core_anchor_25", "core_anchor_50", "core_anchor_75", "core_anchor_100",
)
ACTION_NAMES = V26_ACTION_NAMES + ANCHOR_ACTION_NAMES
CORE_ACTION_NAMES = ANCHOR_ACTION_NAMES[4:]


def _validate(clip, protection, qp, anchor_decoded):
    if not isinstance(clip, np.ndarray):
        raise ValueError("expected uint8 RGB source [T,H,W,3]")
    validate_clip(clip)
    if type(qp) is not int or not 0 <= qp <= 51:
        raise ValueError("invalid QP")
    if (not isinstance(anchor_decoded, np.ndarray)
            or anchor_decoded.dtype != np.uint8
            or anchor_decoded.shape != clip.shape):
        raise ValueError("anchor reconstruction must match uint8 RGB source geometry")
    try:
        mask = np.asarray(protection, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid protection map") from error
    if (mask.shape != clip.shape[1:3] or not np.isfinite(mask).all()
            or np.any((mask < 0) | (mask > 1))):
        raise ValueError("invalid protection map")
    return mask


def build_anchor_actions(clip: np.ndarray, protection: np.ndarray, qp: int,
                         anchor_decoded: np.ndarray | None = None) -> list[Candidate]:
    """Eight convex residual shrink actions; every mask==1 core pixel is exact.

    Uniform blends move .25/.5/.75/1 of the source-to-anchor difference. Core
    variants attenuate this amount by (1-protection), preserving source pixels
    at full protection. The strengths are fixed; QP conditions the reference
    reconstruction supplied by the caller. Rounding uses NumPy round-to-even.
    """
    mask = _validate(clip, protection, qp, anchor_decoded)
    source = clip.astype(np.float32)
    residual = anchor_decoded.astype(np.float32) - source
    actions = []
    for protected in (False, True):
        scale = (1. - mask)[None, ..., None] if protected else 1.
        for percent, amount in ((25, .25), (50, .5), (75, .75), (100, 1.)):
            output = np.rint(source + amount * scale * residual).astype(np.uint8)
            if protected:
                output[:, mask == 1.] = clip[:, mask == 1.]
            prefix = "core" if protected else "uniform"
            actions.append(Candidate(f"{prefix}_anchor_{percent}", output))
    return actions


def build_anchor_bank(clip: np.ndarray, protection: np.ndarray, qp: int,
                      anchor_decoded: np.ndarray | None = None) -> list[Candidate]:
    """Frozen V26 identity+33 prefix, followed by eight reconstruction blends."""
    # Fail before rendering even the legacy prefix if the required anchor is absent.
    _validate(clip, protection, qp, anchor_decoded)
    return (build_stabilized_bank(clip, protection, qp)
            + build_anchor_actions(clip, protection, qp, anchor_decoded))
