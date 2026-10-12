"""Add bounded low-frequency/luma stabilization actions to the frozen V25 bank.

This engineering bank modifies source pixels, never codec syntax or settings.
It retains T/H/W for every added action. Uniform actions and source-exact core
variants are explicit; actual codec bytes and the unchanged AR guard decide
feasibility. The temporal luma window is offline and resets at scene cuts.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import Candidate, validate_clip
from .task_bank import ACTION_NAMES as BASE_ACTION_NAMES, build_task_bank


STABILIZED_ACTION_NAMES = (
    "uniform_dc_soft", "uniform_dc_medium", "uniform_dc_strong",
    "core_dc_soft", "core_dc_medium", "core_dc_strong",
    "uniform_exposure_soft", "uniform_exposure_medium", "uniform_exposure_strong",
    "core_exposure_soft", "core_exposure_medium", "core_exposure_strong",
    "uniform_dc_tiny", "core_dc_tiny", "uniform_exposure_tiny", "core_exposure_tiny",
)
ACTION_NAMES = BASE_ACTION_NAMES + STABILIZED_ACTION_NAMES
CORE_ACTION_NAMES = tuple(n for n in STABILIZED_ACTION_NAMES if n.startswith("core_"))


def max_pixel_change(qp: int, strength: str) -> int:
    """Per-channel integer movement limit for all new actions.

    QP30/40/45/50 limits are respectively soft2/3/4/4,
    medium4/6/7/8 and strong6/9/11/12. Rounding is included.
    Tiny actions are bounded at1 for every QP.
    """
    if type(qp) is not int or not 0 <= qp <= 51 or strength not in ("tiny", "soft", "medium", "strong"):
        raise ValueError("invalid QP/strength")
    q = float(np.clip((qp - 30) / 20., 0., 1.))
    return int(np.ceil({"tiny": 1, "soft": 4, "medium": 8, "strong": 12}[strength] * (.5 + .5*q)))


def _validate(clip, protection, qp):
    validate_clip(clip)
    if type(qp) is not int or not 0 <= qp <= 51:
        raise ValueError("invalid QP")
    mask = np.asarray(protection, np.float32)
    if mask.shape != clip.shape[1:3] or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError("invalid protection map")
    return mask


def _luma_reference(source):
    luma = source @ np.array([.299, .587, .114], np.float32)
    dc = luma.mean((1, 2))
    cuts = np.zeros(len(source), dtype=bool)
    cuts[0] = True
    motion = np.ones(source.shape[:3], np.float32)
    for t in range(1, len(source)):
        difference = source[t] - source[t-1]
        cuts[t] = float(np.abs(difference).mean()) >= 32.
        # Remove robust global brightness change before gating motion, allowing
        # exposure stabilization without interpreting uniform flicker as motion.
        global_shift = np.median(difference, axis=(0, 1))
        normalized_difference = np.abs(difference-global_shift).mean(-1)
        motion[t] = np.clip(1.-normalized_difference/12., 0., 1.)
        if cuts[t]:
            motion[t] = 1.
    reference = dc.copy()
    starts = np.flatnonzero(cuts)
    for start, end in zip(starts, [*starts[1:], len(source)]):
        for t in range(int(start), int(end)):
            reference[t] = np.median(dc[max(int(start), t-2):min(int(end), t+3)])
    return luma, dc, reference, motion


def build_stabilization_actions(clip: np.ndarray, protection: np.ndarray, qp: int) -> list[Candidate]:
    """Sixteen uniform/core actions with bounded tiny/soft/medium/strong levels.

    DC actions subtract part of a sigma4 luma field's departure from frame DC,
    leaving fine texture and edges in place. Exposure actions stabilize frame
    DC toward a five-frame source median inside a shot. Neither action removes
    frames or resizes. Every change is capped per RGB channel before rounding.
    """
    mask = _validate(clip, protection, qp)
    source = clip.astype(np.float32)
    q = float(np.clip((qp-30)/20., 0., 1.))
    luma, dc, reference, motion = _luma_reference(source)
    low = np.stack([cv2.GaussianBlur(frame, (0, 0), 4.,
                                    borderType=cv2.BORDER_REPLICATE) for frame in luma])
    dc_delta = dc[:, None, None]-low
    exposure_delta = (reference-dc)[:, None, None]*motion
    limits = np.array([4., 8., 12.], np.float32)*(.5+.5*q)
    actions = []
    for kind, delta, base_amounts in (("dc", dc_delta, (.20, .35, .50)),
                                      ("exposure", exposure_delta, (.35, .65, 1.))):
        for protected in (False, True):
            for level, amount, limit in zip(("soft", "medium", "strong"), base_amounts, limits):
                shift = np.clip(delta*amount*(.35+.65*q), -limit, limit)
                if protected:
                    shift = shift*(1.-mask)[None]
                output = np.clip(np.rint(source+shift[..., None]), 0, 255).astype(np.uint8)
                if protected:
                    output[:, mask == 1.] = clip[:, mask == 1.]
                prefix = "core" if protected else "uniform"
                actions.append(Candidate(f"{prefix}_{kind}_{level}", output))
    # Append near-identity backoff after the original12, preserving all indexes.
    # Pixel movements are <=1 level even at QP50; amounts are source/QP-only.
    for kind, delta, amount in (("dc", dc_delta, .05), ("exposure", exposure_delta, .15)):
        for protected in (False, True):
            shift = np.clip(delta*amount*(.35+.65*q), -(.5+.5*q), .5+.5*q)
            if protected:
                shift = shift*(1.-mask)[None]
            output = np.clip(np.rint(source+shift[..., None]), 0, 255).astype(np.uint8)
            if protected:
                output[:, mask == 1.] = clip[:, mask == 1.]
            prefix = "core" if protected else "uniform"
            actions.append(Candidate(f"{prefix}_{kind}_tiny", output))
    assert tuple(c.name for c in actions) == STABILIZED_ACTION_NAMES
    return actions


def build_stabilized_bank(clip: np.ndarray, protection: np.ndarray, qp: int) -> list[Candidate]:
    """Frozen identity+17 V25 actions, followed by16 stabilization actions."""
    return build_task_bank(clip, protection, qp)+build_stabilization_actions(clip, protection, qp)
