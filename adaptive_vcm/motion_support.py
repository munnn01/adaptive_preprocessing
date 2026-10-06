"""Camera-compensated RGB optical-flow support for encoder-only suppression.

This is an original density-region adaptation, not codec motion-vector
extraction or a faithful implementation of MoCrop. No region crosses a cut.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .preprocessing import feather, validate_clip


def _consistent(flow: np.ndarray, reverse: np.ndarray):
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    x, y = xx + flow[..., 0], yy + flow[..., 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    sampled = cv2.remap(reverse, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    error = np.linalg.norm(flow + sampled, axis=-1)
    tolerance = np.maximum(1., .10 * (np.linalg.norm(flow, axis=-1) + np.linalg.norm(sampled, axis=-1)))
    return valid & (error <= tolerance), valid


def _flow_pair(first: np.ndarray, second: np.ndarray):
    """Return both coordinate systems' residual density plus auditable quality."""
    settings = (.5, 3, 15, 3, 5, 1.2, 0)
    forward = cv2.calcOpticalFlowFarneback(first, second, None, *settings)
    backward = cv2.calcOpticalFlowFarneback(second, first, None, *settings)
    good_f, valid_f = _consistent(forward, backward)
    good_b, valid_b = _consistent(backward, forward)
    fraction = min(float(good_f.sum() / max(1, valid_f.sum())),
                   float(good_b.sum() / max(1, valid_b.sum())))
    translation = np.median(forward[good_f], axis=0) if good_f.any() else np.zeros(2, np.float32)
    reverse_translation = np.median(backward[good_b], axis=0) if good_b.any() else np.zeros(2, np.float32)
    textured = min(float(first.std()), float(second.std())) >= 2.
    reason = None
    if not textured or fraction < .55 or min(first.shape) < 8:
        reason = "unreliable_flow"
    elif float(np.linalg.norm(translation)) > .2 * min(first.shape):
        reason = "strong_camera_motion"
    # Histograms avoid interpreting a textured global pan as a scene cut.
    a = cv2.calcHist([first], [0], None, [32], [0, 256])
    b = cv2.calcHist([second], [0], None, [32], [0, 256])
    distance = float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    difference = float(np.abs(first.astype(np.float32) - second).mean())
    yy, xx = np.mgrid[:first.shape[0], :first.shape[1]].astype(np.float32)
    warped = cv2.remap(second.astype(np.float32), xx + forward[..., 0], yy + forward[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    photometric = float(np.abs(first.astype(np.float32) - warped)[valid_f].mean()) if valid_f.any() else difference
    cut = ((distance > .65 and difference > 25.) or (difference > 45. and fraction < .15)
           or (difference > 32. and photometric > 28.))
    if not cut and photometric > 20.:
        reason = "unreliable_flow"
    info = {"consistent_fraction": fraction, "global_translation_xy": translation.astype(float).tolist(),
            "histogram_distance": distance, "photometric_mae": photometric,
            "reliable": reason is None and not cut,
            "reason": "scene_cut" if cut else reason}
    if reason is not None or cut:
        zeros = np.zeros(first.shape, np.float32)
        return zeros, zeros.copy(), cut, info

    def density(flow, global_translation, good):
        residual = np.linalg.norm(flow - global_translation, axis=-1)
        # Absolute pixel threshold, never a clip-relative min/max normalization.
        value = np.clip((residual - .5) / 2.5, 0., 1.) * good
        return cv2.GaussianBlur(value.astype(np.float32), (0, 0), .75,
                                borderType=cv2.BORDER_REPLICATE)

    return density(forward, translation, good_f), density(backward, reverse_translation, good_b), cut, info


def _density_region(density: np.ndarray):
    """Choose a deterministic grid rectangle by normalized sum/mean density."""
    h, w = density.shape
    total, maximum = float(density.sum()), float(density.max())
    if maximum < .15 or total < 2.:
        return np.zeros((h, w), np.float32), None
    integral = cv2.integral(density)
    best, rectangle = -1., None
    for ratio in (.25, .40):
        rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
        ys = sorted(set(range(0, h - rh + 1, max(1, rh // 4))) | {h - rh})
        xs = sorted(set(range(0, w - rw + 1, max(1, rw // 4))) | {w - rw})
        for y in ys:
            for x in xs:
                amount = float(integral[y + rh, x + rw] - integral[y, x + rw]
                               - integral[y + rh, x] + integral[y, x])
                score = .6 * amount / total + .4 * amount / (rh * rw * maximum)
                if score > best:
                    best, rectangle = score, [x, y, x + rw, y + rh]
    core = density >= .25
    x1, y1, x2, y2 = rectangle
    core[y1:y2, x1:x2] = True
    core = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return feather(core, radius=4), rectangle


def build_motion_support(clip: np.ndarray, protection: np.ndarray, task: str) -> dict:
    """Return finite THW support and JSON provenance from source RGB only.

    ``cuts[0]`` is always True. OD accepts exactly one frame and bypasses flow.
    Untextured, inconsistent, strong-camera and static segments use the supplied
    semantic support exactly. Exact semantic cores are never weakened.
    """
    validate_clip(clip)
    if task not in ("ar", "od") or (task == "od" and len(clip) != 1):
        raise ValueError("unsupported task or multi-frame OD input")
    semantic = np.asarray(protection, dtype=np.float32)
    if semantic.shape != clip.shape[1:3] or not np.isfinite(semantic).all() or np.any((semantic < 0) | (semantic > 1)):
        raise ValueError("expected finite [0,1] HW semantic protection")
    count, h, w = clip.shape[:3]
    output = np.broadcast_to(semantic, (count, h, w)).copy()
    motion = np.zeros((count, h, w), np.float32)
    cuts = np.zeros(count, bool)
    cuts[0] = True
    metadata = {"task": task, "flow_source": None if task == "od" else "rgb_farneback_forward_backward",
                "camera_compensation": None if task == "od" else "median_translation",
                "region_score": {"normalized_sum": .6, "normalized_mean": .4},
                "fallback_reason": "od_single_frame" if task == "od" else None,
                "pairs": [], "segments": []}
    if task == "od":
        metadata["segments"].append({"start": 0, "end": 1, "mode": "semantic_fallback",
                                     "reason": "od_single_frame", "region_xyxy": None})
        return {"protection": output, "motion": motion, "cuts": cuts, "metadata": metadata}
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in clip]
    for index in range(1, count):
        before, after, cut, info = _flow_pair(gray[index - 1], gray[index])
        cuts[index] = cut
        info["previous_frame"], info["frame"] = index - 1, index
        metadata["pairs"].append(info)
        if not cut:
            motion[index - 1] = np.maximum(motion[index - 1], before)
            motion[index] = np.maximum(motion[index], after)
    starts = np.flatnonzero(cuts).tolist() + [count]
    for start, end in zip(starts[:-1], starts[1:]):
        pairs = metadata["pairs"][start:end - 1]
        reliable = sum(pair["reliable"] for pair in pairs)
        reason = None
        if end - start == 1:
            reason = "single_frame_segment"
        elif reliable < math.ceil(len(pairs) / 2):
            reason = "strong_camera_motion" if any(p["reason"] == "strong_camera_motion" for p in pairs) else "unreliable_flow"
        density = motion[start:end].mean(0)
        region, rectangle = _density_region(density) if reason is None else (np.zeros((h, w), np.float32), None)
        if reason is None and rectangle is None:
            reason = "no_residual_motion"
        if reason is None:
            output[start:end] = np.maximum(output[start:end], region)
        else:
            motion[start:end] = 0
        metadata["segments"].append({"start": start, "end": end,
                                     "mode": "motion_density" if reason is None else "semantic_fallback",
                                     "reason": reason, "reliable_pairs": reliable,
                                     "region_xyxy": rectangle})
    return {"protection": output, "motion": motion, "cuts": cuts, "metadata": metadata}
