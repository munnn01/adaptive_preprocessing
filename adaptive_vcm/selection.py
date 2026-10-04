"""Typed label-free observations; choose one real bitstream with task guards."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Observation:
    name: str
    coded_bytes: int
    # One finite relative task distance per encoder teacher.
    distances: tuple[float, ...]
    preserves_decision: tuple[bool, ...]


def ar_guard(source: np.ndarray, anchor: np.ndarray, trial: np.ndarray,
             confidence: float = .6) -> tuple[float, bool]:
    """KL(source||trial) relative to codec-only; no target class is accepted."""
    vectors = [np.asarray(p, np.float64) for p in (source, anchor, trial)]
    if any(p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or np.any(p < 0)
           or not np.isclose(p.sum(), 1, atol=1e-5) for p in vectors):
        raise ValueError("invalid probability vector")
    if len({len(p) for p in vectors}) != 1 or not 0 <= confidence <= 1:
        raise ValueError("inconsistent probability vectors/confidence")
    s, a, c = vectors
    def kl(p):
        return float(np.sum(s * (np.log(np.maximum(s, 1e-12)) - np.log(np.maximum(p, 1e-12)))))
    decision = ((s.max() < confidence or s.argmax() == c.argmax())
                and (a.max() < confidence or a.argmax() == c.argmax()))
    return kl(c) - kl(a), bool(decision)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    low, high = np.maximum(box[:2], boxes[:, :2]), np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=1)
    area_a = np.maximum(box[2:] - box[:2], 0).prod()
    area_b = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-8)


def detection_distance(source: dict, trial: dict, threshold: float = .25) -> float:
    """Score-weighted box/class preservation with unique matches, not COCO mAP."""
    scores = np.asarray(source["scores"], np.float64)
    boxes = np.asarray(source["boxes"], np.float64).reshape(-1, 4)
    labels = np.asarray(source["labels"])
    other_boxes = np.asarray(trial["boxes"], np.float64).reshape(-1, 4)
    other_labels = np.asarray(trial["labels"])
    other_scores = np.asarray(trial["scores"], np.float64)
    if len(scores) != len(boxes) or len(labels) != len(scores) or len(other_scores) != len(other_boxes) or len(other_labels) != len(other_boxes):
        raise ValueError("inconsistent detections")
    if any(not np.isfinite(x).all() for x in (scores, boxes, other_boxes, other_scores)):
        raise ValueError("nonfinite detections")
    chosen = np.flatnonzero(scores >= threshold)
    if not len(chosen):
        return math.inf  # no reliable source object: caller falls back to identity
    chosen = chosen[np.argsort(-scores[chosen], kind="stable")]
    used: set[int] = set()
    loss = 0.
    for i in chosen:
        eligible = np.array([j for j in range(len(other_boxes))
                             if j not in used and other_labels[j] == labels[i]], dtype=int)
        quality = 0.
        if len(eligible):
            similarity = _iou(boxes[i], other_boxes[eligible]) * np.minimum(other_scores[eligible] / max(scores[i], 1e-8), 1)
            best = int(similarity.argmax())
            quality = float(similarity[best])
            if quality > 0:
                used.add(int(eligible[best]))
        loss += scores[i] * (1 - quality)
    return float(loss / scores[chosen].sum())


def relative_guard(task, source_predictions, anchor_predictions, trial_predictions, cfg):
    """The same label-free constraints serve training probes and selection."""
    if not (len(source_predictions) == len(anchor_predictions) == len(trial_predictions)):
        raise ValueError("inconsistent teacher counts")
    if task == "ar":
        pairs = [ar_guard(s, a, c, cfg["ar_confidence"])
                 for s, a, c in zip(source_predictions, anchor_predictions, trial_predictions)]
        distances = tuple(p[0] for p in pairs)
        decisions = tuple(bool(p[1] and (not cfg.get("ar_require_anchor_decision", False)
                                    or a.argmax() == c.argmax()))
                          for p, a, c in zip(pairs, anchor_predictions, trial_predictions))
        return distances, decisions
    if task != "od" or len(source_predictions) != 1:
        raise ValueError("unsupported task/teacher count")
    s, a, c = source_predictions[0], anchor_predictions[0], trial_predictions[0]
    distance = detection_distance(s, c, cfg["od_score_threshold"]) - detection_distance(s, a, cfg["od_score_threshold"])
    return (distance,), (True,)


def select(observations: list[Observation], slack: float, min_savings: float = .01) -> int:
    """Safe fallback; malformed/missing/nonfinite observations fail closed."""
    if not observations or observations[0].name != "identity":
        raise ValueError("identity must be first")
    if not math.isfinite(slack) or slack < 0 or not 0 <= min_savings < 1:
        raise ValueError("invalid policy")
    anchor = observations[0]
    if anchor.coded_bytes <= 0 or not anchor.distances or len(anchor.distances) != len(anchor.preserves_decision):
        raise ValueError("invalid anchor")
    if any(not math.isfinite(d) for d in anchor.distances) or not all(anchor.preserves_decision):
        raise ValueError("invalid anchor guard")
    feasible = [0]
    for i, candidate in enumerate(observations[1:], 1):
        if candidate.coded_bytes <= 0 or candidate.coded_bytes > anchor.coded_bytes * (1 - min_savings):
            continue
        if len(candidate.distances) != len(anchor.distances) or len(candidate.preserves_decision) != len(anchor.distances):
            continue
        if all(math.isfinite(d) and d <= slack for d in candidate.distances) and all(candidate.preserves_decision):
            feasible.append(i)
    return min(feasible, key=lambda i: (observations[i].coded_bytes, i))
