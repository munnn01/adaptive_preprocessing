"""Frozen V31 preprocessing actions and explicit OD coordinate transforms.

This layer renders transmitted RGB and its recipe; encoding and feasibility
measurement belong to the caller. Registry aliases survive pixel deduplication.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib

import cv2
import numpy as np

from ..conditional_learned import profile_candidates as conditional_profiles
from ..motion_learned import PROFILE_NAMES, profile_candidates, validate_support
from ..preprocessing import make_candidates, validate_clip
from .protocol import CODECS, QPS
from .transport import Recipe


@dataclass(frozen=True)
class Action:
    name: str
    task: str
    size: int | None
    kind: str
    profile: str | None
    repeat_factor: int


_CONTROLS = {
    "ar": ("identity", "area112", "area96", "area112_up", "blur20", "blur40",
           "protected_mild", "protected_strong", "protected_temporal"),
    "od": ("identity", "background2", "background4", "background8", "background12"),
}
_CANONICAL = {"ar": (16, 128, 128), "od": (1, 320, 320)}
_TEMPORAL_PROFILE = "motion_block_lowpass_100"


def action_registry(task: str, arm: str) -> tuple[Action, ...]:
    """Stable descriptor order: old controls, V29-A profiles, then additions."""
    if task not in _CONTROLS or arm not in ("a", "b", "c"):
        raise ValueError("unsupported V31 action task/arm")
    actions = [Action(name, task, {"area112": 112, "area96": 96}.get(name),
                      "control", None, 1) for name in _CONTROLS[task]]
    actions.extend(Action(name, task, None, "profile", name, 1) for name in PROFILE_NAMES)
    if task == "od":
        actions.extend(Action(f"area{size}", task, size, "resize", None, 1)
                       for size in (256, 224, 192))
    if arm in ("b", "c"):
        if task == "ar":
            actions.extend(Action(f"drop2_{size}", task, size, "drop2", None, 2)
                           for size in (128, 112, 96))
            actions.extend(Action(f"temporal_{size}", task, size, "temporal",
                                  _TEMPORAL_PROFILE, 1) for size in (128, 112, 96))
        else:
            actions.extend(Action(f"background8_area{size}", task, size,
                                  "background_resize", "background8", 1)
                           for size in (256, 224, 192))
    return tuple(actions)


def _shape(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(
            type(v) is not int or v <= 0 for v in value):
        raise ValueError(f"{name} must be positive (height,width)")
    return tuple(value)


def _geometry(source_transform, original_shape):
    original = _shape(original_shape, "original_shape")
    if not isinstance(source_transform, (tuple, list)) or len(source_transform) != 6:
        raise ValueError("source_transform requires sx,sy,left,top,canonical_h,canonical_w")
    sx, sy, left, top, height, width = source_transform
    canonical = _shape((height, width), "canonical canvas")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
           for v in (sx, sy, left, top)) or sx <= 0 or sy <= 0 or left < 0 or top < 0:
        raise ValueError("invalid source transform scales/offsets")
    if left + sx * original[1] > width + 1e-6 or top + sy * original[0] > height + 1e-6:
        raise ValueError("source transform content exceeds canonical canvas")
    return tuple(source_transform), original, canonical


def execute_action(source: dict, action: Action, qp: int, support: dict) -> dict:
    """Render a primary action without modifying source pixels or cached support.

    Source requires task, uint8 THW3 rgb, codec, Fraction duration (AR), padded,
    source_fps, and HW control_protection for controls/background compositions.
    OD additionally requires its explicit six-value source_transform and
    original_shape. AR may omit geometry metadata because its source is square.
    An unavailable action returns no RGB/recipe and must never be encoded.
    """
    return _execute_action(source, action, qp, support, {})


def execute_actions(source: dict, registry: tuple[Action, ...], qp: int, support: dict) -> list[dict]:
    """Validated single-action logic with banks scoped to this source/condition.

    Each authoritative renderer runs once. No pixels can be injected by a caller;
    aliases retain independent owned results and their individual wire recipes.
    """
    if not isinstance(registry, tuple) or not registry or len(set(registry)) != len(registry):
        raise ValueError("expected nonempty unique frozen action tuple")
    banks = {}
    return [_execute_action(source, action, qp, support, banks) for action in registry]


def _execute_action(source, action, qp, support, banks):
    if not isinstance(source, dict) or not isinstance(action, Action):
        raise ValueError("expected source dictionary and frozen Action")
    task = source.get("task")
    if task not in _CANONICAL or action.task != task or action not in action_registry(task, "b"):
        raise ValueError("action is outside the frozen task registry")
    if type(qp) is not int or qp not in QPS or source.get("codec") not in CODECS:
        raise ValueError("unsupported frozen codec/QP condition")
    rgb = source.get("rgb")
    if not isinstance(rgb, np.ndarray):
        raise ValueError("source RGB must be an array")
    validate_clip(rgb)
    if rgb.shape[:3] != _CANONICAL[task]:
        raise ValueError("primary source frame count/geometry differs from frozen recipe")
    validate_support(rgb, support, task)
    if type(source.get("padded")) is not bool:
        raise ValueError("source padding must be explicitly recorded")
    height, width = rgb.shape[1:3]
    transform = source.get("source_transform", (1., 1., 0, 0, height, width) if task == "ar" else None)
    original = source.get("original_shape", (height, width) if task == "ar" else None)
    transform, original, canonical = _geometry(transform, original)
    if canonical != (height, width):
        raise ValueError("source transform canonical canvas differs from RGB geometry")
    indices = tuple(range(0, len(rgb), action.repeat_factor))
    result = {"available": False, "reason": None, "rgb": None, "recipe": None,
              "source_shape": (height, width), "coded_shape": None,
              "source_transform": transform, "original_shape": original,
              "sample_indices": indices, "rgb_sha256": None}
    duration = source.get("duration") if task == "ar" else Fraction(1, 25)
    if not isinstance(duration, Fraction) or duration <= 0:
        result["reason"] = "unknown_source_duration"
        return result
    if action.kind == "drop2":
        if source["padded"]:
            result["reason"] = "padded_source"
            return result
        fps = source.get("source_fps")
        if not isinstance(fps, Fraction) or fps <= 0:
            result["reason"] = "unknown_source_fps"
            return result
    if action.kind in ("control", "background_resize"):
        protection = np.asarray(source.get("control_protection"), dtype=np.float32)
        if protection.shape != (height, width) or not np.isfinite(protection).all() or np.any(
                (protection < 0) | (protection > 1)):
            raise ValueError("controls require finite [0,1] HW control_protection")
        control = action.name if action.kind == "control" else "background8"
        if "control" not in banks:
            banks["control"] = {c.name: c.clip for c in make_candidates(rgb, protection, task, qp)}
        pixels = banks["control"][control]
    elif action.kind == "profile":
        if "profile" not in banks:
            owned_support = {key: np.array(support[key], copy=True) for key in ("protection", "motion", "cuts")}
            banks["profile"] = {c.name: c.clip for c in profile_candidates(rgb, owned_support, task, qp, variant="a")}
        pixels = banks["profile"][action.profile]
    elif action.kind == "temporal":
        # V30-C's third slot at full strength; no strength sweep or drop2 mix.
        if "temporal" not in banks:
            owned_support = {key: np.array(support[key], copy=True) for key in ("protection", "motion", "cuts")}
            banks["temporal"] = {c.name: c.clip for c in conditional_profiles(rgb, owned_support, task, qp, variant="c")}
        pixels = banks["temporal"][action.profile]
    else:
        pixels = rgb.copy()
    if action.kind != "control" and action.size is not None and pixels.shape[1:3] != (action.size, action.size):
        pixels = np.stack([cv2.resize(frame, (action.size, action.size), interpolation=cv2.INTER_AREA)
                           for frame in pixels])
    pixels = pixels[::action.repeat_factor].copy()
    count, coded_h, coded_w = pixels.shape[:3]
    recipe = Recipe(task, source["codec"], coded_w, coded_h, count, len(rgb),
                    duration.numerator, duration.denominator, action.repeat_factor)
    return {**result, "available": True, "rgb": pixels, "recipe": recipe,
            "coded_shape": (coded_h, coded_w),
            "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}


def map_detections(prediction: dict, coded_shape: tuple, source_transform: tuple,
                   original_shape: tuple) -> dict:
    """Copy detections into canonical guard and original COCO coordinates.

    Shapes are (height,width). The explicit canonical canvas is essential when
    rounded letterboxing leaves different padding on opposite edges.
    """
    coded_h, coded_w = _shape(coded_shape, "coded_shape")
    transform, original, canonical = _geometry(source_transform, original_shape)
    sx, sy, left, top, canonical_h, canonical_w = transform
    if not isinstance(prediction, dict) or not {"boxes", "scores", "labels"} <= prediction.keys():
        raise ValueError("detections require boxes, scores and labels")
    arrays = {key: np.asarray(prediction[key]) for key in ("boxes", "scores", "labels")}
    boxes = arrays["boxes"]
    # JSON has no shape metadata: legitimate (0,4) detector arrays become [].
    # Normalize only that representation, not malformed Nx0/Nx3 arrays.
    if boxes.ndim == 1 and boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("boxes must be finite Nx4")
    if any(arrays[key].shape != (len(boxes),) for key in ("scores", "labels")):
        raise ValueError("detection class/score count mismatch")
    mapped = {}
    canonical_boxes = boxes.astype(np.float64, copy=True)
    canonical_boxes *= np.array([canonical_w / coded_w, canonical_h / coded_h] * 2)
    canonical_boxes[:, [0, 2]] = np.clip(canonical_boxes[:, [0, 2]], 0, canonical_w)
    canonical_boxes[:, [1, 3]] = np.clip(canonical_boxes[:, [1, 3]], 0, canonical_h)
    original_boxes = (canonical_boxes - np.array([left, top] * 2)) / np.array([sx, sy] * 2)
    original_boxes[:, [0, 2]] = np.clip(original_boxes[:, [0, 2]], 0, original[1])
    original_boxes[:, [1, 3]] = np.clip(original_boxes[:, [1, 3]], 0, original[0])
    for name, coordinates in (("canonical", canonical_boxes), ("original", original_boxes)):
        mapped[name] = {"boxes": coordinates, "scores": arrays["scores"].copy(),
                        "labels": arrays["labels"].copy()}
    return mapped
