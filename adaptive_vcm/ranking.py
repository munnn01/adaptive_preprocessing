"""TRAIN-supervised ranking of fixed pixel preprocessors, without identity CE.

The predictor only proposes actions. Real encoded bytes and the unchanged
encoder-teacher guard remain authoritative at deployment.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .preprocessing import validate_clip


TEACHER_COUNT = 2
CLASS_COUNT = 400
CONTEXT_DIM = 16 + TEACHER_COUNT * (2 * CLASS_COUNT + 12)
CONTEXT_SCHEMA = "teacher-probabilities-and-source-statistics-v1"


def _distribution(value):
    value = np.asarray(value, np.float64)
    if (value.shape != (CLASS_COUNT,) or not np.isfinite(value).all()
            or np.any(value < 0) or not np.isclose(value.sum(), 1., atol=1e-5)):
        raise ValueError("expected a finite 400-class probability distribution")
    return value / value.sum()


def build_rank_context(clip: np.ndarray, qp: int, codec: str, protection: np.ndarray,
                       source_predictions: Sequence[np.ndarray],
                       anchor_predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Identical label-free context for TRAIN and evaluation.

    Probability entries preserve category identity, but never receive a true
    label or held-out evaluator output. The anchor is the actual codec stream.
    """
    validate_clip(clip)
    if type(qp) is not int or not 0 <= qp <= 51 or codec not in ("h264", "h265"):
        raise ValueError("invalid codec/QP")
    mask = np.asarray(protection, np.float32)
    if (mask.shape != clip.shape[1:3] or not np.isfinite(mask).all()
            or np.any((mask < 0) | (mask > 1))):
        raise ValueError("invalid protection map")
    if len(source_predictions) != TEACHER_COUNT or len(anchor_predictions) != TEACHER_COUNT:
        raise ValueError("ranking requires the two registered encoder teachers")
    x = clip.astype(np.float32) / 255.
    motion = np.abs(np.diff(x, axis=0)).mean(axis=-1) if len(x) > 1 else np.zeros((1, *x.shape[1:3]))
    gradients = [np.abs(np.diff(x, axis=axis)).mean() for axis in (1, 2) if x.shape[axis] > 1]
    values = [qp / 51., float(codec == "h265"), math.log2(x.shape[1]) / 10.,
              math.log2(x.shape[2]) / 10., len(x) / 32.,
              *x.mean(axis=(0, 1, 2)), *x.std(axis=(0, 1, 2)),
              float(motion.mean()), float(np.percentile(motion, 95)),
              float(np.mean(gradients)) if gradients else 0.,
              float(mask.mean()), float((mask >= .95).mean())]
    for source, anchor in zip(source_predictions, anchor_predictions):
        s, a = _distribution(source), _distribution(anchor)
        top_s, top_a = np.sort(s)[-2:], np.sort(a)[-2:]
        log_s, log_a = np.log(np.maximum(s, 1e-12)), np.log(np.maximum(a, 1e-12))
        middle = (s + a) / 2
        js = .5 * np.sum(s * (log_s - np.log(np.maximum(middle, 1e-12))))
        js += .5 * np.sum(a * (log_a - np.log(np.maximum(middle, 1e-12))))
        values.extend(np.sqrt(s))
        values.extend(np.sqrt(a))
        values.extend([top_s[-1], top_a[-1], top_s[-1] - top_s[-2], top_a[-1] - top_a[-2],
                       -np.sum(s * log_s) / math.log(CLASS_COUNT),
                       -np.sum(a * log_a) / math.log(CLASS_COUNT),
                       np.clip(np.sum(s * (log_s - log_a)) / 20., 0., 1.),
                       np.clip(np.sum(a * (log_a - log_s)) / 20., 0., 1.),
                       js / math.log(2), float(s.argmax() == a.argmax()),
                       a[s.argmax()], s[a.argmax()]])
    context = np.asarray(values, np.float32)
    if context.shape != (CONTEXT_DIM,) or not np.isfinite(context).all():
        raise ValueError("invalid ranking context")
    return context


def measurement_targets(measurements: Sequence[dict], *, slack: float,
                        min_savings: float = .01):
    """Use every nonidentity action, separating teacher safety from rate gain."""
    if not measurements or measurements[0].get("name", measurements[0].get("profile")) != "identity":
        raise ValueError("identity measurement must be first")
    anchor = measurements[0]
    size = anchor["coded_bytes"]
    count = len(anchor["distances"])
    if (size <= 0 or not count or len(anchor["decisions"]) != count
            or not all(anchor["decisions"]) or not all(math.isfinite(d) for d in anchor["distances"])
            or not math.isfinite(slack) or slack < 0 or not 0 <= min_savings < 1):
        raise ValueError("invalid measured anchor/guard")
    safety, log_rate, eligible = [], [], []
    for row in measurements[1:]:
        rate = row["coded_bytes"] / size
        if row["coded_bytes"] <= 0 or not math.isfinite(rate):
            raise ValueError("invalid measured bytes")
        safe = (len(row["distances"]) == count and len(row["decisions"]) == count
                and all(row["decisions"])
                and all(math.isfinite(d) and d <= slack for d in row["distances"]))
        safety.append(float(safe))
        log_rate.append(math.log(rate))
        eligible.append(float(safe and rate <= 1 - min_savings))
    return tuple(np.asarray(values, np.float32) for values in (safety, log_rate, eligible))


def static_action_order(safety: np.ndarray, log_rate: np.ndarray,
                        min_savings: float = .01) -> list[int]:
    """Matched-budget comparator, frozen using TRAIN mean guarded savings."""
    if safety.ndim != 2 or safety.shape != log_rate.shape or len(safety) < 1:
        raise ValueError("inconsistent measured target matrices")
    if not np.isfinite(safety).all() or not np.isfinite(log_rate).all() or not 0 <= min_savings < 1:
        raise ValueError("invalid measured targets")
    saving = 1 - np.exp(log_rate)
    gains = np.where((safety == 1) & (saving >= min_savings), saving, 0.).mean(axis=0)
    # Stable action-index tie order avoids any preference for a learned family.
    return [int(i + 1) for i in np.argsort(-gains, kind="stable")]


class RankPreprocessor(nn.Module):
    schema = "adaptive-vcm-ranking-v4"
    candidate_name = "learned_rank"
    context_dim = CONTEXT_DIM

    def __init__(self, width: int = 64, task: str = "ar", action_names=None):
        super().__init__()
        if task != "ar" or type(width) is not int or width < 1:
            raise ValueError("ranking supports AR with a positive model width")
        if action_names is None:
            from .task_bank import ACTION_NAMES
            action_names = ACTION_NAMES
        names = tuple(action_names)
        if len(names) < 2 or names[0] != "identity" or len(set(names)) != len(names):
            raise ValueError("action order must contain identity followed by unique actions")
        self.width, self.task, self.action_names = width, task, names
        self.encoder = nn.Sequential(nn.Linear(CONTEXT_DIM, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.safety_head = nn.Linear(width, len(names) - 1)
        self.rate_head = nn.Linear(width, len(names) - 1)
        # A small positive-saving initial rate keeps pairwise gain gradients
        # active. Measurements supervise this value; it is never a byte proxy.
        nn.init.zeros_(self.rate_head.weight)
        nn.init.constant_(self.rate_head.bias, -.03)
        self.register_buffer("safety_log_weight", torch.zeros(len(names) - 1))
        self.static_action_order = list(range(1, len(names)))

    def forward(self, context):
        context = torch.as_tensor(context, device=self.rate_head.weight.device,
                                  dtype=self.rate_head.weight.dtype)
        if context.ndim == 1:
            context = context[None]
        if context.ndim != 2 or context.shape[1] != CONTEXT_DIM or not torch.isfinite(context).all():
            raise ValueError("invalid ranking context tensor")
        hidden = self.encoder(context)
        return self.safety_head(hidden), self.rate_head(hidden)

    def scores(self, safety_logits, log_rate):
        # Undo the fixed BCE class weight when predicting a probability; without
        # this correction rare unsafe actions would look spuriously plausible.
        probability = (safety_logits - self.safety_log_weight).sigmoid()
        return probability * (1 - log_rate.clamp(-10., 10.).exp()).clamp(min=0.)

    @torch.no_grad()
    def rank(self, context, top_k: int = 3) -> list[int]:
        if type(top_k) is not int or not 1 <= top_k < len(self.action_names):
            raise ValueError("invalid proposal budget")
        logits, log_rate = self(context)
        if len(logits) != 1:
            raise ValueError("rank expects one operating point")
        scores = self.scores(logits, log_rate)[0].cpu().numpy()
        # Always offer K actions. Actual guard can reject all and choose identity.
        return [int(i + 1) for i in np.argsort(-scores, kind="stable")[:top_k]]

    def render(self, clip, protection, qp: int, action_index: int):
        from .task_bank import build_task_bank
        if type(action_index) is not int or not 1 <= action_index < len(self.action_names):
            raise ValueError("only nonidentity action proposals can be rendered")
        bank = build_task_bank(clip, protection, qp)
        if tuple(c.name for c in bank) != self.action_names:
            raise ValueError("checkpoint action order differs from pixel bank")
        return bank[action_index]


def ranking_loss(model: RankPreprocessor, context, safety, log_rate,
                 positive_weight, *, min_savings: float = .01):
    """Balanced safety BCE, measured log-rate regression, useful-action ranking."""
    logits, predicted_rate = model(context)
    if safety.shape != logits.shape or log_rate.shape != logits.shape:
        raise ValueError("inconsistent all-action supervision")
    safety_loss = F.binary_cross_entropy_with_logits(logits, safety, pos_weight=positive_weight)
    rate_loss = F.smooth_l1_loss(predicted_rate, log_rate, beta=.1)
    saving = 1 - log_rate.exp()
    utility = torch.where((safety > .5) & (saving >= min_savings), saving, torch.zeros_like(saving))
    pairs = utility[:, :, None] - utility[:, None, :] >= min_savings
    # Log safety plus negative measured log rate supplies smooth gradients even
    # for initially unsafe/overhead proposals. Inference uses expected savings.
    preference = F.logsigmoid(logits - model.safety_log_weight) - predicted_rate
    margins = preference[:, :, None] - preference[:, None, :]
    pair_loss = F.softplus(-margins[pairs]).mean() if pairs.any() else logits.sum() * 0.
    total = safety_loss + 2. * rate_loss + .25 * pair_loss
    return total, {"safety_loss": safety_loss.detach(), "rate_loss": rate_loss.detach(),
                   "pairwise_loss": pair_loss.detach()}


def load_rank_preprocessor(state: dict, task: str = "ar") -> RankPreprocessor:
    from .task_bank import ACTION_NAMES
    if (state.get("schema") != RankPreprocessor.schema or task != "ar" or state.get("task") != task
            or state.get("context_schema") != CONTEXT_SCHEMA
            or state.get("context_dim") != CONTEXT_DIM or tuple(state.get("action_names", ())) != ACTION_NAMES):
        raise ValueError("incompatible ranking checkpoint/context/bank")
    model = RankPreprocessor(state["width"], task, state["action_names"])
    model.load_state_dict(state["model"], strict=True)
    order = state["static_action_order"]
    if sorted(order) != list(range(1, len(ACTION_NAMES))):
        raise ValueError("invalid TRAIN static action order")
    model.static_action_order = list(order)
    return model.eval()
