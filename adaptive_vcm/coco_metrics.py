"""COCO detection metrics and paired image-level uncertainty estimates.

The object-detection probes and the engine evaluator share this module so box
conversion, COCO evaluation and confidence intervals cannot silently diverge.
Bootstrap samples are drawn *with replacement*.  Repeated source images are
assigned fresh image/annotation ids, preserving their multiplicity in COCOeval.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .legacy_bd import bd_rate


def coco_box(box) -> list[float]:
    """Convert a torchvision ``xyxy`` box to COCO ``xywh``."""
    vals = box.tolist() if hasattr(box, "tolist") else list(box)
    x1, y1, x2, y2 = (float(v) for v in vals[:4])
    return [x1, y1, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)]


def coco_map(results, gt_by_id, image_ids, ann_meta):
    """Return COCO mAP@[.5:.95] and mAP@.5 for the supplied predictions."""
    from pycocotools import mask as _mask  # noqa: F401
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": ann_meta.get("info", {}),
        "images": [{"id": int(i)} for i in image_ids],
        "annotations": [a for i in image_ids for a in gt_by_id[int(i)]],
        "categories": ann_meta["categories"],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    if not results:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, dt, "bbox")
    ev.params.imgIds = [int(i) for i in image_ids]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def _remap_bootstrap_sample(
    sampled_ids: Sequence[int],
    gt_by_id: Mapping[int, Sequence[dict]],
) -> tuple[dict[int, list[dict]], list[int], list[tuple[int, int]]]:
    """Give every bootstrap occurrence a unique COCO image/annotation id.

    Returns ``(remapped_gt, new_ids, [(source_id, new_id), ...])``.  The final
    mapping is also used to duplicate predictions and bitrate observations.
    """
    remapped: dict[int, list[dict]] = {}
    mapping: list[tuple[int, int]] = []
    ann_id = 1
    for pos, source_id in enumerate(sampled_ids, start=1):
        new_id = pos
        mapping.append((int(source_id), new_id))
        anns = []
        for ann in gt_by_id[int(source_id)]:
            copied = dict(ann)
            copied["id"] = ann_id
            copied["image_id"] = new_id
            anns.append(copied)
            ann_id += 1
        remapped[new_id] = anns
    return remapped, list(range(1, len(sampled_ids) + 1)), mapping


def paired_bootstrap_detection_bd(
    records: Mapping[str, Mapping[tuple[str, int], Mapping[int, tuple[float, list[dict]]]]],
    *,
    codec: str,
    qps: Sequence[int],
    gt_by_id: Mapping[int, Sequence[dict]],
    image_ids: Sequence[int],
    ann_meta: Mapping,
    arms: Iterable[str],
    n_boot: int,
    seed: int = 0,
    metric_fn: Callable = coco_map,
) -> dict[str, dict[str, float | int | str | None]]:
    """Paired, with-replacement image bootstrap of BD-rate against ``anchor``.

    The same sampled image occurrences are used for the anchor and every test
    arm.  Each arm receives its own distribution and confidence interval; arm
    draws are never pooled together.
    """
    if n_boot <= 0:
        return {}
    ids = [int(i) for i in image_ids]
    if len(ids) < 2:
        return {arm: {"lo": None, "hi": None, "p_lt_zero": None,
                      "n_draws": 0, "method": "paired_image_bootstrap"}
                for arm in arms}

    arms = list(arms)
    draws: dict[str, list[float]] = {arm: [] for arm in arms}
    rng = random.Random(seed)
    for _ in range(int(n_boot)):
        sampled = [rng.choice(ids) for _ in ids]
        gt_draw, draw_ids, mapping = _remap_bootstrap_sample(sampled, gt_by_id)
        curves: dict[str, tuple[list[float], list[float]]] = {}
        for arm in ["anchor", *arms]:
            rates, metrics = [], []
            for qp in qps:
                slot = records[arm][(codec, int(qp))]
                rates.append(float(np.mean([slot[src][0] for src, _ in mapping])))
                preds = []
                for src, new_id in mapping:
                    for pred in slot[src][1]:
                        copied = dict(pred)
                        copied["image_id"] = new_id
                        preds.append(copied)
                metrics.append(float(metric_fn(preds, gt_draw, draw_ids, ann_meta)[0]))
            curves[arm] = (rates, metrics)
        anchor_rate, anchor_metric = curves["anchor"]
        for arm in arms:
            rate, metric = curves[arm]
            value = bd_rate(anchor_rate, anchor_metric, rate, metric)
            if np.isfinite(value):
                draws[arm].append(float(value))

    out = {}
    for arm in arms:
        values = np.asarray(draws[arm], dtype=np.float64)
        out[arm] = {
            "lo": float(np.percentile(values, 2.5)) if values.size else None,
            "hi": float(np.percentile(values, 97.5)) if values.size else None,
            "p_lt_zero": float(np.mean(values < 0.0)) if values.size else None,
            "n_draws": int(values.size),
            "method": "paired_image_bootstrap_with_replacement",
        }
    return out


__all__ = [
    "coco_box",
    "coco_map",
    "paired_bootstrap_detection_bd",
]

