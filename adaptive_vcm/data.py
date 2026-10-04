"""Mount-independent plans, strict decoding and canonical task-label mapping."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image


def canon(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def fingerprint(ids: list[str]) -> str:
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source IDs")
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def partition(identifier: str) -> str:
    bucket = int(hashlib.md5(identifier.encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "dev" if bucket == 1 else "train"


def ar_plan(root: Path, split: str, count: int) -> tuple[list[dict], dict]:
    from torchvision.models.video import R3D_18_Weights
    categories = R3D_18_Weights.KINETICS400_V1.meta["categories"]
    mapping = {canon(name): i for i, name in enumerate(categories)}
    records, unknown = [], set()
    for path in sorted(root.rglob("*.mp4")):
        category = next((p.name for p in path.parents if canon(p.name) in mapping), None)
        if category is None:
            unknown.add(path.parent.name)
            continue
        identifier = f"{category}/{path.name}"
        if partition(identifier) == split:
            records.append({"id": identifier, "path": str(path), "label": mapping[canon(category)]})
    # Freeze a label-independent ordering before running an analyzer.
    records.sort(key=lambda r: hashlib.sha256(("adaptive-v22-screen:" + r["id"]).encode()).hexdigest())
    fingerprint([r["id"] for r in records])
    if count < 1 or len(records) < count:
        raise ValueError(f"requested {count} AR sources; only {len(records)} matched {split}")
    return records[:count], {"unmapped_folders": sorted(unknown), "available": len(records), "split": split}


def read_video(path: str, frames: int, size: int, stride: int = 2) -> np.ndarray:
    if frames < 1 or size < 2 or size % 2 or stride < 1:
        raise ValueError("invalid clip settings")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open source video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, total - frames * stride) // 2
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    picked = []
    try:
        while len(picked) < frames:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            picked.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
            for _ in range(stride - 1):
                if not cap.grab():
                    break
    finally:
        cap.release()
    if not picked:
        raise RuntimeError(f"source decoded zero frames: {path}")
    # Match historical short-clip convention; metadata records the frame recipe.
    while len(picked) < frames:
        picked.append(picked[-1].copy())
    return np.stack(picked)


def od_plan(root: Path, annotation_path: Path, split: str, count: int) -> tuple[list[dict], dict]:
    meta = json.loads(annotation_path.read_text(encoding="utf-8"))
    records = [{"id": str(im["id"]), "image_id": int(im["id"]),
                "path": str(root / im["file_name"]), "width": im["width"], "height": im["height"]}
               for im in meta["images"] if partition(f"coco2017/{im['id']}") == split]
    records.sort(key=lambda r: hashlib.sha256(("adaptive-v22-screen:" + r["id"]).encode()).hexdigest())
    fingerprint([r["id"] for r in records])
    if count < 1 or len(records) < count:
        raise ValueError(f"requested {count} OD images; only {len(records)} matched {split}")
    records = records[:count]
    missing = [r["id"] for r in records if not Path(r["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"selected COCO images missing: {missing[:5]}")
    return records, meta


def read_image(record: dict, size: int) -> tuple[np.ndarray, tuple[float, float, int, int]]:
    if size < 2 or size % 2:
        raise ValueError("image size must be positive and even")
    with Image.open(record["path"]) as source:
        source = source.convert("RGB")
        w, h = source.size
        if (w, h) != (record["width"], record["height"]):
            raise ValueError("COCO source geometry differs from annotations")
        scale = size / max(h, w)
        nw, nh = round(w * scale), round(h * scale)
        left, top = (size - nw) // 2, (size - nh) // 2
        image = Image.new("RGB", (size, size), (0, 0, 0))
        image.paste(source.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
        # Exact effective x/y resize scales matter when rounding differs.
        return np.asarray(image)[None].copy(), (nw / w, nh / h, left, top)
