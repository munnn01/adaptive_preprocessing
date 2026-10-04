import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import adaptive_vcm.evaluate as evaluation
import adaptive_vcm.train as training
from adaptive_vcm.codec import locate_ffmpeg
from adaptive_vcm.evaluate import ROOT


class TinyDetector:
    def __init__(self, name, device):
        self.name = name
        self.model = SimpleNamespace(
            transform=lambda images, targets: (SimpleNamespace(tensors=torch.stack(images)), targets),
            backbone=nn.Conv2d(3, 4, 3, padding=1).to(device).requires_grad_(False))
    def predict(self, clip):
        return {"boxes": np.array([[8., 8., 16., 16.]]),
                "scores": np.array([.9]), "labels": np.array([1])}


@pytest.mark.codec
def test_od_training_and_full_coco_aggregation_use_paired_real_streams(tmp_path, monkeypatch):
    if locate_ffmpeg() is None:
        pytest.skip("FFmpeg unavailable")
    source = np.random.default_rng(18).integers(80, 160, (1, 32, 48, 3), dtype=np.uint8)
    plan = [{"id": str(i), "image_id": i, "path": "fixture", "width": 48, "height": 32} for i in (1, 2)]
    meta = {"images": [{"id": i} for i in (1, 2)], "categories": [{"id": 1, "name": "object"}],
            "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [8, 8, 8, 8],
                             "area": 64, "iscrowd": 0} for i in (1, 2)]}
    cfg = json.loads((ROOT / "configs/v22_screen.json").read_text())
    cfg["qps"] = [30, 40, 50]
    cfg["od_candidates"] = ["identity", "background4"]
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(meta))
    for module in (evaluation, training):
        monkeypatch.setattr(module, "od_plan", lambda *args: (plan, meta))
        monkeypatch.setattr(module, "read_image", lambda *args: (source.copy(), (1, 1, 0, 0)))
        monkeypatch.setattr(module, "DetectionAnalyzer", TinyDetector)
    train_args = argparse.Namespace(config=config, task="od", root=tmp_path, annotations=annotations,
                                    count=2, steps=2, width=8, seed=17, lr=2e-4, rate_weight=.5,
                                    dual_lr=.02, out=tmp_path / "train")
    training.train(train_args)
    state = torch.load(train_args.out / "preprocessor_last.pth", weights_only=True)
    assert state["steps"] == 2 and state["task"] == "od"
    args = argparse.Namespace(config=config, task="od", root=tmp_path, annotations=annotations,
                               count=2, split="dev", codecs=["h264", "h265"], bootstrap=2,
                               checkpoint=train_args.out / "preprocessor_last.pth", save_streams=False,
                               out=tmp_path / "eval")
    summary = evaluation.run(args)
    for codec in args.codecs:
        report = summary["results"][codec]["resnet50"]
        assert report["curves"]["anchor"]["quality"] == pytest.approx([1, 1, 1])
        assert report["bd_rate_pct"] is None
        assert not report["screen_passes"]

