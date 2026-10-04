import argparse
import json
import shutil

import numpy as np
import pytest
import torch

import adaptive_vcm.evaluate as evaluation
import adaptive_vcm.train as training
from adaptive_vcm.evaluate import ROOT
from adaptive_vcm.codec import locate_ffmpeg


class TinyAction:
    """Differentiable fixture; its predictions are not task-accuracy evidence."""
    def __init__(self, name, device):
        self.name, self.device = name, device

    def tensor(self, clip):
        return torch.from_numpy(clip.copy()).to(self.device).float().permute(3, 0, 1, 2)[None] / 255

    def logits(self, x):
        mean = x.mean((1, 2, 3, 4))
        return torch.stack([mean * 2, 1 - mean, mean * 0], 1)

    def probabilities(self, clip):
        return self.logits(self.tensor(clip)).softmax(-1)[0].detach().cpu().numpy()

    def saliency(self, clip):
        return np.zeros(clip.shape[1:3], np.float32)


@pytest.mark.codec
def test_paired_runner_and_training_checkpoint_with_real_codecs(tmp_path, monkeypatch):
    if locate_ffmpeg() is None:
        pytest.skip("FFmpeg unavailable")
    source = np.random.default_rng(3).integers(80, 160, (1, 32, 48, 3), dtype=np.uint8)
    source = np.repeat(source, 4, 0)
    plan = [{"id": "a/1.mp4", "path": "fixture", "label": 0},
            {"id": "a/2.mp4", "path": "fixture", "label": 0}]
    cfg = json.loads((ROOT / "configs/v22_screen.json").read_text())
    cfg["qps"] = [30, 40, 50]
    cfg["ar_candidates"] = ["identity", "protected_mild"]
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    for module in (evaluation, training):
        monkeypatch.setattr(module, "ar_plan", lambda *args: (plan, {}))
        monkeypatch.setattr(module, "read_video", lambda *args: source.copy())
        monkeypatch.setattr(module, "ActionAnalyzer", TinyAction)
    train_args = argparse.Namespace(config=config, task="ar", root=tmp_path, annotations=None,
                                    count=2, steps=2, width=8, seed=17, lr=2e-4, rate_weight=.5,
                                    dual_lr=.02, out=tmp_path / "train")
    training.train(train_args)
    state = torch.load(train_args.out / "preprocessor_last.pth", weights_only=True)
    assert state["steps"] == 2 and state["train_ids_sha256"]
    log = [json.loads(s) for s in (train_args.out / "train.jsonl").read_text().splitlines()]
    assert len(log) == 2 and all(r["gradient_norm"] > 0 for r in log)
    assert all(r["actual_preprocessed_bpp"] > 0 for r in log)
    args = argparse.Namespace(config=config, task="ar", root=tmp_path, annotations=None,
                               count=2, split="dev", codecs=["h264", "h265"], bootstrap=5,
                               checkpoint=train_args.out / "preprocessor_last.pth", save_streams=True,
                               out=tmp_path / "eval")
    summary = evaluation.run(args)
    assert summary["both_codecs_evaluated"] and not summary["target_confirmed"]
    for codec in args.codecs:
        rows = [json.loads(s) for s in (args.out / f"{codec}_rows.jsonl").read_text().splitlines()]
        assert len(rows) == 2 * 2 * 3
        for anchor, selected in zip(rows[::2], rows[1::2]):
            assert anchor["source_sha256"] == selected["source_sha256"]
            assert selected["coded_bytes"] <= anchor["coded_bytes"]
    with pytest.raises(ValueError, match="output must be empty"):
        evaluation.run(args)

