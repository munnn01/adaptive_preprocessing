"""Train the optional adaptive multiscale blend, using TRAIN sources only.

Real x264/x265 decoded pixels are used in the forward pass, with an explicitly
approximate straight-through backward pass. A calibrated gradient/temporal
complexity prior supplies rate gradients; its value is never reported as a
measured bitrate. The final checkpoint requires a separate actual-codec screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .analyzers import ActionAnalyzer, DetectionAnalyzer
from .codec import StandardCodec, reference_bpp
from .data import ar_plan, od_plan, fingerprint, read_video, read_image
from .evaluate import ROOT, code_manifest, validate_config, write_json
from .learned import AdaptiveBlendPreprocessor
from .preprocessing import action_protection, boxes_to_mask, normalize_map


def rate_prior(video: torch.Tensor) -> torch.Tensor:
    """Spatial plus causal temporal activity; not a codec entropy estimator."""
    components = []
    for axis in (2, 3, 4):
        if video.shape[axis] > 1:
            components.append(torch.log1p(64 * torch.diff(video, dim=axis).abs()).mean())
    return torch.stack(components).mean() if components else video.sum() * 0 + 1e-8


def real_codec_ste(video: torch.Tensor, codec: StandardCodec):
    if video.shape[0] != 1:
        raise ValueError("real-codec training currently uses batch size one")
    pixels = video.detach()[0].permute(1, 2, 3, 0).mul(255).round().clamp(0, 255).byte().cpu().numpy()
    result = codec.roundtrip(pixels)
    reconstructed = torch.from_numpy(result.decoded.copy()).to(video).permute(3, 0, 1, 2)[None] / 255
    # Forward equals the decoded bitstream exactly; backward is identity STE.
    return video + (reconstructed - video).detach(), reference_bpp(result.coded_bytes, pixels.shape)


def od_features(teacher, video):
    x = [video[0, :, 0]]
    images, _ = teacher.model.transform(x, None)
    features = teacher.model.backbone(images.tensors)
    return list(features.values()) if isinstance(features, dict) else [features]


def feature_distance(reference, trial):
    return torch.stack([(a.detach() - b).square().mean() / a.detach().square().mean().clamp_min(1e-6)
                        for a, b in zip(reference, trial)]).mean()


def train(args) -> dict:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(cfg)
    if args.steps < 1 or args.count < 1 or args.lr <= 0 or args.rate_weight < 0 or args.dual_lr < 0:
        raise ValueError("invalid training settings")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("training output must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.task == "ar":
        plan, _ = ar_plan(args.root, "train", args.count)
        teachers = [ActionAnalyzer(name, device) for name in cfg["ar_teachers"]]
    else:
        if args.annotations is None:
            raise ValueError("COCO annotations required for locating training images")
        plan, _ = od_plan(args.root, args.annotations, "train", args.count)
        teachers = [DetectionAnalyzer(cfg["od_teacher"], device)]
    model = AdaptiveBlendPreprocessor(args.width).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    groups = [(codec, qp) for codec in ("h264", "h265") for qp in cfg["qps"]]
    # Separate dual/calibration state prevents a difficult QP dominating others.
    duals = {group: 10. for group in groups}
    calibration = {group: None for group in groups}
    manifest = {"schema": "adaptive-vcm-training-v1", "task": args.task, "seed": args.seed,
                "steps": args.steps, "width": args.width, "lr": args.lr,
                "rate_weight": args.rate_weight, "dual_lr": args.dual_lr,
                "train_ids": [r["id"] for r in plan], "train_ids_sha256": fingerprint([r["id"] for r in plan]),
                "config": cfg, "code": code_manifest(), "device": device,
                "source": "fresh initialization; final-LAST only; no dev/test checkpoint selection",
                "gradient_limitations": "Identity STE is approximate. Calibrated rate_prior is a complexity prior, not an entropy model. OD uses encoder backbone-feature preservation, not differentiable COCO mAP."}
    write_json(args.out / "training_manifest.json", manifest)
    order, schedule = [], []
    for step in range(args.steps):
        if step % len(plan) == 0:
            order = rng.permutation(len(plan)).tolist()
        if step % len(groups) == 0:
            schedule = rng.permutation(len(groups)).tolist()
        item = plan[order[step % len(plan)]]
        group = groups[schedule[step % len(groups)]]
        codec_name, qp = group
        clip = (read_video(item["path"], cfg["frames"], cfg["ar_size"], cfg["temporal_stride"]) if args.task == "ar"
                else read_image(item, cfg["od_size"])[0])
        source = torch.from_numpy(clip.copy()).to(device).float().permute(3, 0, 1, 2)[None] / 255
        if args.task == "ar":
            saliency = np.maximum.reduce([normalize_map(m.saliency(clip)) for m in teachers])
            mask = action_protection(clip, saliency)
        else:
            predicted = teachers[0].predict(clip)
            mask = boxes_to_mask(*clip.shape[1:3], predicted["boxes"][predicted["scores"] >= cfg["od_score_threshold"]])
            if not mask.any():
                mask = np.ones(clip.shape[1:3], np.float32)
        protected = torch.from_numpy(mask.copy()).to(source)[None, None, None].expand(1, 1, len(clip), *clip.shape[1:3])
        preprocessed, auxiliary = model(source, source.new_tensor([qp]), source.new_tensor([int(codec_name == "h265")]),
                                        protected, return_aux=True)
        codec = StandardCodec(codec_name, qp, cfg["preset"], cfg["fps"])
        with torch.no_grad():
            anchor, anchor_bpp = real_codec_ste(source, codec)
        reconstructed, actual_bpp = real_codec_ste(preprocessed, codec)
        if args.task == "ar":
            regrets = []
            for teacher in teachers:
                with torch.no_grad():
                    probability = teacher.logits(source).softmax(-1)
                    reference_kl = F.kl_div(teacher.logits(anchor).log_softmax(-1), probability, reduction="batchmean")
                trial_kl = F.kl_div(teacher.logits(reconstructed).log_softmax(-1), probability, reduction="batchmean")
                regrets.append(trial_kl - reference_kl)
        else:
            teacher = teachers[0]
            with torch.no_grad():
                reference_features = od_features(teacher, source)
                reference_loss = feature_distance(reference_features, od_features(teacher, anchor))
            regrets = [feature_distance(reference_features, od_features(teacher, reconstructed)) - reference_loss]
        # Guard the worst encoder teacher; labels are never used in this loss.
        regret = torch.stack(regrets).max()
        proxy = rate_prior(preprocessed)
        observed_scale = actual_bpp / max(float(proxy.detach()), 1e-8)
        previous_scale = calibration[group]
        calibration[group] = observed_scale if previous_scale is None else .9 * previous_scale + .1 * observed_scale
        # Keep the signed regret for restoring a violated dual constraint.
        task_penalty = regret.clamp_min(0)
        loss = args.rate_weight * calibration[group] * proxy + duals[group] * task_penalty
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training objective")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        duals[group] = float(np.clip(duals[group] + args.dual_lr * float(regret.detach()), 0, 100))
        record = {"step": step + 1, "source_id": item["id"], "codec": codec_name, "qp": qp,
                  "loss": float(loss.detach()), "task_regret": float(regret.detach()), "dual": duals[group],
                  "actual_anchor_bpp": anchor_bpp, "actual_preprocessed_bpp": actual_bpp,
                  "same_qp_overhead_pct": (actual_bpp / anchor_bpp - 1) * 100,
                  "rate_prior": float(proxy.detach()), "calibration": calibration[group],
                  "alpha_mean": float(auxiliary["alpha"].detach().mean()), "gradient_norm": float(gradient_norm)}
        with (args.out / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            print(json.dumps(record, allow_nan=False), flush=True)
    checkpoint = {"schema": "adaptive-vcm-blend-v1", "model": model.state_dict(), "width": args.width,
                  "steps": args.steps, "task": args.task, "seed": args.seed,
                  "train_ids_sha256": manifest["train_ids_sha256"], "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
    torch.save(checkpoint, args.out / "preprocessor_last.pth")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ar", "od"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v22_screen.json")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=302001)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rate-weight", type=float, default=.5)
    parser.add_argument("--dual-lr", type=float, default=.02)
    parser.add_argument("--out", type=Path, required=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
