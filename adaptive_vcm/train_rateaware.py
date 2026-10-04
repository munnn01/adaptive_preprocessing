"""Optimize V23 policy through actual codec bytes and frozen teacher probes.

Two-sided SPSA estimates a derivative in the seven policy controls. This is
stochastic black-box optimization, not a differentiable entropy model or an
identity-STE estimate of codec derivatives. TRAIN-only feasible probes supply
an additional policy imitation target. No evaluator or true labels are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .analyzers import ActionAnalyzer, DetectionAnalyzer
from .codec import StandardCodec, reference_bpp
from .data import ar_plan, od_plan, fingerprint, read_video, read_image
from .evaluate import ROOT, code_manifest, validate_config, write_json
from .preprocessing import boxes_to_mask, normalize_map
from .rateaware import RateAwarePreprocessor, semantic_protection
from .selection import Observation, relative_guard, select


def measured_objective(byte_count, anchor_bytes, distances, decisions, slack, dual):
    if byte_count <= 0 or anchor_bytes <= 0 or not distances or len(distances) != len(decisions):
        raise ValueError("invalid measured objective")
    margin = max(d - slack if math.isfinite(d) else 1. for d in distances)
    violation = max(0., margin) / max(slack, 1e-3) + float(not all(decisions))
    relative_rate = math.log(byte_count / anchor_bytes)
    # A task-invalid probe cannot earn credit for saving arbitrarily many bits.
    # This prevents learning an unsafe filter that the final selector rejects.
    rate_credit = relative_rate if violation == 0 else max(0., relative_rate)
    return rate_credit + dual * violation, violation


def spsa_gradient(plus_loss, minus_loss, direction, epsilon):
    if epsilon <= 0 or not all(math.isfinite(v) for v in (plus_loss, minus_loss, epsilon)):
        raise ValueError("invalid finite-difference probe")
    if not ((direction == 1) | (direction == -1)).all():
        raise ValueError("Rademacher direction required")
    return ((plus_loss - minus_loss) / (2 * epsilon)) * direction


def predict(teachers, task, clip):
    return [teacher.probabilities(clip) if task == "ar" else teacher.predict(clip) for teacher in teachers]


def pixels(video):
    return video[0].permute(1, 2, 3, 0).mul(255).round().clamp(0, 255).byte().cpu().numpy()


def train(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(cfg)
    if min(args.steps, args.count) < 1 or args.lr <= 0 or args.epsilon <= 0 or args.probe_every < 1:
        raise ValueError("invalid training budget")
    if not math.isfinite(args.epsilon) or args.dual_lr < 0:
        raise ValueError("invalid optimization settings")
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
            raise ValueError("COCO annotations required")
        plan, _ = od_plan(args.root, args.annotations, "train", args.count)
        teachers = [DetectionAnalyzer(cfg["od_teacher"], device)]
    model = RateAwarePreprocessor(args.width, args.task).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    groups = [(codec, qp) for codec in ("h264", "h265") for qp in cfg["qps"]]
    weights = np.array([1 if qp < 40 else 2 if qp < 45 else 3 for _, qp in groups], float)
    weights /= weights.sum()
    duals = {group: 2. for group in groups}
    slack = cfg["ar_kl_slack"] if args.task == "ar" else cfg["od_distance_slack"]
    manifest = {"schema": "adaptive-vcm-training-v2", "task": args.task, "seed": args.seed,
                "steps": args.steps, "width": args.width, "lr": args.lr, "epsilon": args.epsilon,
                "probe_every": args.probe_every, "dual_lr": args.dual_lr,
                "qp_sampling": {f"{c}/{q}": float(p) for (c, q), p in zip(groups, weights)},
                "train_ids": [r["id"] for r in plan], "train_ids_sha256": fingerprint([r["id"] for r in plan]),
                "config": cfg, "code": code_manifest(), "device": device,
                "source": "fresh initialization; final-LAST only; no DEV/TEST model selection",
                "objective": "task-feasible log(actual_bytes/anchor_bytes) + dual*normalized_teacher_violation; no savings credit for task-invalid probes",
                "gradient_limitations": "Two-sided SPSA in seven control logits; noisy finite differences, with exact codec/teacher forwards. No rate surrogate, pixel STE, or entropy estimate.",
                "mask": "AR squared semantic map with exact >=.95 core and per-frame motion rendering; OD unchanged detector box halo."}
    write_json(args.out / "training_manifest.json", manifest)
    order = []
    for step in range(args.steps):
        if step % len(plan) == 0:
            order = rng.permutation(len(plan)).tolist()
        item = plan[order[step % len(plan)]]
        group = groups[step % len(groups)] if step < len(groups) else groups[int(rng.choice(len(groups), p=weights))]
        codec_name, qp = group
        clip = (read_video(item["path"], cfg["frames"], cfg["ar_size"], cfg["temporal_stride"]) if args.task == "ar"
                else read_image(item, cfg["od_size"])[0])
        source_predictions = predict(teachers, args.task, clip)
        if args.task == "ar":
            semantic = np.maximum.reduce([normalize_map(t.saliency(clip)) for t in teachers])
            mask = semantic_protection(semantic)
        else:
            detections = source_predictions[0]
            mask = boxes_to_mask(*clip.shape[1:3], detections["boxes"][detections["scores"] >= cfg["od_score_threshold"]])
            if not mask.any():
                mask = np.ones(clip.shape[1:3], np.float32)
        source = torch.from_numpy(clip.copy()).to(device).float().permute(3, 0, 1, 2)[None] / 255
        protected = torch.from_numpy(mask.copy()).to(source)[None, None, None].expand(1, 1, len(clip), *clip.shape[1:3])
        qp_tensor, codec_tensor = source.new_tensor([qp]), source.new_tensor([int(codec_name == "h265")])
        controls = model.policy(source, qp_tensor, codec_tensor, protected)
        codec = StandardCodec(codec_name, qp, cfg["preset"], cfg["fps"])
        anchor = codec.roundtrip(clip)
        anchor_predictions = predict(teachers, args.task, anchor.decoded)
        with torch.no_grad():
            bank = model.filter_bank(source, qp_tensor)
        direction = source.new_tensor(rng.choice([-1., 1.], size=controls.shape))
        trial_controls = [controls.detach(), controls.detach() + args.epsilon * direction,
                          controls.detach() - args.epsilon * direction]
        trial_names = ["center", "plus", "minus"]
        if step % args.probe_every == 0:
            # Strong measured targets prevent training only around an invisible edit.
            for expert in (1, 3, 4):
                target = controls.detach().clone()
                target[:, 0] = 2.
                target[:, 1:6] = -4.
                target[:, 1 + expert] = 4.
                trial_controls.append(target)
                trial_names.append(f"expert{expert}")
        measured, outputs, auxiliaries = [], [], []
        for index, trial_control in enumerate(trial_controls):
            with torch.no_grad():
                output, aux = model(source, qp_tensor, codec_tensor, protected,
                                    controls=trial_control, bank=bank, return_aux=True)
                edited = pixels(output)
            stream = codec.roundtrip(edited)
            if stream.data == anchor.data:
                distances, decisions = tuple(0. for _ in teachers), tuple(True for _ in teachers)
            else:
                distances, decisions = relative_guard(args.task, source_predictions, anchor_predictions,
                                                       predict(teachers, args.task, stream.decoded), cfg)
            objective, violation = measured_objective(stream.coded_bytes, anchor.coded_bytes,
                                                       distances, decisions, slack, duals[group])
            measured.append({"name": trial_names[index], "coded_bytes": stream.coded_bytes,
                             "identity_stream": stream.data == anchor.data,
                             "distances": distances, "decisions": decisions,
                             "objective": objective, "violation": violation})
            outputs.append(edited)
            auxiliaries.append(aux)
        gradient = spsa_gradient(measured[1]["objective"], measured[2]["objective"], direction, args.epsilon)
        if len(clip) == 1:
            gradient[:, 6] = 0  # temporal control is inactive on an image
        observations = [Observation("identity", anchor.coded_bytes, tuple(0. for _ in teachers), tuple(True for _ in teachers))]
        observations += [Observation(r["name"], r["coded_bytes"], r["distances"], r["decisions"]) for r in measured]
        chosen = select(observations, slack, cfg["min_savings"])
        best = trial_controls[chosen - 1] if chosen else None
        imitation = (F.mse_loss(controls, best) if best is not None else controls.sum() * 0)
        # Its numerical value is a gradient carrier, never reported as measured RD.
        carrier = (controls * gradient.detach()).sum() + .2 * imitation
        optimizer.zero_grad(set_to_none=True)
        carrier.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        duals[group] = float(np.clip(duals[group] + args.dual_lr * measured[0]["violation"], 2., 25))
        record = {"step": step + 1, "source_id": item["id"], "codec": codec_name, "qp": qp,
                  "actual_anchor_bpp": reference_bpp(anchor.coded_bytes, clip.shape),
                  "actual_preprocessed_bpp": reference_bpp(measured[0]["coded_bytes"], clip.shape),
                  "same_qp_overhead_pct": (measured[0]["coded_bytes"] / anchor.coded_bytes - 1) * 100,
                  "measured_objective": measured[0]["objective"], "teacher_violation": measured[0]["violation"],
                  "spsa_gradient_norm": float(gradient.norm()), "gradient_norm": float(norm),
                  "dual": duals[group], "imitation_target": observations[chosen].name,
                  "alpha_mean": float(auxiliaries[0]["alpha"].mean()),
                  "editable_fraction": float((protected < 1).float().mean()),
                  "changed_pixel_fraction": float(np.any(outputs[0] != clip, axis=-1).mean()),
                  "center_equals_identity_stream": measured[0]["identity_stream"],
                  "probes": [{"name": r["name"], "coded_bytes": r["coded_bytes"], "objective": r["objective"],
                              "violation": r["violation"]} for r in measured]}
        with (args.out / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            print(json.dumps(record, allow_nan=False), flush=True)
    state = {"schema": model.schema, "model": model.state_dict(), "width": args.width, "task": args.task,
             "steps": args.steps, "seed": args.seed, "train_ids_sha256": manifest["train_ids_sha256"],
             "training_config": cfg,
             "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
    torch.save(state, args.out / "preprocessor_last.pth")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["ar", "od"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v23_screen.json")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=302001)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=.5)
    parser.add_argument("--dual-lr", type=float, default=.05)
    parser.add_argument("--probe-every", type=int, default=10)
    parser.add_argument("--out", required=True, type=Path)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
