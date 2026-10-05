"""TRAIN-only all-action codec measurements followed by minibatch replay."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .analyzers import ActionAnalyzer
from .codec import StandardCodec, reference_bpp
from .data import ar_plan, fingerprint, partition, read_video
from .evaluate import ROOT, code_manifest, validate_config, write_json
from .preprocessing import normalize_map
from .ranking import (CONTEXT_DIM, CONTEXT_SCHEMA, RankPreprocessor, build_rank_context,
                      measurement_targets, ranking_loss, static_action_order)
from .rateaware import semantic_protection
from .selection import relative_guard


def _task_bank():
    from . import task_bank
    return task_bank


def _append(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def _measurement_summary(records, names):
    groups = {}
    for codec, qp in sorted({(row["codec"], row["qp"]) for row in records}):
        subset = [row for row in records if (row["codec"], row["qp"]) == (codec, qp)]
        oracle = [row["oracle_saving_pct"] for row in subset]
        groups[f"{codec}/{qp}"] = {
            "records": len(subset), "any_feasible_records": sum(row["feasible_actions"] > 0 for row in subset),
            "feasible_actions": sum(row["feasible_actions"] for row in subset),
            "nonidentity_trials": len(subset) * (len(names) - 1),
            "oracle_mean_saving_pct": float(np.mean(oracle)),
            "oracle_action_counts": dict(Counter(row["oracle_action"] for row in subset))}
    return groups


def train(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(cfg)
    if args.task != "ar" or cfg.get("ar_training") != "all_action_ranking":
        raise ValueError("V25 ranking requires the registered AR all-action recipe")
    if not cfg.get("ar_require_anchor_decision", False) or cfg["min_savings"] != .01:
        raise ValueError("V25 must retain the strict anchor guard and one-percent byte threshold")
    if (min(args.steps, args.count, args.width, args.measurements, args.batch_size) < 1
            or not 0 < args.lr < 1 or args.measurements < 2 * len(cfg["qps"])):
        raise ValueError("invalid collection/replay budget")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("training output must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan, _ = ar_plan(args.root, "train", args.count)
    if any(partition(row["id"]) != "train" for row in plan):
        raise ValueError("measurement plan contains a DEV/TEST source")
    teachers = [ActionAnalyzer(name, device) for name in cfg["ar_teachers"]]
    bank_module = _task_bank()
    names = tuple(bank_module.ACTION_NAMES)
    model = RankPreprocessor(args.width, args.task, names).to(device).train()
    groups = [(codec, qp) for codec in ("h264", "h265") for qp in cfg["qps"]]
    weights = np.array([1 if qp < 40 else 2 if qp < 45 else 3 for _, qp in groups], float)
    weights /= weights.sum()
    manifest = {
        "schema": "adaptive-vcm-training-v4", "task": "ar", "seed": args.seed,
        "steps": args.steps, "measurements": args.measurements, "batch_size": args.batch_size,
        "width": args.width, "lr": args.lr, "device": device,
        "train_ids": [row["id"] for row in plan],
        "train_ids_sha256": fingerprint([row["id"] for row in plan]),
        "config": cfg, "code": code_manifest(), "action_names": list(names),
        "context_dim": CONTEXT_DIM, "context_schema": CONTEXT_SCHEMA,
        "qp_sampling": {f"{c}/{q}": float(p) for (c, q), p in zip(groups, weights)},
        "objective": "all-action balanced teacher-safety BCE + 2*actual log-byte SmoothL1 + .25*feasible pairwise ranking",
        "guard": "unchanged shared relative_guard, strict anchor decisions, actual coded bytes with all headers",
        "replay": "uniform source/group record minibatches; fixed collection before all optimization",
        "source": "fresh initialization; final-LAST only; no DEV/TEST checkpoint or threshold selection",
        "limitations": "Fixed filter bank ranked from encoder teachers; held-out evaluator accuracy still requires independent measurement."}
    write_json(args.out / "training_manifest.json", manifest)
    contexts, safeties, log_rates, eligibilities, records = [], [], [], [], []
    source_order = []
    for measurement in range(args.measurements):
        if measurement % len(plan) == 0:
            source_order = rng.permutation(len(plan)).tolist()
        item = plan[source_order[measurement % len(plan)]]
        group_index = measurement if measurement < len(groups) else int(rng.choice(len(groups), p=weights))
        codec_name, qp = groups[group_index]
        clip = read_video(item["path"], cfg["frames"], cfg["ar_size"], cfg["temporal_stride"])
        source_predictions = [teacher.probabilities(clip) for teacher in teachers]
        semantic = np.maximum.reduce([normalize_map(teacher.saliency(clip)) for teacher in teachers])
        protection = semantic_protection(semantic)
        codec = StandardCodec(codec_name, qp, cfg["preset"], cfg["fps"])
        anchor = codec.roundtrip(clip)
        anchor_predictions = [teacher.probabilities(anchor.decoded) for teacher in teachers]
        contexts.append(build_rank_context(clip, qp, codec_name, protection, source_predictions, anchor_predictions))
        bank = bank_module.build_task_bank(clip, protection, qp)
        if tuple(candidate.name for candidate in bank) != names:
            raise ValueError("action bank order changed during measurement")
        observations = []
        predictions_cache = {hashlib.sha256(anchor.data).hexdigest(): anchor_predictions}
        for index, candidate in enumerate(bank):
            stream = anchor if index == 0 else codec.roundtrip(candidate.clip)
            stream_hash = hashlib.sha256(stream.data).hexdigest()
            if stream_hash not in predictions_cache:
                predictions_cache[stream_hash] = [teacher.probabilities(stream.decoded) for teacher in teachers]
            distances, decisions = relative_guard("ar", source_predictions, anchor_predictions,
                                                  predictions_cache[stream_hash], cfg)
            observations.append({"name": candidate.name, "coded_bytes": stream.coded_bytes,
                                 "distances": [float(d) for d in distances], "decisions": list(decisions),
                                 "identity_stream": stream.data == anchor.data, "stream_sha256": stream_hash,
                                 "shape": list(candidate.clip.shape), "codec_seconds": stream.seconds})
        safety, log_rate, eligible = measurement_targets(observations, slack=cfg["ar_kl_slack"],
                                                        min_savings=cfg["min_savings"])
        safeties.append(safety)
        log_rates.append(log_rate)
        eligibilities.append(eligible)
        utility = np.where(eligible > .5, 1 - np.exp(log_rate), 0.)
        best = int(utility.argmax()) + 1 if utility.max() > 0 else 0
        row = {"measurement": measurement + 1, "source_id": item["id"], "codec": codec_name, "qp": qp,
               "source_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
               "source_shape": list(clip.shape), "context": contexts[-1].tolist(),
               "actual_anchor_bpp": reference_bpp(anchor.coded_bytes, clip.shape),
               "teacher_safe_actions": int(safety.sum()), "feasible_actions": int(eligible.sum()),
               "oracle_action": names[best], "oracle_saving_pct": float(100 * utility.max()),
               "actions": observations}
        records.append(row)
        _append(args.out / "measurements.jsonl", row)
        if measurement == 0 or (measurement + 1) % 25 == 0 or measurement + 1 == args.measurements:
            print(json.dumps({k: v for k, v in row.items() if k not in ("actions", "context")}), flush=True)
    x_np, safety_np, rate_np, eligible_np = map(np.stack, (contexts, safeties, log_rates, eligibilities))
    np.savez_compressed(args.out / "train_records.npz", context=x_np, safety=safety_np,
                        log_rate=rate_np, eligible=eligible_np)
    records_sha256 = hashlib.sha256((args.out / "train_records.npz").read_bytes()).hexdigest()
    x, safety, rate = [torch.as_tensor(values, dtype=torch.float32, device=device)
                       for values in (x_np, safety_np, rate_np)]
    positive_count = safety.sum(0)
    positive_weight = ((len(safety) - positive_count) / positive_count.clamp(min=1.)).clamp(.25, 20.)
    model.safety_log_weight.copy_(positive_weight.log())
    model.static_action_order = static_action_order(safety_np, rate_np, cfg["min_savings"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        batch_indices = rng.integers(0, len(x), size=args.batch_size)
        indices = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
        loss, parts = ranking_loss(model, x[indices], safety[indices], rate[indices],
                                   positive_weight, min_savings=cfg["min_savings"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        row = {"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(norm),
               **{key: float(value) for key, value in parts.items()}}
        _append(args.out / "train.jsonl", row)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            print(json.dumps(row), flush=True)
    model.eval()
    with torch.no_grad():
        logits, predicted_rates = model(x)
        scores = model.scores(logits, predicted_rates).cpu().numpy()
        predicted_order = np.argsort(-scores, axis=1, kind="stable") + 1
        predicted_safety = (logits - model.safety_log_weight).sigmoid().cpu().numpy()
    top_k = int(cfg.get("rank_top_k", 3))
    if not 1 <= top_k < len(names):
        raise ValueError("invalid registered proposal budget")
    utility_np = np.where(eligible_np > .5, 1 - np.exp(rate_np), 0.)
    learned_utility = np.take_along_axis(utility_np, predicted_order[:, :top_k] - 1, axis=1).max(axis=1)
    static_utility = utility_np[:, np.asarray(model.static_action_order[:top_k]) - 1].max(axis=1)
    oracle_utility = utility_np.max(axis=1)
    fit = {
        "scope": "TRAIN resubstitution diagnostics, not DEV/test performance",
        "record_count": len(x), "unique_measured_sources": len({row["source_id"] for row in records}),
        "positive_weight": positive_weight.cpu().tolist(),
        "teacher_safety_accuracy": float(((predicted_safety >= .5) == (safety_np > .5)).mean()),
        "log_rate_mean_absolute_error": float(np.abs(predicted_rates.cpu().numpy() - rate_np).mean()),
        "top_k": top_k, "top1_action_counts": dict(Counter(names[int(i)] for i in predicted_order[:, 0])),
        "topk_action_counts": dict(Counter(names[int(i)] for i in predicted_order[:, :top_k].ravel())),
        "learned_topk_feasible_records": int((learned_utility > 0).sum()),
        "static_topk_feasible_records": int((static_utility > 0).sum()),
        "bank_oracle_feasible_records": int((oracle_utility > 0).sum()),
        "learned_topk_mean_guarded_saving_pct": float(learned_utility.mean() * 100),
        "static_topk_mean_guarded_saving_pct": float(static_utility.mean() * 100),
        "bank_oracle_mean_guarded_saving_pct": float(oracle_utility.mean() * 100),
        "by_codec_qp": _measurement_summary(records, names)}
    write_json(args.out / "fit_diagnostics.json", fit)
    manifest.update({"train_records_sha256": records_sha256,
                     "measurements_sha256": hashlib.sha256((args.out / "measurements.jsonl").read_bytes()).hexdigest(),
                     "static_action_order": model.static_action_order, "fit_diagnostics": fit})
    write_json(args.out / "training_manifest.json", manifest)
    torch.save({"schema": model.schema, "model": model.state_dict(), "width": args.width, "task": "ar",
                "steps": args.steps, "measurements": args.measurements, "seed": args.seed,
                "train_ids_sha256": manifest["train_ids_sha256"], "training_config": cfg,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "context_schema": CONTEXT_SCHEMA, "context_dim": CONTEXT_DIM,
                "action_names": list(names), "static_action_order": model.static_action_order,
                "train_records_sha256": records_sha256}, args.out / "preprocessor_last.pth")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ar"], default="ar")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v25_screen.json")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--measurements", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=302101)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
