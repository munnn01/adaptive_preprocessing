"""Paired, actual-bitstream benchmark of adaptive preprocessing for AR and OD."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from .analyzers import ActionAnalyzer, DetectionAnalyzer
from .codec import StandardCodec, reference_bpp
from .data import ar_plan, od_plan, read_video, read_image, fingerprint
from .metrics import curve_summary, paired_ar_bootstrap
from .preprocessing import Candidate, action_protection, boxes_to_mask, make_candidates, normalize_map
from .selection import Observation, relative_guard, select
from .rateaware import RateAwarePreprocessor, load_preprocessor, semantic_protection
from .profiles import ProfilePreprocessor

ROOT = Path(__file__).resolve().parents[1]
RANKING_SCHEMAS = ('adaptive-vcm-ranking-v4', 'adaptive-vcm-utility-v5')


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def code_manifest() -> dict:
    files = list((ROOT / "adaptive_vcm").glob("*.py"))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for p in sorted(files)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"commit": commit, "files_sha256": hashes}


def choose_stream(clip, protection, task, codec, cfg, teachers, source_predictions, learned=None,
                  *, learned_mask=None, components=False):
    """Run encoder-only selection. There is no ground-truth or evaluator input."""
    if getattr(learned, 'schema', None) in RANKING_SCHEMAS:
        if task != 'ar':
            raise ValueError('ranking recipe is registered for AR only')
        from .rank_selection import choose_rank_stream
        return choose_rank_stream(clip, protection, codec, cfg, teachers, source_predictions,
                                  learned, learned_mask=learned_mask, components=components)
    candidates = make_candidates(clip, protection, task, codec.qp, cfg[f"{task}_candidates"])
    selected_profile = None
    if task == "od" and not np.any(source_predictions[0]["scores"] >= cfg["od_score_threshold"]):
        candidates = candidates[:1]  # unknown foreground: retain the entire source
    foreground_known = task == "ar" or np.any(source_predictions[0]["scores"] >= cfg["od_score_threshold"])
    if learned is not None and foreground_known:
        with torch.no_grad():
            device = next(learned.parameters()).device
            x = torch.from_numpy(clip.copy()).to(device).float().permute(3, 0, 1, 2)[None] / 255
            mask_np = protection if learned_mask is None else learned_mask
            mask = torch.from_numpy(mask_np.copy()).to(device)[None, None, None].expand(1, 1, len(clip), *clip.shape[1:3])
            if isinstance(learned, ProfilePreprocessor):
                output, aux = learned(x, x.new_tensor([codec.qp]), x.new_tensor([int(codec.codec == "h265")]), mask, return_aux=True)
                selected_profile = learned.profiles[int(aux['profile_indices'][0])][0]
            else:
                output = learned(x, x.new_tensor([codec.qp]), x.new_tensor([int(codec.codec == "h265")]), mask)
            pixels = output[0].permute(1, 2, 3, 0).mul(255).round().clamp(0, 255).byte().cpu().numpy()
            candidates.append(Candidate(getattr(learned, "candidate_name", "learned_blend"), pixels))
    encoded, predictions, observations, audit = [], [], [], []
    for candidate in candidates:
        result = codec.roundtrip(candidate.clip)
        encoded.append(result)
        decoded_predictions = (predictions[0] if predictions and result.data == encoded[0].data else
                               [teacher.probabilities(result.decoded) if task == "ar" else teacher.predict(result.decoded)
                                for teacher in teachers])
        predictions.append(decoded_predictions)
        if not observations:
            distances, decisions = tuple(0. for _ in teachers), tuple(True for _ in teachers)
        else:
            distances, decisions = relative_guard(task, source_predictions, predictions[0], decoded_predictions, cfg)
        observations.append(Observation(candidate.name, result.coded_bytes, distances, decisions))
        audit.append({"name": candidate.name, "coded_bytes": result.coded_bytes,
                      "relative_task_distance": [float(d) if np.isfinite(d) else None for d in distances],
                      "preserves_decision": list(decisions), "codec_seconds": result.seconds,
                      "stream_sha256": hashlib.sha256(result.data).hexdigest()})
        if candidate.name == 'learned_profile':
            audit[-1]['profile'] = selected_profile
    chosen = select(observations, cfg["ar_kl_slack"] if task == "ar" else cfg["od_distance_slack"], cfg["min_savings"])
    if components:
        slack = cfg["ar_kl_slack"] if task == "ar" else cfg["od_distance_slack"]
        learned_indices = [i for i, c in enumerate(candidates) if c.name.startswith("learned_")]
        control_indices = [i for i, c in enumerate(candidates) if not c.name.startswith("learned_")]
        controls_chosen = control_indices[select([observations[i] for i in control_indices], slack, cfg["min_savings"])]
        subset = [0, *learned_indices]
        learned_chosen = subset[select([observations[i] for i in subset], slack, cfg["min_savings"])]
        raw = learned_indices[0] if learned_indices else 0
        alternatives = {"controls": (encoded[controls_chosen], candidates[controls_chosen].name),
                        "learned_guarded": (encoded[learned_chosen], candidates[learned_chosen].name),
                        "learned_raw": (encoded[raw], candidates[raw].name)}
        return encoded[0], encoded[chosen], candidates[chosen].name, audit, alternatives
    return encoded[0], encoded[chosen], candidates[chosen].name, audit


def _ar_curves(rows, names, qps, draws, seed):
    output = {}
    for name in names:
        curves = {}
        for arm in ("anchor", "adaptive"):
            curves[arm] = {"bpp": [], "quality": []}
            for qp in qps:
                records = [r for r in rows if r["arm"] == arm and r["qp"] == qp]
                curves[arm]["bpp"].append(float(np.mean([r["bpp"] for r in records])))
                curves[arm]["quality"].append(float(np.mean([r["correct"][name] for r in records])))
        ci = paired_ar_bootstrap(rows, name, qps, draws, seed)
        output[name] = {"curves": curves, **curve_summary(curves["anchor"]["bpp"], curves["anchor"]["quality"],
                       curves["adaptive"]["bpp"], curves["adaptive"]["quality"], ci=ci)}
    return output


def _od_curves(rows, qps, draws, seed, meta, gt, codec):
    from .coco_metrics import coco_map, paired_bootstrap_detection_bd
    ids = sorted(gt)
    records = {arm: {(codec, qp): {int(r["id"]): (r["bpp"], r["predictions"]) for r in rows
                                 if r["arm"] == arm and r["qp"] == qp} for qp in qps}
               for arm in ("anchor", "adaptive")}
    curves = {arm: {"bpp": [], "quality": []} for arm in records}
    for arm in records:
        for qp in qps:
            slot = records[arm][(codec, qp)]
            if set(slot) != set(ids):
                raise ValueError("missing paired OD records")
            curves[arm]["bpp"].append(float(np.mean([slot[i][0] for i in ids])))
            predictions = [p for i in ids for p in slot[i][1]]
            curves[arm]["quality"].append(coco_map(predictions, gt, ids, meta)[0])
    ci = paired_bootstrap_detection_bd(records, codec=codec, qps=qps, gt_by_id=gt,
                                      image_ids=ids, ann_meta=meta, arms=["adaptive"], n_boot=draws, seed=seed).get("adaptive", {})
    ci["finite_fraction"] = ci.get("n_draws", 0) / draws if draws else 0.
    return {"resnet50": {"curves": curves, **curve_summary(curves["anchor"]["bpp"], curves["anchor"]["quality"],
                              curves["adaptive"]["bpp"], curves["adaptive"]["quality"], ci=ci)}}


def _gt_scaled(meta, plan, geometry):
    ids = {r["image_id"] for r in plan}
    gt = {i: [] for i in ids}
    for ann in meta["annotations"]:
        i = ann["image_id"]
        if i not in ids:
            continue
        sx, sy, left, top = geometry[i]
        x, y, w, h = ann["bbox"]
        gt[i].append({**ann, "bbox": [x * sx + left, y * sy + top, w * sx, h * sy],
                      "area": ann.get("area", w * h) * sx * sy})
    return gt


def _od_predictions(detection, image_id):
    from .coco_metrics import coco_box
    return [{"image_id": image_id, "category_id": int(label), "bbox": coco_box(box), "score": float(score)}
            for box, score, label in zip(detection["boxes"], detection["scores"], detection["labels"]) if score >= .05]


def validate_config(cfg):
    if cfg.get("schema") != 1 or cfg.get("target_bd_rate_pct") != -10:
        raise ValueError("unsupported configuration schema/target")
    qps = cfg["qps"]
    if len(qps) < 3 or len(set(qps)) != len(qps) or any(type(q) is not int or not 0 <= q <= 51 for q in qps):
        raise ValueError("need at least three unique valid QPs")
    if cfg["ar_teachers"] != ["r3d_18", "mc3_18"] or cfg["ar_evaluators"] != ["r2plus1d_18", "r3d_18"]:
        raise ValueError("registered AR analyzer roles changed")
    if cfg["od_teacher"] != "mobilenet" or cfg["od_evaluator"] != "resnet50":
        raise ValueError("registered OD analyzer roles changed")


def run(args) -> dict:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(cfg)
    if args.split not in ("dev", "test") or args.count < 1 or args.bootstrap is not None and args.bootstrap < 0:
        raise ValueError("invalid evaluation plan")
    draws = cfg["bootstrap_draws"] if args.bootstrap is None else args.bootstrap
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("output must be empty; refusing to mix experiment evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.task == "ar":
        plan, data_meta = ar_plan(args.root, args.split, args.count)
        teachers = [ActionAnalyzer(name, device) for name in cfg["ar_teachers"]]
        evaluators = {name: next((m for m in teachers if m.name == name), None) or ActionAnalyzer(name, device)
                      for name in cfg["ar_evaluators"]}
    else:
        if args.annotations is None:
            raise ValueError("COCO instances annotation file required")
        plan, data_meta = od_plan(args.root, args.annotations, args.split, args.count)
        teachers = [DetectionAnalyzer(cfg["od_teacher"], device)]
        evaluators = {cfg["od_evaluator"]: DetectionAnalyzer(cfg["od_evaluator"], device)}
    learned = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if state.get("schema") in (RateAwarePreprocessor.schema, ProfilePreprocessor.schema, *RANKING_SCHEMAS) and state.get("training_config") != cfg:
            raise ValueError("checkpoint/evaluation configuration mismatch")
        learned = load_preprocessor(state, args.task).to(device)
        learned.eval()
    ablate = getattr(args, "ablate_learned", False)
    if ablate and learned is None:
        raise ValueError("component evaluation requires a trained checkpoint")
    manifest = {"experiment": cfg["experiment"], "task": args.task, "codecs": args.codecs,
                "quality_axis": "top1" if args.task == "ar" else "COCO mAP@[.50:.95]",
                "config": cfg, "count": len(plan), "split": args.split, "bootstrap_draws": draws,
                "ids": [r["id"] for r in plan], "ids_sha256": fingerprint([r["id"] for r in plan]),
                "device": device, "python": platform.python_version(), "torch": torch.__version__,
                "code": code_manifest(), "data_metadata": data_meta if args.task == "ar" else {"annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest()},
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
                "scope": cfg["scope"], "rate_denominator": "original pre-transform T*H*W pixels",
                "selection_cost": "all candidate encode/decode and teacher calls included in selection_seconds",
                "component_evaluation": ablate,
                "ar_guard_rule": "anchor_relative_v2" if args.task == "ar" else None,
                "component_scope": "learned_raw is ungated; learned_guarded uses identity fallback; component curves have no bootstrap CI"}
    write_json(args.out / "manifest.json", manifest)
    if getattr(learned, 'schema', None) in RANKING_SCHEMAS:
        manifest['component_scope'] = ('controls and static_adaptive use identical fixed controls; '
                                      'static_adaptive adds TRAIN static topK, adaptive adds learned topK; '
                                      'bank_oracle is an audit-only upper bound; learned_raw is ungated')
        manifest['proposal_budget'] = cfg['rank_top_k']
        write_json(args.out / 'manifest.json', manifest)
    all_rows, geometry = {codec: [] for codec in args.codecs}, {}
    component_rows = {codec: [] for codec in args.codecs}
    for position, item in enumerate(plan):
        if args.task == "ar":
            clip = read_video(item["path"], cfg["frames"], cfg["ar_size"], cfg["temporal_stride"])
        else:
            clip, geometry[item["image_id"]] = read_image(item, cfg["od_size"])
        preparation_start = time.perf_counter()
        if args.task == "ar":
            source_predictions = [teacher.probabilities(clip) for teacher in teachers]
            semantic = np.maximum.reduce([normalize_map(teacher.saliency(clip)) for teacher in teachers])
            protection = action_protection(clip, semantic)
            learned_mask = semantic_protection(semantic) if (isinstance(learned, RateAwarePreprocessor)
                           or getattr(learned, 'schema', None) in RANKING_SCHEMAS) else protection
        else:
            source_predictions = [teachers[0].predict(clip)]
            source = source_predictions[0]
            protection = boxes_to_mask(*clip.shape[1:3], source["boxes"][source["scores"] >= cfg["od_score_threshold"]])
            learned_mask = protection
        preparation_seconds = time.perf_counter() - preparation_start
        source_hash = hashlib.sha256(clip.tobytes()).hexdigest()
        for codec_name in args.codecs:
            for qp in cfg["qps"]:
                start = time.perf_counter()
                codec = StandardCodec(codec_name, qp, cfg["preset"], cfg["fps"])
                bundle = choose_stream(clip, protection, args.task, codec, cfg, teachers, source_predictions,
                                       learned, learned_mask=learned_mask, components=ablate)
                anchor, chosen, name, candidates = bundle[:4]
                seconds = time.perf_counter() - start
                new_rows = []
                streams = [("anchor", anchor, "identity"), ("adaptive", chosen, name)]
                if ablate:
                    streams += [(arm, result, candidate) for arm, (result, candidate) in bundle[4].items()]
                prediction_cache = {}
                for arm, result, candidate_name in streams:
                    row = {"id": item["id"], "qp": qp, "codec": codec_name, "arm": arm,
                           "candidate": candidate_name,
                           "source_sha256": source_hash, "coded_bytes": result.coded_bytes,
                           "bpp": reference_bpp(result.coded_bytes, clip.shape),
                           "stream_sha256": hashlib.sha256(result.data).hexdigest(),
                           "source_preparation_seconds": preparation_seconds, "selection_seconds": seconds}
                    stream_hash = row["stream_sha256"]
                    if stream_hash not in prediction_cache:
                        prediction_cache[stream_hash] = ({"correct": {key: int(model.probabilities(result.decoded).argmax() == item["label"])
                                          for key, model in evaluators.items()}} if args.task == "ar" else
                            {"predictions": _od_predictions(evaluators[cfg["od_evaluator"]].predict(result.decoded), item["image_id"])})
                    row.update(prediction_cache[stream_hash])
                    if args.save_streams:
                        path = args.out / "streams" / codec_name / f"{position:05d}-q{qp}-{arm}.{'264' if codec_name == 'h264' else '265'}"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(result.data)
                    if arm in ("anchor", "adaptive"):
                        new_rows.append(row)
                    else:
                        component_rows[codec_name].append(row)
                        with (args.out / f"{codec_name}_components.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                all_rows[codec_name].extend(new_rows)
                with (args.out / f"{codec_name}_rows.jsonl").open("a", encoding="utf-8") as handle:
                    for row in new_rows:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                with (args.out / "selection_audit.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"id": item["id"], "codec": codec_name, "qp": qp,
                                             "selected": name, "candidates": candidates}, allow_nan=False) + "\n")
        print(f"[{args.task}] source {position + 1}/{len(plan)} completed", flush=True)
    results = {}
    gt = _gt_scaled(data_meta, plan, geometry) if args.task == "od" else None
    if gt is not None:
        write_json(args.out / "coco_ground_truth_scaled.json", {"gt": gt, "categories": data_meta["categories"]})
    for codec, rows in all_rows.items():
        results[codec] = (_ar_curves(rows, list(evaluators), cfg["qps"], draws, cfg["seed"]) if args.task == "ar"
                          else _od_curves(rows, cfg["qps"], draws, cfg["seed"], data_meta, gt, codec))
    component_results = {}
    if ablate:
        for codec, rows in all_rows.items():
            component_results[codec] = {}
            arms = sorted({r['arm'] for r in component_rows[codec]})
            for arm in arms:
                paired = [r for r in rows if r["arm"] == "anchor"]
                paired += [{**r, "arm": "adaptive"} for r in component_rows[codec] if r["arm"] == arm]
                component_results[codec][arm] = (_ar_curves(paired, list(evaluators), cfg["qps"], 0, cfg["seed"]) if args.task == "ar" else
                                                _od_curves(paired, cfg["qps"], 0, cfg["seed"], data_meta, gt, codec))
    decision = {"task": args.task, "quality_axis": manifest["quality_axis"], "results": results,
                "candidate_counts": {codec: dict(Counter(r["candidate"] for r in rows if r["arm"] == "adaptive"))
                                     for codec, rows in all_rows.items()},
                "both_codecs_evaluated": set(args.codecs) == {"h264", "h265"},
                "screen_passes": set(args.codecs) == {"h264", "h265"} and all(r["screen_passes"] for models in results.values() for r in models.values()),
                "target_confirmed": False, "scope": cfg["scope"], "component_results": component_results,
                "high_qp_diagnostics": high_qp_diagnostics(all_rows, component_rows, cfg, args.task, results, component_results)}
    if getattr(learned, 'schema', None) in RANKING_SCHEMAS:
        from .rank_selection import ranking_diagnostics
        decision['policy_contribution'] = {c: ranking_diagnostics(rows, component_rows[c], cfg['qps'])
                                           for c, rows in all_rows.items()}
        decision['proposal_budget'] = {'learned': cfg['rank_top_k'], 'static': cfg['rank_top_k'],
                                      'bank_oracle': len(learned.action_names) - 1,
                                      'oracle_scope': 'audit upper bound; unproposed actions excluded from adaptive selection'}
        if getattr(learned, 'schema', None) == 'adaptive-vcm-utility-v5':
            decision['proposal_budget']['group_static'] = cfg['rank_top_k']
            decision['policy_fit'] = {'learned_mix': learned.learned_mix,
                                     'prior_only': learned.learned_mix == 0,
                                     'utility_target_scope': learned.utility_target_scope,
                                     'scope': 'TRAIN sourceblocked CV; fixed pixel filter bank'}
    write_json(args.out / "summary.json", decision)
    print(json.dumps(decision, indent=2, allow_nan=False), flush=True)
    return decision


def high_qp_diagnostics(all_rows, components, cfg, task, results, component_results):
    """Measured byte savings and quality at QP>=40; no pooled codec target."""
    output = {}
    for codec, rows in all_rows.items():
        output[codec] = {}
        for qp in cfg["qps"]:
            if qp < 40:
                continue
            anchor = {r["id"]: r for r in rows if r["arm"] == "anchor" and r["qp"] == qp}
            entries = {}
            arms = ['adaptive'] + sorted({r['arm'] for r in components[codec]})
            for arm in arms:
                trial = [r for r in [*rows, *components[codec]] if r["arm"] == arm and r["qp"] == qp]
                if not trial:
                    continue
                entries[arm] = {"rate_change_pct": (sum(r["coded_bytes"] for r in trial) / sum(r["coded_bytes"] for r in anchor.values()) - 1) * 100,
                                "candidate_counts": dict(Counter(r["candidate"] for r in trial))}
                if task == "ar":
                    entries[arm]["top1_gap_pp"] = {name: float(np.mean([r["correct"][name] - anchor[r["id"]]["correct"][name] for r in trial]) * 100)
                                                         for name in cfg["ar_evaluators"]}
                else:
                    curves = (results[codec] if arm == "adaptive" else component_results[codec][arm])["resnet50"]["curves"]
                    index = cfg["qps"].index(qp)
                    entries[arm]["map_gap_pp"] = (curves["adaptive"]["quality"][index] - curves["anchor"]["quality"][index]) * 100
            output[codec][str(qp)] = entries
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ar", "od"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v22_screen.json")
    parser.add_argument("--count", type=int, default=208)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--codecs", nargs="+", choices=["h264", "h265"], default=["h264", "h265"])
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-streams", action="store_true")
    parser.add_argument("--ablate-learned", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.codecs)) != len(args.codecs):
        parser.error("duplicate codec")
    run(args)


if __name__ == "__main__":
    main()
