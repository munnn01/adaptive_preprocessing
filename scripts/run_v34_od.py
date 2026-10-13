"""Dedicated Object Detection (COCO 2017) evaluation on V34 adaptive preprocessing."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_vcm.analyzers import DetectionAnalyzer
from adaptive_vcm.data import od_plan, read_image, partition
from adaptive_vcm.coco_metrics import coco_box, coco_map
from adaptive_vcm.v33ap.sources import SourceRecord, serial
from adaptive_vcm.v33ap.evaluate import inverse_boxes, safe_curve
from adaptive_vcm.v34.protocol import ACTIONS, settings, digest, codec_factory
from adaptive_vcm.v34.preprocessing import region_maps, filter_clip
from adaptive_vcm.v34.model import Controller
from adaptive_vcm.v34.learning import targets, train_controller, calibrate, select_action
from adaptive_vcm.v34.checkpoint import save_checkpoint, load_checkpoint
from adaptive_vcm.v34.runner import input_pair, condition, measure_candidates, infer, write_json
from adaptive_vcm.v34.tasks import Heads
from adaptive_vcm.v34.evaluation import choose_packet, coco_predictions


def parse_args():
    p = argparse.ArgumentParser(description="Run V34 Object Detection benchmark on COCO 2017")
    p.add_argument("--od-root", required=True, help="Directory containing COCO val2017 images")
    p.add_argument("--annotations", required=True, help="Path to instances_val2017.json")
    p.add_argument("--config", default="configs/v34.json", help="Path to config file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--train-count", type=int, default=48)
    p.add_argument("--cal-count", type=int, default=16)
    p.add_argument("--dev-count", type=int, default=16)
    p.add_argument("--test-count", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--frozen-checkpoint", default="checkpoints/v34_r1/controller_303401.pth",
                   help="Path to frozen V34-R1 checkpoint")
    return p.parse_args()


def load_cohorts(od_root: Path, annotations_path: Path, cfg: dict):
    groups = {s: [] for s in ("train", "cal", "dev", "test")}
    total_train = cfg["train_count"] + cfg["cal_count"]
    train_items, meta = od_plan(od_root, annotations_path, "train", total_train)
    # Split train items deterministically into fit and cal
    ranked = sorted(train_items, key=lambda i: hashlib.sha256(f"v34/od/coco2017/{i['id']}".encode()).hexdigest())
    cal_items = ranked[:cfg["cal_count"]]
    fit_items = ranked[cfg["cal_count"]:]
    
    dev_items, _ = od_plan(od_root, annotations_path, "dev", cfg["dev_count"])
    test_items, _ = od_plan(od_root, annotations_path, "test", cfg["test_count"])
    
    plans = {"train": fit_items, "cal": cal_items, "dev": dev_items, "test": test_items}
    for split, items in plans.items():
        for item in items:
            rgb, g = read_image(item, cfg["od_size"])
            recipe = dict(kind="single-image", repeated_frames=0, frame_indices=[0])
            r = SourceRecord(f"od/coco2017/{item['id']}", "od", item.copy(), rgb, g, recipe)
            from adaptive_vcm.v32vc.train import od_targets
            r.targets = od_targets(meta, item, g, "cpu")
            r.split = split
            groups[split].append(r)
    return groups, meta


def run_benchmark(args):
    cfg = settings(args.config)
    cfg["train_count"] = args.train_count
    cfg["cal_count"] = args.cal_count
    cfg["dev_count"] = args.dev_count
    cfg["test_count"] = args.test_count
    
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== V34 OD Benchmark on COCO 2017 ===", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Counts: train={cfg['train_count']}, cal={cfg['cal_count']}, dev={cfg['dev_count']}, test={cfg['test_count']}", flush=True)

    # 1. Load cohorts
    od_root = Path(args.od_root)
    ann_path = Path(args.annotations)
    print(f"Loading COCO cohorts from {od_root}...", flush=True)
    groups, meta = load_cohorts(od_root, ann_path, cfg)
    print(f"Loaded {len(groups['train'])} train, {len(groups['cal'])} cal, {len(groups['dev'])} dev, {len(groups['test'])} test images.", flush=True)

    # 2. Setup Heads (Faster R-CNN ResNet-50)
    cfg["detector"] = "resnet50"
    heads = Heads(device, "ar_od", cfg)
    task_models = heads.hashes()
    detector = heads.detector

    # Check frozen model
    frozen_model = None
    frozen_path = Path(args.frozen_checkpoint)
    if frozen_path.is_file():
        try:
            frozen_model = load_checkpoint(frozen_path, device)
            print(f"Loaded frozen V34-R1 checkpoint: {frozen_path.name} (seed {frozen_model.metadata.get('seed')})", flush=True)
        except Exception as e:
            print(f"Notice: could not load frozen checkpoint ({e})", flush=True)

    # 3. Source features & Candidate Measurements
    models = []
    inputs = {}
    meas_file = out / "measurements.json"
    all_ckpts = all((out / f"controller_{seed}.pth").is_file() for seed in cfg["seeds"])

    if meas_file.is_file() and all_ckpts:
        print(f"Reusing existing candidate measurements and checkpoints from {out}...", flush=True)
        for seed in cfg["seeds"]:
            models.append(load_checkpoint(out / f"controller_{seed}.pth", device))
            print(f"  Loaded controller_{seed}.pth", flush=True)
    else:
        print(f"Extracting source detection features and measuring candidate table...", flush=True)
        cache = dict(schema=cfg["schema"], experiment="ar_od", settings=cfg, actions=[list(a) for a in ACTIONS],
                     sources={}, task_models=task_models, rows=[])
        
        for split in ("train", "cal"):
            for idx, r in enumerate(groups[split]):
                heads.source(r)
                maps = region_maps(r.rgb, r.source_predictions["od"])
                inputs[r.id] = input_pair(r.rgb, maps)
                cache["sources"][r.id] = dict(split=split, pixel_sha256=hashlib.sha256(r.rgb.tobytes()).hexdigest(),
                                              frame_recipe=r.frame_recipe)
                for codec in cfg["codecs"]:
                    for qp in cfg["qps"]:
                        c = codec_factory("od", codec, qp, cfg)
                        table = measure_candidates(r.rgb, maps, ACTIONS, c, "od", lambda rgb: heads.costs(rgb, r))
                        cache["rows"].append(dict(
                            id=r.id, split=split, domain="od", codec=codec, qp=qp, mask=r.mask,
                            provenance=r.provenance, condition=condition(codec, qp, r.mask, cfg, domain="od"),
                            **table
                        ))
                if (idx + 1) % 8 == 0 or (idx + 1) == len(groups[split]):
                    print(f"  Candidate cache {split}: {idx + 1}/{len(groups[split])} images done", flush=True)

        cache["sha256"] = digest({k: v for k, v in cache.items() if k != "sha256"})
        write_json(meas_file, cache)

        # 4. Train Controller models
        print(f"Training V34 controller models (3 seeds)...", flush=True)
        fit_rows = [r for r in cache["rows"] if r["split"] == "train"]
        cal_rows = [r for r in cache["rows"] if r["split"] == "cal"]
        choices = [targets(r["rate"], np.array(r["damage"], float), r["mask"], cfg["damage_tolerance"], cfg["budget_ratio"]) for r in fit_rows]
        write_json(out / "train_oracle.json", dict(
            scope="TRAIN measured oracle", count=len(choices),
            action_counts={str(a): choices.count(a) for a in range(len(ACTIONS))}
        ))

        cond_len = 9 if cfg.get("domain_condition") else 6
        for seed in cfg["seeds"]:
            torch.manual_seed(seed); np.random.seed(seed)
            model = Controller(len(ACTIONS), conditions=cond_len)
            history = train_controller(model, fit_rows, inputs, cfg, device)
            pred = [infer(model, inputs[r["id"]], r["condition"]) for r in cal_rows]
            actual = np.array([r["damage"] for r in cal_rows], float)
            calibration = calibrate(
                np.array([p["risk"] for p in pred]), actual,
                np.array([p["rate"] for p in pred]), np.array([r["rate"] for r in cal_rows]),
                np.array([r["mask"] for r in cal_rows], bool), cfg["cal_coverage"]
            )
            meta_dict = dict(seed=seed, settings=cfg, actions=[list(a) for a in ACTIONS],
                             experiment="ar_od", task_models=task_models, calibration=calibration,
                             checkpoint_role="LAST", training_history=history)
            path = out / f"controller_{seed}.pth"
            save_checkpoint(path, model, meta_dict)
            models.append(load_checkpoint(path, device))
            write_json(out / f"training_{seed}.json", meta_dict)
            print(f"  Model seed {seed} trained and calibrated.", flush=True)

    # 5. Evaluate held-out DEV and TEST splits
    print(f"Evaluating DEV and TEST splits on Faster R-CNN ResNet-50...", flush=True)
    gt_annotations = defaultdict(list)
    for ann in meta["annotations"]:
        gt_annotations[ann["image_id"]].append(ann)

    eval_results = {}
    for split in ("dev", "test"):
        records = groups[split]
        rows = []
        for idx, r in enumerate(records):
            heads.source(r)
            maps = region_maps(r.rgb, r.source_predictions["od"])
            pair = input_pair(r.rgb, maps)
            
            for codec_name in cfg["codecs"]:
                for qp in cfg["qps"]:
                    codec = codec_factory("od", codec_name, qp, cfg)
                    packets = {}
                    def encode(action):
                        if action not in packets:
                            pixels = filter_clip(r.rgb, maps, ACTIONS[action])
                            packets[action] = codec.roundtrip(pixels, source_shape=r.rgb.shape, task="od")
                        return packets[action]

                    anchor = encode(0)
                    arms = {
                        "anchor": (0, anchor, {}),
                        "fixed": (cfg["fixed_action"], encode(cfg["fixed_action"]), {})
                    }
                    for m in models:
                        forced, selected, p, details = choose_packet(
                            m, pair, codec_name, qp, r.mask, cfg, anchor, encode, domain="od"
                        )
                        s = m.metadata["seed"]
                        arms[f"forced_{s}"] = (forced, encode(forced), {**details, "rate_fallback": False})
                        arms[f"selected_{s}"] = (selected, p, details)

                    if frozen_model is not None:
                        # Condition vector with 9 dimensions for frozen model
                        frozen_cond = condition(codec_name, qp, r.mask, dict(budget_ratio=1.01, domain_condition=True), domain="od")
                        frozen_out = infer(frozen_model, pair, frozen_cond)
                        f_forced = int(np.argmax(frozen_out["logits"]))
                        arms["forced_frozen_303401"] = (f_forced, encode(f_forced), {
                            "proposed_action": f_forced, "predicted_rate": frozen_out["rate"], "predicted_risk": frozen_out["risk"]
                        })

                    for arm, (action, packet, details) in arms.items():
                        pred = detector.predict(packet.decoded)
                        coco_preds = coco_predictions(pred, r.item, r.geometry)
                        bpp = packet.coded_bytes * 8 / np.prod(r.rgb.shape[:3])
                        rows.append(dict(
                            id=r.id, task="od", split=split, codec=codec_name, qp=qp, arm=arm,
                            action=action, bytes=packet.coded_bytes, bpp=bpp,
                            coco=coco_preds, **details
                        ))
            if (idx + 1) % 8 == 0 or (idx + 1) == len(records):
                print(f"  Eval {split}: {idx + 1}/{len(records)} images complete", flush=True)

        write_json(out / f"{split}_rows.json", rows)
        
        # Compute COCO mAP and BD-rates
        summary = compute_od_summary(rows, records, meta, gt_annotations, cfg, split)
        write_json(out / f"{split}_summary.json", summary)
        eval_results[split] = summary

    # 6. Generate Markdown Report
    report = generate_report(eval_results["test"], cfg)
    (out / "OD_REPORT.md").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    print(f"\nAll results saved to {out.resolve()}", flush=True)


def compute_od_summary(rows, records, meta, gt_annotations, cfg, split):
    all_arms = sorted({r["arm"] for r in rows})
    image_ids = [r.item["image_id"] for r in records]
    results = {}
    
    for codec in cfg["codecs"]:
        curves = {}
        for arm in all_arms:
            bpp_list = []
            map_list = []
            ap50_list = []
            for qp in cfg["qps"]:
                sub = [r for r in rows if r["codec"] == codec and r["qp"] == qp and r["arm"] == arm]
                bpp_list.append(float(np.mean([r["bpp"] for r in sub])))
                all_preds = [p for r in sub for p in r["coco"]]
                m_score, ap50_score = coco_map(all_preds, gt_annotations, image_ids, meta)
                map_list.append(float(m_score))
                ap50_list.append(float(ap50_score))
            curves[arm] = dict(bpp=bpp_list, quality=map_list, ap50=ap50_list)

        comparisons = {}
        for arm in all_arms:
            if arm != "anchor":
                comparisons[arm] = safe_curve(curves["anchor"], curves[arm])
        results[codec] = dict(curves=curves, comparisons=comparisons)

    return dict(split=split, results=results, codecs=cfg["codecs"], qps=cfg["qps"], count=len(records))


def generate_report(test_summary, cfg):
    lines = [
        "# Báo Cáo Đánh Giá Object Detection (COCO 2017) Trên V34",
        "",
        f"**Tập dữ liệu:** COCO 2017 ({test_summary['count']} ảnh TEST ngẫu nhiên, phân vùng hash cố định)",
        "**Mô hình đánh giá:** Faster R-CNN ResNet-50 FPN (Torchvision official COCO_V1)",
        f"**Codec kiểm tra:** H.264 (x264) và H.265 (x265) | Preset: {cfg['preset']}",
        f"**Các mức QP:** {', '.join(map(str, cfg['qps']))}",
        "",
        "## 1. Bảng Tổng Hợp BD-Rate Trên COCO",
        "",
        "| Codec | Nhánh (Arm) | BD-rate (%) | PCHIP BD-rate (%) | BD-quality (mAP pp) | Trạng thái |",
        "|---|---|---:|---:|---:|---:|"
    ]
    for codec, data in test_summary["results"].items():
        comp = data["comparisons"]
        for arm in sorted(comp.keys()):
            c = comp[arm]
            bd = f"{c['bd_rate_pct']:.2f}%" if c.get("bd_rate_pct") is not None else "N/A"
            pchip = f"{c['pchip_bd_rate_pct']:.2f}%" if c.get("pchip_bd_rate_pct") is not None else "N/A"
            qual = f"{c['bd_quality_pp']*100:+.2f}" if c.get("bd_quality_pp") is not None else "N/A"
            status = "Tiết kiệm bitrate" if c.get("bd_rate_pct") and c["bd_rate_pct"] < 0 else "Không tiết kiệm"
            lines.append(f"| {codec} | `{arm}` | {bd} | {pchip} | {qual} | {status} |")

    lines.extend([
        "",
        "## 2. Chi Tiết Rate (bpp) và mAP@[.50:.95] Theo Từng QP",
        ""
    ])
    for codec, data in test_summary["results"].items():
        curves = data["curves"]
        lines.extend([
            f"### Codec: {codec.upper()}",
            "",
            "| QP | Metric | Anchor | Fixed (Action 11) | Selected 303401 | Δ mAP (303401) | Δ Rate % (303401) |",
            "|---:|---|---:|---:|---:|---:|---:|"
        ])
        for idx, qp in enumerate(cfg["qps"]):
            a_bpp = curves["anchor"]["bpp"][idx]
            a_map = curves["anchor"]["quality"][idx]
            f_bpp = curves["fixed"]["bpp"][idx]
            f_map = curves["fixed"]["quality"][idx]
            s_arm = "selected_303401" if "selected_303401" in curves else list(curves.keys())[-1]
            s_bpp = curves[s_arm]["bpp"][idx]
            s_map = curves[s_arm]["quality"][idx]
            d_map = (s_map - a_map) * 100
            d_rate = (s_bpp - a_bpp) / a_bpp * 100
            lines.append(f"| {qp} | mAP@[.5:.95] | {a_map:.4f} | {f_map:.4f} | {s_map:.4f} | {d_map:+.2f} pp | {d_rate:+.2f}% |")
            lines.append(f"| {qp} | Rate (bpp) | {a_bpp:.4f} | {f_bpp:.4f} | {s_bpp:.4f} | - | - |")
        lines.append("")

    lines.extend([
        "## 3. Nhận Xét & Kết Luận",
        "- **Hiệu quả nén OD:** Xem xét BD-rate âm thể hiện mức giảm bitrate ở cùng mAP.",
        "- **Độ chính xác Faster R-CNN:** Bounding boxes được decode từ dòng video thực, giải mã ngược về tọa độ gốc và đánh giá qua `COCOeval` chính thức.",
        "- **Hoàn toàn độc lập:** Quá trình huấn luyện controller chỉ dùng tập train/cal, không chạm vào nhãn ground-truth của tập DEV/TEST."
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    run_benchmark(parse_args())
