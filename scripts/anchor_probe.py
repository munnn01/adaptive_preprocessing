"""Synthetic actual-codec capacity diagnostic; never an AR quality benchmark."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from adaptive_vcm.anchor_bank import build_anchor_actions
from adaptive_vcm.codec import StandardCodec
from adaptive_vcm.stabilized_bank import build_stabilized_bank


def sources():
    rng = np.random.default_rng(27)
    mask = np.zeros((128, 128), np.float32)
    mask[40:88, 40:88] = 1
    examples = []
    for block in (1, 2):
        frame = rng.integers(40, 210, (128 // block, 128 // block, 3), dtype=np.uint8)
        frame = np.repeat(np.repeat(frame, block, axis=0), block, axis=1)
        clip = np.stack([np.clip(frame.astype(float) + rng.normal(0, 2, frame.shape), 0, 255)
                         .round().astype(np.uint8) for _ in range(16)])
        clip[:, 48:80, 48:80] = [232, 48, 16]
        examples.append((f"texture{block}", clip, mask))
    y, x = np.mgrid[:128, :128]
    field = 115 + 30 * np.sin(x / 12) + 20 * np.cos(y / 15)
    frame = np.clip(field[..., None] + rng.normal(0, 4, (128, 128, 3)), 20, 230)
    offsets = (0, 10, -10, 6, -6, 10, 0, -10) * 2
    clip = np.clip(np.stack([frame + offset for offset in offsets]), 0, 255).round().astype(np.uint8)
    examples.append(("low_frequency_flicker", clip, mask))
    return examples


def summarize(anchor_bytes, results):
    mild = [r for r in results if r["same_geometry"] and r["source_edit_mae"] <= 10]
    feasible = [r for r in mild if r["coded_bytes"] <= .99 * anchor_bytes]
    return dict(mild_saving_actions=len(feasible),
                best_mild_bytes=min([anchor_bytes] + [r["coded_bytes"] for r in mild]),
                best_guarded_mild_bytes=min([anchor_bytes] + [r["coded_bytes"] for r in feasible]))


def probe(out):
    rows = []
    for name, clip, mask in sources():
        for codec in ("h264", "h265"):
            for qp in (40, 45, 50):
                encoder = StandardCodec(codec, qp)
                anchor = encoder.roundtrip(clip)
                row = dict(source=name, source_sha256=hashlib.sha256(clip.tobytes()).hexdigest(),
                           codec=codec, qp=qp, anchor_bytes=anchor.coded_bytes)
                for label, bank in (("v26", build_stabilized_bank(clip, mask, qp)[1:]),
                                    ("anchor", build_anchor_actions(clip, mask, qp, anchor.decoded))):
                    results = []
                    for candidate in bank:
                        encoded = encoder.roundtrip(candidate.clip)
                        resized = np.stack([cv2.resize(frame, (128, 128)) for frame in candidate.clip])
                        mae = float(np.abs(resized.astype(float) - clip.astype(float)).mean())
                        results.append(dict(name=candidate.name, coded_bytes=encoded.coded_bytes,
                                            stream_sha256=hashlib.sha256(encoded.data).hexdigest(),
                                            same_geometry=candidate.clip.shape == clip.shape,
                                            source_edit_mae=mae,
                                            rate_change_pct=100 * (encoded.coded_bytes / anchor.coded_bytes - 1)))
                    row[label + "_actions"] = results
                    row.update({label + "_" + key: value for key, value in summarize(anchor.coded_bytes, results).items()})
                row["new_bank_extra_guarded_mild_bytes"] = max(
                    0, row["v26_best_guarded_mild_bytes"] - row["anchor_best_guarded_mild_bytes"])
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}), flush=True)
    output = dict(scope="Synthetic16x128x128 RGB, medium H264/H265,25fps QP40/45/50; actual elementary stream bytes including headers",
                  mild_definition="Same geometry and source-space RGB MAE<=10; diagnostic does not establish teacher safety",
                  teacher_feasibility=None, dev_score=None, bd_rate=None, top1=None, rows=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/anchor_probe.json")
    probe(parser.parse_args().out)
