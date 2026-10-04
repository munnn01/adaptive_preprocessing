"""Synthetic codec-mechanism check. This cannot measure Top-1 or COCO mAP."""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_vcm.codec import StandardCodec
from adaptive_vcm.preprocessing import suppress

output = Path(sys.argv[1])
rng = np.random.default_rng(20261004)
source = np.clip(128 + rng.normal(0, 25, (8, 128, 128, 3)), 0, 255).astype(np.uint8)
source[:, 48:80, 48:80] = [240, 40, 40]
protection = np.zeros((128, 128), np.float32)
protection[40:88, 40:88] = 1
trial = suppress(source, protection, sigma=2.5, strength=.8, temporal=.35)
np.testing.assert_array_equal(source[:, 40:88, 40:88], trial[:, 40:88, 40:88])
rows = []
for name in ("h264", "h265"):
    for qp in (30, 40, 50):
        codec = StandardCodec(name, qp)
        anchor, preprocessed = codec.roundtrip(source), codec.roundtrip(trial)
        rows.append({"codec": name, "qp": qp, "anchor_bytes": anchor.coded_bytes,
                     "preprocessed_bytes": preprocessed.coded_bytes,
                     "same_qp_rate_change_pct": (preprocessed.coded_bytes / anchor.coded_bytes - 1) * 100})
report = {"source": "synthetic noisy static background with protected object",
          "seed": 20261004, "exact_precodec_roi_preservation": True,
          "real_codec_frame_count_verified": True, "task_accuracy_measured": False,
          "bd_rate_measured": False, "target_confirmed": False, "rows": rows}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(report, indent=2, allow_nan=False))
