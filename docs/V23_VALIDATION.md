# V23 local validation — 2026-10-04

Windows CPU, Python3.11, `D:/STUDY/AI/envs/ten_env`, FFmpeg x264/x265.

- Full suite: **64 passed**, no skipped codec tests, 44,14s.
- After adding a deterministic notebook cell ID, focused runner suite:
  **5 passed**, 4,18s.
- After preventing rate credit for task-invalid probes, the entire V23 suite:
  **16 passed**, 37,52s; includes actual-codec AR/OD training and ablations.
- Actual-byte synthetic probe: all12 task/codec/QP cases decreased bytes;
  all protected source pixels remained exact before compression.

The first full invocation had a missing parent for pytest's temporary folder;
after creating it the pipeline exposed an AR NumPy-bool JSON serialization
bug. Guard decisions now use native Python bool. The final full run includes
this correction and the high-QP DC expert. No unresolved test failure remains.

These checks validate codec execution, live edit strengths, measured objective
gradients, paired component reporting and checkpoint compatibility. Task
fixtures are tiny deterministic networks, not pretrained accuracy evidence.

The following rates are measured at the same QP/preset/source using the
INITIAL, untrained policy, on a synthetic static textured background with a
flat-color protected rectangle. **No task quality was evaluated. These are
not BD-rate, Kinetics Top-1 or COCO mAP results.**

| Synthetic task | Codec | QP40 byte change | QP45 byte change | QP50 byte change |
|---|---|---:|---:|---:|
| AR16×128×128 | H.264 | −47,84% | −66,68% | −30,71% |
| AR16×128×128 | H.265 | −33,71% | −43,41% | −18,18% |
| OD1×320×320 | H.264 | −93,73% | −92,84% | −79,26% |
| OD1×320×320 | H.265 | −88,75% | −83,89% | −57,29% |

[JSON with byte counts, alpha, changed-pixel fraction and stream hashes](results/v23_synthetic_codec_probe.json).
The synthetic background is deliberately easy to simplify; these large byte
changes should not be extrapolated to natural videos/images or used to assert
the <−10% task BD-rate target. The probe demonstrates that the parameterization
can affect a real high-QP stream rather than round back to identity.
