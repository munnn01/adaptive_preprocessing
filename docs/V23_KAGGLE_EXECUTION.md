# V23 GPU execution receipts — 2026-10-04

Kaggle acknowledged version1 for both private notebooks. Manual status
queries after submission report **KernelWorkerStatus.RUNNING** for both.
Free NvidiaTeslaT4 is requested; GPU availability is asserted by notebook.
This records successful submission/running state, not completed training or
verified task accuracy.

Immutable experiment source: `e31eed85ba1ea67687fd1cdbf3d1e9db4aa1c5d4`.
Subsequent documentation-only commits do not change the pinned notebooks.

| Task | Notebook | Fixed train budget | Paired development evaluation |
|---|---|---|---|
| AR | [qktttttttttt/v23-rateaware-ar-s302001](https://www.kaggle.com/code/qktttttttttt/v23-rateaware-ar-s302001) |512 TRAIN sources;1000steps;seed302001;fresh width24|128 DEV clips;H.264/H.265;QP30/35/40/45/50;2000 paired bootstrap draws;Top-1 r2plus1d_18/r3d_18|
| OD | [baoancut/v23-rateaware-od-s302001](https://www.kaggle.com/code/baoancut/v23-rateaware-od-s302001) |512 TRAIN COCO images;1000steps;seed302001;fresh width24|100 DEV images;H.264/H.265;same5 QPs;200 paired image bootstrap draws;ResNet50 COCO mAP|

Both jobs use `configs/v23_screen.json`, measured-rate SPSA, TRAIN-only
feasible probes and final-LAST checkpoint. Evaluation runs `--ablate-learned`
with raw learned, guarded learned and controls curves. Component estimates
have no bootstrap CI; the primary adaptive arm has the fixed CI budget.
QP40/45/50 also receive explicit byte and task-quality gap diagnostics.

The DEV IDs/config resolution are kept comparable to V22; recipe changes
include neural parameterization, training objective, AR mask and decision
guard. This is a recipe comparison with a registered within-V23 component
ablation, not an architecture-only causal result. OD remains an image pilot.

Prepared payloads passed Bash syntax, commit pin, private GPU metadata and
credential checks. Token remains in the external pool and subprocess env;
no credential, licensed LAB PDF, input dataset or old checkpoint was uploaded.
The source repository and logs explicitly distinguish synthetic codec smoke
from real task experiments. No V23 BD-rate target pass is claimed.

Manual collection:

```bash
python scripts/kaggle_runner.py status --account qktttttttttt --slug v23-rateaware-ar-s302001 --pool /path/to/pool.json
python scripts/kaggle_runner.py download --account qktttttttttt --slug v23-rateaware-ar-s302001 --pool /path/to/pool.json
python scripts/kaggle_runner.py status --account baoancut --slug v23-rateaware-od-s302001 --pool /path/to/pool.json
python scripts/kaggle_runner.py download --account baoancut --slug v23-rateaware-od-s302001 --pool /path/to/pool.json
```

No scheduled polling, unattended follow-up or additional experiment is created.
