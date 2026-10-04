# First V22 GPU execution

Submitted and checked manually on 2026-10-04 (Asia/Bangkok).
Kaggle acknowledged version1 for both private free-T4 notebooks. Latest
API status for both: `KernelWorkerStatus.COMPLETE`; downloaded logs report
`[exit] status=0`. See [audited results](RESULTS_v22_2026-10-04.md).

Immutable experiment source: `14b1e21d9f3e899484d6abfda66aece25e8c4d82`.
Later documentation commits do not change the notebook's pinned source.

| Task | Notebook | Fixed budget | Task axis |
|---|---|---|---|
| AR | [qktttttttttt/v22-adaptive-ar-s302001](https://www.kaggle.com/code/qktttttttttt/v22-adaptive-ar-s302001) |512 TRAIN sources,1000 steps,seed302001;128 DEV clips,both codecs,five QPs,2000 paired bootstrap draws|Top-1, r2plus1d_18 and r3d_18|
| OD | [baoancut/v22-adaptive-od-s302001](https://www.kaggle.com/code/baoancut/v22-adaptive-od-s302001) |512 hash-defined TRAIN images,1000 steps,seed302001;100 DEV images,both codecs,five QPs,200 paired bootstrap draws|COCO mAP, Faster R-CNN ResNet50|

Both jobs produced the 1000-step final-LAST checkpoint and complete paired
development outputs. Metrics and paired bootstrap intervals have been
recomputed locally. Neither task passes the joint target; no independent
confirmation exists. The local fixture tests are not task experiments.
Training seed is302001; evaluation/bootstrap seed is20261004 from the pinned
configuration. OD is a single-frame detection pilot, not a video-OD benchmark.

Prepared notebook payloads passed Bash syntax checks and an exact comparison
against credential values from the external pool. No pool file, credential,
LAB PDF, input dataset or historical checkpoint was uploaded. The notebooks
mount existing datasets and clone the pinned public repository.

To collect manually:

```bash
python scripts/kaggle_runner.py status --account qktttttttttt --slug v22-adaptive-ar-s302001 --pool /path/to/pool.json
python scripts/kaggle_runner.py download --account qktttttttttt --slug v22-adaptive-ar-s302001 --pool /path/to/pool.json
python scripts/kaggle_runner.py status --account baoancut --slug v22-adaptive-od-s302001 --pool /path/to/pool.json
python scripts/kaggle_runner.py download --account baoancut --slug v22-adaptive-od-s302001 --pool /path/to/pool.json
```

No scheduled polling or subsequent experiment is created by this submission.
