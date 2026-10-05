# V25 AR DEV128: completed and audited

Private Kaggle job: [baoancut/v25-ranking-ar-s302101](https://www.kaggle.com/code/baoancut/v25-ranking-ar-s302101).
Executable: `0bfcead429dcde08d0e67cb0589216aa7a357131`.
TRAIN: 512 sources / 512 measured source-codec-QP records, 1,500 replay steps,
seed 302101. Evaluation: fixed DEV128, five QPs, two codecs, 2,000 paired
source bootstrap draws. No TEST evaluation or main promotion.

An independent artifact audit verified all 1,280 operating points and guarded
minimum-byte selections. All 1,280 anchor streams exactly match V24; all
1,280 control streams exactly match V24 adaptive. Comparisons are paired.
One Windows/Linux bootstrap endpoint differed by 5.1e-15; only endpoint
roundoff up to 1e-12 is accepted, with draw counts and other fields exact.

| Codec | Held-out r2plus1d BD-rate | PCHIP | 95% paired bootstrap CI | Worst QP top1 gap |
|---|---:|---:|---:|---:|
| H.264 | -4.1222% | -2.1191% | [-10.8818%, +1.4122%] | -5.46875 pp at QP35 |
| H.265 | -3.0039% | -3.3904% | [-5.8452%, +1.2831%] | -3.90625 pp at QP40 |

Both codec screening gates fail. The r3d evaluator also participates in the
encoder guard; its BD-rates of -7.8088% and -4.0947% do not establish transfer
to an independent evaluator. Neither result reaches the -10% target.

| Codec/QP | Total byte saving vs anchor | Additional policy saving vs controls | Learned selected /128 |
|---|---:|---:|---:|
| H.264/40 | 8.2044% | 0.8559 pp | 15 |
| H.264/45 | 2.9449% | 0.1127 pp | 5 |
| H.264/50 | 0.9345% | 0.3020 pp | 7 |
| H.265/40 | 4.1168% | 0.6415 pp | 15 |
| H.265/45 | 0.8288% | 0.2004 pp | 9 |
| H.265/50 | 0.1452% | 0.0445 pp | 3 |

Learned proposals win 131/1,280 points: 97 pure filters and 34 resampling
actions. They save 43,255 additional bytes over controls, or 0.7166 percentage
points of anchor bytes. At QP40/45/50 combined, the increment is 9,106 bytes,
or 0.4004 pp. However, adaptive uses 12,182 more bytes than global static and
18,901 more than the TRAIN codec/QP static comparator. The latter comparator
has incomplete evaluator scoring and is a rate-only attribution result.
More learned selections therefore do not prove useful conditional ranking.

Evidence is preserved locally under
`D:/STUDY/LAB/bao_1/output/adaptive_v25_results/full`: the original archive,
extracted TRAIN/eval files, `audit.json`, `independent.json` and paired-check
script. The accuracy losses and weak high-QP capacity remain unresolved.
