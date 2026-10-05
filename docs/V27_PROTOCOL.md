# V27 experimental protocol

V27 improves the proposal mechanism and adds codec-aware pixel directions. It
does not promise a safe byte win on all1280DEV points. V26's complete measured
bank provided marginal guarded savings on only482/1280points (236/768 at high
QP), whereas K3 retrieved160 (51high). Both capacity and retrieval need work.

## Registered learning and evaluation

- TRAIN128 clips, all H264/H265 × QP30/35/40/45/50 =1280actual measured banks.
- DEV128 different SHA-partitioned clips, likewise1280points; no fitting on DEV.
- Fixed seed302201; four source-blocked TRAIN folds; all groups of a source stay
  together. The frozen nine recipes use neighborhoods8/16/32 and mix0/.5/1.
- Greedy K3 optimizes complementary per-source utility. Low QP CV selects actual
  saved bytes; high QP CV selects macro codec/QP mean marginal percentage.
- Same K3 global/group greedy portfolios expose actual conditional contribution.
- Forty-two actions retain exact V26prefix34 and add eight uniform/core blends
  towards the same-QP anchor reconstruction. Fully protected core pixels remain
  source-exact. Coefficients are fixed; the conditional portfolio is learned.
- Strict two-teacher anchor_relative_v2, KL slack.1 and >=1%actual stream saving
  including headers remain unchanged. Independent primary evaluatorr2plus1d_18
  determines task quality. TEST stays uninspected until a quality-passing merge
  gate; no main promotion on DEV alone.

## Evidence before a new GPU evaluation

Replay of immutable V26 TRAIN512 measurements: portfolioOOF19530bytes/56points
versus V2618189/52. High-QP3123/25 versus3048/23. Recipe-selected OOF is not an
independent validation score. New synthetic FFmpeg anchor bank18points/756actual
roundtrips had21rate-feasible mildactions at8points but zero extra best bytes
over V26 under a surrogate MAE<=10criterion. Teacher-safe new capacity is unknown.

T4 V26full took6.32h, collection approximately1.84h. V27 caches saliency/source
predictions once per source while measuring ten groups. Local dense-collector
and actual H264/H265 integration checks plus an independent review replace a
second quota-consuming pilot. The full registered job uses baoancut; old qk jobs
remain untouched. GPU results must be audited before improvement is claimed.

```powershell
python -m adaptive_vcm.train_portfolio --task ar --root KINETICS_ROOT --count 128 --measurements 1280 --seed 302201 --out TRAIN_DIR
python -m adaptive_vcm.evaluate --task ar --root KINETICS_ROOT --config configs/v27_screen.json --checkpoint TRAIN_DIR/preprocessor_last.pth --count 128 --split dev --codecs h264 h265 --bootstrap 2000 --ablate-learned --out EVAL_DIR
python -m scripts.audit_v27 --run RUN_DIR --expected-commit IMMUTABLE_SHA --baseline-v26 V26_EVAL_DIR --out AUDIT_JSON
```

Actual job preparation/submission uses scripts/kaggle_runner.py with recipev27,
TRAIN count128/measurements1280, DEV count128 and a full40-character commit.
Credentials remain subprocess environment only and are not exported in artifacts.

## Primary research references

Lu etal., *Preprocessing Enhanced Image Compression for Machine Vision*, IEEE
TCSVT2024, motivates quantization-adaptive semantic preprocessing; see
https://arxiv.org/abs/2206.05650 and the verified citation in LITERATURE_V26.md.
Chadha etal., *Deep Perceptual Preprocessing for Video Coding*, CVPR2021, describes
encoder awareness and virtual codec constraints:
https://openaccess.thecvf.com/content/CVPR2021/papers/Chadha_Deep_Perceptual_Preprocessing_for_Video_Coding_CVPR_2021_paper.pdf.
Zhao etal., *A Preprocessing Framework for Video Machine Vision under Compression*,
DCC2024 proceedings (author preprint uploadedDecember2025), investigates a neural
preprocessor with virtual codec training and standard codecs for testing:
https://arxiv.org/abs/2512.15331. It is a conference paper, not a journal article.
The V27 reconstruction blend/greedy portfolio are engineering hypotheses; these
papers do not establish their rate or task safety.
