# V27: guarded learned portfolio and codec reconstruction preprocessing

The user authorized continued adaptive preprocessing development, GitHub upload
and Kaggle execution. This experiment extends V26, whose audited DEV128 has
only 482/1280 bank actions beating guarded controls, of which K3 retrieves 160.
At QP40/45/50 these numbers are 236 and 51. Covering every operating point with
a proposal is different from winning every point with a smaller safe stream.

## Registered changes

1. Learn a three-action portfolio from measured TRAIN marginal utility. Greedy
selection maximizes weighted per-source best utility rather than selecting
three independent expected scores, so complementary feasible actions can enter
the set. A compact nearest-context kernel adapts TRAIN group weights using the
existing encoder-only 41D source/teacher/actual-anchor-BPP context. Fit scalers,
memory, priors and hyperparameters exclusively inside source-blocked TRAIN CV.
Use a frozen grid: neighbors 8/16/32, mix 0/.5/1; low-QP selection maximizes
OOF actual bytes, high-QP selection maximizes macro codec/QP mean marginal
percentage. Equal results prefer prior-only then larger neighborhoods. No DEV
checkpoint selection. Report global and group greedy portfolios at identical K3.
2. Add eight same-geometry actions that shrink source-to-anchor reconstruction
residual: uniform/core blends .25/.5/.75/1. Full core mask preserves source
pixels exactly. Anchor reconstruction is already available at the encoder.
It may guide pixels and policy before proposed trials, but unproposed candidate
results must never enter primary ranking. Previous 34 actions retain identical
order and pixels. New actions are hypotheses, not promised savings.
3. Measure 128 TRAIN clips in all ten codec/QP groups (1280 records), shuffle
source order by seed, and keep all groups of a source in the same CV fold.
Cache per-source teacher predictions/saliency while visiting its ten groups.
DEV128 has 1280 different points; true labels and evaluator predictions never
enter fitting. Dense coverage trades fewer unique sources for complete QP
trajectories and must be assessed on held-out clips.

## Interfaces and compatibility

`anchor_bank.py`: ACTION_NAMES (V26 prefix + eight),
`build_anchor_bank(clip, protection, qp, anchor_decoded)` and
`build_anchor_actions(clip, protection, qp, anchor_decoded)` return Candidates.
Reject missing/mismatched reconstruction; source/anchor are uint8 RGB THWC.

`portfolio_ranking.py`: PortfolioRankPreprocessor schema
`adaptive-vcm-portfolio-v6`, fit_portfolio_model using V26's exact integer-byte
inputs, load_portfolio_preprocessor. Support rank/proposal_details,
static_action_order, group_static_action_order, checkpoint_state and explicit
registered bank validation. Prior-only status is per operating point.

`train_portfolio.py`: dense collector and source-blocked fitting, registered
config v27_screen.json; final checkpoint includes measured stream/record hashes.
Selection dispatch constructs bank after encoding anchor. Older V25/V26
checkpoints and selectors retain their behavior. Rendering requires anchor
reconstruction for the new bank; fail closed without it.

## Invariants and gates

Two encoder teachers r3d_18/mc3_18; strict anchor_relative_v2, KL slack .1;
>=1% actual byte savings including headers; fixed controls and K3. Independent
primary evaluator r2plus1d_18. Resolution128,16frames,stride2,presetmedium,fps25;
QP30/35/40/45/50. No guard loosening, equal-byte learned credit, DEV memorization,
unproposed bank leakage or TEST inspection. Count neural pixel weights honestly:
this is a learned conditional policy over registered pixel directions.

## Validation and execution

TDD for complementary retrieval, source fold isolation, integer equality,
checkpoint replay, anchor geometry/core preservation, dense group collection,
and immutable Kaggle packaging. Synthetic FFmpeg high-QP probe measures bytes,
without claiming teacher safety. Replay existing 512 TRAIN rows to compare OOF
against V26. Full local suite and independent review precede GitHub push.
Kaggle uses baoancut, TRAIN128*10/DEV128, frozen seed302201, bootstrap2000.
Dense TRAIN4*10 and actual H264/H265 integration checks provide the local smoke
gate. Previous V26 T4 full runtime6.32h establishes the runtime baseline; cache
source teachers across ten groups. Existing qk jobs remain untouched. TEST is a fresh
merge gate only after DEV quality passes. V24 main is not promoted on DEV gains.

## Research basis

QP-conditioned preprocessing and encoder awareness are motivated by Guo Lu et
al., IEEE TCSVT2024 (https://arxiv.org/abs/2206.05650), and Chadha et al.,
CVPR2021 (https://openaccess.thecvf.com/content/CVPR2021/papers/Chadha_Deep_Perceptual_Preprocessing_for_Video_Coding_CVPR_2021_paper.pdf).
The reconstruction blend and portfolio are registered engineering hypotheses;
published gains are not attributed to this implementation.
