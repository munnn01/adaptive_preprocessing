# V25: high-QP capacity and measurable policy contribution

## Registered comparison

Initial material is main commit `308990a`, with executable V24 commit
`6e595f0d30648723db6ad29bd13ae7f7097c5fe2`. Its DEV128 primary adaptive streams
equal controls at every operating point. V24 TRAIN QP50 found only12/279
feasible oracle edits; learned selected identity993/1000 training steps and
all1280 DEV operating points. These are separate capacity and learning failures.

V25 keeps original16 frames,128 source size, stride2,25fps, both codecs at
QP30/35/40/45/50 and medium preset. All elementary-stream headers count, and
resized streams always use original T*H*W as the rate denominator. The two
teachers, strict anchor decision guard, relative KL slack0.1 and minimum1%
actual saving are unchanged. AR primary evaluators remain r2plus1d_18 and r3d_18;
r2plus1d_18 never supplies a training signal. OD continues to use V23.

The objective is increased high-QP actual-byte saving, positive marginal saving
from learned proposals, and more teacher-feasible TRAIN choices while retaining
task accuracy. The final VCM target remains strict BD-rate<-10% for each codec
and each primary analyzer, with all existing quality and uncertainty gates.
More selected points alone do not establish success.

## Two falsifiable changes

The fixed18-action bank (identity plus17) adds mild detail soft thresholding,
chroma denoising, source-motion-gated temporal denoising and resize combinations
at120/104/88 for128 sources. Protected and uniform variants are explicit.
See [bank definitions](V25_BANK.md); this is an engineering hypothesis about
additional feasible rate leverage, not a guarantee.

The ranking model receives source statistics, QP/codec/protection and source
versus anchor predictions from the frozen two encoder teachers. It predicts
safety and measured log-byte ratios for all17 actions. TRAIN collects all
actual action measurements before minibatch replay. It receives no identity
winner-class labels, no true labels, and no primary held-out model outputs.
There is no learned pixel-to-pixel neural filter in V25: learning chooses among
fixed QP-conditioned pixel operators. See [policy objective](V25_POLICY.md).

## Arms and attribution

| Arm | Candidate pool |
|---|---|
| anchor | Exact source encoded by the same standard codec |
| controls | Original V24 analytic controls |
| adaptive | Controls plus learned top3 distinct nonidentity bank actions |
| static_adaptive | Controls plus TRAIN mean-utility static top3 actions |
| learned_guarded | Identity plus learned top3 actions, same actual teacher guard |
| learned_raw | Learned top1 without guard; diagnostic only |
| bank_oracle | Controls plus all17 bank actions; audit upper bound |

All arms reuse identical encoded streams and teacher observations at an
operating point. Full-bank observations cannot influence learned context,
ranking or its primary candidate subset. There is no artificial learned tie
preference. `policy_contribution` records marginal bytes over controls and over
the equal-budget static arm, operating-point counts, wins and losses. Oracle
results do not count as learned results. An ablation run includes full-bank
measurement cost; production adaptive mode encodes only controls and proposals.

## Evidence sequence

1. Local tests and synthetic actual-codec probe establish accounting, safety
   gating and learnability, without task-performance claims.
2. Pilot:80 TRAIN source/group measurements,1500 replay updates, first16 DEV
   sources, full registered codec/QP grid. It is a development diagnostic.
3. Full development:512 TRAIN measurements/sources,1500 updates,128 DEV sources,
   2000 paired bootstrap draws. Final-LAST checkpoint; no DEV model selection.
4. If a candidate warrants promotion, compare against V24 on128 previously
   unused TEST sources (source MD5 bucket0; TRAIN>=2 and DEV1). Run the held-out
   comparison in a fresh checkout, after freezing the candidate. Do not tune on
   TEST. Main is not promoted based only on synthetic evidence or DEV wins.

Arbor research state is kept separately under
`D:/STUDY/LAB/bao_1/output/adaptive_v25_research/.arbor`. Experimental code is
published on a named branch so Kaggle can check out an immutable commit; that
publication does not mean the held-out merge gate has passed.

## Primary literature and scope of inference

Lu et al., [Preprocessing Enhanced Image Compression for Machine Vision](https://arxiv.org/abs/2206.05650)
motivates QP-adaptive semantic preprocessing before an unchanged conventional
codec. Its reported gains are not imported into this experiment.

Pang et al., [Adaptive High-Frequency Preprocessing for Video Coding](https://arxiv.org/abs/2508.08849)
describes predicting filtering strategies using comparisons across types and
strengths. It is a2025 preprint about perceptual quality. V25 borrows the
strategy-selection idea as an inference and replaces human quality supervision
with measured encoder-teacher safety and actual bytes; it is not a reproduction.

Chadha and Andreopoulos, [Deep Perceptual Preprocessing for Video Coding, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chadha_Deep_Perceptual_Preprocessing_for_Video_Coding_CVPR_2021_paper.pdf)
provides a published example of learnable preprocessing ahead of standard video
coding. V25 requires AR validation because perceptual results cannot establish
machine-task accuracy.
