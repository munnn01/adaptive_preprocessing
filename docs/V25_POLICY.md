# V25 all-action ranking policy

V24 measured useful TRAIN edits but predicted identity almost everywhere. V25
removes identity from the classification problem. It collects every bank action's
actual H.264/H.265 bytes and shared encoder-teacher guard result, then replays fixed
source/group records in minibatches. This tests whether fuller supervision avoids
identity collapse; it does not establish a measured DEV improvement on its own.

`RankPreprocessor` has two nonidentity heads: teacher safety logits and log actual
byte ratio. Balanced BCE teaches safety, SmoothL1 teaches the measured log rate,
and pairwise preference loss separates useful feasible actions from alternatives.
The fixed BCE class weight is removed from inferred safety probabilities. Identity
remains an external fallback in the exact-byte selection policy.

The 1,640-dimensional context uses only source pixels, QP, codec identity, source
protection, and the two frozen encoder teachers' source and actual anchor
probabilities. Full 400-class source/anchor distributions preserve conditional
semantic information; confidence, margins, entropies, KL, Jensen-Shannon distance
and class agreement describe codec fragility. No true class label or held-out
evaluator output is a policy input or training target.

The policy always proposes the registered top K nonidentity actions, default three.
Every proposal still undergoes the unchanged actual codec and strict anchor class
guard with relative KL slack 0.1 and minimum byte saving 1%. The bank oracle and
TRAIN-derived static top K comparator must use the same guard and proposal budget
to distinguish a learned ranking benefit from merely testing more actions.

Collection defaults to 512 TRAIN source/group records with QP weights 1/1/2/3/3
for QP 30/35/40/45/50, across each codec. The first ten records cover all registered
codec/QP groups. Replay defaults to 1,500 updates of uniform minibatches of size32,
without DEV/TEST checkpoint selection. The final-LAST checkpoint stores action
order, context schema, TRAIN source fingerprint, measured archive fingerprint,
exact training configuration and the static action ranking. The fixed bank defines
the filtering strengths; learned ranking does not train those coefficients.

`measurements.jsonl` stores source pixel fingerprints, contexts, original geometry,
every action's measured bytes, stream fingerprints and teacher guard observations.
`train_records.npz` preserves all replay tensors. `fit_diagnostics.json` explicitly
reports TRAIN resubstitution only, including codec/QP feasible action counts, oracle
capacity and equal-budget learned/static retrieval. These diagnostics cannot be
used as independent accuracy or BD-rate evidence.

Run `python -m adaptive_vcm.train_ranking --root TRAIN_ROOT --config
configs/v25_screen.json --count 512 --measurements 512 --steps 1500 --batch-size 32
--out EMPTY_DIRECTORY`.
The coordinator evaluates DEV and a source-hash-separated TEST cohort independently
before promoting this candidate over V24. High-QP byte savings and held-out action
accuracy remain empirical questions.
