"""Compare source-blocked TRAIN utility retrieval with the fixed V25 MLP recipe."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adaptive_vcm.ranking import RankPreprocessor, ranking_loss, CONTEXT_SCHEMA, CONTEXT_DIM
from adaptive_vcm.utility_ranking import (fit_utility_model, load_record_directory,
                                          source_folds, _retrieval, _utility)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output file already exists")
    bundle = load_record_directory(args.records)
    if (bundle['manifest']['context_schema'] != CONTEXT_SCHEMA
            or bundle['manifest']['context_dim'] != CONTEXT_DIM):
        raise ValueError('MLP comparison requires legacy V25 measurement context')
    _, report = fit_utility_model(bundle["context"], bundle["safety"], bundle["log_rate"],
        bundle["source_ids"], bundle["action_names"], anchor_bytes=bundle["anchor_bytes"],
        action_bytes=bundle['action_bytes'])
    records = list(map(json.loads, (args.records / "measurements.jsonl").read_text().splitlines()))
    x = torch.tensor(np.asarray([row["context"] for row in records]), dtype=torch.float32)
    safety = torch.tensor(bundle["safety"], dtype=torch.float32)
    rate = torch.tensor(bundle["log_rate"], dtype=torch.float32)
    scores = np.zeros_like(bundle["safety"])
    folds = source_folds(bundle["source_ids"])
    torch.set_num_threads(2)
    for fold in range(4):
        torch.manual_seed(302101 + fold)
        rng = np.random.default_rng(302101 + fold)
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model = RankPreprocessor(64, action_names=bundle["action_names"])
        positives = safety[train].sum(0)
        weight = ((len(train) - positives) / positives.clamp(min=1.)).clamp(.25, 20.)
        model.safety_log_weight.copy_(weight.log())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for _ in range(1500):
            indices = train[rng.integers(0, len(train), size=32)]
            loss, _ = ranking_loss(model, x[indices], safety[indices], rate[indices], weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            logits, predicted_rate = model(x[valid])
            scores[valid] = model.scores(logits, predicted_rate).numpy()
        print(json.dumps({"completed_train_fold": fold}), flush=True)
    target = _utility(bundle["safety"], bundle["log_rate"], .01,
                      anchor_bytes=bundle['anchor_bytes'], action_bytes=bundle['action_bytes'])
    report["legacy_mlp_oof"] = _retrieval(scores, target, bundle["anchor_bytes"], 3)
    report["legacy_mlp_recipe"] = {"width": 64, "steps_per_fold": 1500, "batch_size": 32,
        "lr": .001, "seed": "302101 + fold", "source": "reinitialized inside each source-blocked TRAIN fold"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected_recipe", "selected_oof",
        "global_static_oof", "group_static_oof", "legacy_mlp_oof")}, indent=2))


if __name__ == "__main__":
    main()
