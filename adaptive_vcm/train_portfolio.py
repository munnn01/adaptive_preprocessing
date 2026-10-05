"""Collect a complete TRAIN codec/QP grid and fit a guarded K3 portfolio.

Learning sees actual TRAIN streams, never DEV/TEST or evaluator labels. The
learned policy chooses registered pixel directions; it has no neural pixel CNN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .analyzers import ActionAnalyzer
from .codec import StandardCodec, reference_bpp
from .data import ar_plan, fingerprint, partition, read_video
from .evaluate import ROOT, code_manifest, validate_config, write_json
from .preprocessing import action_protection, make_candidates, normalize_map
from .ranking import measurement_targets
from .rateaware import semantic_protection
from .selection import Observation, relative_guard, select
from .train_utility import _append
from .utility_ranking import CONTEXT_DIM, CONTEXT_SCHEMA, build_utility_context


def dense_schedule(plan, qps, seed):
    """Every TRAIN source visits every codec/QP once; cache one source at a time."""
    if not plan or list(qps) != [30, 35, 40, 45, 50]:
        raise ValueError('complete registered codec/QP grid is required')
    rng = np.random.default_rng(seed)
    groups = [(codec, qp) for codec in ('h264', 'h265') for qp in qps]
    return [(int(index), *groups[int(group)]) for index in rng.permutation(len(plan))
            for group in rng.permutation(len(groups))]


def capacity_diagnostics(records, prefix_size):
    result = {}
    for codec, qp in sorted({(r['codec'], r['qp']) for r in records}):
        rows = [r for r in records if (r['codec'], r['qp']) == (codec, qp)]
        old_points = new_points = old_gain = new_gain = 0
        for row in rows:
            anchor = row['actions'][0]['coded_bytes']
            baseline = row['controls_coded_bytes']
            valid = [a['coded_bytes'] if (all(a['decisions']) and max(a['distances']) <= .1
                     and a['coded_bytes'] <= anchor * .99) else baseline for a in row['actions']]
            old = baseline - min([baseline] + valid[1:prefix_size])
            new = baseline - min([baseline] + valid[1:])
            old_points += old > 0
            new_points += new > 0
            old_gain += old
            new_gain += new
        result[f'{codec}/{qp}'] = dict(records=len(rows), v26_marginal_records=int(old_points),
            expanded_marginal_records=int(new_points), v26_marginal_saved_bytes=int(old_gain),
            expanded_marginal_saved_bytes=int(new_gain))
    return result


def train(args):
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(cfg)
    if (cfg.get('ar_training') != 'source_validated_portfolio' or
            not cfg.get('ar_require_anchor_decision') or cfg.get('ar_guard_rule') != 'anchor_relative_v2'
            or cfg['ar_kl_slack'] != .1 or cfg['min_savings'] != .01 or cfg.get('rank_top_k') != 3
            or cfg['qps'] != [30, 35, 40, 45, 50]):
        raise ValueError('V27 requires the registered portfolio and unchanged strict guard')
    if args.count < 4 or args.measurements != args.count * 10:
        raise ValueError('complete TRAIN codec/QP grid requires count*10 measured records')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('training output must be empty')
    from .anchor_bank import ACTION_NAMES, build_anchor_bank
    from .stabilized_bank import ACTION_NAMES as V26_NAMES
    from .portfolio_ranking import fit_portfolio_model
    torch.set_num_threads(2)
    plan, _ = ar_plan(args.root, 'train', args.count)
    if len(plan) != args.count or any(partition(r['id']) != 'train' for r in plan):
        raise ValueError('incomplete or held-out source in TRAIN collection')
    schedule = dense_schedule(plan, cfg['qps'], args.seed)
    if len(schedule) != args.measurements:
        raise ValueError('complete TRAIN codec/QP grid is missing measured records')
    args.out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teachers = [ActionAnalyzer(name, device) for name in cfg['ar_teachers']]
    manifest = dict(schema='adaptive-vcm-training-v6', task='ar', seed=args.seed,
        measurements=args.measurements, device=device, config=cfg, code=code_manifest(),
        train_ids=[r['id'] for r in plan], train_ids_sha256=fingerprint([r['id'] for r in plan]),
        action_names=list(ACTION_NAMES), context_dim=CONTEXT_DIM, context_schema=CONTEXT_SCHEMA,
        collection='complete TRAIN source x codec x QP; actual streams including headers',
        guard='strict anchor_relative_v2; KL slack .1; >=1% actual byte saving',
        source='TRAIN-only sourceblocked CV; no evaluator labels or DEV/TEST checkpoint selection',
        limitations='Learned conditional portfolio over fixed pixel directions, not a neural pixel CNN.')
    write_json(args.out / 'training_manifest.json', manifest)
    records, contexts, safety_rows, rate_rows, eligible_rows, baselines = [], [], [], [], [], []
    previous_source = None
    for measurement, (source_index, codec_name, qp) in enumerate(schedule, 1):
        item = plan[source_index]
        if previous_source != source_index:
            clip = read_video(item['path'], cfg['frames'], cfg['ar_size'], cfg['temporal_stride'])
            source_predictions = [t.probabilities(clip) for t in teachers]
            semantic = np.maximum.reduce([normalize_map(t.saliency(clip)) for t in teachers])
            protection = semantic_protection(semantic)
            control_protection = action_protection(clip, semantic)
            source_hash = hashlib.sha256(clip.tobytes()).hexdigest()
            previous_source = source_index
        codec = StandardCodec(codec_name, qp, cfg['preset'], cfg['fps'])
        anchor = codec.roundtrip(clip)
        anchor_predictions = [t.probabilities(anchor.decoded) for t in teachers]
        context = build_utility_context(clip, qp, codec_name, protection, source_predictions,
            anchor_predictions, anchor_bpp=reference_bpp(anchor.coded_bytes, clip.shape))
        bank = build_anchor_bank(clip, protection, qp, anchor.decoded)
        if tuple(c.name for c in bank) != tuple(ACTION_NAMES):
            raise ValueError('bank action order changed during collection')
        prediction_cache = {hashlib.sha256(anchor.data).hexdigest(): anchor_predictions}
        pixel_cache = {}
        def measure(candidate, identity=False):
            key = (candidate.clip.shape, hashlib.sha256(candidate.clip.tobytes()).digest())
            stream = anchor if identity else pixel_cache.get(key)
            if stream is None:
                stream = codec.roundtrip(candidate.clip)
            pixel_cache[key] = stream
            stream_hash = hashlib.sha256(stream.data).hexdigest()
            if stream_hash not in prediction_cache:
                prediction_cache[stream_hash] = [t.probabilities(stream.decoded) for t in teachers]
            distances, decisions = relative_guard('ar', source_predictions, anchor_predictions,
                prediction_cache[stream_hash], cfg)
            return dict(name=candidate.name, coded_bytes=stream.coded_bytes,
                distances=[float(d) for d in distances], decisions=list(decisions),
                identity_stream=stream.data == anchor.data, stream_sha256=stream_hash,
                shape=list(candidate.clip.shape), codec_seconds=stream.seconds)
        observations = [measure(c, i == 0) for i, c in enumerate(bank)]
        controls = [measure(c, i == 0) for i, c in enumerate(
            make_candidates(clip, control_protection, 'ar', qp, cfg['ar_candidates']))]
        winner = select([Observation(c['name'], c['coded_bytes'], tuple(c['distances']),
                          tuple(c['decisions'])) for c in controls], cfg['ar_kl_slack'], cfg['min_savings'])
        baseline = math.log(controls[winner]['coded_bytes'] / anchor.coded_bytes)
        safe, rate, eligible = measurement_targets(observations, slack=cfg['ar_kl_slack'],
                                                  min_savings=cfg['min_savings'])
        contexts.append(context); safety_rows.append(safe); rate_rows.append(rate)
        eligible_rows.append(eligible); baselines.append(baseline)
        row = dict(measurement=measurement, source_id=item['id'], codec=codec_name, qp=qp,
            source_sha256=source_hash, source_shape=list(clip.shape), context=context.tolist(),
            anchor_decoded_sha256=hashlib.sha256(anchor.decoded.tobytes()).hexdigest(),
            actual_anchor_bpp=reference_bpp(anchor.coded_bytes, clip.shape),
            controls=controls, controls_selected=controls[winner]['name'],
            controls_coded_bytes=controls[winner]['coded_bytes'], baseline_log_rate=baseline,
            teacher_safe_actions=int(safe.sum()), feasible_actions=int(eligible.sum()), actions=observations)
        records.append(row)
        _append(args.out / 'measurements.jsonl', row)
        if measurement == 1 or measurement % 25 == 0 or measurement == args.measurements:
            print(json.dumps({k: row[k] for k in ('measurement', 'source_id', 'codec', 'qp',
                'teacher_safe_actions', 'feasible_actions', 'controls_coded_bytes')}), flush=True)
    context, safety, rate, eligible = map(np.stack, (contexts, safety_rows, rate_rows, eligible_rows))
    np.savez_compressed(args.out / 'train_records.npz', context=context, safety=safety, log_rate=rate,
        eligible=eligible, baseline_log_rate=np.asarray(baselines, dtype=np.float64))
    model, fit = fit_portfolio_model(context, safety, rate, [r['source_id'] for r in records],
        action_names=ACTION_NAMES, min_savings=cfg['min_savings'], baseline_log_rate=np.asarray(baselines),
        action_bytes=np.asarray([[a['coded_bytes'] for a in r['actions'][1:]] for r in records]),
        anchor_bytes=np.asarray([r['actions'][0]['coded_bytes'] for r in records]),
        control_bytes=np.asarray([r['controls_coded_bytes'] for r in records]))
    capacity = capacity_diagnostics(records, len(V26_NAMES))
    write_json(args.out / 'fit_diagnostics.json', fit)
    write_json(args.out / 'capacity_diagnostics.json', capacity)
    manifest.update(train_records_sha256=hashlib.sha256((args.out / 'train_records.npz').read_bytes()).hexdigest(),
        measurements_sha256=hashlib.sha256((args.out / 'measurements.jsonl').read_bytes()).hexdigest(),
        fit_diagnostics=fit, capacity_diagnostics=capacity, static_action_order=model.static_action_order)
    write_json(args.out / 'training_manifest.json', manifest)
    state = model.checkpoint_state()
    state.update(seed=args.seed, measurements=args.measurements, fit_method='sourceblocked_train_cv_portfolio',
        training_config=cfg, train_ids_sha256=manifest['train_ids_sha256'],
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        measurements_sha256=manifest['measurements_sha256'],
        train_records_sha256=manifest['train_records_sha256'])
    torch.save(state, args.out / 'preprocessor_last.pth')
    print(json.dumps({'fit': fit, 'capacity': capacity}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['ar'], default='ar')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/v27_screen.json')
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--measurements', type=int, default=1280)
    parser.add_argument('--seed', type=int, default=302201)
    parser.add_argument('--out', type=Path, required=True)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
