"""Measure fixed filter strengths on TRAIN and fit a source-validated rate policy.

Candidate feasibility always uses actual streams and the existing two teachers.
No evaluator, true label, DEV source, or TEST source enters policy fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

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
from .task_bank import ACTION_NAMES as V25_NAMES


def _append(path, row):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')


def capacity_diagnostics(records, names, cfg):
    """Paired old/new feasibility on the very same source/codec/QP measurements."""
    result = {}
    for codec, qp in sorted({(row['codec'], row['qp']) for row in records}):
        rows = [r for r in records if (r['codec'], r['qp']) == (codec, qp)]
        old_feasible, all_feasible, new_trials = [], [], []
        old_gain, all_gain = [], []
        for row in rows:
            _, rate, eligible = measurement_targets(row['actions'], slack=cfg['ar_kl_slack'],
                                                     min_savings=cfg['min_savings'])
            gains = np.where(eligible > .5, 1 - np.exp(rate), 0.)
            old = gains[:len(V25_NAMES) - 1]
            old_feasible.append(bool((old > 0).any()))
            all_feasible.append(bool((gains > 0).any()))
            old_gain.append(float(old.max()))
            all_gain.append(float(gains.max()))
            new_trials.append(int(eligible[len(V25_NAMES) - 1:].sum()))
        result[f'{codec}/{qp}'] = {
            'records': len(rows), 'v25_feasible_records': sum(old_feasible),
            'expanded_feasible_records': sum(all_feasible),
            'newly_feasible_records': sum(new and not old for old, new in zip(old_feasible, all_feasible)),
            'new_action_feasible_trials': sum(new_trials),
            'v25_oracle_mean_saving_pct': 100 * float(np.mean(old_gain)),
            'expanded_oracle_mean_saving_pct': 100 * float(np.mean(all_gain)),
            'new_nonidentity_trials': len(rows) * (len(names) - len(V25_NAMES))}
    return result


def train(args):
    from .stabilized_bank import ACTION_NAMES, build_stabilized_bank
    from .utility_ranking import (CONTEXT_DIM, CONTEXT_SCHEMA, build_utility_context,
                                  fit_utility_model)
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(cfg)
    if (cfg.get('ar_training') != 'source_validated_utility' or not cfg.get('ar_require_anchor_decision')
            or cfg['min_savings'] != .01 or cfg.get('rank_top_k') != 3):
        raise ValueError('V26 requires registered utility recipe, strict guard and unchanged K3/1% threshold')
    if getattr(args, 'reuse_records', None) is not None:
        return replay_train(args, cfg)
    if args.count < 4 or args.measurements < 2 * len(cfg['qps']):
        raise ValueError('collection needs at least four TRAIN sources and all codec/QP groups')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('training output must be empty')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plan, _ = ar_plan(args.root, 'train', args.count)
    if any(partition(r['id']) != 'train' for r in plan):
        raise ValueError('TRAIN collection contains a held-out source')
    if tuple(ACTION_NAMES[:len(V25_NAMES)]) != V25_NAMES:
        raise ValueError('expanded bank must preserve the exact V25 prefix')
    teachers = [ActionAnalyzer(name, device) for name in cfg['ar_teachers']]
    groups = [(codec, qp) for codec in ('h264', 'h265') for qp in cfg['qps']]
    weights = np.array([1 if qp < 40 else 2 if qp < 45 else 3 for _, qp in groups], float)
    weights /= weights.sum()
    manifest = {
        'schema': 'adaptive-vcm-training-v5', 'task': 'ar', 'seed': args.seed,
        'measurements': args.measurements, 'device': device, 'config': cfg,
        'train_ids': [r['id'] for r in plan],
        'train_ids_sha256': fingerprint([r['id'] for r in plan]), 'code': code_manifest(),
        'action_names': list(ACTION_NAMES), 'context_dim': CONTEXT_DIM,
        'context_schema': CONTEXT_SCHEMA,
        'qp_sampling': {f'{c}/{q}': float(p) for (c, q), p in zip(groups, weights)},
        'guard': 'unchanged strict anchor decisions, KL slack .1, >=1% actual bytes including headers',
        'source': 'TRAIN-only; fixed sourceblocked CV recipe; no DEV checkpoint/threshold selection',
        'limitations': 'Learned rate policy selects fixed pixel filters; held-out task quality is measured separately.'}
    write_json(args.out / 'training_manifest.json', manifest)
    records, contexts, safety_rows, rate_rows, eligible_rows, baseline_rows = [], [], [], [], [], []
    source_order = []
    for measurement in range(args.measurements):
        if measurement % len(plan) == 0:
            source_order = rng.permutation(len(plan)).tolist()
        item = plan[source_order[measurement % len(plan)]]
        group = measurement if measurement < len(groups) else int(rng.choice(len(groups), p=weights))
        codec_name, qp = groups[group]
        clip = read_video(item['path'], cfg['frames'], cfg['ar_size'], cfg['temporal_stride'])
        source_predictions = [t.probabilities(clip) for t in teachers]
        semantic = np.maximum.reduce([normalize_map(t.saliency(clip)) for t in teachers])
        protection = semantic_protection(semantic)
        control_protection = action_protection(clip, semantic)
        codec = StandardCodec(codec_name, qp, cfg['preset'], cfg['fps'])
        anchor = codec.roundtrip(clip)
        anchor_predictions = [t.probabilities(anchor.decoded) for t in teachers]
        context = build_utility_context(clip, qp, codec_name, protection, source_predictions,
                                        anchor_predictions, anchor_bpp=reference_bpp(anchor.coded_bytes, clip.shape))
        bank = build_stabilized_bank(clip, protection, qp)
        if tuple(c.name for c in bank) != tuple(ACTION_NAMES):
            raise ValueError('bank order changed during collection')
        cache = {hashlib.sha256(anchor.data).hexdigest(): anchor_predictions}
        pixel_cache = {}
        def measure(candidate, identity=False):
            key = (candidate.clip.shape, hashlib.sha256(candidate.clip.tobytes()).digest())
            stream = anchor if identity else pixel_cache.get(key)
            if stream is None:
                stream = codec.roundtrip(candidate.clip)
            pixel_cache[key] = stream
            stream_hash = hashlib.sha256(stream.data).hexdigest()
            if stream_hash not in cache:
                cache[stream_hash] = [t.probabilities(stream.decoded) for t in teachers]
            distances, decisions = relative_guard('ar', source_predictions, anchor_predictions,
                                                   cache[stream_hash], cfg)
            return {'name': candidate.name, 'coded_bytes': stream.coded_bytes,
                    'distances': [float(d) for d in distances], 'decisions': list(decisions),
                    'identity_stream': stream.data == anchor.data, 'stream_sha256': stream_hash,
                    'shape': list(candidate.clip.shape), 'codec_seconds': stream.seconds}
        observations = [measure(c, index == 0) for index, c in enumerate(bank)]
        controls = [measure(c, index == 0) for index, c in enumerate(
            make_candidates(clip, control_protection, 'ar', qp, cfg['ar_candidates']))]
        control_index = select([Observation(r['name'], r['coded_bytes'], tuple(r['distances']),
                                           tuple(r['decisions'])) for r in controls],
                               cfg['ar_kl_slack'], cfg['min_savings'])
        baseline = math.log(controls[control_index]['coded_bytes'] / anchor.coded_bytes)
        safe, rate, eligible = measurement_targets(observations, slack=cfg['ar_kl_slack'],
                                                   min_savings=cfg['min_savings'])
        contexts.append(context)
        safety_rows.append(safe)
        rate_rows.append(rate)
        eligible_rows.append(eligible)
        baseline_rows.append(baseline)
        utility = np.where(eligible > .5, 1 - np.exp(rate), 0.)
        best = int(utility.argmax()) + 1 if utility.max() > 0 else 0
        row = {'measurement': measurement + 1, 'source_id': item['id'], 'codec': codec_name, 'qp': qp,
               'source_sha256': hashlib.sha256(clip.tobytes()).hexdigest(), 'source_shape': list(clip.shape),
               'context': context.tolist(), 'actual_anchor_bpp': reference_bpp(anchor.coded_bytes, clip.shape),
               'teacher_safe_actions': int(safe.sum()), 'feasible_actions': int(eligible.sum()),
               'oracle_action': ACTION_NAMES[best], 'oracle_saving_pct': float(100 * utility.max()),
               'controls': controls, 'controls_selected': controls[control_index]['name'],
               'controls_coded_bytes': controls[control_index]['coded_bytes'],
               'baseline_log_rate': baseline, 'actions': observations}
        records.append(row)
        _append(args.out / 'measurements.jsonl', row)
        if measurement == 0 or (measurement + 1) % 25 == 0 or measurement + 1 == args.measurements:
            print(json.dumps({k: v for k, v in row.items() if k not in ('context', 'actions', 'controls')}), flush=True)
    context, safety, rate, eligible = map(np.stack, (contexts, safety_rows, rate_rows, eligible_rows))
    np.savez_compressed(args.out / 'train_records.npz', context=context, safety=safety,
                        log_rate=rate, eligible=eligible, baseline_log_rate=np.asarray(baseline_rows))
    model, fit = fit_utility_model(context, safety, rate, [r['source_id'] for r in records],
                                   action_names=ACTION_NAMES, min_savings=cfg['min_savings'],
                                   baseline_log_rate=np.asarray(baseline_rows),
                                   action_bytes=np.asarray([[a['coded_bytes'] for a in r['actions'][1:]] for r in records]),
                                   control_bytes=np.asarray([r['controls_coded_bytes'] for r in records]),
                                   anchor_bytes=np.asarray([r['actions'][0]['coded_bytes'] for r in records]))
    write_json(args.out / 'fit_diagnostics.json', fit)
    capacity = capacity_diagnostics(records, ACTION_NAMES, cfg)
    write_json(args.out / 'capacity_diagnostics.json', capacity)
    records_sha = hashlib.sha256((args.out / 'train_records.npz').read_bytes()).hexdigest()
    manifest.update({'train_records_sha256': records_sha,
                     'measurements_sha256': hashlib.sha256((args.out / 'measurements.jsonl').read_bytes()).hexdigest(),
                     'fit_diagnostics': fit, 'capacity_diagnostics': capacity,
                     'static_action_order': model.static_action_order})
    write_json(args.out / 'training_manifest.json', manifest)
    checkpoint = model.checkpoint_state()
    checkpoint.update({'seed': args.seed, 'measurements': args.measurements,
                       'fit_method': 'sourceblocked_train_cv_ridge',
                       'train_ids_sha256': manifest['train_ids_sha256'], 'training_config': cfg,
                       'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                       'train_records_sha256': records_sha})
    torch.save(checkpoint, args.out / 'preprocessor_last.pth')
    print(json.dumps({'fit': fit, 'capacity': capacity}), flush=True)
    return manifest


def replay_train(args, cfg):
    """Refit a numerical repair on immutable TRAIN measurements, without encoding."""
    from .utility_ranking import fit_utility_model, load_record_directory
    bundle = load_record_directory(args.reuse_records)
    original = bundle['manifest']
    if cfg != original['config'] or original['seed'] != args.seed:
        raise ValueError('cached replay must retain the registered config and seed')
    if bundle['baseline_log_rate'] is None:
        raise ValueError('V26 marginal replay requires measured guarded controls')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('replay output must be empty')
    args.out.mkdir(parents=True, exist_ok=True)
    model, fit = fit_utility_model(bundle['context'], bundle['safety'], bundle['log_rate'],
        bundle['source_ids'], bundle['action_names'], anchor_bytes=bundle['anchor_bytes'],
        baseline_log_rate=bundle['baseline_log_rate'], action_bytes=bundle['action_bytes'],
        control_bytes=bundle['control_bytes'])
    for name in ('measurements.jsonl', 'train_records.npz', 'capacity_diagnostics.json'):
        shutil.copyfile(args.reuse_records / name, args.out / name)
    manifest = {**original, 'code': code_manifest(), 'fit_diagnostics': fit,
                'static_action_order': model.static_action_order,
                'replay': {'source_measurements_sha256': bundle['measurements_sha256'],
                           'measurement_code': original['code'],
                           'reason': 'integer-byte numerical repair; identical TRAIN streams'}}
    write_json(args.out / 'training_manifest.json', manifest)
    write_json(args.out / 'fit_diagnostics.json', fit)
    checkpoint = {**model.checkpoint_state(), 'seed': args.seed,
                  'measurements': original['measurements'], 'fit_method': 'sourceblocked_train_cv_ridge',
                  'train_ids_sha256': original['train_ids_sha256'], 'training_config': cfg,
                  'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  'train_records_sha256': original['train_records_sha256']}
    torch.save(checkpoint, args.out / 'preprocessor_last.pth')
    print(json.dumps({'cached_train_refit': True, 'measurements': original['measurements'],
                      'fit': fit}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['ar'], default='ar')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/v26_screen.json')
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--measurements', type=int, default=512)
    parser.add_argument('--seed', type=int, default=302101)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-records', type=Path)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
