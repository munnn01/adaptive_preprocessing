"""TRAIN-only spatial RGB imitation from complete actual H.264/H.265 grids.

Codec measurements label fixed reference profiles; Adam sees cached RGB targets
and source support only. No codec surrogate, evaluator, labels, or DEV selector
enters fitting. Identity is the target whenever a profile adds no guarded bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from .analyzers import ActionAnalyzer, DetectionAnalyzer
from .codec import StandardCodec, reference_bpp
from .data import ar_plan, od_plan, fingerprint, partition, read_video, read_image
from .preprocessing import Candidate, action_protection, boxes_to_mask, make_candidates, normalize_map
from .rateaware import semantic_protection
from .selection import Observation, relative_guard, select

ROOT = Path(__file__).resolve().parents[1]
QPS = (30, 35, 40, 45, 50)
CODECS = ('h264', 'h265')
LEARNING_RATE = 1e-3


def semantic_group_weights(records):
    """Keep every condition, equalize codec/QP groups and present target classes."""
    if not records:
        raise ValueError('nonempty semantic fitting records required')
    keys = [(r['codec'], r['qp'], r['target_profile'] != 'identity') for r in records]
    counts = Counter(keys)
    groups = {(c, q) for c, q, _ in counts}
    classes = Counter((c, q) for c, q, _ in counts)
    return [len(records) / (len(groups) * classes[k[:2]] * counts[k]) for k in keys]


def semantic_loss(output, target, aux, protection, row, group_weight):
    """Distill measured expert/strength; hard cores cannot dilute editable loss."""
    from .motion_learned import EXPERT_NAMES, PROFILE_NAMES
    positive = row['target_profile'] != 'identity'
    if positive and row['target_profile'] not in PROFILE_NAMES:
        raise ValueError('unknown semantic target profile')
    editable = 1 - protection
    denominator = editable.sum().clamp_min(1e-8)
    rgb = ((output - target).square() * editable).sum() / (3 * denominator)
    strength = int(row['target_profile'].rsplit('_', 1)[1]) / 100 if positive else 0.
    alpha = ((aux['raw_alpha'] - strength).square() * editable).sum() / denominator
    mixture = aux['mixture'].clamp_min(1e-8)
    expert = mixture.sum() * 0
    if positive:
        name = row['target_profile'][len('motion_'):].rsplit('_', 1)[0]
        index = EXPERT_NAMES.index(name)
        expert = (-mixture[:, index:index + 1].log() * editable).sum() / denominator
    control_bytes = row['controls_coded_bytes']
    marginal = row['marginal_saved_bytes'] if positive else 0
    if control_bytes <= 0 or marginal < 0 or not math.isfinite(marginal):
        raise ValueError('invalid actual byte utility')
    utility = 1 + min(1., 10 * marginal / control_bytes)
    entropy = -(mixture * mixture.log()).sum(1, keepdim=True)
    top = mixture.topk(2, dim=1).values
    parts = dict(rgb_loss=rgb, alpha_loss=alpha, expert_loss=expert,
                 utility_weight=output.new_tensor(utility),
                 alpha_mean=(aux['alpha'] * editable).sum() / denominator,
                 raw_alpha_mean=(aux['raw_alpha'] * editable).sum() / denominator,
                 expert_entropy=(entropy * editable).sum() / denominator,
                 expert_margin=((top[:, :1] - top[:, 1:]) * editable).sum() / denominator)
    return group_weight * utility * (rgb + .05 * alpha + .01 * expert), parts


def conditional_loss(output, target, aux, protection, row, group_weight, variant):
    """V30: admission on every row; measured action supervision on positives."""
    from .conditional_learned import canonical_target
    expected = canonical_target(row['target_profile'], variant)
    if row.get('target_parameters') != expected:
        raise ValueError('missing or inconsistent canonical target parameters')
    positive = expected['admission']
    editable = 1 - protection
    denominator = editable.sum().clamp_min(1e-8)
    rgb = ((output - target).square() * editable).sum() / (3 * denominator)
    strength, weights = aux['strength'], aux['expert_weights'].clamp_min(1e-8)
    gate_target = torch.full_like(aux['gate_logit'], float(positive))
    gate = F.binary_cross_entropy_with_logits(aux['gate_logit'], gate_target)
    alpha = output.sum() * 0
    expert = output.sum() * 0
    target_entropy = output.new_zeros(())
    if positive:
        alpha = (strength - expected['strength']).square().mean()
        desired = output.new_tensor(expected['expert_weights']).reshape(1,4,1,1,1)
        expert = -(desired * weights.log()).sum(1).mean()
        target_entropy = -(desired * desired.clamp_min(1e-8).log()).sum(1).mean()
    control, marginal = row['controls_coded_bytes'], row['marginal_saved_bytes']
    if (type(control) is not int or control <= 0 or type(marginal) is not int or marginal < 0
            or (positive and marginal <= 0) or (not positive and marginal != 0)):
        raise ValueError('invalid actual byte utility for conditional target')
    utility = 1 + min(1., 10 * marginal / control)
    entropy = -(weights * weights.log()).sum(1).mean()
    top = weights.topk(2, dim=1).values
    parts = dict(rgb_loss=rgb, alpha_loss=alpha, expert_loss=expert, gate_loss=gate,
        expert_kl=expert-target_entropy, target_entropy=target_entropy,
        utility_weight=output.new_tensor(utility),
        alpha_mean=(aux['alpha'] * editable).sum()/denominator,
        raw_alpha_mean=strength.mean(), gate_probability=aux['gate_probability'].mean(),
        expert_entropy=entropy, expert_margin=(top[:,0]-top[:,1]).mean())
    return group_weight * utility * (rgb + .05*alpha + .01*expert + .05*gate), parts


def conditional_bank_summary(records, baseline_names):
    """Actual feasible headroom, independently of model selection counts."""
    output = {}
    for codec, qp in sorted({(r['codec'],r['qp']) for r in records}):
        rows = [r for r in records if (r['codec'],r['qp'])==(codec,qp)]
        baseline_positive = positive = baseline_margin = margin = 0
        for row in rows:
            def saved(actions):
                feasible = [a['coded_bytes'] for a in actions if _eligible(a,
                    row['anchor_coded_bytes'],row['slack'],row['min_savings'],
                    2 if row['task']=='ar' else 1)]
                return max(0,row['controls_coded_bytes']-min(feasible)) if feasible else 0
            old = saved(row.get('baseline_profiles',
                [a for a in row['profiles'] if a['name'] in baseline_names]))
            new = saved(row['profiles'])
            baseline_positive += old > 0
            positive += new > 0
            baseline_margin += old
            margin += new
        output[f'{codec}/{qp}'] = dict(points=len(rows),
            baseline_positive_targets=baseline_positive,positive_targets=positive,
            baseline_marginal_saved_bytes=baseline_margin,marginal_saved_bytes=margin,
            extra_bank_saved_bytes=margin-baseline_margin)
    return output


def calibration_split(train_ids, seed):
    """Reserve a deterministic source-level quarter, independent of plan order."""
    if len(train_ids) < 2 or len(set(train_ids)) != len(train_ids):
        raise ValueError('calibration requires at least two unique TRAIN sources')
    count = max(1, len(train_ids) // 4)
    ranked = sorted(train_ids, key=lambda i: hashlib.sha256(f'{seed}/{i}'.encode()).hexdigest())
    selected = set(ranked[:count])
    return ([i for i in train_ids if i not in selected], [i for i in train_ids if i in selected])


def fit_admission_policy(records, task):
    """Fit nonpositive teacher-distance headroom from actual marginal byte wins."""
    if task not in ('ar', 'od') or not records:
        raise ValueError('complete TRAIN calibration grid required')
    ids = {r['source_id'] for r in records}
    expected = {(i, c, q) for i in ids for c in CODECS for q in QPS}
    actual = [(r['source_id'], r['codec'], r['qp']) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('complete TRAIN calibration grid required')
    policy, slack, teacher_count = {}, .1 if task == 'ar' else .03, 2 if task == 'ar' else 1
    for codec in CODECS:
        for qp in QPS:
            rows = [r for r in records if (r['codec'], r['qp']) == (codec, qp)]
            distances = [max(a['distances']) for r in rows for a in r['proposals']
                         if _eligible(a, r['anchor_coded_bytes'], slack, .01, teacher_count)
                         and a['coded_bytes'] < r['controls_coded_bytes'] and max(a['distances']) <= 0]
            policy[f'{codec}/{qp}'] = dict(threshold=float(np.quantile(distances, .75)) if distances else 0.,
                enabled=bool(distances), calibration_points=len(rows), n_nonworsening=len(distances))
    return policy


def dense_schedule(plan, qps, seed):
    """Visit each source's ten distinct groups together, then release its RGB."""
    if not plan or list(qps) != list(QPS):
        raise ValueError('complete registered codec/QP grid is required')
    rng = np.random.default_rng(seed)
    groups = [(c, q) for c in CODECS for q in QPS]
    return [(int(i), *groups[int(j)]) for i in rng.permutation(len(plan))
            for j in rng.permutation(len(groups))]


def _eligible(action, anchor_bytes, slack, min_savings, teacher_count=None):
    distances, decisions = action.get('distances', []), action.get('decisions', [])
    size = action.get('coded_bytes')
    return (type(size) is int and 0 < size <= anchor_bytes * (1 - min_savings)
            and bool(distances) and len(distances) == len(decisions)
            and (teacher_count is None or len(distances) == teacher_count)
            and all(type(v) is bool and v for v in decisions)
            and all(isinstance(d, (int, float)) and math.isfinite(d) and d <= slack for d in distances))


def _control_winner(controls, slack, min_savings):
    observations = [Observation(a['name'], a['coded_bytes'],
                    tuple(float(d) if d is not None else math.inf for d in a['distances']),
                    tuple(a['decisions'])) for a in controls]
    return select(observations, slack, min_savings)


def choose_training_target(controls, profiles, *, slack, min_savings=.01):
    """A profile must add actual guarded savings beyond the control winner."""
    winner = _control_winner(controls, slack, min_savings)
    anchor, baseline = controls[0]['coded_bytes'], controls[winner]['coded_bytes']
    feasible = [a for a in profiles if _eligible(a, anchor, slack, min_savings,
                len(controls[0]['distances'])) and a['coded_bytes'] < baseline]
    return min(feasible, key=lambda a: a['coded_bytes'])['name'] if feasible else 'identity'


def fit_static_orders(records, profile_names, k=3):
    """Greedy complementary marginal byte coverage, independently per group.

    A rejected profile contributes zero utility. Selecting another profile gets
    credit only for bytes it adds beyond controls and the portfolio so far.
    Equal gains retain registry order; zero-gain fillers keep the same K budget.
    """
    names = list(profile_names)
    if not records or type(k) is not int or not 1 <= k <= len(names) or len(set(names)) != len(names):
        raise ValueError('invalid static portfolio data or K')
    orders = {}
    for codec, qp in sorted({(r['codec'], r['qp']) for r in records}):
        rows = [r for r in records if (r['codec'], r['qp']) == (codec, qp)]
        utilities = []
        for row in rows:
            measured = {a['name']: a for a in row['profiles']}
            if set(measured) != set(names) or len(measured) != len(row['profiles']):
                raise ValueError('inconsistent profile registry in TRAIN records')
            utilities.append([max(0, row['controls_coded_bytes'] - measured[n]['coded_bytes'])
                              if _eligible(measured[n], row['anchor_coded_bytes'], row['slack'], row['min_savings'],
                                           2 if row.get('task') == 'ar' else 1 if row.get('task') == 'od' else None)
                              else 0 for n in names])
        utility = np.asarray(utilities, dtype=np.int64)
        covered = np.zeros(len(rows), dtype=np.int64)
        selected = []
        for _ in range(k):
            remaining = [i for i in range(len(names)) if i not in selected]
            best = max(remaining, key=lambda i: int(np.maximum(utility[:, i] - covered, 0).sum()))
            selected.append(best)
            covered = np.maximum(covered, utility[:, best])
        orders[f'{codec}/{qp}'] = [names[i] for i in selected]
    return orders


def validate_training_records(records, train_ids, task, profile_names, conditional_variant=None):
    """Check actual TRAIN provenance and the complete source/codec/QP matrix."""
    if task not in ('ar', 'od') or not train_ids or any(type(i) is not str for i in train_ids):
        raise ValueError('invalid TRAIN source identities or task')
    fingerprint(train_ids)
    if any(partition(f'coco2017/{i}' if task == 'od' else i) != 'train' for i in train_ids):
        raise ValueError('held-out source in TRAIN collection')
    expected = {(i, c, q) for i in train_ids for c in CODECS for q in QPS}
    actual = [(r.get('source_id'), r.get('codec'), r.get('qp')) for r in records]
    if (len(actual) != len(expected) or set(actual) != expected
            or any(r.get('task') != task for r in records)):
        raise ValueError('complete TRAIN codec/QP grid is required without duplicate points')
    names = list(profile_names)
    teacher_count = 2 if task == 'ar' else 1
    for row in records:
        if [a.get('name') for a in row.get('profiles', [])] != names:
            raise ValueError('inconsistent TRAIN profile registry')
        if any(type(a.get('coded_bytes')) is not int or a['coded_bytes'] <= 0
               for a in row['profiles'] + row.get('controls', [])):
            raise ValueError('TRAIN measurements require positive actual integer bytes')
        controls = row.get('controls', [])
        if (not controls or controls[0].get('name') != 'identity'
                or row.get('anchor_coded_bytes') != controls[0]['coded_bytes']
                or any(len(a.get('distances', [])) != teacher_count or len(a.get('decisions', [])) != teacher_count
                       for a in row['profiles'] + controls)):
            raise ValueError('invalid TRAIN anchor or teacher guard count')
        if row.get('slack') != (.1 if task == 'ar' else .03) or row.get('min_savings') != .01:
            raise ValueError('TRAIN guard policy changed')
        winner = _control_winner(controls, row['slack'], row['min_savings'])
        if (row.get('controls_selected') != controls[winner]['name']
                or row.get('controls_coded_bytes') != controls[winner]['coded_bytes']):
            raise ValueError('inconsistent TRAIN control winner')
        target = choose_training_target(controls, row['profiles'], slack=row['slack'], min_savings=row['min_savings'])
        target_bytes = (controls[0]['coded_bytes'] if target == 'identity'
                        else next(a['coded_bytes'] for a in row['profiles'] if a['name'] == target))
        marginal = controls[winner]['coded_bytes'] - target_bytes if target != 'identity' else 0
        if (row.get('target_profile') != target or row.get('target_coded_bytes') != target_bytes
                or row.get('marginal_saved_bytes') != marginal):
            raise ValueError('inconsistent TRAIN target choice or bytes')
        if conditional_variant is not None:
            from .conditional_learned import canonical_target,profile_registry
            from .motion_learned import PROFILE_NAMES
            if names != [p['name'] for p in profile_registry(conditional_variant)]:
                raise ValueError('inconsistent conditional TRAIN registry')
            if row.get('target_parameters') != canonical_target(target,conditional_variant):
                raise ValueError('inconsistent canonical TRAIN target parameters')
            baseline = row.get('baseline_profiles',[])
            if ([a.get('name') for a in baseline] != list(PROFILE_NAMES)
                    or any(type(a.get('coded_bytes')) is not int or a['coded_bytes']<=0
                        or len(a.get('distances',[]))!=teacher_count
                        or len(a.get('decisions',[]))!=teacher_count for a in baseline)):
                raise ValueError('incomplete actual baseline TRAIN measurements')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _append(path, value):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def _code_manifest():
    hashes = {p.relative_to(ROOT).as_posix(): _sha(p.read_bytes().replace(b'\r\n', b'\n'))
              for p in sorted((ROOT / 'adaptive_vcm').glob('*.py'))}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                        text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return dict(commit=commit, files_sha256=hashes)


def _validate_config(cfg):
    if cfg.get('qps') != list(QPS):
        raise ValueError('complete registered codec/QP grid is required')
    if (cfg.get('schema') != 1 or cfg.get('ar_teachers') != ['r3d_18', 'mc3_18']
            or cfg.get('target_bd_rate_pct') != -10
            or cfg.get('ar_evaluators') != ['r2plus1d_18', 'r3d_18']
            or cfg.get('od_evaluator') != 'resnet50'
            or cfg.get('od_teacher') != 'mobilenet' or cfg.get('ar_kl_slack') != .1
            or cfg.get('od_distance_slack') != .03 or cfg.get('min_savings') != .01
            or cfg.get('ar_confidence') != .6 or cfg.get('od_score_threshold') != .25
            or cfg.get('ar_require_anchor_decision') is not True
            or cfg.get('ar_guard_rule') != 'anchor_relative_v2'
            or cfg.get('ar_mode') != 'motion_spatial' or cfg.get('od_mode') != 'motion_spatial'
            or cfg.get('motion_static_k') != 3 or cfg.get('motion_proposal_scales') != [.5, 1., 1.5]):
        raise ValueError('V28 requires registered frozen teachers and unchanged strict guards')
    variant = cfg.get('v29_variant')
    if variant is not None and (variant not in ('a', 'b', 'c') or cfg.get('experiment') != f'v29-{variant}'):
        raise ValueError('invalid V29 variant or experiment')
    conditional_variant = cfg.get('v30_variant')
    if conditional_variant is not None and (variant is not None or conditional_variant not in ('a','b','c')
            or cfg.get('experiment') != f'v30-{conditional_variant}'):
        raise ValueError('invalid V30 variant or experiment')


def _predict(teachers, clip, task):
    return [teacher.probabilities(clip) if task == 'ar' else teacher.predict(clip) for teacher in teachers]


def _support_hash(support):
    digest = hashlib.sha256()
    for key in ('protection', 'motion', 'cuts'):
        value = np.ascontiguousarray(support[key])
        digest.update(_json_bytes(dict(key=key, shape=list(value.shape), dtype=str(value.dtype))))
        digest.update(value.tobytes())
    digest.update(_json_bytes(support['metadata']))
    return digest.hexdigest()


def _measure_group(clip, support, protection, task, codec_name, qp, cfg, teachers,
                   source_predictions, profile_candidates, profile_names, foreground_known):
    codec = StandardCodec(codec_name, qp, cfg['preset'], cfg['fps'])
    controls = make_candidates(clip, protection, task, qp, cfg[f'{task}_candidates'])
    variant = cfg.get('v29_variant')
    renderer_variant = cfg.get('v30_variant',variant)
    profiles = (profile_candidates(clip, support, task, qp) if renderer_variant is None else
                profile_candidates(clip, support, task, qp, variant=renderer_variant))
    if [c.name for c in profiles] != list(profile_names):
        raise ValueError('profile registry changed during TRAIN collection')
    if not foreground_known:
        controls = [Candidate(c.name, clip) for c in controls]
        profiles = [Candidate(c.name, clip) for c in profiles]
    baseline_profiles = []
    if cfg.get('v30_variant') == 'c':
        from .motion_learned import profile_candidates as original_profiles
        baseline_profiles = original_profiles(clip,support,task,qp,variant='a')
        if not foreground_known:
            baseline_profiles = [Candidate(c.name,clip) for c in baseline_profiles]
    anchor = codec.roundtrip(clip)
    anchor_predictions = _predict(teachers, anchor.decoded, task)
    # Equal source pixels are deterministic codec inputs. Streams and decoded
    # teacher outputs are cached separately; same-stream guards bypass inf-inf.
    anchor_hash = _sha(anchor.data)
    pixel_cache = {(clip.shape, _sha(clip.tobytes())): anchor}
    prediction_cache = {anchor_hash: anchor_predictions}
    slack = cfg['ar_kl_slack'] if task == 'ar' else cfg['od_distance_slack']
    def measure(candidate):
        pixel_hash = _sha(candidate.clip.tobytes())
        key = (candidate.clip.shape, pixel_hash)
        if key not in pixel_cache:
            pixel_cache[key] = codec.roundtrip(candidate.clip)
        stream = pixel_cache[key]
        stream_hash = _sha(stream.data)
        identical = stream.data == anchor.data
        if identical:
            distances, decisions = (0.,) * len(teachers), (True,) * len(teachers)
        else:
            if stream_hash not in prediction_cache:
                prediction_cache[stream_hash] = _predict(teachers, stream.decoded, task)
            distances, decisions = relative_guard(task, source_predictions, anchor_predictions,
                                                  prediction_cache[stream_hash], cfg)
        action = dict(name=candidate.name, coded_bytes=int(stream.coded_bytes),
                      distances=[float(d) if math.isfinite(d) else None for d in distances],
                      decisions=[bool(d) for d in decisions], identity_stream=identical,
                      stream_sha256=stream_hash, pixel_sha256=pixel_hash,
                      shape=list(candidate.clip.shape), codec_seconds=float(stream.seconds))
        action['eligible'] = _eligible(action, anchor.coded_bytes, slack, cfg['min_savings'], len(teachers))
        return action
    measured_controls, measured_profiles = [measure(c) for c in controls], [measure(c) for c in profiles]
    measured_baseline = [measure(c) for c in baseline_profiles]
    winner = _control_winner(measured_controls, slack, cfg['min_savings'])
    target = choose_training_target(measured_controls, measured_profiles, slack=slack, min_savings=cfg['min_savings'])
    target_pixels = clip if target == 'identity' else next(c.clip for c in profiles if c.name == target)
    target_measurement = measured_controls[0] if target == 'identity' else next(a for a in measured_profiles if a['name'] == target)
    result = dict(controls=measured_controls, profiles=measured_profiles,
                anchor_coded_bytes=int(anchor.coded_bytes), anchor_decoded_sha256=_sha(anchor.decoded.tobytes()),
                actual_anchor_bpp=reference_bpp(anchor.coded_bytes, clip.shape),
                controls_selected=measured_controls[winner]['name'],
                controls_coded_bytes=measured_controls[winner]['coded_bytes'],
                target_profile=target, target_sha256=_sha(target_pixels.tobytes()),
                target_coded_bytes=target_measurement['coded_bytes'],
                marginal_saved_bytes=(measured_controls[winner]['coded_bytes'] - target_measurement['coded_bytes']
                                      if target != 'identity' else 0),
                slack=slack, min_savings=cfg['min_savings'], foreground_known=foreground_known)
    if cfg.get('v30_variant'):
        from .conditional_learned import canonical_target
        from .motion_learned import PROFILE_NAMES as baseline_names
        result.update(target_parameters=canonical_target(target,cfg['v30_variant']),
            baseline_profiles=measured_baseline if baseline_profiles else
                [a for a in measured_profiles if a['name'] in baseline_names],
            distinct_codec_encodes=len(pixel_cache),distinct_decoded_teacher_sets=len(prediction_cache),
            distinct_teacher_evaluations=len(prediction_cache)*len(teachers),
            candidate_measurement_slots=len(controls)+len(profiles)+len(baseline_profiles),
            actual_probe_codec_seconds=sum(stream.seconds for stream in pixel_cache.values()))
    return result,target_pixels


def _fit(args, records, device):
    from .motion_learned import MotionAwarePreprocessor
    # Teacher construction and collection must not change initialization.
    torch.manual_seed(args.seed)
    variant = getattr(args, 'variant', None)
    conditional_variant = getattr(args,'conditional_variant',None)
    if conditional_variant is not None:
        from .conditional_learned import ConditionalPreprocessor
        model = ConditionalPreprocessor(args.width,args.task,variant=conditional_variant)
    else:
        model = (MotionAwarePreprocessor(args.width,args.task) if variant is None else
                 MotionAwarePreprocessor(args.width,args.task,variant=variant))
    model = model.to(device).train()
    conditioned = variant is not None or conditional_variant is not None
    group_weights = semantic_group_weights(records) if conditioned else None
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(args.seed)
    steps, gradient_total, edited_total, epoch_logs = 0, 0., 0, []
    for epoch in range(args.epochs):
        epoch_loss = epoch_gradient = 0.
        epoch_edits = epoch_target_edits = epoch_pixels = epoch_positive = 0
        class_diagnostics = {'positive': [], 'identity': []}
        for index in rng.permutation(len(records)):
            row = records[int(index)]
            # One example on device at once; uint8 corpus remains on disk.
            with np.load(args.out / row['source_cache'], allow_pickle=False) as cache:
                source_pixels = cache['source']
                source = torch.from_numpy(source_pixels.copy()).to(device).float().permute(3, 0, 1, 2)[None] / 255
                protection = torch.from_numpy(cache['protection'].copy()).to(source)[None, None]
                motion = torch.from_numpy(cache['motion'].copy()).to(source)[None, None]
                cuts = torch.from_numpy(cache['cuts'].copy()).to(device)[None]
            if row['target_cache'] is None:
                target = source
            else:
                with np.load(args.out / row['target_cache'], allow_pickle=False) as cache:
                    target = torch.from_numpy(cache['target'].copy()).to(source).permute(3, 0, 1, 2)[None] / 255
            semantic_parts = None
            if not conditioned:
                output = model(source, source.new_tensor([row['qp']]),
                               source.new_tensor([int(row['codec'] == 'h265')]), protection,
                               motion=motion, cuts=cuts)
                weight = 2. if row['qp'] >= 40 else 1.
                loss = weight * F.mse_loss(output, target)
            else:
                output, aux = model(source, source.new_tensor([row['qp']]),
                                    source.new_tensor([int(row['codec'] == 'h265')]), protection,
                                    motion=motion, cuts=cuts, return_aux=True)
                weight = 1.
                if conditional_variant is not None:
                    loss,semantic_parts = conditional_loss(output,target,aux,protection,row,
                        group_weights[int(index)],conditional_variant)
                else:
                    loss, semantic_parts = semantic_loss(output,target,aux,protection,row,
                        group_weights[int(index)])
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite RGB imitation objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            head_gradients = None
            if conditional_variant is not None:
                head_gradients = dict(alpha_gradient_norm=float(model.strength_head.weight.grad.norm()),
                    expert_gradient_norm=float(model.expert_head.weight.grad.norm()),
                    gate_gradient_norm=float(model.gate_head.weight.grad.norm()))
            elif variant is not None:
                head = model.head.weight.grad
                head_gradients = dict(alpha_gradient_norm=float(head[:1].norm()),
                                      expert_gradient_norm=float(head[1:].norm()))
            optimizer.step()
            with torch.no_grad():
                edited = int((output.mul(255).round() != source.mul(255).round()).any(1).sum())
                target_edits = int((target.mul(255).round() != source.mul(255).round()).any(1).sum())
            pixels = source.numel() // 3
            steps += 1
            gradient_total += float(norm)
            edited_total += edited
            epoch_loss += float(loss.detach())
            epoch_gradient += float(norm)
            epoch_edits += edited
            epoch_target_edits += target_edits
            epoch_pixels += pixels
            epoch_positive += row['target_profile'] != 'identity'
            log = dict(step=steps, epoch=epoch + 1, source_id=row['source_id'], codec=row['codec'], qp=row['qp'],
                       loss=float(loss.detach()), qp_weight=weight, gradient_norm=float(norm),
                       nonzero_edit_pixels=edited, output_edit_fraction=edited / pixels,
                       target_edit_fraction=target_edits / pixels, target_profile=row['target_profile'])
            if semantic_parts is not None:
                log.update(group_weight=group_weights[int(index)],
                           **{key: float(value.detach()) for key, value in semantic_parts.items()},
                           **head_gradients)
                kind = 'positive' if row['target_profile'] != 'identity' else 'identity'
                diagnostic_keys = ['loss','rgb_loss','alpha_loss','expert_loss','alpha_mean','raw_alpha_mean',
                    'expert_entropy','expert_margin','gradient_norm','alpha_gradient_norm','expert_gradient_norm']
                if conditional_variant is not None:
                    diagnostic_keys += ['gate_loss','gate_probability','gate_gradient_norm','expert_kl']
                class_diagnostics[kind].append({k:log[k] for k in diagnostic_keys})
            _append(args.out / 'train.jsonl', log)
        summary = dict(epoch=epoch + 1, epochs=args.epochs, steps=steps,
                       mean_loss=epoch_loss / len(records), mean_gradient_norm=epoch_gradient / len(records),
                       output_edit_fraction=epoch_edits / epoch_pixels, target_edit_fraction=epoch_target_edits / epoch_pixels,
                       positive_targets=epoch_positive, identity_targets=len(records) - epoch_positive)
        if conditioned:
            for kind, values in class_diagnostics.items():
                summary[f'{kind}_loss'] = dict(points=len(values),
                    **({k: sum(v[k] for v in values) / len(values) for k in values[0]} if values else {}))
        epoch_logs.append(summary)
        _append(args.out / 'epochs.jsonl', summary)
        print(json.dumps(summary), flush=True)
    return model, dict(steps=steps, gradient_norm_sum=gradient_total,
                       mean_gradient_norm=gradient_total / steps, nonzero_edit_pixels_sum=edited_total,
                       epoch_logs=epoch_logs)


def _prediction_digest(predictions):
    return _sha(json.dumps(predictions, sort_keys=True, allow_nan=False,
                           default=lambda value: value.tolist()).encode())


def _calibrate(args, records, model, cfg, teachers, calibration_ids):
    """Measure the final model on disjoint TRAIN sources, with decoded teachers."""
    from .motion_selection import neural_candidates, PROPOSALS
    rows = [r for r in records if r['source_id'] in set(calibration_ids)]
    model.eval()
    measured, previous_cache = [], None
    for row in rows:
        if row['source_cache'] != previous_cache:
            cache_path = args.out / row['source_cache']
            if _sha(cache_path.read_bytes()) != row['source_cache_sha256']:
                raise ValueError('calibration source cache changed')
            with np.load(cache_path, allow_pickle=False) as cache:
                clip = cache['source'].copy()
                support = {key: cache[key].copy() for key in ('protection', 'motion', 'cuts')}
                support['metadata'] = json.loads(str(cache['metadata']))
            if _sha(clip.tobytes()) != row['source_sha256']:
                raise ValueError('calibration source pixels changed')
            source_predictions = _predict(teachers, clip, args.task)
            previous_cache = row['source_cache']
        codec = StandardCodec(row['codec'], row['qp'], cfg['preset'], cfg['fps'])
        anchor = codec.roundtrip(clip)
        if (anchor.coded_bytes != row['anchor_coded_bytes'] or
                _sha(anchor.data) != row['controls'][0]['stream_sha256']):
            raise ValueError('calibration anchor differs from original TRAIN control')
        anchor_predictions = _predict(teachers, anchor.decoded, args.task)
        prediction_cache = {_sha(anchor.data): anchor_predictions}
        candidates = neural_candidates(clip, support, codec, model)
        if [c.name for c in candidates] != [name for name, _ in PROPOSALS]:
            raise ValueError('calibration requires exactly three registered neural proposals')
        proposals = []
        for candidate in candidates:
            stream = codec.roundtrip(candidate.clip)
            stream_hash = _sha(stream.data)
            same = stream.data == anchor.data
            if stream_hash not in prediction_cache:
                prediction_cache[stream_hash] = _predict(teachers, stream.decoded, args.task)
            predictions = prediction_cache[stream_hash]
            distances, decisions = (((0.,) * len(teachers), (True,) * len(teachers)) if same else
                relative_guard(args.task, source_predictions, anchor_predictions, predictions, cfg))
            proposals.append(dict(name=candidate.name, coded_bytes=int(stream.coded_bytes),
                distances=[float(d) if math.isfinite(d) else None for d in distances],
                decisions=[bool(d) for d in decisions], identity_stream=same,
                pixel_sha256=_sha(candidate.clip.tobytes()), stream_sha256=stream_hash,
                decoded_sha256=_sha(stream.decoded.tobytes()), predictions_sha256=_prediction_digest(predictions),
                shape=list(candidate.clip.shape), codec_seconds=float(stream.seconds)))
        result = dict(task=args.task, source_id=row['source_id'], codec=row['codec'], qp=row['qp'],
            source_sha256=row['source_sha256'], source_cache=row['source_cache'],
            source_cache_sha256=row['source_cache_sha256'], support_sha256=row['support_sha256'],
            anchor_coded_bytes=int(anchor.coded_bytes), anchor_stream_sha256=_sha(anchor.data),
            anchor_decoded_sha256=_sha(anchor.decoded.tobytes()),
            source_predictions_sha256=_prediction_digest(source_predictions),
            anchor_predictions_sha256=_prediction_digest(anchor_predictions),
            controls=row['controls'], controls_selected=row['controls_selected'],
            controls_coded_bytes=row['controls_coded_bytes'], proposals=proposals)
        measured.append(result)
        _append(args.out / 'calibration_measurements.jsonl', result)
    policy = fit_admission_policy(measured, args.task)
    return dict(calibration_ids=list(calibration_ids), calibration_ids_sha256=fingerprint(calibration_ids),
                calibration_measurements=len(measured),
                calibration_measurements_sha256=_sha((args.out / 'calibration_measurements.jsonl').read_bytes()),
                admission_policy=policy,
                admission_policy_sha256=_sha(_json_bytes(policy)),
                calibration_rule='TRAIN-only p75 max teacher-relative distance among nonpositive guarded extra-byte actions')


def train(args):
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    _validate_config(cfg)
    if (args.task not in ('ar', 'od') or any(type(v) is not int or v < 1 for v in (args.count, args.epochs, args.width))
            or args.width < 4):
        raise ValueError('invalid TRAIN task or fixed-epoch budget')
    if args.task == 'od' and getattr(args, 'annotations', None) is None:
        raise ValueError('COCO annotations are required for locating TRAIN images')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('training output must be empty')
    plan, _ = (ar_plan(args.root, 'train', args.count) if args.task == 'ar'
               else od_plan(args.root, args.annotations, 'train', args.count))
    train_ids = [r['id'] for r in plan]
    if len(plan) != args.count or any(partition(f'coco2017/{i}' if args.task == 'od' else i) != 'train' for i in train_ids):
        raise ValueError('incomplete or held-out source in TRAIN collection')
    fingerprint(train_ids)
    variant = cfg.get('v29_variant')
    args.variant = variant
    conditional_variant = cfg.get('v30_variant')
    args.conditional_variant = conditional_variant
    fit_ids, calibration_ids = (calibration_split(train_ids, args.seed) if variant == 'c'
                                else (list(train_ids), []))
    schedule = dense_schedule(plan, cfg['qps'], args.seed)
    from .motion_learned import PROFILE_NAMES, profile_candidates
    baseline_names = PROFILE_NAMES
    registry = None
    if conditional_variant is not None:
        from .conditional_learned import profile_registry,profile_candidates
        registry = profile_registry(conditional_variant)
        PROFILE_NAMES = tuple(p['name'] for p in registry)
    from .motion_support import build_motion_support
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'cache/sources').mkdir(parents=True)
    (args.out / 'cache/targets').mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teachers = ([ActionAnalyzer(name, device) for name in cfg['ar_teachers']] if args.task == 'ar'
                else [DetectionAnalyzer(cfg['od_teacher'], device)])
    code = _code_manifest()
    config_bytes=args.config.read_bytes()
    config_sha256 = _sha(config_bytes.replace(b'\r\n',b'\n') if conditional_variant is not None else config_bytes)
    code_sha256 = _sha(_json_bytes(code))
    manifest = dict(schema='adaptive-vcm-training-v7', task=args.task, seed=args.seed,
                    train_count=len(plan), train_ids=train_ids, train_ids_sha256=fingerprint(train_ids),
                    config=cfg, config_sha256=config_sha256, code=code, code_sha256=code_sha256,
                    profile_names=list(PROFILE_NAMES), epochs=args.epochs, width=args.width, device=device,
                    measurements=len(schedule), collection='complete TRAIN source x H.264/H.265 x QP30/35/40/45/50',
                    optimizer='Adam', learning_rate=LEARNING_RATE, high_qp_weight=2.,
                    source='fresh spatial renderer; fixed epochs; final-LAST; no DEV/TEST selection',
                    objective='RGB MSE imitation of actual-codec teacher-feasible extra savings beyond controls; otherwise source identity',
                    limitations='TRAIN teacher-feasible profile targets do not guarantee learned inference feasibility or held-out task quality.')
    if variant is not None:
        manifest.update(schema='adaptive-vcm-training-v8', variant=variant,
            fit_ids=fit_ids, fit_ids_sha256=fingerprint(fit_ids), calibration_ids=calibration_ids,
            fit_count=len(fit_ids), calibration_count=len(calibration_ids),
            objective='editable-normalized RGB imitation + measured expert/strength/gate distillation; balanced codec/QP/target classes; capped actual marginal-byte utility',
            objective_weights=dict(rgb=1., alpha=.05, expert=.01),
            utility_rule='1 + min(1, 10 * actual_marginal_saved_bytes / original_control_bytes)',
            high_qp_weight=1., source='fresh semantic renderer; fixed epochs; final-LAST; no DEV/GT selection')
        if calibration_ids:
            manifest['calibration_ids_sha256'] = fingerprint(calibration_ids)
            manifest['fit_calibration_caveat'] = 'C fits only the non-calibration TRAIN sources; A/B fit all collected TRAIN sources.'
            print(json.dumps(dict(warning=manifest['fit_calibration_caveat'],
                                  fit_count=len(fit_ids), calibration_count=len(calibration_ids))), flush=True)
    if conditional_variant is not None:
        manifest.update(schema='adaptive-vcm-training-v9',variant=conditional_variant,
            profile_registry=registry,fit_ids=fit_ids,fit_ids_sha256=fingerprint(fit_ids),
            calibration_ids=[],fit_count=len(fit_ids),calibration_count=0,
            objective='conditional admission + positive-only measured strength/expert + editable RGB distillation',
            objective_weights=dict(rgb=1.,alpha=.05,expert=.01,gate=.05),
            gate_threshold=.5,utility_rule='1 + min(1, 10 * actual_marginal_saved_bytes / original_control_bytes)',
            high_qp_weight=1.,source='fresh conditional renderer; fixed epochs; final-LAST; no DEV/GT selection',
            baseline_profile_names=list(baseline_names))
        manifest['config_digest_rule']='SHA256 UTF8 file bytes with CRLF normalized to LF'
    _write_json(args.out / 'training_manifest.json', manifest)
    records, previous_source = [], None
    for measurement, (source_index, codec_name, qp) in enumerate(schedule, 1):
        item = plan[source_index]
        if source_index != previous_source:
            clip = (read_video(item['path'], cfg['frames'], cfg['ar_size'], cfg['temporal_stride']) if args.task == 'ar'
                    else read_image(item, cfg['od_size'])[0])
            source_predictions = _predict(teachers, clip, args.task)
            foreground_known = args.task == 'ar' or bool(np.any(source_predictions[0]['scores'] >= cfg['od_score_threshold']))
            if args.task == 'ar':
                semantic = np.maximum.reduce([normalize_map(t.saliency(clip)) for t in teachers])
                control_protection = action_protection(clip, semantic)
                learned_protection = semantic_protection(semantic)
            else:
                prediction = source_predictions[0]
                control_protection = boxes_to_mask(*clip.shape[1:3], prediction['boxes'][prediction['scores'] >= cfg['od_score_threshold']])
                learned_protection = control_protection if foreground_known else np.ones(clip.shape[1:3], np.float32)
            support = build_motion_support(clip, learned_protection, args.task)
            source_hash, support_hash = _sha(clip.tobytes()), _support_hash(support)
            source_cache = f'cache/sources/{source_index:05d}.npz'
            np.savez_compressed(args.out / source_cache, source=clip, protection=support['protection'],
                                motion=support['motion'], cuts=support['cuts'],
                                metadata=np.asarray(json.dumps(support['metadata'], sort_keys=True, allow_nan=False)))
            source_cache_hash = _sha((args.out / source_cache).read_bytes())
            previous_source = source_index
        measured, target = _measure_group(clip, support, control_protection, args.task, codec_name, qp,
                                         cfg, teachers, source_predictions, profile_candidates, PROFILE_NAMES, foreground_known)
        target_cache = None
        if measured['target_profile'] != 'identity':
            target_cache = f'cache/targets/{measurement:06d}.npz'
            np.savez_compressed(args.out / target_cache, target=target)
        row = dict(measurement=measurement, task=args.task, source_id=item['id'], codec=codec_name, qp=qp,
                   source_sha256=source_hash, source_shape=list(clip.shape), source_pixels=int(np.prod(clip.shape[:3])),
                   source_cache=source_cache, source_cache_sha256=source_cache_hash, support_sha256=support_hash,
                   config_sha256=config_sha256, code_sha256=code_sha256, target_cache=target_cache,
                   target_cache_sha256=_sha((args.out / target_cache).read_bytes()) if target_cache else None,
                   **measured)
        records.append(row)
        _append(args.out / 'measurements.jsonl', row)
        if measurement == 1 or measurement % 25 == 0 or measurement == len(schedule):
            print(json.dumps({k: row[k] for k in ('measurement', 'source_id', 'codec', 'qp', 'target_profile', 'marginal_saved_bytes')}), flush=True)
    if conditional_variant is not None:
        validate_training_records(records,train_ids,args.task,PROFILE_NAMES,conditional_variant)
    else:
        validate_training_records(records,train_ids,args.task,PROFILE_NAMES)
    fit_set = set(fit_ids)
    fit_records = [r for r in records if r['source_id'] in fit_set]
    static_orders = fit_static_orders(fit_records, PROFILE_NAMES, k=3)
    baseline_static_orders = None
    if conditional_variant is not None:
        baseline_rows = [dict(r,profiles=r['baseline_profiles']) for r in fit_records]
        baseline_static_orders = fit_static_orders(baseline_rows,baseline_names,k=3)
        headroom = conditional_bank_summary(records,baseline_names)
        costs = dict(distinct_codec_encodes=sum(r['distinct_codec_encodes'] for r in records),
            distinct_teacher_evaluations=sum(r['distinct_teacher_evaluations'] for r in records),
            candidate_measurement_slots=sum(r['candidate_measurement_slots'] for r in records),
            actual_probe_codec_seconds=sum(r['actual_probe_codec_seconds'] for r in records))
        _write_json(args.out/'bank_headroom.json',dict(groups=headroom,costs=costs))
        print(json.dumps(dict(bank_headroom=headroom,actual_collection_costs=costs)),flush=True)
    measurements_hash = _sha((args.out / 'measurements.jsonl').read_bytes())
    train_source_sha256 = {row['source_id']: row['source_sha256'] for row in records}
    if variant != 'c':
        del teachers
    model, fitting = _fit(args, fit_records, device)
    calibration = _calibrate(args, records, model, cfg, teachers, calibration_ids) if variant == 'c' else {}
    if variant == 'c':
        del teachers
    positive = sum(r['target_profile'] != 'identity' for r in records)
    manifest.update(**fitting, measurements_sha256=measurements_hash, static_orders=static_orders,
                    positive_targets=positive, identity_targets=len(records) - positive,
                    marginal_saved_bytes=sum(r['marginal_saved_bytes'] for r in records),
                    train_source_sha256=train_source_sha256)
    if variant is not None:
        manifest.update(fit_positive_targets=sum(r['target_profile'] != 'identity' for r in fit_records),
                        fit_identity_targets=sum(r['target_profile'] == 'identity' for r in fit_records),
                        **calibration)
    if conditional_variant is not None:
        manifest.update(fit_positive_targets=sum(r['target_profile']!='identity' for r in fit_records),
            fit_identity_targets=sum(r['target_profile']=='identity' for r in fit_records),
            baseline_static_orders=baseline_static_orders,bank_headroom=headroom,
            actual_collection_costs=costs,model_parameters=sum(p.numel() for p in model.parameters()))
    _write_json(args.out / 'training_manifest.json', manifest)
    state = dict(schema=model.schema, task=args.task, width=args.width, model=model.state_dict(),
                    steps=fitting['steps'], epochs=args.epochs, measurements=len(records), train_count=len(plan),
                    train_ids=train_ids, train_ids_sha256=manifest['train_ids_sha256'], training_config=cfg,
                    measurements_sha256=measurements_hash, static_orders=static_orders, profile_names=list(PROFILE_NAMES),
                    seed=args.seed, config_sha256=config_sha256, code_sha256=code_sha256,
                    code=code, train_source_sha256=train_source_sha256)
    if variant is not None:
        state.update(variant=variant, fit_ids=fit_ids, fit_ids_sha256=fingerprint(fit_ids),
                     calibration_ids=calibration_ids)
        state.update(calibration)
    if conditional_variant is not None:
        state.update(variant=conditional_variant,profile_registry=registry,
            fit_ids=fit_ids,fit_ids_sha256=fingerprint(fit_ids),calibration_ids=[],
            baseline_static_orders=baseline_static_orders)
    torch.save(state, args.out / 'preprocessor_last.pth')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['ar', 'od'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/v28_screen.json')
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--width', type=int, default=12)
    parser.add_argument('--seed', type=int, default=302801)
    parser.add_argument('--out', type=Path, required=True)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
