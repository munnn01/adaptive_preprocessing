"""TRAIN-only spatial RGB imitation from complete actual H.264/H.265 grids.

Codec measurements label fixed reference profiles; Adam sees cached RGB targets
and source support only. No codec surrogate, evaluator, labels, or DEV selector
enters fitting. Identity is the target whenever a profile adds no guarded bytes.
"""
from __future__ import annotations

import argparse
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


def validate_training_records(records, train_ids, task, profile_names):
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
    profiles = profile_candidates(clip, support, task, qp)
    if [c.name for c in profiles] != list(profile_names):
        raise ValueError('profile registry changed during TRAIN collection')
    if not foreground_known:
        controls = [Candidate(c.name, clip) for c in controls]
        profiles = [Candidate(c.name, clip) for c in profiles]
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
    winner = _control_winner(measured_controls, slack, cfg['min_savings'])
    target = choose_training_target(measured_controls, measured_profiles, slack=slack, min_savings=cfg['min_savings'])
    target_pixels = clip if target == 'identity' else next(c.clip for c in profiles if c.name == target)
    target_measurement = measured_controls[0] if target == 'identity' else next(a for a in measured_profiles if a['name'] == target)
    return dict(controls=measured_controls, profiles=measured_profiles,
                anchor_coded_bytes=int(anchor.coded_bytes), anchor_decoded_sha256=_sha(anchor.decoded.tobytes()),
                actual_anchor_bpp=reference_bpp(anchor.coded_bytes, clip.shape),
                controls_selected=measured_controls[winner]['name'],
                controls_coded_bytes=measured_controls[winner]['coded_bytes'],
                target_profile=target, target_sha256=_sha(target_pixels.tobytes()),
                target_coded_bytes=target_measurement['coded_bytes'],
                marginal_saved_bytes=(measured_controls[winner]['coded_bytes'] - target_measurement['coded_bytes']
                                      if target != 'identity' else 0),
                slack=slack, min_savings=cfg['min_savings'], foreground_known=foreground_known), target_pixels


def _fit(args, records, device):
    from .motion_learned import MotionAwarePreprocessor
    # Teacher construction and collection must not change initialization.
    torch.manual_seed(args.seed)
    model = MotionAwarePreprocessor(args.width, args.task).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(args.seed)
    steps, gradient_total, edited_total, epoch_logs = 0, 0., 0, []
    for epoch in range(args.epochs):
        epoch_loss = epoch_gradient = 0.
        epoch_edits = epoch_target_edits = epoch_pixels = epoch_positive = 0
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
            output = model(source, source.new_tensor([row['qp']]),
                           source.new_tensor([int(row['codec'] == 'h265')]), protection,
                           motion=motion, cuts=cuts)
            weight = 2. if row['qp'] >= 40 else 1.
            loss = weight * F.mse_loss(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite RGB imitation objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
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
            _append(args.out / 'train.jsonl', log)
        summary = dict(epoch=epoch + 1, epochs=args.epochs, steps=steps,
                       mean_loss=epoch_loss / len(records), mean_gradient_norm=epoch_gradient / len(records),
                       output_edit_fraction=epoch_edits / epoch_pixels, target_edit_fraction=epoch_target_edits / epoch_pixels,
                       positive_targets=epoch_positive, identity_targets=len(records) - epoch_positive)
        epoch_logs.append(summary)
        _append(args.out / 'epochs.jsonl', summary)
        print(json.dumps(summary), flush=True)
    return model, dict(steps=steps, gradient_norm_sum=gradient_total,
                       mean_gradient_norm=gradient_total / steps, nonzero_edit_pixels_sum=edited_total,
                       epoch_logs=epoch_logs)


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
    schedule = dense_schedule(plan, cfg['qps'], args.seed)
    from .motion_learned import PROFILE_NAMES, profile_candidates
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
    config_sha256, code_sha256 = _sha(args.config.read_bytes()), _sha(_json_bytes(code))
    manifest = dict(schema='adaptive-vcm-training-v7', task=args.task, seed=args.seed,
                    train_count=len(plan), train_ids=train_ids, train_ids_sha256=fingerprint(train_ids),
                    config=cfg, config_sha256=config_sha256, code=code, code_sha256=code_sha256,
                    profile_names=list(PROFILE_NAMES), epochs=args.epochs, width=args.width, device=device,
                    measurements=len(schedule), collection='complete TRAIN source x H.264/H.265 x QP30/35/40/45/50',
                    optimizer='Adam', learning_rate=LEARNING_RATE, high_qp_weight=2.,
                    source='fresh spatial renderer; fixed epochs; final-LAST; no DEV/TEST selection',
                    objective='RGB MSE imitation of actual-codec teacher-feasible extra savings beyond controls; otherwise source identity',
                    limitations='TRAIN teacher-feasible profile targets do not guarantee learned inference feasibility or held-out task quality.')
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
    validate_training_records(records, train_ids, args.task, PROFILE_NAMES)
    static_orders = fit_static_orders(records, PROFILE_NAMES, k=3)
    measurements_hash = _sha((args.out / 'measurements.jsonl').read_bytes())
    train_source_sha256 = {row['source_id']: row['source_sha256'] for row in records}
    del teachers
    model, fitting = _fit(args, records, device)
    positive = sum(r['target_profile'] != 'identity' for r in records)
    manifest.update(**fitting, measurements_sha256=measurements_hash, static_orders=static_orders,
                    positive_targets=positive, identity_targets=len(records) - positive,
                    marginal_saved_bytes=sum(r['marginal_saved_bytes'] for r in records),
                    train_source_sha256=train_source_sha256)
    _write_json(args.out / 'training_manifest.json', manifest)
    torch.save(dict(schema=model.schema, task=args.task, width=args.width, model=model.state_dict(),
                    steps=fitting['steps'], epochs=args.epochs, measurements=len(records), train_count=len(plan),
                    train_ids=train_ids, train_ids_sha256=manifest['train_ids_sha256'], training_config=cfg,
                    measurements_sha256=measurements_hash, static_orders=static_orders, profile_names=list(PROFILE_NAMES),
                    seed=args.seed, config_sha256=config_sha256, code_sha256=code_sha256,
                    code=code, train_source_sha256=train_source_sha256),
               args.out / 'preprocessor_last.pth')
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
