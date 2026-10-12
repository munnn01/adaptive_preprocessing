"""Evaluate V32 on paired held-out AR/COCO sources using paid wire packets."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from ..analyzers import ActionAnalyzer, DetectionAnalyzer
from ..codec import reference_bpp
from ..data import ar_plan, od_plan, read_video, read_image, fingerprint, partition
from ..evaluate import _ar_curves, _od_curves, _gt_scaled, _od_predictions, write_json
from ..motion_support import build_motion_support
from ..preprocessing import action_protection, boxes_to_mask, normalize_map
from ..rateaware import semantic_protection
from .baseline import FrozenV30Baseline
from .codec import V32Codec
from .protocol import ROOT, load_config, provenance
from .selection import choose_stream


class SourceAudit:
    def __init__(self):
        self.hashes = {}

    def add(self, identifier, rgb):
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if identifier in self.hashes or digest in self.hashes.values():
            raise ValueError('duplicate source ID/pixel content in V32 evaluation')
        self.hashes[identifier] = digest
        return digest


def empty_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('V32 output must be empty; refusing to mix evidence')
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize(rows, ids, task, cfg, draws, meta=None, gt=None, *, has_baseline=False, extra_arms=()):
    """Refuse partial grids before fitting or bootstrapping any branch."""
    fingerprint(ids)
    arms = ['controls', 'simplification', 'retarget', 'v32']
    if has_baseline:
        arms += ['v30b', 'hybrid']
    if len(set(extra_arms)) != len(extra_arms) or set(extra_arms) & {'anchor', *arms}:
        raise ValueError('duplicate/unplanned extra arm grid')
    arms += list(extra_arms)
    expected = {(i, c, q, a) for i in ids for c in cfg['codecs'] for q in cfg['qps'] for a in ['anchor', *arms]}
    keys = [(r['id'], r['codec'], r['qp'], r['arm']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('missing/duplicate/unplanned V32 paired grid')
    results, diagnostics = {}, {}
    for codec in cfg['codecs']:
        subset = [r for r in rows if r['codec'] == codec]
        results[codec], diagnostics[codec] = {}, {}
        for arm in arms:
            paired = [r for r in subset if r['arm'] == 'anchor']
            paired += [{**r, 'arm': 'adaptive'} for r in subset if r['arm'] == arm]
            names = sorted(paired[0]['correct']) if task == 'ar' else []
            results[codec][arm] = (_ar_curves(paired, names, cfg['qps'], draws, cfg['seed']) if task == 'ar'
                else _od_curves(paired, cfg['qps'], draws, cfg['seed'], meta, gt, codec))
            diagnostics[codec][arm] = {}
            for qp in cfg['qps']:
                anchor = [r for r in paired if r['arm'] == 'anchor' and r['qp'] == qp]
                chosen = [r for r in paired if r['arm'] == 'adaptive' and r['qp'] == qp]
                diagnostics[codec][arm][str(qp)] = dict(
                    total_byte_change_pct=(sum(r['coded_bytes'] for r in chosen) / sum(r['coded_bytes'] for r in anchor)-1)*100,
                    candidate_counts=dict(Counter(r['candidate'] for r in chosen)))
    return dict(task=task, results=results, same_qp=diagnostics, complete_paired_grid=True,
                target_confirmed=False, rate_scope='wire header + grid metadata + actual elementary stream',
                selection_scope='all fixed/hybrid probes measured; exploratory encoder selection')


def prepare_source(rgb, task, cfg, teachers):
    """Only source RGB and frozen teachers enter support construction."""
    if task == 'ar':
        predictions = [t.probabilities(rgb) for t in teachers]
        semantic = np.maximum.reduce([normalize_map(t.saliency(rgb)) for t in teachers])
        protection = action_protection(rgb, semantic)
        support = build_motion_support(rgb, semantic_protection(semantic), task)
    else:
        predictions = [teachers[0].predict(rgb)]
        source = predictions[0]
        protection = boxes_to_mask(*rgb.shape[1:3], source['boxes'][source['scores'] >= cfg['od_score_threshold']])
        support = build_motion_support(rgb, protection, task)
    return protection, support, predictions


def run(args, *, extension=None, codec_factory=None):
    # Check this before expensive model initialization, including incomplete args.
    out = empty_output(args.out)
    cfg = load_config(args.config)
    if args.split not in ('dev', 'test') or args.count < 1 or args.bootstrap is not None and args.bootstrap < 0:
        raise ValueError('invalid V32 evaluation plan')
    if bool(args.checkpoint) != bool(args.v30_runtime):
        raise ValueError('pass both --checkpoint and --v30-runtime for frozen V30-B')
    draws = cfg['bootstrap_draws'] if args.bootstrap is None else args.bootstrap
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_num_threads(2)
    baseline = FrozenV30Baseline.load(args.checkpoint, args.v30_runtime, args.task, device) if args.checkpoint else None
    if baseline:
        baseline.validate_config(cfg)
    vc_models, vc_families = {}, {}
    if getattr(args, 'plain_checkpoint', None) or getattr(args, 'joint_checkpoint', None):
        if not getattr(args, 'plain_checkpoint', None) or not getattr(args, 'joint_checkpoint', None):
            raise ValueError('both V32-VC checkpoints required')
        from ..v32vc.checkpoint import load_checkpoint, validate_pair
        from ..v32vc.evaluate import EXTRA_FAMILIES
        vc_models = {arm: load_checkpoint(getattr(args, arm + '_checkpoint'), args.task, device) for arm in ('plain', 'joint')}
        validate_pair(vc_models)
        vc_families = EXTRA_FAMILIES
    plugin=extension or {}
    if plugin and vc_models: raise ValueError('only one evaluation extension allowed')
    if plugin:
        vc_models,vc_families=plugin['models'],plugin['families']
    if args.task == 'ar':
        plan, meta = ar_plan(args.root, args.split, args.count)
        teachers = [ActionAnalyzer(n, device) for n in cfg['ar_teachers']]
        evaluators = {n: next((t for t in teachers if t.name == n), None) or ActionAnalyzer(n, device) for n in cfg['ar_evaluators']}
        if plugin.get('evaluators'): evaluators.update(plugin['evaluators'](device))
    else:
        if args.annotations is None:
            raise ValueError('COCO --annotations required')
        plan, meta = od_plan(args.root, args.annotations, args.split, args.count)
        teachers = [DetectionAnalyzer(cfg['od_teacher'], device)]
        evaluators = {cfg['od_evaluator']: DetectionAnalyzer(cfg['od_evaluator'], device)}
    ids = [item['id'] for item in plan]
    fingerprint(ids)
    if any(partition(item['id'] if args.task == 'ar' else f"coco2017/{item['id']}") != args.split for item in plan):
        raise ValueError('V32 plan contains a source outside the held-out partition')
    manifest = dict(**provenance(cfg), status='running', config=cfg, task=args.task, split=args.split,
        count=len(plan), ids=ids, ids_sha256=fingerprint(ids), source_pixel_sha256={},
        device=device, python=platform.python_version(), torch=str(torch.__version__), bootstrap_draws=draws,
        scope='frozen_v30b_plus_roi' if baseline else 'fixed_roi_only',
        baseline=baseline.manifest if baseline else None,
        data_metadata=meta if args.task == 'ar' else {'annotation_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest()},
        rate_denominator='original canonical source T*H*W pixels',
        receiver='standard H.264/H.265 decode then received-grid inverse retargeting',
        selection_cost='all candidate encodes, both codec decodes, inverse maps and teacher calls included; preparation recorded separately')
    if vc_models:
        manifest.update(scope='v32vc_paired_extension', v32vc={a: m.manifest for a, m in vc_models.items()},
                        extra_arms=list(vc_families), comparison_limit='New FIT48/CAL16 versus historical V30-B TRAIN32; exploratory')
    if plugin: manifest.update(plugin['manifest'])
    write_json(out / 'manifest.json', manifest)
    rows, geometry, audit = [], {}, SourceAudit()
    for position, item in enumerate(plan):
        if args.task == 'ar':
            rgb = read_video(item['path'], cfg['frames'], cfg['ar_size'], cfg['temporal_stride'])
        else:
            rgb, geometry[item['image_id']] = read_image(item, cfg['od_size'])
        source_hash = audit.add(item['id'], rgb)
        if baseline:
            baseline.assert_source(item['id'], rgb)
        for model in vc_models.values():
            model.assert_source(item['id'], rgb)
        start = time.perf_counter()
        protection, support, source_predictions = prepare_source(rgb, args.task, cfg, teachers)
        preparation_seconds = time.perf_counter() - start
        for codec in cfg['codecs']:
            for qp in cfg['qps']:
                actual_codec = (codec_factory or V32Codec)(codec, qp, cfg['preset'], Fraction(cfg['fps']))
                options, extension_seconds = {}, 0.
                if vc_models:
                    from ..v32vc.evaluate import prepare_extension
                    options, extension_seconds = plugin.get('prepare',prepare_extension)(rgb, support, actual_codec, vc_models, cfg)
                bundle = choose_stream(rgb, protection, support, args.task,
                    actual_codec, cfg, teachers, source_predictions, baseline, **options)
                bundle.seconds += extension_seconds
                extension_record = dict(extension_preparation_seconds=extension_seconds) if vc_models else {}
                prediction_cache = {}
                for arm, (packet, name) in {'anchor': (bundle.anchor, 'identity'), **bundle.arms}.items():
                    reused_prediction = packet.packet_sha256 in prediction_cache
                    if packet.packet_sha256 not in prediction_cache:
                        prediction_cache[packet.packet_sha256] = plugin['predict'](packet, item, evaluators, arm, codec, qp) if plugin.get('predict') else (
                            dict(correct={n: int(e.probabilities(packet.decoded).argmax() == item['label']) for n, e in evaluators.items()})
                            if args.task == 'ar' else dict(predictions=_od_predictions(evaluators[cfg['od_evaluator']].predict(packet.decoded), item['image_id'])))
                    receiver_cost = {}
                    if plugin.get('predict'):
                        received = prediction_cache[packet.packet_sha256]['shared_receiver']
                        receiver_cost = dict(prediction_cache_reused=reused_prediction,
                            charged_receiver_seconds=0. if reused_prediction else received['total_prediction_seconds'],
                            charged_head_seconds={} if reused_prediction else received['latency_seconds'])
                    packet_path = Path('packets') / codec / f'{position:05d}-q{qp}-{arm}.v32'
                    (out / packet_path).parent.mkdir(parents=True, exist_ok=True)
                    (out / packet_path).write_bytes(packet.data)
                    row = dict(id=item['id'], codec=codec, qp=qp, arm=arm, candidate=name,
                        source_sha256=source_hash, source_shape=list(rgb.shape), coded_shape=list(packet.coded_shape),
                        elementary_bytes=packet.elementary_bytes, metadata_bytes=packet.metadata_bytes,
                        coded_bytes=packet.coded_bytes, bpp=reference_bpp(packet.coded_bytes, rgb.shape),
                        elementary_bpp=reference_bpp(packet.elementary_bytes, rgb.shape),
                        packet_sha256=packet.packet_sha256, packet_path=packet_path.as_posix(),
                        source_preparation_seconds=preparation_seconds, selection_seconds=bundle.seconds,
                        **prediction_cache[packet.packet_sha256], **extension_record, **receiver_cost)
                    rows.append(row)
                    with (out / 'rows.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(row, allow_nan=False) + '\n')
                record = dict(id=item['id'], codec=codec, qp=qp, selected={a: name for a, (_, name) in bundle.arms.items()},
                    candidates=bundle.audit, source_support=support['metadata'], selection_seconds=bundle.seconds,
                    probe_cost=dict(encodes=sum(r['distinct_encode'] for r in bundle.audit),
                        packet_decodes=sum(r['distinct_packet_decode'] for r in bundle.audit),
                        teacher_calls=sum(r['teacher_calls'] for r in bundle.audit)), **extension_record)
                with (out / 'selection_audit.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
        manifest['source_pixel_sha256'] = audit.hashes.copy()
        write_json(out / 'manifest.json', manifest)
        print(f'[{args.task}] source {position+1}/{len(plan)} complete', flush=True)
    gt = _gt_scaled(meta, plan, geometry) if args.task == 'od' else None
    if gt is not None:
        write_json(out / 'ground_truth_scaled.json', {'gt': gt, 'categories': meta['categories']})
    summary = summarize(rows, ids, args.task, cfg, draws, meta, gt, has_baseline=baseline is not None,
                        extra_arms=tuple(vc_families))
    summary['scope'] = manifest['scope']
    write_json(out / 'summary.json', summary)
    manifest['status'] = 'complete'
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['ar', 'od'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/v32_roi.json')
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--bootstrap', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--v30-runtime', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
