"""Frozen V34-R1 validation on official K400 val, independent of controller history."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from ..data import canon
from .protocol import digest


def validation_annotations(rows, train_ids, categories):
    mapping = {canon(name): i for i, name in enumerate(categories)}
    result = {}
    for row in rows:
        yt = row['youtube_id']
        if yt in train_ids:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', yt) or canon(row['label']) not in mapping:
            raise ValueError('invalid official validation annotation')
        value = dict(label=mapping[canon(row['label'])], time_start=int(row['time_start']), time_end=int(row['time_end']))
        if value['time_end'] <= value['time_start'] or yt in result:
            raise ValueError('ambiguous validation YouTube ID')
        result[yt] = value
    return result


def select_records(records, annotations, exclusions, *, count, seed, shards, shard):
    if count < 1 or shards < 1 or count % shards or not 0 <= shard < shards:
        raise ValueError('invalid independent sample/shard')
    forbidden = set(exclusions.get('youtube_ids', []))
    forbidden.update(Path(i).name[:11] for i in exclusions.get('ids', []))
    available = {}
    for row in records:
        yt = row['youtube_id']
        if yt not in annotations or yt in forbidden:
            continue
        if yt in available:
            raise ValueError('duplicate available YouTube ID')
        available[yt] = {**row, **annotations[yt]}
    if len(available) < count:
        raise ValueError(f'insufficient independent validation sources: {len(available)} < {count}')
    ranked = sorted(available, key=lambda yt: hashlib.sha256(f'v34-independent:{seed}:{yt}'.encode()).hexdigest())
    chosen = ranked[:count]
    audit = dict(global_ids=chosen, global_ids_sha256=digest(chosen), eligible_count=len(available),
                 available_ids_sha256=digest(sorted(available)), count=count, seed=seed, shards=shards, shard=shard,
                 selection='uniform hash rank of available official-val YouTube IDs; no analyzer/accuracy filtering')
    return [available[yt] for yt in chosen[shard::shards]], audit


def frame_inventory(root, annotations, *, minimum_frames):
    """Data-only eligibility before hash selection; never inspect task accuracy."""
    if minimum_frames < 1:
        raise ValueError('positive minimum source frame count required')
    records, counts = [], {}
    for directory, dirs, _ in os.walk(root):
        keep = []
        for name in dirs:
            if name in annotations:
                if name in counts:
                    raise ValueError('duplicate available YouTube ID')
                path = Path(directory)/name
                count = sum(1 for p in path.glob('img_*.jpg') if p.is_file())
                counts[name] = count
                if count >= minimum_frames:
                    records.append(dict(youtube_id=name, path=str(path), source_frame_count=count))
                if len(counts) % 1000 == 0:
                    print('V34 source frame inventory', len(counts), 'eligible', len(records), flush=True)
            elif not re.fullmatch(r'[A-Za-z0-9_-]{11}', name):
                keep.append(name)
        dirs[:] = keep
    return records, dict(minimum_frames=minimum_frames, official_available=len(counts), eligible_sources=len(records),
        frame_counts=dict(sorted(counts.items())), frame_counts_sha256=digest(counts),
        short_sources={k:v for k,v in sorted(counts.items()) if v < minimum_frames},
        eligibility='count provider img_*.jpg files before hash ranking; no task inference, padding or selected-video replacement')


def read_frames(directory, *, frames, stride, size):
    import cv2
    from PIL import Image
    files = sorted(Path(directory).glob('img_*.jpg'))
    if len(files) < frames * stride:
        raise ValueError(f'short frame source; no padding/replacement allowed: {directory}; found {len(files)}, required {frames*stride}')
    start = max(0, len(files) - frames * stride) // 2
    indices = [start + i * stride for i in range(frames)]
    picked, hashes = [], []
    for i in indices:
        with Image.open(files[i]) as im:
            image = np.asarray(im.convert('RGB'))
        picked.append(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA))
        hashes.append(dict(name=files[i].name, sha256=hashlib.sha256(files[i].read_bytes()).hexdigest()))
    return np.stack(picked), dict(kind='provider-predecoded-RGB-center-stride', reported_source_frames=len(files),
        start=start, stride=stride, frame_indices=indices, decoded_unique_frames=frames, repeated_frames=0,
        selected_frame_files=hashes, original_video_cadence='not verified; provider dataset title F10 is not treated as ground-truth FPS')


def audit_pixels(rgb, exclusions, seen):
    sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    if sha in set(exclusions.get('pixel_sha256', [])):
        raise ValueError('historical pixels overlap')
    if sha in seen:
        raise ValueError('duplicate independent pixels')
    seen.add(sha)
    return sha


def paired_arrays(rows, codec, arm, name, qps):
    subset = [r for r in rows if r['codec'] == codec and r['arm'] in ('anchor', arm)]
    keyed = {(r['id'], r['qp'], r['arm']): r for r in subset}
    if len(keyed) != len(subset):
        raise ValueError('duplicate paired grid rows')
    ids = sorted({r['id'] for r in subset if r['arm'] == 'anchor'})
    expected = {(i, q, a) for i in ids for q in qps for a in ('anchor', arm)}
    if not ids or set(keyed) != expected:
        raise ValueError('incomplete or mismatched paired grid')
    arrays = []
    for a in ('anchor', arm):
        rate = np.array([[keyed[i, q, a]['bpp'] for q in qps] for i in ids], float)
        quality = np.array([[keyed[i, q, a]['correct'][name] for q in qps] for i in ids], float)
        if not np.isfinite(rate).all() or (rate <= 0).any() or not np.isin(quality, [0, 1]).all():
            raise ValueError('invalid paired observations')
        arrays.append((rate, quality))
    return ids, arrays


def run(args):
    protocol = json.loads(Path(args.protocol).read_text())
    for name, path, key in [('annotation', args.annotations, 'annotation_csv_sha256'),
                            ('exclusion', args.exclusions, 'exclusions_sha256')]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(name + ' manifest bytes drift')
    import torch
    from torchvision.models.video import Swin3D_T_Weights
    from ..v33ap.sources import SourceRecord
    from .checkpoint import load_checkpoint
    from .tasks import Heads
    from .evaluation import evaluate
    from .runner import write_json
    from ..v32vc.checkpoint import state_hash

    cfg, design = protocol['settings'], protocol['design']
    exclusions = json.loads(Path(args.exclusions).read_text())
    annotation_rows = list(csv.DictReader(Path(args.annotations).open(encoding='utf-8')))
    categories = Swin3D_T_Weights.KINETICS400_V1.meta['categories']
    if digest(categories) != protocol['categories_sha256']:
        raise ValueError('backbone label order drift')
    annotations = {r['youtube_id']: dict(label=int(r['label']), time_start=int(r['time_start']), time_end=int(r['time_end']))
                   for r in annotation_rows}
    if len(annotations) != len(annotation_rows):
        raise ValueError('duplicate pinned official-val annotation')
    minimum_frames = cfg['frames'] * cfg['temporal_stride']
    if design.get('minimum_source_frames') != minimum_frames:
        raise ValueError('minimum source frames must be prospectively declared in repaired protocol')
    records, inventory = frame_inventory(args.root, annotations, minimum_frames=minimum_frames)
    selected, audit = select_records(records, annotations, exclusions, count=design['count'], seed=design['selection_seed'],
                                      shards=design['shards'], shard=args.shard)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('immutable empty holdout output required')
    out.mkdir(parents=True)
    write_json(out/'protocol.json', protocol)
    write_json(out/'selection.json', audit)
    write_json(out/'source_inventory.json', inventory)
    print('V34 independent eligibility', json.dumps(dict(available=inventory['official_available'],
        eligible=inventory['eligible_sources'], excluded_short=len(inventory['short_sources']),
        minimum_frames=minimum_frames, shard=args.shard, selected=len(selected))), flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    models = []
    for seed in cfg['seeds']:
        path = Path(args.checkpoints)/f'controller_{seed}.pth'
        model = load_checkpoint(path, device).requires_grad_(False)
        expected = protocol['checkpoints'][str(seed)]
        if (model.checkpoint_sha256 != expected['sha256'] or model.metadata['settings'] != cfg
            or model.metadata['source_commit'] != protocol['controller_source_commit']
            or model.metadata['seed'] != seed):
            raise ValueError('frozen checkpoint identity/recipe drift')
        models.append(model)
    before = [state_hash(m.state_dict()) for m in models]
    if before != [protocol['checkpoints'][str(s)]['state_sha256'] for s in cfg['seeds']]:
        raise ValueError('frozen controller state drift')
    heads = Heads(device, 'ar_ot', cfg)
    if any(heads.hashes() != m.metadata['task_models'] for m in models):
        raise ValueError('frozen task model/transform drift')
    groups = {'dev': [], 'test': []}
    cohorts, seen = {}, set()
    for item in selected:
        rgb, recipe = read_frames(item['path'], frames=cfg['frames'], stride=cfg['temporal_stride'], size=cfg['ar_size'])
        sha = audit_pixels(rgb, exclusions, seen)
        identifier = 'ar/official-k400-val/' + item['youtube_id']
        record = SourceRecord(identifier, 'ar', item.copy(), rgb, None, recipe)
        record.split = 'test'
        groups['test'].append(record)
        cohorts[identifier] = dict(youtube_id=item['youtube_id'], split='independent_test', pixel_sha256=sha, frame_recipe=recipe)
    write_json(out/'cohorts.json', cohorts)
    print('V34 independent frames validated', len(cohorts), 'frozen evaluation starting', flush=True)
    evaluate(groups, heads, models, cfg, {'ar': {}}, out, None)
    after = [state_hash(m.state_dict()) for m in models]
    if before != after:
        raise ValueError('controller parameter update during independent evaluation')
    summary = json.loads((out/'test_summary.json').read_text())
    summary.update(scope='independent official K400 val; frozen original AR+OT controller; predecoded frame provider; shard results only',
                   target_confirmed=False, independent_design=design, shard=args.shard,
                   uncertainty_limit='global paired-video BD bootstrap only after all four disjoint shards merge')
    write_json(out/'test_summary.json', summary)
    write_json(out/'complete.json', dict(status='complete', shard=args.shard, sources=len(selected),
        global_ids_sha256=audit['global_ids_sha256'], checkpoint_sha256=[m.checkpoint_sha256 for m in models],
        parameter_updates=False, training_performed=False, controller_state_before=before, controller_state_after=after))


def paired_bootstrap(arrays, *, draws, seed, alpha):
    from ..v33ap.evaluate import safe_curve
    (ar, aq), (br, bq) = arrays
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        # The same source index vector is shared across arms and all QPs.
        sample = rng.integers(0, len(ar), len(ar))
        value = safe_curve(dict(bpp=ar[sample].mean(0), quality=aq[sample].mean(0)),
                           dict(bpp=br[sample].mean(0), quality=bq[sample].mean(0))).get('bd_rate_pct')
        boot.append(float(value) if value is not None and np.isfinite(value) else np.nan)
    finite = np.array(boot); finite = finite[np.isfinite(finite)]
    return dict(lo=float(np.quantile(finite, alpha)) if len(finite) else None,
                hi=float(np.quantile(finite, 1-alpha)) if len(finite) else None,
                confidence=1-2*alpha, draws=draws, finite_draws=len(finite), unit='source video',
                familywise='Bonferroni over two codecs for primary Swin seed' if alpha == .0125 else 'secondary descriptive 95%')


def validate_shards(paths, protocol):
    plans = [json.loads((Path(p)/'selection.json').read_text()) for p in paths]
    design, cfg = protocol['design'], protocol['settings']
    if len(plans) != design['shards'] or {p['shard'] for p in plans} != set(range(design['shards'])):
        raise ValueError('all independent shards required')
    ids = plans[0]['global_ids']
    if len(ids) != design['count'] or len(set(ids)) != len(ids):
        raise ValueError('invalid global independent membership')
    if any(p['global_ids'] != ids or p['global_ids_sha256'] != digest(ids)
           or p['available_ids_sha256'] != plans[0]['available_ids_sha256']
           or p['count'] != design['count'] or p['shards'] != design['shards']
           or p['seed'] != design['selection_seed'] for p in plans):
        raise ValueError('independent candidate/sample drift across shards')
    rows, source_ids, pixels = [], [], []
    expected_hashes = [protocol['checkpoints'][str(s)]['sha256'] for s in cfg['seeds']]
    expected_states = [protocol['checkpoints'][str(s)]['state_sha256'] for s in cfg['seeds']]
    arms = ['anchor', 'fixed']+[f'{a}_{s}' for s in cfg['seeds'] for a in ('forced', 'calibrated', 'selected')]
    for path, plan in zip(map(Path, paths), plans):
        if json.loads((path/'protocol.json').read_text()) != protocol:
            raise ValueError('protocol drift across shards')
        status = json.loads((path/'complete.json').read_text())
        if (status.get('status') != 'complete' or status.get('parameter_updates') is not False
            or status.get('training_performed') is not False or status.get('checkpoint_sha256') != expected_hashes
            or status.get('controller_state_before') != expected_states or status.get('controller_state_after') != expected_states
            or status.get('shard') != plan['shard'] or status.get('sources') != design['count']//design['shards']
            or status.get('global_ids_sha256') != digest(ids)):
            raise ValueError('incomplete, misidentified or updated holdout shard state')
        cohorts = json.loads((path/'cohorts.json').read_text())
        expected_ids = {'ar/official-k400-val/'+i for i in ids[plan['shard']::design['shards']]}
        if (set(cohorts) != expected_ids or any(k != 'ar/official-k400-val/'+v['youtube_id'] for k,v in cohorts.items())):
            raise ValueError('deterministic shard cohort membership mismatch')
        shard_rows = json.loads((path/'test_rows.json').read_text())
        keys = [(r['id'], r['codec'], r['qp'], r['arm']) for r in shard_rows]
        expected_grid = {(i,c,q,a) for i in expected_ids for c in cfg['codecs'] for q in cfg['qps'] for a in arms}
        if (len(keys) != len(set(keys)) or set(keys) != expected_grid
            or any(r.get('task') != 'ar' or r.get('split') != 'test' for r in shard_rows)):
            raise ValueError('independent shard row membership/grid mismatch')
        source_ids.extend(v['youtube_id'] for v in cohorts.values())
        pixels.extend(v['pixel_sha256'] for v in cohorts.values())
        rows.extend(shard_rows)
    if (len(source_ids) != design['count'] or len(set(source_ids)) != design['count']
        or set(source_ids) != set(plans[0]['global_ids']) or len(set(pixels)) != design['count']):
        raise ValueError('independent source/pixel membership mismatch')
    return rows


def merge(paths, out, *, protocol_path, draws=None):
    from ..v33ap.evaluate import safe_curve
    from .runner import write_json
    # The preregistered local protocol is authoritative, not a returned shard.
    protocol = json.loads(Path(protocol_path).read_text())
    design = protocol['design']
    expected_draws = design['bootstrap_draws']
    if expected_draws != 5000 or (draws is not None and draws != expected_draws):
        raise ValueError('bootstrap draw count must match preregistered 5000')
    draws = expected_draws
    rows = validate_shards(paths, protocol)
    qps = protocol['settings']['qps']
    report = dict(scope='predeclared independent official K400 validation; no retraining or threshold tuning', n=design['count'],
                  design=design, protocol_sha256=digest(protocol), comparisons={})
    for codec in protocol['settings']['codecs']:
        report['comparisons'][codec] = {}
        for seed in protocol['settings']['seeds']:
            arm = 'selected_' + str(seed)
            report['comparisons'][codec][str(seed)] = {}
            for name in ('swin3d_t', 'r3d_18'):
                ids, arrays = paired_arrays(rows, codec, arm, name, qps)
                if len(ids) != design['count']:
                    raise ValueError('incomplete global paired grid')
                (ar, aq), (br, bq) = arrays
                curve_a = dict(bpp=ar.mean(0).tolist(), quality=aq.mean(0).tolist())
                curve_b = dict(bpp=br.mean(0).tolist(), quality=bq.mean(0).tolist())
                comparison = safe_curve(curve_a, curve_b)
                alpha = .0125 if name == 'swin3d_t' and seed == design['primary_seed'] else .025
                ci = paired_bootstrap(arrays, draws=draws, seed=design['bootstrap_seed'], alpha=alpha)
                raw = [dict(qp=q, corrected=int(((bq[:,j]==1)&(aq[:,j]==0)).sum()),
                            damaged=int(((bq[:,j]==0)&(aq[:,j]==1)).sum()),
                            delta_accuracy_pp=float((bq[:,j]-aq[:,j]).mean()*100),
                            mean_paired_rate_ratio=float((br[:,j]/ar[:,j]).mean())) for j,q in enumerate(qps)]
                report['comparisons'][codec][str(seed)][name] = dict(anchor=curve_a, adaptive=curve_b, comparison=comparison, bootstrap=ci, raw_same_qp=raw)
                print('independent bootstrap', codec, seed, name, flush=True)
    primary = [report['comparisons'][c][str(design['primary_seed'])]['swin3d_t'] for c in protocol['settings']['codecs']]
    report['primary_confirmed'] = all(
        v['comparison'].get('point_passes') is True and v['bootstrap']['hi'] is not None
        and v['bootstrap']['hi'] < 0 and v['bootstrap']['finite_draws'] >= .99*draws for v in primary)
    write_json(Path(out)/'independent_summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    evaluate = sub.add_parser('evaluate')
    for name in ('root', 'protocol', 'annotations', 'exclusions', 'checkpoints', 'out'):
        evaluate.add_argument('--'+name, required=True)
    evaluate.add_argument('--shard', type=int, required=True)
    join = sub.add_parser('merge'); join.add_argument('--paths', nargs='+', required=True)
    join.add_argument('--out', required=True); join.add_argument('--protocol', required=True)
    join.add_argument('--draws', type=int, default=None)
    args = parser.parse_args()
    if args.mode == 'evaluate': run(args)
    else: merge(args.paths, args.out, protocol_path=args.protocol, draws=args.draws)


if __name__ == '__main__':
    main()
