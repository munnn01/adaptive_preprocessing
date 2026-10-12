"""Fixed V32 MOT20 evaluation with an explicit Hungarian IoU reference tracker."""
from __future__ import annotations

import argparse
import configparser
import copy
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from ..analyzers import DetectionAnalyzer
from ..motion_support import build_motion_support
from ..preprocessing import boxes_to_mask, validate_clip
from ..selection import relative_guard
from .codec import V32Codec
from .evaluate import empty_output
from .protocol import ROOT, load_config, provenance
from .selection import choose_stream

TRACKEVAL_COMMIT = '12c8791b303e0a0b50f753af204249e622d0281a'
ARMS = ('anchor', 'controls', 'simplification', 'retarget', 'v32')
TRACKER_RECIPE = dict(name='hungarian_iou_reference', coco_label=1, score_threshold=.25,
                      iou_threshold=.3, max_age_missing_frames=2,
                      motion_model=None, appearance_model=None, training=False)


def person_detections(detections):
    boxes = np.asarray(detections['boxes'], float).reshape(-1, 4)
    scores = np.asarray(detections['scores'], float)
    labels = np.asarray(detections['labels'])
    if (scores.shape != (len(boxes),) or labels.shape != scores.shape
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()):
        raise ValueError('invalid person detection arrays')
    valid = (labels == 1) & (scores >= .25) & np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
    return dict(boxes=boxes[valid].copy(), scores=scores[valid].copy(), labels=labels[valid].copy())


def iou_matrix(first, second):
    first, second = np.asarray(first).reshape(-1, 4), np.asarray(second).reshape(-1, 4)
    low = np.maximum(first[:, None, :2], second[None, :, :2])
    high = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(2)
    area_a = np.maximum(first[:, 2:] - first[:, :2], 0).prod(1)
    area_b = np.maximum(second[:, 2:] - second[:, :2], 0).prod(1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


def match_boxes(first, second):
    """Maximum valid cardinality then total IoU; deterministic stable input order."""
    overlap = iou_matrix(first, second)
    if not overlap.size:
        return []
    rows, cols = linear_sum_assignment(np.where(overlap >= .3, 1 - overlap, 1e6))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if overlap[i, j] >= .3]


class ReferenceTracker:
    """Last-box matching only: this is neither SORT nor ByteTrack nor SOTA."""
    def __init__(self):
        self.frame_number = 0
        self.next_id = 1
        self.tracks = {}

    def step(self, detections):
        self.frame_number += 1
        frame = person_detections(detections)
        self.tracks = {k: v for k, v in self.tracks.items()
                       if self.frame_number - v['frame'] <= 3}
        active = sorted(self.tracks)
        prior_boxes = np.asarray([self.tracks[k]['box'] for k in active]).reshape(-1, 4)
        assigned, links = {}, []
        for previous, current in match_boxes(prior_boxes, frame['boxes']):
            identifier = active[previous]
            track = self.tracks[identifier]
            assigned[current] = identifier
            links.append((track['frame'], track['index'], self.frame_number, current))
        identifiers = []
        for index, box in enumerate(frame['boxes']):
            identifier = assigned.get(index)
            if identifier is None:
                identifier = self.next_id
                self.next_id += 1
            identifiers.append(identifier)
            self.tracks[identifier] = dict(box=box.copy(), frame=self.frame_number, index=index)
        return dict(**frame, track_ids=np.asarray(identifiers, int),
                    frame_number=self.frame_number, associations=links)


class TrackingTeacher:
    """All probes clone a common prefix; only commit advances the anchor state."""
    def __init__(self, detector):
        self.detector = detector
        self.tracker = ReferenceTracker()
        self.probe_predictions = []

    def predict(self, clip):
        validate_clip(clip)
        tracker = copy.deepcopy(self.tracker)
        prefix = {f"{v['frame']}:{v['index']}": k for k, v in tracker.tracks.items()}
        frames = [tracker.step(self.detector.predict(frame[None])) for frame in clip]
        result = dict(frames=frames, scores=np.concatenate([f['scores'] for f in frames]),
                    associations=[link for frame in frames for link in frame['associations']],
                    prefix_track_ids=prefix, _terminal_state=tracker,
                    rgb_sha256=hashlib.sha256(clip.tobytes()).hexdigest())
        self.probe_predictions.append(result)
        return result

    def commit(self, prediction):
        self.tracker = copy.deepcopy(prediction['_terminal_state'])


def tracking_guard_details(task, source_predictions, anchor_predictions, predictions, cfg):
    if task != 'ot' or any(len(p) != 1 for p in (source_predictions, anchor_predictions, predictions)):
        raise ValueError('tracking guard requires one OT teacher')
    source, anchor, trial = source_predictions[0], anchor_predictions[0], predictions[0]
    if not len(source['frames']) == len(anchor['frames']) == len(trial['frames']):
        raise ValueError('tracking guard temporal mismatch')
    details, mappings, distances, decisions = [], {}, [], []
    for source_frame, anchor_frame, trial_frame in zip(source['frames'], anchor['frames'], trial['frames']):
        distance, decision = relative_guard('od', [source_frame], [anchor_frame], [trial_frame], cfg)
        value = float(distance[0])
        if not np.isfinite(value):
            value = float('inf')  # Empty source evidence never admits a nonidentity proposal.
        preserved = bool(decision[0] and value <= cfg['od_distance_slack'])
        distances.append(value)
        decisions.append(preserved)
        number = anchor_frame['frame_number']
        if number != trial_frame['frame_number']:
            raise ValueError('tracking guard frame ID mismatch')
        mappings[number] = {i: j for i, j in match_boxes(anchor_frame['boxes'], trial_frame['boxes'])}
        details.append(dict(frame=number, relative_distance=value if np.isfinite(value) else None,
                            preserves_decision=preserved))
    trial_by_frame = {f['frame_number']: f for f in trial['frames']}
    prefix = anchor.get('prefix_track_ids', {})
    def trial_id(frame, index):
        if frame not in trial_by_frame:
            return prefix.get(f'{frame}:{index}')
        mapped = mappings[frame].get(index)
        return None if mapped is None else int(trial_by_frame[frame]['track_ids'][mapped])
    associations = []
    for f1, i1, f2, i2 in anchor['associations']:
        first, second = trial_id(f1, i1), trial_id(f2, i2)
        associations.append(dict(link=[f1, i1, f2, i2], preserved=first is not None and first == second))
    links_preserved = all(link['preserved'] for link in associations)
    return dict(frames=details, associations=associations, preserves_associations=links_preserved,
                maximum_relative_distance=max(distances),
                preserves_decision=all(decisions) and links_preserved)


def tracking_guard(task, source_predictions, anchor_predictions, predictions, cfg):
    result = tracking_guard_details(task, source_predictions, anchor_predictions, predictions, cfg)
    return (result['maximum_relative_distance'],), (result['preserves_decision'],)


class TrackingGuard:
    def __init__(self):
        self.records = []

    def __call__(self, task, source, anchor, trial, cfg):
        result = tracking_guard_details(task, source, anchor, trial, cfg)
        result['reconstructed_pixel_sha256'] = trial[0].get('rgb_sha256')
        self.records.append(result)
        return (result['maximum_relative_distance'],), (result['preserves_decision'],)


def source_support(rgb, prediction, motion=False):
    """THW detector masks for ROI, their HW union for historical OD controls."""
    validate_clip(rgb)
    if len(prediction['frames']) != len(rgb):
        raise ValueError('support temporal mismatch')
    protection = np.stack([boxes_to_mask(*rgb.shape[1:3], person_detections(frame)['boxes'])
                           for frame in prediction['frames']])
    union = protection.max(0)
    if motion:
        support = build_motion_support(rgb, union, 'ar')
        support['protection'] = np.maximum(protection, support['protection'])
    else:
        cuts = np.zeros(len(rgb), bool)
        cuts[0] = True
        support = dict(protection=protection, motion=np.zeros_like(protection), cuts=cuts,
                       metadata=dict(task='ot', flow_source=None, roi_source='frozen_source_person_boxes'))
    return union, support


def plan_sequences(root, sequences, max_frames):
    if max_frames < 1 or not sequences or len(sequences) != len(set(sequences)):
        raise ValueError('invalid MOT20 sequence/frame scope')
    plan = []
    for name in sequences:
        if name not in ('MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'):
            raise ValueError('select official MOT20 train sequence names')
        path = Path(root) / name
        ini = path / 'seqinfo.ini'
        parser = configparser.ConfigParser()
        if not parser.read(ini):
            raise FileNotFoundError(f'missing seqinfo.ini: {ini}')
        seq = parser['Sequence']
        fps = Fraction(seq['frameRate'])
        total, width, height = (int(seq[k]) for k in ('seqLength', 'imWidth', 'imHeight'))
        if fps != 25 or min(total, width, height) < 1 or seq['name'] != name or seq['imDir'] != 'img1' or seq['imExt'] != '.jpg':
            raise ValueError('invalid MOT20 seqinfo geometry/FPS')
        frames = min(max_frames, total)
        images = [path / 'img1' / f'{number:06}.jpg' for number in range(1, frames + 1)]
        for file in [*images, path / 'gt' / 'gt.txt']:
            if not file.is_file():
                raise FileNotFoundError(f'missing MOT20 file: {file}')
        plan.append(dict(name=name, path=path, images=images, frames=frames, full_frames=total,
                         width=width, height=height, fps=fps))
    return plan


def read_sequence(item, size):
    if size < 2 or size % 2:
        raise ValueError('canonical MOT20 size must be positive and even')
    width, height = item['width'], item['height']
    scale = size / max(width, height)
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    frames = []
    for file in item['images']:
        with Image.open(file) as image:
            if image.size != (width, height):
                raise ValueError('MOT20 image differs from seqinfo geometry')
            canonical = Image.new('RGB', (size, size))
            canonical.paste(image.convert('RGB').resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
            frames.append(np.asarray(canonical).copy())
    return np.stack(frames), (nw / width, nh / height, left, top)


def original_boxes(boxes, geometry, width, height):
    sx, sy, left, top = geometry
    boxes = np.asarray(boxes, float).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / sx, 0, width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / sy, 0, height)
    return boxes


def stage_ground_truth(plan, destination):
    """Called only after label-free selection and evaluator predictions finish."""
    destination, manifest = Path(destination), {}
    for item in plan:
        source = item['path'] / 'gt' / 'gt.txt'
        raw = source.read_bytes()
        selected = []
        for line in raw.decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            columns = line.split(',')
            values = np.asarray(columns, float)
            if len(values) < 9 or not np.isfinite(values).all() or values[0] != int(values[0]) or not 1 <= values[0] <= item['full_frames']:
                raise ValueError('invalid MOT20 GT row')
            if values[0] <= item['frames']:
                selected.append(line)
        staged = destination / item['name']
        (staged / 'gt').mkdir(parents=True, exist_ok=False)
        gt = ('\n'.join(selected) + ('\n' if selected else '')).encode()
        (staged / 'gt' / 'gt.txt').write_bytes(gt)
        ini = configparser.ConfigParser()
        ini.read(item['path'] / 'seqinfo.ini')
        ini['Sequence']['seqLength'] = str(item['frames'])
        with (staged / 'seqinfo.ini').open('w', encoding='utf-8') as handle:
            ini.write(handle)
        manifest[item['name']] = dict(source_sha256=hashlib.sha256(raw).hexdigest(),
            staged_sha256=hashlib.sha256(gt).hexdigest(), selected_rows=len(selected),
            retains_all_classes=True, frames=[1, item['frames']])
    return manifest


def load_trackeval(path):
    path = Path(path).resolve()
    if not (path / 'trackeval' / '__init__.py').is_file():
        raise RuntimeError('TrackEval pinned checkout is required before any model/probe work')
    try:
        commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        clean = subprocess.run(['git', '-C', str(path), 'diff', '--exit-code', 'HEAD', '--', 'trackeval'],
                               capture_output=True, check=False).returncode == 0
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError('TrackEval must be a Git checkout at the pinned commit') from error
    if commit != TRACKEVAL_COMMIT or not clean:
        raise RuntimeError('TrackEval commit or metric source differs from the pinned checkout')
    # This historical official pin uses NumPy aliases removed in NumPy 1.24.
    for alias, builtin in [('float', float), ('int', int), ('bool', bool)]:
        if alias not in np.__dict__:
            setattr(np, alias, builtin)
    sys.path.insert(0, str(path))
    try:
        module = importlib.import_module('trackeval')
    except ImportError as error:
        raise RuntimeError(f'TrackEval dependencies unavailable: {error}') from error
    if Path(module.__file__).resolve() != path / 'trackeval' / '__init__.py':
        raise RuntimeError('a different TrackEval module is already imported')
    return module


def write_mot(path, frames, geometry, width, height):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        for frame in frames:
            boxes = original_boxes(frame['boxes'], geometry, width, height)
            for identifier, box, score in zip(frame['track_ids'], boxes, frame['scores']):
                x1, y1, x2, y2 = box
                if x2 > x1 and y2 > y1:
                    handle.write(f"{frame['frame_number']},{identifier},{x1:.6f},{y1:.6f},"
                                 f"{x2-x1:.6f},{y2-y1:.6f},{score:.6f},-1,-1,-1\n")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')


def model_provenance(analyzer):
    model = getattr(analyzer, 'model', None)
    if model is None:
        return dict(name=analyzer.name, analytic_fixture=True, weights=None, state_sha256=None, frozen=None)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    weights = {'mobilenet': 'FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1',
               'resnet50': 'FasterRCNN_ResNet50_FPN_Weights.COCO_V1'}
    return dict(name=analyzer.name, weights=weights[analyzer.name],
        state_sha256=digest.hexdigest(), frozen=not any(p.requires_grad for p in model.parameters()),
        torch=str(torch.__version__), torchvision=importlib.metadata.version('torchvision'))


def evaluate_mot(package, plan, out, tracker_names):
    out = Path(out)
    dataset_config = dict(GT_FOLDER=str(out / 'ground_truth'), TRACKERS_FOLDER=str(out / 'trackers'),
        OUTPUT_FOLDER=str(out / 'official_metrics'), TRACKERS_TO_EVAL=tracker_names,
        BENCHMARK='MOT20', SPLIT_TO_EVAL='train', SKIP_SPLIT_FOL=True,
        DO_PREPROC=True, PRINT_CONFIG=False, SEQ_INFO={item['name']: item['frames'] for item in plan})
    evaluation_config = dict(PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False,
        USE_PARALLEL=False, BREAK_ON_ERROR=True, LOG_ON_ERROR=None,
        OUTPUT_SUMMARY=True, OUTPUT_DETAILED=True, PLOT_CURVES=False)
    dataset = package.datasets.MotChallenge2DBox(dataset_config)
    evaluator = package.Evaluator(evaluation_config)
    metric_config = dict(PRINT_CONFIG=False, THRESHOLD=.5)
    metrics = [package.metrics.HOTA(), package.metrics.CLEAR(metric_config), package.metrics.Identity(metric_config)]
    results, messages = evaluator.evaluate([dataset], metrics)
    raw = results[dataset.get_name()]
    write_json(out / 'trackeval_raw_results.json', dict(results=raw, messages=messages,
        dataset_config=dataset_config, evaluator_config=evaluation_config, commit=TRACKEVAL_COMMIT))
    def extract(value):
        value = value['pedestrian']
        return dict(HOTA=float(np.mean(value['HOTA']['HOTA']) * 100),
            IDF1=float(value['Identity']['IDF1'] * 100), MOTA=float(value['CLEAR']['MOTA'] * 100),
            IDs=int(value['CLEAR']['IDSW']), FP=int(value['CLEAR']['CLR_FP']),
            FN=int(value['CLEAR']['CLR_FN']), TP=int(value['CLEAR']['CLR_TP']))
    return {name: dict(combined=extract(raw[name]['COMBINED_SEQ']),
                       sequences={item['name']: extract(raw[name][item['name']]) for item in plan})
            for name in tracker_names}


def exploratory_summary(points):
    from ..legacy_bd import bd_rate, bd_metric
    from ..metrics import pchip_bd
    qps = [30, 35, 40, 45, 50]
    expected = {(codec, qp, arm) for codec in ('h264', 'h265') for qp in qps for arm in ARMS}
    keys = [(p['codec'], p['qp'], p['arm']) for p in points]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('incomplete or duplicate tracking paired grid')
    lookup = {key: point for key, point in zip(keys, points)}
    results = {}
    for codec in ('h264', 'h265'):
        results[codec] = {}
        anchor = [lookup[(codec, qp, 'anchor')] for qp in qps]
        ra = [p['bpp'] for p in anchor]
        for arm in ARMS[1:]:
            selected = [lookup[(codec, qp, arm)] for qp in qps]
            rb = [p['bpp'] for p in selected]
            results[codec][arm] = {}
            for metric in ('HOTA', 'IDF1', 'MOTA'):
                qa, qb = ([p['metrics'][metric] / 100 for p in rows] for rows in (anchor, selected))
                results[codec][arm][metric] = json_safe(dict(
                    bd_rate_pct=bd_rate(ra, qa, rb, qb), pchip_bd_rate_pct=pchip_bd(ra, qa, rb, qb),
                    bd_quality_pp=bd_metric(ra, qa, rb, qb) * 100,
                    same_qp_gap_pp=(np.asarray(qb) - qa) * 100,
                    same_qp_rate_pct=(np.asarray(rb) / ra - 1) * 100,
                    qps=qps, anchor_bpp=ra, selected_bpp=rb,
                    anchor_quality_pct=np.asarray(qa)*100, selected_quality_pct=np.asarray(qb)*100))
    return dict(task='ot', scope='fixed_roi_only_tracking', complete_paired_grid=True,
        results=results, points=points, target_confirmed=False, bootstrap_confidence=None,
        interpretation='exploratory contiguous MOT20 prefix; no full-benchmark or statistical target claim',
        rate_scope='wire header + grid metadata + actual elementary stream',
        selection_scope='all fixed candidate probes measured; labels and evaluator excluded')


def run(args, *, extension=None):
    plugin = extension or {}
    out = empty_output(args.out)
    trackeval = load_trackeval(args.trackeval_root)
    cfg = load_config(args.config)
    if args.size < 2 or args.size % 2 or args.size > 4096:
        raise ValueError('MOT20 canonical size must be even and between 2 and 4096')
    plan = plan_sequences(args.root, args.sequences, args.max_frames)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_num_threads(2)
    tracking_config = dict(tracker=TRACKER_RECIPE, canonical_size=args.size,
        window_frames=16, temporal_subsample=1, padding_frames=0, motion_support=args.motion_support,
        max_frames=args.max_frames, sequences=args.sequences, trackeval_commit=TRACKEVAL_COMMIT)
    run_config = dict(base=cfg, tracking=tracking_config)
    manifest = dict(**provenance(run_config), status='running', task='ot',
        scope='fixed_roi_only_tracking', config=run_config, device=device,
        python=platform.python_version(), torch=str(torch.__version__),
        temporal_subsample=1, padding_frames=0, codec_window_frames=16,
        teacher='frozen_coco_mobilenet_fasterrcnn', evaluator='frozen_coco_resnet50_fasterrcnn',
        trackeval_commit=TRACKEVAL_COMMIT, tracker=TRACKER_RECIPE,
        rate_denominator='canonical source T*H*W pixels', sequences={},
        ground_truth_access='only after all label-free selections and evaluator predictions',
        selection_cost='all candidate encodes, received packet decodes and frozen teacher frame calls',
        baseline=None, training=False, target_confirmed=False, bootstrap_confidence=None)
    manifest.update(plugin.get('manifest', {}))
    write_json(out / 'config_snapshot.json', run_config)
    write_json(out / 'manifest.json', manifest)
    teacher_detector = DetectionAnalyzer(cfg['od_teacher'], device)
    evaluator_detector = DetectionAnalyzer(cfg['od_evaluator'], device)
    manifest['models'] = dict(teacher=model_provenance(teacher_detector),
                              evaluator=model_provenance(evaluator_detector))
    write_json(out / 'manifest.json', manifest)
    points = {(codec, qp, arm): dict(codec=codec, qp=qp, arm=arm, coded_bytes=0,
        elementary_bytes=0, metadata_bytes=0, source_pixels=0, prediction_paths={})
        for codec in cfg['codecs'] for qp in cfg['qps'] for arm in ARMS}

    def append(path, value):
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(json_safe(value), allow_nan=False) + '\n')

    for item in plan:
        rgb, geometry = read_sequence(item, args.size)
        name = item['name']
        snapshot = out / 'source_snapshots' / name
        snapshot.mkdir(parents=True)
        original_ini = (item['path'] / 'seqinfo.ini').read_bytes()
        (snapshot / 'seqinfo.ini').write_bytes(original_ini)
        frame_ids = [f'{name}/{number:06}' for number in range(1, len(rgb) + 1)]
        manifest['sequences'][name] = dict(frames=[1, len(rgb)], full_sequence_frames=item['full_frames'],
            truncated=len(rgb) < item['full_frames'], original_width=item['width'], original_height=item['height'],
            fps=[item['fps'].numerator, item['fps'].denominator], letterbox=list(geometry), ids=frame_ids,
            ids_sha256=hashlib.sha256('\n'.join(frame_ids).encode()).hexdigest(),
            seqinfo_sha256=hashlib.sha256(original_ini).hexdigest(),
            frame_file_sha256={identifier: hashlib.sha256(path.read_bytes()).hexdigest()
                               for identifier, path in zip(frame_ids, item['images'])},
            canonical_pixel_sha256={identifier: hashlib.sha256(frame.tobytes()).hexdigest()
                                    for identifier, frame in zip(frame_ids, rgb)})
        write_json(out / 'manifest.json', manifest)
        prepared = []
        clean_teacher = plugin.get('teacher_factory', TrackingTeacher)(teacher_detector)
        for start in range(0, len(rgb), 16):
            clip = rgb[start:start + 16]
            before = time.perf_counter()
            source = clean_teacher.predict(clip)
            union, support = plugin.get('support', source_support)(clip, source, args.motion_support)
            clean_teacher.commit(source)
            clean_teacher.probe_predictions.clear()
            preparation_seconds = time.perf_counter() - before
            prepared.append((start, clip, source, union, support, preparation_seconds))
            append(out / 'preparation_audit.jsonl', dict(id=name, frame_start=start+1,
                frame_end=start+len(clip), source_sha256=source['rgb_sha256'],
                teacher_frame_calls=len(clip), seconds=preparation_seconds,
                cached_across_codec_qp_points=True, source_support=support['metadata']))
        for codec_name in cfg['codecs']:
            for qp in cfg['qps']:
                codec = plugin.get('codec_factory', V32Codec)(codec_name, qp, cfg['preset'], item['fps'])
                teacher = plugin.get('teacher_factory', TrackingTeacher)(teacher_detector)
                trackers = {arm: ReferenceTracker() for arm in ARMS}
                evaluated_frames = {arm: [] for arm in ARMS}
                for start, clip, source, union, support, preparation_seconds in prepared:
                    guard = plugin.get('guard_factory', TrackingGuard)()
                    teacher.probe_predictions.clear()
                    bundle = choose_stream(clip, union, support, 'ot', codec, cfg,
                                           [teacher], [source], guard=guard)
                    anchor_prediction = teacher.probe_predictions[0]
                    teacher.commit(anchor_prediction)
                    identity_details = dict(frames=[dict(frame=f['frame_number'], relative_distance=0.,
                        preserves_decision=True) for f in anchor_prediction['frames']],
                        associations=[dict(link=list(link), preserved=True) for link in anchor_prediction['associations']],
                        preserves_associations=True, maximum_relative_distance=0., preserves_decision=True)
                    details_by_hash = {record['reconstructed_pixel_sha256']: record for record in guard.records}
                    details_by_hash[anchor_prediction['rgb_sha256']] = identity_details
                    for candidate in bundle.audit:
                        candidate['tracking_guard'] = details_by_hash.get(candidate['reconstructed_pixel_sha256'])
                        if candidate['tracking_guard'] is None:
                            raise RuntimeError('missing per-frame tracking guard audit')
                    predictions_cache, evaluator_calls = {}, 0
                    evaluation_start = time.perf_counter()
                    for arm, (packet, candidate) in {'anchor': (bundle.anchor, 'identity'), **bundle.arms}.items():
                        if packet.decoded.shape != clip.shape:
                            raise RuntimeError('evaluator must receive canonical reconstructed coordinates')
                        digest = packet.packet_sha256
                        receiver_record = {}
                        if plugin.get('receiver'):
                            received = plugin['receiver'](packet, sequence=name,
                                arm=f'{codec_name}/q{qp}/{arm}', frame_start=start+1,
                                detector=evaluator_detector, tracker=trackers[arm])
                            evaluated_frames[arm].extend(received['ot']['frames'])
                            evaluator_calls += received['ot']['detector_frame_calls']
                            receiver_record = dict(shared_receiver={k:v for k,v in received.items() if k!='ot'})
                            append(out / 'action_predictions.jsonl', dict(id=name,codec=codec_name,qp=qp,
                                arm=arm,**receiver_record['shared_receiver']))
                        else:
                            if digest not in predictions_cache:
                                predictions_cache[digest] = [evaluator_detector.predict(frame[None]) for frame in packet.decoded]
                                evaluator_calls += len(packet.decoded)
                            evaluated_frames[arm].extend(trackers[arm].step(d) for d in predictions_cache[digest])
                        relative = Path('packets') / name / codec_name / f'q{qp}-{arm}-{start+1:06}.v32'
                        (out / relative).parent.mkdir(parents=True, exist_ok=True)
                        with (out / relative).open('xb') as handle:
                            handle.write(packet.data)
                        point = points[(codec_name, qp, arm)]
                        point['coded_bytes'] += packet.coded_bytes
                        point['elementary_bytes'] += packet.elementary_bytes
                        point['metadata_bytes'] += packet.metadata_bytes
                        point['source_pixels'] += int(np.prod(clip.shape[:3]))
                        append(out / 'rows.jsonl', dict(id=name, frame_start=start+1, frame_end=start+len(clip),
                            codec=codec_name, qp=qp, arm=arm, candidate=candidate,
                            coded_bytes=packet.coded_bytes, elementary_bytes=packet.elementary_bytes,
                            metadata_bytes=packet.metadata_bytes, bpp=8*packet.coded_bytes / np.prod(clip.shape[:3]),
                            source_shape=list(clip.shape), coded_shape=list(packet.coded_shape),
                            reconstructed_shape=list(packet.decoded.shape), packet_sha256=digest,
                            source_sha256=source['rgb_sha256'], packet_path=relative.as_posix(),
                            source_preparation_seconds=preparation_seconds, selection_seconds=bundle.seconds,
                            **receiver_record))
                    append(out / 'selection_audit.jsonl', dict(id=name, codec=codec_name, qp=qp,
                        frame_start=start+1, frame_end=start+len(clip), candidates=bundle.audit,
                        selected={arm: candidate for arm, (_, candidate) in bundle.arms.items()},
                        source_support=support['metadata'], preparation_seconds=preparation_seconds,
                        selection_seconds=bundle.seconds, evaluator_seconds=time.perf_counter()-evaluation_start,
                        probe_cost=dict(encodes=sum(r['distinct_encode'] for r in bundle.audit),
                            packet_decodes=sum(r['distinct_packet_decode'] for r in bundle.audit),
                            teacher_clip_calls=sum(r['teacher_calls'] for r in bundle.audit),
                            teacher_frame_calls=sum(r['teacher_calls'] for r in bundle.audit)*len(clip),
                            source_teacher_frame_calls=0, source_preparation_cached=True,
                            independent_evaluator_frame_calls=evaluator_calls)))
                    print(f'[ot] {name} {codec_name} QP{qp} frames {start+1}-{start+len(clip)} complete', flush=True)
                for arm in ARMS:
                    tracker_name = f'{codec_name}_q{qp}_{arm}'
                    relative = Path('trackers') / tracker_name / 'data' / f'{name}.txt'
                    write_mot(out / relative, evaluated_frames[arm], geometry, item['width'], item['height'])
                    points[(codec_name, qp, arm)]['prediction_paths'][name] = relative.as_posix()
    manifest['ground_truth'] = stage_ground_truth(plan, out / 'ground_truth')
    manifest['status'] = 'scoring'
    write_json(out / 'manifest.json', manifest)
    names = [f'{codec}_q{qp}_{arm}' for codec, qp, arm in points]
    metrics = evaluate_mot(trackeval, plan, out, names)
    for (codec, qp, arm), point in points.items():
        point['bpp'] = 8*point['coded_bytes'] / point['source_pixels']
        point['metrics'] = metrics[f'{codec}_q{qp}_{arm}']['combined']
        point['sequence_metrics'] = metrics[f'{codec}_q{qp}_{arm}']['sequences']
    summary = exploratory_summary(list(points.values()))
    if plugin:
        summary.update({k:v for k,v in plugin.get('manifest',{}).items()
                        if k in ('scope','active_tasks','action_accuracy','action_bd_rate','action_metric_reason')})
    write_json(out / 'summary.json', summary)
    manifest['status'] = 'complete'
    write_json(out / 'manifest.json', manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--sequences', nargs='+', default=['MOT20-01'])
    parser.add_argument('--max-frames', type=int, default=32)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--trackeval-root', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--motion-support', action='store_true')
    result = run(parser.parse_args())
    print(json.dumps(dict(status='complete', task='ot', points=len(result['points']), target_confirmed=False)), flush=True)


if __name__ == '__main__':
    main()
