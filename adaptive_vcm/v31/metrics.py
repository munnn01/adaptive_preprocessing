"""PCHIP-first task curves and paired source bootstrap over frozen choices."""
from __future__ import annotations

import contextlib
import hashlib
import io
from importlib.metadata import version
import math

import numpy as np

from ..coco_metrics import coco_box
from ..legacy_bd import bd_rate
from ..metrics import pchip_bd
from .protocol import CODECS, QPS, canonical_hash


def summarize_curves(reference, test):
    arrays = []
    for curve in (reference, test):
        rate, quality = np.asarray(curve['rate'], float), np.asarray(curve['quality'], float)
        if curve.get('qps') != list(QPS) or rate.shape != (4,) or quality.shape != (4,) or not np.isfinite(rate).all() or not np.isfinite(quality).all() or np.any(rate <= 0):
            raise ValueError('invalid four-QP curve')
        arrays.append((rate, quality))
    (ra, qa), (rb, qb) = arrays
    low, high = max(qa.min(), qb.min()), min(qa.max(), qb.max())
    reason = 'constant quality' if min(len(np.unique(qa)), len(np.unique(qb))) < 2 else 'no common quality overlap' if high <= low else None
    primary = pchip_bd(ra, qa, rb, qb) if reason is None else math.nan
    cubic = bd_rate(ra, qa, rb, qb) if reason is None else math.nan
    return {'primary_interpolation': 'pchip', 'pchip_bd_rate_pct': float(primary) if math.isfinite(primary) else None,
            'cubic_bd_rate_pct': float(cubic) if math.isfinite(cubic) else None,
            'overlap': [float(low), float(high)] if high > low and reason is None else None,
            'reason': reason or (None if math.isfinite(primary) else 'nonfinite interpolation'),
            'same_qp_quality_gap_pp': (qb - qa).tolist(),
            'same_qp_rate_pct': ((rb / ra - 1)*100).tolist()}


def _ar_probabilities(predictions):
    p = np.asarray([item['probabilities'] for item in predictions], dtype=float)
    if p.ndim != 2 or not len(p) or p.shape[1] < 2 or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-5):
        raise ValueError('invalid AR scoring probabilities')
    return p


def score_ar(predictions, labels):
    p = _ar_probabilities(predictions)
    y = np.asarray([labels[item['source_id']] for item in predictions])
    if y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError('invalid AR scoring labels')
    target = np.eye(p.shape[1])[y]
    true_probability = p[np.arange(len(y)), y]
    return {'top1_pct': float(np.mean(p.argmax(axis=1) == y)*100),
            'true_probability': float(true_probability.mean()),
            'nll': float(-np.log(np.maximum(true_probability, 1e-12)).mean()),
            'brier': float(np.sum((p - target)**2, axis=1).mean())}


class COCOMatchCache:
    """Exact image-local matching, followed by complete global AP per sample.

    COCO matching is independent across images. Repeated occurrences reuse
    those records but contribute detections and nonignored GT each time.
    The stable global score ordering and precision envelope are rebuilt, so
    this never averages per-image AP. Only area=all/maxDets100 is reported.
    """
    def __init__(self, predictions, annotations):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        self.source_ids = [item['source_id'] for item in predictions]
        self.id_by_source = {item['source_id']: int(item['image_id']) for item in predictions}
        if not predictions or len(self.id_by_source) != len(predictions) or len(set(self.id_by_source.values())) != len(predictions):
            raise ValueError('COCO source/image identities must be unique')
        gt_by_id = {int(key): value for key, value in annotations['gt_by_id'].items()}
        ids = sorted(self.id_by_source.values())
        gt = COCO()
        gt.dataset = {'info': {}, 'images': [{'id':i} for i in ids], 'categories':annotations['categories'],
                      'annotations': [dict(a) for i in ids for a in gt_by_id[i]]}
        results = []
        for item in predictions:
            boxes, scores, labels = np.asarray(item['boxes'], float).reshape(-1,4), np.asarray(item['scores'],float), np.asarray(item['labels'])
            if scores.ndim != 1 or labels.ndim != 1 or len(boxes) != len(scores) or len(scores) != len(labels) or not np.isfinite(boxes).all() or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1) or labels.dtype.kind not in 'iuf' or (len(labels) and (not np.isfinite(labels).all() or np.any(labels != np.floor(labels)))):
                raise ValueError('invalid original-coordinate COCO prediction')
            results.extend({'image_id':int(item['image_id']), 'category_id':int(label), 'bbox':coco_box(box), 'score':float(score)}
                           for box,score,label in zip(boxes,scores,labels))
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            if results:
                dt = gt.loadRes(results)
            else:
                dt = COCO()
                dt.dataset = {'images':gt.dataset['images'], 'categories':gt.dataset['categories'], 'annotations':[]}
                dt.createIndex()
            ev = COCOeval(gt,dt,'bbox')
            ev.params.imgIds = ids
            ev.params.areaRng = [[0,1e10]]
            ev.params.areaRngLbl = ['all']
            ev.params.maxDets = [100]
            ev.evaluate()
        self.categories = list(ev.params.catIds)
        self.recall_thresholds = ev.params.recThrs.copy()
        self.records = {(self.categories[k],image_id): ev.evalImgs[k*len(ids)+i]
                        for k in range(len(self.categories)) for i,image_id in enumerate(ids)}
        self.identity = {'version':'v31-exact-coco-match-1', 'pycocotools':version('pycocotools'),
                         'area':[0,1e10], 'max_dets':100, 'iou_thresholds':ev.params.iouThrs.tolist(),
                         'recall_thresholds':ev.params.recThrs.tolist(), 'categories':self.categories,
                         'inputs_hash':canonical_hash({'predictions':predictions,'annotations':{
                             'gt_by_id':{str(k):v for k,v in gt_by_id.items()}, 'categories':annotations['categories']}})}

    def score(self, source_occurrences=None):
        # Normal observed COCOeval sorts IDs; bootstrap remapped IDs preserve
        # occurrence order (1..N), including ties between duplicated images.
        if source_occurrences is None:
            occurrences = sorted(self.source_ids,key=lambda s:self.id_by_source[s])
        else:
            occurrences = list(source_occurrences)
        if not occurrences or any(s not in self.id_by_source for s in occurrences):
            raise ValueError('invalid COCO source sample')
        category_precision = []
        for category in self.categories:
            records = [self.records[(category,self.id_by_source[s])] for s in occurrences]
            records = [r for r in records if r is not None]
            if not records:
                continue
            positive_gt = sum(np.count_nonzero(r['gtIgnore'] == 0) for r in records)
            if positive_gt == 0:
                continue
            scores = np.concatenate([r['dtScores'] for r in records])
            order = np.argsort(-scores,kind='mergesort')
            matches = np.concatenate([r['dtMatches'] for r in records],axis=1)[:,order] != 0
            ignored = np.concatenate([r['dtIgnore'] for r in records],axis=1)[:,order]
            tp = np.cumsum(matches & ~ignored,axis=1).astype(float)
            fp = np.cumsum(~matches & ~ignored,axis=1).astype(float)
            recall = tp / positive_gt
            precision = tp / (tp + fp + np.spacing(1))
            if precision.shape[1]:
                precision = np.maximum.accumulate(precision[:,::-1],axis=1)[:,::-1]
            interpolated = np.zeros((10,101),float)
            for t in range(10):
                indices = np.searchsorted(recall[t], self.recall_thresholds,side='left')
                valid = indices < len(precision[t])
                interpolated[t,valid] = precision[t,indices[valid]]
            category_precision.append(interpolated)
        if not category_precision:
            raise ValueError('COCO AP undefined: no nonignored ground truth')
        precision = np.stack(category_precision,axis=2)
        return {'map_pct':float(precision.mean()*100), 'map50_pct':float(precision[0].mean()*100)}


def score_od(predictions, annotations):
    return COCOMatchCache(predictions,annotations).score()


def _grid(rows,task):
    if task not in ('ar','od') or not rows:
        raise ValueError('empty/invalid task grid')
    ids = sorted({row['source_id'] for row in rows})
    lookup = {(row['source_id'],row['codec'],row['qp']):row for row in rows}
    expected = {(s,c,q) for s in ids for c in CODECS for q in QPS}
    if len(lookup) != len(rows) or set(lookup) != expected or any(row['task'] != task for row in rows):
        raise ValueError('missing or duplicate eight-condition grid')
    for s in ids:
        identity = None
        for c in CODECS:
            for q in QPS:
                row = lookup[s,c,q]
                value = canonical_hash({'source':row['source'],'ground_truth':row['ground_truth']})
                if identity is not None and value != identity:
                    raise ValueError('source identity changed within paired grid')
                identity = value
    return ids,lookup


def _normalizer(row,task):
    if task == 'ar':
        pair = row['source']['duration']
        if len(pair) != 2 or any(type(v) is not int or v <= 0 for v in pair):
            raise ValueError('invalid exact AR duration')
        return pair[0]/pair[1]
    shape = row['source']['original_shape']
    if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('invalid original OD geometry')
    return math.prod(shape)


def _choice(row,method):
    index = 0 if method == 'anchor' else row.get('choices',{}).get(method)
    if type(index) is not int or not 0 <= index < len(row['actions']):
        raise ValueError('invalid/missing frozen method choice')
    action = row['actions'][index]
    if action.get('available',True) is not True or type(action.get('total_bytes')) is not int or action['total_bytes'] <= 0:
        raise ValueError('selected method has no valid transmitted packet')
    return action


def _od_annotations(rows):
    categories = rows[0]['ground_truth']['categories']
    gt = {}
    for row in rows:
        ground = row['ground_truth']
        image_id = int(ground['image_id'])
        if ground['categories'] != categories or (image_id in gt and gt[image_id] != ground['annotations']):
            raise ValueError('inconsistent frozen COCO annotations')
        gt[image_id] = ground['annotations']
    return {'gt_by_id':gt, 'categories':categories}


def _condition_data(ids,lookup,task,method,codec,annotations=None):
    model = 'r2plus1d_18' if task == 'ar' else 'resnet50'
    bits, norms, ar_values, caches = [], [], [], []
    for qp in QPS:
        subset = [lookup[s,codec,qp] for s in ids]
        actions = [_choice(row,method) for row in subset]
        bits.append([action['total_bytes']*8 for action in actions])
        norms.append([_normalizer(row,task) for row in subset])
        if task == 'ar':
            p = _ar_probabilities([action['predictions']['evaluators'][model] for action in actions])
            y = np.asarray([row['ground_truth']['label'] for row in subset])
            if y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= p.shape[1]):
                raise ValueError('invalid independent AR labels')
            true = p[np.arange(len(y)),y]
            ar_values.append(np.stack([(p.argmax(axis=1)==y)*100.,true,-np.log(np.maximum(true,1e-12)),
                                       np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)],axis=1))
        else:
            predictions = [{'source_id':row['source_id'],'image_id':row['ground_truth']['image_id'],
                            **action['predictions']['evaluators'][model]['original']}
                           for row,action in zip(subset,actions)]
            caches.append(COCOMatchCache(predictions,annotations))
    return {'bits':np.asarray(bits,float).T,'normalizers':np.asarray(norms,float).T,
            'ar_values':np.stack(ar_values,axis=1) if task == 'ar' else None,'caches':caches}


def _curve(data,task,ids,sample=None):
    indices = np.arange(len(ids)) if sample is None else np.asarray(sample)
    rate = data['bits'][indices].sum(axis=0)/data['normalizers'][indices].sum(axis=0)
    if task == 'ar':
        values = data['ar_values'][indices].mean(axis=0)
        quality = values[:,0].tolist()
        continuous = [dict(zip(('top1_pct','true_probability','nll','brier'),map(float,value))) for value in values]
    else:
        occurrences = None if sample is None else [ids[i] for i in indices]
        continuous = [cache.score(occurrences) for cache in data['caches']]
        quality = [value['map_pct'] for value in continuous]
    return {'qps':list(QPS),'rate':rate.tolist(),'quality':quality,'continuous':continuous,
            'source_count':len(indices),'rate_unit':'bits_per_second' if task == 'ar' else 'bits_per_original_pixel'}


def curves(rows,task,method,codec,source_occurrences=None):
    ids,lookup = _grid(rows,task)
    if codec not in CODECS:
        raise ValueError('invalid curve codec')
    data = _condition_data(ids,lookup,task,method,codec,_od_annotations(rows) if task == 'od' else None)
    sample = [ids.index(s) for s in source_occurrences] if source_occurrences is not None else None
    return _curve(data,task,ids,sample)


def _ci(values,draws):
    valid = [value for value in values if value is not None and math.isfinite(value)]
    fraction = len(valid)/draws if draws else 0.
    return {'lo':float(np.percentile(valid,2.5)) if valid else None,
            'hi':float(np.percentile(valid,97.5)) if valid else None,
            'draws':draws,'finite_draws':len(valid),'invalid_draws':draws-len(valid),
            'finite_fraction':fraction,'sufficient_valid_draws':bool(draws and fraction >= .9)}


def paired_comparisons(rows,task,comparisons,draws,seed,annotations=None):
    ids,lookup = _grid(rows,task)
    if type(draws) is not int or draws < 0 or type(seed) is not int or not comparisons or len(set(comparisons)) != len(comparisons):
        raise ValueError('invalid paired bootstrap request')
    methods = sorted({name for pair in comparisons for name in pair})
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in comparisons):
        raise ValueError('invalid direct comparisons')
    ann = _od_annotations(rows) if task == 'od' else None
    if annotations is not None and canonical_hash({'gt_by_id':{str(k):v for k,v in annotations['gt_by_id'].items()},'categories':annotations['categories']}) != canonical_hash({'gt_by_id':{str(k):v for k,v in ann['gt_by_id'].items()},'categories':ann['categories']}):
        raise ValueError('external annotations differ from frozen rows')
    datasets = {(codec,method):_condition_data(ids,lookup,task,method,codec,ann) for codec in CODECS for method in methods}
    observed = {key:_curve(data,task,ids) for key,data in datasets.items()}
    output, samples = {}, np.random.default_rng(seed).integers(0,len(ids),size=(draws,len(ids)))
    values = {(ref,test,codec):{'pchip':[],'cubic':[]} for ref,test in comparisons for codec in CODECS}
    for sample in samples:
        sampled = {key:_curve(data,task,ids,sample) for key,data in datasets.items()}
        for ref,test in comparisons:
            for codec in CODECS:
                summary = summarize_curves(sampled[codec,ref],sampled[codec,test])
                values[ref,test,codec]['pchip'].append(summary['pchip_bd_rate_pct'])
                values[ref,test,codec]['cubic'].append(summary['cubic_bd_rate_pct'])
    for ref,test in comparisons:
        output[f'{ref}->{test}'] = {}
        for codec in CODECS:
            summary = summarize_curves(observed[codec,ref],observed[codec,test])
            summary.update(pchip_ci=_ci(values[ref,test,codec]['pchip'],draws),
                           cubic_ci=_ci(values[ref,test,codec]['cubic'],draws))
            output[f'{ref}->{test}'][codec] = summary
    return {'task':task,'primary_model':'r2plus1d_18' if task == 'ar' else 'resnet50',
            'curves':{method:{codec:observed[codec,method] for codec in CODECS} for method in methods},
            'comparisons':output,'sampling':{'method':'paired_source_bootstrap','seed':seed,'draws':draws,
                'source_ids':ids,'shared_across_qps_codecs_methods_and_pairs':True,
                'sample_indices_sha256':hashlib.sha256(samples.astype('<i8').tobytes()).hexdigest()},
            'coco_matching_cache':{'used':task == 'od','identities':[cache.identity for data in datasets.values() for cache in data['caches']]}}
