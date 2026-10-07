import copy
import importlib
import importlib.util
import math

import numpy as np
import pytest

from adaptive_vcm.coco_metrics import _remap_bootstrap_sample, coco_map
from adaptive_vcm.v31.protocol import CODECS, QPS


def mod():
    assert importlib.util.find_spec('adaptive_vcm.v31.metrics') is not None, 'paired V31 metrics missing'
    return importlib.import_module('adaptive_vcm.v31.metrics')


def test_hand_derived_ar_metrics():
    g = mod()
    score = g.score_ar([{'source_id':'a', 'probabilities':[.75,.25]},
                        {'source_id':'b', 'probabilities':[.6,.4]}], {'a':0,'b':1})
    assert score['top1_pct'] == 50
    assert score['true_probability'] == pytest.approx(.575)
    assert score['nll'] == pytest.approx(-math.log(.3)/2)
    assert score['brier'] == pytest.approx((.125 + .72)/2)


def test_pchip_primary_known_shift_reverse_and_overlap():
    g = mod()
    ref = {'qps':list(QPS),'rate':[100,80,60,40], 'quality':[90,80,70,60]}
    test = {**ref,'rate':[80,64,48,32]}
    summary = g.summarize_curves(ref,test)
    assert summary['pchip_bd_rate_pct'] == pytest.approx(-20)
    assert summary['cubic_bd_rate_pct'] == pytest.approx(-20)
    assert summary['overlap'] == [60,90]
    assert g.summarize_curves(test,ref)['pchip_bd_rate_pct'] == pytest.approx(25)
    limited = {**test,'quality':[85,75,65,55]}
    assert g.summarize_curves(ref,limited)['overlap'] == [60,85]
    flat = {**test,'quality':[70]*4}
    assert g.summarize_curves(ref,flat)['pchip_bd_rate_pct'] is None
    assert g.summarize_curves(ref,flat)['reason'] == 'constant quality'
    separate = {**test,'quality':[50,40,30,20]}
    assert g.summarize_curves(ref,separate)['reason'] == 'no common quality overlap'
    with pytest.raises(ValueError):
        g.summarize_curves(ref,{**test,'rate':[80,float('nan'),48,32]})


def ar_rows():
    rows = []
    # At each QP lose one of four clips; non-identity packets retain predictions.
    for codec in CODECS:
        for j, qp in enumerate(QPS):
            for i in range(4):
                p = [.8,.2] if i >= j else [.2,.8]
                prediction = {'evaluators': {'r2plus1d_18': {'probabilities':p}}}
                rows.append({'source_id':str(i),'task':'ar','codec':codec,'qp':qp,
                             'ground_truth':{'label':0}, 'source':{'duration':[i+1,1]},
                             'actions':[{'total_bytes':(4-j)*100,'predictions':prediction},
                                        {'total_bytes':(4-j)*80,'predictions':prediction},
                                        {'total_bytes':(4-j)*90,'predictions':prediction}],
                             'choices':{'learned':1,'static':2}})
    return rows


def test_aggregate_rate_is_ratio_of_sums_and_direct_paired_ci():
    g = mod()
    rows = ar_rows()
    curve = g.curves(rows,'ar','anchor','h264')
    assert curve['rate'][0] == pytest.approx(4*400*8/10)
    result = g.paired_comparisons(rows,'ar',[('static','learned'),('anchor','learned')],100,2)
    pair = result['comparisons']['static->learned']['h264']
    assert pair['pchip_bd_rate_pct'] == pytest.approx((80/90-1)*100)
    assert pair['pchip_ci']['lo'] == pytest.approx((80/90-1)*100)
    assert pair['pchip_ci']['hi'] == pytest.approx((80/90-1)*100)
    assert pair['pchip_ci']['finite_fraction'] >= .9
    assert pair['pchip_ci']['sufficient_valid_draws']
    assert result['sampling']['shared_across_qps_codecs_methods_and_pairs'] is True
    assert pair['pchip_bd_rate_pct'] != pytest.approx(-20 - -10)


def test_undefined_draws_are_counted_without_fake_confidence():
    g = mod()
    rows = ar_rows()
    for row in rows:
        for action in row['actions']:
            action['predictions']['evaluators']['r2plus1d_18']['probabilities'] = [.8,.2]
    result = g.paired_comparisons(rows,'ar',[('anchor','learned')],20,4)
    pair = result['comparisons']['anchor->learned']['h264']
    assert pair['pchip_ci']['finite_draws'] == 0
    assert pair['pchip_ci']['invalid_draws'] == 20
    assert pair['pchip_ci']['hi'] is None
    assert not pair['pchip_ci']['sufficient_valid_draws']


def coco_fixture():
    gt = {
        1:[{'id':1,'image_id':1,'category_id':1,'bbox':[0,0,10,10],'area':100,'iscrowd':0},
           {'id':2,'image_id':1,'category_id':1,'bbox':[20,20,10,10],'area':100,'iscrowd':1}],
        2:[{'id':3,'image_id':2,'category_id':1,'bbox':[0,0,10,10],'area':100,'iscrowd':0}],
        3:[]}
    preds = [
        {'source_id':'1','image_id':1,'boxes':[[0,0,10,10],[20,20,30,30],[0,0,10,10]],'scores':[.7,.7,.7],'labels':[1,1,1]},
        {'source_id':'2','image_id':2,'boxes':[[30,30,40,40],[0,0,9,9]],'scores':[.7,.6],'labels':[1,1]},
        {'source_id':'3','image_id':3,'boxes':[[0,0,10,10]],'scores':[.7],'labels':[1]}]
    return preds, {'gt_by_id':gt,'categories':[{'id':1,'name':'object'},{'id':2,'name':'absent'}]}


def actual_resample(preds, annotations, occurrences):
    gt, ids, mapping = _remap_bootstrap_sample(occurrences,annotations['gt_by_id'])
    lookup = {p['image_id']:p for p in preds}
    predictions = []
    for old_id, new_id in mapping:
        p = lookup[old_id]
        predictions.extend({'image_id':new_id,'category_id':label,'score':score,
                            'bbox':[box[0],box[1],box[2]-box[0],box[3]-box[1]]}
                           for box,score,label in zip(p['boxes'],p['scores'],p['labels']))
    return coco_map(predictions,gt,ids,annotations)


@pytest.mark.parametrize('occurrences', [[1,2,3],[2,1,1],[3,3,1],[2,2,2],[1]])
def test_cached_coco_equals_actual_full_remapped_eval(occurrences):
    g = mod()
    preds, annotations = coco_fixture()
    cache = g.COCOMatchCache(preds,annotations)
    expected = actual_resample(preds,annotations,occurrences)
    score = cache.score([str(i) for i in occurrences])
    assert score['map_pct'] == pytest.approx(expected[0]*100,abs=1e-12)
    assert score['map50_pct'] == pytest.approx(expected[1]*100,abs=1e-12)
    assert cache.identity['area'] == [0,1e10] and cache.identity['max_dets'] == 100


def test_coco_empty_predictions_and_native_pixel_rate():
    g = mod()
    preds, annotations = coco_fixture()
    for p in preds:
        p.update(boxes=[],scores=[],labels=[])
    assert g.score_od(preds,annotations)['map_pct'] == 0
    cache = g.COCOMatchCache(preds,annotations)
    assert cache.score(['1','1','2'])['map_pct'] == 0
    rows = []
    for codec in CODECS:
        for qp in QPS:
            for i,p in enumerate(preds):
                rows.append({'task':'od','source_id':p['source_id'],'codec':codec,'qp':qp,
                             'ground_truth':{'image_id':p['image_id'], 'annotations':annotations['gt_by_id'][p['image_id']],
                                             'categories':annotations['categories']},
                             'source':{'original_shape':[10*(i+1),20]},
                             'actions':[{'total_bytes':100,'predictions':{'evaluators':{'resnet50':{'original':p}}}}],
                             'choices':{}})
    assert g.curves(rows,'od','anchor','h264')['rate'][0] == pytest.approx(300*8/1200)


def test_coco_cache_identity_is_portable_canonical_json():
    from adaptive_vcm.v31.protocol import canonical_hash
    g = mod()
    preds,annotations = coco_fixture()
    identity = g.COCOMatchCache(preds,annotations).identity
    assert len(canonical_hash(identity))==64


def test_metrics_fail_on_missing_block_changed_source_or_bad_choice():
    g = mod()
    rows = ar_rows()
    for bad in [rows[1:],rows+[rows[0]]]:
        with pytest.raises(ValueError,match='grid'):
            g.paired_comparisons(bad,'ar',[('anchor','learned')],2,0)
    bad = copy.deepcopy(rows)
    bad[0]['source']['duration'] = [0,1]
    with pytest.raises(ValueError):
        g.curves(bad,'ar','anchor','h264')
    bad = copy.deepcopy(rows)
    bad[0]['choices']['learned'] = 99
    with pytest.raises(ValueError):
        g.curves(bad,'ar','learned','h264')


def test_od_paired_comparison_recomputes_global_ap_for_duplicates():
    g = mod()
    rows = []
    for codec in CODECS:
        for j,qp in enumerate(QPS):
            for i in range(4):
                image_id = i+1
                correct = i >= j
                pred = {'boxes':[[0,0,10,10] if correct else [20,20,30,30]],'scores':[.7],'labels':[1]}
                actions = [{'total_bytes':(4-j)*size,'predictions':{'evaluators':{'resnet50':{'original':pred}}}}
                           for size in (100,80,90)]
                rows.append({'source_id':str(image_id),'task':'od','codec':codec,'qp':qp,
                             'ground_truth':{'image_id':image_id,'categories':[{'id':1,'name':'x'}],
                                 'annotations':[{'id':image_id,'image_id':image_id,'category_id':1,
                                                 'bbox':[0,0,10,10],'area':100,'iscrowd':0}]},
                             'source':{'original_shape':[10,20*(i+1)]},'actions':actions,
                             'choices':{'learned':1,'static':2}})
    curve = g.curves(rows,'od','learned','h264',['2','1','1','4'])
    # Stable equal-score order is [TP,FP,FP,TP] at QP35; global AP is
    # (26*1 + 25*.5)/101, rather than the mean of image-level APs.
    assert curve['quality'] == pytest.approx([100,(26+25*.5)/101*100,26*.25/101*100,26*.25/101*100])
    result = g.paired_comparisons(rows,'od',[('static','learned')],40,2)
    pair = result['comparisons']['static->learned']['h264']
    assert pair['pchip_bd_rate_pct'] == pytest.approx((80/90-1)*100)
    assert pair['pchip_ci']['hi'] == pytest.approx((80/90-1)*100)
    assert pair['pchip_ci']['sufficient_valid_draws']
    assert result['coco_matching_cache']['used']
