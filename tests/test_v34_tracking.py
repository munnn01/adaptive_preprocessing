import numpy as np
import pytest
import os
from pathlib import Path
from adaptive_vcm.v34.tracking import ByteTracker, tracking_cost, ot_split
from adaptive_vcm.v32.tracking import original_boxes

def pred(score=.9,x=10):
    return dict(boxes=np.array([[x,10,x+20,40]],float),scores=np.array([score]),labels=np.array([1]))

def test_bytetrack_retains_low_score_link_and_continues_across_windows():
    tracker=ByteTracker(); first=tracker.step(pred())
    second=tracker.step(pred(.8,11)); third=tracker.step(pred(.2,12))
    assert len(first['track_ids'])==1
    assert np.array_equal(first['track_ids'],second['track_ids'])
    assert np.array_equal(second['track_ids'],third['track_ids'])
    assert third['frame_number']==3
    assert ByteTracker().step(pred())['frame_number']==1

def test_gt_ids_only_affect_scoring_and_canonical_mapping_is_reversible():
    tracker=ByteTracker(); frames=[tracker.step(pred()),tracker.step(pred(.9,11))]
    gt=[{'boxes':f['boxes'].copy(),'track_ids':np.array([99])} for f in frames]
    assert tracking_cost(frames,gt,ground_truth=True)==0
    gt[1]['track_ids']=np.array([100])
    assert tracking_cost(frames,gt,ground_truth=True)>0
    # Effective x/y resize can differ; preserve original box geometry.
    assert np.allclose(original_boxes([[12,14,22,34]],(.5,.4,2,4),100,100),[[20,25,40,75]])

def test_ot_sequences_do_not_mix_training_and_test_and_cal_is_contiguous():
    assert ot_split('MOT20-01',1)=='test'
    assert ot_split('MOT20-03',1)=='dev'
    assert ot_split('MOT20-02',176)=='train'
    assert ot_split('MOT20-02',193)=='cal'
    with pytest.raises(ValueError): ot_split('MOT20-05',1)

@pytest.mark.skipif(not os.environ.get('V34_TRACKEVAL'),reason='requires external pinned TrackEval checkout')
def test_bytetrack_mot_writer_and_official_metrics_agree(tmp_path):
    from tests.mot_fixture import fixture_sequence
    from adaptive_vcm.v32.tracking import plan_sequences,stage_ground_truth,write_mot,load_trackeval,evaluate_mot
    fixture_sequence(tmp_path/'data')
    plan=plan_sequences(tmp_path/'data',['MOT20-01'],3)
    out=tmp_path/'metric'; out.mkdir()
    tracker=ByteTracker()
    frames=[tracker.step(dict(boxes=np.array([[10+i,5,30+i,25]],float),scores=np.array([.9]),labels=np.array([1]))) for i in range(3)]
    write_mot(out/'trackers'/'test'/'data'/'MOT20-01.txt',frames,(1,1,0,0),80,40)
    stage_ground_truth(plan,out/'ground_truth')
    values=evaluate_mot(load_trackeval(Path(os.environ['V34_TRACKEVAL'])),plan,out,['test'])
    assert values['test']['combined']['HOTA']==100
    assert values['test']['combined']['IDF1']==100
