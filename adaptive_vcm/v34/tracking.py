"""Pinned ByteTrack adapter; GT used by scoring only, never track updates."""
import importlib.metadata
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..v32.tracking import iou_matrix,plan_sequences,read_sequence
from ..v33ap.sources import SourceRecord

class ByteTracker:
    def __init__(self):
        import supervision as sv
        if importlib.metadata.version('supervision')!='0.27.0': raise ValueError('requires supervision==0.27.0')
        self.sv=sv; self.frame=0
        self.tracker=sv.ByteTrack(track_activation_threshold=.25,lost_track_buffer=30,
            minimum_matching_threshold=.8,frame_rate=25,minimum_consecutive_frames=1)
    def step(self,prediction):
        p=prediction; boxes=np.asarray(p['boxes'],float).reshape(-1,4); scores=np.asarray(p['scores'],float); labels=np.asarray(p['labels'],int)
        if scores.shape!=(len(boxes),) or labels.shape!=scores.shape or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('invalid tracking detections')
        keep=(labels==1)&(scores>.1)&(boxes[:,2:]>boxes[:,:2]).all(1)
        d=self.sv.Detections(xyxy=boxes[keep],confidence=scores[keep],class_id=labels[keep])
        tracked=self.tracker.update_with_detections(d); self.frame+=1
        return dict(boxes=tracked.xyxy.copy(),scores=tracked.confidence.copy(),labels=tracked.class_id.copy(),
                    track_ids=tracked.tracker_id.copy(),frame_number=self.frame)

def track_frames(predictions,tracker=None):
    tracker=tracker or ByteTracker()
    return [tracker.step(p) for p in predictions]

def tracking_cost(frames,reference,ground_truth=False):
    """Local ID/box surrogate, explicitly not official HOTA or IDF1."""
    if len(frames)!=len(reference): raise ValueError('tracking frame count mismatch')
    misses=fp=switches=count=0; links={}; reverse={}
    for cur,ref in zip(frames,reference):
        a=np.asarray(ref['boxes']).reshape(-1,4); b=np.asarray(cur['boxes']).reshape(-1,4)
        overlap=iou_matrix(a,b); pairs=[]
        if overlap.size:
            ii,jj=linear_sum_assignment(np.where(overlap>=.5,1-overlap,1e6))
            pairs=[(int(i),int(j)) for i,j in zip(ii,jj) if overlap[i,j]>=.5]
        count+=len(a); misses+=len(a)-len(pairs); fp+=len(b)-len(pairs)
        for i,j in pairs:
            key=int(ref['track_ids'][i]); value=int(cur['track_ids'][j])
            if key in links and links[key]!=value: switches+=1
            if value in reverse and reverse[value]!=key: switches+=1
            links[key]=value
            reverse[value]=key
    return (misses+fp+2*switches)/max(count,1)

def ot_split(name,start):
    if name=='MOT20-01': return 'test'
    if name=='MOT20-03': return 'dev'
    if name=='MOT20-02': return 'train' if start<=192 else 'cal'
    raise ValueError('unapproved OT sequence')

def gt_window(path,start,stop,geometry):
    rows=np.loadtxt(path,delimiter=',',ndmin=2); sx,sy,left,top=geometry; frames=[]
    for number in range(start,stop):
        # MOTChallenge pedestrian positives only for the training surrogate.
        valid=rows[(rows[:,0]==number)&(rows[:,6]>0)&(rows[:,7]==1)]
        boxes=valid[:,2:6].copy(); boxes[:,2:]+=boxes[:,:2]
        boxes[:,[0,2]]=boxes[:,[0,2]]*sx+left; boxes[:,[1,3]]=boxes[:,[1,3]]*sy+top
        frames.append(dict(boxes=boxes,track_ids=valid[:,1].astype(int)))
    return frames

def tracking_sources(root,cfg):
    plan=plan_sequences(root,['MOT20-02','MOT20-03','MOT20-01'],cfg['ot_frames'])
    groups={s:[] for s in ('train','cal','dev','test')}
    for item in plan:
        if item['frames']!=cfg['ot_frames'] or item['frames']%cfg['frames']: raise ValueError('complete MOT windows required')
        rgb,g=read_sequence(item,cfg['ot_size'])
        for offset in range(0,len(rgb),cfg['frames']):
            start=offset+1; split=ot_split(item['name'],start)
            source=dict(id=f"{item['name']}/{start:06}",sequence=item['name'],start=start,
                width=item['width'],height=item['height'],path=str(item['path']))
            r=SourceRecord('ot/'+source['id'],'ot',source,rgb[offset:offset+cfg['frames']].copy(),g,
                dict(kind='contiguous-MOT20',frame_indices=list(range(start,start+cfg['frames'])),repeated_frames=0))
            r.split=split
            # DEV/TEST GT is deliberately not read until after selected streams.
            if split in ('train','cal'): r.targets=gt_window(item['path']/'gt/gt.txt',start,start+cfg['frames'],g)
            groups[split].append(r)
    return groups,plan
