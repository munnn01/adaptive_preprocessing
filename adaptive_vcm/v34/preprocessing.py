"""Causal support masks and bounded spatial frequency edits, never spatial warps."""
import cv2
import numpy as np
from ..preprocessing import validate_clip
from ..motion_support import _flow_pair

def region_maps(rgb, predictions, temporal=True):
    validate_clip(rgb)
    t,h,w,_ = rgb.shape
    if len(predictions) != t: raise ValueError('one source prediction per frame required')
    history = np.zeros((h,w),np.float32); previous = None; result=[]
    for frame, pred in zip(rgb,predictions):
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        motion=np.zeros((h,w),np.float32); cut=False
        if previous is not None and temporal:
            _,motion,cut,_ = _flow_pair(previous,gray)
        if cut or not temporal: history.fill(0)
        core=np.zeros((h,w),np.float32)
        boxes=np.asarray(pred['boxes'],np.float32).reshape(-1,4)
        scores=np.asarray(pred['scores'],np.float32)
        if scores.shape!=(len(boxes),) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('invalid source detections')
        for box,score in zip(boxes,scores):
            if score<.25: continue
            x1,y1,x2,y2=box
            if x2<=x1 or y2<=y1: continue
            x1,y1=np.maximum(0,np.floor([x1,y1])).astype(int)
            x2,y2=np.minimum([w,h],np.ceil([x2,y2])).astype(int)
            core[y1:y2,x1:x2]=1
        # Decaying source history: no future union; a cut erases prior objects.
        history=np.maximum(core,history*.7)
        radius=max(3,round(min(h,w)*.06))|1
        context=np.maximum(cv2.dilate(history,np.ones((radius,radius),np.uint8)),motion)
        context=(1-history)*np.clip(context,0,1)
        result.append(np.stack((history,context,1-history-context),-1))
        previous=gray
    return np.stack(result).astype(np.float32)

def filter_clip(rgb,maps,action):
    validate_clip(rgb)
    maps=np.asarray(maps,np.float32); a=np.asarray(action,np.float32)
    if (maps.shape!=rgb.shape or not np.isfinite(maps).all() or maps.min()<0
        or not np.allclose(maps.sum(-1),1,atol=1e-5) or a.shape!=(6,)
        or not np.isfinite(a).all() or a.min()<-.75 or a.max()>.25):
        raise ValueError('invalid V34 maps/action')
    if not a.any(): return rgb.copy()
    coeff=np.einsum('thwr,rb->thwb',maps,a.reshape(3,2))
    result=[]
    for i,frame in enumerate(rgb):
        x=frame.astype(np.float32)
        g1=cv2.GaussianBlur(x,(0,0),.8,borderType=cv2.BORDER_REPLICATE)
        g2=cv2.GaussianBlur(x,(0,0),1.6,borderType=cv2.BORDER_REPLICATE)
        y=x+coeff[i,:,:,0,None]*(x-g1)+coeff[i,:,:,1,None]*(g1-g2)
        result.append(np.rint(np.clip(y,0,255)).astype(np.uint8))
    return np.stack(result)
