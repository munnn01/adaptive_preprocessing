"""Independent frozen heads see only a received packet's reconstructed RGB."""
import hashlib
import time
import numpy as np
import torch
from ..preprocessing import validate_clip
from .tasks import frame_indices


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def timed(fn):
    sync(); before=time.perf_counter(); value=fn(); sync()
    return value,time.perf_counter()-before


def finite_detection(d):
    return all(k in d and np.isfinite(np.asarray(d[k])).all() for k in ('boxes','scores','labels'))


class SharedReceiver:
    def __init__(self,action,detector):
        self.actions=action if isinstance(action,dict) else {action.name:action}
        self.detector=detector; self._cache={}
        for h in [*self.actions.values(),detector]: h.model.eval().requires_grad_(False)

    def predict(self,packet,*,tasks,indices=None):
        if not tasks or len(tasks)!=len(set(tasks)) or not set(tasks)<={'ar','od'}: raise ValueError('unique ar/od tasks required')
        rgb=packet.decoded; validate_clip(rgb)
        if 'ar' in tasks and len(rgb)==1: raise ValueError('single image cannot become an AR clip')
        indices=frame_indices(len(rgb)) if indices is None else indices
        if not isinstance(indices,tuple) or not indices or len(set(indices))!=len(indices) or any(type(i)!=int or not 0<=i<len(rgb) for i in indices): raise ValueError('invalid frame scope')
        digest=hashlib.sha256(rgb.tobytes()).hexdigest(); key=(rgb.shape,rgb.dtype.str,digest,tasks,indices)
        if self._cache.get('key')!=key: self._cache={'key':key}
        sync(); start=time.perf_counter()
        result=dict(packet_sha256=packet.packet_sha256,rgb_sha256=digest,coded_bytes=packet.coded_bytes,
            source_shape=list(rgb.shape),indices=list(indices),ar_accuracy=None,od_map=None,latency_seconds={},cached='predictions' in self._cache)
        if result['cached']:
            result.update(self._cache['predictions']); result['latency_seconds']={k:0. for k in self._cache['latency_keys']}
        else:
            preds={}
            if 'ar' in tasks:
                preds['ar_backbones']={}
                for name,h in self.actions.items():
                    p,secs=timed(lambda:h.probabilities(rgb)); p=np.asarray(p,float)
                    if p.ndim!=1 or len(p)<2 or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-5): raise ValueError('invalid received action probabilities')
                    preds['ar_backbones'][name]=p.tolist(); result['latency_seconds']['ar/'+name]=secs
                preds['ar']=preds['ar_backbones'].get('swin3d_t',next(iter(preds['ar_backbones'].values())))
                preds['ar_packet_sha256']=packet.packet_sha256
            if 'od' in tasks:
                detections,secs=timed(lambda:{i:self.detector.predict(rgb[i:i+1]) for i in indices})
                if any(not finite_detection(d) for d in detections.values()): raise ValueError('nonfinite received detections')
                preds['od']=detections; preds['od_packet_sha256']=packet.packet_sha256; result['latency_seconds']['od']=secs
            self._cache.update(predictions=preds,latency_keys=list(result['latency_seconds'])); result.update(preds)
        # Prediction reuse never reuses a different wire packet's identity.
        for task in tasks: result[task+'_packet_sha256']=packet.packet_sha256
        sync(); result['total_prediction_seconds']=time.perf_counter()-start
        return result
