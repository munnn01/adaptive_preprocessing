"""Frozen task measurements with separate GT and source-consistency provenance."""
import numpy as np
import time
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ..v32vc.train import tensor
from ..v33ap.tasks import TaskHeads,ar_forward,od_forward,frame_indices,prediction,distance
from ..v32.tracking import iou_matrix

def detection_distance(reference,current):
    values=[]
    for ref,cur in zip(reference,current):
        a=np.where(np.asarray(ref['scores'])>=.25)[0]; b=np.where(np.asarray(cur['scores'])>=.1)[0]
        if not len(a): values.append(float(np.mean(np.asarray(cur['scores'])[b])) if len(b) else 0.); continue
        if not len(b): values.append(1.); continue
        overlap=iou_matrix(ref['boxes'][a],cur['boxes'][b])
        overlap*=np.asarray(ref['labels'])[a,None]==np.asarray(cur['labels'])[None,b]
        ii,jj=linear_sum_assignment(1-overlap)
        penalty=np.ones(len(a)); penalty[ii]=1-overlap[ii,jj]
        values.append(float(penalty.mean()))
    return float(np.mean(values))

class Heads(TaskHeads):
    def __init__(self,device,experiment,cfg=None):
        self.cfg=cfg or {}
        detector=None
        if self.cfg.get('detector')=='resnet50':
            from ..analyzers import DetectionAnalyzer
            detector=DetectionAnalyzer('resnet50',device)
        super().__init__(device,detector=detector); self.experiment=experiment
        self.configure_detector()

    def configure_detector(self):
        # Lock detector transform as well as weights; avoid expensive upsampling
        # of every low-resolution AR frame to 800px during candidate search.
        self.detector.model.transform.min_size=(320,)
        self.detector.model.transform.max_size=640
        if getattr(self,'cfg',{}).get('detector_detections'):
            self.detector.model.roi_heads.detections_per_img=self.cfg['detector_detections']

    @contextmanager
    def detector_resolution(self,rgb):
        t=self.detector.model.transform; old=(t.min_size,t.max_size)
        size=getattr(self,'cfg',{}).get('ot_detector_size')
        try:
            if size and min(rgb.shape[1:3])>=size: t.min_size=(size,); t.max_size=size
            yield
        finally: t.min_size,t.max_size=old

    def hashes(self):
        value=super().hashes(); value['od']['transform']={'min_size':[320],'max_size':640}
        if self.cfg.get('ot_detector_size'):
            value['od'].update(ot_transform={'min_size':[self.cfg['ot_detector_size']],'max_size':self.cfg['ot_detector_size']},
                               detections_per_img=self.cfg['detector_detections'])
        return value

    @torch.no_grad()
    def detections(self,rgb):
        x=tensor(rgb,self.device)
        result=[]
        batch=getattr(self,'cfg',{}).get('detector_batch',4)
        with self.detector_resolution(rgb):
            for start in range(0,len(rgb),batch):
                values=self.detector.model([x[0,:,i] for i in range(start,min(start+batch,len(rgb)))])
                result.extend(prediction(v) for v in values)
        return result

    @torch.no_grad()
    def source(self,r):
        start=time.perf_counter()
        r.source_predictions={'od':self.detections(r.rgb)}; r.source_features={}
        r.source_detector_seconds=time.perf_counter()-start
        teacher_start=time.perf_counter()
        if len(r.rgb)>1:
            logits,f=ar_forward(self.action,tensor(r.rgb,self.device))
            r.source_predictions['ar']=logits.softmax(-1)[0].cpu().numpy(); r.source_features['ar']=f.cpu()
        r.mask=[len(r.rgb)>1,self.experiment=='ar_od',self.experiment=='ar_ot' and len(r.rgb)>1]
        r.provenance=[('K400 label + source KL/feature' if r.task=='ar' else 'source KL/feature only') if r.mask[0] else None,
            ('COCO boxes + source feature' if r.task=='od' else 'source detection/feature consistency only') if r.mask[1] else None,
            ('MOT20 GT boxes/IDs, local window surrogate' if r.task=='ot' else 'source tracks consistency only') if r.mask[2] else None]
        if r.mask[1]:
            _,f=od_forward(self.detector,tensor(r.rgb,self.device),frame_indices(len(r.rgb)))
            r.source_features['od']=f.cpu()
        if r.mask[2]:
            from .tracking import track_frames
            r.source_predictions['ot']=track_frames(r.source_predictions['od'])
        r.diagnostic_teacher_seconds=time.perf_counter()-teacher_start

    @torch.no_grad()
    def costs(self,rgb,r,source_only=False):
        x=tensor(rgb,self.device); result=[None]*3
        if r.mask[0]:
            logits,f=ar_forward(self.action,x)
            target=torch.as_tensor(r.source_predictions['ar'],device=self.device)[None]
            value=.25*F.kl_div(logits.log_softmax(-1),target,reduction='batchmean')+.1*distance(f,r.source_features['ar'])
            if r.task=='ar' and not source_only: value=value+F.cross_entropy(logits,torch.tensor([r.item['label']],device=self.device))
            result[0]=float(value)
        if r.mask[1]:
            indices=frame_indices(len(rgb))
            if r.task=='od':
                targets=[{k:v.to(self.device) for k,v in r.targets.items()}]
                # Faster R-CNN's GT loss samples proposals even with frozen
                # weights. Use common random numbers across all candidates.
                devices=[torch.device(self.device).index or 0] if str(self.device).startswith('cuda') else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(303401)
                    loss,f=od_forward(self.detector,x,indices,targets)
                value=float(sum(loss.values()))
            else:
                preds,f=od_forward(self.detector,x,indices)
                value=detection_distance([r.source_predictions['od'][i] for i in indices],[prediction(v) for v in preds])
            result[1]=value+.1*float(distance(f,r.source_features['od']))
        if r.mask[2]:
            from .tracking import track_frames, tracking_cost
            frames=track_frames(self.detections(rgb))
            use_gt=r.task=='ot' and not source_only
            result[2]=tracking_cost(frames,r.targets if use_gt else r.source_predictions['ot'],ground_truth=use_gt)
        return result
