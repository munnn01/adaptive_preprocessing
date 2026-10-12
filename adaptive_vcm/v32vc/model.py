"""One bounded RGB head; no gate, expert bank or decoder network."""
import torch
from torch import nn
import torch.nn.functional as F


def conditions(video, qp, codec, cuts):
    if (video.ndim != 5 or video.shape[1] != 3 or not video.is_floating_point()
            or min(video.shape) < 1 or not torch.isfinite(video).all()
            or (video < 0).any() or (video > 1).any()):
        raise ValueError('finite B3THW RGB in [0,1] required')
    b,_,t,_,_ = video.shape
    qp = torch.as_tensor(qp,device=video.device,dtype=video.dtype).reshape(-1)
    codec = torch.as_tensor(codec,device=video.device,dtype=video.dtype).reshape(-1)
    if qp.numel()!=b or not torch.isfinite(qp).all() or (qp<0).any() or (qp>51).any():
        raise ValueError('one valid QP per sample required')
    if codec.numel()!=b or not ((codec==0)|(codec==1)).all():
        raise ValueError('one codec ID 0/1 per sample required')
    if cuts is None:
        cuts = torch.zeros(b,t,dtype=torch.bool,device=video.device); cuts[:,0] = True
    if cuts.shape!=(b,t) or cuts.dtype!=torch.bool or not cuts[:,0].all():
        raise ValueError('BT scene cuts must start at first frame')
    return qp,codec,cuts.to(video.device)


class ResidualPreprocessor(nn.Module):
    def __init__(self, task, width=12):
        super().__init__()
        if task not in ('ar','od') or type(width)!=int or width<4:
            raise ValueError('invalid task/width')
        self.task,self.width = task,width
        self.trunk = nn.Sequential(nn.Conv2d(7,width,3,padding=1),nn.SiLU(),
                                   nn.Conv2d(width,width,3,padding=1),nn.SiLU())
        self.temporal = nn.Conv3d(width,width,(3,1,1),groups=width) if task=='ar' else None
        self.head = nn.Conv2d(width,3,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self,video,qp,codec,protection,motion,cuts):
        qp,codec,cuts = conditions(video,qp,codec,cuts)
        b,_,t,h,w = video.shape
        if self.task=='od' and t!=1:
            raise ValueError('OD requires one frame')
        for support in (protection,motion):
            if (support.shape!=(b,1,t,h,w) or not torch.isfinite(support).all()
                    or (support<0).any() or (support>1).any()):
                raise ValueError('invalid protection/motion')
        p,m = protection.to(video),motion.to(video)
        extra = torch.cat((p,m,(qp/51).view(b,1,1,1,1).expand(b,1,t,h,w),
                           codec.view(b,1,1,1,1).expand(b,1,t,h,w)),1)
        features = torch.cat((video,extra),1).permute(0,2,1,3,4).reshape(b*t,7,h,w)
        features = self.trunk(features).reshape(b,t,self.width,h,w).permute(0,2,1,3,4)
        if self.temporal is not None:
            batches = []
            for index in range(b):
                starts = torch.nonzero(cuts[index],as_tuple=False).flatten().tolist()+[t]
                segments = [self.temporal(F.pad(features[index:index+1,:,a:z],(0,0,0,0,2,0)))
                            for a,z in zip(starts,starts[1:])]
                batches.append(torch.cat(segments,2))
            features = features+.1*torch.cat(batches,0)
        frames = features.permute(0,2,1,3,4).reshape(b*t,self.width,h,w)
        residual = .1*self.head(frames).tanh().reshape(b,t,3,h,w).permute(0,2,1,3,4)
        result = (video+(1-p)*(1-.75*m)*residual).clamp(0,1)
        return torch.where(p==1,video,result)
