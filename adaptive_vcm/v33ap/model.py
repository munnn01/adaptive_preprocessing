"""A single bounded RGB head, with a causal temporal branch for clips."""
import torch
from torch import nn
import torch.nn.functional as F
from ..v32vc.model import conditions


class JointPreprocessor(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        if type(width)!=int or width<4: raise ValueError('invalid width')
        self.width = width
        self.trunk = nn.Sequential(nn.Conv2d(7,width,3,padding=1),nn.SiLU(),
                                   nn.Conv2d(width,width,3,padding=1),nn.SiLU())
        self.temporal = nn.Conv3d(width,width,(3,1,1),groups=width)
        self.head = nn.Conv2d(width,3,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, video, qp, codec, protection, motion, cuts):
        qp,codec,cuts = conditions(video,qp,codec,cuts)
        b,_,t,h,w = video.shape
        for support in (protection,motion):
            if (support.shape!=(b,1,t,h,w) or not torch.isfinite(support).all()
                or (support<0).any() or (support>1).any()): raise ValueError('invalid support')
        p,m = protection.to(video),motion.to(video)
        extra = torch.cat((p,m,(qp/51).view(b,1,1,1,1).expand(b,1,t,h,w),
                           codec.view(b,1,1,1,1).expand(b,1,t,h,w)),1)
        f = self.trunk(torch.cat((video,extra),1).permute(0,2,1,3,4).reshape(b*t,7,h,w))
        f = f.reshape(b,t,self.width,h,w).permute(0,2,1,3,4)
        if t>1:
            pieces=[]
            for index in range(b):
                starts=torch.nonzero(cuts[index],as_tuple=False).flatten().tolist()+[t]
                pieces.append(torch.cat([self.temporal(F.pad(f[index:index+1,:,a:z],(0,0,0,0,2,0)))
                                         for a,z in zip(starts,starts[1:])],2))
            f=f+.1*torch.cat(pieces,0)
        delta=.1*self.head(f.permute(0,2,1,3,4).reshape(b*t,self.width,h,w)).tanh()
        delta=delta.reshape(b,t,3,h,w).permute(0,2,1,3,4)
        y=(video+(1-p)*(1-.75*m)*delta).clamp(0,1)
        return torch.where(p==1,video,y)
