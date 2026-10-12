"""Compact spatial encoder + temporal GRU predicts finite action/risk/rate."""
import torch
from torch import nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, actions, conditions=6):
        super().__init__(); self.actions=actions; self.conditions=conditions
        self.encoder=nn.Sequential(nn.Conv2d(6,12,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(12,16,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.temporal=nn.GRU(16,24,batch_first=True)
        self.head=nn.Sequential(nn.Linear(24+conditions,32),nn.SiLU(),nn.Linear(32,actions*5))
    def forward(self,x,maps,condition):
        if x.ndim!=5 or maps.shape!=x.shape or condition.shape!=(len(x),self.conditions):
            raise ValueError('controller condition/input shape mismatch')
        b,t,c,h,w=x.shape
        image=torch.cat((x,maps),2).reshape(b*t,6,h,w)
        image=F.interpolate(image,(48,48),mode='bilinear',align_corners=False)
        z=self.encoder(image).reshape(b,t,16)
        _,hidden=self.temporal(z)
        raw=self.head(torch.cat((hidden[-1],condition),1)).reshape(b,self.actions,5)
        return dict(logits=raw[:,:,0],risk=raw[:,:,1:4],rate=F.softplus(raw[:,:,4])+.05)
