"""Independent Video Swin-T AR evaluation on received, inverse-mapped RGB."""
import numpy as np
import torch
from ..analyzers import ActionAnalyzer


class SwinAnalyzer(ActionAnalyzer):
    def __init__(self,device):
        from torchvision.models import video as V
        weights=V.Swin3D_T_Weights.KINETICS400_V1
        self.model=V.swin3d_t(weights=weights).eval().to(device).requires_grad_(False)
        self.transform=weights.transforms(); self.device=device; self.name='swin3d_t'
        self.categories=weights.meta['categories']

    def logits(self,x):
        return self.model(self.transform(x.permute(0,2,1,3,4)))
