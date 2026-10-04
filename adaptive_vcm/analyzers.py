"""Frozen task networks: encoder teachers and separately scored AR/OD evaluators."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ActionAnalyzer:
    def __init__(self, name: str, device: str):
        from torchvision.models import video as V
        table = {"r3d_18": (V.r3d_18, V.R3D_18_Weights.KINETICS400_V1),
                 "mc3_18": (V.mc3_18, V.MC3_18_Weights.KINETICS400_V1),
                 "r2plus1d_18": (V.r2plus1d_18, V.R2Plus1D_18_Weights.KINETICS400_V1)}
        constructor, weights = table[name]
        self.model = constructor(weights=weights).eval().to(device)
        self.model.requires_grad_(False)
        self.device = device
        self.name = name
        self.categories = weights.meta["categories"]
        self.mean = torch.tensor([.43216, .394666, .37645], device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([.22803, .22145, .216989], device=device).view(1, 3, 1, 1, 1)

    def tensor(self, clip):
        return torch.from_numpy(np.ascontiguousarray(clip)).to(self.device).float().permute(3, 0, 1, 2)[None] / 255

    def logits(self, x):
        b, c, t, h, w = x.shape
        if (h, w) != (112, 112):
            x = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w),
                              (112, 112), mode="bilinear", align_corners=False)
            x = x.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
        return self.model((x - self.mean) / self.std)

    @torch.no_grad()
    def probabilities(self, clip):
        return self.logits(self.tensor(clip)).softmax(-1)[0].cpu().numpy()

    def saliency(self, clip):
        """Source-predicted class attribution; ground-truth label is never passed."""
        with torch.enable_grad():
            x = self.tensor(clip).requires_grad_(True)
            logits = self.logits(x)
            objective = logits[0, logits[0].detach().argmax()]
            gradient = torch.autograd.grad(objective, x, only_inputs=True)[0]
            return (gradient * x).abs().mean(1).amax(1)[0].detach().cpu().numpy()


class DetectionAnalyzer:
    def __init__(self, name: str, device: str):
        from torchvision.models import detection as D
        table = {
            "mobilenet": (D.fasterrcnn_mobilenet_v3_large_fpn, D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1),
            "resnet50": (D.fasterrcnn_resnet50_fpn, D.FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
        }
        constructor, weights = table[name]
        self.model = constructor(weights=weights).eval().to(device)
        self.model.requires_grad_(False)
        self.device = device
        self.name = name

    @torch.no_grad()
    def predict(self, clip):
        if len(clip) != 1:
            raise ValueError("COCO OD benchmark requires a single frame")
        x = torch.from_numpy(np.ascontiguousarray(clip[0])).to(self.device).float().permute(2, 0, 1) / 255
        result = self.model([x])[0]
        return {k: v.detach().cpu().numpy() for k, v in result.items()}
