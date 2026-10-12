"""Differentiable resampling of fixed V32 grids; grids themselves have no gradient."""
import torch
import torch.nn.functional as F
from ..v32.roi import validate_grids


def _area_axis(source,target,reference):
    lo = torch.arange(target,device=reference.device,dtype=reference.dtype)*source/target
    hi = (torch.arange(target,device=reference.device,dtype=reference.dtype)+1)*source/target
    pixels = torch.arange(source,device=reference.device,dtype=reference.dtype)
    weights = (torch.minimum(hi[:,None],pixels[None]+1)-torch.maximum(lo[:,None],pixels[None])).clamp_min(0)
    # Float32 interval endpoints can drift: keep the weights a convex average.
    return weights/weights.sum(1,keepdim=True)


def _resize(x,size):
    h,w = x.shape[-2:]; dh,dw = size
    if (h,w)==size: return x
    if dh<=h and dw<=w:
        rows,cols = _area_axis(h,dh,x),_area_axis(w,dw,x)
        pixels = torch.einsum('ih,nchw,jw->ncij',rows,x,cols)
    else:
        pixels = F.interpolate(x,size=size,mode='bilinear',align_corners=False)
    # Accumulation can still overshoot by one ULP at white RGB pixels.
    return pixels.clamp(0,1)


def transform(video,grids,*,inverse=False):
    if video.ndim!=5 or video.shape[1]!=3 or not isinstance(grids,tuple) or not grids:
        raise ValueError('B3THW tensor and nonempty fixed grids required')
    g = grids[0]; t = video.shape[2]
    source = (t,g.y_source[-1],g.x_source[-1],3)
    coded = (t,g.y_coded[-1],g.x_coded[-1],3)
    validate_grids(grids,source,coded)
    if tuple(video.shape[2:]) != ((coded if inverse else source)[0:3]):
        raise ValueError('grid input shape mismatch')
    segments = []
    b = video.shape[0]
    for grid in grids:
        sx,dx = (grid.x_coded,grid.x_source) if inverse else (grid.x_source,grid.x_coded)
        sy,dy = (grid.y_coded,grid.y_source) if inverse else (grid.y_source,grid.y_coded)
        rows = []
        for j in range(len(sy)-1):
            cells = []
            for i in range(len(sx)-1):
                patch = video[:,:,grid.start:grid.stop,sy[j]:sy[j+1],sx[i]:sx[i+1]]
                frames = patch.permute(0,2,1,3,4).reshape(-1,3,patch.shape[-2],patch.shape[-1])
                dh,dw = dy[j+1]-dy[j],dx[i+1]-dx[i]
                pixels = _resize(frames,(dh,dw)).reshape(b,grid.stop-grid.start,3,dh,dw).permute(0,2,1,3,4)
                cells.append(pixels)
            rows.append(torch.cat(cells,-1))
        segments.append(torch.cat(rows,-2))
    return torch.cat(segments,2)
