"""Fixed YUV420 block-DCT surrogate. Entropy estimates are never measured bitrate."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .model import conditions


def dct_basis(n):
    k = torch.arange(n).float()[:,None]; m = torch.arange(n).float()[None]
    d = torch.cos(math.pi*(2*m+1)*k/(2*n))
    d[0] *= math.sqrt(1/n); d[1:] *= math.sqrt(2/n)
    return d


def color_planes(rgb):
    r,g,b = rgb[:,0:1],rgb[:,1:2],rgb[:,2:3]
    y = .299*r+.587*g+.114*b
    cb = .5-.168736*r-.331264*g+.5*b
    cr = .5+.5*r-.418688*g-.081312*b
    return y,F.avg_pool2d(cb,2),F.avg_pool2d(cr,2)


def color_rgb(planes):
    y,cb,cr = planes
    cb = F.interpolate(cb,size=y.shape[-2:],mode='bilinear',align_corners=False)-.5
    cr = F.interpolate(cr,size=y.shape[-2:],mode='bilinear',align_corners=False)-.5
    return torch.cat((y+1.402*cr,y-.344136*cb-.714136*cr,y+1.772*cb),1)


class VirtualCodec(nn.Module):
    def __init__(self,calibration=None):
        super().__init__()
        self.calibration = calibration or {}
        self.register_buffer('basis',dct_basis(8))

    def _blocks(self,x):
        n,c,h,w = x.shape
        padded = F.pad(x,(0,(-w)%8,0,(-h)%8),mode='replicate')
        ph,pw = padded.shape[-2:]
        cells = padded.reshape(n,c,ph//8,8,pw//8,8)
        d = self.basis.to(x)
        coeff = torch.einsum('ku,lv,nciujv->ncikjl',d,d,cells)
        return coeff,(h,w,ph,pw)

    def _inverse(self,coeff,shape):
        h,w,ph,pw = shape; d = self.basis.to(coeff)
        cells = torch.einsum('ku,lv,ncikjl->nciujv',d,d,coeff)
        return cells.reshape(coeff.shape[0],coeff.shape[1],ph,pw)[...,:h,:w]

    def forward(self,video,qp,codec,cuts=None):
        qp,codec,cuts = conditions(video,qp,codec,cuts)
        if video.shape[-1]%2 or video.shape[-2]%2:
            raise ValueError('YUV420 requires even geometry')
        frames,bits = [],video.new_zeros(())
        for sample in range(video.shape[0]):
            group = ('h265' if codec[sample]==1 else 'h264')+'/'+str(int(qp[sample]))
            entry = self.calibration.get(group,{})
            step = float(entry.get('step',.015*2**((float(qp[sample])-30)/6)))
            rate_scale,rate_offset = float(entry.get('rate_scale',1.)),float(entry.get('rate_offset',0.))
            if not all(math.isfinite(v) for v in (step,rate_scale,rate_offset)) or min(step,rate_scale)<=0 or rate_offset<0:
                raise ValueError('invalid frozen virtual calibration')
            prior = None; reconstructed = []; sample_bits = video.new_zeros(())
            for time in range(video.shape[2]):
                planes = color_planes(video[sample:sample+1,:,time])
                decoded = []
                for index,plane in enumerate(planes):
                    pred = torch.full_like(plane,.5) if prior is None or cuts[sample,time] else prior[index]
                    coeff,shape = self._blocks(plane-pred)
                    qstep = step*(1.5 if index else 1.)
                    latent = coeff/qstep
                    quantized = latent+torch.empty_like(latent).uniform_(-.5,.5) if self.training else latent.round()
                    # One Gaussian per frequency, with stable variance for flat/small planes.
                    sigma = latent.square().mean((0,1,2,4),keepdim=True).add(.25).sqrt()
                    upper = .5*(1+torch.erf((quantized+.5)/(sigma*math.sqrt(2))))
                    lower = .5*(1+torch.erf((quantized-.5)/(sigma*math.sqrt(2))))
                    sample_bits = sample_bits-(upper-lower).clamp_min(1e-9).log2().sum()
                    decoded.append((pred+self._inverse(quantized*qstep,shape)).clamp(0,1))
                prior = decoded
                reconstructed.append(color_rgb(decoded).clamp(0,1))
            frames.append(torch.stack(reconstructed,2))
            pixels = video.shape[2]*video.shape[3]*video.shape[4]
            bits = bits+rate_scale*sample_bits+rate_offset*pixels
        return torch.cat(frames,0),bits/(video.shape[0]*video.shape[2]*video.shape[3]*video.shape[4])
