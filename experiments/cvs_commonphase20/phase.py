"""Constant unit rotation; generator never advances the native training RNG."""
import torch
from . import design as d
def apply(raw,step):
 if type(step) is not int or not 1<=step<=4440 or raw.ndim!=3 or raw.shape[1:]!=(2,256) or not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):raise ValueError('Finite real L step IQ required')
 angles=(torch.rand(len(raw),generator=torch.Generator(device='cpu').manual_seed(d.PHASE_SEED+step))*2-1)*.37
 a=angles.to(device=raw.device,dtype=raw.dtype);c=a.cos()[:,None];s=a.sin()[:,None]
 x=torch.stack((c*raw[:,0]-s*raw[:,1],s*raw[:,0]+c*raw[:,1]),1)
 p=raw.square().sum(1).mean(1);q=x.square().sum(1).mean(1);error=float(((p-q).abs()/p.clamp_min(1e-12)).max())
 if not bool(torch.isfinite(x).all()) or error>1e-6:raise ValueError('Unit phase finite/power check failed')
 return x,angles.tolist(),error
