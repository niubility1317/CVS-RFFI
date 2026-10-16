"""Invertible received-IQ scalar gain,no phase/window/noise/nonlinear change."""
import math,torch
from . import design as d
def apply(x,step):
 if isinstance(step,bool) or not isinstance(step,int) or not 1<=step<=4440 or x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):raise ValueError('Finite IQ256 and registered step1..4440 required')
 g=torch.Generator(device='cpu').manual_seed(d.GAIN_SEED+step);values=.9+.2*torch.rand((len(x),1,1),generator=g);values=values.to(device=x.device,dtype=x.dtype);out=x*values
 before=x.square().mean((1,2));expected=before*values.flatten().square();after=out.square().mean((1,2));error=float(((after-expected).abs()/expected.clamp_min(1e-20)).max())
 if not math.isfinite(error) or error>1e-6 or not bool(torch.isfinite(out).all()):raise ValueError('Gain power/finite contract failed')
 return out,values.flatten().tolist(),error
