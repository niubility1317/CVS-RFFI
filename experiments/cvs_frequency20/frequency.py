"""Private packet-frequency draw,no native RNG advancement or IQ renormalizing."""
import math,torch
from . import design as d
def apply(x,step):
 if isinstance(step,bool) or not isinstance(step,int) or not 1<=step<=4440 or x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):raise ValueError('Finite IQ256 and registered step1..4440 required')
 g=torch.Generator(device='cpu').manual_seed(d.FREQUENCY_SEED+step);values=((2.*torch.rand((len(x),1),generator=g)-1.)*d.FREQUENCY_MAX).to(device=x.device,dtype=x.dtype)
 n=torch.arange(256,device=x.device,dtype=x.dtype);phase=(2.*math.pi*values)*n;c,s=phase.cos(),phase.sin();r,i=x.unbind(1);out=torch.stack((r*c-i*s,r*s+i*c),1)
 before=x.square().mean((1,2));after=out.square().mean((1,2));error=float(((after-before).abs()/before.clamp_min(1e-20)).max())
 if not math.isfinite(error) or error>1e-6 or not bool(torch.isfinite(out).all()):raise ValueError('Frequency power/finite contract failed')
 return out,values.flatten().tolist(),error
