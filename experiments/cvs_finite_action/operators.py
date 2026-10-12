"""Explicit digital operations, no RX/TX truth and no satellite simulation."""
import math
import torch
from torch.nn import functional as F
def rms(x):return x.square().sum(1).mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
def quality(x):
 z=torch.complex(x[:,0],x[:,1]);power=torch.fft.fft(z).abs().square().clamp_min(1e-12)
 flat=power.log().mean(-1).exp()/power.mean(-1)
 a,b=z[:,16:],z[:,:-16];coh=(a*b.conj()).sum(-1).abs()/(a.abs().square().sum(-1)*b.abs().square().sum(-1)).clamp_min(1e-24).sqrt()
 return torch.stack((rms(x).squeeze(1).log(),flat,coh.clamp(0,1)),1)
def apply(x,kind,a,seed):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not torch.isfinite(x).all():raise ValueError('Finite IQ256 required')
 if len(a)!=2 or not all(math.isfinite(v) for v in a):raise ValueError('Known finite action parameters required')
 if kind=='spectral':
  if max(abs(v) for v in a)>1.5:raise ValueError('Spectral action out of range')
  f=torch.fft.fftfreq(256,device=x.device,dtype=x.dtype);db=a[0]*torch.cos(2*math.pi*f)+a[1]*torch.cos(4*math.pi*f)
  full=torch.fft.fftshift(torch.fft.ifft(torch.exp(db*math.log(10)/20))).real
  kernel=full[120:137]*torch.hann_window(17,periodic=False,device=x.device,dtype=x.dtype)
  # conv1d uses correlation; flip for the declared zero-padded linear FIR.
  y=F.conv1d(x,kernel.flip(0)[None,None].expand(2,1,17),padding=8,groups=2)
  y=y*(rms(x)/rms(y))[:,None,:]
 elif kind=='timing':
  delta=a[0]
  if abs(delta)>.25 or a[1]!=0:raise ValueError('Timing action out of range')
  shifted=F.pad(x[:,:,:-1],(1,0)) if delta>=0 else F.pad(x[:,:,1:],(0,1))
  y=(1-abs(delta))*x+abs(delta)*shifted;y=y*(rms(x)/rms(y))[:,None,:]
 elif kind=='noise':
  if not 0<=a[0]<=.1 or a[1]!=0:raise ValueError('Noise action out of range')
  gen=torch.Generator(device=x.device).manual_seed(seed)
  noise=torch.randn(x.shape,generator=gen,device=x.device,dtype=x.dtype)
  y=x+noise*(a[0]*rms(x)/math.sqrt(2))[:,None,:]
 else:raise ValueError('Unknown finite action')
 if not torch.isfinite(y).all():raise ValueError('Nonfinite operation')
 return y
