"""Training-only bounded Gaussian attenuation of one fixed STF harmonic pair."""
import math,random
import torch
HALF=128;N=512;SIGMA=1/256;DEPTH=.5;BOUND=.1

def choices(count,seed):
 r=random.Random(int(seed));return [r.randrange(1,7) for _ in range(count)]
def _filter(x,k,pad):
 n=256;ix=torch.arange(-pad,n+pad,device=x.device);j=ix.remainder(2*(n-1));ix=torch.where(j<n,j,2*(n-1)-j).long();z=torch.complex(x[:,0].double(),x[:,1].double());v=z[:,ix];f=torch.fft.fftfreq(v.shape[-1],device=x.device,dtype=torch.float64)[None,:];center=torch.as_tensor(k,device=x.device,dtype=torch.float64)[:,None]/20
 def gaussian(c):
  delta=(f-c+.5).remainder(1)-.5;return torch.exp(-.5*(delta/SIGMA).square())
 h=1-DEPTH*(gaussian(center)+gaussian(-center));y=torch.fft.ifft(torch.fft.fft(v,dim=-1)*h,dim=-1)[:,pad:pad+256];return torch.stack((y.real,y.imag),1)

def apply(x,k,reference=False):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or len(k)!=len(x) or any(type(i) is not int or i not in range(1,7) for i in k):raise ValueError('Finite original float32IQ256 andlabel-independent STF pair1..6')
 with torch.no_grad():
  raw=x.double();power=raw.square().sum((1,2));
  if bool((power<=0).any()):raise ValueError('Nonzero received IQ')
  uncapped=_filter(x,k,HALF);delta=uncapped-raw;relative=torch.sqrt(delta.square().sum((1,2))/power);alpha=torch.clamp(BOUND/relative.clamp_min(1e-30),max=1.);masked=(raw+alpha[:,None,None]*delta).float();gain=torch.sqrt(masked.double().square().sum((1,2))/power);gainview=(raw*gain[:,None,None]).float();actual=torch.sqrt((masked.double()-raw).square().sum((1,2))/power)
  if not bool(torch.isfinite(masked).all()) or not bool(torch.isfinite(gainview).all()) or float(actual.max())>BOUND+1e-6:raise ValueError('Finite bounded allpacket spectral observation')
  stats=dict(harmonic_pair=k,relative_IQ_RMS=actual.tolist(),cap=alpha.tolist(),matched_gain=gain.tolist(),power_match_relative_error=((gainview.double().square().sum((1,2))-masked.double().square().sum((1,2))).abs()/power).tolist(),reference_window_relative_RMS=torch.sqrt((masked.double()[:,:,80:160]-raw[:,:,80:160]).square().sum((1,2))/raw[:,:,80:160].square().sum((1,2)).clamp_min(1e-30)).tolist())
  if reference:
   long=raw+alpha[:,None,None]*(_filter(x,k,512)-raw);error=(masked.double()-long).square().sum((1,2));change=(masked.double()-raw).square().sum((1,2));stats.update(boundary_error_energy=error.tolist(),change_energy=change.tolist(),input_energy=power.tolist())
 return masked,gainview,stats
