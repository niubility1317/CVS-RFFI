"""Same fixed primary observation,re-estimated only on four legal fit RX."""
import math,torch
from . import design as d
def basis(n=32):
 pos=(torch.arange(n,dtype=torch.float64)+.5)*32/n;k=torch.arange(1,9,dtype=torch.float64)
 return torch.cos(math.pi*pos[:,None]*k[None,:]/32)*math.sqrt(2/32)
def observe(x):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Finite IQ256 required')
 z=torch.complex(x[:,0].double(),x[:,1].double());p=torch.fft.fftshift(torch.fft.fft(z),dim=-1).abs().square().reshape(-1,32,8).mean(-1);power=p.mean(-1,keepdim=True)
 if bool((power<=0).any()):raise ValueError('Nonzero packet power required')
 v=(p/power).clamp_min(1e-6).log();return v-v.mean(-1,keepdim=True)
def grid(v,y,rx,day):
 if v.shape!=(5040,8) or set(y.tolist())!=set(range(6)) or set(rx.tolist())!=set(d.FIT) or set(day.tolist())!={1,2,3}:raise ValueError('Fit-only source spectrum dimensions differ')
 m=torch.empty(6,3,4,8,dtype=torch.float64)
 for t in range(6):
  for dd in (1,2,3):
   for ri,r in enumerate(d.FIT):
    cell=v[(y==t)&(day==dd)&(rx==r)]
    if len(cell)!=70:raise ValueError('Incomplete source L cell')
    m[t,dd-1,ri]=cell.mean(0)
 return m
def fit(m,t,dd):
 if m.shape!=(6,3,4,8) or t not in range(6) or dd not in range(3):raise ValueError('Fixed fourRX folds required')
 v=m[torch.arange(6)!=t][:,torch.arange(3)!=dd].mean((0,1));return v-v.mean(0,keepdim=True)
def audit(m):
 folds=[];sse=zero=control=0.;mask=torch.tensor(d.CONTROL,dtype=torch.float64)
 for t in range(6):
  for dd in range(3):
   pred=fit(m,t,dd);actual=m[t,dd]-m[t,dd].mean(0,keepdim=True);p=float((pred-actual).square().sum());z=float(actual.square().sum());c=float((pred*mask-actual).square().sum());sse+=p;zero+=z;control+=c
   folds.append(dict(TX=t,day=dd+1,fit_cells=40,test_cells=4,prediction_SSE=p,zero_SSE=z,control_SSE=c))
 e=m.mean((0,1));e=e-e.mean(0,keepdim=True)
 return dict(folds=folds,prediction_SSE=sse,zero_SSE=zero,control_SSE=control,relative_MSE=sse/zero if zero>1e-20 else None,better_folds=sum(f['prediction_SSE']<f['zero_SSE'] for f in folds),global_fit_effects=e.tolist(),means=m.tolist())
