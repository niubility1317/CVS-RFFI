"""Input-only inference for a source-selected spectral-prior restorer."""
import torch
from torch.nn import functional as F

@torch.no_grad()
def make_prior(anchor,bank,power,unit):
 z=torch.complex(anchor[:,0],anchor[:,1]);c=z[:,64:-64];energy=c.abs().square().sum(1).clamp_min(1e-12);score=(c@unit.conj().T).abs()/energy.sqrt()[:,None];values,index=score.topk(16,dim=1);neighbors=bank[index];gain=(neighbors[:,:,64:-64].conj()*c[:,None,:]).sum(-1)/power[index];weight=F.softmax((values-values[:,:1])/.05,dim=1);p=(neighbors*gain[:,:,None]*weight[:,:,None]).sum(1);return torch.stack((p.real,p.imag),1)

@torch.no_grad()
def apply(model,phase,gate,bank,power,unit,alpha,threshold,x):
 enabled=gate(x).sigmoid()>threshold;out=x.clone()
 if enabled.any():
  a=phase.anchor(x[enabled]);ref=make_prior(a,bank,power,unit);out[enabled]=a+alpha*(model(a,ref)-a)
 return out,enabled
