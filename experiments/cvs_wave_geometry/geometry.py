import torch
from torch import nn

def projection(width=32,seed=2026101207):
 g=torch.Generator().manual_seed(seed)
 return torch.linalg.qr(torch.randn(256,width,generator=g,dtype=torch.float64),mode='reduced').Q.float()

def features(x,kind,k):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not torch.isfinite(x).all():raise ValueError('Expected finite IQ Bx2x256')
 a=x.to(torch.float64)@k.to(device=x.device,dtype=torch.float64)
 energy=a.square().sum((1,2),keepdim=True)
 # A zero packet stays in evaluation. No fitting, cross-packet moments or ridge.
 a=a/energy.sqrt().clamp_min(1e-12)
 i,q=a[:,0],a[:,1]
 ii=i[:,:,None]*i[:,None,:];qq=q[:,:,None]*q[:,None,:]
 iq=i[:,:,None]*q[:,None,:];qi=q[:,:,None]*i[:,None,:]
 if kind=='IQ':
  v=a.flatten(1);v=torch.nn.functional.pad(v,(0,4096-v.shape[1]))
 elif kind=='RAW_REL':v=torch.cat([ii.flatten(1),qq.flatten(1),iq.flatten(1),qi.flatten(1)],1)*32
 elif kind=='COMPLEX_REL':
  re=(ii+qq).flatten(1);im=(qi-iq).flatten(1)
  v=torch.cat([re,im,re,im],1)*16
 elif kind=='GL2_REL':
  c=a@a.transpose(1,2);e,u=torch.linalg.eigh(c)
  keep=e>e[:,-1:].clamp_min(1e-30)*1e-8
  inv=torch.where(keep,1/e.clamp_min(1e-30),torch.zeros_like(e))
  pinv=(u*inv[:,None,:])@u.transpose(1,2)
  p=(a.transpose(1,2)@pinv@a).flatten(1)
  v=torch.cat([p,p,p,p],1)*8
 else:raise ValueError('Unregistered representation')
 if v.shape!=(len(x),4096) or not torch.isfinite(v).all():raise FloatingPointError('Relation representation')
 return v.float()

class Head(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(4096,96),nn.SiLU(),nn.LayerNorm(96),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,6))
 def forward(self,x):return self.net(x)

def diagnostics(x,k):
 a=x.double()@k.double();e=torch.linalg.eigvalsh(a@a.transpose(1,2))
 return dict(count=len(x),numerical_rank_less_than2=int((e[:,0]<=e[:,1].clamp_min(1e-30)*1e-8).sum()),zero_packets=int((a.square().sum((1,2))==0).sum()),minimum_eigen_ratio=float((e[:,0]/e[:,1].clamp_min(1e-30)).min()),scope='numerical rank threshold; exact GL2 proof for full rank without truncation')
