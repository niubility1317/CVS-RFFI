"""Learned separable x[n-m]|x[n-l]|^2 terms, before the native PA gate.

The primitive is third order on component-clipped IQ. SiLU and the downstream
native network are not homogeneous. This is received-IQ feature extraction,
not recovery of unique transmitter or receiver hardware parameters.
"""
import torch
from torch import nn
from torch.nn import functional as F

class CrossDelayResidual(nn.Module):
 def __init__(self,mode,rank=8,memory=9,channels=24):
  super().__init__()
  if mode not in ('conv','cross_delay') or (rank,memory,channels)!=(8,9,24):raise ValueError('Unregistered PA mechanism or capacity')
  self.mode=mode;self.rank=rank;self.memory=memory
  self.a=nn.Parameter(torch.randn(rank,1,memory)*.2)
  self.b=nn.Parameter(torch.randn(rank,1,memory)*.2)
  self.projection=nn.Conv1d(2*rank,channels,1)
  self.gain=nn.Parameter(torch.full((1,channels,1),.05))
 def filters(self):return F.normalize(self.a,dim=2,eps=1e-4),F.normalize(self.b,dim=2,eps=1e-4)
 def filter(self,x,w):return F.conv1d(F.pad(x,(self.memory-1,0)),w.flip(-1))
 def terms(self,x):
  if x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype not in (torch.float32,torch.float64):raise ValueError('Expected floating IQ[B,2,256]')
  q=x.clamp(-2.,2.);a,b=self.filters()
  ui=self.filter(q[:,0:1],a)
  if self.mode=='conv':return torch.cat([ui,self.filter(q[:,1:2],b)],1)
  uq=self.filter(q[:,1:2],a);power=q.square().sum(1,keepdim=True);v=self.filter(power,b)
  return torch.cat([ui*v,uq*v],1)
 def forward(self,x):
  with torch.autocast(device_type=x.device.type,enabled=False):
   return self.gain.tanh()*self.projection(F.silu(self.terms(x)))

class AugmentedPALift(nn.Module):
 def __init__(self,base,mode,seed):
  super().__init__()
  if base.memory_depth!=4 or base.orders!=(1,3,5) or base.clip!=2.:raise ValueError('Native PA lift differs')
  self.base=base
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(seed+51507)
   self.residual=CrossDelayResidual(mode)
 def forward(self,x):return self.base(x)+self.residual(x)
