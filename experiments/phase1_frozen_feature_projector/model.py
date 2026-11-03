"""Frozen-reader feature prediction and one-step IQ feature projection."""
import torch
from torch import nn
from torch.nn import functional as F


class FeatureProjector(nn.Module):
 def __init__(self,reader,width=32):
  super().__init__();self.reader=reader.eval().requires_grad_(False)
  self.feature_head=nn.Sequential(nn.Linear(320,192),nn.SiLU(),nn.Linear(192,160))
  self.entry=nn.Conv1d(5,width,7,padding=3)
  self.layers=nn.ModuleList(nn.Conv1d(width,width,5,dilation=d,padding=2*d) for d in (1,2,4,8))
  self.gate_output=nn.Conv1d(width,4,1);self.step=nn.Parameter(torch.zeros(()));self.latest_target=None
  nn.init.zeros_(self.gate_output.weight);nn.init.zeros_(self.gate_output.bias)
  nn.init.zeros_(self.feature_head[-1].weight);nn.init.zeros_(self.feature_head[-1].bias)

 def train(self,mode=True):
  super().train(mode);self.reader.eval();return self

 def forward(self,received,initial,log_variance):
  n=len(received)
  if received.shape!=(n,2,256) or initial.shape!=received.shape or log_variance.shape!=(n,):
   raise ValueError('Matching received IQ, initial waveform and blind noise required')
  if not all(torch.isfinite(v).all() for v in (received,initial,log_variance)):
   raise FloatingPointError('Finite received evidence required')
  with torch.no_grad():
   raw_features=self.reader.features(received);initial_features=self.reader.features(initial)
  if raw_features.shape!=(n,160) or initial_features.shape!=(n,160):
   raise ValueError('One frozen 160-dimensional feature per waveform required')
  raw_unit=F.normalize(raw_features,dim=1);anchor_unit=F.normalize(initial_features,dim=1)
  target=F.normalize(anchor_unit+.3*self.feature_head(torch.cat((raw_unit,anchor_unit),1)),dim=1)
  self.latest_target=target
  with torch.enable_grad():
   query=initial.detach().clone().requires_grad_(True)
   features=self.reader.features(query)
   discrepancy=(1-F.cosine_similarity(features,target.detach(),dim=1)).sum()
   gradient=torch.autograd.grad(discrepancy,query,create_graph=False)[0].detach()
  if not torch.isfinite(gradient).all():raise FloatingPointError('Finite frozen-reader IQ gradient required')
  norm=gradient.square().mean((1,2),keepdim=True).sqrt()
  direction=torch.where(norm>1e-8,gradient/norm.clamp_min(1e-8),torch.zeros_like(gradient))
  power=received.square().mean((1,2),keepdim=True).clamp_min(1e-8);scale=power.sqrt()
  x=torch.complex(received[:,0],received[:,1])/scale[:,0]
  a=torch.complex(initial[:,0],initial[:,1])/scale[:,0]
  cross=a.conj()*x;noise=((log_variance-power[:,0,0].log()).clamp(-16,8)/8)[:,None].expand(-1,256)
  evidence=torch.stack((torch.log1p(x.abs().square()),torch.log1p(a.abs().square()),
                         cross.real.clamp(-16,16),cross.imag.clamp(-16,16),noise),dim=1)
  h=F.silu(self.entry(evidence))
  for layer in self.layers:h=h+.2*F.silu(layer(h))
  w=.1*self.gate_output(h).tanh();delta=torch.complex(w[:,0],w[:,1])*x+torch.complex(w[:,2],w[:,3])*a
  direct=torch.stack((delta.real,delta.imag),dim=1)*scale
  return initial+direct-.1*self.step.tanh()*direction*scale
