"""DURA-free source-supervised blind channel correction."""
import torch
from torch import nn
from channel_refinement import fir_design,fit_bounded_delta

def as_complex(x):
 if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('IQ [N,2,256] required')
 return torch.complex(x[:,0,:],x[:,1,:])

def channel_features(received,parent_forward,logvar):
 if received.shape!=parent_forward.shape or logvar.shape!=(len(received),):raise ValueError('Matching received and parent IQ required')
 x=as_complex(received);p=as_complex(parent_forward);scale=p.abs().square().mean(1).clamp_min(1e-8).sqrt();x=x/scale[:,None];p=p/scale[:,None];design=fir_design(p);res=x-p;corr=(design.conj().transpose(1,2)@res.unsqueeze(-1)).squeeze(-1)/256;gram=(design.conj().transpose(1,2)@design)/256;power=res.abs().square().mean(1).log1p();feat=torch.cat((corr.real,corr.imag,gram.real.flatten(1),gram.imag.flatten(1),(logvar.clamp(-12,4)/12)[:,None],power[:,None]),1)
 if feat.shape!=(len(received),62) or not torch.isfinite(feat).all():raise ValueError('Finite 62-dimensional blind features required')
 return feat

def oracle_target(received,clean_forward):
 x=as_complex(received);p=as_complex(clean_forward);mask=torch.ones(256,device=x.device,dtype=torch.bool);return fit_bounded_delta(fir_design(p),x-p,mask)

class ChannelRegressor(nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',torch.zeros(62));self.register_buffer('std',torch.ones(62));self.net=nn.Sequential(nn.Linear(62,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,10));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward_features(self,features):
  if features.ndim!=2 or features.shape[1]!=62:raise ValueError('Expected 62-dimensional channel features')
  v=self.net((features-self.mean)/self.std);g=torch.complex(v[:,:5],v[:,5:]);norm=torch.linalg.vector_norm(g,dim=1).clamp_min(1e-12);return g*torch.clamp(.25/norm,max=1)[:,None]
 def forward(self,received,parent_forward,logvar):return self.forward_features(channel_features(received,parent_forward,logvar))
