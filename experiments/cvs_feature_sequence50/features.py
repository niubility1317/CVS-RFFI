"""Class-independent middle-layer distributions and ordered path features.

Every operation is per packet, including the native default classifier. No
extra classifier, RX routing, dataset access, fitted probe, or query update.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from experiments.cvs_reference_identity.model import ReferenceCVS
from experiments.cvs_feature_restitution.features import install as install_counter,layout

ARMS=('MEAN','MEAN_VAR','NONLINEAR_MEAN','RFF','COV','SIG','SIG_TIME')
WINDOWS=4;LOCAL_DIM=6;DIRECTIONS=14;OUTPUT_DIM=160;WIDTH=28

def distribution_descriptor(u,mode,omega):
 if u.ndim!=3 or u.shape[-1]!=6 or omega.shape!=(14,6) or mode not in ARMS[:4]:raise ValueError('Fixed local distribution dimensions/mode differ')
 a=u@omega.T;mu=a.mean(1)
 if mode=='MEAN':return torch.cat((mu,mu),-1)
 if mode=='MEAN_VAR':return torch.cat((mu,a.var(1,unbiased=False)),-1)
 if mode=='NONLINEAR_MEAN':return torch.cat((mu.cos(),mu.sin()),-1)
 return torch.cat((a.cos().mean(1),a.sin().mean(1)),-1)

def path_signature(u,include_time):
 if u.ndim!=3 or u.shape[1]<2 or u.shape[-1]!=6 or not isinstance(include_time,bool):raise ValueError('A six-coordinate path with multiple positions is required')
 if include_time:
  t=torch.linspace(0,1,u.shape[1],device=u.device,dtype=u.dtype)[None,:,None].expand(len(u),-1,-1)
  u=torch.cat((u,t),-1)
 delta=u[:,1:]-u[:,:-1];before=delta.cumsum(1)-delta
 second=.5*((before[:,:,:,None]*delta[:,:,None,:])-(delta[:,:,:,None]*before[:,:,None,:])).sum(1)
 ij=torch.triu_indices(u.shape[-1],u.shape[-1],offset=1,device=u.device)
 return torch.cat((delta.sum(1),second[:,ij[0],ij[1]]),-1)

def moment_descriptor(u):
 if u.ndim!=3 or u.shape[-1]!=6 or u.shape[1]<2:raise ValueError('A six-coordinate multi-position path is required')
 mu=u.mean(1);centered=u-mu[:,None];cov=centered.transpose(1,2)@centered/u.shape[1]
 ij=torch.triu_indices(6,6,device=u.device)
 return torch.cat((mu,cov[:,ij[0],ij[1]]),-1)

class SequencePool(nn.Module):
 def __init__(self,channels,arm,seed):
  super().__init__()
  if arm not in ARMS:raise ValueError('Unregistered sequence pooling arm')
  self.arm=arm;self.channels=channels
  # Keep every original backbone/head/dropout RNG draw identical to control.
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(int(seed)+80411)
   self.local_projection=nn.Conv1d(channels,6,1,bias=False)
   self.output_projection=nn.Linear(4*28,160,bias=False)
  self.output_norm=nn.LayerNorm(160)
  self.gain_raw=nn.Parameter(torch.full((160,),math.atanh(.05/.2)))
  rng=torch.Generator().manual_seed(2026101307)
  omega=torch.randn(14,6,generator=rng)
  omega=omega/omega.norm(dim=1,keepdim=True)*torch.tensor([.5,1.,2.,.5,1.,2.,.5,1.,2.,.5,1.,2.,.5,1.])[:,None]
  self.register_buffer('omega',omega)
  dim=27 if arm=='COV' else 21 if arm=='SIG' else 28
  if dim==28:lift=torch.eye(28)
  else:
   q,_=torch.linalg.qr(torch.randn(28,28,generator=rng));lift=q[:,:dim].T*math.sqrt(28/dim)
  self.register_buffer('descriptor_lift',lift)
 def local_path(self,h):
  if h.ndim!=3 or h.shape[1]!=self.channels or h.shape[2]<8 or h.shape[2]%4:raise ValueError('Four equal multi-position native t2 windows required')
  return torch.tanh(self.local_projection(h)).transpose(1,2)
 def descriptor(self,u):
  if self.arm in ARMS[:4]:v=distribution_descriptor(u,self.arm,self.omega)
  elif self.arm=='COV':v=moment_descriptor(u)
  else:v=path_signature(u,self.arm=='SIG_TIME')
  return v@self.descriptor_lift
 def forward(self,h):
  u=self.local_path(h);b,t,_=u.shape
  windows=u.reshape(b,4,t//4,6).reshape(b*4,t//4,6)
  desc=self.descriptor(windows).reshape(b,112)
  return .2*self.gain_raw.tanh()*self.output_norm(self.output_projection(desc))

class SequenceReference(ReferenceCVS):
 def features(self,x):
  # Call-local capture; deepcopy/EMA/checkpoints retain no closure or sample.
  local={}
  def capture(module,args,output):local['h']=output
  handle=self.core.id_backbone.t2.register_forward_hook(capture)
  try:base=super().features(x)
  finally:handle.remove()
  if set(local)!={'h'}:raise RuntimeError('Actual t2 feature path was not called')
  return base+self.sequence_pool(local['h'])

def install(model,seed,arm):
 model=install_counter(model,seed,norm='none',competition=False)
 core,_,_=layout(model)
 references=[m for m in model.modules() if type(m) is ReferenceCVS]
 if len(references)!=1:raise ValueError('One native reference feature extractor required')
 ref=references[0]
 if arm!='BASE_REF':
  ref.sequence_pool=SequencePool(core.t2.pw.out_channels,arm,seed)
  ref.__class__=SequenceReference
 extra=0 if arm=='BASE_REF' else sum(p.numel() for p in ref.sequence_pool.parameters())
 model._feature_learning_report=dict(arm=arm,layer='t2',local_channels=core.t2.pw.out_channels,local_dimension=6,windows=4,descriptor_width=28,output_embedding=160,extra_parameters=extra,classifier_changed=False,default_head_only=True,separate_head_training=False,per_packet=True,local_projection_bounded='tanh',random_direction_seed=2026101307,random_direction_scales=[.5,1.,2.],gain_initial=.05,gain_bound=.2,ordered=arm in ('SIG','SIG_TIME'),time_coordinate=arm=='SIG_TIME',parameter_matched=arm!='BASE_REF',intrinsic_descriptor_rank_matched=False)
 return model
