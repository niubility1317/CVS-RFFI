import torch
from torch import nn
from . import design as d
from experiments.cvs_nocfo_foundation.model import build as native,groups as native_groups
from experiments.cvs_wave_geometry.geometry import projection

def relations(x,k,kind):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not torch.isfinite(x).all():raise ValueError('Finite per-packet IQ256 required')
 # The same fixed numerical compensation as the existing identity reference.
 # No known excitation, repeated preamble or calibrated oscillator is asserted.
 z=torch.complex(x[:,0].double(),x[:,1].double())
 z=z/z.abs().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
 w=z[:,80:160];cross=(w[:,20:]*w[:,:-20].conj()).sum(-1)
 slope=torch.where(cross.abs()>1e-12,torch.atan2(cross.imag,cross.real)/20,torch.zeros_like(cross.real))
 t=torch.arange(256,device=x.device,dtype=torch.float64)
 z=z*torch.polar(torch.ones_like(t[None,:]*slope[:,None]),-slope[:,None]*t[None,:])
 a=torch.stack((z.real,z.imag),1)@k.double()
 a=a/a.square().sum((1,2),keepdim=True).sqrt().clamp_min(1e-12)
 i,q=a[:,0],a[:,1];ii=i[:,:,None]*i[:,None,:];qq=q[:,:,None]*q[:,None,:]
 iq=i[:,:,None]*q[:,None,:];qi=q[:,:,None]*i[:,None,:]
 if kind=='RAW_REL_BRANCH':v=torch.stack((ii,qq,iq,qi),1)*32
 elif kind=='COMPLEX_REL_BRANCH':
  re=ii+qq;im=qi-iq;v=torch.stack((re,im,re,im),1)*16
 else:raise ValueError('Unregistered relation kind')
 if v.shape!=(len(x),4,32,32) or not torch.isfinite(v).all():raise FloatingPointError('Complement relation output')
 return v.float()

class RelationIdentity(nn.Module):
 def __init__(self,parent,c):
  super().__init__();self.parent=parent;self.config=c
  self._stage2_baseline_args=parent._stage2_baseline_args;self._identity_input_policy=parent._identity_input_policy
  self.register_buffer('projection',projection())
  self.branch=nn.Sequential(nn.Conv2d(4,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.AvgPool2d(2),
   nn.Conv2d(16,24,3,padding=1),nn.GroupNorm(4,24),nn.SiLU(),nn.AdaptiveAvgPool2d(2),nn.Flatten(),
   nn.Linear(96,160),nn.LayerNorm(160))
  self.gain=nn.Parameter(torch.zeros(160));self._current_q=None
  self.layer_info=[dict(level='full256_relation_branch',channels=4,shape=[4,32,32],feature_width=160,
   native_backbone_retained=True,whole_model_invariant=False)]
 def forward_identity_only(self,x):
  out=self.parent.forward_identity_only(x);rel=self.branch(relations(x,self.projection,self.config['arm']))
  joint=out['id_feat_joint']+.2*self.gain.tanh()*rel
  return dict(out,id_feat_joint=joint,tx_logits=self.parent.head.classify(joint))
 def forward(self,x):return self.forward_identity_only(x)['tx_logits']

def build(c,fit_x,device):
 d.validate(c);return RelationIdentity(native(c['source_loader_config'],device),c).to(device)
def groups(model,c):
 pg,names=native_groups(model.parent,c['source_loader_config'])
 names={key:['parent.'+n for n in vv] for key,vv in names.items()}
 extra=[]
 for n,p in model.named_parameters():
  if n=='gain' or n.startswith('branch.'):p.requires_grad_(True);extra.append(p);names.setdefault('relation',[]).append(n)
 pg.append(dict(params=extra,name='relation',lr=.0002));return pg,names
