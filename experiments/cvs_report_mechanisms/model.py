import torch
from torch import nn
from torch.nn import functional as F
from . import design as d
from .frontends import DIMS,features,phase_rotate
from experiments.cvs_nocfo_foundation.model import build as native,groups as native_groups

class Model(nn.Module):
 def __init__(self,c,device):
  super().__init__();self.arm=c['arm'];self.native=native(c['source_loader_config'],device);self._identity_input_policy=self.native._identity_input_policy;self._stage2_baseline_args=self.native._stage2_baseline_args
  self.branch=nn.Sequential(nn.Linear(DIMS[self.arm],128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,160),nn.LayerNorm(160)) if self.arm in DIMS else None
  self.gain=nn.Parameter(torch.zeros(160)) if self.branch is not None else None
  self.to(device)
 @property
 def head(self):return self.native.head
 def forward_identity_only(self,x):
  if self.arm=='C4_ORBIT':
   z=torch.cat([phase_rotate(x,k) for k in range(4)],0);out=self.native.forward_identity_only(z);joint=out['id_feat_joint'].reshape(4,len(x),160).mean(0);return dict(id_feat_joint=joint,tx_logits=self.head.classify(joint))
  out=self.native.forward_identity_only(x)
  if self.branch is not None:
   joint=out['id_feat_joint']+2*self.gain.sigmoid()*self.branch(features(self.arm,x));out=dict(out,id_feat_joint=joint,tx_logits=self.head.classify(joint))
  return out
 def forward(self,x):return self.forward_identity_only(x)['tx_logits']

def build(c,device):d.validate(c);return Model(c,device)
def groups(model,c):
 pg,names=native_groups(model.native,c['source_loader_config']);names={k:['native.'+n for n in v] for k,v in names.items()}
 if model.branch is not None:
  extra=[p for n,p in model.named_parameters() if n.startswith('branch.') or n=='gain'];pg.append(dict(name='report_branch',params=extra,lr=c['peak_lr']));names['report_branch']=[n for n,p in model.named_parameters() if n.startswith('branch.') or n=='gain']
 return pg,names
