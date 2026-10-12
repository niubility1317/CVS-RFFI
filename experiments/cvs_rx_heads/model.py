import torch
from torch import nn
from torch.nn import functional as F
from . import design as d
from experiments.cvs_nocfo_foundation.model import build as native,groups as native_groups
class RXHeads(nn.Module):
 def __init__(self,parent,c):
  super().__init__();self.parent=parent;self.config=c;gen=torch.Generator().manual_seed(c['heads']['initializer_seed']);self.experts=nn.Parameter(torch.randn(4,6,160,generator=gen)*.02)
  self._identity_input_policy=parent._identity_input_policy;self._current_q=None;self.layer_info=[dict(level='nativejoint160classifier',experts=4,source_RX_only_duringtraining=True,inference_RX=False)]
 def training_logits(self,x):
  out=self.parent.forward_identity_only(x);h=F.normalize(out['id_feat_joint'],dim=1);heads=16*torch.einsum('bd,kcd->bkc',h,F.normalize(self.experts,dim=-1));return out['tx_logits'],heads
 def forward(self,x):
  a,b=self.training_logits(x);p=.5*a.softmax(1)+.5*b.softmax(-1).mean(1);return p.clamp_min(1e-12).log()
def build(c,fit_x,device):d.validate(c);return RXHeads(native(c['source_loader_config'],device),c).to(device)
def groups(model,c):
 pg,names=native_groups(model.parent,c['source_loader_config']);names={key:['parent.'+n for n in values] for key,values in names.items()};model.experts.requires_grad_(True);pg.append(dict(params=[model.experts],name='experts',lr=.0002));names['experts']=['experts'];return pg,names
def assignments(c,fit):
 if c['arm']=='RX_HEAD':
  values=sorted(fit['rx'].unique().tolist());a=torch.tensor([values.index(r) for r in fit['rx'].tolist()])
 else:
  perm=torch.randperm(len(fit['ids']),generator=torch.Generator().manual_seed(c['heads']['assignment_seed']));a=torch.empty(len(perm),dtype=torch.long);a[perm]=torch.arange(len(perm))%4
 if sorted(a.bincount(minlength=4).tolist())!=[1260]*4:raise ValueError('Eachsourcehead musthave1260fitL')
 return a
