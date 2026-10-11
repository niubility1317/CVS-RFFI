"""Compatible parent weights, revised physical frontend and optional identity branch."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .reference import install_reference
from .physical import nonlinear_memory_features

class ReportIdentity(nn.Module):
 def __init__(self,base,nonlinear=False):
  super().__init__();self.base=base;self.frontend=install_reference(base)
  self._stage2_baseline_args=base._stage2_baseline_args
  self.nonlinear=nn.Sequential(nn.Linear(17,64),nn.SiLU(),nn.Linear(64,160),nn.LayerNorm(160)) if nonlinear else None
  self.nonlinear_gain=nn.Parameter(torch.zeros(160)) if nonlinear else None

 @property
 def id_backbone(self):return self.base.id_backbone
 @property
 def head(self):return self.id_backbone.encoder.core.id_backbone.cls_head

 def forward_identity_only(self,x):
  out=self.base.forward_identity_only(x)
  if self.nonlinear is not None:
   joint=out['id_feat_joint']+self.nonlinear_gain.tanh()*self.nonlinear(nonlinear_memory_features(x))
   out=dict(out,id_feat_joint=joint,tx_logits=self.head.classify(joint))
  return out

 def forward(self,x):return self.forward_identity_only(x)['tx_logits']

 def representation(self,x):
  values={}
  handles=[self.head.base_norm.register_forward_hook(lambda m,i,o:values.update(identity=o)),self.head.pa_norm.register_forward_hook(lambda m,i,o:values.update(pa=o))]
  try:out=self.forward_identity_only(x)
  finally:
   for h in handles:h.remove()
  reference=next(m.response for m in self.id_backbone.modules() if hasattr(m,'response_projection'))(x)
  h=torch.cat((values['identity'],values['pa'],reference),1)
  if h.shape!=(len(x),349):raise ValueError('Native intermediate349 routing differs')
  return out['tx_logits'],F.normalize(out['id_feat_joint'],dim=1,eps=1e-4),h

def build(c,device,checkpoint=None):
 from experiments.cvs_stage2_decouple.common import build_model
 base=build_model(c['parent_contract'],device)
 model=ReportIdentity(base,c['enabled']['nonlinear']).to(device)
 if checkpoint is not None:
  ck=torch.load(checkpoint,map_location=device,weights_only=False)
  if ck['config']!=c or ck['frontend_version']!=c['frontend']['version']:raise ValueError('Reference/identity checkpoint semantics differ')
  model.load_state_dict(ck['model'],strict=True)
 return model

def groups(model,c):
 from experiments.cvs_stage2_decouple.common import parameter_groups
 result,names=parameter_groups(model.base)
 for group in result:
  group['peak']=c['optimizer']['encoder_peak_lr' if group['name']=='encoder' else 'fusion_head_peak_lr']
 for key in names:names[key]=['base.'+n for n in names[key]]
 if model.nonlinear is not None:
  for n,p in model.named_parameters():
   if n.startswith('nonlinear'):
    p.requires_grad_(True);result[1]['params'].append(p);names['fusion_head'].append(n)
 return result,names

class Teacher:
 def __init__(self,student,decay=.99):
  self.model=deepcopy(student).eval().requires_grad_(False);self.decay=decay;self.updates=0;self.calls=0
 @torch.no_grad()
 def probabilities(self,x):
  self.calls+=1;self.model.eval();return self.model(x).softmax(1).detach()
 @torch.no_grad()
 def update(self,student):
  source=dict(student.named_parameters())
  for n,p in self.model.named_parameters():
   if source[n].requires_grad:p.mul_(self.decay).add_(source[n].detach(),alpha=1-self.decay)
  source_buffers=dict(student.named_buffers())
  for n,b in self.model.named_buffers():b.copy_(source_buffers[n])
  self.updates+=1
