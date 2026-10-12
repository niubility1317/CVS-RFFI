import math
from contextlib import contextmanager
import torch
from torch import nn
from . import design as d
from experiments.cvs_nocfo_foundation.model import build as native,groups as native_groups
from experiments.cvs_finite_action.operators import quality

@contextmanager
def eval_state(model):
 modes=[(m,m.training) for m in model.modules()]
 try:model.eval();yield
 finally:
  for m,mode in modes:m.training=mode

class ConditionedIdentity(nn.Module):
 def __init__(self,parent,c,fit_x,device):
  super().__init__();self.parent=parent;self.config=c;self._stage2_baseline_args=parent._stage2_baseline_args;self._identity_input_policy=parent._identity_input_policy;self._current_q=None;self._handles=[]
  self.maps=nn.ModuleDict();self.reducers=nn.ModuleDict();self.layer_info=[]
  if c['arm']=='CE30':return
  from experiments.cvs_e20_cross_style.style import _paths
  _,paths=_paths(parent.base);modules=dict(parent.base.named_modules());selected={level:('base.'+paths[level],modules[paths[level]]) for level in d.LEVELS}
  shapes={};handles=[m.register_forward_hook(lambda mod,inp,out,level=level:shapes.update({level:tuple(out.shape)})) for level,(name,m) in selected.items()]
  try:
   with eval_state(parent),torch.no_grad():parent(fit_x[:4].to(device))
  finally:
   for h in handles:h.remove()
  if c['arm']=='QUALITY_FILM':
   with torch.no_grad():q=torch.cat([quality(fit_x[i:i+256].to(device)).cpu() for i in range(0,len(fit_x),256)])
   self.register_buffer('q_mean',q.mean(0).to(device));self.register_buffer('q_std',q.std(0,unbiased=False).clamp_min(1e-4).to(device))
  for level,(name,module) in selected.items():
   shape=shapes[level]
   if len(shape)!=3 or shape[1]<=0:raise ValueError('Expected actual channel/time feature')
   channels=shape[1];key=level.replace('.','_');gen=nn.Sequential(nn.Linear(3,16),nn.SiLU(),nn.Linear(16,2*channels));nn.init.zeros_(gen[-1].weight);nn.init.zeros_(gen[-1].bias);self.maps[key]=gen.to(device)
   if c['arm']=='FREE_FILM':self.reducers[key]=nn.Linear(2*channels,3,bias=False).to(device)
   self.layer_info.append(dict(level=level,module=name,channels=channels,shape=list(shape)))
   self._handles.append(module.register_forward_hook(self.hook(key,channels)))
 def hook(self,key,channels):
  def transform(module,inputs,out):
   if self._current_q is None:raise ValueError('Condition FiLM outside own per-packet forward')
   if out.ndim!=3 or out.shape[1]!=channels or len(out)!=len(self._current_q):raise ValueError('FiLM packet/layer shape mismatch')
   if self.config['arm']=='QUALITY_FILM':context=self._current_q
   else:
    moments=torch.cat((out.mean(-1),out.std(-1,unbiased=False)),1);context=self.reducers[key](moments).clamp(-4.,4.)
   scale,shift=self.maps[key](context).chunk(2,1);scale=(.2*scale.tanh()).exp();shift=.1*shift.tanh()
   return scale[:,:,None]*out+shift[:,:,None]
  return transform
 def forward_identity_only(self,x):
  if self.config['arm']=='CE30':return self.parent.forward_identity_only(x)
  if self._current_q is not None:raise ValueError('Unexpected recursive FiLM forward')
  self._current_q=((quality(x)-self.q_mean)/self.q_std).clamp(-4.,4.) if self.config['arm']=='QUALITY_FILM' else x.new_zeros((len(x),3))
  try:return self.parent.forward_identity_only(x)
  finally:self._current_q=None
 def forward(self,x):return self.forward_identity_only(x)['tx_logits']

def build(c,fit_x,device):
 d.validate(c);p=native(c['source_loader_config'],device);m=ConditionedIdentity(p,c,fit_x,device).to(device);return m
def groups(model,c):
 pg,names=native_groups(model.parent,c['source_loader_config'])
 names={k:['parent.'+v for v in values] for k,values in names.items()}
 extras=[]
 for name,p in model.named_parameters():
  if name.startswith(('maps.','reducers.')):p.requires_grad_(True);extras.append(p);names.setdefault('condition',[]).append(name)
 if extras:pg.append(dict(params=extras,name='condition',lr=.0002))
 return pg,names
