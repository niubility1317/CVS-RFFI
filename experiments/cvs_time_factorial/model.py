from contextlib import contextmanager
from copy import deepcopy
import torch
from torch import nn
from . import design as d
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_source_identity_readout.input_policy import apply
from experiments.cvs_feature_multiscale50.model import source_domain_count

class TemporalBlock(nn.Module):
 def __init__(self,old,settings,activation):
  super().__init__()
  if tuple(map(tuple,settings)) not in (((3,1),(5,1),(5,1)),((3,1),(5,1),(5,2))) or activation not in ('none','silu'):raise ValueError('Unregistered temporal factor')
  self.kind='time';self._settings=tuple(map(tuple,settings));self.branch_activation=activation
  ci,co=old.dw.in_channels,old.pw.out_channels
  self.branches=nn.ModuleList([nn.Conv1d(ci,ci,k,padding=(k-1)//2*rate,dilation=rate,groups=ci,bias=False) for k,rate in self._settings])
  self.fuse=nn.Conv1d(3*ci,co,1,bias=False);self.skip=nn.Conv1d(ci,co,1,bias=False)
  self.norm=deepcopy(old.norm);self.act=deepcopy(old.act);self.pool=deepcopy(old.pool);self.drop=deepcopy(old.drop)
 def forward(self,x):
  parts=[b(x) for b in self.branches]
  if self.branch_activation=='silu':parts=[torch.nn.functional.silu(v) for v in parts]
  return self.drop(self.pool(self.act(self.norm(self.fuse(torch.cat(parts,1))+self.skip(x)))))

def install(model,c):
 d.validate(c);s=c['time_factorial']
 if not s['enabled']:raise ValueError('Historical model uses its original builder')
 model._identity_input_policy=apply(model);b=model.id_backbone.encoder.core.id_backbone;head=b.cls_head
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(c['model_seed']+s['private_init_offset'])
  for layer in s['layers']:setattr(b,layer,TemporalBlock(getattr(b,layer),s['settings'],s['branch_activation']))
 if b.cls_head is not head:raise ValueError('Default classifier object changed')
 model._time_initial_parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if any(n.startswith('id_backbone.encoder.core.id_backbone.'+layer+'.') for layer in s['layers'])}
 model._time_factorial_contract=dict(s,head='unchanged cosine6x160',pseudo_EMA='historical unmodified',separate_head_fit=False,outer_torch_RNG_preserved=True)
 return model

@contextmanager
def installed(c,training=False):
 d.validate(c)
 with native_installed(c) as n:
  previous=n.build_baseline_model
  if n._source_feature_auxiliary_hook is not None:raise ValueError('Unregistered auxiliary hook')
  def build(args,device):return install(previous(args,device),c).to(device)
  n.build_baseline_model=build
  try:
   if training:
    from .monitor import installed_monitor
    with installed_monitor(n,c,d.write):yield n
   else:yield n
  finally:n.build_baseline_model=previous

def build(c,device='cpu'):
 a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256
 with installed(c) as n:return n.build_baseline_model(a,torch.device(device))
