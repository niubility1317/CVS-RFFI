from contextlib import contextmanager
import torch
from . import design as d
from .sampler import replace
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_source_identity_readout.input_policy import apply
from experiments.cvs_feature_multiscale50.model import source_domain_count

def install(model,c):
 d.validate(c);model._identity_input_policy=apply(model)
 model._source_neighbor_sampling=dict(c['source_neighbor_sampling'],head='unchanged cosine6x160',pseudo_EMA='historical unmodified',separate_head_fit=False)
 return model

@contextmanager
def installed(c,training=False):
 d.validate(c)
 with native_installed(c) as n:
  previous=n.build_baseline_model;previous_data=n._build_ssdg_wisig_data
  if n._source_feature_auxiliary_hook is not None:raise ValueError('Unregistered feature/loss hook')
  def build(args,device):return install(previous(args,device),c).to(device)
  def data(*args,**kw):
   ctx=previous_data(*args,**kw);ctx['unlabeled_loader']=replace(ctx['unlabeled_loader'],c['source_neighbor_sampling']['mode']);return ctx
  n.build_baseline_model=build
  if training:n._build_ssdg_wisig_data=data
  try:
   if training:
    from experiments.cvs_time_factorial.monitor import installed_monitor
    with installed_monitor(n,c,d.write):yield n
   else:yield n
  finally:n.build_baseline_model=previous;n._build_ssdg_wisig_data=previous_data

def build(c,device='cpu'):
 a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256
 with installed(c) as n:return n.build_baseline_model(a,torch.device(device))
