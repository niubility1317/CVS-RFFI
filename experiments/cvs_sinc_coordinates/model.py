from contextlib import contextmanager
import torch
from . import design as d
from .coordinate import CoordinateSinc
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_source_identity_readout.input_policy import apply
from experiments.cvs_feature_multiscale50.model import source_domain_count
PREFIX='id_backbone.encoder.core.id_backbone.sinc.'

def physical_state(module):return {PREFIX+k:v.detach().cpu().clone() for k,v in dict(low_hz_=module.low_hz_,band_hz_=module.band_hz_,t_=module.t_,window_=module.window_).items()}
def install(model,c):
 d.validate(c);s=c['sinc_coordinates']
 if not s['enabled']:raise ValueError('Historical baseline has its original builder')
 model._identity_input_policy=apply(model);b=model.id_backbone.encoder.core.id_backbone;head=b.cls_head
 b.sinc=CoordinateSinc(b.sinc,s['mode'],s['scale_hz'])
 model._sinc_initial_physical_state=physical_state(b.sinc)
 if head is not b.cls_head:raise ValueError('Default classifier changed')
 model._sinc_coordinates_contract=dict(s,head='unchanged cosine6x160',pseudo_EMA='historical unmodified',separate_head_fit=False)
 return model
@contextmanager
def installed(c,training=False):
 d.validate(c)
 with native_installed(c) as n:
  previous=n.build_baseline_model
  if n._source_feature_auxiliary_hook is not None:raise ValueError('Unregistered training hook')
  def build(args,device):return install(previous(args,device),c).to(device)
  n.build_baseline_model=build
  try:
   if training:
    from experiments.cvs_time_factorial.monitor import installed_monitor
    with installed_monitor(n,c,d.write):yield n
   else:yield n
  finally:n.build_baseline_model=previous
def build(c,device='cpu'):
 a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256
 with installed(c) as n:return n.build_baseline_model(a,torch.device(device))
