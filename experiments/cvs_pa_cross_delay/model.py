from contextlib import contextmanager
import torch
from . import design as d
from .block import AugmentedPALift
from experiments.cvs_source_identity_readout.input_policy import apply
from experiments.cvs_phase1_stack.runtime import installed as native_installed

def install(model,c):
 d.validate(c)
 if not c['pa_cross_delay']['enabled']:raise ValueError('Frozen baseline uses its original model')
 model._identity_input_policy=apply(model)
 backbone=model.id_backbone.encoder.core.id_backbone;head=backbone.cls_head
 if not backbone.use_pa_path or backbone.pa_feature_source!='raw_iq':raise ValueError('Expected native raw-IQ PA branch')
 backbone.pa_lift=AugmentedPALift(backbone.pa_lift,c['pa_cross_delay']['mode'],c['model_seed'])
 if backbone.cls_head is not head:raise ValueError('Default classifier changed')
 model._pa_cross_delay_contract=dict(c['pa_cross_delay'],head='unchanged cosine6x160',pseudo_EMA='historical unmodified',separate_head_fit=False)
 return model

@contextmanager
def installed(c,training=False):
 d.validate(c)
 with native_installed(c) as n:
  previous=n.build_baseline_model
  if n._source_feature_auxiliary_hook is not None:raise ValueError('Unregistered auxiliary training hook')
  def build(args,device):return install(previous(args,device),c).to(device)
  n.build_baseline_model=build
  try:yield n
  finally:n.build_baseline_model=previous

def build(c,device='cpu'):
 a=d.make_args(c,str(device));a.num_domains=5;a.input_len=256
 with installed(c) as n:return n.build_baseline_model(a,torch.device(device))
