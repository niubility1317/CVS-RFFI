from contextlib import contextmanager
import torch
from . import design as d
from .episode import Objective
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_source_identity_readout.input_policy import apply

@contextmanager
def installed(c,training=False):
 d.validate(c)
 if c['arm']=='LEGACY50':raise ValueError('Baseline source belongs to its original run')
 with native_installed(c) as n:
  previous=n.build_baseline_model;hook=n._source_feature_auxiliary_hook
  def build(args,device):
   model=previous(args,device);model._identity_input_policy=apply(model);model._episodic_feature_contract=dict(c['source_episode'],class_head='unchanged cosine6x160',pseudo_EMA='historical unmodified')
   return model
  n.build_baseline_model=build
  if training:
   from experiments.cvs_source_transfer_probe.run import data
   L,V,contract=data();del V
   n._source_feature_auxiliary_hook=Objective(c,L,contract)
  try:yield n
  finally:n.build_baseline_model=previous;n._source_feature_auxiliary_hook=hook

def build(c,device='cpu'):
 a=d.make_args(c,str(device));a.num_domains=5;a.input_len=256
 with installed(c) as n:return n.build_baseline_model(a,torch.device(device))
