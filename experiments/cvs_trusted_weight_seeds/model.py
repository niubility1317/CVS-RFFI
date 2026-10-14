import torch
from experiments.cvs_moment_area50.features import install as add_features
from experiments.cvs_signature_stats50.model import source_domain_count,classifier
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_source_identity_readout.input_policy import apply

def install(model,c):return add_features(model,c['model_seed'],'SIG_TIME')
def build(c,device='cpu',*,design=None):
 if design is not None:design.validate(c);a=design.make_args(c,str(device))
 else:
  from . import design as d
  d.validate(c);a=d.make_args(c,str(device))
 a.num_domains=source_domain_count(a);a.input_len=256;torch.manual_seed(c['model_seed'])
 with installed(c) as native:model=native.build_baseline_model(a,torch.device(device))
 model._identity_input_policy=apply(model);return install(model,c).to(device)
