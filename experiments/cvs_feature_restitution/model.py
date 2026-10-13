from . import design as d
from .features import install as add_features
from experiments.cvs_feature_multiscale50.model import source_domain_count,classifier
from experiments.cvs_phase1_stack.runtime import installed
import torch

def install(model,c):
 d.validate(c);return add_features(model,c['model_seed'],c['normalization'],c['competition'])
def build(c,device):
 d.validate(c);a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256
 torch.manual_seed(c['model_seed'])
 with installed(c) as native:model=native.build_baseline_model(a,torch.device(device))
 from experiments.cvs_source_identity_readout.input_policy import apply
 model._identity_input_policy=apply(model)
 return install(model,c).to(device)
