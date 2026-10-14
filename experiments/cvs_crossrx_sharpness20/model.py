"""All controls share SIG_TIME and the original fixed centered Sinc chart."""
import torch
from . import design as d
from experiments.cvs_feature_sequence50 import model as parent
from experiments.cvs_sigtime_heldcredit.coordinate import CoordinateSinc,PREFIX
from experiments.cvs_phase1_stack.runtime import installed

classifier=parent.classifier;source_domain_count=parent.source_domain_count

def install(model,c):
 d.validate(c);model=parent.install(model,d.structural_config(c['model_seed']))
 weight=classifier(model).weight
 sinc=model.get_submodule(PREFIX);count=sum(p.numel() for p in model.parameters())
 owner,name=PREFIX.rsplit('.',1);replacement=CoordinateSinc(sinc,'CENTER_25K')
 setattr(model.get_submodule(owner),name,replacement.to(sinc.low_hz_.device))
 if classifier(model).weight is not weight or count!=sum(p.numel() for p in model.parameters()):
  raise ValueError('Centered feature chart changed the default head or parameter count')
 model._feature_learning_report.update(sinc_coordinates='CENTER_25K',sinc_scale_hz=25000.,sinc_primary_path=PREFIX,
  extra_mechanism_parameters=0,additional_classifier_parameters=0)
 return model

def build(c,device):
 d.validate(c);a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256
 torch.manual_seed(c['model_seed'])
 with installed(c) as native:model=native.build_baseline_model(a,torch.device(device))
 from experiments.cvs_source_identity_readout.input_policy import apply
 model._identity_input_policy=apply(model)
 return install(model,c).to(device)
