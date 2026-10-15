"""Unchanged single SIG_TIME and its joint default cosine head."""
import torch
from . import design as d
from experiments.cvs_signature_stats50.model import source_domain_count,classifier
from experiments.cvs_moment_area50.features import install as fixed_signature
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_source_identity_readout.input_policy import apply
from experiments.cvs_urban_semantic50.model import load_source


def install(model,c):d.validate(c);return fixed_signature(model,c['model_seed'],'SIG_TIME')


def build(c,device='cpu'):
    a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256;torch.manual_seed(c['model_seed'])
    with installed(c) as native:model=native.build_baseline_model(a,torch.device(device))
    model._identity_input_policy=apply(model);return install(model,c).to(device)


def own_model(c,device='cpu'):
    from pathlib import Path
    d.validate(c);mechanism=d.read(Path(c['output_root'])/'training_mechanism_complete.json')
    if (mechanism['status']!='VERIFIED' or mechanism['updates']!=11100 or mechanism['main_forward_packets']!=9031800
            or mechanism['head_fit_updates'] or mechanism['U_truth_read']):
        raise ValueError('Complete own E50 five-view/default-head source mechanism required')
    return load_source(c,device,lambda:build(c,device))
