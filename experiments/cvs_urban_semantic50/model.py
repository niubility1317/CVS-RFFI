"""Fixed deployment model and default head, with exact read-only control reuse."""
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_signature_stats50.model import source_domain_count,classifier
from experiments.cvs_moment_area50.features import install as fixed_signature
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_source_identity_readout.input_policy import apply


def install(model,c):d.validate(c);return fixed_signature(model,c['model_seed'],'SIG_TIME')


def build(c,device='cpu'):
    a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256;torch.manual_seed(c['model_seed'])
    with installed(c) as native:model=native.build_baseline_model(a,torch.device(device))
    model._identity_input_policy=apply(model);return install(model,c).to(device)


def own_model(c,device='cpu'):
    d.validate(c)
    if c['reused']:
        from experiments.cvs_urban_branch50.model import own_model as original
        return original(c['reuse_source_config'],device)
    return load_source(c,device,lambda:build(c,device))


def load_source(c,device,builder):
    from experiments.cvs_phase1_stack.recover import verify_contract
    source=Path(c['output_root']);done=d.read(source/'completion.json');frozen=d.read(source/'source_frozen.json')
    init=d.read(source/'initialization.json');inner=d.read(source/'inner_source_contract.json')
    if (done['config']!=c or done['epoch']!=50 or done['optimizer_steps']!=11100 or done['target_access']
            or done['head_fit_updates'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access']):
        raise ValueError('Need own fixed final50/default-head source freeze')
    if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact']:
        raise ValueError('Fresh semantic students must be scratch, never target-selected parents')
    verify_contract(d.read(source/'source_contract.json'),d.read(d.SOURCE))
    if (inner['fold_seed']!=2026101503 or inner['counts']!=dict(L_train=5580,L_dev=720,U_s=56700,V=27000)
            or inner['L_dev_updates'] or inner['V_role']!='health_only'):
        raise ValueError('Actual source fold differs from archived controls')
    cp=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if (cp['epoch']!=50 or cp['a1_ema_successful_updates']!=11100 or cp['run_id']!=c['run_id'] or cp['candidate_id']!=c['row_id']
            or cp['split_info']['inner_source_contract']!=inner or not cp['args']['from_scratch'] or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt']):
        raise ValueError('Actual checkpoint lineage/budget differs')
    model=builder();model.load_state_dict(cp['model'],strict=True);return model.eval().requires_grad_(False)
