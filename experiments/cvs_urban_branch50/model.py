"""The identical SIG_TIME/default cosine model; new internal-fold provenance only."""
import torch
from . import design as d
from experiments.cvs_signature_stats50.model import source_domain_count,classifier
from experiments.cvs_moment_area50.features import install as fixed_signature
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_source_identity_readout.input_policy import apply


def install(model,c):
    d.validate(c);return fixed_signature(model,c['model_seed'],'SIG_TIME')


def build(c,device='cpu'):
    a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256
    torch.manual_seed(c['model_seed'])
    with installed(c) as native:model=native.build_baseline_model(a,torch.device(device))
    model._identity_input_policy=apply(model);return install(model,c).to(device)


def own_model(c,device='cpu'):
    from pathlib import Path
    from experiments.cvs_phase1_stack.recover import verify_contract
    d.validate(c);source=Path(c['output_root'])
    done=d.read(source/'completion.json');frozen=d.read(source/'source_frozen.json');init=d.read(source/'initialization.json')
    if (done['config']!=c or done['epoch']!=50 or done['optimizer_steps']!=11100 or done['target_access']
            or done['head_fit_updates'] or frozen['config']!=c or frozen['epoch']!=50
            or frozen['selection']!='final_only' or frozen['target_access']):
        raise ValueError('Need source-frozen fixed E50/default-head checkpoint')
    if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact']:
        raise ValueError('No historical full-L or target-selected ancestors allowed')
    verify_contract(d.read(source/'source_contract.json'),d.read(d.SOURCE))
    inner=d.read(source/'inner_source_contract.json')
    if (inner['fold_seed']!=2026101503 or inner['counts']!=dict(L_train=5580,L_dev=720,U_s=56700,V=27000)
            or inner['L_dev_updates'] or inner['V_role']!='health_only'):
        raise ValueError('Actual internal source fold differs')
    checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if (checkpoint['epoch']!=50 or checkpoint['a1_ema_successful_updates']!=11100
            or checkpoint['run_id']!=d.RUN or checkpoint['candidate_id']!=c['row_id']
            or checkpoint['split_info']['inner_source_contract']!=inner
            or not checkpoint['args']['from_scratch'] or checkpoint['args']['baseline_ckpt'] or checkpoint['args']['teacher_ckpt']):
        raise ValueError('Default checkpoint source fold/identity/scratch budget differs')
    model=build(c,device);model.load_state_dict(checkpoint['model'],strict=True)
    return model.eval().requires_grad_(False)
