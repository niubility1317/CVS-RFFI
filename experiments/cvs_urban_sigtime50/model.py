import torch
from . import design as d
from experiments.cvs_signature_stats50.model import source_domain_count, classifier
from experiments.cvs_moment_area50.features import install as fixed_signature
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_source_identity_readout.input_policy import apply

def install(model,c):
    d.validate(c); return fixed_signature(model,c['model_seed'],'SIG_TIME')

def build(c,device='cpu'):
    a=d.make_args(c,str(device)); a.num_domains=source_domain_count(a); a.input_len=256
    torch.manual_seed(c['model_seed'])
    with installed(c) as n: m=n.build_baseline_model(a,torch.device(device))
    m._identity_input_policy=apply(m); return install(m,c).to(device)

def own_model(c,device='cpu'):
    d.validate(c); source=__import__('pathlib').Path(c['output_root'])
    done=d.read(source/'completion.json'); frozen=d.read(source/'source_frozen.json')
    if done['config']!=c or done['epoch']!=50 or done['optimizer_steps']!=11100 or done['target_access']:
        raise ValueError('Own fixed final50 source provenance differs')
    init=d.read(source/'initialization.json')
    if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['source_roles']!='EXACT_MATCH':
        raise ValueError('Scratch training provenance differs')
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(d.read(source/'source_contract.json'),d.read(d.SOURCE))
    if frozen['config']!=c or frozen['epoch']!=50 or frozen['selection']!='final_only' or frozen['target_access']:
        raise ValueError('Model must be source-frozen before target IQ')
    ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100 or ck['candidate_id']!=c['row_id'] or ck['run_id']!=d.RUN:
        raise ValueError('Checkpoint identity/budget differs')
    if not ck['args']['from_scratch'] or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt']:
        raise ValueError('Unexpected inherited model/teacher')
    m=build(c,device); m.load_state_dict(ck['model'],strict=True)
    return m.eval().requires_grad_(False)
