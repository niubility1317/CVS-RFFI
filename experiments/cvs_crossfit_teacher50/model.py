"""Same actual CVS/SIG_TIME/default head; independently scratch L-only teachers."""
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
    from experiments.cvs_phase1_stack.recover import verify_contract
    d.validate(c);source=Path(c['output_root']);done=d.read(source/'completion.json');frozen=d.read(source/'source_frozen.json')
    init=d.read(source/'initialization.json');inner=d.read(source/'inner_source_contract.json')
    if (done['config']!=c or done['epoch']!=50 or done['optimizer_steps']!=11100 or done['target_access']
            or done['head_fit_updates'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access']):
        raise ValueError('Need own fixed final50/default-head source-frozen teacher')
    if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact']:
        raise ValueError('Whole teacher lineage must be independent scratch')
    verify_contract(d.read(source/'source_contract.json'),d.read(d.SOURCE))
    expected_RX=[3,4,6,8] if c['arm']=='EXCLUDE_RX1' else [1,3,4,6,8]
    if (inner['train_receivers']!=expected_RX or inner['counts']['L_train']!=4464 or inner['L_dev_updates']
            or inner['fold_seed']!=2026101503 or inner['subset_seed']!=d.SUBSET_SEED or inner['arm']!=c['arm']):
        raise ValueError('Teacher actual RX exposure/source subset differs')
    cp=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if (cp['epoch']!=50 or cp['a1_ema_successful_updates']!=11100 or cp['run_id']!=d.RUN or cp['candidate_id']!=c['row_id']
            or cp['split_info']['inner_source_contract']!=inner or not cp['args']['from_scratch']
            or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt'] or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher']):
        raise ValueError('Teacher checkpoint inherits forbidden model/EMA/U exposure')
    model=build(c,device);model.load_state_dict(cp['model'],strict=True);return model.eval().requires_grad_(False)
