"""Checkpoint/data provenance and identity-only construction."""
from pathlib import Path
from types import SimpleNamespace
import math
import torch
from . import design as d

def clean(value):
    if torch.is_tensor(value):return clean(value.detach().cpu().tolist())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value

def verify_parent(c):
    d.validate(c)
    source=Path(c['parent_checkpoint']).parent
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(d.read(source/'source_contract.json'),d.read(d.SOURCE))
    init=d.read(source/'initialization.json');done=d.read(source/'completion.json')
    resolved=d.read(source/'resolved_config.json');actual=d.read(source/'resolved_native_args.json')
    origin=done['checkpoint_origin'];old=origin['config']
    checks=[init['scratch_only'],not init['checkpoint_sources'],not init['ancestors'],not init['target_contact'],
            init['seed']==c['model_seed'],init['source_roles']=='EXACT_MATCH',done['status']=='SOURCE_TRAINED',
            done['epoch']==200,done['logged_steps']==44400,done['optimizer_steps']==44400,not done['target_access'],
            origin['commit']==d.PARENT_COMMIT,origin['mode']=='post_training_diagnostic_recovery',
            old['run_id']==d.PARENT_ORIGIN,old['arm']=='native',old['model_seed']==c['model_seed'],
            not old['checkpoint_sources'],old['model_initialization']=='scratch',not old['target_access'],
            resolved['commit']==d.PARENT_COMMIT,all(resolved.get(k)==v for k,v in old.items()),
            actual['from_scratch'],not actual['baseline_ckpt'],not actual['teacher_ckpt'],actual['seed']==c['model_seed'],
            not actual['use_unlabeled'],not actual['use_ema_teacher'],not actual['use_concat_sat_channel_aug'],
            actual['label_smoothing']==0,actual['label_epochs']==200,actual['pseudo_epochs']==0,
            actual['a1_source_screen_only'],not actual['a1_periodic_target_inputs'],not actual['a1_periodic_target_truth'],
            not actual['a1_periodic_target_start'],not actual['a1_periodic_target_interval'],not actual['a1_final_weak_reference'],
            actual['checkpoint_selection']=='final_only',actual['wisig_pkl']==c['dataset']]
    if not all(checks):raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: parent source/config/CE contract')
    # Historical args include SatViewStage objects. Register the native
    # code directory before deserializing; no data or satellite generator runs.
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()
    ck=torch.load(c['parent_checkpoint'],map_location='cpu',weights_only=False)
    if (ck['epoch']!=200 or ck['run_id']!=d.PARENT_ORIGIN or ck['candidate_id']!=old['row_id'] or
        ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=44400):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: actual parent checkpoint')
    from experiments.cvs_feature_disentangle.source import clean as original_clean
    from experiments.cvs_phase1_stack.runtime import installed
    with installed(c):
        from scripts.train_daot_rc4_baseline import resolved_config
        if original_clean(resolved_config(SimpleNamespace(**ck['args'])))!=actual:
            raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: checkpoint args differ')
    return dict(status='VERIFIED',checkpoint=c['parent_checkpoint'],origin_run=d.PARENT_ORIGIN,
                recovery_run=d.PARENT_RUN,origin_commit=d.PARENT_COMMIT,epoch=200,optimizer_steps=44400,
                source_roles='EXACT_MATCH',target_training_or_selection=False,ancestors=[],seed=c['model_seed'])

def build_model(c,device,checkpoint=None):
    """Strict load precedes creation of any action/reference/cache state."""
    d.validate(c)
    path=checkpoint or c['parent_checkpoint']
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()
    ck=torch.load(path,map_location='cpu',weights_only=False)
    from experiments.cvs_phase1_stack.runtime import installed
    with installed(c) as native:
        model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),torch.device(device))
    model.load_state_dict(ck['model'],strict=True)
    model._stage2_baseline_args=ck['baseline_args']
    return model

def identity_logits(model,x):
    return model.forward_identity_only(x)['tx_logits']

class IdentityView(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return identity_logits(self.model,x)

def margins(logits,y):
    true=logits.gather(1,y[:,None]).squeeze(1)
    other=logits.masked_fill(torch.nn.functional.one_hot(y,logits.shape[1]).bool(),float('-inf')).max(1).values
    return true-other

def positive_huber(z):
    z=z.clamp_min(0)
    return torch.where(z<=1,.5*z.square(),z-.5)

def parameter_groups(model):
    """Train only the identity network; projection/fusion/head use higher LR."""
    for p in model.parameters():p.requires_grad_(False)
    encoder=[];fusion=[];names={'encoder':[],'fusion_head':[]}
    tokens=('cls_head','t_proj','f_proj','pa_proj','fusion','gate','reference_proj','response_proj','response_gain','reference_gate')
    for name,p in model.id_backbone.named_parameters():
        p.requires_grad_(True)
        key='fusion_head' if any(token in name for token in tokens) else 'encoder'
        (fusion if key=='fusion_head' else encoder).append(p);names[key].append('id_backbone.'+name)
    if not encoder or not fusion:raise ValueError('Empty parameter learning-rate group')
    return [dict(params=encoder,name='encoder',peak=3e-5,lr=1e-6),dict(params=fusion,name='fusion_head',peak=1e-4,lr=1e-6)],names

def learning_rate(step,peak):
    if not 1<=step<=8880:raise ValueError('Out-of-budget step')
    if step<=444:return 1e-6+(peak-1e-6)*step/444
    return 1e-6+.5*(peak-1e-6)*(1+math.cos(math.pi*(step-444)/(8880-444)))
