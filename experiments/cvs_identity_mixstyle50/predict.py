"""identity MixStyle source provenance and default-only shared evaluation."""
from pathlib import Path
from . import design as d
from experiments.cvs_feature_multiscale50 import predict as historical

def source_proof(c):
 d.validate(c)
 if c.get('source_config') is not None:return historical.source_proof(c['source_config'])
 p=Path(c['output_root']);done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');init=d.read(p/'initialization.json');args=d.read(p/'resolved_native_args.json')
 if 'args' in args:args=args['args']
 if (done.get('status')!='SOURCE_TRAINED' or done.get('config')!=c or done.get('epoch')!=50 or done.get('logged_steps')!=11100 or done.get('optimizer_steps')!=11100 or done.get('target_access') is not False or done.get('target_evaluated') is not False or done.get('checkpoint_sources')!=[]):raise ValueError('Own identity MixStyle source budget/role differs')
 if init!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']):raise ValueError('Own identity MixStyle scratch provenance differs')
 actual=dict(c['identity_mixstyle'],head='unchanged cosine6x160',separate_head_fit=False)
 if any(resolved.get(k)!=v for k,v in c.items()) or resolved.get('actual_identity_mixstyle')!=actual or resolved.get('identity_input_policy',{}).get('explicit_CFO_identity_input') is not False:raise ValueError('Resolved identity MixStyle/default-head contract differs')
 expected=d.make_args(c,'cpu')
 keys=('epochs','label_epochs','pseudo_epochs','use_unlabeled','use_ema_teacher','ema_decay','pseudo_temporal_gate','pseudo_temporal_mode','pseudo_temporal_window','pseudo_temporal_min_conf','pseudo_strong_agreement','pseudo_domain_gate','pseudo_threshold_mode','pseudo_quantile','label_smoothing','lambda_u','lambda_ent','phase1_lr_schedule','phase1_lr_min','use_concat_sat_channel_aug','use_sat_consistency','lambda_sat_cls','lambda_sat_cons','sat_training_mode','from_scratch','baseline_ckpt','teacher_ckpt','output_dir','run_id','candidate_id','lr','weight_decay','freeze_backbone')
 if any(args.get(k)!=getattr(expected,k) for k in keys):raise ValueError('Actual oldgate/EMA/cos recipe differs')
 frozen=dict(status='SOURCE_FROZEN',config=c,epoch=50,updates=11100,checkpoint=str(p/'final_ssdg.pth'),selection='final_only',target_access=False)
 if d.read(p/'source_frozen.json')!=frozen:raise ValueError('Own identity MixStyle fixed final freeze differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE));return p

def own_model(c,device):
 import torch
 from types import SimpleNamespace
 if c.get('source_config') is not None:return historical.own_model(c['source_config'],device)
 p=source_proof(c)
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules()
 ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100 or ck['run_id']!=d.RUN or ck['candidate_id']!=c['row_id'] or ck['checkpoint_selection']!='final_only':raise ValueError('Own identity MixStyle checkpoint identity differs')
 from .model import installed
 with installed(c) as native:model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),torch.device(device))
 model.load_state_dict(ck['model'],strict=True)
 return model.eval().requires_grad_(False)

def baseline(c,ids):
 import numpy as np
 if c['arm']!=d.BASELINE_ARM:raise ValueError('Only fixed historical baseline can reuse predictions')
 unused,prior_ids=historical.preflight(c['source_config']['row_id'])
 p=Path(d.PROJECT)/'runs'/d.BASELINE_EVAL_RUN/c['row_id']/'prediction/predictions.npz'
 with np.load(p,allow_pickle=False) as saved:
  if set(saved.files)!={'ids','clean'} or not np.array_equal(ids,saved['ids']) or not np.array_equal(ids,prior_ids):raise ValueError('Historical default prediction IDs differ')
  return saved['clean'].copy()

def pipeline():
 from experiments.cvs_default_head.pipeline import DefaultPipeline
 return DefaultPipeline(d,source_proof,own_model,baseline)
def predict(row):return pipeline().predict(row)
def preflight(row):return pipeline().preflight(row)
