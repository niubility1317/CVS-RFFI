"""Fixed final default-head predictions; no fitter or target feedback."""
from pathlib import Path
from . import design as d
from . import model as api
from experiments.cvs_feature_multiscale50 import predict as historical
from experiments.cvs_identity_mixstyle50.predict import baseline as shared_baseline

def source_proof(c):
 d.validate(c)
 if c['source_config'] is not None:return historical.source_proof(c['source_config'])
 p=Path(c['output_root']);done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');init=d.read(p/'initialization.json');args=d.read(p/'actual_native_args.json');args=args.get('args',args)
 if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['epoch']!=50 or done['logged_steps']!=11100 or done['optimizer_steps']!=11100 or done['target_access'] or done['target_evaluated'] or done['checkpoint_sources'] or done['ancestors'] or done['initialization']!='scratch':raise ValueError('Own source budget/provenance differs')
 if not done['native_fitted_probes_disabled'] or done['native_disabled_probe_calls']!=1 or done['head_fit_updates'] or not resolved['native_fitted_probes_disabled'] or done['L_context_steps']!=11100:raise ValueError('Default joint head or actual labeled calls differ')
 if init!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']):raise ValueError('Source initialization differs')
 if resolved['config']!=c or done['feature_extractor']!=resolved['actual_feature_extractor'] or done['identity_input_policy']['explicit_CFO_identity_input']:raise ValueError('Actual feature/input policy differs')
 expected=d.make_args(c,'cpu');keys=('epochs','label_epochs','pseudo_epochs','use_unlabeled','use_ema_teacher','ema_decay','pseudo_temporal_gate','pseudo_temporal_mode','pseudo_temporal_window','pseudo_temporal_min_conf','pseudo_strong_agreement','pseudo_domain_gate','pseudo_threshold_mode','pseudo_quantile','label_smoothing','lambda_u','lambda_ent','phase1_lr_schedule','phase1_lr_min','use_concat_sat_channel_aug','use_sat_consistency','lambda_sat_cls','lambda_sat_cons','sat_training_mode','from_scratch','baseline_ckpt','teacher_ckpt','output_dir','run_id','candidate_id','lr','weight_decay','freeze_backbone')
 if any(args.get(k)!=getattr(expected,k) for k in keys):raise ValueError('Oldgate/EMA/cos actual recipe differs')
 if d.read(p/'source_frozen.json')!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=50,selection='final_only',target_access=False):raise ValueError('Own final freeze differs')
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE));return p

def own_model(c,device):
 if c['source_config'] is not None:return historical.own_model(c['source_config'],device)
 import torch
 p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100 or ck['run_id']!=d.RUN or ck['candidate_id']!=c['row_id'] or ck['checkpoint_selection']!='final_only':raise ValueError('Own checkpoint identity differs')
 model=api.build(c,device);model.load_state_dict(ck['model'],strict=True);return model.eval().requires_grad_(False)
def baseline(c,ids):return shared_baseline(c,ids,design=d)
def pipeline():
 from experiments.cvs_default_head.pipeline import DefaultPipeline
 return DefaultPipeline(d,source_proof,own_model,baseline)
def predict(row):return pipeline().predict(row)
def preflight(row):return pipeline().preflight(row)
