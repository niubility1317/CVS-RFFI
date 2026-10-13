from pathlib import Path
from . import design as d

def source_proof(c):
 d.validate(c)
 if c['baseline_reuse']:
  from experiments.cvs_feature_multiscale50.predict import source_proof as original
  return original(c['baseline_reference'])
 p=Path(c['output_root']);done=d.read(p/'completion.json');initial=d.read(p/'initialization.json');freeze=d.read(p/'source_frozen.json');resolved=d.read(p/'resolved_config.json')
 from .source import require_budget
 from experiments.cvs_feature_multiscale50.predict import validate_recipe_args
 from experiments.cvs_phase1_stack.recover import verify_contract
 require_budget(done['logged_steps'],done['optimizer_steps'])
 if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['epoch']!=50 or done['target_access'] or done['target_evaluated'] or done['checkpoint_sources'] or done['ancestors'] or done['initialization']!='scratch' or done['L_context_steps']!=11100:raise ValueError('New source provenance/budget differs')
 if initial!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']):raise ValueError('Unknown initialization')
 if freeze!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=50,selection='final_only',target_access=False) or resolved['config']!=c:raise ValueError('Source final freeze differs')
 if done['identity_input_policy']['explicit_CFO_identity_input'] or resolved['identity_input_policy']['explicit_CFO_identity_input']:raise ValueError('CFO input policy differs')
 if not done.get('native_fitted_probes_disabled') or done.get('native_disabled_probe_calls')!=1 or done.get('head_fit_updates')!=0 or not resolved.get('native_fitted_probes_disabled'):raise ValueError('Native separate classifier fitting was not disabled')
 report=done['feature_extractor']
 if report['arm']!=c['arm'] or report['classifier_changed'] or report['output_embedding']!=160 or not report['default_head_only'] or report['separate_head_training'] or report['layer']!='t2':raise ValueError('Actual sequence/default-head mechanism differs')
 if report['extra_parameters']!=6*report['local_channels']+18400:raise ValueError('Feature parameter count differs')
 if any(done['mask_counts'].values()):raise ValueError('Sequence route does not use evidence masking')
 if resolved['actual_feature_extractor']!=report or done['baseline_recipe']!='historicaloldgate+EMA+cos' or done['label_epochs']!=33 or done['pseudo_epochs']!=17:raise ValueError('Actual oldgate recipe differs')
 # Use this candidate's native arguments, not a same-name historical configuration.
 args=d.read(p/'actual_native_args.json');args=args.get('args',args);expected=d.make_args(c,'cpu')
 for key in ('run_id','candidate_id','output_dir','epochs','label_epochs','pseudo_epochs'):
  if args.get(key)!=getattr(expected,key):raise ValueError('Native row arguments differ: '+key)
 original=d.baseline_config(c['model_seed']);reference=d.read(Path(original['output_root'])/'actual_native_args.json');reference=reference.get('args',reference)
 for key in ('use_unlabeled','use_ema_teacher','ema_decay','pseudo_temporal_gate','pseudo_temporal_mode','pseudo_temporal_window','pseudo_temporal_min_conf','pseudo_strong_agreement','pseudo_domain_gate','pseudo_threshold_mode','pseudo_quantile','label_smoothing','lambda_u','lambda_ent','phase1_lr_schedule','phase1_lr_min','use_concat_sat_channel_aug','use_sat_consistency','lambda_sat_cls','lambda_sat_cons','sat_training_mode','from_scratch','baseline_ckpt','teacher_ckpt','lr','weight_decay'):
  if args.get(key)!=reference.get(key) or args.get(key)!=getattr(expected,key):raise ValueError('Historical recipe differs: '+key)
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE));return p

def own_model(c,device):
 import torch
 if c['baseline_reuse']:
  from experiments.cvs_feature_multiscale50.predict import own_model as original
  return original(c['baseline_reference'],device)
 from .model import build
 p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=50 or ck['candidate_id']!=c['row_id'] or ck['run_id']!=d.RUN or ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=11100:raise ValueError('Fixed final checkpoint differs')
 model=build(c,device);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False);return model
