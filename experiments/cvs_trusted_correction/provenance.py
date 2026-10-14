"""Read existing own-student/EMA source provenance; no new training inheritance."""
from pathlib import Path
import copy
import torch
from . import design as d

def source(seed):
 p=d.reference_source(seed);done=d.read(p/'completion.json');init=d.read(p/'initialization.json');resolved=d.read(p/'resolved_config.json');freeze=d.read(p/'source_frozen.json');args=d.read(p/'resolved_native_args.json');args=args.get('args',args)
 c=done['config']
 if done['status']!='SOURCE_TRAINED' or done['epoch']!=50 or done['logged_steps']!=11100 or done['optimizer_steps']!=11100 or done['target_access'] or done['target_evaluated'] or done['checkpoint_sources'] or c['model_seed']!=seed or c['arm']!='PAIRED_ORDER' or c['run_id']!=d.REFERENCE_RUN:raise ValueError('Fixed PAIR source provenance/budget differs')
 if init!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=seed):raise ValueError('PAIR unknown ancestry')
 if c['source_neighbor_sampling']['mode']!='PAIRS' or not c['source_neighbor_sampling']['original_gates'] or resolved['actual_source_neighbor_sampling']['mode']!='PAIRS' or resolved['identity_input_policy']['explicit_CFO_identity_input']:raise ValueError('PAIR mechanism/input policy differs')
 if freeze['selection']!='final_only' or freeze['epoch']!=50 or freeze['updates']!=11100 or freeze['target_access'] or freeze['config']!=c:raise ValueError('PAIR source freeze differs')
 original=d.old.make_args(d.baseline_config(seed),'cpu')
 for key in ('epochs','label_epochs','pseudo_epochs','use_unlabeled','use_ema_teacher','ema_decay','pseudo_temporal_gate','pseudo_temporal_mode','pseudo_temporal_window','pseudo_temporal_min_conf','pseudo_strong_agreement','pseudo_domain_gate','pseudo_threshold_mode','pseudo_quantile','label_smoothing','lambda_u','lambda_ent','lr','weight_decay','phase1_lr_schedule','phase1_lr_min','use_concat_sat_channel_aug','use_sat_consistency','baseline_ckpt','teacher_ckpt'):
  if args[key]!=getattr(original,key):raise ValueError('Historical source recipe differs:'+key)
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE));return p

def models(seed):
 from experiments.cvs_feature_multiscale50.model import build
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules();p=source(seed);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100 or ck['candidate_id']!=f'PAIRED_ORDER-s{seed}' or ck['run_id']!=d.REFERENCE_RUN or ck['checkpoint_selection']!='final_only' or ck['ema_model'] is None:raise ValueError('PAIR real final student/EMA differs')
 student=build(d.baseline_config(seed),'cpu');student.load_state_dict(ck['model'],strict=True);student.eval().requires_grad_(False)
 teacher=copy.deepcopy(student);teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval().requires_grad_(False)
 return student,teacher,p
