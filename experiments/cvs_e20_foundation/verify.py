"""Read-only E20 provenance verification; never falls back to E200/E40."""
from pathlib import Path
from types import SimpleNamespace
import torch
from experiments.cvs_ce_epoch2040 import design as baseline

EXECUTION_COMMIT='83b1903fed01854f5abb75646a02e9a5e190a90e'

def config(seed):
 return baseline.config(next(r for r in baseline.rows() if r['epochs']==20 and r['model_seed']==seed))

def validate_metadata(c,init,done,resolved,actual):
 baseline.validate(c)
 checks=[c['epochs']==20,c['updates']==4440,c['arm']=='native',c['model_initialization']=='scratch',
  not c['checkpoint_sources'],not c['target_access'],c['checkpoint_selection']=='final_only',
  init['scratch_only'],not init['checkpoint_sources'],not init['ancestors'],not init['target_contact'],
  init['source_roles']=='EXACT_MATCH',init['seed']==c['model_seed'],
  done['status']=='SOURCE_TRAINED',done['epoch']==20,done['logged_steps']==4440,done['optimizer_steps']==4440,
  done['config']==c,not done['target_access'],not done['checkpoint_sources'],
  resolved['commit']==EXECUTION_COMMIT,all(resolved.get(k)==v for k,v in c.items()),
  actual['from_scratch'],not actual['baseline_ckpt'],not actual['teacher_ckpt'],actual['seed']==c['model_seed'],
  actual['label_epochs']==20,actual['pseudo_epochs']==0,actual['label_smoothing']==0,
  not actual['use_unlabeled'],not actual['use_ema_teacher'],not actual['use_concat_sat_channel_aug'],
  actual['checkpoint_selection']=='final_only',actual['wisig_pkl']==c['dataset'],
  actual['a1_source_screen_only'],not actual['a1_periodic_target_inputs'],not actual['a1_periodic_target_truth'],
  not actual['a1_periodic_target_start'],not actual['a1_periodic_target_interval'],not actual['a1_final_weak_reference']]
 if not all(checks):raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: original E20 metadata differs')
 return True

def verify(seed,build_smoke=False):
 c=config(seed);source=Path(c['output_root']);read=baseline.read
 from experiments.cvs_phase1_stack.recover import verify_contract
 verify_contract(read(source/'source_contract.json'),read(baseline.SOURCE))
 init=read(source/'initialization.json');done=read(source/'completion.json')
 resolved=read(source/'resolved_config.json');actual=read(source/'resolved_native_args.json')
 validate_metadata(c,init,done,resolved,actual)
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules()
 checkpoint=source/'final_ssdg.pth'
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if payload['epoch']!=20 or payload['checkpoint_selection']!='final_only' or payload['a1_ema_successful_updates']!=4440:
  raise ValueError('Actual E20 checkpoint budget/selection differs')
 if payload['run_id']!=baseline.RUN or payload['candidate_id']!=c['row_id']:
  raise ValueError('Actual E20 checkpoint run/seed identity differs')
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_ce_epoch2040.source import clean
 with installed(c) as native:
  from scripts.train_daot_rc4_baseline import resolved_config
  if clean(resolved_config(SimpleNamespace(**payload['args'])))!=actual:
   raise ValueError('Actual E20 checkpoint arguments differ from recorded native args')
  smoke=None
  if build_smoke:
   model=native.build_baseline_model(SimpleNamespace(**payload['baseline_args']),torch.device('cpu'))
   model.load_state_dict(payload['model'],strict=True);model.eval()
   with torch.no_grad():logits=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('E20 strict-load smoke failed')
   smoke=dict(strict_load=True,finite_logits=True,shape=list(logits.shape),target_access=False)
 return dict(status='VERIFIED',seed=seed,checkpoint=str(checkpoint),source_run=baseline.RUN,execution_commit=EXECUTION_COMMIT,
  epoch=20,optimizer_steps=4440,scratch_only=True,ancestors=[],source_roles='EXACT_MATCH',
  training_target_access=False,checkpoint_selection='fixed final20; all original matrix rows evaluated',
  method='pure CE, original gates/reference_response, cosine schedule',auxiliary_inheritance=False,smoke=smoke,
  selection_context='E20 now explicitly selected by user after target-informed epoch-budget exploration; exposed development benchmark, not a new blind confirmation')
