"""Verify saved final10/cumulative30 and its real logs;never rewrite failed native status."""
from pathlib import Path
import gzip,json,math
from . import design as d
W='id_backbone.encoder.core.id_backbone.cls_head.weight'
def validate_card(x,c):
 expected=dict(training_run=d.TRAIN_RUN,row_id=c['row_id'],epoch=10,state_epoch=10,successful_updates=2220,checkpoint_selection='final_only',checkpoint_role='training_final_only',parent=c['training_config']['parent_source'],source_roles_match=True,step_records=2220,step_first=1,step_last=2220,successful_steps=2220,nonfinite_steps=0,U_truth_available=0,epoch_records=10,progress_epoch=10,progress_steps=2220,head_shape=[6,160],head_changed=True,body_changed=True,optimizer_step_min=2220,optimizer_step_max=2220,default_head_joint=True,native_terminal='HELDOUT_EVAL_INCOMPLETE',native_reason='Unexpected checkpoint dependency',selected_final_exists=True,final_blocked=False,own_EMA_parent=True,source_initialization_parent=True)
 if any(x.get(k)!=v for k,v in expected.items()):raise ValueError('Incomplete/mismatched E30 source provenance or different failure')
 return x

def checkpoint(cp,c):
 if cp['run_id']!=d.TRAIN_RUN or cp['candidate_id']!=c['row_id'] or cp['epoch']!=10 or cp['a1_ema_successful_updates']!=2220 or cp['checkpoint_selection']!='final_only' or cp['checkpoint_role']!='training_final_only' or cp.get('ema_model') is None:raise ValueError('Fixed E30 checkpoint identity differs')
 state=cp.get('checkpoint_status',{})
 if state.get('state')!='FINAL_ONLY' or state.get('checkpoint_safe') is not True or state.get('phase1_v2_final_blocked') is not False:raise ValueError('Final checkpoint safety/role differs')
 a=cp['args'];p=c['training_config']['parent_source']
 if a['baseline_ckpt']!=str(Path(p)/'final_ssdg.pth') or a['teacher_ckpt'] or a['from_scratch'] or a.get('a1_scratch_only') is not False or a['freeze_backbone'] or (a['epochs'],a['label_epochs'],a['pseudo_epochs'],a['seed'])!=(10,0,10,d.SEED) or a.get('source_fold_held_rx') is not None or a.get('source_fold_training_receivers') or not a['use_ema_teacher'] or a['use_concat_sat_channel_aug']:raise ValueError('E30 data/parent/head/U actual args differ')
 return cp

def recover(c):
 import torch
 d.validate(c);p=Path(c['source_root']);d.trained.parent_proof(c['arm'])
 roles=d.read(p/'source_contract.json')['role_ids'];roles_ok=roles==d.read(d.SOURCE)['role_ids']
 cp=checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c)
 initial=torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'];steps=[]
 for s in cp['optimizer']['state'].values():
  if 'step' in s:steps.append(int(s['step']))
 if not steps or not all(bool(torch.isfinite(v).all()) for v in cp['model'].values()):raise ValueError('Nonfinite/incomplete final model')
 total=success=nonfinite=utruth=0;first=last=None;L_forward_packets=U_packets=As_packets=0
 with gzip.open(p/'step_metrics.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   x=json.loads(line);m=x['metrics'];step=x['step']
   if first is None:first=step
   if last is not None and step!=last+1:raise ValueError('Incomplete actual step sequence')
   last=step;total+=1;success+=int(m['train/optimizer_step_applied']);nonfinite+=int(m['train/skipped_nonfinite_grad'])+int(m['train/skipped_nonfinite_loss']);utruth+=int(m['train/pseudo_truth_available'])
   if m['concat/L_forwards']!=4:raise ValueError('Matched four L forwards required')
   L_forward_packets+=128*int(m['concat/L_forwards']);U_packets+=int(m['train/pseudo_total']);As_packets+=int(m['concat/As_packets'])
 epochs=[json.loads(x) for x in (p/'epoch_metrics.jsonl').read_text().splitlines() if x.strip()]
 if [x['epoch'] for x in epochs]!=list(range(1,11)):raise ValueError('All10completed epochs required')
 progress=d.read(p/'reference_progress.json');terminal=d.read(p/'phase1_terminal_status.json');held=d.read(p/'frozen_phase1_heldout_eval.json');init=d.read(p/'initialization.json');ema=d.read(p/'EMA_initialization.json')
 if Path(terminal['selected_checkpoint'])!=p/'final_ssdg.pth' or terminal['selection_source']!='training_final_only' or terminal['exit_code']!=7 or held['status']!='FAILED':raise ValueError('Only known saved-final loader failure recoverable')
 if progress['L_packets']!=L_forward_packets or progress['As_packets']!=As_packets or progress['source_RX']!=list(d.FIT) or progress['source_RX_holdout'] is not None or progress['U_truth_read'] or progress['target_access']:raise ValueError('Actual source view budget differs')
 expected_as=284160 if c['arm']=='FULL_AS' else 142080 if c['arm'] in ('B01','B11') else 0
 expected_ag=142080 if c['arm'] in ('B10','B11') else 0
 if As_packets!=expected_as or progress['Ag_packets']!=expected_ag or U_packets!=567000:raise ValueError('Actual L/U/operator counts differ')
 parent=c['training_config']['parent_source'];parentCP=str(Path(parent)/'final_ssdg.pth')
 card=dict(training_run=cp['run_id'],row_id=cp['candidate_id'],epoch=cp['epoch'],state_epoch=cp['state_epoch'],successful_updates=cp['a1_ema_successful_updates'],checkpoint_selection=cp['checkpoint_selection'],checkpoint_role=cp['checkpoint_role'],parent=parent,source_roles_match=roles_ok,step_records=total,step_first=first,step_last=last,successful_steps=success,nonfinite_steps=nonfinite,U_truth_available=utruth,epoch_records=len(epochs),progress_epoch=progress['epoch'],progress_steps=progress['steps'],head_shape=list(cp['model'][W].shape),head_changed=not torch.equal(cp['model'][W],initial[W]),body_changed=any(not torch.equal(v,initial[k]) for k,v in cp['model'].items() if k!=W),optimizer_step_min=min(steps),optimizer_step_max=max(steps),default_head_joint=not cp['args']['freeze_backbone'],native_terminal=terminal['status'],native_reason=held['reason'],selected_final_exists=terminal['selected_checkpoint_exists'],final_blocked=terminal['phase1_v2_final_blocked'],own_EMA_parent=ema['strict_own_parent'] and ema['parent']==parent,source_initialization_parent=not init['scratch_only'] and init['checkpoint_sources']==[parentCP] and init['source_roles']=='EXACT_MATCH' and not init['target_contact'])
 validate_card(card,c)
 value=dict(status='SOURCE_BUDGET_AND_FIXED_CHECKPOINT_VERIFIED',config=c,card=card,source_checkpoint=str(p/'final_ssdg.pth'),new_training_updates=0,head_fit_updates=0,truth_read=False,source_roles={k:len(v) for k,v in roles.items()},L_forward_packets=L_forward_packets,U_observations=U_packets,As_packets=As_packets,Ag_packets=expected_ag,sourceV_accuracy_percent=epochs[-1]['val_tx_acc'],native_status_preserved=True,unavailable_due_posttrain_abort=['inmemory clock MSE/scene coverage','reference_steps detailed records','perstep Sinc-unit records'],source_logs_closed=True)
 d.write(d.BASE/c['row_id']/'source_recovery.json',value);return value

def source(c):
 d.validate(c);v=d.read(d.BASE/c['row_id']/'source_recovery.json')
 if v['status']!='SOURCE_BUDGET_AND_FIXED_CHECKPOINT_VERIFIED' or v['config']!=c or v['new_training_updates'] or v['head_fit_updates'] or v['truth_read']:raise ValueError('No verified fixed source checkpoint')
 validate_card(v['card'],c);return Path(c['source_root'])
def load(c,device):
 import torch
 from experiments.cvs_clock_factorial30 import model as api
 p=source(c);cp=checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c)
 m=api.build(c['training_config'],device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
