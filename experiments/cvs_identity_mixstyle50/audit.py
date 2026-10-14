"""Independent saved-log/default-prediction recount; never trains or selects."""
import argparse,gzip,json,math,time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .predict import source_proof,preflight,baseline

def pseudo_count(step):return 0 if step<=33*222 else 124 if step%222==0 else 256
def style_budget(data,enabled):
 records=data['records']
 if [r['epoch'] for r in records]!=list(range(1,51)) or data['U_truth_read'] or data['head_fit'] or data['query_read'] or data['extra_V_passes']!=2:raise ValueError('Fixed source style epoch budget differs')
 totals={n:dict(calls=0,enabled_calls=0,packet_calls=0,eligible_packets=0,changed_packets=0) for n in d.STYLE_LAYERS}
 for r in records:
  if r['main_L_calls']!=(222 if enabled else 0) or r['teacher_style_calls'] or r['U_style_calls'] or r['eval_style_calls']:raise ValueError('Non-L or missing actual hook call')
  for n in d.STYLE_LAYERS:
   v=r['layers'][n]
   if v['calls']!=(222 if enabled else 0) or v['packet_calls']!=(28416 if enabled else 0) or not 0<=v['changed_packets']<=v['eligible_packets']<=v['packet_calls'] or not 0<=v['enabled_calls']<=v['calls']:raise ValueError('Actual style layer call/count differs')
   for k in totals[n]:totals[n][k]+=v[k]
 for r in records:
  if r['epoch'] in (33,34) and r['source_V_RX_health']['source_val_count']!=27000:raise ValueError('Source eval health shape differs')
 return dict(status='VERIFIED',enabled=enabled,layers=totals,teacher_U_eval_intervention=False,head_fit=False)

def metrics(y,pred,mask):
 cm=np.bincount(y[mask]*6+pred[mask],minlength=36).reshape(6,6);n=int(cm.sum());den=cm.sum(0)+cm.sum(1)
 return cm,n,float(np.trace(cm)/n),float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
def saved_numerical(c,source):
 from types import SimpleNamespace
 from .model import installed
 from experiments.cvs_phase1_overlay.model import native_modules
 from experiments.cvs_feature_diagnostics.sinc_change import compare
 from experiments.cvs_feature_diagnostics.sinc_optimizer import inspect_state
 native_modules();initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'];ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100 or ck['run_id']!=d.RUN or ck['candidate_id']!=c['row_id'] or ck['checkpoint_selection']!='final_only':raise ValueError('Own fixedfinal checkpoint differs')
 with installed(c) as native:
  model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),torch.device('cpu'));model.load_state_dict(ck['model'],strict=True)
  optimizer=inspect_state(model,ck,native);Sinc=type(model.id_backbone.encoder.core.id_backbone.sinc)
 return dict(status='VERIFIED',all_saved_optimizer_shapes_and_order_match=optimizer['all_saved_moment_shapes_match'],optimizer=optimizer,student_Sinc=compare(initial,ck['model'],Sinc),EMA_Sinc=compare(initial,ck['ema_model'],Sinc),student_EMA_Sinc=compare(ck['model'],ck['ema_model'],Sinc),optimizer_updates_by_analysis=0,IQ_read=False,query_read=False,head_fit=False)

def run(completed_only=False):
 from experiments.cvs_report_ssl.evaluate import truth_arrays
 torch.set_num_threads(2);start=time.time();available=[];missing=[]
 # Require scored immutable artifacts before any truth connection, per row.
 for r in d.rows():
  p=d.BASE/r['row_id']
  if (p/'completion.json').is_file() and (p/'scores'/'metrics.json').is_file() and (p/'prediction'/'predictions.npz').is_file():available.append(r)
  else:missing.append(r['row_id'])
 if missing and not completed_only:raise ValueError('Full preregistered matrix still incomplete; no truth read')
 if not available:raise ValueError('No scored row available; no truth read')
 rows=[];all_metrics=[];decisions={};truth=None;common_ids=None;steps_total=0;initials={};proofs={}
 for r in available:
  c=d.config(r);source=source_proof(c);preflight(r['row_id']);proofs[r['row_id']]=source
  epochs=[json.loads(line) for line in (source/'epoch_metrics.jsonl').read_text().splitlines()]
  assert [e['epoch'] for e in epochs]==list(range(1,51)) and all(e['test_tx_total']==0 and e['test_tx_acc'] is None and e['val_tx_total']==27000 for e in epochs)
  accepted=sum(int(e.get('train_pseudo_selected',0)) for e in epochs);total=sum(int(e.get('train_pseudo_total',0)) for e in epochs)
  assert total==963900 and all(e.get('train_pseudo_total',0)==(0 if e['epoch']<=33 else 56700) for e in epochs)
  U=sampling=numerical=None
  if c['source_config'] is None:
   steps=u=selected=0
   with gzip.open(source/'step_metrics.jsonl.gz','rt') as f:
    for line in f:
     v=json.loads(line);steps+=1;m=v['metrics'];assert v['step']==steps and v['epoch']==(steps-1)//222+1
     assert math.isfinite(m['train/loss']) and m['train/skipped_nonfinite_loss']==0 and m['train/skipped_nonfinite_grad']==0
     assert m['train/a1_ema_successful_updates']==steps and abs(m['train/a1_ema_effective_decay']-.999)<1e-10
     assert m['train/pseudo_total']==pseudo_count(steps) and m.get('train/pseudo_truth_available',0)==0
     u+=int(m['train/pseudo_total']);selected+=int(m['train/pseudo_selected'])
   assert steps==11100 and u==total and selected==accepted;steps_total+=steps
   U=d.read(source/'source_U_epoch_diagnostic.json');assert set(map(int,U['epochs']))==set(range(34,51)) and not U['U_truth_read'] and U['new_forward_calls']==0
   assert all(sum(v['count'] for v in group.values())==56700 for group in U['epochs'].values())
   assert sum(v['accepted'] for group in U['epochs'].values() for v in group.values())==accepted
   assert all(sum(v['predicted_class_counts'])==v['count'] and sum(v['accepted_predicted_class_counts'])==v['accepted'] for group in U['epochs'].values() for v in group.values())
   sampling=d.read(source/'source_style_epoch_diagnostic.json');style_budget(sampling,c['identity_mixstyle']['enabled'])
   assert all(q['source_V_RX_health']['source_val_count']==27000 for q in sampling['records'] if q['epoch'] in (33,34))
   numerical=saved_numerical(c,source);initials[c['arm'],c['model_seed']]=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)['model']
  saved_metrics=d.read(d.BASE/r['row_id']/'scores'/'metrics.json')['records'];assert len(saved_metrics)==18
  with np.load(d.BASE/r['row_id']/'prediction'/'predictions.npz',allow_pickle=False) as f:
   ids=f['ids'];pred=f['ORIGINAL_HEAD'];assert set(f.files)=={'ids','ORIGINAL_HEAD'} and len(ids)==168000 and len(np.unique(ids))==168000
   if common_ids is None:common_ids=ids.copy();truth=truth_arrays(d.read(d.TRUTH),ids)
   else:assert np.array_equal(common_ids,ids)
   y,rx,day=truth;decisions[c['arm'],c['model_seed']]=pred.copy()
   if c['source_config'] is not None:assert np.array_equal(pred,baseline(c,ids))
   for v in saved_metrics:
    assert v['readout']=='ORIGINAL_HEAD' and v['view']=='clean'
    mask=np.ones(len(y),dtype=bool) if v['dimension']=='overall' else rx==v['stratum'] if v['dimension']=='receiver' else day==v['stratum'] if v['dimension']=='day' else y==list(d.CLASSES).index(v['stratum'])
    cm,n,acc,f1=metrics(y,pred,mask);assert v['confusion']==cm.tolist() and v['query_count']==n and v['accuracy']==acc and v['macro_f1']==f1
  all_metrics.extend(saved_metrics)
  keys=('epoch','epoch_time_s','val_tx_acc','train_loss_tx_labeled','train_loss_tx_unlabeled','train_pseudo_total','train_pseudo_selected','train_pseudo_truth_available','train_a1_ema_successful_updates','train_reliable_ratio','train_pseudo_conf','train_strong_pass','train_temporal_pass')
  rows.append(dict(row_id=r['row_id'],arm=c['arm'],seed=c['model_seed'],source_reused=c['source_config'] is not None,epochs=50,updates=11100,U_packets=total,U_accepted=accepted,source_epoch_curve=[{k:e[k] for k in keys if k in e} for e in epochs],source_U_diagnostic=U,source_style_diagnostic=sampling,saved_optimizer_and_Sinc=numerical))
 errors=[];equal=[]
 for arm,control in d.PAIRS:
  for seed in d.SEEDS:
   if (arm,seed) not in decisions or (control,seed) not in decisions:continue
   good=decisions[arm,seed]==truth[0];old=decisions[control,seed]==truth[0]
   errors.append(dict(arm=arm,control=control,seed=seed,corrected_errors=int((good&~old).sum()),introduced_errors=int((~good&old).sum()),both_correct=int((good&old).sum()),both_wrong=int((~good&~old).sum()),count=len(good)))
 for seed in d.SEEDS:
  if ('MATCHED_CONTROL',seed) not in initials or ('IDENTITY_MIXSTYLE',seed) not in initials:continue
  a,b=initials['MATCHED_CONTROL',seed],initials['IDENTITY_MIXSTYLE',seed];assert set(a)==set(b) and all(torch.equal(v,b[k]) for k,v in a.items())
  equal.append(dict(seed=seed,actual_initial_states_identical=True,state_tensors=len(a)))
 return dict(status='PARTIAL' if missing else 'VERIFIED',run_id=d.RUN,models=len(rows),missing=missing,rows=rows,metrics=all_metrics,error_transitions=errors,actual_initial_equality=equal,new_step_records_parsed=steps_total,epoch_records_parsed=50*len(rows),prediction_count=168000*len(rows),metric_records_independently_recounted=18*len(rows),truth_last_recount=True,head_fit=False,optimizer_updates_by_analysis=0,scope='Actual complete artifacts only; partial rows never imply full three-seed goal achievement; no result feedback to training',seconds=time.time()-start)

def checks():
 assert pseudo_count(33*222)==0 and pseudo_count(33*222+1)==256 and pseudo_count(34*222)==124
 assert sum(pseudo_count(step) for step in range(1,11101))==963900
 y=np.array([0,1,1,2,5]);p=np.array([0,0,1,2,4]);cm,n,acc,f1=metrics(y,p,np.ones(5,dtype=bool));assert n==5 and acc==.6 and cm.sum()==5 and abs(f1-7/18)<1e-12
 return dict(status='PASS',synthetic_only=True,fixed_label_pseudo_tail_and_total=True,confusion_accuracy_macroF1_checked=True,no_actual_artifact_or_truth_read=True,head_fit=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checks',action='store_true');a=p.parse_args();print(json.dumps(checks() if a.checks else run(False)))
