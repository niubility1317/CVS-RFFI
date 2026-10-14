"""Read-only complete saved-log/default-prediction recount after row scoring."""
import gzip,json,math,time
import numpy as np
from . import design as default_design
from .predict import source_proof,preflight,baseline
from experiments.cvs_identity_mixstyle50.audit import pseudo_count,metrics
from experiments.cvs_pa_cross_delay.score import row_completion,validate_records

def run(*,design=None):
 d=design or default_design;started=time.time()
 # All rows must have scored immutable predictions before truth can be read.
 for row in d.rows():
  c=d.config(row);p=d.BASE/row['row_id'];saved=d.read(p/'scores/metrics.json')
  count=validate_records(c,saved['records'],design=d)
  if d.read(p/'completion.json')!=row_completion(c,count,design=d) or not saved['truth_last'] or saved['independent_recount']!='VERIFIED' or not (p/'prediction/predictions.npz').is_file():raise ValueError('Full scored row matrix required before recount')
 from experiments.cvs_report_ssl.evaluate import truth_arrays
 rows=[];records=[];decisions={};truth=None;common_ids=None;step_total=0
 for row in d.rows():
  c=d.config(row);p=d.BASE/row['row_id'];source=source_proof(c,design=d);preflight(row['row_id'],design=d)
  epochs=[json.loads(line) for line in (source/'epoch_metrics.jsonl').read_text().splitlines()]
  if [e['epoch'] for e in epochs]!=list(range(1,51)) or any(e['test_tx_total']!=0 or e['test_tx_acc'] is not None or e['val_tx_total']!=27000 for e in epochs):raise ValueError('Source epoch health/no-target history differs')
  accepted=sum(int(e.get('train_pseudo_selected',0)) for e in epochs);total=sum(int(e.get('train_pseudo_total',0)) for e in epochs)
  if total!=963900 or any(e.get('train_pseudo_total',0)!=(0 if e['epoch']<=33 else 56700) or e.get('train_pseudo_truth_available',0)!=0 for e in epochs):raise ValueError('Fixed hidden-U budget differs')
  guard=None
  if c['source_config'] is None:
   steps=u=selected=0
   with gzip.open(source/'step_metrics.jsonl.gz','rt') as stream:
    for line in stream:
     v=json.loads(line);steps+=1;m=v['metrics']
     if v['step']!=steps or v['epoch']!=(steps-1)//222+1 or not math.isfinite(m['train/loss']) or m['train/skipped_nonfinite_loss'] or m['train/skipped_nonfinite_grad']:raise ValueError('Actual update/nonfinite record differs')
     if m['train/a1_ema_successful_updates']!=steps or abs(m['train/a1_ema_effective_decay']-.999)>1e-10 or m['train/pseudo_total']!=pseudo_count(steps) or m.get('train/pseudo_truth_available',0)!=0:raise ValueError('Actual oldgate/EMA/hidden-U record differs')
     u+=int(m['train/pseudo_total']);selected+=int(m['train/pseudo_selected'])
   if steps!=11100 or u!=total or selected!=accepted:raise ValueError('Complete saved step/epoch counts differ')
   step_total+=steps;done=d.read(source/'completion.json')
   guard={k:done[k] for k in ('native_fitted_probes_disabled','native_disabled_probe_calls','head_fit_updates','L_context_steps')}
   if guard!=dict(native_fitted_probes_disabled=True,native_disabled_probe_calls=1,head_fit_updates=0,L_context_steps=11100):raise ValueError('Default joint head/probe closure differs')
  saved=d.read(p/'scores/metrics.json')['records']
  with np.load(p/'prediction/predictions.npz',allow_pickle=False) as f:
   ids=f['ids'];pred=f['ORIGINAL_HEAD']
   if set(f.files)!={'ids','ORIGINAL_HEAD'} or len(ids)!=168000 or len(np.unique(ids))!=168000:raise ValueError('Default decisions/physical IDs differ')
   if common_ids is None:common_ids=ids.copy();truth=truth_arrays(d.read(d.TRUTH),ids)
   elif not np.array_equal(common_ids,ids):raise ValueError('Same-row physical IDs differ')
   y,rx,day=truth;decisions[c['arm'],c['model_seed']]=pred.copy()
   if c['source_config'] is not None and not np.array_equal(pred,baseline(c,ids,design=d)):raise ValueError('Historical baseline decisions differ')
   for v in saved:
    if v['readout']!='ORIGINAL_HEAD' or v['view']!='clean':raise ValueError('Default clean readout differs')
    mask=np.ones(len(y),dtype=bool) if v['dimension']=='overall' else rx==v['stratum'] if v['dimension']=='receiver' else day==v['stratum'] if v['dimension']=='day' else y==list(d.CLASSES).index(v['stratum'])
    cm,n,accuracy,f1=metrics(y,pred,mask)
    if v['confusion']!=cm.tolist() or v['query_count']!=n or v['accuracy']!=accuracy or v['macro_f1']!=f1:raise ValueError('Independent metric recount differs')
  records.extend(saved)
  keys=('epoch','epoch_time_s','val_tx_acc','train_loss_tx_labeled','train_loss_tx_unlabeled','train_pseudo_total','train_pseudo_selected','train_pseudo_truth_available','train_reliable_ratio','train_pseudo_conf','train_strong_pass','train_temporal_pass')
  rows.append(dict(row_id=row['row_id'],arm=c['arm'],seed=c['model_seed'],source_reused=c['source_config'] is not None,epochs=50,updates=11100,U_packets=total,U_accepted=accepted,source_epoch_curve=[{k:e[k] for k in keys if k in e} for e in epochs],native_default_head_guard=guard))
 errors=[]
 for arm,control in d.PAIRS:
  for seed in d.SEEDS:
   good=decisions[arm,seed]==truth[0];old=decisions[control,seed]==truth[0]
   errors.append(dict(arm=arm,control=control,seed=seed,corrected_errors=int((good&~old).sum()),introduced_errors=int((~good&old).sum()),both_correct=int((good&old).sum()),both_wrong=int((~good&~old).sum()),count=len(good)))
 return dict(status='VERIFIED',run_id=d.RUN,models=len(rows),rows=rows,metrics=records,error_transitions=errors,new_step_records_parsed=step_total,epoch_records_parsed=50*len(rows),prediction_count=168000*len(rows),metric_records_independently_recounted=18*len(rows),truth_last_recount=True,head_fit=False,native_fitted_probes_disabled_for_new_models=True,optimizer_updates_by_analysis=0,scope='Completed immutable prediction and full source scalar logs only; no new training/target forwards, no feedback to methods/seeds/configuration',seconds=time.time()-started)
