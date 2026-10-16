"""Recount completed F target rows only; no model, raw IQ or raw truth access."""
import json,statistics,time
from pathlib import Path
import numpy as np
RUN='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02';RELEASE='cvs_F_supcon50_clean_20261016_r02';EXECUTION='4df7b94d0fd8ce29056bd8236364e45afa92e07f'
def recount(project):
 project=Path(project);root=project/'runs'/RUN;release=project/'releases'/RELEASE
 if (release/'release_commit.txt').read_text().strip()!=EXECUTION:raise ValueError('Wrong F execution release')
 done=json.loads((root/'completion.json').read_text());result=json.loads((root/'result.json').read_text());registered=json.loads((release/'registration/experiment.json').read_text())
 if done.get('status')!='ARTIFACTS_COMPLETE' or done.get('rows')!=9 or done.get('new_models')!=6 or done.get('new_optimizer_steps')!=66600:raise ValueError('Full fixed F target matrix is not complete')
 if registered['run_id']!=RUN or len(registered['rows'])!=9:raise ValueError('Fixed registration required')
 def normal(v):return json.loads(json.dumps(v).replace(str(project),'PROJECT'))
 rows=[];records=[];correct={};ids=None
 # Verify every prediction completion before reading any independent scores.
 for row in registered['rows']:
  rid=row['row_id'];p=root/rid;rd=json.loads((p/'completion.json').read_text());pr=json.loads((p/'prediction/complete.json').read_text());config=row['config']
  if rd['status']!='MODEL_PREDICTED_SCORED' or normal(rd['config'])!=config or normal(pr['config'])!=config or pr['status']!='PREDICTIONS_COMPLETE' or pr['count']!=168000 or pr['views']!=['clean'] or pr['readouts']!=['NATIVE_HEAD'] or pr['truth_read'] or pr['query_fit'] or pr['head_fit_updates'] or rd['head_fit_updates']:raise ValueError('Complete frozen default-head decisions required before recount')
  if rd['new_training_steps']!=(0 if config['baseline_reuse'] else 11100):raise ValueError('Wrong actual fixed row budget')
 for row in registered['rows']:
  rid=row['row_id'];c=row['config'];p=root/rid;scores=json.loads((p/'scores/metrics.json').read_text())
  if not scores['truth_last'] or len(scores['records'])!=18:raise ValueError('18 independent same-row strata required')
  for r in scores['records']:
   cm=np.asarray(r['confusion']);n=int(cm.sum());hits=int(np.trace(cm));den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()
   if cm.shape!=(6,6) or cm.dtype.kind not in 'iu' or (cm<0).any() or n!=r['query_count'] or abs(hits/n-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Independent integer confusion recount mismatch')
  overall=next(r for r in scores['records'] if r['dimension']=='overall')
  with np.load(p/'scores/correctness.npz',allow_pickle=False) as cor,np.load(p/'prediction/predictions.npz',allow_pickle=False) as pred:
   if ids is None:ids=cor['ids'].copy()
   if set(pred.files)!={'ids','NATIVE_HEAD'} or not np.array_equal(ids,cor['ids']) or not np.array_equal(ids,pred['ids']) or len(ids)!=168000 or cor['NATIVE_HEAD'].dtype.kind!='b' or cor['NATIVE_HEAD'].shape!=(168000,) or pred['NATIVE_HEAD'].shape!=(168000,) or pred['NATIVE_HEAD'].dtype.kind not in 'iu' or pred['NATIVE_HEAD'].min()<0 or pred['NATIVE_HEAD'].max()>=6:raise ValueError('Paired same physical IDs and valid decisions required')
   correct[(c['arm'],c['model_seed'])]=cor['NATIVE_HEAD'].copy()
  hits=int(correct[(c['arm'],c['model_seed'])].sum())
  if hits!=int(np.trace(np.asarray(overall['confusion']))):raise ValueError('Correctness differs from overall confusion')
  rows.append(dict(row_id=rid,arm=c['arm'],model_seed=c['model_seed'],correct=hits,accuracy=hits/168000.,macro_f1=overall['macro_f1'],worst_RX=min(r['accuracy'] for r in scores['records'] if r['dimension']=='receiver'),worst_day=min(r['accuracy'] for r in scores['records'] if r['dimension']=='day')));records+=scores['records']
 if records!=result['records']:raise ValueError('Aggregate differs from immutable same-row scores')
 def stats(v):return dict(mean=statistics.mean(v),sample_SD=statistics.stdev(v),n=len(v))
 summary={}
 for arm in ('BASE','ALL_TX','CROSS_RX'):
  selected=[v for v in rows if v['arm']==arm];summary[arm]=dict(per_seed=selected,**{k:stats([r[k] for r in selected]) for k in ('accuracy','macro_f1','worst_RX','worst_day')})
  for k in ('accuracy','macro_f1','worst_RX','worst_day'):
   for field in ('mean','sample_SD'):
    if abs(summary[arm][k][field]-result['summary'][arm][k][field])>1e-12:raise ValueError('Three-seed summary mismatch')
 comparisons={}
 for a,b in (('ALL_TX','BASE'),('CROSS_RX','BASE'),('CROSS_RX','ALL_TX')):
  values=[]
  for seed in (2026092701,2026092702,2026092703):
   fixed=int((correct[(a,seed)]&~correct[(b,seed)]).sum());broken=int((~correct[(a,seed)]&correct[(b,seed)]).sum());x=next(v for v in rows if v['arm']==a and v['model_seed']==seed);y=next(v for v in rows if v['arm']==b and v['model_seed']==seed);values.append(dict(model_seed=seed,corrected_errors=fixed,introduced_errors=broken,accuracy_difference_pp=(fixed-broken)/1680.,macro_f1_difference_pp=100*(x['macro_f1']-y['macro_f1']),worst_RX_difference_pp=100*(x['worst_RX']-y['worst_RX']),worst_day_difference_pp=100*(x['worst_day']-y['worst_day'])))
  name=a+'-'+b;v=dict(per_seed=values,accuracy_difference_pp=stats([x['accuracy_difference_pp'] for x in values]),positive_seeds=sum(x['accuracy_difference_pp']>0 for x in values));official=result['comparisons'][name]
  if values!=official['per_seed'] or v['positive_seeds']!=official['positive_seeds'] or any(abs(v['accuracy_difference_pp'][k]-official['accuracy_difference_pp'][k])>1e-12 for k in ('mean','sample_SD')):raise ValueError('Independent paired statistics differ')
  comparisons[name]=v
 return dict(status='VERIFIED',run_id=RUN,execution_commit=EXECUTION,rows=9,confusion_matrices_recounted=162,logical_target_decisions=1512000,new_target_decisions=1008000,paired_same_physical_IDs=True,summary=summary,comparisons=comparisons,truth_last=True,source_performance_gate=False,head_fit_updates=0,raw_IQ_reads=0,raw_truth_reads=0,new_model_forwards=0,new_optimizer_updates=0,read_at=time.time(),scope='Arithmetic on completed independent truth-last score/confusion/correctness and fixed prediction IDs; no training or candidate changes')
