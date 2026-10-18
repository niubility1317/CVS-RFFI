"""Source-only fixed decision precedes reading any new formal scores."""
import csv
from pathlib import Path
import numpy as np
from . import design as d

def roots():
 ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 return dict(NOAS30=ref/'B00-s2026092701',AS30=ref/'B01-s2026092701',
  AS_MEAN_CE=d.BASE/'C1-s2026092701',AS_PROBMEAN_CE=d.BASE/'P1-s2026092701')
def cfg(name):
 if name=='NOAS30':return d.reference_config('B00')
 if name=='AS30':return d.reference_config('B01')
 return d.config(next(r for r in d.rows() if r['arm']==('C1' if name=='AS_MEAN_CE' else 'P1')))
def source_partial(error=None,missing=None):
 result=dict(status='SOURCE_READOUT_PARTIAL',source_qualified=False,error=error,missing=missing or [],formal_scores_read=False,target_scores_read=False,
  formal_evaluation_always_required=True,E50_only_after_source_PASS=True,no_automatic_seed_extension=True,NN=0,target_feedback=False)
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def source_decision():
 rr=roots();missing=[str(r/p) for r in rr.values() for p in ('source/completion.json','source_qualification/complete.json') if not (r/p).is_file()]
 if missing:return source_partial(missing=missing)
 vals={}
 for name,r in rr.items():
  c=cfg(name);s,q=d.read(r/'source/completion.json'),d.read(r/'source_qualification/complete.json');a=np.asarray(q['source_accuracies'],float)
  if (s['config']!=c or q['config']!=c or s['target_access'] or q['target_access'] or q['formal_scores_read'] or q['training_updates'] or q['NN']!=12600 or s['final_source_metrics']['source_val_count']!=27000 or a.shape!=(2,3) or not np.isfinite(a).all()):raise ValueError('Exact frozen source-only rows required')
  vals[name]=dict(fullV_clean=s['final_source_metrics']['source_val_accuracy'],source_accuracies=a.tolist(),midlow=float(a[:,1:].mean()),low=float(a[:,2].mean()))
 m,c,no,old=[vals[k] for k in ('AS_PROBMEAN_CE','AS_MEAN_CE','NOAS30','AS30')]
 effects=dict(fullV_clean_probmean_vs_NoAs_pp=100*(m['fullV_clean']-no['fullV_clean']),
  fullV_clean_probmean_vs_mean_pp=100*(m['fullV_clean']-c['fullV_clean']),
  pooled_midlow_probmean_vs_mean_pp=100*(m['midlow']-c['midlow']),low_probmean_vs_mean_pp=100*(m['low']-c['low']),
  pooled_midlow_mean_vs_originalAs30_pp=100*(c['midlow']-old['midlow']))
 checks=dict(clean_within_NoAs_0p1pp=effects['fullV_clean_probmean_vs_NoAs_pp']>=-.1-1e-12,
  midlow_vs_mean_at_least_0p5pp=effects['pooled_midlow_probmean_vs_mean_pp']>=.5-1e-12)
 result=dict(status='SOURCE_READOUT_DECIDED',source_qualified=all(checks.values()),candidate='AS_PROBMEAN_CE',rows=vals,effects=effects,checks=checks,
  formal_scores_read=False,target_scores_read=False,formal_evaluation_always_required=True,target_feedback=False,model_seed_count=1,NN=0,
  E50_total_allowed_only_after_source_PASS=True,E50_requires_same_budget_controls=True,no_automatic_seed_extension=True,
  As_CE_aggregation_only=True,Dropout_KL_unchanged=True)
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 records,values,cost,missing=[],{},{},[]
 for label,root in roots().items():
  reused=label in ('NOAS30','AS30');absent=[str(root/p) for p in ('prediction/complete.json','scoring/complete.json') if not (root/p).is_file()]
  if absent:missing.append(dict(arm=label,artifacts=absent));continue
  prediction,scoring=d.read(root/'prediction/complete.json'),d.read(root/'scoring/complete.json')
  if prediction['status']!='PREDICTIONS_COMPLETE' or prediction['views']!=list(d.VIEWS) or prediction['truth_read'] or prediction['query_fit'] or scoring['status']!='SCORED_COMPLETE' or scoring['views']!=list(d.VIEWS) or not scoring['truth_last']:raise ValueError('All seven fixed predictions before independent truth-last scores required')
  rs=d.read(root/'scoring/scores.json')['results'];values[label]={r['view']:r for r in rs if r['dimension']=='overall'}
  if set(values[label])!=set(d.VIEWS):raise ValueError('Seven actual views required')
  records.extend(dict(r,comparison_arm=label,reused=reused) for r in rs)
  if not reused:cost[label]=dict(new10=d.read(root/'source/result.json'),inherited20=d.read(Path(cfg(label)['parent_source'])/'result.json'))
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['comparison_arm','reused','row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'],extrasaction='ignore');w.writeheader();w.writerows(records)
 source=d.read(d.BASE/'source_matrix_decision.json');status='FORMAL_PARTIAL' if missing else 'SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS' if source['status']=='SOURCE_READOUT_DECIDED' else 'FORMAL_COMPLETE_SOURCE_PARTIAL'
 result=dict(status=status,missing_rows=missing,views=values,cost=cost,source_decision=source,
  reference_formal_artifacts_reused=True,reference_reprediction=0,reference_rescoring=0,new_model_rows=2,
  new_epochs_per_row=10,parent_epochs_per_row=20,cumulative_epochs_per_row=30,new_updates_per_row=2220,
  parent_updates_per_row=4440,cumulative_updates_per_row=6660,shared_parent_count=1,
  model_seed_count=1,source_FAIL_does_not_skip_formal=True,formal_feedback=False,crossseed_confirmed=False,goal_complete=False)
 d.write(d.BASE/'results.json',result);return result
