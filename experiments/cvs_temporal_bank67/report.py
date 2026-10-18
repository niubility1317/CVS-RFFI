"""Same-physical confirmation source decision; formal results never choose the rule."""
import csv
from pathlib import Path
import numpy as np
from . import design as d

def source_partial(error=None,missing=None):
 r=dict(status='SOURCE_READOUT_PARTIAL',source_qualified=False,error=error,missing=missing or [],formal_evaluation_always_required=True,formal_scores_read=False,target_scores_read=False,E50_only_after_source_PASS=True,no_automatic_seed_extension=True,model_seed_count=1,NN=0,target_feedback=False)
 d.write(d.BASE/'source_matrix_decision.json',r);return r

def source_decision():
 ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 roots={'NOAS30':ref/'B00-s2026092701','AS30':ref/'B01-s2026092701',
        'NOAS_BANK':d.BASE/'B00-s2026092701','AS_BANK':d.BASE/'B01-s2026092701'}
 missing=[str(r/p) for r in roots.values() for p in ['source/completion.json','source_qualification/complete.json'] if not (r/p).is_file()]
 if missing:return source_partial(missing=missing)
 vals={}
 for name,r in roots.items():
  arm='B00' if name in ('NOAS30','NOAS_BANK') else 'B01'
  c=d.reference_config(arm) if name in ('NOAS30','AS30') else d.config(next(x for x in d.rows() if x['arm']==arm))
  s,q=d.read(r/'source/completion.json'),d.read(r/'source_qualification/complete.json')
  a=np.asarray(q['source_accuracies'],float)
  if (s['config']!=c or q['config']!=c or s['target_access'] or q['target_access'] or q['formal_scores_read'] or q['training_updates'] or q['NN']!=12600 or s['final_source_metrics']['source_val_count']!=27000 or a.shape!=(2,3) or not np.isfinite(a).all()):raise ValueError('Exact frozen source-only references/candidates required')
  vals[name]=dict(fullV_clean=s['final_source_metrics']['source_val_accuracy'],source_accuracies=a.tolist(),midlow=float(a[:,1:].mean()),low=float(a[:,2].mean()))
 main,control,old,oldno=[vals[k] for k in ['AS_BANK','NOAS_BANK','AS30','NOAS30']]
 effects=dict(fullV_clean_vs_matchedNoAs_bank_pp=100*(main['fullV_clean']-control['fullV_clean']),
  fullV_clean_vs_originalNoAs30_pp=100*(main['fullV_clean']-oldno['fullV_clean']),
  midlow_vs_originalAs30_pp=100*(main['midlow']-old['midlow']),low_vs_originalAs30_pp=100*(main['low']-old['low']),
  midlow_interaction_pp=100*(main['midlow']-control['midlow']-old['midlow']+oldno['midlow']))
 checks=dict(clean_within_matchedNoAs_0p1pp=effects['fullV_clean_vs_matchedNoAs_bank_pp']>=-.1-1e-12,
  clean_within_originalNoAs_0p1pp=effects['fullV_clean_vs_originalNoAs30_pp']>=-.1-1e-12,
  midlow_vs_originalAs30_at_least_0p5pp=effects['midlow_vs_originalAs30_pp']>=.5-1e-12,
  low_not_below_originalAs30=effects['low_vs_originalAs30_pp']>=-1e-12)
 result=dict(status='SOURCE_READOUT_DECIDED',source_qualified=all(checks.values()),candidate='AS_BANK',rows=vals,checks=checks,effects=effects,formal_scores_read=False,target_scores_read=False,formal_evaluation_always_required=True,target_feedback=False,model_seed_count=1,NN=0,E50_total_allowed_only_after_source_PASS=True,E50_requires_same_budget_controls=True,no_automatic_seed_extension=True)
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 records,values,cost,missing=[],{},{},[]
 refs=[('NOAS30',Path(d.PROJECT)/'runs'/d.REFERENCE_RUN/'B00-s2026092701',True),('AS30',Path(d.PROJECT)/'runs'/d.REFERENCE_RUN/'B01-s2026092701',True)]+[('NOAS_BANK' if r['arm']=='B00' else 'AS_BANK',d.BASE/r['row_id'],False) for r in d.rows()]
 for label,root,reused in refs:
  absent=[str(root/p) for p in ('prediction/complete.json','scoring/complete.json') if not (root/p).is_file()]
  if absent:missing.append(dict(arm=label,artifacts=absent));continue
  prediction,scoring=d.read(root/'prediction/complete.json'),d.read(root/'scoring/complete.json')
  if prediction['status']!='PREDICTIONS_COMPLETE' or prediction['views']!=list(d.VIEWS) or prediction['truth_read'] or prediction['query_fit'] or scoring['status']!='SCORED_COMPLETE' or scoring['views']!=list(d.VIEWS) or not scoring['truth_last']:raise ValueError('All seven fixed predictions and independent truth-last scores required')
  rs=d.read(root/'scoring/scores.json')['results'];values[label]={r['view']:r for r in rs if r['dimension']=='overall'}
  if set(values[label])!=set(d.VIEWS):raise ValueError('Seven actual formal views required')
  records.extend(dict(r,comparison_arm=label,reused=reused) for r in rs)
  if not reused:
   c=d.config(next(r for r in d.rows() if r['arm']==('B00' if label=='NOAS_BANK' else 'B01')));cost[label]=dict(new10=d.read(root/'source/result.json'),inherited20=d.read(Path(c['parent_source'])/'result.json'))
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['comparison_arm','reused','row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'],extrasaction='ignore');writer.writeheader();writer.writerows(records)
 source=d.read(d.BASE/'source_matrix_decision.json');status='FORMAL_PARTIAL' if missing else 'SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS' if source['status']=='SOURCE_READOUT_DECIDED' else 'FORMAL_COMPLETE_SOURCE_PARTIAL'
 result=dict(status=status,missing_rows=missing,views=values,cost=cost,source_decision=source,reference_formal_artifacts_reused=True,reference_reprediction=0,reference_rescoring=0,new_model_rows=2,new_epochs_per_row=10,parent_epochs_per_row=20,cumulative_epochs_per_row=30,new_updates_per_row=2220,parent_updates_per_row=4440,cumulative_updates_per_row=6660,model_seed_count=1,source_FAIL_does_not_skip_formal=True,formal_feedback=False,crossseed_confirmed=False,goal_complete=False)
 d.write(d.BASE/'results.json',result);return result
