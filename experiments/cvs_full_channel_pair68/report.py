"""Source qualification for independent full As; target never chooses continuation."""
import csv
from pathlib import Path
import numpy as np
from . import design as d

def logical_rows():
 ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 same=Path(d.PROJECT)/'runs'/d.SAME_RUN/'AS_CE-s2026092701'
 new=d.rows()[0]
 return [('NOAS30',ref/'B00-s2026092701',d.reference_config('B00'),True),
  ('AS30',ref/'B01-s2026092701',d.reference_config('B01'),True),
  ('SAME_CE6',same,d.reference_config('SAME_CE6'),True),
  ('INDEP_CE6',d.BASE/new['row_id'],d.config(new),False)]

def source_partial(error=None,missing=None):
 r=dict(status='SOURCE_READOUT_PARTIAL',source_qualified=False,error=error,missing=missing or [],formal_evaluation_always_required=True,formal_scores_read=False,target_scores_read=False,E50_only_after_source_PASS=True,no_automatic_seed_extension=True,model_seed_count=1,NN=0,target_feedback=False)
 d.write(d.BASE/'source_matrix_decision.json',r);return r

def source_values(root,c):
 s,q=d.read(root/'source/completion.json'),d.read(root/'source_qualification/complete.json')
 a=np.asarray(q['source_accuracies'],float)
 if (s['config']!=c or q['config']!=c or s['target_access'] or q['target_access'] or q['formal_scores_read'] or q['training_updates'] or q['NN']!=12600 or s['final_source_metrics']['source_val_count']!=27000 or a.shape!=(2,3) or not np.isfinite(a).all()):raise ValueError('Exact fixed source-only fullV/panel references required')
 return dict(fullV_clean=s['final_source_metrics']['source_val_accuracy'],source_accuracies=a.tolist(),midlow=float(a[:,1:].mean()),low=float(a[:,2].mean()))

def decide_values(vals):
 new,same,old,noas=[vals[k] for k in ('INDEP_CE6','SAME_CE6','AS30','NOAS30')]
 effects=dict(fullV_clean_vs_NoAs30_pp=100*(new['fullV_clean']-noas['fullV_clean']),
  fullV_clean_vs_same_CE6_pp=100*(new['fullV_clean']-same['fullV_clean']),
  midlow_vs_same_CE6_pp=100*(new['midlow']-same['midlow']),
  midlow_vs_original_As30_pp=100*(new['midlow']-old['midlow']),
  midlow_vs_best_fixed_reference_pp=100*(new['midlow']-max(old['midlow'],same['midlow'])),
  low_vs_same_CE6_pp=100*(new['low']-same['low']),low_vs_original_As30_pp=100*(new['low']-old['low']))
 checks=dict(fullV_clean_vs_NoAs30_at_least_minus0p1pp=effects['fullV_clean_vs_NoAs30_pp']>=-.1-1e-12,
  midlow_vs_best_fixed_reference_at_least_plus0p5pp=effects['midlow_vs_best_fixed_reference_pp']>=.5-1e-12)
 return effects,checks

def source_decision():
 refs=logical_rows();missing=[str(root/p) for _,root,_,_ in refs for p in ('source/completion.json','source_qualification/complete.json') if not (root/p).is_file()]
 if missing:return source_partial(missing=missing)
 vals={label:source_values(root,c) for label,root,c,_ in refs};effects,checks=decide_values(vals)
 result=dict(status='SOURCE_READOUT_DECIDED',source_qualified=all(checks.values()),candidate='INDEP_CE6',rows=vals,checks=checks,effects=effects,
  primary_same_parent_comparison='INDEP_CE6 minus SAME_CE6; both own SAT18 AS_CE20 plus10',original_As30_parent_history='Original P43 fullKL20 differs from AS_CE20; not pure channel-view attribution',
  formal_scores_read=False,target_scores_read=False,formal_evaluation_always_required=True,target_feedback=False,model_seed_count=1,NN=0,
  E50_total_allowed_only_after_source_PASS=True,E50_requires_same_budget_controls=True,no_automatic_seed_extension=True)
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 records,values,cost,missing=[],{},{},[]
 for label,root,c,reused in logical_rows():
  absent=[str(root/p) for p in ('prediction/complete.json','scoring/complete.json') if not (root/p).is_file()]
  if absent:missing.append(dict(arm=label,artifacts=absent));continue
  prediction,scoring=d.read(root/'prediction/complete.json'),d.read(root/'scoring/complete.json')
  if prediction['status']!='PREDICTIONS_COMPLETE' or prediction['views']!=list(d.VIEWS) or prediction['truth_read'] or prediction['query_fit'] or scoring['status']!='SCORED_COMPLETE' or scoring['views']!=list(d.VIEWS) or not scoring['truth_last']:raise ValueError('All seven fixed predictions and independent truth-last scores required')
  rs=d.read(root/'scoring/scores.json')['results'];values[label]={r['view']:r for r in rs if r['dimension']=='overall'}
  if set(values[label])!=set(d.VIEWS):raise ValueError('Seven actual formal views required')
  records.extend(dict(r,comparison_arm=label,reused=reused) for r in rs)
  if not reused:cost[label]=dict(new10=d.read(root/'source/result.json'),inherited20=d.read(Path(c['parent_source'])/'result.json'))
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['comparison_arm','reused','row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'],extrasaction='ignore');writer.writeheader();writer.writerows(records)
 source=d.read(d.BASE/'source_matrix_decision.json');status='FORMAL_PARTIAL' if missing else 'SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS' if source['status']=='SOURCE_READOUT_DECIDED' else 'FORMAL_COMPLETE_SOURCE_PARTIAL'
 result=dict(status=status,missing_rows=missing,views=values,cost=cost,source_decision=source,reference_formal_artifacts_reused=True,reference_reprediction=0,reference_rescoring=0,
  new_model_rows=1,reused_controls=3,new_epochs_per_row=10,parent_epochs_per_row=20,cumulative_epochs_per_row=30,new_updates_per_row=2220,parent_updates_per_row=4440,cumulative_updates_per_row=6660,
  same_parent_mechanism_comparison='INDEP_CE6 versus SAT18 SAME_CE6',originalAs30_has_different_parent_KL_history=True,
  model_seed_count=1,source_FAIL_does_not_skip_formal=True,formal_feedback=False,crossseed_confirmed=False,goal_complete=False)
 d.write(d.BASE/'results.json',result);return result
