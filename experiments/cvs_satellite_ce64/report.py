"""Source-only fixed objective-shape decision before any formal comparison."""
import csv
from pathlib import Path
import numpy as np
from . import design as d

def source_partial(error=None,missing=None):
 result=dict(status='SOURCE_READOUT_PARTIAL',source_qualified=False,error=error,missing=missing or [],formal_evaluation_always_required=True,formal_scores_read=False,target_scores_read=False,no_automatic_budget_or_seed_extension=True,model_seed_count=1,NN=0,target_feedback=False)
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def source_decision():
 ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 paths=[ref/'B00-s2026092701/source/completion.json',ref/'B01-s2026092701/source_qualification/complete.json']
 for row in d.rows():paths.extend([d.BASE/row['row_id']/'source/completion.json',d.BASE/row['row_id']/'source_qualification/complete.json'])
 missing=[str(p) for p in paths if not p.is_file()]
 if missing:return source_partial(missing=missing)
 no,old,*new=[d.read(p) for p in paths]
 if no['target_access'] or old['config']!=d.old_config() or old['target_access'] or old['formal_scores_read'] or old['training_updates'] or old['NN']!=12600 or no['final_source_metrics']['source_val_count']!=27000:raise ValueError('Only matching frozen source references may decide')
 previous=np.asarray(old['source_accuracies'],float)
 if previous.shape!=(2,3) or not np.isfinite(previous).all():raise ValueError('Reference source panel shape differs')
 values={}
 for i,row in enumerate(d.rows()):
  yes,q=new[2*i:2*i+2];c=d.config(row)
  if yes['config']!=c or q['config']!=c or yes['target_access'] or q['target_access'] or q['formal_scores_read'] or q['training_updates'] or q['NN']!=12600 or yes['final_source_metrics']['source_val_count']!=27000:raise ValueError('Only exact frozen source candidate may decide')
  now=np.asarray(q['source_accuracies'],float)
  if now.shape!=(2,3) or not np.isfinite(now).all():raise ValueError('Candidate source panel shape differs')
  values[row['arm']]=dict(fullV_AS_clean_vs_originalNOAS_pp=float(100*(yes['final_source_metrics']['source_val_accuracy']-no['final_source_metrics']['source_val_accuracy'])),mid_low_mean_vs_originalAs30_pp=float(100*(now[:,1:].mean()-previous[:,1:].mean())),low_vs_originalAs30_pp=float(100*(now[:,2].mean()-previous[:,2].mean())),source_accuracies=now.tolist())
 t=values['CE_T2'];vs=float(t['mid_low_mean_vs_originalAs30_pp']-values['CE_HALF']['mid_low_mean_vs_originalAs30_pp'])
 checks=dict(fullV_AS_clean_within_original_NOAS_0p1pp=bool(t['fullV_AS_clean_vs_originalNOAS_pp']>=-.1-1e-12),mid_low_mean_gain_at_least_0p5pp=bool(t['mid_low_mean_vs_originalAs30_pp']>=.5-1e-12),low_not_below_originalAs30=bool(t['low_vs_originalAs30_pp']>=-1e-12),mid_low_gain_beyond_half_CE_at_least_0p5pp=bool(vs>=.5-1e-12))
 result=dict(status='SOURCE_READOUT_DECIDED',checks=checks,source_qualified=bool(all(checks.values())),rows=values,t2_vs_half_mid_low_pp=vs,originalAs30_source_accuracies=previous.tolist(),control_not_promotion_candidate='CE_HALF',formal_scores_read=False,target_scores_read=False,formal_evaluation_always_required=True,no_automatic_budget_or_seed_extension=True,target_feedback=False,model_seed_count=1,NN=0)
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 records,values,cost,missing=[],{},{},[]
 refs=[('NOAS30',Path(d.PROJECT)/'runs'/d.REFERENCE_RUN/'B00-s2026092701',True),('AS30',Path(d.PROJECT)/'runs'/d.REFERENCE_RUN/'B01-s2026092701',True)]+[(r['arm'],d.BASE/r['row_id'],False) for r in d.rows()]
 for label,root,reused in refs:
  absent=[str(root/p) for p in ('prediction/complete.json','scoring/complete.json') if not (root/p).is_file()]
  if absent:missing.append(dict(arm=label,artifacts=absent));continue
  prediction,scoring=d.read(root/'prediction/complete.json'),d.read(root/'scoring/complete.json')
  if prediction['status']!='PREDICTIONS_COMPLETE' or prediction['views']!=list(d.VIEWS) or prediction['truth_read'] or prediction['query_fit'] or scoring['status']!='SCORED_COMPLETE' or scoring['views']!=list(d.VIEWS) or not scoring['truth_last']:raise ValueError('All seven fixed predictions and independent truth-last scores required')
  rs=d.read(root/'scoring/scores.json')['results'];values[label]={r['view']:r for r in rs if r['dimension']=='overall'}
  if set(values[label])!=set(d.VIEWS):raise ValueError('Seven actual formal views required')
  records.extend(dict(r,comparison_arm=label,reused=reused) for r in rs)
  if not reused:
   c=d.config(next(r for r in d.rows() if r['arm']==label));cost[label]=dict(new10=d.read(root/'source/result.json'),inherited20=d.read(Path(c['parent_source'])/'result.json'))
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['comparison_arm','reused','row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'],extrasaction='ignore');writer.writeheader();writer.writerows(records)
 source=d.read(d.BASE/'source_matrix_decision.json');status='FORMAL_PARTIAL' if missing else 'SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS' if source['status']=='SOURCE_READOUT_DECIDED' else 'FORMAL_COMPLETE_SOURCE_PARTIAL'
 result=dict(status=status,missing_rows=missing,views=values,cost=cost,source_decision=source,reference_formal_artifacts_reused=True,reference_reprediction=0,reference_rescoring=0,new_model_rows=2,new_epochs_per_row=10,parent_epochs_per_row=20,cumulative_epochs_per_row=30,new_updates_per_row=2220,parent_updates_per_row=4440,cumulative_updates_per_row=6660,model_seed_count=1,source_FAIL_does_not_skip_formal=True,formal_feedback=False,crossseed_confirmed=False,goal_complete=False)
 d.write(d.BASE/'results.json',result);return result
