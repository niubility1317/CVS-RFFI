"""Save source readout before formal truth; scientific failure never skips all7."""
import csv
import numpy as np
from . import design as d

def source_decision():
 row=d.rows()[0];root=d.BASE/row['row_id'];done=d.read(root/'source/completion.json');q=d.read(root/'source_qualification/complete.json');ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 old_as=d.read(ref/'B01-s2026092701/source_qualification/complete.json');no_as=d.read(ref/'B00-s2026092701/source/completion.json')
 if done['target_access'] or no_as['target_access'] or old_as['config']!=d.old_config() or q['config']!=d.config(row) or q['target_access'] or q['formal_scores_read'] or q['NN']!=12600 or q['training_updates'] or done['final_source_metrics']['source_val_count']!=27000 or no_as['final_source_metrics']['source_val_count']!=27000 or old_as['target_access'] or old_as['formal_scores_read']:raise ValueError('Only frozen exact source readouts may determine source result')
 now=np.asarray(q['source_accuracies'],float);old=np.asarray(old_as['source_accuracies'],float)
 if now.shape!=(2,3) or old.shape!=(2,3) or not np.isfinite(now).all() or not np.isfinite(old).all():raise ValueError('Complete source two-draw/three-scene readout required')
 clean_pp=100*(done['final_source_metrics']['source_val_accuracy']-no_as['final_source_metrics']['source_val_accuracy']);midlow_pp=100*(now[:,1:].mean()-old[:,1:].mean());low_pp=100*(now[:,2].mean()-old[:,2].mean());checks=dict(fullV_clean_within_0p1pp=clean_pp>=-.1-1e-12,mid_low_mean_gain_at_least_0p5pp=midlow_pp>=.5-1e-12,low_not_below_originalAs30=low_pp>=-1e-12)
 result=dict(status='SOURCE_READOUT_DECIDED',checks=checks,source_qualified=all(checks.values()),fullV_clean_vs_NoAs30_pp=clean_pp,mid_low_mean_vs_As30_pp=midlow_pp,low_vs_As30_pp=low_pp,source_accuracies=now.tolist(),originalAs30_source_accuracies=old.tolist(),formal_scores_read=False,formal_evaluation_always_required=True,no_automatic_budget_or_seed_extension=True,target_feedback=False,model_seed_count=1,NN=0)
 d.write(d.BASE/'source_matrix_decision.json',result);return result

from pathlib import Path

def collect():
 records=[];values={};cost={};refs=[('URBAN',d.BASE/d.rows()[0]['row_id']),('AS30',Path(d.PROJECT)/'runs'/d.REFERENCE_RUN/'B01-s2026092701'),('NOAS30',Path(d.PROJECT)/'runs'/d.REFERENCE_RUN/'B00-s2026092701')]
 for label,root in refs:
  prediction=d.read(root/'prediction/complete.json');scoring=d.read(root/'scoring/complete.json')
  if prediction['status']!='PREDICTIONS_COMPLETE' or prediction['views']!=list(d.VIEWS) or prediction['truth_read'] or scoring['status']!='SCORED_COMPLETE' or scoring['views']!=list(d.VIEWS) or not scoring['truth_last']:raise ValueError('All7 immutable predictions and independent truthlast scores required')
  rs=d.read(root/'scoring/scores.json')['results'];values[label]={r['view']:r for r in rs if r['dimension']=='overall'}
  if set(values[label])!=set(d.VIEWS):raise ValueError('Seven actual view results required')
  records.extend(dict(r,comparison_arm=label,reused=label!='URBAN') for r in rs)
 cost['new10']=d.read(d.BASE/d.rows()[0]['row_id']/'source/result.json');cost['inherited20']=d.read(Path(d.old_config()['parent_source'])/'result.json')
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['comparison_arm','reused','row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'],extrasaction='ignore');writer.writeheader();writer.writerows(records)
 source=d.read(d.BASE/'source_matrix_decision.json') if (d.BASE/'source_matrix_decision.json').is_file() else None
 result=dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS' if source else 'FORMAL_COMPLETE_SOURCE_PARTIAL',views=values,cost=cost,source_decision=source,reference_formal_artifacts_reused=True,reference_reprediction=0,reference_rescoring=0,new_epochs=10,parent_epochs=20,cumulative_epochs=30,new_updates=2220,parent_updates=4440,cumulative_updates=6660,model_seed_count=1,source_FAIL_does_not_skip_formal=True,formal_feedback=False,crossseed_confirmed=False,goal_complete=False)
 d.write(d.BASE/'results.json',result);return result
