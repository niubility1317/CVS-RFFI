import os
from . import design as d
def finish():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU aggregation required')
 from experiments.cvs_sigtime_rdrop_legacy.evaluate import preflight
 records=[]
 for row in d.rows():
  rid=row['row_id'];preflight(rid);done=d.read(d.BASE/rid/'completion.json')
  if done['status']!='MODEL_PREDICTED_SCORED' or done['new_training_steps']!=11100:raise ValueError('Complete fixed E50 row required')
  records+=d.read(d.BASE/rid/'scores/metrics.json')['records']
 if len(records)!=72:raise ValueError('All four rows/72 strata required')
 for rid in d.OLD_SCORE_ROWS:
  if d.read(d.ORIGINAL_BASE/rid/'completion.json')['status']!='MODEL_PREDICTED_SCORED':raise ValueError('Original already-started row still lacks score')
 d.write(d.BASE/'result.json',dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,rows=4,records=records,totals=d.totals(),original_run_id=d.ORIGINAL_RUN,original_score_closure_rows=list(d.OLD_SCORE_ROWS),scientific_group_rows=18,logical_matrix_analysis_pending=True,goal_completed=False))
 d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',rows=4,metric_records=72,truth_last=True,head_fit_updates=0,original_score_closures_complete=True));print('TECHNICAL_RECOVERY_FOUR_ROWS_COMPLETE',flush=True)
