"""Freeze source decision before any primary analysis reads formal metric values."""
from . import design as d
def run():
 from .qualify import matrix
 decision=matrix();rows=[]
 for r in d.rows():
  c=d.config(r);v=d.read(d.BASE/r['row_id']/'completion.json')
  if v['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or v['config']!=c or v['views']!=list(d.VIEWS) or v['count']!=168000 or not v['truth_last'] or v['head_fit_updates']:raise ValueError('All two fixed final rows and sevenviews required')
  rows.append(dict(row_id=r['row_id'],status=v['status'],scores='PROJECT/runs/'+d.RUN+'/'+r['row_id']+'/scores/metrics.json'))
 return dict(status='ARTIFACTS_COMPLETE_PENDING_SAME_ROW_ANALYSIS',rows=rows,source_decision=decision['decision'],source_pass=decision['source_pass'],formal_numeric_metrics_read=False,new_IQ_Ag_efficacy_not_yet_proven=True,target_feedback=False,joint_goal_complete=False)
