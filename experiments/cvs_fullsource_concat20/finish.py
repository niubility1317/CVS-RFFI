"""Descriptive fixed matrix; no target-driven promotion, rerun or tuning."""
from . import design as d

def run():
 import numpy as np
 from .evaluate import preflight
 scores={};costs={};summary={};ids=None
 for row in d.rows():
  rid=row['row_id'];c,physical=preflight(rid);v=d.read(d.BASE/rid/'completion.json')
  if v['status']!='TRAINED_DEFAULT_PREDICTED_SCORED':raise ValueError('All registered rows must close')
  s=d.read(d.BASE/rid/'scores/metrics.json');scores[row['arm']]=s['records'];costs[row['arm']]=dict(training=d.read(d.BASE/rid/'source/training_result.json'),prediction=d.read(d.BASE/rid/'prediction/complete.json'),score_CPU_wall_seconds=s['CPU_wall_seconds'])
  summary[row['arm']]={r['view']:dict(accuracy=r['accuracy'],macro_F1=r['macro_f1'],worst_RX_accuracy=min(x['accuracy'] for x in s['records'] if x['view']==r['view'] and x['dimension']=='receiver')) for r in s['records'] if r['dimension']=='overall'}
  if ids is None:ids=physical
  elif not np.array_equal(ids,physical):raise ValueError('Matched physical decisions required')
 comparisons={}
 for a,b in (('CONCAT','REPEAT'),('REPEAT','NATIVE'),('CONCAT','NATIVE')):
  aa=d.BASE/(a+'-s'+str(d.SEED))/'scores/correctness.npz';bb=d.BASE/(b+'-s'+str(d.SEED))/'scores/correctness.npz';part={}
  with np.load(aa,allow_pickle=False) as x,np.load(bb,allow_pickle=False) as y:
   for view in d.VIEWS:
    repairs=int((x[view]&~y[view]).sum());breaks=int((~x[view]&y[view]).sum());part[view]=dict(delta_accuracy_pp=(repairs-breaks)*100/len(ids),repairs=repairs,new_errors=breaks,delta_macro_F1_pp=100*(summary[a][view]['macro_F1']-summary[b][view]['macro_F1']))
  part['urban_mean_gain_pp']=sum(part[v]['delta_accuracy_pp'] for v in d.SCENES[3:])/3;comparisons[a+'-'+b]=part
 gaps={a:{v:100*(s['clean']['accuracy']-s[v]['accuracy']) for v in d.SCENES[:3]} for a,s in summary.items()}
 return dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,model_seed_count=1,model_seed=d.SEED,source_training_RX=list(d.FIT),source_RX_holdout=None,summary=summary,comparisons=comparisons,suburban_gap_to_own_clean_pp=gaps,costs=costs,records=scores,truth_last=True,default_head_only=True,target_feedback=False,new_Ag_proved=False,joint_goal_complete=False,limit='Single model seed/full originalsource;descriptive fixed matrix,not new blind benchmark or proof of Ag/As joint efficacy')
