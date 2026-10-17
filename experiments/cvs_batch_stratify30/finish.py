"""All three fixed endpoints report together;no target feedback to training."""
from . import design as d
def run():
 import numpy as np
 from .evaluate import preflight
 summary={};cost={}
 for row in d.rows():
  rid=row['row_id'];preflight(rid);p=d.BASE/rid;score=d.read(p/'scores/metrics.json');mode=row['batch_mode']
  summary[mode]={x['view']:dict(accuracy=x['accuracy'],macro_F1=x['macro_f1'],worst_RX_accuracy=min(y['accuracy'] for y in score['records'] if y['view']==x['view'] and y['dimension']=='receiver')) for x in score['records'] if x['dimension']=='overall'};cost[mode]=d.read(p/'source/training_result.json')
 paired={};criteria={}
 for reference in ('ORIGINAL','RANDOM'):
  left=d.BASE/('TX_RX_BALANCED-s'+str(d.SEED));right=d.BASE/(reference+'-s'+str(d.SEED));pairs={}
  with np.load(left/'scores/correctness.npz',allow_pickle=False) as a,np.load(right/'scores/correctness.npz',allow_pickle=False) as b:
   if not np.array_equal(a['ids'],b['ids']):raise ValueError('Same physical decisions required')
   for view in d.VIEWS:
    x,y=a[view],b[view];pairs[view]=dict(repairs=int((x&~y).sum()),new_errors=int((~x&y).sum()),delta_accuracy_pp=float((x.astype(int)-y.astype(int)).mean()*100),joint_clean_and_scene_delta_pp=float(((x&a['clean']).astype(int)-(y&b['clean']).astype(int)).mean()*100))
  paired[reference]=pairs;criteria[reference]=dict(clean=pairs['clean']['delta_accuracy_pp']>=0.,low_urban=pairs['practical_low_urban']['delta_accuracy_pp']>=.5,mid_urban=pairs['practical_mid_urban']['delta_accuracy_pp']>=0.,suburban=all(pairs[v]['delta_accuracy_pp']>=-.25 for v in d.VIEWS[1:4]))
 passed=all(v for checks in criteria.values() for v in checks.values())
 return dict(status='ARTIFACTS_COMPLETE',summary=summary,paired_differences=paired,criteria=criteria,scientific_pass=passed,decision='FIXED_SINGLE_SEED_RESULT_ONLY_NO_AUTOMATIC_PROMOTION' if passed else 'CLOSED_FIXED_BATCH_REPARTITION_FAILED_RECOGNITION_BENEFIT',training_cost=cost,parent_model_epochs_shared=20,new_model_epochs=30,new_updates=6660,cumulative_epochs_each=30,new_Ag=False,Ag_efficacy_proved=False,joint_goal_complete=False,target_feedback=False,no_automatic_seed_extension=True)
