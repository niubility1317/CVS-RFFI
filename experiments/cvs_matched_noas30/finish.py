"""Mandatory no-As reference reports all seven views,without candidate ranking."""
from . import design as d
def run():
 from .evaluate import preflight
 rid=d.rows()[0]['row_id'];preflight(rid);R=d.BASE/rid;s=d.read(R/'scores/metrics.json');c=d.read(R/'source/training_result.json')
 if c['As_training_observations']!=0 or c['Ag_training_observations']!=0 or c['extra_supervised_VJP_calls']!=4440 or c['gradient_projection_nonzero']!=0:raise ValueError('Reference must have zero As/Ag and neutral twoVJP measurement')
 summary={r['view']:dict(accuracy=r['accuracy'],macro_F1=r['macro_f1'],worst_RX_accuracy=min(x['accuracy'] for x in s['records'] if x['view']==r['view'] and x['dimension']=='receiver')) for r in s['records'] if r['dimension']=='overall'}
 return dict(status='ARTIFACTS_COMPLETE',summary={'B00':summary},training_cost={'B00':c},reference_for_run=d.config(d.rows()[0])['reference_for_run'],mandatory_reference=True,no_candidate_promotion=True,parent_model_epochs_shared=20,new_model_epochs=10,new_updates=2220,cumulative_epochs_each=30,new_Ag=False,As=False,joint_goal_complete=False,target_feedback=False)
