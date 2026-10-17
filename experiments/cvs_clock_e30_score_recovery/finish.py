"""Fixed factorial and mandatory full-As reference; scores never change the queue."""
from . import design as d

def run():
 import numpy as np
 from .evaluate import preflight
 scores={};paths={};costs={}
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));preflight(c['row_id']);base=d.BASE/c['row_id']
  complete=d.read(base/'completion.json')
  if complete['status']!='TRAINED_DEFAULT_PREDICTED_SCORED':raise ValueError('All registered rows must close')
  scores[arm]=d.read(base/'scores/metrics.json')['records'];paths[arm]=base
  costs[arm]=dict(new_training=False,source_recovery=d.read(base/'source_recovery.json'),prediction=d.read(base/'prediction/complete.json'))
 summary={a:{r['view']:dict(accuracy=r['accuracy'],macro_F1=r['macro_f1'],worst_RX_accuracy=min(x['accuracy'] for x in rows if x['view']==r['view'] and x['dimension']=='receiver')) for r in rows if r['dimension']=='overall'} for a,rows in scores.items()}
 interactions={v:100*(summary['B11'][v]['accuracy']-summary['B10'][v]['accuracy']-summary['B01'][v]['accuracy']+summary['B00'][v]['accuracy']) for v in d.VIEWS}
 comparisons={}
 for a,b in (('B10','B00'),('B01','B00'),('B11','B00'),('B11','B10'),('B11','B01'),('B11','FULL_AS')):
  with np.load(paths[a]/'scores/correctness.npz',allow_pickle=False) as x,np.load(paths[b]/'scores/correctness.npz',allow_pickle=False) as y:
   if not np.array_equal(x['ids'],y['ids']):raise ValueError('Matched physical IDs required')
   comparisons[a+'-'+b]={v:dict(repairs=int((x[v]&~y[v]).sum()),new_errors=int((~x[v]&y[v]).sum()),delta_accuracy_pp=float((x[v].astype(int)-y[v].astype(int)).mean()*100)) for v in d.VIEWS}
 return dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,model_seed_count=1,new_models=0,trained_model_run=d.TRAIN_RUN,reused_models=5,new_model_epochs=0,new_optimizer_updates=0,model_epochs_each=30,prior_training_model_epochs=150,source_RX=list(d.FIT),source_RX_holdout=None,summary=summary,accuracy_interaction_pp=interactions,comparisons=comparisons,costs=costs,records=scores,operator_marginal_fraction=.5,As_halfdose_in_B01_B11=True,full_As_reference_included=True,target_feedback=False,joint_goal_complete=False,limit='Fixed single-seed exploratory factorial;no automatic model promotion or seed/budget rescue')
