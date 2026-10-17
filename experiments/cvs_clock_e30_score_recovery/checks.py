"""Focused source-recovery and truth-closure negative tests;zero IQ/CP/NN/update."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from . import design as d,proof

def good(c):return dict(training_run=d.TRAIN_RUN,row_id=c['row_id'],epoch=10,state_epoch=10,successful_updates=2220,checkpoint_selection='final_only',checkpoint_role='training_final_only',parent=c['training_config']['parent_source'],source_roles_match=True,step_records=2220,step_first=1,step_last=2220,successful_steps=2220,nonfinite_steps=0,U_truth_available=0,epoch_records=10,progress_epoch=10,progress_steps=2220,head_shape=[6,160],head_changed=True,body_changed=True,optimizer_step_min=2220,optimizer_step_max=2220,default_head_joint=True,native_terminal='HELDOUT_EVAL_INCOMPLETE',native_reason='Unexpected checkpoint dependency',selected_final_exists=True,final_blocked=False,own_EMA_parent=True,source_initialization_parent=True)
def protocol():
 n=0
 for row in d.rows():
  c=d.config(row);d.validate(c);x=good(c);proof.validate_card(x,c)
  for key,value in (('training_run','UNKNOWN'),('row_id','OTHER'),('epoch',9),('state_epoch',9),('successful_updates',2219),('checkpoint_selection','best_target'),('checkpoint_role','latest'),('parent','UNKNOWN'),('source_roles_match',False),('step_records',2219),('nonfinite_steps',1),('U_truth_available',1),('head_shape',[6,112]),('head_changed',False),('body_changed',False),('optimizer_step_min',2219),('default_head_joint',False),('native_reason','CHECKPOINT_DATA_CONTRACT_MISMATCH'),('selected_final_exists',False),('final_blocked',True),('own_EMA_parent',False)):
   bad=deepcopy(x);bad[key]=value
   try:proof.validate_card(bad,c)
   except ValueError:n+=1
   else:raise AssertionError('Bad recovery provenance accepted '+key)
  for key,value in (('new_optimizer_updates',1),('training_run_id','OTHER'),('source_root','UNKNOWN')):
   bad=deepcopy(c);bad[key]=value
   try:d.validate(bad)
   except ValueError:n+=1
   else:raise AssertionError('Unregistered recovery accepted')
 return n

def run():
 from . import evaluate as ev
 n=protocol();original_read,original_ids,original_source=d.read,ev.ids,proof.source
 try:
  proof.source=lambda c:None;ev.ids=lambda:np.arange(ev.COUNT)
  valid=dict(status='PREDICTIONS_COMPLETE',config=d.config(d.rows()[0]),count=ev.COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,head_fit_updates=0)
  for key,bad in (('status','PENDING'),('count',5),('views',['clean']),('truth_read',True),('query_fit',True),('head_fit_updates',1)):
   value=dict(valid);value[key]=bad
   def fake(path):
    if Path(path).name=='complete.json' and Path(path).parent.name=='prediction':return value
    raise AssertionError('Truth opened before fixed7view closure')
   d.read=fake
   try:ev.score(d.rows()[0]['row_id'])
   except ValueError:n+=1
   else:raise AssertionError('Truth opened on incomplete predictions')
 finally:d.read,ev.ids,proof.source=original_read,original_ids,original_source
 return dict(status='PASS',negative_cases=n,real_IQ=0,CP_loads=0,NN=0,optimizer_updates=0,target_access=False)
