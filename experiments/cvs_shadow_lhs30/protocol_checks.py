"""Fixed E30/data/parent and truth-last negative checks;synthetic metadata only."""
from copy import deepcopy
import os,sys,types
import numpy as np
from . import design as d
from . import evaluate,proof,qualify
def run():
 failures=0
 def reject(fn):
  nonlocal failures
  try:fn()
  except (ValueError,KeyError,TypeError):failures+=1
  else:raise AssertionError('Invalid fixed experiment accepted')
 configs=[d.config(r) for r in d.rows()]
 assert len(configs)==4 and len({c['parent_source'] for c in configs})==1 and all((c['epochs'],c['cumulative_epochs'],c['updates'])==(10,30,2220) for c in configs)
 for key,value in [('epochs',11),('cumulative_epochs',31),('parent_source','UNKNOWN'),('source_RX_holdout',1),('new_U_rules',1),('new_Ag',True),('shadow_mode','LHS'),('model_seed',2026092702)]:
  c=deepcopy(configs[0]);c[key]=value;reject(lambda:d.validate(c))
 pc=d.parent_config();cp=dict(run_id=pc['run_id'],candidate_id=pc['row_id'],epoch=20,checkpoint_selection='final_only',a1_ema_successful_updates=4440,ema_model={},args=dict(baseline_ckpt='',teacher_ckpt='',source_fold_held_rx=None,freeze_backbone=False));d.validate_parent_checkpoint(cp,configs[0])
 for key,value in [('epoch',19),('run_id','UNKNOWN'),('ema_model',None)]:
  bad=deepcopy(cp);bad[key]=value;reject(lambda:d.validate_parent_checkpoint(bad,configs[0]))
 for key,value in [('baseline_ckpt','UNKNOWN'),('source_fold_held_rx',1),('freeze_backbone',True)]:
  bad=deepcopy(cp);bad['args'][key]=value;reject(lambda:d.validate_parent_checkpoint(bad,configs[0]))
 original_read,original_write=d.read,d.write;original_source=proof.source;original_ids=evaluate.ids;old_visibility=os.environ.get('CUDA_VISIBLE_DEVICES');oldmodule=sys.modules.get('experiments.cvs_report_ssl.evaluate');truth_calls=[];fake=types.ModuleType('experiments.cvs_report_ssl.evaluate');fake.truth_arrays=lambda *a:truth_calls.append(True);fake.metric_rows=lambda *a:None;sys.modules[fake.__name__]=fake;os.environ['CUDA_VISIBLE_DEVICES']=''
 try:
  proof.source=lambda c:None;evaluate.ids=lambda:np.arange(168000);base=dict(status='PREDICTIONS_COMPLETE',config=configs[0],count=168000,views=list(d.VIEWS),truth_read=False,query_fit=False,head_fit_updates=0)
  for key,value in [('views',list(d.VIEWS[:-1])),('count',167999),('truth_read',True),('head_fit_updates',1)]:
   bad=dict(base,**{key:value});d.read=lambda p,bad=bad:bad;reject(lambda:evaluate.score(configs[0]['row_id']))
  assert not truth_calls
 finally:
  d.read=original_read;proof.source=original_source;evaluate.ids=original_ids
  if old_visibility is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  else:os.environ['CUDA_VISIBLE_DEVICES']=old_visibility
  if oldmodule is None:sys.modules.pop(fake.__name__,None)
  else:sys.modules[fake.__name__]=oldmodule
 # The candidate must beat BOTH As controls and every fixed draw separately.
 rows={};written=[]
 for c in configs:
  name=c['row_id'].split('-s')[0];is_lhs=name=='AS_LHS';clean=1420 if is_lhs else 1400;records=[];metrics={}
  for view in d.VIEWS:
   correct=clean if view=='clean' else (1410 if is_lhs else 1380) if view in d.VIEWS[1:4] else (1000 if is_lhs else 990);count=1 if view=='clean' else 2
   for draw in range(count):records.append(dict(view=view,draw=draw,count=1800,correct=correct))
   metrics[view]=dict(accuracy=correct/1800,decisions=1800*count)
  rows[c['row_id']]=dict(metrics=metrics,records=records,source_V_unique=1800,all_RX=list(d.FIT),target=False,model_state_unchanged=True)
 try:
  d.read=lambda p:rows[p.parent.parent.name];d.write=lambda p,v:written.append(v);good=qualify.matrix();assert good['source_pass']
  badrow=rows['AS_LHS-s'+str(d.SEED)];low=next(r for r in badrow['records'] if r['view']=='practical_low_urban' and r['draw']==1);low['correct']=980;badrow['metrics']['practical_low_urban']['accuracy']=(1000+980)/3600;bad=qualify.matrix();assert not bad['source_pass'] and not bad['conditions']['AS_NATIVE_lowurban'] and not bad['conditions']['AS_IID_lowurban']
  badrow['target']=True;reject(qualify.matrix)
 finally:d.read=original_read;d.write=original_write
 return dict(status='PASS',negative_cases=failures,common_NoAs20_parent_count=1,rows=4,new_model_epochs=40,new_updates=8880,truth_reads_during_guard_tests=len(truth_calls),source_positive_and_bad_draw_counterexample=True,new_IQ_Ag=False,real_IQ=0,model_NN=0,updates=0,target_access=False)
if __name__=='__main__':
 import json
 print(json.dumps(run(),allow_nan=False))
