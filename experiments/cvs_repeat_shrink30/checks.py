"""Only changed view/loss/scope and truth-last guards;synthetic inputs only."""
from copy import deepcopy
import os,sys,types
import numpy as np
from . import design as d

def run(require_torch=False):
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,KeyError,TypeError):negative+=1
  else:raise AssertionError('Invalid fixedscope accepted')
 configs=[d.config(r) for r in d.rows()];assert len(configs)==2 and len({c['parent_source'] for c in configs})==1
 for key,value in [('epochs',11),('cumulative_epochs',31),('parent_source','UNKNOWN'),('source_RX_holdout',1),('new_U_rules',1),('As',True),('model_seed',2026092702)]:
  c=deepcopy(configs[0]);c[key]=value;bad(lambda:d.validate(c))
 c=deepcopy(configs[0]);c['IQ_operator']['max_strength']=.2;bad(lambda:d.validate(c))
 c=deepcopy(configs[0]);c['supervised_views']['L_NN_forwards']=4;bad(lambda:d.validate(c))
 from . import proof,evaluate,qualify
 pc=d.parent_config();cp=dict(run_id=pc['run_id'],candidate_id=pc['row_id'],epoch=20,checkpoint_selection='final_only',a1_ema_successful_updates=4440,ema_model={},args=dict(baseline_ckpt='',teacher_ckpt='',source_fold_held_rx=None,freeze_backbone=False));d.validate_parent_checkpoint(cp,configs[0])
 for key,value in [('epoch',19),('run_id','UNKNOWN'),('ema_model',None)]:
  b=deepcopy(cp);b[key]=value;bad(lambda:d.validate_parent_checkpoint(b,configs[0]))
 for key,value in [('baseline_ckpt','UNKNOWN'),('source_fold_held_rx',1),('freeze_backbone',True)]:
  b=deepcopy(cp);b['args'][key]=value;bad(lambda:d.validate_parent_checkpoint(b,configs[0]))
 original_read,original_write=d.read,d.write;original_source=proof.source;original_ids=evaluate.ids;visibility=os.environ.get('CUDA_VISIBLE_DEVICES');oldmodule=sys.modules.get('experiments.cvs_report_ssl.evaluate');truth=[];fake=types.ModuleType('experiments.cvs_report_ssl.evaluate');fake.truth_arrays=lambda *a:truth.append(True);fake.metric_rows=lambda *a:None;sys.modules[fake.__name__]=fake;os.environ['CUDA_VISIBLE_DEVICES']=''
 try:
  proof.source=lambda c:None;evaluate.ids=lambda:np.arange(168000);base=dict(status='PREDICTIONS_COMPLETE',config=configs[0],count=168000,views=list(d.VIEWS),truth_read=False,query_fit=False,head_fit_updates=0)
  for key,value in [('views',list(d.VIEWS[:-1])),('count',167999),('truth_read',True),('head_fit_updates',1)]:
   b=dict(base,**{key:value});d.read=lambda p,b=b:b;bad(lambda:evaluate.score(configs[0]['row_id']))
  assert not truth
 finally:
  d.read=original_read;d.write=original_write;proof.source=original_source;evaluate.ids=original_ids
  if visibility is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  else:os.environ['CUDA_VISIBLE_DEVICES']=visibility
  if oldmodule is None:sys.modules.pop(fake.__name__,None)
  else:sys.modules[fake.__name__]=oldmodule
 rows={}
 for c in configs:
  rate=.91 if c['arm']=='B10' else .9
  rows[c['row_id']]=dict(source_V_unique=27000,all_RX=list(d.FIT),target=False,model_state_unchanged=True,metrics=dict(source_val_accuracy=rate,source_val_rx_accuracy={str(r):rate for r in d.FIT}))
 try:
  d.read=lambda p:rows[p.parent.parent.name];d.write=lambda p,v:None
  assert qualify.matrix()['source_pass']
  rows[configs[1]['row_id']]['metrics']['source_val_accuracy']=.901
  assert not qualify.matrix()['source_pass']
  rows[configs[1]['row_id']]['metrics']['source_val_accuracy']=.91;rows[configs[1]['row_id']]['metrics']['source_val_rx_accuracy']['1']=.89
  assert not qualify.matrix()['source_pass']
  rows[configs[1]['row_id']]['target']=True;bad(qualify.matrix)
 finally:d.read=original_read;d.write=original_write
 output=dict(status='PASS',protocol_negative_cases=negative,common_NoAs20_parent_count=1,rows=2,new_model_epochs=20,new_updates=4440,L_forwards=6,truth_reads_during_guard_tests=len(truth),source_positive_and_no_gain_RX_failure_counterexamples=True,real_IQ=0,real_checkpoint_loads=0,real_model_NN=0,updates=0,target_access=False)
 if require_torch:
  import torch
  from . import model as api
  from .augmentation import Contraction
  original_schedule=api.supcon.schedule;api.supcon.schedule=lambda epoch:0.
  try:
   records=[]
   class Fake(torch.nn.Module):
    def __init__(self,c):
     super().__init__();self.W=torch.nn.Parameter(torch.linspace(-.2,.2,6*512).reshape(6,512));self.inputs=[];self._feature_l_steps=1;self._full_calls=0;self._supcon_pending=None;self._supcon_handles=[];self._supcon_phase=None;self._full_config=c;self._repeat_ag=Contraction(c['new_Ag']);self._full_L_packets=0;self._supcon_calls=0;self._legacy_calls=0;self._legacy_extra_packets=0;self._legacy_positive_KL_calls=0;self._full_records=[]
    def forward_identity_only(self,x,domain_labels=None):self.inputs.append(x.detach().clone());return dict(tx_logits=x.reshape(len(x),512)@self.W.T)
   g=torch.Generator().manual_seed(20261018105);raw=torch.randn(128,2,256,generator=g);labels=torch.arange(128).remainder(6);domain=torch.arange(128).remainder(15);rx=torch.tensor(api.supcon.RX_BY_DOMAIN)[domain];ordinary=raw+.03
   for c in configs:
    m=Fake(c);m._legacy_actual_L_input=ordinary;first=m.forward_identity_only(ordinary,domain_labels=domain);identity=torch.nn.functional.cross_entropy(first['tx_logits'],labels,label_smoothing=.01)
    correction,_=api.hook(m,model=m,labeled_output=first,labels=labels,extra=(None,dict(rx_i=rx)),domains=domain,optimizer_step=1,identity_loss=identity,labeled_count=128,labeled_inputs=raw)
    total=identity+correction;total.backward();assert len(m.inputs)==6 and m._full_L_packets==768 and m.W.grad is not None and bool(torch.isfinite(m.W.grad).all()) and m._legacy_actual_L_input is None
    assert torch.equal(m.inputs[0],m.inputs[1]) and torch.equal(m.inputs[2],m.inputs[3]) and torch.equal(m.inputs[4],m.inputs[5]) and torch.equal(m.inputs[2],raw)
    assert torch.equal(m.inputs[4],raw) is (not c['new_Ag'])
    row=m._full_records[0];expected=.5*row['common_RD']+.25*row['anchor_RD']+.25*row['slot_RD'];assert abs(float(total.detach())-expected)<1e-5 and row['total_L_scale']==1.
    records.append(row['operator']['lambda_values'])
   assert records[0]==records[1]
  finally:api.supcon.schedule=original_schedule
  output.update(actual_hook_synthetic_sixforward_and_loss_scale='PASS',same_IQ_per_pair=True,raw_anchor_unchanged=True,paired_private_lambda_draws_equal=True,synthetic_forward_packets=1536,synthetic_backward_calls=2)
 return output
if __name__=='__main__':
 import json
 print(json.dumps(run()))
