"""Focused default-head margin and fixed source/protocol checks; zero CVS NN."""
import copy,json,tempfile,types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import design as d
from .margin import penalty,captured_native_views
from .protocol import final_metadata,checkpoint

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid fixed margin61 scope accepted')
 assert len(d.rows())==1
 c=d.config(d.rows()[0]);d.validate(c)
 for k,v in [('target_access',True),('As',False),('new_Ag',True),('epochs',20),('updates',4440),('model_seed',2026092702),('parent_source','unknown'),('label_epochs',13),('Ag_mode','WHITE')]:
  z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
 for key,val in [('lambda_pair',1.),('rho',1.),('head_scale',1.),('extra_NN',128),('curriculum',True)]:
  z=copy.deepcopy(c);z['pair_margin'][key]=val;bad(lambda z=z:d.validate(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.config(dict(arm='other',row_id='x',model_seed=d.SEED)))
 roles={'L':['toyL'],'U':['toyU'],'V':['toyV']};done=dict(config=c,epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0);frozen=dict(config=c,selection='final_only',target_access=False);init=dict(scratch_only=False,target_contact=False,checkpoint_sources=c['checkpoint_sources'],source_roles='EXACT_MATCH');contract=dict(role_ids=roles)
 final_metadata(c,done,frozen,init,contract,roles)
 for key,val in [('target_access',True),('epoch',20),('optimizer_steps',4440),('head_fit_updates',1)]:
  z=copy.deepcopy(done);z[key]=val;bad(lambda z=z:final_metadata(c,z,frozen,init,contract,roles))
 cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=d.RUN,candidate_id=c['row_id'],checkpoint_selection='final_only',ema_model={},args=dict(baseline_ckpt=c['parent_source']+'/final_ssdg.pth',teacher_ckpt=''))
 checkpoint(c,cp)
 for key,val in [('run_id',d.REFERENCE_RUN),('candidate_id','B01-s2026092701'),('epoch',20),('ema_model',None)]:
  z=copy.deepcopy(cp);z[key]=val;bad(lambda z=z:checkpoint(c,z))
 # Closed-form loss, stop-gradient, fixed /30 scale and clamp; only student moves.
 y=torch.zeros(64,dtype=torch.long);clean=torch.zeros(64,6);clean[:,0]=30;clean.requires_grad_(True);sat=torch.zeros(64,6,requires_grad=True);meta=[dict(scene='practical_mid_urban',signal_to_added_noise_db=5.) for _ in range(64)]
 loss,info=penalty(clean,sat,y,meta);assert abs(float(loss)-.125)<1e-8;loss.backward()
 assert clean.grad is None and sat.grad is not None and torch.all(sat.grad[:,0]<0) and torch.all(sat.grad[:,1:]>0) and abs(float(sat.grad[:,0].sum())+1/120)<1e-7
 half=clean.detach().clone();half[:,0]=15;double=clean.detach().clone();double[:,0]=600
 assert abs(float(penalty(half,sat.detach(),y,meta)[0])-.0625)<1e-8 and abs(float(penalty(double,sat.detach(),y,meta)[0])-.125)<1e-8
 wrong=clean.detach().clone();wrong[:,1]=60;assert float(penalty(wrong,sat.detach(),y,meta)[0])==0
 for scene,db in [('practical_low_suburban',5.),('practical_low_urban',None)]:
  zero=[dict(scene=scene,signal_to_added_noise_db=db)]*64;assert float(penalty(clean,sat.detach(),y,zero)[0])==0
 values=[float(penalty(clean,sat.detach(),y,[dict(scene='practical_low_urban',signal_to_added_noise_db=db)]*64)[0]) for db in (0.,5.,10.)];assert 0<values[0]<values[1]<values[2]
 for records in [[dict(scene='practical_low_urban')]*64,[dict(scene='invalid',signal_to_added_noise_db=5)]*64,[dict(scene='practical_low_urban',signal_to_added_noise_db=float('inf'))]*64]:bad(lambda records=records:penalty(clean,sat,y,records))
 # This observation wrapper must consume no RNG or change ordinary inputs/outputs.
 class Dummy:
  def __init__(self):self._full_channel=types.SimpleNamespace(apply=lambda x:(x.clone(),meta))
  def forward_identity_only(self,x):return {'tx_logits':x+torch.rand_like(x)}
 m=Dummy();x=torch.ones(128,6);torch.manual_seed(5001);bare=[m.forward_identity_only(x)['tx_logits'] for _ in range(5)];end=torch.get_rng_state().clone();torch.manual_seed(5001)
 with captured_native_views(m) as (captured,observed):
  wrapped=[m.forward_identity_only(x)['tx_logits'] for _ in range(5)];m._full_channel.apply(x[:64])
 assert torch.equal(torch.get_rng_state(),end) and all(torch.equal(a,b) for a,b in zip(bare,wrapped)) and len(captured)==5 and len(observed)==1 and torch.equal(x,torch.ones_like(x)) and 'forward_identity_only' not in vars(m) and 'apply' in vars(m._full_channel)
 original=m.forward_identity_only
 try:
  with captured_native_views(m):raise RuntimeError('toy')
 except RuntimeError:pass
 assert m.forward_identity_only==original
 # Source-only producer schema is exercised with fixed toy rates; formal is absent.
 from . import report
 with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
  base=Path(tmp)/'new';ref=Path(tmp)/'runs'/d.REFERENCE_RUN;row=d.rows()[0];paths=[base/row['row_id']/'source/completion.json',base/row['row_id']/'source_qualification/complete.json',ref/'B00-s2026092701/source/completion.json',ref/'B01-s2026092701/source_qualification/complete.json']
  no=dict(target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.9));yes=dict(config=c,**no);yes['final_source_metrics']=dict(source_val_count=27000,source_val_accuracy=.899)
  q=dict(config=c,target_access=False,formal_scores_read=False,training_updates=0,NN=12600,source_accuracies=[[.9,.75,.6]]*2);old=dict(q,config=d.old_config(),source_accuracies=[[.9,.74,.6]]*2)
  for p,v in zip(paths,[yes,q,no,old]):d.write(p,v)
  with patch.object(d,'BASE',base),patch.object(d,'PROJECT',tmp),patch.object(d,'config',return_value=c):
   v=report.source_decision();assert v['source_qualified'] is True and all(type(x) is bool for x in v['checks'].values());json.dumps(v,allow_nan=False)
   paths[1].unlink();v=report.source_decision();assert v['status']=='SOURCE_READOUT_PARTIAL' and v['source_qualified'] is False and v['formal_evaluation_always_required'] is True
 return dict(status='PASS',protocol_negatives=negative,toy_margin_stopgrad_scale_clamp_q_gradient=True,observation_wrapper_RNG_ordinary_preserved=True,source_bool_JSON=True,CVS_NN=0,IQ=0,CP=0,target=False,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run()))
