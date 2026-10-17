"""Protocol negatives and actual-F replay/remaining-gradient/default-head lifecycle checks."""
import copy,json,random
import numpy as np
import torch
from . import design as d
from .core import direction,consume
from .checkpoint_policy import load
def run(full=True):
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,FloatingPointError):negative+=1
  else:raise AssertionError('Invalid scope accepted')
 c=d.config(d.rows()[1])
 for k,v in [('epochs',11),('updates',2442),('cumulative_epochs',31),('parent_source','other'),('new_losses',1),('new_U_rules',1),('new_neural_forwards',0),('model_seed',2026092702),('As',False),('new_Ag',True),('update_mode','other'),('batch_mode','RANDOM')]:
  a=copy.deepcopy(c);a[k]=v;bad(lambda a=a:d.validate(a))
 for k,v in [('RX',[3,4,6,8]),('L',5580),('U_TX_truth_hidden',False),('source_RX_holdout',1)]:
  a=copy.deepcopy(c);a['data_contract'][k]=v;bad(lambda a=a:d.validate(a))
 a=copy.deepcopy(c);a['update_policy']['radius']=.01;bad(lambda:d.validate(a))
 sentinel=object();calls=[]
 def actual(path,dev):calls.append(path);return 'FINAL'
 assert load('/p/final','cpu','/p/final','/r/final',sentinel,actual) is sentinel and not calls
 assert load('/r/final','cpu','/p/final','/r/final',sentinel,actual)=='FINAL'
 for path in ('/other/final','/r/latest','/p/best'):bad(lambda path=path:load(path,'cpu','/p/final','/r/final',sentinel,actual))
 p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64));delta,norm=direction((('p',p),),(torch.tensor([3.,4.],dtype=torch.float64),),.05);assert norm==5. and torch.allclose(delta[0],torch.tensor([.03,.04],dtype=torch.float64));bad(lambda:direction((('p',p),),(torch.ones(3),),.05));bad(lambda:direction((('p',p),),(torch.tensor([float('nan'),0.]),),.05));bad(lambda:direction((('p',p),),(torch.ones(2),),.1))
 toy=torch.tensor([1.,2.],dtype=torch.float64,requires_grad=True);L=(toy.square().sum());aux=3*toy.sum();old=torch.autograd.grad(L,toy,retain_graph=True)[0];total=torch.autograd.grad(L+aux,toy)[0];new=2*(toy.detach()+.05*old/old.norm());assert torch.allclose(total+(new-old)-new,torch.full((2,),3.,dtype=torch.float64))
 summaries=[];synthetic=0
 if full:
  from . import model as api
  from experiments.cvs_clock_factorial30 import model as original
  from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
  from experiments.cvs_fullsource_concat20 import supcon
  class FakeChannel:
   def apply(self,x,meta,**kw):return x*.9,[dict(scene='practical_mid_urban',state_end='post_sync',receiver_lock_status='ok') for _ in x]
  schedule=supcon.schedule;supcon.schedule=lambda epoch:.1
  try:
   torch.set_num_threads(2);torch.manual_seed(20261018301);raw=torch.randn(128,2,256);ordinary=raw*.7+.13;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains]
   for row in d.rows():
    c=d.config(row);m=api.build(c,'cpu');base=original.build(d.nominal_config(c['arm']),'cpu');base.load_state_dict(m.state_dict(),strict=True);m._full_channel=FakeChannel() if c['As'] else None;base._full_channel=FakeChannel() if c['As'] else None;maxerr=0.
    for step in (1,2):
     records=[]
     for net,replay_flag in ((base,False),(m,True)):
      net.train();net.zero_grad(set_to_none=True);torch.manual_seed(20261018302+step);random.seed(20261018302+step);np.random.seed((20261018302+step)%(2**32))
      with net._phase1_labeled_feature_context():out=net(ordinary,y_tx=y,return_aux=True)
      ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
      corr,_=net._source_labeled_feature_hook(model=net,labeled_output=out,labels=y,extra=(None,dict(rx_i=rx)),domains=domains,optimizer_step=step,identity_loss=ce,labeled_count=128,labeled_inputs=raw)
      head=net.id_backbone.encoder.core.id_backbone.cls_head.weight;((ce+corr)+.03*head.square().sum()).backward();rng=torch.get_rng_state().clone();weights={k:v.detach().clone() for k,v in net.state_dict().items()}
      if replay_flag:consume(net,net.parameters(),5.);synthetic+=1024;assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,weights[k]) for k,v in net.state_dict().items())
      else:synthetic+=512
      grads={k:v.grad.detach().clone() for k,v in net.named_parameters() if v.grad is not None};records.append(grads)
     if row['update_mode']=='REPLAY':
      assert set(records[0])==set(records[1]);maxerr=max(maxerr,max(float((v-records[0][k]).abs().max()) for k,v in records[1].items()));assert maxerr<1e-5
     else:assert any(not torch.equal(v,records[0][k]) for k,v in records[1].items())
     # Both copies remain at the same weights;native-step update is tested once below.
    opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,eps=1e-8,weight_decay=.0001,foreach=False);clip_physical(opt,m.parameters(),5.);opt.step();teacher=copy.deepcopy(m).eval();assert teacher.id_backbone.encoder._sam_owner is teacher and teacher._sam_original_scope.__self__ is teacher and teacher._sam_pending is None and teacher._sam_recording is False and opt._sinc_unit_steps==1
    assert m._sam_steps==2 and m._sam_vjp_calls==4 and m._sam_replay_packets==1024 and m._full_L_packets==1024
    summaries.append(dict(row=row['row_id'],neutral_gradient_max_error=maxerr,replay_relative_error_max=max(v['replay_gradient_relative_difference'] for v in m._sam_records),actual_perturbation_max=max(v['actual_perturbation_L2'] for v in m._sam_records),lifecycle_steps=2,default_head_joint_update=not torch.equal(m.id_backbone.encoder.core.id_backbone.cls_head.weight,base.id_backbone.encoder.core.id_backbone.cls_head.weight)))
  finally:supcon.schedule=schedule
 return dict(status='PASS',protocol_negatives=negative,analytic_SAM=True,remaining_aux_gradient_preserved=True,actual_F=summaries,synthetic_F_forward_packets=synthetic,synthetic_updates=3 if full else 0,extra_F_VJP=12 if full else 0,CP=0,real_IQ=0,real_updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
