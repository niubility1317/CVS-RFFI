"""Fixed noAs contract and neutral full-F repeated ordinary view comparison."""
import copy,json,torch
from . import design as d
from . import model as api
from .checkpoint_policy import load
def run():
 n=0
 def bad(f):
  nonlocal n
  try:f()
  except (ValueError,TypeError,KeyError):n+=1
  else:raise AssertionError('Invalid reference contract accepted')
 c=d.config(d.rows()[0])
 for k,v in [('As',True),('new_Ag',True),('new_losses',1),('new_U_rules',1),('new_neural_forwards',1),('epochs',11),('cumulative_epochs',31),('model_seed',2026092702),('gradient_mode','PROJECT'),('arm','FULL_AS'),('parent_source','CONCAT')]:
  a=copy.deepcopy(c);a[k]=v;bad(lambda a=a:d.validate(a))
 for k,v in [('RX',[3,4,6,8]),('L',5580),('U_TX_truth_hidden',False),('source_RX_holdout',1)]:
  a=copy.deepcopy(c);a['data_contract'][k]=v;bad(lambda a=a:d.validate(a))
 sentinel=object();assert load('/p/final','cpu','/p/final','/r/final',sentinel,None) is sentinel;bad(lambda:load('/other','cpu','/p/final','/r/final',sentinel,None))
 from experiments.cvs_fullsource_concat20 import supcon
 from experiments.cvs_view_pcgrad30.core import consume
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 old=supcon.schedule;supcon.schedule=lambda epoch:.1
 try:
  torch.set_num_threads(2);torch.manual_seed(20261018301);raw=torch.randn(128,2,256);ordinary=raw*.7+.13;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains]
  reference=api.original.build(d.nominal_config(),'cpu');measured=api.build(c,'cpu');measured.load_state_dict(reference.state_dict(),strict=True);gradients=[];states=[];rng=[]
  for i,m in enumerate((reference,measured)):
   assert m._full_channel is None and not getattr(m,'_clock',None);m.train();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,eps=1e-8,weight_decay=.0001,foreach=False);torch.manual_seed(20261018302)
   with m._phase1_labeled_feature_context():out=m(ordinary,y_tx=y,return_aux=True)
   ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01);extra,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(None,dict(rx_i=rx)),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=raw);(ce+extra).backward()
   if i:consume(m,m.parameters(),5.)
   clip_physical(opt,m.parameters(),5.);gradients.append({k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None});rng.append(torch.random.get_rng_state().clone());opt.step();states.append(copy.deepcopy(m.state_dict()))
   assert m._full_L_packets==512 and m._full_records[0]['As_packets']==0
  assert gradients[0].keys()==gradients[1].keys() and all(torch.equal(v,gradients[1][k]) for k,v in gradients[0].items()) and torch.equal(rng[0],rng[1]);assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())
  assert measured._pc_vjp_calls==2 and measured._pc_clips==1 and measured._pc_records[0]['applied']==False
  teacher=copy.deepcopy(measured).eval();assert teacher._pc_pending is None and teacher._source_labeled_feature_hook.__self__ is teacher
 finally:supcon.schedule=old
 return dict(status='PASS',protocol_negatives=n,synthetic_F_NN=1024,synthetic_updates=2,extra_VJP=2,neutral_gradient_and_update_exact=True,same_RNG=True,As_generated=0,IQ=0,CP=0,real_updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
