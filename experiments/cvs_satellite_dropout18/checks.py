"""Focused source scope and native RNG/gradient preservation checks."""
import copy,json
from . import design as d

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError):negative+=1
  else:raise AssertionError('Invalid scope accepted')
 c=d.config(d.rows()[0],20)
 assert all(d.config(r,s)['baseline_reuse'] is False for r in d.rows() for s in (20,30))
 for k,v in [('target_access',True),('As',False),('new_Ag',True),('epochs',31),('updates',6661),('source_RX_holdout',1),('satellite_mode','AS_RD'),('total_L_loss_scale',1.),('model_seed',2026092702),('baseline_reuse',True)]:
  z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
 bad(lambda:d.config(dict(arm='AS_RD',row_id='unregistered',model_seed=d.SEED)))
 try:import torch
 except ImportError:return dict(status='LOCAL_PROTOCOL_PASS_TORCH_UNAVAILABLE',protocol_negatives=negative,CVS_NN=0,real_IQ=0,CP=0,target=False,remote_Torch_checks_required=True)
 from .dropout import Engine
 bad(lambda:Engine('INVALID'))
 x=torch.arange(128*5,dtype=torch.float32).reshape(128,5)/100;x.requires_grad_(True);mask=torch.arange(128)%2==1;module=torch.nn.Dropout(.3);states=[];outputs=[];grads=[]
 for mode in ('AS_CE','AS_DET'):
  e=Engine(mode);torch.manual_seed(9001)
  with e.scope(mask):e.frame=2;v=e.apply(module,x);v.sum().backward();grads.append(x.grad.clone());x.grad=None;outputs.append(v.detach().clone());states.append(torch.get_rng_state().clone())
  assert not e.active and e.mask is None
 assert torch.equal(states[0],states[1]) and torch.equal(outputs[0][~mask],outputs[1][~mask]) and torch.equal(outputs[1][mask],x.detach()[mask]) and torch.equal(grads[1][mask],torch.ones_like(grads[1][mask]))
 for frame in (0,1,4,5):
  e=Engine('AS_DET');torch.manual_seed(9002);ref=module(x);torch.manual_seed(9002)
  with e.scope(mask):e.frame=frame;v=e.apply(module,x)
  assert torch.equal(v,ref)
 e=Engine('AS_DET');bad(lambda:e.scope(torch.ones(127,dtype=torch.bool)).__enter__())
 try:
  with e.scope(mask):raise RuntimeError('synthetic')
 except RuntimeError:pass
 assert not e.active and e.mask is None
 module.eval();e=Engine('AS_DET')
 with e.scope(mask):e.frame=2;v=e.apply(module,x)
 assert torch.equal(v,x)
 return dict(status='PASS',protocol_negatives=negative,global_RNG_matched=True,ordinary_and_raw_unmodified=True,satellite_direct_gradient=True,exception_scope_restored=True,evaluation_unchanged=True,CVS_NN=0,real_IQ=0,CP=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
