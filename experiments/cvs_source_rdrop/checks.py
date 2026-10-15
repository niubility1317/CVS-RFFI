import copy,json
import torch
from . import design as d
from .objective import symmetric_kl

def checks():
 torch.set_num_threads(2);torch.manual_seed(31519)
 a=torch.randn(12,6,requires_grad=True);b=torch.randn(12,6,requires_grad=True)
 value=symmetric_kl(a,b);ga,gb=torch.autograd.grad(value,(a,b))
 assert value>0 and ga.norm()>0 and gb.norm()>0
 assert torch.allclose(value,symmetric_kl(b,a))
 assert float(symmetric_kl(a,a).detach())==0.
 assert torch.allclose(value,symmetric_kl(a+2,b-3),atol=1e-6)
 assert torch.allclose(value,symmetric_kl(a.flip(0),b.flip(0)),atol=1e-6)
 bads=[lambda:symmetric_kl(a[:,:5],b[:,:5]),lambda:symmetric_kl(a,b[:11]),lambda:symmetric_kl(a[:0],b[:0]),
  lambda:symmetric_kl(a.long(),b.long()),lambda:symmetric_kl(a,b.double()),lambda:symmetric_kl(a,torch.full_like(b,float('nan')))]
 for call in bads:
  try:call()
  except ValueError:pass
  else:raise AssertionError('Invalid stochastic logit pair accepted')
 c=d.config(d.rows()[0])
 for key,new in [('role','U_s'),('coefficient',.1),('head_fit_updates',1),('parent_rule','target-best')]:
  changed=copy.deepcopy(c);changed[key]=new
  try:d.validate(changed)
  except ValueError:pass
  else:raise AssertionError('Changed scientific source recipe accepted')
 from experiments.cvs_receiver_dro50 import model as api
 from .audit import measure_batch
 model=api.build(d.source_config(),'cpu').eval()
 before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
 x=torch.randn(12,2,256,generator=torch.Generator().manual_seed(31520));y=torch.arange(12)%6
 result=measure_batch(model,x,y,31521)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
 assert torch.equal(rng,torch.get_rng_state()) and all(p.grad is None for p in model.parameters())
 assert result['features1'].shape==result['features2'].shape==(12,160)
 assert not torch.equal(result['features1'],result['features2']) and result['losses']['SYMMETRIC_KL']>0
 for name in ('PAIRED_CE','SYMMETRIC_KL'):
  assert [v.numel() for v in result['vectors'][name]]==[3456,6912,576,960]
  assert all(torch.isfinite(v).all() and float(v.norm())>0 for v in result['vectors'][name])
 return dict(status='PASS',negative_cases=len(bads)+4,symmetric_KL=True,zero_for_identical_logits=True,
  logit_common_shift_invariance=True,both_predictions_differentiable=True,actual_native_dropout_pair_and_joint160=True,
  actual_t1_t2_SIG_and_existing_W_derivatives=True,model_state_and_RNG_unchanged=True,
  head_fit_updates=0,optimizer_updates=0,real_IQ_or_checkpoint_read=False)
if __name__=='__main__':print(json.dumps(checks()))
