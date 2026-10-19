"""Real-F synthetic one-step oracle, original preservation and negative inputs."""
import json,math
import torch
from .virtual import discardable_step
from experiments.cvs_rdrop_sinc_units import optimizer as units
from experiments.cvs_F_nontrue20 import design as base,model as api

def run():
 torch.set_num_threads(2);torch.manual_seed(20261019622)
 c=base.config(next(r for r in base.rows() if r['arm']=='B1'));m=api.build(c,'cpu').eval()
 initial={k:v.detach().clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
 # Artificial complete gradients isolate actual optimizer algebra from network IO.
 gradients={k:torch.full_like(p,.02 if i%2 else -.03) for i,(k,p) in enumerate(m.named_parameters())}
 clone,w=discardable_step(m,gradients);assert torch.equal(torch.get_rng_state(),rng)
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) and all(p.grad is None for p in m.parameters())
 norm=math.sqrt(sum(float((g.double()/(25000. if k in units.QNAMES else 1.)).square().sum()) for k,g in gradients.items()))
 coefficient=min(1.,5./(norm+1e-6));sinc=m.get_submodule(units.PREFIX[:-1]);origins=dict(zip(units.QNAMES,(sinc.low_origin_hz,sinc.band_origin_hz)))
 maxerror=0.
 for k,p in m.named_parameters():
  g=gradients[k]*coefficient;lr=.0002*(1000. if k in units.QNAMES else 1.);eps=1e-8/(1000. if k in units.QNAMES else 1.)
  expected=p.detach()*(1-.0002*.0001)
  if k in origins:expected=expected-.0002*.0001*origins[k]/25000.
  expected=expected-lr*g/(g.abs()+eps);error=float((expected-dict(clone.named_parameters())[k].detach()).abs().max());maxerror=max(maxerror,error)
  torch.testing.assert_close(dict(clone.named_parameters())[k].detach(),expected,atol=2e-6,rtol=1e-6)
 negative=0
 bad=[dict(gradients,unregistered=torch.ones(1)),{k:g for k,g in gradients.items() if k!=next(iter(gradients))}]
 first=next(iter(gradients));q=dict(gradients);q[first]=torch.full_like(q[first],float('nan'));bad.append(q)
 q=dict(gradients);q[first]=q[first].reshape(-1)[:1];bad.append(q)
 for q in bad:
  try:discardable_step(m,q)
  except ValueError:negative+=1
  else:raise AssertionError('Protocol-negative gradient input accepted')
 assert negative==4
 # One original and one discarded default-head call prove actual wrapper execution.
 x=torch.zeros(2,2,256)
 with torch.no_grad():a=m.forward_identity_only(x)['tx_logits'];b=clone.forward_identity_only(x)['tx_logits']
 assert a.shape==b.shape==(2,6) and torch.isfinite(a).all() and torch.isfinite(b).all()
 generator=torch.Generator(device='cpu').manual_seed(20261019623);x=torch.randn(2,2,256,generator=generator);y=torch.tensor([0,4])
 z=m.forward_identity_only(x)['tx_logits'];loss=torch.nn.functional.cross_entropy(z,y,label_smoothing=.01)
 named=dict(m.named_parameters());grads=torch.autograd.grad(loss,tuple(named.values()),allow_unused=True)
 moved,real=discardable_step(m,dict(zip(named,grads)))
 with torch.no_grad():after=moved.forward_identity_only(x)['tx_logits']
 assert torch.isfinite(after).all() and not torch.equal(after,z.detach())
 assert all(p.grad is None for p in m.parameters()) and torch.equal(torch.get_rng_state(),rng)
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items())
 return dict(status='PASS',physical_AdamW_first_step_oracle=True,max_parameter_abs_error=maxerror,real_F_defaulthead_gradient_and_changed_logits=True,original_state_and_grads_unchanged=True,global_RNG_unchanged=True,nonaliasing=True,virtual_updates=2,original_updates=0,synthetic_NN=8,real_IQ=0,checkpoint_loads=0,target_access=False,protocol_negatives=negative,source_performance_claim=False)
if __name__=='__main__':print(json.dumps(run()))
