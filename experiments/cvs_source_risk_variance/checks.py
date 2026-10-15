import copy,json,torch
from . import design as d
from .objective import risk_variance,RX

def checks():
 torch.set_num_threads(2);torch.manual_seed(31529)
 y=torch.arange(30)%6;rx=torch.tensor(list(RX)).repeat_interleave(6);a=torch.randn(30,6,requires_grad=True)
 ce,var,risks=risk_variance(a,y,rx);manual=torch.stack([torch.nn.functional.cross_entropy(a[rx==r],y[rx==r],label_smoothing=.01) for r in RX])
 assert torch.allclose(risks,manual,atol=1e-6) and torch.allclose(var,manual.var(unbiased=False),atol=1e-6) and torch.allclose(ce,torch.nn.functional.cross_entropy(a,y,label_smoothing=.01))
 assert torch.autograd.grad(var,a,retain_graph=True)[0].norm()>0
 order=torch.randperm(30);assert torch.allclose(risk_variance(a[order],y[order],rx[order])[1],var,atol=1e-7)
 equal=torch.zeros_like(a,requires_grad=True);_,z,_=risk_variance(equal,y,rx);assert float(z)==0 and float(torch.autograd.grad(z,equal)[0].norm())==0
 bads=[lambda:risk_variance(a,y,rx,role='U_s'),lambda:risk_variance(a[:,:5],y,rx),lambda:risk_variance(a[:0],y[:0],rx[:0]),lambda:risk_variance(a,y.float(),rx),lambda:risk_variance(a,y,rx.float()),lambda:risk_variance(a,y,torch.zeros_like(rx)),lambda:risk_variance(a,y+6,rx),lambda:risk_variance(a, y,rx[:-1]),lambda:risk_variance(torch.full_like(a,float('nan')),y,rx)]
 for f in bads:
  try:f()
  except ValueError:pass
  else:raise AssertionError('Illegal source risk accepted')
 c=d.config(d.rows()[0])
 for k,v in [('role','U_s'),('coefficient',10.),('head_fit_updates',1),('receiver_groups',[1,3]),('parent_rule','target-best')]:
  b=copy.deepcopy(c);b[k]=v
  try:d.validate(b)
  except ValueError:pass
  else:raise AssertionError('Changed fixed source recipe accepted')
 from experiments.cvs_receiver_dro50 import model as api
 from .audit import measure_batch
 model=api.build(d.source_config(),'cpu').eval();before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
 x=torch.randn(30,2,256,generator=torch.Generator().manual_seed(31530));out=measure_batch(model,x,y,rx,31531)
 assert out['features'].shape==(30,160) and all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and torch.equal(rng,torch.get_rng_state()) and all(p.grad is None for p in model.parameters())
 for k in ('BALANCED_CE','RISK_VARIANCE'):
  assert [v.numel() for v in out['vectors'][k]]==[3456,6912,576,960] and all(torch.isfinite(v).all() and v.norm()>0 for v in out['vectors'][k])
 from .publish import INSPECT
 compile(INSPECT,'risk_variance_remote_inspect','exec')
 assert 'forward_packets=900' in INSPECT and 'saved_vectors=40' in INSPECT
 return dict(status='PASS',negative_cases=len(bads)+5,risk_formula_and_balanced_CE=True,constant_risk_zero_loss_and_gradient=True,actual_native_t1_t2_SIG_W_derivatives=True,model_state_and_RNG_unchanged=True,real_source_IQ_or_checkpoint_read=False,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(checks()))
