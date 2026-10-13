import copy,json,torch
from torch.nn import functional as F
from .block import CrossDelayResidual
from . import design as d
from .model import build
from experiments.cvs_default_head.predict import logits as default_logits

def primitive_checks():
 torch.set_num_threads(2);torch.manual_seed(74113)
 q=torch.randn(2,2,256,dtype=torch.float64)*.25
 m=CrossDelayResidual('cross_delay').double();a,b=m.filters();value=m.terms(q)
 delays=torch.stack([F.pad(q,(i,0))[...,:256] for i in range(9)],2)
 power=q.square().sum(1,keepdim=True)
 pdelays=torch.stack([F.pad(power,(i,0))[...,:256] for i in range(9)],2)
 u=torch.einsum('brml,rm->brl',delays[:,0:1].expand(-1,8,-1,-1),a[:,0])
 uq=torch.einsum('brml,rm->brl',delays[:,1:2].expand(-1,8,-1,-1),a[:,0])
 v=torch.einsum('brml,rm->brl',pdelays.expand(-1,8,-1,-1),b[:,0])
 expected=torch.cat([u*v,uq*v],1);error=float((value-expected).abs().max())
 assert error<1e-12
 # Explicit off-diagonal m=1,l=3 term, rather than the same delayed sample cubed.
 special=copy.deepcopy(m);special.a.data.zero_();special.b.data.zero_();special.a.data[:,:,1]=1;special.b.data[:,:,3]=1
 off=special.terms(q);sample=F.pad(q,(1,0))[...,:256]*F.pad(power,(3,0))[...,:256]
 assert torch.allclose(off[:,:8],sample[:,0:1].expand(-1,8,-1),atol=1e-12,rtol=1e-12)
 assert torch.allclose(off[:,8:],sample[:,1:2].expand(-1,8,-1),atol=1e-12,rtol=1e-12)
 assert not torch.allclose(sample,F.pad(q,(1,0))[...,:256]*F.pad(power,(1,0))[...,:256])
 # Primitive common-phase equivariance before SiLU/component clipping.
 angle=.37;rot=torch.stack([q[:,0]*torch.cos(torch.tensor(angle))-q[:,1]*torch.sin(torch.tensor(angle)),q[:,0]*torch.sin(torch.tensor(angle))+q[:,1]*torch.cos(torch.tensor(angle))],1)
 angle_t=q.new_tensor(angle);rot=torch.stack([q[:,0]*angle_t.cos()-q[:,1]*angle_t.sin(),q[:,0]*angle_t.sin()+q[:,1]*angle_t.cos()],1)
 actual=m.terms(rot);rvalue=torch.cat([value[:,:8]*angle_t.cos()-value[:,8:]*angle_t.sin(),value[:,:8]*angle_t.sin()+value[:,8:]*angle_t.cos()],1)
 phase_error=float((actual-rvalue).abs().max());assert phase_error<1e-12
 # No future samples or circular wrap; same fixed per-packet operator.
 changed=q.clone();changed[:,:,128:]+=10;assert torch.equal(m.terms(changed)[:,:,:128],value[:,:,:128])
 assert torch.allclose(m(q[:1]),m(q)[:1],atol=1e-12,rtol=1e-12)
 for x in [torch.zeros_like(q),q*1e-20,q*1e4]:assert torch.isfinite(m(x)).all()
 scaled=m.terms(q*.5);assert torch.allclose(scaled,value*.125,atol=1e-12,rtol=1e-12)
 # Analytic factorized primitive and all filters/input are differentiable.
 xx=q.clone().requires_grad_(True);loss=m(xx).square().mean();loss.backward()
 assert torch.isfinite(xx.grad).all() and float(xx.grad.norm())>0
 assert all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0 for p in m.parameters())
 assert sum(p.numel() for p in m.parameters())==576
 negatives=0
 for args in [('bad',8,9,24),('conv',7,9,24),('conv',8,8,24),('conv',8,9,25)]:
  try:CrossDelayResidual(*args)
  except ValueError:negatives+=1
  else:raise AssertionError('Unregistered capacity accepted')
 for x in [q[:,:,0:255],q[:,:1],q.long(),q.half()]:
  try:m.terms(x)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid IQ accepted')
 return dict(status='PASS',mathematical_max_error=error,off_diagonal_delay_term=True,primitive_phase_equivariance_error=phase_error,phase_claim='primitive only; clipping/SiLU/full CVS need not be invariant',cubic_primitive_before_activation=True,causal_zero_padding=True,zero_tiny_clipped_inputs_finite=True,per_packet_independent=True,all_new_gradients_finite_nonzero=True,extra_parameters=576,negative_cases=negatives)

def checks():
 primitive=primitive_checks();torch.manual_seed(74217)
 configs=[d.config(r) for r in d.training_rows()[:2]];models=[build(c) for c in configs]
 encoders=[m.id_backbone.encoder for m in models]
 for k,v in encoders[0].state_dict().items():assert torch.equal(v,encoders[1].state_dict()[k]),k
 x=torch.randn(6,2,256);y=torch.arange(6);reports=[]
 for c,m in zip(configs,models):
  e=m.id_backbone.encoder;h=e.core.id_backbone.cls_head;pa=e.core.id_backbone.pa_lift;clone=copy.deepcopy(m).eval();ce=clone.id_backbone.encoder
  ce.core.id_backbone.pa_lift.residual.gain.data.zero_();baseline=copy.deepcopy(ce);baseline.core.id_backbone.pa_lift=baseline.core.id_backbone.pa_lift.base
  with torch.no_grad():assert torch.equal(ce.features(x),baseline.features(x))
  m.train();m.zero_grad(set_to_none=True);loss=F.cross_entropy(m.forward_identity_only(x)['tx_logits'],y);loss.backward()
  gradients={name:float(p.grad.norm()) if p.grad is not None else None for name,p in pa.residual.named_parameters()}
  assert all(v is not None and v>0 for v in gradients.values()) and all(torch.isfinite(p.grad).all() for p in pa.residual.parameters())
  assert h.weight.shape==(6,160) and h.scale==30. and h.weight.grad is not None and float(h.weight.grad.norm())>0
  m.eval().requires_grad_(False);before={k:v.clone() for k,v in m.state_dict().items()}
  direct=default_logits(m,x)
  with torch.no_grad():expected=h.classify(e.features(x));zero=default_logits(m,torch.zeros(2,2,256))
  assert torch.equal(direct,expected) and torch.isfinite(zero).all()
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
  reports.append(dict(arm=c['arm'],head_type=type(h).__name__,head_shape=[6,160],head_scale=30,extra_feature_parameters=576,feature_branch_gradients=gradients,classifier_joint_CE=True,default_logits_exact=True,separate_head_updates=0,prediction_updates=0))
 negatives=0
 for change in [dict(epochs=51),dict(epochs=20),dict(readouts=['FRESH_COSINE_ALL_L']),dict(fresh_cosine_probe={}),dict(checkpoint_sources=['unknown']),dict(target_access=True),dict(classifier_training='head-only')]:
  c=copy.deepcopy(configs[0]);c.update(change)
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Unregistered recipe/head work accepted')
 return dict(status='PASS',primitive=primitive,rows=reports,matched_initialization=True,zero_gain_exact_baseline=True,negative_config_cases=negatives,real_IQ_or_target_read=False)
if __name__=='__main__':print(json.dumps(checks()))
