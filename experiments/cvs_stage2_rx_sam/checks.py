import argparse,copy,json
import torch
from torch import nn
from . import design as d
from .objective import objective
from .sam import buffers,restore_buffers,rng_state,set_rng,sam_backward
from .model import OriginalIdentity,CurrentStudentTarget,groups
from .receiver import ReceiverActions
from experiments.cvs_stage2_rx_risk.checks import synthetic_aux
from experiments.cvs_stage2_rx_actions.checks import synthetic

def assert_state(a,b):
 assert a.keys()==b.keys()
 assert all(torch.equal(a[k],b[k]) for k in a)

def unit_checks():
 # Analytic scalar quadratic: SAM gradient is w+rho; restoration precedes SGD step.
 m=nn.Linear(1,1,bias=False);m.weight.data.fill_(2.);o=torch.optim.SGD(m.parameters(),lr=.1)
 loss=m.weight.square().sum()/2;loss.backward();v=sam_backward(m,o,.05,buffers(m),lambda:m.weight.square().sum()/2)
 assert torch.equal(m.weight,torch.tensor([[2.]])) and abs(m.weight.grad.item()-2.05)<1e-6
 o.step();assert abs(m.weight.item()-1.795)<1e-6
 # BN/dropout: same stochastic objective at rho0, only first-pass BN effects persist.
 torch.manual_seed(51);m=nn.Sequential(nn.Linear(3,5),nn.BatchNorm1d(5),nn.Dropout(.4),nn.Linear(5,2)).train();o=torch.optim.AdamW(m.parameters(),lr=.001)
 x=torch.randn(8,3);start=buffers(m);rng=rng_state();loss=m(x).square().mean();loss.backward()
 state=copy.deepcopy(m.state_dict());grad=[p.grad.clone() for p in m.parameters()];end=rng_state()
 def closure():set_rng(rng);return m(x).square().mean()
 sam_backward(m,o,0.,start,closure)
 assert_state(m.state_dict(),state);assert torch.equal(rng_state()[0],end[0])
 assert all(torch.equal(p.grad,g) for p,g in zip(m.parameters(),grad))
 # Failure also restores exact parameters, buffers, RNG; caller never steps on failure.
 o.zero_grad();m(x).sum().backward();state=copy.deepcopy(m.state_dict());end=rng_state()
 def fail():
  m(x);raise RuntimeError('intentional closure failure')
 try:sam_backward(m,o,.05,start,fail)
 except RuntimeError as e:assert str(e)=='intentional closure failure'
 else:raise AssertionError('Expected failure')
 assert_state(m.state_dict(),state);assert torch.equal(rng_state()[0],end[0])
 return dict(analytic_gradient=True,exact_restore_normal_and_exception=True,BN_once=True,dropout_replay=True,rho0_gradient_equivalence=True)

def check():
 from experiments.cvs_stage2_decouple.design import make_args
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from experiments.cvs_stage2_rx_risk.objective import objective as old_objective
 from .source import component_gradients
 unit=unit_checks();torch.manual_seed(221);torch.set_num_threads(2)
 L=synthetic();U={k:v for k,v in L.items() if k!='y'};idx=torch.arange(32)*70
 pc=d.previous.previous.previous.original.parent_config(d.SEEDS[0]);a=make_args(pc,'cpu');a.num_domains,a.input_len=5,256
 with numerical_context(d.FULL_FP32_POLICY),installed(pc) as native:base=native.build_baseline_model(a,torch.device('cpu'))
 base._stage2_baseline_args=vars(a);records=[]
 for arm in ('SSL','PHYS','RANK'):
  c=d.config(next(r for r in d.rows() if r['arm']==arm));oldc=d.previous.config(next(r for r in d.previous.rows() if r['arm']==arm))
  m1=OriginalIdentity(copy.deepcopy(base)).train();m2=OriginalIdentity(copy.deepcopy(base)).train();pg,_=groups(m1,c);groups(m2,oldc)
  pre=synthetic_aux();h1=ReceiverActions(c,L,pre);h2=ReceiverActions(oldc,L,pre)
  torch.manual_seed(991);v1,_,weighted,info=objective(c,m1,h1,L,idx,U,idx,CurrentStudentTarget(m1),None,1776,'cpu');v1.backward()
  torch.manual_seed(991);v2=old_objective(oldc,m2,h2,L,idx,U,idx,CurrentStudentTarget(m2),None,1776,'cpu')[0];v2.backward()
  assert torch.equal(v1,v2),(arm,float(v1),float(v2));assert_state(m1.state_dict(),m2.state_dict())
  assert all((p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad) for p,q in zip(m1.parameters(),m2.parameters()))
  c=d.config(next(r for r in d.rows() if r['arm']=='SAM_'+arm));d.validate(c)
  m=OriginalIdentity(copy.deepcopy(base)).train();pg,_=groups(m,c);heads=ReceiverActions(c,L,pre);opt=torch.optim.AdamW(pg,weight_decay=1e-4);target=CurrentStudentTarget(m)
  before=buffers(m);tape={};value,_,weighted,info=objective(c,m,heads,L,idx,U,idx,target,None,1776,'cpu',replay=tape)
  audit=component_gradients(weighted,pg,m);value.backward();expected=[p.grad.clone() if p.grad is not None else None for p in m.parameters()]
  aux=[p.grad.clone() if p.grad is not None else None for p in heads.parameters()];state=copy.deepcopy(m.state_dict());rng=rng_state();target_calls=target.calls
  def replay():return objective(c,m,heads,L,idx,U,idx,target,None,1776,'cpu',replay=tape,reuse=True)[0]
  # No new selection or auxiliary supervision may run at perturbed weights.
  def forbidden(*args,**kwargs):raise AssertionError('Frozen action/critic recomputed')
  heads.choose=forbidden;heads.critic_objective=forbidden
  sam_backward(m,opt,0.,before,replay)
  assert_state(m.state_dict(),state);assert torch.equal(rng_state()[0],rng[0]);assert target.calls==target_calls
  assert all((p.grad is None and g is None) or torch.equal(p.grad,g) for p,g in zip(m.parameters(),expected))
  assert all((p.grad is None and g is None) or torch.equal(p.grad,g) for p,g in zip(heads.parameters(),aux))
  metrics=sam_backward(m,opt,.05,before,replay)
  assert abs(metrics['actual_perturbation_l2']-.05)<1e-5;assert_state(m.state_dict(),state)
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  assert all((p.grad is None and g is None) or torch.equal(p.grad,g) for p,g in zip(heads.parameters(),aux))
  if arm=='RANK':assert info['critic_packets']==32 and all(v==0 for v in audit['weighted_dense_gradient_norms']['critic'].values())
  opt.step();assert any(not torch.equal(v,m.state_dict()[k]) for k,v in state.items())
  records.append(dict(arm=arm,ordinary_matches_prior_loss_and_gradient=True,SAM_finite_update=True,targets_actions_auxiliary_once=True,auxiliary_direct_identity_gradient_zero=True))
 assert len(d.rows())==18 and all(d.validate(d.config(r)) for r in d.rows())
 return dict(status='PASS',rows=18,unit=unit,arms=records,target_access=False)

def preflight():
 from .source import load_source
 from .model import build
 result=check();parents=[]
 for seed in d.SEEDS:
  c=d.config(next(r for r in d.rows() if r['arm']=='SAM_RANK' and r['model_seed']==seed));L,U,V,contract,proof=load_source(c)
  model=build(c,'cpu').train();pg,_=groups(model,c);heads=ReceiverActions(c,L);idx=torch.arange(4);opt=torch.optim.AdamW(pg,weight_decay=1e-4);target=CurrentStudentTarget(model)
  before=buffers(model);tape={};value,_,_,info=objective(c,model,heads,L,idx,U,idx,target,None,1776,'cpu',replay=tape);value.backward();state=copy.deepcopy(model.state_dict())
  metrics=sam_backward(model,opt,.05,before,lambda:objective(c,model,heads,L,idx,U,idx,target,None,1776,'cpu',replay=tape,reuse=True)[0]);assert_state(model.state_dict(),state)
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  parents.append(dict(seed=seed,parent=proof,auxiliary=heads.provenance,real_L_U_SAM_backward=metrics,temporary_weights_restored=True))
 result['parents']=parents;return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();v=preflight() if a.remote_preflight else check();d.write(a.output,v);print(json.dumps(v))
