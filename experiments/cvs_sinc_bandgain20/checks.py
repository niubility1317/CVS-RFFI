"""Actual-network isolation, physical decay and displacement norm controls."""
import json,sys
import torch
from . import design as d,model as api
from .optimizer import gains,adjusted,attach,MODES,PREFIX
from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
from experiments.cvs_sigtime_rdrop_legacy.model import terms
from experiments.cvs_equivariant_identity.precision import numerical_context
W='id_backbone.encoder.core.id_backbone.cls_head.weight'
def protocol_checks():
 from copy import deepcopy
 c=d.config(d.rows()[0]);negative=0
 variants=[]
 for field,value in (('test_views',['clean']),('separate_head_training',True),('baseline_ckpt','forbidden'),('output_root','old_or_target_root')):
  v=deepcopy(c);v[field]=value;variants.append(v)
 v=deepcopy(c);v['fold']['held_RX']=3;variants.append(v)
 for v in variants:
  try:d.validate(v)
  except ValueError:negative+=1
  else:raise ValueError('Forbidden fixed source contract accepted')
 def result(errors,ce):return dict(held_default_head=dict(count=1260,errors=errors,mean_CE=ce),actual_steps=4440,head_fit_updates=0,target_access=False,final_L_CE=.2,feature_witness=dict(feature_export_exact=True,classW_zeroing_leaves_feature_identical=True))
 rows=dict(FULL=result(100,.3),NORM=result(90,.29),BAND=result(80,.28))
 if not d.qualify(rows)['qualified']:raise ValueError('Valid source decision rejected')
 rows['BAND']=result(90,.28)
 if d.qualify(rows)['qualified']:raise ValueError('Tied source errors incorrectly qualify')
 rows['BAND']=result(80,.31)
 if d.qualify(rows)['qualified']:raise ValueError('Worse source CE incorrectly qualifies')
 return dict(status='PASS',protocol_negative_cases=negative,source_gate_cases=3,source_IQ_reads=0,head_fit_updates=0)
def run(device='cpu'):
 torch.set_num_threads(2);states={};rows=[];physical_decay=[];negative=0;reference_gradients=None;reference_initial=None
 for mode in MODES:d.make_args(d.config(next(r for r in d.rows() if r['arm']==mode)),'cpu')
 with numerical_context(d.FULL_FP32_POLICY):
  for mode in MODES:
   c=d.config(next(r for r in d.rows() if r['arm']==mode));m=api.build(c,device).train();initial={n:p.detach().clone() for n,p in m.named_parameters()};params=list(m.parameters());opt=build(m,params,'CENTER_UNITS',lr=.0002,weight_decay=.0001,eps=1e-8);policy=attach(opt,m,mode)
   torch.manual_seed(991);x=torch.randn(8,2,256,device=device);y=torch.arange(8,device=device)%6
   first=m(x,y_tx=y,return_aux=True,domain_labels=None)['tx_logits'];second=m.forward_identity_only(x)['tx_logits'];loss,_,_,_,_=terms(first,second,y,1.);loss.backward()
   if dict(m.named_parameters())[W].grad is None or not bool(torch.isfinite(loss)):raise ValueError('Joint default-head training path missing')
   gradients={n:None if p.grad is None else p.grad.detach().clone() for n,p in m.named_parameters()}
   if reference_gradients is None:reference_gradients=gradients;reference_initial=initial;gradient_difference=0.
   else:
    if any(not torch.equal(initial[n],reference_initial[n]) for n in initial):raise ValueError('Matched controls have different initialization')
    gradient_difference=max(float((g-reference_gradients[n]).abs().max()) for n,g in gradients.items() if g is not None)
    for n,p in m.named_parameters():p.grad=None if reference_gradients[n] is None else reference_gradients[n].clone()
   # Isolate the optimizer intervention with exactly identical gradients.
   # Independent native CUDA forwards/backwards need not be bitwise deterministic.
   clip_physical(opt,params,5.);opt.step();opt.zero_grad(set_to_none=True)
   state={n:p.detach().clone() for n,p in m.named_parameters()};states[mode]=state
   if torch.equal(state[W],initial[W]) or opt._band_gain_steps!=1 or opt._sinc_physical_clip_calls!=1:raise ValueError('Joint actual update missing')
   rows.append(dict(mode=mode,independent_forward_gradient_max_abs_difference=gradient_difference,gain_min=min(policy['weights']),gain_max=max(policy['weights']),actual_step=opt._band_gain_records[-1]))
   # A zero-adaptive step on a fresh optimizer tests that origin decay is
   # neither gain-scaled nor applied twice in any treatment.
   z=api.build(c,device);zp=list(z.parameters());zero=build(z,zp,'CENTER_UNITS',lr=.0002,weight_decay=.0001,eps=1e-8);attach(zero,z,mode);s=z.get_submodule(PREFIX);orig=[s.low_origin_hz.clone(),s.band_origin_hz.clone()]
   for p in zp:p.grad=torch.zeros_like(p)
   clip_physical(zero,zp,5.);zero.step()
   for q,o in zip(zero._sinc_unit_policy['sinc'],orig):
    expected=-(.0002*.0001)*(o/25000.)
    if not torch.allclose(q,expected,atol=2e-11,rtol=2e-7):raise ValueError('Physical origin decay changed')
   physical_decay.append(mode)
  for name in states['FULL']:
   if not name.startswith(PREFIX) and any(not torch.equal(states['FULL'][name],states[a][name]) for a in ('NORM','BAND')):raise ValueError('Gain mechanism changed body/defaultW first update')
  if all(torch.equal(states['FULL'][n],states['BAND'][n]) for n in states['FULL'] if n.startswith(PREFIX)):raise ValueError('Adaptive Sinc intervention missing')
  a=[torch.linspace(-.3,.4,24,device=device)[:,None],torch.linspace(.5,-.1,24,device=device)[:,None]];w=[torch.linspace(.02,.2,24,device=device)[:,None]]*2
  scaled,stats=adjusted(a,w,'NORM');band,_=adjusted(a,w,'BAND');nf=torch.cat(scaled).double().norm();nb=torch.cat(band).double().norm()
  if not torch.allclose(nf,nb,atol=1e-8,rtol=2e-7):raise ValueError('Local adaptive norm control differs')
  for args in ((a,w,'INVALID'),(a,[torch.ones(23,1,device=device)]*2,'BAND'),(a,[torch.zeros(24,1,device=device)]*2,'BAND')):
   try:adjusted(*args)
   except ValueError:negative+=1
   else:raise ValueError('Invalid displacement allowed')
 return dict(status='PASS',device=device,conditions=rows,physical_origin_decay_unscaled_once=physical_decay,body_defaultW_first_update_bitwise_equal_under_identical_gradient=True,local_norm_difference=float(abs(nf-nb)),negative_cases=negative,source_IQ_reads=0,head_fit_updates=0,synthetic_joint_optimizer_steps=6)
if __name__=='__main__':print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else 'cpu'),allow_nan=False))
