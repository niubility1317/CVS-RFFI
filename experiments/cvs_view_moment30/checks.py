"""Focused lifecycle, second-moment recurrence and actual fixed-F integration."""
import copy,json,math
from types import SimpleNamespace
import torch
from . import core

def run(full=True):
 torch.set_num_threads(2)
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,FloatingPointError):negative+=1
  else:raise AssertionError('Invalid optimizer/lifecycle accepted')
 # Independent recurrence includes signed gradients, clipping, decay and unequal view contributions.
 maxerr=0.
 for mode in ('MEASURE','VIEW_SECOND'):
  p=torch.nn.Parameter(torch.tensor([.3,-.7],dtype=torch.float64));m=torch.nn.Module();m.register_parameter('p',p);core.initialize(m,mode)
  opt=torch.optim.AdamW([p],lr=.02,betas=(.8,.9),eps=1e-8,weight_decay=.03,foreach=False);core.attach_optimizer(opt,m)
  w=p.detach().clone();first=torch.zeros_like(w);second=torch.zeros_like(w)
  for step in range(1,9):
   a=torch.tensor([.4*step,-.2],dtype=torch.float64);b=torch.tensor([-.1,.15*step],dtype=torch.float64);extra=torch.tensor([.07,-.13],dtype=torch.float64)
   O=(p*a).sum();S=(p*b).sum();before=torch.random.get_rng_state().clone();core.capture(m,O,S,step);assert torch.equal(before,torch.random.get_rng_state()) and p.grad is None
   (O+S+(p*extra).sum()).backward();g=a+b+extra;norm=torch.nn.utils.clip_grad_norm_(m.parameters(),.6);factor=min(1.,.6/(float(norm)+1e-6));g=g*factor
   core.after_clip(m,list(m.parameters()),.6,norm);assert torch.allclose(p.grad,g,atol=1e-15,rtol=1e-15)
   var=(a-b).square()*factor**2 if mode=='VIEW_SECOND' else torch.zeros_like(g)
   first=.8*first+.2*g;second=.9*second+.1*(g.square()+var);w=w*(1-.02*.03)-.02*(first/(1-.8**step))/((second/(1-.9**step)).sqrt()+1e-8)
   opt.step();opt.zero_grad(set_to_none=True)
   err=float((p-w).abs().max());maxerr=max(err,maxerr);assert err<2e-14
   assert torch.allclose(opt.state[p]['exp_avg'],first,atol=2e-15,rtol=2e-15) and torch.allclose(opt.state[p]['exp_avg_sq'],second,atol=2e-15,rtol=2e-15)
  assert m._vm_steps==8 and m._vm_vjp_calls==16 and len(m._vm_records)==8 and m._vm_pending is None and m._vm_ready is None
 # Equal views leave native AdamW exactly unchanged, including optimizer state and no averaging of parameters.
 p=torch.nn.Parameter(torch.tensor([.2,-.4]));q=torch.nn.Parameter(p.detach().clone());m=torch.nn.Module();m.register_parameter('p',p);core.initialize(m,'VIEW_SECOND');a=torch.optim.AdamW([p],lr=.01,foreach=False);b=torch.optim.AdamW([q],lr=.01,foreach=False);core.attach_optimizer(a,m)
 for step in range(1,4):
  core.capture(m,p.square().sum(),p.square().sum(),step);(2*p.square().sum()).backward();(2*q.square().sum()).backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);torch.nn.utils.clip_grad_norm_([q],5.);core.after_clip(m,[p],5.,norm);a.step();b.step();a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True);assert torch.equal(p,q)
  for k in ('step','exp_avg','exp_avg_sq'):assert torch.equal(a.state[p][k],b.state[q][k])
 bad(lambda:core.initialize(m,'OTHER'))
 bad(lambda:core.after_clip(m,[p],5.,torch.tensor(1.)))
 bad(lambda:core.capture(m,p.sum(),p.sum(),9))
 for options in ({'betas':(.9,0.)},{'amsgrad':True},{'maximize':True},{'capturable':True},{'differentiable':True},{'fused':True}):
  test=torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]);test.param_groups[0].update(options);bad(lambda test=test:core.validate_optimizer(test))
 synthetic=0;integration=[]
 if full:
  from . import design as d,model as api
  from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
  from experiments.cvs_fullsource_concat20 import supcon
  c=d.config(d.rows()[0])
  for key,val in [('epochs',11),('cumulative_epochs',31),('new_losses',1),('new_U_rules',1),('model_seed',7),('moment_mode','BAD')]:
   cc=copy.deepcopy(c);cc[key]=val;bad(lambda cc=cc:d.validate(cc))
  for key,val in [('RX',[3,4,6,8]),('L',5580),('source_RX_holdout',1),('U_TX_truth_hidden',False)]:
   cc=copy.deepcopy(c);cc['data_contract'][key]=val;bad(lambda cc=cc:d.validate(cc))
  class Channel:
   def apply(self,x,meta,**kw):return x*.9,[dict(scene='practical_mid_urban',state_end='PARTIAL',receiver_lock_status='locked') for _ in x]
  old=supcon.schedule;supcon.schedule=lambda epoch:.1
  try:
   torch.manual_seed(20261018401);raw=torch.randn(128,2,256);ordinary=raw*.7+.13;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains];initial=None
   for row in d.rows():
    c=d.config(row);model=api.build(c,'cpu')
    if initial is None:initial=copy.deepcopy(model.state_dict())
    else:model.load_state_dict(initial,strict=True)
    if c['As']:model._full_channel=Channel()
    model.train();opt=build(model,list(model.parameters()),'CENTER_UNITS',lr=.0002,eps=1e-8,weight_decay=.0001,foreach=False);core.attach_optimizer(opt,model);torch.manual_seed(20261018402)
    with model._phase1_labeled_feature_context():out=model(ordinary,y_tx=y,return_aux=True)
    ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01);correction,_=model._source_labeled_feature_hook(model=model,labeled_output=out,labels=y,extra=(None,dict(rx_i=rx)),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=raw)
    (ce+correction).backward();params=list(model.parameters());norm=clip_physical(opt,params,5.);core.after_clip(model,params,5.,norm);before={id(p):(p.grad.detach().clone(),opt.state[p].get('exp_avg_sq',torch.zeros_like(p)).clone()) for p in params if p.grad is not None};pending=model._vm_ready;named,delta,stats,step,elapsed,fac=pending;opt.step()
    for (_,p),var in zip(named,delta):
     if p.grad is None:continue
     group=next(g for g in opt.param_groups if any(q is p for q in g['params']));beta=group['betas'][1];g,v=before[id(p)];expected=beta*v+(1-beta)*(g.square()+(var*fac*fac if c['moment_mode']=='VIEW_SECOND' and var is not None else 0));torch.testing.assert_close(opt.state[p]['exp_avg_sq'],expected,rtol=3e-6,atol=2e-8)
    teacher=copy.deepcopy(model);assert teacher._vm_pending is None and teacher._vm_ready is None and teacher._source_labeled_feature_hook.__self__ is teacher
    assert model._full_L_packets==512 and model._vm_vjp_calls==2 and model._vm_steps==1 and opt._sinc_unit_steps==1 and opt._sinc_physical_clip_calls==1
    assert sum(p.numel() for p in model.parameters())==900722
    synthetic+=512;integration.append(dict(row_id=row['row_id'],parameters=900722,L_forwards=4,VJP=2,default_head_shape=[6,160],Sinc_physical_clip=True))
  finally:supcon.schedule=old
 return dict(status='PASS',negative_cases=negative,manual_recurrence_steps=16,max_parameter_error=maxerr,equal_view_native_steps=3,synthetic_F_forward_packets=synthetic,synthetic_optimizer_updates=19+(4 if full else 0),synthetic_VJP_calls=38+(8 if full else 0),integration=integration,real_IQ=0,CP=0,real_updates=0,U_truth_read=False,target_access=False,new_Ag=False)
if __name__=='__main__':print(json.dumps(run()))
