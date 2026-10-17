"""Replay exact L observations/RNG;replace only L gradient before the original physical clip."""
import random,time
import numpy as np
import torch
from experiments.cvs_fullsource_concat20 import supcon
from experiments.cvs_sigtime_rdrop_legacy.model import terms

def rng_get(x):return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state(x.device).clone() if x.is_cuda else None)
def rng_set(s,x):
 random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2])
 if s[3] is not None:torch.cuda.set_rng_state(s[3],x.device)
def direction(named,gradients,radius):
 if len(named)!=len(gradients) or radius not in (0.,.05):raise ValueError('Fixed ordered gradient/radius required')
 norm2=torch.zeros((),dtype=torch.float64,device=named[0][1].device)
 for (_,p),g in zip(named,gradients):
  if g is not None:
   if g.shape!=p.shape or not bool(torch.isfinite(g).all()):raise FloatingPointError('Invalid supervised gradient')
   norm2+=g.detach().double().square().sum()
 norm=float(norm2.sqrt());scale=radius/norm if norm else 0.
 return tuple(None if g is None else g.detach()*scale for g in gradients),norm
def capture(m,loss,labels,rx,domains,weight,step):
 if m._sam_pending is not None or step!=m._sam_steps+1 or len(m._sam_draws)!=4 or not m._sam_recording:raise ValueError('One actual ordered4forward L objective required')
 m._sam_recording=False;named=tuple((k,p) for k,p in m.named_parameters() if p.requires_grad);params=tuple(p for _,p in named);before=[p.grad for p in params]
 grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
 if any(p.grad is not g for p,g in zip(params,before)):raise ValueError('Capture changed native gradient accumulation')
 g=tuple(None if v is None else v.detach() for v in grads);direction(named,g,m._sam_radius)
 m._sam_pending=dict(named=named,gradients=g,draws=m._sam_draws,labels=labels.detach(),rx=rx.detach(),domains=domains.detach(),weight=weight,step=step,loss=float(loss.detach()))
 m._sam_draws=[];m._sam_vjp_calls+=1
def replay(m,pending):
 logits=[];maps=[]
 for i,(x,state) in enumerate(pending['draws']):
  rng_set(state,x);local={};handles=[]
  try:
   if i<2:
    for level in supcon.LEVELS:
     def grab(module,args,out,level=level):
      if level in local:raise ValueError('Repeated feature layer in replay')
      local[level]=out
     handles.append(m.get_submodule(supcon.PREFIX+level).register_forward_hook(grab))
   logits.append(m.forward_identity_only(x,domain_labels=pending['domains'])['tx_logits'])
   if i<2:maps.append(local)
  finally:
   for h in handles:h.remove()
 a,*_=terms(logits[0],logits[1],pending['labels'],1.);b,*_=terms(logits[2],logits[3],pending['labels'],1.)
 aux=torch.stack([supcon.penalty(v,pending['labels'],pending['rx'],'CROSS_RX')[0] for v in maps]).mean()
 return .5*a+.5*b+pending['weight']*aux
def consume(m,parameters,max_norm):
 params=list(parameters);p=m._sam_pending;step=m._sam_steps+1
 if float(max_norm)!=5. or p is None or p['step']!=step or {id(v) for v in params}!={id(v) for _,v in p['named']}:raise ValueError('One ordered original joint clip5 required')
 x=p['draws'][0][0];outside=rng_get(x);named=p['named'];saved=[v.detach().clone() for _,v in named];buffers={k:v.detach().clone() for k,v in m.named_buffers()};before=[v.grad for _,v in named];deltas,norm=direction(named,p['gradients'],m._sam_radius);start=time.perf_counter()
 try:
  with torch.no_grad():
   for (_,v),delta in zip(named,deltas):
    if delta is not None:v.add_(delta)
  actual_delta_norm=float(sum((v.detach().double()-s.double()).square().sum() for (_,v),s in zip(named,saved)).sqrt())
  with torch.enable_grad(),torch.autocast(device_type=x.device.type,enabled=False):
   loss=replay(m,p);fresh=torch.autograd.grad(loss,tuple(v for _,v in named),allow_unused=True)
  m._sam_vjp_calls+=1;m._sam_replay_packets+=sum(len(v) for v,_ in p['draws'])
  if any(v.grad is not g for (_,v),g in zip(named,before)):raise ValueError('Replay changed native gradient accumulation')
  if any((a is None)!=(b is None) for a,b in zip(p['gradients'],fresh)):raise ValueError('Replay changed used parameter set')
  _,newnorm=direction(named,fresh,0.);difference=float(loss.detach())-p['loss'];error2=sum((a.detach().double()-b.double()).square().sum() for a,b in zip(fresh,p['gradients']) if a is not None);relative=float(error2.sqrt())/max(norm,1e-12)
  if m._sam_radius==0. and (abs(difference)>5e-5+1e-5*abs(p['loss']) or relative>1e-4):raise ValueError('Neutral replay differs from actual L objective')
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite replay loss')
  if any(not torch.equal(v,buffers[k]) for k,v in m.named_buffers()):raise ValueError('Replay modified a persistent model buffer')
 finally:
  with torch.no_grad():
   for (_,v),s in zip(named,saved):v.copy_(s)
  rng_set(outside,x)
 if any(not torch.equal(v.detach(),s) for (_,v),s in zip(named,saved)):raise ValueError('Parameter perturbation was not exactly restored')
 if m._sam_radius:
  with torch.no_grad():
   for (_,v),old,new in zip(named,p['gradients'],fresh):
    if old is not None:
     if v.grad is None:raise ValueError('Native total omitted an L gradient')
     v.grad.add_(new-old)
 m._sam_records.append(dict(step=step,mode=m._sam_mode,radius=m._sam_radius,L_loss=p['loss'],replay_L_loss=float(loss.detach()),L_gradient_norm=norm,replay_gradient_norm=newnorm,replay_gradient_relative_difference=relative,actual_perturbation_L2=actual_delta_norm,weights_exactly_restored=True,global_RNG_restored=True,persistent_buffers_unchanged=True,L_forwards=8,extra_L_forward_packets=512,extra_VJP_calls=2,U_replayed=False,teacher_replayed=False,host_enqueue_seconds=time.perf_counter()-start))
 m._sam_steps=step;m._sam_pending=None
 return params
