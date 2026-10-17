"""One-sided supervised-view projection in the existing physical-gradient coordinates."""
import time
import torch
from experiments.cvs_rdrop_sinc_units.optimizer import QNAMES
MODES=('ORIGINAL','MEASURE','PROJECT')
def vector(named,gradients):
 if len(named)!=len(gradients):raise ValueError('Gradient/parameter alignment differs')
 values=[]
 for (name,p),g in zip(named,gradients):
  if g is not None and (g.shape!=p.shape or not bool(torch.isfinite(g).all())):raise FloatingPointError('Invalid view gradient')
  v=torch.zeros_like(p).detach() if g is None else g.detach()
  values.append(v.reshape(-1).double()/(25000. if name in QNAMES else 1.))
 return torch.cat(values)
def geometry(named,gO,gS):
 a=vector(named,gO);b=vector(named,gS);oo=torch.dot(a,a);ss=torch.dot(b,b);os=torch.dot(a,b)
 if not bool(torch.isfinite(torch.stack((oo,ss,os))).all()):raise FloatingPointError('Nonfinite view gradient geometry')
 alpha=float(-os/oo) if bool(os<0) and bool(oo>0) else 0.
 return dict(dot=float(os),ordinary_norm=float(oo.sqrt()),satellite_norm=float(ss.sqrt()),cosine=float(os/(oo*ss).sqrt()) if bool(oo>0) and bool(ss>0) else None,coefficient=alpha,conflict=bool(os<0),projected_satellite_dot=float(os+alpha*oo),projection_delta_norm=alpha*float(oo.sqrt()))
def capture(model,ordinary,satellite,step):
 if model._pc_pending is not None or model._pc_mode not in MODES or step!=model._pc_clips+1:raise ValueError('One captured L pair per native update required')
 if model._pc_mode=='ORIGINAL':return
 start=time.perf_counter();named=tuple((k,p) for k,p in model.named_parameters() if p.requires_grad);params=tuple(p for _,p in named)
 before=[p.grad for p in params]
 gO=torch.autograd.grad(ordinary,params,retain_graph=True,allow_unused=True)
 gS=torch.autograd.grad(satellite,params,retain_graph=True,allow_unused=True)
 if any(p.grad is not v for p,v in zip(params,before)):raise ValueError('autograd.grad changed native accumulation')
 gO=tuple(None if g is None else g.detach() for g in gO);gS=tuple(None if g is None else g.detach() for g in gS)
 stats=geometry(named,gO,gS);model._pc_vjp_calls+=2
 model._pc_pending=(named,gO,stats,step,time.perf_counter()-start)
def consume(model,parameters,max_norm):
 params=list(parameters)
 if float(max_norm)!=5. or {id(p) for p in params}!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('Only native allparameter physical clip5 is permitted')
 step=model._pc_clips+1;mode=model._pc_mode
 if mode=='ORIGINAL':
  if model._pc_pending is not None:raise ValueError('Original cannot capture gradients')
  record=dict(step=step,mode=mode,extra_VJP_calls=0,applied=False,coefficient=0.)
 else:
  if model._pc_pending is None:raise ValueError('Missing current L view gradients')
  named,gO,stats,captured,elapsed=model._pc_pending
  if captured!=step or {id(p) for _,p in named}!={id(p) for p in params}:raise ValueError('Stale or mismatched view gradients')
  coefficient=stats['coefficient'] if mode=='PROJECT' else 0.
  with torch.no_grad():
   for (_,p),g in zip(named,gO):
    if g is not None:
     if p.grad is None:raise ValueError('Native total backward omitted ordinary gradient')
     if coefficient:p.grad.add_(g,alpha=coefficient)
  record=dict(step=step,mode=mode,extra_VJP_calls=2,applied=bool(coefficient),applied_coefficient=coefficient,capture_host_enqueue_seconds=elapsed,**stats)
 model._pc_pending=None;model._pc_clips=step;model._pc_records.append(record)
 return params
