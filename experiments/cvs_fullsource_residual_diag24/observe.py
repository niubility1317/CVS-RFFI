"""Readonly existing-path hooks and evaluation-mode local CE gradients."""
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .engine import tensor_iq
from .model import modules,gradient,compare_grad
NAMES=('time','frequency','PA','reference','signature')
@contextmanager
def capture(m):
 values={};handles=[]
 for name,module in modules(m).items():
  def record(mod,args,out,name=name):values[name]=out.detach().flatten(1)
  handles.append(module.register_forward_hook(record))
 try:yield values
 finally:
  for h in handles:h.remove()
def forward(m,x,values):
 values.clear();out=m.forward_identity_only(x)
 if set(values)!=set(NAMES) or out['tx_logits'].shape!=(len(x),6) or not torch.isfinite(out['tx_logits']).all():raise ValueError('Actual branch/defaulthead forward invalid')
 return out,{k:v.clone() for k,v in values.items()}
def predict(m,panel,observations,device,out):
 clean=[];branches={k:[] for k in NAMES};sat=np.empty((2,3,11,d.N,6),np.float32);similar=np.empty((2,3,d.N,5,3),np.float32)
 with capture(m) as values,torch.no_grad():
  for start in range(0,d.N,d.BATCH):
   stop=start+d.BATCH;q,b=forward(m,panel['x'][start:stop].to(device),values);clean.extend(q['tx_logits'].cpu().tolist())
   for k,v in b.items():branches[k].append(v.cpu())
  branches={k:torch.cat(v) for k,v in branches.items()}
  for draw in d.DRAWS:
   for si,scene in enumerate(d.SCENES):
    for ci,condition in enumerate(d.CONDITIONS):
     for start in range(0,d.N,d.BATCH):
      stop=start+d.BATCH;q,b=forward(m,tensor_iq(observations[draw,si,ci,start:stop],device),values);sat[draw,si,ci,start:stop]=q['tx_logits'].cpu().tolist()
      if ci==0:
       for ki,k in enumerate(NAMES):
        a=branches[k][start:stop].to(device);z=b[k]
        similar[draw,si,start:stop,ki]=torch.stack((F.cosine_similarity(a,z),a.norm(dim=1),z.norm(dim=1)),1).cpu().tolist()
 np.savez_compressed(out/'predictions.npz',ids=np.asarray(panel['ids']),TX=panel['y'].tolist(),RX=panel['rx'].tolist(),day=panel['day'].tolist(),clean=np.asarray(clean,np.float32),satellite=sat)
 np.savez_compressed(out/'branch_metrics.npz',names=np.asarray(NAMES),metrics=np.asarray(['clean_sat_cosine','clean_norm','sat_norm']),values=similar)
 return np.asarray(clean,np.float32),sat,similar
def gradients(m,panel,observations,device):
 before_flags={p:p.requires_grad for p in m.parameters()};m.eval().requires_grad_(True);named=list(m.named_parameters());out=[]
 try:
  x=panel['x'][:d.GRAD_N].to(device);y=panel['y'][:d.GRAD_N].to(device)
  clean=m.forward_identity_only(x)['tx_logits'];ce=F.cross_entropy(clean,y,label_smoothing=.01);cg=gradient(ce,named)
  for draw in d.DRAWS:
   for si,scene in enumerate(d.SCENES):
    x=tensor_iq(observations[draw,si,0,:d.GRAD_N],device);logits=m.forward_identity_only(x)['tx_logits'];value=F.cross_entropy(logits,y,label_smoothing=.01);sg=gradient(value,named)
    out.append(dict(draw=draw,scene=scene,count=d.GRAD_N,clean_CE=float(ce.detach()),satellite_CE=float(value.detach()),groups=compare_grad(cg,sg),scope='frozen weights, deterministic eval-mode smoothedCE, not stochastic training gradients'))
 finally:
  for p,flag in before_flags.items():p.requires_grad_(flag)
 if any(p.grad is not None for p in m.parameters()):raise ValueError('Readonly autograd accumulated parameter .grad')
 return out
