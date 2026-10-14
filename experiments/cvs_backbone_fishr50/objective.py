"""Fishr-style actual backbone parameter-gradient variances; no fitted readout.
Adaptation differs from original Fishr classifier-gradient/EMA objective.
"""
from contextlib import contextmanager
import time
import torch
from torch.nn import functional as F
from . import design as d

def validate_layout(y,rx,day,ids,role='L_s'):
 y,rx,day=(v.detach().cpu() for v in (y,rx,day))
 if role!='L_s' or len(ids)!=180 or len(set(ids))!=180 or any(v.shape!=(180,) for v in (y,rx,day)):
  raise ValueError('Only180 unique physical labelled-source packets allowed')
 for values,expected in ((y,range(6)),(rx,d.RX),(day,d.DAYS)):
  if set(values.detach().cpu().tolist())!=set(expected):raise ValueError('Source label/RX/day set differs')
 for tx in range(6):
  for dd in d.DAYS:
   for receiver in d.RX:
    if int(((y==tx)&(day==dd)&(rx==receiver)).sum())!=2:raise ValueError('Need2 trueL packets perTX/day/RX')

def variance_terms(G,y,rx,day):
 if G.ndim!=2 or len(G)!=180 or not torch.isfinite(G.detach()).all():raise ValueError('Actual per-packet gradient matrix differs')
 G=G.double();group=y*3+(day-1);means=torch.stack([G[group==i].mean(0) for i in range(18)]);residual=G-means[group];losses={};meta={}
 for name,values in [('RAW',G),('TX_DAY_CENTERED',residual)]:
  vv=torch.stack([values[rx==receiver].var(0,unbiased=True) for receiver in d.RX]);average=vv.mean(0);numerator=(vv-average).square().mean();denominator=average.square().mean();term=numerator/denominator.clamp_min(1e-30)
  if not torch.isfinite(term.detach()):raise FloatingPointError('Nonfinite actual backbone variance objective')
  losses[name]=term;meta[name]=dict(loss=float(term.detach()),numerator=float(numerator.detach()),denominator=float(denominator.detach()),denominator_differentiable=True,denominator_floor=1e-30,variance_trace_by_RX=vv.detach().sum(1).cpu().tolist(),variance_denominator=35,coordinate_count=G.shape[1])
 return losses,meta

@contextmanager
def auxiliary_weight_constant(model):
 head=model.id_backbone.encoder.core.id_backbone.cls_head
 if type(head).__name__!='ResidualPhysicalHead' or head.weight.shape!=(6,160) or head.scale!=30:raise ValueError('Existing default classifier differs')
 before=head.weight.requires_grad
 try:
  head.weight.requires_grad_(False)
  yield head.weight
 finally:head.weight.requires_grad_(before)

def calculate(model,x,y,rx,day,ids,*,create_graph=True):
 validate_layout(y,rx,day,ids)
 if any(m.training for m in model.modules()) or x.shape!=(180,2,256):raise ValueError('Auxiliary180packet eval graph differs')
 named=dict(model.named_parameters());params=[named[n] for n in d.FISHR_PARAMETERS]
 if [p.numel() for p in params]!=[3456,6912] or not all(p.requires_grad for p in params):raise ValueError('Actual backbone derivative objects differ')
 start=time.perf_counter();vectors=[[],[]];CE=[]
 with auxiliary_weight_constant(model):
  for i in range(180):
   out=model.forward_identity_only(x[i:i+1]);loss=F.cross_entropy(out['tx_logits'],y[i:i+1],label_smoothing=.01)
   gg=torch.autograd.grad(loss,params,create_graph=create_graph,retain_graph=create_graph,allow_unused=False)
   for target,g in zip(vectors,gg):target.append(g.flatten())
   CE.append(loss.detach())
 matrices=[torch.stack(v) for v in vectors];terms={key:[] for key in d.LEVELS};stats=[]
 for name,G in zip(d.FISHR_PARAMETERS,matrices):
  losses,detail=variance_terms(G,y,rx,day)
  for key in d.LEVELS:terms[key].append(losses[key])
  stats.append(dict(parameter=name,views=detail))
 losses={key:torch.stack(v).mean() for key,v in terms.items()}
 return losses,dict(parameter_statistics=stats,source_CE_mean=float(torch.stack(CE).mean()),packet_forwards=180,seconds=time.perf_counter()-start,parameter_gradient_coordinates=10368,head_fit_updates=0,auxiliary_classW_constant=True),matrices
