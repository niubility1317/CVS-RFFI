from contextlib import contextmanager
import torch
from experiments.cvs_e20_cross_style.style import _paths
from . import design as d

@contextmanager
def capture(model):
 modules,paths=_paths(model.base);maps={};handles=[]
 try:
  for level in d.LEVELS:
   def hook(module,inputs,out,level=level):
    if level in maps or out.ndim!=3 or not out.is_floating_point():raise ValueError('Expected one real native channel/time map per level')
    maps[level]=out
   handles.append(modules[paths[level]].register_forward_hook(hook))
  yield maps
  if tuple(maps)!=d.LEVELS:raise ValueError('Actual shallow/middle capture missing or reordered')
 finally:
  for h in handles:h.remove()

def penalty(maps,y,rx,c):
 d.validate(c)
 if y.ndim!=1 or rx.shape!=y.shape or y.dtype!=torch.long or rx.dtype!=torch.long or not set(y.tolist())<=set(range(6)) or not set(rx.tolist())<=set(d.RX)-{c['held_rx']}:
  raise ValueError('Alignment requires current fit L TX/RX metadata, held RX excluded')
 if tuple(maps)!=d.LEVELS:raise ValueError('Alignment actual layer list differs')
 losses=[];details=[]
 for level,h in maps.items():
  if h.ndim!=3 or len(h)!=len(y) or not torch.isfinite(h).all():raise ValueError('Alignment feature shape/values differ')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1)
  z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4)
  groups=[];pair_count=0;occurrences=0
  for tx in range(6) if c['arm']=='TX_CONDITIONAL' else (None,):
   mask=torch.ones_like(y,dtype=torch.bool) if tx is None else y==tx
   centroids=[]
   for r in sorted(set(rx.tolist())):
    member=mask&(rx==r);n=int(member.sum())
    if n>=2:centroids.append(z[member].mean(0));occurrences+=n
   if len(centroids)>=2:
    a=torch.stack(centroids);groups.append((a-a.mean(0)).square().mean());pair_count+=len(centroids)*(len(centroids)-1)//2
  loss=torch.stack(groups).mean() if groups else h.sum()*0
  losses.append(loss);details.append(dict(level=level,shape=list(h.shape),eligible_TX_groups=len(groups),RX_pairs=pair_count,group_occurrences=occurrences,loss=float(loss.detach())))
 return torch.stack(losses).mean(),details
