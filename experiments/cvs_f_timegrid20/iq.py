"""Bounded interpolation within original observed samples;no synthetic boundary."""
import random,time
from collections import Counter
import torch
from . import design as d

def transform(x,rho):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):raise ValueError('Finite floating IQ2x256 required')
 rho=torch.as_tensor(rho,dtype=torch.float64,device=x.device)
 if rho.shape!=(len(x),) or not bool(torch.isfinite(rho).all()) or bool(((rho<0)|(rho>d.RHO_MAX)).any()):raise ValueError('One bounded nonnegative rho perpacket required')
 n=torch.arange(256,device=x.device,dtype=torch.float64)[None,:];t=127.5+(1-rho[:,None])*(n-127.5);lo=t.floor().long();hi=(lo+1).clamp(max=255);w=(t-lo).to(x.dtype)[:,None,:];a=x.gather(2,lo[:,None,:].expand(-1,2,-1));b=x.gather(2,hi[:,None,:].expand(-1,2,-1));y=a+(b-a)*w
 if bool(((t<0)|(t>255)).any()) or not bool(torch.isfinite(y).all()):raise ValueError('No external sample or nonfinite output allowed')
 return y
class Runtime:
 def __init__(self,trace):self.trace=trace;self.calls=0;self.packets=0;self.sessions={};self.coverage=Counter();self.max_relative_L2=0.;self.seconds=0.
 def apply(self,x,meta,labels,epoch,step):
  started=time.perf_counter()
  if epoch not in range(1,21) or step!=self.calls+1 or labels.shape!=(len(x),) or bool(((labels<0)|(labels>5)).any()):raise ValueError('One ordered visibleL update only')
  keys=('eq_i','rx_i','day_i','sig_i','base_index');ids=list(zip(*(meta[k].detach().cpu().tolist() for k in keys)));ys=labels.detach().cpu().tolist();rhos=[]
  if len(ids)!=len(x) or len(set(ids))!=len(ids):raise ValueError('Unique actual source physical128 required')
  for eq,rx,day,sig,base in ids:
   if eq!=1 or rx not in d.FIT or day not in (1,2,3):raise ValueError('No U/V/held/target session allowed')
   key=(epoch,eq,rx,day);rho=d.RHO_MIN+(d.RHO_MAX-d.RHO_MIN)*random.Random(d.GRID_SEED+epoch*4096+rx*64+day*8+eq).random()
   if self.sessions.setdefault(key,rho)!=rho:raise ValueError('Session rho changed within epoch')
   rhos.append(rho)
  y=transform(x,rhos);power=x.detach().double().square().sum((1,2));change=(y.detach().double()-x.detach().double()).square().sum((1,2));rel=(change/power.clamp_min(1e-30)).sqrt();ratio=y.detach().double().square().sum((1,2))/power.clamp_min(1e-30)
  if bool((rel>d.L2_BOUND+1e-5).any()) or bool(((power==0)&(change!=0)).any()):raise ValueError('Universal numerical L2 bound/zero preservation differs')
  rels=rel.cpu().tolist();powers=ratio.cpu().tolist()
  import json
  for pid,label,rho,r,p in zip(ids,ys,rhos,rels,powers):
   self.coverage[label,pid[1],pid[2]]+=1;self.max_relative_L2=max(self.max_relative_L2,r);self.trace.write(json.dumps(dict(step=step,epoch=epoch,physical_id=pid,TX=label,rho=rho,max_displacement_samples=127.5*rho,relative_L2=r,power_ratio=p,session=[epoch,pid[0],pid[1],pid[2]],padding=False,normalization=False),allow_nan=False)+'\n')
  self.calls+=1;self.packets+=len(x);self.seconds+=time.perf_counter()-started;return y
 def __deepcopy__(self,memo):return None
 def summary(self):return dict(calls=self.calls,packets=self.packets,sessions=len(self.sessions),coverage_cells=len(self.coverage),max_relative_L2=self.max_relative_L2,construct_seconds=self.seconds,source_only=True,As=False)
