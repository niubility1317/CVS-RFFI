"""Fixed matched donor permutations; two-view centers and total energy preserved."""
import torch
from torch.nn import functional as F
FIT=(3,4,6,8)
def donors(y,meta):
 if y.ndim!=1 or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()):raise ValueError('Visible legal source L identity labels required')
 n=len(y)
 if n<1 or set(('eq_i','rx_i','day_i','base_index'))-set(meta):raise ValueError('Actual L physical metadata required')
 values={k:meta[k].detach().cpu().tolist() for k in ('eq_i','rx_i','day_i','base_index')}
 if any(len(v)!=n for v in values.values()) or len(set(values['base_index']))!=n or any(r not in FIT for r in values['rx_i']) or any(day not in (1,2,3) for day in values['day_i']):raise ValueError('Unique allowed source L metadata required')
 cells={}
 for i,(t,e,day,rx) in enumerate(zip(y.detach().cpu().tolist(),values['eq_i'],values['day_i'],values['rx_i'])):cells.setdefault((t,e,day),{}).setdefault(rx,[]).append(i)
 same=list(range(n));cross=list(range(n));cohort=[]
 for key in sorted(cells):
  receivers={rx:indices[:2] for rx,indices in cells[key].items() if len(indices)>=2}
  order=sorted(receivers)
  if len(order)<2:continue
  for pos,rx in enumerate(order):
   current=receivers[rx];other=receivers[order[(pos+1)%len(order)]];cohort.extend(current)
   for slot,i in enumerate(current):same[i]=current[1-slot];cross[i]=other[slot]
 expected=list(range(n))
 if sorted(same)!=expected or sorted(cross)!=expected:raise ValueError('Donors must use each original residual once')
 return {k:torch.tensor(v,device=y.device,dtype=torch.long) for k,v in dict(SELF=expected,SAME_RX=same,CROSS_RX=cross).items()},len(cohort)
def virtual_features(z1,z2,donor):
 if z1.ndim!=2 or z1.shape!=z2.shape or z1.shape[1]!=160 or donor.shape!=(len(z1),) or donor.dtype!=torch.long or sorted(donor.detach().cpu().tolist())!=list(range(len(z1))):raise ValueError('Matched joint160 and a full donor permutation required')
 if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():raise ValueError('Finite native features required')
 center=(z1+z2)*.5;delta=(z1-z2)*.5;noise=delta[donor]
 return center+noise,center-noise,center,delta
def logits(z,W,scale=30.):
 if W.shape!=(6,160) or z.shape[1]!=160 or scale!=30.:raise ValueError('Original six-class cosine head required')
 return scale*F.linear(F.normalize(z,dim=1,eps=1e-12),F.normalize(W,dim=1,eps=1e-12))
def loss(z1,z2,W,y,donor,classify=None):
 plus,minus,center,delta=virtual_features(z1,z2,donor)
 classify=(lambda z:logits(z,W)) if classify is None else classify
 ce=.5*(F.cross_entropy(classify(plus),y)+F.cross_entropy(classify(minus),y))
 original=(z1.detach().double().square().sum()+z2.detach().double().square().sum())
 changed=(plus.detach().double().square().sum()+minus.detach().double().square().sum())
 error=float((changed-original).abs()/original.clamp_min(1e-12))
 if error>2e-6:raise ValueError('Permutation must preserve pooled two-view feature energy')
 return ce,dict(energy_relative_error=error,mean_center_L2=center.detach().norm(dim=1).mean(),mean_residual_L2=delta.detach().norm(dim=1).mean())
