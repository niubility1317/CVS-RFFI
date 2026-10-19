"""Actual feature displacement: normalized block metric and three source-CE risks."""
from itertools import combinations
import torch
from torch.nn import functional as F
from experiments.cvs_f_weaktx_gradient50.core import select,blocks,PREFIX
MODES=('STEP_NORM','UPDATE_CONE')
def active(step):return step>1110 and step%4==0
def strength(step):return min(max((((step-1)//222+1)-5)/10.,0.),1.)
def contributions(first,second,y):
 if first.shape!=(128,6) or second.shape!=first.shape or y.shape!=(128,) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):raise ValueError('Only visible paired L128 source TX logits')
 ce=.5*(F.cross_entropy(first,y,reduction='none',label_smoothing=.01)+F.cross_entropy(second,y,reduction='none',label_smoothing=.01));masks=(y==1,y==3,(y!=1)&(y!=3))
 return [ce[mask].sum()/128 for mask in masks],[int(mask.sum()) for mask in masks]
def project(x,h):
 if x.ndim!=1 or h.shape!=(3,len(x)) or x.device.type!='cpu' or h.device.type!='cpu' or not all(bool(torch.isfinite(v).all()) for v in (x,h)):raise ValueError('Finite CPU vectors for three fixed class risks')
 x=x.double();h=h.double();norms=h.norm(dim=1);hh=h/norms.clamp_min(1e-300)[:,None];gram=hh@hh.T;b=hh@x;tol=1e-10*max(float(x.norm()),1.);valid=[i for i in range(3) if norms[i]>0];best=x.clone() if bool((b<=tol).all()) else None;distance=0. if best is not None else float('inf');chosen=[]
 for size in range(1,len(valid)+1):
  for subset in combinations(valid,size):
   indices=torch.tensor(subset,dtype=torch.long);small=gram[indices][:,indices];rhs=b[indices];lam=torch.linalg.lstsq(small,rhs,driver='gelsd',rcond=1e-12).solution
   if not bool(torch.isfinite(lam).all()) or bool((lam< -tol).any()):continue
   full=torch.zeros(3,dtype=torch.float64);full[indices]=lam.clamp_min(0);residual=b-gram@full
   if bool((residual>tol).any()):continue
   change=hh.T@full;cost=float(change@change)
   if cost<distance:best=x-change;distance=cost;chosen=list(subset)
 fallback=best is None
 if fallback:best=torch.zeros_like(x)
 return best,dict(solver_fallback=fallback,active_constraints=chosen,raw_normalized_dots=b.tolist(),projected_normalized_dots=(hh@best).tolist(),zero_constraint_rows=int((norms==0).sum()))
def route(before,after,gradients,block_ranges,eta):
 if before.ndim!=1 or after.shape!=before.shape or before.dtype not in (torch.float32,torch.float64) or after.dtype!=before.dtype or gradients.shape!=(3,len(before)) or not 0<=eta<=1 or any(v.device.type!='cpu' or not bool(torch.isfinite(v).all()) for v in (before,after,gradients)):raise ValueError('Finite same-state feature step and actual three source class VJPs required')
 old=before.double();raw=after.double()-old;scales=torch.zeros_like(raw);end=0
 for lo,hi in block_ranges:
  if lo!=end or not lo<hi<=len(raw):raise ValueError('Contiguous full deployed feature blocks required')
  scales[lo:hi]=raw[lo:hi].square().mean().sqrt();end=hi
 if end!=len(raw):raise ValueError('Incomplete deployed step metric')
 denom=scales.clamp_min(1e-300);x=raw/denom;h=gradients.double()*scales;hn=h/h.norm(dim=1).clamp_min(1e-300)[:,None];xp,info=project(x,h);n0=float(x.norm());candidate=(old+scales*(x+eta*(xp-x))).to(before.dtype)
 def coordinates(value):return (value.double()-old)/denom
 def stepnorm(value):return float(coordinates(value).norm())
 storage_projection_fallback=False;relative_residual=float((hn@coordinates(candidate)).max())/max(n0,1e-30)
 if eta==1. and relative_residual>1e-3:
  candidate=before.clone();storage_projection_fallback=True
 target=stepnorm(candidate)
 def rounded(gamma):return (old+raw*gamma).to(before.dtype)
 control=before.clone() if n0==0 or target==0 else rounded(target/n0);best_error=abs(stepnorm(control)-target)/max(target,1e-30);gamma=0. if n0==0 else target/n0
 if best_error>1e-3:
  low=0.;high=max(1.,gamma*1.01)
  for _ in range(4):
   if stepnorm(rounded(high))>=target:break
   high*=2
  for _ in range(40):
   mid=(low+high)/2;value=rounded(mid);norm=stepnorm(value);error=abs(norm-target)/max(target,1e-30)
   if error<best_error:control=value;best_error=error;gamma=mid
   if norm<target:low=mid
   else:high=mid
 match_fallback=best_error>1e-3
 if match_fallback:candidate=after.clone();control=after.clone();target=n0;best_error=0.;gamma=1.
 actual=coordinates(candidate);control_x=coordinates(control);risk=(hn@actual).tolist();control_risk=(hn@control_x).tolist();actual_norm=float(actual.norm());control_cos=float(control_x@x/(control_x.norm()*x.norm())) if control_x.norm()>0 and x.norm()>0 else None;cos=float(actual@x/(actual.norm()*x.norm())) if actual.norm()>0 and x.norm()>0 else None
 info.update(strength=eta,block_metric='same-state native step RMS per feature block; zero-step blocks stay zero',raw_step_normalized_norm=n0,projected_stored_normalized_norm=actual_norm,control_stored_normalized_norm=stepnorm(control),stored_norm_match_error=best_error,control_gamma=gamma,projection_storage_fallback=storage_projection_fallback,storage_match_fallback=match_fallback,stored_normalized_dots=risk,control_stored_normalized_dots=control_risk,control_normalized_correction_fraction=float((control_x-x).norm())/max(n0,1e-30),control_raw_cosine=control_cos,full_strength_max_residual_relative=max(risk)/max(n0,1e-30),constraint_enforced=eta==1. and not match_fallback,normalized_correction_fraction=float((actual-x).norm())/max(n0,1e-30),retained_normalized_norm_ratio=actual_norm/max(n0,1e-30) if n0 else 1.,raw_projected_cosine=cos,actual_correction_nonzero=bool((candidate!=after).any()))
 if not all(bool(torch.isfinite(v).all()) for v in (candidate,control)) or (info['constraint_enforced'] and info['full_strength_max_residual_relative']>1e-3):raise FloatingPointError('Stored feature displacement risk/norm correction nonfinite or invalid')
 return candidate,control,info
