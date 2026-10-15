import math,torch
from torch.nn import functional as F
ARMS=('FULL_KL','FEATURE_KL','SHRINK','DIFFERENCE')
def new_audit():return dict(calls=0,raw_energy=0.,common_energy=0.,output_energy=0.,max_norm_relative_error=0.,max_difference_common_abs=0.)
def project(g1,g2,arm):
 if arm not in ARMS or g1.shape!=g2.shape or not torch.isfinite(g1).all() or not torch.isfinite(g2).all():raise ValueError('Finite paired KL gradients and fixed arm required')
 delta=(g1-g2)*.5
 raw=g1.double().square().sum()+g2.double().square().sum();target=2*delta.double().square().sum()
 if arm=='DIFFERENCE':return delta,-delta
 if arm=='SHRINK':
  scale=(target/raw.clamp_min(1e-300)).sqrt().to(g1.dtype)
  return g1*scale,g2*scale
 return g1,g2
class RoutePair(torch.autograd.Function):
 @staticmethod
 def forward(ctx,z1,z2,arm,audit):
  if arm not in ARMS or z1.shape!=z2.shape or z1.ndim!=2 or z1.shape[1]!=160:raise ValueError('Same paired joint160 features required')
  ctx.arm=arm;ctx.audit=audit
  return z1.view_as(z1),z2.view_as(z2)
 @staticmethod
 def backward(ctx,g1,g2):
  a,b=project(g1,g2,ctx.arm);common=(g1+g2)*.5;delta=(g1-g2)*.5
  raw=float(g1.double().square().sum()+g2.double().square().sum());out=float(a.double().square().sum()+b.double().square().sum());diff=float(2*delta.double().square().sum())
  target=diff if ctx.arm in ('SHRINK','DIFFERENCE') else raw
  audit=ctx.audit;audit['calls']+=1;audit['raw_energy']+=raw;audit['common_energy']+=float(2*common.double().square().sum());audit['output_energy']+=out
  audit['max_norm_relative_error']=max(audit['max_norm_relative_error'],abs(out-target)/max(target,1e-300))
  if ctx.arm=='DIFFERENCE':audit['max_difference_common_abs']=max(audit['max_difference_common_abs'],float((a+b).abs().max()))
  return a,b,None,None
def logits(z,W):
 if z.ndim!=2 or z.shape[1]!=160 or W.shape!=(6,160):raise ValueError('Original joint160/default six-class head required')
 return 30.*F.linear(F.normalize(z,dim=1,eps=1e-4),F.normalize(W,dim=1,eps=1e-4))
def consistency(z1,z2,W,arm,audit):
 a,b=RoutePair.apply(z1,z2,arm,audit);weight=W if arm=='FULL_KL' else W.detach()
 p,q=logits(a,weight).log_softmax(1),logits(b,weight).log_softmax(1)
 return .5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
