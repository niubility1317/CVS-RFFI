"""One normalized true-logit Jacobian row, not probability or feature matching."""
import torch
from torch.nn import functional as F
def row(logits,x,y,*,create_graph=False,retain_graph=False):
 if logits.shape!=(len(x),6) or x.ndim!=3 or x.shape[1:]!=(2,256) or y.shape!=(len(x),) or y.dtype!=torch.long or not x.requires_grad or not bool(((y>=0)&(y<6)).all()) or not torch.isfinite(logits).all():raise ValueError('Finite legal six-class IQ sensitivity inputs required')
 return torch.autograd.grad(logits.gather(1,y[:,None]).sum(),x,create_graph=create_graph,retain_graph=retain_graph)[0]
def direction(student,teacher,accepted):
 if student.shape!=teacher.shape or student.ndim!=3 or student.shape[1:]!=(2,256) or accepted.shape!=(len(student),) or accepted.dtype!=torch.bool or not torch.isfinite(student).all() or not torch.isfinite(teacher).all():raise ValueError('Finite aligned IQ derivatives and detached teacher mask required')
 s=F.normalize(student.flatten(1),dim=1,eps=1e-12);t=F.normalize(teacher.detach().flatten(1),dim=1,eps=1e-12)
 cosine=(s*t).sum(1);loss=(accepted.to(cosine.dtype)*(1.-cosine)).mean()
 return loss,cosine
def geometry(named,g0,g1):
 # Same physical-Hz Sinc metric as the unified global clip; no optimizer step.
 dot=n0=n1=0.
 for (name,p),a,b in zip(named,g0,g1):
  scale=25000. if name.endswith(('.sinc.low_coordinate','.sinc.band_coordinate')) else 1.
  if a is not None:
   if not torch.isfinite(a).all():raise FloatingPointError('Nonfinite first parameter derivative')
   n0+=float((a.detach().double()/scale).square().sum())
  if b is not None:
   if not torch.isfinite(b).all():raise FloatingPointError('Nonfinite second parameter derivative')
   n1+=float((b.detach().double()/scale).square().sum())
  if a is not None and b is not None:dot+=float((a.detach().double()*b.detach().double()).sum())/(scale*scale)
 return dict(dot=dot,norm0=n0**.5,norm1=n1**.5,cosine=dot/(n0*n1)**.5 if n0>0 and n1>0 else None)
