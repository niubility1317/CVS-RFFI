"""Exact native forward; full batch-statistic derivative and size-matched control."""
import torch
from torch.nn import functional as F
from experiments.cvs_f_supcon50.mechanism import LEVELS,loss as native_loss
MODES=('STAT_GRAD','NORM_ONLY')
EPS=1e-4
class _Standardize(torch.autograd.Function):
 @staticmethod
 def forward(ctx,v,mode,record):
  if mode not in MODES or v.ndim!=2:raise ValueError('Fixed descriptor standardization policies only')
  mean=v.mean(0);raw=v.std(0,unbiased=False);sigma=raw.clamp_min(EPS);z=(v-mean)/sigma
  ctx.save_for_backward(z,sigma,raw>=EPS);ctx.mode=mode;ctx.record=record
  record.update(backward_calls=0,channels=len(sigma),clamped_channels=int((raw<EPS).sum()))
  return z
 @staticmethod
 def backward(ctx,g):
  z,sigma,active=ctx.saved_tensors
  native=g/sigma;translation=native.mean(0,keepdim=True);scale=z*(native*z).mean(0,keepdim=True)*active
  full=native-translation-scale;oldnorm=native.norm();fullnorm=full.norm();ratio=fullnorm/oldnorm.clamp_min(1e-30)
  delivered=full if ctx.mode=='STAT_GRAD' else native*ratio
  if not bool(torch.isfinite(delivered).all()):raise FloatingPointError('Nonfinite statistic derivative')
  ctx.record.update(backward_calls=ctx.record['backward_calls']+1,native_descriptor_grad_norm=float(oldnorm),full_descriptor_grad_norm=float(fullnorm),delivered_descriptor_grad_norm=float(delivered.norm()),norm_ratio=float(ratio),translation_removed_norm=float(translation.expand_as(native).norm()),scale_removed_norm=float(scale.norm()),mode=ctx.mode)
  return delivered,None,None
def standardize(v,mode,record):return _Standardize.apply(v,mode,record)
def penalty(maps,y,rx,mode):
 if mode not in MODES or tuple(maps)!=LEVELS:raise ValueError('Fixed policy and original shallow/middle maps required')
 values=[];details=[]
 for level in LEVELS:
  h=maps[level]
  if h.ndim!=3 or len(h)!=len(y) or not bool(torch.isfinite(h).all()):raise ValueError('Finite current L feature maps required')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);backward={};z=standardize(v,mode,backward);value,stats=native_loss(z,y,rx,'CROSS_RX');values.append(value);details.append(dict(level=level,shape=list(h.shape),loss=float(value.detach()),statistic_policy=mode,backward=backward,**stats))
 return torch.stack(values).mean(),details
