import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_report_alignment_clean.alignment import capture

def masks(y,rx,arm):
 same=y[:,None]==y[None,:];nonself=~torch.eye(len(y),dtype=torch.bool,device=y.device)
 if arm=='ALL_TX':positive=same&nonself;denominator=nonself
 elif arm=='CROSS_RX_TX':positive=same&(rx[:,None]!=rx[None,:]);denominator=(~same)|positive
 else:raise ValueError('Unregistered positive rule')
 if not positive.any(1).all() or not ((~same)&denominator).any(1).all():raise ValueError('Every labeled anchor needs positive and otherTX negative')
 return positive,denominator

def loss(z,y,rx,arm,temperature=.2):
 if not torch.isfinite(z).all() or z.ndim!=2 or len(z)!=len(y) or temperature<=0:raise ValueError('Finite contrastive descriptors/temperature required')
 positive,denominator=masks(y,rx,arm);z=F.normalize(z,dim=1,eps=1e-4);logits=z@z.T/temperature
 logden=torch.logsumexp(logits.masked_fill(~denominator,-torch.inf),1)
 n=positive.sum(1);term=logden-(logits*positive).sum(1)/n-n.float().log()
 return term.mean(),n

def penalty(maps,y,rx,c):
 d.validate(c)
 if y.ndim!=1 or rx.shape!=y.shape or y.dtype!=torch.long or rx.dtype!=torch.long or not set(y.tolist())<=set(range(6)) or not set(rx.tolist())<=set(d.RX):raise ValueError('Only legal currentL TX/RX metadata')
 if tuple(maps)!=d.LEVELS:raise ValueError('Actual shallow/middle layers differ')
 values=[];details=[]
 for level,h in maps.items():
  if h.ndim!=3 or len(h)!=len(y) or not torch.isfinite(h).all():raise ValueError('Invalid actual map')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4)
  value,n=loss(z,y,rx,c['arm'],c['contrastive']['temperature']);values.append(value)
  details.append(dict(level=level,shape=list(h.shape),anchors=len(y),positive_min=int(n.min()),positive_max=int(n.max()),loss=float(value.detach())))
 return torch.stack(values).mean(),details
