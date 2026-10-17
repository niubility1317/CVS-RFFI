"""Port of 4df7b94d0f: L-only primary-pair shallow/middle SupCon; no satellite feature constraint."""
from contextlib import contextmanager
from collections.abc import Mapping
from types import MethodType
import torch
from torch.nn import functional as F
LEVELS=('time_down','t2.pool')
PREFIX='id_backbone.encoder.core.id_backbone.'
RX_BY_DOMAIN=(1,1,1,3,3,3,4,4,4,6,6,6,8,8,8)
def schedule(epoch):return .1*min(max((int(epoch)-5)/10.,0.),1.)
def masks(y,rx,mode):
 if mode not in ('ALL_TX','CROSS_RX') or y.dtype!=torch.long or rx.dtype!=torch.long or y.ndim!=1 or rx.shape!=y.shape or not bool(((y>=0)&(y<6)).all()) or not all(int(r) in (1,3,4,6,8) for r in rx.detach().cpu().tolist()):raise ValueError('Current source L TX/RX only')
 same=y[:,None]==y[None,:];other=~torch.eye(len(y),dtype=torch.bool,device=y.device)
 positive=same&other if mode=='ALL_TX' else same&(rx[:,None]!=rx[None,:]);negative=(~same)&other;denominator=other if mode=='ALL_TX' else positive|negative;valid=positive.any(1)&negative.any(1)
 return positive,denominator,valid
def loss(z,y,rx,mode):
 if z.ndim!=2 or len(z)!=len(y) or not bool(torch.isfinite(z).all()):raise ValueError('Finite current L descriptors required')
 positive,denominator,valid=masks(y,rx,mode);n=positive.sum(1)
 if not bool(valid.any()):return z.sum()*0.,dict(valid_anchors=0,total_anchors=len(y),positive_min=0,positive_max=0)
 z=F.normalize(z,dim=1,eps=1e-4);logits=(z@z.T/.2)[valid];pos=positive[valid];den=denominator[valid];count=n[valid];value=(torch.logsumexp(logits.masked_fill(~den,-torch.inf),1)-(logits*pos).sum(1)/count-count.to(z.dtype).log()).mean()
 if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite source contrastive loss')
 return value,dict(valid_anchors=int(valid.sum()),total_anchors=len(y),positive_min=int(count.min()),positive_max=int(count.max()))
def penalty(maps,y,rx,mode):
 if tuple(maps)!=LEVELS:raise ValueError('Both actual shallow/middle F maps required')
 values=[];detail=[]
 for level in LEVELS:
  h=maps[level]
  if h.ndim!=3 or len(h)!=len(y) or not bool(torch.isfinite(h).all()):raise ValueError('Finite current L map required')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4);value,stats=loss(z,y,rx,mode);values.append(value);detail.append(dict(level=level,shape=list(h.shape),loss=float(value.detach()),**stats))
 return torch.stack(values).mean(),detail
