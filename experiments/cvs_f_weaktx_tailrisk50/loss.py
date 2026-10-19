"""Pair-averaged per-physical-anchor feature risk, balanced by weak TX and RX."""
import torch
from torch.nn import functional as F
LEVELS=('time_down','t2.pool')
RXS=(1,3,4,6,8)
WEAK=(1,3)
MODES=('MEAN_WEAK_RX','TAIL_WEAK_RX')

def labels(y,rx):
 if y.dtype!=torch.long or rx.dtype!=torch.long or y.ndim!=1 or rx.shape!=y.shape or not len(y) or not bool(((y>=0)&(y<6)).all()) or not all(int(r) in RXS for r in rx.detach().cpu().tolist()):raise ValueError('Visible source L TX and physical RX required')

def masks(y,rx):
 labels(y,rx);same=y[:,None]==y[None,:];same_rx=rx[:,None]==rx[None,:]
 pos=same&~same_rx;negative=~same&same_rx
 return pos,pos|negative,pos.any(1)&negative.any(1)

def anchor_loss(z,y,rx):
 if z.ndim!=2 or len(z)!=len(y) or not bool(torch.isfinite(z).all()):raise ValueError('Finite current feature descriptors required')
 pos,den,valid=masks(y,rx);out=z.sum(1)*0
 if bool(valid.any()):
  u=F.normalize(z,dim=1,eps=1e-4);logits=(u@u.T/.2)[valid];positive=pos[valid];n=positive.sum(1).to(z.dtype)
  each=torch.logsumexp(logits.masked_fill(~den[valid],-torch.inf),1)-(logits*positive).sum(1)/n-n.log()
  out=out.index_copy(0,torch.where(valid)[0],each)
 if not bool(torch.isfinite(out).all()):raise FloatingPointError('Nonfinite RX-conditional per-anchor loss')
 return out,valid

def risk(per_anchor,y,rx,valid,mode):
 labels(y,rx)
 if mode not in MODES or per_anchor.shape!=y.shape or valid.shape!=y.shape or valid.dtype!=torch.bool or not bool(torch.isfinite(per_anchor).all()):raise ValueError('Registered finite weak TX/RX risk required')
 classes=[];cells=[];available=selected=0
 for c in WEAK:
  groups=[]
  for r in RXS:
   index=torch.where(valid&(y==c)&(rx==r))[0];n=len(index)
   if not n:continue
   values=per_anchor[index];k=n if mode=='MEAN_WEAK_RX' else (n+1)//2
   ix=torch.topk(values.detach(),k,sorted=False).indices if k<n else torch.arange(n,device=index.device)
   value=values[ix].mean();groups.append(value);available+=n;selected+=k
   cells.append(dict(TX=c,RX=r,available=n,selected=k,mean_loss=float(values.detach().mean()),selected_loss=float(value.detach())))
  if not groups:return per_anchor.sum()*0,dict(eligible=False,valid_cells=0,available_anchors=0,selected_anchors=0,cells=[])
  classes.append(torch.stack(groups).mean())
 value=torch.stack(classes).mean()
 if not torch.isfinite(value):raise FloatingPointError('Nonfinite balanced weak feature risk')
 return value,dict(eligible=True,valid_cells=len(cells),available_anchors=available,selected_anchors=selected,selected_fraction=selected/available,cells=cells)

def penalty(first,second,y,rx,mode):
 if tuple(first)!=LEVELS or tuple(second)!=LEVELS or len(y)!=128:raise ValueError('Both unchanged native L128 RDrop feature maps required')
 values=[];details=[]
 for level,width in zip(LEVELS,(48,96)):
  scores=[]
  for maps in (first,second):
   h=maps[level]
   if h.ndim!=3 or h.shape[:2]!=(128,width) or not bool(torch.isfinite(h).all()):raise ValueError('Actual deployed shallow/middle maps required')
   v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4)
   each,valid=anchor_loss(z,y,rx);scores.append(each)
  # One risk per physical L packet: stochastic view losses are averaged first.
  value,info=risk((scores[0]+scores[1])*.5,y,rx,valid,mode);values.append(value);details.append(dict(level=level,loss=float(value.detach()),**info))
 return torch.stack(values).mean(),dict(eligible=all(v['eligible'] for v in details),valid_cells=sum(v['valid_cells'] for v in details),available_anchor_objectives=sum(v['available_anchors'] for v in details),selected_anchor_objectives=sum(v['selected_anchors'] for v in details),levels=details)
