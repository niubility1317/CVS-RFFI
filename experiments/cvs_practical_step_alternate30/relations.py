"""Identical original ORD128 formula, using its actual ordinary maps only."""
import torch
from torch.nn import functional as F
LEVELS=('time_down','t2.pool')
def validate(maps,y,rx):
 if tuple(maps)!=LEVELS or y.dtype!=torch.long or rx.dtype!=torch.long or y.shape!=(128,) or rx.shape!=y.shape or not bool(((y>=0)&(y<6)).all()) or not all(int(r) in (1,3,4,6,8) for r in rx.tolist()):raise ValueError('Only true original ORD128 full-source maps/metadata')
 for h in maps.values():
  if h.ndim!=3 or len(h)!=128:raise ValueError('Actual ordinary L128 shallow/middle maps required')

def penalty(ordinary,y,rx):
 validate(ordinary,y,rx);maps=ordinary;yy,rr=y,rx;mode='ORD128';n=len(yy);same=yy[:,None]==yy[None,:];positive=same&(rr[:,None]!=rr[None,:]);negative=~same;den=positive|negative;valid=positive.any(1)&negative.any(1);count=positive.sum(1);values=[];details=[]
 for level in LEVELS:
  h=maps[level]
  if not bool(torch.isfinite(h).all()):raise ValueError('No nonfinite map/sample may be dropped')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4);z=F.normalize(z,dim=1,eps=1e-4)
  if bool(valid.any()):
   logits=(z@z.T/.2)[valid];c=count[valid];value=(torch.logsumexp(logits.masked_fill(~den[valid],-torch.inf),1)-(logits*positive[valid]).sum(1)/c-c.to(z.dtype).log()).mean()
  else:value=h.sum()*0
  if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite original contrastive objective')
  values.append(value);details.append(dict(level=level,mode=mode,shape=list(h.shape),loss=float(value.detach()),valid_anchors=int(valid.sum()),total_anchors=n,ordinary_anchors=128,additional_anchors=n-128,positive_min=int(count[valid].min()) if valid.any() else 0,positive_max=int(count[valid].max()) if valid.any() else 0,cross_view_positive_pairs=int(positive[:128,128:].sum()) if n>128 else 0,same_RX_positive_pairs=int((positive&(rr[:,None]==rr[None,:])).sum()),true_satellite_anchors=64 if mode=='SAT192' else 0))
 return torch.stack(values).mean(),details
