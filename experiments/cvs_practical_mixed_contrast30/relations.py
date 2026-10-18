"""Existing SupCon formula; only actual IQ view membership differs."""
import torch
from torch.nn import functional as F
LEVELS=('time_down','t2.pool');MODES=('ORD128','MIXED128')
def select_maps(ordinary,satellite,mask,mode):
 if mode not in MODES or mask.dtype!=torch.bool or mask.shape!=(128,) or int(mask.sum())!=64 or tuple(ordinary)!=LEVELS or tuple(satellite)!=LEVELS:raise ValueError('Exactly128 original physical inputs and selected64 required')
 result={}
 for level in LEVELS:
  a,b=ordinary[level],satellite[level]
  if a.ndim!=3 or a.shape!=b.shape or len(a)!=128:raise ValueError('Reuse actual matching existing maps only')
  result[level]=a if mode=='ORD128' else b
 return result

def metadata(y,rx,mask,mode):
 if mode not in MODES or y.dtype!=torch.long or rx.dtype!=torch.long or y.shape!=(128,) or rx.shape!=y.shape or not bool(((y>=0)&(y<6)).all()) or not all(int(r)in(1,3,4,6,8) for r in rx.tolist()) or mask.dtype!=torch.bool or mask.shape!=(128,) or int(mask.sum())!=64:raise ValueError('Only true labeled original full-source L128')
 return y,rx

def penalty(ordinary,satellite,y,rx,mask,mode,*,actual_As=False):
 maps=select_maps(ordinary,satellite,mask,mode);yy,rr=metadata(y,rx,mask,mode);n=len(yy);same=yy[:,None]==yy[None,:];positive=same&(rr[:,None]!=rr[None,:]);negative=~same;den=positive|negative;valid=positive.any(1)&negative.any(1);count=positive.sum(1);values=[];details=[]
 for level in LEVELS:
  h=maps[level]
  if not bool(torch.isfinite(h).all()):raise ValueError('No nonfinite map/sample may be dropped')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4);z=F.normalize(z,dim=1,eps=1e-4)
  if bool(valid.any()):
   logits=(z@z.T/.2)[valid];c=count[valid];value=(torch.logsumexp(logits.masked_fill(~den[valid],-torch.inf),1)-(logits*positive[valid]).sum(1)/c-c.to(z.dtype).log()).mean()
  else:value=h.sum()*0
  if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite original contrastive objective')
  values.append(value);details.append(dict(level=level,mode=mode,shape=list(h.shape),loss=float(value.detach()),valid_anchors=int(valid.sum()),total_anchors=n,ordinary_anchors=128 if mode=='ORD128' else 64,selected_anchors=0 if mode=='ORD128' else 64,additional_anchors=0,anchors_per_physical_input=1,positive_min=int(count[valid].min()) if valid.any() else 0,positive_max=int(count[valid].max()) if valid.any() else 0,cross_view_positive_pairs=int(positive[:128,128:].sum()) if n>128 else 0,same_RX_positive_pairs=int((positive&(rr[:,None]==rr[None,:])).sum()),true_satellite_anchors=64 if mode=='MIXED128' and actual_As else 0,selected_actual_view='As_raw' if actual_As else 'raw',raw_pair_maps_used=0))
 return torch.stack(values).mean(),details
