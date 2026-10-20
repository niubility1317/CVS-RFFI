"""Multi-positive decoupling at original source shallow/middle descriptors."""
import torch
from torch.nn import functional as F
from experiments.cvs_f_supcon50.mechanism import masks,LEVELS
MODES=('Q_DCL','DCL')
def anchors(z,y,rx):
 if z.ndim!=2 or len(z)!=len(y) or not bool(torch.isfinite(z).all()):raise ValueError('Finite current source L descriptor matrix required')
 positive,denominator,valid=masks(y,rx,'CROSS_RX');negative=denominator&~positive;zero=z.sum(1)*0.;out={key:zero.clone() for key in ('NATIVE','Q_DCL','DCL','POS_COMP')};q=torch.zeros_like(zero).detach()
 if bool(valid.any()):
  unit=F.normalize(z,dim=1,eps=1e-4);s=(unit@unit.T/.2)[valid];p=positive[valid];n=negative[valid];count=p.sum(1).to(z.dtype);mean=(s*p).sum(1)/count;lp=torch.logsumexp(s.masked_fill(~p,-torch.inf),1);ln=torch.logsumexp(s.masked_fill(~n,-torch.inf),1);la=torch.logaddexp(lp,ln);qq=(ln-la).exp().detach();decoupled=ln-mean-count.log();competition=lp-mean-count.log();ix=torch.where(valid)[0]
  for key,value in dict(NATIVE=la-mean-count.log(),Q_DCL=qq*decoupled,DCL=decoupled,POS_COMP=competition).items():out[key]=out[key].index_copy(0,ix,value)
  q=q.index_copy(0,ix,qq)
 if not all(bool(torch.isfinite(v).all()) for v in out.values()):raise FloatingPointError('Nonfinite decoupled contrastive objective')
 return out,dict(valid=valid,q=q,positive_count=positive.sum(1))
def penalty(maps,y,rx,mode):
 if mode not in MODES or tuple(maps)!=LEVELS:raise ValueError('Fixed Q_DCL/DCL actual shallow and middle maps only')
 values=[];details=[]
 for level in LEVELS:
  h=maps[level]
  if h.ndim!=3 or len(h)!=len(y) or not bool(torch.isfinite(h).all()):raise ValueError('Finite current L feature maps required')
  v=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1);z=(v-v.detach().mean(0))/v.detach().std(0,unbiased=False).clamp_min(1e-4);each,info=anchors(z,y,rx);valid=info['valid'];q=info['q'];n=info['positive_count'][valid];value=each[mode][valid].mean() if bool(valid.any()) else z.sum()*0.;values.append(value)
  per=[dict(TX=c,count=int(((y==c)&valid).sum()),q_mean=float(q[(y==c)&valid].mean()) if bool(((y==c)&valid).any()) else None) for c in range(6)]
  details.append(dict(level=level,shape=list(h.shape),loss=float(value.detach()),valid_anchors=int(valid.sum()),total_anchors=len(y),positive_min=int(n.min()) if len(n) else 0,positive_max=int(n.max()) if len(n) else 0,q_mean=float(q[valid].mean()) if bool(valid.any()) else None,q_min=float(q[valid].min()) if bool(valid.any()) else None,q_max=float(q[valid].max()) if bool(valid.any()) else None,per_TX_q=per,native_loss=float(each['NATIVE'][valid].mean().detach()) if bool(valid.any()) else 0.,positive_competition_loss=float(each['POS_COMP'][valid].mean().detach()) if bool(valid.any()) else 0.,mode=mode))
 return torch.stack(values).mean(),details
