"""Same smoothed CE/symmetricKL,perpacket weighting before the original128 mean."""
import time

def weighted_terms(first,second,y,weights):
 import torch
 from torch.nn import functional as F
 start=time.perf_counter()
 if first.shape!=second.shape or first.shape!=(128,6) or y.shape!=(128,) or y.dtype!=torch.long or weights.shape!=(128,) or not bool(((y>=0)&(y<6)).all()) or not torch.isfinite(first).all() or not torch.isfinite(second).all() or not torch.isfinite(weights).all() or not bool((weights>0).all()):raise ValueError('Finite sameIQ128 sixclass weighted source labels required')
 ce_a=F.cross_entropy(first,y,label_smoothing=.01,reduction='none');ce_b=F.cross_entropy(second,y,label_smoothing=.01,reduction='none');ce=.5*(ce_a+ce_b)
 p,q=first.log_softmax(1),second.log_softmax(1);kl=.5*((p.exp()-q.exp())*(p-q)).sum(1)
 result=((ce+kl)*weights).mean();weighted_ce=(ce*weights).mean();weighted_kl=(kl*weights).mean()
 return result,weighted_ce,weighted_kl,dict(CE_a=ce_a,CE_b=ce_b,CE=ce,KL=kl),time.perf_counter()-start
