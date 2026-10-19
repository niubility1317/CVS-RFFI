"""Detached channel choice only; no new fitted classifier or trainable mask."""
import math
import torch
WEAK=(1,3)
def eligible(y):
 if y.dtype!=torch.long or y.ndim!=1 or not bool(((y>=0)&(y<6)).all()):raise ValueError('Current labeled source TX required')
 return (y==1)|(y==3)
def channel_mask(activation,gradient,mode,seed):
 if activation.ndim!=3 or activation.shape[1] not in (48,96) or not bool(torch.isfinite(activation).all()):raise ValueError('Actual shallow/middle source maps required')
 b,c,_=activation.shape;k=math.ceil(c*.125)
 if mode=='SALIENCY_WEAK':
  if gradient is None or gradient.shape!=activation.shape or not bool(torch.isfinite(gradient).all()):raise ValueError('Finite actual L true-logit gradient required')
  score=(activation.detach()*gradient.detach()).sum(-1)
  order=torch.argsort(score,dim=1,descending=True,stable=True)
 elif mode=='RANDOM_WEAK':
  generator=torch.Generator(device='cpu');generator.manual_seed(int(seed))
  order=torch.rand(b,c,generator=generator).argsort(1).to(activation.device)
 else:raise ValueError('Registered random/saliency method only')
 mask=torch.ones((b,c,1),device=activation.device,dtype=activation.dtype)
 mask.scatter_(1,order[:,:k,None],0.)
 if mask.requires_grad or not bool((mask.sum(1)==c-k).all()):raise ValueError('Detached exact channel budget required')
 return mask
