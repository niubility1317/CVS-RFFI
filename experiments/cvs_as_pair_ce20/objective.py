"""Mean of two supervised CEs;diagnostic KL is outside autograd."""
import torch
from torch.nn import functional as F

def pair(first,second,y):
 if first.shape!=second.shape or first.shape!=(len(y),6) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()):raise ValueError('VisibleL labels/default6logits required')
 if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):raise ValueError('Finite paireddefaultlogits required')
 ce=.5*(F.cross_entropy(first,y,label_smoothing=.01)+F.cross_entropy(second,y,label_smoothing=.01))
 with torch.no_grad():
  p,q=first.detach().log_softmax(1),second.detach().log_softmax(1);diagnostic=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
 return ce,ce,diagnostic
