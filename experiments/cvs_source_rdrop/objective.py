"""Same-input stochastic dropout consistency; existing six-class default head."""
import torch
from torch.nn import functional as F
def symmetric_kl(a,b):
 if a.ndim!=2 or a.shape!=b.shape or a.shape[1]!=6 or not len(a) or not a.is_floating_point() or a.dtype!=b.dtype or a.device!=b.device:
  raise ValueError('Matching nonempty six-class floating logits required')
 if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Nonfinite dropout logits')
 la,lb=a.log_softmax(1),b.log_softmax(1)
 # Both stochastic predictions remain differentiable; no teacher or target fit.
 return .5*((la.exp()*(la-lb)).sum(1).mean()+(lb.exp()*(lb-la)).sum(1).mean())
