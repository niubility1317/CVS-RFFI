"""Bounded detached difficulty, inside an externally unchanged acceptance mask."""
import torch
from torch.nn import functional as F
def weights(logits,pseudo,accepted):
 if logits.ndim!=2 or len(logits)<1 or logits.shape[1]!=6 or pseudo.shape!=(len(logits),) or accepted.shape!=pseudo.shape or pseudo.dtype!=torch.long or accepted.dtype!=torch.bool or not torch.isfinite(logits).all() or pseudo.min()<0 or pseudo.max()>=6:raise ValueError('Finite default6class/matched originalmask required')
 h=(1-logits.detach().softmax(1).gather(1,pseudo[:,None]).squeeze(1)).clamp(0,1);w=.75+.5*(h/.1).clamp(0,1)
 if w.requires_grad or h.requires_grad:raise ValueError('Weight gradient not stopped')
 return h,w
def losses(logits,pseudo,accepted):
 h,w=weights(logits,pseudo,accepted);ce=F.cross_entropy(logits,pseudo,reduction='none')
 if not accepted.any():uniform=logits.sum()*0;weighted=uniform;effective=torch.zeros_like(w)
 else:
  selected=w[accepted];uniform=ce[accepted].mean();weighted=(selected*ce[accepted]).sum()/selected.sum();effective=torch.where(accepted,w/selected.mean(),torch.zeros_like(w))
 return dict(hardness=h,raw_weight=w,effective_weight=effective,CE=ce,uniform=uniform,weighted=weighted,accept=accepted.clone())
