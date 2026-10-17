"""One optional future U objective; no head/module and no urban-confidence gate."""
import torch
from torch.nn import functional as F

def transfer_loss(student,teacher,hidden_y,tau=.95):
 if hidden_y.dtype!=torch.long or hidden_y.ndim!=1 or not bool((hidden_y==-1).all()):raise ValueError('U TX labels must be hidden -1 sentinels')
 n=len(hidden_y)
 if n==0 or student.shape!=(n,6) or teacher.shape!=student.shape or not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()) or not 0<tau<1:raise ValueError('Finite matched original default-head logits required')
 p=teacher.detach().softmax(1);q=((p.max(1).values-tau)/(1-tau)).clamp(0,1)
 # Batch mean preserves the confidence mass; no normalization by selected count.
 per=F.kl_div(F.log_softmax(student,1),p,reduction='none').sum(1)
 return (q*per).mean(),dict(q=q.detach(),per_sample=per.detach())
