"""Keep a fixed raw-IQ slot and equal physical/view loss normalization."""
import torch
from torch.nn import functional as F


def rd(first,second,y):
    if first.shape!=second.shape or first.shape!=(len(y),6) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()):
        raise ValueError('Legal L labels and default six-class logits required')
    if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):raise ValueError('Finite logits required')
    ce=.5*(F.cross_entropy(first,y,label_smoothing=.01)+F.cross_entropy(second,y,label_smoothing=.01))
    p,q=first.log_softmax(1),second.log_softmax(1)
    kl=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
    return ce+kl,ce,kl


def objective(logits,y):
    if len(logits)!=6:raise ValueError('Three matched pairs required')
    pairs=[rd(logits[i],logits[i+1],y) for i in (0,2,4)]
    total=sum(w*p[0] for w,p in zip((.5,.25,.25),pairs))
    metrics={f'{name}_{metric}':pair[index].detach() for name,pair in zip(('raw','g_slot','s_slot'),pairs) for index,metric in ((1,'CE'),(2,'KL'))}
    return total,metrics
