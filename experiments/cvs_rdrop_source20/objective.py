import torch
from torch.nn import functional as F

def symmetric_kl(a,b):
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=6 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Finite paired default six-class logits required')
    p,q=F.log_softmax(a,dim=1),F.log_softmax(b,dim=1)
    return .5*((p.exp()-q.exp())*(p-q)).sum(1).mean()

def gradient_stats(a,b):
    a,b=a.double(),b.double();aa=float(a.square().sum());bb=float(b.square().sum());dot=float(a@b)
    return dict(CE_norm=aa**.5,auxiliary_norm=bb**.5,auxiliary_over_CE=(bb/aa)**.5 if aa>0 else None,
        cosine=dot/(aa*bb)**.5 if aa*bb>0 else None)
