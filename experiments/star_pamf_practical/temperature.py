"""Source holdout NLL calibration only; no target data or parameter gradients."""
import math
import torch
import torch.nn.functional as F


def fit_temperature(logits, labels, lower=.25, upper=10.):
    z=logits.detach().cpu().double(); y=labels.detach().cpu().long()
    if z.ndim!=2 or len(z)!=len(y) or not len(y) or not torch.isfinite(z).all():
        raise ValueError('Invalid source calibration logits')
    # NLL is convex in inverse temperature. Bounded golden-section search.
    def nll(beta):return float(F.cross_entropy(z*beta,y))
    a,b=1/upper,1/lower; ratio=(math.sqrt(5)-1)/2
    c=b-ratio*(b-a);d=a+ratio*(b-a);fc,fd=nll(c),nll(d)
    for _ in range(48):
        if fc<fd:b,d,fd=d,c,fc;c=b-ratio*(b-a);fc=nll(c)
        else:a,c,fc=c,d,fd;d=a+ratio*(b-a);fd=nll(d)
    candidates=[(nll(beta),beta) for beta in (1.,1/upper,1/lower,(a+b)/2)]
    best,beta=min(candidates);temp=1/beta
    return temp,dict(temperature=temp,nll_before=nll(1.),nll_after=best,samples=len(y),
        lower=lower,upper=upper,boundary=abs(temp-lower)<1e-5 or abs(temp-upper)<1e-5)
