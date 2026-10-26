"""Explicitly noisy hard-pool relabeling; ordinary self-PL outside that pool."""
import torch
from torch.nn import functional as F


def component(logits,labels,q):
    if q not in (0.,.7):raise ValueError('Only preregistered CE or GCE(.7) permitted')
    if (logits.ndim!=2 or logits.shape[1]<2 or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
            or labels.shape!=(len(logits),) or labels.dtype!=torch.long or labels.device!=logits.device
            or bool(((labels<0)|(labels>=logits.shape[1])).any())):
        raise ValueError('Aligned finite logits and registered pseudo-labels required')
    logp=F.log_softmax(logits,dim=1).gather(1,labels[:,None]).squeeze(1)
    return -logp if q==0 else -torch.expm1(q*logp)/q


def mixed_loss(raw_logits,self_label,self_accepted,noisy_label,noisy_mask,*,q,rho,extra_logits=None,uniform_attenuation=False):
    if raw_logits.ndim!=2 or len(raw_logits)<1 or rho not in (0.,.25) or q not in (0.,.7):raise ValueError('Fixed full-U batch and objective required')
    n,c=raw_logits.shape
    for x,dtype in ((self_label,torch.long),(self_accepted,torch.bool),(noisy_label,torch.long),(noisy_mask,torch.bool)):
        if x.shape!=(n,) or x.dtype!=dtype or x.device!=raw_logits.device:raise ValueError('Aligned source U pseudo-labels and boolean masks required')
    if bool(((noisy_label<0)|(noisy_label>=c)).any()):raise ValueError('Noisy targets outside registered classes')
    count=int(noisy_mask.sum())
    if uniform_attenuation and (q!=0 or rho!=0):raise ValueError('Scaled-CE control only for raw CE')
    if not rho and extra_logits is not None:raise ValueError('Single raw route must skip second forward')
    if rho and count and (extra_logits is None or extra_logits.shape!=(count,c) or extra_logits.dtype!=raw_logits.dtype or extra_logits.device!=raw_logits.device):
        raise ValueError('Exactly one extra view per noisy candidate required')
    ordinary=(component(raw_logits,self_label,0.)*(self_accepted&~noisy_mask)).sum()
    noisy=component(raw_logits[noisy_mask],noisy_label[noisy_mask],q)
    if uniform_attenuation and count:
        p=raw_logits[noisy_mask].detach().softmax(1).gather(1,noisy_label[noisy_mask,None]).squeeze(1)
        noisy=noisy*p.pow(.7).mean()
    total=ordinary+(1-rho)*noisy.sum()
    if rho and count:total=total+rho*component(extra_logits,noisy_label[noisy_mask],q).sum()
    # As in ordinary PL, normalization includes every source U packet, not just accepted ones.
    return total/n
