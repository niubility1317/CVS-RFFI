import torch
from torch.nn import functional as F


def posterior_distance(estimated,reference):
    if estimated.ndim!=2 or estimated.shape!=reference.shape or estimated.shape[1]!=6:
        raise ValueError('Matched original six-class defaulthead logits required')
    if not torch.isfinite(estimated).all() or not torch.isfinite(reference).all():raise FloatingPointError('Finite native logits required')
    return (4*F.kl_div(F.log_softmax(estimated/2,dim=1),F.softmax(reference.detach()/2,dim=1),reduction='none').sum(1)).clamp_min(0)
