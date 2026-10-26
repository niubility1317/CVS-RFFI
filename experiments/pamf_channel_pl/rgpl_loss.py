"""Mutually exclusive self/graph U supervision with a fixed total CE dose."""
import torch
from torch.nn import functional as F


def unlabeled_loss(raw_logits,pseudo,accepted,graph_label,certified,*,rho,extra_logits=None):
    if (raw_logits.ndim!=2 or len(raw_logits)<1 or raw_logits.shape[1]<2
            or not raw_logits.is_floating_point() or not bool(torch.isfinite(raw_logits).all())
            or type(rho) not in (float,int) or not 0<=rho<=1):
        raise ValueError('Finite U logits and fixed CE dose split required')
    n,classes=raw_logits.shape
    for value,dtype in ((pseudo,torch.long),(graph_label,torch.long),(accepted,torch.bool),(certified,torch.bool)):
        if value.shape!=(n,) or value.dtype!=dtype or value.device!=raw_logits.device:
            raise ValueError('Aligned pseudo-labels and boolean source masks required')
    if bool(((pseudo<0)|(pseudo>=classes)|(graph_label<0)|(graph_label>=classes)).any()):
        raise ValueError('Pseudo-labels must belong to the registered class set')
    if rho==0:
        if extra_logits is not None:raise ValueError('GraphRaw must skip its second forward')
    elif (extra_logits is None or extra_logits.shape!=(int(certified.sum()),classes)
            or extra_logits.dtype!=raw_logits.dtype or extra_logits.device!=raw_logits.device
            or not bool(torch.isfinite(extra_logits).all())):
        raise ValueError('Exactly one extra view per certified source U required')
    labels=torch.where(certified,graph_label,pseudo)
    weights=torch.where(certified,raw_logits.new_tensor(1-rho),accepted.to(raw_logits.dtype))
    total=(F.cross_entropy(raw_logits,labels,reduction='none')*weights).sum()
    if rho and bool(certified.any()):
        total=total+rho*F.cross_entropy(extra_logits,graph_label[certified],reduction='sum')
    # Preserve ordinary-PL's full-U-batch normalization; no accepted-count scaling.
    return total/n
