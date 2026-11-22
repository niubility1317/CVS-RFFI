"""Source-checked ambiguous labels for LoRa Tri."""
import torch
from torch.nn import functional as F


def candidate_sets(probability, hard_classes, minimum=.95):
    """Keep a two- or three-device set shared by all three views."""
    top=probability.topk(3,dim=-1).indices
    mask=F.one_hot(top,num_classes=probability.shape[-1]).sum(2).bool().all(1)
    size=mask.sum(1)
    weak=torch.ones(mask.shape[-1],device=mask.device,dtype=torch.bool)
    weak[hard_classes]=False
    mass=(mask*probability.mean(1)).sum(1)
    accepted=(size>=2)&(size<=3)&(mask&weak).any(1)&(mass>=minimum)
    return mask&accepted[:,None]


def set_ce(logits, mask, batch_size):
    """Negative log probability that the identity belongs to the set."""
    if not len(logits):
        return logits.new_zeros(())
    total=logits.logsumexp(1)
    supported=logits.masked_fill(~mask,-torch.inf).logsumexp(1)
    return (total-supported).sum()/batch_size
