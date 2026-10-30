"""Same-record target contrast without pseudo TX labels or a class count."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .star_representation_loss import phase_gain_view

TEMPERATURE=.1


def two_views(x,generator):
    """Independent existing phase/gain views of one fixed restored packet."""
    return phase_gain_view(x,generator),phase_gain_view(x,generator)


def instance_contrast(z0,z1):
    if (not torch.is_tensor(z0) or not torch.is_tensor(z1) or z0.ndim!=2 or z0.shape[1]!=160
            or len(z0)<2 or z0.shape!=z1.shape or z0.dtype!=torch.float32 or z1.dtype!=torch.float32
            or z0.device!=z1.device or not bool(torch.isfinite(z0).all()) or not bool(torch.isfinite(z1).all())
            or bool(z0.norm(dim=1).eq(0).any()) or bool(z1.norm(dim=1).eq(0).any())):
        raise ValueError('Two aligned finite nonzero FP32 target unit-feature views with at least two distinct records required')
    with torch.autocast(device_type=z0.device.type,enabled=False):
        a,b=F.normalize(z0,dim=1),F.normalize(z1,dim=1)
        z=torch.cat((a,b));n=len(a);index=torch.arange(2*n,device=z.device)
        similarity=z@z.T/TEMPERATURE
        logits=similarity.masked_fill(torch.eye(2*n,dtype=torch.bool,device=z.device),-torch.inf)
        positive=(index+n)%(2*n);loss=F.cross_entropy(logits,positive)
    if not bool(torch.isfinite(loss)):raise ValueError('Finite symmetric same-record contrastive loss required')
    return dict(loss=loss,positive_cosine=float((a*b).sum(1).mean().detach().cpu()),negative_count=2*n-2,anchors=2*n)
