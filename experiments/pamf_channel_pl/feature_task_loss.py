"""Exact first-order task gradient to existing features, optionally shielding the head."""
import torch
from torch.nn import functional as F


def task_loss(features,logits,target,accepted,shield_head):
    if (features.ndim!=2 or logits.ndim!=2 or len(features)!=len(logits) or not len(features)
            or target.shape!=logits.shape or target.requires_grad or accepted.shape!=(len(features),)
            or accepted.dtype!=torch.bool or not features.requires_grad or not logits.requires_grad
            or not isinstance(shield_head,bool)
            or any(v.device!=features.device for v in (logits,target,accepted))
            or not all(bool(torch.isfinite(v).all()) for v in (features,logits,target))
            or bool((target<0).any()) or not torch.allclose(target.sum(1),torch.ones_like(target[:,0]),atol=1e-5,rtol=1e-5)):
        raise ValueError('Connected training features, finite frozen categorical targets and aligned Boolean mask required')
    ce=(F.cross_entropy(logits,target.argmax(1),reduction='none')*accepted).mean()
    if not shield_head:
        return ce
    gradient=torch.autograd.grad(ce,features,retain_graph=True,allow_unused=True)[0]
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise ValueError('Default-head task must use the exact returned feature graph')
    # Value is CE; the first-order feature/backbone gradient is exactly CE's.
    # No auxiliary gradient reaches classification-path parameters or targets.
    # Parameters already used to construct features still receive gradients.
    return ce.detach()+((features-features.detach())*gradient.detach()).sum()
