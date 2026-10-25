"""Use default-head identity task magnitude to scale cross-RX reference gradients."""
import torch
from torch.nn import functional as F
from .receiver_reference import relation_terms

MAX_SCALE=32.
EPS=1e-12


def task_scaled_loss(features,logits,references,receivers,source_rxs,target,accepted):
    if not torch.is_grad_enabled() or not features.requires_grad or not logits.requires_grad:
        raise ValueError('Actual same-forward student features/head graph required')
    if logits.shape!=target.shape or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite actual default-head logits aligned with raw teacher target required')
    per_packet=relation_terms(features,references,receivers,source_rxs,target,accepted)
    reference=(per_packet*accepted).mean()
    hard_identity=(F.cross_entropy(logits,target.argmax(1),reduction='none')*accepted).mean()
    # These two calls only differentiate at the existing feature tensor. They
    # neither populate parameter.grad nor run the encoder's backward pass.
    gr=torch.autograd.grad(reference,features,retain_graph=True)[0]
    gh=torch.autograd.grad(hard_identity,features,retain_graph=True)[0]
    nr=gr.norm(dim=1);nh=gh.norm(dim=1)
    scale=(nh/(nr+EPS)).clamp(max=MAX_SCALE).detach()
    if not bool(torch.isfinite(scale).all()):raise ValueError('Finite task-based reference scales required')
    loss=(per_packet*scale*accepted).mean()
    count=accepted.sum().clamp_min(1)
    metrics=dict(task_scale_mean=(scale*accepted).sum()/count,
        task_scale_max=scale.max(),task_scale_sum=(scale*accepted).sum(),
        task_scale_cap_fraction=((scale>=MAX_SCALE)&accepted).sum()/count,
        hard_feature_gradient_norm_mean=(nh.detach()*accepted).sum()/count,
        reference_feature_gradient_norm_mean=(nr.detach()*accepted).sum()/count)
    return loss,{key:value.detach() for key,value in metrics.items()}
