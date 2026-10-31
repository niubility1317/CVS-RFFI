"""Same-record soft-target self-distillation; no labels, neighbors or IO."""
import math
import torch
from torch.nn import functional as F


def _finite(x,name):
    if not torch.is_tensor(x) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite floating-point '+name+' required')
    return x


def cosine_logits(z,head):
    _finite(z,'features');_finite(head,'head')
    if z.ndim!=2 or head.ndim!=2 or z.shape[1]!=head.shape[1] or z.device!=head.device or z.dtype!=head.dtype:
        raise ValueError('Aligned feature/head matrices required')
    if bool(z.norm(dim=1).eq(0).any()) or bool(head.norm(dim=1).eq(0).any()):raise ValueError('Nonzero feature/head rows required')
    return F.normalize(z,dim=1)@F.normalize(head,dim=1).T


def teacher_targets(logits,center,temperature=.04):
    _finite(logits,'teacher logits');_finite(center,'center')
    if logits.ndim!=2 or center.shape!=(logits.shape[1],) or center.device!=logits.device or temperature<=0:
        raise ValueError('Aligned old center and teacher logits required')
    with torch.no_grad():return F.softmax((logits.detach()-center.detach())/temperature,dim=1).detach()


def soft_target_ce(logits,targets,temperature=.1):
    _finite(logits,'student logits');_finite(targets,'soft targets')
    if logits.ndim!=2 or not len(logits) or targets.shape!=logits.shape or targets.device!=logits.device or temperature<=0:
        raise ValueError('Aligned same-record soft targets required')
    if bool((targets<0).any()) or not torch.allclose(targets.sum(1),torch.ones(len(targets),device=targets.device),rtol=0,atol=2e-6):
        raise ValueError('Normalized nonnegative soft targets required')
    return -(targets.detach()*F.log_softmax(logits/temperature,dim=1)).sum(1).mean()


@torch.no_grad()
def update_center_(center,teacher_logits,momentum=.9):
    _finite(center,'center');_finite(teacher_logits,'teacher logits')
    if teacher_logits.ndim!=2 or not len(teacher_logits) or center.shape!=(teacher_logits.shape[1],) or not 0<=momentum<1:
        raise ValueError('Aligned full-U logits and center momentum required')
    center.mul_(momentum).add_(teacher_logits.detach().mean(0),alpha=1-momentum)


@torch.no_grad()
def ema_tensor_(teacher,student,momentum=.996):
    _finite(teacher,'teacher parameter');_finite(student,'student parameter')
    if teacher.shape!=student.shape or teacher.dtype!=student.dtype or teacher.device!=student.device or not math.isfinite(momentum) or not 0<=momentum<1:
        raise ValueError('Matching EMA tensors and finite momentum required')
    teacher.mul_(momentum).add_(student.detach(),alpha=1-momentum)


@torch.no_grad()
def ema_update_(teacher,student,momentum=.996):
    left=dict(teacher.named_parameters());right=dict(student.named_parameters())
    if set(left)!=set(right) or any(p.requires_grad or p.grad is not None for p in left.values()):
        raise ValueError('Frozen stop-gradient teacher must match student parameters')
    count=0
    for name,param in right.items():
        if not param.requires_grad:continue
        ema_tensor_(left[name],param,momentum);count+=param.numel()
    # Buffers and frozen Sinc leaves are deliberately neither copied nor EMA'd.
    return count
