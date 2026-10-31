"""Source cross-RX cosine-distance relations, with one frozen-F input VJP."""
import torch
from torch.nn import functional as F
from .signature_iq import signature_features

CAP=.5


def relation_error(signature,target,rx):
    if (signature.ndim!=2 or signature.shape[1]!=160 or target.shape!=signature.shape
        or rx.shape!=(len(signature),) or rx.dtype!=torch.int64):
        raise ValueError('Matched source Bx160 SIG and integer source RX required')
    source=F.normalize(signature,dim=1,eps=1e-4)
    reference=F.normalize(target.detach(),dim=1,eps=1e-4)
    mask=(rx[:,None]!=rx[None,:]).to(signature.dtype)
    peers=mask.sum(1);distance=1-source@source.T;desired=1-reference@reference.T
    denominator=(desired.square()*mask).sum(1)/peers.clamp_min(1)
    error=((distance-desired).square()*mask).sum(1)/peers.clamp_min(1)/denominator.clamp_min(1e-4)
    return error,dict(peers=peers,denominator=denominator,active=peers>0)


def bounded_relation(restored,clean,error):
    if restored.ndim!=3 or restored.shape[1:]!=(2,256) or clean.shape!=restored.shape or error.shape!=(len(restored),):
        raise ValueError('Matched source IQ and per-record relation errors required')
    # The SUM includes both endpoints of every cross-RX pair. The detached
    # per-record cap then preconditions this coupled input derivative; it is
    # not claimed to be the gradient of a weighted global pairwise objective.
    raw=torch.autograd.grad(error.sum(),restored)[0].detach()
    if not torch.isfinite(raw).all() or not torch.isfinite(error).all():raise FloatingPointError('Nonfinite source relation derivative')
    with torch.no_grad():
        norm=raw.flatten(1).norm(dim=1)
        iqnorm=2*(restored.detach()-clean.detach()).flatten(1).norm(dim=1)/(512*clean.detach().square().mean((1,2)).clamp_min(1e-4))
        weight=(CAP*iqnorm/norm.clamp_min(1e-12)).clamp_max(1)
        gradient=raw*weight[:,None,None];value=error.detach()*weight
    injection=(restored*gradient).flatten(1).sum(1)
    return value+(injection-injection.detach()),dict(weight=weight,raw_pressure=norm/iqnorm.clamp_min(1e-12),
        bounded_pressure=gradient.flatten(1).norm(dim=1)/iqnorm.clamp_min(1e-12),pressure_defined=iqnorm>1e-12)


def relation_guide(reader,restored,clean,target,rx):
    signature=signature_features(reader,restored)
    instance=(signature-target.detach()).square().sum(1)/target.detach().square().sum(1).clamp_min(1e-4)
    error,relation=relation_error(signature,target,rx)
    guide,info=bounded_relation(restored,clean,error)
    return guide,dict(info,instance_error=instance.detach(),relation_error=error.detach(),
                      peers=relation['peers'],denominator=relation['denominator'],active=relation['active'])
