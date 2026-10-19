import torch
from . import design as d


def merge_gradients(wave,identity,arm):
    if (arm not in d.ARMS or wave.ndim!=3 or wave.shape[1:]!=(2,256) or wave.shape!=identity.shape or
        wave.dtype!=torch.float32 or identity.dtype!=torch.float32 or wave.device!=identity.device or
        not torch.isfinite(wave).all() or not torch.isfinite(identity).all()):
        raise ValueError('Registered matching finite FP32 IQ output gradients required')
    dot=(wave*identity).sum((1,2));norm2=wave.square().sum((1,2))
    conflict=(dot<0)&(norm2>d.WAVE_NORM_SQUARED_MIN)
    applied=conflict if arm==d.ARMS[1] else torch.zeros_like(conflict)
    coefficient=torch.where(applied,dot/norm2.clamp_min(d.WAVE_NORM_SQUARED_MIN),torch.zeros_like(dot))
    adjusted=identity-coefficient[:,None,None]*wave
    merged=wave+adjusted
    if not torch.isfinite(merged).all():raise FloatingPointError('Nonfinite merged output gradient')
    return merged,dict(conflict=conflict.detach(),applied=applied.detach(),dot_before=dot.detach(),
        dot_after=(wave*adjusted).sum((1,2)).detach(),wave_norm2=norm2.detach(),identity_norm2=identity.square().sum((1,2)).detach(),
        removed_norm2=(coefficient[:,None,None]*wave).square().sum((1,2)).detach())


def output_gradient(estimate,paired,identity,arm):
    if paired.shape!=(len(estimate),) or identity.shape!=paired.shape:raise ValueError('One waveform/identity scalar per current IQ frame required')
    # Each task graph stops at estimate; the shared restorer graph is traversed
    # once afterwards. No Hessian, teacher-state update or inference gradient.
    wave=torch.autograd.grad(paired.sum(),estimate,retain_graph=True)[0]
    semantic=torch.autograd.grad(identity.sum(),estimate)[0]
    merged,stats=merge_gradients(wave,semantic,arm)
    return merged.detach()/len(estimate),stats
