import torch
from . import design as d


def merge_gradients(wave,identity,arm):
    if (arm not in d.ARMS or wave.ndim!=3 or wave.shape[1:]!=(2,256) or wave.shape!=identity.shape or
        wave.dtype!=torch.float32 or identity.dtype!=torch.float32 or wave.device!=identity.device or
        not torch.isfinite(wave).all() or not torch.isfinite(identity).all()):
        raise ValueError('Registered matching finite FP32 IQ output gradients required')
    w2=wave.square().sum((1,2));i2=identity.square().sum((1,2));dot=(wave*identity).sum((1,2))
    if arm==d.ARMS[0]:
        a=w2.sum();b=i2.sum();scale=torch.where(b>d.NORM_SQUARED_MIN,(a/b.clamp_min(d.NORM_SQUARED_MIN)).sqrt().clamp(max=1),torch.ones_like(b)).expand_as(w2)
    else:scale=torch.where(i2>d.NORM_SQUARED_MIN,(w2/i2.clamp_min(d.NORM_SQUARED_MIN)).sqrt().clamp(max=1),torch.ones_like(i2))
    adjusted=identity*scale[:,None,None];merged=wave+adjusted
    if not torch.isfinite(merged).all():raise FloatingPointError('Nonfinite norm-balanced output gradient')
    return merged,dict(conflict=(dot<0).detach(),scaled=((scale<1)&(i2>d.NORM_SQUARED_MIN)).detach(),scale=scale.detach(),
        dot_before=dot.detach(),dot_after=(wave*adjusted).sum((1,2)).detach(),wave_norm2=w2.detach(),
        identity_norm2=i2.detach(),identity_after_norm2=adjusted.square().sum((1,2)).detach())


def output_gradient(estimate,paired,identity,arm):
    if paired.shape!=(len(estimate),) or identity.shape!=paired.shape:raise ValueError('One waveform/identity scalar per IQ frame required')
    wave=torch.autograd.grad(paired.sum(),estimate,retain_graph=True)[0]
    semantic=torch.autograd.grad(identity.sum(),estimate)[0]
    merged,stats=merge_gradients(wave,semantic,arm)
    return merged.detach()/len(estimate),stats
