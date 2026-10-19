import torch
from . import design as d


def merge_gradients(wave,identity,coarse_count,arm):
    d.configuration(arm)
    if (wave.ndim!=1 or wave.shape!=identity.shape or not 0<coarse_count<len(wave) or wave.dtype!=torch.float32 or
        identity.dtype!=torch.float32 or wave.device!=identity.device or not torch.isfinite(wave).all() or not torch.isfinite(identity).all()):
        raise ValueError('Matching finite FP32 restorer vectors and component boundary required')
    ranges=[('all',0,len(wave))] if arm==d.ARMS[0] else [('coarse',0,coarse_count),('correction',coarse_count,len(wave))]
    merged=[];stats={}
    for name,start,end in ranges:
        w=wave[start:end];g=identity[start:end];w2=w.square().sum();i2=g.square().sum();dot=(w*g).sum()
        scale=torch.where(i2>d.NORM_SQUARED_MIN,(w2/i2.clamp_min(d.NORM_SQUARED_MIN)).sqrt().clamp(max=1),torch.ones_like(i2))
        adjusted=scale*g;merged.append(w+adjusted)
        stats[name]=dict(scale=scale.detach(),scaled=((scale<1)&(i2>d.NORM_SQUARED_MIN)).detach(),conflict=(dot<0).detach(),
            wave_norm2=w2.detach(),identity_norm2=i2.detach(),identity_after_norm2=adjusted.square().sum().detach(),
            dot_before=dot.detach(),dot_after=(w*adjusted).sum().detach())
    result=torch.cat(merged).detach()
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite merged parameter gradient')
    return result,stats


def parameter_gradients(model,paired,identity,arm):
    if paired.ndim!=1 or paired.shape!=identity.shape or not len(paired):raise ValueError('Matched per-IQ task losses required')
    named=list(model.named_parameters());parameters=[p for _,p in named];sizes=[p.numel() for p in parameters]
    flags=[name.startswith('coarse.') for name,_ in named];cut=sum(flags)
    if (not 0<cut<len(named) or flags!=[True]*cut+[False]*(len(named)-cut) or
        any(not name.startswith('correction.') for name,_ in named[cut:]) or any(not p.requires_grad for p in parameters)):
        raise ValueError('Registered trainable coarse/correction parameter ordering required')
    wave=torch.autograd.grad(paired.mean(),parameters,retain_graph=True)
    semantic=torch.autograd.grad(identity.mean(),parameters)
    merged,stats=merge_gradients(torch.cat([g.reshape(-1) for g in wave]),torch.cat([g.reshape(-1) for g in semantic]),sum(sizes[:cut]),arm)
    for p,g in zip(parameters,merged.split(sizes)):p.grad=g.reshape_as(p)
    return stats
