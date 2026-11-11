import torch
from experiments.pamf_tri_u_coherence_m1 import objective as base

GuardedCoherenceU=base.GuardedCoherenceU
masked_loss=base.masked_loss


def scales(coherence,mode,min_scale):
    if coherence.ndim!=1 or not bool(torch.isfinite(coherence).all()) or mode not in ('identity','expanded','expanded_uniform'):
        raise ValueError('Finite source coherence and registered mode required')
    if mode=='identity':return torch.ones_like(coherence)
    adaptive=1-((1-coherence.clamp(0,1))*100).clamp(0,1-min_scale)
    return adaptive if mode=='expanded' else adaptive.mean().expand_as(adaptive)
