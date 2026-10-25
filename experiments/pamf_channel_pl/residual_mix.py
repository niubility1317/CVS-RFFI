"""Training-only interpolation of the raw/restored IQ of one physical packet."""
import torch


def mix_iq(raw,restored,alpha):
    if (raw.ndim!=3 or tuple(raw.shape[1:])!=(2,256) or not len(raw) or restored.shape!=raw.shape
            or alpha.shape!=(len(raw),) or raw.dtype!=torch.float32 or restored.dtype!=raw.dtype or alpha.dtype!=raw.dtype
            or raw.requires_grad or restored.requires_grad or alpha.requires_grad
            or restored.device!=raw.device or alpha.device!=raw.device
            or not all(bool(torch.isfinite(v).all()) for v in (raw,restored,alpha))
            or bool((alpha<0).any()) or bool((alpha>1).any())):
        raise ValueError('Aligned detached FP32 raw/restored IQ and per-packet interpolation in [0,1] required')
    # lerp preserves the exact raw and deployed-restoration endpoints.
    return torch.lerp(raw,restored,alpha[:,None,None])


def paired_views(raw,restored,route,generator):
    if (route not in ('MixMix','MixG') or not isinstance(generator,torch.Generator)
            or str(generator.device)!=str(raw.device)):
        raise ValueError('Fixed residual-view route and separate same-device random generator required')
    # Both routes advance the private stream equally, so their first mixed
    # view remains matched over all batches. The native RNG is untouched.
    draws=tuple(torch.rand(len(raw),device=raw.device,dtype=raw.dtype,generator=generator) for _ in range(2))
    alphas=draws if route=='MixMix' else draws[:1]
    views=tuple(mix_iq(raw,restored,a) for a in alphas)
    if route=='MixG':views=views+(restored,)
    return views,alphas
