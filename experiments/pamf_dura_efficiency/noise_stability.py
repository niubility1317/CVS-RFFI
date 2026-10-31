"""Fixed source-only additive-noise stress and its preregistered decision."""
import math
import torch


def paired_noise(received,variance,clean_mask,standard_normal):
    if (received.ndim!=3 or received.shape[1:]!=(2,256) or standard_normal.shape!=received.shape
        or variance.shape!=(len(received),) or clean_mask.shape!=(len(received),) or clean_mask.dtype!=torch.bool
        or not torch.isfinite(variance).all() or (variance<0).any()):
        raise ValueError('Aligned source IQ, nonnegative real-component variance and clean mask required')
    noise=standard_normal*variance.sqrt()[:,None,None]
    noise=torch.where(clean_mask[:,None,None],torch.zeros_like(noise),noise)
    return received+noise,received-noise,noise


def qualifies(rows):
    if len(rows)!=2 or {r['view'] for r in rows}!={'1','20'}:
        raise ValueError('Both fixed source views required')
    return all(math.isfinite(r['noise_gain_ratio']) and math.isfinite(r['primary_MSE_ratio'])
        and r['noise_gain_ratio']>=1.25 and r['primary_MSE_ratio']>=1.10 for r in rows)
