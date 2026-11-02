"""Noise variance estimate normalized by the observed IQ signal power."""
import torch

def relative_logvar(iq,logvar):
 if iq.ndim!=3 or iq.shape[1]!=2 or logvar.shape!=(len(iq),):raise ValueError('IQ [N,2,T] and log variance [N] required')
 power=iq.square().sum(1).mean(1).clamp_min(1e-16)
 if not torch.isfinite(iq).all() or not torch.isfinite(logvar).all():raise ValueError('Finite IQ and variance required')
 return logvar-power.log()
