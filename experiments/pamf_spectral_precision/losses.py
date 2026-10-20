import numpy as np
import torch
from . import design as d
from experiments.pamf_spatial_conditioning.losses import posterior_distance
from experiments.pamf_reception_restore.runner import pair_loss as original_pair_loss


def precision_weights(power,arm):
    if arm not in d.ARMS:raise ValueError('Registered spectral weighting required')
    power=np.asarray(power,np.float64)
    if power.shape!=(256,) or not np.isfinite(power).all() or (power<0).any() or power.mean()<=0:
        raise ValueError('Finite nonnegative nonzero global256-bin source power required')
    if arm==d.ARMS[0]:return np.ones(256,np.float32)
    denominator=power/power.mean()+d.PSD_FLOOR
    low=0.;high=d.MAX_PRECISION*float(denominator.max())
    for _ in range(64):
        middle=(low+high)/2
        if np.clip(middle/denominator,d.MIN_PRECISION,d.MAX_PRECISION).mean()<1:low=middle
        else:high=middle
    return np.clip(((low+high)/2)/denominator,d.MIN_PRECISION,d.MAX_PRECISION).astype(np.float32)


def source_precision(clean,arm):
    if clean.shape!=(63000,2,256):raise ValueError('Original sourceL/U63000x2x256 required')
    power=np.zeros(256,np.float64)
    for offset in range(0,len(clean),512):
        value=np.asarray(clean[offset:offset+512],np.float64)
        spectrum=np.fft.fft(value[:,0]+1j*value[:,1],norm='ortho')
        power+=np.square(np.abs(spectrum)).sum(0)
    power/=len(clean);weights=precision_weights(power,arm)
    state=dict(arm=arm,count=len(clean),roles=['L_s','U_s'],TX_labels_read=0,V_reads=0,target_reads=0,
        rule=d.configuration(arm)['spectral_rule'],weight_min=float(weights.min()),weight_max=float(weights.max()),
        weight_mean=float(weights.astype(np.float64).mean()),source_global_power_mean=float(power.mean()))
    return weights,state


def pair_loss(estimate,clean,weights):
    if (estimate.shape!=clean.shape or estimate.ndim!=3 or estimate.shape[1:]!=(2,256) or
        weights.shape!=(256,) or weights.dtype!=estimate.dtype or weights.device!=estimate.device or
        weights.requires_grad or not torch.isfinite(weights).all() or float(weights.min())<.25 or float(weights.max())>4 or
        abs(float(weights.mean())-1)>1e-5):raise ValueError('MatchingIQ and fixedboundedmean1 spectral weights required')
    paired,terms=original_pair_loss(estimate,clean)
    error=estimate-clean
    spectrum=torch.fft.fft(torch.complex(error[:,0],error[:,1]),norm='ortho')
    energy=clean.square().sum(1).mean(1).clamp_min(2e-4)
    spectral=(spectrum.abs().square()*weights[None]).mean(1)/energy
    return paired+d.SPECTRAL_WEIGHT*spectral,dict(terms,spectral_nmse=spectral)
