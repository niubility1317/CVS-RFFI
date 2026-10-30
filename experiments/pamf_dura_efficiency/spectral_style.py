"""Source-calibrated spectral reweighting; no labels, F or target statistics."""
import torch
from experiments.pamf_restoration.models import complex_iq,pair


def bank_from_psd(power):
    if power.ndim!=2 or len(power)<2 or power.shape[1]<9 or not torch.isfinite(power).all() or (power<0).any():
        raise ValueError('Finite nonnegative source session PSD matrix required')
    # Fixed nine-bin smoothing suppresses packet-specific narrowband variation.
    smoothed=sum(torch.roll(power,shift,-1) for shift in range(-4,5))/9
    floor=smoothed.mean()*.001
    if floor<=0:raise ValueError('Nonzero source signal power required')
    stable=smoothed.clamp_min(floor)
    gain=(stable[None,:,:]/stable[:,None,:]).sqrt() # origin,destination,bin
    return gain,dict(smoothing_bins=9,relative_floor=.001,floor=float(floor),
        floored_fraction=float((smoothed<floor).double().mean()),
        gain_min=float(gain.min()),gain_max=float(gain.max()),
        condition_number_max=float((gain.max(-1).values/gain.min(-1).values).max()))


def spectral_reference(clean,gain):
    value=torch.fft.ifft(torch.fft.fft(complex_iq(clean),norm='ortho')*gain,norm='ortho')
    result=pair(value)
    original=clean.square().mean((1,2),keepdim=True)
    changed=result.square().mean((1,2),keepdim=True)
    return result*(original/changed.clamp_min(1e-12)).sqrt()


def paired_spectral(received,clean,actual_action,gain):
    reference=spectral_reference(clean,gain)
    return received+pair(actual_action(complex_iq(reference-clean))),reference
