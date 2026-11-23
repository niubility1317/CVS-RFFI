"""Test candidate LoRa edits against non-overlapping chirp subsets."""

import torch

from .channel_blind import zero_carrier
from .variance_bound import spectral_contrast, variance_interval
from .weak_signal_probe import observed_noise_rms


def support_gain(before_a, before_b, after_a, after_b, bands):
    """Conservative held-chirp validation of a candidate common waveform update."""
    fft=lambda x:torch.fft.fft(x,norm='ortho')
    difference=fft(before_b-before_a)
    emissions=(fft(after_a-before_a),fft(after_b-before_b))
    error=difference.abs().square()@bands.T
    gains=[]
    for delta,sign in zip(emissions,(1.,-1.),strict=True):
        cross=(2*sign*(difference.conj()*delta).real)@bands.T
        penalty=delta.abs().square()@bands.T
        gains.append((cross-penalty)/error.clamp_min(1e-8))
    return torch.minimum(*gains)


def independent_proposals(prior, observed, coherence, bands, selectors):
    """Build analytic proposals from independent odd/even chirp observations."""
    for start in (0,1):
        y,q=observed[:,start::2],coherence[:,start::2]
        measured=torch.fft.fft(y,norm='ortho')
        contrast,w=spectral_contrast(measured,q)
        variance,_=variance_interval(observed_noise_rms(y).square(),contrast,w,selectors)
        noise=variance@bands
        prediction=prior.mean+prior.innovation(measured-prior.mean,noise,True)
        proposal=y+zero_carrier(torch.fft.ifft(prediction-measured,norm='ortho'))
        yield y, proposal


def observed_support(prior, observed, coherence, bands, selectors):
    """Compute proof scores from independent odd/even chirp proposal fits."""
    half = [(y.mean(1), torch.lerp(y, proposal, .5).mean(1)) for y, proposal in
            independent_proposals(prior, observed, coherence, bands, selectors)]
    return confirmation_scores(support_gain(half[0][0],half[1][0],half[0][1],half[1][1],bands))


def confirmation_scores(gains):
    return torch.stack((gains[..., :3].mean(-1), gains.mean(-1)), dim=-1)
