"""Recover near-carrier complex structure without changing the broadband dose."""

import torch

from .complex_vector import ComplexVector, fft
from .denoise_components import bands


def same_size(direction, reference):
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    dose = torch.linalg.vector_norm(reference, dim=-1, keepdim=True)
    return torch.where(norm > 1e-8, direction * (dose/norm.clamp_min(1e-8)), reference)


def matched_near(prior, target):
    result = prior.clone()
    for band in bands(prior.shape[-1], prior.device):
        result[..., band] = same_size(target[..., band], prior[..., band])
    return result


class NearSpectrum(ComplexVector):
    """Refine two spectral directions while retaining every outer coefficient."""

    def forward(self, observed, baseline, quality, frozen_update):
        current = baseline + torch.fft.ifft(frozen_update, norm='ortho')
        proposal = .3*self.proposal(observed, current, quality)
        result = frozen_update.clone()
        for band in bands(frozen_update.shape[-1], frozen_update.device):
            reference = frozen_update[..., band]
            correction = proposal[..., band]
            base_rms = reference.abs().square().mean(-1, keepdim=True).sqrt()
            step_rms = correction.abs().square().mean(-1, keepdim=True)
            candidate = reference + .9*base_rms*correction*(1.+step_rms).rsqrt()
            result[..., band] = same_size(candidate, reference)
        return result


def near_error(delta, baseline, signal):
    wanted = fft(signal)
    difference = fft(baseline)+delta-wanted
    result = []
    for band in bands(delta.shape[-1], delta.device):
        reference_power = wanted[..., band].abs().square().mean(-1)
        relative = difference[..., band].abs().square().mean(-1) / (.001+reference_power)
        result.append(relative.log1p().mean())
    return .5*(result[0]+result[1])


def same_signal(delta, baseline, signal, to_first):
    """Compare near-band reconstruction errors in a shared physical frame."""
    physical = (baseline-signal + torch.fft.ifft(delta, norm='ortho')) * to_first
    difference = fft(physical[:, 0]-physical[:, 1])
    result = []
    reference = fft(signal[:, 0])
    for band in bands(difference.shape[-1], difference.device):
        residual = difference[..., band].abs().square().sum(-1)
        target = reference[..., band].abs().square().sum(-1)
        result.append((residual/(.001*band.sum()+target)).log1p().mean())
    return .5*(result[0]+result[1])
