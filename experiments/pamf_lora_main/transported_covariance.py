"""Transport estimated native RF noise covariance into the LoRa tracking frame."""

import torch

from .rf_noise_transport import FFT, native_samples, noise_profile
from .spectral_covariance import real_coordinates
from .variance_bound import spectral_contrast, variance_interval


def ac(value):
    return torch.cat((torch.zeros_like(value[..., :1]), value[..., 1:]), dim=-1)


def transported_diagonal(rotation, psd):
    """Diagonal of F R* F* diag(psd) F R F* via cyclic convolution."""
    kernel = torch.fft.fft(rotation.conj(), norm='ortho').abs().square() / rotation.shape[-1]
    return torch.fft.ifft(torch.fft.fft(psd) * torch.fft.fft(kernel)).real


def noise_action(value, rotation, psd, scale):
    shifted = torch.fft.ifft(scale * value, norm='ortho') * rotation
    native = torch.fft.fft(shifted, norm='ortho') * psd
    tracked = torch.fft.ifft(native, norm='ortho') * rotation.conj()
    return ac(scale * torch.fft.fft(tracked, norm='ortho'))


def signal_action(value, prior):
    """Apply the original shared/individual prior as a real-linear spectral operator."""
    total = prior.individual * value + prior.shared * value.sum(1, keepdim=True)
    bins = prior.correlated_bins
    features = real_coordinates(value.index_select(-1, bins))
    common = features.sum(dim=1, keepdim=True)
    factors = prior.factors
    diagonal = prior.shared_real - factors.square().sum(-1)
    correlated = prior.private_real * features + diagonal * common + (common @ factors) @ factors.T
    re, im = (2 * correlated).chunk(2, dim=-1)
    return ac(total.index_copy(-1, bins, torch.complex(re, im)))


def inner(first, second):
    return (first.conj() * second).real.sum(dim=(-2, -1), keepdim=True)


def solve(residual, prior, marginal, rotation, psd, *, steps=8):
    """Fixed-budget real PCG preconditioned by the existing diagonal posterior."""
    scale = (marginal / transported_diagonal(rotation, psd).clamp_min(1e-6)).sqrt()

    def inverse_diagonal(value):
        return (value - prior.innovation(value, marginal, True)) / marginal

    def apply(value):
        return signal_action(value, prior) + noise_action(value, rotation, psd, scale)

    x = inverse_diagonal(residual)
    error = residual - apply(x)
    z = inverse_diagonal(error)
    direction = z
    dot = inner(error, z)
    for _ in range(steps):
        image = apply(direction)
        step = dot / inner(direction, image).clamp_min(1e-20)
        x = x + step * direction
        error = error - step * image
        z = inverse_diagonal(error)
        updated = inner(error, z)
        direction = z + (updated / dot.clamp_min(1e-20)) * direction
        dot = updated
    # Verify the actual linear residual instead of the recursive approximation.
    error = residual - apply(x)
    relative = (inner(error, error) / inner(residual, residual).clamp_min(1e-20)).sqrt()
    return signal_action(x, prior), relative


def native_psd(observed, rotation):
    real = observed.real
    window = torch.hann_window(FFT, device=real.device, dtype=real.dtype)
    frequency = torch.arange(FFT, device=real.device)
    samples, outside, _ = native_samples(observed, rotation, window, frequency)
    profile, variance = noise_profile(samples, outside)
    color = profile / variance.clamp_min(1e-8)
    # Shrink color toward flat so the covariance remains well conditioned.
    color = .5 + .5 * color
    length = observed.shape[-1]
    at = torch.arange(length, device=real.device) * (FFT / length)
    low = at.floor().long()
    return torch.lerp(color[..., low], color[..., (low + 1) % FFT], at - low)


def half_edits(observed, rotation, quality, rms, model):
    """Keep the frozen proposal and noise marginal; change cross-bin covariance."""
    _, proposal, outer = model.propose(observed, quality)
    measured = torch.fft.fft(observed, norm='ortho')
    contrast, weights = spectral_contrast(measured, quality)
    variance, _ = variance_interval(outer, contrast, weights, model.selectors)
    marginal = (variance @ model.bands).clamp_min(1e-7)
    center = torch.fft.fft(proposal, norm='ortho')
    residual = ac(measured - center)
    prior = model.clean_detail
    diagonal = prior.innovation(residual, marginal, True)
    psd = native_psd(observed, rotation)
    transported, error = solve(residual, prior, marginal, rotation, psd)
    accepted = error <= .02
    transported = torch.where(accepted, transported, diagonal)
    control = .5 * torch.fft.ifft(diagonal - residual, norm='ortho')
    candidate = .5 * torch.fft.ifft(transported - residual, norm='ortho')
    control_rms = inner(control * rms, control * rms).sqrt()
    candidate_rms = inner(candidate * rms, candidate * rms).sqrt()
    matched = torch.where(candidate_rms > 1e-10, candidate * control_rms / candidate_rms.clamp_min(1e-10),
                          control)
    return control, candidate, matched, error
