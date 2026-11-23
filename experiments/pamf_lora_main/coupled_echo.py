"""Joint LoRa waveform estimation with a latent short echo in the same solve."""

import torch

from .channel_blind import zero_carrier
from .crossfit_pilot import LAGS
from .factorized_chirps import FactorizedChirps, packet_posterior
from .variance_bound import spectral_contrast, variance_interval


def joint_precision(rhs, noise, shared, individual):
    """Apply the inverse joint observation covariance to one or more vectors."""
    precision = (noise + individual).clamp_min(1e-8).reciprocal()[..., None]
    shared = shared[..., None]
    pooled = shared * (precision * rhs).sum(dim=1, keepdim=True)
    pooled = pooled / (1 + shared * precision.sum(dim=1, keepdim=True))
    return (rhs - pooled) * precision


def short_echo_columns(observed, template, bins, scale):
    carrier = observed.mean(-1, keepdim=True)
    delayed = torch.stack([template.roll(lag, -1) for lag in LAGS], dim=-1)
    waveforms = carrier[..., None] * delayed * template.conj()[..., None]
    spectrum = torch.fft.fft(waveforms, dim=2, norm='ortho').index_select(2, bins)
    return spectrum * scale


def echo_system(columns, noise, shared, individual):
    # B^H A^-1 is obtained from A^-1 B since A is Hermitian.
    weighted = joint_precision(columns, noise, shared, individual).flatten(1, 2)
    flat = columns.flatten(1, 2)
    covariance = flat.mH @ weighted
    dimension = columns.shape[-1]
    small = torch.eye(dimension, device=columns.device, dtype=columns.dtype)
    small = small + .5 * (covariance + covariance.mH)
    action = torch.linalg.solve_ex(small, weighted.mH).result
    log_penalty = torch.linalg.slogdet(small).logabsdet
    return flat, weighted.mH, action, log_penalty


def integrated_echo(residual, system):
    flat, adjoint, action, log_penalty = system
    value = residual.flatten(1)[..., None]
    coefficients = action @ value
    reduction = ((adjoint @ value).mH @ coefficients).real.flatten()
    correction = (flat @ coefficients).reshape(residual.shape)
    return correction, reduction - log_penalty


class CoupledEcho(FactorizedChirps):
    def __init__(self, widths, *, echo):
        super().__init__(widths, coupled=True)
        self.echo = echo
        bins = torch.cat((torch.arange(1, 33), torch.arange(224, 256)))
        self.register_buffer('echo_bins', bins, persistent=False)
        self.register_buffer('echo_scale', torch.tensor([2**.5 * .06 / (1 + abs(lag)) for lag in LAGS]),
                             persistent=False)

    def forward(self, observed, quality, template, *, controls=False):
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        coarse, fine, outside = self.propose(observed, quality, mixed_conv=mixed)
        measured = torch.fft.fft(observed, norm='ortho')
        moment, weights = spectral_contrast(measured, quality)
        band_noise, _ = variance_interval(outside, moment, weights, self.selectors)
        noise = band_noise @ self.bands
        shared, individual = self.clean_detail.shared, self.clean_detail.individual
        bins = self.echo_bins
        if self.echo or controls:
            columns = short_echo_columns(observed, template, bins, self.echo_scale)
            system = echo_system(columns, noise.index_select(-1, bins), shared[bins], individual[bins])
            clean_residual = (measured - self.clean_detail.mean).index_select(-1, bins)
            _, evidence = integrated_echo(clean_residual, system)
            belief = (.25 * evidence).clamp(-12., 12.).sigmoid()[:, None, None]

        def restore(center, echo):
            residual = measured - center
            if echo:
                portion = residual.index_select(-1, bins)
                correction, _ = integrated_echo(portion, system)
                residual = residual.index_copy(-1, bins, portion - belief * correction)
            spectrum = center + packet_posterior(residual, noise, shared, individual, coupled=True)
            return observed + zero_carrier(torch.fft.ifft(spectrum - measured, norm='ortho'))

        initial = restore(torch.fft.fft(coarse, norm='ortho'), self.echo)
        proposal = torch.fft.fft(fine, norm='ortho')
        result = restore(proposal, self.echo)
        if controls:
            mean = self.clean_detail.mean
            analytic_joint = restore(mean, False)
            analytic_echo = restore(mean, True)
            echo_off = restore(proposal, False) if self.echo else result
            return initial, result, analytic_joint, analytic_echo, echo_off, belief
        return initial, result
