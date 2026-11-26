"""Observation-conditioned complex spectral proposal for a fixed W24 parent."""

import torch

from experiments.pamf_lora_main.w24_second_stage import SecondStage


class CovarianceRefiner(SecondStage):
    """Use per-packet spectral statistics as a second input to the trainable prior."""

    def __init__(self, parent, *, cross_chirp, seed):
        super().__init__(parent, conditioned=True, seed=seed)
        self.cross_chirp = cross_chirp
        bins = torch.cat((torch.arange(-64, -8), torch.arange(8, 64))).remainder(384)
        self.register_buffer('detail_bins', bins, persistent=False)

    def observation_state(self, y, first_full):
        spectrum = torch.fft.fft(y, norm='ortho')
        noise = (spectrum[..., self.refiner.noise_band].abs().square().median(-1).values
                 / .69314718056).clamp(1e-5, 4.)[:, None, None, :]
        samples = spectrum[..., self.detail_bins].transpose(1, 2).unflatten(1, (-1, 8))
        power = samples.abs().square()
        # Exclude the coefficient being filtered from its own covariance estimate.
        diagonal = ((power.sum(dim=-2, keepdim=True) - power) / 7).maximum(noise)
        inverse_std = diagonal.rsqrt()
        whitened = samples * inverse_std
        if self.cross_chirp:
            outer = samples.unsqueeze(-1) * samples.conj().unsqueeze(-2)
            cov = (outer.sum(dim=-3, keepdim=True) - outer) / 7
            off = cov - cov.diagonal(dim1=-2, dim2=-1).diag_embed()
            off = off * inverse_std.unsqueeze(-1) * inverse_std.unsqueeze(-2)
            # Keep the first inverse-expansion term contractive even at high coherence.
            row_bound = off.abs().sum(-1).amax(-1, keepdim=True).clamp_min(.5)
            off = off * (.5 / row_bound.unsqueeze(-1))
            whitened = whitened - (off * whitened.unsqueeze(-2)).sum(-1)
        correction = -noise * inverse_std * whitened
        delta = torch.zeros_like(spectrum)
        delta[..., self.detail_bins] = correction.flatten(1, 2).transpose(1, 2)
        return y + torch.fft.ifft(delta, norm='ortho')
