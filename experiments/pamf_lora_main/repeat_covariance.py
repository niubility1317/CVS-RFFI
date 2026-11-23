"""Observe noise covariance across LoRa chirps, then apply one learned two-stage shrinkage."""

import torch
from torch import nn

from .chirp_spectrum_recovery import power


REPEATS = 6
BLOCK = 8
MAX_EDIT = .35
MAX_LOGIT_SHIFT = .75


def shrinkage_head(incoming):
    head = nn.Sequential(nn.Conv1d(incoming, 32, 7, padding=3), nn.SiLU(),
                         nn.Conv1d(32, 32, 5, padding=2), nn.SiLU(),
                         nn.Conv1d(32, REPEATS, 1))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def repeated_noise(clean, rng):
    """Independent and shared generic additive disturbances, each with a colored spectrum."""
    kwargs = dict(device=clean.device, generator=rng)
    size = clean.shape

    def noise(shape):
        sample = torch.complex(torch.randn(shape, **kwargs), torch.randn(shape, **kwargs))
        tap = 1.6 * torch.rand((len(clean), 1, 1), **kwargs) - .8
        return sample + tap * sample.roll(1, -1)

    independent = noise(size)
    shared = noise((len(clean), 1, size[-1]))
    coupling = torch.polar(torch.ones(size[:2], device=clean.device),
                           2 * torch.pi * torch.rand(size[:2], **kwargs))
    dependence = torch.rand((len(clean), 1, 1), **kwargs).square()
    mixed = independent * (1 - dependence).sqrt() + shared * coupling[..., None] * dependence.sqrt()
    dose = 1.35 * torch.rand((*size[:2], 1), **kwargs).square()
    norm = (power(clean).mean(-1, keepdim=True) / power(mixed).mean(-1, keepdim=True).clamp_min(1e-8)).sqrt()
    return clean + mixed * dose * norm


def radial_at_same_dose(observed, restored):
    """Match the per-bin edit length and remove only along the received complex ray."""
    spectrum = torch.fft.fft(observed, norm='ortho')
    corrected = torch.fft.fft(restored, norm='ortho')
    amplitude = spectrum.abs()
    length = (corrected - spectrum).abs()
    shrinkage = length / amplitude.clamp_min(1e-12)
    return torch.fft.ifft(spectrum * (1 - shrinkage), norm='ortho')


class RepeatCovariance(nn.Module):
    def __init__(self, correlated, *, protected=False):
        super().__init__()
        self.correlated = correlated
        self.protected = protected
        self.coarse = shrinkage_head(2 * REPEATS + 3)
        self.fine = shrinkage_head(3 * REPEATS + 3)
        freq = torch.fft.fftshift(torch.fft.fftfreq(256)).abs()
        self.register_buffer('frequency', freq, persistent=False)
        self.register_buffer('outer', torch.nonzero(freq >= .375).flatten(), persistent=False)
        self.register_buffer('eye', torch.eye(REPEATS, dtype=torch.complex64), persistent=False)

    def noise_covariance(self, spectrum):
        side = spectrum.index_select(-1, self.outer)
        energy = power(side).mean(1)
        # Bound the influence of a few strong out-of-band signal or interference bins.
        cap = (3 * energy.median(dim=-1, keepdim=True).values / energy.clamp_min(1e-8)).clamp(max=1)
        side = side * cap[:, None].sqrt()
        covariance = side @ side.mH / cap.sum(dim=-1).clamp_min(1)[:, None, None]
        diagonal = covariance.diagonal(dim1=-2, dim2=-1).real
        diagonal_matrix = torch.diag_embed(diagonal).to(spectrum.dtype)
        estimate = .25 * diagonal_matrix + .75 * covariance if self.correlated else diagonal_matrix
        return estimate + 1e-4 * self.eye, diagonal.mean(dim=-1)

    def forward(self, y, quality):
        spectrum = torch.fft.fftshift(torch.fft.fft(y, norm='ortho'), dim=-1)
        covariance, noise = self.noise_covariance(spectrum)
        lower = torch.linalg.cholesky_ex(covariance, check_errors=False).L
        whitened = torch.linalg.solve_triangular(lower, spectrum, upper=False)
        ac = whitened * (self.frequency > 0)
        blocks = ac.unflatten(-1, (-1, BLOCK)).transpose(1, 2)
        outer = blocks @ blocks.mH / BLOCK
        pooled = (outer.roll(1, dims=1) + 2 * outer + outer.roll(-1, dims=1)) / 4
        levels, axes = torch.linalg.eigh(pooled)
        transformed = axes.mH @ blocks
        representation = power(transformed).transpose(1, 2).flatten(2)
        level = levels.clamp_min(0).transpose(1, 2).repeat_interleave(BLOCK, dim=-1)
        if self.protected:
            prior = -level.clamp_min(.05).log()  # Removal fraction: 1 / (1 + level).
        else:
            prior = level.clamp_min(1).reciprocal().clamp(.02, .98).logit()
        scalar = torch.stack((noise.log1p(), quality.mean(dim=(1, 2))), dim=1)
        side = torch.cat((scalar[:, :, None].expand(-1, -1, y.shape[-1]),
                          self.frequency.expand(len(y), 1, -1)), dim=1)
        features = torch.cat((level.log1p(), representation.log1p(), side), dim=1)
        mixed = self.training and y.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            coarse_change = self.coarse(features)
        coarse_change = coarse_change.float()

        def removal(update):
            if self.protected:
                update = MAX_LOGIT_SHIFT * update.tanh()
            return (prior + update).sigmoid()

        rejection = removal(coarse_change)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            fine_change = self.fine(torch.cat((features, (representation * rejection.square()).log1p()), dim=1))
        if self.protected:
            refined = removal(coarse_change + fine_change.float())
        else:
            refined = (prior + coarse_change + fine_change.float()).sigmoid()
        strength = noise[:, None, None] / (noise[:, None, None] + .006)
        basis = lower[:, None] @ axes

        def restore(shrink):
            weights = (strength * shrink).unflatten(-1, (-1, BLOCK)).transpose(1, 2)
            delta = (basis @ (transformed * weights)).transpose(1, 2).flatten(2)
            if self.protected:
                # The complex edit disk guarantees amplitude retention and limits
                # angular deviation in every original tracked bin.
                radius = MAX_EDIT * spectrum.abs()
                delta = delta * (radius / (radius + delta.abs()).clamp_min(1e-12))
            return torch.fft.ifft(torch.fft.ifftshift(spectrum - delta, dim=-1), norm='ortho')

        return restore(rejection), restore(refined)
