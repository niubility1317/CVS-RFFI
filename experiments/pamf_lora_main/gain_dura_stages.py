"""Fuse a learned real-gain restoration and a DURA prior inside both MAP stages."""
import torch
from torch import nn

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map


class ComplexPriorResidual(nn.Module):
    """Learn a two-quadrature correction from the current state and its innovation."""

    def __init__(self, *, aligned_only=False):
        super().__init__()
        self.aligned_only = aligned_only
        self.temporal = self._branch(padding='replicate')
        self.spectral = self._branch(padding='circular')

    @staticmethod
    def _branch(padding):
        network = nn.Sequential(
            nn.Conv1d(11, 32, 7, padding=3, padding_mode=padding), nn.SiLU(),
            nn.Conv1d(32, 32, 5, dilation=3, padding=6, padding_mode=padding), nn.SiLU(),
            nn.Conv1d(32, 32, 5, dilation=7, padding=14, padding_mode=padding), nn.SiLU(),
            nn.Conv1d(32, 2, 3, padding=1, padding_mode=padding),
        )
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
        return network

    @staticmethod
    def _features(values, scale, noise_feature):
        wide = torch.cat([pair(value / scale) for value in values], dim=1).asinh()
        return torch.cat((wide, noise_feature), dim=1)

    def forward(self, temporal, spectral, variance, direction):
        observed = temporal[0]
        scale = observed.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        logvar = (variance / scale.square()).log().clamp(-16, 4) / 8
        noise_feature = logvar[:, None].expand(-1, 1, observed.shape[-1])
        t = complex_iq(self.temporal(self._features(temporal, scale, noise_feature)))
        f = complex_iq(self.spectral(self._features(spectral, scale, noise_feature)))
        delta = scale * (torch.fft.fft(t, norm='ortho') + f)
        if self.aligned_only:
            axis_energy = direction.abs().square().clamp_min(1e-8 * scale.square())
            delta = ((delta * direction.conj()).real / axis_energy) * direction
        power = delta.abs().square().mean(-1, keepdim=True)
        return delta * (.45 * scale / power.clamp_min(1e-16).sqrt()).clamp(max=1.)


class StageGain(nn.Module):
    """Predict a real spectral trust coefficient; the same module serves both stages."""

    def __init__(self, *, packet_only=False, correction=None):
        super().__init__()
        self.packet_only = packet_only
        self.correction = correction
        self.network = nn.Sequential(
            nn.Conv1d(8, 24, 7, padding=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(24, 24, 9, dilation=2, padding=8, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(24, 1, 5, padding=2, padding_mode='circular'),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def coefficient(self, observed, gain_prior, dura_prior, innovation, noise):
        powers = [x.abs().square() for x in (observed, gain_prior, dura_prior, innovation)]
        difference = (dura_prior - gain_prior).abs().square()
        relative = dura_prior * gain_prior.conj() / ((powers[1] + noise) * (powers[2] + noise)).sqrt()
        features = [torch.log1p(p / noise) for p in (*powers, difference)]
        features += [relative.real, relative.imag, noise.log().expand_as(difference) / 8]
        logits = self.network(torch.stack(features, dim=1))[:, 0]
        if self.packet_only:
            logits = logits.mean(-1, keepdim=True)
        return logits.sigmoid()

    def forward(self, received, gain_prior, context, priors):
        """DURA input is real IQ; its gain reference and returned stages are complex."""
        base, action, variance = context
        observed = complex_iq(received)
        fixed = torch.fft.fft(torch.stack((observed, gain_prior)), norm='ortho')
        y_freq, gain_freq = fixed.unbind(0)
        variance = variance.reshape(len(received), 1)
        noise = variance.clamp_min(1e-5) + .005
        logvar = (variance.log().clamp(-16, 4) / 8)[:, None].expand(-1, 1, observed.shape[-1])
        current = complex_iq(base)
        stages = []
        for prior_core in priors:
            innovation = action.adjoint(observed - action(current))
            residual = pair(innovation)
            residual = residual / (residual.square().mean((1, 2), keepdim=True) + 1e-6).sqrt()
            inputs = torch.cat((received, base, pair(current), residual, logvar), dim=1)
            dura = complex_iq(base + SCALE * prior_core(inputs))
            dura_freq = torch.fft.fft(dura, norm='ortho')
            innovation_freq = torch.fft.fft(innovation, norm='ortho')
            trust = self.coefficient(y_freq, gain_freq, dura_freq, innovation_freq, noise)
            direction = dura_freq - gain_freq
            prior_freq = gain_freq + trust * direction
            if self.correction is not None:
                temporal = (observed, gain_prior, dura, current, innovation)
                spectral = (y_freq, gain_freq, dura_freq, torch.fft.fft(current, norm='ortho'), innovation_freq)
                prior_freq = prior_freq + self.correction(temporal, spectral, variance, direction)
            prior = torch.fft.ifft(prior_freq, norm='ortho')
            solved = weighted_map(action, observed, prior, variance, steps=2, initial=current)
            solve_delta = torch.fft.fft(solved - prior, norm='ortho')
            current = prior + torch.fft.ifft(trust * solve_delta, norm='ortho')
            stages.append(current)
        return stages
