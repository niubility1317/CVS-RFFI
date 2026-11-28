"""Fuse a learned real-gain restoration and a DURA prior inside both MAP stages."""
import torch
from torch import nn

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map


class StageGain(nn.Module):
    """Predict a real spectral trust coefficient; the same module serves both stages."""

    def __init__(self, *, packet_only=False):
        super().__init__()
        self.packet_only = packet_only
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
            prior_freq = gain_freq + trust * (dura_freq - gain_freq)
            prior = torch.fft.ifft(prior_freq, norm='ortho')
            solved = weighted_map(action, observed, prior, variance, steps=2, initial=current)
            solve_delta = torch.fft.fft(solved - prior, norm='ortho')
            current = prior + torch.fft.ifft(trust * solve_delta, norm='ortho')
            stages.append(current)
        return stages
