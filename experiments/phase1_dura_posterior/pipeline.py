"""Frozen posterior estimation and source-gated insertion into DURA."""

import torch
from torch import nn


class PhysicalPosterior(nn.Module):
    """Posterior mixture mean under a frozen estimated action and source prior."""

    def __init__(self, blind, means, covariances, priors, *, action_matrix, posterior, synthesizer=None):
        super().__init__()
        self.blind = blind.eval().requires_grad_(False)
        self.register_buffer('means', means)
        self.register_buffer('covariances', covariances)
        self.register_buffer('priors', priors)
        self.register_buffer('eye', torch.eye(512, dtype=means.dtype, device=means.device))
        self.action_matrix = action_matrix
        self.posterior = posterior
        self.synthesizer = None if synthesizer is None else synthesizer.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.blind.eval()
        if self.synthesizer is not None:
            self.synthesizer.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        action_parameters, logvar = self.blind(x)
        action = self.action_matrix(action_parameters)
        noise = self.eye[None]*logvar.exp()[:, None, None]
        states, logweights = self.posterior(x.reshape(len(x), 512), action, noise,
                                            self.means, self.covariances, self.priors)
        wave = (states*logweights.exp().unsqueeze(-1)).sum(1).reshape_as(x)
        if self.synthesizer is not None:
            candidates = torch.cat((x[:, None], states.reshape(len(x), -1, 2, 256)), dim=1)
            wave = self.synthesizer(wave, candidates, logvar)
        if not torch.isfinite(wave).all() or not torch.isfinite(logvar).all():
            raise ValueError('Finite posterior waveform and estimated noise are required')
        return wave, logvar


class GatedPosteriorDura(nn.Module):
    """Return original DURA for closed gates and corrected DURA for open gates."""

    def __init__(self, integrated, gate, posterior, *, threshold=0.8):
        super().__init__()
        if not 0 < threshold < 1:
            raise ValueError('Gate threshold must be a probability')
        self.integrated = integrated
        self.gate = gate.eval().requires_grad_(False)
        self.posterior = posterior.eval().requires_grad_(False)
        self.threshold = float(threshold)

    def train(self, mode=True):
        super().train(mode)
        self.gate.eval()
        self.posterior.eval()
        self.integrated.dura.eval()
        return self

    def forward(self, received):
        with torch.no_grad():
            probabilities = self.gate(received).sigmoid().reshape(-1)
            if probabilities.shape != (len(received),) or not torch.isfinite(probabilities).all():
                raise ValueError('Finite per-window gate probabilities are required')
            active = probabilities > self.threshold
            base = self.integrated.dura(received)
        if not active.any():
            return base, active
        selected = active.nonzero(as_tuple=True)[0]
        with torch.no_grad():
            posterior, logvar = self.posterior(received[selected])
        corrected = self.integrated(received[selected], posterior, logvar)
        return base.index_copy(0, selected, corrected), active
