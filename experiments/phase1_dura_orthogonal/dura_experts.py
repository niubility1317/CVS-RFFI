"""Source-trained mixture of DURA corrections conditioned on observed physics."""
import torch
from torch import nn

from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral

EXPERT_COUNT = 3


class DuraExperts(nn.Module):
    def __init__(self, width=32, taps=9):
        super().__init__()
        self.experts = nn.ModuleList(DuraMultiscaleSpectral(width=width, taps=taps)
                                     for _ in range(EXPERT_COUNT))
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, EXPERT_COUNT))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape
                or innovation.shape != received.shape or variance.shape != (len(received),)):
            raise ValueError('Matching per-record IQ and variance required')
        energy = anchor.square().mean((1, 2)).clamp_min(1e-12)
        raw_energy = received.square().mean((1, 2)).clamp_min(1e-12)
        innovation_energy = innovation.square().mean((1, 2)).clamp_min(1e-12)
        features = torch.stack((variance.clamp_min(1e-12).log()-energy.log(),
                                innovation_energy.log()-energy.log(),
                                raw_energy.log()-energy.log()), dim=1).clamp(-16, 16)/8
        weights = self.gate(features).softmax(dim=1)
        candidates = torch.stack([expert(received, anchor, innovation, variance, validate=validate)
                                  for expert in self.experts], dim=1)
        return anchor + ((candidates-anchor[:, None])*weights[:, :, None, None]).sum(dim=1)
