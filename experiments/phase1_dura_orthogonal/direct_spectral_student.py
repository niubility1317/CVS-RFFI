"""Independent IQ-only multiscale recovery student; DURA is training-only."""
import torch
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral


class DirectSpectralStudent(DuraMultiscaleSpectral):
    def forward(self, received, anchor=None, innovation=None, variance=None, *, validate=True):
        # Only the received record supplies inference inputs; external context is ignored.
        power = received.square().mean((1, 2)).clamp_min(1e-12)
        return super().forward(received, received, torch.zeros_like(received), power, validate=validate)
