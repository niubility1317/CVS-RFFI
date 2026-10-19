"""Quality logit is trained on an augmentation flag, never on TX identity.

The scalar output is not an identity or receiver readout. The existing TX head
and all restoration weights stay untouched. No normalization state or batch
statistics can couple independent observations at inference.
"""
import torch
from torch import nn


class QualityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 24, 7, stride=2, padding=3), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv1d(24, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.SiLU())
        self.output = nn.Sequential(nn.Flatten(), nn.Linear(32*32, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, received):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Current packet IQ[B,2,256] only')
        normalized = received / received.square().mean((1, 2), keepdim=True).clamp_min(1e-4).sqrt()
        spectrum = torch.fft.fft(torch.complex(normalized[:, 0], normalized[:, 1]), norm='ortho')
        features = torch.cat((normalized, spectrum.real[:, None], spectrum.imag[:, None]), 1)
        return self.output(self.encoder(features)).squeeze(1)


def keep_input(logits):
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise FloatingPointError('One finite quality logit per observation required')
    return logits < 0.


def select_iq(received, restored, logits):
    if received.shape != restored.shape or len(logits) != len(received):
        raise ValueError('Per-observation quality and waveform shapes differ')
    keep = keep_input(logits)
    return torch.where(keep[:, None, None], received, restored), keep
