"""DURA with fixed source-fitted, noise-conditioned complex spectral correction."""
import torch
from torch import nn

from experiments.phase1_dura_orthogonal.source_frequency_regression import spectra


class DuraFrequencyRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weights', torch.zeros((4, 256, 4), dtype=torch.complex64))
        self.register_buffer('limits', torch.zeros(3, dtype=torch.float32))

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        if (received.shape != anchor.shape or received.ndim != 3 or received.shape[1:] != (2, 256)
                or variance.shape != (len(received),)):
            raise ValueError('Matching received IQ, DURA IQ and physical variance required')
        if validate and (not torch.isfinite(received).all() or not torch.isfinite(anchor).all()
                         or not torch.isfinite(variance).all() or (variance <= 0).any()):
            raise ValueError('Finite IQ and positive DURA variance required')
        ratio = variance/received.square().mean((1, 2)).clamp_min(1e-12)
        group = torch.bucketize(ratio, self.limits)
        features, a, _, _ = spectra(received, anchor, anchor)
        recovered = torch.fft.ifft(a + (features*self.weights[group]).sum(-1), dim=-1, norm='ortho')
        return torch.stack((recovered.real, recovered.imag), 1)
