"""DURA residual refinement with separate physical and orthogonal directions."""
import torch
from torch import nn
from torch.nn import functional as F


def split_direction(residual, direction):
    """Split each real-IQ window into a direction-parallel and orthogonal part."""
    if residual.shape != direction.shape or residual.ndim != 3 or residual.shape[1:] != (2, 256):
        raise ValueError('Matching [batch, 2, 256] residual and direction required')
    axes = (1, 2)
    denominator = direction.square().sum(axes, keepdim=True)
    inner = (residual * direction).sum(axes, keepdim=True)
    coefficient = torch.where(denominator > 1e-20, inner / denominator.clamp_min(1e-20), 0.)
    parallel = coefficient * direction
    return parallel, residual - parallel


class OrthogonalRecovery(nn.Module):
    """Trainable source-IQ-only correction around an externally frozen DURA.

    The learned branch can add a physical innovation step and an independent
    waveform correction.  Orthogonalization is per packet in real I/Q space;
    it does not use receiver or identity labels, clean IQ, or other packets.
    The two zero-initialized heads reproduce the DURA input exactly.
    """

    def __init__(self, width=48):
        super().__init__()
        self.entry = nn.Conv1d(7, width, 7, padding=3)
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 5, padding=2*d, dilation=d)
                                    for d in (1, 2, 4, 8, 16))
        self.wave = nn.Conv1d(width, 2, 1)
        self.step = nn.Linear(width, 1)
        nn.init.zeros_(self.wave.weight)
        nn.init.zeros_(self.wave.bias)
        nn.init.zeros_(self.step.weight)
        nn.init.zeros_(self.step.bias)

    def forward(self, received, anchor, innovation, variance):
        if (received.ndim != 3 or received.shape[1:] != (2, 256)
                or anchor.shape != received.shape or innovation.shape != received.shape
                or variance.shape != (len(received),)):
            raise ValueError('Matching current IQ windows and one noise variance required')
        if (not all(torch.isfinite(x).all() for x in (received, anchor, innovation, variance))
                or (variance <= 0).any()):
            raise ValueError('Finite IQ and positive noise variance required')
        scale = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        direction_rms = innovation.square().mean((1, 2), keepdim=True).sqrt()
        direction = innovation / direction_rms.clamp_min(1e-12) * scale
        noise = ((variance.log() - 2*scale[:, 0, 0].log()).clamp(-16, 8)/8)[:, None, None]
        inputs = torch.cat((received / scale, anchor / scale, direction / scale,
                            noise.expand(-1, 1, 256)), dim=1)
        hidden = F.silu(self.entry(inputs))
        for layer in self.layers:
            hidden = hidden + .2 * F.silu(layer(hidden))
        waveform = .5 * scale * self.wave(hidden).tanh()
        _, orthogonal = split_direction(waveform, direction)
        step = .5 * self.step(hidden.mean(-1)).tanh()[:, :, None]
        return anchor + orthogonal + step * direction
