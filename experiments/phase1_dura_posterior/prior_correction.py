"""A phase-equivariant correction for one DURA prior, initialized to zero."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PosteriorPriorCorrection(nn.Module):
    """Use an independently estimated posterior without changing the original prior at initialization.

    The caller supplies only waveforms and an estimated log noise variance.  This
    module neither estimates the posterior nor invokes the DURA data-consistency
    step; those remain explicit parts of the surrounding experiment.
    """

    def __init__(self, width=32):
        super().__init__()
        self.entry = nn.Conv1d(11, width, 7, padding=3)
        self.layers = nn.ModuleList(
            nn.Conv1d(width, width, 5, padding=2*d, dilation=d)
            for d in (1, 2, 4, 8)
        )
        self.coefficients = nn.Conv1d(width, 6, 1)
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)

    def forward(self, received, current, prior, posterior, log_variance):
        shape = received.shape
        if (received.ndim != 3 or shape[1:] != (2, 256)
                or any(x.shape != shape for x in (current, prior, posterior))
                or log_variance.shape != (shape[0],)):
            raise ValueError('Four aligned IQ windows and one log variance per window are required')

        # All conditioning channels are invariant to a common IQ phase rotation.
        streams = torch.stack((received-prior, current-prior, posterior-prior), dim=1)
        power = prior.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        reference = prior / scale
        normalized = streams / scale[:, None]
        rr, ri = reference[:, 0:1], reference[:, 1:2]
        sr, si = normalized[:, :, 0], normalized[:, :, 1]
        magnitudes = torch.cat(((reference.square().sum(1, keepdim=True)+1e-12).sqrt(),
                                (sr.square()+si.square()+1e-12).sqrt()), dim=1)
        dot = sr*rr + si*ri
        cross = si*rr - sr*ri
        relative_noise = (log_variance - power[:, 0, 0].log()).clamp(-16, 8)
        noise = (relative_noise/8)[:, None, None].expand(-1, 1, 256)
        h = F.silu(self.entry(torch.cat((magnitudes, dot, cross, noise), dim=1)))
        for layer in self.layers:
            h = h + 0.2*F.silu(layer(h))
        weights = self.coefficients(h).tanh()
        real, imag = weights[:, :3], weights[:, 3:]
        dr, di = streams[:, :, 0], streams[:, :, 1]
        correction = torch.stack(((real*dr-imag*di).sum(1),
                                  (real*di+imag*dr).sum(1)), dim=1)
        return prior + (0.25/math.sqrt(3))*correction
