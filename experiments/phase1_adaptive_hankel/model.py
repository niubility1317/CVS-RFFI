"""Learned, phase-equivariant singular-spectrum correction of an IQ estimate."""

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveHankel(nn.Module):
    """Adapt singular components using the current window and blind noise estimate.

    The physical estimate and its SVD are treated as fixed source inputs: only
    the small rank-domain network is trained. No identity or template is input.
    """

    def __init__(self, window=32, width=32):
        super().__init__()
        if not 1 <= window <= 128 or width < 1:
            raise ValueError('A valid Hankel window and network width are required')
        self.window = window
        self.entry = nn.Conv1d(4, width, 5, padding=2)
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 5, padding=2*d, dilation=d) for d in (1, 2, 4))
        self.gate_output = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def forward(self, received, initial, logvar):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or initial.shape != received.shape
                or logvar.shape != (len(received),)):
            raise ValueError('Matching IQ windows and one noise estimate are required')
        if not all(torch.isfinite(t).all() for t in (received, initial, logvar)):
            raise FloatingPointError('Finite source evidence is required')
        with torch.no_grad():
            z = torch.complex(initial[:, 0], initial[:, 1])
            columns = 256-self.window+1
            matrix = z.unfold(-1, self.window, 1)
            left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
            total = singular.square().sum(1, keepdim=True).clamp_min(1e-12).sqrt()
            relative = singular/total
            rank = torch.linspace(0, 1, self.window, device=received.device, dtype=received.dtype)[None].expand(len(received), -1)
            power = initial.square().mean((1, 2)).clamp_min(1e-8)
            residual = (received-initial).square().mean((1, 2))/power
            noise = (logvar-power.log()).clamp(-16, 8)/8
            features = torch.stack((relative, rank, noise[:, None].expand_as(relative),
                                    torch.log1p(residual)[:, None].expand_as(relative)), dim=1)
        hidden = F.silu(self.entry(features))
        for layer in self.layers:
            hidden = hidden+0.2*F.silu(layer(hidden))
        adjustment = 0.25*self.gate_output(hidden).tanh()[:, 0]*singular
        delta = (left*adjustment[:, None, :])@right
        index = (torch.arange(columns, device=received.device)[:, None]
                 +torch.arange(self.window, device=received.device)[None, :]).reshape(1, -1)
        signal = torch.zeros_like(z).scatter_add_(1, index.expand(len(z), -1), delta.reshape(len(z), -1))
        count = torch.bincount(index.reshape(-1), minlength=256).to(received)
        signal = signal/count[None]
        output = initial+torch.stack((signal.real, signal.imag), dim=1)
        if not torch.isfinite(output).all():
            raise FloatingPointError('Finite adaptive Hankel output is required')
        return output
