"""DURA-anchored correction conditioned on phase-invariant paired-view features."""
import torch
from torch import nn
from torch.nn import functional as F


class CrossViewDURA(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv1d(12, 48, 7, padding=3)
        self.down1 = nn.Conv1d(48, 64, 5, stride=2, padding=2)
        self.down2 = nn.Conv1d(64, 96, 5, stride=2, padding=2)
        self.latent = nn.Linear(192, 64)
        self.decode = nn.Conv1d(77, 64, 5, padding=2)
        self.blocks = nn.ModuleList(nn.Conv1d(64, 64, 5, padding=2*d, dilation=d)
                                    for d in (1, 2, 4))
        self.gate_output = nn.Conv1d(64, 8, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)
        self.last_latent = None

    def forward(self, received, initial, log_variance):
        if (received.ndim != 3 or received.shape[1:] != (2, 256)
                or initial.shape != received.shape or log_variance.shape != (len(received),)):
            raise ValueError('Matching IQ windows and one blind variance required')
        if not all(torch.isfinite(v).all() for v in (received, initial, log_variance)):
            raise FloatingPointError('Finite single-window evidence required')
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        z = torch.complex(received[:, 0], received[:, 1]) / scale[:, 0]
        a = torch.complex(initial[:, 0], initial[:, 1]) / scale[:, 0]
        cross = z * a.conj()
        local = [z.abs(), a.abs(), cross.real, cross.imag]
        for signal in (z, a):
            for lag in (1, 4):
                prior = torch.cat((signal[:, :lag].flip(1), signal[:, :-lag]), dim=1)
                corr = signal * prior.conj()
                local.extend((corr.real, corr.imag))
        local = torch.stack(local, dim=1)
        h = F.silu(self.entry(local))
        h = F.silu(self.down1(h))
        h = F.silu(self.down2(h))
        summary = torch.cat((h.mean(2), (h.var(2, unbiased=False) + 1e-6).sqrt()), dim=1)
        latent = self.latent(summary)
        self.last_latent = latent
        noise = ((log_variance - power[:, 0, 0].log()).clamp(-16, 8) / 8)[:, None, None]
        features = torch.cat((local, latent[:, :, None].expand(-1, -1, 256),
                              noise.expand(-1, 1, 256)), dim=1)
        d = F.silu(self.decode(features))
        for block in self.blocks:
            d = d + 0.2 * F.silu(block(d))
        w = self.gate_output(d).tanh()
        basis = (a, z, torch.roll(a, 1, 1), torch.roll(a, -1, 1))
        correction = sum(torch.complex(w[:, 2*i], w[:, 2*i+1]) * v
                         for i, v in enumerate(basis))
        return initial + 0.25 * torch.stack((correction.real, correction.imag), dim=1) * scale
