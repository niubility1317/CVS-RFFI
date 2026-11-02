"""Fixed source-only channel augmentation; never applied to target inference."""
import math
import torch
from torch.nn import functional as F


class SourceChannelAugment:
    def __init__(self, seed=2026110419):
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.total = 0
        self.augmented = 0

    def __call__(self, received):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Source IQ windows must have shape [batch,2,256]')
        if not torch.isfinite(received).all():
            raise FloatingPointError('Finite source IQ required')
        n = len(received)
        rand = lambda *shape: torch.rand(*shape, generator=self.generator, device='cpu').to(received.device)
        selected = rand(n) < 0.75
        delay = torch.floor(rand(n) * 12).long() + 1
        echo_mag = 0.35 * rand(n)
        echo_angle = 2 * math.pi * rand(n)
        rate = (2 * rand(n) - 1) * 0.01
        snr_db = 20 + 15 * rand(n)
        padded = F.pad(received, (12, 0))
        index = torch.arange(256, device=received.device)[None, :] + 12 - delay[:, None]
        echo = torch.gather(padded, 2, index[:, None, :].expand(-1, 2, -1))
        z = torch.complex(received[:, 0], received[:, 1])
        e = torch.complex(echo[:, 0], echo[:, 1])
        coefficient = torch.polar(echo_mag, echo_angle)
        phase = rate[:, None] * torch.arange(256, device=received.device)
        disturbed = (z + coefficient[:, None] * e) * torch.polar(torch.ones_like(phase), phase)
        scale = received.square().mean((1, 2)).clamp_min(1e-12).sqrt() * (10 ** (-snr_db / 20))
        noise = torch.randn(n, 2, 256, generator=self.generator, device='cpu').to(received.device)
        mixed = torch.stack((disturbed.real, disturbed.imag), dim=1) + scale[:, None, None] * noise
        self.total += n
        self.augmented += int(selected.sum())
        return torch.where(selected[:, None, None], mixed, received)
