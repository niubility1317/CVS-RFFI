"""Two observation-consistent updates around a trainable whole-packet clean projector."""

import torch
from torch import nn
from torch.nn import functional as F

from .w24_packet_unet import CENTER, block


WIDTH = 64


def zero_dc(value):
    return value - value.mean(-1, keepdim=True)


class CleanProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(block(2, 32, stride=2), block(32, 48, stride=2),
                                     block(48, WIDTH, stride=2))
        self.encode = nn.Linear(WIDTH * (CENTER // 8), WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.context = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        self.decode = nn.Linear(WIDTH, 32 * (CENTER // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose1d(32, 24, 4, stride=2, padding=1), nn.GroupNorm(8, 24), nn.SiLU(),
            nn.ConvTranspose1d(24, 16, 4, stride=2, padding=1), nn.GroupNorm(8, 16), nn.SiLU(),
            nn.Conv1d(16, 2, 5, padding=2),
        )
        nn.init.normal_(self.decoder[-1].weight, std=.002)
        nn.init.zeros_(self.decoder[-1].bias)
        frequency = torch.fft.fftfreq(CENTER)
        self.register_buffer('noise_band', frequency.abs() >= 1 / 3, persistent=False)

    def project(self, center):
        """Map non-carrier shape through a shared encoder/decoder and eight-chirp context."""
        shape = center.shape[:2]
        iq = torch.stack((center.real, center.imag), dim=-2).flatten(0, 1)
        code = self.encode(self.encoder(iq).flatten(1)).unflatten(0, shape)
        normalized = self.norm(code)
        code = code + self.context(normalized, normalized, normalized, need_weights=False)[0]
        decoded = self.decode(F.silu(code)).flatten(0, 1).unflatten(-1, (32, CENTER // 8))
        real, imag = self.decoder(decoded).float().unflatten(0, shape).unbind(-2)
        return zero_dc(torch.complex(real, imag))

    @staticmethod
    def scale(rms):
        return (rms / rms.square().mean(-2, keepdim=True).sqrt().clamp_min(1e-8)).clamp_min(1e-8)

    def project_clean(self, y, rms):
        scale = self.scale(rms)
        return y.mean(-1, keepdim=True) + self.project(zero_dc(y) * scale) / scale

    def forward(self, y, _quality, rms, _edges):
        scale = self.scale(rms)
        observed = zero_dc(y) * scale
        noise = (torch.fft.fft(observed, norm='ortho')[..., self.noise_band].abs().square()
                 .flatten(1).median(-1).values / .69314718056).clamp(1e-6, 4.)[:, None, None]
        common_weight = (noise / 8) / (noise / 8 + .004)
        private_weight = (noise * .875) / (noise * .875 + .018)

        def update(current):
            change = self.project(current) - observed
            common = change.mean(-2, keepdim=True)
            # The same observation fidelity terms anchor both unrolled passes.
            return observed + common_weight * common + private_weight * (change - common)

        first = update(observed)
        second = update(first)
        return tuple(y + 2. * (z - observed) / scale for z in (first, second))

    def propose(self, y, quality, rms, edges):
        return self(y, quality, rms, edges),
