"""LoRa preamble features and balanced sampling for the repaired Tri recipe."""
import numpy as np
import torch
from torch import nn

from .physical import ChirpResponse


def complete_labeled_batches(y, rx, rng):
    """Cover every class/receiver cell exactly once, with four RXs per class."""
    cells = [[np.flatnonzero((y == c) & (rx == r)) for r in range(5)] for c in range(10)]
    sizes = {len(cell) for group in cells for cell in group}
    if len(sizes) != 1 or min(sizes) < 16 or min(sizes) % 16:
        raise ValueError('LoRa balanced batches require equal class/RX cells divisible by 16')
    size = sizes.pop()
    cells = [[rng.permutation(cell) for cell in group] for group in cells]
    for used in range(0, size, 16):
        # Five pairs of classes take turns being absent. In four of the five
        # batches each class advances to the next omitted receiver.
        classes = rng.permutation(10).reshape(5, 2)
        orders = np.stack([rng.permutation(5) for _ in range(10)])
        cursor = np.full((10, 5), used, dtype=np.int64)
        turns = np.zeros(10, dtype=np.int64)
        for step in range(25):
            banned = step % 5
            chosen = classes[np.arange(5) != banned].reshape(-1)
            batch = []
            for c in chosen:
                dropped = orders[c, turns[c] % 5]
                turns[c] += 1
                for r in range(5):
                    if r != dropped:
                        batch.append(cells[c][r][cursor[c, r]])
                        cursor[c, r] += 1
            yield np.asarray(batch, dtype=np.int64)


def packet_phase_curvature(gain):
    phase=gain/gain.abs().clamp_min(1e-5)
    # A constant phase and a common linear phase slope both cancel.
    variation=phase[:,2:]*phase[:,:-2]*phase[:,1:-1].conj().square()
    return torch.cat((variation.real,variation.imag),dim=1)


class AlignedTrace(nn.Module):
    """Remove the common dechirped tone and retain residual amplitude and phase."""

    def __init__(self, *, samples=256):
        super().__init__()
        self.reference = ChirpResponse(samples=samples)
        self.register_buffer('time', torch.arange(samples, dtype=torch.float32), persistent=False)

    def forward(self, x, *, with_drift=False, with_phase=False):
        z = torch.complex(x[:, :, 0], x[:, :, 1])
        baseband = z * self.reference.chirp.conj()
        length = 8*self.time.numel()
        k = torch.fft.fft(baseband, n=length).abs().argmax(-1)
        k = (k + length // 2) % length - length // 2
        coarse = 2 * torch.pi * k[..., None] * self.time / length
        w = baseband * torch.polar(torch.ones_like(coarse), -coarse)
        lag = 3*self.time.numel()//32
        slope = (w[..., lag:] * w[..., :-lag].conj()).mean(-1).angle() / lag
        slope = slope.clamp(-torch.pi / length, torch.pi / length)
        fine = slope[..., None] * self.time
        w = w * torch.polar(torch.ones_like(fine), -fine)
        dc = w.mean(-1, keepdim=True)
        w = w * (dc.conj() / dc.abs().clamp_min(1e-5))
        rms = z.abs().square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
        w = w / rms
        trace = torch.stack((w.real, w.imag, z.abs() / rms), dim=2)
        if with_phase:
            return trace,packet_phase_curvature(dc.squeeze(-1))
        if with_drift:
            # Only variation between chirps enters the model. The common
            # carrier offset, including receiver oscillator bias, is removed.
            rate = self.reference.sample_rate
            tone = k * (rate / length) + slope * (rate / (2 * torch.pi))
            drift = tone - tone.mean(dim=1, keepdim=True)
            return trace, drift
        return trace


class CoherentPacket(nn.Module):
    """Add a common-tone-aligned residual branch to the existing PAMF packet F."""

    def __init__(self, raw):
        super().__init__()
        self.raw = raw
        self.trace = AlignedTrace()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 24, 9, stride=2, padding=4), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv1d(24, 40, 7, stride=2, padding=3), nn.GroupNorm(5, 40), nn.SiLU(),
            nn.Conv1d(40, 48, 5, stride=2, padding=2), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(384, 160), nn.LayerNorm(160), nn.SiLU(),
        )
        self.head = nn.Linear(160, 10)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        logits, fused = self.raw(x)
        traces = self.trace(x)
        features = self.encoder(traces.flatten(0, 1)).reshape(len(x), x.shape[1], -1).mean(1)
        return logits + self.head(features), fused + .10 * features
