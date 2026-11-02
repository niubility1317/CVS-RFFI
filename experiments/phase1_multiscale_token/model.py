"""Input-only multiscale complex spectral filtering with axial attention."""
import torch
from torch import nn


class MultiScaleToken(nn.Module):
    def __init__(self):
        super().__init__()
        width = 32
        self.entry = nn.Linear(2, width)
        self.time_position = nn.Parameter(torch.zeros(1, 32, 1, width))
        self.frequency_position = nn.Parameter(torch.zeros(1, 1, 32, width))
        self.frequency_norm = nn.LayerNorm(width)
        self.frequency_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.time_norm = nn.LayerNorm(width)
        self.time_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.feed_norm = nn.LayerNorm(width)
        self.feed = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, width))
        self.gate_output = nn.Linear(width, 2)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def forward(self, received, initial, log_variance):
        if (received.ndim != 3 or received.shape[1:] != (2, 256)
                or initial.shape != received.shape or log_variance.shape != (len(received),)):
            raise ValueError('Matching IQ windows and one blind log variance required')
        if not all(torch.isfinite(v).all() for v in (received, initial, log_variance)):
            raise FloatingPointError('Finite single-window evidence required')
        batch = len(received)
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        signal = torch.complex(received[:, 0], received[:, 1]) / scale[:, 0]
        noise = ((log_variance - power[:, 0, 0].log()).clamp(-16, 8) / 8)
        corrections = []
        for window in (8, 16, 32):
            frames = 256 // window
            spectrum = torch.fft.fft(signal.reshape(batch, frames, window), dim=-1, norm='ortho')
            magnitude = spectrum.abs().clamp_max(32).log1p()
            evidence = torch.stack((magnitude, noise[:, None, None].expand_as(magnitude)), dim=-1)
            h = self.entry(evidence) + self.time_position[:, :frames] + self.frequency_position[:, :, :window]
            f = self.frequency_norm(h).reshape(batch * frames, window, -1)
            f, _ = self.frequency_attention(f, f, f, need_weights=False)
            h = h + f.reshape(batch, frames, window, -1)
            t = self.time_norm(h).transpose(1, 2).reshape(batch * window, frames, -1)
            t, _ = self.time_attention(t, t, t, need_weights=False)
            h = h + t.reshape(batch, window, frames, -1).transpose(1, 2)
            h = h + self.feed(self.feed_norm(h))
            weight = self.gate_output(h).tanh()
            correction = spectrum * torch.complex(weight[..., 0], weight[..., 1])
            corrections.append(torch.fft.ifft(correction, dim=-1, norm='ortho').reshape(batch, 256))
        delta = sum(corrections) / len(corrections)
        return initial + 0.25 * torch.stack((delta.real, delta.imag), dim=1) * scale
