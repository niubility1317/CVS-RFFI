"""Phase-equivariant axial time-frequency correction around frozen DURA."""
import torch
from torch import nn


class SpectralTokenRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        width = 48
        self.input = nn.Linear(5, width)
        self.time_position = nn.Parameter(torch.zeros(1, 16, 1, width))
        self.frequency_position = nn.Parameter(torch.zeros(1, 1, 16, width))
        self.freq_norm = nn.LayerNorm(width)
        self.freq_attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.time_norm = nn.LayerNorm(width)
        self.time_attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.feed_norm = nn.LayerNorm(width)
        self.feed = nn.Sequential(nn.Linear(width, 96), nn.SiLU(), nn.Linear(96, width))
        self.output = nn.Linear(width, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, received, initial, log_variance):
        shape = received.shape
        if (received.ndim != 3 or shape[1:] != (2, 256) or initial.shape != shape
                or log_variance.shape != (shape[0],)):
            raise ValueError('Matching IQ windows and one blind log variance are required')
        if not all(torch.isfinite(v).all() for v in (received, initial, log_variance)):
            raise FloatingPointError('Finite received evidence is required')
        batch = shape[0]
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        z = torch.complex(received[:, 0], received[:, 1]) / scale[:, 0]
        a = torch.complex(initial[:, 0], initial[:, 1]) / scale[:, 0]
        pivot = z.gather(1, z.abs().argmax(dim=1, keepdim=True))
        unit = pivot / pivot.abs().clamp_min(1e-8)
        zs = torch.fft.fft((z * unit.conj()).reshape(batch, 16, 16), dim=-1, norm='ortho')
        aa = torch.fft.fft((a * unit.conj()).reshape(batch, 16, 16), dim=-1, norm='ortho')
        noise = ((log_variance - power[:, 0, 0].log()).clamp(-16, 8) / 8)[:, None, None].expand(-1, 16, 16)
        evidence = torch.stack((zs.real, zs.imag, aa.real, aa.imag, noise), dim=-1)
        h = self.input(evidence) + self.time_position + self.frequency_position
        f = self.freq_norm(h).reshape(batch * 16, 16, -1)
        f, _ = self.freq_attention(f, f, f, need_weights=False)
        h = h + f.reshape(batch, 16, 16, -1)
        t = self.time_norm(h).transpose(1, 2).reshape(batch * 16, 16, -1)
        t, _ = self.time_attention(t, t, t, need_weights=False)
        h = h + t.reshape(batch, 16, 16, -1).transpose(1, 2)
        h = h + self.feed(self.feed_norm(h))
        delta = self.output(h).tanh()
        spectrum = torch.complex(delta[..., 0], delta[..., 1])
        waveform = torch.fft.ifft(spectrum, dim=-1, norm='ortho').reshape(batch, 256) * unit
        correction = torch.stack((waveform.real, waveform.imag), dim=1) * scale
        return initial + 0.25 * correction
