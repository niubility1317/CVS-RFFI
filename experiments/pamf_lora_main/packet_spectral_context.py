"""Joint LoRa packet context across chirps and contiguous complex spectral patches."""

import torch
from torch import nn

from .channel_blind import zero_carrier
from .spectral_covariance import SharedSpectrumRecovery
from .weak_signal_probe import observed_noise_rms


PATCH = 16
WIDTH = 64


class SpectralAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 2 * WIDTH),
                                 nn.GELU(), nn.Linear(2 * WIDTH, WIDTH))

    def forward(self, tokens):
        normalized = self.attention_norm(tokens)
        attention, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attention
        return tokens + self.mlp(tokens)


class SpectralPatchMixer(nn.Module):
    """Keep the patch encoding identical while changing the attention extent."""

    def __init__(self, *, packet):
        super().__init__()
        self.packet = packet
        self.projection = nn.Linear(8 * PATCH, WIDTH)
        self.frequency_position = nn.Parameter(torch.zeros(1, 1, 256 // PATCH, WIDTH))
        self.chirp_position = nn.Parameter(torch.zeros(1, 6, 1, WIDTH))
        self.blocks = nn.Sequential(SpectralAttention(), SpectralAttention())
        self.readout = nn.Linear(WIDTH, 2 * PATCH)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def mix(self, fields):
        b, c, n, f = fields.shape
        tokens = self.projection(fields.reshape(b, c, n // PATCH, PATCH * f))
        tokens = tokens + self.frequency_position + self.chirp_position
        context = tokens.flatten(1, 2) if self.packet else tokens.flatten(0, 1)
        return self.blocks(context).reshape_as(tokens)

    def forward(self, fields):
        return self.readout(self.mix(fields)).reshape(*fields.shape[:3], 2)


class PacketSpectralContext(SharedSpectrumRecovery):
    """Refine the common physical baseline with observed complex spectral content."""

    def __init__(self, widths, *, packet):
        super().__init__(widths, correlated=True)
        self.spectral_mixer = SpectralPatchMixer(packet=packet)

    def forward(self, observed, quality, rotation):
        coarse, refined = super().forward(observed, quality, rotation)
        variance = observed_noise_rms(observed).square()
        scale = (.02 + variance).sqrt()
        inputs = [torch.fft.fftshift(torch.fft.fft(zero_carrier(v), norm='ortho'), dim=-1)/scale
                  for v in (observed, coarse, refined)]
        shape = observed.real.shape
        signals = [feature for z in inputs for feature in (z.real, z.imag)]
        fields = torch.stack((*signals, quality.expand(shape), variance.log1p().expand(shape)), dim=-1)
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            delta = self.spectral_mixer(fields)
        delta = delta.float() if mixed else delta
        correction = torch.complex(delta[..., 0], delta[..., 1])
        correction = correction * (variance / (.10 + variance)).sqrt()
        correction = correction * (1 + correction.abs().square()).rsqrt()
        timed = torch.fft.ifft(torch.fft.ifftshift(correction, dim=-1), norm='ortho')
        return coarse, refined + zero_carrier(timed)
