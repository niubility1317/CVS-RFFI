"""Single-record time-frequency recovery without a DURA dependency."""
import torch
from torch import nn

WIDTH = 384
LAYERS = 8
HEADS = 8
TOKENS = 64
SCALE = 2.0


class LargeDirectTimeFreq(nn.Module):
    """Joint attention over full-window time and frequency tokens."""
    def __init__(self, width=WIDTH, layers=LAYERS):
        super().__init__()
        if width < HEADS or width % HEADS or layers < 1:
            raise ValueError('Width must be a positive multiple of attention heads')
        def stem():
            return nn.Sequential(nn.Conv1d(2, 96, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(96, 96, 5, padding=2), nn.GELU())
        self.time_stem = stem()
        self.freq_stem = stem()
        self.time_down = nn.Conv1d(96, width, 9, stride=4, padding=4)
        self.freq_down = nn.Conv1d(96, width, 9, stride=4, padding=4)
        self.position = nn.Parameter(torch.zeros(1, 2*TOKENS, width))
        layer = nn.TransformerEncoderLayer(width, HEADS, 4*width, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.decode = nn.Sequential(nn.Conv1d(width+96, 192, 5, padding=2), nn.GELU(),
                                    nn.Conv1d(192, 96, 5, padding=2), nn.GELU())
        self.out = nn.Conv1d(96, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, received, anchor=None, innovation=None, variance=None, *, validate=True):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Received IQ windows required')
        if validate and not torch.isfinite(received).all():
            raise ValueError('Finite received IQ required')
        scale = received.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        normalized = received/scale
        z = torch.complex(normalized[:, 0], normalized[:, 1])
        spectrum = torch.fft.fft(z, dim=-1, norm='ortho')
        frequency = torch.stack((spectrum.real, spectrum.imag), 1)
        local = self.time_stem(normalized)
        freq = self.freq_stem(frequency)
        time_tokens = self.time_down(local).transpose(1, 2)
        freq_tokens = self.freq_down(freq).transpose(1, 2)
        tokens = torch.cat((time_tokens, freq_tokens), 1) + self.position
        context = self.transformer(tokens)[:, :TOKENS].transpose(1, 2).repeat_interleave(4, dim=-1)
        correction = self.out(self.decode(torch.cat((local, context), 1)))
        correction = correction*torch.rsqrt(1+correction.square().sum(1, keepdim=True))*scale*SCALE
        return received + correction
