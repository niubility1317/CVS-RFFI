"""Full-window attention and local waveform restoration around frozen DURA."""
import torch
from torch import nn

from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate

WIDTH = 384
LAYERS = 8
HEADS = 8
TOKENS = 64
SCALE = .25


class LargeContextDura(nn.Module):
    """Learn a residual with full-window attention and a full-rate local path."""
    def __init__(self, width=WIDTH, layers=LAYERS):
        super().__init__()
        if width < HEADS or width % HEADS or layers < 1:
            raise ValueError('Width must be a positive multiple of attention heads')
        self.residual_scale = SCALE
        self.local = nn.Sequential(nn.Conv1d(7, 64, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(64, 64, 5, padding=2), nn.GELU())
        self.down = nn.Conv1d(64, width, 9, stride=4, padding=4)
        self.position = nn.Parameter(torch.zeros(1, TOKENS, width))
        layer = nn.TransformerEncoderLayer(width, HEADS, 4*width, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.global_context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.decode = nn.Sequential(nn.Conv1d(width+64, 128, 5, padding=2), nn.GELU(),
                                    nn.Conv1d(128, 64, 5, padding=2), nn.GELU())
        self.out = nn.Conv1d(64, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape
                or innovation.shape != received.shape or variance.shape != (len(received),)):
            raise ValueError('Matching IQ, DURA context and variance required')
        if validate and (not all(torch.isfinite(v).all() for v in (received, anchor, innovation, variance))
                         or (variance <= 0).any()):
            raise ValueError('Finite IQ and positive variance required')
        power = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12)
        scale = power.sqrt()
        noise = torch.log1p(variance[:, None, None].clamp_min(0.)/power)
        features = torch.cat((received/scale, anchor/scale, innovation/scale,
                              noise.expand(-1, 1, received.shape[-1])), 1)
        local = self.local(features)
        tokens = self.down(local).transpose(1, 2) + self.position
        context = self.global_context(tokens).transpose(1, 2)
        # Repetition is deterministic in CUDA backward; convolution below learns interpolation.
        context = context.repeat_interleave(4, dim=-1)
        delta = self.out(self.decode(torch.cat((local, context), 1)))
        delta = delta*torch.rsqrt(1+delta.square().sum(1, keepdim=True))*scale*self.residual_scale
        return anchor + noise_gate(anchor, variance)*delta
