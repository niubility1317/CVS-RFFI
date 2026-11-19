"""Matched supervision on a full 11,000-point SatIQ waveform."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import build_model
from .wide_model import FEATURES, RawSymbolEncoder


class PaperPAMF(nn.Module):
    def __init__(self, runtime, seed, classes):
        super().__init__()
        self.native = build_model(runtime, seed, 1)
        self.native.core.id_backbone.cls_head.requires_grad_(False)
        self.original_rate = RawSymbolEncoder(symbol_count=11, bounded=False)
        self.head = nn.Linear(2 * FEATURES, classes, bias=False)
        nn.init.xavier_uniform_(self.head.weight)
        bins = torch.arange(129).float()
        cutoff = torch.cos(torch.pi * ((bins - 60) / 20).clamp(0, 1)) * .5 + .5
        self.register_buffer("cutoff", cutoff, persistent=False)

    def embed(self, iq):
        # Keep both paths comparable in norm. The older residual sum became
        # dominated by the original-rate path at later checkpoints.
        centered = iq - iq.mean(dim=(1, 2), keepdim=True)
        rms = centered.square().sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        raw = centered / rms
        common = raw[:, :, :8000]
        spectrum = torch.fft.rfft(common, dim=-1)[..., :129]
        coarse = torch.fft.irfft(spectrum * self.cutoff, n=256, dim=-1) * (256 / 8000)
        native = F.normalize(self.native.features(coarse), dim=1)
        feature = F.normalize(self.original_rate(raw), dim=1)
        return F.normalize(torch.cat((native, feature), dim=1), dim=1)

    def forward(self, iq):
        embedding = self.embed(iq)
        logits = 30.0 * F.linear(embedding, F.normalize(self.head.weight, dim=1))
        return embedding, logits
