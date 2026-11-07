"""DURA correction with a learned, monotone, per-record radial amplitude warp."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral

SEGMENTS = 8
SEGMENT_WIDTH = .5
ENVELOPE_WEIGHT = .5
QUANTILES = 16


def magnitude(iq):
    return iq.square().sum(1, keepdim=True).clamp_min(1e-12).sqrt()


def normalize_magnitude(iq):
    amp = magnitude(iq)
    return amp / amp.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


def monotone_factor(radius, logits):
    """Integral of positive piecewise slopes, expressed as a residual multiplier."""
    if radius.ndim != 3 or radius.shape[1] != 1 or logits.shape != (len(radius), SEGMENTS):
        raise ValueError('Matching radius and eight slope batches required')
    starts = torch.arange(SEGMENTS, device=radius.device, dtype=radius.dtype)[None, :, None] * SEGMENT_WIDTH
    distance = (radius-starts).clamp_min(0.)
    pieces = torch.cat((distance[:, :-1].clamp_max(SEGMENT_WIDTH), distance[:, -1:]), dim=1)
    slopes = logits.clamp(-1.5, 1.5).exp()
    delta = (pieces * (slopes[:, :, None]-1.)).sum(1, keepdim=True)
    return 1. + delta/radius.clamp_min(1e-12)


def envelope_loss(output, clean):
    if output.shape != clean.shape or output.ndim != 3 or output.shape[1:] != (2, 256):
        raise ValueError('Matching 256-sample IQ required')
    positions = torch.linspace(8, 247, QUANTILES, device=output.device).long()
    a = normalize_magnitude(output).sort(dim=-1).values[..., positions]
    b = normalize_magnitude(clean).sort(dim=-1).values[..., positions]
    return (a-b).square().mean()


class DuraMonotoneEnvelope(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.base = DuraMultiscaleSpectral(width=width, taps=9)
        self.entry = nn.Conv1d(4, 16, 7, padding=3)
        self.hidden = nn.Conv1d(16, 16, 5, padding=2)
        self.envelope_head = nn.Linear(16, SEGMENTS)
        nn.init.zeros_(self.envelope_head.weight)
        nn.init.zeros_(self.envelope_head.bias)

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        base = self.base(received, anchor, innovation, variance, validate=validate)
        noise = (variance[:, None, None]/magnitude(anchor).square().mean(-1, keepdim=True).clamp_min(1e-12))
        features = torch.cat((normalize_magnitude(received), normalize_magnitude(base),
                              normalize_magnitude(innovation), torch.log1p(noise).clamp_max(16.).expand(-1, 1, 256)/16.), 1)
        h = F.silu(self.entry(features))
        logits = self.envelope_head(F.silu(self.hidden(h)).mean(-1))
        return base * monotone_factor(normalize_magnitude(base), logits)
