"""Predict per-record widely-linear FIR corrections without clean IQ at inference."""
import torch
from torch import nn
from torch.nn import functional as F


def apply_filter_delta(anchor, coefficients):
    """Apply direct and conjugate complex FIR coefficients to one IQ window."""
    if anchor.ndim != 3 or anchor.shape[1:] != (2, 256) or coefficients.ndim != 4:
        raise ValueError('Aligned IQ and direct/conjugate complex FIR coefficients required')
    n, branch, taps, component = coefficients.shape
    if n != len(anchor) or branch != 2 or component != 2 or taps < 1 or taps % 2 != 1:
        raise ValueError('One odd-length complex filter pair per IQ window required')
    x = torch.complex(anchor[:, 0], anchor[:, 1])
    windows = F.pad(x, (taps//2, taps//2)).unfold(1, taps, 1)
    w = torch.complex(coefficients[..., 0], coefficients[..., 1])
    delta = (windows*w[:, 0, None, :] + windows.conj()*w[:, 1, None, :]).sum(-1)
    return torch.stack((delta.real, delta.imag), dim=1)


def project_delta(anchor, candidate, *, epsilon):
    if anchor.shape != candidate.shape or anchor.ndim != 3 or not 0 < epsilon <= 1:
        raise ValueError('Aligned IQ and positive relative radius required')
    delta = candidate-anchor
    radius = epsilon*anchor.flatten(1).norm(dim=1)
    magnitude = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
    return anchor+delta*(radius/magnitude).clamp(max=1)[:, None, None]


class FilterStudent(nn.Module):
    """Infer bounded per-window FIR coefficients from received and anchor IQ."""
    def __init__(self, *, feature_dim=160, classes=6, width=64, layers=3, heads=4, taps=9, epsilon=.2):
        super().__init__()
        if type(taps) is not int or taps < 1 or taps % 2 != 1 or not 0 < epsilon <= 1:
            raise ValueError('Odd tap count and positive relative radius required')
        self.feature_dim, self.classes, self.taps, self.epsilon = feature_dim, classes, taps, float(epsilon)
        self.input = nn.Conv1d(4, width, 4, stride=4)
        self.position = nn.Parameter(torch.randn(1, 64, width)*.01)
        self.condition = nn.Sequential(nn.Linear(feature_dim+classes, width*2), nn.SiLU(), nn.Linear(width*2, width))
        block = nn.TransformerEncoderLayer(width, heads, width*4, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, 4*taps)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, received, anchor, features, logits):
        n = len(received)
        if received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape:
            raise ValueError('Aligned received and anchor IQ windows required')
        if features.shape != (n, self.feature_dim) or logits.shape != (n, self.classes):
            raise ValueError('One frozen teacher feature and score vector per record required')
        context = torch.cat((F.normalize(features.detach(), dim=1), F.softmax(logits.detach(), dim=1)), dim=1)
        h = self.input(torch.cat((anchor, received), dim=1)).transpose(1, 2)
        h = self.encoder(h+self.position+self.condition(context)[:, None, :])
        coeff = self.output(h.mean(1)).reshape(n, 2, self.taps, 2)
        return project_delta(anchor, anchor+apply_filter_delta(anchor, coeff), epsilon=self.epsilon)
