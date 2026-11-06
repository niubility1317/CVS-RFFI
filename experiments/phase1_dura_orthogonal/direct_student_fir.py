"""Received-IQ-conditioned complex FIR refinement of a frozen source student."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.direct_spectral_student import DirectSpectralStudent

SOURCE = '20261106-phase1-direct-spectral-wide-distill-source-s2026092701-r01'
SOURCE_COMMIT = '18f348500c4db11a7538cb61f6971cf54a41379e'
TAPS = 17
SCALE = .5


class DirectStudentFIR(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.student = DirectSpectralStudent(width=width, taps=9).eval().requires_grad_(False)
        self.features = nn.Sequential(nn.Conv1d(4, width, 9, padding=4), nn.GELU(),
                                      nn.Conv1d(width, width, 9, padding=4), nn.GELU(),
                                      nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.tap_head = nn.Linear(width, 2*TAPS)
        nn.init.zeros_(self.tap_head.weight)
        nn.init.zeros_(self.tap_head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.student.eval()
        return self

    def refine(self, received, base):
        scale = base.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        taps = SCALE*torch.tanh(self.tap_head(self.features(torch.cat((received/scale, base/scale), 1))))
        tr, ti = taps[:, :TAPS, None], taps[:, TAPS:, None]
        windows = F.pad(base/scale, (TAPS//2, TAPS//2)).unfold(2, TAPS, 1)
        real, imag = windows[:, 0].transpose(1, 2), windows[:, 1].transpose(1, 2)
        delta = torch.stack(((real*tr-imag*ti).sum(1), (real*ti+imag*tr).sum(1)), 1)
        return base + scale*delta

    def forward(self, received, anchor=None, innovation=None, variance=None, *, validate=True):
        with torch.no_grad():
            base = self.student(received, validate=validate)
        return self.refine(received, base)
