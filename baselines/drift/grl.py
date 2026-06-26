from __future__ import annotations

import math

import torch
from torch import nn


class _DRIFTGradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float):
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coeff * grad_output, None


def gradient_reverse(x: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    return _DRIFTGradientReverse.apply(x, float(coeff))


class DRIFTGradientReversalLayer(nn.Module):
    def __init__(self, coeff: float = 1.0):
        super().__init__()
        self.coeff = float(coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gradient_reverse(x, self.coeff)


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """Compatibility option for ablations; not the DRIFT paper default."""

    p = min(max(float(progress), 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-float(gamma) * p)) - 1.0
