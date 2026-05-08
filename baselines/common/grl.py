from __future__ import annotations

import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity in the forward pass and `-lambda` gradient in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def gradient_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, float(lambd))


class GradientReversalLayer(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor, lambd: float | None = None) -> torch.Tensor:
        return gradient_reverse(x, self.lambd if lambd is None else float(lambd))


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """DANN schedule from 0 to 1 for adversarial domain training."""

    p = min(1.0, max(0.0, float(progress)))
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0)
