from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    """SimCLR/NT-Xent loss for batches arranged as `[view1_batch, view2_batch]`."""

    def __init__(self, temperature: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2 or features.size(0) % 2 != 0:
            raise ValueError("NTXentLoss expects [2B,D] features.")
        n2 = features.size(0)
        n = n2 // 2
        z = F.normalize(features, dim=1, eps=self.eps)
        logits = z @ z.t() / max(self.temperature, self.eps)
        logits = logits.masked_fill(torch.eye(n2, device=features.device, dtype=torch.bool), -1e9)
        pos = torch.arange(n2, device=features.device)
        pos = torch.where(pos < n, pos + n, pos - n)
        return F.cross_entropy(logits, pos)


def siamese_contrastive_ce_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    *,
    ce_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    temperature: float = 0.05,
):
    loss_cl = NTXentLoss(temperature=temperature)(torch.cat([z1, z2], dim=0))
    loss_ce = 0.5 * (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels))
    loss = float(ce_weight) * loss_ce + float(contrastive_weight) * loss_cl
    return {"loss": loss, "loss_ce": loss_ce, "loss_cl": loss_cl}
