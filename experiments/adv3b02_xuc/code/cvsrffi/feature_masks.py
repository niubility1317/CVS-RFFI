from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=dim, eps=float(eps))


class FeatureMaskRouter(nn.Module):
    """Route a shared feature into TX, RX/domain, and interaction subspaces.

    The router is intentionally auxiliary-first: it returns masked features and
    regularizers, but it does not replace the existing classifier path by
    default.
    """

    def __init__(
        self,
        feat_dim: int,
        *,
        tx_ratio: float = 0.65,
        rx_ratio: float = 0.25,
        int_ratio: float = 0.10,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        if self.feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        self.tx_ratio = float(tx_ratio)
        self.rx_ratio = float(rx_ratio)
        self.int_ratio = float(int_ratio)
        self.temperature = float(temperature)
        init = self._initial_logits()
        self.logits = nn.Parameter(init)

    def _initial_logits(self) -> torch.Tensor:
        ratios = torch.tensor([self.tx_ratio, self.rx_ratio, self.int_ratio], dtype=torch.float32)
        ratios = torch.clamp(ratios, min=1e-4)
        ratios = ratios / ratios.sum().clamp_min(1e-6)
        return ratios.log().view(3, 1).repeat(1, self.feat_dim)

    def current_masks(self) -> Dict[str, torch.Tensor]:
        temp = max(1e-4, float(self.temperature))
        masks = torch.softmax(self.logits / temp, dim=0)
        return {"tx": masks[0], "rx": masks[1], "int": masks[2]}

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if h.ndim != 2:
            raise ValueError("h must have shape [N, D]")
        if h.size(1) != self.feat_dim:
            raise ValueError(f"feature dim mismatch: expected {self.feat_dim}, got {h.size(1)}")
        h = torch.nan_to_num(h.float(), nan=0.0, posinf=0.0, neginf=0.0)
        masks = self.current_masks()
        z_tx = _safe_normalize(h * masks["tx"].view(1, -1), dim=1)
        z_rx = _safe_normalize(h * masks["rx"].view(1, -1), dim=1)
        z_int = _safe_normalize(h * masks["int"].view(1, -1), dim=1)
        return z_tx, z_rx, z_int, masks

    def mask_regularization(
        self,
        *,
        lambda_overlap: float = 1.0,
        lambda_cover: float = 1.0,
        lambda_binary: float = 1.0,
        lambda_balance: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        masks = self.current_masks()
        tx = masks["tx"]
        rx = masks["rx"]
        inter = masks["int"]
        overlap = (tx * rx).abs().mean() + (tx * inter).abs().mean() + (rx * inter).abs().mean()
        cover = (tx + rx + inter - 1.0).abs().mean()
        binary = (tx * (1.0 - tx)).mean() + (rx * (1.0 - rx)).mean() + (inter * (1.0 - inter)).mean()
        target = torch.tensor(
            [self.tx_ratio, self.rx_ratio, self.int_ratio],
            device=tx.device,
            dtype=tx.dtype,
        )
        target = target / target.sum().clamp_min(1e-6)
        means = torch.stack([tx.mean(), rx.mean(), inter.mean()])
        balance = (means - target).pow(2).mean()
        loss = (
            float(lambda_overlap) * overlap
            + float(lambda_cover) * cover
            + float(lambda_binary) * binary
            + float(lambda_balance) * balance
        )
        return loss, {
            "overlap": float(overlap.detach().item()),
            "cover": float(cover.detach().item()),
            "binary": float(binary.detach().item()),
            "balance": float(balance.detach().item()),
            "tx_mean": float(tx.detach().mean().item()),
            "rx_mean": float(rx.detach().mean().item()),
            "int_mean": float(inter.detach().mean().item()),
        }
