from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrInfo = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


class TxPreservingRFFIMixStyle(nn.Module):
    """TX-preserving MixStyle plugin for RFFI domain generalization.

    The module assumes a disentangled feature input:
      z_tx  : transmitter identity feature, kept untouched
      z_rcn : receiver/channel/noise mixed feature, style-randomized

    It mixes only z_rcn statistics, and by default pairs samples from the same
    transmitter but different domains, e.g. same TX + different rx_day. This is
    intentionally more conservative than generic MixStyle because low-level
    RFFI statistics may contain real transmitter fingerprints.
    """

    def __init__(
        self,
        p: float = 0.3,
        alpha: float = 0.1,
        eps: float = 1e-6,
        strength: float = 0.35,
        groups: int = 8,
        require_same_tx: bool = True,
        require_diff_domain: bool = True,
        fallback: str = "skip",
    ):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.strength = float(strength)
        self.groups = int(groups)
        self.require_same_tx = bool(require_same_tx)
        self.require_diff_domain = bool(require_diff_domain)
        self.fallback = str(fallback).lower().strip()
        if self.fallback not in ("skip", "random"):
            raise ValueError("fallback must be 'skip' or 'random'")

    @staticmethod
    def _as_label_vector(labels: Optional[torch.Tensor], batch_size: int, device) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        labels = labels.to(device=device).view(-1)
        if labels.numel() != batch_size:
            return None
        return labels

    @staticmethod
    def _choose_groups(feature_dim: int, requested_groups: int) -> int:
        requested_groups = max(1, min(int(requested_groups), int(feature_dim)))
        for g in range(requested_groups, 0, -1):
            if feature_dim % g == 0:
                return g
        return 1

    def _sample_lambda(self, batch_size: int, shape_rank: int, device, dtype: torch.dtype) -> torch.Tensor:
        shape = (batch_size,) + (1,) * (shape_rank - 1)
        a = torch.full(shape, self.alpha, device=device, dtype=torch.float32)
        beta = torch.distributions.Beta(a, a)
        return beta.sample().to(device=device, dtype=dtype).clamp(0.0, 1.0)

    def _safe_partner_perm(
        self,
        y_tx: Optional[torch.Tensor],
        domain_labels: Optional[torch.Tensor],
        batch_size: int,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.arange(batch_size, device=device)
        perm = idx.clone()
        valid = torch.zeros(batch_size, device=device, dtype=torch.bool)

        y_tx = self._as_label_vector(y_tx, batch_size, device)
        domain_labels = self._as_label_vector(domain_labels, batch_size, device)

        if batch_size <= 1:
            return perm, valid

        random_perm = torch.randperm(batch_size, device=device)
        for i in range(batch_size):
            mask = idx != i
            if self.require_same_tx:
                if y_tx is None:
                    mask = torch.zeros_like(mask)
                else:
                    mask = mask & (y_tx == y_tx[i])
            if self.require_diff_domain:
                if domain_labels is None:
                    mask = torch.zeros_like(mask)
                else:
                    mask = mask & (domain_labels != domain_labels[i])

            candidates = idx[mask]
            if candidates.numel() > 0:
                j = int(torch.randint(candidates.numel(), (1,), device=device).item())
                perm[i] = candidates[j]
                valid[i] = True
            elif self.fallback == "random":
                perm[i] = random_perm[i]
                valid[i] = perm[i] != i

        return perm, valid

    def _normalize_and_mix_2d(
        self,
        z_rcn: torch.Tensor,
        perm: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, dim = z_rcn.shape
        groups = self._choose_groups(dim, self.groups)
        grouped = z_rcn.view(bsz, groups, dim // groups)

        mu = grouped.mean(dim=2, keepdim=True).detach()
        var = (grouped - mu).pow(2).mean(dim=2, keepdim=True)
        sigma = torch.sqrt(var + self.eps).detach()
        normed = (grouped - mu) / sigma.clamp_min(self.eps)

        lam = self._sample_lambda(bsz, grouped.dim(), z_rcn.device, z_rcn.dtype)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sigma_mix = lam * sigma + (1.0 - lam) * sigma[perm]
        mixed = (normed * sigma_mix + mu_mix).view_as(z_rcn)

        valid_view = valid.view(bsz, 1)
        mixed = z_rcn + self.strength * (mixed - z_rcn)
        mixed = torch.where(valid_view, mixed, z_rcn)
        return mixed, lam.view(bsz, -1).mean(dim=1)

    def _normalize_and_mix_3d(
        self,
        z_rcn: torch.Tensor,
        perm: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = z_rcn.size(0)
        mu = z_rcn.mean(dim=2, keepdim=True).detach()
        var = (z_rcn - mu).pow(2).mean(dim=2, keepdim=True)
        sigma = torch.sqrt(var + self.eps).detach()
        normed = (z_rcn - mu) / sigma.clamp_min(self.eps)

        lam = self._sample_lambda(bsz, z_rcn.dim(), z_rcn.device, z_rcn.dtype)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sigma_mix = lam * sigma + (1.0 - lam) * sigma[perm]
        mixed = normed * sigma_mix + mu_mix

        valid_view = valid.view(bsz, 1, 1)
        mixed = z_rcn + self.strength * (mixed - z_rcn)
        mixed = torch.where(valid_view, mixed, z_rcn)
        return mixed, lam.view(bsz, -1).mean(dim=1)

    def forward(
        self,
        z_tx: torch.Tensor,
        z_rcn: torch.Tensor,
        y_tx: Optional[torch.Tensor],
        domain_labels: Optional[torch.Tensor],
        return_info: bool = False,
    ) -> TensorOrInfo:
        if z_tx.size(0) != z_rcn.size(0):
            raise ValueError("z_tx and z_rcn must have the same batch size")

        bsz = z_rcn.size(0)
        empty_info = {
            "applied": torch.zeros((), device=z_rcn.device, dtype=torch.bool),
            "valid_mask": torch.zeros(bsz, device=z_rcn.device, dtype=torch.bool),
            "valid_ratio": torch.zeros((), device=z_rcn.device, dtype=z_rcn.dtype),
        }

        if (not self.training) or self.p <= 0.0 or self.alpha <= 0.0 or self.strength <= 0.0:
            return (z_rcn, empty_info) if return_info else z_rcn
        if bsz <= 1 or z_rcn.dim() not in (2, 3):
            return (z_rcn, empty_info) if return_info else z_rcn
        if torch.rand((), device=z_rcn.device) > self.p:
            return (z_rcn, empty_info) if return_info else z_rcn

        perm, valid = self._safe_partner_perm(y_tx, domain_labels, bsz, z_rcn.device)
        if not bool(valid.any()):
            return (z_rcn, empty_info) if return_info else z_rcn

        if z_rcn.dim() == 2:
            mixed, lam = self._normalize_and_mix_2d(z_rcn, perm, valid)
        else:
            mixed, lam = self._normalize_and_mix_3d(z_rcn, perm, valid)

        info = {
            "applied": torch.ones((), device=z_rcn.device, dtype=torch.bool),
            "perm": perm,
            "lambda": lam,
            "valid_mask": valid,
            "valid_ratio": valid.to(dtype=z_rcn.dtype).mean(),
        }
        return (mixed, info) if return_info else mixed


class TxRcnGatedFusion(nn.Module):
    """Gated residual fusion for TX and receiver/channel/noise features."""

    def __init__(
        self,
        tx_dim: int,
        rcn_dim: int,
        out_dim: Optional[int] = None,
        beta: float = 0.3,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_dim = int(out_dim or tx_dim)
        self.beta = float(beta)
        self.tx_proj = nn.Identity() if tx_dim == out_dim else nn.Linear(tx_dim, out_dim)
        self.rcn_proj = nn.Linear(rcn_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(tx_dim + rcn_dim, out_dim),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, z_tx: torch.Tensor, z_rcn: torch.Tensor) -> torch.Tensor:
        if z_tx.dim() != 2 or z_rcn.dim() != 2:
            raise ValueError("TxRcnGatedFusion expects 2D features shaped [B, D]")
        gate = self.gate(torch.cat([z_tx, z_rcn], dim=1))
        tx = self.tx_proj(z_tx)
        rcn = self.drop(self.rcn_proj(z_rcn))
        return self.norm(tx + self.beta * gate * rcn)


def rffi_mixstyle_consistency_loss(
    logits_clean: torch.Tensor,
    logits_mixed: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL consistency from clean logits to TP-RFFI-MixStyle logits."""

    t = max(float(temperature), 1e-6)
    target = F.softmax(logits_clean.detach() / t, dim=1)
    log_prob = F.log_softmax(logits_mixed / t, dim=1)
    return F.kl_div(log_prob, target, reduction="batchmean") * (t * t)


__all__ = [
    "TxPreservingRFFIMixStyle",
    "TxRcnGatedFusion",
    "rffi_mixstyle_consistency_loss",
]
