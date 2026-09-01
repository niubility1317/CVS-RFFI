"""Permutation-invariant support-only domain state inference for MARC-OT."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SupportDomainState:
    """Frozen support-derived controls for one task before query inference."""

    q: Tensor
    uncertainty: Tensor
    block_gates: Tensor
    block_lrs: Tensor


class SupportSetEncoder(nn.Module):
    """Encode legal support statistics with a shared class-permutation-invariant rule."""

    def __init__(
        self,
        *,
        feature_dim: int,
        coefficient_dim: int,
        block_count: int,
        hidden_dim: int = 64,
        lr_min: float = 1e-4,
        lr_max: float = 1e-2,
    ) -> None:
        super().__init__()
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (feature_dim, coefficient_dim, block_count, hidden_dim)
        ):
            raise ValueError("encoder dimensions must be positive integers")
        if not math.isfinite(lr_min) or not math.isfinite(lr_max) or lr_min <= 0.0 or lr_min > lr_max:
            raise ValueError("learning-rate bounds must be finite and positive")
        self.feature_dim = feature_dim
        self.coefficient_dim = coefficient_dim
        self.block_count = block_count
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, coefficient_dim + 1 + 2 * block_count),
        )

    def _split_outputs(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        q_end = self.coefficient_dim
        uncertainty_end = q_end + 1
        gate_end = uncertainty_end + self.block_count
        return (
            raw[:q_end],
            raw[q_end],
            raw[uncertainty_end:gate_end],
            raw[gate_end:],
        )

    def forward(
        self, features: Tensor, labels: Tensor, physical_tokens: Iterable[object]
    ) -> SupportDomainState:
        """Infer a task state from unique, finite support-only feature rows."""

        if not isinstance(features, Tensor) or not isinstance(labels, Tensor):
            raise ValueError("support features and labels must be torch.Tensor values")
        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != self.feature_dim
            or labels.shape != (features.shape[0],)
        ):
            raise ValueError("support feature geometry drift")
        if not features.is_floating_point() or labels.is_floating_point() or labels.dtype == torch.bool:
            raise ValueError("support features must be floating point and labels must be integer indices")
        if features.device != labels.device:
            raise ValueError("support features and labels must share a device")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("non-finite support features are not allowed")
        try:
            tokens = tuple(physical_tokens)
            unique_count = len(set(tokens))
        except TypeError as error:
            raise ValueError("physical support tokens must be hashable") from error
        if len(tokens) != features.shape[0]:
            raise ValueError("physical support token count drift")
        if unique_count != len(tokens):
            raise ValueError("physical support tokens must be unique")

        sample_h = self.phi(features)
        if not bool(torch.isfinite(sample_h).all()):
            raise ValueError("non-finite support encoder activations")
        class_h = torch.stack(
            [
                torch.cat(
                    (
                        sample_h[labels == class_id].mean(dim=0),
                        sample_h[labels == class_id].var(dim=0, unbiased=False),
                    )
                )
                for class_id in torch.unique(labels, sorted=True)
            ]
        )
        raw = self.rho(class_h.mean(dim=0))
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("non-finite support encoder outputs")
        q, raw_u, raw_gate, raw_lr = self._split_outputs(raw)
        return SupportDomainState(
            q=q,
            uncertainty=torch.sigmoid(raw_u),
            block_gates=torch.sigmoid(raw_gate),
            block_lrs=self.lr_min + (self.lr_max - self.lr_min) * torch.sigmoid(raw_lr),
        )


__all__ = ["SupportDomainState", "SupportSetEncoder"]
