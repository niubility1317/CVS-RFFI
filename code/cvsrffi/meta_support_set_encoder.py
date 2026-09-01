"""Permutation-invariant support-only domain state inference for MARC-OT."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SupportDomainState:
    """Frozen support-derived controls for one task before query inference."""

    q: Tensor
    uncertainty: Tensor
    block_gates: Tensor
    block_lrs: Tensor
    class_means: Tensor = field(default_factory=lambda: torch.empty(0, 0))
    class_diag_variances: Tensor = field(default_factory=lambda: torch.empty(0, 0))
    class_norms: Tensor = field(default_factory=lambda: torch.empty(0, 2))
    class_stat_flags: Tensor = field(default_factory=lambda: torch.empty(0, 4))


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
        if not math.isfinite(lr_min) or not math.isfinite(lr_max) or lr_min <= 0.0 or lr_min >= lr_max:
            raise ValueError("learning-rate bounds must be finite and positive")
        self.feature_dim = feature_dim
        self.coefficient_dim = coefficient_dim
        self.block_count = block_count
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.class_stat_flag_count = 4
        self.class_stat_dim = 2 * feature_dim + 2 + self.class_stat_flag_count
        self.phi = nn.Sequential(
            nn.Linear(self.class_stat_dim, hidden_dim),
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
        self,
        features: Tensor,
        labels: Tensor,
        physical_tokens: Iterable[object],
        effective_mask: Tensor,
    ) -> SupportDomainState:
        """Infer a task state from masked raw per-class support statistics."""

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
        if not isinstance(effective_mask, Tensor):
            raise ValueError("support effective mask must be a torch.Tensor")
        mask = effective_mask.to(device=features.device, dtype=features.dtype)
        if mask.shape != (features.shape[0],) or not bool(torch.isfinite(mask).all()):
            raise ValueError("support effective mask must be finite and row aligned")
        if not bool(((mask == 0.0) | (mask == 1.0)).all()):
            raise ValueError("support effective mask must contain only zero or one")
        try:
            tokens = tuple(physical_tokens)
            unique_count = len(set(tokens))
        except TypeError as error:
            raise ValueError("physical support tokens must be hashable") from error
        if len(tokens) != features.shape[0]:
            raise ValueError("physical support token count drift")
        if unique_count != len(tokens):
            raise ValueError("physical support tokens must be unique")

        class_means: list[Tensor] = []
        class_variances: list[Tensor] = []
        class_norms: list[Tensor] = []
        class_flags: list[Tensor] = []
        class_statistics: list[Tensor] = []
        for class_id in torch.unique(labels, sorted=True):
            selected = (labels == class_id) & (mask == 1.0)
            effective_k = int(selected.sum().item())
            if effective_k < 1:
                raise ValueError("every support class must retain an effective row")
            class_rows = features[selected]
            mean = class_rows.mean(dim=0)
            diagonal_variance = class_rows.var(dim=0, unbiased=False)
            norms = torch.stack((mean.norm(), diagonal_variance.norm()))
            flags = features.new_tensor(
                (1.0, float(effective_k >= 2), float(effective_k >= 5), float(effective_k >= 10))
            )
            class_means.append(mean)
            class_variances.append(diagonal_variance)
            class_norms.append(norms)
            class_flags.append(flags)
            class_statistics.append(
                torch.cat((mean, diagonal_variance, norms, flags), dim=0)
            )

        mean_rows = torch.stack(class_means)
        variance_rows = torch.stack(class_variances)
        norm_rows = torch.stack(class_norms)
        flag_rows = torch.stack(class_flags)
        class_h = self.phi(torch.stack(class_statistics))
        if not bool(torch.isfinite(class_h).all()):
            raise ValueError("non-finite support encoder activations")
        pooled = torch.cat(
            (class_h.mean(dim=0), class_h.var(dim=0, unbiased=False)), dim=0
        )
        raw = self.rho(pooled)
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("non-finite support encoder outputs")
        q, raw_u, raw_gate, raw_lr = self._split_outputs(raw)
        return SupportDomainState(
            q=q,
            uncertainty=torch.sigmoid(raw_u),
            block_gates=torch.sigmoid(raw_gate),
            block_lrs=self.lr_min + (self.lr_max - self.lr_min) * torch.sigmoid(raw_lr),
            class_means=mean_rows,
            class_diag_variances=variance_rows,
            class_norms=norm_rows,
            class_stat_flags=flag_rows,
        )


__all__ = ["SupportDomainState", "SupportSetEncoder"]
