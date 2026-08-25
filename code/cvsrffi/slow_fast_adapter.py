"""Feature-space slow/fast adapters for frozen CVS identity embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import Tensor
from torch.nn import functional as F


class SlowFastCandidate(str, Enum):
    COMMON_SHIFT_R4 = "COMMON_SHIFT_R4"
    FAST_FILM_R8 = "FAST_FILM_R8"
    FAST_LOWRANK_R8 = "FAST_LOWRANK_R8"


def _finite_matrix(value: Tensor, *, name: str) -> Tensor:
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if value.shape[0] < 1 or value.shape[1] < 1 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite and nonempty")
    return value.detach().clone().float()


def _finite_vector(value: Tensor | None, *, name: str, length: int) -> Tensor | None:
    if value is None:
        return None
    if (
        not torch.is_tensor(value)
        or value.ndim != 1
        or value.numel() != int(length)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be a finite rank-sized vector")
    return value.detach().clone().float()


@dataclass(frozen=True)
class SlowFastAdapterState:
    """Frozen aggregate state; no sample-level source rows are representable."""

    candidate: SlowFastCandidate
    slow_u: Tensor
    slow_v: Tensor | None = None
    rho: float = 1.0
    gamma: Tensor | None = None
    beta: Tensor | None = None
    direction_gate: Tensor | None = None
    common_coeff: Tensor | None = None

    def __post_init__(self) -> None:
        candidate = SlowFastCandidate(self.candidate)
        slow_u = _finite_matrix(self.slow_u, name="slow_u")
        feature_dim, rank = (int(value) for value in slow_u.shape)
        rho = float(self.rho)
        if not math.isfinite(rho) or rho < 0.0 or rho > 1.0:
            raise ValueError("rho must be finite in [0, 1]")

        slow_v = None if self.slow_v is None else _finite_matrix(self.slow_v, name="slow_v")
        if slow_v is not None and tuple(slow_v.shape) != (feature_dim, rank):
            raise ValueError("slow_v must match slow_u shape")
        gamma = _finite_vector(self.gamma, name="gamma", length=rank)
        beta = _finite_vector(self.beta, name="beta", length=rank)
        direction_gate = _finite_vector(
            self.direction_gate, name="direction_gate", length=rank
        )
        common_coeff = _finite_vector(
            self.common_coeff, name="common_coeff", length=rank
        )

        if candidate is SlowFastCandidate.COMMON_SHIFT_R4:
            if rank != 4:
                raise ValueError("COMMON_SHIFT_R4 requires rank 4")
            if common_coeff is None:
                raise ValueError("COMMON_SHIFT_R4 requires common_coeff")
            if any(value is not None for value in (slow_v, gamma, beta, direction_gate)):
                raise ValueError("COMMON_SHIFT_R4 accepts only slow_u and common_coeff")
        else:
            if rank != 8:
                raise ValueError("FAST candidates require rank 8")
            if slow_v is None or gamma is None or beta is None:
                raise ValueError("FAST candidates require slow_v, gamma and beta")
            if common_coeff is not None:
                raise ValueError("FAST candidates do not accept common_coeff")
            if candidate is SlowFastCandidate.FAST_FILM_R8 and direction_gate is not None:
                raise ValueError("FAST_FILM_R8 does not accept direction_gate")
            if candidate is SlowFastCandidate.FAST_LOWRANK_R8 and direction_gate is None:
                raise ValueError("FAST_LOWRANK_R8 requires direction_gate")

        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "slow_u", slow_u)
        object.__setattr__(self, "slow_v", slow_v)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "direction_gate", direction_gate)
        object.__setattr__(self, "common_coeff", common_coeff)

    @property
    def feature_dim(self) -> int:
        return int(self.slow_u.shape[0])

    @property
    def rank(self) -> int:
        return int(self.slow_u.shape[1])

    @property
    def fast_parameter_count(self) -> int:
        values = (self.gamma, self.beta, self.direction_gate, self.common_coeff)
        return int(sum(value.numel() for value in values if value is not None))


def apply_slow_fast(features: Tensor, state: SlowFastAdapterState) -> Tensor:
    """Apply one frozen adapter state and return normalized identity features."""

    if (
        not torch.is_tensor(features)
        or features.ndim != 2
        or features.shape[0] < 1
        or not features.is_floating_point()
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("features must be a finite nonempty floating matrix")
    if int(features.shape[1]) != state.feature_dim:
        raise ValueError(
            f"feature width {int(features.shape[1])} does not match frozen width {state.feature_dim}"
        )
    z = features.float()
    u = state.slow_u.to(device=z.device, dtype=z.dtype)
    if state.candidate is SlowFastCandidate.COMMON_SHIFT_R4:
        coeff = state.common_coeff.to(device=z.device, dtype=z.dtype)
        corrected = z - (coeff @ u.transpose(0, 1)).unsqueeze(0)
        return F.normalize(corrected, dim=1, eps=1.0e-8)

    v = state.slow_v.to(device=z.device, dtype=z.dtype)
    gamma = state.gamma.to(device=z.device, dtype=z.dtype)
    beta = state.beta.to(device=z.device, dtype=z.dtype)
    normalized = F.layer_norm(z, (state.feature_dim,))
    hidden = normalized @ v
    latent = (1.0 + gamma) * hidden + beta
    if state.candidate is SlowFastCandidate.FAST_LOWRANK_R8:
        gate = torch.sigmoid(
            state.direction_gate.to(device=z.device, dtype=z.dtype)
        )
        latent = gate * latent
    corrected = z + float(state.rho) * (latent @ u.transpose(0, 1))
    return F.normalize(corrected, dim=1, eps=1.0e-8)


__all__ = [
    "SlowFastAdapterState",
    "SlowFastCandidate",
    "apply_slow_fast",
]
