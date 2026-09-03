from __future__ import annotations

from typing import Any, Mapping

import torch


class SelectiveNuisanceSubspace:
    """Softly remove directions that are nuisance-rich and fingerprint-poor."""

    def __init__(
        self,
        *,
        feature_dim: int,
        max_rank: int,
        weight: float,
        regularization: float = 1e-4,
    ) -> None:
        if int(feature_dim) < 1 or not 1 <= int(max_rank) <= int(feature_dim):
            raise ValueError("invalid selective subspace dimensions")
        if not 0.0 <= float(weight) <= 1.0 or float(regularization) <= 0.0:
            raise ValueError("invalid selective subspace strength")
        self.feature_dim = int(feature_dim)
        self.max_rank = int(max_rank)
        self.weight = float(weight)
        self.regularization = float(regularization)
        self.basis: torch.Tensor | None = None
        self.eigenvalues: torch.Tensor | None = None

    def update(self, nuisance_delta: torch.Tensor, fingerprint_delta: torch.Tensor) -> bool:
        if self.weight == 0.0:
            return False
        nuisance = nuisance_delta.detach().float()
        fingerprint = fingerprint_delta.detach().float()
        if nuisance.ndim != 2 or fingerprint.ndim != 2 or nuisance.shape[1] != self.feature_dim or fingerprint.shape[1] != self.feature_dim:
            raise ValueError("subspace deltas must be rank-2 with feature_dim columns")
        if min(nuisance.shape[0], fingerprint.shape[0]) < 2:
            return False
        nuisance = nuisance - nuisance.mean(dim=0)
        fingerprint = fingerprint - fingerprint.mean(dim=0)
        c_n = nuisance.t() @ nuisance / max(1, nuisance.shape[0] - 1)
        c_f = fingerprint.t() @ fingerprint / max(1, fingerprint.shape[0] - 1)
        eye = torch.eye(self.feature_dim, device=c_n.device)
        f_value, f_vector = torch.linalg.eigh(c_f + self.regularization * eye)
        whitening = f_vector @ torch.diag(f_value.clamp_min(self.regularization).rsqrt()) @ f_vector.t()
        whitened = whitening @ c_n @ whitening
        values, vectors = torch.linalg.eigh(whitened)
        order = values.argsort(descending=True)[: self.max_rank]
        basis = whitening @ vectors[:, order]
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        if not bool(torch.isfinite(basis).all()):
            return False
        self.basis = basis.detach()
        self.eigenvalues = values[order].detach()
        return True

    def project(self, feature: torch.Tensor) -> torch.Tensor:
        if self.weight == 0.0 or self.basis is None:
            return feature
        basis = self.basis.to(device=feature.device, dtype=feature.dtype)
        return feature - self.weight * (feature @ basis) @ basis.t()

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "max_rank": self.max_rank,
            "weight": self.weight,
            "regularization": self.regularization,
            "basis": None if self.basis is None else self.basis.cpu(),
            "eigenvalues": None if self.eigenvalues is None else self.eigenvalues.cpu(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["feature_dim"]) != self.feature_dim or int(state["max_rank"]) != self.max_rank:
            raise ValueError("subspace state dimensions do not match")
        self.weight = float(state["weight"])
        self.regularization = float(state["regularization"])
        self.basis = None if state.get("basis") is None else torch.as_tensor(state["basis"], dtype=torch.float32)
        self.eigenvalues = (
            None if state.get("eigenvalues") is None else torch.as_tensor(state["eigenvalues"], dtype=torch.float32)
        )
