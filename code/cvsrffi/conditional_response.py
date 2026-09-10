"""Bounded conditional responses and exact Gaussian support registration.

State descriptors must be supplied by a separately audited source-only pipeline;
this module does not infer physical excitation from received amplitude.
"""
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import torch
from torch import Tensor, nn


def _finite(x: Tensor, name: str) -> None:
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite floating point")


def _spd(x: Tensor, name: str) -> Tensor:
    _finite(x, name)
    if x.shape[-1] != x.shape[-2] or not torch.allclose(x, x.transpose(-1, -2)):
        raise ValueError(f"{name} must be symmetric")
    chol, info = torch.linalg.cholesky_ex(x)
    if (info != 0).any():
        raise ValueError(f"{name} must be positive definite")
    return chol


class ConditionalResponse(nn.Module):
    """mu_y + shared centered linear response + class-centered slopes.

    fit_state_domain is an explicit source-training operation. Queries never
    update buffers. Responses clamp to the source box; domain_diagnostics must
    accompany inference when interpreting states outside that box.
    """

    def __init__(self, num_classes: int, feature_dim: int, state_dim: int,
                 response_scale: float = 1.0):
        super().__init__()
        if min(num_classes, feature_dim, state_dim) < 1 or not 0 < response_scale < float("inf"):
            raise ValueError("positive dimensions and finite positive response_scale required")
        self.num_classes, self.feature_dim, self.state_dim = num_classes, feature_dim, state_dim
        self.response_scale = response_scale
        self.mean = nn.Parameter(torch.zeros(num_classes, feature_dim))
        self.shared_slopes = nn.Parameter(torch.zeros(feature_dim, state_dim))
        self.class_slopes = nn.Parameter(torch.zeros(num_classes, feature_dim, state_dim))
        self.register_buffer("state_lower", torch.zeros(state_dim))
        self.register_buffer("state_upper", torch.zeros(state_dim))
        self.register_buffer("state_center", torch.zeros(state_dim))
        self.register_buffer("domain_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_state_domain(self, source_states: Tensor, *, source_training: bool) -> None:
        if source_training is not True:
            raise ValueError("domain fit requires explicit source_training=True")
        self._check_state(source_states, require_fit=False)
        if source_states.shape[0] == 0:
            raise ValueError("source states cannot be empty")
        self.state_lower.copy_(source_states.amin(0))
        self.state_upper.copy_(source_states.amax(0))
        self.state_center.copy_(source_states.mean(0))
        self.domain_fitted.fill_(True)

    def _check_state(self, e: Tensor, require_fit: bool = True) -> None:
        _finite(e, "state")
        if e.ndim != 2 or e.shape[1] != self.state_dim:
            raise ValueError("state must have shape [B,state_dim]")
        if require_fit and not bool(self.domain_fitted):
            raise RuntimeError("fit source-training state domain before inference")

    def _slopes(self) -> Tensor:
        return self.response_scale * (self.shared_slopes.unsqueeze(0) +
                                     self.class_slopes - self.class_slopes.mean(0, keepdim=True))

    def forward(self, e: Tensor) -> Tensor:
        self._check_state(e)
        centered = e.clamp(self.state_lower, self.state_upper) - self.state_center
        return self.mean.unsqueeze(0) + torch.einsum("bp,cdp->bcd", centered, self._slopes())

    def basis(self, e: Tensor) -> Tensor:
        """Exact support basis [1, clamp(e)-source_center]."""
        self._check_state(e)
        return torch.cat((e.new_ones(len(e), 1),
                          e.clamp(self.state_lower, self.state_upper)-self.state_center), -1)

    def coefficients(self) -> Tensor:
        """Per-class prior means [C,1+P,D], compatible with basis(e)."""
        return torch.cat((self.mean[:, None, :], self._slopes().transpose(-1, -2)), 1)

    def jacobian(self, e: Tensor) -> Tensor:
        self._check_state(e)
        active = ((e >= self.state_lower) & (e <= self.state_upper)).to(e.dtype)
        return self._slopes().unsqueeze(0) * active[:, None, None, :]

    def state_uncertainty(self, e: Tensor, covariance: Tensor) -> Tensor:
        """First-order J U J^T; U may be positive semidefinite."""
        self._check_state(e)
        if covariance.shape != (len(e), self.state_dim, self.state_dim):
            raise ValueError("state covariance must have shape [B,P,P]")
        _finite(covariance, "state covariance")
        if not torch.allclose(covariance, covariance.transpose(-1, -2)) or (torch.linalg.eigvalsh(covariance) < -1e-7).any():
            raise ValueError("state covariance must be symmetric positive semidefinite")
        j = self.jacobian(e)
        return torch.einsum("bcdp,bpq,bceq->bcde", j, covariance, j)

    def domain_diagnostics(self, e: Tensor) -> dict:
        self._check_state(e)
        excess = e - e.clamp(self.state_lower, self.state_upper)
        return {"in_domain": (excess == 0).all(-1), "outside_distance": excess.norm(dim=-1),
                "clamped_state": e.clamp(self.state_lower, self.state_upper)}


@dataclass(frozen=True)
class SupportResponsePosterior:
    """Frozen registration snapshot; public tensors are defensive copies.

    Coefficients use row-major vec(A), A[P,D]. Prediction covariance is only
    parameter uncertainty; observation/response covariance is added by caller.
    """
    _mean: Tensor
    _precision: Tensor
    _design: Tensor
    physical_ids: tuple
    observation_count: int

    @property
    def mean(self) -> Tensor:
        return self._mean.clone()

    @property
    def precision(self) -> Tensor:
        return self._precision.clone()

    def predict(self, phi: Tensor) -> tuple[Tensor, Tensor]:
        _finite(phi, "phi")
        if phi.ndim != 2 or phi.shape[1] != self._mean.shape[0]:
            raise ValueError("phi must have shape [B,P]")
        output_dtype = phi.dtype
        phi = phi.to(self._precision.dtype)
        d = self._mean.shape[1]
        eye = torch.eye(d, dtype=phi.dtype, device=phi.device)
        h = torch.einsum("bp,df->bdpf", phi, eye).reshape(len(phi), d, -1)
        solved = torch.linalg.solve(self._precision, h.transpose(-1, -2))
        covariance = h @ solved
        covariance = (covariance + covariance.transpose(-1, -2)) * .5
        return (phi @ self._mean).to(output_dtype), covariance.to(output_dtype)

    def coverage(self, phi: Tensor | None = None) -> dict:
        rank = int(torch.linalg.matrix_rank(self._design)) if self._design.numel() else 0
        result = {"physical_shots": len(self.physical_ids), "observed_scalars": self.observation_count,
                  "design_rank": rank, "basis_dim": self._mean.shape[0],
                  "full_state_rank": rank == self._mean.shape[0]}
        if phi is not None:
            _, covariance = self.predict(phi)
            result["leverage"] = covariance.diagonal(dim1=-2, dim2=-1).mean(-1)
        return result

    def to_dict(self) -> dict[str, Tensor]:
        """Detached tensor-only deployment state; physical IDs remain in run metadata.

        The snapshot records the shot count, not raw possibly sensitive IDs.
        Reloaded IDs are synthetic immutable index keys, never usable as new
        source/support provenance evidence.
        """
        return {"mean": self._mean.detach().clone(), "precision": self._precision.detach().clone(),
                "design": self._design.detach().clone(),
                "physical_shots": torch.tensor(len(self.physical_ids)),
                "observation_count": torch.tensor(self.observation_count)}

    @classmethod
    def from_dict(cls, state: dict[str, Tensor]) -> "SupportResponsePosterior":
        mean, precision, design = state['mean'], state['precision'], state['design']
        _finite(mean, 'mean')
        if mean.ndim != 2 or precision.shape != (mean.numel(), mean.numel()):
            raise ValueError('invalid posterior dimensions')
        _spd(precision, 'precision')
        _finite(design, 'design')
        if design.ndim != 2 or design.shape[1] != mean.shape[0]:
            raise ValueError('invalid posterior design')
        shots, count = int(state['physical_shots']), int(state['observation_count'])
        if shots < len(design) or count < 0 or count > shots*mean.shape[1]:
            raise ValueError('invalid posterior counts')
        return cls(mean.detach().clone(), precision.detach().clone(), design.detach().clone(),
                   tuple(f'restored:{i}' for i in range(shots)), count)


def fit_support_response(z: Tensor, phi: Tensor, covariance: Tensor, observed: Tensor,
                         physical_ids: Sequence, *, prior_mean: Tensor | None = None,
                         prior_precision: float | Tensor = 1.0) -> SupportResponsePosterior:
    """Exact regression with marginal observed covariance, never zero filling.

    For same-variance ridge use prior_precision=lambda/sigma**2. Duplicate
    physical IDs are rejected, including repeated augmented views.
    """
    if z.ndim != 2 or phi.ndim != 2 or len(z) != len(phi) or min(z.shape[1], phi.shape[1]) < 1:
        raise ValueError("z[N,D] and phi[N,P] required")
    n, d = z.shape
    p = phi.shape[1]
    _finite(phi, "phi")
    if observed.shape != z.shape or observed.dtype != torch.bool:
        raise ValueError("observed must be bool[N,D]")
    _finite(z[observed], "observed z")
    if covariance.shape != (n, d, d):
        raise ValueError("covariance must have shape [N,D,D]")
    _spd(covariance, "covariance")
    if len(physical_ids) != n or len(set(physical_ids)) != n:
        raise ValueError("physical_ids must identify distinct physical support shots")
    mean = z.new_zeros(p, d) if prior_mean is None else prior_mean
    if mean.shape != (p, d):
        raise ValueError("prior_mean must have shape [P,D]")
    _finite(mean, "prior_mean")
    precision = torch.as_tensor(prior_precision, dtype=z.dtype, device=z.device)
    if precision.ndim == 0:
        precision = torch.eye(p*d, dtype=z.dtype, device=z.device) * precision
    if precision.shape != (p*d, p*d):
        raise ValueError("prior_precision must be scalar or [PD,PD]")
    _spd(precision, "prior_precision")
    # Validate supplied matrices before any transformation. Accumulate/solve in
    # FP64: dense PD precision systems amplify FP32 cancellation and asymmetric
    # BLAS roundoff even though each exact H' V^-1 H is symmetric.
    precision = precision.double().clone()
    mean, phi, covariance, z = mean.double(), phi.double(), covariance.double(), z.double()
    rhs = precision @ mean.reshape(-1)
    identity = torch.eye(d, dtype=z.dtype, device=z.device)
    for i in range(n):
        mask = observed[i]
        if not mask.any():
            continue
        h = torch.einsum("p,od->opd", phi[i], identity[mask]).reshape(int(mask.sum()), p*d)
        # Select covariance BEFORE inverse/solve: this is marginal inference.
        selected = covariance[i][mask][:, mask]
        chol = torch.linalg.cholesky(selected)
        update = h.T @ torch.cholesky_solve(h, chol)
        precision = precision + (update + update.T) * .5
        rhs = rhs + h.T @ torch.cholesky_solve(z[i, mask, None], chol).squeeze(-1)
    posterior_mean = torch.cholesky_solve(rhs[:, None], torch.linalg.cholesky(precision)).reshape(p, d)
    return SupportResponsePosterior(posterior_mean.clone(), precision.clone(),
                                    phi[observed.any(-1)].detach().clone(), tuple(physical_ids),
                                    int(observed.sum()))


def matched_state_cross_rx_differences(z: Tensor, tx: Sequence, receiver: Sequence,
                                       state: Sequence) -> list[dict]:
    """Source diagnostic over group means in supplied matched state strata.

    No arbitrary sample pairing, no training loss, no physical disentanglement
    claim. Caller must supply credible matched-state labels and source rows.
    """
    _finite(z, "z")
    if z.ndim != 2 or any(len(v) != len(z) for v in (tx, receiver, state)):
        raise ValueError("z[N,D] and N group labels required")
    groups = {}
    for i, key in enumerate(zip(tx, receiver, state)):
        groups.setdefault(key, []).append(i)
    means = {key: z[indices].mean(0).detach() for key, indices in groups.items()}
    result = []
    for s in dict.fromkeys(state):
        for a, b in combinations(dict.fromkeys(tx), 2):
            rxs = [r for r in dict.fromkeys(receiver) if (a,r,s) in means and (b,r,s) in means]
            for r1, r2 in combinations(rxs, 2):
                diff = means[a,r1,s] - means[b,r1,s] - means[a,r2,s] + means[b,r2,s]
                result.append({"tx_pair": (a,b), "receiver_pair": (r1,r2), "state": s,
                               "difference": diff, "squared_norm": diff.square().sum()})
    return result
