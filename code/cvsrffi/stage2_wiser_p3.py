"""Support-only P3 old-class risk objectives for WISER-RF."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import torch
import torch.nn.functional as functional

from cvsrffi.stage2_binova_d92 import differentiable_old_d92_logits


@dataclass(frozen=True)
class CrossFitFold:
    """The support rows used to fit and validate one cross-fit D92 model."""

    fit_indices: torch.Tensor
    validation_indices: torch.Tensor


@dataclass(frozen=True)
class P3LossResult:
    """Support-only out-of-fold P3 objective and its classwise constraints."""

    total: torch.Tensor
    mean_risk: torch.Tensor
    soft_floor: torch.Tensor
    class_risk: torch.Tensor
    violation: torch.Tensor
    oof_logits: torch.Tensor
    oof_predictions: torch.Tensor


def _token_order(seed: int, tokens: Sequence[str], indices: torch.Tensor) -> torch.Tensor:
    ordered = sorted(
        indices.tolist(),
        key=lambda index: sha256(f"{seed}:{tokens[index]}".encode("utf-8")).digest(),
    )
    return torch.tensor(ordered, dtype=torch.long, device=indices.device)


def stratified_crossfit_indices(
    labels: torch.Tensor,
    support_tokens: Sequence[str],
    *,
    fold_count: int,
    seed: int,
) -> Sequence[CrossFitFold]:
    """Split support rows by class with token-stable validation membership.

    Tokens, rather than package rows, bind each physical support observation to a
    fold. Consequently a valid package reorder cannot change membership.
    """

    values = torch.as_tensor(labels).view(-1).long()
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if len(support_tokens) != len(values):
        raise ValueError("support_tokens must align one-to-one with labels")
    tokens = tuple(str(token) for token in support_tokens)
    if len(set(tokens)) != len(tokens):
        raise ValueError("support_tokens must be unique")
    if len(values) == 0:
        raise ValueError("cross-fitting requires at least one support row")

    validation_parts: list[list[torch.Tensor]] = [[] for _ in range(fold_count)]
    for class_id in torch.unique(values, sorted=True).tolist():
        indices = torch.where(values == int(class_id))[0]
        ordered = _token_order(int(seed), tokens, indices)
        for fold_index, chunk in enumerate(torch.tensor_split(ordered, fold_count)):
            validation_parts[fold_index].append(chunk)

    all_indices = torch.arange(len(values), device=values.device)
    folds: list[CrossFitFold] = []
    for parts in validation_parts:
        validation = torch.sort(torch.cat(parts)).values
        fit_mask = torch.ones(len(values), dtype=torch.bool, device=values.device)
        fit_mask[validation] = False
        folds.append(CrossFitFold(all_indices[fit_mask], validation))
    return tuple(folds)


def frozen_class_risk(oof_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return per-class mean held-out cross-entropy without updating any state."""

    logits = torch.as_tensor(oof_logits)
    targets = torch.as_tensor(labels, dtype=torch.long, device=logits.device).view(-1)
    if logits.ndim != 2 or logits.shape[0] != len(targets):
        raise ValueError("oof_logits and labels must have matching row counts")
    if logits.shape[0] == 0:
        raise ValueError("class risk requires at least one out-of-fold row")
    losses = functional.cross_entropy(logits, targets, reduction="none")
    risks = []
    for class_id in range(logits.shape[1]):
        selected = losses[targets == class_id]
        if selected.numel() == 0:
            raise ValueError("each registered class requires held-out support rows")
        risks.append(selected.mean())
    return torch.stack(risks)


def _validate_folds(folds: Sequence[CrossFitFold], row_count: int, device: torch.device) -> None:
    assigned = torch.zeros(row_count, dtype=torch.bool, device=device)
    all_indices = torch.arange(row_count, device=device)
    for fold in folds:
        fit = torch.as_tensor(fold.fit_indices, dtype=torch.long, device=device).view(-1)
        validation = torch.as_tensor(fold.validation_indices, dtype=torch.long, device=device).view(-1)
        if validation.numel() == 0:
            raise ValueError("each cross-fit fold requires validation support rows")
        if bool((validation < 0).any()) or bool((validation >= row_count).any()):
            raise ValueError("cross-fit validation indices are out of range")
        if bool(assigned[validation].any()):
            raise ValueError("each support row must be validated exactly once")
        expected_fit = all_indices[~torch.isin(all_indices, validation)]
        if not torch.equal(torch.sort(fit).values, expected_fit):
            raise ValueError("cross-fit fit indices must be the validation complement")
        assigned[validation] = True
    if not bool(assigned.all()):
        raise ValueError("each support row must be validated exactly once")


def cross_fitted_p3_loss(
    identity: torch.Tensor,
    fft: torch.Tensor,
    labels: torch.Tensor,
    *,
    folds: Sequence[CrossFitFold],
    baseline_class_risk: torch.Tensor,
    class_duals: torch.Tensor,
    epsilon: torch.Tensor,
    rho: float,
    beta: float,
    tau: float,
) -> P3LossResult:
    """Fit D92 only on support folds and optimize the resulting OOF class risk."""

    identities = torch.as_tensor(identity)
    fft_rows = torch.as_tensor(fft, device=identities.device)
    targets = torch.as_tensor(labels, dtype=torch.long, device=identities.device).view(-1)
    if identities.ndim != 2 or fft_rows.ndim != 2 or len(identities) != len(fft_rows):
        raise ValueError("identity and fft support rows must align")
    if len(identities) != len(targets):
        raise ValueError("support labels must align with feature rows")
    if rho < 0.0 or beta < 0.0 or tau <= 0.0:
        raise ValueError("rho and beta must be nonnegative and tau must be positive")
    _validate_folds(folds, len(targets), identities.device)

    oof_logits: torch.Tensor | None = None
    for fold in folds:
        fit = torch.as_tensor(fold.fit_indices, dtype=torch.long, device=identities.device)
        validation = torch.as_tensor(
            fold.validation_indices, dtype=torch.long, device=identities.device
        )
        logits = differentiable_old_d92_logits(
            identities[fit], fft_rows[fit], targets[fit], identities[validation], fft_rows[validation]
        )
        if oof_logits is None:
            oof_logits = torch.empty(
                (len(targets), logits.shape[1]), dtype=logits.dtype, device=logits.device
            )
        oof_logits[validation] = logits
    if oof_logits is None:
        raise ValueError("at least one cross-fit fold is required")

    class_risk = frozen_class_risk(oof_logits, targets)
    reference = torch.as_tensor(
        baseline_class_risk, dtype=class_risk.dtype, device=class_risk.device
    ).detach().view(-1)
    duals = torch.as_tensor(
        class_duals, dtype=class_risk.dtype, device=class_risk.device
    ).detach().view(-1)
    allowance = torch.as_tensor(
        epsilon, dtype=class_risk.dtype, device=class_risk.device
    ).detach().view(-1)
    if not (len(reference) == len(duals) == len(allowance) == len(class_risk)):
        raise ValueError("class-risk vectors must match the registered class count")

    violation = torch.relu(class_risk - reference - allowance)
    mean_risk = class_risk.mean()
    soft_floor = float(tau) * torch.logsumexp(class_risk / float(tau), dim=0)
    total = (
        mean_risk
        + float(beta) * soft_floor
        + torch.sum(duals * violation)
        + 0.5 * float(rho) * torch.sum(violation.square())
    )
    return P3LossResult(
        total=total,
        mean_risk=mean_risk,
        soft_floor=soft_floor,
        class_risk=class_risk,
        violation=violation,
        oof_logits=oof_logits,
        oof_predictions=oof_logits.argmax(dim=1),
    )


def update_nonnegative_duals(
    class_duals: torch.Tensor, violation: torch.Tensor, *, rate: float
) -> torch.Tensor:
    """Apply a projected ascent update for the fixed support-risk constraints."""

    if rate < 0.0:
        raise ValueError("rate must be nonnegative")
    duals = torch.as_tensor(class_duals)
    values = torch.as_tensor(violation, dtype=duals.dtype, device=duals.device)
    if duals.shape != values.shape:
        raise ValueError("class_duals and violation must have matching shapes")
    with torch.no_grad():
        return torch.clamp_min(duals + float(rate) * values, 0.0).detach()


def _shared_domain_inputs(
    target_features: torch.Tensor,
    labels: torch.Tensor,
    source_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.as_tensor(target_features)
    target_labels = torch.as_tensor(labels, dtype=torch.long, device=target.device).view(-1)
    source = torch.as_tensor(source_points, device=target.device)
    if target.ndim != 2 or target.shape[0] != target_labels.numel():
        raise ValueError("target features/labels must align")
    if source.ndim != 3 or source.shape[2] != target.shape[1]:
        raise ValueError("source points must be [domain,class,feature]")
    if source.shape[0] < 1 or source.shape[1] < 1:
        raise ValueError("source points require at least one domain and class")
    if not torch.isfinite(target).all() or not torch.isfinite(source).all():
        raise ValueError("shared domain inputs must be finite")
    class_count = int(source.shape[1])
    if (
        target_labels.numel() < class_count
        or bool((target_labels < 0).any())
        or bool((target_labels >= class_count).any())
    ):
        raise ValueError("target labels must index the registered source classes")
    if any(not bool((target_labels == class_id).any()) for class_id in range(class_count)):
        raise ValueError("target support is missing a registered class")
    return target, target_labels, source


def _class_centers(target: torch.Tensor, labels: torch.Tensor, class_count: int) -> torch.Tensor:
    return torch.stack(
        [target[labels == class_id].mean(dim=0) for class_id in range(class_count)], dim=0
    )


def infer_shared_domain_weights(
    target_features: torch.Tensor,
    labels: torch.Tensor,
    source_points: torch.Tensor,
    *,
    steps: int = 80,
    learning_rate: float = 0.1,
    l2: float = 0.01,
) -> torch.Tensor:
    """Fit one detached domain simplex from frozen initial target support features."""

    if int(steps) < 1 or learning_rate <= 0.0 or l2 < 0.0:
        raise ValueError("shared-domain optimizer parameters must be positive")
    target, target_labels, source = _shared_domain_inputs(
        target_features, labels, source_points
    )
    frozen_target = functional.normalize(target.detach().float(), dim=-1)
    frozen_source = functional.normalize(source.detach().float(), dim=-1)
    class_count = int(frozen_source.shape[1])
    target_centers = functional.normalize(
        _class_centers(frozen_target, target_labels, class_count), dim=-1
    ).detach()
    logits = torch.zeros(
        frozen_source.shape[0], device=frozen_target.device, dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam((logits,), lr=float(learning_rate))
    with torch.enable_grad():
        for _ in range(int(steps)):
            weights = torch.softmax(logits, dim=0)
            source_centers = functional.normalize(
                torch.einsum("d,dcf->cf", weights, frozen_source), dim=-1
            )
            objective = (target_centers - source_centers).square().mean()
            objective = objective + float(l2) * weights.square().sum()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
    return torch.softmax(logits.detach(), dim=0).detach()


def shared_domain_manifold_loss(
    target_features: torch.Tensor,
    labels: torch.Tensor,
    source_points: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Align current target class centers to a fixed shared source-domain mixture."""

    target, target_labels, source = _shared_domain_inputs(
        target_features, labels, source_points
    )
    frozen_weights = torch.as_tensor(weights, device=target.device).detach().float().view(-1)
    if (
        frozen_weights.shape != (source.shape[0],)
        or not torch.isfinite(frozen_weights).all()
        or bool((frozen_weights < 0).any())
        or not torch.isclose(frozen_weights.sum(), torch.tensor(1.0, device=target.device))
    ):
        raise ValueError("shared domain weights must be one finite simplex")
    current_centers = functional.normalize(
        _class_centers(functional.normalize(target.float(), dim=-1), target_labels, int(source.shape[1])),
        dim=-1,
    )
    frozen_source = functional.normalize(source.detach().float(), dim=-1)
    source_centers = functional.normalize(
        torch.einsum("d,dcf->cf", frozen_weights, frozen_source), dim=-1
    )
    return (current_centers - source_centers).square().mean()


__all__ = [
    "CrossFitFold",
    "P3LossResult",
    "cross_fitted_p3_loss",
    "frozen_class_risk",
    "infer_shared_domain_weights",
    "shared_domain_manifold_loss",
    "stratified_crossfit_indices",
    "update_nonnegative_duals",
]
