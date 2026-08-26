"""Source-fitted receiver/LEO bases with support-only analytic context adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .slow_fast_cache import GroundFeatureCache


_LEO_VIEWS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _finite_matrix(value: Tensor, *, name: str) -> Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] < 1
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be a finite nonempty floating matrix")
    return value.detach().clone().float()


def _geometric_median(rows: Tensor, *, iterations: int = 32) -> Tensor:
    if rows.ndim != 2 or rows.shape[0] < 1:
        raise ValueError("geometric median requires a nonempty row matrix")
    estimate = rows.median(dim=0).values
    for _ in range(int(iterations)):
        distances = torch.linalg.vector_norm(rows - estimate, dim=1).clamp_min(1.0e-6)
        weights = distances.reciprocal()
        updated = (rows * weights[:, None]).sum(dim=0) / weights.sum()
        if float(torch.linalg.vector_norm(updated - estimate)) <= 1.0e-7:
            estimate = updated
            break
        estimate = updated
    return estimate


def _robust_basis(rows: Tensor, *, rank: int, remove: Tensor | None = None) -> Tensor:
    values = rows.float()
    if remove is not None:
        values = values - (values @ remove) @ remove.transpose(0, 1)
    norms = torch.linalg.vector_norm(values, dim=1)
    positive = norms[norms > 1.0e-8]
    if positive.numel() < int(rank):
        raise ValueError("not enough nonzero domain rows for requested rank")
    clip = torch.quantile(positive, 0.9)
    values = values * (clip / norms.clamp_min(clip)).clamp(max=1.0)[:, None]
    _left, _singular, right = torch.linalg.svd(values, full_matrices=False)
    basis = right[: int(rank)].transpose(0, 1).contiguous()
    if remove is not None:
        basis = basis - remove @ (remove.transpose(0, 1) @ basis)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
    return basis[:, : int(rank)].detach()


@dataclass(frozen=True)
class FactoredSlowFastState:
    """Aggregate deployment state; no source sample rows are representable."""

    receiver_basis: Tensor
    leo_basis: Tensor
    geometric_centers: Tensor
    decision_prototypes: Tensor
    class_ids: Tensor
    ridge_receiver: float = 0.1
    ridge_leo: float = 0.1

    def __post_init__(self) -> None:
        receiver = _finite_matrix(self.receiver_basis, name="receiver_basis")
        leo = _finite_matrix(self.leo_basis, name="leo_basis")
        centers = _finite_matrix(self.geometric_centers, name="geometric_centers")
        prototypes = _finite_matrix(self.decision_prototypes, name="decision_prototypes")
        if receiver.shape[0] != leo.shape[0] or centers.shape[1] != receiver.shape[0]:
            raise ValueError("factored basis and geometric center widths must match")
        if prototypes.shape != centers.shape:
            raise ValueError("decision prototypes and geometric centers must align")
        if (
            not torch.is_tensor(self.class_ids)
            or self.class_ids.ndim != 1
            or self.class_ids.numel() != centers.shape[0]
            or self.class_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or len(set(int(value) for value in self.class_ids.tolist())) != self.class_ids.numel()
        ):
            raise ValueError("class_ids must be a unique row-aligned integer vector")
        ridge_rx = float(self.ridge_receiver)
        ridge_leo = float(self.ridge_leo)
        if not math.isfinite(ridge_rx) or not math.isfinite(ridge_leo) or ridge_rx <= 0.0 or ridge_leo <= 0.0:
            raise ValueError("factored ridge values must be finite and positive")
        object.__setattr__(self, "receiver_basis", receiver)
        object.__setattr__(self, "leo_basis", leo)
        object.__setattr__(self, "geometric_centers", F.normalize(centers, dim=1))
        object.__setattr__(self, "decision_prototypes", F.normalize(prototypes, dim=1))
        object.__setattr__(self, "class_ids", self.class_ids.detach().clone().long())
        object.__setattr__(self, "ridge_receiver", ridge_rx)
        object.__setattr__(self, "ridge_leo", ridge_leo)

    @property
    def feature_dim(self) -> int:
        return int(self.receiver_basis.shape[0])

    @property
    def fast_parameter_count(self) -> int:
        return int(self.receiver_basis.shape[1] + self.leo_basis.shape[1])

    @property
    def basis(self) -> Tensor:
        return torch.cat((self.receiver_basis, self.leo_basis), dim=1)


def _training_indices(cache: GroundFeatureCache, excluded: Iterable[object]) -> list[int]:
    excluded_strings = {str(value) for value in excluded}
    indices = [index for index, receiver in enumerate(cache.receivers) if str(receiver) not in excluded_strings]
    if not indices:
        raise ValueError("factored slow state has no training receivers")
    return indices


def _geometric_centers(cache: GroundFeatureCache, indices: list[int], class_ids: Tensor) -> Tensor:
    centers: list[Tensor] = []
    for class_row in range(int(class_ids.numel())):
        rows = [cache.features[index] for index in indices if cache.views[index] == "clean" and int(cache.labels[index]) == class_row]
        if not rows:
            raise ValueError("every registered class needs clean rows for geometric centers")
        centers.append(F.normalize(_geometric_median(torch.stack(rows)), dim=0))
    return torch.stack(centers)


def _receiver_vectors(
    cache: GroundFeatureCache,
    indices: list[int],
    centers: Tensor,
    class_ids: Tensor,
) -> tuple[Tensor, list[str]]:
    receivers = sorted({str(cache.receivers[index]) for index in indices})
    vectors: list[Tensor] = []
    for receiver in receivers:
        class_residuals: list[Tensor] = []
        for class_row in range(int(class_ids.numel())):
            rows = [
                cache.features[index]
                for index in indices
                if str(cache.receivers[index]) == receiver
                and cache.views[index] == "clean"
                and int(cache.labels[index]) == class_row
            ]
            if not rows:
                raise ValueError("every fit receiver must cover all registered classes in clean")
            center = centers[class_row]
            residual = _geometric_median(torch.stack(rows)) - center
            residual = residual - torch.dot(residual, center) * center
            class_residuals.append(residual)
        vectors.append(_geometric_median(torch.stack(class_residuals)))
    return torch.stack(vectors), receivers


def _paired_leo_residuals(
    cache: GroundFeatureCache,
    indices: list[int],
    *,
    scene: str | None = None,
) -> Tensor:
    allowed = set(indices)
    grouped: dict[str, dict[str, int]] = {}
    for index in indices:
        grouped.setdefault(cache.physical_sample_ids[index], {})[cache.views[index]] = index
    residuals: list[Tensor] = []
    for views in grouped.values():
        clean_index = views.get("clean")
        if clean_index is None:
            continue
        clean = F.normalize(cache.features[clean_index].float(), dim=0)
        names = (scene,) if scene is not None else _LEO_VIEWS
        for name in names:
            leo_index = views.get(name)
            if leo_index is None or leo_index not in allowed:
                continue
            delta = F.normalize(cache.features[leo_index].float(), dim=0) - clean
            delta = delta - torch.dot(delta, clean) * clean
            residuals.append(delta)
    if not residuals:
        raise ValueError("factored LEO basis requires clean/LEO physical pairs")
    return torch.stack(residuals)


def fit_factored_state(
    cache: GroundFeatureCache,
    decision_prototypes: Tensor,
    class_ids: Tensor,
    *,
    excluded_receiver: object | None = None,
    excluded_receivers: Iterable[object] = (),
    rank_rx: int = 4,
    rank_leo: int = 4,
    ridge_receiver: float = 0.1,
    ridge_leo: float = 0.1,
) -> tuple[FactoredSlowFastState, dict[str, Any]]:
    excluded = list(excluded_receivers)
    if excluded_receiver is not None:
        excluded.append(excluded_receiver)
    indices = _training_indices(cache, excluded)
    class_tensor = class_ids.detach().clone().long()
    prototypes = _finite_matrix(decision_prototypes, name="decision_prototypes")
    if prototypes.shape[0] != class_tensor.numel() or prototypes.shape[1] != cache.feature_dim:
        raise ValueError("decision prototypes must align with class_ids and cache width")
    cache_classes = tuple(sorted(int(value) for value in torch.unique(cache.labels).tolist()))
    if cache_classes != tuple(range(int(class_tensor.numel()))):
        raise ValueError("ground cache labels must be contiguous prototype rows")
    centers = _geometric_centers(cache, indices, class_tensor)
    receiver_rows, fit_receivers = _receiver_vectors(cache, indices, centers, class_tensor)
    receiver_basis = _robust_basis(receiver_rows, rank=int(rank_rx))
    leo_rows = _paired_leo_residuals(cache, indices)
    leo_basis = _robust_basis(leo_rows, rank=int(rank_leo), remove=receiver_basis)
    state = FactoredSlowFastState(
        receiver_basis=receiver_basis,
        leo_basis=leo_basis,
        geometric_centers=centers,
        decision_prototypes=prototypes,
        class_ids=class_tensor,
        ridge_receiver=ridge_receiver,
        ridge_leo=ridge_leo,
    )
    return state, {
        "schema": "cvs.factored_slow_fast.state_fit.v1",
        "fit_receivers": fit_receivers,
        "excluded_receivers": sorted(str(value) for value in excluded),
        "rank_receiver": int(rank_rx),
        "rank_leo": int(rank_leo),
        "fast_parameter_count": state.fast_parameter_count,
        "source_rows_used": len(indices),
        "paired_leo_rows": int(leo_rows.shape[0]),
    }


def solve_factored_context(
    support_features: Tensor,
    support_class_ids: Tensor,
    state: FactoredSlowFastState,
) -> tuple[Tensor, dict[str, Any]]:
    features = _finite_matrix(support_features, name="support_features")
    if features.shape[1] != state.feature_dim:
        raise ValueError("support feature width does not match factored state")
    if not torch.is_tensor(support_class_ids) or support_class_ids.ndim != 1 or support_class_ids.numel() != features.shape[0]:
        raise ValueError("support class IDs must align with support rows")
    row_by_id = {int(value): row for row, value in enumerate(state.class_ids.tolist())}
    observed = {int(value) for value in support_class_ids.tolist()}
    if not observed or not observed <= set(row_by_id):
        raise ValueError("domain context may use registered old classes only")
    basis = state.basis.to(features)
    ridge = torch.cat(
        (
            torch.full((state.receiver_basis.shape[1],), state.ridge_receiver, device=features.device, dtype=features.dtype),
            torch.full((state.leo_basis.shape[1],), state.ridge_leo, device=features.device, dtype=features.dtype),
        )
    )
    system = basis.transpose(0, 1) @ basis + torch.diag(ridge)
    codes: list[Tensor] = []
    residuals: list[Tensor] = []
    for class_id in sorted(observed):
        mask = support_class_ids.long() == int(class_id)
        location = _geometric_median(features[mask])
        residual = location - state.geometric_centers[row_by_id[class_id]].to(features)
        residuals.append(residual)
        codes.append(torch.linalg.solve(system, basis.transpose(0, 1) @ residual))
    per_class_codes = torch.stack(codes)
    context = _geometric_median(per_class_codes)
    common_residual = _geometric_median(torch.stack(residuals))
    explained = basis @ context
    coverage = float(explained.square().sum() / common_residual.square().sum().clamp_min(1.0e-8))
    disagreement = float(torch.linalg.vector_norm(per_class_codes - context, dim=1).median())
    return context, {
        "schema": "cvs.factored_slow_fast.context.v1",
        "old_class_count": len(observed),
        "fast_parameter_count": int(context.numel()),
        "coverage": coverage,
        "class_code_disagreement": disagreement,
        "support_shift_norm": float(torch.linalg.vector_norm(common_residual)),
        "query_updates": 0,
        "optimizer_state_bytes": 0,
        "per_class_codes": per_class_codes.detach().cpu(),
    }


def apply_factored_context(features: Tensor, state: FactoredSlowFastState, context: Tensor) -> Tensor:
    rows = _finite_matrix(features, name="features")
    if rows.shape[1] != state.feature_dim or not torch.is_tensor(context) or context.ndim != 1 or context.numel() != state.fast_parameter_count:
        raise ValueError("feature/context shape does not match factored state")
    correction = state.basis.to(rows) @ context.to(rows)
    return F.normalize(rows - correction.unsqueeze(0), dim=1, eps=1.0e-8)


def _margins(features: Tensor, labels: Tensor, prototypes: Tensor) -> Tensor:
    scores = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).transpose(0, 1)
    true_scores = scores.gather(1, labels.long()[:, None]).squeeze(1)
    competitors = scores.clone()
    competitors.scatter_(1, labels.long()[:, None], float("-inf"))
    return true_scores - competitors.max(dim=1).values


def support_safety_diagnostics(
    baseline_features: Tensor,
    adapted_features: Tensor,
    label_rows: Tensor,
    decision_prototypes: Tensor,
    *,
    coverage: float,
    disagreement: float,
    min_coverage: float,
    max_disagreement: float,
    min_correct_margin_q10: float,
    min_wrong_margin_median: float,
    min_class_margin_cvar: float,
) -> dict[str, Any]:
    baseline = _finite_matrix(baseline_features, name="baseline_features")
    adapted = _finite_matrix(adapted_features, name="adapted_features")
    prototypes = _finite_matrix(decision_prototypes, name="decision_prototypes")
    if baseline.shape != adapted.shape or label_rows.numel() != baseline.shape[0]:
        raise ValueError("support safety inputs must align")
    before = _margins(baseline, label_rows, prototypes)
    after = _margins(adapted, label_rows, prototypes)
    correct = before > 0.0
    wrong = ~correct
    flips = int((correct & (after <= 0.0)).sum())
    ratios = after[correct] / before[correct].clamp_min(1.0e-8)
    ratio_q10 = float(torch.quantile(ratios, 0.1)) if ratios.numel() else math.inf
    wrong_delta_median = float((after[wrong] - before[wrong]).median()) if wrong.any() else math.inf
    class_cvars: list[float] = []
    for class_id in torch.unique(label_rows).tolist():
        delta = after[label_rows == int(class_id)] - before[label_rows == int(class_id)]
        count = max(1, int(math.ceil(0.2 * delta.numel())))
        class_cvars.append(float(torch.sort(delta).values[:count].mean()))
    worst_cvar = min(class_cvars)
    baseline_scores = F.normalize(baseline, dim=1) @ F.normalize(prototypes, dim=1).transpose(0, 1)
    adapted_scores = F.normalize(adapted, dim=1) @ F.normalize(prototypes, dim=1).transpose(0, 1)
    support_utility = float(
        F.cross_entropy(8.0 * baseline_scores, label_rows.long())
        - F.cross_entropy(8.0 * adapted_scores, label_rows.long())
    )
    reasons: list[str] = []
    if flips:
        reasons.append("CORRECT_TO_WRONG")
    if ratio_q10 < float(min_correct_margin_q10):
        reasons.append("CORRECT_MARGIN_Q10")
    if wrong_delta_median <= float(min_wrong_margin_median):
        reasons.append("WRONG_MARGIN_MEDIAN")
    if worst_cvar < float(min_class_margin_cvar):
        reasons.append("CLASS_MARGIN_CVAR20")
    if float(coverage) < float(min_coverage):
        reasons.append("BASIS_COVERAGE")
    if float(disagreement) > float(max_disagreement):
        reasons.append("CLASS_CODE_DISAGREEMENT")
    return {
        "correct_to_wrong_flips": flips,
        "correct_margin_ratio_q10": ratio_q10,
        "wrong_margin_delta_median": wrong_delta_median,
        "worst_class_margin_cvar20": worst_cvar,
        "support_utility": support_utility,
        "coverage": float(coverage),
        "class_code_disagreement": float(disagreement),
        "safe_to_commit": not reasons,
        "rejection_reasons": reasons,
    }


def basis_scene_diagnostics(
    cache: GroundFeatureCache,
    state: FactoredSlowFastState,
    *,
    excluded_receiver: object | None = None,
) -> dict[str, Any]:
    indices = _training_indices(cache, () if excluded_receiver is None else (excluded_receiver,))
    scene_bases: dict[str, Tensor] = {}
    explained: dict[str, float] = {}
    shared = state.leo_basis
    for scene in _LEO_VIEWS:
        rows = _paired_leo_residuals(cache, indices, scene=scene)
        residualized = rows - (rows @ state.receiver_basis) @ state.receiver_basis.transpose(0, 1)
        rank = min(int(state.leo_basis.shape[1]), int((torch.linalg.vector_norm(residualized, dim=1) > 1.0e-8).sum()))
        scene_bases[scene] = _robust_basis(residualized, rank=max(1, rank), remove=state.receiver_basis)
        numerator = ((residualized @ shared) @ shared.transpose(0, 1)).square().sum()
        explained[scene] = float((numerator / residualized.square().sum().clamp_min(1.0e-8)).clamp(0.0, 1.0))
    angles: dict[str, float] = {}
    for left_index, left in enumerate(_LEO_VIEWS):
        for right in _LEO_VIEWS[left_index + 1 :]:
            singular = torch.linalg.svdvals(scene_bases[left].transpose(0, 1) @ scene_bases[right]).clamp(0.0, 1.0)
            angle = torch.rad2deg(torch.acos(singular.min()))
            angles[f"{left}__{right}"] = float(angle)
    return {
        "schema": "cvs.factored_slow_fast.basis_diagnostics.v1",
        "explained_ratio_by_scene": explained,
        "principal_angle_deg": angles,
    }


__all__ = [
    "FactoredSlowFastState",
    "apply_factored_context",
    "basis_scene_diagnostics",
    "fit_factored_state",
    "solve_factored_context",
    "support_safety_diagnostics",
]
