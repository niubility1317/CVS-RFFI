"""Role-blind MIRAGE open-world geometry and three-state decision boundary.

The head consumes only a frozen normalized identity embedding, registered
class geometry, and an optional episode mask.  It deliberately has no access
to source samples, target roles, truth, quotas, or threshold-search logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


UNKNOWN_LABEL = -1
DEFER_LABEL = -2

_DIAG_FLOOR = 1e-4
_NORMALIZATION_TOLERANCE = 1e-3


@dataclass(frozen=True)
class DecisionThresholds:
    """Frozen global thresholds for quality, registration, and unknown risk."""

    tau_q: float
    tau_reg: float
    tau_unk: float

    def __post_init__(self) -> None:
        for field_name in ("tau_q", "tau_reg", "tau_unk"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
        if float(self.tau_reg) > float(self.tau_unk):
            raise ValueError("tau_reg must not exceed tau_unk")


@dataclass
class OpenHeadOutput:
    """Per-class geometry plus scalar open-world evidence for one batch."""

    class_scores: Tensor
    class_distances: Tensor
    radius_margins: Tensor
    energy: Tensor
    unknown_risk: Tensor


@dataclass
class DecisionResult:
    """Independent per-query registered, unknown, or defer decisions."""

    labels: Tensor
    explicit_unknown: Tensor
    registered: Tensor
    deferred: Tensor


def _require_positive_integer(value: int, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        lower_bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {lower_bound}")


def _require_finite(value: Tensor, name: str) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite tensor at {name}")
    return value


class MIRAGEOpenHead(nn.Module):
    """Score frozen identity embeddings against PSD per-class open-world geometry.

    The precision matrix for registered class ``c`` is represented as
    ``diag(softplus(log_diag_precision[c]) + 1e-4) + L_c L_c^T``.  This keeps
    the Mahalanobis form PSD for both diagonal-only and optional low-rank use.
    """

    def __init__(self, *, num_classes: int, feature_dim: int, covariance_rank: int = 0) -> None:
        super().__init__()
        _require_positive_integer(num_classes, "num_classes")
        _require_positive_integer(feature_dim, "feature_dim")
        _require_positive_integer(covariance_rank, "covariance_rank", allow_zero=True)

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.covariance_rank = covariance_rank
        self.prototypes = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        self.log_radius = nn.Parameter(torch.zeros(num_classes))
        self.log_diag_precision = nn.Parameter(torch.zeros(num_classes, feature_dim))
        if covariance_rank == 0:
            self.register_parameter("low_rank_factor", None)
        else:
            self.low_rank_factor = nn.Parameter(torch.zeros(num_classes, feature_dim, covariance_rank))
        self.log_risk_weights = nn.Parameter(torch.zeros(3))
        self.risk_bias = nn.Parameter(torch.zeros(()))

    def _validate_z_id(self, z_id: Tensor) -> None:
        if not isinstance(z_id, Tensor):
            raise TypeError("z_id must be a torch.Tensor")
        if not z_id.is_floating_point():
            raise TypeError("z_id must use a floating dtype")
        if z_id.ndim != 2 or z_id.shape[1] != self.feature_dim:
            raise ValueError(f"z_id must have shape [B, D] with D={self.feature_dim}")
        if z_id.shape[0] < 1:
            raise ValueError("z_id batch must be non-empty")
        if z_id.device != self.prototypes.device:
            raise ValueError("z_id device must match head parameters")
        if not bool(torch.isfinite(z_id).all()):
            raise ValueError("z_id must be finite")
        norms = torch.linalg.vector_norm(z_id, dim=1)
        if not bool(torch.isfinite(norms).all()):
            raise ValueError("z_id norms must be finite")
        if not bool(torch.allclose(norms, torch.ones_like(norms), rtol=_NORMALIZATION_TOLERANCE, atol=_NORMALIZATION_TOLERANCE)):
            raise ValueError("z_id rows must be approximately L2-normalized")

    def _registered_indices(self, class_mask: Tensor | None, *, device: torch.device) -> Tensor:
        if class_mask is None:
            return torch.arange(self.num_classes, device=device)
        if not isinstance(class_mask, Tensor):
            raise TypeError("class_mask must be a torch.Tensor")
        if class_mask.dtype != torch.bool:
            raise TypeError("class_mask must use torch.bool")
        if class_mask.ndim != 1 or class_mask.shape[0] != self.num_classes:
            raise ValueError(f"class_mask must have shape [{self.num_classes}]")
        if class_mask.device != device:
            raise ValueError("class_mask device must match z_id")
        registered_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
        if registered_indices.numel() == 0:
            raise ValueError("class_mask must retain at least one registered class")
        return registered_indices

    def forward(self, z_id: Tensor, *, class_mask: Tensor | None = None) -> OpenHeadOutput:
        """Return masked class geometry and role-blind unknown evidence for ``z_id``."""

        self._validate_z_id(z_id)
        registered_indices = self._registered_indices(class_mask, device=z_id.device)
        dtype = z_id.dtype
        batch_size = z_id.shape[0]

        # Select before computing any geometry: an episode's proxy row must not
        # affect its scores, energy, minimum distance, or unknown risk.
        active_prototypes = self.prototypes.index_select(0, registered_indices).to(dtype=dtype)
        prototype_norms = torch.linalg.vector_norm(active_prototypes, dim=1, keepdim=True)
        _require_finite(prototype_norms, "prototype_norms")
        if bool((prototype_norms <= torch.finfo(dtype).eps).any()):
            raise FloatingPointError("registered prototype has zero norm")
        active_prototypes = active_prototypes / prototype_norms
        _require_finite(active_prototypes, "normalized_prototypes")

        active_radii = functional.softplus(
            self.log_radius.index_select(0, registered_indices).to(dtype=dtype)
        )
        active_precision = functional.softplus(
            self.log_diag_precision.index_select(0, registered_indices).to(dtype=dtype)
        ) + _DIAG_FLOOR
        _require_finite(active_radii, "radii")
        _require_finite(active_precision, "diagonal_precision")

        deltas = z_id[:, None, :] - active_prototypes[None, :, :]
        active_distances = (deltas.square() * active_precision[None, :, :]).sum(dim=-1)
        if self.low_rank_factor is not None:
            active_factor = self.low_rank_factor.index_select(0, registered_indices).to(dtype=dtype)
            low_rank_projection = torch.einsum("bad,adr->bar", deltas, active_factor)
            active_distances = active_distances + low_rank_projection.square().sum(dim=-1)
        _require_finite(active_distances, "class_distances")
        active_margins = active_distances - active_radii[None, :]
        active_scores = z_id @ active_prototypes.transpose(0, 1)
        _require_finite(active_margins, "radius_margins")
        _require_finite(active_scores, "class_scores")

        fill_low = torch.finfo(dtype).min
        fill_high = torch.finfo(dtype).max
        class_scores = torch.full((batch_size, self.num_classes), fill_low, dtype=dtype, device=z_id.device)
        class_distances = torch.full((batch_size, self.num_classes), fill_high, dtype=dtype, device=z_id.device)
        radius_margins = torch.full((batch_size, self.num_classes), fill_high, dtype=dtype, device=z_id.device)
        class_scores = class_scores.index_copy(1, registered_indices, active_scores)
        class_distances = class_distances.index_copy(1, registered_indices, active_distances)
        radius_margins = radius_margins.index_copy(1, registered_indices, active_margins)

        energy = -torch.logsumexp(active_scores, dim=1)
        minimum_distance = active_distances.amin(dim=1)
        minimum_margin = active_margins.amin(dim=1)
        normalized_minimum_distance = minimum_distance / (1.0 + minimum_distance)
        normalized_positive_margin = functional.relu(minimum_margin)
        normalized_positive_margin = normalized_positive_margin / (1.0 + normalized_positive_margin)
        normalized_energy = functional.softplus(energy)
        risk_weights = functional.softplus(self.log_risk_weights.to(dtype=dtype))
        risk_logit = (
            risk_weights[0] * normalized_minimum_distance
            + risk_weights[1] * normalized_positive_margin
            + risk_weights[2] * normalized_energy
            + self.risk_bias.to(dtype=dtype)
        )
        unknown_risk = torch.sigmoid(risk_logit)
        _require_finite(energy, "energy")
        _require_finite(unknown_risk, "unknown_risk")
        return OpenHeadOutput(
            class_scores=class_scores,
            class_distances=class_distances,
            radius_margins=radius_margins,
            energy=energy,
            unknown_risk=unknown_risk,
        )


def _validate_decision_output(output: OpenHeadOutput) -> tuple[int, int]:
    if not isinstance(output, OpenHeadOutput):
        raise TypeError("output must be an OpenHeadOutput")
    tensors = {
        "class_scores": output.class_scores,
        "class_distances": output.class_distances,
        "radius_margins": output.radius_margins,
        "energy": output.energy,
        "unknown_risk": output.unknown_risk,
    }
    for name, value in tensors.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    if output.class_scores.ndim != 2:
        raise ValueError("class_scores must have shape [B, C]")
    batch_size, num_classes = output.class_scores.shape
    if batch_size < 1 or num_classes < 1:
        raise ValueError("class_scores must have non-empty batch and class dimensions")
    for name in ("class_distances", "radius_margins"):
        if tensors[name].shape != (batch_size, num_classes):
            raise ValueError(f"{name} must have shape [B, C]")
    for name in ("energy", "unknown_risk"):
        if tensors[name].shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")
    if not bool(((output.unknown_risk >= 0.0) & (output.unknown_risk <= 1.0)).all()):
        raise ValueError("unknown_risk must lie in [0, 1]")
    return batch_size, num_classes


def decide(output: OpenHeadOutput, *, quality: Tensor, thresholds: DecisionThresholds) -> DecisionResult:
    """Apply the frozen per-query quality/registered/unknown/defer policy.

    Low quality always defers.  A registered label needs both low risk and an
    in-radius highest-score class.  Explicit unknown additionally requires the
    query to fall outside every registered class radius; all remaining cases
    defer and never count as explicit unknown.
    """

    batch_size, _ = _validate_decision_output(output)
    if not isinstance(thresholds, DecisionThresholds):
        raise TypeError("thresholds must be DecisionThresholds")
    if not isinstance(quality, Tensor):
        raise TypeError("quality must be a torch.Tensor")
    if not quality.is_floating_point():
        raise TypeError("quality must use a floating dtype")
    if quality.ndim != 1 or quality.shape[0] != batch_size:
        raise ValueError("quality must have shape [B]")
    if quality.device != output.class_scores.device:
        raise ValueError("quality device must match output")
    if not bool(torch.isfinite(quality).all()):
        raise ValueError("quality must be finite")
    if not bool(((quality >= 0.0) & (quality <= 1.0)).all()):
        raise ValueError("quality must lie in [0, 1]")

    best_labels = output.class_scores.argmax(dim=1)
    best_margins = output.radius_margins.gather(1, best_labels[:, None]).squeeze(1)
    outside_registered_support = output.radius_margins.amin(dim=1) > 0.0
    quality_ok = quality >= float(thresholds.tau_q)
    registered = quality_ok & (output.unknown_risk <= float(thresholds.tau_reg)) & (best_margins <= 0.0)
    explicit_unknown = (
        quality_ok
        & ~registered
        & (output.unknown_risk >= float(thresholds.tau_unk))
        & outside_registered_support
    )
    deferred = ~(registered | explicit_unknown)
    labels = torch.full_like(best_labels, DEFER_LABEL)
    labels = torch.where(registered, best_labels, labels)
    labels = torch.where(explicit_unknown, torch.full_like(labels, UNKNOWN_LABEL), labels)
    return DecisionResult(
        labels=labels,
        explicit_unknown=explicit_unknown,
        registered=registered,
        deferred=deferred,
    )


__all__ = [
    "DEFER_LABEL",
    "UNKNOWN_LABEL",
    "DecisionResult",
    "DecisionThresholds",
    "MIRAGEOpenHead",
    "OpenHeadOutput",
    "decide",
]
