"""Feature-level few-shot adaptation utilities for spaceborne CVS-RFFI.

The functions in this module operate on already extracted `z_id` features. They
are intentionally small and side-effect free so the new-TX enrollment and target
receiver calibration logic can be validated before launching full model runs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


UNKNOWN_LABEL = -1
LIFECYCLE_QUARANTINE = "quarantine"
LIFECYCLE_ACTIVE_LOCAL = "active_local"
LIFECYCLE_GROUND_CONFIRMED = "ground_confirmed"
LIFECYCLE_REVOKED = "revoked"


@dataclass(frozen=True)
class OpenSetGateConfig:
    """Open-set decision policy over prototype scores.

    `cosine` preserves the historical max-cosine threshold behavior. `combined`
    applies every configured gate that has a usable threshold/statistic.
    Mahalanobis and OpenMax-style gates use per-class statistics stored in the
    PrototypeSet metadata when explicit thresholds are not supplied.
    """

    mode: str = "cosine"
    min_cosine: float | None = None
    min_margin: float | None = None
    max_mahalanobis: float | None = None
    openmax_tail_size: int = 20
    openmax_quantile: float = 0.95
    openmax_min_threshold: float = 0.02
    mahalanobis_eps: float = 1e-4


@dataclass
class PrototypeSet:
    labels: torch.Tensor
    vectors: torch.Tensor
    counts: torch.Tensor
    metadata: dict = field(default_factory=dict)

    def index_of(self, label: int) -> int:
        matches = (self.labels.cpu() == int(label)).nonzero(as_tuple=False).flatten()
        if matches.numel() != 1:
            raise KeyError(f"prototype label not found or duplicated: {label}")
        return int(matches[0].item())

    def label_values(self) -> set[int]:
        return {int(v) for v in self.labels.cpu().tolist()}


@dataclass
class ClassState:
    class_id: int
    group: str
    prototype: torch.Tensor
    mask: torch.Tensor
    subspace: torch.Tensor
    covariance_diag: torch.Tensor
    thresholds: dict = field(default_factory=dict)
    evt_params: dict = field(default_factory=dict)
    support_quality: float = 1.0
    source_weight: float = 0.0
    support_anchors: torch.Tensor | None = None


@dataclass
class SiameseAnchorVerifier:
    anchor_features: torch.Tensor
    anchor_labels: torch.Tensor
    threshold: float
    scale: float = 20.0

    def same_probability(self, features: torch.Tensor, labels: torch.Tensor | Iterable[int]) -> torch.Tensor:
        x = normalize_rows(torch.as_tensor(features).float())
        labels = _labels_tensor(labels, device=x.device)
        anchors = normalize_rows(torch.as_tensor(self.anchor_features).float()).to(x.device)
        anchor_labels = _labels_tensor(self.anchor_labels, device=x.device)
        probs = []
        for row, label in zip(x, labels):
            mask = anchor_labels == int(label.item())
            if not bool(mask.any().item()):
                probs.append(torch.tensor(0.0, dtype=x.dtype, device=x.device))
                continue
            sim = torch.max(anchors[mask] @ row)
            probs.append(torch.sigmoid(float(self.scale) * (sim - float(self.threshold))))
        return torch.stack(probs, dim=0)


@dataclass
class PredictionResult:
    predicted_labels: torch.Tensor
    scores: torch.Tensor
    accepted: torch.Tensor
    candidate_labels: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
    margins: torch.Tensor | None = None
    mahalanobis: torch.Tensor | None = None
    openmax_distance: torch.Tensor | None = None
    gate_reasons: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    energy: torch.Tensor | None = None
    subspace_residual: torch.Tensor | None = None
    seen_new_evidence: torch.Tensor | None = None
    seen_new_support_affinity: torch.Tensor | None = None
    seen_new_support_residual: torch.Tensor | None = None
    seen_new_anchor_similarity: torch.Tensor | None = None
    seen_new_anchor_delta: torch.Tensor | None = None


@dataclass
class AdaptationResult:
    prototype_set: PrototypeSet
    predicted_labels: torch.Tensor
    scores: torch.Tensor
    accepted: torch.Tensor
    metrics: dict[str, float]
    telemetry: dict = field(default_factory=dict)
    candidate_labels: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
    margins: torch.Tensor | None = None
    mahalanobis: torch.Tensor | None = None
    openmax_distance: torch.Tensor | None = None
    gate_reasons: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    energy: torch.Tensor | None = None
    subspace_residual: torch.Tensor | None = None
    seen_new_evidence: torch.Tensor | None = None
    seen_new_support_affinity: torch.Tensor | None = None
    seen_new_support_residual: torch.Tensor | None = None
    seen_new_anchor_similarity: torch.Tensor | None = None
    seen_new_anchor_delta: torch.Tensor | None = None


@dataclass
class NewClassRecord:
    label: int
    state: str
    support_count: int
    prototype_version: int = 1
    adapter_version: int = 0
    reason: str = ""
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class NewClassLifecycleManager:
    """Small deterministic state machine for on-orbit new-TX enrollment."""

    allowed_transitions = {
        LIFECYCLE_QUARANTINE: {LIFECYCLE_ACTIVE_LOCAL, LIFECYCLE_REVOKED},
        LIFECYCLE_ACTIVE_LOCAL: {LIFECYCLE_GROUND_CONFIRMED, LIFECYCLE_REVOKED, LIFECYCLE_QUARANTINE},
        LIFECYCLE_GROUND_CONFIRMED: {LIFECYCLE_REVOKED},
        LIFECYCLE_REVOKED: set(),
    }

    def enroll(
        self,
        label: int,
        *,
        support_count: int,
        initial_state: str = LIFECYCLE_QUARANTINE,
        prototype_version: int = 1,
        reason: str = "fewshot_support_enrollment",
    ) -> NewClassRecord:
        if initial_state not in self.allowed_transitions:
            raise ValueError(f"unknown lifecycle state: {initial_state}")
        record = NewClassRecord(
            label=int(label),
            state=LIFECYCLE_QUARANTINE,
            support_count=int(support_count),
            prototype_version=int(prototype_version),
            reason=str(reason),
            history=[
                {
                    "from": None,
                    "to": LIFECYCLE_QUARANTINE,
                    "reason": str(reason),
                    "prototype_version": int(prototype_version),
                }
            ],
        )
        if initial_state != LIFECYCLE_QUARANTINE:
            record = self.transition(record, initial_state, reason=f"initial_{initial_state}")
        return record

    def transition(
        self,
        record: NewClassRecord,
        new_state: str,
        *,
        reason: str = "",
        prototype_version: int | None = None,
        adapter_version: int | None = None,
    ) -> NewClassRecord:
        new_state = str(new_state)
        if new_state not in self.allowed_transitions:
            raise ValueError(f"unknown lifecycle state: {new_state}")
        if new_state not in self.allowed_transitions.get(record.state, set()):
            raise ValueError(f"invalid lifecycle transition: {record.state} -> {new_state}")
        updated = NewClassRecord(
            label=int(record.label),
            state=new_state,
            support_count=int(record.support_count),
            prototype_version=int(record.prototype_version if prototype_version is None else prototype_version),
            adapter_version=int(record.adapter_version if adapter_version is None else adapter_version),
            reason=str(reason),
            history=list(record.history),
        )
        updated.history.append(
            {
                "from": record.state,
                "to": new_state,
                "reason": str(reason),
                "prototype_version": updated.prototype_version,
                "adapter_version": updated.adapter_version,
            }
        )
        return updated


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected [N, D] tensor, got shape={tuple(x.shape)}")
    return F.normalize(x.float(), p=2, dim=1, eps=eps)


def _labels_tensor(labels: torch.Tensor | Iterable[int], *, device=None) -> torch.Tensor:
    out = torch.as_tensor(labels, dtype=torch.long, device=device)
    if out.ndim != 1:
        raise ValueError(f"expected [N] labels, got shape={tuple(out.shape)}")
    return out


def _normalize_stage_name(stage: str) -> str:
    token = str(stage).strip().upper().replace("_", "-")
    if token in {"A", "STAGE2-A", "STAGE2A", "STAGE2-A-ZERO-LABEL-DEPLOY"}:
        return "Stage2-A"
    if token in {"B", "STAGE2-B", "STAGE2B", "STAGE2-B-OLD-LABEL-CALIBRATION"}:
        return "Stage2-B"
    if token in {"C", "STAGE2-C", "STAGE2C", "STAGE2-C-OLD-NEW-ENROLLMENT"}:
        return "Stage2-C"
    raise ValueError(f"unknown Stage2 protocol stage: {stage}")


def validate_stage2_protocol(
    stage: str,
    *,
    use_target_old_support: bool = False,
    use_target_new_support: bool = False,
    use_unknown_query_for_threshold_calibration: bool = False,
) -> dict:
    """Validate Stage2 data visibility before calibration or registration."""

    normalized = _normalize_stage_name(stage)
    if normalized == "Stage2-A" and bool(use_target_old_support):
        raise ValueError("Stage2-A cannot use target-old support")
    if normalized in {"Stage2-A", "Stage2-B"} and bool(use_target_new_support):
        raise ValueError(f"{normalized} cannot use target-new support")
    if bool(use_unknown_query_for_threshold_calibration):
        raise ValueError("unknown query samples must not be used for threshold calibration")
    return {
        "stage": normalized,
        "target_old_support_allowed": normalized in {"Stage2-B", "Stage2-C"},
        "target_new_support_allowed": normalized == "Stage2-C",
        "unknown_query_threshold_calibration_allowed": False,
    }


def _orthonormal_columns(matrix: torch.Tensor, max_rank: int) -> torch.Tensor:
    matrix = torch.as_tensor(matrix).float()
    if matrix.ndim != 2:
        raise ValueError(f"expected rank-2 matrix, got shape={tuple(matrix.shape)}")
    dim = int(matrix.shape[-1])
    rank = max(0, min(int(max_rank), dim, int(matrix.shape[0])))
    if rank <= 0 or matrix.numel() == 0:
        return torch.zeros(dim, 0, dtype=matrix.dtype, device=matrix.device)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    if torch.allclose(centered, torch.zeros_like(centered)):
        centered = matrix
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:rank].T.contiguous()
    except RuntimeError:
        basis, _ = torch.linalg.qr(centered.T, mode="reduced")
        basis = basis[:, :rank].contiguous()
    return F.normalize(basis, p=2, dim=0, eps=1e-12)


def estimate_orbit_subspace(
    source_prototypes: PrototypeSet,
    target_old_support: torch.Tensor,
    target_old_labels: torch.Tensor | Iterable[int],
    *,
    orbit_rank: int = 2,
) -> torch.Tensor:
    """Estimate shared target-old residual directions against source prototypes."""

    dim = int(source_prototypes.vectors.shape[1])
    support = torch.as_tensor(target_old_support).float().to(source_prototypes.vectors.device)
    labels = _labels_tensor(target_old_labels, device=source_prototypes.vectors.device)
    if support.numel() == 0 or labels.numel() == 0 or int(orbit_rank) <= 0:
        return torch.zeros(dim, 0, dtype=source_prototypes.vectors.dtype, device=source_prototypes.vectors.device)
    support = normalize_rows(support)
    residuals = []
    for feature, label in zip(support, labels):
        try:
            idx = source_prototypes.index_of(int(label.item()))
        except KeyError:
            continue
        residuals.append(feature - source_prototypes.vectors[idx].to(device=feature.device, dtype=feature.dtype))
    if not residuals:
        return torch.zeros(dim, 0, dtype=source_prototypes.vectors.dtype, device=source_prototypes.vectors.device)
    return _orthonormal_columns(torch.stack(residuals, dim=0), max_rank=int(orbit_rank))


def _class_conditioned_masks(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_labels: list[int],
    *,
    dim: int,
    active_ratio: float = 0.25,
    eps: float = 1e-6,
) -> dict[int, torch.Tensor]:
    features = torch.as_tensor(features).float()
    labels = _labels_tensor(labels, device=features.device)
    if features.numel() == 0 or labels.numel() == 0:
        return {int(label): torch.ones(dim, dtype=torch.float32, device=features.device) for label in class_labels}
    k = max(1, min(int(dim), int(torch.ceil(torch.tensor(float(active_ratio) * float(dim))).item())))
    out = {}
    global_var = features.var(dim=0, unbiased=False) if features.size(0) > 1 else torch.zeros(dim, device=features.device)
    for label in class_labels:
        class_mask = labels == int(label)
        members = features[class_mask]
        if members.numel() == 0:
            out[int(label)] = torch.ones(dim, dtype=torch.float32, device=features.device)
            continue
        others = features[~class_mask]
        mu = members.mean(dim=0)
        within = members.var(dim=0, unbiased=False) if members.size(0) > 1 else torch.zeros_like(mu)
        if others.numel() > 0:
            sep = (mu - others.mean(dim=0)).abs()
        else:
            sep = mu.abs()
        reliability = sep / (within + global_var + float(eps))
        idx = torch.topk(reliability, k=k, largest=True).indices
        mask = torch.zeros(dim, dtype=torch.float32, device=features.device)
        mask[idx] = 1.0
        out[int(label)] = mask
    return out


def _metadata_threshold(prototypes: PrototypeSet, key: str, label: int) -> float | None:
    value = prototypes.metadata.get(key)
    if not torch.is_tensor(value):
        return None
    try:
        idx = prototypes.index_of(int(label))
    except KeyError:
        return None
    item = value.detach().cpu().reshape(value.shape[0], -1)[idx, 0].item()
    return float(item)


def _metadata_vector(prototypes: PrototypeSet, key: str, label: int, fallback: torch.Tensor) -> torch.Tensor:
    value = prototypes.metadata.get(key)
    if not torch.is_tensor(value):
        return fallback.clone()
    try:
        idx = prototypes.index_of(int(label))
    except KeyError:
        return fallback.clone()
    return value[idx].to(device=fallback.device, dtype=fallback.dtype).clone()


def _class_cosine_distances(features: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    return (1.0 - (features @ normalize_rows(prototype.view(1, -1)).squeeze(0))).clamp_min(0.0)


def _tail_threshold(
    distances: torch.Tensor,
    *,
    tail_size: int,
    quantile: float,
    min_threshold: float,
) -> torch.Tensor:
    if distances.numel() <= 0:
        return torch.tensor(float(min_threshold), dtype=torch.float32)
    tail_size = max(1, min(int(tail_size), int(distances.numel())))
    tail = torch.topk(distances.float(), k=tail_size, largest=True).values
    q = min(1.0, max(0.0, float(quantile)))
    threshold = torch.quantile(tail, q)
    return threshold.clamp_min(float(min_threshold))


def _weibull_cv(shape: float) -> float:
    g1 = math.exp(math.lgamma(1.0 + 1.0 / float(shape)))
    g2 = math.exp(math.lgamma(1.0 + 2.0 / float(shape)))
    return math.sqrt(max(g2 / max(g1 * g1, 1e-12) - 1.0, 0.0))


def fit_weibull_tail(
    distances: torch.Tensor | Iterable[float],
    *,
    tail_size: int = 20,
    target_far: float = 0.05,
    min_scale: float = 1e-6,
) -> dict[str, float]:
    """Fit a two-parameter Weibull tail with method-of-moments.

    This is intentionally dependency-free for on-orbit/edge use. It uses only
    known calibration/support distances; callers must not pass unknown query
    samples into this function.
    """

    values = torch.as_tensor(list(distances) if not torch.is_tensor(distances) else distances, dtype=torch.float32).reshape(-1)
    values = values[torch.isfinite(values)].clamp_min(0.0)
    if values.numel() == 0:
        values = torch.tensor([float(min_scale)], dtype=torch.float32)
    tail_size = max(1, min(int(tail_size), int(values.numel())))
    tail = torch.topk(values, k=tail_size, largest=True).values.clamp_min(float(min_scale))
    mean = float(tail.mean().item())
    std = float(tail.std(unbiased=False).item())
    cv = std / max(mean, float(min_scale))
    if cv <= 1e-6:
        shape = 10.0
    else:
        lo, hi = 0.1, 25.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _weibull_cv(mid) > cv:
                lo = mid
            else:
                hi = mid
        shape = 0.5 * (lo + hi)
    gamma_1 = math.exp(math.lgamma(1.0 + 1.0 / shape))
    scale = max(mean / max(gamma_1, 1e-12), float(min_scale))
    far = min(0.999999, max(1e-6, float(target_far)))
    threshold = scale * ((-math.log(far)) ** (1.0 / shape))
    return {
        "fit": "weibull_moments",
        "shape": float(shape),
        "scale": float(scale),
        "threshold": float(max(threshold, float(tail.max().item()))),
        "tail_size": int(tail_size),
        "target_far": float(target_far),
    }


def _mahalanobis_threshold(members: torch.Tensor, prototype: torch.Tensor, diag_var: torch.Tensor) -> torch.Tensor:
    centered = members - prototype.view(1, -1)
    d2 = (centered.pow(2) / diag_var.view(1, -1)).sum(dim=1)
    if d2.numel() <= 0:
        return torch.tensor(float("nan"), dtype=torch.float32, device=members.device)
    return d2.sqrt().max().clamp_min(1.0)


def build_prototype_set(
    features: torch.Tensor,
    labels: torch.Tensor | Iterable[int],
    *,
    gate_config: OpenSetGateConfig | None = None,
) -> PrototypeSet:
    features = normalize_rows(torch.as_tensor(features).float())
    labels = _labels_tensor(labels, device=features.device)
    if features.size(0) != labels.numel():
        raise ValueError("features and labels have different lengths")

    class_labels = sorted({int(v) for v in labels.cpu().tolist() if int(v) != UNKNOWN_LABEL})
    if not class_labels:
        raise ValueError("no known labels available for prototype construction")

    vectors = []
    counts = []
    diag_vars = []
    openmax_thresholds = []
    mahalanobis_thresholds = []
    support_radius = {}
    gate_config = gate_config or OpenSetGateConfig(mode="cosine")
    for label in class_labels:
        mask = labels == int(label)
        members = features[mask]
        mean = members.mean(dim=0, keepdim=True)
        proto = normalize_rows(mean).squeeze(0)
        centered = members - proto.view(1, -1)
        if int(mask.sum().item()) > 1:
            diag_var = centered.pow(2).mean(dim=0).clamp_min(float(gate_config.mahalanobis_eps))
        else:
            diag_var = torch.full_like(proto, float(gate_config.mahalanobis_eps))
        distances = _class_cosine_distances(members, proto)
        vectors.append(normalize_rows(mean).squeeze(0))
        counts.append(int(mask.sum().item()))
        diag_vars.append(diag_var)
        openmax_thresholds.append(
            _tail_threshold(
                distances,
                tail_size=int(gate_config.openmax_tail_size),
                quantile=float(gate_config.openmax_quantile),
                min_threshold=float(gate_config.openmax_min_threshold),
            ).to(device=features.device)
        )
        mahalanobis_thresholds.append(_mahalanobis_threshold(members, proto, diag_var))
        support_radius[int(label)] = {
            "cosine_distance_max": float(distances.max().item()) if distances.numel() else float("nan"),
            "cosine_distance_mean": float(distances.mean().item()) if distances.numel() else float("nan"),
        }

    return PrototypeSet(
        labels=torch.tensor(class_labels, dtype=torch.long, device=features.device),
        vectors=torch.stack(vectors, dim=0),
        counts=torch.tensor(counts, dtype=torch.long, device=features.device),
        metadata={
            "diag_var": torch.stack(diag_vars, dim=0),
            "openmax_thresholds": torch.stack(openmax_thresholds, dim=0),
            "mahalanobis_thresholds": torch.stack(mahalanobis_thresholds, dim=0),
            "support_radius": support_radius,
            "gate_fit": {
                "openmax_tail_size": int(gate_config.openmax_tail_size),
                "openmax_quantile": float(gate_config.openmax_quantile),
                "openmax_min_threshold": float(gate_config.openmax_min_threshold),
                "mahalanobis_eps": float(gate_config.mahalanobis_eps),
            },
        },
    )


def _cat_metadata_tensor(source: PrototypeSet, new: PrototypeSet, key: str) -> torch.Tensor | None:
    left = source.metadata.get(key)
    right = new.metadata.get(key)
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.cat([left.to(source.vectors.device), right.to(source.vectors.device)], dim=0)
    return None


def merge_prototype_sets(source: PrototypeSet, new: PrototypeSet) -> PrototypeSet:
    overlap = source.label_values() & new.label_values()
    if overlap:
        raise ValueError(f"new prototypes overlap source labels: {sorted(overlap)}")
    metadata = {"source_labels": sorted(source.label_values()), "new_labels": sorted(new.label_values())}
    for key in ("diag_var", "openmax_thresholds", "mahalanobis_thresholds"):
        value = _cat_metadata_tensor(source, new, key)
        if value is not None:
            metadata[key] = value
    metadata["support_radius"] = {
        **dict(source.metadata.get("support_radius", {})),
        **dict(new.metadata.get("support_radius", {})),
    }
    if "gate_fit" in source.metadata or "gate_fit" in new.metadata:
        metadata["gate_fit"] = {"source": source.metadata.get("gate_fit", {}), "new": new.metadata.get("gate_fit", {})}
    return PrototypeSet(
        labels=torch.cat([source.labels, new.labels], dim=0),
        vectors=torch.cat([source.vectors, new.vectors], dim=0),
        counts=torch.cat([source.counts, new.counts], dim=0),
        metadata=metadata,
    )


class OrbitAdaptiveMSEHead(nn.Module):
    """Feature-level Orbit-Adaptive Masked Subspace Energy head."""

    def __init__(
        self,
        dim: int,
        class_states: Mapping[int, ClassState],
        *,
        alpha_cosine: float = 1.0,
        beta_residual: float = 1.0,
        eta_mahalanobis: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.class_states = {int(k): v for k, v in class_states.items()}
        if not self.class_states:
            raise ValueError("OrbitAdaptiveMSEHead requires at least one class state")
        self.class_order = sorted(self.class_states)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_cosine), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(float(beta_residual), dtype=torch.float32))
        self.eta = nn.Parameter(torch.tensor(float(eta_mahalanobis), dtype=torch.float32))
        self.temperature = nn.Parameter(torch.tensor(float(temperature), dtype=torch.float32))

    def _soft_mixture_metrics(
        self,
        h: torch.Tensor,
        state: ClassState,
        prototype: torch.Tensor,
        mask: torch.Tensor,
        covariance: torch.Tensor,
        subspace: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if not bool(state.thresholds.get("soft_mixture_score_enabled", False)):
            return None
        anchors = state.support_anchors
        if anchors is None or not hasattr(anchors, "numel") or int(anchors.numel()) == 0:
            return None
        weight = max(0.0, min(1.0, float(state.thresholds.get("soft_mixture_score_weight", 1.0))))
        if weight <= 0.0:
            return None
        anchor_bank = torch.as_tensor(anchors, dtype=h.dtype, device=h.device)
        if anchor_bank.ndim == 1:
            anchor_bank = anchor_bank.view(1, -1)
        if anchor_bank.ndim != 2 or anchor_bank.size(1) != self.dim:
            return None
        bank = normalize_rows(torch.cat([prototype, anchor_bank], dim=0))
        max_anchors = int(state.thresholds.get("soft_mixture_max_anchors", bank.size(0)))
        if max_anchors > 0 and bank.size(0) > max_anchors:
            bank = torch.cat([bank[:1], bank[1:max_anchors]], dim=0)
        k = max(1, min(int(state.thresholds.get("soft_mixture_topk", 2)), int(bank.size(0))))
        temp = max(1e-4, float(state.thresholds.get("soft_mixture_temperature", 0.10)))
        masked_h = mask * h
        masked_bank = mask * bank
        sim = F.cosine_similarity(masked_h[:, None, :], masked_bank[None, :, :], dim=-1, eps=1e-12)
        values, indices = sim.topk(k, dim=1)
        weights = torch.softmax(values / temp, dim=1)
        selected = bank[indices]
        mixture = normalize_rows((weights.unsqueeze(-1) * selected).sum(dim=1))
        masked_mixture = mask * mixture
        cos = F.cosine_similarity(masked_h, masked_mixture, dim=-1, eps=1e-12)
        delta = mask * (h - mixture)
        if subspace.numel() > 0 and subspace.shape[1] > 0:
            u = F.normalize(subspace, p=2, dim=0, eps=1e-12)
            projection = (delta @ u) @ u.T
        else:
            projection = torch.zeros_like(delta)
        residual = (delta - projection).pow(2).sum(dim=-1)
        maha = (mask * (h - mixture).pow(2) / covariance).sum(dim=-1)
        return {
            "cos": cos,
            "residual": residual,
            "maha": maha,
            "weight": torch.full_like(cos, weight),
            "anchor_count": torch.full_like(cos, float(bank.size(0))),
        }

    def _anchor_density_metrics(
        self,
        h: torch.Tensor,
        state: ClassState,
        prototype: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        anchors = state.support_anchors
        if anchors is None or not hasattr(anchors, "numel") or int(anchors.numel()) == 0:
            return None
        anchor_bank = torch.as_tensor(anchors, dtype=h.dtype, device=h.device)
        if anchor_bank.ndim == 1:
            anchor_bank = anchor_bank.view(1, -1)
        if anchor_bank.ndim != 2 or anchor_bank.size(1) != self.dim:
            return None
        bank = normalize_rows(torch.cat([prototype, anchor_bank], dim=0))
        max_anchors = int(state.thresholds.get("anchor_density_max_anchors", bank.size(0)))
        if max_anchors > 0 and bank.size(0) > max_anchors:
            bank = torch.cat([bank[:1], bank[1:max_anchors]], dim=0)
        k = max(1, min(int(state.thresholds.get("anchor_density_topk", 3)), int(bank.size(0))))
        temp = max(1e-4, float(state.thresholds.get("anchor_density_temperature", 0.08)))
        sim = F.cosine_similarity((mask * h)[:, None, :], (mask * bank)[None, :, :], dim=-1, eps=1e-12)
        values = sim.topk(k, dim=1).values
        density = temp * torch.logsumexp(values / temp, dim=1)
        return {
            "anchor_density": density,
            "anchor_density_best": values[:, 0],
            "anchor_density_count": torch.full_like(density, float(bank.size(0))),
        }

    def compute_class_scores(self, h: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
        h = normalize_rows(torch.as_tensor(h).float())
        if h.size(1) != self.dim:
            raise ValueError(f"expected feature dim {self.dim}, got {h.size(1)}")
        out: dict[int, dict[str, torch.Tensor]] = {}
        for label in self.class_order:
            state = self.class_states[label]
            prototype = torch.as_tensor(state.prototype, dtype=h.dtype, device=h.device).view(1, -1)
            mask = torch.as_tensor(state.mask, dtype=h.dtype, device=h.device).view(1, -1)
            covariance = torch.as_tensor(state.covariance_diag, dtype=h.dtype, device=h.device).view(1, -1).clamp_min(1e-12)
            subspace = torch.as_tensor(state.subspace, dtype=h.dtype, device=h.device)
            if prototype.size(1) != self.dim or mask.size(1) != self.dim or covariance.size(1) != self.dim:
                raise ValueError(f"class {label} state dimension does not match head dim {self.dim}")
            if subspace.numel() > 0 and subspace.shape[0] != self.dim:
                raise ValueError(f"class {label} subspace dimension does not match head dim {self.dim}")

            masked_h = mask * h
            masked_p = mask * prototype
            cos = F.cosine_similarity(masked_h, masked_p.expand_as(masked_h), dim=-1, eps=1e-12)
            delta = mask * (h - prototype)
            if subspace.numel() > 0 and subspace.shape[1] > 0:
                u = F.normalize(subspace, p=2, dim=0, eps=1e-12)
                projection = (delta @ u) @ u.T
            else:
                projection = torch.zeros_like(delta)
            residual = (delta - projection).pow(2).sum(dim=-1)
            maha = (mask * (h - prototype).pow(2) / covariance).sum(dim=-1)
            bias = float(state.thresholds.get("bias", 0.0))
            score = self.alpha * cos - self.beta * residual - self.eta * maha + bias
            row = {
                "score": score,
                "cos": cos,
                "residual": residual,
                "maha": maha,
            }
            mixture = self._soft_mixture_metrics(h, state, prototype, mask, covariance, subspace)
            if mixture is not None:
                mix_score = self.alpha * mixture["cos"] - self.beta * mixture["residual"] - self.eta * mixture["maha"] + bias
                mix_weight = mixture["weight"]
                row.update(
                    {
                        "hard_score": score,
                        "hard_cos": cos,
                        "hard_residual": residual,
                        "hard_maha": maha,
                        "soft_mixture_score": mix_score,
                        "soft_mixture_cos": mixture["cos"],
                        "soft_mixture_residual": mixture["residual"],
                        "soft_mixture_maha": mixture["maha"],
                        "soft_mixture_anchor_count": mixture["anchor_count"],
                        "score": (1.0 - mix_weight) * score + mix_weight * mix_score,
                        "cos": (1.0 - mix_weight) * cos + mix_weight * mixture["cos"],
                        "residual": (1.0 - mix_weight) * residual + mix_weight * mixture["residual"],
                        "maha": (1.0 - mix_weight) * maha + mix_weight * mixture["maha"],
                    }
                )
            density = self._anchor_density_metrics(h, state, prototype, mask)
            if density is not None:
                row.update(density)
            out[int(label)] = row
        return out

    def score_matrix(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
        per_class = self.compute_class_scores(h)
        matrix = torch.stack([per_class[label]["score"] for label in self.class_order], dim=-1)
        return matrix, per_class

    def energy(self, scores: torch.Tensor) -> torch.Tensor:
        temp = F.softplus(self.temperature).clamp_min(1e-4)
        return -temp * torch.logsumexp(scores / temp, dim=-1)


class LowRankTargetAdapter(nn.Module):
    """Small feature-level adapter for on-orbit few-shot support updates."""

    def __init__(self, dim: int, *, rank: int = 2, alpha: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rank = max(1, min(int(rank), int(dim)))
        self.alpha = float(alpha)
        self.down = nn.Linear(self.dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.dim, bias=True)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = normalize_rows(torch.as_tensor(features).float())
        delta = self.up(self.down(x))
        return normalize_rows(x + float(self.alpha) * delta)


class ResidualMLPTargetAdapter(nn.Module):
    """Small nonlinear residual adapter for feature-space repair."""

    def __init__(self, dim: int, *, rank: int = 2, alpha: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        hidden = max(2, min(int(rank) * 2, int(dim)))
        self.rank = hidden
        self.alpha = float(alpha)
        self.net = nn.Sequential(
            nn.Linear(self.dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, self.dim, bias=True),
        )
        nn.init.normal_(self.net[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = normalize_rows(torch.as_tensor(features).float())
        delta = self.net(x)
        return normalize_rows(x + float(self.alpha) * delta)


def _prototype_bank_for_target_adapter(
    source_prototypes: PrototypeSet,
    support: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    old_labels = source_prototypes.label_values()
    class_labels = sorted({int(v) for v in labels.detach().cpu().tolist() if int(v) != UNKNOWN_LABEL} | old_labels)
    vectors = []
    for label in class_labels:
        if int(label) in old_labels:
            vectors.append(source_prototypes.vectors[source_prototypes.index_of(int(label))].detach().float())
        else:
            members = support[labels == int(label)]
            vectors.append(normalize_rows(members.mean(dim=0, keepdim=True)).squeeze(0).detach().float())
    label_tensor = torch.tensor(class_labels, dtype=torch.long, device=support.device)
    proto = normalize_rows(torch.stack(vectors, dim=0).to(support.device))
    mapped = torch.tensor([class_labels.index(int(v)) for v in labels.detach().cpu().tolist()], dtype=torch.long, device=support.device)
    return label_tensor, proto, mapped


def _class_constrained_soft_prototype_loss(
    adapter: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor | Iterable[int],
    anchor_features: torch.Tensor,
    anchor_labels: torch.Tensor | Iterable[int],
    *,
    topk: int = 2,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Softly assign known samples to same-class anchors without cross-class mixing."""

    features = normalize_rows(torch.as_tensor(features).float())
    labels = _labels_tensor(labels, device=features.device)
    anchors = normalize_rows(torch.as_tensor(anchor_features).float()).to(features.device)
    anchor_labels = _labels_tensor(anchor_labels, device=features.device)
    zero = torch.zeros((), dtype=features.dtype, device=features.device)
    if features.numel() == 0 or anchors.numel() == 0 or labels.numel() == 0 or anchor_labels.numel() == 0:
        return zero
    known = labels != UNKNOWN_LABEL
    anchor_known = anchor_labels != UNKNOWN_LABEL
    if not bool(known.any().item()) or not bool(anchor_known.any().item()):
        return zero
    features = features[known]
    labels = labels[known]
    anchors = anchors[anchor_known]
    anchor_labels = anchor_labels[anchor_known]
    same_class = labels.view(-1, 1) == anchor_labels.view(1, -1)
    if not bool(same_class.any().item()):
        return zero

    tau = max(float(temperature), 1.0e-4)
    with torch.no_grad():
        target_logits = (features.detach() @ anchors.detach().T) / tau
        target_logits = target_logits.masked_fill(~same_class, -float("inf"))
        k = max(1, min(int(topk), target_logits.size(1)))
        cutoff = target_logits.topk(k=k, dim=1).values[:, -1].view(-1, 1)
        target_logits = target_logits.masked_fill(target_logits < cutoff, -float("inf"))
        valid = torch.isfinite(target_logits).any(dim=1)
        if not bool(valid.any().item()):
            return zero
        target = torch.zeros_like(target_logits)
        target[valid] = F.softmax(target_logits[valid], dim=1)

    adapted_features = normalize_rows(adapter(features))
    adapted_anchors = normalize_rows(adapter(anchors.detach()))
    log_probs = F.log_softmax((adapted_features @ adapted_anchors.T) / tau, dim=1)
    return -(target[valid] * log_probs[valid]).sum(dim=1).mean()


def _soft_prototype_mixture_boundary_loss(
    adapter: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor | Iterable[int],
    anchor_features: torch.Tensor,
    anchor_labels: torch.Tensor | Iterable[int],
    *,
    topk: int = 2,
    temperature: float = 0.10,
    margin: float = 0.15,
) -> torch.Tensor:
    """Pull samples to a same-class prototype mixture and away from other classes."""

    features = normalize_rows(torch.as_tensor(features).float())
    labels = _labels_tensor(labels, device=features.device)
    anchors = normalize_rows(torch.as_tensor(anchor_features).float()).to(features.device)
    anchor_labels = _labels_tensor(anchor_labels, device=features.device)
    zero = torch.zeros((), dtype=features.dtype, device=features.device)
    if features.numel() == 0 or anchors.numel() == 0 or labels.numel() == 0 or anchor_labels.numel() == 0:
        return zero
    known = labels != UNKNOWN_LABEL
    anchor_known = anchor_labels != UNKNOWN_LABEL
    if not bool(known.any().item()) or not bool(anchor_known.any().item()):
        return zero
    features = features[known]
    labels = labels[known]
    anchors = anchors[anchor_known]
    anchor_labels = anchor_labels[anchor_known]
    same_class = labels.view(-1, 1) == anchor_labels.view(1, -1)
    other_class = ~same_class
    valid = same_class.any(dim=1)
    if not bool(valid.any().item()):
        return zero

    tau = max(float(temperature), 1.0e-4)
    with torch.no_grad():
        raw_logits = (features.detach() @ anchors.detach().T) / tau
        target_logits = raw_logits.masked_fill(~same_class, -float("inf"))
        k = max(1, min(int(topk), target_logits.size(1)))
        cutoff = target_logits.topk(k=k, dim=1).values[:, -1].view(-1, 1)
        target_logits = target_logits.masked_fill(target_logits < cutoff, -float("inf"))
        valid = torch.isfinite(target_logits).any(dim=1)
        if not bool(valid.any().item()):
            return zero
        target = torch.zeros_like(target_logits)
        target[valid] = F.softmax(target_logits[valid], dim=1)

    adapted_features = normalize_rows(adapter(features[valid]))
    adapted_anchors = normalize_rows(adapter(anchors.detach()))
    mixture = normalize_rows(target[valid] @ adapted_anchors)
    mixture_similarity = F.cosine_similarity(adapted_features, mixture, dim=-1, eps=1e-12)
    pull_loss = (1.0 - mixture_similarity).pow(2).mean()

    other_mask = other_class[valid]
    if not bool(other_mask.any().item()):
        return pull_loss
    other_scores = adapted_features @ adapted_anchors.T
    other_scores = other_scores.masked_fill(~other_mask, -float("inf"))
    best_other = other_scores.max(dim=1).values
    finite = torch.isfinite(best_other)
    if not bool(finite.any().item()):
        return pull_loss
    boundary_loss = F.relu(best_other[finite] - mixture_similarity[finite] + float(margin)).pow(2).mean()
    return pull_loss + boundary_loss


def _support_center_leave_one_out_loss(
    adapter: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor | Iterable[int],
    *,
    temperature: float = 0.10,
    margin: float = 0.10,
) -> torch.Tensor:
    """Train support samples against leave-one-out class centers."""

    features = normalize_rows(torch.as_tensor(features).float())
    labels = _labels_tensor(labels, device=features.device)
    zero = torch.zeros((), dtype=features.dtype, device=features.device)
    if features.numel() == 0 or labels.numel() == 0 or features.size(0) != labels.numel():
        return zero
    keep = labels != UNKNOWN_LABEL
    if not bool(keep.any().item()):
        return zero
    features = features[keep]
    labels = labels[keep]
    class_values = sorted({int(v) for v in labels.detach().cpu().tolist()})
    if len(class_values) < 2:
        return zero

    adapted = normalize_rows(adapter(features))
    tau = max(float(temperature), 1.0e-4)
    logits_rows: list[torch.Tensor] = []
    targets: list[int] = []
    for row_idx in range(adapted.size(0)):
        centers = []
        row_label = int(labels[row_idx].item())
        for class_idx, label_value in enumerate(class_values):
            mask = labels == int(label_value)
            if int(label_value) == row_label and int(mask.sum().item()) > 1:
                mask = mask.clone()
                mask[row_idx] = False
            centers.append(normalize_rows(adapted[mask].mean(dim=0, keepdim=True)).squeeze(0))
            if int(label_value) == row_label:
                targets.append(class_idx)
        center_matrix = torch.stack(centers, dim=0)
        logits_rows.append(adapted[row_idx].view(1, -1) @ center_matrix.T)
    logits = torch.cat(logits_rows, dim=0)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=features.device)
    ce_loss = F.cross_entropy(logits / tau, target_tensor)
    true_score = logits.gather(1, target_tensor.view(-1, 1)).squeeze(1)
    other_mask = torch.ones_like(logits, dtype=torch.bool)
    other_mask.scatter_(1, target_tensor.view(-1, 1), False)
    best_other = logits.masked_fill(~other_mask, -float("inf")).max(dim=1).values
    finite = torch.isfinite(best_other)
    if not bool(finite.any().item()):
        return ce_loss
    margin_loss = F.relu(best_other[finite] - true_score[finite] + float(margin)).pow(2).mean()
    return ce_loss + margin_loss


def _support_leave_one_out_proxy_stats(
    adapter: nn.Module,
    features: torch.Tensor,
    mapped: torch.Tensor,
    labels: torch.Tensor,
    old_labels: set[int],
) -> dict[str, float]:
    """Query-free support generalization proxy for adapter strength selection."""

    features = normalize_rows(torch.as_tensor(features).float())
    mapped = torch.as_tensor(mapped, dtype=torch.long, device=features.device)
    labels = _labels_tensor(labels, device=features.device)
    empty = {
        "support_cv_acc": float("nan"),
        "support_cv_old_acc": float("nan"),
        "support_cv_seen_new_acc": float("nan"),
        "support_cv_margin_mean": float("nan"),
        "support_cv_margin_p10": float("nan"),
        "support_cv_valid_count": 0.0,
    }
    if features.numel() == 0 or mapped.numel() == 0 or labels.numel() == 0:
        return empty
    if features.size(0) != mapped.numel() or mapped.numel() != labels.numel():
        return empty
    keep = labels != UNKNOWN_LABEL
    if not bool(keep.any().item()):
        return empty
    features = features[keep]
    mapped = mapped[keep]
    labels = labels[keep]
    class_values = sorted({int(v) for v in mapped.detach().cpu().tolist()})
    if len(class_values) < 2:
        return empty

    adapted = normalize_rows(adapter(features))
    logits_rows: list[torch.Tensor] = []
    targets: list[int] = []
    row_old_flags: list[bool] = []
    for row_idx in range(adapted.size(0)):
        row_class = int(mapped[row_idx].item())
        same_count = int((mapped == row_class).sum().item())
        if same_count <= 1:
            continue
        centers = []
        target_index = None
        for class_idx, class_value in enumerate(class_values):
            mask = mapped == int(class_value)
            if int(class_value) == row_class:
                mask = mask.clone()
                mask[row_idx] = False
                target_index = class_idx
            if not bool(mask.any().item()):
                centers = []
                break
            centers.append(normalize_rows(adapted[mask].mean(dim=0, keepdim=True)).squeeze(0))
        if not centers or target_index is None:
            continue
        center_matrix = torch.stack(centers, dim=0)
        logits_rows.append(adapted[row_idx].view(1, -1) @ center_matrix.T)
        targets.append(int(target_index))
        row_old_flags.append(int(labels[row_idx].item()) in old_labels)
    if not logits_rows:
        return empty

    logits = torch.cat(logits_rows, dim=0)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=features.device)
    predicted = logits.argmax(dim=1)
    correct = predicted == target_tensor
    true_score = logits.gather(1, target_tensor.view(-1, 1)).squeeze(1)
    other_mask = torch.ones_like(logits, dtype=torch.bool)
    other_mask.scatter_(1, target_tensor.view(-1, 1), False)
    best_other = logits.masked_fill(~other_mask, -float("inf")).max(dim=1).values
    margin = true_score - best_other
    finite = torch.isfinite(margin)
    old_mask = torch.tensor(row_old_flags, dtype=torch.bool, device=features.device)
    seen_new_mask = ~old_mask

    def _masked_acc(mask: torch.Tensor) -> float:
        valid = mask & torch.isfinite(logits).any(dim=1)
        if not bool(valid.any().item()):
            return float("nan")
        return float(correct[valid].float().mean().detach().cpu().item())

    if bool(finite.any().item()):
        margin_finite = margin[finite]
        margin_mean = float(margin_finite.mean().detach().cpu().item())
        margin_p10 = float(torch.quantile(margin_finite, 0.10).detach().cpu().item())
    else:
        margin_mean = float("nan")
        margin_p10 = float("nan")
    return {
        "support_cv_acc": _masked_acc(torch.ones_like(old_mask, dtype=torch.bool)),
        "support_cv_old_acc": _masked_acc(old_mask),
        "support_cv_seen_new_acc": _masked_acc(seen_new_mask),
        "support_cv_margin_mean": margin_mean,
        "support_cv_margin_p10": margin_p10,
        "support_cv_valid_count": float(logits.size(0)),
    }


def _void_background_competition_loss(
    adapter: nn.Module,
    known_features: torch.Tensor,
    known_mapped: torch.Tensor,
    class_prototypes: torch.Tensor,
    pseudo_unknown: torch.Tensor,
    *,
    scale: float = 16.0,
) -> torch.Tensor:
    """Train pseudo-unknown samples as an explicit background competitor."""

    zero = torch.zeros((), dtype=class_prototypes.dtype, device=class_prototypes.device)
    if pseudo_unknown.numel() == 0 or known_features.numel() == 0 or known_mapped.numel() == 0:
        return zero
    known = normalize_rows(torch.as_tensor(known_features).float()).to(class_prototypes.device)
    pseudo = normalize_rows(torch.as_tensor(pseudo_unknown).float()).to(class_prototypes.device)
    mapped = torch.as_tensor(known_mapped, dtype=torch.long, device=class_prototypes.device)
    if known.size(0) != mapped.numel() or class_prototypes.numel() == 0:
        return zero

    adapted_known = normalize_rows(adapter(known))
    adapted_pseudo = normalize_rows(adapter(pseudo))
    void_anchors = adapted_pseudo.detach()
    known_class_logits = float(scale) * (adapted_known @ class_prototypes.T)
    known_void_logit = float(scale) * (adapted_known @ void_anchors.T).max(dim=1).values.view(-1, 1)
    pseudo_class_logits = float(scale) * (adapted_pseudo @ class_prototypes.T)
    pseudo_void_logit = float(scale) * (adapted_pseudo @ void_anchors.T).max(dim=1).values.view(-1, 1)
    known_logits = torch.cat([known_class_logits, known_void_logit], dim=1)
    pseudo_logits = torch.cat([pseudo_class_logits, pseudo_void_logit], dim=1)
    pseudo_labels = torch.full((pseudo_logits.size(0),), int(class_prototypes.size(0)), dtype=torch.long, device=pseudo_logits.device)
    return 0.5 * (F.cross_entropy(known_logits, mapped) + F.cross_entropy(pseudo_logits, pseudo_labels))


def _negative_anchor_background_loss(
    adapter: nn.Module,
    known_features: torch.Tensor,
    known_mapped: torch.Tensor,
    class_prototypes: torch.Tensor,
    pseudo_unknown: torch.Tensor,
    *,
    margin: float = 0.12,
    temperature: float = 0.10,
    max_anchors: int = 256,
) -> tuple[torch.Tensor, int]:
    """Build a query-free background anchor basin while keeping known samples out of it."""

    zero = torch.zeros((), dtype=class_prototypes.dtype, device=class_prototypes.device)
    if pseudo_unknown.numel() == 0 or known_features.numel() == 0 or known_mapped.numel() == 0:
        return zero, 0
    if class_prototypes.numel() == 0:
        return zero, 0
    known = normalize_rows(torch.as_tensor(known_features).float()).to(class_prototypes.device)
    pseudo = normalize_rows(torch.as_tensor(pseudo_unknown).float()).to(class_prototypes.device)
    mapped = torch.as_tensor(known_mapped, dtype=torch.long, device=class_prototypes.device)
    if known.size(0) != mapped.numel():
        return zero, 0

    limit = max(1, int(max_anchors))
    if pseudo.size(0) > limit:
        pseudo = pseudo[:limit]
    adapted_known = normalize_rows(adapter(known))
    adapted_pseudo = normalize_rows(adapter(pseudo))
    bg_centroid = normalize_rows(adapted_pseudo.detach().mean(dim=0, keepdim=True))
    bg_anchors = torch.cat([bg_centroid, adapted_pseudo.detach()], dim=0)
    proto = normalize_rows(class_prototypes).to(class_prototypes.device)
    tau = max(float(temperature), 1.0e-4)

    known_scores = adapted_known @ proto.T
    known_true = known_scores.gather(1, mapped.view(-1, 1)).squeeze(1)
    known_bg = (adapted_known @ bg_anchors.T).max(dim=1).values
    known_margin_loss = F.relu(known_bg - known_true + float(margin)).pow(2).mean()

    pseudo_known = (adapted_pseudo @ proto.T).max(dim=1).values
    pseudo_bg = (adapted_pseudo @ bg_anchors.T).max(dim=1).values
    pseudo_margin_loss = F.relu(pseudo_known - pseudo_bg + float(margin)).pow(2).mean()

    known_logits = torch.stack([known_true, known_bg], dim=1) / tau
    known_targets = torch.zeros((known_logits.size(0),), dtype=torch.long, device=known_logits.device)
    pseudo_logits = torch.stack([pseudo_known, pseudo_bg], dim=1) / tau
    pseudo_targets = torch.ones((pseudo_logits.size(0),), dtype=torch.long, device=pseudo_logits.device)
    ce_loss = 0.5 * (F.cross_entropy(known_logits, known_targets) + F.cross_entropy(pseudo_logits, pseudo_targets))
    return ce_loss + known_margin_loss + pseudo_margin_loss, int(known.size(0) + pseudo.size(0))


def _three_way_decision_head_loss(
    adapter: nn.Module,
    support: torch.Tensor,
    labels: torch.Tensor,
    mapped: torch.Tensor,
    class_labels: torch.Tensor,
    class_prototypes: torch.Tensor,
    pseudo_unknown: torch.Tensor,
    old_labels: set[int],
    *,
    temperature: float = 0.10,
    known_margin: float = 0.08,
    background_margin: float = 0.08,
    support_ce_weight: float = 1.0,
    pseudo_ce_weight: float = 0.35,
    support_background_margin_weight: float = 1.0,
    pseudo_margin_weight: float = 0.50,
) -> tuple[torch.Tensor, int]:
    """Train an explicit old / seen-new / background scoring geometry.

    The background class is built only from query-free pseudo-unknown anchors.
    Unknown query labels stay eval-only; this loss gives rejection a learnable
    surrogate target instead of leaving it to post-hoc thresholds.
    """

    zero = torch.zeros((), dtype=class_prototypes.dtype, device=class_prototypes.device)
    if support.numel() == 0 or labels.numel() == 0 or mapped.numel() == 0 or class_prototypes.numel() == 0:
        return zero, 0
    if pseudo_unknown.numel() == 0 or pseudo_unknown.ndim != 2 or int(pseudo_unknown.shape[1]) != int(support.shape[1]):
        return zero, 0
    old_mask_cols = torch.tensor(
        [int(v) in old_labels for v in class_labels.detach().cpu().tolist()],
        dtype=torch.bool,
        device=class_prototypes.device,
    )
    seen_mask_cols = ~old_mask_cols
    if not bool(old_mask_cols.any().item()):
        return zero, 0

    tau = max(float(temperature), 1.0e-4)
    adapted_support = normalize_rows(adapter(support))
    adapted_pseudo = normalize_rows(adapter(pseudo_unknown.to(support.device)))
    proto = normalize_rows(class_prototypes).to(support.device)
    bg_anchor = adapted_pseudo.detach()

    support_scores = adapted_support @ proto.T
    support_old = support_scores[:, old_mask_cols].max(dim=1).values
    support_bg = (adapted_support @ bg_anchor.T).max(dim=1).values

    pseudo_scores = adapted_pseudo @ proto.T
    pseudo_old = pseudo_scores[:, old_mask_cols].max(dim=1).values
    pseudo_bg = (adapted_pseudo @ bg_anchor.T).max(dim=1).values

    if bool(seen_mask_cols.any().item()):
        support_seen = support_scores[:, seen_mask_cols].max(dim=1).values
        support_logits = torch.stack([support_old, support_seen, support_bg], dim=1) / tau
        support_targets = torch.tensor(
            [0 if int(v) in old_labels else 1 for v in labels.detach().cpu().tolist()],
            dtype=torch.long,
            device=support.device,
        )
        pseudo_seen = pseudo_scores[:, seen_mask_cols].max(dim=1).values
        pseudo_logits = torch.stack([pseudo_old, pseudo_seen, pseudo_bg], dim=1) / tau
        pseudo_targets = torch.full((pseudo_logits.size(0),), 2, dtype=torch.long, device=support.device)
        true_known = torch.where(support_targets == 0, support_old, support_seen)
        other_known = torch.where(support_targets == 0, support_seen, support_old)
        pseudo_known = torch.maximum(pseudo_old, pseudo_seen)
    else:
        # Stage2-B can be old/unknown-only. In that case train the same
        # query-free background pressure as an old-vs-background head instead
        # of silently dropping the loss because no seen-new class exists.
        support_logits = torch.stack([support_old, support_bg], dim=1) / tau
        support_targets = torch.zeros((support_logits.size(0),), dtype=torch.long, device=support.device)
        pseudo_logits = torch.stack([pseudo_old, pseudo_bg], dim=1) / tau
        pseudo_targets = torch.ones((pseudo_logits.size(0),), dtype=torch.long, device=support.device)
        true_known = support_old
        other_known = support_old.detach()
        pseudo_known = pseudo_old

    support_ce = F.cross_entropy(support_logits, support_targets)
    pseudo_ce = F.cross_entropy(pseudo_logits, pseudo_targets)
    known_margin_loss = F.relu(other_known - true_known + float(known_margin)).pow(2).mean()
    background_margin_loss = F.relu(support_bg - true_known + float(background_margin)).pow(2).mean()
    pseudo_margin_loss = F.relu(pseudo_known - pseudo_bg + float(background_margin)).pow(2).mean()
    loss = (
        float(support_ce_weight) * support_ce
        + float(pseudo_ce_weight) * pseudo_ce
        + known_margin_loss
        + float(support_background_margin_weight) * background_margin_loss
        + float(pseudo_margin_weight) * pseudo_margin_loss
    )
    return loss, int(support.size(0) + pseudo_unknown.size(0))


def _known_coverage_margin_loss(
    adapter: nn.Module,
    known_features: torch.Tensor,
    known_mapped: torch.Tensor,
    class_prototypes: torch.Tensor,
    *,
    margin: float = 0.12,
    min_true_affinity: float = 0.35,
    max_samples: int = 256,
) -> tuple[torch.Tensor, int]:
    """Keep known/source/bridge samples covered before unknown rejection is optimized."""

    zero = torch.zeros((), dtype=class_prototypes.dtype, device=class_prototypes.device)
    if known_features.numel() == 0 or known_mapped.numel() == 0 or class_prototypes.numel() == 0:
        return zero, 0
    features = normalize_rows(torch.as_tensor(known_features).float()).to(class_prototypes.device)
    mapped = torch.as_tensor(known_mapped, dtype=torch.long, device=class_prototypes.device)
    if features.size(0) != mapped.numel():
        return zero, 0
    limit = max(1, int(max_samples))
    if features.size(0) > limit:
        features = features[:limit]
        mapped = mapped[:limit]
    logits = normalize_rows(adapter(features)) @ normalize_rows(class_prototypes).T
    if logits.size(1) < 2:
        return zero, int(features.size(0))
    true_score = logits.gather(1, mapped.view(-1, 1)).squeeze(1)
    other_mask = torch.ones_like(logits, dtype=torch.bool)
    other_mask.scatter_(1, mapped.view(-1, 1), False)
    best_other = logits.masked_fill(~other_mask, -float("inf")).max(dim=1).values
    finite = torch.isfinite(best_other)
    if not bool(finite.any().item()):
        return zero, 0
    margin_loss = F.relu(best_other[finite] - true_score[finite] + float(margin)).pow(2)
    coverage_loss = F.relu(float(min_true_affinity) - true_score[finite]).pow(2)
    return margin_loss.mean() + coverage_loss.mean(), int(finite.sum().item())


def _source_leave_one_out_unknown_boundary_loss(
    adapter: nn.Module,
    source_features: torch.Tensor,
    source_labels: torch.Tensor | Iterable[int],
    class_labels: torch.Tensor,
    class_prototypes: torch.Tensor,
    *,
    unknown_margin: float = 0.35,
    interclass_margin: float = 0.08,
    max_samples_per_class: int = 24,
) -> tuple[torch.Tensor, int]:
    """Use each source-old class as a held-out unknown relative to other known prototypes."""

    zero = torch.zeros((), dtype=class_prototypes.dtype, device=class_prototypes.device)
    if source_features.numel() == 0 or class_prototypes.numel() == 0:
        return zero, 0
    features = normalize_rows(torch.as_tensor(source_features).float()).to(class_prototypes.device)
    labels = _labels_tensor(source_labels, device=class_prototypes.device)
    if features.size(0) != labels.numel():
        return zero, 0
    class_values = [int(v) for v in class_labels.detach().cpu().tolist()]
    class_index = {label: idx for idx, label in enumerate(class_values)}
    keep_indices: list[int] = []
    per_class_limit = max(1, int(max_samples_per_class))
    for label in sorted({int(v) for v in labels.detach().cpu().tolist()}):
        if label == UNKNOWN_LABEL or label not in class_index:
            continue
        label_indices = torch.nonzero(labels == int(label), as_tuple=False).flatten()
        if label_indices.numel() == 0:
            continue
        keep_indices.extend(int(v) for v in label_indices[:per_class_limit].detach().cpu().tolist())
    if not keep_indices:
        return zero, 0

    idx_tensor = torch.tensor(keep_indices, dtype=torch.long, device=features.device)
    selected = features[idx_tensor]
    selected_labels = labels[idx_tensor]
    logits = normalize_rows(adapter(selected)) @ normalize_rows(class_prototypes).T
    true_indices = torch.tensor(
        [class_index[int(label)] for label in selected_labels.detach().cpu().tolist()],
        dtype=torch.long,
        device=features.device,
    )
    true_affinity = logits.gather(1, true_indices.view(-1, 1)).squeeze(1)
    other_mask = torch.ones_like(logits, dtype=torch.bool)
    other_mask.scatter_(1, true_indices.view(-1, 1), False)
    if not bool(other_mask.any().item()):
        return zero, 0
    best_other = logits.masked_fill(~other_mask, -float("inf")).max(dim=1).values
    finite = torch.isfinite(best_other)
    if not bool(finite.any().item()):
        return zero, 0
    heldout_unknown = F.relu(best_other[finite] - float(unknown_margin)).pow(2).mean()
    interclass_gap = F.relu(best_other[finite] - true_affinity[finite] + float(interclass_margin)).pow(2).mean()
    return heldout_unknown + interclass_gap, int(finite.sum().item())


def _prototype_pseudo_unknown_features(
    prototypes: torch.Tensor,
    *,
    samples_per_pair: int = 2,
    offset_scale: float = 0.15,
) -> torch.Tensor:
    proto = normalize_rows(torch.as_tensor(prototypes).float())
    if proto.size(0) < 2 or int(samples_per_pair) <= 0:
        return torch.empty((0, proto.size(1)), dtype=proto.dtype, device=proto.device)
    samples = []
    for left in range(proto.size(0)):
        for right in range(left + 1, proto.size(0)):
            p_left = proto[left]
            p_right = proto[right]
            midpoint = normalize_rows((p_left + p_right).view(1, -1)).squeeze(0)
            direction = normalize_rows((p_left - p_right).view(1, -1)).squeeze(0)
            for idx in range(max(1, int(samples_per_pair))):
                branch = idx % 4
                if branch == 0:
                    sample = midpoint + float(offset_scale) * direction
                elif branch == 1:
                    sample = midpoint - float(offset_scale) * direction
                elif branch == 2:
                    sample = p_left + float(offset_scale) * direction
                else:
                    sample = p_right - float(offset_scale) * direction
                samples.append(normalize_rows(sample.view(1, -1)).squeeze(0))
    return torch.stack(samples, dim=0)


def _source_feature_boundary_pseudo_unknown_features(
    features: torch.Tensor | None,
    labels: torch.Tensor | Iterable[int] | None,
    *,
    samples_per_pair: int = 0,
    offset_scale: float = 0.20,
) -> torch.Tensor:
    """Build query-free boundary negatives from source old-class feature spread."""

    if features is None or labels is None or int(samples_per_pair) <= 0:
        return torch.empty((0, 0), dtype=torch.float32)
    x = normalize_rows(torch.as_tensor(features).float())
    y = _labels_tensor(labels, device=x.device)
    if x.numel() == 0 or y.numel() == 0 or x.size(0) != y.numel():
        return torch.empty((0, x.shape[1] if x.ndim == 2 else 0), dtype=x.dtype, device=x.device)
    class_values = sorted({int(v) for v in y.detach().cpu().tolist() if int(v) != UNKNOWN_LABEL})
    if len(class_values) < 2:
        return torch.empty((0, x.shape[1]), dtype=x.dtype, device=x.device)

    by_class = {label: x[y == int(label)] for label in class_values}
    samples = []
    per_pair = max(1, int(samples_per_pair))
    for left_idx, left in enumerate(class_values):
        left_members = by_class[int(left)]
        if left_members.numel() == 0:
            continue
        for right in class_values[left_idx + 1 :]:
            right_members = by_class[int(right)]
            if right_members.numel() == 0:
                continue
            sims = left_members @ right_members.T
            flat = torch.topk(sims.reshape(-1), k=min(per_pair, sims.numel()), largest=True).indices
            for flat_idx in flat.tolist():
                li = int(flat_idx // right_members.size(0))
                ri = int(flat_idx % right_members.size(0))
                a = left_members[li]
                b = right_members[ri]
                midpoint = normalize_rows((a + b).view(1, -1)).squeeze(0)
                direction = normalize_rows((a - b).view(1, -1)).squeeze(0)
                samples.append(normalize_rows((midpoint + float(offset_scale) * direction).view(1, -1)).squeeze(0))
                samples.append(normalize_rows((midpoint - float(offset_scale) * direction).view(1, -1)).squeeze(0))
    return torch.stack(samples, dim=0) if samples else torch.empty((0, x.shape[1]), dtype=x.dtype, device=x.device)


def _prototype_old_neighborhood_features(
    source_prototypes: PrototypeSet,
    *,
    samples_per_class: int = 2,
    radius: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build query-free old-class neighborhood samples around source prototypes."""

    proto = normalize_rows(torch.as_tensor(source_prototypes.vectors).float())
    labels = source_prototypes.labels.detach().long().to(proto.device)
    if proto.size(0) == 0 or int(samples_per_class) <= 0:
        return (
            torch.empty((0, proto.size(1)), dtype=proto.dtype, device=proto.device),
            torch.empty((0,), dtype=torch.long, device=proto.device),
        )
    samples = []
    sample_labels = []
    for idx in range(proto.size(0)):
        p = proto[idx]
        if proto.size(0) > 1:
            sims = proto @ p
            sims[idx] = -float("inf")
            nearest = proto[int(torch.argmax(sims).item())]
            direction = normalize_rows((p - nearest).view(1, -1)).squeeze(0)
        else:
            direction = p
        for item in range(max(1, int(samples_per_class))):
            sign = -1.0 if item % 2 else 1.0
            scale = float(radius) * (1.0 + 0.5 * float(item // 2))
            sample = normalize_rows((p + sign * scale * direction).view(1, -1)).squeeze(0)
            samples.append(sample)
            sample_labels.append(int(labels[idx].item()))
    return torch.stack(samples, dim=0), torch.tensor(sample_labels, dtype=torch.long, device=proto.device)


def _target_shift_pseudo_unknown_features(
    source_prototypes: PrototypeSet,
    target_support: torch.Tensor,
    target_labels: torch.Tensor | Iterable[int],
    *,
    samples_per_class: int = 0,
    offset_scale: float = 0.20,
) -> torch.Tensor:
    """Build old-support hard negatives without reading unknown query samples."""

    source_vectors = normalize_rows(torch.as_tensor(source_prototypes.vectors).float())
    labels = _labels_tensor(target_labels, device=source_vectors.device)
    support = normalize_rows(torch.as_tensor(target_support).float()).to(source_vectors.device)
    if support.numel() == 0 or labels.numel() == 0 or int(samples_per_class) <= 0:
        return torch.empty((0, source_vectors.shape[1]), dtype=source_vectors.dtype, device=source_vectors.device)
    old_labels = source_prototypes.label_values()
    samples = []
    for label in sorted(old_labels):
        support_mask = labels == int(label)
        if not bool(support_mask.any().item()):
            continue
        source_idx = source_prototypes.index_of(int(label))
        source_vec = source_vectors[source_idx]
        members = support[support_mask]
        target_mean = normalize_rows(members.mean(dim=0, keepdim=True)).squeeze(0)
        shift_direction = target_mean - source_vec
        if float(shift_direction.norm().item()) <= 1.0e-6:
            if source_vectors.shape[0] > 1:
                sims = source_vectors @ source_vec
                sims[source_idx] = -float("inf")
                nearest = source_vectors[int(torch.argmax(sims).item())]
                shift_direction = source_vec - nearest
            else:
                shift_direction = source_vec
        shift_direction = normalize_rows(shift_direction.view(1, -1)).squeeze(0)
        nearest_direction = shift_direction
        if source_vectors.shape[0] > 1:
            sims = source_vectors @ target_mean
            sims[source_idx] = -float("inf")
            nearest = source_vectors[int(torch.argmax(sims).item())]
            nearest_direction = normalize_rows((target_mean - nearest).view(1, -1)).squeeze(0)
        for idx in range(max(1, int(samples_per_class))):
            anchor = members[idx % members.shape[0]]
            scale = float(offset_scale) * (1.0 + 0.25 * float(idx // 3))
            branch = idx % 3
            if branch == 0:
                sample = target_mean + scale * shift_direction
            elif branch == 1:
                sample = anchor + scale * shift_direction
            else:
                sample = target_mean + 0.5 * scale * shift_direction + 0.5 * scale * nearest_direction
            samples.append(normalize_rows(sample.view(1, -1)).squeeze(0))
    if not samples:
        return torch.empty((0, source_vectors.shape[1]), dtype=source_vectors.dtype, device=source_vectors.device)
    return torch.stack(samples, dim=0)


def _target_old_bridge_features(
    source_prototypes: PrototypeSet,
    target_support: torch.Tensor,
    target_labels: torch.Tensor | Iterable[int],
    *,
    samples_per_class: int = 0,
    max_mix: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build query-free old samples between source prototypes and target-old support."""

    source_vectors = normalize_rows(torch.as_tensor(source_prototypes.vectors).float())
    labels = _labels_tensor(target_labels, device=source_vectors.device)
    support = normalize_rows(torch.as_tensor(target_support).float()).to(source_vectors.device)
    if support.numel() == 0 or labels.numel() == 0 or int(samples_per_class) <= 0:
        return (
            torch.empty((0, source_vectors.shape[1]), dtype=source_vectors.dtype, device=source_vectors.device),
            torch.empty((0,), dtype=torch.long, device=source_vectors.device),
        )
    samples = []
    sample_labels = []
    for label in sorted(source_prototypes.label_values()):
        support_mask = labels == int(label)
        if not bool(support_mask.any().item()):
            continue
        source_vec = source_vectors[source_prototypes.index_of(int(label))]
        members = support[support_mask]
        target_mean = normalize_rows(members.mean(dim=0, keepdim=True)).squeeze(0)
        for idx in range(max(1, int(samples_per_class))):
            anchor = target_mean if idx % 2 == 0 else members[idx % members.shape[0]]
            alpha = min(float(max_mix), (float(idx + 1) / float(max(2, int(samples_per_class) + 1))) * float(max_mix))
            sample = normalize_rows(((1.0 - alpha) * source_vec + alpha * anchor).view(1, -1)).squeeze(0)
            samples.append(sample)
            sample_labels.append(int(label))
    if not samples:
        return (
            torch.empty((0, source_vectors.shape[1]), dtype=source_vectors.dtype, device=source_vectors.device),
            torch.empty((0,), dtype=torch.long, device=source_vectors.device),
        )
    return torch.stack(samples, dim=0), torch.tensor(sample_labels, dtype=torch.long, device=source_vectors.device)


def _target_halo_pseudo_unknown_features(
    source_prototypes: PrototypeSet,
    target_support: torch.Tensor,
    target_labels: torch.Tensor | Iterable[int],
    *,
    samples_per_class: int = 0,
    offset_scale: float = 0.35,
) -> torch.Tensor:
    """Build hard negatives just outside target-old support, toward competing old prototypes."""

    source_vectors = normalize_rows(torch.as_tensor(source_prototypes.vectors).float())
    labels = _labels_tensor(target_labels, device=source_vectors.device)
    support = normalize_rows(torch.as_tensor(target_support).float()).to(source_vectors.device)
    if support.numel() == 0 or labels.numel() == 0 or int(samples_per_class) <= 0:
        return torch.empty((0, source_vectors.shape[1]), dtype=source_vectors.dtype, device=source_vectors.device)
    samples = []
    for label in sorted(source_prototypes.label_values()):
        support_mask = labels == int(label)
        if not bool(support_mask.any().item()):
            continue
        source_idx = source_prototypes.index_of(int(label))
        source_vec = source_vectors[source_idx]
        members = support[support_mask]
        target_mean = normalize_rows(members.mean(dim=0, keepdim=True)).squeeze(0)
        shift_direction = target_mean - source_vec
        if float(shift_direction.norm().item()) <= 1.0e-6:
            shift_direction = source_vec
        shift_direction = normalize_rows(shift_direction.view(1, -1)).squeeze(0)
        if source_vectors.shape[0] > 1:
            sims = source_vectors @ target_mean
            sims[source_idx] = -float("inf")
            nearest = source_vectors[int(torch.argmax(sims).item())]
            toward_competitor = nearest - target_mean
        else:
            toward_competitor = -shift_direction
        if float(toward_competitor.norm().item()) <= 1.0e-6:
            toward_competitor = -shift_direction
        toward_competitor = normalize_rows(toward_competitor.view(1, -1)).squeeze(0)
        for idx in range(max(1, int(samples_per_class))):
            anchor = members[idx % members.shape[0]]
            scale = float(offset_scale) * (1.0 + 0.25 * float(idx // 4))
            branch = idx % 4
            if branch == 0:
                sample = target_mean + scale * toward_competitor
            elif branch == 1:
                sample = anchor + scale * toward_competitor
            elif branch == 2:
                sample = target_mean + 0.5 * scale * toward_competitor + 0.5 * scale * shift_direction
            else:
                sample = anchor + 0.5 * scale * toward_competitor - 0.25 * scale * shift_direction
            samples.append(normalize_rows(sample.view(1, -1)).squeeze(0))
    if not samples:
        return torch.empty((0, source_vectors.shape[1]), dtype=source_vectors.dtype, device=source_vectors.device)
    return torch.stack(samples, dim=0)


def _target_support_ring_pseudo_unknown_features(
    source_prototypes: PrototypeSet,
    target_support: torch.Tensor,
    target_labels: torch.Tensor | Iterable[int],
    *,
    samples_per_class: int = 0,
    offset_scale: float = 0.45,
) -> torch.Tensor:
    """Build near-support hard negatives just outside compact target-old support."""

    source_vectors = normalize_rows(torch.as_tensor(source_prototypes.vectors).float())
    labels = _labels_tensor(target_labels, device=source_vectors.device)
    support = normalize_rows(torch.as_tensor(target_support).float()).to(source_vectors.device)
    if support.numel() == 0 or labels.numel() == 0 or int(samples_per_class) <= 0:
        return torch.empty((0, source_vectors.shape[1]), dtype=source_vectors.dtype, device=source_vectors.device)
    samples = []
    for label in sorted(source_prototypes.label_values()):
        support_mask = labels == int(label)
        if not bool(support_mask.any().item()):
            continue
        source_idx = source_prototypes.index_of(int(label))
        source_vec = source_vectors[source_idx]
        members = support[support_mask]
        target_mean = normalize_rows(members.mean(dim=0, keepdim=True)).squeeze(0)
        if source_vectors.shape[0] > 1:
            sims = source_vectors @ target_mean
            sims[source_idx] = -float("inf")
            nearest = source_vectors[int(torch.argmax(sims).item())]
        else:
            nearest = -source_vec
        away_from_source = target_mean - source_vec
        if float(away_from_source.norm().item()) <= 1.0e-6:
            away_from_source = target_mean - nearest
        if float(away_from_source.norm().item()) <= 1.0e-6:
            away_from_source = target_mean
        away_from_source = normalize_rows(away_from_source.view(1, -1)).squeeze(0)
        toward_competitor = nearest - target_mean
        if float(toward_competitor.norm().item()) <= 1.0e-6:
            toward_competitor = -away_from_source
        toward_competitor = normalize_rows(toward_competitor.view(1, -1)).squeeze(0)
        for idx in range(max(1, int(samples_per_class))):
            anchor = members[idx % members.shape[0]]
            radial = anchor - target_mean
            if float(radial.norm().item()) <= 1.0e-6:
                radial = away_from_source
            radial = normalize_rows(radial.view(1, -1)).squeeze(0)
            scale = float(offset_scale) * (1.0 + 0.20 * float(idx // 4))
            branch = idx % 4
            if branch == 0:
                sample = anchor + scale * radial
            elif branch == 1:
                sample = anchor + 0.7 * scale * away_from_source
            elif branch == 2:
                sample = target_mean + 0.5 * scale * radial + 0.5 * scale * toward_competitor
            else:
                sample = anchor + 0.5 * scale * radial + 0.5 * scale * toward_competitor
            samples.append(normalize_rows(sample.view(1, -1)).squeeze(0))
    if not samples:
        return torch.empty((0, source_vectors.shape[1]), dtype=source_vectors.dtype, device=source_vectors.device)
    return torch.stack(samples, dim=0)


def _target_shift_pseudo_unknown_features_from_states(
    class_states: Mapping[int, ClassState],
    *,
    samples_per_class: int = 0,
    offset_scale: float = 0.20,
) -> torch.Tensor:
    labels = sorted(int(k) for k in class_states)
    if not labels:
        return torch.empty((0, 0), dtype=torch.float32)
    first = torch.as_tensor(class_states[labels[0]].prototype).float()
    if int(samples_per_class) <= 0:
        return torch.empty((0, first.numel()), dtype=first.dtype, device=first.device)
    prototypes = {
        int(label): normalize_rows(torch.as_tensor(class_states[int(label)].prototype).float().view(1, -1)).squeeze(0)
        for label in labels
    }
    proto_matrix = torch.stack([prototypes[int(label)] for label in labels], dim=0).to(first.device)
    samples = []
    for label in labels:
        state = class_states[int(label)]
        if str(state.group) != "old" or state.support_anchors is None:
            continue
        anchors = normalize_rows(torch.as_tensor(state.support_anchors).float()).to(first.device)
        if anchors.numel() == 0:
            continue
        prototype = prototypes[int(label)].to(first.device)
        target_mean = normalize_rows(anchors.mean(dim=0, keepdim=True)).squeeze(0)
        shift_direction = target_mean - prototype
        if float(shift_direction.norm().item()) <= 1.0e-6:
            if proto_matrix.shape[0] > 1:
                idx = labels.index(int(label))
                sims = proto_matrix @ prototype
                sims[idx] = -float("inf")
                nearest = proto_matrix[int(torch.argmax(sims).item())]
                shift_direction = prototype - nearest
            else:
                shift_direction = prototype
        shift_direction = normalize_rows(shift_direction.view(1, -1)).squeeze(0)
        nearest_direction = shift_direction
        if proto_matrix.shape[0] > 1:
            idx = labels.index(int(label))
            sims = proto_matrix @ target_mean
            sims[idx] = -float("inf")
            nearest = proto_matrix[int(torch.argmax(sims).item())]
            nearest_direction = normalize_rows((target_mean - nearest).view(1, -1)).squeeze(0)
        for sample_idx in range(max(1, int(samples_per_class))):
            anchor = anchors[sample_idx % anchors.shape[0]]
            scale = float(offset_scale) * (1.0 + 0.25 * float(sample_idx // 3))
            branch = sample_idx % 3
            if branch == 0:
                sample = target_mean + scale * shift_direction
            elif branch == 1:
                sample = anchor + scale * shift_direction
            else:
                sample = target_mean + 0.5 * scale * shift_direction + 0.5 * scale * nearest_direction
            samples.append(normalize_rows(sample.view(1, -1)).squeeze(0))
    if not samples:
        return torch.empty((0, first.numel()), dtype=first.dtype, device=first.device)
    return torch.stack(samples, dim=0)


def _target_halo_pseudo_unknown_features_from_states(
    class_states: Mapping[int, ClassState],
    *,
    samples_per_class: int = 0,
    offset_scale: float = 0.35,
) -> torch.Tensor:
    labels = sorted(int(k) for k in class_states)
    if not labels:
        return torch.empty((0, 0), dtype=torch.float32)
    first = torch.as_tensor(class_states[labels[0]].prototype).float()
    if int(samples_per_class) <= 0:
        return torch.empty((0, first.numel()), dtype=first.dtype, device=first.device)
    prototypes = {
        int(label): normalize_rows(torch.as_tensor(class_states[int(label)].prototype).float().view(1, -1)).squeeze(0)
        for label in labels
    }
    proto_matrix = torch.stack([prototypes[int(label)] for label in labels], dim=0).to(first.device)
    samples = []
    for label in labels:
        state = class_states[int(label)]
        if str(state.group) != "old" or state.support_anchors is None:
            continue
        anchors = normalize_rows(torch.as_tensor(state.support_anchors).float()).to(first.device)
        if anchors.numel() == 0:
            continue
        prototype = prototypes[int(label)].to(first.device)
        target_mean = normalize_rows(anchors.mean(dim=0, keepdim=True)).squeeze(0)
        shift_direction = target_mean - prototype
        if float(shift_direction.norm().item()) <= 1.0e-6:
            shift_direction = prototype
        shift_direction = normalize_rows(shift_direction.view(1, -1)).squeeze(0)
        if proto_matrix.shape[0] > 1:
            idx = labels.index(int(label))
            sims = proto_matrix @ target_mean
            sims[idx] = -float("inf")
            nearest = proto_matrix[int(torch.argmax(sims).item())]
            toward_competitor = nearest - target_mean
        else:
            toward_competitor = -shift_direction
        if float(toward_competitor.norm().item()) <= 1.0e-6:
            toward_competitor = -shift_direction
        toward_competitor = normalize_rows(toward_competitor.view(1, -1)).squeeze(0)
        for sample_idx in range(max(1, int(samples_per_class))):
            anchor = anchors[sample_idx % anchors.shape[0]]
            scale = float(offset_scale) * (1.0 + 0.25 * float(sample_idx // 4))
            branch = sample_idx % 4
            if branch == 0:
                sample = target_mean + scale * toward_competitor
            elif branch == 1:
                sample = anchor + scale * toward_competitor
            elif branch == 2:
                sample = target_mean + 0.5 * scale * toward_competitor + 0.5 * scale * shift_direction
            else:
                sample = anchor + 0.5 * scale * toward_competitor - 0.25 * scale * shift_direction
            samples.append(normalize_rows(sample.view(1, -1)).squeeze(0))
    if not samples:
        return torch.empty((0, first.numel()), dtype=first.dtype, device=first.device)
    return torch.stack(samples, dim=0)


def _target_support_ring_pseudo_unknown_features_from_states(
    class_states: Mapping[int, ClassState],
    *,
    samples_per_class: int = 0,
    offset_scale: float = 0.45,
) -> torch.Tensor:
    labels = sorted(int(k) for k in class_states)
    if not labels:
        return torch.empty((0, 0), dtype=torch.float32)
    first = torch.as_tensor(class_states[labels[0]].prototype).float()
    if int(samples_per_class) <= 0:
        return torch.empty((0, first.numel()), dtype=first.dtype, device=first.device)
    prototypes = {
        int(label): normalize_rows(torch.as_tensor(class_states[int(label)].prototype).float().view(1, -1)).squeeze(0)
        for label in labels
    }
    proto_matrix = torch.stack([prototypes[int(label)] for label in labels], dim=0).to(first.device)
    samples = []
    for label in labels:
        state = class_states[int(label)]
        if str(state.group) != "old" or state.support_anchors is None:
            continue
        anchors = normalize_rows(torch.as_tensor(state.support_anchors).float()).to(first.device)
        if anchors.numel() == 0:
            continue
        prototype = prototypes[int(label)].to(first.device)
        target_mean = normalize_rows(anchors.mean(dim=0, keepdim=True)).squeeze(0)
        if proto_matrix.shape[0] > 1:
            idx = labels.index(int(label))
            sims = proto_matrix @ target_mean
            sims[idx] = -float("inf")
            nearest = proto_matrix[int(torch.argmax(sims).item())]
        else:
            nearest = -prototype
        away_from_source = target_mean - prototype
        if float(away_from_source.norm().item()) <= 1.0e-6:
            away_from_source = target_mean - nearest
        if float(away_from_source.norm().item()) <= 1.0e-6:
            away_from_source = target_mean
        away_from_source = normalize_rows(away_from_source.view(1, -1)).squeeze(0)
        toward_competitor = nearest - target_mean
        if float(toward_competitor.norm().item()) <= 1.0e-6:
            toward_competitor = -away_from_source
        toward_competitor = normalize_rows(toward_competitor.view(1, -1)).squeeze(0)
        for sample_idx in range(max(1, int(samples_per_class))):
            anchor = anchors[sample_idx % anchors.shape[0]]
            radial = anchor - target_mean
            if float(radial.norm().item()) <= 1.0e-6:
                radial = away_from_source
            radial = normalize_rows(radial.view(1, -1)).squeeze(0)
            scale = float(offset_scale) * (1.0 + 0.20 * float(sample_idx // 4))
            branch = sample_idx % 4
            if branch == 0:
                sample = anchor + scale * radial
            elif branch == 1:
                sample = anchor + 0.7 * scale * away_from_source
            elif branch == 2:
                sample = target_mean + 0.5 * scale * radial + 0.5 * scale * toward_competitor
            else:
                sample = anchor + 0.5 * scale * radial + 0.5 * scale * toward_competitor
            samples.append(normalize_rows(sample.view(1, -1)).squeeze(0))
    if not samples:
        return torch.empty((0, first.numel()), dtype=first.dtype, device=first.device)
    return torch.stack(samples, dim=0)


def _adapter_proxy_accuracy(logits: torch.Tensor, mapped_labels: torch.Tensor) -> float:
    if logits.numel() == 0 or mapped_labels.numel() == 0:
        return float("nan")
    predicted = logits.argmax(dim=1)
    return float((predicted == mapped_labels).float().mean().detach().cpu().item())


def _finite_or(value: float, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bounded01(value: float, default: float = 0.0) -> float:
    out = _finite_or(value, default)
    return max(0.0, min(1.0, out))


def _select_adapter_alpha_by_proxy(
    adapter: nn.Module,
    *,
    support: torch.Tensor,
    mapped: torch.Tensor,
    labels: torch.Tensor,
    class_prototypes: torch.Tensor,
    source_retention_features: torch.Tensor,
    source_retention_mapped: torch.Tensor,
    pseudo_unknown: torch.Tensor,
    unknown_moat_margin: float,
    old_bridge: torch.Tensor,
    old_bridge_mapped: torch.Tensor,
    old_labels: set[int],
    policy: str,
    old_acc_target: float = 0.90,
    seen_new_acc_target: float = 0.75,
    candidate_alphas: Iterable[float] | None = None,
) -> dict:
    """Select adapter strength from support/source/surrogate signals only."""

    policy_name = str(policy or "final").strip().lower()
    if policy_name in {"", "none", "final", "last"}:
        adapter.alpha = 1.0
        return {
            "policy": "final",
            "selected_alpha": 1.0,
            "reason": "use_final_trained_adapter",
            "candidates": [],
        }

    alphas = list(candidate_alphas or (0.0, 0.25, 0.5, 0.75, 1.0))
    if not alphas:
        alphas = [1.0]
    support_old_mask = torch.tensor(
        [int(v) in old_labels for v in labels.detach().cpu().tolist()],
        dtype=torch.bool,
        device=support.device,
    )
    support_new_mask = ~support_old_mask
    candidates: list[dict] = []
    with torch.no_grad():
        for alpha in alphas:
            adapter.alpha = float(alpha)
            support_logits = 16.0 * (adapter(support) @ class_prototypes.T)
            support_acc = _adapter_proxy_accuracy(support_logits, mapped)
            support_old_acc = (
                _adapter_proxy_accuracy(support_logits[support_old_mask], mapped[support_old_mask])
                if bool(support_old_mask.any().item())
                else float("nan")
            )
            support_new_acc = (
                _adapter_proxy_accuracy(support_logits[support_new_mask], mapped[support_new_mask])
                if bool(support_new_mask.any().item())
                else float("nan")
            )
            source_acc = float("nan")
            if source_retention_features.numel() > 0 and source_retention_mapped.numel() > 0:
                source_logits = 16.0 * (adapter(source_retention_features) @ class_prototypes.T)
                source_acc = _adapter_proxy_accuracy(source_logits, source_retention_mapped)
            bridge_acc = float("nan")
            if old_bridge.numel() > 0 and old_bridge_mapped.numel() > 0:
                bridge_logits = 16.0 * (adapter(old_bridge) @ class_prototypes.T)
                bridge_acc = _adapter_proxy_accuracy(bridge_logits, old_bridge_mapped)
            pseudo_mean = float("nan")
            pseudo_exceed = float("nan")
            if pseudo_unknown.numel() > 0:
                pseudo_affinity = adapter(pseudo_unknown) @ class_prototypes.T
                pseudo_max = pseudo_affinity.max(dim=1).values
                pseudo_mean = float(pseudo_max.mean().detach().cpu().item())
                pseudo_exceed = float((pseudo_max > float(unknown_moat_margin)).float().mean().detach().cpu().item())
            legacy_proxy_score = (
                3.0 * _finite_or(support_old_acc, _finite_or(support_acc))
                + 1.0 * _finite_or(support_new_acc, 0.0)
                + 2.0 * _finite_or(source_acc, 1.0)
                + 1.0 * _finite_or(bridge_acc, _finite_or(source_acc, 1.0))
                - 3.0 * _finite_or(pseudo_exceed, 0.0)
                - 0.50 * _finite_or(pseudo_mean, 0.0)
                - 0.02 * float(alpha)
            )
            boundary_guard_score = (
                4.0 * _bounded01(source_acc, 1.0)
                + 3.0 * _bounded01(bridge_acc, _finite_or(source_acc, 1.0))
                + 2.0 * _bounded01(support_old_acc, _finite_or(support_acc, 0.0))
                + 0.50 * _bounded01(support_new_acc, 0.0)
                - 6.0 * _bounded01(pseudo_exceed, 0.0)
                - 1.50 * max(0.0, _finite_or(pseudo_mean, 0.0) - float(unknown_moat_margin))
                - 0.35 * float(alpha)
            )
            retention_floor = min(
                _bounded01(source_acc, 1.0),
                _bounded01(bridge_acc, _finite_or(source_acc, 1.0)),
            )
            support_old = _bounded01(support_old_acc, _finite_or(support_acc, 0.0))
            support_seen_new = _bounded01(support_new_acc, 0.0)
            has_seen_new_support = bool(support_new_mask.any().item())
            support_cv_stats = _support_leave_one_out_proxy_stats(adapter, support, mapped, labels, old_labels)
            support_cv_acc = _bounded01(support_cv_stats.get("support_cv_acc"), _finite_or(support_acc, 0.0))
            support_cv_old_acc = _bounded01(support_cv_stats.get("support_cv_old_acc"), support_old)
            support_cv_seen_new_acc = _bounded01(support_cv_stats.get("support_cv_seen_new_acc"), support_seen_new)
            support_cv_margin_mean = _finite_or(support_cv_stats.get("support_cv_margin_mean"), -1.0)
            support_cv_margin_p10 = _finite_or(support_cv_stats.get("support_cv_margin_p10"), -1.0)
            support_cv_valid_count = int(_finite_or(support_cv_stats.get("support_cv_valid_count"), 0.0))
            support_cv_overfit = max(0.0, _bounded01(support_acc, 0.0) - support_cv_acc)
            support_cv_overfit = max(support_cv_overfit, support_old - support_cv_old_acc)
            if has_seen_new_support:
                support_cv_overfit = max(support_cv_overfit, support_seen_new - support_cv_seen_new_acc)
            pseudo_risk = _bounded01(pseudo_exceed, 0.0)
            pseudo_margin_excess = max(0.0, _finite_or(pseudo_mean, 0.0) - float(unknown_moat_margin))
            support_overfit = max(0.0, support_old - retention_floor - 0.03)
            if math.isfinite(_finite_or(support_acc, float("nan"))):
                support_overfit = max(support_overfit, _bounded01(support_acc, 0.0) - retention_floor - 0.03)
            old_support_floor = max(0.72, min(0.98, float(old_acc_target) - 0.05))
            retention_floor_target = max(0.68, min(0.96, float(old_acc_target) - 0.10))
            seen_new_floor = max(0.40, min(0.90, float(seen_new_acc_target) - 0.15))
            known_constraint_deficit = (
                max(0.0, retention_floor_target - retention_floor)
                + max(0.0, old_support_floor - support_old)
                + (max(0.0, seen_new_floor - support_seen_new) if has_seen_new_support else 0.0)
            )
            pseudo_risk_cap = 0.55 if has_seen_new_support else 0.45
            pseudo_constraint_deficit = max(0.0, pseudo_risk - pseudo_risk_cap)
            known_feasible = known_constraint_deficit <= 1.0e-6
            support_cv_old_floor = max(0.58, min(0.92, float(old_acc_target) - 0.18))
            support_cv_seen_new_floor = max(0.32, min(0.80, float(seen_new_acc_target) - 0.22))
            support_cv_margin_floor = -0.04 if has_seen_new_support else -0.02
            support_cv_deficit = (
                max(0.0, support_cv_old_floor - support_cv_old_acc)
                + (max(0.0, support_cv_seen_new_floor - support_cv_seen_new_acc) if has_seen_new_support else 0.0)
                + max(0.0, support_cv_margin_floor - support_cv_margin_p10)
            )
            support_cv_feasible = support_cv_valid_count > 0 and support_cv_deficit <= 1.0e-6
            # This selector is intentionally more conservative than target_boundary_guard:
            # next48ee showed support accuracy can rise while old-query accuracy collapses.
            retention_risk_score = (
                4.5 * retention_floor
                + 1.20 * support_old
                + 0.35 * support_seen_new
                - 4.0 * pseudo_risk
                - 1.25 * pseudo_margin_excess
                - 3.0 * support_overfit
                - 0.85 * float(alpha) * float(alpha)
            )
            constrained_retention_score = (
                9.0 * (1.0 - known_constraint_deficit)
                + 2.0 * retention_floor
                + 1.5 * support_old
                + 0.5 * support_seen_new
                - 2.0 * pseudo_constraint_deficit
                - 1.0 * pseudo_risk
                - 0.75 * pseudo_margin_excess
                - 0.50 * float(alpha) * float(alpha)
            )
            support_cv_risk_score = (
                4.0 * retention_floor
                + 2.0 * support_cv_old_acc
                + 0.8 * support_cv_seen_new_acc
                + 0.8 * support_cv_acc
                + 0.6 * max(-0.5, min(0.5, support_cv_margin_mean))
                - 3.0 * support_cv_overfit
                - 3.5 * pseudo_risk
                - 1.0 * pseudo_margin_excess
                - 0.70 * float(alpha) * float(alpha)
            )
            support_cv_constrained_score = (
                9.0 * (1.0 - known_constraint_deficit)
                + 7.0 * (1.0 - support_cv_deficit)
                + 2.0 * retention_floor
                + 1.5 * support_cv_old_acc
                + 0.5 * support_cv_seen_new_acc
                - 2.5 * support_cv_overfit
                - 1.2 * pseudo_constraint_deficit
                - 0.7 * pseudo_risk
                - 0.55 * float(alpha) * float(alpha)
            )
            proxy_score = retention_risk_score if policy_name in {
                "retention_risk_balanced",
                "retention_balanced",
                "risk_balanced",
                "identity_preserving_risk",
            } else constrained_retention_score if policy_name in {
                "constrained_retention_risk",
                "known_coverage_constrained",
                "coverage_constrained",
                "identity_preserving",
            } else support_cv_constrained_score if policy_name in {
                "support_cv_constrained",
                "support_cv_identity",
                "identity_preserving_cv",
            } else support_cv_risk_score if policy_name in {
                "support_cv_risk_balanced",
                "support_cv_balanced",
                "identity_preserving_risk_cv",
            } else boundary_guard_score if policy_name in {
                "target_boundary_guard",
                "boundary_guard",
                "conservative_proxy",
            } else legacy_proxy_score
            candidates.append(
                {
                    "alpha": float(alpha),
                    "proxy_score": float(proxy_score),
                    "legacy_proxy_score": float(legacy_proxy_score),
                    "boundary_guard_score": float(boundary_guard_score),
                    "retention_risk_score": float(retention_risk_score),
                    "constrained_retention_score": float(constrained_retention_score),
                    "known_feasible": bool(known_feasible),
                    "known_constraint_deficit": float(known_constraint_deficit),
                    "pseudo_constraint_deficit": float(pseudo_constraint_deficit),
                    "old_support_floor": float(old_support_floor),
                    "retention_floor_target": float(retention_floor_target),
                    "seen_new_floor": float(seen_new_floor) if has_seen_new_support else None,
                    "retention_floor": float(retention_floor),
                    "support_cv_score": float(support_cv_constrained_score if policy_name in {"support_cv_constrained", "support_cv_identity", "identity_preserving_cv"} else support_cv_risk_score),
                    "support_cv_feasible": bool(support_cv_feasible),
                    "support_cv_deficit": float(support_cv_deficit),
                    "support_cv_old_floor": float(support_cv_old_floor),
                    "support_cv_seen_new_floor": float(support_cv_seen_new_floor) if has_seen_new_support else None,
                    "support_cv_margin_floor": float(support_cv_margin_floor),
                    "support_cv_acc": float(support_cv_stats.get("support_cv_acc", float("nan"))),
                    "support_cv_old_acc": float(support_cv_stats.get("support_cv_old_acc", float("nan"))),
                    "support_cv_seen_new_acc": float(support_cv_stats.get("support_cv_seen_new_acc", float("nan"))),
                    "support_cv_margin_mean": float(support_cv_stats.get("support_cv_margin_mean", float("nan"))),
                    "support_cv_margin_p10": float(support_cv_stats.get("support_cv_margin_p10", float("nan"))),
                    "support_cv_valid_count": int(support_cv_valid_count),
                    "support_cv_overfit_penalty": float(support_cv_overfit),
                    "support_overfit_penalty": float(support_overfit),
                    "support_acc": float(support_acc),
                    "support_old_acc": float(support_old_acc),
                    "support_seen_new_acc": float(support_new_acc),
                    "source_retention_acc": float(source_acc),
                    "old_bridge_acc": float(bridge_acc),
                    "pseudo_unknown_affinity_mean": float(pseudo_mean),
                    "pseudo_unknown_margin_exceed_rate": float(pseudo_exceed),
                }
            )
    if policy_name in {
        "constrained_retention_risk",
        "known_coverage_constrained",
        "coverage_constrained",
        "identity_preserving",
        "support_cv_constrained",
        "support_cv_identity",
        "identity_preserving_cv",
    }:
        if policy_name in {"support_cv_constrained", "support_cv_identity", "identity_preserving_cv"}:
            feasible = [
                item
                for item in candidates
                if bool(item.get("known_feasible", False)) and bool(item.get("support_cv_feasible", False))
            ]
        else:
            feasible = [item for item in candidates if bool(item.get("known_feasible", False))]
        if feasible:
            selected = max(
                feasible,
                key=lambda item: (
                    -_bounded01(item.get("pseudo_unknown_margin_exceed_rate", 0.0), 0.0),
                    -max(0.0, _finite_or(item.get("pseudo_unknown_affinity_mean", 0.0), 0.0) - float(unknown_moat_margin)),
                    -float(item.get("support_cv_deficit", 0.0)),
                    _bounded01(item.get("support_cv_old_acc", item.get("support_old_acc", 0.0)), 0.0),
                    _finite_or(item.get("support_cv_margin_p10", -1.0), -1.0),
                    _bounded01(item.get("retention_floor", 0.0), 0.0),
                    _bounded01(item.get("support_old_acc", 0.0), 0.0),
                    -float(item.get("alpha", 0.0)),
                ),
            )
        else:
            selected = min(
                candidates,
                key=lambda item: (
                    float(item.get("known_constraint_deficit", 999.0)),
                    float(item.get("support_cv_deficit", 999.0)),
                    float(item.get("pseudo_constraint_deficit", 999.0)),
                    float(item.get("alpha", 0.0)),
                ),
            ) if candidates else {"alpha": 1.0}
    else:
        selected = max(candidates, key=lambda item: item["proxy_score"]) if candidates else {"alpha": 1.0}
    adapter.alpha = float(selected["alpha"])
    selected_policy = (
        "constrained_retention_risk"
        if policy_name in {"constrained_retention_risk", "known_coverage_constrained", "coverage_constrained", "identity_preserving"}
        else "support_cv_constrained"
        if policy_name in {"support_cv_constrained", "support_cv_identity", "identity_preserving_cv"}
        else "support_cv_risk_balanced"
        if policy_name in {"support_cv_risk_balanced", "support_cv_balanced", "identity_preserving_risk_cv"}
        else "retention_risk_balanced"
        if policy_name in {"retention_risk_balanced", "retention_balanced", "risk_balanced", "identity_preserving_risk"}
        else "target_boundary_guard"
        if policy_name in {
        "target_boundary_guard",
        "boundary_guard",
        "conservative_proxy",
        }
        else "proxy_line_search"
    )
    return {
        "policy": selected_policy,
        "selected_alpha": float(selected["alpha"]),
        "selected_proxy_score": float(selected.get("proxy_score", float("nan"))),
        "selected_known_feasible": bool(selected.get("known_feasible", False)),
        "selected_known_constraint_deficit": float(selected.get("known_constraint_deficit", float("nan"))),
        "selected_pseudo_constraint_deficit": float(selected.get("pseudo_constraint_deficit", float("nan"))),
        "selected_support_cv_feasible": bool(selected.get("support_cv_feasible", False)),
        "selected_support_cv_deficit": float(selected.get("support_cv_deficit", float("nan"))),
        "selected_support_cv_acc": float(selected.get("support_cv_acc", float("nan"))),
        "selected_support_cv_old_acc": float(selected.get("support_cv_old_acc", float("nan"))),
        "selected_support_cv_seen_new_acc": float(selected.get("support_cv_seen_new_acc", float("nan"))),
        "selected_support_cv_margin_p10": float(selected.get("support_cv_margin_p10", float("nan"))),
        "selected_support_cv_overfit_penalty": float(selected.get("support_cv_overfit_penalty", float("nan"))),
        "reason": (
            "hard_known_coverage_feasibility_then_surrogate_unknown_risk"
            if selected_policy == "constrained_retention_risk"
            else "support_leave_one_out_feasibility_then_old_retention_and_surrogate_unknown_risk"
            if selected_policy == "support_cv_constrained"
            else "support_leave_one_out_retention_first_with_surrogate_unknown_risk_penalty"
            if selected_policy == "support_cv_risk_balanced"
            else "old_retention_first_with_support_overfit_and_surrogate_unknown_risk_penalty"
            if selected_policy == "retention_risk_balanced"
            else
            "old_source_bridge_retention_with_surrogate_unknown_risk_and_alpha_penalty"
            if selected_policy == "target_boundary_guard"
            else "support_source_retention_vs_surrogate_unknown_proxy"
        ),
        "candidates": candidates,
    }


def fit_low_compute_target_adapter(
    source_prototypes: PrototypeSet,
    target_support: torch.Tensor,
    target_labels: torch.Tensor | Iterable[int],
    *,
    source_adapter_features: torch.Tensor | None = None,
    source_adapter_labels: torch.Tensor | Iterable[int] | None = None,
    source_boundary_pseudo_unknown_samples_per_pair: int = 0,
    source_boundary_pseudo_unknown_offset_scale: float = 0.20,
    rank: int = 2,
    steps: int = 80,
    lr: float = 0.05,
    source_anchor_weight: float = 0.05,
    source_ce_weight: float = 0.10,
    unknown_moat_weight: float = 0.10,
    unknown_moat_margin: float = 0.45,
    pseudo_unknown_samples_per_pair: int = 2,
    pseudo_unknown_offset_scale: float = 0.15,
    pseudo_unknown_target_shift_samples_per_class: int = 0,
    pseudo_unknown_target_shift_offset_scale: float = 0.20,
    pseudo_unknown_target_halo_samples_per_class: int = 0,
    pseudo_unknown_target_halo_offset_scale: float = 0.35,
    pseudo_unknown_target_ring_samples_per_class: int = 0,
    pseudo_unknown_target_ring_offset_scale: float = 0.45,
    old_bridge_weight: float = 0.10,
    old_bridge_samples_per_class: int = 2,
    old_bridge_max_mix: float = 0.85,
    adapter_kind: str = "low_rank",
    support_contrast_weight: float = 0.0,
    support_contrast_negative_margin: float = 0.78,
    support_contrast_positive_margin: float = 0.88,
    support_center_ce_weight: float = 0.0,
    support_center_temperature: float = 0.10,
    support_center_margin: float = 0.10,
    soft_proto_weight: float = 0.0,
    soft_proto_topk: int = 2,
    soft_proto_temperature: float = 0.10,
    soft_proto_boundary_weight: float = 0.0,
    soft_proto_boundary_margin: float = 0.15,
    void_background_weight: float = 0.0,
    negative_anchor_weight: float = 0.0,
    negative_anchor_margin: float = 0.12,
    negative_anchor_temperature: float = 0.10,
    negative_anchor_max_anchors: int = 256,
    three_way_head_weight: float = 0.0,
    three_way_head_temperature: float = 0.10,
    three_way_head_known_margin: float = 0.08,
    three_way_head_background_margin: float = 0.08,
    three_way_head_support_ce_weight: float = 1.0,
    three_way_head_pseudo_ce_weight: float = 0.35,
    three_way_head_support_background_margin_weight: float = 1.0,
    three_way_head_pseudo_margin_weight: float = 0.50,
    old_neighborhood_weight: float = 0.10,
    old_neighborhood_samples_per_class: int = 2,
    old_neighborhood_radius: float = 0.06,
    old_surrogate_margin_weight: float = 0.05,
    old_surrogate_margin: float = 0.10,
    source_looo_unknown_weight: float = 0.0,
    source_looo_unknown_margin: float = 0.35,
    source_looo_interclass_margin: float = 0.08,
    source_looo_max_samples_per_class: int = 24,
    known_coverage_weight: float = 0.0,
    known_coverage_margin: float = 0.12,
    known_coverage_min_affinity: float = 0.35,
    known_coverage_max_samples: int = 256,
    adapter_selection_policy: str = "final",
    old_acc_target: float = 0.90,
    seen_new_acc_target: float = 0.75,
) -> tuple[nn.Module, dict]:
    support = normalize_rows(torch.as_tensor(target_support).float()).to(source_prototypes.vectors.device)
    labels = _labels_tensor(target_labels, device=support.device)
    if support.size(0) != labels.numel():
        raise ValueError("target support features and labels must have equal lengths")
    if bool((labels == UNKNOWN_LABEL).any().item()):
        raise ValueError("unknown query samples must not be used for target-adapter training")
    old_labels = source_prototypes.label_values()
    validate_stage2_protocol(
        "Stage2-C",
        use_target_old_support=bool(any(int(v) in old_labels for v in labels.detach().cpu().tolist())),
        use_target_new_support=bool(any(int(v) not in old_labels for v in labels.detach().cpu().tolist())),
        use_unknown_query_for_threshold_calibration=False,
    )
    class_labels, class_prototypes, mapped = _prototype_bank_for_target_adapter(source_prototypes, support, labels)
    adapter_kind = str(adapter_kind).lower().strip()
    if adapter_kind in {"residual_mlp", "mlp", "nonlinear"}:
        adapter = ResidualMLPTargetAdapter(int(support.shape[1]), rank=int(rank), alpha=1.0).to(support.device)
        adapter_kind = "residual_mlp"
    elif adapter_kind in {"low_rank", "linear", "lora"}:
        adapter = LowRankTargetAdapter(int(support.shape[1]), rank=int(rank), alpha=1.0).to(support.device)
        adapter_kind = "low_rank"
    else:
        raise ValueError(f"unknown adapter_kind: {adapter_kind}")
    torch.manual_seed(0)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=float(lr))
    source_vectors = source_prototypes.vectors.detach().float().to(support.device)
    class_index = {int(label): idx for idx, label in enumerate(class_labels.detach().cpu().tolist())}
    source_mapped = torch.tensor(
        [class_index[int(label)] for label in source_prototypes.labels.detach().cpu().tolist()],
        dtype=torch.long,
        device=support.device,
    )
    source_adapter = torch.empty((0, support.shape[1]), dtype=support.dtype, device=support.device)
    source_adapter_mapped = torch.empty((0,), dtype=torch.long, device=support.device)
    source_adapter_label_tensor = torch.empty((0,), dtype=torch.long, device=support.device)
    if source_adapter_features is not None and source_adapter_labels is not None:
        source_adapter_raw = normalize_rows(torch.as_tensor(source_adapter_features).float()).to(support.device)
        source_adapter_label_tensor = _labels_tensor(source_adapter_labels, device=support.device)
        if source_adapter_raw.size(0) != source_adapter_label_tensor.numel():
            raise ValueError("source adapter features and labels must have equal lengths")
        keep = torch.tensor(
            [int(v) in class_index for v in source_adapter_label_tensor.detach().cpu().tolist()],
            dtype=torch.bool,
            device=support.device,
        )
        if bool(keep.any().item()):
            source_adapter = source_adapter_raw[keep].detach()
            source_adapter_label_tensor = source_adapter_label_tensor[keep].detach()
            source_adapter_mapped = torch.tensor(
                [class_index[int(label)] for label in source_adapter_label_tensor.detach().cpu().tolist()],
                dtype=torch.long,
                device=support.device,
            )
    geometry_pseudo_unknown = _prototype_pseudo_unknown_features(
        class_prototypes.detach(),
        samples_per_pair=int(pseudo_unknown_samples_per_pair),
        offset_scale=float(pseudo_unknown_offset_scale),
    ).to(support.device)
    source_boundary_pseudo_unknown = _source_feature_boundary_pseudo_unknown_features(
        source_adapter,
        source_adapter_label_tensor,
        samples_per_pair=int(source_boundary_pseudo_unknown_samples_per_pair),
        offset_scale=float(source_boundary_pseudo_unknown_offset_scale),
    ).to(support.device)
    target_shift_pseudo_unknown = _target_shift_pseudo_unknown_features(
        source_prototypes,
        support,
        labels,
        samples_per_class=int(pseudo_unknown_target_shift_samples_per_class),
        offset_scale=float(pseudo_unknown_target_shift_offset_scale),
    ).to(support.device)
    target_halo_pseudo_unknown = _target_halo_pseudo_unknown_features(
        source_prototypes,
        support,
        labels,
        samples_per_class=int(pseudo_unknown_target_halo_samples_per_class),
        offset_scale=float(pseudo_unknown_target_halo_offset_scale),
    ).to(support.device)
    target_ring_pseudo_unknown = _target_support_ring_pseudo_unknown_features(
        source_prototypes,
        support,
        labels,
        samples_per_class=int(pseudo_unknown_target_ring_samples_per_class),
        offset_scale=float(pseudo_unknown_target_ring_offset_scale),
    ).to(support.device)
    pseudo_unknown_parts = [
        part
        for part in (
            geometry_pseudo_unknown,
            source_boundary_pseudo_unknown,
            target_shift_pseudo_unknown,
            target_halo_pseudo_unknown,
            target_ring_pseudo_unknown,
        )
        if part.numel() > 0
    ]
    pseudo_unknown = (
        torch.cat(pseudo_unknown_parts, dim=0)
        if pseudo_unknown_parts
        else torch.empty((0, support.shape[1]), dtype=support.dtype, device=support.device)
    )
    old_bridge, old_bridge_labels = _target_old_bridge_features(
        source_prototypes,
        support,
        labels,
        samples_per_class=int(old_bridge_samples_per_class),
        max_mix=float(old_bridge_max_mix),
    )
    old_bridge = old_bridge.to(support.device)
    old_bridge_mapped = torch.empty((0,), dtype=torch.long, device=support.device)
    if old_bridge_labels.numel() > 0:
        old_bridge_mapped = torch.tensor(
            [class_index[int(label)] for label in old_bridge_labels.detach().cpu().tolist()],
            dtype=torch.long,
            device=support.device,
        )
    old_neighborhood, old_neighborhood_labels = _prototype_old_neighborhood_features(
        source_prototypes,
        samples_per_class=int(old_neighborhood_samples_per_class),
        radius=float(old_neighborhood_radius),
    )
    old_neighborhood = old_neighborhood.to(support.device)
    old_neighborhood_mapped = torch.empty((0,), dtype=torch.long, device=support.device)
    if old_neighborhood_labels.numel() > 0:
        old_neighborhood_mapped = torch.tensor(
            [class_index[int(label)] for label in old_neighborhood_labels.detach().cpu().tolist()],
            dtype=torch.long,
            device=support.device,
        )
    old_margin_features = [source_vectors]
    old_margin_mapped = [source_mapped]
    if old_bridge.numel() > 0:
        old_margin_features.append(old_bridge)
        old_margin_mapped.append(old_bridge_mapped)
    if old_neighborhood.numel() > 0:
        old_margin_features.append(old_neighborhood)
        old_margin_mapped.append(old_neighborhood_mapped)
    support_old_mask = torch.tensor(
        [int(v) in old_labels for v in labels.detach().cpu().tolist()],
        dtype=torch.bool,
        device=support.device,
    )
    if bool(support_old_mask.any().item()):
        old_margin_features.append(support[support_old_mask].detach())
        old_margin_mapped.append(mapped[support_old_mask].detach())
    old_anchor_bank = support[support_old_mask].detach() if bool(support_old_mask.any().item()) else torch.empty((0, support.shape[1]), dtype=support.dtype, device=support.device)
    old_anchor_labels = labels[support_old_mask].detach() if bool(support_old_mask.any().item()) else torch.empty((0,), dtype=torch.long, device=support.device)
    old_bridge_anchor_mapped = torch.empty((0,), dtype=torch.long, device=support.device)
    if old_bridge_labels.numel() > 0:
        old_bridge_anchor_mapped = old_bridge_labels.detach().to(device=support.device, dtype=torch.long)
    old_margin_features_tensor = (
        torch.cat(old_margin_features, dim=0)
        if old_margin_features
        else torch.empty((0, support.shape[1]), dtype=support.dtype, device=support.device)
    )
    old_margin_mapped_tensor = (
        torch.cat(old_margin_mapped, dim=0)
        if old_margin_mapped
        else torch.empty((0,), dtype=torch.long, device=support.device)
    )
    soft_proto_anchor_features = torch.cat([source_vectors.detach(), support.detach()], dim=0)
    soft_proto_anchor_labels = torch.cat([source_prototypes.labels.detach().to(support.device), labels.detach()], dim=0)
    soft_proto_train_features = [support.detach(), source_vectors.detach()]
    soft_proto_train_labels = [labels.detach(), source_prototypes.labels.detach().to(support.device)]
    if old_bridge.numel() > 0 and old_bridge_labels.numel() > 0:
        soft_proto_train_features.append(old_bridge.detach())
        soft_proto_train_labels.append(old_bridge_labels.detach().to(support.device))
    if old_neighborhood.numel() > 0 and old_neighborhood_labels.numel() > 0:
        soft_proto_train_features.append(old_neighborhood.detach())
        soft_proto_train_labels.append(old_neighborhood_labels.detach().to(support.device))
    soft_proto_train_features_tensor = torch.cat(soft_proto_train_features, dim=0)
    soft_proto_train_labels_tensor = torch.cat(soft_proto_train_labels, dim=0)
    loss_trace = []
    for step_idx in range(max(0, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        adapted = adapter(support)
        logits = 16.0 * (adapted @ class_prototypes.T)
        loss_ce = F.cross_entropy(logits, mapped)
        loss_anchor_raw = loss_ce.detach() * 0.0
        if source_anchor_weight > 0:
            anchor_features = source_adapter if source_adapter.numel() > 0 else source_vectors
            anchored = adapter(anchor_features)
            loss_anchor_raw = (anchored - anchor_features).pow(2).mean()
        loss_anchor = float(source_anchor_weight) * loss_anchor_raw
        loss_source_ce_raw = loss_ce.detach() * 0.0
        if source_ce_weight > 0:
            if source_adapter.numel() > 0 and source_adapter_mapped.numel() > 0:
                source_logits = 16.0 * (adapter(source_adapter) @ class_prototypes.T)
                loss_source_ce_raw = F.cross_entropy(source_logits, source_adapter_mapped)
            elif source_vectors.numel() > 0:
                source_logits = 16.0 * (adapter(source_vectors) @ class_prototypes.T)
                loss_source_ce_raw = F.cross_entropy(source_logits, source_mapped)
        loss_source_ce = float(source_ce_weight) * loss_source_ce_raw
        loss_old_bridge_raw = loss_ce.detach() * 0.0
        if old_bridge_weight > 0 and old_bridge.numel() > 0:
            old_bridge_logits = 16.0 * (adapter(old_bridge) @ class_prototypes.T)
            loss_old_bridge_raw = F.cross_entropy(old_bridge_logits, old_bridge_mapped)
        loss_old_bridge = float(old_bridge_weight) * loss_old_bridge_raw
        loss_unknown_moat_raw = loss_ce.detach() * 0.0
        if unknown_moat_weight > 0 and pseudo_unknown.numel() > 0:
            pseudo_affinity = adapter(pseudo_unknown) @ class_prototypes.T
            loss_unknown_moat_raw = F.relu(
                pseudo_affinity.max(dim=1).values - float(unknown_moat_margin)
            ).pow(2).mean()
        loss_unknown_moat = float(unknown_moat_weight) * loss_unknown_moat_raw
        loss_old_neighborhood_raw = loss_ce.detach() * 0.0
        if old_neighborhood_weight > 0 and old_neighborhood.numel() > 0:
            old_neighborhood_logits = 16.0 * (adapter(old_neighborhood) @ class_prototypes.T)
            loss_old_neighborhood_raw = F.cross_entropy(old_neighborhood_logits, old_neighborhood_mapped)
        loss_old_neighborhood = float(old_neighborhood_weight) * loss_old_neighborhood_raw
        loss_old_surrogate_margin_raw = loss_ce.detach() * 0.0
        if (
            old_surrogate_margin_weight > 0
            and pseudo_unknown.numel() > 0
            and old_margin_features_tensor.numel() > 0
            and old_margin_mapped_tensor.numel() > 0
        ):
            old_margin_logits = adapter(old_margin_features_tensor) @ class_prototypes.T
            old_true_affinity = old_margin_logits.gather(1, old_margin_mapped_tensor.view(-1, 1)).squeeze(1)
            pseudo_max_affinity = (adapter(pseudo_unknown) @ class_prototypes.T).max(dim=1).values
            loss_old_surrogate_margin_raw = F.relu(
                pseudo_max_affinity.view(-1, 1) - old_true_affinity.view(1, -1) + float(old_surrogate_margin)
            ).pow(2).mean()
        loss_old_surrogate_margin = float(old_surrogate_margin_weight) * loss_old_surrogate_margin_raw
        loss_support_contrast_negative_raw = loss_ce.detach() * 0.0
        loss_support_contrast_positive_raw = loss_ce.detach() * 0.0
        if support_contrast_weight > 0 and old_anchor_bank.numel() > 0:
            adapted_anchors = normalize_rows(adapter(old_anchor_bank))
            if pseudo_unknown.numel() > 0:
                pseudo_anchor = normalize_rows(adapter(pseudo_unknown)) @ adapted_anchors.T
                loss_support_contrast_negative_raw = F.relu(
                    pseudo_anchor.max(dim=1).values - float(support_contrast_negative_margin)
                ).pow(2).mean()
            positive_parts = []
            positive_labels = []
            if bool(support_old_mask.any().item()):
                positive_parts.append(support[support_old_mask].detach())
                positive_labels.append(labels[support_old_mask].detach())
            if old_bridge.numel() > 0 and old_bridge_anchor_mapped.numel() > 0:
                positive_parts.append(old_bridge.detach())
                positive_labels.append(old_bridge_anchor_mapped.detach())
            if positive_parts:
                positive_features = torch.cat(positive_parts, dim=0)
                positive_label_values = torch.cat(positive_labels, dim=0)
                adapted_positive = normalize_rows(adapter(positive_features))
                sims = adapted_positive @ adapted_anchors.T
                same_label = positive_label_values.view(-1, 1) == old_anchor_labels.view(1, -1)
                if bool(same_label.any().item()):
                    same_sims = sims.masked_fill(~same_label, -float("inf"))
                    positive_best = same_sims.max(dim=1).values
                    finite = torch.isfinite(positive_best)
                    if bool(finite.any().item()):
                        loss_support_contrast_positive_raw = F.relu(
                            float(support_contrast_positive_margin) - positive_best[finite]
                        ).pow(2).mean()
        loss_support_contrast = float(support_contrast_weight) * (
            loss_support_contrast_negative_raw + loss_support_contrast_positive_raw
        )
        loss_support_center_raw = loss_ce.detach() * 0.0
        if support_center_ce_weight > 0:
            loss_support_center_raw = _support_center_leave_one_out_loss(
                adapter,
                support,
                labels,
                temperature=float(support_center_temperature),
                margin=float(support_center_margin),
            )
        loss_support_center = float(support_center_ce_weight) * loss_support_center_raw
        loss_soft_proto_raw = loss_ce.detach() * 0.0
        if soft_proto_weight > 0 and soft_proto_anchor_features.numel() > 0 and soft_proto_train_features_tensor.numel() > 0:
            loss_soft_proto_raw = _class_constrained_soft_prototype_loss(
                adapter,
                soft_proto_train_features_tensor,
                soft_proto_train_labels_tensor,
                soft_proto_anchor_features,
                soft_proto_anchor_labels,
                topk=int(soft_proto_topk),
                temperature=float(soft_proto_temperature),
            )
        loss_soft_proto = float(soft_proto_weight) * loss_soft_proto_raw
        loss_soft_proto_boundary_raw = loss_ce.detach() * 0.0
        if (
            soft_proto_boundary_weight > 0
            and soft_proto_anchor_features.numel() > 0
            and soft_proto_train_features_tensor.numel() > 0
        ):
            loss_soft_proto_boundary_raw = _soft_prototype_mixture_boundary_loss(
                adapter,
                soft_proto_train_features_tensor,
                soft_proto_train_labels_tensor,
                soft_proto_anchor_features,
                soft_proto_anchor_labels,
                topk=int(soft_proto_topk),
                temperature=float(soft_proto_temperature),
                margin=float(soft_proto_boundary_margin),
            )
        loss_soft_proto_boundary = float(soft_proto_boundary_weight) * loss_soft_proto_boundary_raw
        loss_void_background_raw = loss_ce.detach() * 0.0
        if void_background_weight > 0 and pseudo_unknown.numel() > 0 and old_margin_features_tensor.numel() > 0:
            loss_void_background_raw = _void_background_competition_loss(
                adapter,
                old_margin_features_tensor,
                old_margin_mapped_tensor,
                class_prototypes,
                pseudo_unknown,
            )
        loss_void_background = float(void_background_weight) * loss_void_background_raw
        loss_negative_anchor_raw = loss_ce.detach() * 0.0
        negative_anchor_count = 0
        if negative_anchor_weight > 0 and pseudo_unknown.numel() > 0 and old_margin_features_tensor.numel() > 0:
            loss_negative_anchor_raw, negative_anchor_count = _negative_anchor_background_loss(
                adapter,
                old_margin_features_tensor,
                old_margin_mapped_tensor,
                class_prototypes,
                pseudo_unknown,
                margin=float(negative_anchor_margin),
                temperature=float(negative_anchor_temperature),
                max_anchors=int(negative_anchor_max_anchors),
            )
        loss_negative_anchor = float(negative_anchor_weight) * loss_negative_anchor_raw
        loss_three_way_head_raw = loss_ce.detach() * 0.0
        three_way_head_count = 0
        if three_way_head_weight > 0:
            loss_three_way_head_raw, three_way_head_count = _three_way_decision_head_loss(
                adapter,
                support,
                labels,
                mapped,
                class_labels,
                class_prototypes,
                pseudo_unknown,
                old_labels,
                temperature=float(three_way_head_temperature),
                known_margin=float(three_way_head_known_margin),
                background_margin=float(three_way_head_background_margin),
                support_ce_weight=float(three_way_head_support_ce_weight),
                pseudo_ce_weight=float(three_way_head_pseudo_ce_weight),
                support_background_margin_weight=float(three_way_head_support_background_margin_weight),
                pseudo_margin_weight=float(three_way_head_pseudo_margin_weight),
            )
        loss_three_way_head = float(three_way_head_weight) * loss_three_way_head_raw
        loss_known_coverage_raw = loss_ce.detach() * 0.0
        known_coverage_count = 0
        if known_coverage_weight > 0 and old_margin_features_tensor.numel() > 0 and old_margin_mapped_tensor.numel() > 0:
            loss_known_coverage_raw, known_coverage_count = _known_coverage_margin_loss(
                adapter,
                old_margin_features_tensor,
                old_margin_mapped_tensor,
                class_prototypes,
                margin=float(known_coverage_margin),
                min_true_affinity=float(known_coverage_min_affinity),
                max_samples=int(known_coverage_max_samples),
            )
        loss_known_coverage = float(known_coverage_weight) * loss_known_coverage_raw
        loss_source_looo_raw = loss_ce.detach() * 0.0
        source_looo_sample_count = 0
        if source_looo_unknown_weight > 0 and source_adapter.numel() > 0 and source_adapter_label_tensor.numel() > 0:
            loss_source_looo_raw, source_looo_sample_count = _source_leave_one_out_unknown_boundary_loss(
                adapter,
                source_adapter,
                source_adapter_label_tensor,
                class_labels,
                class_prototypes,
                unknown_margin=float(source_looo_unknown_margin),
                interclass_margin=float(source_looo_interclass_margin),
                max_samples_per_class=int(source_looo_max_samples_per_class),
            )
        loss_source_looo = float(source_looo_unknown_weight) * loss_source_looo_raw
        loss = (
            loss_ce
            + loss_anchor
            + loss_source_ce
            + loss_old_bridge
            + loss_unknown_moat
            + loss_old_neighborhood
            + loss_old_surrogate_margin
            + loss_support_contrast
            + loss_support_center
            + loss_soft_proto
            + loss_soft_proto_boundary
            + loss_void_background
            + loss_negative_anchor
            + loss_three_way_head
            + loss_known_coverage
            + loss_source_looo
        )
        loss.backward()
        grad_sq = 0.0
        grad_seen = 0
        for param in adapter.parameters():
            if param.grad is None:
                continue
            value = float(param.grad.detach().float().norm(2).item())
            grad_sq += value * value
            grad_seen += 1
        grad_norm = grad_sq ** 0.5 if grad_seen > 0 else float("nan")
        optimizer.step()
        with torch.no_grad():
            step_adapted = adapter(support)
            step_predicted = class_labels[(step_adapted @ class_prototypes.T).argmax(dim=1)]
            step_acc = (step_predicted == labels).float().mean()
        loss_trace.append(
            {
                "step": int(step_idx + 1),
                "loss_total": float(loss.detach().cpu().item()),
                "loss_ce": float(loss_ce.detach().cpu().item()),
                "loss_source_anchor_raw": float(loss_anchor_raw.detach().cpu().item()),
                "loss_source_anchor_weighted": float(loss_anchor.detach().cpu().item()),
                "source_anchor_weight": float(source_anchor_weight),
                "loss_source_ce_raw": float(loss_source_ce_raw.detach().cpu().item()),
                "loss_source_ce_weighted": float(loss_source_ce.detach().cpu().item()),
                "source_ce_weight": float(source_ce_weight),
                "loss_old_bridge_raw": float(loss_old_bridge_raw.detach().cpu().item()),
                "loss_old_bridge_weighted": float(loss_old_bridge.detach().cpu().item()),
                "old_bridge_weight": float(old_bridge_weight),
                "old_bridge_count": int(old_bridge.shape[0]),
                "loss_unknown_moat_raw": float(loss_unknown_moat_raw.detach().cpu().item()),
                "loss_unknown_moat_weighted": float(loss_unknown_moat.detach().cpu().item()),
                "unknown_moat_weight": float(unknown_moat_weight),
                "unknown_moat_margin": float(unknown_moat_margin),
                "pseudo_unknown_count": int(pseudo_unknown.shape[0]),
                "pseudo_unknown_geometry_count": int(geometry_pseudo_unknown.shape[0]),
                "pseudo_unknown_source_boundary_count": int(source_boundary_pseudo_unknown.shape[0]),
                "pseudo_unknown_target_shift_count": int(target_shift_pseudo_unknown.shape[0]),
                "pseudo_unknown_target_halo_count": int(target_halo_pseudo_unknown.shape[0]),
                "pseudo_unknown_target_ring_count": int(target_ring_pseudo_unknown.shape[0]),
                "loss_old_neighborhood_raw": float(loss_old_neighborhood_raw.detach().cpu().item()),
                "loss_old_neighborhood_weighted": float(loss_old_neighborhood.detach().cpu().item()),
                "old_neighborhood_weight": float(old_neighborhood_weight),
                "old_neighborhood_count": int(old_neighborhood.shape[0]),
                "loss_old_surrogate_margin_raw": float(loss_old_surrogate_margin_raw.detach().cpu().item()),
                "loss_old_surrogate_margin_weighted": float(loss_old_surrogate_margin.detach().cpu().item()),
                "old_surrogate_margin_weight": float(old_surrogate_margin_weight),
                "old_surrogate_margin": float(old_surrogate_margin),
                "loss_support_contrast_negative_raw": float(loss_support_contrast_negative_raw.detach().cpu().item()),
                "loss_support_contrast_positive_raw": float(loss_support_contrast_positive_raw.detach().cpu().item()),
                "loss_support_contrast_weighted": float(loss_support_contrast.detach().cpu().item()),
                "support_contrast_weight": float(support_contrast_weight),
                "support_contrast_negative_margin": float(support_contrast_negative_margin),
                "support_contrast_positive_margin": float(support_contrast_positive_margin),
                "loss_support_center_raw": float(loss_support_center_raw.detach().cpu().item()),
                "loss_support_center_weighted": float(loss_support_center.detach().cpu().item()),
                "support_center_ce_weight": float(support_center_ce_weight),
                "support_center_temperature": float(support_center_temperature),
                "support_center_margin": float(support_center_margin),
                "loss_soft_proto_raw": float(loss_soft_proto_raw.detach().cpu().item()),
                "loss_soft_proto_weighted": float(loss_soft_proto.detach().cpu().item()),
                "soft_proto_weight": float(soft_proto_weight),
                "soft_proto_topk": int(soft_proto_topk),
                "soft_proto_temperature": float(soft_proto_temperature),
                "soft_proto_anchor_count": int(soft_proto_anchor_features.shape[0]),
                "loss_soft_proto_boundary_raw": float(loss_soft_proto_boundary_raw.detach().cpu().item()),
                "loss_soft_proto_boundary_weighted": float(loss_soft_proto_boundary.detach().cpu().item()),
                "soft_proto_boundary_weight": float(soft_proto_boundary_weight),
                "soft_proto_boundary_margin": float(soft_proto_boundary_margin),
                "loss_void_background_raw": float(loss_void_background_raw.detach().cpu().item()),
                "loss_void_background_weighted": float(loss_void_background.detach().cpu().item()),
                "void_background_weight": float(void_background_weight),
                "loss_negative_anchor_background_raw": float(loss_negative_anchor_raw.detach().cpu().item()),
                "loss_negative_anchor_background_weighted": float(loss_negative_anchor.detach().cpu().item()),
                "negative_anchor_weight": float(negative_anchor_weight),
                "negative_anchor_margin": float(negative_anchor_margin),
                "negative_anchor_temperature": float(negative_anchor_temperature),
                "negative_anchor_count": int(negative_anchor_count),
                "loss_three_way_head_raw": float(loss_three_way_head_raw.detach().cpu().item()),
                "loss_three_way_head_weighted": float(loss_three_way_head.detach().cpu().item()),
                "three_way_head_weight": float(three_way_head_weight),
                "three_way_head_temperature": float(three_way_head_temperature),
                "three_way_head_known_margin": float(three_way_head_known_margin),
                "three_way_head_background_margin": float(three_way_head_background_margin),
                "three_way_head_support_ce_weight": float(three_way_head_support_ce_weight),
                "three_way_head_pseudo_ce_weight": float(three_way_head_pseudo_ce_weight),
                "three_way_head_support_background_margin_weight": float(three_way_head_support_background_margin_weight),
                "three_way_head_pseudo_margin_weight": float(three_way_head_pseudo_margin_weight),
                "three_way_head_count": int(three_way_head_count),
                "loss_known_coverage_raw": float(loss_known_coverage_raw.detach().cpu().item()),
                "loss_known_coverage_weighted": float(loss_known_coverage.detach().cpu().item()),
                "known_coverage_weight": float(known_coverage_weight),
                "known_coverage_margin": float(known_coverage_margin),
                "known_coverage_min_affinity": float(known_coverage_min_affinity),
                "known_coverage_count": int(known_coverage_count),
                "loss_source_looo_unknown_raw": float(loss_source_looo_raw.detach().cpu().item()),
                "loss_source_looo_unknown_weighted": float(loss_source_looo.detach().cpu().item()),
                "source_looo_unknown_weight": float(source_looo_unknown_weight),
                "source_looo_unknown_margin": float(source_looo_unknown_margin),
                "source_looo_interclass_margin": float(source_looo_interclass_margin),
                "source_looo_sample_count": int(source_looo_sample_count),
                "lr": float(lr),
                "grad_norm": float(grad_norm),
                "support_acc": float(step_acc.detach().cpu().item()),
            }
        )
    selection_features = source_adapter if source_adapter.numel() > 0 else source_vectors
    selection_mapped = source_adapter_mapped if source_adapter_mapped.numel() > 0 else source_mapped
    adapter_selection = _select_adapter_alpha_by_proxy(
        adapter,
        support=support,
        mapped=mapped,
        labels=labels,
        class_prototypes=class_prototypes,
        source_retention_features=selection_features,
        source_retention_mapped=selection_mapped,
        pseudo_unknown=pseudo_unknown,
        unknown_moat_margin=float(unknown_moat_margin),
        old_bridge=old_bridge,
        old_bridge_mapped=old_bridge_mapped,
        old_labels=old_labels,
        policy=str(adapter_selection_policy),
        old_acc_target=float(old_acc_target),
        seen_new_acc_target=float(seen_new_acc_target),
    )
    with torch.no_grad():
        adapted = adapter(support)
        predicted = class_labels[(adapted @ class_prototypes.T).argmax(dim=1)]
        old_mask = torch.tensor([int(v) in old_labels for v in labels.detach().cpu().tolist()], dtype=torch.bool, device=labels.device)
        new_mask = ~old_mask
        old_acc = _accuracy_for_mask(labels, predicted, old_mask) if bool(old_mask.any().item()) else float("nan")
        new_acc = _accuracy_for_mask(labels, predicted, new_mask) if bool(new_mask.any().item()) else float("nan")
    telemetry = {
        "training_scope": "fewshot_target_old_and_seen_new_support_only",
        "compute_profile": (
            "feature_level_residual_mlp_adapter_no_backbone_update"
            if adapter_kind == "residual_mlp"
            else "feature_level_low_rank_adapter_no_backbone_update"
        ),
        "loss_trace_schema": "target_adapter_step_loss_v1",
        "adapter_kind": str(adapter_kind),
        "rank": int(adapter.rank),
        "selected_alpha": float(adapter.alpha),
        "adapter_selection_policy": adapter_selection.get("policy"),
        "adapter_selection": adapter_selection,
        "steps": int(steps),
        "optimizer": "Adam",
        "lr": float(lr),
        "source_adapter_feature_count": int(source_adapter.shape[0]),
        "source_adapter_label_count": int(source_adapter_mapped.numel()),
        "source_boundary_pseudo_unknown_samples_per_pair": int(source_boundary_pseudo_unknown_samples_per_pair),
        "source_boundary_pseudo_unknown_offset_scale": float(source_boundary_pseudo_unknown_offset_scale),
        "source_anchor_weight": float(source_anchor_weight),
        "source_ce_weight": float(source_ce_weight),
        "unknown_moat_weight": float(unknown_moat_weight),
        "unknown_moat_margin": float(unknown_moat_margin),
        "pseudo_unknown_samples_per_pair": int(pseudo_unknown_samples_per_pair),
        "pseudo_unknown_offset_scale": float(pseudo_unknown_offset_scale),
        "pseudo_unknown_source_boundary_count": int(source_boundary_pseudo_unknown.shape[0]),
        "pseudo_unknown_target_shift_samples_per_class": int(pseudo_unknown_target_shift_samples_per_class),
        "pseudo_unknown_target_shift_offset_scale": float(pseudo_unknown_target_shift_offset_scale),
        "pseudo_unknown_target_halo_samples_per_class": int(pseudo_unknown_target_halo_samples_per_class),
        "pseudo_unknown_target_halo_offset_scale": float(pseudo_unknown_target_halo_offset_scale),
        "pseudo_unknown_target_ring_samples_per_class": int(pseudo_unknown_target_ring_samples_per_class),
        "pseudo_unknown_target_ring_offset_scale": float(pseudo_unknown_target_ring_offset_scale),
        "pseudo_unknown_geometry_count": int(geometry_pseudo_unknown.shape[0]),
        "pseudo_unknown_target_shift_count": int(target_shift_pseudo_unknown.shape[0]),
        "pseudo_unknown_target_halo_count": int(target_halo_pseudo_unknown.shape[0]),
        "pseudo_unknown_target_ring_count": int(target_ring_pseudo_unknown.shape[0]),
        "pseudo_unknown_count": int(pseudo_unknown.shape[0]),
        "old_bridge_weight": float(old_bridge_weight),
        "old_bridge_samples_per_class": int(old_bridge_samples_per_class),
        "old_bridge_max_mix": float(old_bridge_max_mix),
        "old_bridge_count": int(old_bridge.shape[0]),
        "old_neighborhood_weight": float(old_neighborhood_weight),
        "old_neighborhood_samples_per_class": int(old_neighborhood_samples_per_class),
        "old_neighborhood_radius": float(old_neighborhood_radius),
        "old_neighborhood_count": int(old_neighborhood.shape[0]),
        "old_surrogate_margin_weight": float(old_surrogate_margin_weight),
        "old_surrogate_margin": float(old_surrogate_margin),
        "old_surrogate_margin_count": int(old_margin_features_tensor.shape[0]),
        "source_looo_unknown_weight": float(source_looo_unknown_weight),
        "source_looo_unknown_margin": float(source_looo_unknown_margin),
        "source_looo_interclass_margin": float(source_looo_interclass_margin),
        "source_looo_max_samples_per_class": int(source_looo_max_samples_per_class),
        "source_looo_sample_count": int(loss_trace[-1]["source_looo_sample_count"]) if loss_trace else 0,
        "support_contrast_weight": float(support_contrast_weight),
        "support_contrast_negative_margin": float(support_contrast_negative_margin),
        "support_contrast_positive_margin": float(support_contrast_positive_margin),
        "support_contrast_anchor_count": int(old_anchor_bank.shape[0]),
        "support_center_ce_weight": float(support_center_ce_weight),
        "support_center_temperature": float(support_center_temperature),
        "support_center_margin": float(support_center_margin),
        "support_center_class_count": len({int(v) for v in labels.detach().cpu().tolist()}),
        "soft_proto_weight": float(soft_proto_weight),
        "soft_proto_topk": int(soft_proto_topk),
        "soft_proto_temperature": float(soft_proto_temperature),
        "soft_proto_boundary_weight": float(soft_proto_boundary_weight),
        "soft_proto_boundary_margin": float(soft_proto_boundary_margin),
        "void_background_weight": float(void_background_weight),
        "negative_anchor_weight": float(negative_anchor_weight),
        "negative_anchor_margin": float(negative_anchor_margin),
        "negative_anchor_temperature": float(negative_anchor_temperature),
        "negative_anchor_max_anchors": int(negative_anchor_max_anchors),
        "negative_anchor_count": int(loss_trace[-1]["negative_anchor_count"]) if loss_trace else 0,
        "three_way_head_weight": float(three_way_head_weight),
        "three_way_head_temperature": float(three_way_head_temperature),
        "three_way_head_known_margin": float(three_way_head_known_margin),
        "three_way_head_background_margin": float(three_way_head_background_margin),
        "three_way_head_support_ce_weight": float(three_way_head_support_ce_weight),
        "three_way_head_pseudo_ce_weight": float(three_way_head_pseudo_ce_weight),
        "three_way_head_support_background_margin_weight": float(three_way_head_support_background_margin_weight),
        "three_way_head_pseudo_margin_weight": float(three_way_head_pseudo_margin_weight),
        "three_way_head_count": int(loss_trace[-1]["three_way_head_count"]) if loss_trace else 0,
        "known_coverage_weight": float(known_coverage_weight),
        "known_coverage_margin": float(known_coverage_margin),
        "known_coverage_min_affinity": float(known_coverage_min_affinity),
        "known_coverage_max_samples": int(known_coverage_max_samples),
        "known_coverage_count": int(loss_trace[-1]["known_coverage_count"]) if loss_trace else 0,
        "soft_proto_anchor_count": int(soft_proto_anchor_features.shape[0]),
        "soft_proto_train_count": int(soft_proto_train_features_tensor.shape[0]),
        "loss_profile": "target_support_ce+source_old_retention_ce+source_anchor+old_target_bridge_ce+pseudo_unknown_moat+old_neighborhood_retention+old_surrogate_margin+target_support_contrast"
        + ("+support_center_leave_one_out_ce_margin" if float(support_center_ce_weight) > 0 else "")
        + ("+class_constrained_soft_prototype_mixture" if float(soft_proto_weight) > 0 else "")
        + ("+soft_prototype_mixture_boundary" if float(soft_proto_boundary_weight) > 0 else "")
        + ("+void_background_competition" if float(void_background_weight) > 0 else "")
        + ("+negative_anchor_background_basin" if float(negative_anchor_weight) > 0 else "")
        + ("+three_way_old_seen_background_head" if float(three_way_head_weight) > 0 else "")
        + ("+known_coverage_margin" if float(known_coverage_weight) > 0 else "")
        + ("+source_leave_one_old_out_unknown_boundary" if float(source_looo_unknown_weight) > 0 else ""),
        "trainable_parameters": int(sum(p.numel() for p in adapter.parameters() if p.requires_grad)),
        "old_acc_target": float(old_acc_target),
        "seen_new_acc_target": float(seen_new_acc_target),
        "support_old_acc": float(old_acc),
        "support_seen_new_acc": float(new_acc),
        "loss_trace": loss_trace,
        "loss_initial": float(loss_trace[0]["loss_total"]) if loss_trace else float("nan"),
        "loss_final": float(loss_trace[-1]["loss_total"]) if loss_trace else float("nan"),
        "loss_terms": {
            "target_support_ce": float(loss_trace[-1]["loss_ce"]) if loss_trace else float("nan"),
            "source_anchor_weighted": float(loss_trace[-1]["loss_source_anchor_weighted"]) if loss_trace else float("nan"),
            "source_old_ce_weighted": float(loss_trace[-1]["loss_source_ce_weighted"]) if loss_trace else float("nan"),
            "old_target_bridge_ce_weighted": float(loss_trace[-1]["loss_old_bridge_weighted"]) if loss_trace else float("nan"),
            "pseudo_unknown_moat_weighted": float(loss_trace[-1]["loss_unknown_moat_weighted"]) if loss_trace else float("nan"),
            "old_neighborhood_retention_weighted": float(loss_trace[-1]["loss_old_neighborhood_weighted"]) if loss_trace else float("nan"),
            "old_surrogate_margin_weighted": float(loss_trace[-1]["loss_old_surrogate_margin_weighted"]) if loss_trace else float("nan"),
            "target_support_contrast_weighted": float(loss_trace[-1]["loss_support_contrast_weighted"]) if loss_trace else float("nan"),
            "support_center_leave_one_out_weighted": (
                float(loss_trace[-1]["loss_support_center_weighted"]) if loss_trace else float("nan")
            ),
            "class_constrained_soft_prototype_mixture_weighted": float(loss_trace[-1]["loss_soft_proto_weighted"]) if loss_trace else float("nan"),
            "soft_prototype_mixture_boundary_weighted": (
                float(loss_trace[-1]["loss_soft_proto_boundary_weighted"]) if loss_trace else float("nan")
            ),
            "void_background_competition_weighted": (
                float(loss_trace[-1]["loss_void_background_weighted"]) if loss_trace else float("nan")
            ),
            "negative_anchor_background_basin_weighted": (
                float(loss_trace[-1]["loss_negative_anchor_background_weighted"]) if loss_trace else float("nan")
            ),
            "three_way_old_seen_background_head_weighted": (
                float(loss_trace[-1]["loss_three_way_head_weighted"]) if loss_trace else float("nan")
            ),
            "known_coverage_margin_weighted": (
                float(loss_trace[-1]["loss_known_coverage_weighted"]) if loss_trace else float("nan")
            ),
            "source_leave_one_old_out_unknown_boundary_weighted": (
                float(loss_trace[-1]["loss_source_looo_unknown_weighted"]) if loss_trace else float("nan")
            ),
        },
        "class_labels": class_labels.detach().cpu(),
        "class_prototypes": class_prototypes.detach().cpu(),
    }
    return adapter.cpu(), telemetry


def _state_threshold(state: ClassState, name: str, default: float) -> float:
    value = state.thresholds.get(name)
    if value is None:
        return float(default)
    return float(value)


def _state_evt_tail_threshold(state: ClassState, default: float) -> float:
    value = state.evt_params.get("tail_threshold", state.thresholds.get("max_evt_tail"))
    if value is None:
        return float(default)
    return float(value)


def _seen_new_evidence_score(cosine: torch.Tensor, residual: torch.Tensor, mahalanobis: torch.Tensor) -> torch.Tensor:
    """Support-only confidence for separating registered seen-new classes from unknowns."""

    return cosine - residual - 0.05 * torch.sqrt(mahalanobis.clamp_min(0.0))


def _old_support_evidence_score(
    cosine: torch.Tensor,
    anchor_similarity: torch.Tensor,
    residual: torch.Tensor,
    mahalanobis: torch.Tensor,
) -> torch.Tensor:
    """Support-derived old-class evidence calibrated without eval unknown labels."""

    return 0.5 * cosine + 0.5 * anchor_similarity - residual - 0.05 * torch.sqrt(mahalanobis.clamp_min(0.0))


def _support_anchor_similarity(features: torch.Tensor, state: ClassState) -> torch.Tensor:
    anchors = state.support_anchors
    if anchors is None or not torch.is_tensor(anchors) or anchors.numel() == 0:
        prototype = torch.as_tensor(state.prototype, dtype=features.dtype, device=features.device).view(1, -1)
        return features @ normalize_rows(prototype).T.squeeze(1)
    anchor_bank = normalize_rows(torch.as_tensor(anchors, dtype=features.dtype, device=features.device))
    return (features @ anchor_bank.T).max(dim=1).values


def _leave_one_out_anchor_similarity(members: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    support = normalize_rows(torch.as_tensor(members).float())
    anchor_bank = normalize_rows(torch.as_tensor(anchors).float()).to(support.device)
    sims = support @ anchor_bank.T
    if anchor_bank.size(0) <= 1:
        return sims.max(dim=1).values
    top2 = sims.topk(k=min(2, sims.size(1)), dim=1).values
    return top2[:, 1]


def _old_support_quality_stats(
    members: torch.Tensor | None,
    source_vec: torch.Tensor,
    *,
    kappa: float = 3.0,
) -> dict[str, float]:
    if members is None or not torch.is_tensor(members) or members.numel() == 0:
        return {
            "support_count": 0.0,
            "support_quality": 0.0,
            "support_source_similarity_mean": 0.0,
            "support_source_mean_similarity": 0.0,
            "support_compactness": 0.0,
            "support_count_term": 0.0,
        }

    support = normalize_rows(torch.as_tensor(members).float())
    source = normalize_rows(torch.as_tensor(source_vec).float().view(1, -1)).squeeze(0).to(support.device)
    n = int(support.shape[0])
    source_sims = support @ source
    source_mean_similarity = float(source_sims.mean().item())
    target_mean = normalize_rows(support.mean(dim=0, keepdim=True)).squeeze(0)
    source_target_similarity = float(torch.dot(target_mean, source).item())
    if n > 1:
        sims = support @ support.T
        tri = torch.triu_indices(n, n, offset=1, device=support.device)
        compactness = float(sims[tri[0], tri[1]].mean().item()) if tri.numel() > 0 else source_target_similarity
    else:
        compactness = source_target_similarity

    source_term = max(0.0, min(1.0, (source_target_similarity + 1.0) * 0.5))
    compactness_term = max(0.0, min(1.0, (compactness + 1.0) * 0.5))
    count_term = n / (n + max(0.0, float(kappa)))
    quality = max(0.0, min(1.0, 0.50 * source_term + 0.30 * compactness_term + 0.20 * count_term))
    return {
        "support_count": float(n),
        "support_quality": float(quality),
        "support_source_similarity_mean": float(source_mean_similarity),
        "support_source_mean_similarity": float(source_target_similarity),
        "support_compactness": float(compactness),
        "support_count_term": float(count_term),
    }


def _prototype_set_from_class_states(class_states: Mapping[int, ClassState]) -> PrototypeSet:
    labels = sorted(int(k) for k in class_states)
    vectors = [normalize_rows(torch.as_tensor(class_states[label].prototype).float().view(1, -1)).squeeze(0) for label in labels]
    diag = [torch.as_tensor(class_states[label].covariance_diag).float() for label in labels]
    openmax = [
        torch.tensor(float(class_states[label].evt_params.get("tail_threshold", class_states[label].thresholds.get("max_residual", 1.0))))
        for label in labels
    ]
    maha = [torch.tensor(float(class_states[label].thresholds.get("max_mahalanobis", 1.0e6))) for label in labels]
    return PrototypeSet(
        labels=torch.tensor(labels, dtype=torch.long, device=vectors[0].device),
        vectors=torch.stack(vectors, dim=0),
        counts=torch.tensor([0 for _ in labels], dtype=torch.long, device=vectors[0].device),
        metadata={
            "diag_var": torch.stack(diag, dim=0),
            "openmax_thresholds": torch.stack(openmax, dim=0).to(vectors[0].device),
            "mahalanobis_thresholds": torch.stack(maha, dim=0).to(vectors[0].device),
            "oa_mse_groups": {int(label): str(class_states[label].group) for label in labels},
        },
    )


def predict_with_oa_mse_head(
    features: torch.Tensor,
    head: OrbitAdaptiveMSEHead,
    *,
    quality_scores: torch.Tensor | Iterable[float] | None = None,
    quality_threshold: float | None = None,
) -> PredictionResult:
    h = normalize_rows(torch.as_tensor(features).float())
    score_matrix, per_class = head.score_matrix(h)
    topk = score_matrix.topk(min(2, score_matrix.size(1)), dim=1)
    top_scores = topk.values[:, 0]
    top_order_idx = topk.indices[:, 0]
    if topk.values.size(1) > 1:
        margins = topk.values[:, 0] - topk.values[:, 1]
    else:
        margins = topk.values[:, 0]
    energies = head.energy(score_matrix)
    group_by_label = {int(label): str(head.class_states[int(label)].group) for label in head.class_order}
    old_indices = [idx for idx, label in enumerate(head.class_order) if group_by_label[int(label)] == "old"]
    seen_new_indices = [idx for idx, label in enumerate(head.class_order) if group_by_label[int(label)] == "seen_new"]

    def best_group_scores(indices: list[int]) -> torch.Tensor:
        if not indices:
            return torch.full_like(top_scores, float("nan"))
        return score_matrix[:, indices].max(dim=1).values

    best_old_score = best_group_scores(old_indices)
    best_seen_new_score = best_group_scores(seen_new_indices)
    label_tensor = torch.tensor(head.class_order, dtype=torch.long, device=score_matrix.device)

    def best_group_labels(indices: list[int]) -> torch.Tensor:
        if not indices:
            return torch.full((score_matrix.size(0),), UNKNOWN_LABEL, dtype=torch.long, device=score_matrix.device)
        local_best = torch.argmax(score_matrix[:, indices], dim=1)
        group_labels = torch.tensor([head.class_order[idx] for idx in indices], dtype=torch.long, device=score_matrix.device)
        return group_labels[local_best]

    diagnostics: dict[str, torch.Tensor] = {
        "best_old_label": best_group_labels(old_indices).detach(),
        "best_seen_new_label": best_group_labels(seen_new_indices).detach(),
        "best_old_score": best_old_score.detach(),
        "best_seen_new_score": best_seen_new_score.detach(),
        "seen_new_minus_old_score": (best_seen_new_score - best_old_score).detach(),
    }

    selected_labels = torch.tensor([head.class_order[int(i)] for i in top_order_idx.tolist()], dtype=torch.long)
    accepted = torch.zeros_like(top_scores, dtype=torch.bool)
    predicted = torch.full_like(selected_labels, UNKNOWN_LABEL)
    decisions: list[str] = []
    reasons: list[str] = []
    residuals = []
    mahalas = []
    openmax_distances = []
    residual_deltas = []
    mahalanobis_deltas = []
    margin_deltas = []
    energy_deltas = []
    evt_deltas = []
    min_accept_deltas = []
    q = None if quality_scores is None else torch.as_tensor(quality_scores, dtype=torch.float32).reshape(-1)
    seen_new_labels = [label for label in head.class_order if str(head.class_states[int(label)].group) == "seen_new"]
    seen_new_evidence = None
    seen_new_support_affinity = None
    seen_new_support_residual = None
    seen_new_anchor_similarity = None
    seen_new_anchor_delta = None
    seen_new_evidence_delta = None
    seen_new_affinity_delta = None
    seen_new_residual_delta = None
    old_support_anchor_similarity = None
    old_support_anchor_delta = None
    old_support_anchor_margin = None
    old_support_evidence = None
    old_support_evidence_delta = None
    old_surrogate_evidence_delta = None
    old_surrogate_reject_evidence_delta = None
    old_support_quality = None
    old_support_quality_delta = None
    anchor_density_selected = None
    anchor_density_margin = None
    anchor_density_delta = None
    anchor_density_margin_delta = None
    support_knn_label = None
    support_knn_score = None
    support_knn_margin = None
    support_knn_old_score = None
    support_knn_seen_new_score = None
    support_knn_seen_new_minus_old = None
    support_knn_topk = None
    soft_mixture_score_selected = None
    soft_mixture_cos_selected = None
    soft_mixture_residual_selected = None
    soft_mixture_maha_selected = None
    soft_mixture_score_margin = None
    soft_mixture_consistency_pass = None
    density_cols = []
    soft_score_cols = []
    soft_cos_cols = []
    soft_residual_cols = []
    soft_maha_cols = []
    for label in head.class_order:
        row = per_class[int(label)]
        if "anchor_density" in row:
            density_cols.append(row["anchor_density"].detach())
        else:
            density_cols.append(torch.full_like(top_scores, -float("inf")))
        soft_score_cols.append(row.get("soft_mixture_score", torch.full_like(top_scores, -float("inf"))).detach())
        soft_cos_cols.append(row.get("soft_mixture_cos", torch.full_like(top_scores, -float("inf"))).detach())
        soft_residual_cols.append(row.get("soft_mixture_residual", torch.full_like(top_scores, float("inf"))).detach())
        soft_maha_cols.append(row.get("soft_mixture_maha", torch.full_like(top_scores, float("inf"))).detach())
    if density_cols:
        density_matrix = torch.stack(density_cols, dim=1)
        anchor_density_selected = density_matrix.gather(1, top_order_idx.view(-1, 1)).squeeze(1)
        support_knn_score, support_knn_idx = density_matrix.max(dim=1)
        support_knn_label = label_tensor[support_knn_idx]
        support_knn_topk = torch.tensor(
            [
                int(head.class_states[int(head.class_order[int(idx)])].thresholds.get("anchor_density_topk", 3))
                for idx in support_knn_idx.detach().cpu().tolist()
            ],
            dtype=torch.float32,
            device=score_matrix.device,
        )
        if density_matrix.size(1) > 1:
            density_top2 = density_matrix.topk(2, dim=1).values
            anchor_density_margin = density_top2[:, 0] - density_top2[:, 1]
            support_knn_margin = anchor_density_margin
        else:
            anchor_density_margin = torch.full_like(anchor_density_selected, 1.0)
            support_knn_margin = torch.full_like(support_knn_score, 1.0)
        support_knn_old_score = best_group_scores(old_indices) if old_indices else torch.full_like(support_knn_score, float("nan"))
        support_knn_seen_new_score = (
            best_group_scores(seen_new_indices) if seen_new_indices else torch.full_like(support_knn_score, float("nan"))
        )
        if old_indices:
            support_knn_old_score = density_matrix[:, old_indices].max(dim=1).values
        if seen_new_indices:
            support_knn_seen_new_score = density_matrix[:, seen_new_indices].max(dim=1).values
        support_knn_seen_new_minus_old = support_knn_seen_new_score - support_knn_old_score
    if soft_score_cols:
        soft_score_matrix = torch.stack(soft_score_cols, dim=1)
        soft_mixture_score_selected = soft_score_matrix.gather(1, top_order_idx.view(-1, 1)).squeeze(1)
        soft_mixture_cos_selected = torch.stack(soft_cos_cols, dim=1).gather(1, top_order_idx.view(-1, 1)).squeeze(1)
        soft_mixture_residual_selected = torch.stack(soft_residual_cols, dim=1).gather(1, top_order_idx.view(-1, 1)).squeeze(1)
        soft_mixture_maha_selected = torch.stack(soft_maha_cols, dim=1).gather(1, top_order_idx.view(-1, 1)).squeeze(1)
        if soft_score_matrix.size(1) > 1:
            soft_top2 = soft_score_matrix.topk(2, dim=1).values
            soft_mixture_score_margin = soft_top2[:, 0] - soft_top2[:, 1]
        else:
            soft_mixture_score_margin = torch.full_like(soft_mixture_score_selected, 1.0)
        soft_mixture_consistency_pass = torch.ones_like(soft_mixture_score_selected, dtype=torch.bool)
    if old_indices:
        anchor_cols = []
        anchor_delta_cols = []
        evidence_cols = []
        evidence_delta_cols = []
        surrogate_delta_cols = []
        surrogate_reject_delta_cols = []
        support_quality_cols = []
        support_quality_delta_cols = []
        drift_cos_cols = []
        drift_dist_cols = []
        effective_rho_cols = []
        support_count_cols = []
        support_compactness_cols = []
        for label in [head.class_order[idx] for idx in old_indices]:
            state = head.class_states[int(label)]
            cos = per_class[int(label)]["cos"].detach()
            residual = per_class[int(label)]["residual"].detach()
            maha = per_class[int(label)]["maha"].detach()
            anchor = _support_anchor_similarity(h, state).detach()
            min_anchor = _state_threshold(state, "min_old_support_anchor_similarity", -1.0)
            evidence = _old_support_evidence_score(cos, anchor, residual, maha)
            min_evidence = _state_threshold(state, "min_old_support_evidence", -1.0)
            min_surrogate_evidence = _state_threshold(state, "min_old_surrogate_evidence", -1.0)
            min_surrogate_reject_evidence = _state_threshold(state, "min_old_surrogate_reject_evidence", min_surrogate_evidence)
            min_override_quality = _state_threshold(state, "min_old_anchor_override_quality", 1.0)
            quality = torch.full_like(cos, float(state.support_quality))
            anchor_cols.append(anchor)
            anchor_delta_cols.append(anchor - float(min_anchor))
            evidence_cols.append(evidence)
            evidence_delta_cols.append(evidence - float(min_evidence))
            surrogate_delta_cols.append(evidence - float(min_surrogate_evidence))
            surrogate_reject_delta_cols.append(evidence - float(min_surrogate_reject_evidence))
            support_quality_cols.append(quality)
            support_quality_delta_cols.append(quality - float(min_override_quality))
            drift_cos = float(state.thresholds.get("support_source_mean_similarity", 1.0))
            drift_cos_cols.append(torch.full_like(cos, drift_cos))
            drift_dist_cols.append(torch.full_like(cos, 1.0 - drift_cos))
            effective_rho_cols.append(torch.full_like(cos, float(state.thresholds.get("effective_rho", 0.0))))
            support_count_cols.append(torch.full_like(cos, float(state.thresholds.get("support_count", 0.0))))
            support_compactness_cols.append(torch.full_like(cos, float(state.thresholds.get("support_compactness", 0.0))))
        anchor_matrix = torch.stack(anchor_cols, dim=1)
        best_old = torch.argmax(score_matrix[:, old_indices], dim=1)
        old_support_anchor_similarity = anchor_matrix.gather(1, best_old.view(-1, 1)).squeeze(1)
        if anchor_matrix.size(1) > 1:
            anchor_top2 = anchor_matrix.topk(2, dim=1).values
            old_support_anchor_margin = anchor_top2[:, 0] - anchor_top2[:, 1]
        else:
            old_support_anchor_margin = torch.full_like(old_support_anchor_similarity, 1.0)
        old_support_anchor_delta = torch.stack(anchor_delta_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1)
        old_support_evidence = torch.stack(evidence_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1)
        old_support_evidence_delta = torch.stack(evidence_delta_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1)
        old_surrogate_evidence_delta = torch.stack(surrogate_delta_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1)
        old_surrogate_reject_evidence_delta = torch.stack(surrogate_reject_delta_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1)
        old_support_quality = torch.stack(support_quality_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1)
        old_support_quality_delta = torch.stack(support_quality_delta_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1)
        diagnostics["old_drift_cos"] = torch.stack(drift_cos_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1).detach()
        diagnostics["old_drift_dist"] = torch.stack(drift_dist_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1).detach()
        diagnostics["old_effective_rho"] = torch.stack(effective_rho_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1).detach()
        diagnostics["old_support_count"] = torch.stack(support_count_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1).detach()
        diagnostics["old_support_compactness"] = torch.stack(support_compactness_cols, dim=1).gather(1, best_old.view(-1, 1)).squeeze(1).detach()
    if seen_new_labels:
        evidence_cols = []
        affinity_cols = []
        residual_cols = []
        anchor_cols = []
        anchor_delta_cols = []
        evidence_delta_cols = []
        affinity_delta_cols = []
        residual_delta_cols = []
        for label in seen_new_labels:
            state = head.class_states[int(label)]
            cos = per_class[int(label)]["cos"].detach()
            residual = per_class[int(label)]["residual"].detach()
            maha = per_class[int(label)]["maha"].detach()
            anchor = _support_anchor_similarity(h, state).detach()
            evidence = _seen_new_evidence_score(cos, residual, maha)
            min_affinity = _state_threshold(state, "min_seen_new_support_affinity", -1.0)
            max_support_residual = _state_threshold(state, "max_seen_new_support_residual", 1.0)
            min_evidence = _state_threshold(state, "min_seen_new_evidence", -1.0)
            min_anchor = _state_threshold(state, "min_seen_new_anchor_similarity", -1.0)
            evidence_cols.append(evidence)
            affinity_cols.append(cos)
            residual_cols.append(residual)
            anchor_cols.append(anchor)
            anchor_delta_cols.append(anchor - float(min_anchor))
            evidence_delta_cols.append(evidence - float(min_evidence))
            affinity_delta_cols.append(cos - float(min_affinity))
            residual_delta_cols.append(float(max_support_residual) - residual)
        evidence_matrix = torch.stack(evidence_cols, dim=1)
        best_seen_new = torch.argmax(evidence_matrix, dim=1)
        seen_new_evidence = evidence_matrix.gather(1, best_seen_new.view(-1, 1)).squeeze(1)
        seen_new_support_affinity = torch.stack(affinity_cols, dim=1).gather(1, best_seen_new.view(-1, 1)).squeeze(1)
        seen_new_support_residual = torch.stack(residual_cols, dim=1).gather(1, best_seen_new.view(-1, 1)).squeeze(1)
        seen_new_anchor_similarity = torch.stack(anchor_cols, dim=1).gather(1, best_seen_new.view(-1, 1)).squeeze(1)
        seen_new_anchor_delta = torch.stack(anchor_delta_cols, dim=1).gather(1, best_seen_new.view(-1, 1)).squeeze(1)
        seen_new_evidence_delta = torch.stack(evidence_delta_cols, dim=1).gather(1, best_seen_new.view(-1, 1)).squeeze(1)
        seen_new_affinity_delta = torch.stack(affinity_delta_cols, dim=1).gather(1, best_seen_new.view(-1, 1)).squeeze(1)
        seen_new_residual_delta = torch.stack(residual_delta_cols, dim=1).gather(1, best_seen_new.view(-1, 1)).squeeze(1)

    for row, label in enumerate(selected_labels.tolist()):
        state = head.class_states[int(label)]
        residual = per_class[int(label)]["residual"][row].detach()
        maha = per_class[int(label)]["maha"][row].detach()
        residuals.append(residual)
        mahalas.append(maha)
        openmax_distances.append(residual.clamp_min(0.0))
        max_residual = _state_threshold(state, "max_residual", float("inf"))
        max_mahalanobis = _state_threshold(state, "max_mahalanobis", float("inf"))
        min_margin = _state_threshold(state, "min_margin", -float("inf"))
        max_energy = _state_threshold(state, "max_energy", float("inf"))
        surrogate_reject_energy = _state_threshold(state, "surrogate_reject_energy", float("inf"))
        max_evt = _state_evt_tail_threshold(state, max_residual)
        residual_delta = torch.tensor(max_residual - float(residual.item()), dtype=torch.float32)
        maha_delta = torch.tensor(max_mahalanobis - float(maha.item()), dtype=torch.float32)
        margin_delta = torch.tensor(float(margins[row].item()) - min_margin, dtype=torch.float32)
        energy_delta = torch.tensor(max_energy - float(energies[row].item()), dtype=torch.float32)
        evt_delta = torch.tensor(max_evt - float(residual.item()), dtype=torch.float32)
        residual_deltas.append(residual_delta)
        mahalanobis_deltas.append(maha_delta)
        margin_deltas.append(margin_delta)
        energy_deltas.append(energy_delta)
        evt_deltas.append(evt_delta)
        min_accept_deltas.append(torch.stack([residual_delta, maha_delta, margin_delta, energy_delta, evt_delta]).min())
        if q is not None and quality_threshold is not None and row < q.numel() and float(q[row].item()) < float(quality_threshold):
            decisions.append("defer")
            reasons.append("low_quality")
            continue
        if bool(state.thresholds.get("soft_mixture_consistency_gate_enabled", False)):
            mix_cos = (
                soft_mixture_cos_selected[row].detach()
                if soft_mixture_cos_selected is not None
                else torch.tensor(-float("inf"), dtype=torch.float32)
            )
            mix_residual = (
                soft_mixture_residual_selected[row].detach()
                if soft_mixture_residual_selected is not None
                else torch.tensor(float("inf"), dtype=torch.float32)
            )
            mix_margin = (
                soft_mixture_score_margin[row].detach()
                if soft_mixture_score_margin is not None
                else torch.tensor(-float("inf"), dtype=torch.float32)
            )
            min_mix_cos = _state_threshold(state, "soft_mixture_min_cos", -float("inf"))
            max_mix_residual = _state_threshold(state, "soft_mixture_max_residual", float("inf"))
            min_mix_margin = _state_threshold(state, "soft_mixture_min_margin", -float("inf"))
            mixture_failed = (
                float(mix_cos.item()) < min_mix_cos
                or float(mix_residual.item()) > max_mix_residual
                or float(mix_margin.item()) < min_mix_margin
            )
            if soft_mixture_consistency_pass is not None:
                soft_mixture_consistency_pass[row] = not mixture_failed
            if mixture_failed:
                action = str(state.thresholds.get("soft_mixture_consistency_action", "uncertain")).lower()
                if action == "reject":
                    decisions.append("reject")
                    reasons.append("soft_mixture_consistency_reject")
                elif action == "defer":
                    decisions.append("defer")
                    reasons.append("soft_mixture_consistency_defer")
                else:
                    decisions.append("uncertain")
                    reasons.append("soft_mixture_consistency_uncertain")
                continue
        if bool(state.thresholds.get("anchor_density_gate_enabled", False)):
            min_density = _state_threshold(state, "min_anchor_density", -float("inf"))
            min_density_margin = _state_threshold(state, "min_anchor_density_margin", -float("inf"))
            density = (
                anchor_density_selected[row].detach()
                if anchor_density_selected is not None
                else torch.tensor(-float("inf"), dtype=torch.float32)
            )
            density_margin = (
                anchor_density_margin[row].detach()
                if anchor_density_margin is not None
                else torch.tensor(-float("inf"), dtype=torch.float32)
            )
            density_failed = float(density.item()) < min_density
            margin_failed = float(density_margin.item()) < min_density_margin
            if density_failed or margin_failed:
                action = str(state.thresholds.get("anchor_density_gate_action", "uncertain")).lower()
                decisions.append("reject" if action == "reject" else "uncertain")
                reasons.append("anchor_density_reject" if action == "reject" else "anchor_density_uncertain")
                continue
        if str(state.group) == "seen_new":
            seen_gate_fields = (
                "min_seen_new_support_affinity",
                "max_seen_new_support_residual",
                "min_seen_new_evidence",
                "min_seen_new_anchor_similarity",
            )
            has_seen_new_gate = any(name in state.thresholds for name in seen_gate_fields)
            if has_seen_new_gate:
                affinity = per_class[int(label)]["cos"][row].detach()
                evidence = _seen_new_evidence_score(affinity, residual, maha)
                anchor = _support_anchor_similarity(h[row].view(1, -1), state)[0].detach()
                min_affinity = _state_threshold(state, "min_seen_new_support_affinity", -float("inf"))
                max_support_residual = _state_threshold(state, "max_seen_new_support_residual", float("inf"))
                min_evidence = _state_threshold(state, "min_seen_new_evidence", -float("inf"))
                min_anchor = _state_threshold(state, "min_seen_new_anchor_similarity", -float("inf"))
                anchor_failed = float(anchor.item()) < min_anchor
                geometry_failed = (
                    float(affinity.item()) < min_affinity
                    or float(residual.item()) > max_support_residual
                    or float(evidence.item()) < min_evidence
                )
                if anchor_failed or geometry_failed:
                    decisions.append("reject")
                    reasons.append("seen_new_anchor_reject" if anchor_failed else "seen_new_evidence_reject")
                    continue
        if str(state.group) == "old" and "min_old_support_anchor_similarity" in state.thresholds:
            anchor = _support_anchor_similarity(h[row].view(1, -1), state)[0].detach()
            min_anchor = _state_threshold(state, "min_old_support_anchor_similarity", -float("inf"))
            if float(anchor.item()) < min_anchor:
                decisions.append("uncertain")
                reasons.append("old_support_anchor_uncertain")
                continue
        if str(state.group) == "old":
            old_gate_fields = ("min_old_support_evidence", "min_old_surrogate_evidence", "min_old_surrogate_reject_evidence")
            if any(name in state.thresholds for name in old_gate_fields):
                affinity = per_class[int(label)]["cos"][row].detach()
                anchor = _support_anchor_similarity(h[row].view(1, -1), state)[0].detach()
                evidence = _old_support_evidence_score(affinity, anchor, residual, maha)
                min_support_evidence = _state_threshold(state, "min_old_support_evidence", -float("inf"))
                min_surrogate_evidence = _state_threshold(state, "min_old_surrogate_evidence", -float("inf"))
                min_surrogate_reject_evidence = _state_threshold(
                    state,
                    "min_old_surrogate_reject_evidence",
                    min_surrogate_evidence,
                )
                if float(evidence.item()) < min_support_evidence:
                    decisions.append("uncertain")
                    reasons.append("old_support_evidence_uncertain")
                    continue
                if float(evidence.item()) < min_surrogate_reject_evidence:
                    decisions.append("reject")
                    reasons.append("old_surrogate_evidence_reject")
                    continue
                if float(evidence.item()) < min_surrogate_evidence:
                    decisions.append("uncertain")
                    reasons.append("old_surrogate_evidence_uncertain")
                    continue
        if float(energies[row].item()) >= surrogate_reject_energy:
            allow_old_anchor_override = False
            if str(state.group) == "old" and "min_old_support_anchor_similarity" in state.thresholds:
                anchor = _support_anchor_similarity(h[row].view(1, -1), state)[0].detach()
                min_anchor = _state_threshold(state, "min_old_support_anchor_similarity", -float("inf"))
                affinity = per_class[int(label)]["cos"][row].detach()
                evidence = _old_support_evidence_score(affinity, anchor, residual, maha)
                min_support_evidence = _state_threshold(state, "min_old_support_evidence", -float("inf"))
                min_surrogate_evidence = _state_threshold(state, "min_old_surrogate_evidence", -float("inf"))
                min_surrogate_reject_evidence = _state_threshold(
                    state,
                    "min_old_surrogate_reject_evidence",
                    min_surrogate_evidence,
                )
                min_override_quality = _state_threshold(state, "min_old_anchor_override_quality", float("inf"))
                allow_old_anchor_override = (
                    float(state.support_quality) >= min_override_quality
                    and float(anchor.item()) >= min_anchor
                    and float(evidence.item()) >= min_support_evidence
                    and float(evidence.item()) >= min_surrogate_evidence
                    and float(evidence.item()) >= min_surrogate_reject_evidence
                )
            if not allow_old_anchor_override:
                decisions.append("reject")
                reasons.append("surrogate_energy_reject")
                continue
        passes_accept = (
            float(residual.item()) <= max_residual
            and float(maha.item()) <= max_mahalanobis
            and float(margins[row].item()) >= min_margin
            and float(energies[row].item()) <= max_energy
            and float(residual.item()) <= max_evt
        )
        if passes_accept:
            accepted[row] = True
            predicted[row] = int(label)
            decisions.append("accept")
            reasons.append("accepted")
            continue

        reject_residual = _state_threshold(state, "reject_residual", max_residual)
        reject_mahalanobis = _state_threshold(state, "reject_mahalanobis", max_mahalanobis)
        reject_energy = _state_threshold(state, "reject_energy", float("inf"))
        if (
            float(residual.item()) > reject_residual
            or float(maha.item()) > reject_mahalanobis
            or float(energies[row].item()) > reject_energy
        ):
            decisions.append("reject")
            reasons.append("oa_mse_reject")
        else:
            decisions.append("uncertain")
            reasons.append("oa_mse_uncertain")

    diagnostics.update(
        {
            "residual_delta": torch.stack(residual_deltas).detach() if residual_deltas else torch.empty(0),
            "mahalanobis_delta": torch.stack(mahalanobis_deltas).detach() if mahalanobis_deltas else torch.empty(0),
            "margin_delta": torch.stack(margin_deltas).detach() if margin_deltas else torch.empty(0),
            "energy_delta": torch.stack(energy_deltas).detach() if energy_deltas else torch.empty(0),
            "evt_delta": torch.stack(evt_deltas).detach() if evt_deltas else torch.empty(0),
            "min_accept_delta": torch.stack(min_accept_deltas).detach() if min_accept_deltas else torch.empty(0),
        }
    )
    if seen_new_evidence_delta is not None:
        diagnostics["seen_new_evidence_delta"] = seen_new_evidence_delta.detach()
    if seen_new_affinity_delta is not None:
        diagnostics["seen_new_affinity_delta"] = seen_new_affinity_delta.detach()
    if seen_new_residual_delta is not None:
        diagnostics["seen_new_residual_delta"] = seen_new_residual_delta.detach()
    if old_support_anchor_similarity is not None:
        diagnostics["old_support_anchor_similarity"] = old_support_anchor_similarity.detach()
    if old_support_anchor_margin is not None:
        diagnostics["old_support_anchor_margin"] = old_support_anchor_margin.detach()
    if old_support_anchor_delta is not None:
        diagnostics["old_support_anchor_delta"] = old_support_anchor_delta.detach()
    if old_support_evidence is not None:
        diagnostics["old_support_evidence"] = old_support_evidence.detach()
    if old_support_evidence_delta is not None:
        diagnostics["old_support_evidence_delta"] = old_support_evidence_delta.detach()
    if old_surrogate_evidence_delta is not None:
        diagnostics["old_surrogate_evidence_delta"] = old_surrogate_evidence_delta.detach()
    if old_surrogate_reject_evidence_delta is not None:
        diagnostics["old_surrogate_reject_evidence_delta"] = old_surrogate_reject_evidence_delta.detach()
    if old_support_quality is not None:
        diagnostics["old_support_quality"] = old_support_quality.detach()
    if old_support_quality_delta is not None:
        diagnostics["old_support_quality_delta"] = old_support_quality_delta.detach()
    if anchor_density_selected is not None:
        diagnostics["anchor_density"] = anchor_density_selected.detach()
        diagnostics["support_knn_label"] = support_knn_label.detach()
        diagnostics["support_knn_score"] = support_knn_score.detach()
        diagnostics["support_knn_margin"] = support_knn_margin.detach()
        diagnostics["support_knn_old_score"] = support_knn_old_score.detach()
        diagnostics["support_knn_seen_new_score"] = support_knn_seen_new_score.detach()
        diagnostics["support_knn_seen_new_minus_old"] = support_knn_seen_new_minus_old.detach()
        diagnostics["support_knn_topk"] = support_knn_topk.detach()
        density_delta_cols = []
        margin_delta_cols = []
        for label in head.class_order:
            state = head.class_states[int(label)]
            min_density = _state_threshold(state, "min_anchor_density", -float("inf"))
            min_density_margin = _state_threshold(state, "min_anchor_density_margin", -float("inf"))
            density_delta_cols.append(density_matrix[:, head.class_order.index(label)] - float(min_density))
            margin_delta_cols.append(torch.full_like(top_scores, float("inf") if min_density_margin == -float("inf") else -float(min_density_margin)))
        anchor_density_delta = torch.stack(density_delta_cols, dim=1).gather(1, top_order_idx.view(-1, 1)).squeeze(1)
        anchor_density_margin_delta = anchor_density_margin - torch.tensor(
            [
                _state_threshold(head.class_states[int(head.class_order[int(i)])], "min_anchor_density_margin", -float("inf"))
                for i in top_order_idx.tolist()
            ],
            dtype=anchor_density_margin.dtype,
            device=anchor_density_margin.device,
        )
        diagnostics["anchor_density_margin"] = anchor_density_margin.detach()
        diagnostics["anchor_density_delta"] = anchor_density_delta.detach()
        diagnostics["anchor_density_margin_delta"] = anchor_density_margin_delta.detach()
    if soft_mixture_score_selected is not None:
        diagnostics["soft_mixture_score"] = soft_mixture_score_selected.detach()
        diagnostics["soft_mixture_cos"] = soft_mixture_cos_selected.detach()
        diagnostics["soft_mixture_residual"] = soft_mixture_residual_selected.detach()
        diagnostics["soft_mixture_maha"] = soft_mixture_maha_selected.detach()
        diagnostics["soft_mixture_score_margin"] = soft_mixture_score_margin.detach()
        diagnostics["soft_mixture_consistency_pass_mask"] = soft_mixture_consistency_pass.detach()
    return PredictionResult(
        predicted_labels=predicted.cpu(),
        scores=top_scores.detach().cpu(),
        accepted=accepted.cpu(),
        candidate_labels=selected_labels.detach().cpu(),
        diagnostics={key: value.detach().cpu() for key, value in diagnostics.items()},
        margins=margins.detach().cpu(),
        mahalanobis=torch.stack(mahalas).detach().cpu() if mahalas else None,
        openmax_distance=torch.stack(openmax_distances).detach().cpu() if openmax_distances else None,
        gate_reasons=reasons,
        decisions=decisions,
        energy=energies.detach().cpu(),
        subspace_residual=torch.stack(residuals).detach().cpu() if residuals else None,
        seen_new_evidence=seen_new_evidence.detach().cpu() if seen_new_evidence is not None else None,
        seen_new_support_affinity=seen_new_support_affinity.detach().cpu() if seen_new_support_affinity is not None else None,
        seen_new_support_residual=seen_new_support_residual.detach().cpu() if seen_new_support_residual is not None else None,
        seen_new_anchor_similarity=seen_new_anchor_similarity.detach().cpu() if seen_new_anchor_similarity is not None else None,
        seen_new_anchor_delta=seen_new_anchor_delta.detach().cpu() if seen_new_anchor_delta is not None else None,
    )


def _old80_first_head_scores(
    features: torch.Tensor,
    class_states: Mapping[int, ClassState],
    *,
    mode: str,
    fusion_rho: float,
    knn_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    mode = str(mode).lower()
    if mode == "support_cv_select":
        mode, cv_stats = _select_old80_first_head_mode(
            class_states,
            fusion_rho=float(fusion_rho),
            knn_k=int(knn_k),
        )
    else:
        cv_stats = {"support_cv_acc": float("nan"), "support_cv_count": 0.0}
    old_items = [
        (int(label), state)
        for label, state in sorted(class_states.items())
        if str(state.group) == "old"
    ]
    if not old_items:
        raise ValueError("OLD80_FIRST head requires at least one old class state")
    x = normalize_rows(torch.as_tensor(features).float())
    device = x.device
    labels = torch.tensor([label for label, _ in old_items], dtype=torch.long, device=device)

    if mode.startswith("support_knn"):
        anchors = []
        anchor_labels = []
        for label, state in old_items:
            support = getattr(state, "support_anchors", None)
            if support is None or int(torch.as_tensor(support).numel()) == 0:
                support = torch.as_tensor(state.prototype).float().view(1, -1)
            support = normalize_rows(torch.as_tensor(support).float()).to(device)
            anchors.append(support)
            anchor_labels.extend([int(label)] * int(support.shape[0]))
        anchor_matrix = torch.cat(anchors, dim=0)
        anchor_label_tensor = torch.tensor(anchor_labels, dtype=torch.long, device=device)
        sims = x @ anchor_matrix.T
        k = min(max(1, int(knn_k)), int(anchor_matrix.shape[0]))
        top = sims.topk(k, dim=1)
        class_scores = []
        for label in labels.tolist():
            label_match = anchor_label_tensor[top.indices] == int(label)
            label_scores = torch.where(label_match, top.values, torch.full_like(top.values, -float("inf")))
            class_scores.append(label_scores.max(dim=1).values)
        score_matrix = torch.stack(class_scores, dim=1)
    else:
        vectors = []
        for _, state in old_items:
            proto = normalize_rows(torch.as_tensor(state.prototype).float().view(1, -1)).squeeze(0)
            support = getattr(state, "support_anchors", None)
            if support is not None and int(torch.as_tensor(support).numel()) > 0 and mode in {"support_centroid", "fused_centroid"}:
                support_mean = normalize_rows(torch.as_tensor(support).float().mean(dim=0, keepdim=True)).squeeze(0)
                if mode == "support_centroid":
                    proto = support_mean
                else:
                    rho = max(0.0, min(1.0, float(fusion_rho)))
                    proto = normalize_rows((rho * support_mean + (1.0 - rho) * proto).view(1, -1)).squeeze(0)
            vectors.append(proto.to(device))
        proto_matrix = normalize_rows(torch.stack(vectors, dim=0))
        score_matrix = x @ proto_matrix.T

    topk = score_matrix.topk(min(2, score_matrix.size(1)), dim=1)
    scores = topk.values[:, 0]
    indices = topk.indices[:, 0]
    margins = topk.values[:, 0] - topk.values[:, 1] if topk.values.size(1) > 1 else topk.values[:, 0]
    return labels[indices], scores, margins, labels, {
        **cv_stats,
        "selected_mode_code": float(
            {
                "fused_centroid": 1,
                "support_centroid": 2,
                "support_knn1": 3,
                "support_knn3": 4,
            }.get(mode, 0)
        ),
    }


def _old80_first_support_cv_accuracy(
    class_states: Mapping[int, ClassState],
    *,
    mode: str,
    fusion_rho: float,
    knn_k: int,
) -> tuple[float, int]:
    support_rows = []
    support_labels = []
    for label, state in sorted(class_states.items()):
        if str(state.group) != "old":
            continue
        support = getattr(state, "support_anchors", None)
        if support is None or int(torch.as_tensor(support).numel()) == 0:
            continue
        support = normalize_rows(torch.as_tensor(support).float())
        support_rows.append(support)
        support_labels.extend([int(label)] * int(support.shape[0]))
    if not support_rows:
        return float("nan"), 0
    support_matrix = torch.cat(support_rows, dim=0)
    label_tensor = torch.tensor(support_labels, dtype=torch.long)
    correct = 0
    evaluated = 0
    for row in range(int(support_matrix.shape[0])):
        cv_states = {}
        for label, state in class_states.items():
            if str(state.group) != "old":
                continue
            anchors = getattr(state, "support_anchors", None)
            if anchors is None or int(torch.as_tensor(anchors).numel()) == 0:
                cv_states[int(label)] = state
                continue
            anchors = normalize_rows(torch.as_tensor(anchors).float())
            keep = torch.ones(int(anchors.shape[0]), dtype=torch.bool)
            if int(label) == int(label_tensor[row].item()):
                class_index = int((label_tensor[:row] == int(label)).sum().item())
                if class_index < int(keep.numel()):
                    keep[class_index] = False
            cv_states[int(label)] = replace(state, support_anchors=anchors[keep] if bool(keep.any().item()) else None)
        pred_labels, _, _, _, _ = _old80_first_head_scores(
            support_matrix[row : row + 1],
            cv_states,
            mode=mode,
            fusion_rho=float(fusion_rho),
            knn_k=int(knn_k),
        )
        correct += int(int(pred_labels[0].item()) == int(label_tensor[row].item()))
        evaluated += 1
    return float(correct / max(1, evaluated)), evaluated


def _select_old80_first_head_mode(
    class_states: Mapping[int, ClassState],
    *,
    fusion_rho: float,
    knn_k: int,
) -> tuple[str, dict[str, float]]:
    candidates = ("fused_centroid", "support_centroid", "support_knn3")
    rows = []
    for idx, mode in enumerate(candidates):
        acc, count = _old80_first_support_cv_accuracy(
            class_states,
            mode=mode,
            fusion_rho=float(fusion_rho),
            knn_k=int(knn_k),
        )
        score = -1.0 if math.isnan(acc) else float(acc)
        rows.append((score, -idx, mode, count))
    best_score, _, best_mode, count = max(rows)
    return best_mode, {
        "support_cv_acc": float("nan") if best_score < 0.0 else float(best_score),
        "support_cv_count": float(count),
    }


def apply_old80_first_head(
    features: torch.Tensor,
    result: PredictionResult,
    class_states: Mapping[int, ClassState],
    *,
    mode: str = "disabled",
    apply_policy: str = "replace_all",
    fusion_rho: float = 0.75,
    knn_k: int = 3,
) -> PredictionResult:
    """OLD80_FIRST old-class head using only source old states and target-old support."""

    mode = str(mode).lower()
    if mode in {"", "disabled", "none"}:
        return result
    labels, scores, margins, _, stats = _old80_first_head_scores(
        features,
        class_states,
        mode=mode,
        fusion_rho=float(fusion_rho),
        knn_k=int(knn_k),
    )
    predicted = result.predicted_labels.detach().cpu().clone()
    accepted = result.accepted.detach().cpu().clone().bool()
    candidate = (
        result.candidate_labels.detach().cpu().clone()
        if result.candidate_labels is not None
        else predicted.clone()
    )
    out_scores = result.scores.detach().cpu().clone()
    out_margins = (
        result.margins.detach().cpu().clone()
        if result.margins is not None
        else torch.full_like(out_scores, float("nan"))
    )
    policy = str(apply_policy).lower()
    if policy == "replace_all":
        apply_mask = torch.ones_like(accepted, dtype=torch.bool)
    elif policy == "replace_all_except_seen_new_override":
        apply_mask = torch.ones_like(accepted, dtype=torch.bool)
        seen_new_override = result.diagnostics.get("seen_new_registration_override_mask") if result.diagnostics else None
        if torch.is_tensor(seen_new_override):
            seen_new_override = seen_new_override.detach().cpu().bool().reshape(-1)
            if int(seen_new_override.numel()) >= int(apply_mask.numel()):
                apply_mask &= ~seen_new_override[: int(apply_mask.numel())]
    elif policy == "rescue_rejected":
        apply_mask = (~accepted) | (predicted == UNKNOWN_LABEL)
    elif policy == "replace_unknown":
        apply_mask = predicted == UNKNOWN_LABEL
    else:
        raise ValueError(f"unknown OLD80_FIRST apply_policy: {apply_policy}")
    labels = labels.detach().cpu()
    scores = scores.detach().cpu()
    margins = margins.detach().cpu()
    predicted[apply_mask] = labels[apply_mask]
    candidate[apply_mask] = labels[apply_mask]
    accepted[apply_mask] = True
    out_scores[apply_mask] = scores[apply_mask]
    out_margins[apply_mask] = margins[apply_mask]
    decisions = list(result.decisions or [])
    if len(decisions) != int(accepted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in result.accepted.detach().cpu().tolist()]
    reasons = list(result.gate_reasons or [])
    if len(reasons) != int(accepted.numel()):
        reasons = ["accepted" if bool(v) else "rejected" for v in result.accepted.detach().cpu().tolist()]
    for row in torch.nonzero(apply_mask, as_tuple=False).flatten().tolist():
        decisions[int(row)] = "accept"
        reasons[int(row)] = f"old80_first_{mode}"
    diagnostics = dict(result.diagnostics or {})
    n = int(accepted.numel())
    diagnostics["old80_first_label"] = labels
    diagnostics["old80_first_score"] = scores
    diagnostics["old80_first_margin"] = margins
    diagnostics["old80_first_applied_mask"] = apply_mask.detach().cpu()
    diagnostics["old80_first_support_cv_acc"] = torch.full((n,), float(stats.get("support_cv_acc", float("nan"))))
    diagnostics["old80_first_support_cv_count"] = torch.full((n,), float(stats.get("support_cv_count", 0.0)))
    diagnostics["old80_first_mode_code"] = torch.full((n,), float(stats.get("selected_mode_code", 0.0)))
    return replace(
        result,
        predicted_labels=predicted,
        candidate_labels=candidate,
        scores=out_scores,
        margins=out_margins,
        accepted=accepted,
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def _resolve_gate_config(
    unknown_threshold: float | None,
    gate_config: OpenSetGateConfig | None,
) -> OpenSetGateConfig:
    if gate_config is None:
        mode = "cosine" if unknown_threshold is not None else "none"
        return OpenSetGateConfig(mode=mode, min_cosine=unknown_threshold)
    if unknown_threshold is not None and gate_config.min_cosine is None:
        return replace(gate_config, min_cosine=float(unknown_threshold))
    return gate_config


def _selected_metadata_threshold(prototypes: PrototypeSet, key: str, indices: torch.Tensor) -> torch.Tensor | None:
    value = prototypes.metadata.get(key)
    if not torch.is_tensor(value):
        return None
    return value.to(device=indices.device)[indices]


def _mahalanobis_to_prototypes(x: torch.Tensor, prototypes: PrototypeSet) -> torch.Tensor | None:
    diag_var = prototypes.metadata.get("diag_var")
    if not torch.is_tensor(diag_var):
        return None
    proto = normalize_rows(prototypes.vectors).to(device=x.device, dtype=x.dtype)
    diag_var = diag_var.to(device=x.device, dtype=x.dtype).clamp_min(1e-12)
    centered = x.unsqueeze(1) - proto.unsqueeze(0)
    return (centered.pow(2) / diag_var.unsqueeze(0)).sum(dim=-1).sqrt()


def register_old_classes(
    source_prototypes: PrototypeSet,
    target_old_support: torch.Tensor,
    target_old_labels: torch.Tensor | Iterable[int],
    *,
    stage: str = "Stage2-B",
    orbit_rank: int = 2,
    active_ratio: float = 0.25,
    kappa: float = 3.0,
    old_anchor_override_min_quality: float = 0.55,
    gate_config: OpenSetGateConfig | None = None,
) -> tuple[dict[int, ClassState], torch.Tensor]:
    support = torch.as_tensor(target_old_support).float().to(source_prototypes.vectors.device)
    labels = _labels_tensor(target_old_labels, device=source_prototypes.vectors.device)
    validate_stage2_protocol(
        stage,
        use_target_old_support=bool(support.numel() and labels.numel()),
        use_target_new_support=False,
        use_unknown_query_for_threshold_calibration=False,
    )
    if support.numel() > 0:
        support = normalize_rows(support)
    dim = int(source_prototypes.vectors.shape[1])
    u_orbit = estimate_orbit_subspace(source_prototypes, support, labels, orbit_rank=int(orbit_rank))

    mask_features = [source_prototypes.vectors.detach().float()]
    mask_labels = [source_prototypes.labels.detach().long()]
    if support.numel() > 0 and labels.numel() > 0:
        valid_support_mask = torch.tensor(
            [int(v) in source_prototypes.label_values() for v in labels.detach().cpu().tolist()],
            dtype=torch.bool,
            device=labels.device,
        )
        if bool(valid_support_mask.any().item()):
            mask_features.append(support[valid_support_mask].detach().float())
            mask_labels.append(labels[valid_support_mask].detach().long())
    masks = _class_conditioned_masks(
        torch.cat(mask_features, dim=0),
        torch.cat(mask_labels, dim=0),
        sorted(source_prototypes.label_values()),
        dim=dim,
        active_ratio=float(active_ratio),
    )

    states: dict[int, ClassState] = {}
    gate_config = gate_config or OpenSetGateConfig(mode="combined", min_cosine=0.0)
    for label in sorted(source_prototypes.label_values()):
        src_idx = source_prototypes.index_of(int(label))
        source_vec = source_prototypes.vectors[src_idx]
        support_mask = labels == int(label)
        n = int(support_mask.sum().item()) if labels.numel() else 0
        if n > 0:
            target_members = support[support_mask].detach().clone()
            target_mean = normalize_rows(target_members.mean(dim=0, keepdim=True)).squeeze(0)
            support_quality_stats = _old_support_quality_stats(target_members, source_vec, kappa=float(kappa))
            base_rho = n / (n + max(0.0, float(kappa)))
            rho = base_rho * float(support_quality_stats["support_quality"])
            prototype = normalize_rows((rho * target_mean + (1.0 - rho) * source_vec).view(1, -1)).squeeze(0)
        else:
            target_members = None
            support_quality_stats = _old_support_quality_stats(None, source_vec, kappa=float(kappa))
            base_rho = 0.0
            rho = 0.0
            prototype = source_vec.clone()
        covariance = _metadata_vector(source_prototypes, "diag_var", int(label), torch.full_like(prototype, gate_config.mahalanobis_eps))
        if n > 1:
            target_var = (support[support_mask] - prototype.view(1, -1)).pow(2).mean(dim=0).clamp_min(float(gate_config.mahalanobis_eps))
            covariance = ((1.0 - rho) * covariance + rho * target_var).clamp_min(float(gate_config.mahalanobis_eps))
        openmax_threshold = _metadata_threshold(source_prototypes, "openmax_thresholds", int(label))
        maha_threshold = _metadata_threshold(source_prototypes, "mahalanobis_thresholds", int(label))
        max_residual = float(openmax_threshold if openmax_threshold is not None else 1.0)
        max_maha = float(maha_threshold if maha_threshold is not None else 1.0e6)
        source_weight = float(1.0 - rho)
        states[int(label)] = ClassState(
            class_id=int(label),
            group="old",
            prototype=prototype,
            mask=masks[int(label)].to(device=prototype.device, dtype=prototype.dtype),
            subspace=u_orbit[:, : min(u_orbit.shape[1], max(0, int(orbit_rank)))].to(device=prototype.device, dtype=prototype.dtype),
            covariance_diag=covariance,
            thresholds={
                "max_residual": max_residual,
                "reject_residual": max_residual,
                "max_mahalanobis": max_maha,
                "reject_mahalanobis": max_maha,
                "min_margin": float(gate_config.min_margin or 0.0),
                "source_weight": source_weight,
                "base_rho": float(base_rho),
                "effective_rho": float(rho),
                "support_count": float(support_quality_stats["support_count"]),
                "support_quality": float(support_quality_stats["support_quality"]),
                "support_source_similarity_mean": float(support_quality_stats["support_source_similarity_mean"]),
                "support_source_mean_similarity": float(support_quality_stats["support_source_mean_similarity"]),
                "support_compactness": float(support_quality_stats["support_compactness"]),
                "support_count_term": float(support_quality_stats["support_count_term"]),
                "min_old_anchor_override_quality": float(old_anchor_override_min_quality),
            },
            evt_params={
                "tail_threshold": max_residual,
                "support_quality_gate": {
                    "fit": "target_old_support_source_alignment_compactness_count",
                    "support_quality": float(support_quality_stats["support_quality"]),
                    "support_source_similarity_mean": float(support_quality_stats["support_source_similarity_mean"]),
                    "support_source_mean_similarity": float(support_quality_stats["support_source_mean_similarity"]),
                    "support_compactness": float(support_quality_stats["support_compactness"]),
                    "support_count_term": float(support_quality_stats["support_count_term"]),
                    "base_rho": float(base_rho),
                    "effective_rho": float(rho),
                    "min_old_anchor_override_quality": float(old_anchor_override_min_quality),
                },
            },
            support_quality=float(support_quality_stats["support_quality"]),
            source_weight=source_weight,
            support_anchors=target_members,
        )
    return states, u_orbit


def register_new_classes(
    target_new_support: torch.Tensor,
    target_new_labels: torch.Tensor | Iterable[int],
    old_class_states: Mapping[int, ClassState],
    u_orbit: torch.Tensor,
    source_global_stats: Mapping | None = None,
    config: Mapping | None = None,
    *,
    stage: str = "Stage2-C",
    active_ratio: float = 0.25,
    gate_config: OpenSetGateConfig | None = None,
) -> dict[int, ClassState]:
    del source_global_stats
    config = dict(config or {})
    support = normalize_rows(torch.as_tensor(target_new_support).float())
    labels = _labels_tensor(target_new_labels, device=support.device)
    validate_stage2_protocol(stage, use_target_new_support=bool(support.numel()), use_unknown_query_for_threshold_calibration=False)
    old_labels = {int(v) for v in old_class_states}
    new_labels = sorted({int(v) for v in labels.cpu().tolist() if int(v) != UNKNOWN_LABEL})
    overlap = old_labels & set(new_labels)
    if overlap:
        raise ValueError(f"new classes overlap old class states: {sorted(overlap)}")
    gate_config = gate_config or OpenSetGateConfig(mode="combined", min_cosine=0.0)
    dim = int(support.shape[1])
    old_proto_features = []
    old_proto_labels = []
    for old_label, state in old_class_states.items():
        old_proto_features.append(torch.as_tensor(state.prototype).float().to(support.device))
        old_proto_labels.append(int(old_label))
    mask_features = torch.cat([torch.stack(old_proto_features, dim=0), support], dim=0) if old_proto_features else support
    mask_labels = torch.cat([torch.tensor(old_proto_labels, dtype=torch.long, device=support.device), labels], dim=0) if old_proto_labels else labels
    masks = _class_conditioned_masks(mask_features, mask_labels, new_labels, dim=dim, active_ratio=float(active_ratio))

    states: dict[int, ClassState] = {}
    max_total_rank = int(config.get("max_total_rank", 4))
    for label in new_labels:
        members = support[labels == int(label)]
        prototype = normalize_rows(members.mean(dim=0, keepdim=True)).squeeze(0)
        if members.size(0) > 1:
            covariance = (members - prototype.view(1, -1)).pow(2).mean(dim=0).clamp_min(float(gate_config.mahalanobis_eps))
        else:
            covariance = torch.full_like(prototype, float(gate_config.mahalanobis_eps))
        basis_rows = []
        orbit_basis = torch.as_tensor(u_orbit, dtype=prototype.dtype, device=prototype.device)
        if orbit_basis.numel() > 0 and orbit_basis.shape[1] > 0:
            basis_rows.append(orbit_basis.T)
        if members.size(0) > 1:
            basis_rows.append((members - prototype.view(1, -1)).detach())
        if basis_rows:
            subspace = _orthonormal_columns(torch.cat(basis_rows, dim=0), max_rank=max_total_rank).to(
                device=prototype.device,
                dtype=prototype.dtype,
            )
        else:
            subspace = torch.zeros(dim, 0, dtype=prototype.dtype, device=prototype.device)
        states[int(label)] = ClassState(
            class_id=int(label),
            group="seen_new",
            prototype=prototype,
            mask=masks[int(label)].to(device=prototype.device, dtype=prototype.dtype),
            subspace=subspace,
            covariance_diag=covariance,
            thresholds={
                "max_residual": 1.0,
                "reject_residual": 1.0,
                "max_mahalanobis": 1.0e6,
                "reject_mahalanobis": 1.0e6,
                "min_margin": float(gate_config.min_margin or 0.0),
                "source_weight": 0.0,
            },
            evt_params={"tail_threshold": 1.0},
            support_quality=1.0,
            source_weight=0.0,
            support_anchors=members.detach().clone(),
        )
    return states


def calibrate_thresholds(
    class_states: Mapping[int, ClassState],
    calibration_known: torch.Tensor,
    calibration_labels: torch.Tensor | Iterable[int],
    *,
    surrogate_unknown: torch.Tensor | None = None,
    target_far: float = 0.05,
    mode: str = "balanced",
    evt_mode: str = "openmax_tail",
    unknown_source: str = "surrogate",
    old_retention_quantile: float = 0.95,
    old_surrogate_evidence_margin: float = 0.0,
    old_surrogate_reject_relax: float = 0.0,
    support_retention_guard: bool = False,
    support_retention_guard_quantile: float = 0.05,
    support_retention_guard_slack: float = 0.02,
) -> dict[int, ClassState]:
    if surrogate_unknown is not None and str(unknown_source) != "surrogate":
        raise ValueError("unknown query samples must not be used for threshold calibration")
    if not class_states:
        return {}
    dim = int(next(iter(class_states.values())).prototype.numel())
    head = OrbitAdaptiveMSEHead(dim=dim, class_states=class_states)
    known = normalize_rows(torch.as_tensor(calibration_known).float())
    labels = _labels_tensor(calibration_labels, device=known.device)
    known_scores, per_class = head.score_matrix(known)
    known_energy = head.energy(known_scores)
    updated = {int(k): v for k, v in class_states.items()}
    q_hi = 0.90 if str(mode) == "safe" else 0.95
    q_old = min(1.0, max(0.0, float(old_retention_quantile)))
    for label, state in updated.items():
        mask = labels == int(label)
        if not bool(mask.any().item()):
            continue
        residual = per_class[int(label)]["residual"][mask]
        maha = per_class[int(label)]["maha"][mask]
        cos = per_class[int(label)]["cos"][mask]
        state.thresholds["max_residual"] = float(torch.quantile(residual, q_hi).item())
        state.thresholds["reject_residual"] = state.thresholds["max_residual"]
        state.thresholds["max_mahalanobis"] = float(torch.quantile(maha, q_hi).item())
        state.thresholds["reject_mahalanobis"] = state.thresholds["max_mahalanobis"]
        if str(state.group) == "old" and state.support_anchors is not None and torch.is_tensor(state.support_anchors) and state.support_anchors.numel() > 0:
            q_low_old = max(0.0, min(1.0, 1.0 - q_old))
            anchor_similarity = _leave_one_out_anchor_similarity(known[mask], state.support_anchors)
            old_evidence = _old_support_evidence_score(cos, anchor_similarity, residual, maha)
            state.thresholds["min_old_support_anchor_similarity"] = max(
                -1.0,
                float(torch.quantile(anchor_similarity, q_low_old).item()) - 0.05,
            )
            state.thresholds["min_old_support_evidence"] = float(torch.quantile(old_evidence, q_low_old).item()) - 0.05
        if str(state.group) == "seen_new":
            q_low = 0.10 if str(mode) == "safe" else 0.05
            evidence = _seen_new_evidence_score(cos, residual, maha)
            support_residual = max(float(torch.quantile(residual, q_hi).item()) * 2.0, 0.02)
            state.thresholds["min_seen_new_support_affinity"] = max(
                -1.0,
                float(torch.quantile(cos, q_low).item()) - 0.05,
            )
            state.thresholds["max_seen_new_support_residual"] = support_residual
            state.thresholds["min_seen_new_evidence"] = float(torch.quantile(evidence, q_low).item()) - max(
                0.05,
                0.5 * support_residual,
            )
            if state.support_anchors is not None and torch.is_tensor(state.support_anchors) and state.support_anchors.numel() > 0:
                anchor_similarity = _leave_one_out_anchor_similarity(known[mask], state.support_anchors)
                state.thresholds["min_seen_new_anchor_similarity"] = max(
                    -1.0,
                    float(torch.quantile(anchor_similarity, q_low).item()) - 0.05,
                )
        if str(evt_mode).lower() == "weibull":
            weibull = fit_weibull_tail(
                residual.detach(),
                tail_size=min(20, max(1, int(residual.numel()))),
                target_far=float(target_far),
            )
            state.evt_params.update(
                {
                    "fit": str(weibull["fit"]),
                    "weibull_shape": float(weibull["shape"]),
                    "weibull_scale": float(weibull["scale"]),
                    "tail_threshold": float(weibull["threshold"]),
                    "target_far": float(target_far),
                }
            )
            state.thresholds["max_residual"] = float(weibull["threshold"])
            state.thresholds["reject_residual"] = float(weibull["threshold"])
        else:
            state.evt_params["tail_threshold"] = state.thresholds["max_residual"]
    if surrogate_unknown is not None:
        unknown = normalize_rows(torch.as_tensor(surrogate_unknown).float())
        if unknown.numel() > 0:
            unknown_scores, unknown_per_class = head.score_matrix(unknown)
            unknown_energy = head.energy(unknown_scores)
            far_q = max(0.0, min(1.0, float(target_far)))
            surrogate_evidence_q = 1.0 - far_q
            known_accept_cap = float(torch.quantile(known_energy, q_old).item())
            unknown_accept_cap = float(torch.quantile(unknown_energy, far_q).item())
            energy_threshold = max(known_accept_cap, unknown_accept_cap)
            for label, state in updated.items():
                state.thresholds["max_energy"] = float(energy_threshold)
                state.thresholds["reject_energy"] = float(energy_threshold)
                state.thresholds["surrogate_reject_energy"] = float(unknown_accept_cap)
                if (
                    str(state.group) == "old"
                    and state.support_anchors is not None
                    and torch.is_tensor(state.support_anchors)
                    and state.support_anchors.numel() > 0
                ):
                    unknown_cos = unknown_per_class[int(label)]["cos"].detach()
                    unknown_residual = unknown_per_class[int(label)]["residual"].detach()
                    unknown_maha = unknown_per_class[int(label)]["maha"].detach()
                    unknown_anchor = _support_anchor_similarity(unknown, state).detach()
                    unknown_old_evidence = _old_support_evidence_score(
                        unknown_cos,
                        unknown_anchor,
                        unknown_residual,
                        unknown_maha,
                    )
                    state.thresholds["min_old_surrogate_evidence"] = float(
                        torch.quantile(unknown_old_evidence, surrogate_evidence_q).item()
                    ) + float(old_surrogate_evidence_margin)
                    reject_relax = max(0.0, float(old_surrogate_reject_relax))
                    state.thresholds["min_old_surrogate_reject_evidence"] = float(
                        state.thresholds["min_old_surrogate_evidence"]
                    ) - reject_relax
                    support_guard_threshold = None
                    if bool(support_retention_guard) and bool((labels == int(label)).any().item()):
                        support_mask = labels == int(label)
                        support_cos = per_class[int(label)]["cos"][support_mask].detach()
                        support_residual = per_class[int(label)]["residual"][support_mask].detach()
                        support_maha = per_class[int(label)]["maha"][support_mask].detach()
                        support_anchor = _leave_one_out_anchor_similarity(
                            known[support_mask],
                            state.support_anchors,
                        ).detach()
                        support_old_evidence = _old_support_evidence_score(
                            support_cos,
                            support_anchor,
                            support_residual,
                            support_maha,
                        )
                        guard_q = max(0.0, min(1.0, float(support_retention_guard_quantile)))
                        support_guard_threshold = float(torch.quantile(support_old_evidence, guard_q).item()) - max(
                            0.0,
                            float(support_retention_guard_slack),
                        )
                        state.thresholds["min_old_surrogate_reject_evidence"] = min(
                            float(state.thresholds["min_old_surrogate_reject_evidence"]),
                            support_guard_threshold,
                        )
                    state.evt_params["old_surrogate_evidence_gate"] = {
                        "fit": "support_derived_old_vs_surrogate_unknown_evidence",
                        "surrogate_unknown_quantile": float(surrogate_evidence_q),
                        "surrogate_unknown_evidence_threshold": float(state.thresholds["min_old_surrogate_evidence"]),
                        "surrogate_unknown_reject_threshold": float(state.thresholds["min_old_surrogate_reject_evidence"]),
                        "margin": float(old_surrogate_evidence_margin),
                        "reject_relax": float(reject_relax),
                        "support_retention_guard_enabled": bool(support_retention_guard),
                        "support_retention_guard_quantile": float(support_retention_guard_quantile),
                        "support_retention_guard_slack": float(support_retention_guard_slack),
                        "support_retention_guard_threshold": support_guard_threshold,
                        "unknown_source": str(unknown_source),
                        "action": "reject_below_relaxed_surrogate_threshold_uncertain_between_reject_and_surrogate_threshold",
                    }
                state.evt_params["energy_gate"] = {
                    "fit": "old_retention_constrained_known_quantile_and_surrogate_unknown_low_quantile",
                    "known_quantile": float(q_old),
                    "surrogate_unknown_quantile": float(far_q),
                    "known_accept_cap": float(known_accept_cap),
                    "surrogate_unknown_accept_cap": float(unknown_accept_cap),
                    "threshold": float(energy_threshold),
                    "surrogate_reject_energy": float(unknown_accept_cap),
                    "target_far": float(target_far),
                    "unknown_source": str(unknown_source),
                    "old_retention_constrained": True,
                }
    return updated


def calibrate_anchor_density_gates(
    class_states: Mapping[int, ClassState],
    calibration_known: torch.Tensor,
    calibration_labels: torch.Tensor | Iterable[int],
    *,
    enabled: bool = False,
    topk: int = 3,
    temperature: float = 0.08,
    min_quantile: float = 0.05,
    margin_quantile: float = 0.05,
    action: str = "uncertain",
) -> dict:
    """Fit a support-anchor one-class density gate without using unknown query labels."""

    telemetry = {
        "enabled": bool(enabled),
        "source": "target_support_or_source_anchor_known_calibration_only",
        "unknown_query_threshold_calibration": False,
        "class_count": 0,
        "topk": int(topk),
        "temperature": float(temperature),
        "min_quantile": float(min_quantile),
        "margin_quantile": float(margin_quantile),
        "action": str(action),
    }
    if not enabled or not class_states:
        return telemetry
    known = normalize_rows(torch.as_tensor(calibration_known).float())
    labels = _labels_tensor(calibration_labels, device=known.device)
    if known.numel() == 0 or labels.numel() == 0:
        telemetry["reason"] = "empty_known_calibration"
        return telemetry
    for state in class_states.values():
        state.thresholds["anchor_density_topk"] = int(topk)
        state.thresholds["anchor_density_temperature"] = float(temperature)
    head = OrbitAdaptiveMSEHead(dim=int(next(iter(class_states.values())).prototype.numel()), class_states=class_states)
    _, per_class = head.score_matrix(known)
    density_cols = []
    for label in head.class_order:
        row = per_class[int(label)]
        if "anchor_density" in row:
            density_cols.append(row["anchor_density"].detach())
        else:
            density_cols.append(torch.full((known.size(0),), -float("inf"), dtype=known.dtype, device=known.device))
    density_matrix = torch.stack(density_cols, dim=1)
    q_density = max(0.0, min(1.0, float(min_quantile)))
    q_margin = max(0.0, min(1.0, float(margin_quantile)))
    updated_classes = 0
    for class_col, label in enumerate(head.class_order):
        state = class_states[int(label)]
        mask = labels == int(label)
        if not bool(mask.any().item()):
            continue
        own_density = density_matrix[mask, class_col]
        finite_own = own_density[torch.isfinite(own_density)]
        if finite_own.numel() == 0:
            continue
        if density_matrix.size(1) > 1:
            competitor = density_matrix[mask].clone()
            competitor[:, class_col] = -float("inf")
            best_competitor = competitor.max(dim=1).values
            margin = own_density - best_competitor
            finite_margin = margin[torch.isfinite(margin)]
        else:
            finite_margin = torch.ones_like(finite_own)
        state.thresholds["anchor_density_gate_enabled"] = True
        state.thresholds["anchor_density_topk"] = int(topk)
        state.thresholds["anchor_density_temperature"] = float(temperature)
        state.thresholds["min_anchor_density"] = float(torch.quantile(finite_own, q_density).item()) - 0.02
        state.thresholds["min_anchor_density_margin"] = (
            float(torch.quantile(finite_margin, q_margin).item()) - 0.01
            if finite_margin.numel() > 0
            else -float("inf")
        )
        state.thresholds["anchor_density_gate_action"] = str(action)
        updated_classes += 1
    telemetry["class_count"] = int(updated_classes)
    return telemetry


def calibrate_class_envelope_gates(
    class_states: Mapping[int, ClassState],
    calibration_known: torch.Tensor,
    calibration_labels: torch.Tensor | Iterable[int],
    *,
    enabled: bool = False,
    evidence_quantile: float = 0.05,
    residual_quantile: float = 0.95,
    score_quantile: float = 0.05,
    margin_quantile: float = 0.05,
    evidence_slack: float = 0.02,
    residual_slack: float = 0.02,
    score_slack: float = 0.05,
    margin_slack: float = 0.02,
    min_failures: int = 1,
    action: str = "reject",
) -> dict:
    """Fit query-free per-class source/support acceptance envelopes.

    The envelope uses only known calibration rows from source old classes and
    allowed target support. Unknown query labels are never used for threshold
    fitting.
    """

    telemetry = {
        "enabled": bool(enabled),
        "source": "source_old_plus_allowed_target_support_known_calibration_only",
        "unknown_query_threshold_calibration": False,
        "class_count": 0,
        "evidence_quantile": float(evidence_quantile),
        "residual_quantile": float(residual_quantile),
        "score_quantile": float(score_quantile),
        "margin_quantile": float(margin_quantile),
        "evidence_slack": float(evidence_slack),
        "residual_slack": float(residual_slack),
        "score_slack": float(score_slack),
        "margin_slack": float(margin_slack),
        "min_failures": int(min_failures),
        "action": str(action),
    }
    if not enabled or not class_states:
        return telemetry
    known = normalize_rows(torch.as_tensor(calibration_known).float())
    labels = _labels_tensor(calibration_labels, device=known.device)
    if known.numel() == 0 or labels.numel() == 0:
        telemetry["reason"] = "empty_known_calibration"
        return telemetry
    head = OrbitAdaptiveMSEHead(dim=int(next(iter(class_states.values())).prototype.numel()), class_states=class_states)
    score_matrix, per_class = head.score_matrix(known)
    q_evidence = max(0.0, min(1.0, float(evidence_quantile)))
    q_residual = max(0.0, min(1.0, float(residual_quantile)))
    q_score = max(0.0, min(1.0, float(score_quantile)))
    q_margin = max(0.0, min(1.0, float(margin_quantile)))
    updated_classes = 0
    for class_col, label in enumerate(head.class_order):
        state = class_states[int(label)]
        mask = labels == int(label)
        if not bool(mask.any().item()):
            continue
        row = per_class[int(label)]
        residual = row["residual"].detach()[mask]
        score = score_matrix[:, class_col].detach()[mask]
        if str(state.group) == "old":
            anchor = _support_anchor_similarity(known[mask], state).detach()
            evidence = _old_support_evidence_score(row["cos"].detach()[mask], anchor, residual, row["maha"].detach()[mask])
        else:
            evidence = _seen_new_evidence_score(row["cos"].detach()[mask], residual, row["maha"].detach()[mask])
        if score_matrix.size(1) > 1:
            competitor = score_matrix[mask].detach().clone()
            competitor[:, class_col] = -float("inf")
            margin = score - competitor.max(dim=1).values
            finite_margin = margin[torch.isfinite(margin)]
        else:
            finite_margin = torch.ones_like(score)
        finite_evidence = evidence[torch.isfinite(evidence)]
        finite_residual = residual[torch.isfinite(residual)]
        finite_score = score[torch.isfinite(score)]
        if finite_evidence.numel() == 0 or finite_residual.numel() == 0 or finite_score.numel() == 0:
            continue
        state.thresholds["class_envelope_gate_enabled"] = True
        state.thresholds["min_class_envelope_evidence"] = float(torch.quantile(finite_evidence, q_evidence).item()) - float(evidence_slack)
        state.thresholds["max_class_envelope_residual"] = float(torch.quantile(finite_residual, q_residual).item()) + float(residual_slack)
        state.thresholds["min_class_envelope_score"] = float(torch.quantile(finite_score, q_score).item()) - float(score_slack)
        state.thresholds["min_class_envelope_margin"] = (
            float(torch.quantile(finite_margin, q_margin).item()) - float(margin_slack)
            if finite_margin.numel() > 0
            else -float("inf")
        )
        state.thresholds["class_envelope_min_failures"] = max(1, int(min_failures))
        state.thresholds["class_envelope_action"] = str(action)
        updated_classes += 1
    telemetry["class_count"] = int(updated_classes)
    return telemetry


def generate_pseudo_unknown_features(
    class_states: Mapping[int, ClassState],
    *,
    samples_per_pair: int = 2,
    offset_scale: float = 0.15,
    target_shift_samples_per_class: int = 0,
    target_shift_offset_scale: float = 0.20,
    target_halo_samples_per_class: int = 0,
    target_halo_offset_scale: float = 0.35,
    target_ring_samples_per_class: int = 0,
    target_ring_offset_scale: float = 0.45,
) -> torch.Tensor:
    """Generate query-free pseudo-unknown features from registered class geometry."""

    labels = sorted(int(k) for k in class_states)
    if len(labels) < 2:
        raise ValueError("pseudo-unknown generation requires at least two registered classes")
    samples = []
    for left_i, left in enumerate(labels):
        for right in labels[left_i + 1:]:
            p_left = torch.as_tensor(class_states[left].prototype).float()
            p_right = torch.as_tensor(class_states[right].prototype).float().to(p_left.device)
            midpoint = normalize_rows((p_left + p_right).view(1, -1)).squeeze(0)
            direction = normalize_rows((p_left - p_right).view(1, -1)).squeeze(0)
            for i in range(max(1, int(samples_per_pair))):
                branch = i % 4
                if branch == 0:
                    sample = midpoint + float(offset_scale) * direction
                elif branch == 1:
                    sample = midpoint - float(offset_scale) * direction
                elif branch == 2:
                    sample = p_left + float(offset_scale) * direction
                else:
                    sample = p_right - float(offset_scale) * direction
                samples.append(normalize_rows(sample.view(1, -1)).squeeze(0))
    geometry = torch.stack(samples, dim=0)
    target_shift = _target_shift_pseudo_unknown_features_from_states(
        class_states,
        samples_per_class=int(target_shift_samples_per_class),
        offset_scale=float(target_shift_offset_scale),
    ).to(geometry.device)
    target_halo = _target_halo_pseudo_unknown_features_from_states(
        class_states,
        samples_per_class=int(target_halo_samples_per_class),
        offset_scale=float(target_halo_offset_scale),
    ).to(geometry.device)
    target_ring = _target_support_ring_pseudo_unknown_features_from_states(
        class_states,
        samples_per_class=int(target_ring_samples_per_class),
        offset_scale=float(target_ring_offset_scale),
    ).to(geometry.device)
    parts = [part for part in (geometry, target_shift, target_halo, target_ring) if part.numel() > 0]
    return torch.cat(parts, dim=0)


def fit_siamese_verifier(
    support_features: torch.Tensor,
    support_labels: torch.Tensor | Iterable[int],
    *,
    quantile: float = 0.10,
    scale: float = 20.0,
) -> SiameseAnchorVerifier:
    """Build an ambiguous-only anchor verifier from few-shot labeled support."""

    support = normalize_rows(torch.as_tensor(support_features).float())
    labels = _labels_tensor(support_labels, device=support.device)
    if support.size(0) != labels.numel():
        raise ValueError("support features and labels must have equal lengths")
    if bool((labels == UNKNOWN_LABEL).any().item()):
        raise ValueError("unknown query samples must not be used for Siamese verifier fitting")
    same_sims = []
    for label in sorted({int(v) for v in labels.detach().cpu().tolist()}):
        idx = torch.nonzero(labels == int(label), as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        feats = support[idx]
        sim = feats @ feats.T
        tri = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
        same_sims.append(sim[tri[0], tri[1]])
    if same_sims:
        same = torch.cat(same_sims, dim=0)
        threshold = float(torch.quantile(same, min(1.0, max(0.0, float(quantile)))).item())
    else:
        threshold = 0.80
    return SiameseAnchorVerifier(
        anchor_features=support.detach().cpu(),
        anchor_labels=labels.detach().cpu(),
        threshold=float(threshold),
        scale=float(scale),
    )


def apply_siamese_verifier_to_ambiguous(
    features: torch.Tensor,
    result: PredictionResult,
    class_states: Mapping[int, ClassState],
    verifier: SiameseAnchorVerifier,
    *,
    threshold: float = 0.50,
    unknown_risk_veto: bool = False,
    unknown_risk_veto_mode: str = "any",
    min_old_support_evidence_delta: float | None = None,
    min_old_surrogate_reject_delta: float | None = None,
    min_energy_delta: float | None = None,
    min_mahalanobis_delta: float | None = None,
    min_accept_delta: float | None = None,
    min_old_support_anchor_margin: float | None = None,
    min_veto_failures: int = 1,
) -> PredictionResult:
    """Verify only uncertain OA-MSE rows, preserving accept/reject/defer rows."""

    x = normalize_rows(torch.as_tensor(features).float())
    if x.size(0) != result.predicted_labels.numel():
        raise ValueError("features and prediction result must have equal lengths")
    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    diagnostics = dict(result.diagnostics or {})
    labels = sorted(int(k) for k in class_states)
    prototypes = normalize_rows(torch.stack([torch.as_tensor(class_states[k].prototype).float() for k in labels], dim=0))
    pair_label = torch.full_like(predicted, UNKNOWN_LABEL)
    pair_prob = torch.full(predicted.shape, float("nan"), dtype=torch.float32)
    pair_threshold = torch.full(predicted.shape, float(threshold), dtype=torch.float32)
    pair_called = torch.zeros(predicted.shape, dtype=torch.bool)
    pair_veto = torch.zeros(predicted.shape, dtype=torch.bool)

    def diagnostic_value(name: str, row: int) -> float | None:
        value = (result.diagnostics or {}).get(name)
        if value is None or not torch.is_tensor(value) or int(value.numel()) <= row:
            return None
        return float(value.reshape(-1)[row].item())

    veto_rules = (
        ("old_support_evidence_delta", min_old_support_evidence_delta),
        ("old_surrogate_reject_evidence_delta", min_old_surrogate_reject_delta),
        ("energy_delta", min_energy_delta),
        ("mahalanobis_delta", min_mahalanobis_delta),
        ("min_accept_delta", min_accept_delta),
    )
    veto_mode = str(unknown_risk_veto_mode or "any").lower()
    for row, decision in enumerate(decisions):
        if str(decision) != "uncertain":
            continue
        sims = x[row].view(1, -1) @ prototypes.T
        label = int(labels[int(torch.argmax(sims, dim=1).item())])
        prob = verifier.same_probability(x[row].view(1, -1), torch.tensor([label]))[0]
        pair_label[row] = int(label)
        pair_prob[row] = float(prob.item())
        pair_called[row] = True
        if float(prob.item()) >= float(threshold):
            veto = False
            if bool(unknown_risk_veto):
                failed_rules = 0
                for name, minimum in veto_rules:
                    if minimum is None:
                        continue
                    value = diagnostic_value(name, row)
                    if value is not None and value < float(minimum):
                        failed_rules += 1
                        if veto_mode == "any":
                            veto = True
                            break
                if veto_mode == "coupled":
                    anchor_margin_failed = False
                    if min_old_support_anchor_margin is not None:
                        anchor_margin = diagnostic_value("old_support_anchor_margin", row)
                        anchor_margin_failed = (
                            anchor_margin is not None and anchor_margin < float(min_old_support_anchor_margin)
                        )
                    veto = failed_rules >= max(1, int(min_veto_failures)) and anchor_margin_failed
            if veto:
                pair_veto[row] = True
                predicted[row] = UNKNOWN_LABEL
                accepted[row] = False
                decisions[row] = "reject"
                reasons[row] = "siamese_unknown_risk_coupled_reject" if veto_mode == "coupled" else "siamese_unknown_risk_reject"
                continue
            predicted[row] = label
            accepted[row] = True
            decisions[row] = "accept"
            reasons[row] = "siamese_verified"
        else:
            predicted[row] = UNKNOWN_LABEL
            accepted[row] = False
            decisions[row] = "reject"
            reasons[row] = "siamese_reject"
    diagnostics["pair_verifier_label"] = pair_label.detach().cpu()
    diagnostics["pair_verifier_prob"] = pair_prob.detach().cpu()
    diagnostics["pair_verifier_threshold"] = pair_threshold.detach().cpu()
    diagnostics["pair_verifier_called_mask"] = pair_called.detach().cpu()
    diagnostics["pair_verifier_veto_mask"] = pair_veto.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        diagnostics=diagnostics,
        decisions=decisions,
        gate_reasons=reasons,
    )


def apply_old_unknown_acceptance_guard(
    result: PredictionResult,
    class_states: Mapping[int, ClassState],
    *,
    enabled: bool = False,
    min_old_support_evidence_delta: float | None = None,
    min_old_surrogate_reject_delta: float | None = None,
    min_energy_delta: float | None = None,
    min_mahalanobis_delta: float | None = None,
    min_accept_delta: float | None = None,
    min_old_support_anchor_margin: float | None = None,
    min_best_old_score: float | None = None,
    min_margin: float | None = None,
    min_guard_failures: int = 1,
) -> PredictionResult:
    """Reject accepted old-like rows whose old evidence remains weak."""

    if not bool(enabled):
        return result
    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())
    reject_mask = torch.zeros_like(accepted, dtype=torch.bool)

    def diagnostic_value(name: str, row: int) -> float | None:
        value = (result.diagnostics or {}).get(name)
        if value is None or not torch.is_tensor(value) or int(value.numel()) <= row:
            return None
        return float(value.reshape(-1)[row].item())

    guard_rules = (
        ("old_support_evidence_delta", min_old_support_evidence_delta),
        ("old_surrogate_reject_evidence_delta", min_old_surrogate_reject_delta),
        ("energy_delta", min_energy_delta),
        ("mahalanobis_delta", min_mahalanobis_delta),
        ("min_accept_delta", min_accept_delta),
        ("old_support_anchor_margin", min_old_support_anchor_margin),
        ("best_old_score", min_best_old_score),
    )
    threshold = max(1, int(min_guard_failures))
    for row, label in enumerate(predicted.tolist()):
        state = class_states.get(int(label))
        if state is None or str(state.group) != "old" or not bool(accepted[row].item()):
            continue
        failed_rules = 0
        for name, minimum in guard_rules:
            if minimum is None:
                continue
            value = diagnostic_value(name, row)
            if value is not None and value < float(minimum):
                failed_rules += 1
        if min_margin is not None:
            margin = None
            if result.margins is not None and torch.is_tensor(result.margins) and int(result.margins.numel()) > row:
                margin = float(result.margins.reshape(-1)[row].item())
            if margin is not None and margin < float(min_margin):
                failed_rules += 1
        if failed_rules >= threshold:
            predicted[row] = UNKNOWN_LABEL
            accepted[row] = False
            decisions[row] = "reject"
            reasons[row] = "old_unknown_acceptance_guard_reject"
            reject_mask[row] = True

    diagnostics = dict(result.diagnostics or {})
    diagnostics["old_unknown_acceptance_guard_reject_mask"] = reject_mask.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_old_primary_acceptance_gate(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    min_old_support_evidence_delta: float = 0.0,
    min_old_support_anchor_delta: float = -0.02,
    min_old_support_anchor_margin: float = 0.0,
    min_score_margin: float = 0.0,
    require_soft_mixture: bool = False,
    min_soft_mixture_margin: float = -float("inf"),
    min_soft_mixture_cos: float = -float("inf"),
    max_soft_mixture_residual: float = float("inf"),
    require_support_knn: bool = True,
    require_support_knn_label_match: bool = True,
    min_support_knn_margin: float = 0.0,
    max_support_knn_seen_new_minus_old: float | None = None,
    min_old_drift_cos: float = -float("inf"),
    max_old_drift_dist: float = float("inf"),
    require_class_envelope: bool = False,
    unknown_veto_background_score: float = 0.86,
    unknown_veto_background_margin: float = 0.10,
    unknown_veto_min_sources: int = 1,
    fail_action: str = "defer",
    unknown_veto_action: str = "reject",
    promote_rescue_candidates: bool = False,
) -> PredictionResult:
    """Make final old accepts depend on high-consistency old evidence.

    This is intentionally terminal: it runs after the rescue/arbitration
    stages. By default it only preserves or vetoes old accepts. When
    promote_rescue_candidates is enabled, retention-rescue eligibility can
    become an old accept only if all old-primary consistency checks pass and no
    unknown-risk veto is active.
    """

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result
    if x.size(0) != result.predicted_labels.numel():
        raise ValueError("features and prediction result must have equal lengths")

    score_matrix, per_class = head.score_matrix(x)
    label_to_col = {int(label): idx for idx, label in enumerate(head.class_order)}
    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    n = int(predicted.numel())

    def diagnostic_tensor(name: str, *, fill: float = float("nan")) -> torch.Tensor:
        value = (result.diagnostics or {}).get(name)
        if value is None or not torch.is_tensor(value):
            return torch.full((n,), float(fill), dtype=torch.float32)
        flat = value.detach().cpu().float().reshape(-1)
        if int(flat.numel()) >= n:
            return flat[:n]
        padded = torch.full((n,), float(fill), dtype=torch.float32)
        padded[: int(flat.numel())] = flat
        return padded

    def diagnostic_bool(name: str) -> torch.Tensor:
        value = (result.diagnostics or {}).get(name)
        if value is None or not torch.is_tensor(value):
            return torch.zeros((n,), dtype=torch.bool)
        flat = value.detach().cpu().bool().reshape(-1)
        if int(flat.numel()) >= n:
            return flat[:n]
        padded = torch.zeros((n,), dtype=torch.bool)
        padded[: int(flat.numel())] = flat
        return padded

    old_evidence_delta = diagnostic_tensor("old_support_evidence_delta")
    old_anchor_delta = diagnostic_tensor("old_support_anchor_delta")
    old_anchor_margin_diag = diagnostic_tensor("old_support_anchor_margin")
    old_drift_cos = diagnostic_tensor("old_drift_cos")
    old_drift_dist = diagnostic_tensor("old_drift_dist")
    support_knn_label = diagnostic_tensor("support_knn_label", fill=float(UNKNOWN_LABEL))
    support_knn_margin = diagnostic_tensor("support_knn_margin")
    support_knn_seen_new_minus_old = diagnostic_tensor("support_knn_seen_new_minus_old")
    soft_mixture_cos = diagnostic_tensor("soft_mixture_cos")
    soft_mixture_residual = diagnostic_tensor("soft_mixture_residual")
    soft_mixture_margin = diagnostic_tensor("soft_mixture_score_margin")
    soft_mixture_base_pass = diagnostic_bool("soft_mixture_consistency_pass_mask")
    if "soft_mixture_consistency_pass_mask" not in (result.diagnostics or {}):
        soft_mixture_base_pass = torch.ones((n,), dtype=torch.bool)
    envelope_label = diagnostic_tensor("class_envelope_label", fill=float(UNKNOWN_LABEL))
    envelope_failure_count = diagnostic_tensor("class_envelope_failure_count")
    envelope_reject = diagnostic_bool("class_envelope_reject_mask")
    rescue_eligible = diagnostic_bool("retention_rescue_eligible_mask")

    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    background_available = bool(
        background.numel() > 0 and background.ndim == 2 and int(background.shape[1]) == int(x.shape[1])
    )
    if background_available:
        background = background.to(x.device)
        background_score = torch.max(x @ background.T, dim=1).values.detach().cpu()
    else:
        background_score = torch.full((n,), -float("inf"), dtype=torch.float32)
    known_rows = [
        torch.as_tensor(state.prototype).float()
        for state in head.class_states.values()
        if state is not None and torch.as_tensor(state.prototype).numel() == x.shape[1]
    ]
    if known_rows:
        known = normalize_rows(torch.stack(known_rows, dim=0)).to(x.device)
        known_score = torch.max(x @ known.T, dim=1).values.detach().cpu()
    else:
        known_score = torch.zeros((n,), dtype=torch.float32)
    background_margin = background_score - known_score

    prior_veto_sources = [
        "pair_verifier_veto_mask",
        "old_unknown_acceptance_guard_reject_mask",
        "source_looo_reject_mask",
        "density_shell_reject_mask",
        "identity_consensus_reject_mask",
        "three_way_reject_mask",
        "support_conformal_reject_mask",
        "support_reconstruction_reject_mask",
        "pre_reject_arbitration_reject_mask",
        "pre_reject_arbitration_background_reject_risk_mask",
        "pre_reject_arbitration_background_defer_risk_mask",
        "pre_reject_arbitration_extreme_background_mask",
        "two_branch_background_reject_mask",
        "void_background_reject_mask",
    ]
    prior_veto_count = torch.zeros((n,), dtype=torch.float32)
    for name in prior_veto_sources:
        prior_veto_count += diagnostic_bool(name).float()
    direct_background_veto = (background_score >= float(unknown_veto_background_score)) & (
        background_margin >= float(unknown_veto_background_margin)
    )
    unknown_veto = (prior_veto_count + direct_background_veto.float()) >= float(max(1, int(unknown_veto_min_sources)))

    old_primary_label = torch.full_like(predicted, UNKNOWN_LABEL)
    score_margin_values = torch.full((n,), float("nan"), dtype=torch.float32)
    old_candidate_mask = torch.zeros_like(accepted, dtype=torch.bool)
    evidence_pass = torch.zeros_like(accepted, dtype=torch.bool)
    anchor_delta_pass = torch.zeros_like(accepted, dtype=torch.bool)
    anchor_margin_pass = torch.zeros_like(accepted, dtype=torch.bool)
    score_margin_pass = torch.zeros_like(accepted, dtype=torch.bool)
    soft_mixture_pass = torch.zeros_like(accepted, dtype=torch.bool)
    support_knn_pass = torch.zeros_like(accepted, dtype=torch.bool)
    drift_pass = torch.zeros_like(accepted, dtype=torch.bool)
    class_envelope_pass = torch.zeros_like(accepted, dtype=torch.bool)
    consistency_pass = torch.zeros_like(accepted, dtype=torch.bool)
    blocked_accept = torch.zeros_like(accepted, dtype=torch.bool)
    unknown_veto_applied = torch.zeros_like(accepted, dtype=torch.bool)
    rescue_promoted = torch.zeros_like(accepted, dtype=torch.bool)
    rescue_blocked = torch.zeros_like(accepted, dtype=torch.bool)
    fail_action_norm = str(fail_action).lower().strip()
    if fail_action_norm not in {"reject", "defer", "uncertain"}:
        fail_action_norm = "defer"
    veto_action_norm = str(unknown_veto_action).lower().strip()
    if veto_action_norm not in {"reject", "defer", "uncertain"}:
        veto_action_norm = "reject"

    for row_idx in range(n):
        pred_label = int(predicted[row_idx].item())
        cand_label = int(candidate_labels[row_idx].item())
        pred_state = head.class_states.get(pred_label)
        cand_state = head.class_states.get(cand_label)
        if pred_state is not None and str(pred_state.group) == "old":
            label = pred_label
            state = pred_state
        elif cand_state is not None and str(cand_state.group) == "old":
            label = cand_label
            state = cand_state
        else:
            continue
        col = label_to_col.get(int(label))
        if col is None:
            continue
        old_candidate_mask[row_idx] = True
        old_primary_label[row_idx] = int(label)
        row_scores = score_matrix[row_idx]
        if row_scores.numel() > 1:
            competitor = row_scores.detach().clone()
            competitor[col] = -float("inf")
            margin = row_scores[col].detach() - competitor.max()
        else:
            margin = torch.tensor(1.0, dtype=torch.float32)
        score_margin_values[row_idx] = float(margin.detach().cpu().item())

        evidence_ok = float(old_evidence_delta[row_idx].item()) >= float(min_old_support_evidence_delta)
        anchor_delta_ok = float(old_anchor_delta[row_idx].item()) >= float(min_old_support_anchor_delta)
        anchor_margin_ok = float(old_anchor_margin_diag[row_idx].item()) >= float(min_old_support_anchor_margin)
        score_margin_ok = float(score_margin_values[row_idx].item()) >= float(min_score_margin)
        evidence_pass[row_idx] = evidence_ok
        anchor_delta_pass[row_idx] = anchor_delta_ok
        anchor_margin_pass[row_idx] = anchor_margin_ok
        score_margin_pass[row_idx] = score_margin_ok

        soft_has_values = (
            math.isfinite(float(soft_mixture_margin[row_idx].item()))
            or math.isfinite(float(soft_mixture_cos[row_idx].item()))
            or math.isfinite(float(soft_mixture_residual[row_idx].item()))
        )
        soft_ok = bool(soft_mixture_base_pass[row_idx].item())
        soft_ok = soft_ok and float(soft_mixture_margin[row_idx].item()) >= float(min_soft_mixture_margin)
        soft_ok = soft_ok and float(soft_mixture_cos[row_idx].item()) >= float(min_soft_mixture_cos)
        soft_ok = soft_ok and float(soft_mixture_residual[row_idx].item()) <= float(max_soft_mixture_residual)
        if bool(require_soft_mixture) and not bool(soft_has_values):
            soft_ok = False
        elif not bool(require_soft_mixture) and not bool(soft_has_values):
            soft_ok = True
        soft_mixture_pass[row_idx] = bool(soft_ok)

        if bool(require_support_knn):
            knn_checks = []
            if bool(require_support_knn_label_match):
                knn_checks.append(int(support_knn_label[row_idx].item()) == int(label))
            knn_checks.append(float(support_knn_margin[row_idx].item()) >= float(min_support_knn_margin))
            if max_support_knn_seen_new_minus_old is not None:
                knn_checks.append(
                    math.isfinite(float(support_knn_seen_new_minus_old[row_idx].item()))
                    and float(support_knn_seen_new_minus_old[row_idx].item())
                    <= float(max_support_knn_seen_new_minus_old)
                )
            knn_ok = bool(knn_checks) and all(knn_checks)
        else:
            knn_ok = True
        support_knn_pass[row_idx] = bool(knn_ok)

        drift_ok = (
            float(old_drift_cos[row_idx].item()) >= float(min_old_drift_cos)
            and float(old_drift_dist[row_idx].item()) <= float(max_old_drift_dist)
        )
        drift_pass[row_idx] = bool(drift_ok)

        if bool(require_class_envelope):
            envelope_enabled = bool(state.thresholds.get("class_envelope_gate_enabled", False))
            min_failures = max(1, int(state.thresholds.get("class_envelope_min_failures", 1)))
            envelope_ok = (
                envelope_enabled
                and int(envelope_label[row_idx].item()) == int(label)
                and float(envelope_failure_count[row_idx].item()) < float(min_failures)
                and not bool(envelope_reject[row_idx].item())
            )
        else:
            envelope_ok = True
        class_envelope_pass[row_idx] = bool(envelope_ok)

        row_pass = all(
            bool(mask[row_idx].item())
            for mask in (
                evidence_pass,
                anchor_delta_pass,
                anchor_margin_pass,
                score_margin_pass,
                soft_mixture_pass,
                support_knn_pass,
                drift_pass,
                class_envelope_pass,
            )
        )
        consistency_pass[row_idx] = bool(row_pass)

        is_accepted_old = bool(accepted[row_idx].item()) and pred_state is not None and str(pred_state.group) == "old"
        is_rescue_candidate = (
            bool(promote_rescue_candidates)
            and bool(rescue_eligible[row_idx].item())
            and not bool(accepted[row_idx].item())
            and cand_state is not None
            and str(cand_state.group) == "old"
        )
        if is_accepted_old or is_rescue_candidate:
            if bool(unknown_veto[row_idx].item()):
                predicted[row_idx] = UNKNOWN_LABEL
                accepted[row_idx] = False
                decisions[row_idx] = veto_action_norm
                reasons[row_idx] = "old_primary_unknown_veto"
                unknown_veto_applied[row_idx] = True
            elif not bool(row_pass):
                predicted[row_idx] = UNKNOWN_LABEL
                accepted[row_idx] = False
                decisions[row_idx] = fail_action_norm
                reasons[row_idx] = f"old_primary_consistency_{fail_action_norm}"
                if is_accepted_old:
                    blocked_accept[row_idx] = True
                else:
                    rescue_blocked[row_idx] = True
            elif is_rescue_candidate:
                predicted[row_idx] = int(label)
                accepted[row_idx] = True
                decisions[row_idx] = "accept"
                reasons[row_idx] = "old_primary_rescue_consensus_accept"
                rescue_promoted[row_idx] = True

    diagnostics = dict(result.diagnostics or {})
    diagnostics["old_primary_label"] = old_primary_label.detach().cpu()
    diagnostics["old_primary_candidate_mask"] = old_candidate_mask.detach().cpu()
    diagnostics["old_primary_evidence_delta"] = old_evidence_delta.detach().cpu()
    diagnostics["old_primary_anchor_delta"] = old_anchor_delta.detach().cpu()
    diagnostics["old_primary_anchor_margin"] = old_anchor_margin_diag.detach().cpu()
    diagnostics["old_primary_score_margin"] = score_margin_values.detach().cpu()
    diagnostics["old_primary_soft_mixture_margin"] = soft_mixture_margin.detach().cpu()
    diagnostics["old_primary_soft_mixture_cos"] = soft_mixture_cos.detach().cpu()
    diagnostics["old_primary_soft_mixture_residual"] = soft_mixture_residual.detach().cpu()
    diagnostics["old_primary_support_knn_label"] = support_knn_label.detach().cpu()
    diagnostics["old_primary_support_knn_margin"] = support_knn_margin.detach().cpu()
    diagnostics["old_primary_support_knn_seen_new_minus_old"] = support_knn_seen_new_minus_old.detach().cpu()
    diagnostics["old_primary_drift_cos"] = old_drift_cos.detach().cpu()
    diagnostics["old_primary_drift_dist"] = old_drift_dist.detach().cpu()
    diagnostics["old_primary_background_score"] = background_score.detach().cpu()
    diagnostics["old_primary_background_margin"] = background_margin.detach().cpu()
    diagnostics["old_primary_prior_veto_count"] = prior_veto_count.detach().cpu()
    diagnostics["old_primary_evidence_pass_mask"] = evidence_pass.detach().cpu()
    diagnostics["old_primary_anchor_delta_pass_mask"] = anchor_delta_pass.detach().cpu()
    diagnostics["old_primary_anchor_margin_pass_mask"] = anchor_margin_pass.detach().cpu()
    diagnostics["old_primary_score_margin_pass_mask"] = score_margin_pass.detach().cpu()
    diagnostics["old_primary_soft_mixture_pass_mask"] = soft_mixture_pass.detach().cpu()
    diagnostics["old_primary_support_knn_pass_mask"] = support_knn_pass.detach().cpu()
    diagnostics["old_primary_drift_pass_mask"] = drift_pass.detach().cpu()
    diagnostics["old_primary_class_envelope_pass_mask"] = class_envelope_pass.detach().cpu()
    diagnostics["old_primary_consistency_pass_mask"] = consistency_pass.detach().cpu()
    diagnostics["old_primary_unknown_veto_mask"] = unknown_veto.detach().cpu()
    diagnostics["old_primary_unknown_veto_applied_mask"] = unknown_veto_applied.detach().cpu()
    diagnostics["old_primary_blocked_accept_mask"] = blocked_accept.detach().cpu()
    diagnostics["old_primary_rescue_promoted_mask"] = rescue_promoted.detach().cpu()
    diagnostics["old_primary_rescue_blocked_mask"] = rescue_blocked.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_source_looo_unknown_risk_arbitration(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    source_features: torch.Tensor,
    source_labels: torch.Tensor | Iterable[int],
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    risk_quantile: float = 0.85,
    risk_slack: float = 0.0,
    min_score_margin: float = 0.02,
    min_known_evidence_delta: float = -0.08,
    background_score: float = 0.86,
    background_margin: float = 0.10,
    reject_min_failures: int = 2,
    reject_action: str = "reject",
) -> PredictionResult:
    """Reject known accepts that look no stronger than source LOOO impostors.

    Calibration uses source known classes only. For every source sample, its
    true class is temporarily hidden and the best remaining class score becomes
    a query-free impostor score. Target rows are then rejected only when their
    selected known score fails that source leave-one-old-out floor and support
    or pseudo-background evidence also looks unsafe.
    """

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result

    source_x = normalize_rows(torch.as_tensor(source_features).float())
    source_y = _labels_tensor(source_labels, device=source_x.device)
    if source_x.numel() == 0 or source_x.ndim != 2 or source_y.numel() == 0:
        return result
    if int(source_x.shape[1]) != int(x.shape[1]):
        return result

    label_to_col = {int(label): idx for idx, label in enumerate(head.class_order)}
    known_source_mask = torch.tensor(
        [int(label.item()) in label_to_col for label in source_y],
        dtype=torch.bool,
        device=source_x.device,
    )
    if not bool(known_source_mask.any().item()) or len(label_to_col) < 2:
        return result
    source_x = source_x[known_source_mask]
    source_y = source_y[known_source_mask]

    source_scores, _ = head.score_matrix(source_x.to(x.device))
    source_y = source_y.to(x.device)
    source_true_cols = torch.tensor(
        [label_to_col[int(label.item())] for label in source_y],
        dtype=torch.long,
        device=x.device,
    )
    masked_source_scores = source_scores.detach().clone()
    masked_source_scores[torch.arange(masked_source_scores.size(0), device=x.device), source_true_cols] = -float("inf")
    source_impostor_score = masked_source_scores.max(dim=1).values
    finite_impostor = source_impostor_score[torch.isfinite(source_impostor_score)]
    if finite_impostor.numel() == 0:
        return result
    q = min(1.0, max(0.0, float(risk_quantile)))
    impostor_floor = float(torch.quantile(finite_impostor.detach().float().cpu(), q).item()) + float(risk_slack)

    score_matrix, _ = head.score_matrix(x)
    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    background_available = bool(
        background.numel() > 0 and background.ndim == 2 and int(background.shape[1]) == int(x.shape[1])
    )
    if background_available:
        background = background.to(x.device)
        bg_score = torch.max(x @ background.T, dim=1).values
    else:
        bg_score = torch.full((int(x.shape[0]),), -float("inf"), dtype=torch.float32, device=x.device)
    known_rows = [
        torch.as_tensor(state.prototype).float()
        for state in head.class_states.values()
        if state is not None and torch.as_tensor(state.prototype).numel() == x.shape[1]
    ]
    if known_rows:
        known = normalize_rows(torch.stack(known_rows, dim=0)).to(x.device)
        known_score = torch.max(x @ known.T, dim=1).values
    else:
        known_score = torch.zeros_like(bg_score)
    bg_margin = bg_score - known_score

    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    def diagnostic_tensor(name: str) -> torch.Tensor:
        value = (result.diagnostics or {}).get(name)
        n = int(predicted.numel())
        if value is None or not torch.is_tensor(value):
            return torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
        flat = value.detach().float().reshape(-1).to(x.device)
        if int(flat.numel()) >= n:
            return flat[:n]
        padded = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
        padded[: int(flat.numel())] = flat
        return padded

    old_evidence_delta = diagnostic_tensor("old_support_evidence_delta")
    seen_evidence_delta = diagnostic_tensor("seen_new_evidence_delta")

    n = int(x.shape[0])
    risk_label = torch.full((n,), UNKNOWN_LABEL, dtype=torch.long, device=x.device)
    selected_score = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    second_score = torch.full_like(selected_score, float("nan"))
    score_margin = torch.full_like(selected_score, float("nan"))
    score_floor = torch.full_like(selected_score, float(impostor_floor))
    risk_margin = torch.full_like(selected_score, float("nan"))
    known_evidence_delta = torch.full_like(selected_score, float("nan"))
    pass_mask = torch.zeros((n,), dtype=torch.bool, device=x.device)
    reject_mask = torch.zeros((n,), dtype=torch.bool, device=x.device)
    failure_count = torch.zeros((n,), dtype=torch.long, device=x.device)

    for row in range(n):
        label = int(candidate_labels[row].item())
        state = head.class_states.get(label)
        col = label_to_col.get(label)
        if state is None or col is None or str(state.group) not in {"old", "seen_new"}:
            continue
        row_scores = score_matrix[row].detach()
        own_score = row_scores[col]
        competitor = row_scores.clone()
        competitor[col] = -float("inf")
        other_score = competitor.max() if int(competitor.numel()) > 1 else torch.tensor(-float("inf"), device=x.device)
        margin = own_score - other_score
        if str(state.group) == "old":
            evidence_delta = old_evidence_delta[row]
        else:
            evidence_delta = seen_evidence_delta[row]
        weak_known = bool(torch.isfinite(evidence_delta).item()) and float(evidence_delta.item()) < float(min_known_evidence_delta)
        low_floor = float(own_score.item()) < float(impostor_floor)
        low_margin = float(margin.item()) < float(min_score_margin)
        bg_risk = float(bg_score[row].item()) >= float(background_score) and float(bg_margin[row].item()) >= float(background_margin)
        failures = int(low_floor) + int(low_margin) + int(weak_known) + int(bg_risk)

        risk_label[row] = label
        selected_score[row] = own_score
        second_score[row] = other_score
        score_margin[row] = margin
        risk_margin[row] = own_score - float(impostor_floor)
        known_evidence_delta[row] = evidence_delta
        failure_count[row] = int(failures)
        pass_mask[row] = failures == 0
        if bool(accepted[row].item()) and failures >= max(1, int(reject_min_failures)) and (low_floor or bg_risk):
            reject_mask[row] = True

    action = "defer" if str(reject_action).lower() == "defer" else "reject"
    for row in torch.nonzero(reject_mask.detach().cpu(), as_tuple=False).reshape(-1).tolist():
        predicted[row] = UNKNOWN_LABEL
        accepted[row] = False
        decisions[row] = action
        reasons[row] = "source_looo_unknown_risk_reject" if action == "reject" else "source_looo_unknown_risk_defer"

    diagnostics = dict(result.diagnostics or {})
    diagnostics["source_looo_risk_label"] = risk_label.detach().cpu()
    diagnostics["source_looo_risk_score"] = selected_score.detach().cpu()
    diagnostics["source_looo_risk_floor"] = score_floor.detach().cpu()
    diagnostics["source_looo_risk_margin"] = risk_margin.detach().cpu()
    diagnostics["source_looo_second_score"] = second_score.detach().cpu()
    diagnostics["source_looo_score_margin"] = score_margin.detach().cpu()
    diagnostics["source_looo_known_evidence_delta"] = known_evidence_delta.detach().cpu()
    diagnostics["source_looo_background_score"] = bg_score.detach().cpu()
    diagnostics["source_looo_background_margin"] = bg_margin.detach().cpu()
    diagnostics["source_looo_pass_mask"] = pass_mask.detach().cpu()
    diagnostics["source_looo_reject_mask"] = reject_mask.detach().cpu()
    diagnostics["source_looo_failure_count"] = failure_count.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_class_envelope_gate(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    *,
    enabled: bool = False,
) -> PredictionResult:
    """Reject accepted rows outside the query-free class source/support envelope."""

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result
    score_matrix, per_class = head.score_matrix(x)
    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    envelope_label = torch.full_like(predicted, UNKNOWN_LABEL)
    envelope_evidence = torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
    envelope_residual = torch.full_like(envelope_evidence, float("nan"))
    envelope_score = torch.full_like(envelope_evidence, float("nan"))
    envelope_margin = torch.full_like(envelope_evidence, float("nan"))
    envelope_failure_count = torch.zeros_like(envelope_evidence)
    envelope_reject_mask = torch.zeros_like(accepted, dtype=torch.bool)
    threshold_by_label = {int(label): idx for idx, label in enumerate(head.class_order)}

    for row_idx, label in enumerate(candidate_labels.tolist()):
        state = head.class_states.get(int(label))
        if state is None or not bool(state.thresholds.get("class_envelope_gate_enabled", False)):
            continue
        col = threshold_by_label.get(int(label))
        if col is None:
            continue
        row_scores = score_matrix[row_idx]
        score = row_scores[col].detach()
        if row_scores.numel() > 1:
            competitor = row_scores.detach().clone()
            competitor[col] = -float("inf")
            margin = score - competitor.max()
        else:
            margin = torch.tensor(1.0, dtype=score.dtype, device=score.device)
        residual = per_class[int(label)]["residual"][row_idx].detach()
        maha = per_class[int(label)]["maha"][row_idx].detach()
        cos = per_class[int(label)]["cos"][row_idx].detach()
        if str(state.group) == "old":
            anchor = _support_anchor_similarity(x[row_idx].view(1, -1), state)[0].detach()
            evidence = _old_support_evidence_score(cos, anchor, residual, maha)
        else:
            evidence = _seen_new_evidence_score(cos, residual, maha)
        min_evidence = _state_threshold(state, "min_class_envelope_evidence", -float("inf"))
        max_residual = _state_threshold(state, "max_class_envelope_residual", float("inf"))
        min_score = _state_threshold(state, "min_class_envelope_score", -float("inf"))
        min_margin = _state_threshold(state, "min_class_envelope_margin", -float("inf"))
        failures = 0
        failures += int(float(evidence.item()) < min_evidence)
        failures += int(float(residual.item()) > max_residual)
        failures += int(float(score.item()) < min_score)
        failures += int(float(margin.item()) < min_margin)
        envelope_label[row_idx] = int(label)
        envelope_evidence[row_idx] = float(evidence.item())
        envelope_residual[row_idx] = float(residual.item())
        envelope_score[row_idx] = float(score.item())
        envelope_margin[row_idx] = float(margin.item())
        envelope_failure_count[row_idx] = float(failures)
        min_failures = max(1, int(state.thresholds.get("class_envelope_min_failures", 1)))
        if bool(accepted[row_idx].item()) and failures >= min_failures:
            action = str(state.thresholds.get("class_envelope_action", "reject")).lower()
            predicted[row_idx] = UNKNOWN_LABEL
            accepted[row_idx] = False
            decisions[row_idx] = "reject" if action == "reject" else "uncertain"
            reasons[row_idx] = "class_envelope_reject" if action == "reject" else "class_envelope_uncertain"
            envelope_reject_mask[row_idx] = True

    diagnostics = dict(result.diagnostics or {})
    diagnostics["class_envelope_label"] = envelope_label.detach().cpu()
    diagnostics["class_envelope_evidence"] = envelope_evidence.detach().cpu()
    diagnostics["class_envelope_residual"] = envelope_residual.detach().cpu()
    diagnostics["class_envelope_score"] = envelope_score.detach().cpu()
    diagnostics["class_envelope_margin"] = envelope_margin.detach().cpu()
    diagnostics["class_envelope_failure_count"] = envelope_failure_count.detach().cpu()
    diagnostics["class_envelope_reject_mask"] = envelope_reject_mask.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_retention_rescue_gate(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    old_min_evidence_delta: float = 0.02,
    old_min_anchor_delta: float = -0.01,
    old_min_anchor_margin: float = 0.0,
    old_min_score_margin: float = 0.0,
    seen_new_min_evidence_delta: float = 0.02,
    seen_new_min_anchor_delta: float = 0.0,
    seen_new_min_score_margin: float = -0.02,
    max_background_score: float = 0.70,
    max_background_margin: float = 0.06,
    direct_accept: bool = True,
) -> PredictionResult:
    """Mark or recover rejected known rows when support evidence beats background risk.

    The gate is query-label free: it uses the candidate label from the OA-MSE
    head, source/allowed-support class states, existing support diagnostics,
    and pseudo-background anchors. Unknown query labels are never used. When
    direct_accept is false, this gate only marks rescue eligibility; a later
    old-primary consensus gate must make the final accept decision.
    """

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result
    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    background_available = bool(background.numel() > 0 and background.ndim == 2 and int(background.shape[1]) == int(x.shape[1]))
    if background_available:
        background = background.to(x.device)
        background_score = torch.max(x @ background.T, dim=1).values
    else:
        background_score = torch.full((int(x.shape[0]),), -float("inf"), dtype=torch.float32, device=x.device)
    known_rows = [
        torch.as_tensor(state.prototype).float()
        for state in head.class_states.values()
        if state is not None and torch.as_tensor(state.prototype).numel() == x.shape[1]
    ]
    if known_rows:
        known = normalize_rows(torch.stack(known_rows, dim=0)).to(x.device)
        known_score = torch.max(x @ known.T, dim=1).values
    else:
        known_score = torch.zeros_like(background_score)
    background_margin = background_score - known_score

    score_matrix, _ = head.score_matrix(x)
    label_to_col = {int(label): idx for idx, label in enumerate(head.class_order)}
    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    def diagnostic_tensor(name: str) -> torch.Tensor:
        value = (result.diagnostics or {}).get(name)
        if value is None or not torch.is_tensor(value):
            return torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
        flat = value.detach().cpu().float().reshape(-1)
        if int(flat.numel()) >= int(predicted.numel()):
            return flat[: int(predicted.numel())]
        padded = torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
        padded[: int(flat.numel())] = flat
        return padded

    old_evidence_delta = diagnostic_tensor("old_support_evidence_delta")
    old_anchor_delta = diagnostic_tensor("old_support_anchor_delta")
    old_anchor_margin = diagnostic_tensor("old_support_anchor_margin")
    seen_evidence_delta = diagnostic_tensor("seen_new_evidence_delta")
    seen_anchor_delta = diagnostic_tensor("seen_new_anchor_delta")

    rescue_label = torch.full_like(predicted, UNKNOWN_LABEL)
    rescue_score = torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
    rescue_margin = torch.full_like(rescue_score, float("nan"))
    rescue_evidence_delta = torch.full_like(rescue_score, float("nan"))
    rescue_anchor_delta = torch.full_like(rescue_score, float("nan"))
    rescue_anchor_margin = torch.full_like(rescue_score, float("nan"))
    eligible_mask = torch.zeros_like(accepted, dtype=torch.bool)
    rescue_mask = torch.zeros_like(accepted, dtype=torch.bool)

    for row_idx, label in enumerate(candidate_labels.tolist()):
        state = head.class_states.get(int(label))
        col = label_to_col.get(int(label))
        if state is None or col is None:
            continue
        row_scores = score_matrix[row_idx]
        score = row_scores[col].detach()
        if row_scores.numel() > 1:
            competitor = row_scores.detach().clone()
            competitor[col] = -float("inf")
            margin = score - competitor.max()
        else:
            margin = torch.tensor(1.0, dtype=score.dtype, device=score.device)
        rescue_label[row_idx] = int(label)
        rescue_score[row_idx] = float(score.item())
        rescue_margin[row_idx] = float(margin.item())
        bg_ok = (
            float(background_score[row_idx].detach().cpu().item()) <= float(max_background_score)
            and float(background_margin[row_idx].detach().cpu().item()) <= float(max_background_margin)
        )
        if str(state.group) == "old":
            rescue_evidence_delta[row_idx] = float(old_evidence_delta[row_idx].item())
            rescue_anchor_delta[row_idx] = float(old_anchor_delta[row_idx].item())
            rescue_anchor_margin[row_idx] = float(old_anchor_margin[row_idx].item())
            evidence_ok = (
                float(old_evidence_delta[row_idx].item()) >= float(old_min_evidence_delta)
                and float(old_anchor_delta[row_idx].item()) >= float(old_min_anchor_delta)
                and float(old_anchor_margin[row_idx].item()) >= float(old_min_anchor_margin)
                and float(margin.detach().cpu().item()) >= float(old_min_score_margin)
            )
        elif str(state.group) == "seen_new":
            rescue_evidence_delta[row_idx] = float(seen_evidence_delta[row_idx].item())
            rescue_anchor_delta[row_idx] = float(seen_anchor_delta[row_idx].item())
            evidence_ok = (
                float(seen_evidence_delta[row_idx].item()) >= float(seen_new_min_evidence_delta)
                and float(seen_anchor_delta[row_idx].item()) >= float(seen_new_min_anchor_delta)
                and float(margin.detach().cpu().item()) >= float(seen_new_min_score_margin)
            )
        else:
            evidence_ok = False
        eligible = bool(evidence_ok and bg_ok)
        eligible_mask[row_idx] = eligible
        if eligible and bool(direct_accept) and not bool(accepted[row_idx].item()):
            predicted[row_idx] = int(label)
            accepted[row_idx] = True
            decisions[row_idx] = "accept"
            reasons[row_idx] = "retention_rescue_accept"
            rescue_mask[row_idx] = True

    diagnostics = dict(result.diagnostics or {})
    diagnostics["retention_rescue_label"] = rescue_label.detach().cpu()
    diagnostics["retention_rescue_score"] = rescue_score.detach().cpu()
    diagnostics["retention_rescue_margin"] = rescue_margin.detach().cpu()
    diagnostics["retention_rescue_evidence_delta"] = rescue_evidence_delta.detach().cpu()
    diagnostics["retention_rescue_anchor_delta"] = rescue_anchor_delta.detach().cpu()
    diagnostics["retention_rescue_anchor_margin"] = rescue_anchor_margin.detach().cpu()
    diagnostics["retention_rescue_background_score"] = background_score.detach().cpu()
    diagnostics["retention_rescue_background_margin"] = background_margin.detach().cpu()
    diagnostics["retention_rescue_eligible_mask"] = eligible_mask.detach().cpu()
    diagnostics["retention_rescue_accept_mask"] = rescue_mask.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_pre_reject_defer_arbitration(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    old_min_evidence_delta: float = 0.0,
    old_min_anchor_delta: float = -0.02,
    old_min_anchor_margin: float = 0.0,
    old_min_score_margin: float = -0.02,
    seen_new_min_evidence_delta: float = 0.0,
    seen_new_min_anchor_delta: float = 0.0,
    seen_new_min_score_margin: float = -0.05,
    max_background_score: float = 0.74,
    max_background_margin: float = 0.10,
    defer_background_score: float = 0.70,
    defer_background_margin: float = 0.04,
    reject_background_score: float = 0.82,
    reject_background_margin: float = 0.12,
    defer_action: str = "uncertain",
    support_neighborhood_retention: bool = False,
    support_neighborhood_old_min_evidence_delta: float = 0.02,
    support_neighborhood_old_min_anchor_delta: float = -0.04,
    support_neighborhood_old_min_anchor_margin: float = -0.02,
    support_neighborhood_old_min_score_margin: float = -0.04,
    support_neighborhood_seen_new_min_evidence_delta: float = 0.02,
    support_neighborhood_seen_new_min_anchor_delta: float = -0.04,
    support_neighborhood_seen_new_min_score_margin: float = -0.08,
    support_neighborhood_max_background_score: float = 0.96,
    support_neighborhood_max_background_margin: float = 0.30,
    support_neighborhood_require_source_looo_pass: bool = False,
    support_neighborhood_source_looo_max_failures: int = 0,
) -> PredictionResult:
    """Arbitrate known evidence against pseudo-background before hard rejection.

    This runs before the ambiguous-row verifier. It is query-label free: known
    evidence comes from source/allowed support diagnostics, while background
    risk comes from pseudo-unknown anchors generated without unknown query
    labels.
    """

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result
    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    background_available = bool(
        background.numel() > 0 and background.ndim == 2 and int(background.shape[1]) == int(x.shape[1])
    )
    if background_available:
        background = background.to(x.device)
        background_score = torch.max(x @ background.T, dim=1).values
    else:
        background_score = torch.full((int(x.shape[0]),), -float("inf"), dtype=torch.float32, device=x.device)
    known_rows = [
        torch.as_tensor(state.prototype).float()
        for state in head.class_states.values()
        if state is not None and torch.as_tensor(state.prototype).numel() == x.shape[1]
    ]
    if known_rows:
        known = normalize_rows(torch.stack(known_rows, dim=0)).to(x.device)
        known_score = torch.max(x @ known.T, dim=1).values
    else:
        known_score = torch.zeros_like(background_score)
    background_margin = background_score - known_score
    background_available_mask = torch.full_like(background_score, bool(background_available), dtype=torch.bool)
    background_accept_ok = background_available_mask & (background_score <= float(max_background_score)) & (
        background_margin <= float(max_background_margin)
    )
    background_defer_risk = background_available_mask & (background_score >= float(defer_background_score)) & (
        background_margin >= float(defer_background_margin)
    )
    background_reject_risk = background_available_mask & (background_score >= float(reject_background_score)) & (
        background_margin >= float(reject_background_margin)
    )

    score_matrix, _ = head.score_matrix(x)
    label_to_col = {int(label): idx for idx, label in enumerate(head.class_order)}
    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    def diagnostic_tensor(name: str) -> torch.Tensor:
        value = (result.diagnostics or {}).get(name)
        if value is None or not torch.is_tensor(value):
            return torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
        flat = value.detach().cpu().float().reshape(-1)
        if int(flat.numel()) >= int(predicted.numel()):
            return flat[: int(predicted.numel())]
        padded = torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
        padded[: int(flat.numel())] = flat
        return padded

    old_evidence_delta = diagnostic_tensor("old_support_evidence_delta")
    old_anchor_delta = diagnostic_tensor("old_support_anchor_delta")
    old_anchor_margin = diagnostic_tensor("old_support_anchor_margin")
    seen_evidence_delta = diagnostic_tensor("seen_new_evidence_delta")
    seen_anchor_delta = diagnostic_tensor("seen_new_anchor_delta")
    source_looo_failure_count = diagnostic_tensor("source_looo_failure_count")
    source_looo_reject_value = (result.diagnostics or {}).get("source_looo_reject_mask")
    if source_looo_reject_value is not None and torch.is_tensor(source_looo_reject_value):
        source_looo_reject = source_looo_reject_value.detach().cpu().bool().reshape(-1)
        if int(source_looo_reject.numel()) < int(predicted.numel()):
            padded = torch.zeros((int(predicted.numel()),), dtype=torch.bool)
            padded[: int(source_looo_reject.numel())] = source_looo_reject
            source_looo_reject = padded
        else:
            source_looo_reject = source_looo_reject[: int(predicted.numel())]
    else:
        source_looo_reject = torch.zeros((int(predicted.numel()),), dtype=torch.bool)

    arbitration_label = torch.full_like(predicted, UNKNOWN_LABEL)
    arbitration_score = torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
    arbitration_margin = torch.full_like(arbitration_score, float("nan"))
    arbitration_evidence_delta = torch.full_like(arbitration_score, float("nan"))
    arbitration_anchor_delta = torch.full_like(arbitration_score, float("nan"))
    arbitration_anchor_margin = torch.full_like(arbitration_score, float("nan"))
    evidence_ok_mask = torch.zeros_like(accepted, dtype=torch.bool)
    accept_mask = torch.zeros_like(accepted, dtype=torch.bool)
    reject_mask = torch.zeros_like(accepted, dtype=torch.bool)
    defer_mask = torch.zeros_like(accepted, dtype=torch.bool)
    uncertain_mask = torch.zeros_like(accepted, dtype=torch.bool)
    support_retention_mask = torch.zeros_like(accepted, dtype=torch.bool)
    extreme_background_mask = torch.zeros_like(accepted, dtype=torch.bool)
    support_retention_source_looo_block_mask = torch.zeros_like(accepted, dtype=torch.bool)
    defer_action_norm = str(defer_action).lower().strip()
    if defer_action_norm not in {"uncertain", "defer"}:
        defer_action_norm = "uncertain"

    for row_idx, label in enumerate(candidate_labels.tolist()):
        state = head.class_states.get(int(label))
        col = label_to_col.get(int(label))
        if state is None or col is None:
            continue
        row_scores = score_matrix[row_idx]
        score = row_scores[col].detach()
        if row_scores.numel() > 1:
            competitor = row_scores.detach().clone()
            competitor[col] = -float("inf")
            margin = score - competitor.max()
        else:
            margin = torch.tensor(1.0, dtype=score.dtype, device=score.device)
        arbitration_label[row_idx] = int(label)
        arbitration_score[row_idx] = float(score.item())
        arbitration_margin[row_idx] = float(margin.item())
        if str(state.group) == "old":
            arbitration_evidence_delta[row_idx] = float(old_evidence_delta[row_idx].item())
            arbitration_anchor_delta[row_idx] = float(old_anchor_delta[row_idx].item())
            arbitration_anchor_margin[row_idx] = float(old_anchor_margin[row_idx].item())
            evidence_ok = (
                float(old_evidence_delta[row_idx].item()) >= float(old_min_evidence_delta)
                and float(old_anchor_delta[row_idx].item()) >= float(old_min_anchor_delta)
                and float(old_anchor_margin[row_idx].item()) >= float(old_min_anchor_margin)
                and float(margin.detach().cpu().item()) >= float(old_min_score_margin)
            )
        elif str(state.group) == "seen_new":
            arbitration_evidence_delta[row_idx] = float(seen_evidence_delta[row_idx].item())
            arbitration_anchor_delta[row_idx] = float(seen_anchor_delta[row_idx].item())
            evidence_ok = (
                float(seen_evidence_delta[row_idx].item()) >= float(seen_new_min_evidence_delta)
                and float(seen_anchor_delta[row_idx].item()) >= float(seen_new_min_anchor_delta)
                and float(margin.detach().cpu().item()) >= float(seen_new_min_score_margin)
            )
        else:
            evidence_ok = False
        evidence_ok_mask[row_idx] = bool(evidence_ok)
        background_is_extreme = bool(background_available_mask[row_idx].detach().cpu().item()) and (
            float(background_score[row_idx].detach().cpu().item()) > float(support_neighborhood_max_background_score)
            or float(background_margin[row_idx].detach().cpu().item()) > float(support_neighborhood_max_background_margin)
        )
        extreme_background_mask[row_idx] = bool(background_is_extreme)
        support_retention_ok = False
        if bool(support_neighborhood_retention) and not bool(background_is_extreme):
            source_looo_blocked = False
            if bool(support_neighborhood_require_source_looo_pass):
                source_failures = float(source_looo_failure_count[row_idx].item())
                source_rejected = bool(source_looo_reject[row_idx].item())
                source_looo_blocked = source_rejected or (
                    math.isfinite(source_failures)
                    and source_failures > float(max(0, int(support_neighborhood_source_looo_max_failures)))
                )
            support_retention_source_looo_block_mask[row_idx] = bool(source_looo_blocked)
            if str(state.group) == "old":
                support_retention_ok = (
                    not bool(source_looo_blocked)
                    and float(old_evidence_delta[row_idx].item()) >= float(support_neighborhood_old_min_evidence_delta)
                    and float(old_anchor_delta[row_idx].item()) >= float(support_neighborhood_old_min_anchor_delta)
                    and float(old_anchor_margin[row_idx].item()) >= float(support_neighborhood_old_min_anchor_margin)
                    and float(margin.detach().cpu().item()) >= float(support_neighborhood_old_min_score_margin)
                )
            elif str(state.group) == "seen_new":
                support_retention_ok = (
                    not bool(source_looo_blocked)
                    and float(seen_evidence_delta[row_idx].item()) >= float(support_neighborhood_seen_new_min_evidence_delta)
                    and float(seen_anchor_delta[row_idx].item()) >= float(support_neighborhood_seen_new_min_anchor_delta)
                    and float(margin.detach().cpu().item()) >= float(support_neighborhood_seen_new_min_score_margin)
                )
        if bool(support_retention_ok):
            predicted[row_idx] = int(label)
            accepted[row_idx] = True
            decisions[row_idx] = "accept"
            reasons[row_idx] = "pre_reject_support_neighborhood_retention_accept"
            accept_mask[row_idx] = True
            support_retention_mask[row_idx] = True
            continue
        if bool(evidence_ok) and bool(background_accept_ok[row_idx].detach().cpu().item()):
            predicted[row_idx] = int(label)
            accepted[row_idx] = True
            decisions[row_idx] = "accept"
            reasons[row_idx] = "pre_reject_arbitration_accept"
            accept_mask[row_idx] = True
            continue
        if bool(background_reject_risk[row_idx].detach().cpu().item()) and not bool(evidence_ok):
            predicted[row_idx] = UNKNOWN_LABEL
            accepted[row_idx] = False
            decisions[row_idx] = "reject"
            reasons[row_idx] = "pre_reject_arbitration_background_reject"
            reject_mask[row_idx] = True
            continue
        if bool(background_defer_risk[row_idx].detach().cpu().item()) or str(decisions[row_idx]) == "reject":
            predicted[row_idx] = UNKNOWN_LABEL
            accepted[row_idx] = False
            decisions[row_idx] = defer_action_norm
            reasons[row_idx] = "pre_reject_arbitration_defer"
            if defer_action_norm == "defer":
                defer_mask[row_idx] = True
            else:
                uncertain_mask[row_idx] = True

    diagnostics = dict(result.diagnostics or {})
    diagnostics["pre_reject_arbitration_label"] = arbitration_label.detach().cpu()
    diagnostics["pre_reject_arbitration_score"] = arbitration_score.detach().cpu()
    diagnostics["pre_reject_arbitration_margin"] = arbitration_margin.detach().cpu()
    diagnostics["pre_reject_arbitration_evidence_delta"] = arbitration_evidence_delta.detach().cpu()
    diagnostics["pre_reject_arbitration_anchor_delta"] = arbitration_anchor_delta.detach().cpu()
    diagnostics["pre_reject_arbitration_anchor_margin"] = arbitration_anchor_margin.detach().cpu()
    diagnostics["pre_reject_arbitration_background_score"] = background_score.detach().cpu()
    diagnostics["pre_reject_arbitration_background_margin"] = background_margin.detach().cpu()
    diagnostics["pre_reject_arbitration_background_available_mask"] = background_available_mask.detach().cpu()
    diagnostics["pre_reject_arbitration_background_accept_ok_mask"] = background_accept_ok.detach().cpu()
    diagnostics["pre_reject_arbitration_background_defer_risk_mask"] = background_defer_risk.detach().cpu()
    diagnostics["pre_reject_arbitration_background_reject_risk_mask"] = background_reject_risk.detach().cpu()
    diagnostics["pre_reject_arbitration_evidence_ok_mask"] = evidence_ok_mask.detach().cpu()
    diagnostics["pre_reject_arbitration_support_retention_mask"] = support_retention_mask.detach().cpu()
    diagnostics["pre_reject_arbitration_extreme_background_mask"] = extreme_background_mask.detach().cpu()
    diagnostics["pre_reject_arbitration_support_retention_source_looo_block_mask"] = (
        support_retention_source_looo_block_mask.detach().cpu()
    )
    diagnostics["pre_reject_arbitration_accept_mask"] = accept_mask.detach().cpu()
    diagnostics["pre_reject_arbitration_reject_mask"] = reject_mask.detach().cpu()
    diagnostics["pre_reject_arbitration_defer_mask"] = defer_mask.detach().cpu()
    diagnostics["pre_reject_arbitration_uncertain_mask"] = uncertain_mask.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_three_way_decision_head(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    temperature: float = 0.10,
    accept_prob: float = 0.50,
    reject_prob: float = 0.55,
    defer_prob: float = 0.45,
    known_background_margin: float = 0.02,
    reject_margin: float = 0.04,
    old_seen_margin: float = -0.06,
    defer_action: str = "uncertain",
    known_floor_enabled: bool = False,
    known_floor_action: str = "defer",
    known_floor_old_min_evidence_delta: float = -0.04,
    known_floor_old_min_anchor_delta: float = -0.08,
    known_floor_old_min_anchor_margin: float = -0.04,
    known_floor_old_min_score_margin: float = -0.12,
    known_floor_seen_new_min_evidence_delta: float = -0.04,
    known_floor_seen_new_min_anchor_delta: float = -0.08,
    known_floor_seen_new_min_score_margin: float = -0.12,
    known_floor_background_override_prob: float = 0.995,
    known_floor_background_override_margin: float = 1.00,
    decision_policy: str = "background_competition",
) -> PredictionResult:
    """Apply an explicit old / seen-new / background decision head.

    The head uses support/source class states and query-free pseudo-background
    anchors. It does not look at unknown query labels or fit thresholds from
    eval-only unknowns.
    """

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    if x.numel() == 0 or x.ndim != 2:
        return result
    background_available = bool(
        background.numel() > 0 and background.ndim == 2 and int(background.shape[1]) == int(x.shape[1])
    )
    score_matrix, _ = head.score_matrix(x)
    old_cols = [idx for idx, label in enumerate(head.class_order) if str(head.class_states[int(label)].group) == "old"]
    seen_cols = [idx for idx, label in enumerate(head.class_order) if str(head.class_states[int(label)].group) == "seen_new"]
    if not old_cols or not background_available:
        return result
    old_scores = score_matrix[:, old_cols]
    best_old_score, best_old_pos = old_scores.max(dim=1)
    old_labels = torch.tensor([head.class_order[int(old_cols[int(v)])] for v in best_old_pos.detach().cpu().tolist()], dtype=torch.long, device=x.device)
    seen_available = bool(seen_cols)
    if seen_available:
        seen_scores = score_matrix[:, seen_cols]
        best_seen_score, best_seen_pos = seen_scores.max(dim=1)
        seen_labels = torch.tensor([head.class_order[int(seen_cols[int(v)])] for v in best_seen_pos.detach().cpu().tolist()], dtype=torch.long, device=x.device)
    else:
        best_seen_score = torch.full_like(best_old_score, -1.0e6)
        seen_labels = old_labels.clone()
    background = background.to(x.device)
    background_score = torch.max(x @ background.T, dim=1).values
    tau = max(float(temperature), 1.0e-4)
    if seen_available:
        logits = torch.stack([best_old_score, best_seen_score, background_score], dim=1) / tau
        probs = F.softmax(logits, dim=1)
        old_prob = probs[:, 0]
        seen_prob = probs[:, 1]
        background_prob = probs[:, 2]
        known_pair_logits = torch.stack([best_old_score, best_seen_score], dim=1) / tau
        known_pair_probs = F.softmax(known_pair_logits, dim=1)
        old_known_prob = known_pair_probs[:, 0]
        seen_known_prob = known_pair_probs[:, 1]
        best_known_prob_class_first = torch.maximum(old_known_prob, seen_known_prob)
    else:
        logits = torch.stack([best_old_score, background_score], dim=1) / tau
        probs = F.softmax(logits, dim=1)
        old_prob = probs[:, 0]
        seen_prob = torch.zeros_like(old_prob)
        background_prob = probs[:, 1]
        old_known_prob = torch.ones_like(old_prob)
        seen_known_prob = torch.zeros_like(old_prob)
        best_known_prob_class_first = old_known_prob
    best_known_score = torch.maximum(best_old_score, best_seen_score)
    policy_norm = str(decision_policy).lower().strip()
    if policy_norm not in {"background_competition", "class_first", "evidence_balanced"}:
        policy_norm = "background_competition"
    best_known_prob = best_known_prob_class_first if policy_norm == "class_first" else torch.maximum(old_prob, seen_prob)
    best_known_is_seen = (
        (seen_known_prob > old_known_prob if policy_norm == "class_first" else seen_prob > old_prob)
        if seen_available
        else torch.zeros_like(old_prob, dtype=torch.bool)
    )
    best_known_label = torch.where(best_known_is_seen, seen_labels, old_labels)
    background_margin = background_score - best_known_score
    known_background_gap = best_known_score - background_score
    old_seen_gap = (best_old_score - best_seen_score).abs()
    diagnostics = dict(result.diagnostics or {})

    def _diag_tensor(name: str, default: float = float("nan")) -> torch.Tensor:
        value = diagnostics.get(name)
        if value is None:
            return torch.full_like(best_known_score, float(default), dtype=torch.float32)
        tensor = torch.as_tensor(value, dtype=torch.float32, device=x.device).reshape(-1)
        if int(tensor.numel()) != int(best_known_score.numel()):
            return torch.full_like(best_known_score, float(default), dtype=torch.float32)
        return tensor

    known_floor_action_norm = str(known_floor_action).lower().strip()
    if known_floor_action_norm not in {"accept", "defer", "uncertain"}:
        known_floor_action_norm = "defer"
    old_evidence_delta = _diag_tensor("old_support_evidence_delta")
    old_anchor_delta = _diag_tensor("old_support_anchor_delta")
    old_anchor_margin = _diag_tensor("old_support_anchor_margin")
    seen_evidence_delta = _diag_tensor("seen_new_evidence_delta")
    seen_anchor_delta = _diag_tensor("seen_new_anchor_delta")
    old_score_margin = best_old_score - best_seen_score
    seen_score_margin = best_seen_score - best_old_score
    old_floor = (
        (old_evidence_delta >= float(known_floor_old_min_evidence_delta))
        & (old_anchor_delta >= float(known_floor_old_min_anchor_delta))
        & (old_anchor_margin >= float(known_floor_old_min_anchor_margin))
        & (old_score_margin >= float(known_floor_old_min_score_margin))
    )
    seen_floor = (
        (seen_evidence_delta >= float(known_floor_seen_new_min_evidence_delta))
        & (seen_anchor_delta >= float(known_floor_seen_new_min_anchor_delta))
        & (seen_score_margin >= float(known_floor_seen_new_min_score_margin))
        if seen_available
        else torch.zeros_like(old_floor, dtype=torch.bool)
    )
    known_floor_mask = torch.zeros_like(best_known_score, dtype=torch.bool)
    if bool(known_floor_enabled):
        known_floor_mask = torch.where(best_known_is_seen, seen_floor, old_floor)
    known_evidence_mask = torch.where(best_known_is_seen, seen_floor, old_floor)
    extreme_background_mask = (
        (background_prob >= float(known_floor_background_override_prob))
        & (background_margin >= float(known_floor_background_override_margin))
    )

    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    base_reject_mask = (background_prob >= float(reject_prob)) & (background_margin >= float(reject_margin))
    if policy_norm == "class_first":
        class_first_known_evidence_mask = known_floor_mask | (best_known_prob >= float(accept_prob))
        reject_mask = extreme_background_mask | (base_reject_mask & ~class_first_known_evidence_mask)
        accept_mask = class_first_known_evidence_mask & ~reject_mask
    elif policy_norm == "evidence_balanced":
        class_first_known_evidence_mask = known_evidence_mask & (best_known_prob_class_first >= float(accept_prob))
        reject_mask = extreme_background_mask | (base_reject_mask & ~known_evidence_mask)
        accept_mask = (
            class_first_known_evidence_mask
            & ~reject_mask
            & (known_background_gap >= float(known_background_margin))
        )
    else:
        class_first_known_evidence_mask = torch.zeros_like(best_known_score, dtype=torch.bool)
        reject_mask = base_reject_mask & (~known_floor_mask | extreme_background_mask)
        accept_mask = (best_known_prob >= float(accept_prob)) & (known_background_gap >= float(known_background_margin))
    floor_accept_mask = (
        known_floor_mask
        & ~extreme_background_mask
        & ~reject_mask
        & ~accept_mask
        & (known_floor_action_norm == "accept")
    )
    accept_mask = accept_mask | floor_accept_mask
    floor_defer_mask = (
        known_floor_mask
        & ~extreme_background_mask
        & ~reject_mask
        & ~accept_mask
        & (known_floor_action_norm in {"defer", "uncertain"})
    )
    defer_mask = (
        ~reject_mask
        & ~accept_mask
        & (floor_defer_mask | (background_prob >= float(defer_prob)) | (old_seen_gap <= float(old_seen_margin)))
    )
    defer_action_norm = str(defer_action).lower().strip()
    if defer_action_norm not in {"uncertain", "defer"}:
        defer_action_norm = "uncertain"

    for row in range(int(predicted.numel())):
        if bool(reject_mask[row].detach().cpu().item()):
            predicted[row] = UNKNOWN_LABEL
            accepted[row] = False
            decisions[row] = "reject"
            reasons[row] = "three_way_head_background_reject"
        elif bool(accept_mask[row].detach().cpu().item()):
            label = int(best_known_label[row].detach().cpu().item())
            predicted[row] = label
            candidate_labels[row] = label
            accepted[row] = True
            decisions[row] = "accept"
            reasons[row] = "three_way_head_accept"
        elif bool(defer_mask[row].detach().cpu().item()):
            predicted[row] = UNKNOWN_LABEL
            accepted[row] = False
            action = known_floor_action_norm if bool(floor_defer_mask[row].detach().cpu().item()) else defer_action_norm
            decisions[row] = action
            reasons[row] = "three_way_known_floor_defer" if bool(floor_defer_mask[row].detach().cpu().item()) else "three_way_head_defer"

    diagnostics["three_way_old_score"] = best_old_score.detach().cpu()
    diagnostics["three_way_seen_new_score"] = best_seen_score.detach().cpu()
    diagnostics["three_way_background_score"] = background_score.detach().cpu()
    diagnostics["three_way_old_prob"] = old_prob.detach().cpu()
    diagnostics["three_way_seen_new_prob"] = seen_prob.detach().cpu()
    diagnostics["three_way_background_prob"] = background_prob.detach().cpu()
    diagnostics["three_way_old_known_prob"] = old_known_prob.detach().cpu()
    diagnostics["three_way_seen_new_known_prob"] = seen_known_prob.detach().cpu()
    diagnostics["three_way_known_prob_class_first"] = best_known_prob_class_first.detach().cpu()
    diagnostics["three_way_known_background_gap"] = known_background_gap.detach().cpu()
    diagnostics["three_way_background_margin"] = background_margin.detach().cpu()
    diagnostics["three_way_old_seen_gap"] = old_seen_gap.detach().cpu()
    diagnostics["three_way_label"] = best_known_label.detach().cpu()
    diagnostics["three_way_base_reject_mask"] = base_reject_mask.detach().cpu()
    diagnostics["three_way_known_floor_mask"] = known_floor_mask.detach().cpu()
    diagnostics["three_way_old_floor_mask"] = old_floor.detach().cpu()
    diagnostics["three_way_seen_new_floor_mask"] = seen_floor.detach().cpu()
    diagnostics["three_way_evidence_balanced_known_evidence_mask"] = known_evidence_mask.detach().cpu()
    diagnostics["three_way_extreme_background_mask"] = extreme_background_mask.detach().cpu()
    diagnostics["three_way_floor_accept_mask"] = floor_accept_mask.detach().cpu()
    diagnostics["three_way_floor_defer_mask"] = floor_defer_mask.detach().cpu()
    diagnostics["three_way_class_first_known_evidence_mask"] = class_first_known_evidence_mask.detach().cpu()
    diagnostics["three_way_reject_suppressed_by_floor_mask"] = (base_reject_mask & known_floor_mask & ~extreme_background_mask).detach().cpu()
    diagnostics["three_way_accept_mask"] = accept_mask.detach().cpu()
    diagnostics["three_way_reject_mask"] = reject_mask.detach().cpu()
    diagnostics["three_way_defer_mask"] = defer_mask.detach().cpu()
    diagnostics["three_way_background_available_mask"] = torch.full_like(background_prob, True, dtype=torch.bool).detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_density_shell_inlier_gate(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    old_min_evidence_delta: float = -0.04,
    old_min_anchor_delta: float = -0.08,
    old_min_density_delta: float = -0.06,
    seen_new_min_evidence_delta: float = -0.04,
    seen_new_min_anchor_delta: float = -0.08,
    seen_new_min_density_delta: float = -0.06,
    accept_background_margin: float = 0.18,
    reject_background_score: float = 0.86,
    reject_background_margin: float = 0.14,
    reject_min_failed_shells: int = 2,
) -> PredictionResult:
    """Inlier-first class density shell arbitration.

    This gate is fitted from source/allowed-support class states only. It first
    asks whether a query is inside an old or seen-new class shell, then lets
    pseudo-background risk reject rows only when no class shell is plausible.
    Unknown query labels are never used for calibration.
    """

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result

    score_matrix, per_class = head.score_matrix(x)
    old_cols = [idx for idx, label in enumerate(head.class_order) if str(head.class_states[int(label)].group) == "old"]
    seen_cols = [idx for idx, label in enumerate(head.class_order) if str(head.class_states[int(label)].group) == "seen_new"]
    if not old_cols and not seen_cols:
        return result

    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    background_available = bool(
        background.numel() > 0 and background.ndim == 2 and int(background.shape[1]) == int(x.shape[1])
    )
    if background_available:
        background = background.to(x.device)
        background_score = torch.max(x @ background.T, dim=1).values
    else:
        background_score = torch.full((int(x.shape[0]),), -float("inf"), dtype=torch.float32, device=x.device)
    known_rows = [
        torch.as_tensor(state.prototype).float()
        for state in head.class_states.values()
        if state is not None and torch.as_tensor(state.prototype).numel() == x.shape[1]
    ]
    if known_rows:
        known = normalize_rows(torch.stack(known_rows, dim=0)).to(x.device)
        known_score = torch.max(x @ known.T, dim=1).values
    else:
        known_score = torch.zeros_like(background_score)
    background_margin = background_score - known_score

    def _best_shell(cols: list[int], group_name: str) -> dict[str, torch.Tensor]:
        if not cols:
            n = int(x.shape[0])
            return {
                "label": torch.full((n,), UNKNOWN_LABEL, dtype=torch.long, device=x.device),
                "score": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "margin": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "evidence_delta": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "anchor_delta": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "density_delta": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "pass": torch.zeros((n,), dtype=torch.bool, device=x.device),
            }
        group_scores = score_matrix[:, cols]
        best_score, best_pos = group_scores.max(dim=1)
        best_labels = torch.tensor(
            [head.class_order[int(cols[int(v)])] for v in best_pos.detach().cpu().tolist()],
            dtype=torch.long,
            device=x.device,
        )
        if score_matrix.size(1) > 1:
            competitor = score_matrix.detach().clone()
            for col in cols:
                competitor[:, col] = -float("inf")
            margin = best_score - competitor.max(dim=1).values
        else:
            margin = torch.ones_like(best_score)
        evidence_delta = torch.empty_like(best_score)
        anchor_delta = torch.empty_like(best_score)
        density_delta = torch.empty_like(best_score)
        passed = torch.zeros_like(best_score, dtype=torch.bool)
        for row, label in enumerate(best_labels.detach().cpu().tolist()):
            state = head.class_states[int(label)]
            row_values = per_class[int(label)]
            cos = row_values["cos"][row].detach()
            residual = row_values["residual"][row].detach()
            maha = row_values["maha"][row].detach()
            anchor = _support_anchor_similarity(x[row].view(1, -1), state)[0].detach()
            if group_name == "old":
                evidence = _old_support_evidence_score(cos, anchor, residual, maha)
                min_evidence = _state_threshold(state, "min_old_support_evidence", -float("inf"))
                min_anchor = _state_threshold(state, "min_old_support_anchor_similarity", -float("inf"))
                evidence_floor = float(old_min_evidence_delta)
                anchor_floor = float(old_min_anchor_delta)
                density_floor = float(old_min_density_delta)
            else:
                evidence = _seen_new_evidence_score(cos, residual, maha)
                min_evidence = _state_threshold(state, "min_seen_new_evidence", -float("inf"))
                min_anchor = _state_threshold(state, "min_seen_new_anchor_similarity", -float("inf"))
                evidence_floor = float(seen_new_min_evidence_delta)
                anchor_floor = float(seen_new_min_anchor_delta)
                density_floor = float(seen_new_min_density_delta)
            if "anchor_density" in row_values:
                density = row_values["anchor_density"][row].detach()
            else:
                density = torch.tensor(float("inf"), dtype=torch.float32, device=x.device)
            min_density = _state_threshold(state, "min_anchor_density", -float("inf"))
            evidence_delta[row] = float(evidence.item()) - float(min_evidence)
            anchor_delta[row] = float(anchor.item()) - float(min_anchor)
            density_delta[row] = float(density.item()) - float(min_density)
            passed[row] = (
                evidence_delta[row] >= evidence_floor
                and anchor_delta[row] >= anchor_floor
                and density_delta[row] >= density_floor
            )
        return {
            "label": best_labels,
            "score": best_score,
            "margin": margin,
            "evidence_delta": evidence_delta,
            "anchor_delta": anchor_delta,
            "density_delta": density_delta,
            "pass": passed,
        }

    old_shell = _best_shell(old_cols, "old")
    seen_shell = _best_shell(seen_cols, "seen_new")
    choose_seen = seen_shell["pass"] & (~old_shell["pass"] | (seen_shell["score"] >= old_shell["score"]))
    choose_old = old_shell["pass"] & ~choose_seen
    shell_accept = (choose_old | choose_seen) & (background_margin <= float(accept_background_margin))
    no_shell = ~(old_shell["pass"] | seen_shell["pass"])
    failed_shells = (old_shell["evidence_delta"] < float(old_min_evidence_delta)).long()
    failed_shells += (old_shell["density_delta"] < float(old_min_density_delta)).long()
    failed_shells += (seen_shell["evidence_delta"] < float(seen_new_min_evidence_delta)).long()
    failed_shells += (seen_shell["density_delta"] < float(seen_new_min_density_delta)).long()
    reject_mask = (
        no_shell
        & (failed_shells >= max(1, int(reject_min_failed_shells)))
        & (background_score >= float(reject_background_score))
        & (background_margin >= float(reject_background_margin))
    )

    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    chosen_label = torch.full_like(predicted, UNKNOWN_LABEL)
    chosen_label[choose_old.detach().cpu()] = old_shell["label"].detach().cpu()[choose_old.detach().cpu()]
    chosen_label[choose_seen.detach().cpu()] = seen_shell["label"].detach().cpu()[choose_seen.detach().cpu()]
    for row in torch.nonzero(shell_accept.detach().cpu(), as_tuple=False).reshape(-1).tolist():
        label = int(chosen_label[row].item())
        if label != UNKNOWN_LABEL:
            predicted[row] = label
            candidate_labels[row] = label
            accepted[row] = True
            decisions[row] = "accept"
            reasons[row] = "density_shell_inlier_accept"
    for row in torch.nonzero(reject_mask.detach().cpu(), as_tuple=False).reshape(-1).tolist():
        predicted[row] = UNKNOWN_LABEL
        accepted[row] = False
        decisions[row] = "reject"
        reasons[row] = "density_shell_open_space_reject"

    diagnostics = dict(result.diagnostics or {})
    diagnostics["density_shell_old_label"] = old_shell["label"].detach().cpu()
    diagnostics["density_shell_seen_new_label"] = seen_shell["label"].detach().cpu()
    diagnostics["density_shell_chosen_label"] = chosen_label.detach().cpu()
    diagnostics["density_shell_old_score"] = old_shell["score"].detach().cpu()
    diagnostics["density_shell_seen_new_score"] = seen_shell["score"].detach().cpu()
    diagnostics["density_shell_old_evidence_delta"] = old_shell["evidence_delta"].detach().cpu()
    diagnostics["density_shell_seen_new_evidence_delta"] = seen_shell["evidence_delta"].detach().cpu()
    diagnostics["density_shell_old_anchor_delta"] = old_shell["anchor_delta"].detach().cpu()
    diagnostics["density_shell_seen_new_anchor_delta"] = seen_shell["anchor_delta"].detach().cpu()
    diagnostics["density_shell_old_density_delta"] = old_shell["density_delta"].detach().cpu()
    diagnostics["density_shell_seen_new_density_delta"] = seen_shell["density_delta"].detach().cpu()
    diagnostics["density_shell_background_score"] = background_score.detach().cpu()
    diagnostics["density_shell_background_margin"] = background_margin.detach().cpu()
    diagnostics["density_shell_old_pass_mask"] = old_shell["pass"].detach().cpu()
    diagnostics["density_shell_seen_new_pass_mask"] = seen_shell["pass"].detach().cpu()
    diagnostics["density_shell_accept_mask"] = shell_accept.detach().cpu()
    diagnostics["density_shell_reject_mask"] = reject_mask.detach().cpu()
    diagnostics["density_shell_failed_shell_count"] = failed_shells.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_identity_consensus_arbitration(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    old_min_evidence_delta: float = -0.06,
    old_min_anchor_delta: float = -0.10,
    old_min_density_delta: float = -0.08,
    seen_new_min_evidence_delta: float = -0.04,
    seen_new_min_anchor_delta: float = -0.08,
    seen_new_min_density_delta: float = -0.06,
    min_identity_margin: float = -0.05,
    background_accept_margin: float = 0.22,
    reject_background_score: float = 0.90,
    reject_background_margin: float = 0.18,
    reject_min_identity_failures: int = 4,
    support_background_cap_enabled: bool = False,
    support_background_cap_quantile: float = 0.90,
    support_background_cap_slack: float = 0.05,
    support_background_cap_min_anchors: int = 2,
) -> PredictionResult:
    """Identity-first arbitration between old, seen-new, and open space.

    The arbitration is calibrated only from source plus allowed target support.
    Unknown query samples are never used. Unlike density-shell gating, the
    selected label is based on a fused identity score rather than raw head score,
    so old/seen-new evidence can override a high pseudo-background score only
    when class identity is internally consistent.
    """

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result

    score_matrix, per_class = head.score_matrix(x)
    old_cols = [idx for idx, label in enumerate(head.class_order) if str(head.class_states[int(label)].group) == "old"]
    seen_cols = [idx for idx, label in enumerate(head.class_order) if str(head.class_states[int(label)].group) == "seen_new"]
    if not old_cols and not seen_cols:
        return result

    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    background_available = bool(
        background.numel() > 0 and background.ndim == 2 and int(background.shape[1]) == int(x.shape[1])
    )
    if background_available:
        background = background.to(x.device)
        background_score = torch.max(x @ background.T, dim=1).values
    else:
        background_score = torch.full((int(x.shape[0]),), -float("inf"), dtype=torch.float32, device=x.device)

    known_rows = [
        torch.as_tensor(state.prototype).float()
        for state in head.class_states.values()
        if state is not None and torch.as_tensor(state.prototype).numel() == x.shape[1]
    ]
    if known_rows:
        known = normalize_rows(torch.stack(known_rows, dim=0)).to(x.device)
        known_score = torch.max(x @ known.T, dim=1).values
    else:
        known_score = torch.zeros_like(background_score)
    background_margin = background_score - known_score
    support_background_caps: dict[int, torch.Tensor] = {}
    if bool(support_background_cap_enabled) and background_available:
        q = max(0.0, min(1.0, float(support_background_cap_quantile)))
        min_anchors = max(1, int(support_background_cap_min_anchors))
        for label, state in head.class_states.items():
            anchors = state.support_anchors
            if anchors is None or torch.as_tensor(anchors).numel() == 0:
                anchors = torch.as_tensor(state.prototype).float().view(1, -1)
            anchors = normalize_rows(torch.as_tensor(anchors).float()).to(x.device)
            if anchors.ndim != 2 or anchors.size(1) != x.size(1) or anchors.size(0) < min_anchors:
                continue
            anchor_bg = torch.max(anchors @ background.T, dim=1).values
            support_background_caps[int(label)] = torch.quantile(anchor_bg, q) + float(support_background_cap_slack)

    def _best_identity(cols: list[int], group_name: str) -> dict[str, torch.Tensor]:
        n = int(x.shape[0])
        if not cols:
            return {
                "label": torch.full((n,), UNKNOWN_LABEL, dtype=torch.long, device=x.device),
                "head_score": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "identity_score": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "evidence_delta": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "anchor_delta": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "density_delta": torch.full((n,), -float("inf"), dtype=torch.float32, device=x.device),
                "pass": torch.zeros((n,), dtype=torch.bool, device=x.device),
            }
        group_scores = score_matrix[:, cols]
        best_score, best_pos = group_scores.max(dim=1)
        labels = torch.tensor(
            [head.class_order[int(cols[int(v)])] for v in best_pos.detach().cpu().tolist()],
            dtype=torch.long,
            device=x.device,
        )
        evidence_delta = torch.empty_like(best_score)
        anchor_delta = torch.empty_like(best_score)
        density_delta = torch.empty_like(best_score)
        identity_score = torch.empty_like(best_score)
        passed = torch.zeros_like(best_score, dtype=torch.bool)
        for row, label in enumerate(labels.detach().cpu().tolist()):
            state = head.class_states[int(label)]
            row_values = per_class[int(label)]
            cos = row_values["cos"][row].detach()
            residual = row_values["residual"][row].detach()
            maha = row_values["maha"][row].detach()
            anchor = _support_anchor_similarity(x[row].view(1, -1), state)[0].detach()
            if group_name == "old":
                evidence = _old_support_evidence_score(cos, anchor, residual, maha)
                min_evidence = _state_threshold(state, "min_old_support_evidence", -float("inf"))
                min_anchor = _state_threshold(state, "min_old_support_anchor_similarity", -float("inf"))
                evidence_floor = float(old_min_evidence_delta)
                anchor_floor = float(old_min_anchor_delta)
                density_floor = float(old_min_density_delta)
                density_weight = 0.25
            else:
                evidence = _seen_new_evidence_score(cos, residual, maha)
                min_evidence = _state_threshold(state, "min_seen_new_evidence", -float("inf"))
                min_anchor = _state_threshold(state, "min_seen_new_anchor_similarity", -float("inf"))
                evidence_floor = float(seen_new_min_evidence_delta)
                anchor_floor = float(seen_new_min_anchor_delta)
                density_floor = float(seen_new_min_density_delta)
                density_weight = 0.20
            if "anchor_density" in row_values:
                density = row_values["anchor_density"][row].detach()
            else:
                density = torch.tensor(float("inf"), dtype=torch.float32, device=x.device)
            min_density = _state_threshold(state, "min_anchor_density", -float("inf"))
            evidence_delta[row] = float(evidence.item()) - float(min_evidence)
            anchor_delta[row] = float(anchor.item()) - float(min_anchor)
            density_delta[row] = float(density.item()) - float(min_density)
            density_term = torch.clamp(density_delta[row], min=-0.35, max=0.35)
            identity_score[row] = (
                best_score[row]
                + 0.55 * torch.clamp(evidence_delta[row], min=-0.50, max=0.50)
                + 0.35 * torch.clamp(anchor_delta[row], min=-0.50, max=0.50)
                + density_weight * density_term
            )
            passed[row] = (
                evidence_delta[row] >= evidence_floor
                and anchor_delta[row] >= anchor_floor
                and density_delta[row] >= density_floor
            )
        return {
            "label": labels,
            "head_score": best_score,
            "identity_score": identity_score,
            "evidence_delta": evidence_delta,
            "anchor_delta": anchor_delta,
            "density_delta": density_delta,
            "pass": passed,
        }

    old_identity = _best_identity(old_cols, "old")
    seen_identity = _best_identity(seen_cols, "seen_new")
    identity_margin = old_identity["identity_score"] - seen_identity["identity_score"]
    choose_old = old_identity["pass"] & (
        ~seen_identity["pass"] | (identity_margin >= float(min_identity_margin))
    )
    choose_seen = seen_identity["pass"] & ~choose_old
    chosen_label = torch.full((int(x.shape[0]),), UNKNOWN_LABEL, dtype=torch.long, device=x.device)
    chosen_score = torch.full((int(x.shape[0]),), -float("inf"), dtype=torch.float32, device=x.device)
    chosen_label[choose_old] = old_identity["label"][choose_old]
    chosen_score[choose_old] = old_identity["identity_score"][choose_old]
    chosen_label[choose_seen] = seen_identity["label"][choose_seen]
    chosen_score[choose_seen] = seen_identity["identity_score"][choose_seen]
    identity_accept = (choose_old | choose_seen) & (background_margin <= float(background_accept_margin))
    chosen_background_cap = torch.full_like(background_score, float("inf"))
    if support_background_caps:
        for label, cap in support_background_caps.items():
            chosen_background_cap = torch.where(
                chosen_label == int(label),
                torch.full_like(chosen_background_cap, float(cap.detach().cpu().item())),
                chosen_background_cap,
            )
        identity_accept = identity_accept & (background_score <= chosen_background_cap)

    identity_failures = (old_identity["evidence_delta"] < float(old_min_evidence_delta)).long()
    identity_failures += (old_identity["anchor_delta"] < float(old_min_anchor_delta)).long()
    identity_failures += (old_identity["density_delta"] < float(old_min_density_delta)).long()
    identity_failures += (seen_identity["evidence_delta"] < float(seen_new_min_evidence_delta)).long()
    identity_failures += (seen_identity["anchor_delta"] < float(seen_new_min_anchor_delta)).long()
    identity_failures += (seen_identity["density_delta"] < float(seen_new_min_density_delta)).long()
    no_identity = ~(old_identity["pass"] | seen_identity["pass"])
    reject_mask = (
        no_identity
        & (identity_failures >= max(1, int(reject_min_identity_failures)))
        & (background_score >= float(reject_background_score))
        & (background_margin >= float(reject_background_margin))
    )

    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    for row in torch.nonzero(identity_accept.detach().cpu(), as_tuple=False).reshape(-1).tolist():
        label = int(chosen_label[row].item())
        if label != UNKNOWN_LABEL:
            predicted[row] = label
            candidate_labels[row] = label
            accepted[row] = True
            decisions[row] = "accept"
            reasons[row] = "identity_consensus_accept"
    for row in torch.nonzero(reject_mask.detach().cpu(), as_tuple=False).reshape(-1).tolist():
        predicted[row] = UNKNOWN_LABEL
        accepted[row] = False
        decisions[row] = "reject"
        reasons[row] = "identity_consensus_open_space_reject"

    diagnostics = dict(result.diagnostics or {})
    diagnostics["identity_consensus_old_label"] = old_identity["label"].detach().cpu()
    diagnostics["identity_consensus_seen_new_label"] = seen_identity["label"].detach().cpu()
    diagnostics["identity_consensus_chosen_label"] = chosen_label.detach().cpu()
    diagnostics["identity_consensus_old_score"] = old_identity["identity_score"].detach().cpu()
    diagnostics["identity_consensus_seen_new_score"] = seen_identity["identity_score"].detach().cpu()
    diagnostics["identity_consensus_chosen_score"] = chosen_score.detach().cpu()
    diagnostics["identity_consensus_margin"] = identity_margin.detach().cpu()
    diagnostics["identity_consensus_old_evidence_delta"] = old_identity["evidence_delta"].detach().cpu()
    diagnostics["identity_consensus_seen_new_evidence_delta"] = seen_identity["evidence_delta"].detach().cpu()
    diagnostics["identity_consensus_old_anchor_delta"] = old_identity["anchor_delta"].detach().cpu()
    diagnostics["identity_consensus_seen_new_anchor_delta"] = seen_identity["anchor_delta"].detach().cpu()
    diagnostics["identity_consensus_old_density_delta"] = old_identity["density_delta"].detach().cpu()
    diagnostics["identity_consensus_seen_new_density_delta"] = seen_identity["density_delta"].detach().cpu()
    diagnostics["identity_consensus_background_score"] = background_score.detach().cpu()
    diagnostics["identity_consensus_background_margin"] = background_margin.detach().cpu()
    diagnostics["identity_consensus_support_background_cap"] = chosen_background_cap.detach().cpu()
    diagnostics["identity_consensus_support_background_cap_pass_mask"] = (background_score <= chosen_background_cap).detach().cpu()
    diagnostics["identity_consensus_old_pass_mask"] = old_identity["pass"].detach().cpu()
    diagnostics["identity_consensus_seen_new_pass_mask"] = seen_identity["pass"].detach().cpu()
    diagnostics["identity_consensus_accept_mask"] = identity_accept.detach().cpu()
    diagnostics["identity_consensus_reject_mask"] = reject_mask.detach().cpu()
    diagnostics["identity_consensus_failure_count"] = identity_failures.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_support_conformal_arbitration(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    calibration_quantile: float = 0.05,
    conformity_slack: float = 0.12,
    anchor_margin_slack: float = 0.06,
    background_score: float = 0.82,
    background_margin: float = 0.08,
    hard_reject_margin: float = 0.18,
    reject_min_failures: int = 2,
    reject_action: str = "reject",
) -> PredictionResult:
    """Class-conditional support-conformal veto for accepted known predictions.

    The calibration set is the support/source geometry already allowed by the
    Stage2 protocol. It does not use unknown query labels. The gate asks whether
    the predicted class can explain the sample under its own support geometry;
    background evidence is only used to decide whether a local conformity failure
    should become an open-set rejection/defer.
    """

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result

    score_matrix, per_class = head.score_matrix(x)
    label_to_col = {int(label): idx for idx, label in enumerate(head.class_order)}
    all_anchor_rows = []
    for state in head.class_states.values():
        anchors = state.support_anchors
        if anchors is not None and hasattr(anchors, "numel") and int(anchors.numel()) > 0:
            all_anchor_rows.append(torch.as_tensor(anchors).float())
        else:
            all_anchor_rows.append(torch.as_tensor(state.prototype).float().view(1, -1))
    all_anchors = normalize_rows(torch.cat(all_anchor_rows, dim=0)).to(x.device) if all_anchor_rows else None

    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    background_available = bool(
        background.numel() > 0 and background.ndim == 2 and int(background.shape[1]) == int(x.shape[1])
    )
    if background_available:
        background = background.to(x.device)
        bg_score = torch.max(x @ background.T, dim=1).values
    else:
        bg_score = torch.full((int(x.shape[0]),), -float("inf"), dtype=torch.float32, device=x.device)
    known_rows = [torch.as_tensor(state.prototype).float() for state in head.class_states.values()]
    known = normalize_rows(torch.stack(known_rows, dim=0)).to(x.device) if known_rows else None
    known_score = torch.max(x @ known.T, dim=1).values if known is not None else torch.zeros_like(bg_score)
    bg_margin = bg_score - known_score

    def _class_conformity(rows: torch.Tensor, label: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = normalize_rows(torch.as_tensor(rows).float()).to(x.device)
        if rows.ndim == 1:
            rows = rows.view(1, -1)
        state = head.class_states[int(label)]
        scores, values_by_class = head.score_matrix(rows)
        col = label_to_col[int(label)]
        raw_score = scores[:, col]
        row_values = values_by_class[int(label)]
        cos = row_values["cos"]
        residual = row_values["residual"]
        maha = row_values["maha"]
        anchor = _support_anchor_similarity(rows, state).to(x.device)
        if str(state.group) == "old":
            evidence = _old_support_evidence_score(cos, anchor, residual, maha)
            min_evidence = _state_threshold(state, "min_old_support_evidence", -float("inf"))
            min_anchor = _state_threshold(state, "min_old_support_anchor_similarity", -float("inf"))
            evidence_weight = 0.48
            anchor_weight = 0.34
        else:
            evidence = _seen_new_evidence_score(cos, residual, maha)
            min_evidence = _state_threshold(state, "min_seen_new_evidence", -float("inf"))
            min_anchor = _state_threshold(state, "min_seen_new_anchor_similarity", -float("inf"))
            evidence_weight = 0.42
            anchor_weight = 0.30
        if "anchor_density" in row_values:
            density = row_values["anchor_density"].to(x.device)
        else:
            density = torch.full_like(raw_score, float("inf"))
        min_density = _state_threshold(state, "min_anchor_density", -float("inf"))
        if all_anchors is not None:
            own = anchor
            all_sim = rows @ all_anchors.T
            other = torch.topk(all_sim, k=min(2, int(all_sim.shape[1])), dim=1).values
            other_best = other[:, -1] if int(other.shape[1]) > 1 else torch.zeros_like(own)
            anchor_margin = own - other_best
        else:
            anchor_margin = torch.zeros_like(raw_score)
        evidence_delta = torch.as_tensor(evidence - float(min_evidence), dtype=torch.float32, device=x.device)
        anchor_delta = torch.as_tensor(anchor - float(min_anchor), dtype=torch.float32, device=x.device)
        density_delta = torch.as_tensor(density - float(min_density), dtype=torch.float32, device=x.device)
        conformity = (
            raw_score
            + evidence_weight * torch.clamp(evidence_delta, min=-0.60, max=0.60)
            + anchor_weight * torch.clamp(anchor_delta, min=-0.60, max=0.60)
            + 0.18 * torch.clamp(density_delta, min=-0.40, max=0.40)
            + 0.18 * torch.clamp(anchor_margin, min=-0.40, max=0.40)
        )
        return conformity, anchor_margin

    floors: dict[int, float] = {}
    anchor_floors: dict[int, float] = {}
    q = min(max(float(calibration_quantile), 0.0), 1.0)
    for label, state in head.class_states.items():
        anchors = state.support_anchors
        if anchors is not None and hasattr(anchors, "numel") and int(anchors.numel()) > 0:
            cal_rows = torch.as_tensor(anchors).float()
            cal_rows = torch.cat([cal_rows, torch.as_tensor(state.prototype).float().view(1, -1)], dim=0)
        else:
            cal_rows = torch.as_tensor(state.prototype).float().view(1, -1)
        conformity, anchor_margin_values = _class_conformity(cal_rows, int(label))
        floors[int(label)] = float(torch.quantile(conformity.detach().float().cpu(), q).item()) - float(conformity_slack)
        anchor_floors[int(label)] = (
            float(torch.quantile(anchor_margin_values.detach().float().cpu(), q).item()) - float(anchor_margin_slack)
        )

    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    n = int(x.shape[0])
    conf_label = torch.full((n,), UNKNOWN_LABEL, dtype=torch.long, device=x.device)
    conf_score = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    conf_floor = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    conf_margin = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    anchor_margin_values = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    anchor_margin_floor = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    passed = torch.zeros((n,), dtype=torch.bool, device=x.device)
    reject_mask = torch.zeros((n,), dtype=torch.bool, device=x.device)
    failure_count = torch.zeros((n,), dtype=torch.long, device=x.device)

    for row in range(n):
        label = int(predicted[row].item())
        if label == UNKNOWN_LABEL or label not in head.class_states:
            continue
        state = head.class_states[label]
        if str(state.group) not in {"old", "seen_new"}:
            continue
        conformity, row_anchor_margin = _class_conformity(x[row].view(1, -1), label)
        conf_label[row] = label
        conf_score[row] = conformity[0]
        conf_floor[row] = float(floors[label])
        conf_margin[row] = conformity[0] - float(floors[label])
        anchor_margin_values[row] = row_anchor_margin[0]
        anchor_margin_floor[row] = float(anchor_floors[label])
        local_failures = int(conf_margin[row].item() < 0.0)
        local_failures += int(anchor_margin_values[row].item() < float(anchor_floors[label]))
        local_failures += int(bg_score[row].item() >= float(background_score) and bg_margin[row].item() >= float(background_margin))
        failure_count[row] = int(local_failures)
        passed[row] = local_failures == 0
        severe_local_fail = conf_margin[row].item() <= -abs(float(hard_reject_margin))
        background_risk = bg_score[row].item() >= float(background_score) and bg_margin[row].item() >= float(background_margin)
        if bool(accepted[row].item()) and local_failures >= max(1, int(reject_min_failures)) and (background_risk or severe_local_fail):
            reject_mask[row] = True

    action = "defer" if str(reject_action).lower() == "defer" else "reject"
    for row in torch.nonzero(reject_mask.detach().cpu(), as_tuple=False).reshape(-1).tolist():
        predicted[row] = UNKNOWN_LABEL
        accepted[row] = False
        decisions[row] = action
        reasons[row] = "support_conformal_open_space_reject" if action == "reject" else "support_conformal_defer"

    diagnostics = dict(result.diagnostics or {})
    diagnostics["support_conformal_label"] = conf_label.detach().cpu()
    diagnostics["support_conformal_score"] = conf_score.detach().cpu()
    diagnostics["support_conformal_floor"] = conf_floor.detach().cpu()
    diagnostics["support_conformal_margin"] = conf_margin.detach().cpu()
    diagnostics["support_conformal_anchor_margin"] = anchor_margin_values.detach().cpu()
    diagnostics["support_conformal_anchor_margin_floor"] = anchor_margin_floor.detach().cpu()
    diagnostics["support_conformal_background_score"] = bg_score.detach().cpu()
    diagnostics["support_conformal_background_margin"] = bg_margin.detach().cpu()
    diagnostics["support_conformal_pass_mask"] = passed.detach().cpu()
    diagnostics["support_conformal_reject_mask"] = reject_mask.detach().cpu()
    diagnostics["support_conformal_failure_count"] = failure_count.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def _class_reconstruction_profile(
    rows: torch.Tensor,
    anchors: torch.Tensor,
    *,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = normalize_rows(torch.as_tensor(rows).float())
    anchors = normalize_rows(torch.as_tensor(anchors).float()).to(rows.device)
    if rows.ndim == 1:
        rows = rows.view(1, -1)
    if anchors.ndim == 1:
        anchors = anchors.view(1, -1)
    center = normalize_rows(anchors.mean(dim=0, keepdim=True)).squeeze(0)
    centered = anchors - center
    max_rank = max(0, min(int(rank), int(anchors.shape[0]) - 1, int(rows.shape[1])))
    if max_rank <= 0 or centered.numel() == 0:
        reconstructed = center.view(1, -1).expand_as(rows)
    else:
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            basis = vh[:max_rank].T.contiguous()
            delta = rows - center
            reconstructed = center + (delta @ basis) @ basis.T
        except RuntimeError:
            reconstructed = center.view(1, -1).expand_as(rows)
    residual = torch.linalg.vector_norm(rows - reconstructed, ord=2, dim=1)
    center_cos = rows @ center.view(-1, 1)
    return residual, center_cos.squeeze(1), center


def apply_support_reconstruction_arbitration(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    rank: int = 2,
    residual_quantile: float = 0.95,
    residual_slack: float = 0.04,
    min_residual_floor: float = 0.03,
    negative_scale: float = 0.55,
    negative_margin: float = -0.02,
    hard_residual_margin: float = 0.08,
    background_score: float = 0.86,
    background_margin: float = 0.12,
    reject_min_failures: int = 2,
    reject_action: str = "reject",
) -> PredictionResult:
    """Support-only class reconstruction veto for accepted known predictions."""

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result

    class_banks: dict[int, torch.Tensor] = {}
    class_centers: dict[int, torch.Tensor] = {}
    residual_ceilings: dict[int, float] = {}
    q = min(max(float(residual_quantile), 0.0), 1.0)
    for label, state in head.class_states.items():
        anchors = state.support_anchors
        rows = []
        if anchors is not None and torch.is_tensor(anchors) and int(anchors.numel()) > 0:
            rows.append(torch.as_tensor(anchors).float())
        rows.append(torch.as_tensor(state.prototype).float().view(1, -1))
        bank = normalize_rows(torch.cat(rows, dim=0)).to(x.device)
        class_banks[int(label)] = bank
        cal_residual, _, center = _class_reconstruction_profile(bank, bank, rank=int(rank))
        class_centers[int(label)] = center.detach()
        residual_ceilings[int(label)] = max(
            float(min_residual_floor),
            float(torch.quantile(cal_residual.detach().float().cpu(), q).item()) + float(residual_slack),
        )

    all_center_rows = []
    for center in class_centers.values():
        all_center_rows.append(center.view(1, -1))
    all_centers = normalize_rows(torch.cat(all_center_rows, dim=0)).to(x.device) if all_center_rows else None

    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    background_available = bool(
        background.numel() > 0 and background.ndim == 2 and int(background.shape[1]) == int(x.shape[1])
    )
    if background_available:
        background = background.to(x.device)
        bg_score = torch.max(x @ background.T, dim=1).values
    else:
        bg_score = torch.full((int(x.shape[0]),), -float("inf"), dtype=torch.float32, device=x.device)
    known_score = torch.max(x @ all_centers.T, dim=1).values if all_centers is not None else torch.zeros_like(bg_score)
    bg_margin = bg_score - known_score

    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    n = int(x.shape[0])
    recon_label = torch.full((n,), UNKNOWN_LABEL, dtype=torch.long, device=x.device)
    residual_value = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    residual_ceiling = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    residual_margin = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    center_cosine = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    negative_score = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    negative_margin_value = torch.full((n,), float("nan"), dtype=torch.float32, device=x.device)
    passed = torch.zeros((n,), dtype=torch.bool, device=x.device)
    reject_mask = torch.zeros((n,), dtype=torch.bool, device=x.device)
    failure_count = torch.zeros((n,), dtype=torch.long, device=x.device)

    for row in range(n):
        label = int(predicted[row].item())
        if label == UNKNOWN_LABEL or label not in head.class_states or label not in class_banks:
            continue
        state = head.class_states[label]
        if str(state.group) not in {"old", "seen_new"}:
            continue
        residual, center_cos, own_center = _class_reconstruction_profile(
            x[row].view(1, -1),
            class_banks[label],
            rank=int(rank),
        )
        recon_label[row] = label
        residual_value[row] = residual[0]
        center_cosine[row] = center_cos[0]
        residual_ceiling[row] = float(residual_ceilings[label])
        residual_margin[row] = float(residual_ceilings[label]) - residual[0]

        other_centers = [center for other_label, center in class_centers.items() if int(other_label) != label]
        if other_centers:
            center = normalize_rows(own_center.view(1, -1)).squeeze(0)
            other = normalize_rows(torch.stack(other_centers, dim=0)).to(x.device)
            midpoint = normalize_rows((1.0 - float(negative_scale)) * center.view(1, -1) + float(negative_scale) * other)
            reciprocal = normalize_rows(center.view(1, -1) + float(negative_scale) * (center.view(1, -1) - other))
            neg_bank = normalize_rows(torch.cat([midpoint, reciprocal], dim=0))
            neg_score = torch.max(x[row].view(1, -1) @ neg_bank.T, dim=1).values[0]
        else:
            neg_score = torch.tensor(-float("inf"), dtype=torch.float32, device=x.device)
        own_anchor = _support_anchor_similarity(x[row].view(1, -1), state).to(x.device)[0]
        negative_score[row] = neg_score
        negative_margin_value[row] = neg_score - own_anchor

        local_failures = int(residual_margin[row].item() < 0.0)
        local_failures += int(negative_margin_value[row].item() >= float(negative_margin))
        local_failures += int(bg_score[row].item() >= float(background_score) and bg_margin[row].item() >= float(background_margin))
        failure_count[row] = int(local_failures)
        passed[row] = local_failures == 0
        severe_residual = residual_margin[row].item() <= -abs(float(hard_residual_margin))
        negative_risk = negative_margin_value[row].item() >= float(negative_margin)
        background_risk = bg_score[row].item() >= float(background_score) and bg_margin[row].item() >= float(background_margin)
        if bool(accepted[row].item()) and local_failures >= max(1, int(reject_min_failures)) and (
            severe_residual or negative_risk or background_risk
        ):
            reject_mask[row] = True

    action = "defer" if str(reject_action).lower() == "defer" else "reject"
    for row in torch.nonzero(reject_mask.detach().cpu(), as_tuple=False).reshape(-1).tolist():
        predicted[row] = UNKNOWN_LABEL
        accepted[row] = False
        decisions[row] = action
        reasons[row] = "support_reconstruction_open_space_reject" if action == "reject" else "support_reconstruction_defer"

    diagnostics = dict(result.diagnostics or {})
    diagnostics["support_reconstruction_label"] = recon_label.detach().cpu()
    diagnostics["support_reconstruction_residual"] = residual_value.detach().cpu()
    diagnostics["support_reconstruction_residual_ceiling"] = residual_ceiling.detach().cpu()
    diagnostics["support_reconstruction_residual_margin"] = residual_margin.detach().cpu()
    diagnostics["support_reconstruction_center_cosine"] = center_cosine.detach().cpu()
    diagnostics["support_reconstruction_negative_score"] = negative_score.detach().cpu()
    diagnostics["support_reconstruction_negative_margin"] = negative_margin_value.detach().cpu()
    diagnostics["support_reconstruction_background_score"] = bg_score.detach().cpu()
    diagnostics["support_reconstruction_background_margin"] = bg_margin.detach().cpu()
    diagnostics["support_reconstruction_pass_mask"] = passed.detach().cpu()
    diagnostics["support_reconstruction_reject_mask"] = reject_mask.detach().cpu()
    diagnostics["support_reconstruction_failure_count"] = failure_count.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_two_branch_background_guard(
    features: torch.Tensor,
    result: PredictionResult,
    class_states: Mapping[int, ClassState],
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    min_background_score: float = 0.62,
    min_background_margin: float = -0.02,
    old_support_evidence_delta: float = 0.0,
    old_support_anchor_delta: float = -0.02,
    old_support_anchor_margin: float = 0.0,
    seen_new_evidence_delta: float = 0.0,
    seen_new_anchor_delta: float = 0.0,
) -> PredictionResult:
    """Veto accepted rows only when background risk is high and support evidence is weak."""

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    if x.numel() == 0 or background.numel() == 0 or x.ndim != 2 or background.ndim != 2:
        return result
    known_rows = [
        torch.as_tensor(state.prototype).float()
        for state in class_states.values()
        if state is not None and torch.as_tensor(state.prototype).numel() == x.shape[1]
    ]
    if not known_rows:
        return result
    known = normalize_rows(torch.stack(known_rows, dim=0)).to(x.device)
    background = background.to(x.device)
    background_score = torch.max(x @ background.T, dim=1).values
    known_score = torch.max(x @ known.T, dim=1).values
    background_margin = background_score - known_score
    background_risk = (background_score >= float(min_background_score)) & (
        background_margin >= float(min_background_margin)
    )

    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    def diagnostic_tensor(name: str) -> torch.Tensor:
        value = (result.diagnostics or {}).get(name)
        if value is None or not torch.is_tensor(value):
            return torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
        flat = value.detach().cpu().float().reshape(-1)
        if int(flat.numel()) >= int(predicted.numel()):
            return flat[: int(predicted.numel())]
        padded = torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
        padded[: int(flat.numel())] = flat
        return padded

    old_evidence_delta = diagnostic_tensor("old_support_evidence_delta")
    old_anchor_delta = diagnostic_tensor("old_support_anchor_delta")
    old_anchor_margin_diag = diagnostic_tensor("old_support_anchor_margin")
    seen_evidence_delta = diagnostic_tensor("seen_new_evidence_delta")
    seen_anchor_delta = diagnostic_tensor("seen_new_anchor_delta")
    support_override = torch.zeros_like(accepted, dtype=torch.bool)
    for row, label in enumerate(predicted.tolist()):
        state = class_states.get(int(label))
        if state is None:
            continue
        if str(state.group) == "old":
            support_override[row] = (
                float(old_evidence_delta[row].item()) >= float(old_support_evidence_delta)
                and float(old_anchor_delta[row].item()) >= float(old_support_anchor_delta)
                and float(old_anchor_margin_diag[row].item()) >= float(old_support_anchor_margin)
            )
        elif str(state.group) == "seen_new":
            support_override[row] = (
                float(seen_evidence_delta[row].item()) >= float(seen_new_evidence_delta)
                and float(seen_anchor_delta[row].item()) >= float(seen_new_anchor_delta)
            )

    reject_mask = accepted & background_risk.detach().cpu() & ~support_override
    for row in torch.nonzero(reject_mask, as_tuple=False).reshape(-1).tolist():
        predicted[row] = UNKNOWN_LABEL
        accepted[row] = False
        decisions[row] = "reject"
        reasons[row] = "two_branch_background_guard_reject"

    diagnostics = dict(result.diagnostics or {})
    diagnostics["two_branch_background_score"] = background_score.detach().cpu()
    diagnostics["two_branch_known_score"] = known_score.detach().cpu()
    diagnostics["two_branch_background_margin"] = background_margin.detach().cpu()
    diagnostics["two_branch_background_risk_mask"] = background_risk.detach().cpu()
    diagnostics["two_branch_support_override_mask"] = support_override.detach().cpu()
    diagnostics["two_branch_background_reject_mask"] = reject_mask.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_seen_new_registration_override(
    features: torch.Tensor,
    result: PredictionResult,
    head: OrbitAdaptiveMSEHead,
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    min_evidence_delta: float = 0.0,
    min_anchor_delta: float = 0.0,
    min_affinity_delta: float = -0.02,
    min_residual_delta: float = -0.02,
    min_score_margin: float = -0.10,
    min_seen_vs_old_evidence_margin: float = 0.02,
    max_background_score: float = 0.72,
    max_background_margin: float = 0.08,
    min_support_knn_seen_new_minus_old: float | None = None,
    min_support_knn_margin: float | None = None,
) -> PredictionResult:
    """Accept strong seen-new support evidence before old/unknown veto stages.

    The override is calibrated only from registered seen-new support geometry,
    old support/source evidence, and query-free pseudo-background anchors. It
    never fits thresholds from unknown query labels.
    """

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    if x.numel() == 0 or x.ndim != 2:
        return result

    score_matrix, per_class = head.score_matrix(x)
    group_by_label = {int(label): str(head.class_states[int(label)].group) for label in head.class_order}
    seen_labels = [int(label) for label in head.class_order if group_by_label[int(label)] == "seen_new"]
    old_labels = [int(label) for label in head.class_order if group_by_label[int(label)] == "old"]
    if not seen_labels:
        return result

    def _threshold_delta(value: torch.Tensor, threshold: float) -> torch.Tensor:
        return value.detach() - float(threshold)

    seen_evidence_cols = []
    seen_anchor_delta_cols = []
    seen_affinity_delta_cols = []
    seen_residual_delta_cols = []
    seen_score_cols = []
    for label in seen_labels:
        state = head.class_states[int(label)]
        cos = per_class[int(label)]["cos"].detach()
        residual = per_class[int(label)]["residual"].detach()
        maha = per_class[int(label)]["maha"].detach()
        anchor = _support_anchor_similarity(x, state).detach()
        evidence = _seen_new_evidence_score(cos, residual, maha)
        min_affinity = _state_threshold(state, "min_seen_new_support_affinity", -1.0)
        max_residual = _state_threshold(state, "max_seen_new_support_residual", 1.0)
        min_evidence = _state_threshold(state, "min_seen_new_evidence", -1.0)
        min_anchor = _state_threshold(state, "min_seen_new_anchor_similarity", -1.0)
        seen_evidence_cols.append(_threshold_delta(evidence, min_evidence))
        seen_anchor_delta_cols.append(_threshold_delta(anchor, min_anchor))
        seen_affinity_delta_cols.append(_threshold_delta(cos, min_affinity))
        seen_residual_delta_cols.append(float(max_residual) - residual.detach())
        seen_score_cols.append(score_matrix[:, head.class_order.index(int(label))].detach())

    seen_evidence_delta_matrix = torch.stack(seen_evidence_cols, dim=1)
    best_seen_idx = torch.argmax(seen_evidence_delta_matrix, dim=1)
    best_seen_labels = torch.tensor(
        [seen_labels[int(idx)] for idx in best_seen_idx.detach().cpu().tolist()],
        dtype=torch.long,
        device=x.device,
    )
    seen_evidence_delta = seen_evidence_delta_matrix.gather(1, best_seen_idx.view(-1, 1)).squeeze(1)
    seen_anchor_delta = torch.stack(seen_anchor_delta_cols, dim=1).gather(1, best_seen_idx.view(-1, 1)).squeeze(1)
    seen_affinity_delta = torch.stack(seen_affinity_delta_cols, dim=1).gather(1, best_seen_idx.view(-1, 1)).squeeze(1)
    seen_residual_delta = torch.stack(seen_residual_delta_cols, dim=1).gather(1, best_seen_idx.view(-1, 1)).squeeze(1)
    seen_score = torch.stack(seen_score_cols, dim=1).gather(1, best_seen_idx.view(-1, 1)).squeeze(1)

    if old_labels:
        old_evidence_delta_cols = []
        old_score_cols = []
        for label in old_labels:
            state = head.class_states[int(label)]
            cos = per_class[int(label)]["cos"].detach()
            residual = per_class[int(label)]["residual"].detach()
            maha = per_class[int(label)]["maha"].detach()
            anchor = _support_anchor_similarity(x, state).detach()
            evidence = _old_support_evidence_score(cos, anchor, residual, maha)
            min_old_evidence = _state_threshold(state, "min_old_support_evidence", -1.0)
            old_evidence_delta_cols.append(_threshold_delta(evidence, min_old_evidence))
            old_score_cols.append(score_matrix[:, head.class_order.index(int(label))].detach())
        best_old_evidence_delta = torch.stack(old_evidence_delta_cols, dim=1).max(dim=1).values
        best_old_score = torch.stack(old_score_cols, dim=1).max(dim=1).values
    else:
        best_old_evidence_delta = torch.full_like(seen_evidence_delta, -float("inf"))
        best_old_score = torch.full_like(seen_score, -float("inf"))

    background_score = torch.full_like(seen_score, -float("inf"))
    background_margin = torch.full_like(seen_score, -float("inf"))
    background_risk = torch.zeros_like(seen_score, dtype=torch.bool)
    background = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    if background.numel() > 0 and background.ndim == 2 and background.shape[1] == x.shape[1]:
        background = background.to(x.device)
        background_score = torch.max(x @ background.T, dim=1).values
        known_proto = normalize_rows(
            torch.stack([torch.as_tensor(state.prototype).float() for state in head.class_states.values()], dim=0)
        ).to(x.device)
        known_score = torch.max(x @ known_proto.T, dim=1).values
        background_margin = background_score - known_score
        background_risk = (background_score >= float(max_background_score)) & (
            background_margin >= float(max_background_margin)
        )

    seen_vs_old_evidence = seen_evidence_delta - best_old_evidence_delta
    seen_vs_old_score = seen_score - best_old_score
    diagnostics_source = dict(result.diagnostics or {})

    def _diagnostic_vector(name: str, default: float) -> torch.Tensor:
        value = diagnostics_source.get(name)
        if value is None or not torch.is_tensor(value):
            return torch.full_like(seen_score, float(default))
        flat = value.detach().float().reshape(-1).to(seen_score.device)
        if int(flat.numel()) >= int(seen_score.numel()):
            return flat[: int(seen_score.numel())]
        padded = torch.full_like(seen_score, float(default))
        padded[: int(flat.numel())] = flat
        return padded

    support_knn_seen_new_minus_old = _diagnostic_vector("support_knn_seen_new_minus_old", -float("inf"))
    support_knn_margin = _diagnostic_vector("support_knn_margin", float("inf"))
    support_knn_ok = torch.ones_like(seen_score, dtype=torch.bool)
    if min_support_knn_seen_new_minus_old is not None:
        support_knn_ok &= support_knn_seen_new_minus_old >= float(min_support_knn_seen_new_minus_old)
    if min_support_knn_margin is not None:
        support_knn_ok &= support_knn_margin >= float(min_support_knn_margin)

    override_mask = (
        (seen_evidence_delta >= float(min_evidence_delta))
        & (seen_anchor_delta >= float(min_anchor_delta))
        & (seen_affinity_delta >= float(min_affinity_delta))
        & (seen_residual_delta >= float(min_residual_delta))
        & (seen_vs_old_score >= float(min_score_margin))
        & (seen_vs_old_evidence >= float(min_seen_vs_old_evidence_margin))
        & support_knn_ok
        & ~background_risk
    )

    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    candidate_labels = (
        result.candidate_labels.clone().long()
        if result.candidate_labels is not None
        else predicted.clone().long()
    )
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    for row in torch.nonzero(override_mask.detach().cpu(), as_tuple=False).reshape(-1).tolist():
        label = int(best_seen_labels[row].item())
        predicted[row] = label
        candidate_labels[row] = label
        accepted[row] = True
        decisions[row] = "accept"
        reasons[row] = "seen_new_registration_override"

    diagnostics = diagnostics_source
    diagnostics["seen_new_override_label"] = best_seen_labels.detach().cpu()
    diagnostics["seen_new_override_evidence_delta"] = seen_evidence_delta.detach().cpu()
    diagnostics["seen_new_override_anchor_delta"] = seen_anchor_delta.detach().cpu()
    diagnostics["seen_new_override_affinity_delta"] = seen_affinity_delta.detach().cpu()
    diagnostics["seen_new_override_residual_delta"] = seen_residual_delta.detach().cpu()
    diagnostics["seen_new_override_seen_minus_old_evidence"] = seen_vs_old_evidence.detach().cpu()
    diagnostics["seen_new_override_seen_minus_old_score"] = seen_vs_old_score.detach().cpu()
    diagnostics["seen_new_override_support_knn_seen_new_minus_old"] = support_knn_seen_new_minus_old.detach().cpu()
    diagnostics["seen_new_override_support_knn_margin"] = support_knn_margin.detach().cpu()
    diagnostics["seen_new_override_support_knn_pass_mask"] = support_knn_ok.detach().cpu()
    diagnostics["seen_new_override_background_score"] = background_score.detach().cpu()
    diagnostics["seen_new_override_background_margin"] = background_margin.detach().cpu()
    diagnostics["seen_new_override_background_risk_mask"] = background_risk.detach().cpu()
    diagnostics["seen_new_registration_override_mask"] = override_mask.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        candidate_labels=candidate_labels.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def apply_pseudo_unknown_void_gate(
    features: torch.Tensor,
    result: PredictionResult,
    class_states: Mapping[int, ClassState],
    pseudo_unknown: torch.Tensor,
    *,
    enabled: bool = False,
    min_void_score: float = 0.55,
    min_void_margin: float = 0.05,
    old_support_evidence_delta: float | None = None,
    old_support_anchor_delta: float | None = None,
    old_support_anchor_margin: float | None = None,
    seen_new_evidence_delta: float | None = None,
    seen_new_anchor_delta: float | None = None,
) -> PredictionResult:
    """Reject accepted rows that are closer to protocol-safe pseudo-unknown anchors than known prototypes."""

    if not bool(enabled):
        return result
    x = normalize_rows(torch.as_tensor(features).float())
    void = normalize_rows(torch.as_tensor(pseudo_unknown).float())
    if x.numel() == 0 or void.numel() == 0 or x.ndim != 2 or void.ndim != 2:
        return result
    known_rows = [
        torch.as_tensor(state.prototype).float()
        for state in class_states.values()
        if state is not None and torch.as_tensor(state.prototype).numel() == x.shape[1]
    ]
    if not known_rows:
        return result
    known = normalize_rows(torch.stack(known_rows, dim=0)).to(x.device)
    void = void.to(x.device)
    void_score = torch.max(x @ void.T, dim=1).values
    known_score = torch.max(x @ known.T, dim=1).values
    void_margin = void_score - known_score

    predicted = result.predicted_labels.clone().long()
    accepted = result.accepted.clone().bool()
    decisions = list(result.decisions)
    reasons = list(result.gate_reasons)
    if len(decisions) != int(predicted.numel()):
        decisions = ["accept" if bool(v) else "reject" for v in accepted.tolist()]
    if len(reasons) != int(predicted.numel()):
        reasons = [""] * int(predicted.numel())

    def diagnostic_tensor(name: str) -> torch.Tensor:
        value = (result.diagnostics or {}).get(name)
        if value is None or not torch.is_tensor(value):
            return torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
        flat = value.detach().cpu().float().reshape(-1)
        if int(flat.numel()) >= int(predicted.numel()):
            return flat[: int(predicted.numel())]
        padded = torch.full((int(predicted.numel()),), float("nan"), dtype=torch.float32)
        padded[: int(flat.numel())] = flat
        return padded

    old_evidence_delta = diagnostic_tensor("old_support_evidence_delta")
    old_anchor_delta_diag = diagnostic_tensor("old_support_anchor_delta")
    old_anchor_margin_diag = diagnostic_tensor("old_support_anchor_margin")
    seen_evidence_delta = diagnostic_tensor("seen_new_evidence_delta")
    seen_anchor_delta_diag = diagnostic_tensor("seen_new_anchor_delta")
    support_override = torch.zeros_like(accepted, dtype=torch.bool)
    for row, label in enumerate(predicted.tolist()):
        state = class_states.get(int(label))
        if state is None:
            continue
        if str(state.group) == "old":
            checks = []
            if old_support_evidence_delta is not None:
                checks.append(float(old_evidence_delta[row].item()) >= float(old_support_evidence_delta))
            if old_support_anchor_delta is not None:
                checks.append(float(old_anchor_delta_diag[row].item()) >= float(old_support_anchor_delta))
            if old_support_anchor_margin is not None:
                checks.append(float(old_anchor_margin_diag[row].item()) >= float(old_support_anchor_margin))
            support_override[row] = bool(checks) and all(checks)
        elif str(state.group) == "seen_new":
            checks = []
            if seen_new_evidence_delta is not None:
                checks.append(float(seen_evidence_delta[row].item()) >= float(seen_new_evidence_delta))
            if seen_new_anchor_delta is not None:
                checks.append(float(seen_anchor_delta_diag[row].item()) >= float(seen_new_anchor_delta))
            support_override[row] = bool(checks) and all(checks)

    reject_mask = accepted & (void_score >= float(min_void_score)) & (void_margin >= float(min_void_margin)) & ~support_override
    for row in torch.nonzero(reject_mask, as_tuple=False).reshape(-1).tolist():
        predicted[row] = UNKNOWN_LABEL
        accepted[row] = False
        decisions[row] = "reject"
        reasons[row] = "pseudo_unknown_void_gate_reject"

    diagnostics = dict(result.diagnostics or {})
    diagnostics["void_background_score"] = void_score.detach().cpu()
    diagnostics["void_background_known_score"] = known_score.detach().cpu()
    diagnostics["void_background_margin"] = void_margin.detach().cpu()
    diagnostics["void_background_support_override_mask"] = support_override.detach().cpu()
    diagnostics["void_background_reject_mask"] = reject_mask.detach().cpu()
    return replace(
        result,
        predicted_labels=predicted.cpu(),
        accepted=accepted.cpu(),
        decisions=decisions,
        gate_reasons=reasons,
        diagnostics=diagnostics,
    )


def accepted_only_online_update(
    class_states: Mapping[int, ClassState],
    features: torch.Tensor,
    result: PredictionResult,
    *,
    momentum: float = 0.05,
) -> tuple[dict[int, ClassState], dict]:
    """Update prototypes only from accepted old/seen-new rows."""

    x = normalize_rows(torch.as_tensor(features).float())
    if x.size(0) != result.predicted_labels.numel():
        raise ValueError("features and prediction result must have equal lengths")
    updated = {int(k): v for k, v in class_states.items()}
    updated_counts: dict[int, int] = {}
    skipped: dict[str, int] = {}
    for row, (label, accepted, decision) in enumerate(
        zip(result.predicted_labels.tolist(), result.accepted.tolist(), result.decisions)
    ):
        decision = str(decision)
        if not bool(accepted) or decision != "accept" or int(label) == UNKNOWN_LABEL or int(label) not in updated:
            skipped[decision] = skipped.get(decision, 0) + 1
            continue
        state = updated[int(label)]
        proto = normalize_rows(
            ((1.0 - float(momentum)) * torch.as_tensor(state.prototype).float() + float(momentum) * x[row]).view(1, -1)
        ).squeeze(0)
        updated[int(label)] = ClassState(
            class_id=state.class_id,
            group=state.group,
            prototype=proto,
            mask=state.mask.clone(),
            subspace=state.subspace.clone(),
            covariance_diag=state.covariance_diag.clone(),
            thresholds=dict(state.thresholds),
            evt_params=dict(state.evt_params),
            support_quality=state.support_quality,
            source_weight=state.source_weight,
            support_anchors=None if state.support_anchors is None else state.support_anchors.clone(),
        )
        updated_counts[int(label)] = updated_counts.get(int(label), 0) + 1
    return updated, {
        "update_policy": "accepted_only",
        "updated_classes": updated_counts,
        "skipped_decisions": skipped,
        "momentum": float(momentum),
    }


def predict_with_prototypes(
    features: torch.Tensor,
    prototypes: PrototypeSet,
    *,
    unknown_threshold: float | None = None,
    gate_config: OpenSetGateConfig | None = None,
) -> PredictionResult:
    gate_config = _resolve_gate_config(unknown_threshold, gate_config)
    mode = str(gate_config.mode).lower()
    x = normalize_rows(torch.as_tensor(features).float()).to(prototypes.vectors.device)
    scores_all = x @ normalize_rows(prototypes.vectors).T
    topk = scores_all.topk(min(2, scores_all.size(1)), dim=1)
    scores = topk.values[:, 0]
    indices = topk.indices[:, 0]
    if topk.values.size(1) > 1:
        margins = topk.values[:, 0] - topk.values[:, 1]
    else:
        margins = topk.values[:, 0]
    accepted = torch.ones_like(scores, dtype=torch.bool)
    reject_reasons: list[list[str]] = [[] for _ in range(scores.numel())]

    def apply_mask(mask: torch.Tensor, reason: str) -> None:
        nonlocal accepted
        mask = mask.to(device=accepted.device, dtype=torch.bool)
        accepted = accepted & mask
        for row in (~mask).nonzero(as_tuple=False).flatten().tolist():
            reject_reasons[int(row)].append(reason)

    if mode not in {"none", "cosine", "margin", "mahalanobis", "openmax", "evt", "combined", "oa_mse"}:
        raise ValueError(f"unknown open-set gate mode: {gate_config.mode}")

    gate_mode = "combined" if mode == "oa_mse" else mode

    if gate_mode in {"cosine", "combined"} and gate_config.min_cosine is not None:
        apply_mask(scores >= float(gate_config.min_cosine), "low_cosine")
    if gate_mode in {"margin", "combined"} and gate_config.min_margin is not None:
        apply_mask(margins >= float(gate_config.min_margin), "low_margin")

    mahalanobis_all = _mahalanobis_to_prototypes(x, prototypes)
    selected_mahal = None
    if mahalanobis_all is not None:
        selected_mahal = mahalanobis_all.gather(1, indices.view(-1, 1)).squeeze(1)
    if gate_mode in {"mahalanobis", "combined"}:
        if selected_mahal is None:
            raise ValueError("Mahalanobis gate requested but PrototypeSet metadata lacks diag_var")
        if gate_config.max_mahalanobis is not None:
            mahal_threshold = torch.full_like(selected_mahal, float(gate_config.max_mahalanobis))
        else:
            mahal_threshold = _selected_metadata_threshold(prototypes, "mahalanobis_thresholds", indices)
            if mahal_threshold is None:
                raise ValueError("Mahalanobis gate requested but PrototypeSet metadata lacks thresholds")
        apply_mask(selected_mahal <= mahal_threshold.to(device=selected_mahal.device), "high_mahalanobis")

    openmax_distance = (1.0 - scores).clamp_min(0.0)
    if gate_mode in {"openmax", "evt", "combined"}:
        openmax_threshold = _selected_metadata_threshold(prototypes, "openmax_thresholds", indices)
        if openmax_threshold is None:
            if gate_config.min_cosine is None:
                raise ValueError("OpenMax-style gate requested but PrototypeSet metadata lacks thresholds")
            openmax_threshold = torch.full_like(openmax_distance, 1.0 - float(gate_config.min_cosine))
        apply_mask(openmax_distance <= openmax_threshold.to(device=openmax_distance.device), "openmax_tail")

    labels = prototypes.labels[indices].clone()
    labels = torch.where(accepted, labels, torch.full_like(labels, UNKNOWN_LABEL))
    reasons = ["accepted" if not item else ",".join(item) for item in reject_reasons]
    return PredictionResult(
        predicted_labels=labels.cpu(),
        scores=scores.detach().cpu(),
        accepted=accepted.cpu(),
        margins=margins.detach().cpu(),
        mahalanobis=selected_mahal.detach().cpu() if selected_mahal is not None else None,
        openmax_distance=openmax_distance.detach().cpu(),
        gate_reasons=reasons,
        decisions=["accept" if bool(v) else "reject" for v in accepted.detach().cpu().tolist()],
    )


def _accuracy_for_mask(true_labels: torch.Tensor, predicted_labels: torch.Tensor, mask: torch.Tensor) -> float:
    denom = int(mask.sum().item())
    if denom == 0:
        return float("nan")
    return float((predicted_labels[mask] == true_labels[mask]).float().mean().item())


def compute_open_set_metrics(
    *,
    true_labels: torch.Tensor | Iterable[int],
    predicted_labels: torch.Tensor | Iterable[int],
    accepted: torch.Tensor | Iterable[bool],
    old_labels: set[int] | None = None,
    new_labels: set[int] | None = None,
) -> dict[str, float]:
    true_labels = _labels_tensor(true_labels).cpu()
    predicted_labels = _labels_tensor(predicted_labels).cpu()
    accepted = torch.as_tensor(accepted, dtype=torch.bool).cpu()
    if true_labels.numel() != predicted_labels.numel() or true_labels.numel() != accepted.numel():
        raise ValueError("true_labels, predicted_labels, and accepted must have equal lengths")

    total = max(1, int(true_labels.numel()))
    known_mask = true_labels != UNKNOWN_LABEL
    unknown_mask = true_labels == UNKNOWN_LABEL
    accepted_count = int(accepted.sum().item())

    metrics = {
        "coverage": float(accepted_count / total),
        "full_accuracy": float((predicted_labels == true_labels).float().mean().item()) if total else float("nan"),
        "accepted_accuracy": _accuracy_for_mask(true_labels, predicted_labels, accepted) if accepted_count else float("nan"),
        "known_accuracy": _accuracy_for_mask(true_labels, predicted_labels, known_mask),
        "unknown_rejection_rate": _accuracy_for_mask(true_labels, predicted_labels, unknown_mask),
    }
    if int(unknown_mask.sum().item()) > 0:
        metrics["unknown_false_accept_rate"] = 1.0 - metrics["unknown_rejection_rate"]
    else:
        metrics["unknown_false_accept_rate"] = float("nan")

    if old_labels is not None:
        old_mask = torch.tensor([int(v) in old_labels for v in true_labels.tolist()], dtype=torch.bool)
        metrics["old_class_accuracy"] = _accuracy_for_mask(true_labels, predicted_labels, old_mask)
    if new_labels is not None:
        new_mask = torch.tensor([int(v) in new_labels for v in true_labels.tolist()], dtype=torch.bool)
        metrics["new_class_accuracy"] = _accuracy_for_mask(true_labels, predicted_labels, new_mask)
    return metrics


def run_sfe_enrollment(
    source_prototypes: PrototypeSet,
    support_features: torch.Tensor,
    support_labels: torch.Tensor | Iterable[int],
    query_features: torch.Tensor,
    query_labels: torch.Tensor | Iterable[int],
    *,
    unknown_threshold: float = 0.0,
    gate_config: OpenSetGateConfig | None = None,
    lifecycle_initial_state: str = LIFECYCLE_QUARANTINE,
) -> AdaptationResult:
    gate_config = _resolve_gate_config(unknown_threshold, gate_config)
    new_prototypes = build_prototype_set(support_features, support_labels, gate_config=gate_config)
    all_prototypes = merge_prototype_sets(source_prototypes, new_prototypes)
    pred = predict_with_prototypes(query_features, all_prototypes, gate_config=gate_config)
    metrics = compute_open_set_metrics(
        true_labels=query_labels,
        predicted_labels=pred.predicted_labels,
        accepted=pred.accepted,
        old_labels=source_prototypes.label_values(),
        new_labels=new_prototypes.label_values(),
    )
    manager = NewClassLifecycleManager()
    lifecycle_records = []
    for label, count in zip(new_prototypes.labels.cpu().tolist(), new_prototypes.counts.cpu().tolist()):
        lifecycle_records.append(
            manager.enroll(
                int(label),
                support_count=int(count),
                initial_state=lifecycle_initial_state,
            ).to_dict()
        )
    telemetry = {
        "gate": asdict(gate_config),
        "gate_reasons": pred.gate_reasons,
        "new_class_lifecycle": lifecycle_records,
    }
    return AdaptationResult(
        all_prototypes,
        pred.predicted_labels,
        pred.scores,
        pred.accepted,
        metrics,
        telemetry,
        margins=pred.margins,
        mahalanobis=pred.mahalanobis,
        openmax_distance=pred.openmax_distance,
        gate_reasons=pred.gate_reasons,
    )


def shrink_target_prototypes(
    source_prototypes: PrototypeSet,
    support_features: torch.Tensor,
    support_labels: torch.Tensor | Iterable[int],
    *,
    kappa: float = 3.0,
    drift_by_label: Mapping[int, float] | None = None,
    gate_config: OpenSetGateConfig | None = None,
) -> PrototypeSet:
    support_features = normalize_rows(torch.as_tensor(support_features).float()).to(source_prototypes.vectors.device)
    support_labels = _labels_tensor(support_labels, device=source_prototypes.vectors.device)
    drift_by_label = dict(drift_by_label or {})

    vectors = []
    counts = []
    shrinkage = {}
    for label, source_vec in zip(source_prototypes.labels.tolist(), source_prototypes.vectors):
        mask = support_labels == int(label)
        n = int(mask.sum().item())
        if n > 0:
            target_mean = normalize_rows(support_features[mask].mean(dim=0, keepdim=True)).squeeze(0)
            drift = max(0.0, float(drift_by_label.get(int(label), 0.0)))
            rho = n / (n + max(0.0, float(kappa)) + drift)
            adapted = normalize_rows((rho * target_mean + (1.0 - rho) * source_vec).unsqueeze(0)).squeeze(0)
        else:
            rho = 0.0
            adapted = source_vec
        vectors.append(adapted)
        counts.append(n)
        shrinkage[int(label)] = {"rho": rho, "target_count": n, "drift": float(drift_by_label.get(int(label), 0.0))}

    gate_config = gate_config or OpenSetGateConfig(mode="cosine")
    out = PrototypeSet(
        labels=source_prototypes.labels.clone(),
        vectors=torch.stack(vectors, dim=0),
        counts=torch.tensor(counts, dtype=torch.long, device=source_prototypes.vectors.device),
        metadata={"shrinkage": shrinkage, "source_labels": sorted(source_prototypes.label_values())},
    )
    if "diag_var" in source_prototypes.metadata:
        out.metadata["diag_var"] = source_prototypes.metadata["diag_var"]
    if "openmax_thresholds" in source_prototypes.metadata:
        out.metadata["openmax_thresholds"] = source_prototypes.metadata["openmax_thresholds"]
    if "mahalanobis_thresholds" in source_prototypes.metadata:
        out.metadata["mahalanobis_thresholds"] = source_prototypes.metadata["mahalanobis_thresholds"]
    out.metadata["gate_fit"] = dict(source_prototypes.metadata.get("gate_fit", asdict(gate_config)))
    return out


def run_ftrc_calibration(
    source_prototypes: PrototypeSet,
    support_features: torch.Tensor,
    support_labels: torch.Tensor | Iterable[int],
    query_features: torch.Tensor,
    query_labels: torch.Tensor | Iterable[int],
    *,
    kappa: float = 3.0,
    drift_by_label: Mapping[int, float] | None = None,
    unknown_threshold: float | None = None,
    gate_config: OpenSetGateConfig | None = None,
) -> AdaptationResult:
    gate_config = _resolve_gate_config(unknown_threshold, gate_config)
    adapted = shrink_target_prototypes(
        source_prototypes,
        support_features,
        support_labels,
        kappa=kappa,
        drift_by_label=drift_by_label,
        gate_config=gate_config,
    )
    pred = predict_with_prototypes(query_features, adapted, gate_config=gate_config)
    metrics = compute_open_set_metrics(
        true_labels=query_labels,
        predicted_labels=pred.predicted_labels,
        accepted=pred.accepted,
        old_labels=source_prototypes.label_values(),
    )
    telemetry = {"gate": asdict(gate_config), "gate_reasons": pred.gate_reasons}
    return AdaptationResult(
        adapted,
        pred.predicted_labels,
        pred.scores,
        pred.accepted,
        metrics,
        telemetry,
        margins=pred.margins,
        mahalanobis=pred.mahalanobis,
        openmax_distance=pred.openmax_distance,
        gate_reasons=pred.gate_reasons,
    )
