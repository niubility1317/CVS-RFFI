"""G0-anchored support-only residual calibration for ERBT-IDR M2.5.

The historical D92/R1 affine score remains the primary decision.  This module
can add a small class-centred local-support residual only for low-margin
queries.  Residual strength is selected from a frozen grid using support
jackknife evidence; K1 and K2 are exact base fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_m24_compiler import M24InferenceState
from cvsrffi.stage2_m24_features import IF_DIM, physical_if256


B1 = "M25-B1-G0-BOUNDED-LOCAL-RESIDUAL"
B2 = "M25-B2-G0-SHRINKAGE-RADIUS-RESIDUAL"
B3 = "M25-B3-G0-STABLE-DUAL-PROTOTYPE-RESIDUAL"
M25_ANCHORED_ARMS = (B1, B2, B3)

STRENGTH_GRID = (0.0, 0.02, 0.04, 0.08)
MARGIN_GATE = 0.10
TRUE_MARGIN_P10_TOLERANCE = 0.005
RADIUS_PRIOR_DOF = 8.0
SPLIT_SSE_REDUCTION_MIN = 0.25
SPLIT_JACKKNIFE_COSINE_MIN = 0.90
SPLIT_JACKKNIFE_ASSIGNMENT_MIN = 0.80
PROTOTYPE_TEMPERATURE = 12.0
_EPS = 1.0e-12


class M25AnchoredResidualError(ValueError):
    """Raised when the anchored residual state or support geometry is invalid."""


def _unit_rows(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] <= 0 or not np.isfinite(rows).all():
        raise M25AnchoredResidualError("feature rows must be finite and nonempty")
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norm <= _EPS):
        raise M25AnchoredResidualError("feature rows must be nondegenerate")
    return rows / norm


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def anchored_arm_config_hash(arm: str) -> str:
    if arm not in M25_ANCHORED_ARMS:
        raise M25AnchoredResidualError("unknown anchored residual arm")
    payload = {
        "schema": "cvs.erbt_idr.m25.anchored_residual_config.v1",
        "arm": arm,
        "protocol_schema": "p2_min_v1",
        "base": "P2-A1_NO_RF32_R1",
        "strength_grid": list(STRENGTH_GRID),
        "margin_gate": MARGIN_GATE,
        "true_margin_p10_tolerance": TRUE_MARGIN_P10_TOLERANCE,
        "radius_prior_dof": RADIUS_PRIOR_DOF,
        "split_sse_reduction_min": SPLIT_SSE_REDUCTION_MIN,
        "split_jackknife_cosine_min": SPLIT_JACKKNIFE_COSINE_MIN,
        "split_jackknife_assignment_min": SPLIT_JACKKNIFE_ASSIGNMENT_MIN,
        "query_fit_access": False,
        "query_policy": "independent_all_registered_class_argmax",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _two_cluster(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = _unit_rows(rows)
    similarity = features @ features.T
    np.fill_diagonal(similarity, np.inf)
    left, right = np.unravel_index(int(np.argmin(similarity)), similarity.shape)
    seeds = np.stack([features[left], features[right]])
    assignment = np.argmax(features @ seeds.T, axis=1)
    if len(np.unique(assignment)) != 2:
        return _unit_rows(np.mean(features, axis=0, keepdims=True)), np.zeros(len(features), dtype=np.int64), np.asarray([len(features)])
    centres = _unit_rows(
        np.stack([np.mean(features[assignment == index], axis=0) for index in range(2)])
    )
    assignment = np.argmax(features @ centres.T, axis=1)
    counts = np.bincount(assignment, minlength=2)
    centres = _unit_rows(
        np.stack([np.mean(features[assignment == index], axis=0) for index in range(2)])
    )
    return centres, assignment, counts


def _sse(rows: np.ndarray, centres: np.ndarray, assignment: np.ndarray) -> float:
    residual = np.asarray(rows, dtype=np.float64) - centres[assignment]
    return float(np.sum(np.square(residual)))


def _matched_two_centres(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direct = float(reference[0] @ candidate[0] + reference[1] @ candidate[1])
    swapped = float(reference[0] @ candidate[1] + reference[1] @ candidate[0])
    if swapped > direct:
        return candidate[[1, 0]], np.asarray([1, 0], dtype=np.int64)
    return candidate, np.asarray([0, 1], dtype=np.int64)


def fit_stable_prototypes(rows: Any) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    features = _unit_rows(rows)
    single = _unit_rows(np.mean(features, axis=0, keepdims=True))
    base_assignment = np.zeros(len(features), dtype=np.int64)
    single_sse = _sse(features, single, base_assignment)
    audit: dict[str, Any] = {
        "split_accepted": False,
        "sample_count": int(len(features)),
        "cluster_sizes": [int(len(features))],
        "sse_reduction_fraction": 0.0,
        "jackknife_min_matched_cosine": 0.0,
        "jackknife_min_assignment_agreement": 0.0,
    }
    if len(features) < 4 or single_sse <= _EPS:
        return single.astype(np.float32), np.ones(1, dtype=np.float32), MappingProxyType(audit)

    centres, assignment, counts = _two_cluster(features)
    if len(centres) != 2 or int(np.min(counts)) < 2:
        audit["cluster_sizes"] = counts.astype(int).tolist()
        return single.astype(np.float32), np.ones(1, dtype=np.float32), MappingProxyType(audit)
    split_sse = _sse(features, centres, assignment)
    reduction = float((single_sse - split_sse) / max(single_sse, _EPS))
    audit["cluster_sizes"] = counts.astype(int).tolist()
    audit["sse_reduction_fraction"] = reduction
    if reduction < SPLIT_SSE_REDUCTION_MIN:
        return single.astype(np.float32), np.ones(1, dtype=np.float32), MappingProxyType(audit)

    matched_cosines: list[float] = []
    assignment_agreements: list[float] = []
    for held in range(len(features)):
        keep = np.arange(len(features)) != held
        folded, folded_assignment, folded_counts = _two_cluster(features[keep])
        if len(folded) != 2 or int(np.min(folded_counts)) < 2:
            return single.astype(np.float32), np.ones(1, dtype=np.float32), MappingProxyType(audit)
        matched, mapping = _matched_two_centres(centres, folded)
        matched_cosines.append(float(np.min(np.sum(centres * matched, axis=1))))
        remapped_assignment = mapping[folded_assignment]
        assignment_agreements.append(float(np.mean(remapped_assignment == assignment[keep])))
    min_cosine = float(min(matched_cosines))
    min_agreement = float(min(assignment_agreements))
    audit["jackknife_min_matched_cosine"] = min_cosine
    audit["jackknife_min_assignment_agreement"] = min_agreement
    if (
        min_cosine < SPLIT_JACKKNIFE_COSINE_MIN
        or min_agreement < SPLIT_JACKKNIFE_ASSIGNMENT_MIN
    ):
        return single.astype(np.float32), np.ones(1, dtype=np.float32), MappingProxyType(audit)

    weights = counts.astype(np.float64) / float(np.sum(counts))
    audit["split_accepted"] = True
    return centres.astype(np.float32), weights.astype(np.float32), MappingProxyType(audit)


@dataclass(frozen=True)
class LocalEvidenceModel:
    prototypes: np.ndarray
    prototype_counts: np.ndarray
    prototype_weights: np.ndarray
    radius_squared: np.ndarray
    use_radius: bool
    split_audits: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        prototypes = np.asarray(self.prototypes, dtype=np.float32)
        counts = np.asarray(self.prototype_counts, dtype=np.int64)
        weights = np.asarray(self.prototype_weights, dtype=np.float32)
        radius = np.asarray(self.radius_squared, dtype=np.float32)
        if (
            prototypes.ndim != 3
            or prototypes.shape[2] != IF_DIM
            or counts.shape != (prototypes.shape[0],)
            or weights.shape != prototypes.shape[:2]
            or radius.shape != (prototypes.shape[0],)
            or np.any(counts < 1)
            or np.any(counts > prototypes.shape[1])
            or np.any(radius <= 0.0)
            or not np.isfinite(prototypes).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(radius).all()
        ):
            raise M25AnchoredResidualError("local evidence state drift")
        object.__setattr__(self, "prototypes", _readonly(prototypes, np.float32))
        object.__setattr__(self, "prototype_counts", _readonly(counts, np.int64))
        object.__setattr__(self, "prototype_weights", _readonly(weights, np.float32))
        object.__setattr__(self, "radius_squared", _readonly(radius, np.float32))
        object.__setattr__(self, "split_audits", tuple(MappingProxyType(dict(item)) for item in self.split_audits))

    @property
    def state_bytes(self) -> int:
        return int(
            self.prototypes.nbytes
            + self.prototype_counts.nbytes
            + self.prototype_weights.nbytes
            + self.radius_squared.nbytes
        )

    @property
    def prototype_total(self) -> int:
        return int(np.sum(self.prototype_counts))

    def score(self, value: Any) -> np.ndarray:
        rows = _unit_rows(value)
        result = np.empty((len(rows), len(self.prototype_counts)), dtype=np.float64)
        for class_index, count in enumerate(self.prototype_counts.tolist()):
            similarity = rows @ self.prototypes[class_index, :count].astype(np.float64).T
            if self.use_radius:
                radius = float(self.radius_squared[class_index])
                component = -(2.0 * (1.0 - np.clip(similarity, -1.0, 1.0))) / (2.0 * radius)
                component -= 0.5 * np.log(radius)
            else:
                component = similarity
            if count == 1:
                result[:, class_index] = component[:, 0]
            else:
                weights = self.prototype_weights[class_index, :count].astype(np.float64)
                scaled = PROTOTYPE_TEMPERATURE * component + np.log(np.maximum(weights, _EPS))[None, :]
                maximum = np.max(scaled, axis=1)
                result[:, class_index] = (
                    maximum + np.log(np.sum(np.exp(scaled - maximum[:, None]), axis=1))
                ) / PROTOTYPE_TEMPERATURE
        return result.astype(np.float32)


def build_local_evidence_model(
    metric_support: Any,
    targets: Any,
    class_count: int,
    *,
    arm: str,
    k_shot: int,
) -> LocalEvidenceModel:
    if arm not in M25_ANCHORED_ARMS or int(class_count) < 2 or int(k_shot) < 1:
        raise M25AnchoredResidualError("local evidence configuration drift")
    rows = _unit_rows(metric_support)
    labels = np.asarray(targets, dtype=np.int64)
    if labels.shape != (len(rows),) or np.any(labels < 0) or np.any(labels >= int(class_count)):
        raise M25AnchoredResidualError("local evidence target drift")
    members = [rows[labels == index] for index in range(int(class_count))]
    if any(len(item) < 1 for item in members):
        raise M25AnchoredResidualError("each registered class requires support")

    local_prototypes: list[np.ndarray] = []
    local_weights: list[np.ndarray] = []
    split_audits: list[Mapping[str, Any]] = []
    for class_rows in members:
        if arm == B3 and len(class_rows) >= 4:
            prototypes, weights, split_audit = fit_stable_prototypes(class_rows)
        else:
            prototypes = _unit_rows(np.mean(class_rows, axis=0, keepdims=True)).astype(np.float32)
            weights = np.ones(1, dtype=np.float32)
            split_audit = MappingProxyType({"split_accepted": False, "sample_count": int(len(class_rows))})
        local_prototypes.append(prototypes)
        local_weights.append(weights)
        split_audits.append(split_audit)

    maximum = max(len(item) for item in local_prototypes)
    prototypes = np.zeros((int(class_count), maximum, IF_DIM), dtype=np.float32)
    weights = np.zeros((int(class_count), maximum), dtype=np.float32)
    counts = np.empty(int(class_count), dtype=np.int64)
    raw_radius = np.empty(int(class_count), dtype=np.float64)
    for index, class_rows in enumerate(members):
        count = len(local_prototypes[index])
        counts[index] = count
        prototypes[index, :count] = local_prototypes[index]
        weights[index, :count] = local_weights[index]
        nearest = np.max(class_rows @ local_prototypes[index].astype(np.float64).T, axis=1)
        raw_radius[index] = float(np.mean(2.0 * (1.0 - np.clip(nearest, -1.0, 1.0))))
    positive = raw_radius[raw_radius > _EPS]
    pooled = float(np.median(positive)) if len(positive) else 1.0e-3
    rho = float(RADIUS_PRIOR_DOF / (RADIUS_PRIOR_DOF + max(int(k_shot) - 1, 0)))
    radius = (1.0 - rho) * raw_radius + rho * pooled
    radius = np.maximum(radius, max(0.1 * pooled, 1.0e-4))
    return LocalEvidenceModel(
        prototypes=prototypes,
        prototype_counts=counts,
        prototype_weights=weights,
        radius_squared=radius,
        use_radius=arm in {B2, B3},
        split_audits=tuple(split_audits),
    )


def _normalized_residual(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 2 or not np.isfinite(rows).all():
        raise M25AnchoredResidualError("residual scores must be finite N x C")
    centred = rows - np.mean(rows, axis=1, keepdims=True)
    scale = np.max(np.abs(centred), axis=1, keepdims=True)
    return centred / np.maximum(scale, _EPS)


def apply_bounded_residual(
    base_scores: Any,
    local_scores: Any,
    *,
    strength: float,
    margin_gate: float = MARGIN_GATE,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    base = np.asarray(base_scores, dtype=np.float64)
    local = np.asarray(local_scores, dtype=np.float64)
    if base.shape != local.shape or base.ndim != 2 or base.shape[1] < 2:
        raise M25AnchoredResidualError("base/local score geometry drift")
    if float(strength) < 0.0 or float(margin_gate) < 0.0:
        raise M25AnchoredResidualError("residual strength and margin gate must be nonnegative")
    ordered = np.partition(base, -2, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    gate = margin <= float(margin_gate)
    delta = float(strength) * _normalized_residual(local)
    delta[~gate] = 0.0
    adjusted = base + delta
    return adjusted, MappingProxyType(
        {
            "gated_query_count": int(np.sum(gate)),
            "query_count": int(len(base)),
            "max_logit_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
            "strength": float(strength),
            "margin_gate": float(margin_gate),
        }
    )


def _cross_entropy(scores: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(scores, dtype=np.float64)[mask]
    labels = np.asarray(targets, dtype=np.int64)[mask]
    if len(selected) == 0:
        return 0.0
    maximum = np.max(selected, axis=1, keepdims=True)
    logsum = maximum[:, 0] + np.log(np.sum(np.exp(selected - maximum), axis=1))
    return float(np.mean(logsum - selected[np.arange(len(selected)), labels]))


def _true_margin(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    rows = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    masked = rows.copy()
    masked[np.arange(len(rows)), labels] = -np.inf
    return rows[np.arange(len(rows)), labels] - np.max(masked, axis=1)


def select_residual_strength(
    base_scores: Any,
    local_scores: Any,
    targets: Any,
    *,
    old_class_count: int,
    k_shot: int,
) -> tuple[float, Mapping[str, Any]]:
    base = np.asarray(base_scores, dtype=np.float64)
    local = np.asarray(local_scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if base.shape != local.shape or labels.shape != (len(base),):
        raise M25AnchoredResidualError("support selector geometry drift")
    if int(k_shot) < 5:
        return 0.0, MappingProxyType(
            {
                "selected_strength": 0.0,
                "fallback_to_zero": True,
                "fallback_reason": "K_LT_5_EXACT_G0",
                "selected_old_ce_delta": 0.0,
                "selected_new_ce_delta": 0.0,
                "selected_true_margin_p10_delta": 0.0,
                "candidate_audits": [],
            }
        )
    old_mask = labels < int(old_class_count)
    new_mask = ~old_mask
    base_old_ce = _cross_entropy(base, labels, old_mask)
    base_new_ce = _cross_entropy(base, labels, new_mask)
    base_margin = _true_margin(base, labels)
    base_correct = np.argmax(base, axis=1) == labels
    candidates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for strength in STRENGTH_GRID:
        candidate, application = apply_bounded_residual(
            base, local, strength=float(strength), margin_gate=MARGIN_GATE
        )
        old_delta = _cross_entropy(candidate, labels, old_mask) - base_old_ce
        new_delta = _cross_entropy(candidate, labels, new_mask) - base_new_ce
        margin_delta = float(np.percentile(_true_margin(candidate, labels), 10) - np.percentile(base_margin, 10))
        candidate_correct = np.argmax(candidate, axis=1) == labels
        harm = int(np.sum(base_correct & ~candidate_correct))
        row = {
            "strength": float(strength),
            "balanced_ce": float(0.5 * (_cross_entropy(candidate, labels, old_mask) + _cross_entropy(candidate, labels, new_mask))),
            "old_ce_delta": old_delta,
            "new_ce_delta": new_delta,
            "true_margin_p10_delta": margin_delta,
            "support_harm": harm,
            "application": dict(application),
        }
        candidates.append(row)
        if (
            old_delta <= 1.0e-12
            and new_delta <= 1.0e-12
            and margin_delta >= -TRUE_MARGIN_P10_TOLERANCE
            and harm == 0
        ):
            accepted.append(row)
    selected = min(accepted, key=lambda item: (item["balanced_ce"], item["strength"])) if accepted else candidates[0]
    strength = float(selected["strength"])
    return strength, MappingProxyType(
        {
            "selected_strength": strength,
            "fallback_to_zero": strength == 0.0,
            "fallback_reason": "SUPPORT_JACKKNIFE_SELECTED_ZERO" if strength == 0.0 else "SUPPORT_JACKKNIFE_ACCEPTED",
            "selected_old_ce_delta": float(selected["old_ce_delta"]),
            "selected_new_ce_delta": float(selected["new_ce_delta"]),
            "selected_true_margin_p10_delta": float(selected["true_margin_p10_delta"]),
            "candidate_audits": candidates,
        }
    )


def _metric_features(base_state: M24InferenceState, blocks: Any) -> np.ndarray:
    features = physical_if256(blocks).astype(np.float64)
    log_diag = np.asarray(base_state.input_log_diag_fp32, dtype=np.float64)
    if log_diag.size:
        features *= np.exp(log_diag)[None, :]
    return _unit_rows(features).astype(np.float32)


@dataclass(frozen=True)
class M25AnchoredResidualState:
    classes: tuple[str, ...]
    arm: str
    base_state: M24InferenceState
    local_model: LocalEvidenceModel
    selected_strength: float
    margin_gate: float
    domain_digest: str
    config_hash: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.arm not in M25_ANCHORED_ARMS
            or tuple(self.base_state.classes) != tuple(self.classes)
            or len(self.classes) != len(self.local_model.prototype_counts)
            or float(self.selected_strength) not in STRENGTH_GRID
        ):
            raise M25AnchoredResidualError("anchored residual inference state drift")
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    @property
    def feature_dim(self) -> int:
        return IF_DIM

    @property
    def state_bytes(self) -> int:
        base_bytes = int(
            self.base_state.compiled_affine_state.state_bytes
            + self.base_state.input_log_diag_fp32.nbytes
        )
        return base_bytes + self.local_model.state_bytes

    def metric_features(self, blocks: Any) -> np.ndarray:
        return _metric_features(self.base_state, blocks)

    def score(self, blocks: Any) -> np.ndarray:
        physical = physical_if256(blocks)
        base_scores = self.base_state.score(physical)
        if self.selected_strength == 0.0:
            return base_scores
        local_scores = self.local_model.score(self.metric_features(blocks))
        adjusted, _application = apply_bounded_residual(
            base_scores,
            local_scores,
            strength=self.selected_strength,
            margin_gate=self.margin_gate,
        )
        return adjusted.astype(np.float32)

    def predict(self, blocks: Any) -> np.ndarray:
        return np.asarray(self.classes)[np.argmax(self.score(blocks), axis=1)]


def fit_m25_anchored_residual(
    *,
    arm: str,
    base_state: M24InferenceState,
    support_blocks: Any,
    support_labels: Any,
    classes: Sequence[str],
    k_shot: int,
    old_class_count: int,
    domain_digest: str,
) -> tuple[M25AnchoredResidualState, Mapping[str, Any]]:
    if arm not in M25_ANCHORED_ARMS:
        raise M25AnchoredResidualError("unknown anchored residual arm")
    blocks = np.asarray(support_blocks, dtype=np.float64)
    labels = np.asarray(support_labels).astype(str)
    registry = tuple(str(item) for item in classes)
    if (
        tuple(base_state.classes) != registry
        or blocks.ndim != 2
        or blocks.shape[1] < IF_DIM
        or len(blocks) != len(labels)
        or not 0 < int(old_class_count) <= len(registry)
    ):
        raise M25AnchoredResidualError("anchored support geometry drift")
    lookup = {name: index for index, name in enumerate(registry)}
    if set(labels.tolist()) != set(registry):
        raise M25AnchoredResidualError("anchored support registry drift")
    targets = np.asarray([lookup[name] for name in labels.tolist()], dtype=np.int64)
    if len(blocks) != len(registry) * int(k_shot) or any(
        int(np.sum(targets == index)) != int(k_shot) for index in range(len(registry))
    ):
        raise M25AnchoredResidualError("support must be exact class-symmetric K-shot")

    metric_support = _metric_features(base_state, blocks)
    model = build_local_evidence_model(
        metric_support, targets, len(registry), arm=arm, k_shot=int(k_shot)
    )
    base_support_scores = base_state.score(physical_if256(blocks))
    if int(k_shot) < 5:
        selected = 0.0
        selection = {
            "selected_strength": 0.0,
            "fallback_to_zero": True,
            "fallback_reason": "K_LT_5_EXACT_G0",
            "selected_old_ce_delta": 0.0,
            "selected_new_ce_delta": 0.0,
            "selected_true_margin_p10_delta": 0.0,
            "candidate_audits": [],
        }
        jackknife_scores = np.zeros_like(base_support_scores)
    else:
        jackknife_scores = np.empty_like(base_support_scores, dtype=np.float32)
        for held in range(len(blocks)):
            keep = np.arange(len(blocks)) != held
            folded_model = build_local_evidence_model(
                metric_support[keep], targets[keep], len(registry), arm=arm, k_shot=int(k_shot)
            )
            jackknife_scores[held] = folded_model.score(metric_support[held : held + 1])[0]
        selected, selection_raw = select_residual_strength(
            base_support_scores,
            jackknife_scores,
            targets,
            old_class_count=int(old_class_count),
            k_shot=int(k_shot),
        )
        selection = dict(selection_raw)

    quantization = dict(base_state.audit.get("quantization", {}))
    resource = {
        "compiled_inference_state_bytes": int(
            base_state.compiled_affine_state.state_bytes
            + base_state.input_log_diag_fp32.nbytes
            + model.state_bytes
        ),
        "persistent_update_state_bytes": 0,
        "transient_registration_workspace_peak_bytes": int(
            metric_support.nbytes + jackknife_scores.nbytes
        ),
    }
    audit = {
        "schema": "cvs.erbt_idr.m25.anchored_residual_fit_audit.v1",
        "arm": arm,
        "k_shot": int(k_shot),
        "feature_dim": IF_DIM,
        "support_only": True,
        "query_rows_used": 0,
        "base_method": "P2-A1_NO_RF32_R1",
        "selected_strength": float(selected),
        "margin_gate": MARGIN_GATE,
        "fallback_reason": selection["fallback_reason"],
        "selection": selection,
        "prototype_count_by_class": model.prototype_counts.astype(int).tolist(),
        "prototype_weight_by_class": [
            model.prototype_weights[index, : int(count)].astype(float).tolist()
            for index, count in enumerate(model.prototype_counts)
        ],
        "radius_squared_by_class": model.radius_squared.astype(float).tolist(),
        "split_audit_by_class": [dict(item) for item in model.split_audits],
        "quantization": quantization,
        "resource": resource,
    }
    state = M25AnchoredResidualState(
        classes=registry,
        arm=arm,
        base_state=base_state,
        local_model=model,
        selected_strength=float(selected),
        margin_gate=MARGIN_GATE,
        domain_digest=str(domain_digest),
        config_hash=anchored_arm_config_hash(arm),
        audit=audit,
    )
    return state, MappingProxyType(audit)


__all__ = [
    "B1",
    "B2",
    "B3",
    "M25_ANCHORED_ARMS",
    "M25AnchoredResidualError",
    "M25AnchoredResidualState",
    "LocalEvidenceModel",
    "anchored_arm_config_hash",
    "apply_bounded_residual",
    "build_local_evidence_model",
    "fit_m25_anchored_residual",
    "fit_stable_prototypes",
    "select_residual_strength",
]
