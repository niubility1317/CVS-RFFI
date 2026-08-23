"""G0-anchored target-domain spectral robust centres for ERBT-IDR M2.6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_m24_compiler import M24InferenceState
from cvsrffi.stage2_m24_features import IF_DIM, physical_if256
from cvsrffi.stage2_m26_spectral_anchor import (
    CHECKPOINT_SHA256_PATTERN,
    ENVELOPE_DIM,
    FFT_DIM,
    GEOMETRY_DIM,
    IDENTITY_DIM,
    RIPPLE_DIM,
    Phase1SpectralAnchor,
    fft_envelope_ripple,
    fft_magnitude_geometry,
)


T1 = "M26-T1-G0-IDENTITY-DOMAIN-RESIDUAL"
T2 = "M26-T2-G0-SPECTRAL-DECOMP-RESIDUAL"
T3 = "M26-T3-G0-JOINT-TARGET-SHIFT-RESIDUAL"
T4 = "M26-T4-G0-MAGNITUDE-GEOMETRY-RESIDUAL"
T5 = "M26-T5-G0-JOINT-MAGNITUDE-GEOMETRY-TARGET-SHIFT-RESIDUAL"
M26_ARMS = (T1, T2, T3, T4, T5)
STRENGTH_GRID = (0.0, 0.01, 0.02, 0.04)
MARGIN_GATE = 0.10
TRUE_MARGIN_P10_TOLERANCE = 0.005
IDENTITY_SHIFT_CAP = 0.35
ENVELOPE_SHIFT_CAP = 0.35
GEOMETRY_SHIFT_CAP = 0.35
INTERACTION_CAP = 0.15
_EPS = 1.0e-12


class M26TDSRCError(ValueError):
    pass


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _unit_rows(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] <= 0 or not np.isfinite(rows).all():
        raise M26TDSRCError("feature rows must be finite and nonempty")
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norm <= _EPS):
        raise M26TDSRCError("feature rows must be nondegenerate")
    return rows / norm


def _cauchy_center(value: Any) -> np.ndarray:
    rows = _unit_rows(value)
    centre = np.mean(rows, axis=0)
    centre /= max(float(np.linalg.norm(centre)), _EPS)
    for _ in range(10):
        residual = np.linalg.norm(rows - centre[None, :], axis=1)
        positive = residual[residual > _EPS]
        scale = float(np.median(positive)) if len(positive) else 1.0
        weight = 1.0 / (1.0 + np.square(residual / max(2.3849 * scale, _EPS)))
        updated = np.sum(weight[:, None] * rows, axis=0) / max(float(np.sum(weight)), _EPS)
        updated /= max(float(np.linalg.norm(updated)), _EPS)
        if float(np.linalg.norm(updated - centre)) <= 1.0e-8:
            centre = updated
            break
        centre = updated
    return centre


def _robust_vector_mean(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or not np.isfinite(rows).all():
        raise M26TDSRCError("domain-shift rows must contain at least two finite classes")
    centre = np.median(rows, axis=0)
    for _ in range(10):
        residual = np.linalg.norm(rows - centre[None, :], axis=1)
        positive = residual[residual > _EPS]
        scale = float(np.median(positive)) if len(positive) else 1.0
        weight = 1.0 / (1.0 + np.square(residual / max(2.3849 * scale, _EPS)))
        updated = np.sum(weight[:, None] * rows, axis=0) / max(float(np.sum(weight)), _EPS)
        if float(np.linalg.norm(updated - centre)) <= 1.0e-8:
            centre = updated
            break
        centre = updated
    return centre


def _cap_vector(value: Any, *, cap: float, reference_norms: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    refs = np.asarray(reference_norms, dtype=np.float64)
    positive = refs[refs > _EPS]
    adaptive = float(np.median(positive)) if len(positive) else 0.0
    allowed = min(float(cap), adaptive) if adaptive > 0.0 else 0.0
    if norm <= _EPS or allowed <= 0.0:
        return np.zeros_like(vector)
    return vector * min(1.0, allowed / norm)


def _blocks(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < IF_DIM or not np.isfinite(rows).all():
        raise M26TDSRCError("M2.6 blocks must be finite N x >=256")
    identity = _unit_rows(rows[:, :IDENTITY_DIM])
    fft = rows[:, IDENTITY_DIM : IDENTITY_DIM + FFT_DIM]
    envelope, ripple = fft_envelope_ripple(fft)
    geometry = fft_magnitude_geometry(fft)
    return (
        identity,
        envelope.astype(np.float64),
        ripple.astype(np.float64),
        geometry.astype(np.float64),
    )


def _class_centres(rows: np.ndarray, labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    result = []
    for name in classes:
        members = rows[labels == str(name)]
        if len(members) < 1:
            raise M26TDSRCError("every registered class requires support")
        result.append(_cauchy_center(members))
    return np.stack(result)


def _loo_reliability(
    source: np.ndarray, target: np.ndarray, delta: np.ndarray, *, cap: float
) -> tuple[float, np.ndarray]:
    gains: list[float] = []
    for held in range(len(source)):
        keep = np.arange(len(source)) != held
        raw = _robust_vector_mean(delta[keep])
        shift = _cap_vector(
            raw,
            cap=float(cap),
            reference_norms=np.linalg.norm(delta[keep], axis=1),
        )
        transported = source[held] + shift
        transported /= max(float(np.linalg.norm(transported)), _EPS)
        gains.append(float(target[held] @ transported - target[held] @ source[held]))
    gain = np.asarray(gains, dtype=np.float64)
    positive_fraction = float(np.mean(gain > 0.0))
    positive_median = max(0.0, float(np.median(gain)))
    reliability = float(np.clip(positive_fraction * positive_median / 0.02, 0.0, 1.0))
    return reliability, gain


@dataclass(frozen=True)
class TargetDomainState:
    old_classes: tuple[str, ...]
    source_identity_centres: np.ndarray
    source_envelope_centres: np.ndarray
    source_geometry_centres: np.ndarray
    target_identity_centres: np.ndarray
    target_envelope_centres: np.ndarray
    target_geometry_centres: np.ndarray
    shared_identity_shift: np.ndarray
    shared_envelope_shift: np.ndarray
    shared_geometry_shift: np.ndarray
    identity_interaction: np.ndarray
    envelope_interaction: np.ndarray
    geometry_interaction: np.ndarray
    identity_reliability: float
    envelope_reliability: float
    geometry_reliability: float
    digest: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.old_classes)
        shapes = {
            "source_identity_centres": (count, IDENTITY_DIM),
            "source_envelope_centres": (count, ENVELOPE_DIM),
            "source_geometry_centres": (count, GEOMETRY_DIM),
            "target_identity_centres": (count, IDENTITY_DIM),
            "target_envelope_centres": (count, ENVELOPE_DIM),
            "target_geometry_centres": (count, GEOMETRY_DIM),
            "shared_identity_shift": (IDENTITY_DIM,),
            "shared_envelope_shift": (ENVELOPE_DIM,),
            "shared_geometry_shift": (GEOMETRY_DIM,),
            "identity_interaction": (count, IDENTITY_DIM),
            "envelope_interaction": (count, ENVELOPE_DIM),
            "geometry_interaction": (count, GEOMETRY_DIM),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise M26TDSRCError("target domain state geometry drift")
            object.__setattr__(self, name, _readonly(value, np.float32))
        if (
            not 0.0 <= float(self.identity_reliability) <= 1.0
            or not 0.0 <= float(self.envelope_reliability) <= 1.0
            or not 0.0 <= float(self.geometry_reliability) <= 1.0
        ):
            raise M26TDSRCError("target domain reliability drift")
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def estimate_target_domain_state(
    old_support_blocks: Any,
    old_support_labels: Any,
    old_classes: Sequence[str],
    source_anchor: Phase1SpectralAnchor,
) -> TargetDomainState:
    classes = tuple(str(item) for item in old_classes)
    labels = np.asarray(old_support_labels).astype(str)
    blocks = np.asarray(old_support_blocks, dtype=np.float64)
    if (
        classes != source_anchor.class_registry
        or blocks.ndim != 2
        or blocks.shape[1] < IF_DIM
        or len(blocks) != len(labels)
        or set(labels.tolist()) != set(classes)
    ):
        raise M26TDSRCError("old support/source anchor registry drift")
    identity, envelope, _ripple, geometry = _blocks(blocks)
    target_identity = _class_centres(identity, labels, classes)
    target_envelope = _class_centres(envelope, labels, classes)
    target_geometry = _class_centres(geometry, labels, classes)
    source = source_anchor.centres().astype(np.float64)
    source_identity = _unit_rows(source[:, :IDENTITY_DIM])
    source_envelope, _source_ripple = fft_envelope_ripple(source[:, IDENTITY_DIM:])
    source_envelope = source_envelope.astype(np.float64)
    source_geometry = fft_magnitude_geometry(source[:, IDENTITY_DIM:]).astype(np.float64)
    identity_delta = target_identity - source_identity
    envelope_delta = target_envelope - source_envelope
    geometry_delta = target_geometry - source_geometry
    shared_identity = _cap_vector(
        _robust_vector_mean(identity_delta),
        cap=IDENTITY_SHIFT_CAP,
        reference_norms=np.linalg.norm(identity_delta, axis=1),
    )
    shared_envelope = _cap_vector(
        _robust_vector_mean(envelope_delta),
        cap=ENVELOPE_SHIFT_CAP,
        reference_norms=np.linalg.norm(envelope_delta, axis=1),
    )
    shared_geometry = _cap_vector(
        _robust_vector_mean(geometry_delta),
        cap=GEOMETRY_SHIFT_CAP,
        reference_norms=np.linalg.norm(geometry_delta, axis=1),
    )
    identity_reliability, identity_gain = _loo_reliability(
        source_identity, target_identity, identity_delta, cap=IDENTITY_SHIFT_CAP
    )
    envelope_reliability, envelope_gain = _loo_reliability(
        source_envelope, target_envelope, envelope_delta, cap=ENVELOPE_SHIFT_CAP
    )
    geometry_reliability, geometry_gain = _loo_reliability(
        source_geometry, target_geometry, geometry_delta, cap=GEOMETRY_SHIFT_CAP
    )
    identity_interaction = identity_delta - shared_identity[None, :]
    envelope_interaction = envelope_delta - shared_envelope[None, :]
    geometry_interaction = geometry_delta - shared_geometry[None, :]
    for matrix in (identity_interaction, envelope_interaction, geometry_interaction):
        norm = np.linalg.norm(matrix, axis=1)
        scale = np.minimum(1.0, INTERACTION_CAP / np.maximum(norm, _EPS))
        matrix *= scale[:, None]
    digest_source = {
        "schema": "cvs.erbt_idr.m26.target_domain_state.v1",
        "classes": list(classes),
        "identity_reliability": identity_reliability,
        "envelope_reliability": envelope_reliability,
        "geometry_reliability": geometry_reliability,
        "shared_identity_shift": shared_identity.astype(float).tolist(),
        "shared_envelope_shift": shared_envelope.astype(float).tolist(),
        "shared_geometry_shift": shared_geometry.astype(float).tolist(),
    }
    digest = hashlib.sha256(
        json.dumps(digest_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = {
        **digest_source,
        "identity_loo_gain": identity_gain.astype(float).tolist(),
        "envelope_loo_gain": envelope_gain.astype(float).tolist(),
        "geometry_loo_gain": geometry_gain.astype(float).tolist(),
        "identity_shift_norm": float(np.linalg.norm(shared_identity)),
        "envelope_shift_norm": float(np.linalg.norm(shared_envelope)),
        "geometry_shift_norm": float(np.linalg.norm(shared_geometry)),
        "new_support_rows_used": 0,
        "query_rows_used": 0,
        "source_member_rows_available": False,
    }
    return TargetDomainState(
        old_classes=classes,
        source_identity_centres=source_identity,
        source_envelope_centres=source_envelope,
        source_geometry_centres=source_geometry,
        target_identity_centres=target_identity,
        target_envelope_centres=target_envelope,
        target_geometry_centres=target_geometry,
        shared_identity_shift=shared_identity,
        shared_envelope_shift=shared_envelope,
        shared_geometry_shift=shared_geometry,
        identity_interaction=identity_interaction,
        envelope_interaction=envelope_interaction,
        geometry_interaction=geometry_interaction,
        identity_reliability=identity_reliability,
        envelope_reliability=envelope_reliability,
        geometry_reliability=geometry_reliability,
        digest=digest,
        audit=audit,
    )


def _transported_centres(
    domain: TargetDomainState,
    support_identity: np.ndarray,
    support_envelope: np.ndarray,
    support_geometry: np.ndarray,
    *,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_count = len(domain.old_classes)
    interaction_shrink = float(k_shot / (k_shot + 8.0))
    identity = np.array(support_identity, dtype=np.float64, copy=True)
    envelope = np.array(support_envelope, dtype=np.float64, copy=True)
    geometry = np.array(support_geometry, dtype=np.float64, copy=True)
    for index in range(old_count):
        identity[index] = (
            domain.source_identity_centres[index]
            + float(domain.identity_reliability) * domain.shared_identity_shift
            + float(domain.identity_reliability) * interaction_shrink * domain.identity_interaction[index]
        )
        envelope[index] = (
            domain.source_envelope_centres[index]
            + float(domain.envelope_reliability) * domain.shared_envelope_shift
            + float(domain.envelope_reliability) * interaction_shrink * domain.envelope_interaction[index]
        )
        geometry[index] = (
            domain.source_geometry_centres[index]
            + float(domain.geometry_reliability) * domain.shared_geometry_shift
            + float(domain.geometry_reliability) * interaction_shrink * domain.geometry_interaction[index]
        )
    return _unit_rows(identity), _unit_rows(envelope), _unit_rows(geometry)


def _residual_scores(
    arm: str,
    blocks: Any,
    identity_centres: np.ndarray,
    envelope_centres: np.ndarray,
    ripple_centres: np.ndarray,
    geometry_centres: np.ndarray,
) -> np.ndarray:
    identity, envelope, ripple, geometry = _blocks(blocks)
    identity_score = identity @ identity_centres.T
    envelope_score = envelope @ envelope_centres.T
    ripple_score = ripple @ ripple_centres.T
    geometry_score = geometry @ geometry_centres.T
    if arm == T1:
        return identity_score
    if arm == T2:
        return (ENVELOPE_DIM / FFT_DIM) * envelope_score + (RIPPLE_DIM / FFT_DIM) * ripple_score
    if arm == T3:
        return (
            (IDENTITY_DIM / IF_DIM) * identity_score
            + (ENVELOPE_DIM / IF_DIM) * envelope_score
            + (RIPPLE_DIM / IF_DIM) * ripple_score
        )
    if arm == T4:
        return geometry_score
    if arm == T5:
        return (IDENTITY_DIM / IF_DIM) * identity_score + (GEOMETRY_DIM / IF_DIM) * geometry_score
    raise M26TDSRCError("unknown M2.6 arm")


def _normalized_residual(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 2 or not np.isfinite(rows).all():
        raise M26TDSRCError("residual scores must be finite N x C")
    centred = rows - np.mean(rows, axis=1, keepdims=True)
    scale = np.max(np.abs(centred), axis=1, keepdims=True)
    return centred / np.maximum(scale, _EPS)


def apply_m26_bounded_residual(
    base_scores: Any,
    residual_scores: Any,
    *,
    strength: float,
    margin_gate: float = MARGIN_GATE,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    base = np.asarray(base_scores, dtype=np.float64)
    residual = np.asarray(residual_scores, dtype=np.float64)
    if base.shape != residual.shape or base.ndim != 2 or base.shape[1] < 2:
        raise M26TDSRCError("base/residual score geometry drift")
    if float(strength) < 0.0 or float(margin_gate) < 0.0:
        raise M26TDSRCError("strength and margin gate must be nonnegative")
    ordered = np.partition(base, -2, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    gate = margin <= float(margin_gate)
    delta = float(strength) * _normalized_residual(residual)
    delta[~gate] = 0.0
    return base + delta, MappingProxyType(
        {
            "gated_query_count": int(np.sum(gate)),
            "gated_query_fraction": float(np.mean(gate)) if len(gate) else 0.0,
            "adjusted_query_count": int(np.sum(np.any(np.abs(delta) > 0.0, axis=1))),
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


def _select_strength(
    base_scores: np.ndarray,
    residual_scores: np.ndarray,
    targets: np.ndarray,
    *,
    old_class_count: int,
) -> tuple[float, Mapping[str, Any]]:
    old_mask = targets < int(old_class_count)
    new_mask = ~old_mask
    base_old = _cross_entropy(base_scores, targets, old_mask)
    base_new = _cross_entropy(base_scores, targets, new_mask)
    base_margin = _true_margin(base_scores, targets)
    base_correct = np.argmax(base_scores, axis=1) == targets
    candidates = []
    accepted = []
    for strength in STRENGTH_GRID:
        candidate, application = apply_m26_bounded_residual(
            base_scores, residual_scores, strength=strength
        )
        old_delta = _cross_entropy(candidate, targets, old_mask) - base_old
        new_delta = _cross_entropy(candidate, targets, new_mask) - base_new
        margin_delta = float(
            np.percentile(_true_margin(candidate, targets), 10)
            - np.percentile(base_margin, 10)
        )
        harm = int(np.sum(base_correct & (np.argmax(candidate, axis=1) != targets)))
        row = {
            "strength": float(strength),
            "balanced_ce": float(
                0.5
                * (
                    _cross_entropy(candidate, targets, old_mask)
                    + _cross_entropy(candidate, targets, new_mask)
                )
            ),
            "old_ce_delta": float(old_delta),
            "new_ce_delta": float(new_delta),
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
            "fallback_reason": "SUPPORT_LOO_SELECTED_ZERO" if strength == 0.0 else "SUPPORT_LOO_ACCEPTED",
            "selected_old_ce_delta": float(selected["old_ce_delta"]),
            "selected_new_ce_delta": float(selected["new_ce_delta"]),
            "selected_true_margin_p10_delta": float(selected["true_margin_p10_delta"]),
            "candidate_audits": candidates,
        }
    )


def m26_arm_config_hash(arm: str, anchor_component_id: str) -> str:
    if arm not in M26_ARMS:
        raise M26TDSRCError("unknown M2.6 arm")
    anchor_id = str(anchor_component_id).lower()
    if CHECKPOINT_SHA256_PATTERN.fullmatch(anchor_id) is None:
        raise M26TDSRCError("anchor component identity drift")
    payload = {
        "schema": "cvs.erbt_idr.m26.td_src256_config.v2",
        "arm": arm,
        "source_anchor_component_id": anchor_id,
        "protocol_schema": "p2_min_v1",
        "base": "P2-A1_NO_RF32_R1",
        "fft_split": [ENVELOPE_DIM, RIPPLE_DIM],
        "magnitude_geometry_split": [32, 32, 32],
        "strength_grid": list(STRENGTH_GRID),
        "margin_gate": MARGIN_GATE,
        "query_fit_access": False,
        "query_policy": "independent_all_registered_class_argmax",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class M26TDSRCState:
    classes: tuple[str, ...]
    arm: str
    base_state: M24InferenceState
    domain_state: TargetDomainState
    identity_centres: np.ndarray
    envelope_centres: np.ndarray
    ripple_centres: np.ndarray
    geometry_centres: np.ndarray
    selected_strength: float
    margin_gate: float
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.classes)
        if (
            self.arm not in M26_ARMS
            or tuple(self.base_state.classes) != tuple(self.classes)
            or float(self.selected_strength) not in STRENGTH_GRID
        ):
            raise M26TDSRCError("M2.6 inference state drift")
        for name, width in (
            ("identity_centres", IDENTITY_DIM),
            ("envelope_centres", ENVELOPE_DIM),
            ("ripple_centres", RIPPLE_DIM),
            ("geometry_centres", GEOMETRY_DIM),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (count, width) or not np.isfinite(value).all():
                raise M26TDSRCError("M2.6 decision centre drift")
            object.__setattr__(self, name, _readonly(value, np.float32))
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    @property
    def feature_dim(self) -> int:
        return IF_DIM

    @property
    def state_bytes(self) -> int:
        base = int(
            self.base_state.compiled_affine_state.state_bytes
            + self.base_state.input_log_diag_fp32.nbytes
        )
        return base + int(
            self.identity_centres.nbytes
            + self.envelope_centres.nbytes
            + self.ripple_centres.nbytes
            + self.geometry_centres.nbytes
        )

    def metric_features(self, blocks: Any) -> np.ndarray:
        identity, envelope, ripple, geometry = _blocks(blocks)
        if self.arm == T1:
            joined = identity
        elif self.arm == T2:
            joined = np.concatenate(
                [np.sqrt(ENVELOPE_DIM / FFT_DIM) * envelope, np.sqrt(RIPPLE_DIM / FFT_DIM) * ripple],
                axis=1,
            )
        elif self.arm == T3:
            joined = np.concatenate(
                [
                    np.sqrt(IDENTITY_DIM / IF_DIM) * identity,
                    np.sqrt(ENVELOPE_DIM / IF_DIM) * envelope,
                    np.sqrt(RIPPLE_DIM / IF_DIM) * ripple,
                ],
                axis=1,
            )
        elif self.arm == T4:
            joined = geometry
        else:
            joined = np.concatenate(
                [
                    np.sqrt(IDENTITY_DIM / IF_DIM) * identity,
                    np.sqrt(GEOMETRY_DIM / IF_DIM) * geometry,
                ],
                axis=1,
            )
        return _unit_rows(joined).astype(np.float32)

    def residual_scores(self, blocks: Any) -> np.ndarray:
        return _residual_scores(
            self.arm,
            blocks,
            self.identity_centres,
            self.envelope_centres,
            self.ripple_centres,
            self.geometry_centres,
        ).astype(np.float32)

    def score_with_audit(self, blocks: Any) -> tuple[np.ndarray, Mapping[str, Any]]:
        base = self.base_state.score(physical_if256(blocks))
        if self.selected_strength == 0.0:
            residual = np.zeros_like(base, dtype=np.float32)
        else:
            residual = self.residual_scores(blocks)
        adjusted, application = apply_m26_bounded_residual(
            base,
            residual,
            strength=self.selected_strength,
            margin_gate=self.margin_gate,
        )
        return adjusted.astype(np.float32), application

    def score(self, blocks: Any) -> np.ndarray:
        scores, _application = self.score_with_audit(blocks)
        return scores

    def predict(self, blocks: Any) -> np.ndarray:
        return np.asarray(self.classes)[np.argmax(self.score(blocks), axis=1)]


def fit_m26_td_src256(
    *,
    arm: str,
    base_state: M24InferenceState,
    support_blocks: Any,
    support_labels: Any,
    classes: Sequence[str],
    k_shot: int,
    old_class_count: int,
    source_anchor: Phase1SpectralAnchor,
    domain_digest: str,
) -> tuple[M26TDSRCState, Mapping[str, Any]]:
    if arm not in M26_ARMS:
        raise M26TDSRCError("unknown M2.6 arm")
    blocks = np.asarray(support_blocks, dtype=np.float64)
    labels = np.asarray(support_labels).astype(str)
    registry = tuple(str(item) for item in classes)
    if (
        tuple(base_state.classes) != registry
        or blocks.ndim != 2
        or blocks.shape[1] < IF_DIM
        or len(blocks) != len(labels)
        or not 0 < int(old_class_count) <= len(registry)
        or tuple(registry[: int(old_class_count)]) != source_anchor.class_registry
        or len(blocks) != len(registry) * int(k_shot)
        or set(labels.tolist()) != set(registry)
    ):
        raise M26TDSRCError("M2.6 support/registry geometry drift")
    lookup = {name: index for index, name in enumerate(registry)}
    targets = np.asarray([lookup[name] for name in labels.tolist()], dtype=np.int64)
    if any(int(np.sum(targets == index)) != int(k_shot) for index in range(len(registry))):
        raise M26TDSRCError("support must be exact class-symmetric K-shot")
    old_mask = targets < int(old_class_count)
    domain = estimate_target_domain_state(
        blocks[old_mask], labels[old_mask], registry[: int(old_class_count)], source_anchor
    )
    identity, envelope, ripple, geometry = _blocks(blocks)
    support_identity = _class_centres(identity, labels, registry)
    support_envelope = _class_centres(envelope, labels, registry)
    support_ripple = _class_centres(ripple, labels, registry)
    support_geometry = _class_centres(geometry, labels, registry)
    transported_identity, transported_envelope, transported_geometry = _transported_centres(
        domain, support_identity, support_envelope, support_geometry, k_shot=int(k_shot)
    )
    decision_identity = transported_identity if arm in {T1, T3, T5} else support_identity
    decision_envelope = transported_envelope if arm == T3 else support_envelope
    decision_ripple = support_ripple
    decision_geometry = transported_geometry if arm == T5 else support_geometry
    base_support_scores = base_state.score(physical_if256(blocks))
    if int(k_shot) == 1:
        selected = 0.0
        selection = {
            "selected_strength": 0.0,
            "fallback_to_zero": True,
            "fallback_reason": "K1_EXACT_B0",
            "selected_old_ce_delta": 0.0,
            "selected_new_ce_delta": 0.0,
            "selected_true_margin_p10_delta": 0.0,
            "candidate_audits": [],
        }
        loo_scores = np.zeros_like(base_support_scores)
    else:
        loo_scores = np.empty_like(base_support_scores, dtype=np.float32)
        for held in range(len(blocks)):
            keep = np.arange(len(blocks)) != held
            folded_labels = labels[keep]
            folded_identity, folded_envelope, folded_ripple, folded_geometry = _blocks(blocks[keep])
            folded_support_identity = _class_centres(folded_identity, folded_labels, registry)
            folded_support_envelope = _class_centres(folded_envelope, folded_labels, registry)
            folded_support_ripple = _class_centres(folded_ripple, folded_labels, registry)
            folded_support_geometry = _class_centres(folded_geometry, folded_labels, registry)
            folded_domain = domain
            if targets[held] < int(old_class_count):
                folded_old = np.asarray([lookup[name] < int(old_class_count) for name in folded_labels])
                folded_domain = estimate_target_domain_state(
                    blocks[keep][folded_old],
                    folded_labels[folded_old],
                    registry[: int(old_class_count)],
                    source_anchor,
                )
            folded_transported_identity, folded_transported_envelope, folded_transported_geometry = _transported_centres(
                folded_domain,
                folded_support_identity,
                folded_support_envelope,
                folded_support_geometry,
                k_shot=max(1, int(k_shot) - 1),
            )
            folded_decision_identity = folded_transported_identity if arm in {T1, T3, T5} else folded_support_identity
            folded_decision_envelope = folded_transported_envelope if arm == T3 else folded_support_envelope
            folded_decision_geometry = folded_transported_geometry if arm == T5 else folded_support_geometry
            loo_scores[held] = _residual_scores(
                arm,
                blocks[held : held + 1],
                folded_decision_identity,
                folded_decision_envelope,
                folded_support_ripple,
                folded_decision_geometry,
            )[0]
        selected, selection_raw = _select_strength(
            base_support_scores,
            loo_scores,
            targets,
            old_class_count=int(old_class_count),
        )
        selection = dict(selection_raw)
        if arm == T1 and domain.identity_reliability <= 0.0:
            selected = 0.0
            selection["selected_strength"] = 0.0
            selection["fallback_to_zero"] = True
            selection["fallback_reason"] = "IDENTITY_DOMAIN_LOO_REJECTED"
        if arm == T3 and domain.identity_reliability <= 0.0 and domain.envelope_reliability <= 0.0:
            selected = 0.0
            selection["selected_strength"] = 0.0
            selection["fallback_to_zero"] = True
            selection["fallback_reason"] = "JOINT_DOMAIN_LOO_REJECTED"
        if arm == T5 and domain.identity_reliability <= 0.0 and domain.geometry_reliability <= 0.0:
            selected = 0.0
            selection["selected_strength"] = 0.0
            selection["fallback_to_zero"] = True
            selection["fallback_reason"] = "JOINT_GEOMETRY_DOMAIN_LOO_REJECTED"
    active_blocks = {
        T1: ["identity160"],
        T2: ["fft_envelope32", "fft_ripple64"],
        T3: ["identity160", "fft_envelope32", "fft_ripple64"],
        T4: ["fft_magnitude_geometry96"],
        T5: ["identity160", "fft_magnitude_geometry96"],
    }[arm]
    base_bytes = int(
        base_state.compiled_affine_state.state_bytes
        + base_state.input_log_diag_fp32.nbytes
    )
    centre_bytes = int(
        decision_identity.nbytes
        + decision_envelope.nbytes
        + decision_ripple.nbytes
        + decision_geometry.nbytes
    )
    audit = {
        "schema": "cvs.erbt_idr.m26.td_src256_fit_audit.v1",
        "arm": arm,
        "k_shot": int(k_shot),
        "feature_dim": IF_DIM,
        "support_only": True,
        "query_rows_used": 0,
        "base_method": "P2-A1_NO_RF32_R1",
        "rf32_consumed": False,
        "active_blocks": active_blocks,
        "selected_strength": float(selected),
        "margin_gate": MARGIN_GATE,
        "fallback_reason": selection["fallback_reason"],
        "selection": selection,
        "domain_state": dict(domain.audit),
        "domain_state_digest": domain.digest,
        "external_domain_digest": str(domain_digest),
        "support_and_decision_centres_separated": True,
        "quantization": dict(base_state.audit.get("quantization", {})),
        "resource": {
            "compiled_inference_state_bytes": base_bytes + centre_bytes,
            "persistent_update_state_bytes": 0,
            "transient_registration_workspace_peak_bytes": int(
                identity.nbytes
                + envelope.nbytes
                + ripple.nbytes
                + geometry.nbytes
                + loo_scores.nbytes
            ),
            "residual_query_mac": int(
                {
                    T1: IDENTITY_DIM,
                    T2: FFT_DIM,
                    T3: IF_DIM,
                    T4: GEOMETRY_DIM,
                    T5: IF_DIM,
                }[arm]
                * len(registry)
            ) if selected > 0.0 else 0,
        },
    }
    state = M26TDSRCState(
        classes=registry,
        arm=arm,
        base_state=base_state,
        domain_state=domain,
        identity_centres=decision_identity,
        envelope_centres=decision_envelope,
        ripple_centres=decision_ripple,
        geometry_centres=decision_geometry,
        selected_strength=float(selected),
        margin_gate=MARGIN_GATE,
        audit=audit,
    )
    return state, MappingProxyType(audit)


__all__ = [
    "M26_ARMS",
    "M26TDSRCError",
    "M26TDSRCState",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "TargetDomainState",
    "apply_m26_bounded_residual",
    "estimate_target_domain_state",
    "fit_m26_td_src256",
    "m26_arm_config_hash",
]
