"""Support-only non-equivalent heads for ERBT-IDR M2.4.

The module deliberately avoids the historical P2-A1 covariance/LDA fitter.
It uses a frozen balanced identity/FFT geometry, an optional rank-one hard
projection estimated only from support, a class-symmetric uncertainty bias,
and an optional class-count-normalized two-prototype score.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_d74_orthogonal_nuisance_removal import (
    D74ProjectionError,
    fit_orthogonal_nuisance_direction,
)
from cvsrffi.stage2_m24_features import FFT_DIM, IDENTITY_DIM, IF_DIM


G1 = "M24-G1-FROZEN-BALANCED-PROTOTYPE"
G2 = "M24-G2-ORTHOGONAL-NUISANCE"
G3 = "M24-G3-CLASS-UNCERTAINTY"
G4 = "M24-G4-LOCAL-DUAL-PROTOTYPE"
M24_INVARIANCE_ARMS = (G1, G2, G3, G4)

UNCERTAINTY_WEIGHT = 0.04
UNCERTAINTY_CAP = 0.12
PROTOTYPE_TEMPERATURE = 12.0
_EPS = 1.0e-12


class M24InvarianceBreakingError(ValueError):
    """Raised when a support-only invariance-breaking head is invalid."""


def _unit_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(rows).all() or np.any(norm <= _EPS):
        raise M24InvarianceBreakingError("feature rows must be finite and nondegenerate")
    return rows / norm


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def balanced_if256(value: Any) -> np.ndarray:
    """Return unit([unit(identity160); unit(FFT96)]) with 50/50 block energy."""

    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] <= 0 or rows.shape[1] < IF_DIM:
        raise M24InvarianceBreakingError("M2.4 balanced features require N x >=256")
    joined = np.concatenate(
        [_unit_rows(rows[:, :IDENTITY_DIM]), _unit_rows(rows[:, IDENTITY_DIM:IF_DIM])],
        axis=1,
    )
    return _unit_rows(joined).astype(np.float32)


def invariance_arm_config_hash(arm: str) -> str:
    if arm not in M24_INVARIANCE_ARMS:
        raise M24InvarianceBreakingError("unknown invariance-breaking arm")
    payload = {
        "schema": "cvs.erbt_idr.m24.invariance_arm_config.v1",
        "arm": arm,
        "protocol_schema": "p2_min_v1",
        "feature": "unit_concat_unit_identity160_unit_fft96",
        "target_covariance_fit": False,
        "projection": "support_only_centroid_orthogonal_rank1_hard",
        "uncertainty_weight": UNCERTAINTY_WEIGHT,
        "uncertainty_cap": UNCERTAINTY_CAP,
        "prototype_temperature": PROTOTYPE_TEMPERATURE,
        "query_fit_access": False,
        "query_policy": "independent_all_registered_class_argmax",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _apply_projection(rows: np.ndarray, direction: np.ndarray) -> np.ndarray:
    projected = np.asarray(rows, dtype=np.float64)
    if np.any(direction):
        projected = projected - np.outer(projected @ direction, direction)
    return _unit_rows(projected)


def _class_centres(rows: np.ndarray, targets: np.ndarray, class_count: int) -> np.ndarray:
    return _unit_rows(
        np.stack([np.mean(rows[targets == index], axis=0) for index in range(class_count)])
    )


def _two_prototypes(rows: np.ndarray) -> np.ndarray:
    similarity = np.asarray(rows, dtype=np.float64) @ np.asarray(rows, dtype=np.float64).T
    np.fill_diagonal(similarity, np.inf)
    left, right = np.unravel_index(int(np.argmin(similarity)), similarity.shape)
    seeds = np.stack([rows[left], rows[right]])
    assignment = np.argmax(rows @ seeds.T, axis=1)
    if len(np.unique(assignment)) != 2:
        assignment = np.zeros(len(rows), dtype=np.int64)
        assignment[right] = 1
    centres = np.stack(
        [np.mean(rows[assignment == index], axis=0) for index in range(2)]
    )
    return _unit_rows(centres)


@dataclass(frozen=True)
class M24InvariantState:
    classes: tuple[str, ...]
    arm: str
    prototypes: np.ndarray
    prototype_counts: np.ndarray
    nuisance_direction: np.ndarray
    uncertainty_penalty: np.ndarray
    domain_digest: str
    config_hash: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        class_count = len(self.classes)
        prototypes = np.asarray(self.prototypes, dtype=np.float32)
        counts = np.asarray(self.prototype_counts, dtype=np.int64)
        direction = np.asarray(self.nuisance_direction, dtype=np.float32)
        penalty = np.asarray(self.uncertainty_penalty, dtype=np.float32)
        if (
            self.arm not in M24_INVARIANCE_ARMS
            or prototypes.ndim != 3
            or prototypes.shape[0] != class_count
            or prototypes.shape[2] != IF_DIM
            or counts.shape != (class_count,)
            or np.any((counts < 1) | (counts > prototypes.shape[1]))
            or direction.shape != (IF_DIM,)
            or penalty.shape != (class_count,)
            or not np.isfinite(prototypes).all()
            or not np.isfinite(direction).all()
            or not np.isfinite(penalty).all()
        ):
            raise M24InvarianceBreakingError("invariance-breaking inference state drift")
        object.__setattr__(self, "prototypes", _readonly(prototypes, np.float32))
        object.__setattr__(self, "prototype_counts", _readonly(counts, np.int64))
        object.__setattr__(self, "nuisance_direction", _readonly(direction, np.float32))
        object.__setattr__(self, "uncertainty_penalty", _readonly(penalty, np.float32))
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    @property
    def feature_dim(self) -> int:
        return IF_DIM

    @property
    def state_bytes(self) -> int:
        return int(
            self.prototypes.nbytes
            + self.prototype_counts.nbytes
            + self.nuisance_direction.nbytes
            + self.uncertainty_penalty.nbytes
        )

    def transform(self, blocks: Any) -> np.ndarray:
        return _apply_projection(
            balanced_if256(blocks).astype(np.float64),
            self.nuisance_direction.astype(np.float64),
        ).astype(np.float32)

    def score(self, blocks: Any) -> np.ndarray:
        rows = self.transform(blocks).astype(np.float64)
        output = np.empty((len(rows), len(self.classes)), dtype=np.float64)
        for class_index, count in enumerate(self.prototype_counts.tolist()):
            logits = rows @ self.prototypes[class_index, :count].astype(np.float64).T
            if count == 1:
                pooled = logits[:, 0]
            else:
                scaled = PROTOTYPE_TEMPERATURE * logits
                maximum = np.max(scaled, axis=1)
                pooled = (
                    maximum
                    + np.log(np.mean(np.exp(scaled - maximum[:, None]), axis=1))
                ) / PROTOTYPE_TEMPERATURE
            output[:, class_index] = pooled - float(self.uncertainty_penalty[class_index])
        return output.astype(np.float32)

    def predict(self, blocks: Any) -> np.ndarray:
        return np.asarray(self.classes)[np.argmax(self.score(blocks), axis=1)]


def fit_m24_invariance_breaking(
    *,
    arm: str,
    support_blocks: Any,
    support_labels: Any,
    classes: Sequence[str],
    k_shot: int,
    domain_digest: str,
) -> tuple[M24InvariantState, Mapping[str, Any]]:
    """Fit one head from exact symmetric support without query or truth access."""

    if arm not in M24_INVARIANCE_ARMS:
        raise M24InvarianceBreakingError("unknown invariance-breaking arm")
    blocks = np.asarray(support_blocks, dtype=np.float64)
    labels = np.asarray(support_labels).astype(str)
    registry = tuple(str(item) for item in classes)
    if not registry or int(k_shot) < 1 or blocks.ndim != 2 or len(blocks) != len(labels):
        raise M24InvarianceBreakingError("support geometry drift")
    lookup = {name: index for index, name in enumerate(registry)}
    if set(labels.tolist()) != set(registry):
        raise M24InvarianceBreakingError("support class registry drift")
    targets = np.asarray([lookup[item] for item in labels.tolist()], dtype=np.int64)
    if len(blocks) != len(registry) * int(k_shot) or any(
        int(np.sum(targets == index)) != int(k_shot) for index in range(len(registry))
    ):
        raise M24InvarianceBreakingError("support must be exact class-symmetric K-shot")

    balanced = balanced_if256(blocks).astype(np.float64)
    direction = np.zeros(IF_DIM, dtype=np.float64)
    projection_audit: dict[str, Any] = {
        "status": "off_frozen_balanced_metric",
        "projection_active": False,
        "projection_removed_rank": 0,
        "projection_rank": IF_DIM,
        "query_rows_used": 0,
    }
    if arm in {G2, G3, G4} and int(k_shot) >= 2:
        try:
            direction_value, _projected, raw_audit = fit_orthogonal_nuisance_direction(
                balanced, targets, len(registry), int(k_shot)
            )
            direction = np.asarray(direction_value, dtype=np.float64)
            projection_audit = dict(raw_audit)
        except D74ProjectionError as exc:
            projection_audit = {
                "status": "degenerate_support_safe_fallback_to_g1",
                "projection_active": False,
                "projection_removed_rank": 0,
                "projection_rank": IF_DIM,
                "reason": str(exc),
                "query_rows_used": 0,
            }
    projected = _apply_projection(balanced, direction)
    centres = _class_centres(projected, targets, len(registry))

    spread = np.asarray(
        [
            np.mean(1.0 - np.clip(projected[targets == index] @ centres[index], -1.0, 1.0))
            for index in range(len(registry))
        ],
        dtype=np.float64,
    )
    penalty = np.zeros(len(registry), dtype=np.float64)
    if arm in {G3, G4} and int(k_shot) >= 2:
        positive = spread[spread > _EPS]
        scale = float(np.median(positive)) if len(positive) else 1.0
        penalty = np.clip(
            UNCERTAINTY_WEIGHT * (spread / max(scale, _EPS)) / np.sqrt(float(k_shot)),
            0.0,
            UNCERTAINTY_CAP,
        )

    prototype_count = 2 if arm == G4 and int(k_shot) >= 5 else 1
    prototype_rows = np.zeros((len(registry), prototype_count, IF_DIM), dtype=np.float64)
    counts = np.full(len(registry), prototype_count, dtype=np.int64)
    for class_index in range(len(registry)):
        members = projected[targets == class_index]
        if prototype_count == 2:
            prototype_rows[class_index] = _two_prototypes(members)
        else:
            prototype_rows[class_index, 0] = centres[class_index]

    legacy_fft_fraction = 16.0 / 17.0
    support_delta = np.linalg.norm(
        balanced - np.asarray(
            np.concatenate(
                [
                    _unit_rows(blocks[:, :IDENTITY_DIM]),
                    4.0 * _unit_rows(blocks[:, IDENTITY_DIM:IF_DIM]),
                ],
                axis=1,
            )
            / np.sqrt(17.0),
            dtype=np.float64,
        ),
        axis=1,
    )
    k_specialization = (
        "K1_FROZEN_BALANCED_PROTOTYPE"
        if int(k_shot) == 1
        else "K2_PROJECTED_SINGLE_PROTOTYPE"
        if int(k_shot) == 2
        else "K_GE_5_LOCAL_DUAL_PROTOTYPE"
        if prototype_count == 2
        else "K_GE_3_PROJECTED_SINGLE_PROTOTYPE"
    )
    audit = {
        "schema": "cvs.erbt_idr.m24.invariance_fit_audit.v1",
        "arm": arm,
        "k_shot": int(k_shot),
        "feature_dim": IF_DIM,
        "support_only": True,
        "query_rows_used": 0,
        "target_covariance_fit": False,
        "historical_f1_fallback": False,
        "frozen_metric": "balanced_identity_fft_cosine",
        "identity_energy_fraction": 0.5,
        "fft_energy_fraction": 0.5,
        "legacy_fft_energy_fraction": legacy_fft_fraction,
        "support_feature_delta_l2_mean_vs_legacy": float(np.mean(support_delta)),
        "projection": projection_audit,
        "uncertainty_spread": spread.tolist(),
        "uncertainty_penalty": penalty.tolist(),
        "prototype_count_by_class": counts.tolist(),
        "prototype_pooling": "class_count_normalized_logmeanexp",
        "prototype_temperature": PROTOTYPE_TEMPERATURE,
        "k_specialization": k_specialization,
        "state_non_equivalence": {
            "Z_support_mean_l2_delta_vs_legacy": float(np.mean(support_delta)),
            "metric_rank_loss": int(projection_audit["projection_removed_rank"]),
            "mu_delta_l2_vs_g1": float(np.linalg.norm(centres - _class_centres(balanced, targets, len(registry)))),
            "W_delta_status": "NONLINEAR_MULTI_PROTOTYPE" if prototype_count == 2 else "PROTOTYPE_HEAD_NOT_LDA",
            "b_delta_l2": float(np.linalg.norm(penalty)),
        },
        "resource": {},
        "quantization": {
            "r_p50": 0.0,
            "r_p95": 0.0,
            "r_p99": 0.0,
            "r_max": 0.0,
            "fraction_r_gt_0_1": 0.0,
            "fraction_r_gt_0_5": 0.0,
            "max_logit_abs_error": 0.0,
        },
    }
    state = M24InvariantState(
        classes=registry,
        arm=arm,
        prototypes=prototype_rows,
        prototype_counts=counts,
        nuisance_direction=direction,
        uncertainty_penalty=penalty,
        domain_digest=str(domain_digest),
        config_hash=invariance_arm_config_hash(arm),
        audit=audit,
    )
    resource = {
        "compiled_inference_state_bytes": state.state_bytes,
        "persistent_update_state_bytes": 0,
        "transient_registration_workspace_peak_bytes": int(
            balanced.nbytes + projected.nbytes + centres.nbytes
        ),
    }
    final_audit = MappingProxyType({**audit, "resource": resource})
    return state, final_audit


__all__ = [
    "G1",
    "G2",
    "G3",
    "G4",
    "M24_INVARIANCE_ARMS",
    "M24InvariantState",
    "M24InvarianceBreakingError",
    "balanced_if256",
    "fit_m24_invariance_breaking",
    "invariance_arm_config_hash",
]
