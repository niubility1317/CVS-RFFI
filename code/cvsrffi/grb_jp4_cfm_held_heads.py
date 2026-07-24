"""Fixed heads for the non-promotable GRB-JP4 Phase1-held falsifier."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.covariance import ledoit_wolf

from cvsrffi.stage2_zid_student_t_qknn import normalize_zid_rows


Z_DIM = 160
SCHEMA = "cvs.phase1.grb_jp4_cfm.held_d92_head.v1"
TASK_WEIGHT = 0.5
STATE_LIMIT_BYTES = 262_144


class GRBJP4HeldHeadError(ValueError):
    """Raised when a held-proxy head or its immutable state drifts."""


def _canon(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _registry(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or len(set(result)) != len(result) or any(not item for item in result):
        raise GRBJP4HeldHeadError(f"{name} must contain unique non-empty handles")
    return result


@dataclass(frozen=True, slots=True)
class HeldD92State:
    classes: tuple[str, ...]
    old_classes: tuple[str, ...]
    new_classes: tuple[str, ...]
    k_shot: int
    coefficient_fp32: np.ndarray
    intercept_fp32: np.ndarray
    old_covariance_fp32: np.ndarray
    new_covariance_fp32: np.ndarray
    active: bool
    audit: Mapping[str, Any]
    state_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes, "classes")
        old = _registry(self.old_classes, "old_classes")
        new = _registry(self.new_classes, "new_classes")
        if (
            self.schema != SCHEMA
            or classes != old + new
            or len(old) < 1
            or len(new) < 1
            or type(self.k_shot) is not int
            or self.k_shot not in (1, 5, 10)
        ):
            raise GRBJP4HeldHeadError("D92 held registry/K contract drift")
        coefficient = np.asarray(self.coefficient_fp32)
        intercept = np.asarray(self.intercept_fp32)
        old_cov = np.asarray(self.old_covariance_fp32)
        new_cov = np.asarray(self.new_covariance_fp32)
        if (
            coefficient.dtype != np.float32
            or coefficient.shape != (len(classes), Z_DIM)
            or intercept.dtype != np.float32
            or intercept.shape != (len(classes),)
            or old_cov.dtype != np.float32
            or old_cov.shape != (Z_DIM, Z_DIM)
            or new_cov.dtype != np.float32
            or new_cov.shape != (Z_DIM, Z_DIM)
            or not all(
                np.isfinite(value).all()
                for value in (coefficient, intercept, old_cov, new_cov)
            )
        ):
            raise GRBJP4HeldHeadError("D92 held numeric state drift")
        expected = _state_digest(
            classes,
            old,
            new,
            self.k_shot,
            coefficient,
            intercept,
            old_cov,
            new_cov,
            bool(self.active),
            self.audit,
        )
        if self.state_sha256 != expected:
            raise GRBJP4HeldHeadError("D92 held state receipt drift")
        for field in (
            "coefficient_fp32",
            "intercept_fp32",
            "old_covariance_fp32",
            "new_covariance_fp32",
        ):
            value = np.ascontiguousarray(getattr(self, field)).copy()
            value.setflags(write=False)
            object.__setattr__(self, field, value)

    @property
    def numeric_state_bytes(self) -> int:
        return int(
            self.coefficient_fp32.nbytes
            + self.intercept_fp32.nbytes
            + self.old_covariance_fp32.nbytes
            + self.new_covariance_fp32.nbytes
        )


def _state_digest(
    classes: tuple[str, ...],
    old: tuple[str, ...],
    new: tuple[str, ...],
    k_shot: int,
    coefficient: np.ndarray,
    intercept: np.ndarray,
    old_covariance: np.ndarray,
    new_covariance: np.ndarray,
    active: bool,
    audit: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    header = _canon(
        {
            "schema": SCHEMA,
            "classes": classes,
            "old_classes": old,
            "new_classes": new,
            "K": k_shot,
            "active": active,
            "audit": dict(audit),
        }
    )
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for value in (coefficient, intercept, old_covariance, new_covariance):
        array = np.ascontiguousarray(value, dtype=np.float32)
        descriptor = _canon({"dtype": array.dtype.str, "shape": array.shape})
        raw = array.tobytes(order="C")
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _within_class_residuals(
    rows: np.ndarray,
    labels: tuple[str, ...],
    classes: tuple[str, ...],
) -> np.ndarray:
    pieces = []
    for class_id in classes:
        local = rows[np.asarray([label == class_id for label in labels])]
        if len(local) < 1:
            raise GRBJP4HeldHeadError("D92 held class lacks support")
        pieces.append(local - local.mean(axis=0, keepdims=True))
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float64)


def fit_held_d92_head(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    *,
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    k_shot: int,
) -> HeldD92State:
    """Fit fixed equal-task shared covariance; K1 marks exact qKNN fallback."""

    old = _registry(old_classes, "old_classes")
    new = _registry(new_classes, "new_classes")
    classes = old + new
    if set(old) & set(new):
        raise GRBJP4HeldHeadError("old/new held registries must be disjoint")
    if type(k_shot) is not int or k_shot not in (1, 5, 10):
        raise GRBJP4HeldHeadError("D92 held K must be 1,5,10")
    rows = normalize_zid_rows(np.asarray(support_zid)).astype(np.float64)
    labels = tuple(str(value) for value in support_labels)
    if (
        len(labels) != len(rows)
        or len(rows) != len(classes) * k_shot
        or any(labels.count(class_id) != k_shot for class_id in classes)
        or any(label not in classes for label in labels)
    ):
        raise GRBJP4HeldHeadError("D92 held support must be balanced over registry")

    means = np.stack([rows[np.asarray([x == c for x in labels])].mean(0) for c in classes])
    if k_shot == 1:
        coefficient = np.zeros((len(classes), Z_DIM), dtype=np.float32)
        intercept = np.zeros(len(classes), dtype=np.float32)
        old_covariance = np.eye(Z_DIM, dtype=np.float32)
        new_covariance = np.eye(Z_DIM, dtype=np.float32)
        audit: dict[str, Any] = {
            "status": "k1_exact_qknn_fallback",
            "active": False,
            "old_task_weight": TASK_WEIGHT,
            "new_task_weight": TASK_WEIGHT,
            "query_rows_used_for_fit": 0,
            "single_affine_head": False,
        }
    else:
        old_residual = _within_class_residuals(rows, labels, old)
        new_residual = _within_class_residuals(rows, labels, new)
        old_covariance64, old_shrink = ledoit_wolf(
            old_residual, assume_centered=True
        )
        new_covariance64, new_shrink = ledoit_wolf(
            new_residual, assume_centered=True
        )
        covariance = TASK_WEIGHT * old_covariance64 + TASK_WEIGHT * new_covariance64
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if (
            not np.isfinite(eigenvalues).all()
            or float(eigenvalues[0]) <= 0.0
        ):
            raise GRBJP4HeldHeadError("D92 held shared covariance is not positive")
        coefficient64 = np.linalg.solve(covariance, means.T).T
        intercept64 = -0.5 * np.einsum("ij,ij->i", means, coefficient64)
        coefficient64 -= coefficient64.mean(axis=0, keepdims=True)
        intercept64 -= intercept64.mean()
        coefficient = np.ascontiguousarray(coefficient64, dtype=np.float32)
        intercept = np.ascontiguousarray(intercept64, dtype=np.float32)
        old_covariance = np.ascontiguousarray(old_covariance64, dtype=np.float32)
        new_covariance = np.ascontiguousarray(new_covariance64, dtype=np.float32)
        audit = {
            "status": "registration_balanced_active",
            "active": True,
            "formula": "Sigma=0.5*Sigma_old_LW+0.5*Sigma_new_LW",
            "old_task_weight": TASK_WEIGHT,
            "new_task_weight": TASK_WEIGHT,
            "old_ledoit_wolf_shrinkage": float(old_shrink),
            "new_ledoit_wolf_shrinkage": float(new_shrink),
            "eigenvalue_min": float(eigenvalues[0]),
            "eigenvalue_max": float(eigenvalues[-1]),
            "query_rows_used_for_fit": 0,
            "single_affine_head": True,
            "head_hyperparameter_scan_count": 0,
        }
    state_bytes = int(
        coefficient.nbytes
        + intercept.nbytes
        + old_covariance.nbytes
        + new_covariance.nbytes
    )
    if state_bytes > STATE_LIMIT_BYTES:
        raise GRBJP4HeldHeadError("D92 held numeric state exceeds 256KiB")
    audit["numeric_state_bytes"] = state_bytes
    audit["state_limit_bytes"] = STATE_LIMIT_BYTES
    digest = _state_digest(
        classes,
        old,
        new,
        k_shot,
        coefficient,
        intercept,
        old_covariance,
        new_covariance,
        k_shot > 1,
        audit,
    )
    return HeldD92State(
        classes=classes,
        old_classes=old,
        new_classes=new,
        k_shot=k_shot,
        coefficient_fp32=coefficient,
        intercept_fp32=intercept,
        old_covariance_fp32=old_covariance,
        new_covariance_fp32=new_covariance,
        active=k_shot > 1,
        audit=audit,
        state_sha256=digest,
    )


def score_held_d92_head(state: HeldD92State, query_zid: np.ndarray) -> np.ndarray:
    if type(state) is not HeldD92State or not state.active:
        raise GRBJP4HeldHeadError("inactive K1 D92 state must use exact qKNN scores")
    query = normalize_zid_rows(np.asarray(query_zid)).astype(np.float64)
    logits = (
        query @ state.coefficient_fp32.astype(np.float64).T
        + state.intercept_fp32.astype(np.float64)
    )
    if not np.isfinite(logits).all():
        raise GRBJP4HeldHeadError("D92 held logits became non-finite")
    return np.ascontiguousarray(logits, dtype=np.float32)


def d92_resource_receipt(state: HeldD92State) -> dict[str, Any]:
    if type(state) is not HeldD92State:
        raise GRBJP4HeldHeadError("D92 resource receipt requires typed state")
    registry_bytes = len(_canon({"classes": state.classes}))
    receipt_bytes = len(state.state_sha256)
    full = state.numeric_state_bytes + registry_bytes + receipt_bytes
    if full > STATE_LIMIT_BYTES:
        raise GRBJP4HeldHeadError("D92 held full state exceeds 256KiB")
    return {
        "numeric_state_bytes": state.numeric_state_bytes,
        "registry_bytes": registry_bytes,
        "receipt_bytes": receipt_bytes,
        "full_head_state_bytes": full,
        "state_limit_bytes": STATE_LIMIT_BYTES,
        "post_backbone_mac_per_query": (
            len(state.classes) * Z_DIM + len(state.classes)
            if state.active
            else 0
        ),
        "query_rows_used_for_fit": 0,
    }


__all__ = [
    "GRBJP4HeldHeadError",
    "HeldD92State",
    "SCHEMA",
    "d92_resource_receipt",
    "fit_held_d92_head",
    "score_held_d92_head",
]
