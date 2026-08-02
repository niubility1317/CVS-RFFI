"""Minimal typed Phase1 aggregate consumed by D112 SEAM-qKNN.

This module contains no source-row builder and no target adaptation logic.  It
only validates a decoded, already aggregated Phase1 component and binds its
numeric contents to one immutable receipt used by the support-only scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.phase1.d112.seam_bundle.v1"
FEATURE_SCHEMA = "ADV3B02:z_id:unit_l2:160:v1"
FEATURE_DIM = 160
SHARED_RANK = 3
OLD_CLASS_COUNT = 6
G0_COMPONENT_STATE = "NONFORMAL_G0_FUNCTIONAL_ONLY"
NUMERIC_EPSILON = 64.0 * float(np.finfo(np.float32).eps)
EPSILON_VARIANCE_R = NUMERIC_EPSILON**2 / SHARED_RANK
EPSILON_VARIANCE_AMB = NUMERIC_EPSILON**2 / FEATURE_DIM
IDENTITY_FIELDS = (
    "component_state",
    "global_bundle_valid",
    "global_invalid_reason",
    "formal_phase2_eligible",
    "performance_claim_allowed",
    "performance_metrics_allowed",
    "target_access_allowed",
    "query_rows_used_for_fit",
    "checkpoint_sha256",
    "source_aggregate_sha256",
)


class D112BundleError(ValueError):
    """Raised when the immutable D112 Phase1 component is malformed."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: str, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise D112BundleError(f"{field} must be a lowercase SHA256")
    return text


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _unit_rows(value: np.ndarray, expected_shape: tuple[int, ...], field: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.shape != expected_shape or not np.isfinite(rows).all():
        raise D112BundleError(f"{field} shape/finite check failed")
    flat = rows.reshape((-1, FEATURE_DIM))
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    if np.any(norms <= NUMERIC_EPSILON):
        raise D112BundleError(f"{field} contains a degenerate vector")
    return (flat / norms).reshape(expected_shape)


def _canonical_basis(value: np.ndarray, q0: np.ndarray) -> np.ndarray:
    basis = np.asarray(value, dtype=np.float64)
    if basis.shape != (SHARED_RANK, FEATURE_DIM) or not np.isfinite(basis).all():
        raise D112BundleError("U must be finite [3,160]")
    tangent = basis - (basis @ q0)[:, None] * q0[None, :]
    left, singular, right = np.linalg.svd(tangent, full_matrices=False)
    if singular[-1] <= max(NUMERIC_EPSILON, singular[0] * 1.0e-6):
        raise D112BundleError("decoded U lost its frozen rank-three geometry")
    result = left @ right
    for row_index in range(SHARED_RANK):
        pivot = int(np.argmax(np.abs(result[row_index])))
        if result[row_index, pivot] < 0.0:
            result[row_index] *= -1.0
    if not np.allclose(result @ result.T, np.eye(SHARED_RANK), atol=1.0e-10, rtol=0.0):
        raise D112BundleError("decoded U orthogonalization failed")
    return result


def _content_payload(
    *,
    class_registry: tuple[str, ...],
    g: np.ndarray,
    q0: np.ndarray,
    U: np.ndarray,
    sigma0_r: np.ndarray,
    sigma0_amb: np.ndarray,
    v_g_r: np.ndarray,
    v_g_amb: np.ndarray,
    tau_h_r: float,
    g_quantization_l2_error_bound: np.ndarray,
    q0_quantization_l2_error_bound: float,
    U_operator_error_upper_bound: float,
    endpoint_quantization_chord_mse: np.ndarray,
    manifest_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "class_registry": list(class_registry),
        "arrays": {
            name: _array_receipt(array)
            for name, array in (
                ("g", g),
                ("q0", q0),
                ("U", U),
                ("sigma0_r", sigma0_r),
                ("sigma0_amb", sigma0_amb),
                ("v_g_r", v_g_r),
                ("v_g_amb", v_g_amb),
                ("g_quantization_l2_error_bound", g_quantization_l2_error_bound),
                ("endpoint_quantization_chord_mse", endpoint_quantization_chord_mse),
            )
        },
        "tau_h_r": float(tau_h_r),
        "q0_quantization_l2_error_bound": float(q0_quantization_l2_error_bound),
        "U_operator_error_upper_bound": float(U_operator_error_upper_bound),
        "manifest_identity": {
            field: manifest_identity[field] for field in IDENTITY_FIELDS
        },
    }


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if any(field not in manifest for field in IDENTITY_FIELDS):
        raise D112BundleError("D112 manifest identity field missing")
    result = {field: manifest[field] for field in IDENTITY_FIELDS}
    _require_sha256(str(result["checkpoint_sha256"]), "checkpoint_sha256")
    _require_sha256(str(result["source_aggregate_sha256"]), "source_aggregate_sha256")
    if (
        result["component_state"] != G0_COMPONENT_STATE
        or type(result["global_bundle_valid"]) is not bool
        or result["formal_phase2_eligible"] is not False
        or result["performance_claim_allowed"] is not False
        or result["performance_metrics_allowed"] is not False
        or result["target_access_allowed"] is not False
        or result["query_rows_used_for_fit"] != 0
    ):
        raise D112BundleError("D112 G0 manifest permission identity drift")
    reason = str(result["global_invalid_reason"])
    if (
        result["global_bundle_valid"] is True
        and reason != "NONE"
    ) or (
        result["global_bundle_valid"] is False
        and reason != "PHASE1_SPHERE_CHART_OR_RANK_DEGENERATE"
    ):
        raise D112BundleError("D112 global validity reason drift")
    return result


@dataclass(frozen=True, slots=True)
class D112Bundle:
    class_registry: tuple[str, ...]
    g: np.ndarray
    q0: np.ndarray
    U: np.ndarray
    sigma0_r: np.ndarray
    sigma0_amb: np.ndarray
    v_g_r: np.ndarray
    v_g_amb: np.ndarray
    tau_h_r: float
    g_quantization_l2_error_bound: np.ndarray
    q0_quantization_l2_error_bound: float
    U_operator_error_upper_bound: float
    endpoint_quantization_chord_mse: np.ndarray
    manifest: Mapping[str, Any]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise D112BundleError("D112 bundle schema drift")
        if (
            len(self.class_registry) != OLD_CLASS_COUNT
            or len(set(self.class_registry)) != OLD_CLASS_COUNT
            or any(not isinstance(value, str) or not value for value in self.class_registry)
        ):
            raise D112BundleError("D112 bundle requires six unique opaque class handles")
        expected_vector = (OLD_CLASS_COUNT,)
        if (
            self.g.shape != (OLD_CLASS_COUNT, FEATURE_DIM)
            or self.q0.shape != (FEATURE_DIM,)
            or self.U.shape != (SHARED_RANK, FEATURE_DIM)
            or any(
                array.shape != expected_vector
                for array in (
                    self.sigma0_r,
                    self.sigma0_amb,
                    self.v_g_r,
                    self.v_g_amb,
                    self.g_quantization_l2_error_bound,
                    self.endpoint_quantization_chord_mse,
                )
            )
        ):
            raise D112BundleError("D112 bundle array shape drift")
        arrays = (
            self.g,
            self.q0,
            self.U,
            self.sigma0_r,
            self.sigma0_amb,
            self.v_g_r,
            self.v_g_amb,
            self.g_quantization_l2_error_bound,
            self.endpoint_quantization_chord_mse,
        )
        if any(array.flags.writeable or not np.isfinite(array).all() for array in arrays):
            raise D112BundleError("D112 bundle arrays must be finite and deeply readonly")
        if any(
            np.any(array <= 0.0)
            for array in (self.sigma0_r, self.sigma0_amb, self.v_g_r, self.v_g_amb)
        ) or any(
            np.any(array < 0.0)
            for array in (
                self.g_quantization_l2_error_bound,
                self.endpoint_quantization_chord_mse,
            )
        ):
            raise D112BundleError("D112 variance/error assets are invalid")
        if (
            np.any(self.sigma0_r < EPSILON_VARIANCE_R)
            or np.any(self.v_g_r < EPSILON_VARIANCE_R)
            or np.any(self.sigma0_amb < EPSILON_VARIANCE_AMB)
            or np.any(self.v_g_amb < EPSILON_VARIANCE_AMB)
        ):
            raise D112BundleError("D112 variance asset is below the frozen numeric floor")
        scalars = (
            self.tau_h_r,
            self.q0_quantization_l2_error_bound,
            self.U_operator_error_upper_bound,
        )
        if not all(math.isfinite(float(value)) for value in scalars) or float(self.tau_h_r) <= 0.0 or any(
            float(value) < 0.0 for value in scalars[1:]
        ):
            raise D112BundleError("D112 scalar asset is invalid")
        if not isinstance(self.manifest, Mapping):
            raise D112BundleError("D112 manifest must be a mapping")
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        expected_root = _canonical_sha256(
            _content_payload(
                class_registry=self.class_registry,
                g=self.g,
                q0=self.q0,
                U=self.U,
                sigma0_r=self.sigma0_r,
                sigma0_amb=self.sigma0_amb,
                v_g_r=self.v_g_r,
                v_g_amb=self.v_g_amb,
                tau_h_r=self.tau_h_r,
                g_quantization_l2_error_bound=self.g_quantization_l2_error_bound,
                q0_quantization_l2_error_bound=self.q0_quantization_l2_error_bound,
                U_operator_error_upper_bound=self.U_operator_error_upper_bound,
                endpoint_quantization_chord_mse=self.endpoint_quantization_chord_mse,
                manifest_identity=_manifest_identity(self.manifest),
            )
        )
        if _require_sha256(str(self.manifest.get("content_root_sha256", "")), "content_root_sha256") != expected_root:
            raise D112BundleError("D112 manifest/content root binding drift")


def build_d112_g0_bundle(
    *,
    class_registry: Sequence[str],
    g: np.ndarray,
    q0: np.ndarray,
    U: np.ndarray,
    sigma0_r: np.ndarray,
    sigma0_amb: np.ndarray,
    v_g_r: np.ndarray,
    v_g_amb: np.ndarray,
    tau_h_r: float,
    checkpoint_sha256: str,
    source_aggregate_sha256: str,
    global_bundle_valid: bool = True,
    global_invalid_reason: str | None = None,
    g_quantization_l2_error_bound: np.ndarray | None = None,
    q0_quantization_l2_error_bound: float = 0.0,
    U_operator_error_upper_bound: float = 0.0,
    endpoint_quantization_chord_mse: np.ndarray | None = None,
) -> D112Bundle:
    """Construct the typed G0 surface from caller-owned Phase1 aggregates."""

    registry = tuple(str(value) for value in class_registry)
    decoded_q0 = _unit_rows(np.asarray(q0)[None, :], (1, FEATURE_DIM), "q0")[0]
    arrays = {
        "g": _readonly(_unit_rows(g, (OLD_CLASS_COUNT, FEATURE_DIM), "g"), np.float32),
        "q0": _readonly(decoded_q0, np.float32),
        "U": _readonly(_canonical_basis(U, decoded_q0), np.float32),
        "sigma0_r": _readonly(sigma0_r, np.float32),
        "sigma0_amb": _readonly(sigma0_amb, np.float32),
        "v_g_r": _readonly(v_g_r, np.float32),
        "v_g_amb": _readonly(v_g_amb, np.float32),
        "g_quantization_l2_error_bound": _readonly(
            np.zeros(OLD_CLASS_COUNT) if g_quantization_l2_error_bound is None else g_quantization_l2_error_bound,
            np.float32,
        ),
        "endpoint_quantization_chord_mse": _readonly(
            np.zeros(OLD_CLASS_COUNT) if endpoint_quantization_chord_mse is None else endpoint_quantization_chord_mse,
            np.float32,
        ),
    }
    reason = (
        "NONE"
        if global_bundle_valid
        else "PHASE1_SPHERE_CHART_OR_RANK_DEGENERATE"
    )
    if global_invalid_reason is not None and str(global_invalid_reason) != reason:
        raise D112BundleError("caller-supplied global invalid reason drift")
    manifest_values = {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "component_state": G0_COMPONENT_STATE,
        "global_bundle_valid": bool(global_bundle_valid),
        "global_invalid_reason": reason,
        "formal_phase2_eligible": False,
        "performance_claim_allowed": False,
        "performance_metrics_allowed": False,
        "target_access_allowed": False,
        "query_rows_used_for_fit": 0,
        "checkpoint_sha256": _require_sha256(checkpoint_sha256, "checkpoint_sha256"),
        "source_aggregate_sha256": _require_sha256(
            source_aggregate_sha256, "source_aggregate_sha256"
        ),
    }
    root = _canonical_sha256(
        _content_payload(
            class_registry=registry,
            tau_h_r=float(tau_h_r),
            q0_quantization_l2_error_bound=float(q0_quantization_l2_error_bound),
            U_operator_error_upper_bound=float(U_operator_error_upper_bound),
            manifest_identity=manifest_values,
            **arrays,
        )
    )
    manifest_values["content_root_sha256"] = root
    manifest = MappingProxyType(manifest_values)
    return D112Bundle(
        class_registry=registry,
        tau_h_r=float(tau_h_r),
        q0_quantization_l2_error_bound=float(q0_quantization_l2_error_bound),
        U_operator_error_upper_bound=float(U_operator_error_upper_bound),
        manifest=manifest,
        **arrays,
    )


__all__ = [
    "D112Bundle",
    "D112BundleError",
    "FEATURE_DIM",
    "FEATURE_SCHEMA",
    "EPSILON_VARIANCE_AMB",
    "EPSILON_VARIANCE_R",
    "G0_COMPONENT_STATE",
    "OLD_CLASS_COUNT",
    "SCHEMA",
    "SHARED_RANK",
    "build_d112_g0_bundle",
]
