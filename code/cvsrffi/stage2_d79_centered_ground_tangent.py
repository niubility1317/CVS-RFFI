"""Support-centered ground-tangent correction for D79."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


D78_PATH = Path(__file__).with_name("stage2_d78_ground_tangent_worstclass_margin.py")
SPEC = importlib.util.spec_from_file_location("d79_d78_core", D78_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D79 could not load locked D78 core")
d78 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d78
SPEC.loader.exec_module(d78)

OPTIMIZER_STEPS = d78.OPTIMIZER_STEPS
FW_ITERATIONS = OPTIMIZER_STEPS
ground_domain_tangent_basis = d78.ground_domain_tangent_basis


class D79CenteredTangentError(RuntimeError):
    """Raised when the centered tangent compile contract drifts."""


LDAFit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def fit_centered_ground_tangent_margin(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    base_coefficient: np.ndarray,
    tangent_basis: np.ndarray,
    lda_fit: LDAFit,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit D78 on centered support and return affine-equivalent W/b residuals."""

    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise D79CenteredTangentError("D79 support feature drift")
    support_center = np.mean(x, axis=0, dtype=np.float64)
    centered = np.ascontiguousarray(x - support_center[None, :], dtype=np.float64)
    center_error = float(np.max(np.abs(np.mean(centered, axis=0))))
    tolerance = float(
        128.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(x))))
    )
    if center_error > tolerance:
        raise D79CenteredTangentError("D79 support centering drift")
    coefficient_residual, inherited = d78.fit_ground_tangent_worstclass_margin(
        centered,
        labels,
        class_count,
        k_shot,
        base_coefficient=base_coefficient,
        tangent_basis=tangent_basis,
        lda_fit=lda_fit,
    )
    delta_w64 = np.asarray(coefficient_residual, dtype=np.float64)
    delta_b64 = -(delta_w64 @ support_center)
    delta_b = np.ascontiguousarray(delta_b64, dtype=np.float32)
    center_logits = delta_w64 @ support_center + delta_b64
    if float(np.max(np.abs(center_logits))) > tolerance:
        raise D79CenteredTangentError("D79 centered affine equivalence drift")
    audit = dict(inherited)
    if audit["status"] == "ground_tangent_worstclass_top2_margin_active":
        audit["status"] = "centered_ground_tangent_worstclass_top2_margin_active"
    elif audit["status"] == "zero_projected_gradient_exact_d62_fallback":
        audit["status"] = "centered_zero_projected_gradient_exact_d62_fallback"
    audit.update(
        {
            "schema": "cvs.phase2.d79.centered_ground_tangent_margin_audit.v1",
            "support_centering_enabled": True,
            "support_center_sha256": _sha256(support_center.astype(np.float64)),
            "centered_support_sha256": _sha256(centered.astype(np.float64)),
            "centered_support_mean_max_abs": center_error,
            "bias_residual_frobenius": float(np.linalg.norm(delta_b64)),
            "bias_residual_sha256": _sha256(delta_b.astype(np.float32)),
            "residual_logit_at_support_center_max_abs": float(
                np.max(np.abs(center_logits))
            ),
            "centered_affine_compile": True,
            "bias_residual_fp32": [float(value) for value in delta_b],
            "old_new_role_specific_branch": False,
            "query_rows_used": 0,
        }
    )
    return coefficient_residual, delta_b, audit


def fit_ground_preconditioned_common_descent(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    base_coefficient: np.ndarray,
    preconditioner: np.ndarray,
    lda_fit: LDAFit,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility entry consumed by the inherited integration scaffold."""

    delta_w, delta_b, audit = fit_centered_ground_tangent_margin(
        rows,
        labels,
        class_count,
        k_shot,
        base_coefficient=base_coefficient,
        tangent_basis=preconditioner,
        lda_fit=lda_fit,
    )
    facade = sys.modules.get("d79_centered_core_facade")
    if facade is None or not hasattr(facade, "set_bias_residual"):
        raise D79CenteredTangentError("D79 compile facade is unavailable")
    facade.set_bias_residual(delta_b)
    return delta_w, audit


__all__ = [
    "D79CenteredTangentError",
    "OPTIMIZER_STEPS",
    "fit_centered_ground_tangent_margin",
    "ground_domain_tangent_basis",
]
