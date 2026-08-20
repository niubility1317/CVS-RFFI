"""Independent support-to-head refit for the physical IF256 M2.4 baseline."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_ablation_executors import fit_stage2_ablation
from cvsrffi.stage2_ablation_quantization import decode_affine_state
from cvsrffi.stage2_m24_features import IF_DIM, physical_if256
from cvsrffi.stage2_m24_safe_residual import (
    D1_REFIT,
    M24RegistrationWorkspace,
    M24SafeResidualError,
    arm_config_hash,
)
from cvsrffi.stage2_m24_compiler import M24InferenceState, compile_m24_head


def _balanced_registry(
    labels: np.ndarray, classes: tuple[str, ...], old_class_count: int
) -> tuple[np.ndarray, int]:
    if (
        not 1 < int(old_class_count) <= len(classes)
        or len(set(classes)) != len(classes)
        or set(labels.tolist()) != set(classes)
    ):
        raise M24SafeResidualError("D1-REFIT registry drift")
    lookup = {value: index for index, value in enumerate(classes)}
    targets = np.asarray([lookup[value] for value in labels], dtype=np.int64)
    counts = np.bincount(targets, minlength=len(classes))
    if len(set(counts.tolist())) != 1 or int(counts.min()) < 1:
        raise M24SafeResidualError("D1-REFIT requires balanced K-shot support")
    return targets, int(counts[0])


def fit_m24_d1_refit(
    *,
    support_blocks: Any,
    support_labels: Any,
    classes: Sequence[str],
    old_class_count: int,
    domain_digest: str,
    ground_basis: Any,
    ground_spectral_weights: Any,
    ground_audit: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
) -> tuple[M24InferenceState, Mapping[str, Any], M24RegistrationWorkspace]:
    """Fit P2-A1 lifecycle geometry afresh from support, then compile IF256.

    No fitted historical head, coefficient, bias, prediction, or query row is an
    input.  The existing P2-A1 numerical builder is reused as the locked lower
    level estimator so the experiment isolates representation/refit provenance
    rather than introducing a second approximate implementation.
    """

    blocks = np.asarray(support_blocks, dtype=np.float64)
    labels = np.asarray(support_labels).astype(str)
    registry = tuple(str(item) for item in classes)
    if (
        blocks.ndim != 2
        or blocks.shape[1] < IF_DIM
        or labels.shape != (len(blocks),)
        or not np.isfinite(blocks[:, :IF_DIM]).all()
    ):
        raise M24SafeResidualError("D1-REFIT support geometry drift")
    targets, k_shot = _balanced_registry(labels, registry, int(old_class_count))
    physical = physical_if256(blocks)
    padded = np.concatenate(
        [physical, np.zeros((len(physical), 32), dtype=np.float32)], axis=1
    )
    old_mask = targets < int(old_class_count)
    old_classes = registry[: int(old_class_count)]
    new_classes = registry[int(old_class_count) :]
    kwargs = {
        "old_support_features": padded[old_mask],
        "old_support_labels": labels[old_mask],
        "old_classes": old_classes,
        "ground_basis": ground_basis,
        "ground_spectral_weights": ground_spectral_weights,
        "ground_audit": ground_audit,
        "seed": int(seed),
        "device": device,
    }
    if new_classes:
        fresh = fit_stage2_ablation(
            ablation_id="P2-A1",
            new_support_features=padded[~old_mask],
            new_support_labels=labels[~old_mask],
            new_classes=new_classes,
            **kwargs,
        )
    else:
        fresh = fit_stage2_ablation(ablation_id="P2-S2B-FULL", **kwargs)
    if fresh.compiled_affine_state is None:
        coefficient = np.asarray(fresh.coefficient_fp32, dtype=np.float32)
        bias = np.asarray(fresh.intercept_fp32, dtype=np.float32)
    else:
        coefficient, bias = decode_affine_state(fresh.compiled_affine_state)
    if coefficient.shape != (len(registry), 288) or bias.shape != (len(registry),):
        raise M24SafeResidualError("D1-REFIT fresh affine head geometry drift")
    state, resource, quantization = compile_m24_head(
        np.asarray(coefficient[:, :IF_DIM], dtype=np.float32),
        np.asarray(bias, dtype=np.float32),
        classes=registry,
        domain_digest=str(domain_digest),
        config_hash=arm_config_hash(D1_REFIT),
        support_features=physical,
        transient_workspace_bytes=0,
        block_sizes=(160, 96),
        input_log_diag=np.asarray(fresh.log_diag_fp32[:IF_DIM], dtype=np.float32),
    )
    workspace = M24RegistrationWorkspace(
        support_center=np.empty((0, IF_DIM)),
        decision_center=np.empty((0, IF_DIM)),
        covariance_center=np.empty((0, IF_DIM)),
        support_weights=np.empty(0),
        covariance_diagonal=np.empty(0),
    )
    audit = MappingProxyType({
        "schema": "cvs.erbt_idr.m24.d1_refit_audit.v1",
        "arm": D1_REFIT,
        "method_identity": "P2_A1_NUMERICS_FRESH_SUPPORT_REFIT_IF256",
        "fresh_support_refit": True,
        "historical_fitted_head_inputs": 0,
        "query_rows_used": 0,
        "support_only": True,
        "k_shot": k_shot,
        "registered_class_count": len(registry),
        "old_class_count": int(old_class_count),
        "new_class_count": len(new_classes),
        "source_fit_ablation": "P2-A1" if new_classes else "P2-S2B-FULL",
        "source_fit_feature_profile": "identity160_fft96_beta4_blocknorm_globalnorm",
        "quantization": quantization,
        "resource": resource,
        "fresh_fit_resource": dict(fresh.resource),
        "fresh_fit_audit": {
            key: value
            for key, value in fresh.audit.items()
            if key not in {"d81_actual_coefficient_fp32", "d81_actual_intercept_fp32"}
        },
        "modules": {},
        "whole_candidate_safety": {
            "whole_candidate_fallback_to_f1": False,
            "reason": "independent_support_refit",
        },
    })
    return state, audit, workspace


__all__ = ["fit_m24_d1_refit"]
