"""Cross-ground-class consensus nuisance templates for robust support centers."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

try:
    from . import stage2_d81_ground_nuisance_cauchy_center as d81
except ImportError:  # Standalone probe loading before the runtime package bootstrap.
    _D81_PATH = Path(__file__).with_name(
        "stage2_d81_ground_nuisance_cauchy_center.py"
    )
    _SPEC = importlib.util.spec_from_file_location("d84_d81_center_core", _D81_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError("D84 could not load the D81 center-only primitive")
    d81 = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = d81
    _SPEC.loader.exec_module(d81)


Z_DIM = 160


class D84GroundConsensusError(RuntimeError):
    """Raised when the parameter-free D84 closure drifts."""


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def ground_crossclass_consensus_templates(
    domain_class_prototypes: np.ndarray,
    domain_class_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep only domain drift that agrees across compressed ground classes."""

    prototypes = np.asarray(domain_class_prototypes, dtype=np.float64)
    mask = np.asarray(domain_class_mask, dtype=bool)
    if (
        prototypes.ndim != 3
        or prototypes.shape[2] != Z_DIM
        or mask.shape != prototypes.shape[:2]
        or prototypes.shape[0] < 2
        or prototypes.shape[1] < 2
        or not np.isfinite(prototypes).all()
        or np.any(np.sum(mask, axis=0) < 2)
    ):
        raise D84GroundConsensusError("D84 ground prototype tensor drift")
    registry_domain_count = int(prototypes.shape[0])
    active_domains = np.sum(mask, axis=1) >= 2
    if int(np.sum(active_domains)) < 2:
        raise D84GroundConsensusError("D84 lacks active ground domains")
    prototypes = prototypes[active_domains]
    mask = mask[active_domains]
    active = np.where(mask[..., None], prototypes, 0.0)
    class_counts = np.sum(mask, axis=0, dtype=np.float64)
    class_means = np.sum(active, axis=0) / class_counts[:, None]
    residual = np.where(
        mask[..., None], prototypes - class_means[None, :, :], 0.0
    )
    # Canonicalize the class-reduced residual multiset per domain before any
    # floating-point reduction.  This makes the sealed template bytes
    # independent of the ground registry's class ordering, not only close in
    # real arithmetic.
    canonical_residual = np.zeros_like(residual)
    canonical_mask = np.zeros_like(mask)
    for domain_index in range(prototypes.shape[0]):
        vectors = [
            np.ascontiguousarray(residual[domain_index, class_index])
            for class_index in np.flatnonzero(mask[domain_index])
        ]
        vectors.sort(key=lambda value: value.view(np.uint8).tobytes())
        count = len(vectors)
        canonical_residual[domain_index, :count] = np.stack(vectors)
        canonical_mask[domain_index, :count] = True
    residual, mask = canonical_residual, canonical_mask
    domain_counts = np.sum(mask, axis=1, dtype=np.float64)
    consensus = np.sum(residual, axis=1) / domain_counts[:, None]
    disagreement = np.sum(
        np.where(
            mask[..., None],
            np.square(residual - consensus[:, None, :]),
            0.0,
        ),
        axis=(1, 2),
    ) / (domain_counts * Z_DIM)
    signal = np.sum(np.square(consensus), axis=1) / Z_DIM
    epsilon = np.finfo(np.float64).eps * max(
        1.0, float(np.max(signal + disagreement))
    )
    reliability = signal / (signal + disagreement + epsilon)
    norm = np.linalg.norm(consensus, axis=1)
    tolerance = (
        max(float(np.max(norm)), 1.0)
        * max(prototypes.shape)
        * np.finfo(np.float64).eps
    )
    retained = (norm > tolerance) & (reliability > 0.0)
    if int(np.sum(retained)) < 2:
        raise D84GroundConsensusError("D84 lacks cross-class consensus domains")
    templates = (consensus[retained] / norm[retained, None]).T
    raw_weight = reliability[retained]
    weights = raw_weight / np.sum(raw_weight)
    if (
        not np.isfinite(templates).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-14)
    ):
        raise D84GroundConsensusError("D84 consensus template numerical drift")
    templates = np.ascontiguousarray(templates, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    templates.setflags(write=False)
    weights.setflags(write=False)
    audit = {
        "schema": "cvs.phase2.d84.ground_crossclass_consensus_templates.v1",
        "registry_domain_count": registry_domain_count,
        "domain_count": int(prototypes.shape[0]),
        "ground_class_count": int(prototypes.shape[1]),
        "active_domain_class_cells": int(np.sum(mask)),
        "retained_domain_template_count": int(np.sum(retained)),
        "template_policy": "class_center_then_domain_crossclass_consensus",
        "weight_policy": "normalized_signal_over_signal_plus_crossclass_disagreement",
        "rank_scan_count": 0,
        "weight_scan_count": 0,
        "hyperparameter_count": 0,
        "reliability_min": float(np.min(reliability[retained])),
        "reliability_mean": float(np.mean(reliability[retained])),
        "reliability_max": float(np.max(reliability[retained])),
        "spectral_weight_min": float(np.min(weights)),
        "spectral_weight_max": float(np.max(weights)),
        "template_sha256": _sha256_array(templates),
        "weight_sha256": _sha256_array(weights),
        "ground_class_centers_discarded": True,
        "ground_class_score_access": False,
        "ground_target_identity_mapping_access": False,
        "old_new_role_specific_branch": False,
    }
    return templates, weights, audit


def translate_to_consensus_robust_centers(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    templates: np.ndarray,
    consensus_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply D81's center-only Cauchy rule using D84 consensus directions."""

    transformed, base_audit = d81.translate_to_robust_centers(
        rows,
        labels,
        class_count,
        k_shot,
        templates,
        consensus_weights,
    )
    audit = dict(base_audit)
    audit.update(
        {
            "schema": "cvs.phase2.d84.support_consensus_center_translation.v1",
            "center_formula": "one_step_classwise_cauchy_cross_ground_class_consensus_energy",
            "energy_scale_policy": "per_target_class_mean_consensus_template_energy",
            "ground_template_semantics": "cross_ground_class_agreeing_domain_drift_only",
            "d84_class_id_specific_formula": False,
            "d84_old_new_role_specific_branch": False,
            "d84_query_rows_used": 0,
            "d84_hyperparameter_count": 0,
            "d84_weight_scan_count": 0,
        }
    )
    return transformed, audit


def build_consensus_center_component_fit(
    component_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    templates: np.ndarray,
    consensus_weights: np.ndarray,
    template_audit: dict[str, Any],
    component_arm: str,
    collector: list[dict[str, Any]],
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Wrap every D62 full/block OOF scope with the D84 center translation."""

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        transformed, transform_audit = translate_to_consensus_robust_centers(
            rows,
            labels,
            class_count,
            k_shot,
            templates,
            consensus_weights,
        )
        coefficient, intercept, base_audit = component_fit(
            transformed, labels, class_count, k_shot
        )
        collector.append(
            {
                "component_arm": component_arm,
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "center_shift_l2_max": transform_audit["center_shift_l2_max"],
                "normalized_weight_min": transform_audit["normalized_weight_min"],
            }
        )
        audit = dict(base_audit)
        audit.update(
            {
                "d84_component_arm": component_arm,
                "d84_ground_template_sha256": template_audit["template_sha256"],
                "d84_ground_weight_sha256": template_audit["weight_sha256"],
                "d84_retained_domain_template_count": template_audit[
                    "retained_domain_template_count"
                ],
                "d84_template_policy": template_audit["template_policy"],
                "d84_transform_audit": transform_audit,
            }
        )
        return coefficient, intercept, audit

    return fit


__all__ = [
    "D84GroundConsensusError",
    "Z_DIM",
    "build_consensus_center_component_fit",
    "ground_crossclass_consensus_templates",
    "translate_to_consensus_robust_centers",
]
