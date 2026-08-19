"""ERBT-IDR M2.4 F1-SafeResidual support-only head fitting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_m24_center import estimate_centers
from cvsrffi.stage2_m24_compiler import M24InferenceState, compile_m24_head
from cvsrffi.stage2_m24_covariance import relative_psd_jitter
from cvsrffi.stage2_m24_features import IF_DIM, physical_if256
from cvsrffi.stage2_m24_prior_transport import gated_old_prior
from cvsrffi.stage2_m24_quality import ess_safe_weights, if_residual_reliability
from cvsrffi.stage2_m24_rf_residual import safe_rf_residual
from cvsrffi.stage2_m24_uncertainty import normalized_capped_penalty


RF_LITE_DIM = 10
COMPACT_DIM = IF_DIM + RF_LITE_DIM

D0 = "M24-D0-HISTORICAL-F1"
D1 = "M24-D1-PHYSICAL256-F1"
D2 = "M24-D2-RELATIVE-PSD-JITTER"
D3 = "M24-D3-RF-QUALITY-CENTER"
D4 = "M24-D4-RF-QUALITY-COVARIANCE"
D5 = "M24-D5-IF-RESIDUAL-RELIABILITY"
D6 = "M24-D6-GATED-GROUND-PRIOR"
D7 = "M24-D7-NUISANCE-COVARIANCE"
D8 = "M24-D8-NORMALIZED-UNCERTAINTY"
D9 = "M24-D9-RF-LITE-DIAG-RESIDUAL"
D10 = "M24-D10-RF-LITE-SAFE-GATE"
M24_ARMS = (D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10)


class M24SafeResidualError(ValueError):
    pass


@dataclass(frozen=True)
class M24RegistrationWorkspace:
    support_center: np.ndarray
    decision_center: np.ndarray
    covariance_center: np.ndarray
    support_weights: np.ndarray
    covariance_diagonal: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(sum(value.nbytes for value in (
            self.support_center,
            self.decision_center,
            self.covariance_center,
            self.support_weights,
            self.covariance_diagonal,
        )))


def arm_config_hash(arm: str) -> str:
    if arm not in M24_ARMS:
        raise M24SafeResidualError("unknown M2.4 arm")
    value = {
        "schema": "cvs.erbt_idr.m24.arm_config.v1",
        "arm": arm,
        "protocol_schema": "p2_min_v1",
        "feature": "normalize_unit_id160_concat_4x_unit_fft96_globalnorm",
        "query_fit_access": False,
        "query_policy": "independent_all_registered_class_argmax",
        "alpha_max": 0.1,
        "persistent_update_state_bytes": 0,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_query_features(blocks: Any, *, feature_dim: int) -> np.ndarray:
    rows = np.asarray(blocks, dtype=np.float64)
    base = physical_if256(rows)
    if feature_dim == IF_DIM:
        return base
    if feature_dim != COMPACT_DIM or rows.shape[1] < COMPACT_DIM:
        raise M24SafeResidualError("query feature dimension drift")
    rf = rows[:, IF_DIM:COMPACT_DIM]
    rf = rf / np.maximum(np.linalg.norm(rf, axis=1, keepdims=True), 1.0e-12)
    return np.concatenate([base, rf], axis=1).astype(np.float32)


def _targets(labels: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    lookup = {item: index for index, item in enumerate(classes)}
    try:
        return np.asarray([lookup[str(item)] for item in labels], dtype=np.int64)
    except KeyError as exc:
        raise M24SafeResidualError("support label outside registry") from exc


def _class_weights(
    values: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    k_shot: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    result = np.empty(len(values), dtype=np.float64)
    audits: list[dict[str, float]] = []
    for item in classes:
        mask = labels == item
        if int(np.sum(mask)) != k_shot:
            raise M24SafeResidualError("support is not balanced K-shot")
        local, audit = ess_safe_weights(values[mask], k_shot=k_shot)
        result[mask] = local / len(classes)
        audits.append(audit)
    return result, audits


def _candidate_is_safe(
    base_coefficient: np.ndarray,
    base_bias: np.ndarray,
    candidate_coefficient: np.ndarray,
    candidate_bias: np.ndarray,
    support: np.ndarray,
    targets: np.ndarray,
) -> tuple[bool, dict[str, float | int]]:
    base_feature_dim = base_coefficient.shape[1]
    base_scores = support[:, :base_feature_dim] @ base_coefficient.T + base_bias[None, :]
    candidate_scores = support @ candidate_coefficient.T + candidate_bias[None, :]
    base_prediction = np.argmax(base_scores, axis=1)
    candidate_prediction = np.argmax(candidate_scores, axis=1)
    base_correct = base_prediction == targets
    candidate_correct = candidate_prediction == targets
    help_count = int(np.sum(~base_correct & candidate_correct))
    harm_count = int(np.sum(base_correct & ~candidate_correct))
    safe = harm_count == 0 and int(np.sum(candidate_correct)) >= int(np.sum(base_correct))
    return safe, {
        "support_base_correct": int(np.sum(base_correct)),
        "support_candidate_correct": int(np.sum(candidate_correct)),
        "support_help": help_count,
        "support_harm": harm_count,
    }


def fit_m24_safe_residual(
    *,
    arm: str,
    support_blocks: Any,
    support_labels: Any,
    classes: Sequence[str],
    support_quality: Any,
    k_shot: int,
    old_class_count: int,
    f1_coefficient: Any,
    f1_bias: Any,
    domain_digest: str,
    ground_prior_identity: Any | None = None,
    nuisance_covariance_identity: Any | None = None,
) -> tuple[M24InferenceState, Mapping[str, Any], M24RegistrationWorkspace]:
    if arm not in M24_ARMS[1:]:
        raise M24SafeResidualError("D0 is executed by the historical F1 path")
    blocks = np.asarray(support_blocks, dtype=np.float64)
    labels = np.asarray(support_labels).astype(str)
    registry = tuple(str(item) for item in classes)
    quality = np.asarray(support_quality, dtype=np.float64)
    if blocks.ndim != 2 or blocks.shape[1] < COMPACT_DIM or len(blocks) != len(labels) or quality.shape != (len(blocks),):
        raise M24SafeResidualError("support geometry drift")
    targets = _targets(labels, registry)
    base_rows = physical_if256(blocks)
    base_coefficient = np.asarray(f1_coefficient, dtype=np.float64)
    base_bias = np.asarray(f1_bias, dtype=np.float64)
    if base_coefficient.shape != (len(registry), IF_DIM) or base_bias.shape != (len(registry),):
        raise M24SafeResidualError("F1 reference head geometry drift")

    if arm == D1 or k_shot == 1:
        coefficient = base_coefficient
        bias = base_bias
        feature_dim = IF_DIM
        support_for_compile = base_rows
        workspace = M24RegistrationWorkspace(
            support_center=np.empty((0, IF_DIM)),
            decision_center=np.empty((0, IF_DIM)),
            covariance_center=np.empty((0, IF_DIM)),
            support_weights=np.empty(0),
            covariance_diagonal=np.empty(0),
        )
        safety = {"whole_candidate_fallback_to_f1": False, "reason": "exact_f1" if arm == D1 else "forced_f1_k1"}
        module_audit: dict[str, Any] = {
            "quality_center_enabled": False,
            "quality_covariance_enabled": False,
            "if_residual_enabled": False,
            "prior_enabled": False,
            "nuisance_enabled": False,
            "uncertainty_enabled": False,
            "rf_enabled": False,
        }
    else:
        raw_center = quality if arm == D3 else np.ones(len(blocks))
        raw_covariance = quality if arm == D4 else np.ones(len(blocks))
        if arm == D5:
            residual_reliability = if_residual_reliability(base_rows, labels, registry)
            raw_center = residual_reliability
            raw_covariance = residual_reliability
        center_weights, center_ess = _class_weights(raw_center, labels, registry, k_shot)
        covariance_weights, covariance_ess = _class_weights(raw_covariance, labels, registry, k_shot)
        support_center, decision_center, covariance_center = estimate_centers(
            base_rows,
            labels,
            registry,
            center_weights=center_weights,
            covariance_weights=covariance_weights,
        )
        prior_gate = np.zeros(old_class_count)
        prior_audit: Mapping[str, Any] = {"mode": "off", "fallback_count": old_class_count}
        if arm == D6 and ground_prior_identity is not None and old_class_count > 0:
            ground = np.asarray(ground_prior_identity, dtype=np.float64)
            if ground.shape != (old_class_count, 160):
                raise M24SafeResidualError("ground prior geometry drift")
            old_prior = np.concatenate([ground, support_center[:old_class_count, 160:]], axis=1)
            old_prior = old_prior / np.maximum(np.linalg.norm(old_prior, axis=1, keepdims=True), 1.0e-12)
            fused, prior_gate, prior_audit = gated_old_prior(
                decision_center[:old_class_count],
                old_prior,
                k_shot=k_shot,
                support_rows=base_rows[targets < old_class_count],
                support_targets=targets[targets < old_class_count],
            )
            decision_center = np.array(decision_center, copy=True)
            decision_center[:old_class_count] = fused

        residual = base_rows - covariance_center[targets]
        covariance_diagonal = np.sum(covariance_weights[:, None] * np.square(residual), axis=0)
        if arm == D7 and nuisance_covariance_identity is not None:
            nuisance = np.asarray(nuisance_covariance_identity, dtype=np.float64)
            if nuisance.shape != (160, 160):
                raise M24SafeResidualError("nuisance covariance geometry drift")
            covariance_diagonal[:160] += 0.1 * np.maximum(np.diag(nuisance), 0.0)
        repaired, covariance_audit = relative_psd_jitter(np.diag(covariance_diagonal), relative_floor=1.0e-4)
        covariance_diagonal = np.diag(repaired)
        precision_diagonal = 1.0 / covariance_diagonal
        coefficient_if = decision_center * precision_diagonal[None, :]
        bias_if = -0.5 * np.sum(decision_center * coefficient_if, axis=1)
        uncertainty_penalty = np.zeros(len(registry))
        if arm == D8:
            uncertainty = np.stack([
                np.mean(np.square(base_rows[targets == index] - support_center[index][None, :]), axis=0) / k_shot
                for index in range(len(registry))
            ])
            uncertainty_penalty = normalized_capped_penalty(uncertainty, np.diag(precision_diagonal), cap=0.2)
            bias_if -= uncertainty_penalty

        coefficient = coefficient_if
        bias = bias_if
        feature_dim = IF_DIM
        support_for_compile = base_rows
        rf_audit: Mapping[str, Any] = {"mode": "off", "alpha_max": 0.0}
        if arm in {D9, D10} and k_shot >= 5:
            rf_rows = blocks[:, IF_DIM:COMPACT_DIM]
            rf_rows = rf_rows / np.maximum(np.linalg.norm(rf_rows, axis=1, keepdims=True), 1.0e-12)
            rf_center = np.stack([np.mean(rf_rows[targets == index], axis=0) for index in range(len(registry))])
            rf_variance = np.mean(np.square(rf_rows - rf_center[targets]), axis=0)
            rf_variance = np.maximum(rf_variance, 1.0e-4 * max(float(np.mean(rf_variance)), 1.0e-12))
            raw_rf = rf_center / rf_variance[None, :]
            raw_rf_bias = -0.5 * np.sum(rf_center * raw_rf, axis=1)
            if arm == D9:
                rf_coefficient = 0.1 * raw_rf
                rf_bias = 0.1 * raw_rf_bias
                rf_audit = {"mode": "fixed_bounded_diag", "alpha_max": 0.1}
            else:
                rf_coefficient, rf_bias, rf_audit = safe_rf_residual(
                    coefficient_if,
                    bias_if,
                    raw_rf,
                    raw_rf_bias,
                    base_rows,
                    rf_rows,
                    targets,
                    k_shot=k_shot,
                    alpha_max=0.1,
                )
            coefficient = np.concatenate([coefficient_if, rf_coefficient], axis=1)
            bias = bias_if + rf_bias
            feature_dim = COMPACT_DIM
            support_for_compile = np.concatenate([base_rows, rf_rows], axis=1)

        safe, safety_counts = _candidate_is_safe(
            base_coefficient,
            base_bias,
            np.asarray(coefficient),
            np.asarray(bias),
            support_for_compile,
            targets,
        )
        if not safe:
            coefficient = base_coefficient
            bias = base_bias
            feature_dim = IF_DIM
            support_for_compile = base_rows
        safety = {**safety_counts, "whole_candidate_fallback_to_f1": not safe, "reason": "support_no_harm" if safe else "support_harm"}
        workspace = M24RegistrationWorkspace(
            support_center=support_center,
            decision_center=decision_center,
            covariance_center=covariance_center,
            support_weights=center_weights,
            covariance_diagonal=covariance_diagonal,
        )
        module_audit = {
            "quality_center_enabled": arm == D3,
            "quality_covariance_enabled": arm == D4,
            "if_residual_enabled": arm == D5,
            "prior_enabled": arm == D6,
            "nuisance_enabled": arm == D7,
            "uncertainty_enabled": arm == D8,
            "rf_enabled": arm in {D9, D10},
            "center_ess": center_ess,
            "covariance_ess": covariance_ess,
            "prior_gate_by_old_class": prior_gate.tolist(),
            "prior_audit": dict(prior_audit),
            "covariance_audit": covariance_audit,
            "uncertainty_penalty": uncertainty_penalty.tolist(),
            "rf_audit": dict(rf_audit),
        }

    state, resource, quantization = compile_m24_head(
        np.asarray(coefficient, dtype=np.float32),
        np.asarray(bias, dtype=np.float32),
        classes=registry,
        domain_digest=domain_digest,
        config_hash=arm_config_hash(arm),
        support_features=np.asarray(support_for_compile, dtype=np.float32),
        transient_workspace_bytes=workspace.nbytes,
        block_sizes=(160, 96) if feature_dim == IF_DIM else (160, 96, 10),
    )
    audit = MappingProxyType({
        "schema": "cvs.erbt_idr.m24.fit_audit.v1",
        "arm": arm,
        "k_shot": int(k_shot),
        "feature_dim": int(feature_dim),
        "query_rows_used": 0,
        "support_only": True,
        "whole_candidate_safety": safety,
        "modules": module_audit,
        "quantization": quantization,
        "resource": resource,
    })
    return state, audit, workspace


__all__ = [
    "COMPACT_DIM", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "M24_ARMS", "M24RegistrationWorkspace", "M24SafeResidualError", "arm_config_hash",
    "fit_m24_safe_residual", "prepare_query_features",
]
