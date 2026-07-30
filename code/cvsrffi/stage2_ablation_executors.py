"""Support-only numerical executors for the frozen Phase2 ablation catalog.

The module consumes feature rows that were already produced from sealed
received-IQ packages.  It never accepts query rows in ``fit_stage2_ablation``;
query features are passed only to the immutable fitted state's ``score`` or
``predict`` methods.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_ablation_quantization as quantization
from cvsrffi.stage2_ablation_factory import get_stage2_arm


class Stage2AblationExecutionError(RuntimeError):
    """Raised when an arm cannot satisfy its frozen numerical contract."""


def _rows(value: Any, *, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != d42.FEATURE_DIM
        or len(rows) == 0
        or not np.isfinite(rows).all()
    ):
        raise Stage2AblationExecutionError(f"{name} feature rows drift")
    return rows


def _registry(
    rows: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    *,
    name: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, int]:
    registry = tuple(str(value) for value in classes)
    label_rows = np.asarray([str(value) for value in labels])
    if (
        not registry
        or len(set(registry)) != len(registry)
        or label_rows.shape != (len(rows),)
        or set(label_rows.tolist()) != set(registry)
    ):
        raise Stage2AblationExecutionError(f"{name} registry drift")
    lookup = {value: index for index, value in enumerate(registry)}
    targets = np.asarray([lookup[value] for value in label_rows], dtype=np.int64)
    counts = np.bincount(targets, minlength=len(registry))
    if len(set(counts.tolist())) != 1 or int(counts.min()) <= 0:
        raise Stage2AblationExecutionError(f"{name} K-shot balance drift")
    return label_rows, registry, targets, int(counts[0])


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or bool(np.any(norm <= 1e-12)):
        raise Stage2AblationExecutionError("feature normalization is degenerate")
    return np.asarray(values / norm, dtype=np.float32)


def _class_means(
    rows: np.ndarray, targets: np.ndarray, class_count: int
) -> np.ndarray:
    return np.stack(
        [rows[targets == index].mean(axis=0) for index in range(class_count)]
    ).astype(np.float32)


def _affine_cosine(
    rows: np.ndarray, targets: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized = _normalize(rows)
    means = _normalize(_class_means(normalized, targets, class_count))
    return (
        means,
        np.zeros(class_count, dtype=np.float32),
        {"numerical_method": "cosine_nearest_centroid"},
    )


def _affine_euclidean(
    rows: np.ndarray, targets: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    means = _class_means(rows, targets, class_count).astype(np.float64)
    return (
        np.asarray(2.0 * means, dtype=np.float32),
        np.asarray(-np.sum(means**2, axis=1), dtype=np.float32),
        {"numerical_method": "euclidean_nearest_centroid"},
    )


def _affine_diagonal_lda(
    rows: np.ndarray, targets: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(rows, dtype=np.float64)
    means = _class_means(rows, targets, class_count).astype(np.float64)
    residual = values - means[targets]
    variance = np.mean(residual**2, axis=0)
    ridge = max(float(np.mean(variance)) * 1e-3, 1e-6)
    precision = 1.0 / (variance + ridge)
    coefficient = means * precision[None, :]
    intercept = -0.5 * np.sum(means * coefficient, axis=1)
    return (
        np.asarray(coefficient, dtype=np.float32),
        np.asarray(intercept, dtype=np.float32),
        {
            "numerical_method": "diagonal_lda_fixed_ridge",
            "diagonal_ridge": ridge,
        },
    )


def _affine_pooled_lw(
    rows: np.ndarray, targets: np.ndarray, class_count: int, k_shot: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    coefficient, intercept, audit = d42._fit_equal_prior_lda(
        rows, targets, class_count, k_shot
    )
    result = dict(audit)
    result["numerical_method"] = "pooled_ledoit_wolf_lda"
    return coefficient, intercept, result


def _identity_only_task_balanced(
    rows: np.ndarray,
    targets: np.ndarray,
    old_count: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = _normalize(rows[:, :160]).astype(np.float64)
    means = _class_means(values.astype(np.float32), targets, class_count).astype(
        np.float64
    )
    if class_count == old_count:
        residual = values - means[targets]
        covariance = np.cov(residual, rowvar=False, bias=True)
        status = "stage2b_identity_only"
    else:
        old_mask = targets < old_count
        new_mask = ~old_mask
        old_residual = values[old_mask] - means[targets[old_mask]]
        new_residual = values[new_mask] - means[targets[new_mask]]
        old_covariance = np.cov(old_residual, rowvar=False, bias=True)
        new_covariance = np.cov(new_residual, rowvar=False, bias=True)
        covariance = 0.5 * old_covariance + 0.5 * new_covariance
        status = "stage2c_identity_only_task_balanced"
    covariance = np.atleast_2d(np.asarray(covariance, dtype=np.float64))
    ridge = max(float(np.trace(covariance)) / 160.0 * 1e-3, 1e-6)
    covariance = 0.5 * (covariance + covariance.T) + ridge * np.eye(160)
    coefficient160 = np.linalg.solve(covariance, means.T).T
    intercept = -0.5 * np.diag(means @ coefficient160.T)
    coefficient = np.zeros((class_count, d42.FEATURE_DIM), dtype=np.float32)
    coefficient[:, :160] = coefficient160.astype(np.float32)
    return (
        coefficient,
        intercept.astype(np.float32),
        {
            "numerical_method": status,
            "feature_profile": "identity160_only",
            "ridge": ridge,
        },
    )


def _affine_trainable_adapter_head(
    rows: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    seed: int,
    device: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    """Fit a frozen-backbone low-rank adapter from support only."""

    from cvsrffi import (
        stage2_trainable_lowrank_support_adapter as adapter,
    )

    ranks = np.empty(len(labels), dtype=np.int64)
    next_rank = {value: 0 for value in set(labels.tolist())}
    for index, value in enumerate(labels.tolist()):
        ranks[index] = next_rank[value]
        next_rank[value] += 1
    hyperparameters = adapter.AdapterHyperparameters(
        candidate_id="p2_base_adapter_head_r8_e12",
        rank=8,
        epochs=12,
        learning_rate=0.02,
        temperature=0.10,
        prototype_weight=1.0,
        supervised_contrastive_weight=0.25,
        identity_weight=5.0,
        factor_weight=0.02,
        seed=int(seed),
    )
    u, v, gate, trace = adapter._train_once(
        {"fixed_received_iq": np.asarray(rows, dtype=np.float32)},
        labels,
        ranks,
        k_shot=int(k_shot),
        hyperparameters=hyperparameters,
        device=torch.device(device),
        trace_context={
            "phase": "locked_support_only_baseline_fit",
            "fold": None,
        },
    )
    adapted = adapter._adapt_numpy(
        np.asarray(rows, dtype=np.float32),
        u,
        v,
        gate,
    )
    coefficient, intercept, cosine_audit = _affine_cosine(
        adapted,
        targets,
        class_count,
    )
    parameter_count = int(u.size + v.size + gate.size)
    audit = {
        **cosine_audit,
        "numerical_method": (
            "frozen_backbone_lowrank_r8_adapter_equal_prior_head"
        ),
        "adapter_rank": 8,
        "adapter_epochs": 12,
        "adapter_trainable_parameters": parameter_count,
        "adapter_query_rows_used": 0,
        "adapter_support_only": True,
    }
    return (
        coefficient,
        intercept,
        u,
        v,
        gate,
        tuple(dict(row) for row in trace),
        audit,
    )


def _ground_inputs(
    ground_basis: Any | None,
    ground_spectral_weights: Any | None,
    ground_audit: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    basis = np.asarray(ground_basis, dtype=np.float64)
    weights = np.asarray(ground_spectral_weights, dtype=np.float64)
    audit = dict(ground_audit or {})
    if (
        basis.ndim != 2
        or basis.shape[0] != 160
        or weights.shape != (basis.shape[1],)
        or not np.isfinite(basis).all()
        or not np.isfinite(weights).all()
        or bool(np.any(weights <= 0))
    ):
        raise Stage2AblationExecutionError(
            "ground spectrum is required for this arm"
        )
    required = {
        "d81_basis_sha256",
        "d81_spectral_weight_sha256",
        "d81_participation_ratio_effective_rank",
        "d81_retained_rank",
        "d81_rank_policy",
        "ground_component_input_count",
        "ground_statistic_semantics",
    }
    if any(name not in audit for name in required):
        raise Stage2AblationExecutionError("ground spectrum audit drift")
    return basis, weights, audit


def _metric(
    old_rows: np.ndarray,
    old_targets: np.ndarray,
    old_count: int,
    *,
    enabled: bool,
    seed: int,
    device: Any,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...], dict[str, Any]]:
    if not enabled:
        return (
            np.zeros(d42.FEATURE_DIM, dtype=np.float32),
            (),
            {
                "metric_enabled": False,
                "metric_optimizer_steps": 0,
                "estimated_adaptation_macs": 0,
                "peak_cuda_memory_bytes": 0,
            },
        )
    log_diag, trace, resource = d42._fit_old_only_b3_metric(
        old_rows,
        old_targets,
        old_count,
        seed=int(seed),
        device=torch.device(device),
    )
    evidence = dict(resource)
    evidence["metric_enabled"] = True
    evidence["metric_optimizer_steps"] = len(trace)
    return log_diag, tuple(dict(row) for row in trace), evidence


def _component_builder(
    ablation_id: str,
    *,
    ground_basis: np.ndarray,
    ground_weights: np.ndarray,
    ground_audit: dict[str, Any],
):
    from scripts import probe_d43_structured_covariance as d43
    from scripts import probe_d44_full_block_rms_fusion as d44
    from scripts import probe_d46_classwise_loo_reliability_fusion as d46
    from scripts import probe_d62_crossfitted_fisher_row_splice as d62
    from scripts import probe_d81_ground_nuisance_cauchy_center as d81
    from scripts import probe_d92_registration_balanced_covariance as d92

    if ablation_id == "P2-B0":
        return (
            d92.build_d92_fit(
                d42,
                np.empty((160, 0), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                {},
                apply_ground_center=False,
                allow_fp32_centering_argmax_drift=True,
            )[0],
            "d92_d46_d62_without_ground_robust_center",
        )
    if ablation_id == "P2-C3":
        return (
            d81.build_d81_fit(
                d42,
                ground_basis,
                ground_weights,
                ground_audit,
                allow_fp32_centering_argmax_drift=True,
            )[0],
            "d81_all_classes_equal_covariance",
        )

    original_d62_builder = d92.d62.build_d62_fit
    try:
        if ablation_id == "P2-D0":
            d92.d62.build_d62_fit = lambda module, **_kwargs: (
                module._fit_equal_prior_lda,
                [],
            )
        elif ablation_id == "P2-D1":
            d92.d62.build_d62_fit = lambda module, **_kwargs: (
                d43.build_structured_fit(
                    module,
                    "block3_centered",
                    allow_fp32_centering_argmax_drift=True,
                ),
                [],
            )
        elif ablation_id == "P2-D2":
            d92.d62.build_d62_fit = lambda module, **_kwargs: (
                d44.build_full_block_rms_fit(
                    module,
                    allow_fp32_centering_argmax_drift=True,
                ),
                [],
            )
        elif ablation_id == "P2-E0":
            d92.d62.build_d62_fit = lambda module, **_kwargs: (
                d46.build_classwise_loo_reliability_fit(
                    module,
                    allow_fp32_centering_argmax_drift=True,
                ),
                [],
            )
        fit = d92.build_d92_fit(
            d42,
            ground_basis,
            ground_weights,
            ground_audit,
            allow_fp32_centering_argmax_drift=True,
        )[0]
    finally:
        d92.d62.build_d62_fit = original_d62_builder
    return fit, {
        "P2-D0": "d92_d81_full_only",
        "P2-D1": "d92_d81_block3_only",
        "P2-D2": "d92_d81_full_block_fixed_half",
        "P2-E0": "d92_d81_d46_without_fisher",
    }.get(ablation_id, "d92_d81_d46_d62_full")


def _fit_with_fp32_centering_audit(
    fit: Any,
    transformed: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Record the explicit per-fit FP32 gauge-roundoff policy."""

    coefficient, intercept, audit = fit(
        transformed, targets, class_count, k_shot
    )
    evidence = dict(audit)
    evidence["stage2_ablation_fp32_centering_argmax_drift_allowed"] = True
    return coefficient, intercept, evidence


@dataclass(frozen=True)
class Stage2AblationFittedState:
    ablation_id: str
    stage: str
    classes: tuple[str, ...]
    old_class_count: int
    feature_profile: str
    score_kind: str
    log_diag_fp32: np.ndarray
    coefficient_fp32: np.ndarray
    intercept_fp32: np.ndarray
    compiled_affine_state: quantization.CompiledAffineState | None
    support_bank_fp32: np.ndarray
    support_targets: np.ndarray
    adapter_u_fp32: np.ndarray
    adapter_v_fp32: np.ndarray
    adapter_gate_fp32: np.ndarray
    audit: Mapping[str, Any]
    resource: Mapping[str, Any]
    training_trace: tuple[Mapping[str, Any], ...]

    def _prepared(self, features: Any) -> np.ndarray:
        values = _rows(features, name="query")
        if self.feature_profile == "identity160_only":
            prepared = np.zeros_like(values)
            prepared[:, :160] = _normalize(values[:, :160])
        else:
            prepared = values
        prepared = d42._transform(prepared, self.log_diag_fp32)
        if self.score_kind == "adapter_cosine_affine":
            from cvsrffi import (
                stage2_trainable_lowrank_support_adapter as adapter,
            )

            prepared = adapter._adapt_numpy(
                prepared,
                self.adapter_u_fp32,
                self.adapter_v_fp32,
                self.adapter_gate_fp32,
            )
        return prepared

    def score(self, features: Any) -> np.ndarray:
        rows = self._prepared(features)
        if self.score_kind == "compiled_affine":
            if self.compiled_affine_state is None:
                raise Stage2AblationExecutionError(
                    "compiled affine state is missing"
                )
            scores = quantization.score_affine_state(
                self.compiled_affine_state, rows
            )
        elif self.score_kind == "affine":
            scores = (
                rows @ self.coefficient_fp32.T
                + self.intercept_fp32[None, :]
            )
        elif self.score_kind in {
            "cosine_affine",
            "adapter_cosine_affine",
        }:
            scores = (
                _normalize(rows) @ self.coefficient_fp32.T
                + self.intercept_fp32[None, :]
            )
        elif self.score_kind == "qknn":
            query = _normalize(rows)
            bank = _normalize(self.support_bank_fp32)
            pairwise = query @ bank.T
            scores = np.stack(
                [
                    np.max(pairwise[:, self.support_targets == index], axis=1)
                    for index in range(len(self.classes))
                ],
                axis=1,
            )
        else:
            raise Stage2AblationExecutionError("unknown fitted score kind")
        if scores.shape != (len(rows), len(self.classes)) or not np.isfinite(
            scores
        ).all():
            raise Stage2AblationExecutionError("query scoring drift")
        return np.asarray(scores, dtype=np.float32)

    def predict(self, features: Any) -> np.ndarray:
        scores = self.score(features)
        return np.asarray(self.classes)[np.argmax(scores, axis=1)]


def fit_stage2_ablation(
    *,
    ablation_id: str,
    old_support_features: Any | None,
    old_support_labels: Sequence[str] | None,
    old_classes: Sequence[str],
    new_support_features: Any | None = None,
    new_support_labels: Sequence[str] | None = None,
    new_classes: Sequence[str] = (),
    deployment_prototypes: Any | None = None,
    ground_basis: Any | None = None,
    ground_spectral_weights: Any | None = None,
    ground_audit: Mapping[str, Any] | None = None,
    seed: int,
    device: Any = "cpu",
) -> Stage2AblationFittedState:
    """Fit one frozen arm from deployment state and legal support only."""

    spec = get_stage2_arm(ablation_id)
    old_registry = tuple(str(value) for value in old_classes)
    if spec.stage == "stage2a":
        prototypes = _rows(deployment_prototypes, name="deployment prototype")
        if len(prototypes) != len(old_registry) or len(old_registry) < 2:
            raise Stage2AblationExecutionError("Stage2-A deployment registry drift")
        coefficient = _normalize(prototypes)
        return Stage2AblationFittedState(
            ablation_id=ablation_id,
            stage=spec.stage,
            classes=old_registry,
            old_class_count=len(old_registry),
            feature_profile="full288",
            score_kind="cosine_affine",
            log_diag_fp32=np.zeros(d42.FEATURE_DIM, dtype=np.float32),
            coefficient_fp32=coefficient,
            intercept_fp32=np.zeros(len(old_registry), dtype=np.float32),
            compiled_affine_state=None,
            support_bank_fp32=np.empty((0, d42.FEATURE_DIM), dtype=np.float32),
            support_targets=np.empty(0, dtype=np.int64),
            adapter_u_fp32=np.empty(
                (d42.FEATURE_DIM, 0), dtype=np.float32
            ),
            adapter_v_fp32=np.empty(
                (d42.FEATURE_DIM, 0), dtype=np.float32
            ),
            adapter_gate_fp32=np.empty(0, dtype=np.float32),
            audit={
                "support_only": True,
                "target_support_rows_used": 0,
                "query_rows_used": 0,
                "numerical_method": "frozen_phase1_cosine_prototype",
            },
            resource={
                "optimizer_steps": 0,
                "trainable_parameters": 0,
                "persistent_state_bytes": int(coefficient.nbytes),
            },
            training_trace=(),
        )

    old_rows = _rows(old_support_features, name="old support")
    _, old_registry, old_targets, old_k = _registry(
        old_rows,
        () if old_support_labels is None else old_support_labels,
        old_registry,
        name="old support",
    )
    if spec.stage == "stage2c":
        new_rows = _rows(new_support_features, name="new support")
        _, new_registry, new_targets, new_k = _registry(
            new_rows,
            () if new_support_labels is None else new_support_labels,
            new_classes,
            name="new support",
        )
        if old_k != new_k or set(old_registry) & set(new_registry):
            raise Stage2AblationExecutionError("Stage2-C old/new registry drift")
        rows = np.concatenate([old_rows, new_rows], axis=0)
        targets = np.concatenate(
            [old_targets, new_targets + len(old_registry)]
        )
        classes = old_registry + new_registry
        k_shot = old_k
    else:
        rows, targets, classes, k_shot = (
            old_rows,
            old_targets,
            old_registry,
            old_k,
        )

    feature_profile = (
        "identity160_only" if ablation_id == "P2-A0" else "full288"
    )
    if feature_profile == "identity160_only":
        model_rows = np.zeros_like(rows)
        model_rows[:, :160] = _normalize(rows[:, :160])
        model_old_rows = np.zeros_like(old_rows)
        model_old_rows[:, :160] = _normalize(old_rows[:, :160])
    else:
        model_rows = rows
        model_old_rows = old_rows
    metric_enabled = ablation_id not in {
        "P2-S2B-PROTO",
        "P2-S2B-DIAGOFF",
        "P2-BASE-COSINE",
        "P2-BASE-EUCLIDEAN",
        "P2-BASE-QKNN",
        "P2-BASE-DIAG-LDA",
        "P2-BASE-POOLED-LW-LDA",
        "P2-BASE-FULL-BLOCK-LDA",
        "P2-BASE-ADAPTER-HEAD",
    }
    log_diag, trace, metric_resource = _metric(
        model_old_rows,
        old_targets,
        len(old_registry),
        enabled=metric_enabled,
        seed=seed,
        device=device,
    )
    transformed = d42._transform(model_rows, log_diag)
    score_kind = "affine"
    support_bank = np.empty((0, d42.FEATURE_DIM), dtype=np.float32)
    support_targets = np.empty(0, dtype=np.int64)
    adapter_u = np.empty((d42.FEATURE_DIM, 0), dtype=np.float32)
    adapter_v = np.empty((d42.FEATURE_DIM, 0), dtype=np.float32)
    adapter_gate = np.empty(0, dtype=np.float32)
    started = time.perf_counter()

    if ablation_id in {"P2-S2B-PROTO", "P2-BASE-COSINE"}:
        coefficient, intercept, audit = _affine_cosine(
            transformed, targets, len(classes)
        )
        score_kind = "cosine_affine"
    elif ablation_id == "P2-BASE-EUCLIDEAN":
        coefficient, intercept, audit = _affine_euclidean(
            transformed, targets, len(classes)
        )
    elif ablation_id == "P2-BASE-QKNN":
        coefficient = np.empty((0, d42.FEATURE_DIM), dtype=np.float32)
        intercept = np.empty(0, dtype=np.float32)
        support_bank = transformed.astype(np.float32)
        support_targets = targets.astype(np.int64)
        score_kind = "qknn"
        audit = {"numerical_method": "single_qknn_top1_cosine"}
    elif ablation_id == "P2-BASE-DIAG-LDA":
        coefficient, intercept, audit = _affine_diagonal_lda(
            transformed, targets, len(classes)
        )
    elif ablation_id == "P2-BASE-POOLED-LW-LDA":
        coefficient, intercept, audit = _affine_pooled_lw(
            transformed, targets, len(classes), k_shot
        )
    elif ablation_id == "P2-BASE-ADAPTER-HEAD":
        labels = (
            np.concatenate(
                [
                    np.asarray(old_support_labels).astype(str),
                    np.asarray(new_support_labels).astype(str),
                ]
            )
            if spec.stage == "stage2c"
            else np.asarray(old_support_labels).astype(str)
        )
        (
            coefficient,
            intercept,
            adapter_u,
            adapter_v,
            adapter_gate,
            adapter_trace,
            audit,
        ) = _affine_trainable_adapter_head(
            transformed,
            labels,
            targets,
            len(classes),
            k_shot,
            seed=seed,
            device=device,
        )
        trace = tuple(trace) + tuple(adapter_trace)
        score_kind = "adapter_cosine_affine"
        metric_resource = {
            **metric_resource,
            "metric_enabled": False,
            "optimizer_steps": 12,
            "trainable_parameters": int(
                adapter_u.size + adapter_v.size + adapter_gate.size
            ),
            "estimated_adaptation_macs": int(
                12 * len(rows) * 2 * d42.FEATURE_DIM * 8
            ),
        }
    else:
        needs_ground = ablation_id not in {
            "P2-BASE-FULL-BLOCK-LDA",
            "P2-BASE-ADAPTER-HEAD",
            "P2-B0",
        }
        if needs_ground:
            basis, weights, basis_audit = _ground_inputs(
                ground_basis, ground_spectral_weights, ground_audit
            )
        else:
            basis = np.empty((160, 0), dtype=np.float64)
            weights = np.empty(0, dtype=np.float64)
            basis_audit = {}
        if ablation_id == "P2-BASE-FULL-BLOCK-LDA":
            from scripts import probe_d46_classwise_loo_reliability_fusion as d46

            fit = d46.build_classwise_loo_reliability_fit(
                d42,
                allow_fp32_centering_argmax_drift=True,
            )
            method = "full_block_shrinkage_lda_no_robust_center"
        elif ablation_id == "P2-S2B-DIAGOFF":
            from scripts import probe_d81_ground_nuisance_cauchy_center as d81

            fit = d81.build_d81_fit(
                d42,
                basis,
                weights,
                basis_audit,
                allow_fp32_centering_argmax_drift=True,
            )[0]
            method = "stage2b_d81_diag_metric_off"
        else:
            physical_id = "P2-FULL" if ablation_id == "P2-F3" else ablation_id
            fit, method = _component_builder(
                physical_id,
                ground_basis=basis,
                ground_weights=weights,
                ground_audit=basis_audit,
            )
        coefficient, intercept, audit = _fit_with_fp32_centering_audit(
            fit, transformed, targets, len(classes), k_shot
        )
        audit["numerical_method"] = method

    fit_seconds = time.perf_counter() - started
    quantization_arm = (
        quantization.F3
        if ablation_id
        in {
            "P2-FULL",
            "P2-F3",
            "P2-A0",
            "P2-B0",
            "P2-C3",
            "P2-D0",
            "P2-D1",
            "P2-D2",
            "P2-E0",
        }
        else ablation_id
        if ablation_id in quantization.SUPPORTED_ARMS
        else None
    )
    compiled_affine_state = None
    stored_coefficient = np.asarray(coefficient, dtype=np.float32)
    stored_intercept = np.asarray(intercept, dtype=np.float32)
    if quantization_arm is not None:
        if score_kind != "affine":
            raise Stage2AblationExecutionError(
                "quantization arms require an affine numerical head"
            )
        compiled_affine_state = quantization.compile_affine_state(
            stored_coefficient,
            stored_intercept,
            arm_id=quantization_arm,
        )
        stored_coefficient = np.empty(
            (0, d42.FEATURE_DIM), dtype=np.float32
        )
        stored_intercept = np.empty(0, dtype=np.float32)
        score_kind = "compiled_affine"

    persistent_head_bytes = (
        compiled_affine_state.state_bytes
        if compiled_affine_state is not None
        else int(stored_coefficient.nbytes + stored_intercept.nbytes)
    )
    persistent_bytes = int(
        persistent_head_bytes
        + log_diag.nbytes
        + support_bank.nbytes
        + support_targets.nbytes
        + adapter_u.nbytes
        + adapter_v.nbytes
        + adapter_gate.nbytes
    )
    merged_audit = dict(audit)
    merged_audit.update(
        {
            "ablation_id": ablation_id,
            "stage": spec.stage,
            "support_only": True,
            "query_rows_used": 0,
            "old_k_shot": old_k,
            "new_k_shot": k_shot if spec.stage == "stage2c" else 0,
            "registered_class_count": len(classes),
            "class_id_specific_branch": False,
            "label_permutation_equivariant": True,
            "quantization_arm": quantization_arm,
            "has_fp32_coefficient_sidecar": (
                compiled_affine_state.has_fp32_coefficient_sidecar
                if compiled_affine_state is not None
                else False
            ),
        }
    )
    resource = {
        **metric_resource,
        "fit_seconds": float(fit_seconds),
        "persistent_state_bytes": persistent_bytes,
        "persistent_head_state_bytes": persistent_head_bytes,
        "quantization_arm": quantization_arm,
        "integer_kernel_used": False,
        "formal_int8_acceleration_claim_allowed": False,
        "deployment_claim": (
            "storage_compression_only"
            if quantization_arm in {quantization.F2, quantization.F3}
            else "not_an_int8_acceleration_claim"
        ),
        "registered_class_count": len(classes),
        "old_k_shot": old_k,
        "new_k_shot": k_shot if spec.stage == "stage2c" else 0,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
    }
    return Stage2AblationFittedState(
        ablation_id=ablation_id,
        stage=spec.stage,
        classes=classes,
        old_class_count=len(old_registry),
        feature_profile=feature_profile,
        score_kind=score_kind,
        log_diag_fp32=np.asarray(log_diag, dtype=np.float32),
        coefficient_fp32=stored_coefficient,
        intercept_fp32=stored_intercept,
        compiled_affine_state=compiled_affine_state,
        support_bank_fp32=np.asarray(support_bank, dtype=np.float32),
        support_targets=np.asarray(support_targets, dtype=np.int64),
        adapter_u_fp32=np.asarray(adapter_u, dtype=np.float32),
        adapter_v_fp32=np.asarray(adapter_v, dtype=np.float32),
        adapter_gate_fp32=np.asarray(adapter_gate, dtype=np.float32),
        audit=merged_audit,
        resource=resource,
        training_trace=trace,
    )


__all__ = [
    "Stage2AblationExecutionError",
    "Stage2AblationFittedState",
    "fit_stage2_ablation",
]
