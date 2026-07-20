"""D93 paired-ground transport evaluation on sealed p2_min_v1 row pairs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d93_paired_ground_transport import (
    D93PairedGroundTransport,
    fit_paired_ground_transport,
    transform_registered_features,
)


CANDIDATE_D93_INTERACTION = "d93_paired_ground_transport_interaction"
CANDIDATE_D93_SCALE = "d93_paired_ground_transport_scale"
CANDIDATES_D93 = (CANDIDATE_D93_INTERACTION, CANDIDATE_D93_SCALE)
CANDIDATE_D94_COVERAGE = "d94_paired_ground_transport_coverage_shrink"
CANDIDATES_D94 = (CANDIDATE_D94_COVERAGE,)
CANDIDATES_GROUND_TRANSPORT = (*CANDIDATES_D93, *CANDIDATES_D94)
SCHEMA_D93 = "cvs.phase2.d93.full_query_evaluation.v1"
SCHEMA_D94 = "cvs.phase2.d94.full_query_evaluation.v1"


class D93QueryEvaluationError(ValueError):
    """Raised when the D93 support-only or deployment closure drifts."""


@dataclass(frozen=True)
class D93State:
    base_state: d42.D42UnifiedShrinkageLDAState
    transport: D93PairedGroundTransport

    @property
    def persistent_state_bytes(self) -> int:
        return int(
            self.base_state.persistent_state_bytes
            + self.transport.incremental_state_bytes
        )

    @property
    def registry_state_bytes(self) -> int:
        return int(self.base_state.registry_state_bytes)

    @property
    def classes(self) -> tuple[str, ...]:
        return self.base_state.classes

    @property
    def old_class_count(self) -> int:
        return int(self.base_state.old_class_count)


@dataclass(frozen=True)
class D93Result:
    before_state: D93State
    state: D93State
    matched_fp32_before_state: D93State
    matched_fp32_state: D93State
    training_trace: tuple[dict[str, Any], ...]
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


def _normalize(rows: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float64)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if (
        value.ndim != 2
        or not np.isfinite(value).all()
        or not np.isfinite(norm).all()
        or bool(np.any(norm <= 1.0e-10))
    ):
        raise D93QueryEvaluationError(f"D93 {name} drift")
    return value / norm


def _old_targets(labels: Sequence[str], registry: Sequence[str]) -> np.ndarray:
    classes = tuple(str(item) for item in registry)
    index = {handle: position for position, handle in enumerate(classes)}
    values = np.asarray([index.get(str(label), -1) for label in labels], dtype=np.int64)
    if bool(np.any(values < 0)) or not np.array_equal(
        np.unique(values), np.arange(len(classes), dtype=np.int64)
    ):
        raise D93QueryEvaluationError("D93 target-old registry drift")
    return values


def _build_guarded_d93_base_fit(
    module: Any, d62_probe: Any
) -> tuple[Callable[..., Any], list[dict[str, Any]]]:
    """Use D62 unless transformed support triggers its exact D43 PD failure."""

    locked_d42_fit = module._fit_equal_prior_lda
    primary_fit, call_records = d62_probe.build_d62_fit(module)

    def fit(*args: Any, **kwargs: Any):
        try:
            return primary_fit(*args, **kwargs)
        except RuntimeError as exc:
            if (
                exc.__class__.__name__ != "D43ProbeError"
                or str(exc) != "D43 structured covariance is not positive definite"
            ):
                raise
            coefficients, intercept, audit = locked_d42_fit(*args, **kwargs)
            guarded_audit = dict(audit)
            guarded_audit.update(
                {
                    "d93_support_covariance_fallback": (
                        "locked_d42_auto_shrinkage_on_exact_d43_nonpositive_definite"
                    ),
                    "d93_d43_nonpositive_definite_observed": True,
                    "d93_fallback_query_rows_used": 0,
                }
            )
            return coefficients, intercept, guarded_audit

    return fit, call_records


def build_d93_top_level_fit(
    base_top_level_fit: Callable[..., d42.D42UnifiedShrinkageLDAResult],
    *,
    ground_prototypes: np.ndarray,
    ground_mask: np.ndarray,
    ground_classes: Sequence[str],
    target_old_tx_labels: Sequence[str] | None = None,
    ground_audit: Mapping[str, Any],
    include_nuisance_scale: bool,
    coverage_controlled_update: bool = False,
) -> Callable[..., D93Result]:
    """Wrap the existing D42/D62 top-level fitter with the D93 transport."""

    locked_ground_classes = tuple(str(item) for item in ground_classes)
    locked_target_old_tx_labels = (
        tuple(str(item) for item in target_old_tx_labels)
        if target_old_tx_labels is not None
        else locked_ground_classes
    )

    def fit(
        old_support_features: np.ndarray,
        old_support_labels: Sequence[str],
        old_classes: Sequence[str],
        new_support_features: np.ndarray,
        new_support_labels: Sequence[str],
        new_classes: Sequence[str],
        *,
        seed: int,
        device: Any = "cpu",
        config: Any = None,
    ) -> D93Result:
        old_registry = tuple(str(item) for item in old_classes)
        if (
            len(set(old_registry)) != len(old_registry)
            or len(set(locked_ground_classes)) != len(locked_ground_classes)
            or len(set(locked_target_old_tx_labels))
            != len(locked_target_old_tx_labels)
            or len(old_registry) != len(locked_target_old_tx_labels)
            or set(locked_target_old_tx_labels) != set(locked_ground_classes)
        ):
            raise D93QueryEvaluationError("D93 ground/target-old class binding drift")
        ground_index = {handle: index for index, handle in enumerate(locked_ground_classes)}
        ground_order = np.asarray(
            [ground_index[handle] for handle in locked_target_old_tx_labels],
            dtype=np.int64,
        )
        ordered_ground_prototypes = np.asarray(ground_prototypes)[:, ground_order]
        ordered_ground_mask = np.asarray(ground_mask)[:, ground_order]
        old_rows = np.asarray(old_support_features, dtype=np.float32)
        new_rows = np.asarray(new_support_features, dtype=np.float32)
        if (
            old_rows.ndim != 2
            or new_rows.ndim != 2
            or old_rows.shape[1] != 288
            or new_rows.shape[1] != 288
        ):
            raise D93QueryEvaluationError("D93 registered support feature drift")
        old_z160 = _normalize(old_rows[:, :160], "target-old z160").astype(np.float32)
        old_targets = _old_targets(old_support_labels, old_registry)
        transport = fit_paired_ground_transport(
            ordered_ground_prototypes,
            ordered_ground_mask,
            old_z160,
            old_targets,
            include_nuisance_scale=bool(include_nuisance_scale),
            coverage_controlled_update=bool(coverage_controlled_update),
        )
        transformed_old = transform_registered_features(old_rows, transport)
        transformed_new = transform_registered_features(new_rows, transport)
        base = base_top_level_fit(
            transformed_old,
            old_support_labels,
            old_registry,
            transformed_new,
            new_support_labels,
            tuple(str(item) for item in new_classes),
            seed=int(seed),
            device=device,
            config=config,
        )
        transport_audit = {
            **dict(transport.audit),
            "ground_component_manifest_sha256": str(
                ground_audit["component_manifest_sha256"]
            ),
            "ground_component_npz_sha256": str(
                ground_audit["component_npz_sha256"]
            ),
            "ground_component_logical_state_bytes": int(
                ground_audit["ground_int8_component_logical_state_bytes"]
            ),
            "ground_registry_reordered_to_target_old": bool(
                locked_target_old_tx_labels != locked_ground_classes
            ),
            "ground_to_target_binding_policy": (
                "registered_target_old_tx_label_order_to_opaque_class_index"
            ),
            "ground_to_target_binding_class_count": len(old_registry),
            "target_old_new_final_prototypes_from_target_support_only": True,
            "target_support_input_is_fixed_received_iq_only": True,
            "target_clean_iq_access": False,
            "target_new_clean_iq_access": False,
            "same_physical_iq_multi_channel_views": False,
            "phase2_channel_simulator_calls": 0,
            "transport_applied_to_target_old": True,
            "transport_applied_to_target_new": True,
            "transport_applied_to_query_by_state": True,
            "fft96_rf32_transport_policy": "identity_same_received_iq_auxiliary",
        }
        geometry = {
            **dict(base.geometry_audit),
            "schema": "cvs.phase2.d93.paired_ground_transport_geometry.v1",
            "d93_transport_audit": transport_audit,
            "d93_ground_direct_query_score_access": False,
            "d93_target_prototype_overwrite_from_ground": False,
            "d93_label_permutation_equivariant": True,
        }
        resource = dict(base.resource_audit)
        resource.update(
            {
                "schema": "cvs.phase2.d93.paired_ground_transport_resource.v1",
                "trainable_parameters": int(
                    resource["trainable_parameters"] + transport.parameter_count
                ),
                "d93_closed_form_parameter_count": int(transport.parameter_count),
                "d93_optimizer_steps": 0,
                "persistent_state_bytes": int(
                    base.state.persistent_state_bytes
                    + transport.incremental_state_bytes
                ),
                "d93_incremental_state_bytes": int(
                    transport.incremental_state_bytes
                ),
                "estimated_macs_per_query": int(
                    resource["estimated_macs_per_query"]
                    + transport.macs_per_query
                ),
                "d93_transport_macs_per_query": int(transport.macs_per_query),
                "ground_int8_component_input_count": int(
                    ground_audit["ground_component_input_count"]
                ),
                "ground_int8_update_access": False,
                "ground_aggregate_prototypes_only": True,
                "target_clean_iq_access": False,
                "same_physical_iq_multi_channel_views": False,
                "phase2_channel_simulator_calls": 0,
                "query_rows_used_for_transport_fit": 0,
                "query_dependent_batch_optimization": False,
                "dense_query_graph_bytes": 0,
                "formal_target_vectors_int8_no_fp32_sidecar": True,
            }
        )
        trace = (
            {
                "stage": "d93_closed_form_paired_ground_transport",
                "optimizer_step": 0,
                "mode": transport.audit["mode"],
                "paired_rmse": transport.audit["paired_rmse"],
                "translation_only_rmse": transport.audit[
                    "translation_only_rmse"
                ],
                "operator_condition_number": transport.audit[
                    "operator_condition_number"
                ],
                "query_rows_used": 0,
            },
            *tuple(dict(row) for row in base.training_trace),
        )
        return D93Result(
            before_state=D93State(base.before_state, transport),
            state=D93State(base.state, transport),
            matched_fp32_before_state=D93State(
                base.matched_fp32_before_state, transport
            ),
            matched_fp32_state=D93State(base.matched_fp32_state, transport),
            training_trace=trace,
            geometry_audit=geometry,
            resource_audit=resource,
        )

    return fit


def score_d93(state: D93State, features: np.ndarray) -> np.ndarray:
    if not isinstance(state, D93State):
        raise D93QueryEvaluationError("D93 score state drift")
    transformed = transform_registered_features(features, state.transport)
    return d42.score_d42_unified_shrinkage_lda(state.base_state, transformed)


def predict_d93(state: D93State, features: np.ndarray) -> np.ndarray:
    return np.asarray(state.classes)[np.argmax(score_d93(state, features), axis=1)]


def _load_ground_component(
    component_dir: Path, manifest_sha256: str
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    from scripts import probe_d66_ground_domain_reliability_residual as d66

    _, audit = d66.load_ground_domain_reliability(
        Path(component_dir), str(manifest_sha256), 288
    )
    npz_path = Path(component_dir) / d66.NPZ_NAME
    with np.load(npz_path, allow_pickle=False) as payload:
        q = np.asarray(payload["domain_class_q"], dtype=np.int8)
        scales = np.asarray(payload["domain_class_scale"], dtype=np.float16)
        mask = np.asarray(payload["domain_class_mask"], dtype=np.uint8)
        domains = np.asarray(payload["domain_registry"]).astype(str)
        classes = np.asarray(payload["class_registry"]).astype(str)
    domain_order = np.argsort(domains, kind="stable")
    class_order = np.argsort(classes, kind="stable")
    prototypes = (
        q[domain_order][:, class_order].astype(np.float32)
        * scales[domain_order][:, class_order].astype(np.float32)[..., None]
    )
    ordered_mask = mask[domain_order][:, class_order]
    ordered_classes = tuple(classes[class_order].tolist())
    if int(ordered_mask.sum()) != int(audit["ground_active_domain_class_cells"]):
        raise D93QueryEvaluationError("D93 ground active-cell drift")
    return (
        prototypes,
        ordered_mask,
        ordered_classes,
        {
            **dict(audit),
            "ground_component_input_count": int(ordered_mask.sum()),
            "component_npz_sha256": str(audit["component_npz_sha256"]),
        },
    )


def _audit_fit(
    result: D93Result,
    *,
    scenario: str,
    k_shot: int,
    old_count: int,
    class_count: int,
) -> dict[str, Any]:
    transport = result.geometry_audit.get("d93_transport_audit", {})
    if (
        transport.get("ground_to_target_identity_pairing_used") is not True
        or transport.get("ground_direct_query_score_access") is not False
        or transport.get("ground_aggregate_prototypes_only") is not True
        or transport.get("target_clean_iq_access") is not False
        or transport.get("target_new_clean_iq_access") is not False
        or transport.get("same_physical_iq_multi_channel_views") is not False
        or int(transport.get("phase2_channel_simulator_calls", -1)) != 0
        or transport.get("target_old_new_final_prototypes_from_target_support_only")
        is not True
        or int(transport.get("query_rows_used", -1)) != 0
        or int(transport.get("k_shot", -1)) != int(k_shot)
        or int(transport.get("target_old_class_count", -1)) != int(old_count)
        or float(transport.get("operator_condition_number", np.inf)) > 3.1
    ):
        raise D93QueryEvaluationError("D93 support-only transport closure drift")
    before = result.geometry_audit["before_covariance_audit"]
    after = result.geometry_audit["final_covariance_audit"]
    return {
        "scenario": str(scenario),
        "k_shot": int(k_shot),
        "old_class_count": int(old_count),
        "registered_class_count": int(class_count),
        "k1_unit_covariance_fallback": bool(
            result.geometry_audit["k1_unit_covariance_fallback"]
        ),
        "before_covariance_policy": str(before.get("covariance_policy")),
        "after_covariance_policy": str(after.get("covariance_policy")),
        # Compatibility field for the D81 publication scaffold only.  D93's
        # non-identity behavior is recorded by operator_update_spectral_norm.
        "before_center_shift_l2_max": 0.0,
        "after_center_shift_l2_max": 0.0,
        "d93_transport_mode": str(transport["mode"]),
        "d93_operator_update_spectral_norm": float(
            transport["update_spectral_norm"]
        ),
        "d93_operator_condition_number": float(
            transport["operator_condition_number"]
        ),
        "d93_paired_rmse": float(transport["paired_rmse"]),
        "d93_translation_only_rmse": float(
            transport["translation_only_rmse"]
        ),
        "d93_target_shift_ground_nuisance_coverage": float(
            transport["target_shift_ground_nuisance_coverage"]
        ),
        "d93_target_shift_out_of_ground_nuisance_energy_ratio": float(
            transport["target_shift_out_of_ground_nuisance_energy_ratio"]
        ),
        "d93_ground_effective_domain_count_by_class": list(
            transport["ground_effective_domain_count_by_class"]
        ),
        "d93_ground_stable_rank_by_class": list(
            transport["ground_stable_rank_by_class"]
        ),
        "d93_ground_near_duplicate_pair_fraction_by_class": list(
            transport[
                "ground_near_duplicate_pair_fraction_cosine_eps_1e_4_by_class"
            ]
        ),
        "d93_ground_nuisance_participation_ratio": float(
            transport["nuisance_participation_ratio"]
        ),
        "d93_ground_nuisance_retained_rank": int(
            transport["nuisance_retained_rank"]
        ),
        "d93_ground_to_target_binding_policy": str(
            transport["ground_to_target_binding_policy"]
        ),
        "d93_k1_nonidentity": bool(
            int(k_shot) != 1
            or float(transport["update_spectral_norm"]) > 1.0e-8
            or float(np.linalg.norm(result.state.transport.translation_fp32))
            > 1.0e-8
        ),
        "before_state_bytes": int(result.before_state.persistent_state_bytes),
        "after_state_bytes": int(result.state.persistent_state_bytes),
        "training_trace": [dict(row) for row in result.training_trace],
        "resource_audit": dict(result.resource_audit),
    }


def run_d93_query_evaluation(
    *, candidate: str, target_old_tx_labels: Sequence[str], **kwargs: Any
) -> dict[str, Any]:
    """Reuse the sealed D81 I/O scaffold while replacing its method hooks."""

    if str(candidate) not in CANDIDATES_GROUND_TRANSPORT:
        raise D93QueryEvaluationError("unknown D93/D94 candidate")
    from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe
    from scripts import probe_d62_crossfitted_fisher_row_splice as d62_probe
    from cvsrffi import stage2_d81_query_evaluation as d81_eval

    include_scale = str(candidate) == CANDIDATE_D93_SCALE
    coverage_controlled_update = str(candidate) == CANDIDATE_D94_COVERAGE
    selected_schema = SCHEMA_D94 if coverage_controlled_update else SCHEMA_D93
    locked_target_old_tx_labels = tuple(str(item) for item in target_old_tx_labels)
    if (
        not locked_target_old_tx_labels
        or len(set(locked_target_old_tx_labels)) != len(locked_target_old_tx_labels)
    ):
        raise D93QueryEvaluationError("D93 target-old TX registry drift")
    holder: dict[str, Any] = {}
    original = {
        "load": d81_probe.load_ground_basis,
        "build": d81_probe.build_d81_fit,
        "top_fit": d81_eval.fit_d42_unified_shrinkage_lda,
        "predict": d81_eval.predict_d42_unified_shrinkage_lda,
        "audit": d81_eval._audit_fit,
        "publish": d81_eval._publish_state,
        "candidate": d81_eval.CANDIDATE_D81,
        "schema": d81_eval.SCHEMA,
    }

    def load(component_dir: Path, manifest_sha: str, feature_dim: int):
        if int(feature_dim) != 288:
            raise D93QueryEvaluationError("D93 feature dimension drift")
        prototypes, mask, classes, audit = _load_ground_component(
            component_dir, manifest_sha
        )
        holder.update(
            {
                "prototypes": prototypes,
                "mask": mask,
                "classes": classes,
                "audit": audit,
            }
        )
        return prototypes, mask, audit

    def build(module: Any, _prototypes: Any, _mask: Any, _audit: Any):
        base_fit, call_records = _build_guarded_d93_base_fit(module, d62_probe)
        holder["base_fit"] = base_fit
        return base_fit, call_records, []

    def top_fit(*args: Any, **fit_kwargs: Any):
        required = {"prototypes", "mask", "classes", "audit"}
        if not required <= set(holder):
            raise D93QueryEvaluationError("D93 ground loader ordering drift")
        wrapper = build_d93_top_level_fit(
            original["top_fit"],
            ground_prototypes=holder["prototypes"],
            ground_mask=holder["mask"],
            ground_classes=holder["classes"],
            target_old_tx_labels=locked_target_old_tx_labels,
            ground_audit=holder["audit"],
            include_nuisance_scale=include_scale,
            coverage_controlled_update=coverage_controlled_update,
        )
        return wrapper(*args, **fit_kwargs)

    def publish(*args: Any, **publish_kwargs: Any):
        resource = dict(publish_kwargs["resource"])
        resource.update(
            {
                "k1_strict_identity_pass": False,
                "d93_k1_nonidentity_required": True,
                "d93_candidate": str(candidate),
                "query_extra_macs_for_ground_component": int(
                    max(
                        row["resource_audit"]["d93_transport_macs_per_query"]
                        for row in publish_kwargs["fit_audit"]
                    )
                ),
            }
        )
        publish_kwargs["resource"] = resource
        return original["publish"](*args, **publish_kwargs)

    try:
        d81_probe.load_ground_basis = load
        d81_probe.build_d81_fit = build
        d81_eval.fit_d42_unified_shrinkage_lda = top_fit
        d81_eval.predict_d42_unified_shrinkage_lda = predict_d93
        d81_eval._audit_fit = _audit_fit
        d81_eval._publish_state = publish
        d81_eval.CANDIDATE_D81 = str(candidate)
        d81_eval.SCHEMA = selected_schema
        result = d81_eval.run_d81_query_evaluation(**kwargs)
    finally:
        d81_probe.load_ground_basis = original["load"]
        d81_probe.build_d81_fit = original["build"]
        d81_eval.fit_d42_unified_shrinkage_lda = original["top_fit"]
        d81_eval.predict_d42_unified_shrinkage_lda = original["predict"]
        d81_eval._audit_fit = original["audit"]
        d81_eval._publish_state = original["publish"]
        d81_eval.CANDIDATE_D81 = original["candidate"]
        d81_eval.SCHEMA = original["schema"]
    returned_resource = dict(result.get("resource", {}))
    returned_resource.update(
        {
            "k1_strict_identity_pass": False,
            "d93_k1_nonidentity_required": True,
            "d93_candidate": str(candidate),
            "target_clean_iq_access": False,
            "same_physical_iq_multi_channel_views": False,
            "phase2_channel_simulator_calls": 0,
        }
    )
    return {
        **dict(result),
        "schema": selected_schema,
        "candidate": str(candidate),
        "resource": returned_resource,
        "d93_mode": "interaction_plus_nuisance_scale"
        if include_scale
        else (
            "coverage_controlled_interaction"
            if coverage_controlled_update
            else "interaction_only"
        ),
    }


__all__ = [
    "CANDIDATES_D93",
    "CANDIDATES_D94",
    "CANDIDATES_GROUND_TRANSPORT",
    "CANDIDATE_D93_INTERACTION",
    "CANDIDATE_D93_SCALE",
    "CANDIDATE_D94_COVERAGE",
    "D93QueryEvaluationError",
    "D93Result",
    "D93State",
    "build_d93_top_level_fit",
    "predict_d93",
    "run_d93_query_evaluation",
    "score_d93",
]
