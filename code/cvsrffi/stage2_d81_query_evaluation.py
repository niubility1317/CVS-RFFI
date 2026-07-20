"""Locked full-query evaluation for D81 on sealed p2_min_v1 row pairs.

The module fits D81 from registered support only.  It produces immutable
before/after prediction artifacts before the offline scorer is allowed to join
query truth.  Every scenario is fitted independently with the same locked
formula, and every query is scored independently against all classes currently
registered in that state.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cvsrffi.somph_diagnostic_bundle_loader import (
    load_verified_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_d42_unified_shrinkage_lda import (
    D42UnifiedShrinkageLDAConfig,
    fit_d42_unified_shrinkage_lda,
    predict_d42_unified_shrinkage_lda,
)
from cvsrffi.stage2_diag_cosine_exploration import (
    _descriptor,
    _device,
    _output_root,
    _sha256_file,
    _validate_matched_packages,
    _write_json_new,
    _write_npz_new,
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_predictor_runtime import load_torchscript_backbone_same_fd


CANDIDATE_D81 = "d81_ground_nuisance_cauchy_center"
SCHEMA = "cvs.phase2.d81.full_query_evaluation.v1"


class D81QueryEvaluationError(ValueError):
    """Raised when a sealed row or the locked D81 evaluation drifts."""


def _canonical_bytes(value: Mapping[str, Any] | list[Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _registered_handles(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    rows = manifest.get("registered_classes")
    if not isinstance(rows, list):
        raise D81QueryEvaluationError("registered class manifest drift")
    handles = tuple(str(row.get("class_handle", "")) for row in rows)
    if not handles or any(not value for value in handles) or len(set(handles)) != len(handles):
        raise D81QueryEvaluationError("registered class handle drift")
    return handles


def _require_cross_state_lock(
    before_enrollment: Mapping[str, Any],
    before_apply: Mapping[str, Any],
    after_enrollment: Mapping[str, Any],
    after_apply: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    for field in (
        "receiver",
        "seed",
        "k_shot",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
        "row_handle",
        "row_manifest_sha256",
    ):
        before_value = before_apply.get(field, before_enrollment.get(field))
        after_value = after_apply.get(field, after_enrollment.get(field))
        if before_value != after_value:
            raise D81QueryEvaluationError(f"before/after {field} drift")
    if (
        str(before_enrollment.get("registration_state")) != "before"
        or str(after_enrollment.get("registration_state")) != "after"
        or str(before_apply.get("registration_state")) != "before"
        or str(after_apply.get("registration_state")) != "after"
    ):
        raise D81QueryEvaluationError("before/after registration state drift")
    old_classes = _registered_handles(before_enrollment)
    all_classes = _registered_handles(after_enrollment)
    if all_classes[: len(old_classes)] != old_classes:
        raise D81QueryEvaluationError("old registered prefix drift")
    new_count = len(all_classes) - len(old_classes)
    if len(old_classes) != 6 or new_count not in (5, 10, 20):
        raise D81QueryEvaluationError("D81 125 class-count lock drift")
    k_shot = int(after_enrollment.get("k_shot", -1))
    if k_shot not in (1, 5, 10):
        raise D81QueryEvaluationError("D81 125 K-shot lock drift")
    return old_classes, all_classes, k_shot


def _support_features(
    payload: Mapping[str, np.ndarray],
    *,
    model: Any,
    runtime_device: Any,
    class_handles: tuple[str, ...],
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    class_indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
    mask = ranks < int(k_shot)
    if (
        ranks.shape != class_indices.shape
        or ranks.ndim != 1
        or int(np.sum(mask)) != int(k_shot) * len(class_handles)
        or int(class_indices[mask].min()) != 0
        or int(class_indices[mask].max()) != len(class_handles) - 1
    ):
        raise D81QueryEvaluationError("D81 support assignment drift")
    iq = np.asarray(payload["support_leo_weak_iq"], dtype=np.float32)[mask]
    zid = forward_zid160(model, iq, device=runtime_device, batch_size=64)
    features = registered_feature(iq, zid)
    labels = np.asarray(class_handles, dtype=str)[class_indices[mask]]
    return features, labels, class_indices[mask], int(len(iq))


def _query_features(
    payload: Mapping[str, np.ndarray], *, model: Any, runtime_device: Any
) -> tuple[np.ndarray, np.ndarray, int]:
    iq = np.asarray(payload["query_leo_weak_iq"], dtype=np.float32)
    tokens = np.asarray(payload["query_tokens"]).astype(str)
    if iq.ndim < 2 or tokens.ndim != 1 or len(iq) != len(tokens) or len(iq) == 0:
        raise D81QueryEvaluationError("D81 query payload drift")
    zid = forward_zid160(model, iq, device=runtime_device, batch_size=1)
    return registered_feature(iq, zid), tokens, int(len(iq))


def _audit_fit(
    result: Any,
    *,
    scenario: str,
    k_shot: int,
    old_count: int,
    class_count: int,
) -> dict[str, Any]:
    geometry = result.geometry_audit
    before = geometry.get("before_covariance_audit", {})
    after = geometry.get("final_covariance_audit", {})
    required = (
        "d81_ground_int8_component_used",
        "d81_ground_component_update_access",
        "d81_old_new_role_specific_branch",
        "d81_class_id_specific_formula",
        "d81_uses_outer_held_or_query",
        "d81_query_rows_used",
        "d81_single_affine_state_only",
    )
    for audit, expected_count in ((before, old_count), (after, class_count)):
        transform = audit.get("d81_transform_audit", {})
        if (
            any(name not in audit for name in required)
            or audit["d81_ground_int8_component_used"] is not True
            or audit["d81_ground_component_update_access"] is not False
            or audit["d81_old_new_role_specific_branch"] is not False
            or audit["d81_class_id_specific_formula"] is not False
            or audit["d81_uses_outer_held_or_query"] is not False
            or int(audit["d81_query_rows_used"]) != 0
            or audit["d81_single_affine_state_only"] is not True
            or int(transform.get("class_count", -1)) != expected_count
            or int(transform.get("k_shot", -1)) != int(k_shot)
            or transform.get("uses_outer_held_or_query") is not False
            or float(transform.get("within_class_residual_max_abs_error", 1.0)) > 2e-12
            or float(transform.get("fft96_rf32_max_abs_error", 1.0)) != 0.0
        ):
            raise D81QueryEvaluationError("D81 support-only fit closure drift")
        if int(k_shot) == 1 and float(transform.get("center_shift_l2_max", 1.0)) > 1e-12:
            raise D81QueryEvaluationError("D81 K1 robust center must be identity")
    return {
        "scenario": scenario,
        "k_shot": int(k_shot),
        "old_class_count": int(old_count),
        "registered_class_count": int(class_count),
        "k1_unit_covariance_fallback": bool(geometry["k1_unit_covariance_fallback"]),
        "before_covariance_policy": str(before.get("covariance_policy")),
        "after_covariance_policy": str(after.get("covariance_policy")),
        "before_center_shift_l2_max": float(
            before["d81_transform_audit"]["center_shift_l2_max"]
        ),
        "after_center_shift_l2_max": float(
            after["d81_transform_audit"]["center_shift_l2_max"]
        ),
        "before_effective_sample_size_min": float(
            min(before["d81_transform_audit"]["effective_sample_size_by_class"])
        ),
        "after_effective_sample_size_min": float(
            min(after["d81_transform_audit"]["effective_sample_size_by_class"])
        ),
        "before_state_bytes": int(result.before_state.persistent_state_bytes),
        "after_state_bytes": int(result.state.persistent_state_bytes),
        "training_trace": [dict(row) for row in result.training_trace],
        "resource_audit": dict(result.resource_audit),
    }


def _publish_state(
    output: Path,
    *,
    state: str,
    manifest: Mapping[str, Any],
    apply_manifest: Mapping[str, Any],
    enrollment_audit: Mapping[str, Any],
    apply_audit: Mapping[str, Any],
    enrollment_seal_sha256: str,
    apply_seal_sha256: str,
    query_tokens: list[np.ndarray],
    scenarios: list[np.ndarray],
    predictions: list[np.ndarray],
    fit_audit: list[dict[str, Any]],
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    destination = _output_root(output)
    prediction_sha256 = _write_npz_new(
        destination / "prediction_artifact.npz",
        query_tokens=np.concatenate(query_tokens).astype(str),
        scenarios=np.concatenate(scenarios).astype(str),
        predicted_class_handles=np.concatenate(predictions).astype(str),
    )
    fit_audit_sha256 = _write_json_new(destination / "fit_audit.json", fit_audit)
    resource_sha256 = _write_json_new(destination / "resource_audit.json", dict(resource))
    receipt = {
        "schema": "cvs.phase2.diag_cosine_exploration_receipt.v1",
        "status": "CONFIRMATION_PREDICTION_COMPLETE_UNVERIFIED_GROUND_COMPONENT",
        "claim_scope": "confirmation_stability_screen_development_evidence_only",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "candidate": {"name": CANDIDATE_D81},
        "registration_state": state,
        "receiver": manifest["receiver"],
        "seed": manifest["seed"],
        "k_shot": manifest["k_shot"],
        "registered_class_count": len(_registered_handles(manifest)),
        "row_handle": apply_manifest["row_handle"],
        "row_manifest_sha256": apply_manifest["row_manifest_sha256"],
        "phase1_checkpoint_sha256": manifest["phase1_checkpoint_sha256"],
        "feature_runtime_sha256": manifest["feature_runtime_sha256"],
        "method_lock_sha256": manifest["method_lock_sha256"],
        "enrollment_package_root_sha256": manifest["package_root_sha256"],
        "enrollment_package_seal_sha256": str(enrollment_seal_sha256).lower(),
        "apply_package_root_sha256": apply_manifest["package_root_sha256"],
        "apply_package_seal_sha256": str(apply_seal_sha256).lower(),
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_source_sample_access": False,
        "phase2_source_cache_access": False,
        "phase2_source_label_access": False,
        "phase2_source_derived_signal_access": False,
        "phase2_source_replay": False,
        "phase2_external_source_adapter_access": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "query_truth_present_in_predictor": False,
        "query_truth_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "query_query_graph_used": False,
        "query_decision_policy": "per_sample_all_registered_classes",
        "support_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "query_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "ground_component_status": "UNVERIFIED",
        "preopen_audit": {"enrollment": enrollment_audit, "apply": apply_audit},
        "resource": dict(resource),
        "artifacts": {
            "prediction_artifact.npz": prediction_sha256,
            "fit_audit.json": fit_audit_sha256,
            "resource_audit.json": resource_sha256,
        },
    }
    receipt_sha256 = _write_json_new(destination / "execution_receipt.json", receipt)
    members = [
        {
            "relative_path": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(destination.iterdir(), key=lambda value: value.name)
    ]
    commit = {
        "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
        "members": members,
        "artifact_root_sha256": hashlib.sha256(_canonical_bytes(members)).hexdigest(),
        "execution_receipt_sha256": receipt_sha256,
        "prediction_artifact_sha256": prediction_sha256,
    }
    commit_sha256 = _write_json_new(destination / "COMMIT.json", commit)
    return {
        "registration_state": state,
        "prediction_artifact_sha256": prediction_sha256,
        "execution_receipt_sha256": receipt_sha256,
        "commit_sha256": commit_sha256,
        "output_root": str(destination),
    }


def run_d81_query_evaluation(
    *,
    before_enrollment_package_root: str | Path,
    before_enrollment_seal_path: str | Path,
    before_enrollment_seal_sha256: str,
    before_apply_package_root: str | Path,
    before_apply_seal_path: str | Path,
    before_apply_seal_sha256: str,
    after_enrollment_package_root: str | Path,
    after_enrollment_seal_path: str | Path,
    after_enrollment_seal_sha256: str,
    after_apply_package_root: str | Path,
    after_apply_seal_path: str | Path,
    after_apply_seal_sha256: str,
    ground_component_dir: str | Path,
    ground_manifest_sha256: str,
    output_root: str | Path,
    device: str,
) -> dict[str, Any]:
    """Fit locked D81 from support and publish before/after query predictions."""

    from scripts import probe_d81_ground_nuisance_cauchy_center as probe
    from cvsrffi import stage2_d42_unified_shrinkage_lda as d42

    before_support, before_manifest, before_enrollment_audit = (
        load_verified_somph_predictor_bundle(
            before_enrollment_package_root,
            detached_seal_path=before_enrollment_seal_path,
            expected_seal_sha256=str(before_enrollment_seal_sha256).lower(),
        )
    )
    before_query, before_apply, before_apply_audit = load_verified_somph_predictor_bundle(
        before_apply_package_root,
        detached_seal_path=before_apply_seal_path,
        expected_seal_sha256=str(before_apply_seal_sha256).lower(),
    )
    after_support, after_manifest, after_enrollment_audit = (
        load_verified_somph_predictor_bundle(
            after_enrollment_package_root,
            detached_seal_path=after_enrollment_seal_path,
            expected_seal_sha256=str(after_enrollment_seal_sha256).lower(),
        )
    )
    after_query, after_apply, after_apply_audit = load_verified_somph_predictor_bundle(
        after_apply_package_root,
        detached_seal_path=after_apply_seal_path,
        expected_seal_sha256=str(after_apply_seal_sha256).lower(),
    )
    _validate_matched_packages(before_manifest, before_apply)
    _validate_matched_packages(after_manifest, after_apply)
    old_classes, all_classes, k_shot = _require_cross_state_lock(
        before_manifest, before_apply, after_manifest, after_apply
    )
    runtime_device = _device(device)
    model = load_torchscript_backbone_same_fd(
        after_enrollment_package_root,
        _descriptor(after_manifest, "feature_runtime"),
        device=runtime_device,
    )
    basis, spectral_weights, ground_audit = probe.load_ground_basis(
        Path(ground_component_dir), str(ground_manifest_sha256), 288
    )
    d81_fit, call_records, transform_records = probe.build_d81_fit(
        d42, basis, spectral_weights, ground_audit
    )
    original_fit = d42._fit_equal_prior_lda
    before_tokens: list[np.ndarray] = []
    before_scenarios: list[np.ndarray] = []
    before_predictions: list[np.ndarray] = []
    after_tokens: list[np.ndarray] = []
    after_scenarios: list[np.ndarray] = []
    after_predictions: list[np.ndarray] = []
    fit_audit: list[dict[str, Any]] = []
    support_forward_count = 0
    before_query_forward_count = 0
    after_query_forward_count = 0
    scoring_seconds = 0.0
    peak_state_bytes = 0
    try:
        d42._fit_equal_prior_lda = d81_fit
        for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
            support_x, support_y, class_indices, support_count = _support_features(
                after_support[scenario],
                model=model,
                runtime_device=runtime_device,
                class_handles=all_classes,
                k_shot=k_shot,
            )
            old_mask = class_indices < len(old_classes)
            new_mask = ~old_mask
            result = fit_d42_unified_shrinkage_lda(
                support_x[old_mask],
                support_y[old_mask],
                old_classes,
                support_x[new_mask],
                support_y[new_mask],
                all_classes[len(old_classes) :],
                seed=int(after_manifest["seed"]) + scenario_index,
                device=runtime_device,
                config=D42UnifiedShrinkageLDAConfig(),
            )
            fit_audit.append(
                _audit_fit(
                    result,
                    scenario=scenario,
                    k_shot=k_shot,
                    old_count=len(old_classes),
                    class_count=len(all_classes),
                )
            )
            support_forward_count += support_count
            peak_state_bytes = max(
                peak_state_bytes,
                int(result.before_state.persistent_state_bytes),
                int(result.state.persistent_state_bytes),
            )
            before_x, before_token, before_count = _query_features(
                before_query[scenario], model=model, runtime_device=runtime_device
            )
            after_x, after_token, after_count = _query_features(
                after_query[scenario], model=model, runtime_device=runtime_device
            )
            started = time.perf_counter()
            before_pred = predict_d42_unified_shrinkage_lda(
                result.before_state, before_x
            )
            after_pred = predict_d42_unified_shrinkage_lda(result.state, after_x)
            scoring_seconds += time.perf_counter() - started
            if int(k_shot) == 1:
                before_audit = result.geometry_audit["before_covariance_audit"]
                after_audit = result.geometry_audit["final_covariance_audit"]
                if (
                    before_audit["d62_boundary_status"] != "k1_k2_exact_d46_fallback"
                    or after_audit["d62_boundary_status"] != "k1_k2_exact_d46_fallback"
                ):
                    raise D81QueryEvaluationError("D81 K1 exact fallback drift")
            before_tokens.append(before_token)
            before_scenarios.append(np.asarray([scenario] * before_count))
            before_predictions.append(np.asarray(before_pred).astype(str))
            after_tokens.append(after_token)
            after_scenarios.append(np.asarray([scenario] * after_count))
            after_predictions.append(np.asarray(after_pred).astype(str))
            before_query_forward_count += before_count
            after_query_forward_count += after_count
    finally:
        d42._fit_equal_prior_lda = original_fit
    if _sha256_file(Path(ground_component_dir) / probe.d66.NPZ_NAME) != ground_audit[
        "component_npz_sha256"
    ]:
        raise D81QueryEvaluationError("D81 ground component changed during evaluation")
    common_resource = {
        "trainable_parameters": int(max(row["resource_audit"]["trainable_parameters"] for row in fit_audit)),
        "adaptation_epochs": int(max(row["resource_audit"]["adaptation_epochs"] for row in fit_audit)),
        "optimizer_steps": int(max(row["resource_audit"]["optimizer_steps"] for row in fit_audit)),
        "persistent_state_bytes_peak": int(peak_state_bytes + ground_audit["ground_int8_component_logical_state_bytes"]),
        "ground_component_logical_state_bytes": int(ground_audit["ground_int8_component_logical_state_bytes"]),
        "ground_component_input_count": int(ground_audit["ground_component_input_count"]),
        "ground_component_update_access": False,
        "support_backbone_forward_count": int(support_forward_count),
        "before_query_backbone_forward_count": int(before_query_forward_count),
        "after_query_backbone_forward_count": int(after_query_forward_count),
        "query_backbone_forwards_per_sample": 1,
        "fft_extractions_per_query": 1,
        "score_matrix_latency_sec": float(scoring_seconds),
        "score_matrix_latency_per_query_ms": float(
            1000.0 * scoring_seconds / max(1, before_query_forward_count + after_query_forward_count)
        ),
        "dense_query_graph_bytes": 0,
        "query_extra_macs_for_ground_component": 0,
        "ground_component_fit_execution_count": len(call_records),
        "support_center_transform_execution_count": len(transform_records),
        "k1_strict_identity_pass": bool(
            k_shot != 1
            or all(
                max(row["before_center_shift_l2_max"], row["after_center_shift_l2_max"])
                <= 1e-12
                for row in fit_audit
            )
        ),
    }
    output = Path(output_root)
    if output.exists() and (
        not output.is_dir() or output.is_symlink() or any(output.iterdir())
    ):
        raise D81QueryEvaluationError(
            f"D81 evaluation output is not an empty directory: {output}"
        )
    states = {
        "before": _publish_state(
            output / "before",
            state="before",
            manifest=before_manifest,
            apply_manifest=before_apply,
            enrollment_audit=before_enrollment_audit,
            apply_audit=before_apply_audit,
            enrollment_seal_sha256=before_enrollment_seal_sha256,
            apply_seal_sha256=before_apply_seal_sha256,
            query_tokens=before_tokens,
            scenarios=before_scenarios,
            predictions=before_predictions,
            fit_audit=fit_audit,
            resource=common_resource,
        ),
        "after": _publish_state(
            output / "after",
            state="after",
            manifest=after_manifest,
            apply_manifest=after_apply,
            enrollment_audit=after_enrollment_audit,
            apply_audit=after_apply_audit,
            enrollment_seal_sha256=after_enrollment_seal_sha256,
            apply_seal_sha256=after_apply_seal_sha256,
            query_tokens=after_tokens,
            scenarios=after_scenarios,
            predictions=after_predictions,
            fit_audit=fit_audit,
            resource=common_resource,
        ),
    }
    return {
        "schema": SCHEMA,
        "status": "CONFIRMATION_PREDICTIONS_COMPLETE_UNVERIFIED_GROUND_COMPONENT",
        "candidate": CANDIDATE_D81,
        "receiver": after_manifest["receiver"],
        "seed": after_manifest["seed"],
        "k_shot": k_shot,
        "new_class_count": len(all_classes) - len(old_classes),
        "states": states,
        "resource": common_resource,
        "ground_audit": ground_audit,
    }


__all__ = [
    "CANDIDATE_D81",
    "D81QueryEvaluationError",
    "run_d81_query_evaluation",
]
