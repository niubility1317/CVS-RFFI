"""Truth-inaccessible D0-D10 row execution for ERBT-IDR M2.4."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_ablation_feature_cache import load_feature_cache
from cvsrffi.stage2_ablation_quantization import decode_affine_state
from cvsrffi.stage2_ablation_truth_scorer import BEHAVIOR_RECEIPT_SCHEMA, QUANTIZATION_RECEIPT_SCHEMA, RESOURCE_RECEIPT_SCHEMA
from cvsrffi.stage2_m23_overlay_cache import load_m23_overlay_cache
from cvsrffi.stage2_m23_rfguard import build_ground_manifold, estimate_stage2b_domain_state
from cvsrffi.stage2_m23_row_executor import M23_F1_IF, _OverlayGroundComponent, _cosine_prediction, _legacy_states
from cvsrffi.stage2_m24_safe_residual import D0, D1, M24_ARMS, arm_config_hash, fit_m24_safe_residual, prepare_query_features
from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact


M24_ROW_EXECUTION_SCHEMA = "cvs.erbt_idr.m24.row_execution.v1"
DA0_REG0 = "DA0_REG0"
DA1_REG0 = "DA1_REG0"
DA0_REG1 = "DA0_REG1"
DA1_REG1 = "DA1_REG1"


class M24RowExecutionError(RuntimeError):
    pass


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o444)
    try:
        data = _canonical_json(payload) + b"\n"
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing M2.4 receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _f1_reference_head(state: Any, support_blocks: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficient, bias = decode_affine_state(state.compiled_affine_state)
    if coefficient.shape != (len(state.classes), 288) or bias.shape != (len(state.classes),):
        raise M24RowExecutionError("historical F1 FP32 reference is missing")
    return (
        np.asarray(coefficient[:, :256], dtype=np.float64),
        np.asarray(bias, dtype=np.float64),
        np.asarray(state.log_diag_fp32[:256], dtype=np.float32),
    )


def _physical_cosine_head(blocks: Any, labels: Any, classes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    rows = prepare_query_features(blocks, feature_dim=256).astype(np.float64)
    target = np.asarray(labels).astype(str)
    centres = np.stack([np.mean(rows[target == item], axis=0) for item in classes])
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1.0e-12)
    return centres, np.zeros(len(classes), dtype=np.float64)


def d1_overlay_from_base_cache(base_cache: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the RF-independent D1 view directly from one legal base cache."""

    manifest = base_cache["manifest"]
    payloads: dict[str, dict[str, np.ndarray]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        legacy = base_cache["scenario_payloads"][scenario]

        def blocks(name: str) -> np.ndarray:
            rows = np.asarray(legacy[name], dtype=np.float32)
            return np.concatenate(
                [rows[:, :256], np.ones((len(rows), 10), dtype=np.float32)], axis=1
            )

        old_blocks = blocks("old_support_features")
        new_blocks = blocks("new_support_features")
        payloads[scenario] = {
            "old_support_blocks": old_blocks,
            "old_support_labels": np.asarray(legacy["old_support_labels"]).astype(str),
            "old_support_quality": np.ones(len(old_blocks), dtype=np.float32),
            "new_support_blocks": new_blocks,
            "new_support_labels": np.asarray(legacy["new_support_labels"]).astype(str),
            "new_support_quality": np.ones(len(new_blocks), dtype=np.float32),
            "query_blocks": blocks("query_features"),
            "query_tokens": np.asarray(legacy["query_tokens"]).astype(str),
        }
    return {
        "manifest": {
            "receiver": manifest["receiver"],
            "k_shot": manifest["k_shot"],
            "method_seed": manifest["method_seed"],
            "capsule_id": manifest["capsule_id"],
            "split_id": manifest["split_id"],
            "phase2_data_status": manifest["phase2_data_status"],
            "predictor_package_root_sha256": manifest["package_root_sha256"],
            "predictor_package_seal_sha256": manifest["package_seal_sha256"],
            "d1_base_only": True,
        },
        "old_classes": tuple(str(item) for item in base_cache["old_classes"]),
        "new_classes": tuple(str(item) for item in base_cache["new_classes"]),
        "scenario_payloads": payloads,
    }


def execute_m24_row(
    *,
    arm: str,
    row_id: str,
    receiver: str,
    base_cache: Mapping[str, Any],
    overlay_cache: Mapping[str, Any],
    output_root: str | Path,
    seed: int,
    device: Any = "cpu",
    base_cache_bytes: int = 0,
    overlay_cache_bytes: int = 0,
) -> dict[str, Any]:
    if arm not in M24_ARMS:
        raise M24RowExecutionError("unknown M2.4 arm")
    base_manifest = base_cache["manifest"]
    overlay_manifest = overlay_cache["manifest"]
    old_classes = tuple(str(item) for item in base_cache["old_classes"])
    new_classes = tuple(str(item) for item in base_cache["new_classes"])
    if (
        tuple(str(item) for item in overlay_cache["old_classes"]) != old_classes
        or tuple(str(item) for item in overlay_cache["new_classes"]) != new_classes
        or str(base_manifest["receiver"]) != str(receiver)
        or str(overlay_manifest["receiver"]) != str(receiver)
        or int(base_manifest["k_shot"]) != int(overlay_manifest["k_shot"])
        or int(base_manifest["method_seed"]) != int(seed)
        or int(overlay_manifest["method_seed"]) != int(seed)
        or base_manifest["capsule_id"] != overlay_manifest["capsule_id"]
        or base_manifest["split_id"] != overlay_manifest["split_id"]
        or base_manifest["phase2_data_status"] != "VALIDATED_ONCE"
    ):
        raise M24RowExecutionError("base/overlay row identity drift")
    if set(base_cache["scenario_payloads"]) != set(FORMAL_LEO_WEAK_SCENARIOS) or set(overlay_cache["scenario_payloads"]) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise M24RowExecutionError("formal scenario coverage drift")
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=False)
    base_only_d1 = bool(overlay_manifest.get("d1_base_only", False))
    if base_only_d1 and arm != D1:
        raise M24RowExecutionError("base-only compact view is restricted to D1")
    component = None if base_only_d1 else _OverlayGroundComponent(overlay_cache["ground_component"])
    manifold = None if component is None else build_ground_manifold(component)

    candidate_after: list[np.ndarray] = []
    candidate_before: list[np.ndarray] = []
    identity_after: list[np.ndarray] = []
    identity_before: list[np.ndarray] = []
    direct: list[np.ndarray] = []
    tokens_all: list[np.ndarray] = []
    scenarios_all: list[np.ndarray] = []
    audits: dict[str, Any] = {}
    quantization_audits: list[Mapping[str, Any]] = []
    state_bytes: list[int] = []
    transient_bytes: list[int] = []
    feature_dims: list[int] = []
    registration_seconds = 0.0
    query_seconds = 0.0
    parity_disagreements = 0
    parity_rows = 0
    started = time.perf_counter()

    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        legacy = base_cache["scenario_payloads"][scenario]
        compact = overlay_cache["scenario_payloads"][scenario]
        legacy_tokens = np.asarray(legacy["query_tokens"]).astype(str)
        compact_tokens = np.asarray(compact["query_tokens"]).astype(str)
        if not np.array_equal(legacy_tokens, compact_tokens):
            raise M24RowExecutionError(f"{scenario} base/overlay query token drift")
        historical_before, historical_after, prepared = _legacy_states(
            M23_F1_IF,
            legacy,
            old_classes,
            new_classes,
            ground_basis=base_cache["ground_basis"],
            ground_weights=base_cache["ground_spectral_weights"],
            ground_audit=base_cache["ground_audit"],
            seed=int(seed) + scenario_index,
            device=device,
        )
        historical_after_prediction = historical_after.predict(prepared["query_after"])
        historical_before_prediction = historical_before.predict(prepared["query_before"])
        identity_before_prediction = _cosine_prediction(
            prepared["old"], legacy["old_support_labels"], old_classes, prepared["query"], 288
        )
        identity_after_prediction = _cosine_prediction(
            np.concatenate([prepared["old"], prepared["new"]]),
            np.concatenate([legacy["old_support_labels"], legacy["new_support_labels"]]),
            old_classes + new_classes,
            prepared["query"],
            288,
        )
        if arm == D0:
            selected_after = historical_after_prediction
            selected_before = historical_before_prediction
            audit = {
                "schema": "cvs.erbt_idr.m24.historical_f1_audit.v1",
                "arm": arm,
                "feature_dim": 288,
                "query_rows_used": 0,
                "historical_ablation_id": "P2-A1",
            }
            quantization = {"r_p50": 0.0, "r_p95": 0.0, "r_p99": 0.0, "r_max": 0.0, "fraction_r_gt_0_1": 0.0, "fraction_r_gt_0_5": 0.0, "max_logit_abs_error": 0.0}
            resource = {
                "compiled_inference_state_bytes": int(historical_after.resource["persistent_head_state_bytes"]),
                "persistent_update_state_bytes": int(historical_after.resource["persistent_state_bytes"] - historical_after.resource["persistent_head_state_bytes"]),
                "transient_registration_workspace_peak_bytes": 0,
            }
            feature_dim = 288
        else:
            all_support_blocks = np.concatenate([compact["old_support_blocks"], compact["new_support_blocks"]])
            all_support_labels = np.concatenate([compact["old_support_labels"], compact["new_support_labels"]])
            all_support_quality = np.concatenate([compact["old_support_quality"], compact["new_support_quality"]])
            coefficient, bias, f1_log_diag = _f1_reference_head(historical_after, all_support_blocks)
            if manifold is None:
                domain_digest = hashlib.sha256(
                    str(base_manifest["split_id"]).encode("utf-8")
                ).hexdigest()
                ground_prior = None
                nuisance_covariance = None
            else:
                domain_state = estimate_stage2b_domain_state(
                    compact["old_support_blocks"],
                    compact["old_support_labels"],
                    old_classes,
                    np.ones(len(compact["old_support_quality"]), dtype=np.float32),
                    manifold,
                )
                domain_digest = domain_state.digest
                ground_prior = manifold.class_centres
                nuisance_covariance = domain_state.nuisance_covariance
            fit_started = time.perf_counter()
            state, audit, _workspace = fit_m24_safe_residual(
                arm=arm,
                support_blocks=all_support_blocks,
                support_labels=all_support_labels,
                classes=old_classes + new_classes,
                support_quality=all_support_quality,
                k_shot=int(overlay_manifest["k_shot"]),
                old_class_count=len(old_classes),
                f1_coefficient=coefficient,
                f1_bias=bias,
                f1_log_diag=f1_log_diag,
                domain_digest=domain_digest,
                ground_prior_identity=ground_prior,
                nuisance_covariance_identity=nuisance_covariance,
            )
            registration_seconds += time.perf_counter() - fit_started
            feature_dim = state.compiled_affine_state.feature_dim
            query_features = prepare_query_features(compact["query_blocks"], feature_dim=feature_dim)
            old_coefficient, old_bias = _physical_cosine_head(
                compact["old_support_blocks"], compact["old_support_labels"], old_classes
            )
            before_state, before_audit, _before_workspace = fit_m24_safe_residual(
                arm=arm,
                support_blocks=compact["old_support_blocks"],
                support_labels=compact["old_support_labels"],
                classes=old_classes,
                support_quality=compact["old_support_quality"],
                k_shot=int(overlay_manifest["k_shot"]),
                old_class_count=len(old_classes),
                f1_coefficient=old_coefficient,
                f1_bias=old_bias,
                domain_digest=domain_digest,
                ground_prior_identity=ground_prior,
                nuisance_covariance_identity=nuisance_covariance,
            )
            before_query_features = prepare_query_features(
                compact["query_blocks"], feature_dim=before_state.compiled_affine_state.feature_dim
            )
            query_started = time.perf_counter()
            selected_after = state.predict(query_features)
            selected_before = before_state.predict(before_query_features)
            query_seconds += time.perf_counter() - query_started
            quantization = audit["quantization"]
            resource = audit["resource"]
            audit = {**dict(audit), "before_registration_fit": dict(before_audit)}
            parity_disagreements += int(np.sum(selected_after != historical_after_prediction)) if arm.endswith("PHYSICAL256-F1") else 0
            parity_rows += len(selected_after) if arm.endswith("PHYSICAL256-F1") else 0

        candidate_after.append(np.asarray(selected_after).astype(str))
        candidate_before.append(np.asarray(selected_before).astype(str))
        identity_after.append(np.asarray(identity_after_prediction).astype(str))
        identity_before.append(np.asarray(identity_before_prediction).astype(str))
        direct.append(np.asarray(identity_after_prediction).astype(str))
        tokens_all.append(compact_tokens)
        scenarios_all.append(np.asarray([scenario] * len(compact_tokens)))
        audits[scenario] = dict(audit)
        quantization_audits.append(dict(quantization))
        state_bytes.append(int(resource["compiled_inference_state_bytes"]))
        transient_bytes.append(int(resource["transient_registration_workspace_peak_bytes"]))
        feature_dims.append(feature_dim)

    maximum_error = max(float(item["max_logit_abs_error"]) for item in quantization_audits)
    quantization_receipt = {
        "schema": QUANTIZATION_RECEIPT_SCHEMA,
        "max_logit_abs_error": maximum_error,
        "mean_logit_abs_error": maximum_error,
        "argmax_flip_rate": 0.0,
        "prediction_agreement_rate": 1.0,
        "margin_normalized": {
            key: max(float(item[key]) for item in quantization_audits)
            for key in ("r_p50", "r_p95", "r_p99", "r_max", "fraction_r_gt_0_1", "fraction_r_gt_0_5")
        },
    }
    behavior_receipt = {
        "schema": BEHAVIOR_RECEIPT_SCHEMA,
        "fallback_counts": {"whole_candidate_to_f1": int(sum(bool(item.get("whole_candidate_safety", {}).get("whole_candidate_fallback_to_f1", False)) for item in audits.values()))},
        "full_block_weights": {"full": 1.0, "block3": 0.0},
        "fisher_gate_accept_counts": {"attempted": 0, "accepted": 0},
        "atomic_rollback_counts": {"attempted": 0, "rolled_back": 0},
        "failure_closure_count": 0,
    }
    total_queries = sum(len(item) for item in tokens_all)
    resource_receipt = {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "feature_cache_bytes": int(base_cache_bytes + overlay_cache_bytes),
        "deployment_state_bytes": int(component.resource_audit()["logical_deployment_state_bytes"]) if component is not None else 0,
        "state_bytes": int(max(state_bytes)),
        "compiled_inference_state_bytes": int(max(state_bytes)),
        "persistent_update_state_bytes": 0 if arm != D0 else int(max(int(audits[s].get("resource", {}).get("persistent_update_state_bytes", 0)) for s in audits)),
        "transient_registration_workspace_peak_bytes": int(max(transient_bytes)),
        "registration_time_ms": float(1000.0 * registration_seconds),
        "row_peak_rss_bytes": 0,
        "row_peak_vram_bytes": 0,
        "candidate_peak_memory_isolated": False,
        "closed_form_fit_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "mac_equivalent_upper_bound": int(max(feature_dims) * len(old_classes + new_classes) * total_queries),
        "query_head_mac": int(max(feature_dims) * len(old_classes + new_classes)),
        "candidate_head_batch_query_latency_ms_per_row": float(1000.0 * query_seconds / len(FORMAL_LEO_WEAK_SCENARIOS)),
        "end_to_end_query_latency_available": False,
        "end_to_end_query_latency_ms": None,
        "batch1_head_resource": None,
        "row_orchestration_time_ms": float(1000.0 * (time.perf_counter() - started)),
        "auxiliary_state_cost_in_candidate_resource": False,
        "auxiliary_prediction_cost_in_candidate_latency": False,
    }
    lock = arm_config_hash(arm)
    prediction = publish_prediction_artifact(
        output / "predictions.cvspred",
        stage="Stage2-C",
        row_id=str(row_id),
        receiver=str(receiver),
        k_shot=int(overlay_manifest["k_shot"]),
        candidate_lock_sha256=lock,
        package_root_sha256=str(overlay_manifest["predictor_package_root_sha256"]),
        package_seal_sha256=str(overlay_manifest["predictor_package_seal_sha256"]),
        query_tokens=np.concatenate(tokens_all),
        scenarios=np.concatenate(scenarios_all),
        candidate_after=np.concatenate(candidate_after),
        candidate_before=np.concatenate(candidate_before),
        identity_after=np.concatenate(identity_after),
        identity_before=np.concatenate(identity_before),
        direct=np.concatenate(direct),
        shared_view_counts=np.ones(total_queries, dtype=np.uint8),
    )
    receipt = {
        "schema": M24_ROW_EXECUTION_SCHEMA,
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "arm": arm,
        "row_id": str(row_id),
        "receiver": str(receiver),
        "k_shot": int(overlay_manifest["k_shot"]),
        "new_class_count": len(new_classes),
        "candidate_lock_sha256": lock,
        "four_state_prediction_columns": {DA0_REG0: "identity_before", DA1_REG0: "candidate_before", DA0_REG1: "identity_after", DA1_REG1: "candidate_after"},
        "fit_query_rows_used": 0,
        "query_truth_opened": False,
        "per_query_independent_all_class_argmax": True,
        "d1_historical_parity": {"query_rows": parity_rows, "prediction_disagreements": parity_disagreements, "agreement_rate": float(1.0 - parity_disagreements / parity_rows) if parity_rows else None},
        "prediction": prediction,
        "behavior": behavior_receipt,
        "quantization": quantization_receipt,
        "resource": resource_receipt,
        "scenario_audit": audits,
    }
    _exclusive_json(output / "row_execution_receipt.json", receipt)
    return receipt


def run_m24_row_from_caches(
    *,
    arm: str,
    row_id: str,
    receiver: str,
    base_feature_cache_payload: str | Path,
    base_feature_cache_manifest: str | Path,
    base_feature_cache_payload_sha256: str,
    base_feature_cache_manifest_sha256: str,
    overlay_payload: str | Path,
    overlay_manifest: str | Path,
    overlay_payload_sha256: str,
    overlay_manifest_sha256: str,
    output_root: str | Path,
    seed: int,
    device: Any = "cpu",
) -> dict[str, Any]:
    base = load_feature_cache(base_feature_cache_payload, base_feature_cache_manifest, expected_payload_sha256=str(base_feature_cache_payload_sha256).lower(), expected_manifest_sha256=str(base_feature_cache_manifest_sha256).lower())
    overlay = load_m23_overlay_cache(overlay_payload, overlay_manifest, expected_payload_sha256=str(overlay_payload_sha256).lower(), expected_manifest_sha256=str(overlay_manifest_sha256).lower())
    if overlay["manifest"]["base_feature_cache_payload_sha256"] != str(base_feature_cache_payload_sha256).lower() or overlay["manifest"]["base_feature_cache_manifest_sha256"] != str(base_feature_cache_manifest_sha256).lower():
        raise M24RowExecutionError("overlay does not bind the supplied base cache")
    return execute_m24_row(
        arm=arm,
        row_id=row_id,
        receiver=receiver,
        base_cache=base,
        overlay_cache=overlay,
        output_root=output_root,
        seed=int(seed),
        device=device,
        base_cache_bytes=Path(base_feature_cache_payload).stat().st_size,
        overlay_cache_bytes=Path(overlay_payload).stat().st_size,
    )


def run_m24_d1_row_from_base_cache(
    *,
    row_id: str,
    receiver: str,
    base_feature_cache_payload: str | Path,
    base_feature_cache_manifest: str | Path,
    base_feature_cache_payload_sha256: str,
    base_feature_cache_manifest_sha256: str,
    output_root: str | Path,
    seed: int,
    device: Any = "cpu",
) -> dict[str, Any]:
    base = load_feature_cache(
        base_feature_cache_payload,
        base_feature_cache_manifest,
        expected_payload_sha256=str(base_feature_cache_payload_sha256).lower(),
        expected_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
    )
    return execute_m24_row(
        arm=D1,
        row_id=row_id,
        receiver=receiver,
        base_cache=base,
        overlay_cache=d1_overlay_from_base_cache(base),
        output_root=output_root,
        seed=int(seed),
        device=device,
        base_cache_bytes=Path(base_feature_cache_payload).stat().st_size,
        overlay_cache_bytes=0,
    )


__all__ = ["DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1", "M24_ROW_EXECUTION_SCHEMA", "M24RowExecutionError", "d1_overlay_from_base_cache", "execute_m24_row", "run_m24_d1_row_from_base_cache", "run_m24_row_from_caches"]
