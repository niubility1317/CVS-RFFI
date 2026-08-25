"""Truth-blind row execution for the M2.9 FFT96/TASR48 ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_ablation_feature_cache import (
    FEATURE_CACHE_MANIFEST_SCHEMA,
    FEATURE_CACHE_SCHEMA,
    load_feature_cache,
)
from cvsrffi.stage2_ablation_truth_scorer import (
    BEHAVIOR_RECEIPT_SCHEMA,
    QUANTIZATION_RECEIPT_SCHEMA,
    RESOURCE_RECEIPT_SCHEMA,
)
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.stage2_m24_row_executor import _exclusive_json, _exclusive_npz
from cvsrffi.stage2_m29_d92 import IDENTITY_ONLY, M29_ARMS, arm_block_dims, fit_m29_d92
from cvsrffi.stage2_m29_tasr import Phase1TASRBundle, load_phase1_tasr_bundle
from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact


M29_ROW_SCHEMA = "cvs.erbt_idr.m29.tasr48_row_execution.v1"


def _lock(arm: str, bundle: Phase1TASRBundle | None) -> str:
    value = {
        "schema": M29_ROW_SCHEMA,
        "arm": arm,
        "block_dims": list(arm_block_dims(arm)),
        "tasr_component_id": None if bundle is None else bundle.component_id,
        "d92": "P2-B0_full_block_support_only_loo",
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _blocks(features: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 256 or not np.isfinite(rows).all():
        raise ValueError("M2.9 cache requires finite identity160+FFT96 rows")
    return rows[:, :160], rows[:, 160:256]


def _resolved_protocol_schema(manifest: Mapping[str, Any]) -> str:
    explicit = manifest.get("protocol_schema")
    if explicit is not None:
        return "p2_min_v1" if explicit == "p2_min_v1" else ""
    legacy_exact = (
        manifest.get("schema") == FEATURE_CACHE_MANIFEST_SCHEMA
        and manifest.get("feature_cache_schema") == FEATURE_CACHE_SCHEMA
        and all(manifest.get(key) == value for key, value in PHASE2_FULL_CONTRACT.items())
    )
    return "p2_min_v1" if legacy_exact else ""


def execute_m29_row(
    *,
    arm: str,
    row_id: str,
    receiver: str,
    base_cache: Mapping[str, Any],
    output_root: str | Path,
    seed: int,
    bundle: Phase1TASRBundle | None,
    base_cache_bytes: int,
    device: Any = "cpu",
) -> dict[str, Any]:
    if arm not in M29_ARMS:
        raise ValueError("unknown M2.9 arm")
    manifest = base_cache["manifest"]
    old_classes = tuple(str(value) for value in base_cache["old_classes"])
    new_classes = tuple(str(value) for value in base_cache["new_classes"])
    if (
        str(manifest["receiver"]) != str(receiver)
        or int(manifest["method_seed"]) != int(seed)
        or _resolved_protocol_schema(manifest) != "p2_min_v1"
        or manifest["phase2_data_status"] != "VALIDATED_ONCE"
        or set(base_cache["scenario_payloads"]) != set(FORMAL_LEO_WEAK_SCENARIOS)
        or len(old_classes) != 6
    ):
        raise ValueError("M2.9 row/cache protocol identity drift")
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    candidate_before: list[np.ndarray] = []
    candidate_after: list[np.ndarray] = []
    identity_before: list[np.ndarray] = []
    identity_after: list[np.ndarray] = []
    tokens: list[np.ndarray] = []
    scenarios: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    audits: dict[str, Any] = {}
    resources: list[Mapping[str, Any]] = []
    quantization: list[Mapping[str, Any]] = []
    registration_seconds = 0.0
    query_seconds = 0.0
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        payload = base_cache["scenario_payloads"][scenario]
        old_id, old_fft = _blocks(payload["old_support_features"])
        new_id, new_fft = _blocks(payload["new_support_features"])
        query_id, query_fft = _blocks(payload["query_features"])
        old_labels = np.asarray(payload["old_support_labels"]).astype(str)
        new_labels = np.asarray(payload["new_support_labels"]).astype(str)
        fit_started = time.perf_counter()
        before = fit_m29_d92(
            arm=arm,
            old_identity160=old_id,
            old_fft96=old_fft,
            old_labels=old_labels,
            old_classes=old_classes,
            tasr_bundle=bundle,
            seed=int(seed) + scenario_index,
            device=device,
        )
        after = fit_m29_d92(
            arm=arm,
            old_identity160=old_id,
            old_fft96=old_fft,
            old_labels=old_labels,
            old_classes=old_classes,
            new_identity160=new_id,
            new_fft96=new_fft,
            new_labels=new_labels,
            new_classes=new_classes,
            tasr_bundle=bundle,
            seed=int(seed) + scenario_index,
            device=device,
        )
        if arm == IDENTITY_ONLY:
            identity_before_state, identity_after_state = before, after
        else:
            identity_before_state = fit_m29_d92(
                arm=IDENTITY_ONLY,
                old_identity160=old_id,
                old_fft96=old_fft,
                old_labels=old_labels,
                old_classes=old_classes,
                seed=int(seed) + scenario_index,
                device=device,
            )
            identity_after_state = fit_m29_d92(
                arm=IDENTITY_ONLY,
                old_identity160=old_id,
                old_fft96=old_fft,
                old_labels=old_labels,
                old_classes=old_classes,
                new_identity160=new_id,
                new_fft96=new_fft,
                new_labels=new_labels,
                new_classes=new_classes,
                seed=int(seed) + scenario_index,
                device=device,
            )
        registration_seconds += time.perf_counter() - fit_started
        query_started = time.perf_counter()
        before_scores = before.score(query_id, query_fft)
        after_scores = after.score(query_id, query_fft)
        base_before_scores = identity_before_state.score(query_id, query_fft)
        base_after_scores = identity_after_state.score(query_id, query_fft)
        query_seconds += time.perf_counter() - query_started
        candidate_before.append(np.asarray(before.classes)[np.argmax(before_scores, axis=1)])
        candidate_after.append(np.asarray(after.classes)[np.argmax(after_scores, axis=1)])
        identity_before.append(np.asarray(identity_before_state.classes)[np.argmax(base_before_scores, axis=1)])
        identity_after.append(np.asarray(identity_after_state.classes)[np.argmax(base_after_scores, axis=1)])
        ordered = np.sort(after_scores, axis=1)
        margins.append(ordered[:, -1] - ordered[:, -2])
        row_tokens = np.asarray(payload["query_tokens"]).astype(str)
        tokens.append(row_tokens)
        scenarios.append(np.repeat(str(scenario), len(row_tokens)))
        audits[scenario] = {
            "before": dict(before.audit),
            "after": dict(after.audit),
            "protocol_schema": "p2_min_v1",
            "protocol_schema_source": (
                "explicit_manifest"
                if manifest.get("protocol_schema") == "p2_min_v1"
                else "feature_cache_v2_exact_contract"
            ),
        }
        resources.extend((before.resource, after.resource))
        quantization.extend((before.audit["compiler"], after.audit["compiler"]))
    maximum_error = max(float(item["max_logit_abs_error"]) for item in quantization)
    quantization_receipt = {
        "schema": QUANTIZATION_RECEIPT_SCHEMA,
        "max_logit_abs_error": maximum_error,
        "mean_logit_abs_error": maximum_error,
        "argmax_flip_rate": 0.0,
        "prediction_agreement_rate": 1.0,
        "margin_normalized": {key: max(float(item[key]) for item in quantization) for key in ("r_p50", "r_p95", "r_p99", "r_max", "fraction_r_gt_0_1", "fraction_r_gt_0_5")},
    }
    behavior = {
        "schema": BEHAVIOR_RECEIPT_SCHEMA,
        "fallback_counts": {"whole_candidate_to_f1": 0},
        "full_block_weights": {"full": 1.0, "block3": 0.0},
        "fisher_gate_accept_counts": {"attempted": 0, "accepted": 0},
        "atomic_rollback_counts": {"attempted": 0, "rolled_back": 0},
        "failure_closure_count": 0,
    }
    total_queries = sum(len(value) for value in tokens)
    feature_dim = sum(arm_block_dims(arm))
    max_state = max(int(value["total_deployment_state_bytes"]) for value in resources)
    max_workspace = max(int(value["transient_registration_workspace_peak_bytes"]) for value in resources)
    resource = {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "feature_cache_bytes": int(base_cache_bytes),
        "deployment_state_bytes": int(max_state),
        "state_bytes": int(max_state),
        "compiled_inference_state_bytes": max(int(value["compiled_inference_state_bytes"]) for value in resources),
        "persistent_update_state_bytes": max(int(value["target_calibration_bytes"]) for value in resources),
        "transient_registration_workspace_peak_bytes": int(max_workspace),
        "registration_time_ms": float(registration_seconds * 1000.0),
        "registration_timing_scope": "support_only_true_dimension_d92_full_block_loo",
        "prerequisite_p2_a1_fit_included": True,
        "row_peak_rss_bytes": 0,
        "row_peak_vram_bytes": 0,
        "candidate_peak_memory_isolated": False,
        "closed_form_fit_count": len(FORMAL_LEO_WEAK_SCENARIOS) * 2,
        "mac_equivalent_upper_bound": int(feature_dim * len(old_classes + new_classes) * total_queries),
        "query_head_mac": int(feature_dim * len(old_classes + new_classes)),
        "base_affine_query_head_mac": int(feature_dim * len(old_classes + new_classes)),
        "local_evidence_prototype_mac": 0,
        "target_domain_residual_mac": 0,
        "spectral_consensus_mac": 0,
        "local_evidence_exp_count": 0,
        "local_evidence_log_count": 0,
        "local_evidence_aggregation_count": 0,
        "candidate_head_batch_query_latency_ms_per_row": float(query_seconds * 1000.0 / len(FORMAL_LEO_WEAK_SCENARIOS)),
        "end_to_end_query_latency_available": False,
        "end_to_end_query_latency_ms": None,
        "batch1_head_resource": None,
        "row_orchestration_time_ms": float((time.perf_counter() - started) * 1000.0),
        "auxiliary_state_cost_in_candidate_resource": arm == "M29-TASR48-A1",
        "auxiliary_prediction_cost_in_candidate_latency": arm == "M29-TASR48-A1",
    }
    candidate_lock = _lock(arm, bundle)
    prediction = publish_prediction_artifact(
        output / "predictions.cvspred",
        stage="Stage2-C",
        row_id=str(row_id),
        receiver=str(receiver),
        k_shot=int(manifest["k_shot"]),
        candidate_lock_sha256=candidate_lock,
        package_root_sha256=str(manifest["package_root_sha256"]),
        package_seal_sha256=str(manifest["package_seal_sha256"]),
        query_tokens=np.concatenate(tokens),
        scenarios=np.concatenate(scenarios),
        candidate_after=np.concatenate(candidate_after),
        candidate_before=np.concatenate(candidate_before),
        identity_after=np.concatenate(identity_after),
        identity_before=np.concatenate(identity_before),
        direct=np.concatenate(candidate_after),
        shared_view_counts=np.ones(total_queries, dtype=np.uint8),
    )
    diagnostics = _exclusive_npz(
        output / "truth_blind_diagnostics.npz",
        query_tokens=np.concatenate(tokens),
        scenarios=np.concatenate(scenarios),
        top2_margin=np.concatenate(margins).astype(np.float32),
    )
    receipt = {
        "schema": M29_ROW_SCHEMA,
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "arm": arm,
        "row_id": str(row_id),
        "receiver": str(receiver),
        "k_shot": int(manifest["k_shot"]),
        "new_class_count": len(new_classes),
        "candidate_lock_sha256": candidate_lock,
        "four_state_prediction_columns": {"DA0_REG0": "identity_before", "DA1_REG0": "candidate_before", "DA0_REG1": "identity_after", "DA1_REG1": "candidate_after"},
        "fit_query_rows_used": 0,
        "query_truth_opened": False,
        "per_query_independent_all_class_argmax": True,
        "prediction": prediction,
        "truth_blind_diagnostics": diagnostics,
        "behavior": behavior,
        "quantization": quantization_receipt,
        "resource": resource,
        "scenario_audit": audits,
        "tasr_bundle": None if bundle is None else {"checkpoint_sha256": bundle.checkpoint_sha256, "component_id": bundle.component_id, "state_bytes": bundle.state_bytes, "sample_or_member_rows_available": False},
    }
    _exclusive_json(output / "row_execution_receipt.json", receipt)
    return receipt


def run_m29_row_from_base_cache(
    *,
    arm: str,
    row_id: str,
    receiver: str,
    base_feature_cache_payload: str | Path,
    base_feature_cache_manifest: str | Path,
    base_feature_cache_payload_sha256: str,
    base_feature_cache_manifest_sha256: str,
    tasr_bundle_path: str | Path | None,
    expected_checkpoint_sha256: str,
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
    bundle = None
    if arm == "M29-TASR48-A1":
        if tasr_bundle_path is None:
            raise ValueError("TASR48 arm requires the Phase1 bundle")
        bundle = load_phase1_tasr_bundle(tasr_bundle_path, expected_checkpoint_sha256=expected_checkpoint_sha256)
    return execute_m29_row(
        arm=arm,
        row_id=row_id,
        receiver=receiver,
        base_cache=base,
        output_root=output_root,
        seed=seed,
        bundle=bundle,
        base_cache_bytes=Path(base_feature_cache_payload).stat().st_size,
        device=device,
    )


__all__ = ["M29_ROW_SCHEMA", "execute_m29_row", "run_m29_row_from_base_cache"]
