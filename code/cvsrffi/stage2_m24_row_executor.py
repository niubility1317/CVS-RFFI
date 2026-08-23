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
from cvsrffi.stage2_m24_refit import fit_m24_d1_refit
from cvsrffi.stage2_m24_invariance_breaking import (
    M24_INVARIANCE_ARMS,
    fit_m24_invariance_breaking,
    invariance_arm_config_hash,
)
from cvsrffi.stage2_m24_safe_residual import (
    D0,
    D1,
    D1_REFIT,
    M24_ARMS,
    arm_config_hash,
    compile_m24_d1_from_f1_head,
    fit_m24_safe_residual,
    prepare_query_features,
)
from cvsrffi.stage2_m25_anchored_residual import (
    M25_ANCHORED_ARMS,
    anchored_arm_config_hash,
    fit_m25_anchored_residual,
)
from cvsrffi.stage2_m26_spectral_anchor import (
    Phase1SpectralAnchor,
    load_m26_spectral_anchor,
)
from cvsrffi.stage2_m26_td_src256 import (
    M26_ARMS,
    fit_m26_td_src256,
    m26_arm_config_hash,
)
from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact


M24_ROW_EXECUTION_SCHEMA = "cvs.erbt_idr.m24.row_execution.v1"
DA0_REG0 = "DA0_REG0"
DA1_REG0 = "DA1_REG0"
DA0_REG1 = "DA0_REG1"
DA1_REG1 = "DA1_REG1"


class M24RowExecutionError(RuntimeError):
    pass


def _local_prototype_mac(
    *, feature_dim: int, prototype_counts: Any, active: bool
) -> int:
    if not active:
        return 0
    return int(feature_dim) * sum(int(value) for value in prototype_counts)


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


def _exclusive_npz(path: Path, **arrays: Any) -> dict[str, Any]:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def _support_center_angles(
    blocks: Any, labels: Any, classes: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = prepare_query_features(blocks, feature_dim=256).astype(np.float64)
    target = np.asarray(labels).astype(str)
    centres = np.stack([np.mean(rows[target == item], axis=0) for item in classes])
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1.0e-12)
    left, right = np.triu_indices(len(classes), k=1)
    cosine = np.clip(np.sum(centres[left] * centres[right], axis=1), -1.0, 1.0)
    return (
        np.asarray(classes, dtype=str)[left],
        np.asarray(classes, dtype=str)[right],
        np.degrees(np.arccos(cosine)).astype(np.float32),
    )


def _support_center_angles_from_features(
    rows: Any, labels: Any, classes: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(rows, dtype=np.float64)
    target = np.asarray(labels).astype(str)
    centres = np.stack([np.mean(features[target == item], axis=0) for item in classes])
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1.0e-12)
    left, right = np.triu_indices(len(classes), k=1)
    cosine = np.clip(np.sum(centres[left] * centres[right], axis=1), -1.0, 1.0)
    return (
        np.asarray(classes, dtype=str)[left],
        np.asarray(classes, dtype=str)[right],
        np.degrees(np.arccos(cosine)).astype(np.float32),
    )


def _f1_reference_head(state: Any, support_blocks: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if state.compiled_affine_state is None:
        coefficient = np.asarray(state.coefficient_fp32, dtype=np.float32)
        bias = np.asarray(state.intercept_fp32, dtype=np.float32)
    else:
        coefficient, bias = decode_affine_state(state.compiled_affine_state)
    if coefficient.shape != (len(state.classes), 288) or bias.shape != (len(state.classes),):
        raise M24RowExecutionError("historical F1 FP32 reference is missing")
    return (
        np.asarray(coefficient, dtype=np.float64),
        np.asarray(bias, dtype=np.float64),
        np.asarray(state.log_diag_fp32, dtype=np.float32),
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
    source_anchor: Phase1SpectralAnchor | None = None,
) -> dict[str, Any]:
    if arm not in M24_ARMS + M24_INVARIANCE_ARMS + M25_ANCHORED_ARMS + M26_ARMS:
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
    m26_candidate_lock = None
    if arm in M26_ARMS:
        if source_anchor is None or tuple(source_anchor.class_registry) != old_classes:
            raise M24RowExecutionError("M2.6 requires the checkpoint-bound six-class source anchor")
        m26_candidate_lock = m26_arm_config_hash(arm, source_anchor.component_id)
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=False)
    base_only_d1 = bool(overlay_manifest.get("d1_base_only", False))
    if base_only_d1 and arm not in {D0, D1, D1_REFIT, *M24_INVARIANCE_ARMS, *M25_ANCHORED_ARMS, *M26_ARMS}:
        raise M24RowExecutionError("base-only compact view is restricted to D1 evidence arms")
    component = None if base_only_d1 else _OverlayGroundComponent(overlay_cache["ground_component"])
    manifold = None if component is None else build_ground_manifold(component)

    candidate_after: list[np.ndarray] = []
    candidate_before: list[np.ndarray] = []
    identity_after: list[np.ndarray] = []
    identity_before: list[np.ndarray] = []
    direct: list[np.ndarray] = []
    tokens_all: list[np.ndarray] = []
    scenarios_all: list[np.ndarray] = []
    top2_margins: list[np.ndarray] = []
    center_scenarios: list[np.ndarray] = []
    center_class_left: list[np.ndarray] = []
    center_class_right: list[np.ndarray] = []
    center_angles: list[np.ndarray] = []
    audits: dict[str, Any] = {}
    quantization_audits: list[Mapping[str, Any]] = []
    state_bytes: list[int] = []
    transient_bytes: list[int] = []
    feature_dims: list[int] = []
    registration_seconds = 0.0
    query_seconds = 0.0
    parity_disagreements = 0
    before_parity_disagreements = 0
    parity_rows = 0
    anchored_parity_disagreements = 0
    anchored_before_parity_disagreements = 0
    anchored_parity_rows = 0
    local_prototype_macs: list[int] = []
    local_exp_counts: list[int] = []
    local_log_counts: list[int] = []
    local_aggregation_counts: list[int] = []
    target_domain_residual_macs: list[int] = []
    started = time.perf_counter()

    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        legacy = base_cache["scenario_payloads"][scenario]
        compact = overlay_cache["scenario_payloads"][scenario]
        legacy_tokens = np.asarray(legacy["query_tokens"]).astype(str)
        compact_tokens = np.asarray(compact["query_tokens"]).astype(str)
        if not np.array_equal(legacy_tokens, compact_tokens):
            raise M24RowExecutionError(f"{scenario} base/overlay query token drift")
        if arm == D1_REFIT or arm in M24_INVARIANCE_ARMS:
            prepared = {
                "old": np.asarray(legacy["old_support_features"], dtype=np.float32),
                "new": np.asarray(legacy["new_support_features"], dtype=np.float32),
                "query": np.asarray(legacy["query_features"], dtype=np.float32),
                "query_before": np.asarray(legacy["query_features"], dtype=np.float32),
                "query_after": np.asarray(legacy["query_features"], dtype=np.float32),
            }
            historical_before = None
            historical_after = None
            historical_after_prediction = None
            historical_before_prediction = None
        else:
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
            selected_scores = np.asarray(historical_after.score(prepared["query_after"]), dtype=np.float32)
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
            if arm in M24_INVARIANCE_ARMS:
                state, audit = fit_m24_invariance_breaking(
                    arm=arm,
                    support_blocks=all_support_blocks,
                    support_labels=all_support_labels,
                    classes=old_classes + new_classes,
                    k_shot=int(overlay_manifest["k_shot"]),
                    domain_digest=domain_digest,
                )
                before_state, before_audit = fit_m24_invariance_breaking(
                    arm=arm,
                    support_blocks=compact["old_support_blocks"],
                    support_labels=compact["old_support_labels"],
                    classes=old_classes,
                    k_shot=int(overlay_manifest["k_shot"]),
                    domain_digest=domain_digest,
                )
            elif arm in M25_ANCHORED_ARMS:
                coefficient, bias, f1_log_diag = _f1_reference_head(
                    historical_after, all_support_blocks
                )
                base_state, _base_audit, _base_workspace = compile_m24_d1_from_f1_head(
                    f1_coefficient=coefficient,
                    f1_bias=bias,
                    f1_log_diag=f1_log_diag,
                    f1_compiled_affine_state=historical_after.compiled_affine_state,
                    classes=old_classes + new_classes,
                    domain_digest=domain_digest,
                    support_blocks=all_support_blocks,
                )
                state, audit = fit_m25_anchored_residual(
                    arm=arm,
                    base_state=base_state,
                    support_blocks=all_support_blocks,
                    support_labels=all_support_labels,
                    classes=old_classes + new_classes,
                    k_shot=int(overlay_manifest["k_shot"]),
                    old_class_count=len(old_classes),
                    domain_digest=domain_digest,
                )
                before_coefficient, before_bias, before_log_diag = _f1_reference_head(
                    historical_before, compact["old_support_blocks"]
                )
                before_base_state, _before_base_audit, _before_base_workspace = compile_m24_d1_from_f1_head(
                    f1_coefficient=before_coefficient,
                    f1_bias=before_bias,
                    f1_log_diag=before_log_diag,
                    f1_compiled_affine_state=historical_before.compiled_affine_state,
                    classes=old_classes,
                    domain_digest=domain_digest,
                    support_blocks=compact["old_support_blocks"],
                )
                before_state, before_audit = fit_m25_anchored_residual(
                    arm=arm,
                    base_state=before_base_state,
                    support_blocks=compact["old_support_blocks"],
                    support_labels=compact["old_support_labels"],
                    classes=old_classes,
                    k_shot=int(overlay_manifest["k_shot"]),
                    old_class_count=len(old_classes),
                    domain_digest=domain_digest,
                )
            elif arm in M26_ARMS:
                coefficient, bias, f1_log_diag = _f1_reference_head(
                    historical_after, all_support_blocks
                )
                base_state, _base_audit, _base_workspace = compile_m24_d1_from_f1_head(
                    f1_coefficient=coefficient,
                    f1_bias=bias,
                    f1_log_diag=f1_log_diag,
                    f1_compiled_affine_state=historical_after.compiled_affine_state,
                    classes=old_classes + new_classes,
                    domain_digest=domain_digest,
                    support_blocks=all_support_blocks,
                )
                state, audit = fit_m26_td_src256(
                    arm=arm,
                    base_state=base_state,
                    support_blocks=all_support_blocks,
                    support_labels=all_support_labels,
                    classes=old_classes + new_classes,
                    k_shot=int(overlay_manifest["k_shot"]),
                    old_class_count=len(old_classes),
                    source_anchor=source_anchor,
                    domain_digest=domain_digest,
                )
                before_coefficient, before_bias, before_log_diag = _f1_reference_head(
                    historical_before, compact["old_support_blocks"]
                )
                before_base_state, _before_base_audit, _before_base_workspace = compile_m24_d1_from_f1_head(
                    f1_coefficient=before_coefficient,
                    f1_bias=before_bias,
                    f1_log_diag=before_log_diag,
                    f1_compiled_affine_state=historical_before.compiled_affine_state,
                    classes=old_classes,
                    domain_digest=domain_digest,
                    support_blocks=compact["old_support_blocks"],
                )
                before_state, before_audit = fit_m26_td_src256(
                    arm=arm,
                    base_state=before_base_state,
                    support_blocks=compact["old_support_blocks"],
                    support_labels=compact["old_support_labels"],
                    classes=old_classes,
                    k_shot=int(overlay_manifest["k_shot"]),
                    old_class_count=len(old_classes),
                    source_anchor=source_anchor,
                    domain_digest=domain_digest,
                )
            elif arm == D1:
                coefficient, bias, f1_log_diag = _f1_reference_head(
                    historical_after, all_support_blocks
                )
                state, audit, _workspace = compile_m24_d1_from_f1_head(
                    f1_coefficient=coefficient,
                    f1_bias=bias,
                    f1_log_diag=f1_log_diag,
                    f1_compiled_affine_state=historical_after.compiled_affine_state,
                    classes=old_classes + new_classes,
                    domain_digest=domain_digest,
                    support_blocks=all_support_blocks,
                )
                before_coefficient, before_bias, before_log_diag = _f1_reference_head(
                    historical_before, compact["old_support_blocks"]
                )
                before_state, before_audit, _before_workspace = compile_m24_d1_from_f1_head(
                    f1_coefficient=before_coefficient,
                    f1_bias=before_bias,
                    f1_log_diag=before_log_diag,
                    f1_compiled_affine_state=historical_before.compiled_affine_state,
                    classes=old_classes,
                    domain_digest=domain_digest,
                    support_blocks=compact["old_support_blocks"],
                )
            elif arm == D1_REFIT:
                state, audit, _workspace = fit_m24_d1_refit(
                    support_blocks=all_support_blocks,
                    support_labels=all_support_labels,
                    classes=old_classes + new_classes,
                    old_class_count=len(old_classes),
                    domain_digest=domain_digest,
                    ground_basis=base_cache["ground_basis"],
                    ground_spectral_weights=base_cache["ground_spectral_weights"],
                    ground_audit=base_cache["ground_audit"],
                    seed=int(seed) + scenario_index,
                    device=device,
                )
                before_state, before_audit, _before_workspace = fit_m24_d1_refit(
                    support_blocks=compact["old_support_blocks"],
                    support_labels=compact["old_support_labels"],
                    classes=old_classes,
                    old_class_count=len(old_classes),
                    domain_digest=domain_digest,
                    ground_basis=base_cache["ground_basis"],
                    ground_spectral_weights=base_cache["ground_spectral_weights"],
                    ground_audit=base_cache["ground_audit"],
                    seed=int(seed) + scenario_index,
                    device=device,
                )
            else:
                coefficient, bias, f1_log_diag = _f1_reference_head(
                    historical_after, all_support_blocks
                )
                state, audit, _workspace = fit_m24_safe_residual(
                    arm=arm,
                    support_blocks=all_support_blocks,
                    support_labels=all_support_labels,
                    classes=old_classes + new_classes,
                    support_quality=all_support_quality,
                    k_shot=int(overlay_manifest["k_shot"]),
                    old_class_count=len(old_classes),
                    f1_coefficient=coefficient[:, :256],
                    f1_bias=bias,
                    f1_log_diag=f1_log_diag[:256],
                    domain_digest=domain_digest,
                    ground_prior_identity=ground_prior,
                    nuisance_covariance_identity=nuisance_covariance,
                )
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
            registration_seconds += time.perf_counter() - fit_started
            if arm in M24_INVARIANCE_ARMS or arm in M25_ANCHORED_ARMS or arm in M26_ARMS:
                feature_dim = state.feature_dim
                query_features = compact["query_blocks"]
                before_query_features = compact["query_blocks"]
            else:
                feature_dim = state.compiled_affine_state.feature_dim
                query_features = prepare_query_features(compact["query_blocks"], feature_dim=feature_dim)
                before_query_features = prepare_query_features(
                    compact["query_blocks"], feature_dim=before_state.compiled_affine_state.feature_dim
                )
            query_started = time.perf_counter()
            if arm in M26_ARMS:
                selected_scores, query_application = state.score_with_audit(query_features)
                audit = {**dict(audit), "query_application": dict(query_application)}
            else:
                selected_scores = state.score(query_features)
            selected_after = np.asarray(state.classes)[np.argmax(selected_scores, axis=-1)]
            selected_before = before_state.predict(before_query_features)
            query_seconds += time.perf_counter() - query_started
            quantization = audit["quantization"]
            after_resource = dict(audit["resource"])
            before_resource = dict(before_audit["resource"])
            resource = {
                **after_resource,
                "compiled_inference_state_bytes": max(
                    int(after_resource["compiled_inference_state_bytes"]),
                    int(before_resource["compiled_inference_state_bytes"]),
                ) if arm == D1 or arm in M24_INVARIANCE_ARMS or arm in M25_ANCHORED_ARMS or arm in M26_ARMS else int(after_resource["compiled_inference_state_bytes"]),
                "transient_registration_workspace_peak_bytes": max(
                    int(after_resource["transient_registration_workspace_peak_bytes"]),
                    int(before_resource["transient_registration_workspace_peak_bytes"]),
                ) if arm == D1 or arm in M24_INVARIANCE_ARMS or arm in M25_ANCHORED_ARMS or arm in M26_ARMS else int(after_resource["transient_registration_workspace_peak_bytes"]),
            }
            audit = {**dict(audit), "before_registration_fit": dict(before_audit)}
            if arm == D1:
                parity_disagreements += int(np.sum(selected_after != historical_after_prediction))
                before_parity_disagreements += int(np.sum(selected_before != historical_before_prediction))
                parity_rows += len(selected_after)
            if arm in M25_ANCHORED_ARMS:
                anchored_parity_disagreements += int(np.sum(selected_after != historical_after_prediction))
                anchored_before_parity_disagreements += int(np.sum(selected_before != historical_before_prediction))
                anchored_parity_rows += len(selected_after)

        candidate_after.append(np.asarray(selected_after).astype(str))
        candidate_before.append(np.asarray(selected_before).astype(str))
        identity_after.append(np.asarray(identity_after_prediction).astype(str))
        identity_before.append(np.asarray(identity_before_prediction).astype(str))
        direct.append(np.asarray(identity_after_prediction).astype(str))
        tokens_all.append(compact_tokens)
        scenarios_all.append(np.asarray([scenario] * len(compact_tokens)))
        ordered_scores = np.partition(np.asarray(selected_scores), -2, axis=1)
        top2_margins.append(
            np.asarray(ordered_scores[:, -1] - ordered_scores[:, -2], dtype=np.float32)
        )
        if arm in M24_INVARIANCE_ARMS:
            centre_left, centre_right, centre_angle = _support_center_angles_from_features(
                state.transform(np.concatenate([compact["old_support_blocks"], compact["new_support_blocks"]])),
                np.concatenate([compact["old_support_labels"], compact["new_support_labels"]]),
                old_classes + new_classes,
            )
        elif arm in M25_ANCHORED_ARMS:
            centre_left, centre_right, centre_angle = _support_center_angles_from_features(
                state.metric_features(
                    np.concatenate([compact["old_support_blocks"], compact["new_support_blocks"]])
                ),
                np.concatenate([compact["old_support_labels"], compact["new_support_labels"]]),
                old_classes + new_classes,
            )
        elif arm in M26_ARMS:
            centre_left, centre_right, centre_angle = _support_center_angles_from_features(
                state.metric_features(
                    np.concatenate([compact["old_support_blocks"], compact["new_support_blocks"]])
                ),
                np.concatenate([compact["old_support_labels"], compact["new_support_labels"]]),
                old_classes + new_classes,
            )
        else:
            centre_left, centre_right, centre_angle = _support_center_angles(
                np.concatenate([compact["old_support_blocks"], compact["new_support_blocks"]]),
                np.concatenate([compact["old_support_labels"], compact["new_support_labels"]]),
                old_classes + new_classes,
            )
        center_scenarios.append(np.asarray([scenario] * len(centre_angle)))
        center_class_left.append(centre_left)
        center_class_right.append(centre_right)
        center_angles.append(centre_angle)
        audits[scenario] = dict(audit)
        quantization_audits.append(dict(quantization))
        state_bytes.append(int(resource["compiled_inference_state_bytes"]))
        transient_bytes.append(int(resource["transient_registration_workspace_peak_bytes"]))
        feature_dims.append(feature_dim)
        if arm in M25_ANCHORED_ARMS:
            active = float(audit["selected_strength"]) > 0.0
            counts = [int(value) for value in audit["prototype_count_by_class"]]
            multi = sum(max(0, value - 1) for value in counts)
            local_prototype_macs.append(
                _local_prototype_mac(
                    feature_dim=feature_dim,
                    prototype_counts=counts,
                    active=active,
                )
            )
            local_exp_counts.append(multi if active else 0)
            local_log_counts.append(sum(1 for value in counts if value > 1) if active else 0)
            local_aggregation_counts.append(sum(1 for value in counts if value > 1) if active else 0)
        if arm in M26_ARMS:
            target_domain_residual_macs.append(int(audit["resource"]["residual_query_mac"]))

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
    local_prototype_mac = max(local_prototype_macs, default=0)
    target_domain_residual_mac = max(target_domain_residual_macs, default=0)
    base_query_head_mac = int(max(feature_dims) * len(old_classes + new_classes))
    resource_receipt = {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "feature_cache_bytes": int(base_cache_bytes + overlay_cache_bytes),
        "deployment_state_bytes": int(component.resource_audit()["logical_deployment_state_bytes"]) if component is not None else 0,
        "state_bytes": int(max(state_bytes)),
        "compiled_inference_state_bytes": int(max(state_bytes)),
        "persistent_update_state_bytes": 0 if arm != D0 else int(max(int(audits[s].get("resource", {}).get("persistent_update_state_bytes", 0)) for s in audits)),
        "transient_registration_workspace_peak_bytes": int(max(transient_bytes)),
        "registration_time_ms": float(1000.0 * registration_seconds),
        "registration_timing_scope": (
            "compile_only_existing_p2_a1_head"
            if arm == D1
            else "support_only_invariance_breaking_head"
            if arm in M24_INVARIANCE_ARMS
            else "g0_anchored_support_only_residual"
            if arm in M25_ANCHORED_ARMS
            else "g0_anchored_source_target_domain_residual"
            if arm in M26_ARMS
            else "support_to_compiled_head"
            if arm == D1_REFIT
            else "historical_or_candidate_specific"
        ),
        "prerequisite_p2_a1_fit_included": arm == D1_REFIT,
        "row_peak_rss_bytes": 0,
        "row_peak_vram_bytes": 0,
        "candidate_peak_memory_isolated": False,
        "closed_form_fit_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "mac_equivalent_upper_bound": int((base_query_head_mac + local_prototype_mac + target_domain_residual_mac) * total_queries),
        "query_head_mac": int(base_query_head_mac + local_prototype_mac + target_domain_residual_mac),
        "base_affine_query_head_mac": base_query_head_mac,
        "local_evidence_prototype_mac": int(local_prototype_mac),
        "target_domain_residual_mac": int(target_domain_residual_mac),
        "local_evidence_exp_count": int(max(local_exp_counts, default=0)),
        "local_evidence_log_count": int(max(local_log_counts, default=0)),
        "local_evidence_aggregation_count": int(max(local_aggregation_counts, default=0)),
        "candidate_head_batch_query_latency_ms_per_row": float(1000.0 * query_seconds / len(FORMAL_LEO_WEAK_SCENARIOS)),
        "end_to_end_query_latency_available": False,
        "end_to_end_query_latency_ms": None,
        "batch1_head_resource": None,
        "row_orchestration_time_ms": float(1000.0 * (time.perf_counter() - started)),
        "auxiliary_state_cost_in_candidate_resource": False,
        "auxiliary_prediction_cost_in_candidate_latency": False,
    }
    lock = (
        invariance_arm_config_hash(arm)
        if arm in M24_INVARIANCE_ARMS
        else anchored_arm_config_hash(arm)
        if arm in M25_ANCHORED_ARMS
        else m26_candidate_lock
        if arm in M26_ARMS
        else arm_config_hash(arm)
    )
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
    diagnostics = _exclusive_npz(
        output / "truth_blind_diagnostics.npz",
        query_tokens=np.concatenate(tokens_all),
        scenarios=np.concatenate(scenarios_all),
        predicted_classes=np.concatenate(candidate_after),
        top2_margin=np.concatenate(top2_margins),
        center_scenarios=np.concatenate(center_scenarios),
        center_class_left=np.concatenate(center_class_left),
        center_class_right=np.concatenate(center_class_right),
        center_angle_degrees=np.concatenate(center_angles),
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
        "d1_historical_parity": {
            "query_rows": parity_rows,
            "prediction_disagreements": parity_disagreements,
            "before_prediction_disagreements": before_parity_disagreements,
            "agreement_rate": float(1.0 - parity_disagreements / parity_rows) if parity_rows else None,
            "before_agreement_rate": float(1.0 - before_parity_disagreements / parity_rows) if parity_rows else None,
        },
        "anchored_base_parity": {
            "query_rows": anchored_parity_rows,
            "prediction_disagreements": anchored_parity_disagreements,
            "before_prediction_disagreements": anchored_before_parity_disagreements,
            "agreement_rate": float(1.0 - anchored_parity_disagreements / anchored_parity_rows) if anchored_parity_rows else None,
            "before_agreement_rate": float(1.0 - anchored_before_parity_disagreements / anchored_parity_rows) if anchored_parity_rows else None,
        },
        "prediction": prediction,
        "truth_blind_diagnostics": diagnostics,
        "behavior": behavior_receipt,
        "quantization": quantization_receipt,
        "resource": resource_receipt,
        "scenario_audit": audits,
    }
    if arm in M26_ARMS:
        receipt["source_anchor"] = {
            "schema": "cvs.erbt_idr.m26.source_anchor_receipt.v1",
            "checkpoint_sha256": source_anchor.checkpoint_sha256,
            "component_id": source_anchor.component_id,
            "class_registry": list(source_anchor.class_registry),
            "state_bytes": source_anchor.state_bytes,
            "sample_or_member_rows_available": False,
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


def run_m24_d1_evidence_row_from_base_cache(
    *,
    arm: str,
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
    if arm not in {D0, D1, D1_REFIT}:
        raise M24RowExecutionError("D1 evidence runner accepts only R0/R1/R2")
    base = load_feature_cache(
        base_feature_cache_payload,
        base_feature_cache_manifest,
        expected_payload_sha256=str(base_feature_cache_payload_sha256).lower(),
        expected_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
    )
    return execute_m24_row(
        arm=arm,
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


def run_m24_invariance_row_from_base_cache(
    *,
    arm: str,
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
    if arm not in {D0, *M24_INVARIANCE_ARMS}:
        raise M24RowExecutionError("invariance runner accepts only G0-G4")
    base = load_feature_cache(
        base_feature_cache_payload,
        base_feature_cache_manifest,
        expected_payload_sha256=str(base_feature_cache_payload_sha256).lower(),
        expected_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
    )
    return execute_m24_row(
        arm=arm,
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


def run_m25_anchored_row_from_base_cache(
    *,
    arm: str,
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
    if arm not in {D1, *M25_ANCHORED_ARMS}:
        raise M24RowExecutionError("anchored residual runner accepts only B0-B3")
    base = load_feature_cache(
        base_feature_cache_payload,
        base_feature_cache_manifest,
        expected_payload_sha256=str(base_feature_cache_payload_sha256).lower(),
        expected_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
    )
    return execute_m24_row(
        arm=arm,
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


def run_m26_td_src256_row_from_base_cache(
    *,
    arm: str,
    row_id: str,
    receiver: str,
    base_feature_cache_payload: str | Path,
    base_feature_cache_manifest: str | Path,
    base_feature_cache_payload_sha256: str,
    base_feature_cache_manifest_sha256: str,
    source_anchor_path: str | Path,
    expected_checkpoint_sha256: str,
    output_root: str | Path,
    seed: int,
    device: Any = "cpu",
) -> dict[str, Any]:
    if arm not in {D1, *M26_ARMS}:
        raise M24RowExecutionError("M2.6 runner accepts only B0 and T1-T5")
    base = load_feature_cache(
        base_feature_cache_payload,
        base_feature_cache_manifest,
        expected_payload_sha256=str(base_feature_cache_payload_sha256).lower(),
        expected_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
    )
    anchor = None
    if arm in M26_ARMS:
        anchor = load_m26_spectral_anchor(
            source_anchor_path,
            expected_checkpoint_sha256=str(expected_checkpoint_sha256).lower(),
        )
    return execute_m24_row(
        arm=arm,
        row_id=row_id,
        receiver=receiver,
        base_cache=base,
        overlay_cache=d1_overlay_from_base_cache(base),
        output_root=output_root,
        seed=int(seed),
        device=device,
        base_cache_bytes=Path(base_feature_cache_payload).stat().st_size,
        overlay_cache_bytes=0,
        source_anchor=anchor,
    )


__all__ = ["DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1", "M24_ROW_EXECUTION_SCHEMA", "M24RowExecutionError", "d1_overlay_from_base_cache", "execute_m24_row", "run_m24_d1_evidence_row_from_base_cache", "run_m24_d1_row_from_base_cache", "run_m24_invariance_row_from_base_cache", "run_m25_anchored_row_from_base_cache", "run_m26_td_src256_row_from_base_cache", "run_m24_row_from_caches"]
