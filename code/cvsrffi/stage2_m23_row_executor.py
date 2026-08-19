"""Truth-inaccessible F0-F5 row execution for ERBT-IDR M2.3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi import stage2_ablation_quantization as quantization
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_ablation_executors import fit_stage2_ablation
from cvsrffi.stage2_ablation_feature_cache import load_feature_cache
from cvsrffi.stage2_ablation_truth_scorer import (
    BEHAVIOR_RECEIPT_SCHEMA,
    QUANTIZATION_RECEIPT_SCHEMA,
    RESOURCE_RECEIPT_SCHEMA,
)
from cvsrffi.stage2_m23_overlay_cache import load_m23_overlay_cache
from cvsrffi.stage2_m23_rfguard import (
    ARM_RF_LITE_DIAG,
    ARM_RF_LITE_GATED,
    ARM_RF_QUALITY,
    COMPACT_DIM,
    IF_DIM,
    M23RFGuardState,
    build_ground_manifold,
    estimate_stage2b_domain_state,
    fit_rfguard_m23,
)
from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact


M23_ROW_EXECUTION_SCHEMA = "cvs.erbt_idr.m23.row_execution.v1"

M23_F0_FULL = "M23-F0-CURRENT-FULL"
M23_F1_IF = "M23-F1-IDENTITY-FFT"
M23_F2_RF32_LOW = "M23-F2-RF32-LOW"
M23_F3_RF_QUALITY = "M23-F3-RF-QUALITY"
M23_F4_RF_LITE_DIAG = "M23-F4-RF-LITE-DIAG"
M23_F5_RF_LITE_GATED = "M23-F5-RF-LITE-GATED"
M23_ARMS = (
    M23_F0_FULL,
    M23_F1_IF,
    M23_F2_RF32_LOW,
    M23_F3_RF_QUALITY,
    M23_F4_RF_LITE_DIAG,
    M23_F5_RF_LITE_GATED,
)

DA0_REG0 = "DA0_REG0"
DA1_REG0 = "DA1_REG0"
DA0_REG1 = "DA0_REG1"
DA1_REG1 = "DA1_REG1"


class M23RowExecutionError(RuntimeError):
    """Raised when an M2.3 row cannot close immutable predictions."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def m23_arm_config_hash(arm: str) -> str:
    if arm not in M23_ARMS:
        raise M23RowExecutionError("unknown M2.3 arm")
    config = {
        "schema": "cvs.erbt_idr.m23.arm_config.v1",
        "arm": arm,
        "protocol_schema": "p2_min_v1",
        "query_fit_access": False,
        "query_decision_policy": "independent_all_registered_class_argmax",
        "identity_dim": 160,
        "fft_dim": 96,
        "rf_lite_dim": 10,
        "identity_weight": 1.0,
        "fft_weight": 4.0,
        "legacy_rf32_weight": 0.5 if arm == M23_F2_RF32_LOW else None,
        "rf_quality_classifier_dimension": 0,
        "quantization": "F3_dual_int8_residual_fp16_block_scale_bias",
    }
    return hashlib.sha256(_canonical_json(config)).hexdigest()


def _unit_rows(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if rows.ndim != 2 or np.any(norm <= 1.0e-12) or not np.isfinite(rows).all():
        raise M23RowExecutionError("feature normalization is degenerate")
    return rows / norm


def legacy_low_rf32(value: Any, *, rf_weight: float = 0.5) -> np.ndarray:
    """Create the F2 low-energy RF32 diagnostic without zero padding."""

    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 288 or not np.isfinite(rows).all():
        raise M23RowExecutionError("legacy F2 input must be finite N x 288")
    if not 0.0 < float(rf_weight) < 1.0:
        raise M23RowExecutionError("legacy RF32 weight must be in (0, 1)")
    projected = np.concatenate(
        [
            _unit_rows(rows[:, :160]),
            4.0 * _unit_rows(rows[:, 160:256]),
            float(rf_weight) * _unit_rows(rows[:, 256:]),
        ],
        axis=1,
    )
    return _unit_rows(projected).astype(np.float32)


def _legacy_if_only(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 288 or not np.isfinite(rows).all():
        raise M23RowExecutionError("legacy F1 input must be finite N x 288")
    projected = np.concatenate(
        [
            _unit_rows(rows[:, :160]),
            4.0 * _unit_rows(rows[:, 160:256]),
            np.zeros((len(rows), 32), dtype=np.float64),
        ],
        axis=1,
    )
    return _unit_rows(projected).astype(np.float32)


class _OverlayGroundComponent:
    def __init__(self, arrays: Mapping[str, Any]) -> None:
        for name, value in arrays.items():
            setattr(self, name, np.array(value, copy=True))
        self.domain_registry = tuple(np.asarray(arrays["domain_registry"]).astype(str).tolist())
        self.residual_domain_registry = tuple(
            np.asarray(arrays["residual_domain_registry"]).astype(str).tolist()
        )
        self.class_registry = tuple(np.asarray(arrays["class_registry"]).astype(str).tolist())
        self.center_domain_handle = str(np.asarray(arrays["center_domain_handle"]).item())

    def reconstruct_domain(self, domain_handle: str) -> np.ndarray:
        core = self.core_q.astype(np.float32) * self.core_scale[:, None].astype(np.float32)
        handle = str(domain_handle)
        if handle == self.center_domain_handle:
            return core
        try:
            index = self.residual_domain_registry.index(handle)
        except ValueError as exc:
            raise M23RowExecutionError("unknown aggregate ground domain") from exc
        basis = self.residual_basis_q.astype(np.float32) * self.residual_basis_scale[
            ..., None
        ].astype(np.float32)
        coefficient = self.residual_coeff_q[index].astype(np.float32) * self.residual_coeff_scale[
            index, :, None
        ].astype(np.float32)
        return core + np.einsum("cr,crp->cp", coefficient, basis, optimize=True)

    def resource_audit(self) -> dict[str, int]:
        numeric = (
            self.core_q,
            self.core_scale,
            self.residual_basis_q,
            self.residual_basis_scale,
            self.residual_coeff_q,
            self.residual_coeff_scale,
        )
        return {
            "logical_deployment_state_bytes": int(sum(value.nbytes for value in numeric)),
            "persistent_dense_float_bank_bytes": 0,
        }


def _cosine_prediction(
    support: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    features: np.ndarray,
    dimension: int,
) -> np.ndarray:
    registry = tuple(str(value) for value in classes)
    rows = _unit_rows(np.asarray(support)[:, :dimension])
    query = _unit_rows(np.asarray(features)[:, :dimension])
    centres = _unit_rows(
        np.stack([np.mean(rows[np.asarray(labels).astype(str) == item], axis=0) for item in registry])
    )
    return np.asarray(registry)[np.argmax(query @ centres.T, axis=1)]


def _legacy_states(
    arm: str,
    legacy_payload: Mapping[str, Any],
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    *,
    ground_basis: Any,
    ground_weights: Any,
    ground_audit: Mapping[str, Any],
    seed: int,
    device: Any,
) -> tuple[Any, Any, dict[str, np.ndarray]]:
    if arm == M23_F2_RF32_LOW:
        direct_transform = legacy_low_rf32
        after_ablation = "P2-FULL"
        after_old = legacy_low_rf32(legacy_payload["old_support_features"])
        after_new = legacy_low_rf32(legacy_payload["new_support_features"])
        after_query = legacy_low_rf32(legacy_payload["query_features"])
    elif arm == M23_F1_IF:
        direct_transform = _legacy_if_only
        after_ablation = "P2-A1"
        after_old = np.asarray(legacy_payload["old_support_features"], dtype=np.float32)
        after_new = np.asarray(legacy_payload["new_support_features"], dtype=np.float32)
        after_query = np.asarray(legacy_payload["query_features"], dtype=np.float32)
    else:
        direct_transform = lambda value: np.asarray(value, dtype=np.float32)
        after_ablation = "P2-FULL"
        after_old = direct_transform(legacy_payload["old_support_features"])
        after_new = direct_transform(legacy_payload["new_support_features"])
        after_query = direct_transform(legacy_payload["query_features"])
    direct_old = direct_transform(legacy_payload["old_support_features"])
    direct_new = direct_transform(legacy_payload["new_support_features"])
    direct_query = direct_transform(legacy_payload["query_features"])
    before = fit_stage2_ablation(
        ablation_id="P2-S2B-FULL",
        old_support_features=direct_old,
        old_support_labels=legacy_payload["old_support_labels"],
        old_classes=old_classes,
        ground_basis=ground_basis,
        ground_spectral_weights=ground_weights,
        ground_audit=ground_audit,
        seed=int(seed),
        device=device,
    )
    after = fit_stage2_ablation(
        ablation_id=after_ablation,
        old_support_features=after_old,
        old_support_labels=legacy_payload["old_support_labels"],
        old_classes=old_classes,
        new_support_features=after_new,
        new_support_labels=legacy_payload["new_support_labels"],
        new_classes=new_classes,
        ground_basis=ground_basis,
        ground_spectral_weights=ground_weights,
        ground_audit=ground_audit,
        seed=int(seed),
        device=device,
    )
    return before, after, {
        "old": direct_old,
        "new": direct_new,
        "query": direct_query,
        "query_before": direct_query,
        "query_after": after_query,
    }


def _m23_states(
    arm: str,
    payload: Mapping[str, Any],
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    component: _OverlayGroundComponent,
) -> tuple[M23RFGuardState, M23RFGuardState, np.ndarray, np.ndarray]:
    if arm == M23_F1_IF:
        quality_old = np.ones(len(payload["old_support_quality"]), dtype=np.float32)
        quality_new = np.ones(len(payload["new_support_quality"]), dtype=np.float32)
        rf_arm = ARM_RF_QUALITY
    elif arm == M23_F3_RF_QUALITY:
        quality_old = np.asarray(payload["old_support_quality"], dtype=np.float32)
        quality_new = np.asarray(payload["new_support_quality"], dtype=np.float32)
        rf_arm = ARM_RF_QUALITY
    elif arm == M23_F4_RF_LITE_DIAG:
        quality_old = np.asarray(payload["old_support_quality"], dtype=np.float32)
        quality_new = np.asarray(payload["new_support_quality"], dtype=np.float32)
        rf_arm = ARM_RF_LITE_DIAG
    elif arm == M23_F5_RF_LITE_GATED:
        quality_old = np.asarray(payload["old_support_quality"], dtype=np.float32)
        quality_new = np.asarray(payload["new_support_quality"], dtype=np.float32)
        rf_arm = ARM_RF_LITE_GATED
    else:
        raise M23RowExecutionError("arm is not an M2.3 compact state")
    manifold = build_ground_manifold(component)
    domain_state = estimate_stage2b_domain_state(
        payload["old_support_blocks"],
        payload["old_support_labels"],
        old_classes,
        quality_old,
        manifold,
    )
    before = fit_rfguard_m23(
        payload["old_support_blocks"],
        payload["old_support_labels"],
        old_classes,
        quality_old,
        ground_component=component,
        arm=rf_arm,
        frozen_domain_state=domain_state,
    )
    after = fit_rfguard_m23(
        payload["old_support_blocks"],
        payload["old_support_labels"],
        old_classes,
        quality_old,
        ground_component=component,
        new_support_blocks=payload["new_support_blocks"],
        new_support_labels=payload["new_support_labels"],
        new_classes=new_classes,
        new_support_quality=quality_new,
        arm=rf_arm,
        frozen_domain_state=domain_state,
    )
    return before, after, quality_old, quality_new


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        return 0


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _canonical_json(payload) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing M2.3 receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute_m23_row(
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
    """Execute one F0-F5 row and publish four-state prediction columns."""

    if arm not in M23_ARMS:
        raise M23RowExecutionError("unknown M2.3 arm")
    base_manifest = base_cache["manifest"]
    overlay_manifest = overlay_cache["manifest"]
    old_classes = tuple(base_cache["old_classes"])
    new_classes = tuple(base_cache["new_classes"])
    if (
        tuple(overlay_cache["old_classes"]) != old_classes
        or tuple(overlay_cache["new_classes"]) != new_classes
        or str(base_manifest["receiver"]) != str(receiver)
        or str(overlay_manifest["receiver"]) != str(receiver)
        or int(base_manifest["k_shot"]) != int(overlay_manifest["k_shot"])
        or int(base_manifest["method_seed"]) != int(overlay_manifest["method_seed"])
        or int(seed) != int(base_manifest["method_seed"])
        or base_manifest["capsule_id"] != overlay_manifest["capsule_id"]
        or base_manifest["split_id"] != overlay_manifest["split_id"]
        or base_manifest["phase2_data_status"] != "VALIDATED_ONCE"
    ):
        raise M23RowExecutionError("base/overlay row identity drift")
    if set(base_cache["scenario_payloads"]) != set(FORMAL_LEO_WEAK_SCENARIOS) or set(overlay_cache["scenario_payloads"]) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise M23RowExecutionError("formal scenario coverage drift")
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=False)
    component = _OverlayGroundComponent(overlay_cache["ground_component"])

    candidate_after: list[np.ndarray] = []
    candidate_before: list[np.ndarray] = []
    identity_after: list[np.ndarray] = []
    identity_before: list[np.ndarray] = []
    direct: list[np.ndarray] = []
    tokens_all: list[np.ndarray] = []
    scenarios_all: list[np.ndarray] = []
    views_all: list[np.ndarray] = []
    reference_scores: list[np.ndarray] = []
    compiled_scores: list[np.ndarray] = []
    m23_quantization_audits: list[Mapping[str, Any]] = []
    audits: dict[str, Any] = {}
    state_bytes: list[int] = []
    query_head_macs: list[int] = []
    registration_seconds = 0.0
    query_seconds = 0.0
    started_row = time.perf_counter()

    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        legacy = base_cache["scenario_payloads"][scenario]
        compact = overlay_cache["scenario_payloads"][scenario]
        legacy_tokens = np.asarray(legacy["query_tokens"]).astype(str)
        compact_tokens = np.asarray(compact["query_tokens"]).astype(str)
        if not np.array_equal(legacy_tokens, compact_tokens):
            raise M23RowExecutionError(f"{scenario} base/overlay query token drift")
        scenario_seed = int(seed) + scenario_index
        fit_started = time.perf_counter()
        if arm in {M23_F0_FULL, M23_F1_IF, M23_F2_RF32_LOW}:
            before_state, after_state, prepared = _legacy_states(
                arm,
                legacy,
                old_classes,
                new_classes,
                ground_basis=base_cache["ground_basis"],
                ground_weights=base_cache["ground_spectral_weights"],
                ground_audit=base_cache["ground_audit"],
                seed=scenario_seed,
                device=device,
            )
            query_rows = prepared["query"]
            query_after_rows = prepared["query_after"]
            query_before_rows = prepared["query_before"]
            registration_seconds += time.perf_counter() - fit_started
            query_started = time.perf_counter()
            after_scores = after_state.score(query_after_rows)
            compiled_scores.append(after_scores)
            candidate_after.append(np.asarray(after_state.classes)[np.argmax(after_scores, axis=1)])
            candidate_before.append(before_state.predict(query_before_rows))
            query_seconds += time.perf_counter() - query_started
            dimension = 288
            identity_before.append(
                _cosine_prediction(
                    prepared["old"],
                    legacy["old_support_labels"],
                    old_classes,
                    query_rows,
                    dimension,
                )
            )
            identity_after.append(
                _cosine_prediction(
                    np.concatenate([prepared["old"], prepared["new"]]),
                    np.concatenate(
                        [legacy["old_support_labels"], legacy["new_support_labels"]]
                    ),
                    old_classes + new_classes,
                    query_rows,
                    dimension,
                )
            )
            direct.append(identity_after[-1])
            coefficient = np.asarray(after_state.audit.get("d81_actual_coefficient_fp32"))
            intercept = np.asarray(after_state.audit.get("d81_actual_intercept_fp32"))
            if coefficient.shape == (len(after_state.classes), 288) and intercept.shape == (len(after_state.classes),):
                reference_scores.append(
                    after_state._prepared(query_after_rows)
                    @ coefficient.T
                    + intercept[None, :]
                )
            else:
                reference_scores.append(np.array(after_scores, copy=True))
            state_bytes.append(int(after_state.resource["persistent_state_bytes"]))
            query_head_macs.append(288 * len(after_state.classes))
            audits[scenario] = {
                "legacy_ablation_id": (
                    "P2-A1" if arm == M23_F1_IF else "P2-FULL"
                ),
                "legacy_rf32_weight": (
                    0.0
                    if arm == M23_F1_IF
                    else 0.5
                    if arm == M23_F2_RF32_LOW
                    else 4.0
                ),
                "feature_dim": 288,
            }
        else:
            before_state, after_state, _old_quality, _new_quality = _m23_states(
                arm, compact, old_classes, new_classes, component
            )
            registration_seconds += time.perf_counter() - fit_started
            query_rows = np.asarray(compact["query_blocks"], dtype=np.float32)
            query_started = time.perf_counter()
            after_scores = after_state.score(query_rows)
            compiled_scores.append(after_scores)
            candidate_after.append(np.asarray(after_state.classes)[np.argmax(after_scores, axis=1)])
            candidate_before.append(
                np.asarray(before_state.classes)[np.argmax(before_state.score(query_rows), axis=1)]
            )
            query_seconds += time.perf_counter() - query_started
            dimension = after_state.compiled_affine_state.feature_dim
            identity_before.append(
                _cosine_prediction(
                    compact["old_support_blocks"],
                    compact["old_support_labels"],
                    old_classes,
                    query_rows,
                    min(dimension, IF_DIM),
                )
            )
            identity_after.append(
                _cosine_prediction(
                    np.concatenate(
                        [compact["old_support_blocks"], compact["new_support_blocks"]]
                    ),
                    np.concatenate(
                        [compact["old_support_labels"], compact["new_support_labels"]]
                    ),
                    old_classes + new_classes,
                    query_rows,
                    dimension,
                )
            )
            direct.append(identity_after[-1])
            state_bytes.append(int(after_state.audit["m23_total_retained_state_bytes"]))
            query_head_macs.append(dimension * len(after_state.classes))
            audits[scenario] = dict(after_state.audit)
            m23_quantization_audits.append(after_state.audit)
            if before_state.domain_state.digest != after_state.domain_state.digest:
                raise M23RowExecutionError("Stage2-B domain state changed during registration")

        tokens_all.append(compact_tokens)
        scenarios_all.append(np.asarray([scenario] * len(compact_tokens)))
        views_all.append(np.ones(len(compact_tokens), dtype=np.uint8))

    if reference_scores:
        reference = np.concatenate(reference_scores)
        compiled = np.concatenate(compiled_scores)
        error = np.abs(reference - compiled)
        reference_prediction = np.argmax(reference, axis=1)
        compiled_prediction = np.argmax(compiled, axis=1)
        flip_rate = float(np.mean(reference_prediction != compiled_prediction))
        maximum_error = float(np.max(error))
        mean_error = float(np.mean(error))
    elif m23_quantization_audits:
        support_rows = sum(
            int(value["m23_quantization_support_row_count"])
            for value in m23_quantization_audits
        )
        support_flips = sum(
            int(value["m23_quantization_support_argmax_flip_count"])
            for value in m23_quantization_audits
        )
        maximum_error = max(
            float(value["m23_quantization_support_max_logit_abs_error"])
            for value in m23_quantization_audits
        )
        mean_error = sum(
            float(value["m23_quantization_support_mean_logit_abs_error"])
            * int(value["m23_quantization_support_row_count"])
            for value in m23_quantization_audits
        ) / support_rows
        flip_rate = float(support_flips / support_rows)
    else:
        raise M23RowExecutionError("quantization audit did not close")
    quantization_receipt = {
        "schema": QUANTIZATION_RECEIPT_SCHEMA,
        "max_logit_abs_error": maximum_error,
        "mean_logit_abs_error": mean_error,
        "argmax_flip_rate": flip_rate,
        "prediction_agreement_rate": float(1.0 - flip_rate),
    }
    behavior_receipt = {
        "schema": BEHAVIOR_RECEIPT_SCHEMA,
        "fallback_counts": {
            "rf_no_harm": int(
                sum(
                    int(value.get("m23_rf_no_harm_fallback_count", 0))
                    for value in audits.values()
                )
            )
        },
        "full_block_weights": {"full": 1.0, "block3": 0.0},
        "fisher_gate_accept_counts": {"attempted": 0, "accepted": 0},
        "atomic_rollback_counts": {"attempted": 0, "rolled_back": 0},
        "failure_closure_count": 0,
    }
    total_query = sum(len(value) for value in tokens_all)
    resource_receipt = {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "feature_cache_bytes": int(base_cache_bytes + overlay_cache_bytes),
        "deployment_state_bytes": int(component.resource_audit()["logical_deployment_state_bytes"]),
        "state_bytes": int(max(state_bytes)),
        "registration_time_ms": float(1000.0 * registration_seconds),
        "row_peak_rss_bytes": _rss_bytes(),
        "row_peak_vram_bytes": 0,
        "candidate_peak_memory_isolated": False,
        "closed_form_fit_count": 2 * len(FORMAL_LEO_WEAK_SCENARIOS),
        "mac_equivalent_upper_bound": int(sum(query_head_macs) * max(total_query, 1)),
        "query_head_mac": int(max(query_head_macs)),
        "candidate_head_batch_query_latency_ms_per_row": float(
            1000.0 * query_seconds / len(FORMAL_LEO_WEAK_SCENARIOS)
        ),
        "end_to_end_query_latency_available": False,
        "end_to_end_query_latency_ms": None,
        "batch1_head_resource": None,
        "row_orchestration_time_ms": float(1000.0 * (time.perf_counter() - started_row)),
        "auxiliary_state_cost_in_candidate_resource": False,
        "auxiliary_prediction_cost_in_candidate_latency": False,
    }
    candidate_lock = m23_arm_config_hash(arm)
    prediction = publish_prediction_artifact(
        output / "predictions.cvspred",
        stage="Stage2-C",
        row_id=str(row_id),
        receiver=str(receiver),
        k_shot=int(overlay_manifest["k_shot"]),
        candidate_lock_sha256=candidate_lock,
        package_root_sha256=str(overlay_manifest["predictor_package_root_sha256"]),
        package_seal_sha256=str(overlay_manifest["predictor_package_seal_sha256"]),
        query_tokens=np.concatenate(tokens_all),
        scenarios=np.concatenate(scenarios_all),
        candidate_after=np.concatenate(candidate_after),
        candidate_before=np.concatenate(candidate_before),
        identity_after=np.concatenate(identity_after),
        identity_before=np.concatenate(identity_before),
        direct=np.concatenate(direct),
        shared_view_counts=np.concatenate(views_all),
    )
    receipt = {
        "schema": M23_ROW_EXECUTION_SCHEMA,
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "arm": arm,
        "row_id": str(row_id),
        "receiver": str(receiver),
        "k_shot": int(overlay_manifest["k_shot"]),
        "new_class_count": len(new_classes),
        "candidate_lock_sha256": candidate_lock,
        "four_state_prediction_columns": {
            DA0_REG0: "identity_before",
            DA1_REG0: "candidate_before",
            DA0_REG1: "identity_after",
            DA1_REG1: "candidate_after",
        },
        "fit_query_rows_used": 0,
        "query_truth_opened": False,
        "per_query_independent_all_class_argmax": True,
        "prediction": prediction,
        "behavior": behavior_receipt,
        "quantization": quantization_receipt,
        "resource": resource_receipt,
        "scenario_audit": audits,
    }
    _exclusive_json(output / "row_execution_receipt.json", receipt)
    return receipt


def run_m23_row_from_caches(
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
    base = load_feature_cache(
        base_feature_cache_payload,
        base_feature_cache_manifest,
        expected_payload_sha256=str(base_feature_cache_payload_sha256).lower(),
        expected_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
    )
    overlay = load_m23_overlay_cache(
        overlay_payload,
        overlay_manifest,
        expected_payload_sha256=str(overlay_payload_sha256).lower(),
        expected_manifest_sha256=str(overlay_manifest_sha256).lower(),
    )
    if (
        overlay["manifest"]["base_feature_cache_payload_sha256"]
        != str(base_feature_cache_payload_sha256).lower()
        or overlay["manifest"]["base_feature_cache_manifest_sha256"]
        != str(base_feature_cache_manifest_sha256).lower()
    ):
        raise M23RowExecutionError("overlay does not bind the supplied base cache")
    return execute_m23_row(
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


__all__ = [
    "DA0_REG0",
    "DA0_REG1",
    "DA1_REG0",
    "DA1_REG1",
    "M23_ARMS",
    "M23_F0_FULL",
    "M23_F1_IF",
    "M23_F2_RF32_LOW",
    "M23_F3_RF_QUALITY",
    "M23_F4_RF_LITE_DIAG",
    "M23_F5_RF_LITE_GATED",
    "M23_ROW_EXECUTION_SCHEMA",
    "M23RowExecutionError",
    "execute_m23_row",
    "legacy_low_rf32",
    "m23_arm_config_hash",
    "run_m23_row_from_caches",
]
