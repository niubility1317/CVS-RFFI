"""Truth-inaccessible feature-row execution for the full Phase2 ablation.

The executor receives only frozen deployment state, legal support features,
unlabelled query features, and opaque query/class handles. It fits each state
from deployment/support inputs, makes independent all-class predictions, and
atomically publishes one immutable ``.cvspred`` artifact before any scorer is
allowed to open query truth.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cvsrffi import stage2_ablation_quantization as quantization
from cvsrffi.stage2_ablation_executors import (
    Stage2AblationFittedState,
    fit_stage2_ablation,
)
from cvsrffi.stage2_ablation_factory import get_stage2_arm
from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS


ROW_EXECUTION_SCHEMA = "cvs.full_ablation.phase2.row_execution.v1"
BEHAVIOR_RECEIPT_SCHEMA = (
    "cvs.full_ablation.phase2.behavior_receipt.v1"
)
QUANTIZATION_RECEIPT_SCHEMA = (
    "cvs.full_ablation.phase2.quantization_receipt.v1"
)
RESOURCE_RECEIPT_SCHEMA = (
    "cvs.full_ablation.phase2.resource_receipt.v1"
)


class Stage2AblationRowExecutionError(RuntimeError):
    """Raised when a feature row cannot close its prediction artifact."""


def _prepare_cuda_memory_audit(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value if os.name == "nt" else value * 1024
        except (ImportError, OSError, ValueError):
            return 0


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
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
                raise OSError("short write while publishing row receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _array(
    payload: Mapping[str, Any], name: str, *, rows: int | None = None
) -> np.ndarray:
    if name not in payload:
        raise Stage2AblationRowExecutionError(
            f"scenario payload lacks {name}"
        )
    value = np.asarray(payload[name])
    if rows is not None and len(value) != rows:
        raise Stage2AblationRowExecutionError(
            f"scenario payload {name} row count drift"
        )
    return value


def _fit(
    ablation_id: str,
    *,
    payload: Mapping[str, Any],
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    deployment_prototypes: Any,
    ground_basis: Any,
    ground_spectral_weights: Any,
    ground_audit: Mapping[str, Any],
    seed: int,
    device: Any,
) -> Stage2AblationFittedState:
    stage = get_stage2_arm(ablation_id).stage
    if stage == "stage2a":
        return fit_stage2_ablation(
            ablation_id=ablation_id,
            old_support_features=None,
            old_support_labels=None,
            old_classes=old_classes,
            deployment_prototypes=deployment_prototypes,
            seed=seed,
            device=device,
        )
    kwargs: dict[str, Any] = {
        "ablation_id": ablation_id,
        "old_support_features": _array(
            payload, "old_support_features"
        ),
        "old_support_labels": _array(payload, "old_support_labels"),
        "old_classes": old_classes,
        "ground_basis": ground_basis,
        "ground_spectral_weights": ground_spectral_weights,
        "ground_audit": ground_audit,
        "seed": seed,
        "device": device,
    }
    if stage == "stage2c":
        kwargs.update(
            {
                "new_support_features": _array(
                    payload, "new_support_features"
                ),
                "new_support_labels": _array(
                    payload, "new_support_labels"
                ),
                "new_classes": new_classes,
            }
        )
    return fit_stage2_ablation(**kwargs)


def _full_block_weights(
    state: Stage2AblationFittedState,
) -> dict[str, float]:
    audit = state.audit
    full = audit.get("d46_full_weight_by_class")
    block = audit.get("d46_block_weight_by_class")
    if full is not None and block is not None:
        full_value = float(np.mean(np.asarray(full, dtype=np.float64)))
        block_value = float(np.mean(np.asarray(block, dtype=np.float64)))
        total = full_value + block_value
        if np.isfinite(total) and total > 0.0:
            return {
                "full": full_value / total,
                "block3": block_value / total,
            }
    if state.ablation_id == "P2-D0":
        return {"full": 1.0, "block3": 0.0}
    if state.ablation_id == "P2-D1":
        return {"full": 0.0, "block3": 1.0}
    return {"full": 0.5, "block3": 0.5}


def _behavior_receipt(
    candidate_states: Sequence[Stage2AblationFittedState],
) -> dict[str, Any]:
    fallback = 0
    attempted = 0
    accepted = 0
    atomic_attempted = 0
    rolled_back = 0
    for state in candidate_states:
        audit = state.audit
        if (
            audit.get("d92_status") == "k1_k2_exact_d81_fallback"
            or audit.get("d62_boundary_status")
            == "k1_k2_exact_d46_fallback"
            or audit.get("k_le2_unit_covariance_fallback") is True
        ):
            fallback += 1
        if "d62_final_accept_mask" in audit:
            mask = np.asarray(
                audit["d62_final_accept_mask"], dtype=bool
            )
            attempted += int(mask.size)
            accepted += int(np.sum(mask))
            atomic_attempted += 1
            rolled_back += int(
                not bool(audit.get("d62_joint_atomic_safe", False))
            )
    return {
        "schema": BEHAVIOR_RECEIPT_SCHEMA,
        "fallback_counts": {
            "k_le2_exact": int(fallback),
            "other": 0,
        },
        "full_block_weights": _full_block_weights(candidate_states[-1]),
        "fisher_gate_accept_counts": {
            "attempted": attempted,
            "accepted": accepted,
        },
        "atomic_rollback_counts": {
            "attempted": atomic_attempted,
            "rolled_back": rolled_back,
        },
        "failure_closure_count": 0,
    }


def _quantization_receipt(
    *,
    candidate: Stage2AblationFittedState,
    reference: Stage2AblationFittedState | None,
    query_features: Sequence[np.ndarray],
) -> dict[str, Any]:
    if reference is None:
        return {
            "schema": QUANTIZATION_RECEIPT_SCHEMA,
            "max_logit_abs_error": 0.0,
            "mean_logit_abs_error": 0.0,
            "argmax_flip_rate": 0.0,
            "prediction_agreement_rate": 1.0,
        }
    candidate_scores = np.concatenate(
        [candidate.score(rows) for rows in query_features], axis=0
    )
    reference_scores = np.concatenate(
        [reference.score(rows) for rows in query_features], axis=0
    )
    error = np.abs(candidate_scores - reference_scores)
    changed = np.argmax(candidate_scores, axis=1) != np.argmax(
        reference_scores, axis=1
    )
    flip_rate = float(np.mean(changed))
    return {
        "schema": QUANTIZATION_RECEIPT_SCHEMA,
        "max_logit_abs_error": float(np.max(error)),
        "mean_logit_abs_error": float(np.mean(error)),
        "argmax_flip_rate": flip_rate,
        "prediction_agreement_rate": float(1.0 - flip_rate),
    }


def _resource_receipt(
    *,
    candidate_states: Sequence[Stage2AblationFittedState],
    feature_cache_bytes: int,
    deployment_state_bytes: int,
    peak_rss_bytes: int,
    peak_vram_bytes: int,
    registration_seconds: float,
    query_seconds: float,
    query_count: int,
    row_orchestration_seconds: float,
) -> dict[str, Any]:
    candidate = candidate_states[-1]
    batch1_resource = (
        dict(
            quantization.resource_report(
                candidate.compiled_affine_state,
                query_feature=np.zeros((1, 288), dtype=np.float32),
                latency_repeats=5,
                latency_warmup=1,
            )
        )
        if candidate.compiled_affine_state is not None
        else None
    )
    registered_count = len(candidate.classes)
    feature_dim = 288
    return {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "feature_cache_bytes": int(feature_cache_bytes),
        "deployment_state_bytes": int(deployment_state_bytes),
        "state_bytes": int(candidate.resource["persistent_state_bytes"]),
        "registration_time_ms": float(1000.0 * registration_seconds),
        "row_peak_rss_bytes": int(peak_rss_bytes),
        "row_peak_vram_bytes": int(peak_vram_bytes),
        "candidate_peak_memory_isolated": False,
        "closed_form_fit_count": len(candidate_states),
        "mac_equivalent_upper_bound": int(
            sum(
                int(state.resource.get("estimated_adaptation_macs", 0))
                for state in candidate_states
            )
        ),
        "query_head_mac": int(registered_count * feature_dim),
        "candidate_head_batch_query_latency_ms_per_row": float(
            1000.0 * query_seconds / max(1, query_count)
        ),
        "end_to_end_query_latency_available": False,
        "end_to_end_query_latency_ms": None,
        "batch1_head_resource": batch1_resource,
        "row_orchestration_time_ms": float(
            1000.0 * row_orchestration_seconds
        ),
        "auxiliary_state_cost_in_candidate_resource": False,
        "auxiliary_prediction_cost_in_candidate_latency": False,
    }


def execute_feature_row(
    *,
    ablation_id: str,
    row_id: str,
    receiver: str,
    candidate_lock_sha256: str,
    package_root_sha256: str,
    package_seal_sha256: str,
    input_identity: Mapping[str, Any],
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    scenario_payloads: Mapping[str, Mapping[str, Any]],
    deployment_prototypes_by_scenario: Mapping[str, Any],
    ground_basis: Any,
    ground_spectral_weights: Any,
    ground_audit: Mapping[str, Any],
    output_root: str | Path,
    seed: int,
    device: Any = "cpu",
    shared_view_count: int = 1,
    feature_cache_bytes: int = 0,
    deployment_state_bytes: int = 0,
    peak_rss_bytes: int = 0,
    peak_vram_bytes: int = 0,
) -> dict[str, Any]:
    """Execute one logical row without opening or accepting query truth."""

    spec = get_stage2_arm(ablation_id)
    required_input_identity = {
        "stage_scope",
        "k_shot",
        "new_class_count",
        "method_seed",
        "support_seed",
        "query_seed",
        "new_class_draw_seed",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "phase1_bundle_sha256",
        "phase1_prototype_sha256",
        "feature_cache_payload_sha256",
        "feature_cache_manifest_sha256",
    }
    if (
        not isinstance(input_identity, Mapping)
        or set(input_identity) != required_input_identity
        or input_identity.get("stage_scope") != spec.stage
    ):
        raise Stage2AblationRowExecutionError(
            "row input identity exact schema drift"
        )
    immutable_input_identity = dict(input_identity)
    row_orchestration_started = time.perf_counter()
    if set(scenario_payloads) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise Stage2AblationRowExecutionError(
            "scenario payloads must exactly cover the three formal scenarios"
        )
    if set(deployment_prototypes_by_scenario) != set(
        FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise Stage2AblationRowExecutionError(
            "deployment prototypes must exactly cover the three scenarios"
        )
    if shared_view_count not in {1, 3, 5}:
        raise Stage2AblationRowExecutionError(
            "shared_view_count must be 1, 3, or 5"
        )
    old_registry = tuple(str(value) for value in old_classes)
    new_registry = tuple(str(value) for value in new_classes)
    if (
        len(old_registry) != 6
        or len(set(old_registry)) != len(old_registry)
        or set(old_registry) & set(new_registry)
    ):
        raise Stage2AblationRowExecutionError("class registry drift")
    if spec.stage == "stage2c" and len(new_registry) not in {5, 10, 20}:
        raise Stage2AblationRowExecutionError(
            "Stage2-C new-class count drift"
        )
    if spec.stage != "stage2c" and new_registry:
        raise Stage2AblationRowExecutionError(
            "Stage2-A/B cannot register new classes"
        )

    runtime_device = torch.device(device)
    if runtime_device.type == "cuda":
        if not torch.cuda.is_available():
            raise Stage2AblationRowExecutionError(
                "CUDA device requested but unavailable"
            )
        _prepare_cuda_memory_audit(runtime_device)
    output = Path(output_root).absolute()
    if output.exists() and (
        not output.is_dir() or output.is_symlink() or any(output.iterdir())
    ):
        raise Stage2AblationRowExecutionError(
            "row output root must be absent or an empty directory"
        )
    output.mkdir(parents=True, exist_ok=True)

    all_tokens: list[np.ndarray] = []
    all_scenarios: list[np.ndarray] = []
    candidate_after: list[np.ndarray] = []
    candidate_before: list[np.ndarray] = []
    identity_after: list[np.ndarray] = []
    identity_before: list[np.ndarray] = []
    direct: list[np.ndarray] = []
    views: list[np.ndarray] = []
    quantization_reference_scores: list[np.ndarray] = []
    quantization_candidate_scores: list[np.ndarray] = []
    query_features: list[np.ndarray] = []
    candidate_states: list[Stage2AblationFittedState] = []
    candidate_registration_seconds = 0.0
    candidate_query_seconds = 0.0
    k_shot: int | None = None

    for scenario_index, scenario in enumerate(
        FORMAL_LEO_WEAK_SCENARIOS
    ):
        payload = scenario_payloads[scenario]
        query = np.asarray(
            _array(payload, "query_features"), dtype=np.float32
        )
        tokens = np.asarray(_array(payload, "query_tokens", rows=len(query))).astype(
            str
        )
        scenario_seed = int(seed) + scenario_index
        prototypes = deployment_prototypes_by_scenario[scenario]

        direct_started = time.perf_counter()
        direct_state = _fit(
            "P2-S2A",
            payload=payload,
            old_classes=old_registry,
            new_classes=(),
            deployment_prototypes=prototypes,
            ground_basis=ground_basis,
            ground_spectral_weights=ground_spectral_weights,
            ground_audit=ground_audit,
            seed=scenario_seed,
            device=device,
        )
        if spec.stage == "stage2a":
            candidate_registration_seconds += (
                time.perf_counter() - direct_started
            )
            before_state = after_state = identity_before_state = (
                identity_after_state
            ) = direct_state
            current_k = 0
        elif spec.stage == "stage2b":
            candidate_started = time.perf_counter()
            after_state = _fit(
                ablation_id,
                payload=payload,
                old_classes=old_registry,
                new_classes=(),
                deployment_prototypes=prototypes,
                ground_basis=ground_basis,
                ground_spectral_weights=ground_spectral_weights,
                ground_audit=ground_audit,
                seed=scenario_seed,
                device=device,
            )
            candidate_registration_seconds += (
                time.perf_counter() - candidate_started
            )
            identity_after_state = _fit(
                "P2-S2B-PROTO",
                payload=payload,
                old_classes=old_registry,
                new_classes=(),
                deployment_prototypes=prototypes,
                ground_basis=ground_basis,
                ground_spectral_weights=ground_spectral_weights,
                ground_audit=ground_audit,
                seed=scenario_seed,
                device=device,
            )
            before_state = identity_before_state = direct_state
            current_k = int(after_state.audit["old_k_shot"])
        else:
            before_state = _fit(
                "P2-S2B-FULL",
                payload=payload,
                old_classes=old_registry,
                new_classes=(),
                deployment_prototypes=prototypes,
                ground_basis=ground_basis,
                ground_spectral_weights=ground_spectral_weights,
                ground_audit=ground_audit,
                seed=scenario_seed,
                device=device,
            )
            identity_before_state = _fit(
                "P2-S2B-PROTO",
                payload=payload,
                old_classes=old_registry,
                new_classes=(),
                deployment_prototypes=prototypes,
                ground_basis=ground_basis,
                ground_spectral_weights=ground_spectral_weights,
                ground_audit=ground_audit,
                seed=scenario_seed,
                device=device,
            )
            candidate_started = time.perf_counter()
            after_state = _fit(
                ablation_id,
                payload=payload,
                old_classes=old_registry,
                new_classes=new_registry,
                deployment_prototypes=prototypes,
                ground_basis=ground_basis,
                ground_spectral_weights=ground_spectral_weights,
                ground_audit=ground_audit,
                seed=scenario_seed,
                device=device,
            )
            candidate_registration_seconds += (
                time.perf_counter() - candidate_started
            )
            identity_after_state = _fit(
                "P2-A0",
                payload=payload,
                old_classes=old_registry,
                new_classes=new_registry,
                deployment_prototypes=prototypes,
                ground_basis=ground_basis,
                ground_spectral_weights=ground_spectral_weights,
                ground_audit=ground_audit,
                seed=scenario_seed,
                device=device,
            )
            current_k = int(after_state.audit["old_k_shot"])
        if k_shot is None:
            k_shot = current_k
        elif k_shot != current_k:
            raise Stage2AblationRowExecutionError(
                "K-shot drift across scenarios"
            )

        started = time.perf_counter()
        candidate_score = after_state.score(query)
        candidate_prediction = np.asarray(after_state.classes)[
            np.argmax(candidate_score, axis=1)
        ]
        candidate_query_seconds += time.perf_counter() - started
        candidate_after.append(candidate_prediction)
        candidate_before.append(before_state.predict(query))
        identity_after.append(identity_after_state.predict(query))
        identity_before.append(identity_before_state.predict(query))
        direct.append(direct_state.predict(query))
        if after_state.compiled_affine_state is not None:
            audit = after_state.audit
            coefficient = np.asarray(
                audit.get("d81_actual_coefficient_fp32"),
                dtype=np.float32,
            )
            intercept = np.asarray(
                audit.get("d81_actual_intercept_fp32"),
                dtype=np.float32,
            )
            if (
                coefficient.shape
                != (len(after_state.classes), query.shape[1])
                or intercept.shape != (len(after_state.classes),)
            ):
                raise Stage2AblationRowExecutionError(
                    "same-fit FP32 quantization reference is missing"
                )
            prepared = after_state._prepared(query)
            quantization_reference_scores.append(
                prepared @ coefficient.T + intercept[None, :]
            )
            quantization_candidate_scores.append(candidate_score)
            if isinstance(audit, dict):
                audit.pop("d81_actual_coefficient_fp32", None)
                audit.pop("d81_actual_intercept_fp32", None)
        all_tokens.append(tokens)
        all_scenarios.append(np.asarray([scenario] * len(query)))
        views.append(
            np.full(len(query), shared_view_count, dtype=np.uint8)
        )
        query_features.append(query)
        candidate_states.append(after_state)

    if k_shot is None:
        raise Stage2AblationRowExecutionError("row produced no scenario state")
    token_sets = [set(values.tolist()) for values in all_tokens]
    if any(
        token_sets[left] & token_sets[right]
        for left in range(len(token_sets))
        for right in range(left + 1, len(token_sets))
    ):
        raise Stage2AblationRowExecutionError(
            "formal scenario query tokens must be pairwise disjoint"
        )

    prediction = publish_prediction_artifact(
        output / "predictions.cvspred",
        stage={
            "stage2a": "Stage2-A",
            "stage2b": "Stage2-B",
            "stage2c": "Stage2-C",
        }[spec.stage],
        row_id=row_id,
        receiver=receiver,
        k_shot=k_shot,
        candidate_lock_sha256=candidate_lock_sha256,
        package_root_sha256=package_root_sha256,
        package_seal_sha256=package_seal_sha256,
        query_tokens=np.concatenate(all_tokens),
        scenarios=np.concatenate(all_scenarios),
        candidate_after=np.concatenate(candidate_after),
        candidate_before=np.concatenate(candidate_before),
        identity_after=np.concatenate(identity_after),
        identity_before=np.concatenate(identity_before),
        direct=np.concatenate(direct),
        shared_view_counts=np.concatenate(views),
    )
    behavior = _behavior_receipt(candidate_states)
    if quantization_reference_scores:
        candidate_scores = np.concatenate(
            quantization_candidate_scores, axis=0
        )
        reference_scores = np.concatenate(
            quantization_reference_scores, axis=0
        )
        error = np.abs(candidate_scores - reference_scores)
        changed = np.argmax(candidate_scores, axis=1) != np.argmax(
            reference_scores, axis=1
        )
        flip_rate = float(np.mean(changed))
        quantization_receipt = {
            "schema": QUANTIZATION_RECEIPT_SCHEMA,
            "max_logit_abs_error": float(np.max(error)),
            "mean_logit_abs_error": float(np.mean(error)),
            "argmax_flip_rate": flip_rate,
            "prediction_agreement_rate": float(1.0 - flip_rate),
        }
    else:
        quantization_receipt = _quantization_receipt(
            candidate=candidate_states[-1],
            reference=None,
            query_features=query_features,
        )
    resource = _resource_receipt(
        candidate_states=candidate_states,
        feature_cache_bytes=feature_cache_bytes,
        deployment_state_bytes=deployment_state_bytes,
        peak_rss_bytes=(
            int(peak_rss_bytes)
            if int(peak_rss_bytes) > 0
            else _rss_bytes()
        ),
        peak_vram_bytes=(
            int(peak_vram_bytes)
            if int(peak_vram_bytes) > 0
            else int(torch.cuda.max_memory_allocated(runtime_device))
            if runtime_device.type == "cuda"
            else 0
        ),
        registration_seconds=candidate_registration_seconds,
        query_seconds=candidate_query_seconds,
        query_count=sum(len(value) for value in query_features),
        row_orchestration_seconds=(
            time.perf_counter() - row_orchestration_started
        ),
    )
    receipt = {
        "schema": ROW_EXECUTION_SCHEMA,
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "ablation_id": ablation_id,
        "stage": spec.stage,
        "row_id": row_id,
        "receiver": receiver,
        "k_shot": k_shot,
        "input_identity": immutable_input_identity,
        "scenario_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "query_rows_by_scenario": {
            scenario: len(tokens)
            for scenario, tokens in zip(
                FORMAL_LEO_WEAK_SCENARIOS, all_tokens
            )
        },
        "fit_query_rows_used": 0,
        "query_truth_opened": False,
        "prediction": prediction,
        "behavior": behavior,
        "quantization": quantization_receipt,
        "resource": resource,
    }
    _exclusive_json(output / "row_execution_receipt.json", receipt)
    return receipt


__all__ = [
    "ROW_EXECUTION_SCHEMA",
    "Stage2AblationRowExecutionError",
    "execute_feature_row",
]
