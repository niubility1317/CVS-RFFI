"""Run D13 on sealed strict-K10 enrollment packages without opening query."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi.somph_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    load_verified_somph_predictor_bundle,
)
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_joint_residual_logit_head import (  # noqa: E402
    _build_runtime_authorized_feature_artifact_internal,
)
from cvsrffi.stage2_new_logit_intrusion_guard import (  # noqa: E402
    IntrusionGuardHyperparameters,
    NewLogitIntrusionGuardState,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    predict_all_registered,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)


MAX_SERIALIZED_STATE_BYTES = 256 * 1024
D11_V6_COMMIT_SHA256 = (
    "d9f0a0afc15d5d8e554cae01e6c0a5663ecfbcca7a303136c5d9d56fe3ec58f2"
)
D11_V6_SUPPORT_AUDIT_SHA256 = (
    "75bf5a61c86007e018d5088fbbab1a3eea9dfecc09bb310cc816a7ad1e77f1e4"
)
D11_V6_REPORT_SHA256 = (
    "eb58d6d91dd5bc3aaed9940584a799d2e3f3367fe86518ad5347f2761b48d740"
)


class D13RunnerError(ValueError):
    """Raised when the support-only D13 protocol fails closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def _write_json_new(path: Path, value: Any) -> str:
    raw = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(path)
    return hashlib.sha256(raw).hexdigest()


def _write_jsonl_new(path: Path, rows: list[Mapping[str, Any]]) -> str:
    raw = b"".join(_canonical(dict(row)) + b"\n" for row in rows)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(path)
    return hashlib.sha256(raw).hexdigest()


def _write_text_new(path: Path, text: str) -> str:
    raw = text.encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(path)
    return hashlib.sha256(raw).hexdigest()


def _member(manifest: Mapping[str, Any], kind: str) -> dict[str, Any]:
    rows = [dict(row) for row in manifest["members"] if row.get("kind") == kind]
    if len(rows) != 1:
        raise D13RunnerError(f"enrollment member drift: {kind}")
    return rows[0]


def _validate_manifest(
    manifest: Mapping[str, Any], *, registration_state: str
) -> None:
    required_false = (
        "clean_sample_access",
        "clean_derived_signal_access",
        "phase2_clean_dataset_reachable",
        "phase2_clean_cache_reachable",
        "phase2_clean_control_flow_reachable",
        "phase2_source_sample_access",
        "phase2_source_derived_signal_access",
        "phase2_source_cache_access",
        "phase2_source_label_access",
        "phase2_source_replay",
        "phase2_additional_leo_channel_state_generation",
        "phase2_post_reception_view_counts_as_additional_physical_sample",
        "phase2_query_post_reception_view_fit_access",
        "phase2_query_role_oracle_access",
        "phase2_query_true_batch_class_count_access",
        "phase2_query_class_quota_access",
        "phase2_query_batch_global_assignment",
    )
    if (
        manifest.get("profile") != "enrollment_only"
        or manifest.get("registration_state") != registration_state
        or int(manifest.get("k_shot", -1)) != 10
        or manifest.get("phase2_sample_view_policy")
        != "leo_weak_only_no_clean_access"
        or manifest.get("phase2_physical_sample_observation_policy")
        != "single_leo_weak_observation_per_physical_sample"
        or manifest.get("phase2_pretrained_artifact_policy")
        != "sealed_phase1_checkpoint_only"
        or manifest.get("phase2_query_decision_policy")
        != "per_sample_all_registered_classes"
        or any(bool(manifest.get(field, True)) for field in required_false)
        or tuple(manifest.get("target_channel_scenarios", ()))
        != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise D13RunnerError("enrollment package protocol drift")
    kinds = {str(row.get("kind")) for row in manifest.get("members", ())}
    expected = {
        "feature_runtime",
        "method_lock",
        "overlay_provenance",
        *{f"support:{scenario}" for scenario in FORMAL_LEO_WEAK_SCENARIOS},
    }
    if kinds != expected or any(
        any(token in kind.lower() for token in ("query", "truth", "score", "scorer"))
        for kind in kinds
    ):
        raise D13RunnerError("enrollment member allowlist drift")


def _load_enrollment(
    root: Path, seal: Path, *, registration_state: str
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    if root.name != "enrollment_only" or "enrollment" not in seal.name:
        raise D13RunnerError("D13 accepts enrollment-only paths")
    payloads, manifest, audit = load_verified_somph_predictor_bundle(
        root,
        detached_seal_path=seal,
        expected_seal_sha256=_sha256_file(seal),
    )
    _validate_manifest(manifest, registration_state=registration_state)
    return payloads, manifest, audit


def _payload_rows(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    scenario: str,
) -> dict[str, np.ndarray]:
    handles = np.asarray(
        [str(row["class_handle"]) for row in manifest["registered_classes"]]
    )
    indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
    labels = handles[indices].astype(str)
    ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    order = np.asarray(
        sorted(range(len(labels)), key=lambda index: (labels[index], int(ranks[index]))),
        dtype=np.int64,
    )
    rows = {
        "iq": np.asarray(payload["support_leo_weak_iq"], dtype=np.float32)[order],
        "labels": labels[order],
        "ranks": ranks[order],
        "tokens": np.asarray(payload["support_tokens"]).astype(str)[order],
        "hashes": np.asarray(payload["support_post_channel_iq_sha256"]).astype(str)[
            order
        ],
    }
    classes, counts = np.unique(rows["labels"], return_counts=True)
    computed = np.asarray(
        [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in rows["iq"]
        ]
    )
    if (
        rows["iq"].ndim != 3
        or rows["iq"].shape[1] != 2
        or not np.isfinite(rows["iq"]).all()
        or set(counts.tolist()) != {10}
        or any(
            set(rows["ranks"][rows["labels"] == label].tolist()) != set(range(10))
            for label in classes
        )
        or not np.array_equal(computed, rows["hashes"])
        or len(set(rows["tokens"].tolist())) != len(rows["tokens"])
        or len(set(rows["hashes"].tolist())) != len(rows["hashes"])
    ):
        raise D13RunnerError(f"strict K10 payload drift: {scenario}")
    return rows


def _base_feature(
    model: torch.nn.Module, device: torch.device, iq: np.ndarray
) -> np.ndarray:
    if len(iq) != 1:
        raise D13RunnerError("D13 extraction requires physical batch=1")
    zid = forward_zid160(model, iq, device=device, batch_size=1)
    return registered_feature(iq, zid)


def _build_feature_artifact(
    model: torch.nn.Module,
    device: torch.device,
    rows: Mapping[str, np.ndarray],
    *,
    runtime_sha256: str,
    feature_code_sha256: str,
    checkpoint_sha256: str,
    reuse_by_token: Mapping[str, np.ndarray] | None = None,
):
    cursor = 0
    reuse = {} if reuse_by_token is None else dict(reuse_by_token)

    def extract_single(iq: np.ndarray) -> np.ndarray:
        nonlocal cursor
        token = str(rows["tokens"][cursor])
        cursor += 1
        if token in reuse:
            return np.asarray(reuse[token], dtype=np.float32)[None, :]
        return _base_feature(model, device, iq)

    return _build_runtime_authorized_feature_artifact_internal(
        rows["iq"],
        physical_sample_ids=rows["tokens"].tolist(),
        parent_received_iq_sha256=rows["hashes"].tolist(),
        sealed_runtime_sha256=runtime_sha256,
        feature_code_sha256=feature_code_sha256,
        sealed_phase1_checkpoint_sha256=checkpoint_sha256,
        extract_single_received_iq=extract_single,
        operator_id="base",
        view_seed=0,
    )


def _feature_provenance(
    rows: Mapping[str, np.ndarray], artifact
) -> list[dict[str, Any]]:
    return [
        {
            "physical_sample_id": str(rows["tokens"][index]),
            "parent_received_iq_sha256": str(rows["hashes"][index]),
            "operator_id": artifact.operator_id,
            "view_seed": artifact.view_seed,
            "feature_sha256": artifact.per_row_feature_sha256[index],
            "sealed_runtime_sha256": artifact.sealed_runtime_sha256,
            "sealed_phase1_checkpoint_sha256": (
                artifact.sealed_phase1_checkpoint_sha256
            ),
            "d13_feature_code_sha256": artifact.feature_code_sha256,
        }
        for index in range(len(rows["tokens"]))
    ]


def _state_metadata(state: NewLogitIntrusionGuardState) -> dict[str, Any]:
    hp = state.hyperparameters
    return {
        "schema": state.schema,
        "candidate_id": state.candidate_id,
        "classes": list(state.classes),
        "feature_dim": state.feature_dim,
        "k_shot": state.k_shot,
        "old_class_count": state.old_class_count,
        "registration_generation": state.registration_generation,
        "hyperparameters": {
            "mode": hp.mode,
            "old_risk_quantile": hp.old_risk_quantile,
            "new_room_quantile": hp.new_room_quantile,
            "safety": hp.safety,
            "cap": hp.cap,
            "new_floor_margin": hp.new_floor_margin,
            "hinge_strength": hp.hinge_strength,
            "force_zero": hp.force_zero,
        },
        "resource": dict(state.resource),
        "support_feature_artifact_sha256": state.support_feature_artifact_sha256,
        "support_selection_sha256": state.support_selection_sha256,
        "sealed_runtime_sha256": state.sealed_runtime_sha256,
        "feature_code_sha256": state.feature_code_sha256,
        "sealed_phase1_checkpoint_sha256": state.sealed_phase1_checkpoint_sha256,
        "operator_id": state.operator_id,
        "view_seed": state.view_seed,
        "calibration_diagnostics": [
            dict(value) for value in state.calibration_diagnostics
        ],
        "state_content_sha256": state.state_content_sha256,
    }


def _write_state(
    output: Path, *, stem: str, state: NewLogitIntrusionGuardState
) -> dict[str, Any]:
    npz = output / f"{stem}.npz"
    with npz.open("xb") as handle:
        np.savez(
            handle,
            prototypes=state.prototypes,
            new_logit_penalties=state.new_logit_penalties,
            hinge_thresholds=state.hinge_thresholds,
            hinge_strengths=state.hinge_strengths,
        )
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(npz)
    metadata = output / f"{stem}.json"
    metadata_sha256 = _write_json_new(metadata, _state_metadata(state))
    with np.load(npz, allow_pickle=False) as loaded:
        if any(
            not np.array_equal(loaded[name], getattr(state, name))
            for name in (
                "prototypes",
                "new_logit_penalties",
                "hinge_thresholds",
                "hinge_strengths",
            )
        ):
            raise D13RunnerError("sealed state NPZ readback mismatch")
    loaded_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    if loaded_metadata.get("state_content_sha256") != state.state_content_sha256:
        raise D13RunnerError("sealed state metadata readback mismatch")
    serialized_bytes = npz.stat().st_size + metadata.stat().st_size
    if serialized_bytes > MAX_SERIALIZED_STATE_BYTES:
        raise D13RunnerError("serialized state exceeds 256KiB")
    return {
        "npz_sha256": _sha256_file(npz),
        "metadata_sha256": metadata_sha256,
        "npz_file_bytes": npz.stat().st_size,
        "metadata_file_bytes": metadata.stat().st_size,
        "serialized_state_total_bytes": serialized_bytes,
        "serialized_state_under_256kib": True,
        "content_verified_after_write": True,
    }


def _measure_support_row_pareto(
    state: NewLogitIntrusionGuardState,
    probe_artifact,
    qknn_support_features: np.ndarray,
    *,
    repeats: int = 500,
) -> dict[str, Any]:
    support = np.array(qknn_support_features, dtype=np.float32, copy=True)
    support /= np.maximum(np.linalg.norm(support, axis=1, keepdims=True), 1.0e-8)
    query = np.array(probe_artifact.features, dtype=np.float32, copy=True)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1.0e-8)
    for _ in range(20):
        predict_all_registered(state, probe_artifact)
        _ = query @ support.T
    start = time.perf_counter()
    for _ in range(repeats):
        predict_all_registered(state, probe_artifact)
    d13_ms = (time.perf_counter() - start) * 1000.0 / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        _ = query @ support.T
    qknn_ms = (time.perf_counter() - start) * 1000.0 / repeats
    d13_macs = int(state.resource["prototype_cosine_mac_per_sample"])
    qknn_macs = int(len(support) * state.feature_dim)
    return {
        "benchmark_input": "one_support_row_resource_probe_no_query_open",
        "repeats": repeats,
        "d13_head_latency_ms": d13_ms,
        "identity_single_qknn_latency_ms": qknn_ms,
        "latency_delta_percent": 100.0 * (d13_ms / qknn_ms - 1.0),
        "d13_exact_macs": d13_macs,
        "identity_single_qknn_exact_macs": qknn_macs,
        "mac_delta_percent": 100.0 * (d13_macs / qknn_macs - 1.0),
        "d13_array_state_bytes": int(state.resource["persistent_state_bytes"]),
        "d13_incremental_guard_state_bytes": int(
            state.resource["incremental_guard_state_bytes"]
        ),
        "identity_single_qknn_state_bytes": int(support.nbytes),
        "state_delta_percent": 100.0
        * (int(state.resource["persistent_state_bytes"]) / support.nbytes - 1.0),
        "guard_subtractions_per_sample": int(
            state.resource["guard_subtractions_per_sample"]
        ),
        "guard_relu_per_sample": int(state.resource["guard_relu_per_sample"]),
    }


def _candidate_lock(candidates: tuple[IntrusionGuardHyperparameters, ...]) -> dict[str, Any]:
    rows = [
        {
            "candidate_id": value.candidate_id,
            "mode": value.mode,
            "old_risk_quantile": value.old_risk_quantile,
            "new_room_quantile": value.new_room_quantile,
            "safety": value.safety,
            "cap": value.cap,
            "new_floor_margin": value.new_floor_margin,
            "hinge_strength": value.hinge_strength,
            "force_zero": value.force_zero,
            "quantile_method": "linear",
        }
        for value in candidates
    ]
    return {
        "selection_scope": "one_hyperparameter_arm_shared_by_all_three_scenarios",
        "candidates": rows,
        "lock_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }


def _load_d11_v6_reference(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    commit_path = root / "COMMIT.json"
    audit_path = root / "support_audit.json"
    report_path = root / "report.md"
    actual = {
        "commit_sha256": _sha256_file(commit_path),
        "support_audit_sha256": _sha256_file(audit_path),
        "report_sha256": _sha256_file(report_path),
    }
    expected = {
        "commit_sha256": D11_V6_COMMIT_SHA256,
        "support_audit_sha256": D11_V6_SUPPORT_AUDIT_SHA256,
        "report_sha256": D11_V6_REPORT_SHA256,
    }
    if actual != expected:
        raise D13RunnerError("D11-v6 immutable reference hash drift")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        commit.get("status") != "SUPPORT_ONLY_D11_NOT_SELECTED_NO_QUERY_OPEN"
        or commit.get("support_audit_sha256") != actual["support_audit_sha256"]
        or commit.get("report_sha256") != actual["report_sha256"]
        or bool(commit.get("query_opened", True))
    ):
        raise D13RunnerError("D11-v6 reference COMMIT content drift")
    reference = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        new = audit["scenario_results"][scenario][
            "joint_registration_leave_two_out"
        ]["after_new"]
        reference[scenario] = {
            "overall": float(new["overall_accuracy"]),
            "floor": float(new["min_class_accuracy"]),
            "per_class_accuracy": {
                str(label): float(value)
                for label, value in new["per_class_accuracy"].items()
            },
        }
    provenance = {
        "artifact_root": str(root.resolve()),
        **actual,
        "reference_status": commit["status"],
        "query_opened": False,
        "values_read_from_support_audit": True,
    }
    return reference, provenance


def _candidates() -> tuple[IntrusionGuardHyperparameters, ...]:
    return (
        IntrusionGuardHyperparameters(
            candidate_id="d13_delta0_base",
            mode="constant",
            safety=0.0,
            cap=0.0,
            new_floor_margin=0.0,
            force_zero=True,
        ),
        IntrusionGuardHyperparameters(
            candidate_id="d13_const_q80_r10_s0_c10",
            mode="constant",
            old_risk_quantile=0.80,
            new_room_quantile=0.10,
            safety=0.0,
            cap=0.10,
            new_floor_margin=0.01,
        ),
        IntrusionGuardHyperparameters(
            candidate_id="d13_const_q90_r10_s1_c20",
            mode="constant",
            old_risk_quantile=0.90,
            new_room_quantile=0.10,
            safety=0.01,
            cap=0.20,
            new_floor_margin=0.01,
        ),
        IntrusionGuardHyperparameters(
            candidate_id="d13_const_q100_r10_s1_c40",
            mode="constant",
            old_risk_quantile=1.0,
            new_room_quantile=0.10,
            safety=0.01,
            cap=0.40,
            new_floor_margin=0.01,
        ),
        IntrusionGuardHyperparameters(
            candidate_id="d13_const_q90_r25_s0_c20",
            mode="constant",
            old_risk_quantile=0.90,
            new_room_quantile=0.25,
            safety=0.0,
            cap=0.20,
            new_floor_margin=0.01,
        ),
        IntrusionGuardHyperparameters(
            candidate_id="d13_const_q100_r25_s0_c40",
            mode="constant",
            old_risk_quantile=1.0,
            new_room_quantile=0.25,
            safety=0.0,
            cap=0.40,
            new_floor_margin=0.01,
        ),
    )


def run(
    *,
    before_root: Path,
    before_seal: Path,
    after_root: Path,
    after_seal: Path,
    d11_reference_root: Path,
    output: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    if output.exists():
        raise D13RunnerError("output already exists")
    before_payloads, before_manifest, before_preopen = _load_enrollment(
        before_root, before_seal, registration_state="before"
    )
    after_payloads, after_manifest, after_preopen = _load_enrollment(
        after_root, after_seal, registration_state="after"
    )
    if (
        before_manifest["receiver"] != after_manifest["receiver"]
        or int(before_manifest["seed"]) != int(after_manifest["seed"])
        or before_manifest["feature_runtime_sha256"]
        != after_manifest["feature_runtime_sha256"]
        or before_manifest["phase1_checkpoint_sha256"]
        != after_manifest["phase1_checkpoint_sha256"]
    ):
        raise D13RunnerError("before/after package binding drift")
    before_handles = {
        str(row["class_handle"]) for row in before_manifest["registered_classes"]
    }
    after_handles = {
        str(row["class_handle"]) for row in after_manifest["registered_classes"]
    }
    if not before_handles < after_handles:
        raise D13RunnerError("new-class registration set drift")
    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root, _member(before_manifest, "feature_runtime"), device=device
    )
    output.mkdir(parents=True)
    module_path = CODE / "cvsrffi" / "stage2_new_logit_intrusion_guard.py"
    runner_path = Path(__file__).resolve()
    artifact_provider_path = (
        CODE / "cvsrffi" / "stage2_joint_residual_logit_head.py"
    )
    registered_feature_path = (
        CODE / "cvsrffi" / "stage2_diag_cosine_exploration.py"
    )
    module_sha256 = _sha256_file(module_path)
    runner_sha256 = _sha256_file(runner_path)
    artifact_provider_sha256 = _sha256_file(artifact_provider_path)
    registered_feature_sha256 = _sha256_file(registered_feature_path)
    feature_code_sha256 = hashlib.sha256(
        _canonical(
            {
                "d13_module_sha256": module_sha256,
                "d13_runner_sha256": runner_sha256,
                "d12_artifact_provider_sha256": artifact_provider_sha256,
                "registered_feature_module_sha256": registered_feature_sha256,
            }
        )
    ).hexdigest()
    runtime_sha256 = str(before_manifest["feature_runtime_sha256"])
    checkpoint_sha256 = str(before_manifest["phase1_checkpoint_sha256"])
    candidates = _candidates()
    hyperparameter_lock = _candidate_lock(candidates)
    tracemalloc.start()
    run_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    contexts: dict[str, dict[str, Any]] = {}
    seen_tokens: set[str] = set()
    seen_hashes: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = _payload_rows(
            before_payloads[scenario], before_manifest, scenario=scenario
        )
        after_rows = _payload_rows(
            after_payloads[scenario], after_manifest, scenario=scenario
        )
        current_tokens = set(after_rows["tokens"].tolist())
        current_hashes = set(after_rows["hashes"].tolist())
        if current_tokens & seen_tokens or current_hashes & seen_hashes:
            raise D13RunnerError("physical support reused across LEO scenarios")
        seen_tokens.update(current_tokens)
        seen_hashes.update(current_hashes)
        feature_start = time.perf_counter()
        before_artifact = _build_feature_artifact(
            model,
            device,
            before_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            checkpoint_sha256=checkpoint_sha256,
        )
        old_feature_by_token = {
            token: before_artifact.features[index]
            for index, token in enumerate(before_artifact.physical_sample_ids)
        }
        after_artifact = _build_feature_artifact(
            model,
            device,
            after_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            checkpoint_sha256=checkpoint_sha256,
            reuse_by_token=old_feature_by_token,
        )
        contexts[scenario] = {
            "before_rows": before_rows,
            "after_rows": after_rows,
            "before_artifact": before_artifact,
            "after_artifact": after_artifact,
            "feature_seconds": time.perf_counter() - feature_start,
            "before_provenance": _feature_provenance(before_rows, before_artifact),
            "after_provenance": _feature_provenance(after_rows, after_artifact),
        }
    d11_reference, d11_reference_provenance = _load_d11_v6_reference(
        d11_reference_root
    )
    trace: list[dict[str, Any]] = []
    candidate_rows = []
    evaluations: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        evaluations[candidate.candidate_id] = {}
        scenario_gates = {}
        floors = []
        h_values = []
        joint_values = []
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            context = contexts[scenario]
            joint, joint_trace = evaluate_joint_leave_two_out(
                context["before_artifact"],
                context["before_rows"]["labels"],
                context["before_rows"]["ranks"],
                context["after_artifact"],
                context["after_rows"]["labels"],
                context["after_rows"]["ranks"],
                hyperparameters=candidate,
            )
            evaluations[candidate.candidate_id][scenario] = joint
            trace.extend(
                {
                    "scenario": scenario,
                    "hyperparameter_lock_sha256": hyperparameter_lock[
                        "lock_sha256"
                    ],
                    **row,
                }
                for row in joint_trace
            )
            reference = d11_reference[scenario]
            per_new_non_degrade = bool(
                joint["new_per_class_non_degraded_vs_base_cosine"]
                and joint["new_no_class_collapsed_to_zero"]
            )
            gate = (
                not candidate.force_zero
                and bool(joint["all_new_class_calibration_feasible"])
                and bool(joint["old_score_columns_bitwise_unchanged"])
                and bool(joint["old_per_class_non_degraded_vs_before"])
                and bool(joint["old_per_class_non_degraded_vs_base_cosine"])
                and float(joint["old_forgetting"]) <= 1.0e-12
                and per_new_non_degrade
                and float(joint["after_new"]["overall_accuracy"]) + 1.0e-12
                >= float(joint["base_after_new"]["overall_accuracy"])
                and float(joint["after_new"]["min_class_accuracy"]) + 1.0e-12
                >= float(joint["base_after_new"]["min_class_accuracy"])
                and float(joint["after_new"]["overall_accuracy"]) + 1.0e-12
                >= reference["overall"]
                and float(joint["after_new"]["min_class_accuracy"]) + 1.0e-12
                >= reference["floor"]
                and float(joint["h_old_new"]) + 1.0e-12
                >= float(joint["base_h_old_new"])
                and float(joint["joint_accuracy"]) + 1.0e-12
                >= float(joint["base_joint_accuracy"])
            )
            scenario_gates[scenario] = bool(gate)
            floors.append(float(joint["after_new"]["min_class_accuracy"]))
            h_values.append(float(joint["h_old_new"]))
            joint_values.append(float(joint["joint_accuracy"]))
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "mode": candidate.mode,
                "force_zero": candidate.force_zero,
                "all_scenario_gate_pass": all(scenario_gates.values()),
                "scenario_gate_pass": scenario_gates,
                "worst_scenario_new_floor": min(floors),
                "mean_h_old_new": float(np.mean(h_values)),
                "mean_joint_accuracy": float(np.mean(joint_values)),
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            row["all_scenario_gate_pass"],
            row["worst_scenario_new_floor"],
            row["mean_h_old_new"],
            row["mean_joint_accuracy"],
            row["candidate_id"],
        ),
        reverse=True,
    )
    passing = [
        row
        for row in candidate_rows
        if row["all_scenario_gate_pass"] and not row["force_zero"]
    ]
    selected_id = (
        str(passing[0]["candidate_id"]) if passing else "d13_delta0_base"
    )
    selected_hp = next(
        value for value in candidates if value.candidate_id == selected_id
    )
    selected_row = next(
        row for row in candidate_rows if row["candidate_id"] == selected_id
    )
    promotion_ready = bool(passing and not selected_hp.force_zero)
    scenario_results = {}
    state_hashes = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        context = contexts[scenario]
        fit_start = time.perf_counter()
        fitted = fit_before_after_locked(
            context["before_artifact"],
            context["before_rows"]["labels"],
            context["before_rows"]["ranks"],
            context["after_artifact"],
            context["after_rows"]["labels"],
            context["after_rows"]["ranks"],
            k_shot=10,
            hyperparameters=selected_hp,
        )
        trace.extend(
            {
                "scenario": scenario,
                "hyperparameter_lock_sha256": hyperparameter_lock["lock_sha256"],
                **row,
            }
            for row in fitted.trace
        )
        probe_rows = {
            key: np.asarray(value)[:1]
            for key, value in context["after_rows"].items()
        }
        probe_feature = context["after_artifact"].features[:1]

        def extract_probe(_: np.ndarray) -> np.ndarray:
            return probe_feature

        probe_artifact = _build_runtime_authorized_feature_artifact_internal(
            probe_rows["iq"],
            physical_sample_ids=probe_rows["tokens"].tolist(),
            parent_received_iq_sha256=probe_rows["hashes"].tolist(),
            sealed_runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            sealed_phase1_checkpoint_sha256=checkpoint_sha256,
            extract_single_received_iq=extract_probe,
            operator_id="base",
            view_seed=0,
        )
        joint = evaluations[selected_id][scenario]
        scenario_results[scenario] = {
            "joint_leave_two_out": joint,
            "support_gate_pass": selected_row["scenario_gate_pass"][scenario],
            "resource_before": dict(fitted.before_state.resource),
            "resource_after": dict(fitted.after_state.resource),
            "pareto_vs_identity_single_qknn": _measure_support_row_pareto(
                fitted.after_state,
                probe_artifact,
                context["after_artifact"].features,
            ),
            "before_feature_provenance": context["before_provenance"],
            "after_feature_provenance": context["after_provenance"],
            "old_lineage_feature_prototype_score_exact_freeze_locked": True,
            "measured": {
                "feature_extraction_seconds": context["feature_seconds"],
                "selected_full_fit_seconds": time.perf_counter() - fit_start,
            },
        }
        state_hashes[f"{scenario}:before:k10"] = _write_state(
            output,
            stem=f"state_{scenario}_before_k10",
            state=fitted.before_state,
        )
        state_hashes[f"{scenario}:after:k10"] = _write_state(
            output,
            stem=f"state_{scenario}_after_k10",
            state=fitted.after_state,
        )
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    status = (
        "SUPPORT_ONLY_D13_SELECTED_NO_QUERY_OPEN"
        if promotion_ready
        else "SUPPORT_ONLY_D13_NOT_SELECTED_NO_QUERY_OPEN"
    )
    training_log_sha256 = _write_jsonl_new(output / "training_log.jsonl", trace)
    audit = {
        "schema": "cvs.phase2.d13_support_only_audit.v1",
        "status": status,
        "claim_scope": "development_support_only_no_query_performance_claim",
        "receiver": after_manifest["receiver"],
        "seed": int(after_manifest["seed"]),
        "k_shot": 10,
        "view_policy": "one_fixed_received_iq_base_view_no_new_channel_state",
        "unified_hyperparameter_selection": {
            "selected_candidate_id": selected_id,
            "same_candidate_all_scenarios": True,
            "hyperparameter_lock_sha256": hyperparameter_lock["lock_sha256"],
            "candidate_rows": candidate_rows,
            "delta0_fallback_policy": (
                "if_all_positive_guards_fail_save_true_zero_guard_and_no_go"
            ),
            "delta0_fallback_is_d13_improvement": False,
        },
        "hyperparameter_lock": hyperparameter_lock,
        "scenario_results": scenario_results,
        "d11_v6_joint_new_reference": d11_reference,
        "d11_v6_reference_provenance": d11_reference_provenance,
        "promotion_ready_for_single_query_candidate": promotion_ready,
        "query_package_opened": False,
        "query_truth_opened": False,
        "query_prediction_opened": False,
        "query_score_opened": False,
        "scorer_opened": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "source_derived_signal_access": False,
        "additional_leo_channel_state_generation": False,
        "post_reception_views_count_as_additional_k": False,
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "runtime_authorization": {
            "feature_extraction_mode": (
                "runner_internal_actual_iq_sha_physical_batch1_no_public_mapping"
            ),
            "sealed_runtime_sha256": runtime_sha256,
            "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
            "d13_module_sha256": module_sha256,
            "d13_runner_sha256": runner_sha256,
            "d12_artifact_provider_sha256": artifact_provider_sha256,
            "registered_feature_module_sha256": registered_feature_sha256,
            "combined_feature_code_sha256": feature_code_sha256,
        },
        "state_sha256": state_hashes,
        "training_log_sha256": training_log_sha256,
        "measured_run_resource": {
            "wall_seconds": time.perf_counter() - run_start,
            "peak_python_tracemalloc_bytes": int(peak_python),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            ),
            "device": str(device),
        },
        "before_package_root_sha256": before_manifest["package_root_sha256"],
        "after_package_root_sha256": after_manifest["package_root_sha256"],
        "before_seal_sha256": _sha256_file(before_seal),
        "after_seal_sha256": _sha256_file(after_seal),
        "preopen_audit": {"before": before_preopen, "after": after_preopen},
    }
    audit_sha256 = _write_json_new(output / "support_audit.json", audit)
    lines = [
        "# D13类条件new-logit侵入保护支持集审计",
        "",
        f"状态：`{status}`。只打开严格K10 before/after enrollment-only包，未打开query、truth、prediction、score或scorer。",
        "",
        f"三场景统一选择`{selected_id}`，hyperparameter lock SHA为`{hyperparameter_lock['lock_sha256']}`。全部正guard失败时只保存真实delta0并维持NO-GO。",
        "",
        "|场景|alpha0 old/new/H|D13 old/floor|D13 new/floor|joint/H|forgetting|old逐类>=before/base|new逐类>=alpha0且无0类|calibration feasible|guard bytes|总数组state|门|",
        "|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        row = scenario_results[scenario]
        joint = row["joint_leave_two_out"]
        resource = row["resource_after"]
        lines.append(
            f"|`{scenario}`|"
            f"{joint['base_after_old']['overall_accuracy']:.4f}/"
            f"{joint['base_after_new']['overall_accuracy']:.4f}/"
            f"{joint['base_h_old_new']:.4f}|"
            f"{joint['after_old']['overall_accuracy']:.4f}/"
            f"{joint['after_old']['min_class_accuracy']:.4f}|"
            f"{joint['after_new']['overall_accuracy']:.4f}/"
            f"{joint['after_new']['min_class_accuracy']:.4f}|"
            f"{joint['joint_accuracy']:.4f}/{joint['h_old_new']:.4f}|"
            f"{joint['old_forgetting']:.4f}|"
            f"{joint['old_per_class_non_degraded_vs_before']}/"
            f"{joint['old_per_class_non_degraded_vs_base_cosine']}|"
            f"{joint['new_per_class_non_degraded_vs_base_cosine']}/"
            f"{joint['new_no_class_collapsed_to_zero']}|"
            f"{joint['all_new_class_calibration_feasible']}|"
            f"{resource['incremental_guard_state_bytes']}|"
            f"{resource['persistent_state_bytes']}|"
            f"{row['support_gate_pass']}|"
        )
    lines.extend(
        [
            "",
            "D13为0参数、0epoch闭式support-only guard。K1/K5未从K10切片，必须等待各自独立strict sealed package。本轮不开放query。",
            "",
        ]
    )
    report_sha256 = _write_text_new(output / "report.md", "\n".join(lines))
    commit = {
        "schema": "cvs.phase2.d13_support_only_commit.v1",
        "status": status,
        "support_audit_sha256": audit_sha256,
        "training_log_sha256": training_log_sha256,
        "report_sha256": report_sha256,
        "state_sha256": state_hashes,
        "promotion_ready_for_single_query_candidate": promotion_ready,
        "query_opened": False,
        "selected_candidate_id": selected_id,
        "hyperparameter_lock_sha256": hyperparameter_lock["lock_sha256"],
        "d11_v6_reference_provenance": d11_reference_provenance,
        "sealed_runtime_sha256": runtime_sha256,
        "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
        "d13_module_sha256": module_sha256,
        "d13_runner_sha256": runner_sha256,
        "d12_artifact_provider_sha256": artifact_provider_sha256,
        "registered_feature_module_sha256": registered_feature_sha256,
        "combined_feature_code_sha256": feature_code_sha256,
    }
    commit_sha256 = _write_json_new(output / "COMMIT.json", commit)
    return {"commit_sha256": commit_sha256, **commit}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--d11-reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                before_root=args.before_root,
                before_seal=args.before_seal,
                after_root=args.after_root,
                after_seal=args.after_seal,
                d11_reference_root=args.d11_reference_root,
                output=args.output,
                device_name=args.device,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
