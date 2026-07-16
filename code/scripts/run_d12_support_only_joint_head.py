"""Run D12 on sealed strict-K10 enrollment packages without opening query."""

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
    JointResidualLogitHeadState,
    ResidualHeadHyperparameters,
    _build_runtime_authorized_feature_artifact_internal,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    predict_all_registered,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)


class D12RunnerError(ValueError):
    """Raised when the support-only D12 protocol fails closed."""


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
        raise D12RunnerError(f"enrollment member drift: {kind}")
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
        raise D12RunnerError("enrollment package protocol drift")
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
        raise D12RunnerError("enrollment member allowlist drift")


def _load_enrollment(
    root: Path, seal: Path, *, registration_state: str
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    if root.name != "enrollment_only" or "enrollment" not in seal.name:
        raise D12RunnerError("D12 accepts enrollment-only paths")
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
        raise D12RunnerError(f"strict K10 payload drift: {scenario}")
    return rows


def _base_feature(
    model: torch.nn.Module, device: torch.device, iq: np.ndarray
) -> np.ndarray:
    if len(iq) != 1:
        raise D12RunnerError("D12 extraction requires physical batch=1")
    zid = forward_zid160(model, iq, device=device, batch_size=1)
    return registered_feature(iq, zid)


def _build_feature_artifact(
    model: torch.nn.Module,
    device: torch.device,
    rows: Mapping[str, np.ndarray],
    *,
    runtime_sha256: str,
    feature_code_sha256: str,
    phase1_checkpoint_sha256: str,
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
        sealed_phase1_checkpoint_sha256=phase1_checkpoint_sha256,
        extract_single_received_iq=extract_single,
        operator_id="base",
        view_seed=0,
    )


def _feature_provenance(
    rows: Mapping[str, np.ndarray],
    artifact,
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
            "d12_feature_code_sha256": artifact.feature_code_sha256,
        }
        for index in range(len(rows["tokens"]))
    ]


def _state_metadata(state: JointResidualLogitHeadState) -> dict[str, Any]:
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
            "rank": hp.rank,
            "epochs": hp.epochs,
            "learning_rate": hp.learning_rate,
            "alpha": hp.alpha,
            "temperature": hp.temperature,
            "old_logit_distillation_weight": hp.old_logit_distillation_weight,
            "residual_identity_weight": hp.residual_identity_weight,
            "factor_weight": hp.factor_weight,
            "seed": hp.seed,
        },
        "resource": dict(state.resource),
        "support_feature_artifact_sha256": state.support_feature_artifact_sha256,
        "support_selection_sha256": state.support_selection_sha256,
        "sealed_runtime_sha256": state.sealed_runtime_sha256,
        "feature_code_sha256": state.feature_code_sha256,
        "sealed_phase1_checkpoint_sha256": state.sealed_phase1_checkpoint_sha256,
        "operator_id": state.operator_id,
        "view_seed": state.view_seed,
        "state_content_sha256": state.state_content_sha256,
    }


def _write_state(
    output: Path, *, stem: str, state: JointResidualLogitHeadState
) -> dict[str, Any]:
    npz = output / f"{stem}.npz"
    with npz.open("xb") as handle:
        np.savez(
            handle,
            prototypes=state.prototypes,
            w1=state.w1,
            w2=state.w2,
        )
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(npz)
    metadata = output / f"{stem}.json"
    metadata_sha256 = _write_json_new(metadata, _state_metadata(state))
    with np.load(npz, allow_pickle=False) as loaded:
        if (
            not np.array_equal(loaded["prototypes"], state.prototypes)
            or not np.array_equal(loaded["w1"], state.w1)
            or not np.array_equal(loaded["w2"], state.w2)
        ):
            raise D12RunnerError("sealed state NPZ readback mismatch")
    loaded_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    if loaded_metadata.get("state_content_sha256") != state.state_content_sha256:
        raise D12RunnerError("sealed state metadata readback mismatch")
    return {
        "npz_sha256": _sha256_file(npz),
        "metadata_sha256": metadata_sha256,
        "npz_file_bytes": npz.stat().st_size,
        "metadata_file_bytes": metadata.stat().st_size,
        "content_verified_after_write": True,
    }


def _measure_support_row_pareto(
    state: JointResidualLogitHeadState,
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
    d12_ms = (time.perf_counter() - start) * 1000.0 / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        _ = query @ support.T
    qknn_ms = (time.perf_counter() - start) * 1000.0 / repeats
    d12_macs = int(
        state.resource["residual_head_mac_per_sample"]
        + state.resource["prototype_cosine_mac_per_sample"]
    )
    qknn_macs = int(len(support) * state.feature_dim)
    d12_bytes = int(state.resource["persistent_state_bytes"])
    qknn_bytes = int(support.nbytes)
    return {
        "benchmark_input": "one_support_row_resource_probe_no_query_open",
        "repeats": repeats,
        "d12_head_latency_ms": d12_ms,
        "identity_single_qknn_latency_ms": qknn_ms,
        "latency_delta_percent": 100.0 * (d12_ms / qknn_ms - 1.0),
        "d12_exact_macs": d12_macs,
        "identity_single_qknn_exact_macs": qknn_macs,
        "mac_delta_percent": 100.0 * (d12_macs / qknn_macs - 1.0),
        "d12_state_bytes": d12_bytes,
        "identity_single_qknn_state_bytes": qknn_bytes,
        "state_delta_percent": 100.0 * (d12_bytes / qknn_bytes - 1.0),
    }


def _base_h(joint: Mapping[str, Any]) -> float:
    old = float(joint["base_after_old"]["overall_accuracy"])
    new = float(joint["base_after_new"]["overall_accuracy"])
    return 0.0 if old + new <= 0.0 else 2.0 * old * new / (old + new)


def run(
    *,
    before_root: Path,
    before_seal: Path,
    after_root: Path,
    after_seal: Path,
    output: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    if output.exists():
        raise D12RunnerError("output already exists")
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
        raise D12RunnerError("before/after package binding drift")
    before_handles = {
        str(row["class_handle"]) for row in before_manifest["registered_classes"]
    }
    after_handles = {
        str(row["class_handle"]) for row in after_manifest["registered_classes"]
    }
    if not before_handles < after_handles:
        raise D12RunnerError("new-class registration set drift")
    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root, _member(before_manifest, "feature_runtime"), device=device
    )
    candidates = (
        ResidualHeadHyperparameters(
            candidate_id="d12_alpha0_base_cosine_fallback",
            epochs=0,
            learning_rate=0.0,
            alpha=0.0,
            old_logit_distillation_weight=0.0,
            residual_identity_weight=0.0,
            factor_weight=0.0,
        ),
        ResidualHeadHyperparameters(
            candidate_id="d12_rank8_a005_strong_distill",
            epochs=12,
            learning_rate=0.01,
            alpha=0.05,
            old_logit_distillation_weight=8.0,
            residual_identity_weight=4.0,
            factor_weight=0.02,
        ),
        ResidualHeadHyperparameters(
            candidate_id="d12_rank8_a010_balanced",
            epochs=12,
            learning_rate=0.01,
            alpha=0.10,
            old_logit_distillation_weight=8.0,
            residual_identity_weight=4.0,
            factor_weight=0.02,
        ),
        ResidualHeadHyperparameters(
            candidate_id="d12_rank8_a015_collision",
            epochs=12,
            learning_rate=0.005,
            alpha=0.15,
            old_logit_distillation_weight=12.0,
            residual_identity_weight=6.0,
            factor_weight=0.03,
        ),
    )
    output.mkdir(parents=True)
    module_path = CODE / "cvsrffi" / "stage2_joint_residual_logit_head.py"
    runner_path = Path(__file__).resolve()
    module_sha256 = _sha256_file(module_path)
    runner_sha256 = _sha256_file(runner_path)
    registered_feature_sha256 = _sha256_file(
        CODE / "cvsrffi" / "stage2_diag_cosine_exploration.py"
    )
    feature_code_sha256 = hashlib.sha256(
        _canonical(
            {
                "d12_module_sha256": module_sha256,
                "d12_runner_sha256": runner_sha256,
                "registered_feature_module_sha256": registered_feature_sha256,
            }
        )
    ).hexdigest()
    runtime_sha256 = str(before_manifest["feature_runtime_sha256"])
    checkpoint_sha256 = str(before_manifest["phase1_checkpoint_sha256"])
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
            raise D12RunnerError("physical support reused across LEO scenarios")
        seen_tokens.update(current_tokens)
        seen_hashes.update(current_hashes)
        feature_start = time.perf_counter()
        before_artifact = _build_feature_artifact(
            model,
            device,
            before_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            phase1_checkpoint_sha256=checkpoint_sha256,
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
            phase1_checkpoint_sha256=checkpoint_sha256,
            reuse_by_token=old_feature_by_token,
        )
        contexts[scenario] = {
            "before_rows": before_rows,
            "after_rows": after_rows,
            "before_artifact": before_artifact,
            "after_artifact": after_artifact,
            "feature_seconds": time.perf_counter() - feature_start,
            "before_provenance": _feature_provenance(
                before_rows, before_artifact
            ),
            "after_provenance": _feature_provenance(after_rows, after_artifact),
        }
    d11_v6_joint_new_reference = {
        "leo_clear_weak": {"overall": 0.52, "floor": 0.10},
        "leo_low_elev_weak": {"overall": 0.46, "floor": 0.20},
        "leo_rain_weak": {"overall": 0.60, "floor": 0.40},
    }
    trace: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        evaluations[candidate.candidate_id] = {}
        scenario_gates: dict[str, bool] = {}
        new_floors = []
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
                device=device,
            )
            evaluations[candidate.candidate_id][scenario] = joint
            trace.extend(
                {"scenario": scenario, **row} for row in joint_trace
            )
            reference = d11_v6_joint_new_reference[scenario]
            new_improved = (
                float(joint["after_new"]["overall_accuracy"])
                > reference["overall"] + 1.0e-12
                or float(joint["after_new"]["min_class_accuracy"])
                > reference["floor"] + 1.0e-12
            )
            gate = (
                candidate.alpha > 0.0
                and bool(joint["old_per_class_non_degraded_vs_before"])
                and bool(joint["old_per_class_non_degraded_vs_base_cosine"])
                and float(joint["old_forgetting"]) <= 1.0e-12
                and float(joint["after_new"]["overall_accuracy"])
                >= reference["overall"]
                and float(joint["after_new"]["min_class_accuracy"])
                >= reference["floor"]
                and new_improved
                and float(joint["h_old_new"]) > _base_h(joint) + 1.0e-12
            )
            scenario_gates[scenario] = gate
            new_floors.append(float(joint["after_new"]["min_class_accuracy"]))
            h_values.append(float(joint["h_old_new"]))
            joint_values.append(float(joint["joint_accuracy"]))
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "alpha": candidate.alpha,
                "is_alpha0_base_fallback": candidate.alpha == 0.0,
                "all_scenario_gate_pass": all(scenario_gates.values()),
                "scenario_gate_pass": scenario_gates,
                "worst_scenario_new_floor": min(new_floors),
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
    passing_positive = [
        row
        for row in candidate_rows
        if row["all_scenario_gate_pass"] and not row["is_alpha0_base_fallback"]
    ]
    if passing_positive:
        selected_id = str(passing_positive[0]["candidate_id"])
    else:
        selected_id = "d12_alpha0_base_cosine_fallback"
    selected_hp = next(
        candidate for candidate in candidates if candidate.candidate_id == selected_id
    )
    promotion_ready = bool(
        passing_positive and selected_hp.alpha > 0.0
    )
    selected_candidate_row = next(
        row for row in candidate_rows if row["candidate_id"] == selected_id
    )
    scenario_results: dict[str, Any] = {}
    state_hashes: dict[str, Any] = {}
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
            device=device,
        )
        trace.extend({"scenario": scenario, **row} for row in fitted.trace)
        probe_rows = {
            key: np.asarray(value)[:1]
            for key, value in context["after_rows"].items()
        }
        probe_feature = context["after_artifact"].features[:1]
        cursor = 0

        def extract_probe(_: np.ndarray) -> np.ndarray:
            nonlocal cursor
            cursor += 1
            return probe_feature

        probe_artifact = _build_runtime_authorized_feature_artifact_internal(
            probe_rows["iq"],
            physical_sample_ids=probe_rows["tokens"].tolist(),
            parent_received_iq_sha256=probe_rows["hashes"].tolist(),
            sealed_runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            sealed_phase1_checkpoint_sha256=checkpoint_sha256,
            extract_single_received_iq=extract_probe,
        )
        joint = evaluations[selected_id][scenario]
        scenario_results[scenario] = {
            "joint_leave_two_out": joint,
            "base_cosine_h_old_new": _base_h(joint),
            "support_gate_pass": selected_candidate_row["scenario_gate_pass"][
                scenario
            ],
            "resource_before": dict(fitted.before_state.resource),
            "resource_after": dict(fitted.after_state.resource),
            "pareto_vs_identity_single_qknn": _measure_support_row_pareto(
                fitted.after_state,
                probe_artifact,
                context["after_artifact"].features,
            ),
            "before_feature_provenance": context["before_provenance"],
            "after_feature_provenance": context["after_provenance"],
            "old_lineage_and_feature_exact_reuse_locked": True,
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
        "SUPPORT_ONLY_D12_SELECTED_NO_QUERY_OPEN"
        if promotion_ready
        else "SUPPORT_ONLY_D12_NOT_SELECTED_NO_QUERY_OPEN"
    )
    training_log_sha256 = _write_jsonl_new(output / "training_log.jsonl", trace)
    audit = {
        "schema": "cvs.phase2.d12_support_only_audit.v1",
        "status": status,
        "claim_scope": "development_support_only_no_query_performance_claim",
        "receiver": after_manifest["receiver"],
        "seed": int(after_manifest["seed"]),
        "k_shot": 10,
        "view_policy": "one_fixed_received_iq_base_view_no_new_channel_state",
        "unified_hyperparameter_selection": {
            "selected_candidate_id": selected_id,
            "same_candidate_all_scenarios": True,
            "candidate_rows": candidate_rows,
            "candidate_alphas": [candidate.alpha for candidate in candidates],
            "alpha0_fallback_policy": (
                "if_all_positive_alpha_fail_report_base_cosine_fallback_as_no_go"
            ),
            "alpha0_fallback_is_d12_improvement": False,
        },
        "scenario_results": scenario_results,
        "d10_after_support_reference": {
            "comparison_status": (
                "NOT_COMPARABLE_D10_VALUES_ARE_NOT_JOINT_HELD_NEW_ONLY"
            )
        },
        "d11_v6_joint_new_reference": d11_v6_joint_new_reference,
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
            "d12_module_sha256": module_sha256,
            "d12_runner_sha256": runner_sha256,
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
        "# D12联合注册残差logit head支持集审计",
        "",
        f"状态：`{status}`。只打开严格K10 before/after enrollment-only包，未打开query、truth、prediction、score或scorer。",
        "",
        f"三场景统一选择`{selected_id}`。若正alpha候选未通过，alpha0只作为base cosine安全回退并维持NO-GO。",
        "",
        "|场景|Base old/new/H|D12 old/floor|D12 new/floor|joint/H|ΔH vs base|forgetting|old逐类不低于before/base|D11-v6 joint-new参考|参数|epoch|state bytes|门|",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        row = scenario_results[scenario]
        joint = row["joint_leave_two_out"]
        resource = row["resource_after"]
        reference = d11_v6_joint_new_reference[scenario]
        lines.append(
            f"|`{scenario}`|"
            f"{joint['base_after_old']['overall_accuracy']:.4f}/"
            f"{joint['base_after_new']['overall_accuracy']:.4f}/"
            f"{row['base_cosine_h_old_new']:.4f}|"
            f"{joint['after_old']['overall_accuracy']:.4f}/"
            f"{joint['after_old']['min_class_accuracy']:.4f}|"
            f"{joint['after_new']['overall_accuracy']:.4f}/"
            f"{joint['after_new']['min_class_accuracy']:.4f}|"
            f"{joint['joint_accuracy']:.4f}/{joint['h_old_new']:.4f}|"
            f"{joint['delta_vs_base_h_old_new']:.4f}|"
            f"{joint['old_forgetting']:.4f}|"
            f"{joint['old_per_class_non_degraded_vs_before']}/"
            f"{joint['old_per_class_non_degraded_vs_base_cosine']}|"
            f"{reference['overall']:.4f}/{reference['floor']:.4f}|"
            f"{resource['trainable_parameters']}|"
            f"{resource['adapt_epochs']}|"
            f"{resource['persistent_state_bytes']}|"
            f"{row['support_gate_pass']}|"
        )
    lines.extend(
        [
            "",
            "K1/K5未从K10内存切前缀；必须等待各自独立strict sealed package。D12本轮不开放query。",
            "",
        ]
    )
    report_sha256 = _write_text_new(output / "report.md", "\n".join(lines))
    commit = {
        "schema": "cvs.phase2.d12_support_only_commit.v1",
        "status": status,
        "support_audit_sha256": audit_sha256,
        "training_log_sha256": training_log_sha256,
        "report_sha256": report_sha256,
        "state_sha256": state_hashes,
        "promotion_ready_for_single_query_candidate": promotion_ready,
        "query_opened": False,
        "selected_candidate_id": selected_id,
        "sealed_runtime_sha256": runtime_sha256,
        "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
        "d12_module_sha256": module_sha256,
        "d12_runner_sha256": runner_sha256,
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
