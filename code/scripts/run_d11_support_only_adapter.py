"""Run D11 on sealed strict-K10 enrollment packages without opening query."""

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
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)
from cvsrffi.stage2_trainable_lowrank_support_adapter import (  # noqa: E402
    AdapterHyperparameters,
    TrainableLowRankAdapterState,
    _build_validated_feature_artifact_internal,
    evaluate_joint_registration_leave_two_out,
    predict_all_registered,
    register_new_classes,
    select_and_fit_k10,
)


class D11RunnerError(ValueError):
    """Raised when the enrollment-only protocol fails closed."""


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
        raise D11RunnerError(f"enrollment member drift: {kind}")
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
        raise D11RunnerError("enrollment package protocol drift")
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
        raise D11RunnerError("enrollment member allowlist drift")


def _load_enrollment(
    root: Path, seal: Path, *, registration_state: str
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    if root.name != "enrollment_only" or "enrollment" not in seal.name:
        raise D11RunnerError("D11 accepts enrollment-only paths")
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
        "hashes": np.asarray(payload["support_post_channel_iq_sha256"]).astype(str)[order],
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
        raise D11RunnerError(f"strict K10 payload drift: {scenario}")
    return rows


def _base_features(
    model: torch.nn.Module, device: torch.device, iq: np.ndarray
) -> dict[str, np.ndarray]:
    # Support extraction is deliberately per physical sample.  This prevents
    # batch-composition state from entering the deployable feature fingerprint
    # and matches the future one-row query callback.
    zid = forward_zid160(model, iq, device=device, batch_size=1)
    return {"base": registered_feature(iq, zid)}


def _build_feature_artifact(
    model: torch.nn.Module,
    device: torch.device,
    rows: Mapping[str, np.ndarray],
    *,
    runtime_sha256: str,
    feature_code_sha256: str,
    phase1_checkpoint_sha256: str,
):
    def extract_single(iq: np.ndarray, view_id: str) -> np.ndarray:
        if view_id != "base" or len(iq) != 1:
            raise D11RunnerError("D11 authorized extractor accepts one base row")
        return _base_features(model, device, iq)["base"]

    return _build_validated_feature_artifact_internal(
        rows["iq"],
        physical_sample_ids=rows["tokens"].tolist(),
        parent_received_iq_sha256=rows["hashes"].tolist(),
        sealed_runtime_sha256=runtime_sha256,
        feature_code_sha256=feature_code_sha256,
        sealed_phase1_checkpoint_sha256=phase1_checkpoint_sha256,
        view_seed_by_id={"base": 0},
        extract_single_received_iq_view=extract_single,
    )


def _feature_provenance(
    rows: Mapping[str, np.ndarray],
    features: np.ndarray,
    *,
    runtime_sha256: str,
    module_sha256: str,
    phase1_checkpoint_sha256: str,
) -> list[dict[str, Any]]:
    if len(features) != len(rows["hashes"]):
        raise D11RunnerError("feature provenance alignment drift")
    return [
        {
            "physical_sample_id": str(rows["tokens"][index]),
            "parent_received_iq_sha256": str(rows["hashes"][index]),
            "operator_id": "base",
            "view_seed": 0,
            "feature_sha256": hashlib.sha256(
                np.ascontiguousarray(features[index]).tobytes()
            ).hexdigest(),
            "sealed_runtime_sha256": runtime_sha256,
            "sealed_phase1_checkpoint_sha256": phase1_checkpoint_sha256,
            "d11_feature_code_sha256": module_sha256,
        }
        for index in range(len(features))
    ]


def _state_metadata(state: TrainableLowRankAdapterState) -> dict[str, Any]:
    hp = state.hyperparameters
    return {
        "schema": state.schema,
        "candidate_id": state.candidate_id,
        "classes": list(state.classes),
        "feature_dim": state.feature_dim,
        "k_shot": state.k_shot,
        "view_ids": list(state.view_ids),
        "old_class_count": state.old_class_count,
        "registration_generation": state.registration_generation,
        "hyperparameters": {
            "rank": hp.rank,
            "epochs": hp.epochs,
            "learning_rate": hp.learning_rate,
            "temperature": hp.temperature,
            "prototype_weight": hp.prototype_weight,
            "supervised_contrastive_weight": hp.supervised_contrastive_weight,
            "identity_weight": hp.identity_weight,
            "factor_weight": hp.factor_weight,
            "seed": hp.seed,
        },
        "resource": dict(state.resource),
        "support_feature_artifact_sha256": state.support_feature_artifact_sha256,
        "sealed_runtime_sha256": state.sealed_runtime_sha256,
        "feature_code_sha256": state.feature_code_sha256,
        "sealed_phase1_checkpoint_sha256": state.sealed_phase1_checkpoint_sha256,
        "state_content_sha256": state.state_content_sha256,
        "old_prototypes_sha256": hashlib.sha256(
            np.ascontiguousarray(state.prototypes[: state.old_class_count]).tobytes()
        ).hexdigest(),
    }


def _write_state(
    output: Path, *, stem: str, state: TrainableLowRankAdapterState
) -> dict[str, str]:
    npz = output / f"{stem}.npz"
    with npz.open("xb") as handle:
        np.savez(
            handle,
            low_rank_u=state.low_rank_u,
            low_rank_v=state.low_rank_v,
            gate=state.gate,
            prototypes=state.prototypes,
        )
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(npz)
    metadata = output / f"{stem}.json"
    return {
        "npz_sha256": _sha256_file(npz),
        "metadata_sha256": _write_json_new(metadata, _state_metadata(state)),
        "npz_file_bytes": npz.stat().st_size,
        "metadata_file_bytes": metadata.stat().st_size,
    }


def _measure_support_row_pareto(
    state: TrainableLowRankAdapterState,
    query_artifact,
    qknn_support_features: np.ndarray,
    *,
    repeats: int = 500,
) -> dict[str, Any]:
    support = np.array(qknn_support_features, dtype=np.float32, copy=True)
    support /= np.maximum(np.linalg.norm(support, axis=1, keepdims=True), 1.0e-8)
    query = np.asarray(query_artifact.features_by_view["base"], dtype=np.float32)
    for _ in range(20):
        predict_all_registered(state, query_artifact)
        _ = query @ support.T
    start = time.perf_counter()
    for _ in range(repeats):
        predict_all_registered(state, query_artifact)
    d11_ms = (time.perf_counter() - start) * 1000.0 / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        _ = query @ support.T
    qknn_ms = (time.perf_counter() - start) * 1000.0 / repeats
    qknn_macs = int(len(support) * state.feature_dim)
    d11_macs = int(
        state.resource["adapter_mac_per_view"] + len(state.classes) * state.feature_dim
    )
    qknn_state_bytes = int(support.nbytes)
    d11_state_bytes = int(state.resource["persistent_state_bytes"])
    return {
        "benchmark_input": "one_held_out_resource_probe_support_row_no_query_open",
        "repeats": repeats,
        "d11_adapter_plus_prototype_latency_ms": d11_ms,
        "identity_single_qknn_latency_ms": qknn_ms,
        "latency_delta_percent": 100.0 * (d11_ms / qknn_ms - 1.0),
        "d11_exact_macs": d11_macs,
        "identity_single_qknn_exact_macs": qknn_macs,
        "mac_delta_percent": 100.0 * (d11_macs / qknn_macs - 1.0),
        "d11_state_bytes": d11_state_bytes,
        "identity_single_qknn_state_bytes": qknn_state_bytes,
        "state_delta_percent": 100.0 * (d11_state_bytes / qknn_state_bytes - 1.0),
    }


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
        raise D11RunnerError("output already exists")
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
        raise D11RunnerError("before/after binding drift")
    before_handles = {
        str(row["class_handle"]) for row in before_manifest["registered_classes"]
    }
    after_handles = {
        str(row["class_handle"]) for row in after_manifest["registered_classes"]
    }
    if not before_handles < after_handles:
        raise D11RunnerError("new-class registration set drift")
    device = torch.device(
        "cuda:0" if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root, _member(before_manifest, "feature_runtime"), device=device
    )
    candidates = (
        AdapterHyperparameters(
            candidate_id="d11_rank8_identity_strong",
            rank=8,
            epochs=12,
            learning_rate=0.005,
            identity_weight=20.0,
            factor_weight=0.05,
            seed=20260717,
        ),
        AdapterHyperparameters(
            candidate_id="d11_rank8_floor_seek",
            rank=8,
            epochs=12,
            learning_rate=0.02,
            identity_weight=5.0,
            factor_weight=0.02,
            seed=20260717,
        ),
    )
    output.mkdir(parents=True)
    module_path = CODE / "cvsrffi" / "stage2_trainable_lowrank_support_adapter.py"
    runner_path = Path(__file__).resolve()
    module_sha256 = _sha256_file(module_path)
    runner_sha256 = _sha256_file(runner_path)
    diag_feature_sha256 = _sha256_file(
        CODE / "cvsrffi" / "stage2_diag_cosine_exploration.py"
    )
    feature_code_sha256 = hashlib.sha256(
        _canonical(
            {
                "d11_module_sha256": module_sha256,
                "d11_runner_sha256": runner_sha256,
                "registered_feature_module_sha256": diag_feature_sha256,
            }
        )
    ).hexdigest()
    runtime_sha256 = str(before_manifest["feature_runtime_sha256"])
    phase1_checkpoint_sha256 = str(before_manifest["phase1_checkpoint_sha256"])
    tracemalloc.start()
    run_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trace: list[dict[str, Any]] = []
    scenario_results: dict[str, Any] = {}
    state_hashes: dict[str, Any] = {}
    prior_tokens: set[str] = set()
    prior_hashes: set[str] = set()
    contexts: dict[str, dict[str, Any]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        scenario_start = time.perf_counter()
        before_rows = _payload_rows(
            before_payloads[scenario], before_manifest, scenario=scenario
        )
        after_rows = _payload_rows(
            after_payloads[scenario], after_manifest, scenario=scenario
        )
        current_tokens = set(after_rows["tokens"].tolist())
        current_hashes = set(after_rows["hashes"].tolist())
        if current_tokens.intersection(prior_tokens) or current_hashes.intersection(prior_hashes):
            raise D11RunnerError("physical support reused across LEO scenarios")
        prior_tokens.update(current_tokens)
        prior_hashes.update(current_hashes)
        feature_start = time.perf_counter()
        before_artifact = _build_feature_artifact(
            model,
            device,
            before_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            phase1_checkpoint_sha256=phase1_checkpoint_sha256,
        )
        after_artifact = _build_feature_artifact(
            model,
            device,
            after_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            phase1_checkpoint_sha256=phase1_checkpoint_sha256,
        )
        before_features = before_artifact.features_by_view
        after_features = after_artifact.features_by_view
        before_provenance = _feature_provenance(
            before_rows,
            before_features["base"],
            runtime_sha256=runtime_sha256,
            module_sha256=feature_code_sha256,
            phase1_checkpoint_sha256=phase1_checkpoint_sha256,
        )
        after_provenance = _feature_provenance(
            after_rows,
            after_features["base"],
            runtime_sha256=runtime_sha256,
            module_sha256=feature_code_sha256,
            phase1_checkpoint_sha256=phase1_checkpoint_sha256,
        )
        feature_seconds = time.perf_counter() - feature_start
        old_indices_after = np.asarray(
            [label in before_handles for label in after_rows["labels"]], dtype=bool
        )
        prior_by_token = {
            str(token): (str(label), str(digest), before_features["base"][index])
            for index, (token, label, digest) in enumerate(
                zip(before_rows["tokens"], before_rows["labels"], before_rows["hashes"])
            )
        }
        max_old_feature_reextract_abs_diff = 0.0
        for index in np.flatnonzero(old_indices_after):
            token = str(after_rows["tokens"][index])
            prior = prior_by_token.get(token)
            if (
                prior is None
                or prior[0] != str(after_rows["labels"][index])
                or prior[1] != str(after_rows["hashes"][index])
            ):
                raise D11RunnerError("old support lineage changed after registration")
            before_feature_sha = hashlib.sha256(
                np.ascontiguousarray(prior[2]).tobytes()
            ).hexdigest()
            reextract_diff = float(
                np.max(np.abs(after_features["base"][index] - prior[2]))
            )
            max_old_feature_reextract_abs_diff = max(
                max_old_feature_reextract_abs_diff, reextract_diff
            )
            if reextract_diff > 1.0e-6:
                raise D11RunnerError("after old feature re-extraction drift exceeded tolerance")
            if (
                hashlib.sha256(
                    np.ascontiguousarray(prior[2]).tobytes()
                ).hexdigest()
                != before_feature_sha
            ):
                raise D11RunnerError("after old feature fingerprint lock failed")
            after_provenance[index]["locked_old_feature_sha256"] = before_feature_sha
            after_provenance[index]["locked_old_feature_reused_for_state"] = True
        new_indices = ~old_indices_after
        new_rows = {
            key: np.asarray(value)[new_indices] for key, value in after_rows.items()
        }
        new_artifact = _build_feature_artifact(
            model,
            device,
            new_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            phase1_checkpoint_sha256=phase1_checkpoint_sha256,
        )
        contexts[scenario] = {
            "scenario_start": scenario_start,
            "feature_seconds": feature_seconds,
            "before_rows": before_rows,
            "before_features": before_features,
            "before_artifact": before_artifact,
            "before_provenance": before_provenance,
            "after_provenance": after_provenance,
            "after_artifact": after_artifact,
            "after_rows": after_rows,
            "new_features": new_artifact.features_by_view,
            "new_artifact": new_artifact,
            "new_labels": after_rows["labels"][new_indices],
            "new_ranks": after_rows["ranks"][new_indices],
            "max_old_feature_reextract_abs_diff": (
                max_old_feature_reextract_abs_diff
            ),
        }

    candidate_results: dict[str, dict[str, Any]] = {}
    joint_selection_rows: list[dict[str, Any]] = []
    d10_after_reference = {
        "leo_clear_weak": {"overall": 0.50, "floor": 0.20},
        "leo_low_elev_weak": {"overall": 0.46, "floor": 0.10},
        "leo_rain_weak": {"overall": 0.70, "floor": 0.50},
    }
    for candidate in candidates:
        candidate_results[candidate.candidate_id] = {}
        scenario_gate_pass = True
        scenario_floors: list[float] = []
        scenario_overall: list[float] = []
        scenario_harmonic: list[float] = []
        scenario_joint_gates: dict[str, bool] = {}
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            context = contexts[scenario]
            train_start = time.perf_counter()
            result = select_and_fit_k10(
                context["before_artifact"],
                context["before_rows"]["labels"],
                context["before_rows"]["ranks"],
                candidates=(candidate,),
                device=device,
                require_gate=False,
            )
            train_seconds = time.perf_counter() - train_start
            candidate_results[candidate.candidate_id][scenario] = {
                "result": result,
                "train_seconds": train_seconds,
            }
            for row in result.trace:
                trace.append({"scenario": scenario, **row})
            selected_row = result.validation["candidates"][0]
            joint_result, joint_trace = evaluate_joint_registration_leave_two_out(
                context["before_artifact"],
                context["before_rows"]["labels"],
                context["before_rows"]["ranks"],
                context["new_artifact"],
                context["new_labels"],
                context["new_ranks"],
                hyperparameters=candidate,
                device=device,
            )
            for row in joint_trace:
                trace.append({"scenario": scenario, **row})
            candidate_results[candidate.candidate_id][scenario][
                "joint_result"
            ] = joint_result
            reference = d10_after_reference[scenario]
            joint_gate = (
                bool(selected_row["gate_pass"])
                and bool(joint_result["old_per_class_non_degraded"])
                and joint_result["after_new"]["overall_accuracy"]
                >= reference["overall"]
                and joint_result["after_new"]["min_class_accuracy"]
                >= reference["floor"]
                and (
                    scenario != "leo_low_elev_weak"
                    or joint_result["after_new"]["min_class_accuracy"]
                    > reference["floor"]
                )
            )
            scenario_joint_gates[scenario] = joint_gate
            scenario_gate_pass &= joint_gate
            scenario_floors.append(
                float(joint_result["after_new"]["min_class_accuracy"])
            )
            scenario_overall.append(
                float(joint_result["after_new"]["overall_accuracy"])
            )
            scenario_harmonic.append(float(joint_result["h_old_new"]))
        joint_selection_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "all_scenario_identity_gate_pass": scenario_gate_pass,
                "worst_scenario_floor_accuracy": min(scenario_floors),
                "mean_scenario_overall_accuracy": float(np.mean(scenario_overall)),
                "mean_scenario_h_old_new": float(np.mean(scenario_harmonic)),
                "scenario_joint_gate_pass": scenario_joint_gates,
                "scenario_floor_accuracy": {
                    scenario: scenario_floors[index]
                    for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
                },
                "scenario_overall_accuracy": {
                    scenario: scenario_overall[index]
                    for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
                },
            }
        )
    joint_selection_rows.sort(
        key=lambda row: (
            row["all_scenario_identity_gate_pass"],
            row["worst_scenario_floor_accuracy"],
            row["mean_scenario_h_old_new"],
            row["mean_scenario_overall_accuracy"],
            row["candidate_id"],
        ),
        reverse=True,
    )
    selected_candidate_id = str(joint_selection_rows[0]["candidate_id"])
    all_gate_pass = bool(
        joint_selection_rows[0]["all_scenario_identity_gate_pass"]
    )
    old_lock_pass = True
    selected_hyperparameters = next(
        item for item in candidates if item.candidate_id == selected_candidate_id
    )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        context = contexts[scenario]
        unified_refit_start = time.perf_counter()
        before_result = select_and_fit_k10(
            context["before_artifact"],
            context["before_rows"]["labels"],
            context["before_rows"]["ranks"],
            candidates=(selected_hyperparameters,),
            device=device,
            require_gate=False,
        )
        unified_refit_seconds = time.perf_counter() - unified_refit_start
        for row in before_result.trace:
            trace.append(
                {
                    "scenario": scenario,
                    "unified_hyperparameter_refit": True,
                    **row,
                }
            )
        registration_start = time.perf_counter()
        registration_validation = candidate_results[selected_candidate_id][scenario][
            "joint_result"
        ]
        after_state = register_new_classes(
            before_result.state,
            context["new_artifact"],
            context["new_labels"],
            context["new_ranks"],
            k_shot=10,
            expected_old_support_feature_artifact_sha256=(
                before_result.state.support_feature_artifact_sha256
            ),
        )
        registration_seconds = time.perf_counter() - registration_start
        lock_ok = (
            np.array_equal(after_state.low_rank_u, before_result.state.low_rank_u)
            and np.array_equal(after_state.low_rank_v, before_result.state.low_rank_v)
            and np.array_equal(after_state.gate, before_result.state.gate)
            and np.array_equal(
                after_state.prototypes[: len(before_result.state.classes)],
                before_result.state.prototypes,
            )
        )
        probe_rows = {
            key: np.asarray(value)[:1] for key, value in context["before_rows"].items()
        }
        probe_artifact = _build_feature_artifact(
            model,
            device,
            probe_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            phase1_checkpoint_sha256=phase1_checkpoint_sha256,
        )
        pareto = _measure_support_row_pareto(
            after_state,
            probe_artifact,
            context["after_artifact"].features_by_view["base"],
        )
        old_lock_pass &= lock_ok
        scenario_results[scenario] = {
            "before_selection": dict(before_result.validation),
            "joint_unified_selection": {
                "policy": (
                    "all_scenario_identity_gate_then_worst_scenario_floor_"
                    "then_mean_overall"
                ),
                "selected_candidate_id": selected_candidate_id,
                "candidate_rows": joint_selection_rows,
            },
            "joint_registration_leave_two_out": registration_validation,
            "after_old_adapter_and_prototypes_bitwise_locked": lock_ok,
            "resource_before": dict(before_result.state.resource),
            "resource_after": dict(after_state.resource),
            "measured_pareto_vs_identity_single_qknn": pareto,
            "selected_candidate_id": selected_candidate_id,
            "before_feature_provenance": context["before_provenance"],
            "after_feature_provenance": context["after_provenance"],
            "after_old_feature_fingerprint_locked": True,
            "max_old_feature_reextract_abs_diff": context[
                "max_old_feature_reextract_abs_diff"
            ],
            "measured": {
                "feature_extraction_seconds": context["feature_seconds"],
                "adapter_selection_and_fit_seconds": unified_refit_seconds,
                "registration_seconds": registration_seconds,
                "scenario_total_seconds": (
                    context["feature_seconds"]
                    + unified_refit_seconds
                    + registration_seconds
                ),
                "exact_adapter_mac_per_view_from_runtime_shapes": int(
                    2
                    * before_result.state.feature_dim
                    * before_result.state.hyperparameters.rank
                    + before_result.state.hyperparameters.rank
                ),
            },
        }
        state_hashes[f"{scenario}:before:k10"] = _write_state(
            output,
            stem=f"state_{scenario}_before_k10",
            state=before_result.state,
        )
        state_hashes[f"{scenario}:after:k10"] = _write_state(
            output,
            stem=f"state_{scenario}_after_k10",
            state=after_state,
        )
    d10_pass = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        joint = scenario_results[scenario]["joint_registration_leave_two_out"]
        metric = joint["after_new"]
        reference = d10_after_reference[scenario]
        d10_pass[scenario] = {
            "overall_non_degraded": metric["overall_accuracy"] >= reference["overall"],
            "floor_non_degraded": metric["min_class_accuracy"] >= reference["floor"],
            "low_floor_strictly_improved": (
                scenario != "leo_low_elev_weak"
                or metric["min_class_accuracy"] > reference["floor"]
            ),
            "old_per_class_non_degraded": joint["old_per_class_non_degraded"],
        }
    promotion_ready = (
        all_gate_pass
        and old_lock_pass
        and all(all(row.values()) for row in d10_pass.values())
    )
    status = (
        "SUPPORT_ONLY_D11_SELECTED_NO_QUERY_OPEN"
        if promotion_ready
        else "SUPPORT_ONLY_D11_NOT_SELECTED_NO_QUERY_OPEN"
    )
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    measured_run = {
        "wall_seconds": time.perf_counter() - run_start,
        "peak_python_tracemalloc_bytes": int(peak_python_bytes),
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
    }
    training_log_sha = _write_jsonl_new(output / "training_log.jsonl", trace)
    audit = {
        "schema": "cvs.phase2.d11_support_only_audit.v1",
        "status": status,
        "claim_scope": "support_only_no_query_performance_claim",
        "receiver": after_manifest["receiver"],
        "seed": int(after_manifest["seed"]),
        "k_shot": 10,
        "view_ids": ["base"],
        "view_policy": "one_fixed_received_iq_view_no_new_channel_state",
        "unified_k10_hyperparameter_selection": {
            "selected_candidate_id": selected_candidate_id,
            "policy": (
                "all_scenario_identity_gate_then_worst_scenario_floor_"
                "then_mean_overall"
            ),
            "candidate_rows": joint_selection_rows,
            "same_hyperparameters_refit_for_all_scenarios": True,
        },
        "scenario_results": scenario_results,
        "d10_after_support_reference": d10_after_reference,
        "d10_comparison_gate": d10_pass,
        "measured_run_resource": measured_run,
        "runtime_authorization": {
            "feature_extraction_mode": "internal_sealed_runtime_no_external_feature_input",
            "sealed_runtime_sha256": runtime_sha256,
            "sealed_phase1_checkpoint_sha256": phase1_checkpoint_sha256,
            "d11_module_sha256": module_sha256,
            "d11_runner_sha256": runner_sha256,
            "registered_feature_module_sha256": diag_feature_sha256,
            "combined_feature_code_sha256": feature_code_sha256,
            "support_feature_binding_fields": [
                "physical_sample_id",
                "parent_received_iq_sha256",
                "operator_id",
                "view_seed",
                "feature_sha256",
                "sealed_runtime_sha256",
                "d11_feature_code_sha256",
            ],
            "future_query_callback_rows": 1,
        },
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
        "k1_k5_status": (
            "DEFERRED_UNTIL_INDEPENDENT_STRICT_SEALED_PACKAGES;"
            "NO_K10_PREFIX_SLICING_OR_SURPLUS_REACHABILITY"
        ),
        "opened_package_profiles": [
            "before:enrollment_only",
            "after:enrollment_only",
        ],
        "state_sha256": state_hashes,
        "training_log_sha256": training_log_sha,
        "before_package_root_sha256": before_manifest["package_root_sha256"],
        "after_package_root_sha256": after_manifest["package_root_sha256"],
        "before_seal_sha256": _sha256_file(before_seal),
        "after_seal_sha256": _sha256_file(after_seal),
        "preopen_audit": {"before": before_preopen, "after": after_preopen},
    }
    audit_sha = _write_json_new(output / "support_audit.json", audit)
    lines = [
        "# D11极轻量可训练低秩adapter支持集审计",
        "",
        f"状态：`{status}`。仅打开严格K10 before/after enrollment-only包；未打开query、truth、prediction、score或scorer。",
        "",
        "每个物理IQ仅有包内单一LEO_weak观测。本轮最轻配置只使用其`base`视图；未生成第二LEO信道状态，view不增加K。",
        "",
        "|场景|Before L2O overall/floor|After-old overall/floor|After-new overall/floor|H(old,new)|D10 after overall/floor|rank|参数|epoch|state bytes|旧状态冻结|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        row = scenario_results[scenario]
        before = next(
            item
            for item in row["before_selection"]["candidates"]
            if item["candidate_id"] == row["selected_candidate_id"]
        )
        joint = row["joint_registration_leave_two_out"]
        after_old = joint["after_old"]
        after = joint["after_new"]
        reference = d10_after_reference[scenario]
        resource = row["resource_after"]
        lines.append(
            f"|`{scenario}`|{before['overall_accuracy']:.4f}/{before['min_class_accuracy']:.4f}|"
            f"{after_old['overall_accuracy']:.4f}/{after_old['min_class_accuracy']:.4f}|"
            f"{after['overall_accuracy']:.4f}/{after['min_class_accuracy']:.4f}|"
            f"{joint['h_old_new']:.4f}|"
            f"{reference['overall']:.4f}/{reference['floor']:.4f}|8|"
            f"{resource['trainable_parameters']}|{resource['adapt_epochs']}|"
            f"{resource['persistent_state_bytes']}|{row['after_old_adapter_and_prototypes_bitwise_locked']}|"
        )
    lines.extend(
        [
            "",
            "K1/K5没有从本次K10内存切前缀：必须等待各自独立strict sealed enrollment包，复用K10锁定结构与超参数后再训练。",
            "",
        ]
    )
    report_sha = _write_text_new(output / "report.md", "\n".join(lines))
    commit = {
        "schema": "cvs.phase2.d11_support_only_commit.v1",
        "status": status,
        "support_audit_sha256": audit_sha,
        "training_log_sha256": training_log_sha,
        "report_sha256": report_sha,
        "state_sha256": state_hashes,
        "promotion_ready_for_single_query_candidate": promotion_ready,
        "query_opened": False,
        "sealed_runtime_sha256": runtime_sha256,
        "sealed_phase1_checkpoint_sha256": before_manifest[
            "phase1_checkpoint_sha256"
        ],
        "d11_module_sha256": module_sha256,
        "d11_runner_sha256": runner_sha256,
        "registered_feature_module_sha256": diag_feature_sha256,
        "combined_feature_code_sha256": feature_code_sha256,
    }
    commit_sha = _write_json_new(output / "COMMIT.json", commit)
    return {"commit_sha256": commit_sha, **commit}


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
