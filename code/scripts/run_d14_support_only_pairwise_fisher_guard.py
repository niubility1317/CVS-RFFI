"""Run D14 on sealed strict-K10 enrollment packages without opening query."""

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

from cvsrffi.somph_diagnostic_bundle_loader import (  # noqa: E402
    load_verified_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS  # noqa: E402
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_joint_residual_logit_head import (  # noqa: E402
    _build_runtime_authorized_feature_artifact_internal,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)
from cvsrffi.stage2_sparse_pairwise_fisher_guard import (  # noqa: E402
    RIVAL_NEW_QUANTILE,
    RIVAL_OLD_QUANTILE,
    SparsePairwiseFisherHyperparameters,
    SparsePairwiseFisherState,
    _score_numpy,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    load_sparse_pairwise_fisher_state,
)


MAX_SERIALIZED_STATE_BYTES = 256 * 1024
PREFERRED_SERIALIZED_STATE_BYTES = 80 * 1024


class D14RunnerError(ValueError):
    """Raised when the support-only D14 protocol fails closed."""


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
        raise D14RunnerError(f"enrollment member drift: {kind}")
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
        raise D14RunnerError("enrollment package protocol drift")
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
        raise D14RunnerError("enrollment member allowlist drift")


def _load_enrollment(
    root: Path,
    seal: Path,
    *,
    registration_state: str,
    expected_seal_sha256: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    if root.name != "enrollment_only" or "enrollment" not in seal.name:
        raise D14RunnerError("D14 accepts enrollment-only paths")
    payloads, manifest, audit = load_verified_somph_predictor_bundle(
        root,
        detached_seal_path=seal,
        expected_seal_sha256=expected_seal_sha256,
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
        raise D14RunnerError(f"strict K10 payload drift: {scenario}")
    return rows


def _base_feature(
    model: torch.nn.Module, device: torch.device, iq: np.ndarray
) -> np.ndarray:
    if len(iq) != 1:
        raise D14RunnerError("D14 extraction requires physical batch=1")
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


def _feature_provenance(rows: Mapping[str, np.ndarray], artifact) -> list[dict[str, Any]]:
    return [
        {
            "physical_sample_id": str(rows["tokens"][index]),
            "parent_received_iq_sha256": str(rows["hashes"][index]),
            "operator_id": "base",
            "view_seed": 0,
            "feature_sha256": artifact.per_row_feature_sha256[index],
            "sealed_runtime_sha256": artifact.sealed_runtime_sha256,
            "sealed_phase1_checkpoint_sha256": (
                artifact.sealed_phase1_checkpoint_sha256
            ),
        }
        for index in range(len(rows["tokens"]))
    ]


def _state_arrays(state: SparsePairwiseFisherState) -> dict[str, np.ndarray]:
    return {
        "prototypes": state.prototypes,
        "old_edge_pairs": state.old_edge_pairs,
        "old_edge_directions": state.old_edge_directions,
        "old_edge_bias": state.old_edge_bias,
        "new_rivals": state.new_rivals,
        "new_edge_directions": state.new_edge_directions,
        "new_edge_bias": state.new_edge_bias,
    }


def _state_metadata(state: SparsePairwiseFisherState) -> dict[str, Any]:
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
            "operator_id": hp.operator_id,
            "ridge": hp.ridge,
            "gamma_old": hp.gamma_old,
            "gamma_new": hp.gamma_new,
            "select_band_old": hp.select_band_old,
            "band_old": hp.band_old,
            "band_new": hp.band_new,
            "max_old_edges": hp.max_old_edges,
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
        "state_content_sha256": state.state_content_sha256,
    }


def _write_state(
    output: Path, *, stem: str, state: SparsePairwiseFisherState
) -> dict[str, Any]:
    npz = output / f"{stem}.npz"
    with npz.open("xb") as handle:
        np.savez(handle, **_state_arrays(state))
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(npz)
    metadata = output / f"{stem}.json"
    metadata_sha256 = _write_json_new(metadata, _state_metadata(state))
    npz_sha256 = _sha256_file(npz)
    with np.load(npz, allow_pickle=False) as loaded:
        if set(loaded.files) != set(_state_arrays(state)) or any(
            not np.array_equal(loaded[name], value)
            for name, value in _state_arrays(state).items()
        ):
            raise D14RunnerError("sealed state NPZ readback mismatch")
    loaded_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    if loaded_metadata.get("state_content_sha256") != state.state_content_sha256:
        raise D14RunnerError("sealed state metadata readback mismatch")
    rebuilt = load_sparse_pairwise_fisher_state(
        npz,
        metadata,
        expected_npz_sha256=npz_sha256,
        expected_metadata_sha256=metadata_sha256,
    )
    rng = np.random.default_rng(20260717)
    probe = rng.normal(size=(7, state.feature_dim)).astype(np.float32)
    if (
        rebuilt.state_content_sha256 != state.state_content_sha256
        or not np.array_equal(_score_numpy(probe, rebuilt), _score_numpy(probe, state))
        or not np.array_equal(
            np.argmax(_score_numpy(probe, rebuilt), axis=1),
            np.argmax(_score_numpy(probe, state), axis=1),
        )
    ):
        raise D14RunnerError("rebuilt state prediction roundtrip mismatch")
    serialized_bytes = npz.stat().st_size + metadata.stat().st_size
    if serialized_bytes > MAX_SERIALIZED_STATE_BYTES:
        raise D14RunnerError("serialized state exceeds 256KiB")
    return {
        "npz_sha256": npz_sha256,
        "metadata_sha256": metadata_sha256,
        "npz_file_bytes": npz.stat().st_size,
        "metadata_file_bytes": metadata.stat().st_size,
        "serialized_state_total_bytes": serialized_bytes,
        "serialized_state_under_256kib": True,
        "serialized_state_preferred_under_80kib": (
            serialized_bytes <= PREFERRED_SERIALIZED_STATE_BYTES
        ),
        "content_verified_after_write": True,
        "state_rebuilt_and_prediction_bitwise_verified": True,
    }


def _select_candidate(
    candidate_rows: list[Mapping[str, Any]],
    candidates: tuple[SparsePairwiseFisherHyperparameters, ...],
) -> tuple[str, bool]:
    passing = [
        row
        for row in candidate_rows
        if bool(row["all_scenario_gate_pass"]) and not bool(row["force_zero"])
    ]
    selected_id = (
        str(passing[0]["candidate_id"])
        if passing
        else "d14_z0_true_zero_base"
    )
    selected = next(value for value in candidates if value.candidate_id == selected_id)
    return selected_id, bool(passing and not selected.force_zero)


def _authority_pass(
    before_preopen: Mapping[str, Any],
    after_preopen: Mapping[str, Any],
    *,
    authority_evidence: Path | None,
    expected_authority_evidence_sha256: str | None,
    before_package_root_sha256: str,
    after_package_root_sha256: str,
    before_seal_sha256: str,
    after_seal_sha256: str,
) -> tuple[bool, dict[str, Any]]:
    preopen_pass = all(
        bool(value.get("formal_launch_authority"))
        and bool(value.get("formal_metric_claim_allowed"))
        and value.get("control_state") == "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS"
        for value in (before_preopen, after_preopen)
    )
    if authority_evidence is None:
        return False, {
            "status": "DIAGNOSTIC_SUPPORT_ONLY_NO_AUTHORITY_EVIDENCE",
            "preopen_formal_authority_pass": preopen_pass,
        }
    if expected_authority_evidence_sha256 is None:
        raise D14RunnerError("authority evidence requires external expected SHA")
    actual = _sha256_file(authority_evidence)
    evidence = json.loads(authority_evidence.read_text(encoding="utf-8"))
    binding_pass = (
        actual == expected_authority_evidence_sha256
        and evidence.get("status") == "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS"
        and bool(evidence.get("formal_launch_authority"))
        and evidence.get("before_package_root_sha256")
        == before_package_root_sha256
        and evidence.get("after_package_root_sha256") == after_package_root_sha256
        and evidence.get("before_seal_sha256") == before_seal_sha256
        and evidence.get("after_seal_sha256") == after_seal_sha256
    )
    return bool(preopen_pass and binding_pass), {
        "status": (
            "FORMAL_AUTHORITY_PASS"
            if preopen_pass and binding_pass
            else "FORMAL_AUTHORITY_FAIL_CLOSED"
        ),
        "preopen_formal_authority_pass": preopen_pass,
        "external_authority_binding_pass": binding_pass,
        "authority_evidence_sha256": actual,
        "expected_authority_evidence_sha256": expected_authority_evidence_sha256,
    }


def load_committed_state(
    root: Path,
    *,
    state_key: str,
    expected_commit_sha256: str,
    require_formal_promotion: bool = True,
) -> SparsePairwiseFisherState:
    """Load one deployable state through an externally pinned D14 COMMIT."""

    commit_path = root / "COMMIT.json"
    if (
        len(expected_commit_sha256) != 64
        or _sha256_file(commit_path) != expected_commit_sha256
    ):
        raise D14RunnerError("external D14 COMMIT hash mismatch")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if (
        bool(commit.get("query_opened", True))
        or state_key not in commit.get("state_sha256", {})
    ):
        raise D14RunnerError("D14 COMMIT state binding drift")
    if require_formal_promotion:
        if (
            commit.get("status")
            != "SUPPORT_ONLY_D14_FORMAL_SELECTED_NO_QUERY_OPEN"
            or not bool(commit.get("promotion_ready_for_single_query_candidate"))
            or not bool(commit.get("support_candidate_gate_pass_before_authority"))
            or not bool(commit.get("formal_launch_authority"))
        ):
            raise D14RunnerError("D14 COMMIT lacks formal promotion authority")
    else:
        if (
            bool(commit.get("promotion_ready_for_single_query_candidate", False))
            or bool(commit.get("formal_launch_authority", False))
            or "DIAGNOSTIC" not in str(commit.get("status", ""))
        ):
            raise D14RunnerError("D14 diagnostic COMMIT state drift")
    parts = state_key.split(":")
    if len(parts) != 3:
        raise D14RunnerError("D14 committed state key drift")
    scenario, phase, k_name = parts
    if k_name != "k10" or phase not in {"before", "after"}:
        raise D14RunnerError("D14 committed state key drift")
    stem = f"state_{scenario}_{phase}_k10"
    binding = commit["state_sha256"][state_key]
    state = load_sparse_pairwise_fisher_state(
        root / f"{stem}.npz",
        root / f"{stem}.json",
        expected_npz_sha256=str(binding["npz_sha256"]),
        expected_metadata_sha256=str(binding["metadata_sha256"]),
    )
    if (
        state.candidate_id != commit.get("selected_candidate_id")
        or state.hyperparameters.candidate_id != commit.get("selected_candidate_id")
        or state.sealed_runtime_sha256 != commit.get("sealed_runtime_sha256")
        or state.sealed_phase1_checkpoint_sha256
        != commit.get("sealed_phase1_checkpoint_sha256")
        or state.feature_code_sha256 != commit.get("combined_feature_code_sha256")
    ):
        raise D14RunnerError("D14 committed state metadata binding drift")
    return state


def _candidates() -> tuple[SparsePairwiseFisherHyperparameters, ...]:
    return (
        SparsePairwiseFisherHyperparameters(
            candidate_id="d14_z0_true_zero_base",
            ridge=0.05,
            gamma_old=0.0,
            gamma_new=0.0,
            select_band_old=0.0,
            band_old=0.0,
            band_new=0.0,
            max_old_edges=0,
            force_zero=True,
        ),
        SparsePairwiseFisherHyperparameters(
            candidate_id="d14_c1_balanced_light",
            ridge=0.05,
            gamma_old=0.025,
            gamma_new=0.025,
            select_band_old=0.20,
            band_old=0.10,
            band_new=0.10,
            max_old_edges=3,
        ),
        SparsePairwiseFisherHyperparameters(
            candidate_id="d14_c2_balanced",
            ridge=0.05,
            gamma_old=0.05,
            gamma_new=0.05,
            select_band_old=0.20,
            band_old=0.20,
            band_new=0.20,
            max_old_edges=3,
        ),
        SparsePairwiseFisherHyperparameters(
            candidate_id="d14_c3_floor_first",
            ridge=0.05,
            gamma_old=0.075,
            gamma_new=0.025,
            select_band_old=0.20,
            band_old=0.20,
            band_new=0.10,
            max_old_edges=3,
        ),
    )


def _candidate_lock(
    candidates: tuple[SparsePairwiseFisherHyperparameters, ...],
) -> dict[str, Any]:
    rows = [
        {
            "candidate_id": value.candidate_id,
            "operator_id": value.operator_id,
            "ridge": value.ridge,
            "gamma_old": value.gamma_old,
            "gamma_new": value.gamma_new,
            "band_select_old": value.select_band_old,
            "band_old": value.band_old,
            "band_new": value.band_new,
            "max_old_edges": value.max_old_edges,
            "force_zero": value.force_zero,
            "rival_old_quantile": RIVAL_OLD_QUANTILE,
            "rival_new_quantile": RIVAL_NEW_QUANTILE,
            "quantile_method": "linear",
        }
        for value in candidates
    ]
    return {
        "selection_scope": "one_base_operator_hyperparameter_arm_shared_by_all_three_scenarios",
        "operator_scope": "base_only_mvp_d10_auxiliary_operators_deferred",
        "candidates": rows,
        "lock_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }


def _measure_support_row_pareto(
    state: SparsePairwiseFisherState,
    probe_feature: np.ndarray,
    support_features: np.ndarray,
    *,
    repeats: int = 500,
) -> dict[str, Any]:
    support = _normalise(support_features)
    query = _normalise(probe_feature)
    for _ in range(20):
        _score_numpy(probe_feature, state)
        _ = query @ support.T
    start = time.perf_counter()
    for _ in range(repeats):
        _score_numpy(probe_feature, state)
    d14_ms = (time.perf_counter() - start) * 1000.0 / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        _ = query @ support.T
    qknn_ms = (time.perf_counter() - start) * 1000.0 / repeats
    d14_macs = int(
        state.resource["prototype_cosine_mac_per_sample"]
        + state.resource["fisher_edge_mac_per_sample_upper_bound"]
    )
    qknn_macs = int(len(support) * state.feature_dim)
    return {
        "benchmark_input": "one_support_row_resource_probe_no_query_open",
        "repeats": repeats,
        "d14_head_latency_ms": d14_ms,
        "identity_single_qknn_latency_ms": qknn_ms,
        "latency_delta_percent": 100.0 * (d14_ms / qknn_ms - 1.0),
        "d14_upper_bound_macs": d14_macs,
        "identity_single_qknn_exact_macs": qknn_macs,
        "mac_delta_percent": 100.0 * (d14_macs / qknn_macs - 1.0),
        "d14_array_state_bytes": int(
            state.resource["persistent_array_state_bytes"]
        ),
        "identity_single_qknn_state_bytes": int(support_features.nbytes),
        "backbone_forwards_per_physical_sample": 1,
        "fft_branches_per_physical_sample": 0,
    }


def _normalise(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def run(
    *,
    before_root: Path,
    before_seal: Path,
    expected_before_seal_sha256: str,
    after_root: Path,
    after_seal: Path,
    expected_after_seal_sha256: str,
    output: Path,
    device_name: str = "auto",
    mode: str = "development_select",
    authority_evidence: Path | None = None,
    expected_authority_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    if mode != "development_select":
        raise D14RunnerError(
            "this runner is development_select only; confirmation must apply one locked candidate"
        )
    if (
        len(expected_before_seal_sha256) != 64
        or len(expected_after_seal_sha256) != 64
    ):
        raise D14RunnerError("external expected enrollment seal SHA required")
    if output.exists():
        raise D14RunnerError("output path already exists")
    before_payloads, before_manifest, before_preopen = _load_enrollment(
        before_root,
        before_seal,
        registration_state="before",
        expected_seal_sha256=expected_before_seal_sha256,
    )
    after_payloads, after_manifest, after_preopen = _load_enrollment(
        after_root,
        after_seal,
        registration_state="after",
        expected_seal_sha256=expected_after_seal_sha256,
    )
    if (
        before_manifest["receiver"] != after_manifest["receiver"]
        or int(before_manifest["seed"]) != int(after_manifest["seed"])
        or before_manifest["feature_runtime_sha256"]
        != after_manifest["feature_runtime_sha256"]
        or before_manifest["phase1_checkpoint_sha256"]
        != after_manifest["phase1_checkpoint_sha256"]
    ):
        raise D14RunnerError("before/after package binding drift")
    before_handles = {
        str(row["class_handle"]) for row in before_manifest["registered_classes"]
    }
    after_handles = {
        str(row["class_handle"]) for row in after_manifest["registered_classes"]
    }
    if not before_handles < after_handles:
        raise D14RunnerError("new-class registration set drift")
    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root, _member(before_manifest, "feature_runtime"), device=device
    )
    output.mkdir(parents=True)
    module_path = CODE / "cvsrffi" / "stage2_sparse_pairwise_fisher_guard.py"
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
                "d14_module_sha256": module_sha256,
                "d14_runner_sha256": runner_sha256,
                "d12_artifact_provider_sha256": artifact_provider_sha256,
                "registered_feature_module_sha256": registered_feature_sha256,
                "operator_id": "base",
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
            raise D14RunnerError("physical support reused across LEO scenarios")
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
    trace: list[dict[str, Any]] = []
    candidate_rows = []
    evaluations: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        evaluations[candidate.candidate_id] = {}
        scenario_gates = {}
        old_floors = []
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
            gate = (
                not candidate.force_zero
                and bool(joint["before_old_per_class_non_degraded_vs_base"])
                and bool(joint["after_old_per_class_non_degraded_vs_before"])
                and bool(joint["after_old_per_class_non_degraded_vs_base"])
                and bool(joint["after_new_per_class_non_degraded_vs_base"])
                and bool(joint["old_score_columns_bitwise_equal_before_after"])
                and bool(joint["all_old_edges_endpoint_disjoint"])
                and int(joint["max_old_edge_count"]) <= 3
                and float(joint["old_forgetting"]) <= 1.0e-12
                and float(joint["after_old"]["min_class_accuracy"]) + 1.0e-12
                >= float(joint["before_old"]["min_class_accuracy"])
                and float(joint["after_new"]["min_class_accuracy"]) + 1.0e-12
                >= float(joint["base_after_new"]["min_class_accuracy"])
                and float(joint["h_old_new"]) + 1.0e-12
                >= float(joint["base_h_old_new"])
                and float(joint["joint_accuracy"]) + 1.0e-12
                >= float(joint["base_joint_accuracy"])
            )
            scenario_gates[scenario] = bool(gate)
            old_floors.append(float(joint["after_old"]["min_class_accuracy"]))
            new_floors.append(float(joint["after_new"]["min_class_accuracy"]))
            h_values.append(float(joint["h_old_new"]))
            joint_values.append(float(joint["joint_accuracy"]))
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "force_zero": candidate.force_zero,
                "all_scenario_gate_pass": all(scenario_gates.values()),
                "scenario_gate_pass": scenario_gates,
                "worst_scenario_old_floor": min(old_floors),
                "worst_scenario_new_floor": min(new_floors),
                "mean_h_old_new": float(np.mean(h_values)),
                "mean_joint_accuracy": float(np.mean(joint_values)),
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            row["all_scenario_gate_pass"],
            min(row["worst_scenario_old_floor"], row["worst_scenario_new_floor"]),
            row["mean_h_old_new"],
            row["mean_joint_accuracy"],
            row["candidate_id"],
        ),
        reverse=True,
    )
    selected_id, support_candidate_pass = _select_candidate(
        candidate_rows, candidates
    )
    selected_hp = next(
        value for value in candidates if value.candidate_id == selected_id
    )
    selected_row = next(
        row for row in candidate_rows if row["candidate_id"] == selected_id
    )
    before_actual_seal_sha256 = _sha256_file(before_seal)
    after_actual_seal_sha256 = _sha256_file(after_seal)
    formal_authority, authority_audit = _authority_pass(
        before_preopen,
        after_preopen,
        authority_evidence=authority_evidence,
        expected_authority_evidence_sha256=expected_authority_evidence_sha256,
        before_package_root_sha256=str(before_manifest["package_root_sha256"]),
        after_package_root_sha256=str(after_manifest["package_root_sha256"]),
        before_seal_sha256=before_actual_seal_sha256,
        after_seal_sha256=after_actual_seal_sha256,
    )
    promotion_ready = bool(support_candidate_pass and formal_authority)
    scenario_results = {}
    state_hashes = {}
    all_serialized_preferred = True
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
        before_serialized = _write_state(
            output,
            stem=f"state_{scenario}_before_k10",
            state=fitted.before_state,
        )
        after_serialized = _write_state(
            output,
            stem=f"state_{scenario}_after_k10",
            state=fitted.after_state,
        )
        all_serialized_preferred = all_serialized_preferred and bool(
            before_serialized["serialized_state_preferred_under_80kib"]
            and after_serialized["serialized_state_preferred_under_80kib"]
        )
        state_hashes[f"{scenario}:before:k10"] = before_serialized
        state_hashes[f"{scenario}:after:k10"] = after_serialized
        joint = evaluations[selected_id][scenario]
        scenario_results[scenario] = {
            "joint_leave_two_out": joint,
            "support_gate_pass": selected_row["scenario_gate_pass"][scenario],
            "resource_before": dict(fitted.before_state.resource),
            "resource_after": dict(fitted.after_state.resource),
            "pareto_vs_identity_single_qknn": _measure_support_row_pareto(
                fitted.after_state,
                context["after_artifact"].features[:1],
                context["after_artifact"].features,
            ),
            "before_feature_provenance": context["before_provenance"],
            "after_feature_provenance": context["after_provenance"],
            "old_prototype_pair_and_score_exact_freeze_locked": True,
            "measured": {
                "feature_extraction_seconds": context["feature_seconds"],
                "selected_full_fit_seconds": time.perf_counter() - fit_start,
            },
        }
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    status = (
        "SUPPORT_ONLY_D14_FORMAL_SELECTED_NO_QUERY_OPEN"
        if promotion_ready
        else (
            "SUPPORT_ONLY_D14_DIAGNOSTIC_SELECTED_NO_PROMOTION_NO_QUERY_OPEN"
            if support_candidate_pass
            else "SUPPORT_ONLY_D14_DIAGNOSTIC_NOT_SELECTED_NO_QUERY_OPEN"
        )
    )
    training_log_sha256 = _write_jsonl_new(output / "training_log.jsonl", trace)
    audit = {
        "schema": "cvs.phase2.d14_support_only_audit.v1",
        "diagnostic_only": True,
        "status": status,
        "claim_scope": "development_diagnostic_support_only_no_query_claim",
        "runner_mode": mode,
        "receiver": after_manifest["receiver"],
        "seed": int(after_manifest["seed"]),
        "k_shot": 10,
        "view_policy": "one_fixed_received_iq_base_view_no_new_channel_state",
        "base_only_mvp": True,
        "auxiliary_operator_status": "deferred_no_complete_provenance_or_global_winner",
        "unified_hyperparameter_selection": {
            "selected_candidate_id": selected_id,
            "same_candidate_all_scenarios": True,
            "hyperparameter_lock_sha256": hyperparameter_lock["lock_sha256"],
            "candidate_rows": candidate_rows,
            "true_zero_fallback_policy": (
                "if_all_positive_candidates_fail_save_base_empty_edges_gamma0"
            ),
            "true_zero_fallback_is_d14_improvement": False,
        },
        "hyperparameter_lock": hyperparameter_lock,
        "scenario_results": scenario_results,
        "promotion_ready_for_single_query_candidate": promotion_ready,
        "support_candidate_gate_pass_before_authority": support_candidate_pass,
        "formal_launch_authority": formal_authority,
        "authority_audit": authority_audit,
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
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "runtime_authorization": {
            "feature_extraction_mode": (
                "runner_internal_actual_iq_sha_physical_batch1_no_public_mapping"
            ),
            "sealed_runtime_sha256": runtime_sha256,
            "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
            "d14_module_sha256": module_sha256,
            "d14_runner_sha256": runner_sha256,
            "d12_artifact_provider_sha256": artifact_provider_sha256,
            "registered_feature_module_sha256": registered_feature_sha256,
            "combined_feature_code_sha256": feature_code_sha256,
        },
        "state_sha256": state_hashes,
        "all_serialized_states_under_256kib": True,
        "all_serialized_states_preferred_under_80kib": all_serialized_preferred,
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
        "before_seal_sha256": before_actual_seal_sha256,
        "after_seal_sha256": after_actual_seal_sha256,
        "expected_before_seal_sha256": expected_before_seal_sha256,
        "expected_after_seal_sha256": expected_after_seal_sha256,
        "preopen_audit": {"before": before_preopen, "after": after_preopen},
    }
    audit_sha256 = _write_json_new(output / "support_audit.json", audit)
    lines = [
        "# D14稀疏pairwise Fisher双阶段门支持集审计",
        "",
        f"状态：`{status}`。只打开strict K10 before/after enrollment-only包，未打开query、truth、prediction、score或scorer。",
        "",
        f"三场景统一选择`{selected_id}`；operator固定`base`；hyperparameter lock SHA为`{hyperparameter_lock['lock_sha256']}`。",
        "",
        "|场景|Before old/floor|After old/floor|alpha0 new/floor|D14 new/floor|joint/H|forgetting|old/new逐类门|old score锁|edge old/new|array state|门|",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        row = scenario_results[scenario]
        joint = row["joint_leave_two_out"]
        resource = row["resource_after"]
        lines.append(
            f"|`{scenario}`|"
            f"{joint['before_old']['overall_accuracy']:.4f}/"
            f"{joint['before_old']['min_class_accuracy']:.4f}|"
            f"{joint['after_old']['overall_accuracy']:.4f}/"
            f"{joint['after_old']['min_class_accuracy']:.4f}|"
            f"{joint['base_after_new']['overall_accuracy']:.4f}/"
            f"{joint['base_after_new']['min_class_accuracy']:.4f}|"
            f"{joint['after_new']['overall_accuracy']:.4f}/"
            f"{joint['after_new']['min_class_accuracy']:.4f}|"
            f"{joint['joint_accuracy']:.4f}/{joint['h_old_new']:.4f}|"
            f"{joint['old_forgetting']:.4f}|"
            f"{joint['after_old_per_class_non_degraded_vs_before']}/"
            f"{joint['after_new_per_class_non_degraded_vs_base']}|"
            f"{joint['old_score_columns_bitwise_equal_before_after']}|"
            f"{resource['old_edge_count']}/{resource['new_edge_count']}|"
            f"{resource['persistent_array_state_bytes']}|"
            f"{row['support_gate_pass']}|"
        )
    lines.extend(
        [
            "",
            "D14为0参数、0epoch、base-only闭式support方法。WL-IQ/FFT-EQ因缺少完整统一provenance和三场景global winner暂缓。K1/K5/K20不得从K10切片，必须等待各自独立exact-K package与K10锁定candidate rebuild。本轮不开放query。",
            "",
        ]
    )
    report_sha256 = _write_text_new(output / "report.md", "\n".join(lines))
    commit = {
        "schema": "cvs.phase2.d14_support_only_commit.v1",
        "diagnostic_only": True,
        "status": status,
        "support_audit_sha256": audit_sha256,
        "training_log_sha256": training_log_sha256,
        "report_sha256": report_sha256,
        "state_sha256": state_hashes,
        "promotion_ready_for_single_query_candidate": promotion_ready,
        "support_candidate_gate_pass_before_authority": support_candidate_pass,
        "formal_launch_authority": formal_authority,
        "query_opened": False,
        "selected_candidate_id": selected_id,
        "hyperparameter_lock_sha256": hyperparameter_lock["lock_sha256"],
        "all_serialized_states_under_256kib": True,
        "all_serialized_states_preferred_under_80kib": all_serialized_preferred,
        "sealed_runtime_sha256": runtime_sha256,
        "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
        "d14_module_sha256": module_sha256,
        "d14_runner_sha256": runner_sha256,
        "d12_artifact_provider_sha256": artifact_provider_sha256,
        "registered_feature_module_sha256": registered_feature_sha256,
        "combined_feature_code_sha256": feature_code_sha256,
        "runner_mode": mode,
        "expected_before_seal_sha256": expected_before_seal_sha256,
        "expected_after_seal_sha256": expected_after_seal_sha256,
    }
    commit_sha256 = _write_json_new(output / "COMMIT.json", commit)
    return {"commit_sha256": commit_sha256, **commit}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--before-seal-sha256", required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--after-seal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mode", choices=("development_select",), required=True
    )
    parser.add_argument("--authority-evidence", type=Path)
    parser.add_argument("--authority-evidence-sha256")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                before_root=args.before_root,
                before_seal=args.before_seal,
                expected_before_seal_sha256=args.before_seal_sha256,
                after_root=args.after_root,
                after_seal=args.after_seal,
                expected_after_seal_sha256=args.after_seal_sha256,
                output=args.output,
                device_name=args.device,
                mode=args.mode,
                authority_evidence=args.authority_evidence,
                expected_authority_evidence_sha256=(
                    args.authority_evidence_sha256
                ),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
