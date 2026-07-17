"""Run D18-CMRAE on sealed strict-K10 enrollment support only.

This executable deliberately has no query, truth, scorer, or formal-evaluation
input.  It first requires the signed path-free v2 no-IQ-open preflight,
then reuses the D14 package materializer, binds every fixed received-IQ row to
the package's overlay provenance, and keeps output claims support-only.
Candidate selection is the D18 three-scene atomic selector; a failed positive
route therefore lands the canonical true-zero state.
"""

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
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (CODE, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cvsrffi.somph_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA,
    finalize_somph_enrollment_authority_after_materialization,
    materialize_somph_enrollment_with_signed_authority,
)
from cvsrffi.stage2_cmrae import (  # noqa: E402
    MAX_ADAPTER_STATE_BYTES,
    MAX_FULL_SERIALIZED_STATE_BYTES,
    CmraeSceneSupport,
    CmraeState,
    _build_runtime_authorized_received_iq_artifact_internal,
    _features,
    _normalize,
    _prototype_scores,
    _seal_runtime_authorized_backbone_internal,
    fit_before_after_locked,
    load_state_bytes,
    preregistered_candidates,
    select_k10_candidate_three_scene,
    serialize_state_bytes,
)


SUPPORT_QUERY_DISJOINTNESS_STATUS = "SUPPORT_ONLY_NO_QUERY_CLAIM"
FORMAL_SUPPORT_ADAPTATION_SCOPE = "formal_support_adaptation_state_only"
from cvsrffi.stage2_predictor_runtime import load_torchscript_backbone_same_fd  # noqa: E402
from run_d14_support_only_pairwise_fisher_guard import (  # noqa: E402
    _base_feature,
    _canonical,
    _member,
    _payload_rows,
    _sha256_file,
    _write_json_new,
    _write_jsonl_new,
    _write_text_new,
)


class D18RunnerError(ValueError):
    """Raised when the D18 support-only runner fails closed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def _safe_verified_json_member(
    root: Path, manifest: Mapping[str, Any], *, kind: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    member = _member(manifest, kind)
    root_resolved = root.resolve(strict=True)
    path = (root / str(member["relative_path"])).resolve(strict=True)
    if (
        path.parent != root_resolved
        or path.is_symlink()
        or path.name != str(member["relative_path"])
    ):
        raise D18RunnerError(f"unsafe sealed member path: {kind}")
    with path.open("rb") as handle:
        raw = handle.read()
        descriptor_stat = os.fstat(handle.fileno())
    digest = hashlib.sha256(raw).hexdigest()
    if (
        digest != str(member["sha256"])
        or len(raw) != int(member["size_bytes"])
        or descriptor_stat.st_size != len(raw)
    ):
        raise D18RunnerError(f"sealed member hash/size drift: {kind}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise D18RunnerError(f"sealed JSON parse failed: {kind}") from error
    return value, {
        "kind": kind,
        "relative_path": path.name,
        "sha256": digest,
        "size_bytes": len(raw),
        "same_file_descriptor_hash_and_parse": True,
    }


def _overlay_index(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    value, audit = _safe_verified_json_member(
        root, manifest, kind="overlay_provenance"
    )
    if (
        value.get("schema") != "cvs.phase2.somph_overlay_provenance.v1"
        or value.get("receiver") != manifest.get("receiver")
        or int(value.get("seed", -1)) != int(manifest.get("seed", -2))
        or not isinstance(value.get("samples"), list)
    ):
        raise D18RunnerError("overlay provenance envelope drift")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    overlay_tokens: set[str] = set()
    for row in value["samples"]:
        token = str(row.get("sample_token", ""))
        parent = str(row.get("post_channel_iq_sha256", ""))
        scenario = str(row.get("scenario", ""))
        overlay = str(row.get("overlay_token", ""))
        key = (token, parent, scenario)
        if (
            key in result
            or scenario not in FORMAL_LEO_WEAK_SCENARIOS
            or len(parent) != 64
            or not overlay.startswith("oid_")
            or len(overlay) != 68
            or any(ch not in "0123456789abcdef" for ch in overlay[4:])
            or overlay in overlay_tokens
            or not isinstance(row.get("satellite_seed"), int)
        ):
            raise D18RunnerError("overlay provenance row drift")
        result[key] = dict(row)
        overlay_tokens.add(overlay)
    audit.update(
        {
            "sample_count": len(result),
            "unique_overlay_token_count": len(overlay_tokens),
            "row_schema_and_uniqueness_verified": True,
        }
    )
    return result, audit


def _rows_with_overlay(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    overlay: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    scenario: str,
) -> dict[str, np.ndarray]:
    # D14 performs the actual-IQ SHA, exact-K10, physical-ID and parent-SHA checks.
    rows = dict(_payload_rows(payload, manifest, scenario=scenario))
    handles = np.asarray(
        [str(row["class_handle"]) for row in manifest["registered_classes"]]
    )
    raw_labels = handles[
        np.asarray(payload["support_class_indices"], dtype=np.int64)
    ].astype(str)
    raw_ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    order = np.asarray(
        sorted(
            range(len(raw_labels)),
            key=lambda index: (raw_labels[index], int(raw_ranks[index])),
        ),
        dtype=np.int64,
    )
    if not {
        "support_overlay_tokens",
        "support_satellite_seeds",
    }.issubset(payload):
        raise D18RunnerError("sealed support overlay fields missing")
    payload_overlays = np.asarray(payload["support_overlay_tokens"]).astype(str)[order]
    payload_seeds = np.asarray(
        payload["support_satellite_seeds"], dtype=np.int64
    )[order]
    bound_source_provenance: list[str] = []
    bound_source_caches: list[str] = []
    bound_provenance_records: list[str] = []
    bound_seeds: list[int] = []
    for index, (token, parent) in enumerate(
        zip(rows["tokens"].tolist(), rows["hashes"].tolist())
    ):
        key = (str(token), str(parent), scenario)
        item = overlay.get(key)
        if item is None:
            raise D18RunnerError("support row absent from overlay provenance")
        overlay_token = str(item["overlay_token"])
        satellite_seed = int(item["satellite_seed"])
        if (
            overlay_token != str(payload_overlays[index])
            or satellite_seed != int(payload_seeds[index])
            or len(str(item.get("source_leo_provenance_sha256", ""))) != 64
            or len(str(item.get("source_leo_cache_sha256", ""))) != 64
        ):
            raise D18RunnerError("support NPZ/overlay provenance binding drift")
        bound_source_provenance.append(
            str(item["source_leo_provenance_sha256"])
        )
        bound_source_caches.append(str(item["source_leo_cache_sha256"]))
        bound_provenance_records.append(
            hashlib.sha256(_canonical(dict(item))).hexdigest()
        )
        bound_seeds.append(satellite_seed)
    rows["overlay_tokens"] = payload_overlays
    rows["source_leo_provenance_sha256"] = np.asarray(
        bound_source_provenance
    )
    rows["source_leo_cache_sha256"] = np.asarray(bound_source_caches)
    rows["overlay_provenance_record_sha256"] = np.asarray(
        bound_provenance_records
    )
    rows["satellite_seeds"] = np.asarray(bound_seeds, dtype=np.int64)
    return rows


def _manifest_binding(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    if (
        before.get("receiver") != after.get("receiver")
        or int(before.get("seed", -1)) != int(after.get("seed", -2))
        or int(before.get("k_shot", -1)) != 10
        or int(after.get("k_shot", -1)) != 10
        or before.get("feature_runtime_sha256")
        != after.get("feature_runtime_sha256")
        or before.get("phase1_checkpoint_sha256")
        != after.get("phase1_checkpoint_sha256")
    ):
        raise D18RunnerError("before/after package binding drift")
    old = {str(row["class_handle"]) for row in before["registered_classes"]}
    all_classes = {
        str(row["class_handle"]) for row in after["registered_classes"]
    }
    if not old < all_classes:
        raise D18RunnerError("real new-class registration set required")


def _require_post_materialization_authority(
    before_audit: Mapping[str, Any], after_audit: Mapping[str, Any]
) -> None:
    required_state = "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS"
    for registration_state, audit in (
        ("before", before_audit),
        ("after", after_audit),
    ):
        if (
            audit.get("iq_payload_materialized") is not True
            or audit.get("iq_archive_opened") is not True
            or audit.get("np_load_invoked") is not True
            or audit.get("formal_launch_authority") is not True
            or audit.get("formal_metric_claim_allowed") is not False
            or audit.get("support_query_disjointness_status")
            != SUPPORT_QUERY_DISJOINTNESS_STATUS
            or audit.get("signed_path_free_runtime_authorization_verified")
            is not True
            or audit.get("runtime_authorization_schema")
            != SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA
            or audit.get("phase2_clean_dataset_reachable") is not False
            or audit.get("phase2_clean_cache_reachable") is not False
            or audit.get("phase2_clean_control_flow_reachable") is not False
            or audit.get("status") != required_state
            or audit.get("control_state") != required_state
            or audit.get("phase2_protocol_evidence_status") != required_state
            or len(str(audit.get("post_materialization_audit_sha256", ""))) != 64
        ):
            raise D18RunnerError(
                "formal authority finalizer required before D18 selection: "
                f"{registration_state}"
            )


def _old_reuse(
    before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray]
) -> None:
    old_classes = set(np.asarray(before["labels"]).astype(str).tolist())

    def keyed(rows: Mapping[str, np.ndarray]) -> dict[tuple[str, int], tuple[Any, ...]]:
        return {
            (str(rows["labels"][i]), int(rows["ranks"][i])): (
                str(rows["tokens"][i]),
                str(rows["hashes"][i]),
                str(rows["overlay_tokens"][i]),
                int(rows["satellite_seeds"][i]),
            )
            for i in range(len(rows["labels"]))
            if str(rows["labels"][i]) in old_classes
        }

    if keyed(before) != keyed(after):
        raise D18RunnerError("before/after old support exact reuse drift")


def _cross_scene_disjointness(
    rows: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    scenarios = tuple(FORMAL_LEO_WEAK_SCENARIOS)
    for left_index, left in enumerate(scenarios):
        for right in scenarios[left_index + 1 :]:
            overlap = {
                "physical_sample_id": len(
                    set(rows[left]["tokens"].tolist())
                    & set(rows[right]["tokens"].tolist())
                ),
                "parent_received_iq_sha256": len(
                    set(rows[left]["hashes"].tolist())
                    & set(rows[right]["hashes"].tolist())
                ),
                "overlay_token": len(
                    set(rows[left]["overlay_tokens"].tolist())
                    & set(rows[right]["overlay_tokens"].tolist())
                ),
            }
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "overlap_count": overlap,
                    "pass": not any(overlap.values()),
                }
            )
    if not all(row["pass"] for row in pairs):
        raise D18RunnerError("cross-scene physical/parent/overlay reuse")
    return {"pairs": pairs, "all_pairwise_disjoint": True}


def _selection_records(rows: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "class_handle": str(rows["labels"][index]),
            "rank_within_class": int(rows["ranks"][index]),
            "physical_sample_id": str(rows["tokens"][index]),
            "parent_received_iq_sha256": str(rows["hashes"][index]),
            "overlay_token": str(rows["overlay_tokens"][index]),
            "satellite_seed": int(rows["satellite_seeds"][index]),
            "source_leo_cache_sha256": str(
                rows["source_leo_cache_sha256"][index]
            ),
            "source_leo_provenance_sha256": str(
                rows["source_leo_provenance_sha256"][index]
            ),
            "overlay_provenance_record_sha256": str(
                rows["overlay_provenance_record_sha256"][index]
            ),
        }
        for index in range(len(rows["labels"]))
    ]


def _build_k10_selection_authority_anchor(
    *,
    before_manifest: Mapping[str, Any],
    after_manifest: Mapping[str, Any],
    before_authority_audit: Mapping[str, Any],
    after_authority_audit: Mapping[str, Any],
    before_seal_sha256: str,
    after_seal_sha256: str,
    code_hashes: Mapping[str, str],
    rows_by_scenario: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, Any]:
    selections = {
        scenario: {
            state: _selection_records(rows_by_scenario[scenario][state])
            for state in ("before", "after")
        }
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    payload = {
        "schema": "cvs.phase2.d18_k10_selection_authority_anchor.v1",
        "before_package_root_sha256": before_manifest["package_root_sha256"],
        "after_package_root_sha256": after_manifest["package_root_sha256"],
        "before_detached_seal_sha256": before_seal_sha256,
        "after_detached_seal_sha256": after_seal_sha256,
        "before_manifest_sha256": before_authority_audit["manifest_sha256"],
        "after_manifest_sha256": after_authority_audit["manifest_sha256"],
        "before_authority_audit": _jsonable(before_authority_audit),
        "after_authority_audit": _jsonable(after_authority_audit),
        "before_authority_audit_sha256": hashlib.sha256(
            _canonical(before_authority_audit)
        ).hexdigest(),
        "after_authority_audit_sha256": hashlib.sha256(
            _canonical(after_authority_audit)
        ).hexdigest(),
        "sealed_runtime_sha256": before_manifest["feature_runtime_sha256"],
        "sealed_phase1_checkpoint_sha256": before_manifest[
            "phase1_checkpoint_sha256"
        ],
        "code_hashes": dict(code_hashes),
        "selection_records": selections,
        "formal_launch_authority": bool(
            before_authority_audit["formal_launch_authority"]
            and after_authority_audit["formal_launch_authority"]
        ),
        "formal_metric_claim_allowed": bool(
            before_authority_audit["formal_metric_claim_allowed"]
            and after_authority_audit["formal_metric_claim_allowed"]
        ),
        "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
        "query_opened": False,
        "performance_claim_allowed": False,
    }
    return {
        "payload": payload,
        "k10_selection_authority_anchor_sha256": hashlib.sha256(
            _canonical(payload)
        ).hexdigest(),
    }


def _artifact(
    rows: Mapping[str, np.ndarray],
    *,
    scenario: str,
    runtime_sha256: str,
    checkpoint_sha256: str,
    feature_code_sha256: str,
):
    return _build_runtime_authorized_received_iq_artifact_internal(
        np.asarray(rows["iq"], dtype=np.float32),
        physical_sample_ids=rows["tokens"].tolist(),
        parent_received_iq_sha256=rows["hashes"].tolist(),
        overlay_tokens=rows["overlay_tokens"].tolist(),
        source_leo_provenance_sha256=rows[
            "source_leo_provenance_sha256"
        ].tolist(),
        source_leo_cache_sha256=rows["source_leo_cache_sha256"].tolist(),
        target_channel_views=[scenario] * len(rows["iq"]),
        satellite_seeds=rows["satellite_seeds"].tolist(),
        overlay_provenance_sha256=rows[
            "overlay_provenance_record_sha256"
        ].tolist(),
        sealed_runtime_sha256=runtime_sha256,
        sealed_phase1_checkpoint_sha256=checkpoint_sha256,
        feature_code_sha256=feature_code_sha256,
        purpose="support",
    )


def _write_state_roundtrip(path: Path, state: CmraeState) -> dict[str, Any]:
    if path.exists():
        raise D18RunnerError("state output path already exists")
    path.mkdir(parents=True, exist_ok=False)
    payload, digest = serialize_state_bytes(state)
    state_path = path / "state.npz"
    with state_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(state_path)
    with state_path.open("rb") as handle:
        disk_payload = handle.read()
        disk_stat = os.fstat(handle.fileno())
    if (
        hashlib.sha256(disk_payload).hexdigest() != digest
        or len(disk_payload) != disk_stat.st_size
    ):
        raise D18RunnerError("state disk SHA/size readback drift")
    rebuilt = load_state_bytes(disk_payload, expected_sha256=digest)
    rng = np.random.default_rng(20260717)
    probe = rng.normal(size=(7, state.feature_dim)).astype(np.float32)
    original_scores = _prototype_scores(_normalize(probe), state.prototypes)
    rebuilt_scores = _prototype_scores(_normalize(probe), rebuilt.prototypes)
    if (
        rebuilt.state_content_sha256 != state.state_content_sha256
        or not np.array_equal(
            rebuilt.common_dct_coefficients, state.common_dct_coefficients
        )
        or not np.array_equal(rebuilt.prototypes, state.prototypes)
        or not np.array_equal(original_scores, rebuilt_scores)
    ):
        raise D18RunnerError("state semantic roundtrip drift")
    commit = {
        "schema": "cvs.phase2.d18_state_commit.v1",
        "state_npz_sha256": digest,
        "state_npz_bytes": len(payload),
        "state_content_sha256": state.state_content_sha256,
        "selection_authority_anchor_sha256": (
            state.selection_authority_anchor_sha256
        ),
        "candidate_id": state.hyperparameters.candidate_id,
        "registration_generation": state.registration_generation,
        "authority_scope": state.authority_scope,
        "artifact_scope": FORMAL_SUPPORT_ADAPTATION_SCOPE,
        "formal_launch_authority": True,
        "formal_support_adaptation_state": True,
        "formal_metric_claim_allowed": False,
        "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
        "query_opened": False,
        "performance_claim_allowed": False,
    }
    commit_sha = _write_json_new(path / "COMMIT", commit)
    if {value.name for value in path.iterdir()} != {"state.npz", "COMMIT"}:
        raise D18RunnerError("state directory member allowlist drift")
    with (path / "COMMIT").open("rb") as handle:
        disk_commit = handle.read()
        commit_stat = os.fstat(handle.fileno())
    if (
        hashlib.sha256(disk_commit).hexdigest() != commit_sha
        or len(disk_commit) != commit_stat.st_size
        or json.loads(disk_commit.decode("utf-8")) != commit
    ):
        raise D18RunnerError("state COMMIT disk readback drift")
    envelope_bytes = len(payload) + (path / "COMMIT").stat().st_size
    return {
        **commit,
        "commit_sha256": commit_sha,
        "actual_full_serialized_state_bytes": len(payload),
        "external_commit_envelope_bytes": envelope_bytes,
        "adapter_array_state_bytes": int(
            state.common_dct_coefficients.nbytes
        ),
        "actual_full_state_under_256kib": (
            len(payload) <= MAX_FULL_SERIALIZED_STATE_BYTES
        ),
        "adapter_state_under_16kib": (
            state.common_dct_coefficients.nbytes < MAX_ADAPTER_STATE_BYTES
        ),
        "semantic_roundtrip_verified": True,
        "disk_reopen_sha_and_allowlist_verified": True,
        "fixed_probe_score_bitwise_roundtrip_verified": True,
    }


def _support_inventory(
    rows: Mapping[str, np.ndarray],
    *,
    old_classes: set[str],
    scenario: str,
    state: CmraeState,
) -> list[dict[str, Any]]:
    inventory = [
        {
            "scenario": scenario,
            "class_handle": str(rows["labels"][index]),
            "lifecycle": (
                "target_old"
                if str(rows["labels"][index]) in old_classes
                else "target_new"
            ),
            "rank_within_class": int(rows["ranks"][index]),
            "physical_sample_id": str(rows["tokens"][index]),
            "parent_received_iq_sha256": str(rows["hashes"][index]),
            "overlay_token": str(rows["overlay_tokens"][index]),
            "satellite_seed": int(rows["satellite_seeds"][index]),
            "source_leo_cache_sha256": str(
                rows["source_leo_cache_sha256"][index]
            ),
            "source_leo_provenance_sha256": str(
                rows["source_leo_provenance_sha256"][index]
            ),
            "overlay_provenance_record_sha256": str(
                rows["overlay_provenance_record_sha256"][index]
            ),
            "operator_id": state.operator_id,
            "view_seed": 0,
            "view_seed_policy": "deterministic_fixed_zero_no_random_view",
            "post_reception_view_used": bool(
                not state.hyperparameters.force_zero
            ),
            "post_reception_view_count": 1,
            "post_reception_view_counts_as_additional_physical_sample": False,
            "additional_leo_channel_state_generation": False,
            "counts_as_one_physical_support": True,
        }
        for index in range(len(rows["labels"]))
    ]
    _validate_support_inventory(inventory, state=state, scenario=scenario)
    return inventory


def _validate_support_inventory(
    inventory: list[Mapping[str, Any]],
    *,
    state: CmraeState,
    scenario: str,
) -> None:
    physical_ids = [str(row.get("physical_sample_id", "")) for row in inventory]
    parents = [str(row.get("parent_received_iq_sha256", "")) for row in inventory]
    expected_used = bool(not state.hyperparameters.force_zero)
    if (
        not inventory
        or len(physical_ids) != len(set(physical_ids))
        or len(parents) != len(set(parents))
        or any(
            row.get("scenario") != scenario
            or len(str(row.get("parent_received_iq_sha256", ""))) != 64
            or len(str(row.get("source_leo_cache_sha256", ""))) != 64
            or len(str(row.get("source_leo_provenance_sha256", ""))) != 64
            or len(str(row.get("overlay_provenance_record_sha256", ""))) != 64
            or row.get("operator_id") != state.operator_id
            or row.get("view_seed") != 0
            or row.get("view_seed_policy")
            != "deterministic_fixed_zero_no_random_view"
            or row.get("post_reception_view_used") is not expected_used
            or row.get("post_reception_view_count") != 1
            or row.get(
                "post_reception_view_counts_as_additional_physical_sample"
            )
            is not False
            or row.get("additional_leo_channel_state_generation") is not False
            or row.get("counts_as_one_physical_support") is not True
            for row in inventory
        )
    ):
        raise D18RunnerError("support inventory fixed-view binding drift")


def _latency_pareto(
    state: CmraeState,
    artifact,
    backbone,
    *,
    repeats: int = 25,
) -> dict[str, Any]:
    base_support = _normalize(backbone.extract(artifact.received_iq))
    probe_iq = artifact.received_iq[:1]
    probe_kwargs = {
        "physical_sample_ids": artifact.physical_sample_ids[:1],
        "parent_received_iq_sha256": artifact.parent_received_iq_sha256[:1],
        "overlay_tokens": artifact.overlay_tokens[:1],
        "source_leo_provenance_sha256": (
            artifact.source_leo_provenance_sha256[:1]
        ),
        "source_leo_cache_sha256": artifact.source_leo_cache_sha256[:1],
        "target_channel_views": artifact.target_channel_views[:1],
        "satellite_seeds": artifact.satellite_seeds[:1],
        "overlay_provenance_sha256": artifact.overlay_provenance_sha256[:1],
        "sealed_runtime_sha256": artifact.sealed_runtime_sha256,
        "sealed_phase1_checkpoint_sha256": (
            artifact.sealed_phase1_checkpoint_sha256
        ),
        "feature_code_sha256": artifact.feature_code_sha256,
        "purpose": "support",
    }
    probe = _build_runtime_authorized_received_iq_artifact_internal(
        probe_iq, **probe_kwargs
    )

    def cmrae_once() -> None:
        feature = _features(
            probe, state.common_dct_coefficients, state.hyperparameters, backbone
        )
        _prototype_scores(feature, state.prototypes)

    def qknn_once() -> None:
        feature = _normalize(backbone.extract(probe_iq))
        np.einsum("bd,kd->bk", feature, base_support, optimize=False)

    cmrae_once()
    qknn_once()
    cmrae_elapsed = []
    qknn_elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        cmrae_once()
        cmrae_elapsed.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        qknn_once()
        qknn_elapsed.append(time.perf_counter_ns() - start)
    cmrae_macs = int(
        state.resource["prototype_mac_per_query"]
        + state.resource["dct_reconstruction_macs_per_query"]
        + state.resource["row_rms_restore_macs_per_query"]
    )
    qknn_macs = int(len(base_support) * state.feature_dim)
    payload, _ = serialize_state_bytes(state)
    return {
        "probe": "one_registered_support_observation_resource_only_no_accuracy_claim",
        "repeats": repeats,
        "physical_batch_size": 1,
        "cmrae_end_to_end_mean_ms": float(np.mean(cmrae_elapsed) / 1e6),
        "cmrae_end_to_end_median_ms": float(np.median(cmrae_elapsed) / 1e6),
        "cmrae_end_to_end_p95_ms": float(np.percentile(cmrae_elapsed, 95) / 1e6),
        "identity_single_qknn_end_to_end_mean_ms": float(
            np.mean(qknn_elapsed) / 1e6
        ),
        "identity_single_qknn_end_to_end_median_ms": float(
            np.median(qknn_elapsed) / 1e6
        ),
        "identity_single_qknn_end_to_end_p95_ms": float(
            np.percentile(qknn_elapsed, 95) / 1e6
        ),
        "cmrae_non_fft_macs_per_sample": cmrae_macs,
        "identity_single_qknn_macs_per_sample": qknn_macs,
        "cmrae_estimated_fft_complex_ops_per_sample": int(
            state.resource["estimated_fft_complex_ops_per_query"]
        ),
        "cmrae_actual_serialized_state_bytes": len(payload),
        "identity_single_qknn_numeric_state_bytes_fp32": int(
            base_support.nbytes
        ),
        "state_comparison_scope": (
            "d18_full_serialized_payload_vs_identity_qknn_fp32_numeric_"
            "support_matrix_full_serializer_unavailable"
        ),
        "identity_single_qknn_full_serialized_state_bytes": None,
        "state_bytes_direct_pareto_claim_allowed": False,
        "backbone_forwards_per_timed_sample": 1,
        "additional_leo_state_generated": False,
        "repeated_calls_are_deterministic_benchmark_reexecution_not_extra_k": True,
    }


def run(
    *,
    before_root: Path,
    before_seal: Path,
    expected_before_seal_sha256: str,
    before_formal_policy: Path,
    before_formal_policy_authorization: Path,
    before_signed_policy_authorization_envelope: Path,
    expected_before_signed_policy_authorization_envelope_sha256: str,
    after_root: Path,
    after_seal: Path,
    expected_after_seal_sha256: str,
    after_formal_policy: Path,
    after_formal_policy_authorization: Path,
    after_signed_policy_authorization_envelope: Path,
    expected_after_signed_policy_authorization_envelope_sha256: str,
    output: Path,
    device_name: str = "auto",
    mode: str = "development_select",
) -> dict[str, Any]:
    if mode != "development_select":
        raise D18RunnerError("D18 runner is development_select only")
    if (
        len(expected_before_seal_sha256) != 64
        or len(expected_after_seal_sha256) != 64
    ):
        raise D18RunnerError("external detached seal SHA required")
    if output.exists():
        raise D18RunnerError("output path already exists")

    # Each atomic entry performs the pinned signature preflight internally and
    # only then opens/materializes IQ.  The runner has no audit/capability
    # handoff surface and cannot substitute a synthetic verifier.
    before_evidence = materialize_somph_enrollment_with_signed_authority(
        before_root,
        detached_seal_path=before_seal,
        expected_seal_sha256=expected_before_seal_sha256,
        formal_policy_path=before_formal_policy,
        formal_policy_authorization_path=before_formal_policy_authorization,
        signed_policy_authorization_envelope_path=(
            before_signed_policy_authorization_envelope
        ),
        expected_signed_policy_authorization_envelope_sha256=(
            expected_before_signed_policy_authorization_envelope_sha256
        ),
    )
    after_evidence = materialize_somph_enrollment_with_signed_authority(
        after_root,
        detached_seal_path=after_seal,
        expected_seal_sha256=expected_after_seal_sha256,
        formal_policy_path=after_formal_policy,
        formal_policy_authorization_path=after_formal_policy_authorization,
        signed_policy_authorization_envelope_path=(
            after_signed_policy_authorization_envelope
        ),
        expected_signed_policy_authorization_envelope_sha256=(
            expected_after_signed_policy_authorization_envelope_sha256
        ),
    )
    before_authority_final = (
        finalize_somph_enrollment_authority_after_materialization(
            before_evidence
        )
    )
    after_authority_final = (
        finalize_somph_enrollment_authority_after_materialization(
            after_evidence
        )
    )
    before_payloads = before_evidence.materialized_payloads
    after_payloads = after_evidence.materialized_payloads
    before_manifest = before_evidence.manifest
    after_manifest = after_evidence.manifest
    _require_post_materialization_authority(
        before_authority_final, after_authority_final
    )
    _manifest_binding(before_manifest, after_manifest)
    before_overlay, before_overlay_audit = _overlay_index(
        before_root, before_manifest
    )
    after_overlay, after_overlay_audit = _overlay_index(after_root, after_manifest)

    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root, _member(before_manifest, "feature_runtime"), device=device
    )
    module_path = CODE / "cvsrffi" / "stage2_cmrae.py"
    runner_path = Path(__file__).resolve()
    d14_path = CODE / "scripts" / "run_d14_support_only_pairwise_fisher_guard.py"
    feature_path = CODE / "cvsrffi" / "stage2_diag_cosine_exploration.py"
    code_hashes = {
        "d18_module_sha256": _sha256_file(module_path),
        "d18_runner_sha256": _sha256_file(runner_path),
        "reused_d14_support_helpers_sha256": _sha256_file(d14_path),
        "registered_feature_module_sha256": _sha256_file(feature_path),
    }
    feature_code_sha256 = hashlib.sha256(
        _canonical(
            {
                **code_hashes,
                "operator": "cmrae_dct8_fixed_received_iq",
                "feature_path": "d14_actual_iq_physical_batch1",
            }
        )
    ).hexdigest()
    runtime_sha256 = str(before_manifest["feature_runtime_sha256"])
    checkpoint_sha256 = str(before_manifest["phase1_checkpoint_sha256"])
    backbone_forward_counter = {"count": 0}

    def extract_single(iq: np.ndarray) -> np.ndarray:
        if len(iq) != 1:
            raise D18RunnerError("physical-batch-one backbone required")
        backbone_forward_counter["count"] += 1
        return _base_feature(model, device, iq)

    backbone = _seal_runtime_authorized_backbone_internal(
        extract_single,
        feature_code_sha256=feature_code_sha256,
        sealed_phase1_checkpoint_sha256=checkpoint_sha256,
    )

    output.mkdir(parents=True, exist_ok=False)
    tracemalloc.start()
    run_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scenes: list[CmraeSceneSupport] = []
    contexts: dict[str, dict[str, Any]] = {}
    after_rows_all: dict[str, Mapping[str, np.ndarray]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = _rows_with_overlay(
            before_payloads[scenario],
            before_manifest,
            before_overlay,
            scenario=scenario,
        )
        after_rows = _rows_with_overlay(
            after_payloads[scenario],
            after_manifest,
            after_overlay,
            scenario=scenario,
        )
        _old_reuse(before_rows, after_rows)
        before_artifact = _artifact(
            before_rows,
            scenario=scenario,
            runtime_sha256=runtime_sha256,
            checkpoint_sha256=checkpoint_sha256,
            feature_code_sha256=feature_code_sha256,
        )
        after_artifact = _artifact(
            after_rows,
            scenario=scenario,
            runtime_sha256=runtime_sha256,
            checkpoint_sha256=checkpoint_sha256,
            feature_code_sha256=feature_code_sha256,
        )
        scene = CmraeSceneSupport(
            scenario,
            before_artifact,
            before_rows["labels"],
            before_rows["ranks"],
            after_artifact,
            after_rows["labels"],
            after_rows["ranks"],
        )
        scenes.append(scene)
        contexts[scenario] = {
            "before_rows": before_rows,
            "after_rows": after_rows,
            "before_artifact": before_artifact,
            "after_artifact": after_artifact,
        }
        after_rows_all[scenario] = after_rows
    disjointness = _cross_scene_disjointness(after_rows_all)
    anchor = _build_k10_selection_authority_anchor(
        before_manifest=before_manifest,
        after_manifest=after_manifest,
        before_authority_audit=before_authority_final,
        after_authority_audit=after_authority_final,
        before_seal_sha256=_sha256_file(before_seal),
        after_seal_sha256=_sha256_file(after_seal),
        code_hashes={
            **code_hashes,
            "sealed_runtime_sha256": runtime_sha256,
            "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
            "combined_feature_code_sha256": feature_code_sha256,
        },
        rows_by_scenario={
            scenario: {
                "before": contexts[scenario]["before_rows"],
                "after": contexts[scenario]["after_rows"],
            }
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
    )
    authority_anchor_sha256 = anchor[
        "k10_selection_authority_anchor_sha256"
    ]
    selection_forward_start = backbone_forward_counter["count"]
    selection = select_k10_candidate_three_scene(
        scenes,
        backbone=backbone,
        selection_authority_anchor_sha256=authority_anchor_sha256,
    )
    selection_forward_count = (
        backbone_forward_counter["count"] - selection_forward_start
    )
    selected_id = selection.selected_hyperparameters.candidate_id
    positive_selected = not selection.selected_hyperparameters.force_zero
    selected_eval = next(
        row for row in selection.evaluations if row["candidate_id"] == selected_id
    )
    deployment_forward_counter = {"count": 0}

    def deployment_extract_single(iq: np.ndarray) -> np.ndarray:
        if len(iq) != 1:
            raise D18RunnerError("physical-batch-one backbone required")
        deployment_forward_counter["count"] += 1
        return _base_feature(model, device, iq)

    deployment_backbone = _seal_runtime_authorized_backbone_internal(
        deployment_extract_single,
        feature_code_sha256=feature_code_sha256,
        sealed_phase1_checkpoint_sha256=checkpoint_sha256,
    )
    deployment_fits = []
    deployment_fit_resources = {}
    for scene, selected_fit in zip(scenes, selection.fitted_scenes):
        start_count = deployment_forward_counter["count"]
        measured_fit = fit_before_after_locked(
            scene.before_artifact,
            scene.before_labels,
            scene.before_ranks,
            scene.after_artifact,
            scene.after_labels,
            scene.after_ranks,
            k_shot=10,
            hyperparameters=selection.selected_hyperparameters,
            backbone=deployment_backbone,
            k10_lock_certificate=selection.k10_lock_certificate,
            expected_selection_authority_anchor_sha256=(
                authority_anchor_sha256
            ),
        )
        measured_count = deployment_forward_counter["count"] - start_count
        unique_support_count = len(scene.after_artifact.physical_sample_ids)
        if measured_count != unique_support_count:
            raise D18RunnerError(
                "selected full deployment fit did not use one backbone forward "
                "per unique physical support"
            )
        if (
            measured_fit.before_state.state_content_sha256
            != selected_fit.before_state.state_content_sha256
            or measured_fit.after_state.state_content_sha256
            != selected_fit.after_state.state_content_sha256
        ):
            raise D18RunnerError("selected full deployment fit state reexecution drift")
        deployment_fits.append(measured_fit)
        deployment_fit_resources[scene.scene_id] = {
            "measured_backbone_forward_count": measured_count,
            "unique_physical_support_count": unique_support_count,
            "one_forward_per_unique_physical_support_verified": True,
            "measurement_reexecution_counts_as_additional_k": False,
            "state_sha_reexecution_verified": True,
        }
    fit_by_scene = {
        scene.scene_id: fit for scene, fit in zip(scenes, deployment_fits)
    }
    eval_by_scene = {
        row["scene_id"]: row for row in selected_eval["scene_results"]
    }
    trace = [
        {
            "phase": "sealed_package_preopen",
            "authority_final_before": before_authority_final,
            "authority_final_after": after_authority_final,
            "materialization_evidence_before_sha256": (
                before_evidence.evidence_sha256
            ),
            "materialization_evidence_after_sha256": (
                after_evidence.evidence_sha256
            ),
            "before_package_structure": {
                "manifest": before_manifest,
                "detached_seal_sha256": expected_before_seal_sha256,
            },
            "after_package_structure": {
                "manifest": after_manifest,
                "detached_seal_sha256": expected_after_seal_sha256,
            },
            "overlay_before": before_overlay_audit,
            "overlay_after": after_overlay_audit,
        },
        {
            "phase": "candidate_lock",
            "candidates": [
                {
                    "candidate_id": hp.candidate_id,
                    "lambda_equalizer": hp.lambda_equalizer,
                    "dct_rank": hp.dct_rank,
                    "tau": hp.tau,
                    "force_zero": hp.force_zero,
                }
                for hp in preregistered_candidates()
            ],
        },
        *selection.trace,
        {
            "phase": "selection_resource",
            "physical_batch1_backbone_forward_calls": selection_forward_count,
            "includes_three_candidates_five_outer_folds_and_selected_full_fit": True,
        },
        {
            "phase": "selected_full_deployment_fit_resource",
            "separate_from_development_outer_selection": True,
            "scenarios": deployment_fit_resources,
            "total_measured_backbone_forward_count": int(
                deployment_forward_counter["count"]
            ),
        },
    ]
    old_classes = {
        str(row["class_handle"])
        for row in before_manifest["registered_classes"]
    }
    scenario_results: dict[str, Any] = {}
    inventories: list[dict[str, Any]] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        fit = fit_by_scene[scenario]
        trace.extend(
            {"scene_id": scenario, **row}
            for row in fit.trace
        )
        context = contexts[scenario]
        state_serialization = {
            "before": _write_state_roundtrip(
                output / "states" / scenario / "before", fit.before_state
            ),
            "after": _write_state_roundtrip(
                output / "states" / scenario / "after", fit.after_state
            ),
        }
        pareto = _latency_pareto(
            fit.after_state,
            context["after_artifact"],
            backbone,
        )
        inventory = _support_inventory(
            context["after_rows"],
            old_classes=old_classes,
            scenario=scenario,
            state=fit.after_state,
        )
        inventories.extend(inventory)
        scenario_results[scenario] = {
            "support_outer_l2o": eval_by_scene[scenario],
            "before_state_content_sha256": fit.before_state.state_content_sha256,
            "after_state_content_sha256": fit.after_state.state_content_sha256,
            "equalizer_and_old_prototype_lock": {
                "equalizer_bitwise": bool(
                    np.array_equal(
                        fit.before_state.common_dct_coefficients,
                        fit.after_state.common_dct_coefficients,
                    )
                ),
                "old_prototypes_bitwise": bool(
                    np.array_equal(
                        fit.before_state.prototypes,
                        fit.after_state.prototypes[
                            : fit.after_state.old_class_count
                        ],
                    )
                ),
            },
            "state_serialization": state_serialization,
            "pareto_vs_identity_single_qknn": pareto,
            "support_inventory_count": len(inventory),
        }
    inventory_sha256 = _write_json_new(
        output / "support_inventory.json", inventories
    )
    trace.append(
        {
            "phase": "selected_state_and_resource_audit",
            "selected_candidate_id": selected_id,
            "scenario_results": scenario_results,
        }
    )
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    training_log_sha256 = _write_jsonl_new(
        output / "training_log.jsonl", _jsonable(trace)
    )
    status = (
        "FORMAL_SUPPORT_ADAPTATION_D18_SELECTED_NO_QUERY_NO_METRIC"
        if positive_selected
        else "FORMAL_SUPPORT_ADAPTATION_D18_TRUE_Z0_NO_QUERY_NO_METRIC"
    )
    lock_certificate = selection.k10_lock_certificate
    audit = _jsonable(
        {
            "schema": "cvs.phase2.d18_support_only_audit.v1",
            "status": status,
            "claim_scope": FORMAL_SUPPORT_ADAPTATION_SCOPE,
            "authority": FORMAL_SUPPORT_ADAPTATION_SCOPE,
            "formal_launch_authority": True,
            "formal_support_adaptation_state": True,
            "formal_metric_claim_allowed": False,
            "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
            "query_opened": False,
            "performance_claim_allowed": False,
            "promotion_ready_for_query": False,
            "receiver": after_manifest["receiver"],
            "seed": int(after_manifest["seed"]),
            "k_shot": 10,
            "selected_candidate_id": selected_id,
            "support_candidate_gate_pass": positive_selected,
            "k10_lock_certificate": {
                "schema": lock_certificate.schema,
                "selected_candidate_id": lock_certificate.selected_candidate_id,
                "scene_prefix_locks": lock_certificate.scene_prefix_locks,
                "k10_selection_authority_sha256": (
                    lock_certificate.k10_selection_authority_sha256
                ),
                "authority_scope": lock_certificate.authority_scope,
                "certificate_sha256": lock_certificate.certificate_sha256,
                "low_k_certificate_discloses_no_row_ids_or_parent_hashes": True,
            },
            "k10_selection_authority_anchor": anchor,
            "candidate_evaluations": selection.evaluations,
            "scenario_results": scenario_results,
            "cross_scenario_support_disjointness": disjointness,
            "support_inventory_sha256": inventory_sha256,
            "support_inventory_rows": len(inventories),
            "runtime_authorization": {
                "loader": (
                    "preflight_bound_token_sealed_verified_materializer"
                ),
                "received_iq_binding": (
                    "authority_materializer_same_fd_actual_iq_sha_plus_"
                    "d14_payload_rows_recompute"
                ),
                "overlay_binding": "same_fd_hash_verified_overlay_member_then_sample_token_parent_scenario_crosscheck",
                "physical_batch_size": 1,
                "sealed_runtime_sha256": runtime_sha256,
                "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
                "feature_code_sha256": feature_code_sha256,
                **code_hashes,
            },
            "preopen_audit": {
                "authority_final_before": before_authority_final,
                "authority_final_after": after_authority_final,
                "materialization_evidence_before_sha256": (
                    before_evidence.evidence_sha256
                ),
                "materialization_evidence_after_sha256": (
                    after_evidence.evidence_sha256
                ),
                "before_package_structure": {
                    "manifest": before_manifest,
                    "detached_seal_sha256": expected_before_seal_sha256,
                },
                "after_package_structure": {
                    "manifest": after_manifest,
                    "detached_seal_sha256": expected_after_seal_sha256,
                },
                "before_overlay": before_overlay_audit,
                "after_overlay": after_overlay_audit,
            },
            "runtime_access_audit": {
                "opened_profiles": ["before:enrollment_only", "after:enrollment_only"],
                "query_package_opened": False,
                "query_truth_opened": False,
                "scorer_opened": False,
                "clean_sample_access": False,
                "clean_derived_signal_access": False,
                "source_sample_access": False,
                "source_derived_signal_access": False,
                "additional_leo_channel_state_generation": False,
                "post_reception_view_counts_as_additional_k": False,
                "total_development_selection_backbone_forward_calls": (
                    selection_forward_count
                ),
                "selected_full_deployment_fit": deployment_fit_resources,
                "selected_full_deployment_fit_backbone_forward_calls": int(
                    deployment_forward_counter["count"]
                ),
            },
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "phase2_physical_sample_observation_policy": "single_leo_weak_observation_per_physical_sample",
            "phase2_cross_scenario_physical_sample_reuse": False,
            "phase2_additional_leo_channel_state_generation": False,
            "phase2_post_reception_view_from_fixed_received_iq_only": True,
            "phase2_post_reception_view_counts_as_additional_physical_sample": False,
            "phase2_query_decision_policy": "per_sample_all_registered_classes",
            "phase2_query_role_oracle_access": False,
            "phase2_query_true_batch_class_count_access": False,
            "phase2_query_class_quota_access": False,
            "phase2_query_batch_global_assignment": False,
            "phase2_clean_dataset_reachable": False,
            "phase2_clean_cache_reachable": False,
            "phase2_clean_control_flow_reachable": False,
            "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
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
            "expected_before_seal_sha256": expected_before_seal_sha256,
            "expected_after_seal_sha256": expected_after_seal_sha256,
            "before_authority_commit_sha256": before_authority_final[
                "authority_commit_sha256"
            ],
            "after_authority_commit_sha256": after_authority_final[
                "authority_commit_sha256"
            ],
            "expected_before_signed_policy_authorization_envelope_sha256": (
                expected_before_signed_policy_authorization_envelope_sha256
            ),
            "expected_after_signed_policy_authorization_envelope_sha256": (
                expected_after_signed_policy_authorization_envelope_sha256
            ),
        }
    )
    audit_sha256 = _write_json_new(output / "support_audit.json", audit)
    lines = [
        "# D18-CMRAE strict-K10 support-only开发审计",
        "",
        f"状态：`{status}`。产物只具有formal support adaptation state权限；未打开query、truth或scorer，不允许formal metric或performance claim。",
        "",
        f"三场景统一candidate：`{selected_id}`；K10 lock certificate SHA：`{lock_certificate.certificate_sha256}`。",
        "",
        "|场景|Before old/floor|After old/floor|seen-new/floor|joint/H|forgetting|实际After state bytes|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        value = scenario_results[scenario]
        result = value["support_outer_l2o"]
        before_pc = result["aggregate_before_old_per_class"]
        after_pc = result["aggregate_after_old_per_class"]
        new_pc = result["aggregate_seen_new_per_class"]
        before = float(np.mean(list(before_pc.values())))
        after = float(np.mean(list(after_pc.values())))
        new = float(np.mean(list(new_pc.values())))
        harmonic = 2 * after * new / max(after + new, 1e-8)
        lines.append(
            f"|`{scenario}`|{before:.4f}/{min(before_pc.values()):.4f}|"
            f"{after:.4f}/{min(after_pc.values()):.4f}|"
            f"{new:.4f}/{min(new_pc.values()):.4f}|"
            f"{0.5 * (after + new):.4f}/{harmonic:.4f}|"
            f"{before - after:.4f}|"
            f"{value['state_serialization']['after']['actual_full_serialized_state_bytes']}|"
        )
    lines.extend(
        [
            "",
            "|场景|D18/qKNN MAC|D18 FFT complex ops|D18 mean/median/P95 ms|qKNN mean/median/P95 ms|D18 full serialized bytes|qKNN FP32 numeric bytes|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        pareto = scenario_results[scenario]["pareto_vs_identity_single_qknn"]
        lines.append(
            f"|`{scenario}`|"
            f"{pareto['cmrae_non_fft_macs_per_sample']}/"
            f"{pareto['identity_single_qknn_macs_per_sample']}|"
            f"{pareto['cmrae_estimated_fft_complex_ops_per_sample']}|"
            f"{pareto['cmrae_end_to_end_mean_ms']:.4f}/"
            f"{pareto['cmrae_end_to_end_median_ms']:.4f}/"
            f"{pareto['cmrae_end_to_end_p95_ms']:.4f}|"
            f"{pareto['identity_single_qknn_end_to_end_mean_ms']:.4f}/"
            f"{pareto['identity_single_qknn_end_to_end_median_ms']:.4f}/"
            f"{pareto['identity_single_qknn_end_to_end_p95_ms']:.4f}|"
            f"{pareto['cmrae_actual_serialized_state_bytes']}|"
            f"{pareto['identity_single_qknn_numeric_state_bytes_fp32']}|"
        )
    measured = audit["measured_run_resource"]
    lines.extend(
        [
            "",
            f"运行峰值：CUDA allocated={measured['peak_cuda_allocated_bytes']}B，CUDA reserved={measured['peak_cuda_reserved_bytes']}B，Python tracemalloc={measured['peak_python_tracemalloc_bytes']}B，wall={measured['wall_seconds']:.3f}s。",
            "",
            "状态口径不是直接Pareto：D18列为包含metadata的完整压缩serialized payload；qKNN列仅为FP32 support feature numeric matrix，当前没有matched qKNN完整serializer，因此禁止据两列直接声称状态Pareto。",
            "",
            "这些数值仅为注册support上的outer leave-two-out开发选择证据，不是query准确率或正式确认结果。任一scene/fold/class/floor/H/forgetting门失败时由核心选择器原子回退true Z0。",
            "",
            "CMRAE只变换已经密封的固定LEO_weak received IQ；每个view仍计为原来的一个physical support，不生成第二种LEO状态。",
            "",
        ]
    )
    report_sha256 = _write_text_new(output / "report.md", "\n".join(lines))
    receipt = {
        "schema": "cvs.phase2.d18_support_only_receipt.v1",
        "status": status,
        "authority": FORMAL_SUPPORT_ADAPTATION_SCOPE,
        "formal_launch_authority": True,
        "formal_support_adaptation_state": True,
        "formal_metric_claim_allowed": False,
        "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
        "performance_claim_allowed": False,
        "promotion_ready_for_query": False,
        "selected_candidate_id": selected_id,
        "support_candidate_gate_pass": positive_selected,
        "k10_lock_certificate_sha256": lock_certificate.certificate_sha256,
        "k10_selection_authority_anchor_sha256": authority_anchor_sha256,
        "support_audit_sha256": audit_sha256,
        "training_log_sha256": training_log_sha256,
        "support_inventory_sha256": inventory_sha256,
        "report_sha256": report_sha256,
        "query_opened": False,
        "expected_before_seal_sha256": expected_before_seal_sha256,
        "expected_after_seal_sha256": expected_after_seal_sha256,
        "before_authority_commit_sha256": (
            before_authority_final["authority_commit_sha256"]
        ),
        "after_authority_commit_sha256": (
            after_authority_final["authority_commit_sha256"]
        ),
        **code_hashes,
    }
    receipt_sha256 = _write_json_new(output / "RECEIPT.json", receipt)
    return {"receipt_sha256": receipt_sha256, **receipt}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--before-seal-sha256", required=True)
    parser.add_argument("--before-formal-policy", type=Path, required=True)
    parser.add_argument(
        "--before-formal-policy-authorization", type=Path, required=True
    )
    parser.add_argument(
        "--before-signed-policy-authorization-envelope", type=Path, required=True
    )
    parser.add_argument(
        "--before-signed-policy-authorization-envelope-sha256", required=True
    )
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--after-seal-sha256", required=True)
    parser.add_argument("--after-formal-policy", type=Path, required=True)
    parser.add_argument(
        "--after-formal-policy-authorization", type=Path, required=True
    )
    parser.add_argument(
        "--after-signed-policy-authorization-envelope", type=Path, required=True
    )
    parser.add_argument(
        "--after-signed-policy-authorization-envelope-sha256", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("development_select",), required=True)
    args = parser.parse_args()
    result = run(
        before_root=args.before_root,
        before_seal=args.before_seal,
        expected_before_seal_sha256=args.before_seal_sha256,
        before_formal_policy=args.before_formal_policy,
        before_formal_policy_authorization=(
            args.before_formal_policy_authorization
        ),
        before_signed_policy_authorization_envelope=(
            args.before_signed_policy_authorization_envelope
        ),
        expected_before_signed_policy_authorization_envelope_sha256=(
            args.before_signed_policy_authorization_envelope_sha256
        ),
        after_root=args.after_root,
        after_seal=args.after_seal,
        expected_after_seal_sha256=args.after_seal_sha256,
        after_formal_policy=args.after_formal_policy,
        after_formal_policy_authorization=(
            args.after_formal_policy_authorization
        ),
        after_signed_policy_authorization_envelope=(
            args.after_signed_policy_authorization_envelope
        ),
        expected_after_signed_policy_authorization_envelope_sha256=(
            args.after_signed_policy_authorization_envelope_sha256
        ),
        output=args.output,
        device_name=args.device,
        mode=args.mode,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
