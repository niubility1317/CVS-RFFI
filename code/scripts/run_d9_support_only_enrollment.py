#!/usr/bin/env python3
"""Run D9 only on verified sealed enrollment support packages.

The CLI intentionally has no query, truth, prediction, score, or scorer
argument.  It opens one before and one after ``enrollment_only`` package,
locks D9 at K10, proves nested K1/K5 prototype-only rebuilds, and writes new
immutable state/audit/COMMIT artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
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
from cvsrffi.stage2_class_conditional_iq_head import (  # noqa: E402
    OPERATORS,
    apply_received_iq_operator,
)
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_floor_sparse_operator_fusion import (  # noqa: E402
    FloorSparseOperatorFusionState,
    build_operator_feature_provenance,
    extend_floor_sparse_operator_fusion,
    fit_floor_sparse_operator_fusion,
    rebuild_locked_floor_sparse_prototypes,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)


class D9SupportRunnerError(ValueError):
    """Raised when enrollment-only or immutable-output invariants drift."""


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


def _write_json_new(path: Path, payload: Any) -> str:
    raw = _canonical(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _member(manifest: Mapping[str, Any], kind: str) -> dict[str, Any]:
    rows = [dict(row) for row in manifest["members"] if row.get("kind") == kind]
    if len(rows) != 1:
        raise D9SupportRunnerError(f"enrollment member drift: {kind}")
    return rows[0]


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    registration_state: str,
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
        or not bool(
            manifest.get(
                "phase2_post_reception_view_from_fixed_received_iq_only"
            )
        )
        or any(bool(manifest.get(field, True)) for field in required_false)
        or tuple(manifest.get("target_channel_scenarios", ()))
        != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise D9SupportRunnerError("enrollment package protocol drift")
    kinds = {str(row.get("kind")) for row in manifest.get("members", ())}
    required_kinds = {
        "feature_runtime",
        "method_lock",
        "overlay_provenance",
        *{f"support:{scenario}" for scenario in FORMAL_LEO_WEAK_SCENARIOS},
    }
    if kinds != required_kinds or any(
        any(
            token in kind.lower()
            for token in ("query", "truth", "score", "prediction", "scorer")
        )
        for kind in kinds
    ):
        raise D9SupportRunnerError(
            "enrollment package member allowlist drift"
        )


def _load_enrollment(
    root: Path,
    seal: Path,
    *,
    registration_state: str,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, Any],
]:
    if root.name != "enrollment_only" or "enrollment" not in seal.name:
        raise D9SupportRunnerError("runner accepts enrollment-only paths")
    payloads, manifest, preopen_audit = (
        load_verified_somph_predictor_bundle(
            root,
            detached_seal_path=seal,
            expected_seal_sha256=_sha256_file(seal),
        )
    )
    _validate_manifest(
        manifest, registration_state=registration_state
    )
    return payloads, manifest, preopen_audit


def _payload_rows(
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    scenario: str,
) -> dict[str, np.ndarray]:
    class_handles = np.asarray(
        [
            str(row["class_handle"])
            for row in manifest["registered_classes"]
        ]
    )
    ranks = np.asarray(
        payload["support_rank_within_class"], dtype=np.int64
    )
    indices = np.asarray(
        payload["support_class_indices"], dtype=np.int64
    )
    raw_labels = class_handles[indices].astype(str)
    raw_classes, raw_counts = np.unique(
        raw_labels, return_counts=True
    )
    if (
        set(raw_classes.tolist())
        != {
            str(row["class_handle"])
            for row in manifest["registered_classes"]
        }
        or set(raw_counts.tolist()) != {10}
        or set(ranks.tolist()) != set(range(10))
    ):
        raise D9SupportRunnerError(
            f"strict K10-only support reachability drift: {scenario}"
        )
    source_indices = np.arange(len(indices), dtype=np.int64)
    labels = raw_labels
    order = np.asarray(
        sorted(
            range(len(source_indices)),
            key=lambda index: (
                str(labels[index]),
                int(ranks[source_indices[index]]),
            ),
        ),
        dtype=np.int64,
    )
    selected = source_indices[order]
    rows = {
        "iq": np.asarray(
            payload["support_leo_weak_iq"], dtype=np.float32
        )[selected],
        "labels": class_handles[indices[selected]].astype(str),
        "ranks": ranks[selected],
        "tokens": np.asarray(payload["support_tokens"]).astype(str)[
            selected
        ],
        "overlay_tokens": np.asarray(
            payload["support_overlay_tokens"]
        ).astype(str)[selected],
        "satellite_seeds": np.asarray(
            payload["support_satellite_seeds"], dtype=np.int64
        )[selected],
        "hashes": np.asarray(
            payload["support_post_channel_iq_sha256"]
        ).astype(str)[selected],
    }
    computed_hashes = np.asarray(
        [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in rows["iq"]
        ]
    )
    classes, counts = np.unique(rows["labels"], return_counts=True)
    if (
        rows["iq"].ndim != 3
        or rows["iq"].shape[1] != 2
        or not np.isfinite(rows["iq"]).all()
        or not np.array_equal(computed_hashes, rows["hashes"])
        or len(set(rows["tokens"].tolist())) != len(rows["tokens"])
        or len(set(rows["overlay_tokens"].tolist()))
        != len(rows["overlay_tokens"])
        or len(set(rows["hashes"].tolist())) != len(rows["hashes"])
        or set(classes.tolist())
        != {
            str(row["class_handle"])
            for row in manifest["registered_classes"]
        }
        or set(counts.tolist()) != {10}
        or any(
            not str(token).startswith("sid_")
            for token in rows["tokens"]
        )
        or any(
            not str(token).startswith("oid_")
            for token in rows["overlay_tokens"]
        )
    ):
        raise D9SupportRunnerError(
            f"sealed K10 support payload drift: {scenario}"
        )
    return rows


def _extract_operator_features(
    model: torch.nn.Module,
    device: torch.device,
    iq: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for operator in OPERATORS:
        view = apply_received_iq_operator(iq, operator)
        zid = forward_zid160(
            model, view, device=device, batch_size=64
        )
        result[operator] = registered_feature(view, zid)
    return result


def _state_metadata(
    state: FloorSparseOperatorFusionState,
) -> dict[str, Any]:
    class_rows = []
    for class_index, handle in enumerate(state.classes):
        components = []
        for slot in range(state.operator_indices.shape[1]):
            weight = float(state.weights[class_index, slot])
            operator_index = int(state.operator_indices[class_index, slot])
            if weight > 0.0:
                components.append(
                    {
                        "operator_id": OPERATORS[operator_index],
                        "weight": weight,
                        "prototype_sha256": _array_sha256(
                            state.prototypes[class_index, slot]
                        ),
                    }
                )
        class_rows.append(
            {
                "class_handle": handle,
                "components": components,
            }
        )
    return {
        "schema": state.schema,
        "classes": list(state.classes),
        "class_rows": class_rows,
        "calibrations": [
            {
                "operator_id": value.operator_id,
                "center": value.center,
                "scale": value.scale,
            }
            for value in state.calibrations
        ],
        "feature_dim": state.feature_dim,
        "used_operators": list(state.used_operators),
        "old_class_count": state.old_class_count,
        "registration_generation": state.registration_generation,
        "current_k": state.current_k,
        "selection_lock_k": state.selection_lock_k,
        "selection_lock_sha256": state.selection_lock_sha256,
        "support_lineage": [
            {
                "class_handle": label,
                "physical_sample_id": token,
                "parent_received_iq_sha256": digest,
            }
            for label, token, digest in state.support_lineage
        ],
        "resource": state.resource_audit(),
    }


def _write_state_new(
    output: Path,
    *,
    stem: str,
    state: FloorSparseOperatorFusionState,
) -> dict[str, str]:
    npz_path = output / f"{stem}.npz"
    with npz_path.open("xb") as handle:
        np.savez(
            handle,
            operator_indices=state.operator_indices,
            weights=state.weights,
            prototypes=state.prototypes,
        )
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(npz_path)
    metadata_path = output / f"{stem}.json"
    metadata_sha = _write_json_new(
        metadata_path, _state_metadata(state)
    )
    return {
        "npz_sha256": _sha256_file(npz_path),
        "metadata_sha256": metadata_sha,
    }


def _selection_rows(
    state: FloorSparseOperatorFusionState,
    *,
    scenario: str,
    registration_state: str,
) -> list[dict[str, Any]]:
    selection = state.support_audit["selection"]
    key = (
        "per_class_selection"
        if registration_state == "before"
        else "per_new_class_selection"
    )
    rows = []
    by_class = {row["class_handle"]: row for row in selection[key]}
    for class_index, handle in enumerate(state.classes):
        trace = by_class.get(handle)
        components = [
            {
                "operator_id": OPERATORS[
                    int(state.operator_indices[class_index, slot])
                ],
                "weight": float(state.weights[class_index, slot]),
            }
            for slot in range(state.operator_indices.shape[1])
            if float(state.weights[class_index, slot]) > 0.0
        ]
        rows.append(
            {
                "scenario": scenario,
                "registration_state": registration_state,
                "class_handle": handle,
                "lifecycle": (
                    "old_locked"
                    if registration_state == "after"
                    and class_index < state.old_class_count
                    else "registered"
                ),
                "components": components,
                "selected_candidate_id": (
                    None
                    if trace is None
                    else trace["selected_candidate_id"]
                ),
                "floor_priority_resolved": (
                    False if trace is None else trace["floor_priority"]
                ),
                "candidate_evidence": (
                    [] if trace is None else trace["candidate_evidence"]
                ),
            }
        )
    return rows


def _nested_indices(
    rows: Mapping[str, np.ndarray], k: int
) -> np.ndarray:
    return np.flatnonzero(np.asarray(rows["ranks"]) < int(k))


def _nested_proof(
    locked: FloorSparseOperatorFusionState,
    rows: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    *,
    scenario: str,
    registration_state: str,
    k: int,
    output: Path,
) -> dict[str, Any]:
    indices = _nested_indices(rows, k)
    nested_features = {
        operator: value[indices] for operator, value in features.items()
    }
    nested_hashes = tuple(rows["hashes"][indices].tolist())
    provenance = build_operator_feature_provenance(
        nested_hashes, view_seed=0
    )
    rebuilt = rebuild_locked_floor_sparse_prototypes(
        locked,
        nested_features,
        provenance,
        rows["labels"][indices].tolist(),
        physical_sample_ids=rows["tokens"][indices].tolist(),
        parent_received_iq_sha256=nested_hashes,
    )
    stem = f"state_{scenario}_{registration_state}_k{k}"
    hashes = _write_state_new(output, stem=stem, state=rebuilt)
    return {
        "scenario": scenario,
        "registration_state": registration_state,
        "k": k,
        "support_count": len(indices),
        "selection_lock_sha256": rebuilt.selection_lock_sha256,
        "operator_indices_bitwise_locked": (
            rebuilt.operator_indices is locked.operator_indices
        ),
        "weights_bitwise_locked": rebuilt.weights is locked.weights,
        "calibrations_locked": rebuilt.calibrations == locked.calibrations,
        "only_prototypes_rebuilt": True,
        "k10_lineage_prefix_verified": True,
        "state_sha256": hashes,
    }


def _old_lineage_reuse(
    before_rows: Mapping[str, np.ndarray],
    after_rows: Mapping[str, np.ndarray],
    before_features: Mapping[str, np.ndarray],
    after_features: dict[str, np.ndarray],
    old_handles: set[str],
) -> None:
    prior = {
        str(token): (
            str(label),
            str(digest),
            np.asarray(iq),
            {
                operator: before_features[operator][index]
                for operator in OPERATORS
            },
        )
        for index, (token, label, digest, iq) in enumerate(
            zip(
                before_rows["tokens"],
                before_rows["labels"],
                before_rows["hashes"],
                before_rows["iq"],
            )
        )
    }
    for index, (token, label, digest, iq) in enumerate(
        zip(
            after_rows["tokens"],
            after_rows["labels"],
            after_rows["hashes"],
            after_rows["iq"],
        )
    ):
        if str(label) not in old_handles:
            continue
        value = prior.get(str(token))
        if (
            value is None
            or value[0] != str(label)
            or value[1] != str(digest)
            or not np.array_equal(value[2], iq)
        ):
            raise D9SupportRunnerError(
                "old support lineage/IQ changed across registration"
            )
        for operator in OPERATORS:
            after_features[operator][index] = value[3][operator]


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
        raise D9SupportRunnerError("output already exists")
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
        raise D9SupportRunnerError("before/after package binding drift")
    before_handles = {
        str(row["class_handle"])
        for row in before_manifest["registered_classes"]
    }
    after_handles = {
        str(row["class_handle"])
        for row in after_manifest["registered_classes"]
    }
    if not before_handles < after_handles:
        raise D9SupportRunnerError("absent-class registration drift")
    if device_name == "auto":
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_name)
    model = load_torchscript_backbone_same_fd(
        before_root,
        _member(before_manifest, "feature_runtime"),
        device=device,
    )
    output.mkdir(parents=True)
    state_hashes: dict[str, Any] = {}
    scenario_results: dict[str, Any] = {}
    selection_rows: list[dict[str, Any]] = []
    nested_proofs: list[dict[str, Any]] = []
    prior_scenario_tokens: set[str] = set()
    prior_scenario_hashes: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = _payload_rows(
            before_payloads[scenario],
            before_manifest,
            scenario=scenario,
        )
        after_rows = _payload_rows(
            after_payloads[scenario],
            after_manifest,
            scenario=scenario,
        )
        scenario_tokens = set(before_rows["tokens"].tolist()) | set(
            after_rows["tokens"].tolist()
        )
        scenario_hashes = set(before_rows["hashes"].tolist()) | set(
            after_rows["hashes"].tolist()
        )
        if (
            scenario_tokens.intersection(prior_scenario_tokens)
            or scenario_hashes.intersection(prior_scenario_hashes)
        ):
            raise D9SupportRunnerError(
                "cross-scenario support lineage reuse"
            )
        prior_scenario_tokens.update(scenario_tokens)
        prior_scenario_hashes.update(scenario_hashes)
        before_features = _extract_operator_features(
            model, device, before_rows["iq"]
        )
        after_features = _extract_operator_features(
            model, device, after_rows["iq"]
        )
        _old_lineage_reuse(
            before_rows,
            after_rows,
            before_features,
            after_features,
            before_handles,
        )
        before_provenance = build_operator_feature_provenance(
            before_rows["hashes"].tolist(), view_seed=0
        )
        after_provenance = build_operator_feature_provenance(
            after_rows["hashes"].tolist(), view_seed=0
        )
        before_state = fit_floor_sparse_operator_fusion(
            before_features,
            before_provenance,
            before_rows["labels"].tolist(),
            physical_sample_ids=before_rows["tokens"].tolist(),
            parent_received_iq_sha256=before_rows["hashes"].tolist(),
            base_resource_audit={
                "persistent_state_bytes": 0,
                "estimated_head_macs_per_query": 0,
            },
            floor_priority_classes=(),
        )
        after_state = extend_floor_sparse_operator_fusion(
            before_state,
            after_features,
            after_provenance,
            after_rows["labels"].tolist(),
            physical_sample_ids=after_rows["tokens"].tolist(),
            parent_received_iq_sha256=after_rows["hashes"].tolist(),
            floor_priority_classes=(),
        )
        for registration_state, state in (
            ("before", before_state),
            ("after", after_state),
        ):
            stem = f"state_{scenario}_{registration_state}_k10"
            state_hashes[f"{scenario}:{registration_state}:k10"] = (
                _write_state_new(output, stem=stem, state=state)
            )
            source_rows = (
                before_rows
                if registration_state == "before"
                else after_rows
            )
            source_features = (
                before_features
                if registration_state == "before"
                else after_features
            )
            for k in (1, 5):
                nested_proofs.append(
                    _nested_proof(
                        state,
                        source_rows,
                        source_features,
                        scenario=scenario,
                        registration_state=registration_state,
                        k=k,
                        output=output,
                    )
                )
            selection_rows.extend(
                _selection_rows(
                    state,
                    scenario=scenario,
                    registration_state=registration_state,
                )
            )
        scenario_results[scenario] = {
            "before": {
                "registered_class_count": before_state.class_count,
                "selection": before_state.support_audit["selection"],
                "resource": before_state.resource_audit(),
            },
            "after": {
                "registered_class_count": after_state.class_count,
                "selection": after_state.support_audit["selection"],
                "resource": after_state.resource_audit(),
                "old_state_bitwise_locked": True,
                "old_support_lineage_verified": True,
            },
        }
    audit = {
        "schema": "cvs.phase2.d9_support_only_runner_audit.v1",
        "diagnostic_only": True,
        "status": "SUPPORT_ONLY_D9_LOCKED_NO_QUERY_OPEN",
        "claim_scope": (
            "support_selection_and_state_only_no_query_performance_claim"
        ),
        "receiver": after_manifest["receiver"],
        "seed": int(after_manifest["seed"]),
        "k_shot": 10,
        "new_class_count": (
            int(after_manifest["registered_class_count"])
            - int(before_manifest["registered_class_count"])
        ),
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "opened_package_profiles": [
            "before:enrollment_only",
            "after:enrollment_only",
        ],
        "opened_member_kinds": sorted(
            {
                str(row["kind"])
                for row in before_manifest["members"]
            }
            | {
                str(row["kind"])
                for row in after_manifest["members"]
            }
        ),
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
        "operator_feature_view_seed": 0,
        "operator_feature_binding_fields": [
            "parent_received_iq_sha256",
            "operator_id",
            "view_seed",
        ],
        "focus_tx_resolution": (
            "unavailable_inside_sealed_enrollment_package_opaque_handles_only"
        ),
        "focus_tx_performance_claim": False,
        "scenario_results": scenario_results,
        "per_class_candidate_rows": selection_rows,
        "nested_k_rebuild_proofs": nested_proofs,
        "state_sha256": state_hashes,
        "before_package_root_sha256": before_manifest[
            "package_root_sha256"
        ],
        "after_package_root_sha256": after_manifest[
            "package_root_sha256"
        ],
        "before_seal_sha256": _sha256_file(before_seal),
        "after_seal_sha256": _sha256_file(after_seal),
        "preopen_audit": {
            "before": before_preopen,
            "after": after_preopen,
        },
        "base_resource_policy": (
            "sealed_phase1_checkpoint_reported_separately_not_counted_as_"
            "mutable_adapter_state; D9 combined state counts every D9 tensor"
        ),
        "sealed_feature_runtime_file_bytes": _member(
            before_manifest, "feature_runtime"
        )["size_bytes"],
    }
    audit_sha = _write_json_new(output / "support_audit.json", audit)
    report_lines = [
        "# D9 sealed enrollment support-only锁定",
        "",
        "状态：只打开before/after的`enrollment_only`包，未打开query、truth、"
        "prediction、score或scorer。本artifact不包含query性能结论。",
        "",
        "三个固定received-IQ operator均由包内唯一LEO_weak IQ计算；每个operator"
        " feature逐样本绑定父IQ SHA、operator ID和固定view seed 0，view不增加K，"
        "不生成额外LEO信道状态。",
        "",
        "|场景|状态|类数|support overall baseline/final|support floor baseline/final|"
        "状态bytes|去重operator数|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        for registration_state in ("before", "after"):
            row = scenario_results[scenario][registration_state]
            selection = row["selection"]
            baseline = (
                selection["baseline"]
                if registration_state == "before"
                else selection["baseline_new"]
            )
            final = (
                selection["combined_final"]
                if registration_state == "before"
                else selection["combined_final_new"]
            )
            resource = row["resource"]
            report_lines.append(
                f"|`{scenario}`|{registration_state}|"
                f"{row['registered_class_count']}|"
                f"{baseline['overall_accuracy']:.4f}/"
                f"{final['overall_accuracy']:.4f}|"
                f"{baseline['min_class_accuracy']:.4f}/"
                f"{final['min_class_accuracy']:.4f}|"
                f"{resource['combined_persistent_state_bytes']}|"
                f"{resource['used_operator_count']}|"
            )
    report_lines.extend(
        [
            "",
            "K1/K5仅按K10有序物理support lineage前缀重建prototype；operator indices、"
            "weights、calibrations与selection lock保持锁定。After旧类状态及旧support"
            " lineage已验证，新增类只追加。",
            "",
            "包内类别仅为opaque handle，没有真实TX标签。为保持只读sealed enrollment"
            "边界，本runner未从外部cache反查`20-19/1-18`；逐类候选和floor门完整保留，"
            "但不作TX定向性能声明。",
            "",
        ]
    )
    report_sha = _write_text_new(
        output / "report.md", "\n".join(report_lines)
    )
    commit = {
        "schema": "cvs.phase2.d9_support_only_commit.v1",
        "diagnostic_only": True,
        "status": audit["status"],
        "support_audit_sha256": audit_sha,
        "report_sha256": report_sha,
        "state_sha256": state_hashes,
        "query_package_opened": False,
        "query_truth_opened": False,
        "scorer_opened": False,
        "independent_performance_claim": False,
    }
    commit_sha = _write_json_new(output / "COMMIT.json", commit)
    return {
        "status": commit["status"],
        "output": str(output),
        "commit_sha256": commit_sha,
        "support_audit_sha256": audit_sha,
        "scenario_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "nested_k_proof_count": len(nested_proofs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lock D9 from before/after sealed enrollment support only"
        )
    )
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        before_root=args.before_root.resolve(),
        before_seal=args.before_seal.resolve(),
        after_root=args.after_root.resolve(),
        after_seal=args.after_seal.resolve(),
        output=args.output.resolve(),
        device_name=str(args.device),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
