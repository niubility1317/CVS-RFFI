"""Isolated scorer for SOMP-H single-stream Stage2 prediction artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import stage2_metric_scorer as _base
from .somph_prediction_artifact import verify_somph_prediction_artifact


SCORING_MANIFEST_SCHEMA = "cvs.phase2.somph_scoring_manifest.v1"
EVIDENCE_MANIFEST_SCHEMA = "cvs.phase2.somph_evidence_manifest.v1"
RESOURCE_AUDIT_SCHEMA = "cvs.phase2.somph_resource_audit.v1"
PREOPEN_AUDIT_SCHEMA = "cvs.phase2.somph_preopen_audit.v1"
RUNTIME_ACCESS_AUDIT_SCHEMA = "cvs.phase2.somph_runtime_access_audit.v1"
REGISTRATION_PAIR_SCHEMA = "cvs.phase2.somph_registration_pair.v1"
FORMAL_ROWS_SCHEMA = "cvs.phase2.somph_formal_metric_rows.v1"
FORMAL_PREDICTIONS_SCHEMA = "cvs.phase2.somph_formal_scored_predictions.v1"
SCORING_RECEIPT_SCHEMA = "cvs.phase2.somph_scoring_receipt.v1"
SOMPH_TRUTH_SIDECAR_SCHEMA = "cvs.phase2.query_truth_sidecar.v2"

FORMAL_OLD_TX_LABELS = (
    "14-10", "14-7", "20-15", "20-19", "6-15", "8-20",
)
FORMAL_NEW20_TX_LABELS = (
    "1-16", "1-18", "18-10", "14-11", "8-3",
    "18-8", "10-10", "16-19", "20-12", "4-10",
    "13-14", "2-5", "1-8", "19-13", "19-9",
    "3-8", "19-8", "11-19", "2-16", "19-6",
)

_MANIFEST_KEYS = {
    "schema",
    "stage",
    "receiver",
    "seed",
    "k_shot",
    "new_class_count",
    "expected_query_per_tx",
    "scenarios",
    "old_tx_labels",
    "new_tx_labels",
    "truth_sidecar_json",
    "truth_sidecar_sha256",
    "evidence_manifest_json",
    "evidence_manifest_sha256",
    "scorer_output_must_not_feed_predictor",
}

_PHASE2_CONTRACT = {
    "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
    "clean_sample_access": False,
    "clean_derived_signal_access": False,
    "phase2_clean_dataset_reachable": False,
    "phase2_clean_cache_reachable": False,
    "phase2_clean_control_flow_reachable": False,
    "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
    "phase2_query_decision_policy": "per_sample_all_registered_classes",
    "phase2_query_role_oracle_access": False,
    "phase2_query_true_batch_class_count_access": False,
    "phase2_query_class_quota_access": False,
    "phase2_query_batch_global_assignment": False,
    "query_labels_used_for_fit": False,
    "target_query_used_for_training": False,
    "target_query_used_for_model_selection": False,
    "dense_query_graph_used": False,
}


class SomphScoringError(ValueError):
    """Raised when isolated SOMP-H scoring evidence fails closed."""


def _load_bound_json(
    parent: Path,
    leaf_value: Any,
    expected_sha256: Any,
    *,
    context: str,
) -> tuple[dict[str, Any], Path]:
    if not _base._is_sha256(expected_sha256):
        raise SomphScoringError(f"{context} detached hash drift")
    try:
        leaf = _base._relative_leaf(leaf_value, context=f"{context} path")
        path = _base._regular_file(parent / leaf, context=context)
        with _base._open_regular_same_fd(path, context=context) as handle:
            actual_hash, _size = _base._hash_handle(handle)
            _base._validate_hash_value(
                actual_hash, expected_sha256, context=f"{context} detached hash"
            )
            handle.seek(0)
            payload = _base._load_json_handle(handle, context=context)
    except _base.Stage2ScoringError as exc:
        raise SomphScoringError(str(exc)) from exc
    return payload, path


def _validate_resource_audit(
    payload: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
) -> None:
    keys = {
        "schema", "status", "head_capsule_sha256", "trainable_parameters",
        "updated_original_parameters", "adaptation_epochs", "optimizer_steps",
        "optimizer_state_bytes", "optimizer_state_deployment_required",
        "query_rows_used_for_fit", "clean_sample_access",
        "clean_derived_signal_access", "candidate_state_bytes_fp16",
        "active_scenario_state_bytes_fp16", "candidate_state_cap_bytes",
        "candidate_extra_macs_per_query",
        "capsule_array_bytes_including_registry_and_audit",
        "base_checkpoint_state_bytes", "base_backbone_macs_per_forward",
        "total_deployment_state_bytes", "total_macs_per_query",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        raise SomphScoringError("SOMP-H resource audit exact schema drift")
    if payload["schema"] != RESOURCE_AUDIT_SCHEMA or payload["status"] != "PASS":
        raise SomphScoringError("SOMP-H resource audit status/schema drift")
    if payload["head_capsule_sha256"] != binding["head_capsule_sha256"]:
        raise SomphScoringError("SOMP-H resource audit/head capsule hash mismatch")
    zero_fields = (
        "trainable_parameters", "updated_original_parameters", "adaptation_epochs",
        "optimizer_steps", "optimizer_state_bytes", "query_rows_used_for_fit",
    )
    if any(payload[field] != 0 for field in zero_fields):
        raise SomphScoringError("SOMP-H zero-adaptation resource claim drift")
    if payload["optimizer_state_deployment_required"] is not False:
        raise SomphScoringError("SOMP-H optimizer deployment claim drift")
    if payload["clean_sample_access"] is not False or payload["clean_derived_signal_access"] is not False:
        raise SomphScoringError("SOMP-H resource audit reports forbidden clean access")
    integer_fields = (
        "candidate_state_bytes_fp16", "active_scenario_state_bytes_fp16",
        "candidate_state_cap_bytes", "candidate_extra_macs_per_query",
        "capsule_array_bytes_including_registry_and_audit",
        "base_checkpoint_state_bytes", "base_backbone_macs_per_forward",
        "total_deployment_state_bytes", "total_macs_per_query",
    )
    if any(
        not isinstance(payload[field], int)
        or isinstance(payload[field], bool)
        or payload[field] < 0
        for field in integer_fields
    ):
        raise SomphScoringError("SOMP-H resource audit numeric field drift")
    if payload["candidate_state_bytes_fp16"] > payload["candidate_state_cap_bytes"]:
        raise SomphScoringError("SOMP-H candidate state exceeds the locked cap")
    if payload["total_deployment_state_bytes"] != (
        payload["base_checkpoint_state_bytes"]
        + payload["active_scenario_state_bytes_fp16"]
    ):
        raise SomphScoringError("SOMP-H total deployment state arithmetic drift")
    if payload["total_macs_per_query"] != (
        payload["base_backbone_macs_per_forward"]
        + payload["candidate_extra_macs_per_query"]
    ):
        raise SomphScoringError("SOMP-H total MAC arithmetic drift")


def _validate_preopen_audit(
    payload: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    expected_overlays: Mapping[str, str],
) -> None:
    keys = {
        "schema", "status", "package_root_sha256", "package_seal_sha256",
        "phase2_sample_view_policy", "all_members_preoverlaid_leo_weak",
        "clean_sample_access", "clean_derived_signal_access",
        "phase2_clean_dataset_reachable", "phase2_clean_cache_reachable",
        "phase2_clean_control_flow_reachable",
        "leo_weak_member_sha256_by_scenario",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        raise SomphScoringError("SOMP-H pre-open audit exact schema drift")
    if payload["schema"] != PREOPEN_AUDIT_SCHEMA or payload["status"] != "PASS":
        raise SomphScoringError("SOMP-H pre-open audit status/schema drift")
    if payload["package_root_sha256"] != binding["package_root_sha256"] or payload[
        "package_seal_sha256"
    ] != binding["package_seal_sha256"]:
        raise SomphScoringError("SOMP-H pre-open package binding mismatch")
    if payload["phase2_sample_view_policy"] != "leo_weak_only_no_clean_access":
        raise SomphScoringError("SOMP-H pre-open sample-view policy drift")
    if payload["all_members_preoverlaid_leo_weak"] is not True:
        raise SomphScoringError("SOMP-H package contains a non-LEO_weak input")
    for field in (
        "clean_sample_access", "clean_derived_signal_access",
        "phase2_clean_dataset_reachable", "phase2_clean_cache_reachable",
        "phase2_clean_control_flow_reachable",
    ):
        if payload[field] is not False:
            raise SomphScoringError(f"SOMP-H pre-open forbidden access drift: {field}")
    if payload["leo_weak_member_sha256_by_scenario"] != expected_overlays:
        raise SomphScoringError("SOMP-H pre-open LEO_weak member binding drift")


def _validate_runtime_access_audit(payload: Mapping[str, Any]) -> None:
    keys = {
        "schema", "status", "opened_input_paths", "allowed_input_roots",
        "forbidden_open_count", "clean_sample_access",
        "clean_derived_signal_access", "truth_sidecar_access",
        "scoring_manifest_access", "query_role_oracle_access",
        "query_true_batch_class_count_access", "query_class_quota_access",
        "query_batch_global_assignment",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        raise SomphScoringError("SOMP-H runtime access audit exact schema drift")
    if payload["schema"] != RUNTIME_ACCESS_AUDIT_SCHEMA or payload["status"] != "PASS":
        raise SomphScoringError("SOMP-H runtime access audit status/schema drift")
    if payload["forbidden_open_count"] != 0:
        raise SomphScoringError("SOMP-H runtime access audit found a forbidden open")
    for field in (
        "clean_sample_access", "clean_derived_signal_access", "truth_sidecar_access",
        "scoring_manifest_access", "query_role_oracle_access",
        "query_true_batch_class_count_access", "query_class_quota_access",
        "query_batch_global_assignment",
    ):
        if payload[field] is not False:
            raise SomphScoringError(f"SOMP-H runtime forbidden access drift: {field}")
    paths = payload["opened_input_paths"]
    roots = payload["allowed_input_roots"]
    if (
        not isinstance(paths, list)
        or not paths
        or not all(isinstance(value, str) and value for value in paths)
        or not isinstance(roots, list)
        or not roots
        or not all(isinstance(value, str) and value for value in roots)
    ):
        raise SomphScoringError("SOMP-H runtime path ledger drift")
    normalized_roots = tuple(value.replace("\\", "/").rstrip("/").lower() for value in roots)
    forbidden_tokens = (
        "/clean/", "/raw/", "truth_sidecar", "scoring_manifest",
        "manysig", "manytx",
    )
    for value in paths:
        normalized = value.replace("\\", "/").lower()
        if not any(
            normalized == root or normalized.startswith(f"{root}/")
            for root in normalized_roots
        ):
            raise SomphScoringError("SOMP-H runtime opened an input outside the allowlist")
        if any(token in normalized for token in forbidden_tokens):
            raise SomphScoringError("SOMP-H runtime opened a forbidden input path")


def _validate_registration_pair(
    payload: Mapping[str, Any],
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    keys = {
        "schema", "row_manifest_sha256", "before_binding_sha256",
        "after_binding_sha256", "old_support_physical_ids_sha256_before",
        "old_support_physical_ids_sha256_after",
        "old_query_physical_ids_sha256_before",
        "old_query_physical_ids_sha256_after",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        raise SomphScoringError("SOMP-H registration pair exact schema drift")
    if payload["schema"] != REGISTRATION_PAIR_SCHEMA:
        raise SomphScoringError("SOMP-H registration pair schema drift")
    if payload["row_manifest_sha256"] != before["row_manifest_sha256"]:
        raise SomphScoringError("SOMP-H registration pair row hash mismatch")
    if payload["before_binding_sha256"] != before["stage_input_binding_sha256"]:
        raise SomphScoringError("SOMP-H registration pair before binding mismatch")
    if payload["after_binding_sha256"] != after["stage_input_binding_sha256"]:
        raise SomphScoringError("SOMP-H registration pair after binding mismatch")
    for field in (
        "old_support_physical_ids_sha256_before",
        "old_support_physical_ids_sha256_after",
        "old_query_physical_ids_sha256_before",
        "old_query_physical_ids_sha256_after",
    ):
        if not _base._is_sha256(payload[field]):
            raise SomphScoringError("SOMP-H registration pair physical-ID hash drift")
    if payload["old_support_physical_ids_sha256_before"] != payload[
        "old_support_physical_ids_sha256_after"
    ]:
        raise SomphScoringError("SOMP-H registration pair old support mismatch")
    if payload["old_query_physical_ids_sha256_before"] != payload[
        "old_query_physical_ids_sha256_after"
    ]:
        raise SomphScoringError("SOMP-H registration pair old query mismatch")


def _validate_evidence_bundle(
    manifest_path: Path,
    scoring_manifest: Mapping[str, Any],
    bindings: list[Mapping[str, Any]],
    verified_artifacts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence, evidence_path = _load_bound_json(
        manifest_path.parent,
        scoring_manifest["evidence_manifest_json"],
        scoring_manifest["evidence_manifest_sha256"],
        context="SOMP-H evidence manifest",
    )
    keys = {
        "schema", "stage", "protocol_policy_json", "protocol_policy_sha256",
        "scenarios",
        "satellite_seed_by_scenario", "state_evidence",
        "registration_pair_json", "registration_pair_sha256",
    }
    if set(evidence) != keys or evidence["schema"] != EVIDENCE_MANIFEST_SCHEMA:
        raise SomphScoringError("SOMP-H evidence manifest exact schema drift")
    if evidence["stage"] != scoring_manifest["stage"]:
        raise SomphScoringError("SOMP-H evidence/scoring stage mismatch")
    policy, policy_path = _load_bound_json(
        evidence_path.parent,
        evidence["protocol_policy_json"],
        evidence["protocol_policy_sha256"],
        context="SOMP-H protocol policy",
    )
    if policy != _PHASE2_CONTRACT:
        raise SomphScoringError("SOMP-H Phase2 protocol policy drift")
    if tuple(evidence["scenarios"]) != tuple(_base.FORMAL_LEO_WEAK_SCENARIOS):
        raise SomphScoringError("SOMP-H evidence scenario registry drift")
    seeds = evidence["satellite_seed_by_scenario"]
    if (
        not isinstance(seeds, dict)
        or tuple(seeds) != tuple(_base.FORMAL_LEO_WEAK_SCENARIOS)
        or any(
            not isinstance(seeds[scenario], int)
            or isinstance(seeds[scenario], bool)
            or seeds[scenario] < 0
            for scenario in _base.FORMAL_LEO_WEAK_SCENARIOS
        )
    ):
        raise SomphScoringError("SOMP-H evidence satellite seed drift")
    by_state = evidence["state_evidence"]
    expected_states = [binding["registration_state"] for binding in bindings]
    if not isinstance(by_state, dict) or set(by_state) != set(expected_states):
        raise SomphScoringError("SOMP-H evidence registration-state registry drift")
    state_audit: dict[str, Any] = {}
    state_keys = {
        "method_lock_sha256", "row_manifest_sha256",
        "stage_input_binding_sha256", "package_root_sha256",
        "package_seal_sha256", "feature_runtime_sha256",
        "head_capsule_sha256", "leo_weak_member_sha256_by_scenario",
        "prediction_artifact_sha256", "prediction_seal_sha256",
        "resource_audit_json", "resource_audit_sha256",
        "preopen_audit_json", "preopen_audit_sha256",
        "runtime_access_audit_json", "runtime_access_audit_sha256",
    }
    verified_by_state = {
        binding["registration_state"]: verified
        for binding, verified in zip(bindings, verified_artifacts)
    }
    for binding in bindings:
        state = binding["registration_state"]
        item = by_state[state]
        if not isinstance(item, dict) or set(item) != state_keys:
            raise SomphScoringError("SOMP-H state evidence exact schema drift")
        for field in (
            "method_lock_sha256", "row_manifest_sha256",
            "stage_input_binding_sha256", "package_root_sha256",
            "package_seal_sha256", "feature_runtime_sha256",
            "head_capsule_sha256",
        ):
            if item[field] != binding[field]:
                raise SomphScoringError(f"SOMP-H evidence/artifact mismatch: {field}")
        if item["prediction_artifact_sha256"] != verified_by_state[state][
            "artifact_sha256"
        ]:
            raise SomphScoringError("SOMP-H post-run evidence artifact hash mismatch")
        if item["prediction_seal_sha256"] != verified_by_state[state]["seal_sha256"]:
            raise SomphScoringError("SOMP-H post-run evidence seal hash mismatch")
        if binding["protocol_policy_sha256"] != evidence["protocol_policy_sha256"]:
            raise SomphScoringError("SOMP-H artifact/protocol policy hash mismatch")
        overlays = item["leo_weak_member_sha256_by_scenario"]
        if (
            not isinstance(overlays, dict)
            or tuple(overlays) != tuple(_base.FORMAL_LEO_WEAK_SCENARIOS)
            or any(not _base._is_sha256(overlays[scenario]) for scenario in overlays)
        ):
            raise SomphScoringError("SOMP-H LEO_weak member evidence drift")
        resource, resource_path = _load_bound_json(
            evidence_path.parent,
            item["resource_audit_json"],
            item["resource_audit_sha256"],
            context=f"SOMP-H {state} resource audit",
        )
        preopen, preopen_path = _load_bound_json(
            evidence_path.parent,
            item["preopen_audit_json"],
            item["preopen_audit_sha256"],
            context=f"SOMP-H {state} pre-open audit",
        )
        runtime, runtime_path = _load_bound_json(
            evidence_path.parent,
            item["runtime_access_audit_json"],
            item["runtime_access_audit_sha256"],
            context=f"SOMP-H {state} runtime access audit",
        )
        _validate_resource_audit(resource, binding=binding)
        _validate_preopen_audit(
            preopen, binding=binding, expected_overlays=overlays
        )
        _validate_runtime_access_audit(runtime)
        state_audit[state] = {
            "resource_audit": str(resource_path),
            "resource_audit_sha256": item["resource_audit_sha256"],
            "preopen_audit": str(preopen_path),
            "preopen_audit_sha256": item["preopen_audit_sha256"],
            "runtime_access_audit": str(runtime_path),
            "runtime_access_audit_sha256": item["runtime_access_audit_sha256"],
        }
    if scoring_manifest["stage"] == "stage2c":
        if not isinstance(evidence["registration_pair_json"], str) or not _base._is_sha256(
            evidence["registration_pair_sha256"]
        ):
            raise SomphScoringError("SOMP-H registration pair reference drift")
        pair, pair_path = _load_bound_json(
            evidence_path.parent,
            evidence["registration_pair_json"],
            evidence["registration_pair_sha256"],
            context="SOMP-H registration pair",
        )
        _validate_registration_pair(pair, before=bindings[0], after=bindings[1])
        pair_audit = {
            "registration_pair": str(pair_path),
            "registration_pair_sha256": evidence["registration_pair_sha256"],
            "old_query_physical_ids_sha256": pair[
                "old_query_physical_ids_sha256_before"
            ],
        }
    else:
        if evidence["registration_pair_json"] is not None or evidence[
            "registration_pair_sha256"
        ] is not None:
            raise SomphScoringError("SOMP-H Stage2-B must not carry a registration pair")
        pair_audit = {
            "registration_pair": None,
            "registration_pair_sha256": None,
            "old_query_physical_ids_sha256": None,
        }
    return {
        "evidence_manifest": str(evidence_path),
        "evidence_manifest_sha256": scoring_manifest["evidence_manifest_sha256"],
        "protocol_policy": str(policy_path),
        "protocol_policy_sha256": evidence["protocol_policy_sha256"],
        "state_audit": state_audit,
        **pair_audit,
        "phase2_protocol_evidence_status": (
            "STRUCTURAL_ONLY_REAL_INPUT_RECOMPUTE_REQUIRED"
        ),
    }


def _load_scoring_inputs(
    scoring_manifest_path: str | Path,
    *,
    expected_scoring_manifest_sha256: str,
    bindings: list[Mapping[str, Any]],
    verified_artifacts: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = _base._regular_file(
        Path(scoring_manifest_path), context="SOMP-H scoring manifest"
    )
    try:
        with _base._open_regular_same_fd(
            manifest_path, context="SOMP-H scoring manifest"
        ) as handle:
            actual_hash, _size = _base._hash_handle(handle)
            _base._validate_hash_value(
                actual_hash,
                expected_scoring_manifest_sha256,
                context="SOMP-H scoring manifest detached hash",
            )
            handle.seek(0)
            manifest = _base._load_json_handle(
                handle, context="SOMP-H scoring manifest"
            )
    except _base.Stage2ScoringError as exc:
        raise SomphScoringError(str(exc)) from exc
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema") != SCORING_MANIFEST_SCHEMA:
        raise SomphScoringError("SOMP-H scoring manifest exact schema drift")
    if manifest["stage"] not in {"stage2b", "stage2c"}:
        raise SomphScoringError("SOMP-H scoring manifest stage drift")
    if not isinstance(manifest["receiver"], str) or not manifest["receiver"]:
        raise SomphScoringError("SOMP-H scoring receiver drift")
    for field in ("seed", "k_shot", "new_class_count", "expected_query_per_tx"):
        value = manifest[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise SomphScoringError(f"SOMP-H scoring {field} must be an integer")
    if manifest["k_shot"] <= 0 or manifest["expected_query_per_tx"] != 20:
        raise SomphScoringError("SOMP-H formal scorer requires Q20 per TX")
    if tuple(manifest["scenarios"]) != tuple(_base.FORMAL_LEO_WEAK_SCENARIOS):
        raise SomphScoringError("SOMP-H formal scenario sequence drift")
    old_labels = manifest["old_tx_labels"]
    new_labels = manifest["new_tx_labels"]
    if tuple(old_labels) != FORMAL_OLD_TX_LABELS:
        raise SomphScoringError("SOMP-H old TX registry drift")
    if not isinstance(new_labels, list) or len(set(new_labels)) != len(new_labels):
        raise SomphScoringError("SOMP-H new TX registry drift")
    if set(old_labels) & set(new_labels):
        raise SomphScoringError("SOMP-H old/new TX registries overlap")
    expected_new = 0 if manifest["stage"] == "stage2b" else manifest["new_class_count"]
    if manifest["new_class_count"] not in {0, 5, 10, 20} or len(new_labels) != expected_new:
        raise SomphScoringError("SOMP-H new TX count/registry drift")
    if tuple(new_labels) != FORMAL_NEW20_TX_LABELS[:expected_new]:
        raise SomphScoringError("SOMP-H nested new TX registry drift")
    if manifest["stage"] == "stage2b" and manifest["new_class_count"] != 0:
        raise SomphScoringError("SOMP-H Stage2-B requires new_count=0")
    if manifest["scorer_output_must_not_feed_predictor"] is not True:
        raise SomphScoringError("SOMP-H scorer feedback guard is not locked")
    for field in ("truth_sidecar_sha256", "evidence_manifest_sha256"):
        if not _base._is_sha256(manifest[field]):
            raise SomphScoringError(f"SOMP-H scoring {field} drift")
    registration_pair = (
        manifest["stage"] == "stage2c"
        and len(bindings) == 2
        and {
            binding["registration_state"] for binding in bindings
        }
        == {"before_registration", "after_registration"}
    )
    for binding in bindings:
        expected_stage = None
        if registration_pair:
            expected_stage = (
                "Stage2-B"
                if binding["registration_state"] == "before_registration"
                else "Stage2-C"
            )
        _validate_common_binding(
            binding,
            manifest,
            expected_stage=expected_stage,
        )
    evidence_audit = _validate_evidence_bundle(
        manifest_path, manifest, bindings, verified_artifacts
    )
    # Truth is intentionally opened only after immutable predictions and all
    # LEO_weak-only/clean-unreachable evidence files verify.
    try:
        truth_leaf = _base._relative_leaf(
            manifest["truth_sidecar_json"], context="SOMP-H truth sidecar path"
        )
        truth_path = _base._regular_file(
            manifest_path.parent / truth_leaf, context="SOMP-H truth sidecar"
        )
        with _base._open_regular_same_fd(
            truth_path, context="SOMP-H truth sidecar"
        ) as handle:
            truth_hash, _size = _base._hash_handle(handle)
            _base._validate_hash_value(
                truth_hash,
                manifest["truth_sidecar_sha256"],
                context="SOMP-H truth sidecar detached hash",
            )
            handle.seek(0)
            truth = _base._load_json_handle(handle, context="SOMP-H truth sidecar")
        truth = _base._exact_object(
            truth, _base.TRUTH_TOP_LEVEL_KEYS, context="SOMP-H truth sidecar"
        )
        if truth["schema"] != SOMPH_TRUTH_SIDECAR_SCHEMA:
            raise SomphScoringError("SOMP-H truth sidecar schema drift")
        _base._validate_truth_rows(truth, require_scenario=False)
    except _base.Stage2ScoringError as exc:
        raise SomphScoringError(str(exc)) from exc
    if truth["stage"] != manifest["stage"]:
        raise SomphScoringError("SOMP-H scoring manifest/truth stage mismatch")
    if truth["receiver"] != manifest["receiver"] or truth["seed"] != manifest["seed"]:
        raise SomphScoringError("SOMP-H scoring manifest/truth cell mismatch")
    truth_labels = {row["transmitter_label"] for row in truth["rows"]}
    expected_labels = set(old_labels) | set(new_labels)
    if truth_labels != expected_labels:
        raise SomphScoringError("SOMP-H truth TX registry does not match scoring manifest")
    handles_by_label: dict[str, str] = {}
    for row in truth["rows"]:
        label = row["transmitter_label"]
        handle = row["true_class_handle"]
        previous = handles_by_label.setdefault(label, handle)
        if previous != handle:
            raise SomphScoringError(
                "SOMP-H truth maps one TX label to multiple class handles"
            )
    for binding in bindings:
        labels = list(old_labels)
        if binding["registration_state"] == "after_registration":
            labels.extend(new_labels)
        expected_snapshot = _base.sha256_bytes(
            _base.canonical_json_bytes(
                [handles_by_label[label] for label in labels]
            )
        )
        if binding["registry_snapshot_sha256"] != expected_snapshot:
            raise SomphScoringError(
                "SOMP-H prediction registry snapshot does not match scorer truth registry"
            )
    if manifest["stage"] == "stage2b" and any(
        row["evaluation_role"] != "target_old" for row in truth["rows"]
    ):
        raise SomphScoringError("SOMP-H Stage2-B truth must contain target-old only")
    counts: dict[str, int] = {}
    for row in truth["rows"]:
        label = row["transmitter_label"]
        counts[label] = counts.get(label, 0) + 1
    if set(counts.values()) != {manifest["expected_query_per_tx"]}:
        raise SomphScoringError("SOMP-H scorer-side per-TX query coverage drift")
    if manifest["stage"] == "stage2c":
        old_rows = sorted(
            (
                row
                for row in truth["rows"]
                if row["evaluation_role"] == "target_old"
            ),
            key=lambda row: row["query_token"],
        )
        if any(
            not isinstance(row.get("physical_sample_id"), str)
            or not row["physical_sample_id"]
            for row in old_rows
        ):
            raise SomphScoringError(
                "SOMP-H Stage2-C truth lacks old-query physical sample IDs"
            )
        physical_digest = _base.sha256_bytes(
            _base.canonical_json_bytes(
                [row["physical_sample_id"] for row in old_rows]
            )
        )
        if physical_digest != evidence_audit["old_query_physical_ids_sha256"]:
            raise SomphScoringError(
                "SOMP-H registration pair is detached from scored old queries"
            )
    return manifest, truth, {
        "scoring_manifest": str(manifest_path),
        "scoring_manifest_sha256": expected_scoring_manifest_sha256,
        "truth_sidecar": str(truth_path),
        "truth_sidecar_sha256": manifest["truth_sidecar_sha256"],
        **evidence_audit,
    }


def _verify_artifact(
    path: str | Path,
    *,
    artifact_sha256: str,
    seal_sha256: str,
) -> dict[str, Any]:
    try:
        return verify_somph_prediction_artifact(
            path,
            expected_artifact_sha256=artifact_sha256,
            expected_seal_sha256=seal_sha256,
        )
    except ValueError as exc:
        raise SomphScoringError(f"SOMP-H prediction verification failed: {exc}") from exc


def _binding(verified: Mapping[str, Any]) -> dict[str, Any]:
    manifest = verified["manifest"]
    return {
        key: manifest[key]
        for key in (
            "stage", "registration_state", "row_id", "receiver", "seed", "k_shot",
            "registered_class_count", "registry_snapshot_sha256",
            "method_lock_sha256", "row_manifest_sha256",
            "stage_input_binding_sha256", "package_root_sha256", "package_seal_sha256",
            "feature_runtime_sha256", "head_capsule_sha256", "protocol_policy_sha256",
            "resource_receipt",
        )
    }


def _validate_common_binding(
    binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_stage: str | None = None,
) -> None:
    if expected_stage is None:
        expected_stage = (
            "Stage2-B" if manifest["stage"] == "stage2b" else "Stage2-C"
        )
    for field, expected in (
        ("stage", expected_stage),
        ("receiver", manifest["receiver"]),
        ("seed", manifest["seed"]),
        ("k_shot", manifest["k_shot"]),
    ):
        if binding[field] != expected:
            raise SomphScoringError(f"SOMP-H prediction/scoring binding mismatch: {field}")
    expected_registered = len(manifest["old_tx_labels"])
    if binding["registration_state"] == "after_registration":
        expected_registered += manifest["new_class_count"]
    if binding["registered_class_count"] != expected_registered:
        raise SomphScoringError(
            "SOMP-H prediction/scoring registered class count mismatch"
        )


def _scenario_arrays(verified: Mapping[str, Any], scenario: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = verified["arrays"]
    mask = np.asarray(arrays["scenarios"]).astype(str) == scenario
    return (
        np.asarray(arrays["query_tokens"]).astype(str)[mask],
        np.asarray(arrays["predicted_class_handles"]).astype(str)[mask],
    )


def _class_metrics(
    predictions: np.ndarray,
    ordered_truth: list[Mapping[str, Any]],
    labels: list[str],
) -> tuple[dict[str, float], dict[str, int]]:
    accuracy: dict[str, float] = {}
    counts: dict[str, int] = {}
    truth_handles = np.asarray([row["true_class_handle"] for row in ordered_truth]).astype(str)
    for label in labels:
        mask = np.asarray([row["transmitter_label"] == label for row in ordered_truth])
        counts[label] = int(np.sum(mask))
        accuracy[label] = _base._accuracy(predictions[mask], truth_handles[mask])
    return accuracy, counts


def _validate_prediction_handles(
    predictions: np.ndarray,
    truth: Mapping[str, Any],
    *,
    allowed_roles: set[str],
) -> dict[str, str]:
    role_by_handle = {
        row["true_class_handle"]: row["evaluation_role"] for row in truth["rows"]
    }
    unknown = sorted(set(predictions.tolist()) - set(role_by_handle))
    if unknown:
        raise SomphScoringError("SOMP-H prediction references an unregistered class handle")
    disallowed = sorted(
        value for value in set(predictions.tolist()) if role_by_handle[value] not in allowed_roles
    )
    if disallowed:
        raise SomphScoringError("SOMP-H prediction references a class outside its registration state")
    return role_by_handle


def _confusion_matrix(
    predictions: np.ndarray,
    ordered_truth: list[Mapping[str, Any]],
    labels: list[str],
) -> dict[str, dict[str, int]]:
    tx_by_handle = {
        row["true_class_handle"]: row["transmitter_label"] for row in ordered_truth
    }
    matrix = {
        true_label: {predicted_label: 0 for predicted_label in labels}
        for true_label in labels
    }
    for predicted, truth_row in zip(predictions.tolist(), ordered_truth):
        matrix[truth_row["transmitter_label"]][tx_by_handle[predicted]] += 1
    return matrix


def _receipt_hashes(payload: Mapping[str, Any]) -> str:
    return _base.sha256_bytes(_base.canonical_json_bytes(payload) + b"\n")


def score_somph_stage2b(
    prediction_artifact_path: str | Path,
    scoring_manifest_path: str | Path,
    *,
    expected_prediction_artifact_sha256: str,
    expected_prediction_seal_sha256: str,
    expected_scoring_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Score one frozen Stage2-B old-only SOMP-H prediction artifact."""
    verified = _verify_artifact(
        prediction_artifact_path,
        artifact_sha256=expected_prediction_artifact_sha256,
        seal_sha256=expected_prediction_seal_sha256,
    )
    # Truth is opened only after the immutable prediction verifies.
    binding = _binding(verified)
    manifest, truth, scoring_audit = _load_scoring_inputs(
        scoring_manifest_path,
        expected_scoring_manifest_sha256=expected_scoring_manifest_sha256,
        bindings=[binding],
        verified_artifacts=[verified],
    )
    if manifest["stage"] != "stage2b":
        raise SomphScoringError("score_somph_stage2b requires a Stage2-B manifest")
    if binding["registration_state"] != "before_registration":
        raise SomphScoringError("SOMP-H Stage2-B artifact must be before_registration")
    truth_by_token = {row["query_token"]: row for row in truth["rows"]}
    rows: list[dict[str, Any]] = []
    predictions_output: list[dict[str, Any]] = []
    for scenario in manifest["scenarios"]:
        tokens, predictions = _scenario_arrays(verified, scenario)
        if set(tokens.tolist()) != set(truth_by_token):
            raise SomphScoringError("SOMP-H Stage2-B prediction/truth token mismatch")
        ordered_truth = [truth_by_token[token] for token in tokens.tolist()]
        _validate_prediction_handles(predictions, truth, allowed_roles={"target_old"})
        truth_handles = np.asarray([row["true_class_handle"] for row in ordered_truth]).astype(str)
        old_acc = _base._accuracy(predictions, truth_handles)
        old_class_acc, old_class_count = _class_metrics(
            predictions, ordered_truth, manifest["old_tx_labels"]
        )
        rows.append({
            "row_id": binding["row_id"],
            "stage": "stage2b",
            "receiver_label": binding["receiver"],
            "seed": binding["seed"],
            "scenario": scenario,
            "k_shot": binding["k_shot"],
            "new_class_count": 0,
            "query_count": len(tokens),
            "old_acc_before_increment": old_acc,
            "old_acc_after_increment": None,
            "min_old_class_acc_before": min(old_class_acc.values()),
            "seen_new_acc": None,
            "H_old_new": None,
            "old_class_acc_before": old_class_acc,
            "old_class_count": old_class_count,
            "before_old_confusion_matrix_counts": _confusion_matrix(
                predictions, ordered_truth, manifest["old_tx_labels"]
            ),
        })
        for token, predicted, truth_row in zip(tokens.tolist(), predictions.tolist(), ordered_truth):
            predictions_output.append({
                "row_id": binding["row_id"],
                "stage": "stage2b",
                "registration_state": "before_registration",
                "receiver_label": binding["receiver"],
                "scenario": scenario,
                "query_token": token,
                "evaluation_role": "target_old",
                "transmitter_label": truth_row["transmitter_label"],
                "true_class_handle": truth_row["true_class_handle"],
                "predicted_class_handle": predicted,
                "correct": int(predicted == truth_row["true_class_handle"]),
            })
    return _finalize(
        rows=rows,
        predictions=predictions_output,
        bindings=[binding],
        verified=[verified],
        scoring_audit=scoring_audit,
    )


def score_somph_registration_pair(
    before_prediction_artifact_path: str | Path,
    after_prediction_artifact_path: str | Path,
    scoring_manifest_path: str | Path,
    *,
    expected_before_artifact_sha256: str,
    expected_before_seal_sha256: str,
    expected_after_artifact_sha256: str,
    expected_after_seal_sha256: str,
    expected_scoring_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify two frozen streams, then join truth and score registration."""
    before_verified = _verify_artifact(
        before_prediction_artifact_path,
        artifact_sha256=expected_before_artifact_sha256,
        seal_sha256=expected_before_seal_sha256,
    )
    after_verified = _verify_artifact(
        after_prediction_artifact_path,
        artifact_sha256=expected_after_artifact_sha256,
        seal_sha256=expected_after_seal_sha256,
    )
    # The scorer-side truth/quota manifest is opened only after both artifacts verify.
    before = _binding(before_verified)
    after = _binding(after_verified)
    manifest, truth, scoring_audit = _load_scoring_inputs(
        scoring_manifest_path,
        expected_scoring_manifest_sha256=expected_scoring_manifest_sha256,
        bindings=[before, after],
        verified_artifacts=[before_verified, after_verified],
    )
    if manifest["stage"] != "stage2c":
        raise SomphScoringError("registration-pair scoring requires Stage2-C")
    if before["registration_state"] != "before_registration" or after["registration_state"] != "after_registration":
        raise SomphScoringError("SOMP-H registration-state ordering drift")
    if before["stage"] != "Stage2-B" or after["stage"] != "Stage2-C":
        raise SomphScoringError(
            "SOMP-H registration pair must transition Stage2-B to Stage2-C"
        )
    for field in (
        "row_id", "receiver", "seed", "k_shot",
        "method_lock_sha256", "row_manifest_sha256", "feature_runtime_sha256",
        "protocol_policy_sha256",
    ):
        if before[field] != after[field]:
            raise SomphScoringError(f"SOMP-H before/after binding mismatch: {field}")
    truth_by_token = {row["query_token"]: row for row in truth["rows"]}
    old_tokens = {
        row["query_token"] for row in truth["rows"] if row["evaluation_role"] == "target_old"
    }
    all_tokens = set(truth_by_token)
    rows: list[dict[str, Any]] = []
    predictions_output: list[dict[str, Any]] = []
    for scenario in manifest["scenarios"]:
        before_tokens, before_predictions = _scenario_arrays(before_verified, scenario)
        after_tokens, after_predictions = _scenario_arrays(after_verified, scenario)
        if set(before_tokens.tolist()) != old_tokens:
            raise SomphScoringError("SOMP-H before artifact must contain matched old queries only")
        if set(after_tokens.tolist()) != all_tokens:
            raise SomphScoringError("SOMP-H after artifact must contain all old/new queries")
        before_truth = [truth_by_token[token] for token in before_tokens.tolist()]
        after_truth = [truth_by_token[token] for token in after_tokens.tolist()]
        role_by_handle = _validate_prediction_handles(
            before_predictions, truth, allowed_roles={"target_old"}
        )
        role_by_handle.update(
            _validate_prediction_handles(
                after_predictions, truth, allowed_roles={"target_old", "target_new"}
            )
        )
        before_true_handles = np.asarray(
            [row["true_class_handle"] for row in before_truth]
        ).astype(str)
        after_true_handles = np.asarray(
            [row["true_class_handle"] for row in after_truth]
        ).astype(str)
        after_old_mask = np.asarray(
            [row["evaluation_role"] == "target_old" for row in after_truth]
        )
        after_new_mask = ~after_old_mask
        old_before = _base._accuracy(before_predictions, before_true_handles)
        old_after = _base._accuracy(
            after_predictions[after_old_mask], after_true_handles[after_old_mask]
        )
        seen_new = _base._accuracy(
            after_predictions[after_new_mask], after_true_handles[after_new_mask]
        )
        old_class_before, old_class_count = _class_metrics(
            before_predictions, before_truth, manifest["old_tx_labels"]
        )
        old_class_after, _ = _class_metrics(
            after_predictions, after_truth, manifest["old_tx_labels"]
        )
        new_class_acc, new_class_count = _class_metrics(
            after_predictions, after_truth, manifest["new_tx_labels"]
        )
        old_class_forgetting = {
            label: old_class_before[label] - old_class_after[label]
            for label in manifest["old_tx_labels"]
        }
        old_pred_roles = np.asarray(
            [role_by_handle[value] for value in after_predictions[after_old_mask].tolist()]
        )
        new_pred_roles = np.asarray(
            [role_by_handle[value] for value in after_predictions[after_new_mask].tolist()]
        )
        old_to_new_count = int(np.sum(old_pred_roles == "target_new"))
        new_to_old_count = int(np.sum(new_pred_roles == "target_old"))
        before_confusion = _confusion_matrix(
            before_predictions, before_truth, manifest["old_tx_labels"]
        )
        after_confusion = _confusion_matrix(
            after_predictions,
            after_truth,
            [*manifest["old_tx_labels"], *manifest["new_tx_labels"]],
        )
        rows.append({
            "row_id": before["row_id"],
            "stage": "stage2c",
            "receiver_label": before["receiver"],
            "seed": before["seed"],
            "scenario": scenario,
            "k_shot": before["k_shot"],
            "new_class_count": manifest["new_class_count"],
            "old_query_count": len(before_tokens),
            "new_query_count": int(np.sum(after_new_mask)),
            "old_acc_before_increment": old_before,
            "old_acc_after_increment": old_after,
            "average_forgetting": old_before - old_after,
            "old_adaptation_gain": old_after - old_before,
            "min_old_class_acc_before": min(old_class_before.values()),
            "min_old_class_acc_after": min(old_class_after.values()),
            "seen_new_acc": seen_new,
            "min_seen_new_class_acc": min(new_class_acc.values()),
            "H_old_new": _base._harmonic(old_after, seen_new),
            "old_class_acc_before": old_class_before,
            "old_class_acc_after": old_class_after,
            "old_class_forgetting": old_class_forgetting,
            "old_class_count": old_class_count,
            "seen_new_class_acc": new_class_acc,
            "seen_new_class_count": new_class_count,
            "old_to_new_count": old_to_new_count,
            "old_to_new_rate": old_to_new_count / max(int(np.sum(after_old_mask)), 1),
            "new_to_old_count": new_to_old_count,
            "new_to_old_rate": new_to_old_count / max(int(np.sum(after_new_mask)), 1),
            "before_old_confusion_matrix_counts": before_confusion,
            "after_all_confusion_matrix_counts": after_confusion,
        })
        for state, tokens, values, ordered_truth in (
            ("before_registration", before_tokens, before_predictions, before_truth),
            ("after_registration", after_tokens, after_predictions, after_truth),
        ):
            for token, predicted, truth_row in zip(tokens.tolist(), values.tolist(), ordered_truth):
                predictions_output.append({
                    "row_id": before["row_id"],
                    "stage": "stage2c",
                    "registration_state": state,
                    "receiver_label": before["receiver"],
                    "scenario": scenario,
                    "query_token": token,
                    "evaluation_role": truth_row["evaluation_role"],
                    "transmitter_label": truth_row["transmitter_label"],
                    "true_class_handle": truth_row["true_class_handle"],
                    "predicted_class_handle": predicted,
                    "correct": int(predicted == truth_row["true_class_handle"]),
                })
    return _finalize(
        rows=rows,
        predictions=predictions_output,
        bindings=[before, after],
        verified=[before_verified, after_verified],
        scoring_audit=scoring_audit,
    )


def _finalize(
    *,
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    bindings: list[Mapping[str, Any]],
    verified: list[Mapping[str, Any]],
    scoring_audit: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows_payload = {"schema": FORMAL_ROWS_SCHEMA, "rows": rows}
    predictions_payload = {
        "schema": FORMAL_PREDICTIONS_SCHEMA,
        "predictions": predictions,
    }
    receipt = {
        "schema": SCORING_RECEIPT_SCHEMA,
        "status": "LOCAL_PROTOCOL_REPAIR_REQUIRED",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "row_id": bindings[0]["row_id"],
        "stage": rows[0]["stage"],
        "receiver": bindings[0]["receiver"],
        "seed": bindings[0]["seed"],
        "k_shot": bindings[0]["k_shot"],
        "new_class_count": rows[0]["new_class_count"],
        "protocol_policy_sha256": bindings[0]["protocol_policy_sha256"],
        "prediction_artifact_sha256_by_state": {
            binding["registration_state"]: item["artifact_sha256"]
            for binding, item in zip(bindings, verified)
        },
        "prediction_seal_sha256_by_state": {
            binding["registration_state"]: item["seal_sha256"]
            for binding, item in zip(bindings, verified)
        },
        "resource_audit_sha256_by_state": {
            binding["registration_state"]: scoring_audit["state_audit"][
                binding["registration_state"]
            ]["resource_audit_sha256"]
            for binding in bindings
        },
        "scoring_manifest_sha256": scoring_audit["scoring_manifest_sha256"],
        "truth_sidecar_sha256": scoring_audit["truth_sidecar_sha256"],
        "phase2_protocol_evidence_status": scoring_audit[
            "phase2_protocol_evidence_status"
        ],
        "evidence_manifest_sha256": scoring_audit["evidence_manifest_sha256"],
        "registration_pair_sha256": scoring_audit[
            "registration_pair_sha256"
        ],
        "state_evidence_audit": scoring_audit["state_audit"],
        "scenario_count": len(_base.FORMAL_LEO_WEAK_SCENARIOS),
        "formal_row_count": len(rows),
        "formal_prediction_count": len(predictions),
        "join_policy": "exact_scenario_opaque_query_token_after_prediction_freeze",
        "truth_join_after_all_predictions_verified": True,
        "scorer_output_must_not_feed_predictor": True,
        "formal_rows_sha256": _receipt_hashes(rows_payload),
        "formal_predictions_sha256": _receipt_hashes(predictions_payload),
    }
    if not all(
        math.isfinite(value)
        for row in rows
        for value in row.values()
        if isinstance(value, float)
    ):
        raise SomphScoringError("non-finite SOMP-H formal metric")
    return rows_payload, predictions_payload, receipt


write_somph_scoring_outputs_exclusive = _base.write_scoring_outputs_exclusive


__all__ = [
    "FORMAL_PREDICTIONS_SCHEMA",
    "FORMAL_ROWS_SCHEMA",
    "SCORING_MANIFEST_SCHEMA",
    "SCORING_RECEIPT_SCHEMA",
    "SOMPH_TRUTH_SIDECAR_SCHEMA",
    "SomphScoringError",
    "score_somph_registration_pair",
    "score_somph_stage2b",
    "write_somph_scoring_outputs_exclusive",
]
