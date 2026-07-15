"""Fail-closed Phase2 runtime and predictor-request contract.

This module is intentionally free of dataset, training, Torch, and legacy runner
imports so the strict predictor can validate its request before materializing IQ.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


PHASE2_BASE_CONTRACT: dict[str, Any] = {
    "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
    "clean_sample_access": False,
    "clean_derived_signal_access": False,
}

PHASE2_CLEAN_RUNTIME_CONTRACT: dict[str, Any] = {
    "phase2_clean_dataset_reachable": False,
    "phase2_clean_cache_reachable": False,
    "phase2_clean_control_flow_reachable": False,
    "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
}

PHASE2_QUERY_DECISION_CONTRACT: dict[str, Any] = {
    "phase2_query_decision_policy": "per_sample_all_registered_classes",
    "phase2_query_role_oracle_access": False,
    "phase2_query_true_batch_class_count_access": False,
    "phase2_query_class_quota_access": False,
    "phase2_query_batch_global_assignment": False,
}

PHASE2_FULL_CONTRACT: dict[str, Any] = {
    **PHASE2_BASE_CONTRACT,
    **PHASE2_CLEAN_RUNTIME_CONTRACT,
    **PHASE2_QUERY_DECISION_CONTRACT,
}

DEPRECATED_QUERY_CLASS_COUNT_FIELD = "phase2_query_class_count_access"

RUNTIME_EVIDENCE_REQUIRED_FIELDS = (
    "sealed_inference_package_sha256",
    "package_root_sha256",
    "runtime_code_sha256",
    "artifact_member_allowlist_sha256",
    "filesystem_access_audit_sha256",
    "os_isolation_mode",
    "preopen_audit_status",
    "predict_score_process_isolation",
)

ALLOWED_OS_ISOLATION_MODES = {
    "container_readonly_mounts",
    "bwrap_readonly_mounts",
    "dedicated_uid_readonly_package",
    "equivalent_verified_isolation",
}

PREDICTOR_REQUEST_ALLOWED_KEYS = {
    "schema_version",
    "request_id",
    "row_id",
    "receiver",
    "scenario",
    "k_shot",
    "satellite_seed",
    "candidate_lock_sha256",
    "package_root_sha256",
    "runtime_code_sha256",
    "registered_class_count",
    "registered_classes",
    "support_artifact",
    "query_artifact",
    "checkpoint_artifact",
    "adapter_artifact",
    "head_artifact",
    "tta_policy",
    "tta_policy_sha256",
    "output_contract",
    "phase2_runtime_isolation_evidence",
    *PHASE2_FULL_CONTRACT.keys(),
}

FORBIDDEN_PREDICTOR_KEY_TOKENS = (
    "truth",
    "ground_truth",
    "dataset_role",
    "evaluation_role",
    "query_role",
    "query_per_tx",
    "query_class_histogram",
    "true_batch_class",
    "class_quota",
    "label_order",
    "class_block",
    "raw_label",
    "tx_id",
    "build_spec",
    "source_plan",
    "truth_sidecar",
    "clean_cache",
    "clean_dataset",
)

FORBIDDEN_PREDICTOR_VALUE_TOKENS = (
    "dataset_wigsig",
    ".pkl",
    "build_spec",
    "truth_sidecar",
    "clean_cache",
)


class Phase2ContractError(ValueError):
    """Raised before Phase2 sample materialization when a contract fails."""


def _is_exact_false(value: Any) -> bool:
    return value is False


def _walk(value: Any, path: str = "request"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, str(key), child
            yield from _walk(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def validate_phase2_contract(
    record: Mapping[str, Any],
    *,
    require_runtime_evidence: bool = True,
) -> None:
    """Validate all 12 Phase2 fields and the runtime evidence bundle."""

    errors: list[str] = []
    for field, expected in PHASE2_FULL_CONTRACT.items():
        if field not in record:
            errors.append(f"missing:{field}")
            continue
        actual = record[field]
        if expected is False:
            if not _is_exact_false(actual):
                errors.append(f"must_be_false:{field}")
        elif actual != expected:
            errors.append(f"unexpected_value:{field}")

    if DEPRECATED_QUERY_CLASS_COUNT_FIELD in record:
        errors.append(f"deprecated_field:{DEPRECATED_QUERY_CLASS_COUNT_FIELD}")

    if require_runtime_evidence:
        evidence = record.get("phase2_runtime_isolation_evidence")
        if not isinstance(evidence, Mapping):
            errors.append("missing:phase2_runtime_isolation_evidence")
        else:
            for field in RUNTIME_EVIDENCE_REQUIRED_FIELDS:
                if evidence.get(field) in (None, "", []):
                    errors.append(f"missing_runtime_evidence:{field}")
                elif field.endswith("_sha256") and re.fullmatch(
                    r"[0-9a-f]{64}", str(evidence.get(field)).lower()
                ) is None:
                    errors.append(f"invalid_runtime_evidence_sha256:{field}")
            if evidence.get("preopen_audit_status") != "PASS":
                errors.append("runtime_evidence_not_pass:preopen_audit_status")
            if evidence.get("predict_score_process_isolation") is not True:
                errors.append("runtime_evidence_not_true:predict_score_process_isolation")
            if evidence.get("os_isolation_mode") not in ALLOWED_OS_ISOLATION_MODES:
                errors.append("runtime_evidence_unverified:os_isolation_mode")

    if errors:
        raise Phase2ContractError(";".join(errors))


def validate_predictor_request(request: Mapping[str, Any]) -> None:
    """Reject truth, role, quota, raw-path, and legacy-build signals pre-open."""

    validate_phase2_contract(request, require_runtime_evidence=True)
    unknown = sorted(set(request) - PREDICTOR_REQUEST_ALLOWED_KEYS)
    if unknown:
        raise Phase2ContractError(f"predictor_request_unknown_keys:{','.join(unknown)}")

    registered_count = request.get("registered_class_count")
    registered_classes = request.get("registered_classes")
    if not isinstance(registered_count, int) or registered_count < 1:
        raise Phase2ContractError("registered_class_count_must_be_positive_int")
    if not isinstance(registered_classes, list) or len(registered_classes) != registered_count:
        raise Phase2ContractError("registered_class_registry_size_mismatch")

    for path, key, value in _walk(request):
        lowered_key = key.lower()
        if key not in PHASE2_FULL_CONTRACT and any(
            token in lowered_key for token in FORBIDDEN_PREDICTOR_KEY_TOKENS
        ):
            raise Phase2ContractError(f"forbidden_predictor_key:{path}")
        if isinstance(value, str):
            lowered_value = value.lower()
            if any(token in lowered_value for token in FORBIDDEN_PREDICTOR_VALUE_TOKENS):
                raise Phase2ContractError(f"forbidden_predictor_value:{path}")


def classify_legacy_phase2_record(record: Mapping[str, Any]) -> str:
    """Separate unverified historical evidence from confirmed protocol invalidity."""

    confirmed_clean = any(
        record.get(field) is True
        for field in (
            "clean_sample_access",
            "clean_derived_signal_access",
            "phase2_clean_dataset_reachable",
            "phase2_clean_cache_reachable",
            "phase2_clean_control_flow_reachable",
        )
    )
    if confirmed_clean:
        return "PROTOCOL_INVALID_FOR_PHASE2"
    try:
        validate_phase2_contract(record, require_runtime_evidence=True)
    except Phase2ContractError:
        return "PHASE2_RUNTIME_ISOLATION_UNVERIFIED"
    return "PHASE2_RUNTIME_ISOLATION_VERIFIED"
