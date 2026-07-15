"""Fail-closed Phase2 runtime and predictor-request contracts.

The strict predictor imports this module before it opens any IQ payload.  The
module deliberately has no dataset, training, Torch, or legacy-runner imports.
Pre-run isolation evidence and post-run access/seal evidence are separate: a
request cannot claim a hash for a filesystem access ledger that does not exist
until prediction has completed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
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
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OPAQUE_HANDLE_RE = re.compile(r"(?:cls|qid|sid|oid)_[0-9a-f]{32,64}")

PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS = (
    "sealed_inference_package_sha256",
    "package_root_sha256",
    "runtime_code_sha256",
    "artifact_member_allowlist_sha256",
    "os_isolation_mode",
    "os_isolation_attestation_sha256",
    "preopen_audit_status",
    "preopen_audit_receipt_sha256",
    "predict_score_process_isolation",
)

POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS = (
    *PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    "filesystem_access_audit_sha256",
    "filesystem_access_audit_status",
    "prediction_artifact_sha256",
    "prediction_seal_sha256",
)

# Compatibility name for callers that enumerate the complete formal evidence.
RUNTIME_EVIDENCE_REQUIRED_FIELDS = POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS

ALLOWED_OS_ISOLATION_MODES = {
    "container_readonly_mounts",
    "bwrap_readonly_mounts",
    "dedicated_uid_readonly_package",
    "equivalent_verified_isolation",
}

ARTIFACT_DESCRIPTOR_REQUIRED_KEYS = {
    "relative_path",
    "sha256",
    "size_bytes",
    "artifact_role",
    "schema",
}

PREDICTOR_REQUEST_REQUIRED_KEYS = {
    "schema_version",
    "request_id",
    "row_id",
    "stage",
    "receiver",
    "scenarios",
    "k_shot",
    "satellite_seed",
    "candidate_lock_sha256",
    "package_root_sha256",
    "runtime_code_sha256",
    "registered_class_count",
    "registered_classes",
    "support_artifacts",
    "query_artifacts",
    "checkpoint_artifact",
    "adapter_artifact",
    "head_artifact",
    "tta_policy",
    "tta_policy_sha256",
    "output_contract",
    "phase2_runtime_isolation_evidence",
    *PHASE2_FULL_CONTRACT.keys(),
}
PREDICTOR_REQUEST_ALLOWED_KEYS = set(PREDICTOR_REQUEST_REQUIRED_KEYS)

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
    "manysig",
    "manytx",
)

# The adaptive TTA runtime requires this guard in its config.  It may cross the
# predictor boundary only as a literal negative declaration.
NEGATIVE_ONLY_PREDICTOR_GUARD_KEYS = frozenset(
    {"uses_class_quota", "uses_query_role"}
)


class Phase2ContractError(ValueError):
    """Raised before Phase2 sample materialization when a contract fails."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.lower()) is not None


def _walk(value: Any, path: str = "request"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, str(key), child
            yield from _walk(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _validate_runtime_evidence(evidence: Any, *, phase: str) -> list[str]:
    if not isinstance(evidence, Mapping):
        return ["missing:phase2_runtime_isolation_evidence"]
    if phase not in {"pre_run", "post_run"}:
        return [f"invalid_runtime_evidence_phase:{phase}"]
    required = (
        PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS
        if phase == "pre_run"
        else POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS
    )
    errors: list[str] = []
    for field in required:
        value = evidence.get(field)
        if value in (None, "", []):
            errors.append(f"missing_runtime_evidence:{field}")
        elif field.endswith("_sha256") and not _is_sha256(value):
            errors.append(f"invalid_runtime_evidence_sha256:{field}")
    unexpected = sorted(set(evidence) - set(required))
    if unexpected:
        errors.append(f"unexpected_runtime_evidence:{','.join(unexpected)}")
    if evidence.get("preopen_audit_status") != "PASS":
        errors.append("runtime_evidence_not_pass:preopen_audit_status")
    if evidence.get("predict_score_process_isolation") is not True:
        errors.append("runtime_evidence_not_true:predict_score_process_isolation")
    if evidence.get("os_isolation_mode") not in ALLOWED_OS_ISOLATION_MODES:
        errors.append("runtime_evidence_unverified:os_isolation_mode")
    if phase == "post_run" and evidence.get("filesystem_access_audit_status") != "PASS":
        errors.append("runtime_evidence_not_pass:filesystem_access_audit_status")
    return errors


def validate_phase2_contract(
    record: Mapping[str, Any],
    *,
    require_runtime_evidence: bool | None = None,
    evidence_phase: str | None = None,
) -> None:
    """Validate all 12 Phase2 fields and the requested evidence phase.

    ``require_runtime_evidence`` is retained for existing callers.  ``True``
    means pre-run evidence; use ``evidence_phase='post_run'`` for a completed
    formal prediction artifact.
    """

    if evidence_phase is None:
        evidence_phase = "pre_run" if require_runtime_evidence is not False else "none"
    errors: list[str] = []
    for field, expected in PHASE2_FULL_CONTRACT.items():
        if field not in record:
            errors.append(f"missing:{field}")
            continue
        actual = record[field]
        if expected is False:
            if actual is not False:
                errors.append(f"must_be_false:{field}")
        elif actual != expected:
            errors.append(f"unexpected_value:{field}")

    if DEPRECATED_QUERY_CLASS_COUNT_FIELD in record:
        errors.append(f"deprecated_field:{DEPRECATED_QUERY_CLASS_COUNT_FIELD}")
    if evidence_phase != "none":
        errors.extend(
            _validate_runtime_evidence(
                record.get("phase2_runtime_isolation_evidence"), phase=evidence_phase
            )
        )
    if errors:
        raise Phase2ContractError(";".join(errors))


def _validate_relative_path(value: Any, *, path: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Phase2ContractError(f"artifact_relative_path_invalid:{path}")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise Phase2ContractError(f"artifact_relative_path_invalid:{path}")


def _validate_artifact_descriptor(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping):
        raise Phase2ContractError(f"artifact_descriptor_not_object:{path}")
    missing = sorted(ARTIFACT_DESCRIPTOR_REQUIRED_KEYS - set(value))
    unknown = sorted(set(value) - ARTIFACT_DESCRIPTOR_REQUIRED_KEYS)
    if missing or unknown:
        raise Phase2ContractError(
            f"artifact_descriptor_schema:{path}:missing={missing}:unknown={unknown}"
        )
    _validate_relative_path(value["relative_path"], path=f"{path}.relative_path")
    if not _is_sha256(value["sha256"]):
        raise Phase2ContractError(f"artifact_descriptor_sha256:{path}")
    if not isinstance(value["size_bytes"], int) or value["size_bytes"] < 0:
        raise Phase2ContractError(f"artifact_descriptor_size:{path}")
    for key in ("artifact_role", "schema"):
        if not isinstance(value[key], str) or not value[key]:
            raise Phase2ContractError(f"artifact_descriptor_{key}:{path}")


def validate_predictor_request(request: Mapping[str, Any]) -> None:
    """Reject truth, role, quota, raw-path, and legacy-build signals pre-open."""

    validate_phase2_contract(request, evidence_phase="pre_run")
    missing = sorted(PREDICTOR_REQUEST_REQUIRED_KEYS - set(request))
    unknown = sorted(set(request) - PREDICTOR_REQUEST_ALLOWED_KEYS)
    if missing or unknown:
        raise Phase2ContractError(
            f"predictor_request_schema:missing={missing}:unknown={unknown}"
        )
    if request.get("schema_version") != "cvs.phase2.predict_request.v2":
        raise Phase2ContractError("predictor_request_schema_version")
    if request.get("stage") not in {"stage2b", "stage2c"}:
        raise Phase2ContractError("predictor_request_stage")
    formal_scenarios = [
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    ]
    if request.get("scenarios") != formal_scenarios:
        raise Phase2ContractError("predictor_request_scenarios")
    if not isinstance(request.get("k_shot"), int) or request["k_shot"] < 1:
        raise Phase2ContractError("k_shot_must_be_positive_int")
    for field in (
        "candidate_lock_sha256",
        "package_root_sha256",
        "runtime_code_sha256",
        "tta_policy_sha256",
    ):
        if not _is_sha256(request.get(field)):
            raise Phase2ContractError(f"predictor_request_sha256:{field}")

    evidence = request["phase2_runtime_isolation_evidence"]
    for field in ("package_root_sha256", "runtime_code_sha256"):
        if request[field] != evidence[field]:
            raise Phase2ContractError(f"runtime_evidence_cross_digest_mismatch:{field}")

    registered_count = request.get("registered_class_count")
    registered_classes = request.get("registered_classes")
    if not isinstance(registered_count, int) or registered_count < 1:
        raise Phase2ContractError("registered_class_count_must_be_positive_int")
    if not isinstance(registered_classes, list) or len(registered_classes) != registered_count:
        raise Phase2ContractError("registered_class_registry_size_mismatch")
    seen_handles: set[str] = set()
    for expected_index, item in enumerate(registered_classes):
        if not isinstance(item, Mapping) or set(item) != {"class_index", "class_handle"}:
            raise Phase2ContractError("registered_class_registry_schema")
        if item.get("class_index") != expected_index:
            raise Phase2ContractError("registered_class_registry_index_order")
        handle = item.get("class_handle")
        if not isinstance(handle, str) or OPAQUE_HANDLE_RE.fullmatch(handle) is None:
            raise Phase2ContractError("registered_class_handle_not_opaque")
        if handle in seen_handles:
            raise Phase2ContractError("registered_class_handle_duplicate")
        seen_handles.add(handle)

    for field in ("checkpoint_artifact", "adapter_artifact", "head_artifact"):
        _validate_artifact_descriptor(request[field], path=f"request.{field}")
    for field, prefix in (
        ("support_artifacts", "support:"),
        ("query_artifacts", "query:"),
    ):
        descriptors = request[field]
        if not isinstance(descriptors, list) or len(descriptors) != len(formal_scenarios):
            raise Phase2ContractError(f"artifact_descriptor_list:{field}")
        for index, descriptor in enumerate(descriptors):
            _validate_artifact_descriptor(
                descriptor, path=f"request.{field}[{index}]"
            )
            if descriptor["artifact_role"] != prefix + formal_scenarios[index]:
                raise Phase2ContractError(f"artifact_descriptor_role_order:{field}")
    output = request["output_contract"]
    if not isinstance(output, Mapping) or set(output) != {
        "schema",
        "relative_path",
        "sealed_immutable_required",
    }:
        raise Phase2ContractError("output_contract_schema")
    _validate_relative_path(output["relative_path"], path="request.output_contract.relative_path")
    if output.get("schema") != "cvs.phase2.prediction.v2":
        raise Phase2ContractError("output_contract_version")
    if output.get("sealed_immutable_required") is not True:
        raise Phase2ContractError("output_contract_must_be_immutable")

    for path, key, value in _walk(request):
        lowered_key = key.lower()
        if key not in PHASE2_FULL_CONTRACT and any(
            token in lowered_key for token in FORBIDDEN_PREDICTOR_KEY_TOKENS
        ):
            if key not in NEGATIVE_ONLY_PREDICTOR_GUARD_KEYS or value is not False:
                raise Phase2ContractError(f"forbidden_predictor_key:{path}")
        if isinstance(value, str):
            lowered_value = value.lower()
            if any(token in lowered_value for token in FORBIDDEN_PREDICTOR_VALUE_TOKENS):
                raise Phase2ContractError(f"forbidden_predictor_value:{path}")


def classify_legacy_phase2_record(record: Mapping[str, Any]) -> str:
    """Separate unverified history from confirmed protocol invalidity."""

    if any(
        record.get(field) is True
        for field in (
            "clean_sample_access",
            "clean_derived_signal_access",
            "phase2_clean_dataset_reachable",
            "phase2_clean_cache_reachable",
            "phase2_clean_control_flow_reachable",
        )
    ):
        return "PROTOCOL_INVALID_FOR_PHASE2"
    if any(
        record.get(field) is True
        for field in (
            "phase2_query_role_oracle_access",
            "phase2_query_true_batch_class_count_access",
            "phase2_query_class_quota_access",
            "phase2_query_batch_global_assignment",
        )
    ):
        return "PROTOCOL_INVALID_FOR_DEPLOYMENT"
    try:
        validate_phase2_contract(record, evidence_phase="post_run")
    except Phase2ContractError:
        return "UNVERIFIED_UNDER_CURRENT_PROTOCOL"
    return "PROTOCOL_VALID"
