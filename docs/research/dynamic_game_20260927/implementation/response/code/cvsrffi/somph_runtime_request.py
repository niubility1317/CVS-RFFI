"""Minimal fail-closed requests for isolated SOMP-H enrollment and apply.

These requests are intentionally not experiment manifests.  They contain only
the external trust roots and the small set of process-local controls required
to execute one isolated stage.  Matrix metadata, data locations, query
composition, executable selection, and scoring information are excluded.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from cvsrffi.phase2_runtime_contract import (
    PHASE2_FULL_CONTRACT,
    Phase2ContractError,
    validate_phase2_contract,
)


SOMPH_ENROLLMENT_REQUEST_SCHEMA = "cvs.phase2.somph_enrollment_request.v1"
SOMPH_APPLY_REQUEST_SCHEMA = "cvs.phase2.somph_apply_request.v1"
SOMPH_APPLY_BATCH_SIZE = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROW_HANDLE_RE = re.compile(r"row_[0-9a-f]{64}")
_LEAF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DEVICE_RE = re.compile(r"(?:cpu|cuda:(?:[0-9]|[12][0-9]|3[01]))")
_MAX_SUPPORT_BATCH_SIZE = 256

_ENROLLMENT_KEYS = {
    "schema",
    "package_seal_sha256",
    "head_output_leaf",
    "device",
    "support_batch_size",
    *PHASE2_FULL_CONTRACT.keys(),
}
_APPLY_KEYS = {
    "schema",
    "package_seal_sha256",
    "head_capsule_sha256",
    "head_enrollment_binding_sha256",
    "row_handle",
    "row_manifest_sha256",
    "prediction_output_leaf",
    "device",
    *PHASE2_FULL_CONTRACT.keys(),
}

# These tokens are forbidden both as extra keys and in user-controlled string
# values.  The official Phase2 contract is validated separately and is the
# sole permitted occurrence of its locked clean-unreachability vocabulary.
_FORBIDDEN_TOKENS = (
    "truth",
    "label",
    "role",
    "old",
    "new",
    "boundary",
    "query",
    "query_count",
    "query_size",
    "num_query",
    "quota",
    "order",
    "dataset",
    "cache",
    "raw",
    "clean",
    "path",
    "entrypoint",
    "entry_point",
    "runner",
    "callable",
    "executable",
    "command",
    "argv",
    "module",
    "script",
    "python",
    "main.py",
    ".exe",
    ".bat",
    ".cmd",
    ".ps1",
    ".sh",
    "loader",
    "scorer",
)


class SomphRuntimeRequestError(Phase2ContractError):
    """Raised before an isolated SOMP-H process opens its sealed package."""


def _fail(message: str) -> None:
    raise SomphRuntimeRequestError(message)


def _validate_exact_mapping(
    request: Mapping[str, Any],
    *,
    allowed_keys: set[str],
    schema: str,
) -> None:
    if not isinstance(request, Mapping):
        _fail("somph_runtime_request_not_object")
    if any(not isinstance(key, str) for key in request):
        _fail("somph_runtime_request_key_not_string")

    actual_keys = set(request)
    missing = sorted(allowed_keys - actual_keys)
    unknown = sorted(actual_keys - allowed_keys)
    if missing or unknown:
        _fail(
            "somph_runtime_request_schema:"
            f"missing={missing}:unknown={unknown}"
        )
    if request.get("schema") != schema:
        _fail("somph_runtime_request_schema_version")

    try:
        validate_phase2_contract(request, evidence_phase="none")
    except Phase2ContractError as exc:
        _fail(f"somph_runtime_request_phase2_contract:{exc}")


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"somph_runtime_request_sha256:{field}")
    return value


def _validate_leaf(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or _LEAF_RE.fullmatch(value) is None
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        _fail(f"somph_runtime_request_output_leaf:{field}")
    return value


def _validate_device(value: Any) -> str:
    if not isinstance(value, str) or _DEVICE_RE.fullmatch(value) is None:
        _fail("somph_runtime_request_device")
    return value


def _scan_user_controlled_strings(
    request: Mapping[str, Any],
    *,
    non_contract_keys: set[str],
) -> None:
    for key in non_contract_keys:
        value = request[key]
        lowered_key = key.lower()
        if any(token in lowered_key for token in _FORBIDDEN_TOKENS):
            _fail(f"somph_runtime_request_forbidden_key:{key}")
        if isinstance(value, str):
            lowered_value = value.lower()
            if any(token in lowered_value for token in _FORBIDDEN_TOKENS):
                _fail(f"somph_runtime_request_forbidden_value:{key}")


def validate_somph_enrollment_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an enrollment-only request and return a detached safe copy."""

    _validate_exact_mapping(
        request,
        allowed_keys=_ENROLLMENT_KEYS,
        schema=SOMPH_ENROLLMENT_REQUEST_SCHEMA,
    )
    _scan_user_controlled_strings(
        request,
        non_contract_keys=_ENROLLMENT_KEYS - set(PHASE2_FULL_CONTRACT),
    )
    package_seal = _validate_sha256(
        request["package_seal_sha256"], field="package_seal_sha256"
    )
    output_leaf = _validate_leaf(request["head_output_leaf"], field="head_output_leaf")
    device = _validate_device(request["device"])
    batch_size = request["support_batch_size"]
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= _MAX_SUPPORT_BATCH_SIZE
    ):
        _fail("somph_runtime_request_support_batch_size")

    return {
        "schema": SOMPH_ENROLLMENT_REQUEST_SCHEMA,
        "package_seal_sha256": package_seal,
        "head_output_leaf": output_leaf,
        "device": device,
        "support_batch_size": batch_size,
        **PHASE2_FULL_CONTRACT,
    }


def validate_somph_apply_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an apply-only request and return a detached safe copy.

    Apply batching is deliberately absent from the request surface.  Callers
    must use :data:`SOMPH_APPLY_BATCH_SIZE`, whose fixed value is one.
    """

    _validate_exact_mapping(
        request,
        allowed_keys=_APPLY_KEYS,
        schema=SOMPH_APPLY_REQUEST_SCHEMA,
    )
    _scan_user_controlled_strings(
        request,
        non_contract_keys=_APPLY_KEYS - set(PHASE2_FULL_CONTRACT),
    )
    package_seal = _validate_sha256(
        request["package_seal_sha256"], field="package_seal_sha256"
    )
    head_capsule = _validate_sha256(
        request["head_capsule_sha256"], field="head_capsule_sha256"
    )
    binding = _validate_sha256(
        request["head_enrollment_binding_sha256"],
        field="head_enrollment_binding_sha256",
    )
    row_handle = request["row_handle"]
    if (
        not isinstance(row_handle, str)
        or _ROW_HANDLE_RE.fullmatch(row_handle) is None
    ):
        _fail("somph_runtime_request_row_handle")
    row_manifest = _validate_sha256(
        request["row_manifest_sha256"],
        field="row_manifest_sha256",
    )
    output_leaf = _validate_leaf(
        request["prediction_output_leaf"], field="prediction_output_leaf"
    )
    device = _validate_device(request["device"])

    return {
        "schema": SOMPH_APPLY_REQUEST_SCHEMA,
        "package_seal_sha256": package_seal,
        "head_capsule_sha256": head_capsule,
        "head_enrollment_binding_sha256": binding,
        "row_handle": row_handle,
        "row_manifest_sha256": row_manifest,
        "prediction_output_leaf": output_leaf,
        "device": device,
        **PHASE2_FULL_CONTRACT,
    }


def validate_somph_runtime_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch an exact request schema without accepting aliases."""

    if not isinstance(request, Mapping):
        _fail("somph_runtime_request_not_object")
    schema = request.get("schema")
    if schema == SOMPH_ENROLLMENT_REQUEST_SCHEMA:
        return validate_somph_enrollment_request(request)
    if schema == SOMPH_APPLY_REQUEST_SCHEMA:
        return validate_somph_apply_request(request)
    _fail("somph_runtime_request_schema_version")
