"""Licensed, non-promotable role-Oracle decoder for frozen D92 scores.

The decoder receives one already-computed score row per query.  It derives a
normal all-registered-class prediction and a licensed upper-bound prediction
from the *same* score matrix.  The only extra information accepted by the
upper-bound branch is one ``old``/``new`` role token per row; no transmitter
identity, truth label, class quota, or batch class count is accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import numpy as np


ROLE_CAPSULE_SCHEMA = "cvs.phase2.d92.licensed_role_oracle.role_only.v1"
OUTPUT_SCHEMA = "cvs.phase2.d92.licensed_role_oracle.decoding.v1"
LICENSE_STATUS = "LICENSED_ORACLE_UPPER_BOUND_NON_PROMOTABLE"
_CAPSULE_KEYS = frozenset({"schema", "rows"})
_CAPSULE_ROW_KEYS = frozenset({"query_token", "evaluation_role"})
_VALID_ROLES = frozenset({"target_old", "target_new"})
_TRUTH_SCHEMA = "cvs.phase2.query_truth_sidecar.v2"
_TRUTH_TOP_KEYS = frozenset({"schema", "stage", "receiver", "seed", "rows"})
_TRUTH_REQUIRED_ROW_KEYS = frozenset(
    {
        "query_token",
        "true_class_index",
        "true_class_handle",
        "transmitter_label",
        "evaluation_role",
        "receiver_label",
    }
)
_TRUTH_OPTIONAL_ROW_KEYS = frozenset(
    {"day_label", "signal_label", "physical_sample_id", "scenario"}
)
_QUERY_TOKEN_RE = re.compile(r"qid_[0-9a-f]{32,64}")


class D92LicensedRoleOracleError(ValueError):
    """Raised when the licensed role-only decoding contract is violated."""


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def project_and_seal_d92_role_only_capsule(
    truth_sidecar_path: str | Path, role_capsule_path: str | Path
) -> dict[str, Any]:
    """Project a scorer-side truth sidecar into an exclusive role-only capsule.

    The returned receipt carries file hashes and the path.  The source truth
    hash is also embedded so the exact three-field capsule binds its source.
    """

    truth_path = Path(truth_sidecar_path)
    output_path = Path(role_capsule_path)
    try:
        truth_bytes = truth_path.read_bytes()
        truth = json.loads(truth_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D92LicensedRoleOracleError("truth sidecar is not readable UTF-8 JSON") from exc
    if type(truth) is not dict or set(truth) != _TRUTH_TOP_KEYS:
        raise D92LicensedRoleOracleError("truth sidecar top-level exact schema drift")
    if truth["schema"] != _TRUTH_SCHEMA or truth["stage"] not in {
        "stage2b",
        "stage2c",
    }:
        raise D92LicensedRoleOracleError("truth sidecar identity drift")
    if (
        type(truth["receiver"]) is not str
        or not truth["receiver"]
        or type(truth["seed"]) is not int
        or type(truth["rows"]) is not list
        or not truth["rows"]
    ):
        raise D92LicensedRoleOracleError("truth sidecar metadata is malformed")

    seen_tokens: set[str] = set()
    for row in truth["rows"]:
        if type(row) is not dict:
            raise D92LicensedRoleOracleError("truth sidecar row must be an object")
        keys = set(row)
        if not _TRUTH_REQUIRED_ROW_KEYS.issubset(keys) or not keys.issubset(
            _TRUTH_REQUIRED_ROW_KEYS | _TRUTH_OPTIONAL_ROW_KEYS
        ):
            raise D92LicensedRoleOracleError("truth sidecar row exact schema drift")
        token = row["query_token"]
        if type(token) is not str or _QUERY_TOKEN_RE.fullmatch(token) is None:
            raise D92LicensedRoleOracleError("truth query_token is not opaque")
        if token in seen_tokens:
            raise D92LicensedRoleOracleError("duplicate truth query_token")
        seen_tokens.add(token)
        role = row["evaluation_role"]
        if role not in _VALID_ROLES:
            raise D92LicensedRoleOracleError("truth evaluation_role contamination")

    capsule = {
        "schema": ROLE_CAPSULE_SCHEMA,
        "rows": [
            {
                "query_token": row["query_token"],
                "evaluation_role": row["evaluation_role"],
            }
            for row in truth["rows"]
        ],
    }
    capsule_bytes = _canonical_json(capsule)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(output_path, flags, 0o600)
    except OSError as exc:
        raise D92LicensedRoleOracleError(
            "role-only capsule path must be new and exclusively creatable"
        ) from exc
    try:
        view = memoryview(capsule_bytes)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short role-only capsule write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(output_path, 0o444)
    if stat.S_IMODE(os.lstat(output_path).st_mode) & 0o222:
        raise D92LicensedRoleOracleError("role-only capsule is not read-only")
    return {
        "path": str(output_path),
        "source_truth_sha256": _sha256(truth_bytes),
        "capsule_sha256": _sha256(capsule_bytes),
        "readonly": True,
    }


def _classes_and_old_prefix(
    classes: Sequence[Any], old_registry_prefix: Sequence[Any]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if isinstance(classes, (str, bytes)) or isinstance(
        old_registry_prefix, (str, bytes)
    ):
        raise D92LicensedRoleOracleError("classes must be explicit sequences")
    try:
        registry = tuple(classes)
        old_prefix = tuple(old_registry_prefix)
    except TypeError as exc:
        raise D92LicensedRoleOracleError(
            "classes and old_registry_prefix must be finite sequences"
        ) from exc
    if len(registry) < 2 or not 0 < len(old_prefix) < len(registry):
        raise D92LicensedRoleOracleError(
            "old_registry_prefix must define a non-empty proper registry prefix"
        )
    try:
        unique = len(set(registry)) == len(registry)
    except TypeError as exc:
        raise D92LicensedRoleOracleError("class labels must be hashable") from exc
    if not unique:
        raise D92LicensedRoleOracleError("registered class labels must be unique")
    if registry[: len(old_prefix)] != old_prefix:
        raise D92LicensedRoleOracleError(
            "old_registry_prefix must exactly match the registered-class prefix"
        )
    return registry, old_prefix


def _query_tokens(query_tokens: Sequence[str], rows: int) -> tuple[str, ...]:
    if isinstance(query_tokens, (str, bytes)) or not isinstance(query_tokens, Sequence):
        raise D92LicensedRoleOracleError("query_tokens must be an explicit sequence")
    tokens = tuple(query_tokens)
    if (
        len(tokens) != rows
        or any(type(token) is not str for token in tokens)
        or any(_QUERY_TOKEN_RE.fullmatch(token) is None for token in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        raise D92LicensedRoleOracleError(
            "query_tokens must be unique opaque tokens aligned with score rows"
        )
    return tokens


def _role_tokens(
    role_only_capsule: Mapping[str, Any], query_tokens: tuple[str, ...]
) -> tuple[str, ...]:
    if not isinstance(role_only_capsule, Mapping):
        raise D92LicensedRoleOracleError("role-only capsule must be a mapping")
    if set(role_only_capsule) != _CAPSULE_KEYS:
        raise D92LicensedRoleOracleError(
            "role-only capsule top-level exact schema drift"
        )
    if role_only_capsule["schema"] != ROLE_CAPSULE_SCHEMA:
        raise D92LicensedRoleOracleError("role-only capsule schema drift")
    rows = role_only_capsule["rows"]
    if type(rows) is not list or not rows:
        raise D92LicensedRoleOracleError("role-only capsule rows must be non-empty")
    by_token: dict[str, str] = {}
    for row in rows:
        if type(row) is not dict or set(row) != _CAPSULE_ROW_KEYS:
            raise D92LicensedRoleOracleError("role-only capsule row exact schema drift")
        token = row["query_token"]
        role = row["evaluation_role"]
        if type(token) is not str or _QUERY_TOKEN_RE.fullmatch(token) is None:
            raise D92LicensedRoleOracleError("role-only capsule query_token drift")
        if token in by_token:
            raise D92LicensedRoleOracleError("duplicate role-only capsule query_token")
        if role not in _VALID_ROLES:
            raise D92LicensedRoleOracleError("role-only capsule evaluation_role drift")
        by_token[token] = role
    if set(by_token) != set(query_tokens):
        raise D92LicensedRoleOracleError(
            "role-only capsule has missing or extra query tokens"
        )
    return tuple(by_token[token] for token in query_tokens)


def decode_d92_licensed_role_oracle(
    scores: np.ndarray,
    classes: Sequence[Any],
    old_registry_prefix: Sequence[Any],
    query_tokens: Sequence[str],
    role_only_capsule: Mapping[str, Any],
) -> dict[str, Any]:
    """Decode paired non-Oracle and role-Oracle predictions from frozen scores.

    Ties follow NumPy's deterministic first-index rule in both branches.  Each
    upper-bound prediction depends only on its own score row and role token.
    """

    registry, old_prefix = _classes_and_old_prefix(classes, old_registry_prefix)
    matrix = np.asarray(scores)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(registry)
        or matrix.shape[0] < 1
        or matrix.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(matrix).all()
    ):
        raise D92LicensedRoleOracleError(
            "scores must be a non-empty finite numeric rows-by-registry matrix"
        )
    aligned_query_tokens = _query_tokens(query_tokens, int(matrix.shape[0]))
    roles = _role_tokens(role_only_capsule, aligned_query_tokens)

    baseline_indices = np.argmax(matrix, axis=1).astype(np.int64, copy=False)
    old_size = len(old_prefix)
    oracle_indices = np.empty(matrix.shape[0], dtype=np.int64)
    for row_index, role in enumerate(roles):
        row = matrix[row_index]
        if role == "target_old":
            oracle_indices[row_index] = int(np.argmax(row[:old_size]))
        else:
            oracle_indices[row_index] = old_size + int(np.argmax(row[old_size:]))

    baseline_predictions = tuple(registry[int(index)] for index in baseline_indices)
    oracle_predictions = tuple(registry[int(index)] for index in oracle_indices)
    audit = {
        "license_status": LICENSE_STATUS,
        "promotion_eligible": False,
        "protocol_legal_performance_claim": False,
        "paired_baseline_from_identical_scores": True,
        "query_role_oracle_access": True,
        "query_role_only_access": True,
        "query_tx_id_access": False,
        "query_truth_label_access": False,
        "query_class_quota_access": False,
        "query_true_batch_class_count_access": False,
        "query_batch_reassignment": False,
        "query_independent_decisions": True,
        "oracle_scope": "argmax_within_role_registry_only",
    }
    return {
        "schema": OUTPUT_SCHEMA,
        "license_status": LICENSE_STATUS,
        "classes": registry,
        "old_registry_prefix": old_prefix,
        "baseline_indices": tuple(int(index) for index in baseline_indices),
        "baseline_predictions": baseline_predictions,
        "role_oracle_indices": tuple(int(index) for index in oracle_indices),
        "role_oracle_predictions": oracle_predictions,
        "audit": audit,
    }


def decode_d92_all_registry_baseline(
    scores: np.ndarray, classes: Sequence[Any]
) -> tuple[Any, ...]:
    """Decode the no-Oracle baseline without opening any role capsule."""

    if isinstance(classes, (str, bytes)):
        raise D92LicensedRoleOracleError("classes must be an explicit sequence")
    registry = tuple(classes)
    try:
        unique = len(set(registry)) == len(registry)
    except TypeError as exc:
        raise D92LicensedRoleOracleError("class labels must be hashable") from exc
    matrix = np.asarray(scores)
    if (
        len(registry) < 2
        or not unique
        or matrix.ndim != 2
        or matrix.shape != (matrix.shape[0], len(registry))
        or matrix.shape[0] < 1
        or matrix.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(matrix).all()
    ):
        raise D92LicensedRoleOracleError("invalid all-registry baseline inputs")
    return tuple(registry[int(index)] for index in np.argmax(matrix, axis=1))


__all__ = [
    "D92LicensedRoleOracleError",
    "LICENSE_STATUS",
    "OUTPUT_SCHEMA",
    "ROLE_CAPSULE_SCHEMA",
    "decode_d92_all_registry_baseline",
    "decode_d92_licensed_role_oracle",
    "project_and_seal_d92_role_only_capsule",
]
