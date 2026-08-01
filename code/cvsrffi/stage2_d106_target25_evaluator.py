"""Strict, truth-free single-state four-arm evaluator for D106 Target25.

Feature rows enter only through a SHA-pinned external NPZ/receipt loader, and
row identity enters only through a SHA-pinned canonical plan-state loader.
The evaluator itself only orchestrates the public RDCE, Student-t qKNN, and
RCMR APIs; it contains no alternative DA or head mathematics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cvsrffi.stage2_d106_k_conditioned_router import TARGET25_ROW_SCHEMA
from cvsrffi.stage2_d106_rcmr_2v_qknn import (
    D106RCMR2VBinding,
    build_d106_rcmr_2v_state,
    prepare_d106_rcmr_2v_scoring_context,
    score_d106_rcmr_2v_query,
)
from cvsrffi.stage2_d106_rdce_asset import D106RDCEAsset, Z_DIM
from cvsrffi.stage2_d106_rdce_runtime import (
    D106RDCESupportRows,
    fit_d106_rdce_runtime,
    prepare_d106_rdce_scoring_context,
    transform_d106_rdce_query,
    transform_d106_rdce_zid,
)
from cvsrffi.stage2_lpo_rc_qknn import validate_lpo_rc_physical_id_disjointness
from cvsrffi.stage2_zid_student_t_qknn import (
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
ALLOWED_K = (1, 5, 10)
PAIRED_FEATURE_SCHEMA = "cvs.phase2.d106.target25.paired_features.v1"
PLAN_STATE_SCHEMA = "cvs.phase2.d106.target25.plan_state.v1"
FEATURE_MEMBERS = (
    "support_plus",
    "support_signed",
    "query_plus",
    "query_signed",
    "support_physical_ids",
    "query_physical_ids",
)
FEATURE_ARRAY_NAMES = FEATURE_MEMBERS[:4]
PAIRED_FEATURE_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "feature_archive_name",
        "feature_archive_sha256",
        "archive_members",
        "received_iq_package_seal_sha256",
        "checkpoint_sha256",
        "runtime_sha256",
        "forward_receipt_sha256",
        "ordered_support_physical_ids_sha256",
        "ordered_query_physical_ids_sha256",
        "array_receipts",
        "query_truth_access",
        "query_role_access",
        "performance_metric_access",
    }
)
PLAN_STATE_KEYS = frozenset(
    {
        "schema",
        "row_id",
        "receiver",
        "scene",
        "active_k",
        "registered_classes",
        "capsule_id",
        "split_id",
        "validator_receipt_sha256",
        "seed",
        "support_physical_root_sha256",
        "query_physical_root_sha256",
        "paired_feature_receipt_sha256",
    }
)

_FEATURE_LOADER_TOKEN = object()
_PLAN_LOADER_TOKEN = object()


class D106Target25EvaluatorError(ValueError):
    """Raised when strict Target25 feature, plan, or prediction closure drifts."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D106Target25EvaluatorError("canonical JSON payload is invalid") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D106Target25EvaluatorError(f"{name} must be a lowercase SHA256")
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise D106Target25EvaluatorError(f"{name} must be non-empty builtin text")
    return value


def _tokens(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise D106Target25EvaluatorError(f"{name} must be an ordered sequence")
    result = tuple(value)
    if not result or any(type(item) is not str or not item for item in result):
        raise D106Target25EvaluatorError(
            f"{name} must contain non-empty builtin strings"
        )
    if len(set(result)) != len(result):
        raise D106Target25EvaluatorError(f"{name} must contain unique values")
    return result


def _float32_rows(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise D106Target25EvaluatorError(f"{name} must be a numpy array")
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] != Z_DIM
        or not np.isfinite(value).all()
    ):
        raise D106Target25EvaluatorError(
            f"{name} must be finite float32 [N,{Z_DIM}]"
        )
    return np.ascontiguousarray(value, dtype=np.float32)


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _ordered_root(values: Sequence[str]) -> str:
    return _sha256(list(values))


def _physical_root(values: Sequence[str]) -> str:
    return _sha256(sorted(values))


def _regular_new_path(path: str | Path, name: str) -> Path:
    result = Path(path)
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"immutable {name} path already exists: {result}")
    if not result.parent.is_dir() or result.parent.is_symlink():
        raise D106Target25EvaluatorError(f"{name} parent must be a regular directory")
    return result


def _regular_existing_path(path: str | Path, name: str) -> Path:
    result = Path(path)
    if not result.is_file() or result.is_symlink():
        raise D106Target25EvaluatorError(f"{name} must be a regular non-symlink file")
    return result


@dataclass(frozen=True, slots=True)
class D106PairedFeatureRows:
    support_plus: np.ndarray
    support_signed: np.ndarray
    query_plus: np.ndarray
    query_signed: np.ndarray
    support_physical_ids: tuple[str, ...]
    query_physical_ids: tuple[str, ...]
    received_iq_package_seal_sha256: str
    checkpoint_sha256: str
    runtime_sha256: str
    forward_receipt_sha256: str
    feature_archive_name: str
    feature_archive_sha256: str
    receipt_sha256: str
    _loader_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._loader_token is not _FEATURE_LOADER_TOKEN:
            raise D106Target25EvaluatorError(
                "paired features require strict-loader construction"
            )
        support_plus = _float32_rows(self.support_plus, "support_plus")
        support_signed = _float32_rows(self.support_signed, "support_signed")
        query_plus = _float32_rows(self.query_plus, "query_plus")
        query_signed = _float32_rows(self.query_signed, "query_signed")
        if support_plus.shape != support_signed.shape or query_plus.shape != query_signed.shape:
            raise D106Target25EvaluatorError("paired feature shape closure drift")
        if not np.array_equal(
            support_plus, np.maximum(support_signed, np.float32(0.0))
        ) or not np.array_equal(
            query_plus, np.maximum(query_signed, np.float32(0.0))
        ):
            raise D106Target25EvaluatorError("paired plus views must equal ReLU(signed)")
        support_ids = _tokens(self.support_physical_ids, "support_physical_ids")
        query_ids = _tokens(self.query_physical_ids, "query_physical_ids")
        if len(support_ids) != len(support_plus) or len(query_ids) != len(query_plus):
            raise D106Target25EvaluatorError("paired feature physical-ID closure drift")
        validate_lpo_rc_physical_id_disjointness(support_ids, query_ids)
        for name in (
            "received_iq_package_seal_sha256",
            "checkpoint_sha256",
            "runtime_sha256",
            "forward_receipt_sha256",
            "feature_archive_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "feature_archive_name",
            _text(self.feature_archive_name, "feature_archive_name"),
        )
        object.__setattr__(self, "support_plus", _immutable_array(support_plus))
        object.__setattr__(self, "support_signed", _immutable_array(support_signed))
        object.__setattr__(self, "query_plus", _immutable_array(query_plus))
        object.__setattr__(self, "query_signed", _immutable_array(query_signed))
        object.__setattr__(self, "support_physical_ids", support_ids)
        object.__setattr__(self, "query_physical_ids", query_ids)
        if self.receipt_sha256 != _sha256(self.receipt_payload):
            raise D106Target25EvaluatorError("paired-feature typed receipt drift")

    @property
    def support_physical_root_sha256(self) -> str:
        return _physical_root(self.support_physical_ids)

    @property
    def query_physical_root_sha256(self) -> str:
        return _physical_root(self.query_physical_ids)

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": PAIRED_FEATURE_SCHEMA,
            "feature_archive_name": self.feature_archive_name,
            "feature_archive_sha256": self.feature_archive_sha256,
            "archive_members": list(FEATURE_MEMBERS),
            "received_iq_package_seal_sha256": (
                self.received_iq_package_seal_sha256
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
            "runtime_sha256": self.runtime_sha256,
            "forward_receipt_sha256": self.forward_receipt_sha256,
            "ordered_support_physical_ids_sha256": _ordered_root(
                self.support_physical_ids
            ),
            "ordered_query_physical_ids_sha256": _ordered_root(
                self.query_physical_ids
            ),
            "array_receipts": {
                name: _array_receipt(getattr(self, name))
                for name in FEATURE_ARRAY_NAMES
            },
            "query_truth_access": False,
            "query_role_access": False,
            "performance_metric_access": False,
        }


@dataclass(frozen=True, slots=True)
class D106Target25PlanState:
    row_id: str
    receiver: str
    scene: str
    active_k: int
    registered_classes: tuple[str, ...]
    capsule_id: str
    split_id: str
    validator_receipt_sha256: str
    seed: int
    support_physical_root_sha256: str
    query_physical_root_sha256: str
    paired_feature_receipt_sha256: str
    receipt_sha256: str
    _loader_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._loader_token is not _PLAN_LOADER_TOKEN:
            raise D106Target25EvaluatorError(
                "Target25 plan state requires strict-loader construction"
            )
        object.__setattr__(self, "row_id", _text(self.row_id, "row_id"))
        object.__setattr__(self, "receiver", _text(self.receiver, "receiver"))
        object.__setattr__(self, "scene", _text(self.scene, "scene"))
        if type(self.active_k) is not int or self.active_k not in ALLOWED_K:
            raise D106Target25EvaluatorError("plan active_k must be exactly 1, 5, or 10")
        if type(self.seed) is not int or self.seed < 0:
            raise D106Target25EvaluatorError("plan seed must be a non-negative integer")
        object.__setattr__(
            self,
            "registered_classes",
            _tokens(self.registered_classes, "registered_classes"),
        )
        for name in (
            "capsule_id",
            "split_id",
            "validator_receipt_sha256",
            "support_physical_root_sha256",
            "query_physical_root_sha256",
            "paired_feature_receipt_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        if self.receipt_sha256 != _sha256(self.receipt_payload):
            raise D106Target25EvaluatorError("Target25 typed plan receipt drift")

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": PLAN_STATE_SCHEMA,
            "row_id": self.row_id,
            "receiver": self.receiver,
            "scene": self.scene,
            "active_k": self.active_k,
            "registered_classes": list(self.registered_classes),
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "validator_receipt_sha256": self.validator_receipt_sha256,
            "seed": self.seed,
            "support_physical_root_sha256": self.support_physical_root_sha256,
            "query_physical_root_sha256": self.query_physical_root_sha256,
            "paired_feature_receipt_sha256": self.paired_feature_receipt_sha256,
        }


def publish_d106_paired_features(
    feature_path: str | Path,
    receipt_path: str | Path,
    *,
    received_iq_package_seal_sha256: str,
    checkpoint_sha256: str,
    runtime_sha256: str,
    forward_receipt_sha256: str,
    support_plus: np.ndarray,
    support_signed: np.ndarray,
    query_plus: np.ndarray,
    query_signed: np.ndarray,
    support_physical_ids: Sequence[str],
    query_physical_ids: Sequence[str],
) -> dict[str, str]:
    """Write one non-overwriting same-forward feature NPZ and canonical receipt."""

    feature = _regular_new_path(feature_path, "paired-feature NPZ")
    receipt = _regular_new_path(receipt_path, "paired-feature receipt")
    plus_support = _float32_rows(support_plus, "support_plus")
    signed_support = _float32_rows(support_signed, "support_signed")
    plus_query = _float32_rows(query_plus, "query_plus")
    signed_query = _float32_rows(query_signed, "query_signed")
    if plus_support.shape != signed_support.shape or plus_query.shape != signed_query.shape:
        raise D106Target25EvaluatorError("paired publisher shape closure drift")
    if not np.array_equal(
        plus_support, np.maximum(signed_support, np.float32(0.0))
    ) or not np.array_equal(
        plus_query, np.maximum(signed_query, np.float32(0.0))
    ):
        raise D106Target25EvaluatorError("publisher plus views must equal ReLU(signed)")
    support_ids = _tokens(support_physical_ids, "support_physical_ids")
    query_ids = _tokens(query_physical_ids, "query_physical_ids")
    if len(support_ids) != len(plus_support) or len(query_ids) != len(plus_query):
        raise D106Target25EvaluatorError("publisher physical-ID closure drift")
    validate_lpo_rc_physical_id_disjointness(support_ids, query_ids)
    seals = {
        name: _require_sha256(value, name)
        for name, value in {
            "received_iq_package_seal_sha256": received_iq_package_seal_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "runtime_sha256": runtime_sha256,
            "forward_receipt_sha256": forward_receipt_sha256,
        }.items()
    }
    support_id_array = np.asarray(support_ids, dtype=np.str_)
    query_id_array = np.asarray(query_ids, dtype=np.str_)
    with feature.open("xb") as handle:
        np.savez_compressed(
            handle,
            support_plus=plus_support,
            support_signed=signed_support,
            query_plus=plus_query,
            query_signed=signed_query,
            support_physical_ids=support_id_array,
            query_physical_ids=query_id_array,
        )
    feature_sha = _file_sha256(feature)
    document = {
        "schema": PAIRED_FEATURE_SCHEMA,
        "feature_archive_name": feature.name,
        "feature_archive_sha256": feature_sha,
        "archive_members": list(FEATURE_MEMBERS),
        **seals,
        "ordered_support_physical_ids_sha256": _ordered_root(support_ids),
        "ordered_query_physical_ids_sha256": _ordered_root(query_ids),
        "array_receipts": {
            name: _array_receipt(value)
            for name, value in {
                "support_plus": plus_support,
                "support_signed": signed_support,
                "query_plus": plus_query,
                "query_signed": signed_query,
            }.items()
        },
        "query_truth_access": False,
        "query_role_access": False,
        "performance_metric_access": False,
    }
    receipt_bytes = _canonical_bytes(document)
    with receipt.open("xb") as handle:
        handle.write(receipt_bytes)
    return {
        "feature_archive_sha256": feature_sha,
        "feature_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }


def load_d106_paired_features(
    feature_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_receipt_sha256: str,
) -> D106PairedFeatureRows:
    """Load only an exact NPZ bound by an externally pinned canonical receipt."""

    feature = _regular_existing_path(feature_path, "paired-feature NPZ")
    receipt = _regular_existing_path(receipt_path, "paired-feature receipt")
    expected = _require_sha256(expected_receipt_sha256, "expected_receipt_sha256")
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise D106Target25EvaluatorError("paired-feature external receipt SHA mismatch")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Target25EvaluatorError("paired-feature receipt is not UTF-8 JSON") from error
    if type(document) is not dict or raw != _canonical_bytes(document):
        raise D106Target25EvaluatorError("paired-feature receipt is not canonical JSON")
    if set(document) != PAIRED_FEATURE_RECEIPT_KEYS:
        raise D106Target25EvaluatorError("paired-feature receipt field closure drift")
    if (
        document["schema"] != PAIRED_FEATURE_SCHEMA
        or document["feature_archive_name"] != feature.name
        or document["feature_archive_sha256"] != _file_sha256(feature)
        or document["archive_members"] != list(FEATURE_MEMBERS)
        or document["query_truth_access"] is not False
        or document["query_role_access"] is not False
        or document["performance_metric_access"] is not False
    ):
        raise D106Target25EvaluatorError("paired-feature receipt binding drift")
    for name in (
        "received_iq_package_seal_sha256",
        "checkpoint_sha256",
        "runtime_sha256",
        "forward_receipt_sha256",
        "feature_archive_sha256",
        "ordered_support_physical_ids_sha256",
        "ordered_query_physical_ids_sha256",
    ):
        _require_sha256(document[name], name)
    arrays_receipt = document["array_receipts"]
    if not isinstance(arrays_receipt, Mapping) or set(arrays_receipt) != set(FEATURE_ARRAY_NAMES):
        raise D106Target25EvaluatorError("paired-feature array receipt closure drift")
    try:
        with np.load(feature, allow_pickle=False) as archive:
            if set(archive.files) != set(FEATURE_MEMBERS):
                raise D106Target25EvaluatorError("paired-feature NPZ member closure drift")
            arrays = {name: np.array(archive[name], copy=True) for name in FEATURE_MEMBERS}
    except D106Target25EvaluatorError:
        raise
    except (OSError, ValueError) as error:
        raise D106Target25EvaluatorError("paired-feature NPZ cannot be loaded") from error
    for name in FEATURE_ARRAY_NAMES:
        if arrays_receipt[name] != _array_receipt(arrays[name]):
            raise D106Target25EvaluatorError(f"{name} receipt mismatch")
    support_id_array = arrays["support_physical_ids"]
    query_id_array = arrays["query_physical_ids"]
    if (
        support_id_array.ndim != 1
        or query_id_array.ndim != 1
        or support_id_array.dtype.kind != "U"
        or query_id_array.dtype.kind != "U"
    ):
        raise D106Target25EvaluatorError("paired-feature physical-ID dtype/shape drift")
    support_ids = tuple(support_id_array.tolist())
    query_ids = tuple(query_id_array.tolist())
    if (
        _ordered_root(support_ids) != document["ordered_support_physical_ids_sha256"]
        or _ordered_root(query_ids) != document["ordered_query_physical_ids_sha256"]
    ):
        raise D106Target25EvaluatorError("paired-feature ordered physical-ID receipt drift")
    return D106PairedFeatureRows(
        support_plus=arrays["support_plus"],
        support_signed=arrays["support_signed"],
        query_plus=arrays["query_plus"],
        query_signed=arrays["query_signed"],
        support_physical_ids=support_ids,
        query_physical_ids=query_ids,
        received_iq_package_seal_sha256=document["received_iq_package_seal_sha256"],
        checkpoint_sha256=document["checkpoint_sha256"],
        runtime_sha256=document["runtime_sha256"],
        forward_receipt_sha256=document["forward_receipt_sha256"],
        feature_archive_name=document["feature_archive_name"],
        feature_archive_sha256=document["feature_archive_sha256"],
        receipt_sha256=expected,
        _loader_token=_FEATURE_LOADER_TOKEN,
    )


def publish_d106_target25_plan_state(
    path: str | Path, *, projection: Mapping[str, Any]
) -> str:
    """Write one immutable canonical single-state plan projection."""

    target = _regular_new_path(path, "Target25 plan state")
    if not isinstance(projection, Mapping) or set(projection) != PLAN_STATE_KEYS:
        raise D106Target25EvaluatorError("Target25 plan projection field closure drift")
    payload = _canonical_bytes(dict(projection))
    with target.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def load_d106_target25_plan_state(
    path: str | Path, *, expected_receipt_sha256: str
) -> D106Target25PlanState:
    """Load one externally SHA-pinned canonical single-state plan projection."""

    source = _regular_existing_path(path, "Target25 plan state")
    expected = _require_sha256(expected_receipt_sha256, "expected_receipt_sha256")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise D106Target25EvaluatorError("Target25 plan external receipt SHA mismatch")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Target25EvaluatorError("Target25 plan is not UTF-8 JSON") from error
    if type(document) is not dict or raw != _canonical_bytes(document):
        raise D106Target25EvaluatorError("Target25 plan is not canonical JSON")
    if set(document) != PLAN_STATE_KEYS or document["schema"] != PLAN_STATE_SCHEMA:
        raise D106Target25EvaluatorError("Target25 plan schema/field closure drift")
    return D106Target25PlanState(
        row_id=document["row_id"],
        receiver=document["receiver"],
        scene=document["scene"],
        active_k=document["active_k"],
        registered_classes=tuple(document["registered_classes"]),
        capsule_id=document["capsule_id"],
        split_id=document["split_id"],
        validator_receipt_sha256=document["validator_receipt_sha256"],
        seed=document["seed"],
        support_physical_root_sha256=document["support_physical_root_sha256"],
        query_physical_root_sha256=document["query_physical_root_sha256"],
        paired_feature_receipt_sha256=document["paired_feature_receipt_sha256"],
        receipt_sha256=expected,
        _loader_token=_PLAN_LOADER_TOKEN,
    )


def _unique_student_t_predictions(
    logits: np.ndarray, registry: tuple[str, ...]
) -> list[str]:
    values = np.asarray(logits)
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[1] != len(registry)
        or not np.isfinite(values).all()
    ):
        raise D106Target25EvaluatorError("Student-t logits layout drift")
    predictions: list[str] = []
    for row in values:
        winners = np.flatnonzero(row == np.float32(np.max(row)))
        if len(winners) != 1:
            raise D106Target25EvaluatorError(
                "Student-t cross-class maximum tie is fail-closed"
            )
        predictions.append(registry[int(winners[0])])
    return predictions


def _paired_view_receipt(
    *,
    support_physical_ids: tuple[str, ...],
    support_plus: np.ndarray,
    support_signed: np.ndarray,
    da_receipt_sha256: str,
) -> str:
    return _sha256(
        {
            "schema": "cvs.phase2.d106.target25.paired_support_views.v1",
            "support_physical_ids": list(support_physical_ids),
            "support_plus": _array_receipt(support_plus),
            "support_signed": _array_receipt(support_signed),
            "da_receipt_sha256": da_receipt_sha256,
        }
    )


def _identity_view_receipt(features: D106PairedFeatureRows) -> str:
    return _sha256(
        {
            "schema": "cvs.phase2.d106.target25.identity_views.v1",
            "paired_feature_receipt_sha256": features.receipt_sha256,
            "support_plus": _array_receipt(features.support_plus),
            "support_signed": _array_receipt(features.support_signed),
            "query_plus": _array_receipt(features.query_plus),
            "query_signed": _array_receipt(features.query_signed),
        }
    )


def _validate_plan_feature_support_binding(
    plan: D106Target25PlanState,
    features: D106PairedFeatureRows,
    support: D106RDCESupportRows,
) -> tuple[str, ...]:
    if type(plan) is not D106Target25PlanState or plan._loader_token is not _PLAN_LOADER_TOKEN:
        raise D106Target25EvaluatorError("exact strict-loader Target25 plan required")
    if type(features) is not D106PairedFeatureRows or features._loader_token is not _FEATURE_LOADER_TOKEN:
        raise D106Target25EvaluatorError("exact strict-loader paired features required")
    if type(support) is not D106RDCESupportRows:
        raise D106Target25EvaluatorError("exact D106RDCESupportRows required")
    if (
        plan.receipt_sha256 != _sha256(plan.receipt_payload)
        or features.receipt_sha256 != _sha256(features.receipt_payload)
    ):
        raise D106Target25EvaluatorError("typed plan/feature receipt revalidation drift")
    handle = support.split_handle
    registry = tuple(support.qknn_bank.classes)
    support_ids = tuple(str(value) for value in support.support_physical_ids.tolist())
    if (
        plan.paired_feature_receipt_sha256 != features.receipt_sha256
        or plan.row_id != support.row_id
        or plan.seed != support.seed
        or plan.active_k != support.qknn_bank.active_k
        or plan.registered_classes != registry
        or plan.capsule_id != handle.capsule_id
        or plan.split_id != handle.split_id
        or plan.validator_receipt_sha256 != handle.validator_receipt_sha256
        or plan.support_physical_root_sha256 != handle.support_physical_root_sha256
        or plan.query_physical_root_sha256 != handle.query_physical_root_sha256
        or plan.support_physical_root_sha256 != features.support_physical_root_sha256
        or plan.query_physical_root_sha256 != features.query_physical_root_sha256
        or support_ids != features.support_physical_ids
        or not np.array_equal(support.support_z_id, features.support_plus)
        or len(features.support_plus) != len(registry) * plan.active_k
    ):
        raise D106Target25EvaluatorError("plan/features/support exact binding drift")
    return registry


def evaluate_d106_target25_state(
    *,
    plan_state: D106Target25PlanState,
    paired_features: D106PairedFeatureRows,
    support_rows: D106RDCESupportRows,
    rdce_asset: D106RDCEAsset,
    rdce_row_authority: Any,
    rcmr_method_lock: Any,
) -> dict[str, Any]:
    """Evaluate all four arms from strict plan/features without truth access."""

    registry = _validate_plan_feature_support_binding(
        plan_state, paired_features, support_rows
    )
    support_labels = tuple(str(value) for value in support_rows.support_labels.tolist())
    support_ids = paired_features.support_physical_ids
    query_ids = paired_features.query_physical_ids
    support_plus = paired_features.support_plus
    support_signed = paired_features.support_signed
    query_plus = paired_features.query_plus
    query_signed = paired_features.query_signed

    rdce_state = fit_d106_rdce_runtime(
        rdce_asset, support_rows, row_authority=rdce_row_authority
    )
    rdce_context = prepare_d106_rdce_scoring_context(rdce_state)
    da_support_plus = transform_d106_rdce_zid(
        rdce_state, support_plus, context=rdce_context
    )
    da_support_signed = transform_d106_rdce_zid(
        rdce_state, support_signed, context=rdce_context
    )
    da_query_plus = transform_d106_rdce_query(
        rdce_state, query_plus, context=rdce_context
    )
    da_query_signed = transform_d106_rdce_query(
        rdce_state, query_signed, context=rdce_context
    )

    bank = support_rows.qknn_bank
    qknn_lock = bank.config
    identity_metric = identity_shared_psd_metric(config=qknn_lock)
    da_bank = build_typed_zid_support_bank(
        da_support_plus, support_labels, registry, config=qknn_lock
    )
    arm_predictions: dict[str, list[str]] = {
        "M0": _unique_student_t_predictions(
            score_zid_student_t_logits(bank, query_plus, metric=identity_metric),
            registry,
        ),
        "M_DA": _unique_student_t_predictions(
            score_zid_student_t_logits(
                da_bank, da_query_plus, metric=identity_metric
            ),
            registry,
        ),
    }
    identity_da_receipt = _sha256(
        {
            "schema": "cvs.phase2.d106.target25.identity_da.v1",
            "mapping": "identity",
            "query_state_updates": 0,
        }
    )
    bindings = {
        "M_HEAD": D106RCMR2VBinding(
            capsule_id=plan_state.capsule_id,
            split_id=plan_state.split_id,
            validator_receipt_sha256=plan_state.validator_receipt_sha256,
            support_physical_root_sha256=plan_state.support_physical_root_sha256,
            row_id=plan_state.row_id,
            seed=plan_state.seed,
            active_k=plan_state.active_k,
            da_receipt_sha256=identity_da_receipt,
            paired_view_receipt_sha256=_paired_view_receipt(
                support_physical_ids=support_ids,
                support_plus=support_plus,
                support_signed=support_signed,
                da_receipt_sha256=identity_da_receipt,
            ),
        ),
        "M_JOINT": D106RCMR2VBinding(
            capsule_id=plan_state.capsule_id,
            split_id=plan_state.split_id,
            validator_receipt_sha256=plan_state.validator_receipt_sha256,
            support_physical_root_sha256=plan_state.support_physical_root_sha256,
            row_id=plan_state.row_id,
            seed=plan_state.seed,
            active_k=plan_state.active_k,
            da_receipt_sha256=rdce_state.runtime_receipt_sha256,
            paired_view_receipt_sha256=_paired_view_receipt(
                support_physical_ids=support_ids,
                support_plus=da_support_plus,
                support_signed=da_support_signed,
                da_receipt_sha256=rdce_state.runtime_receipt_sha256,
            ),
        ),
    }
    states = {
        "M_HEAD": build_d106_rcmr_2v_state(
            support_plus,
            support_signed,
            support_labels,
            support_ids,
            registry,
            binding=bindings["M_HEAD"],
            method_lock=rcmr_method_lock,
        ),
        "M_JOINT": build_d106_rcmr_2v_state(
            da_support_plus,
            da_support_signed,
            support_labels,
            support_ids,
            registry,
            binding=bindings["M_JOINT"],
            method_lock=rcmr_method_lock,
        ),
    }
    contexts = {
        arm: prepare_d106_rcmr_2v_scoring_context(state)
        for arm, state in states.items()
    }
    arm_predictions["M_HEAD"] = [
        score_d106_rcmr_2v_query(
            states["M_HEAD"],
            plus,
            signed,
            da_receipt_sha256=identity_da_receipt,
            context=contexts["M_HEAD"],
        ).predicted_class
        for plus, signed in zip(query_plus, query_signed, strict=True)
    ]
    arm_predictions["M_JOINT"] = [
        score_d106_rcmr_2v_query(
            states["M_JOINT"],
            plus,
            signed,
            da_receipt_sha256=rdce_state.runtime_receipt_sha256,
            context=contexts["M_JOINT"],
        ).predicted_class
        for plus, signed in zip(da_query_plus, da_query_signed, strict=True)
    ]
    if set(arm_predictions) != set(ARMS) or any(
        len(value) != len(query_ids) for value in arm_predictions.values()
    ):
        raise D106Target25EvaluatorError("four-arm prediction closure drift")
    method_lock_sha256 = getattr(rcmr_method_lock, "document_sha256", None)
    _require_sha256(method_lock_sha256, "RCMR method-lock receipt")
    row: dict[str, Any] = {
        "schema": TARGET25_ROW_SCHEMA,
        "row_id": plan_state.row_id,
        "receiver": plan_state.receiver,
        "scene": plan_state.scene,
        "K": plan_state.active_k,
        "registered_classes": list(registry),
        "query_physical_ids": list(query_ids),
        "arm_predictions": arm_predictions,
        "shared_component_receipts": {
            "target25_plan_state_sha256": plan_state.receipt_sha256,
            "paired_feature_receipt_sha256": paired_features.receipt_sha256,
            "paired_feature_archive_sha256": paired_features.feature_archive_sha256,
            "received_iq_package_seal_sha256": (
                paired_features.received_iq_package_seal_sha256
            ),
            "checkpoint_sha256": paired_features.checkpoint_sha256,
            "runtime_sha256": paired_features.runtime_sha256,
            "forward_receipt_sha256": paired_features.forward_receipt_sha256,
            "M_DA_M_JOINT_rdce_state_sha256": rdce_state.runtime_receipt_sha256,
            "M0_M_HEAD_identity_view_sha256": _identity_view_receipt(
                paired_features
            ),
            "M0_M_DA_student_t_lock_sha256": qknn_lock.lock_digest,
            "M_HEAD_M_JOINT_rcmr_method_lock_sha256": method_lock_sha256,
            "M_HEAD_state_sha256": states["M_HEAD"].state_receipt_sha256,
            "M_JOINT_state_sha256": states["M_JOINT"].state_receipt_sha256,
        },
        "query_truth_access": False,
        "query_role_access": False,
        "query_selection": False,
        "query_state_updates": 0,
    }
    row["prediction_receipt_sha256"] = _sha256(row)
    return row


__all__ = [
    "ALLOWED_K",
    "ARMS",
    "D106PairedFeatureRows",
    "D106Target25EvaluatorError",
    "D106Target25PlanState",
    "FEATURE_ARRAY_NAMES",
    "FEATURE_MEMBERS",
    "PAIRED_FEATURE_RECEIPT_KEYS",
    "PAIRED_FEATURE_SCHEMA",
    "PLAN_STATE_KEYS",
    "PLAN_STATE_SCHEMA",
    "evaluate_d106_target25_state",
    "load_d106_paired_features",
    "load_d106_target25_plan_state",
    "publish_d106_paired_features",
    "publish_d106_target25_plan_state",
]
