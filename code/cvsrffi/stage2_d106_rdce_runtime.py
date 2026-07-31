"""Query-immutable D106 Phase2 runtime with typed support-row binding.

The formal fit API accepts no loose support receipt and no query, role, source,
or cache argument.  It requires actual target rows plus a canonical external
row-authority document, validated-once split handle, and the exact qKNN bank
made from those rows.  Private math-only fits are non-deployable.  Only a
compact FP16 attenuation vector and immutable bindings survive formal fit.
Deployment phi uses two rank-three projections rather than a dense 160 by 160
product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping
import weakref

import numpy as np

from cvsrffi.stage2_d106_rdce_asset import (
    CANDIDATE_ID,
    D106RDCEAsset,
    D106RDCEAssetError,
    D106RDCEScientificRejectReceipt,
    EPSILON,
    FORMAL_DEPLOYMENT_STATUS,
    NON_DEPLOYABLE_MATH_STATUS,
    RDCE_RANK,
    Z_DIM,
    decode_d106_rdce_basis,
    decode_d106_rdce_tau,
    make_d106_rdce_scientific_reject,
)
from cvsrffi.stage2_lpo_rc_qknn import TypedValidatedOnceP2SplitHandle
from cvsrffi.stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    build_typed_zid_support_bank,
)


ALLOWED_K = (1, 5, 10)
K1_ATTENUATION = 0.3
GAMMA = 0.2
SUPPORT_SCATTER_EPSILON = 1.0e-8
MIN_ATTENUATION = 0.05
MAX_ATTENUATION = 0.95
MIN_METRIC_EIGENVALUE = 0.05
RUNTIME_SCHEMA = "cvs.phase2.d106.rdce_gtsm_runtime.v3"
SUPPORT_BINDING_SCHEMA = "cvs.phase2.d106.rdce_gtsm_support_binding.v1"
RUNTIME_BINDING_SCHEMA = "cvs.phase2.d106.rdce_gtsm_runtime_binding.v1"
RUNTIME_WIRE_SCHEMA = "cvs.phase2.d106.rdce_gtsm_runtime_wire.v2"
RESOURCE_RECEIPT_SCHEMA = "cvs.phase2.d106.rdce_gtsm_resource_receipt.v3"
RUNTIME_WIRE_MAGIC = b"CVSD106RT\x00\x02"
ROW_AUTHORITY_SCHEMA = "cvs.phase2.d106.rdce_gtsm_row_authority.v1"
_ROW_AUTHORITY_LOADER_TOKEN = object()
_SCORING_CONTEXT_LOADER_TOKEN = object()

# The nearest binary16 value above 0.05 and below 0.95 protects the specified
# metric floor after sealing/replay.  No full-precision attenuation survives.
MIN_ATTENUATION_FP16 = np.nextafter(
    np.float16(MIN_ATTENUATION), np.float16(np.inf), dtype=np.float16
)
MAX_ATTENUATION_FP16 = np.nextafter(
    np.float16(MAX_ATTENUATION), np.float16(0.0), dtype=np.float16
)


class D106RDCERuntimeError(D106RDCEAssetError):
    """Raised when target support binding or runtime state drifts."""


class _RuntimeScientificReject(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str:
        raise D106RDCERuntimeError(f"{name} must be an exact string SHA256")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise D106RDCERuntimeError(f"{name} must be a lowercase SHA256")
    return value


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _require_nonempty_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise D106RDCERuntimeError(f"{name} must be a non-empty exact string")
    return value


def _typed_strings(value: Any, name: str, rows: int) -> tuple[str, ...]:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) != rows:
        raise D106RDCERuntimeError(
            f"{name} must be a one-dimensional typed array aligned to support_z_id"
        )
    if value.dtype.kind not in {"U", "S"}:
        raise D106RDCERuntimeError(f"{name} must use a unicode or bytes numpy dtype")
    decoded: list[str] = []
    for item in value.tolist():
        if isinstance(item, bytes):
            try:
                decoded.append(item.decode("utf-8"))
            except UnicodeDecodeError as error:
                raise D106RDCERuntimeError(
                    f"{name} byte values must be UTF-8"
                ) from error
        else:
            decoded.append(str(item))
    result = tuple(decoded)
    if any(not item for item in result):
        raise D106RDCERuntimeError(f"{name} values must be non-empty")
    return result


def _normalized_support(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise D106RDCERuntimeError("support_z_id must be a numpy float32 array")
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or value.shape[1] != Z_DIM
        or value.shape[0] < 1
        or not np.isfinite(value).all()
    ):
        raise D106RDCERuntimeError(f"support_z_id must be finite float32 [N,{Z_DIM}]")
    rows = np.ascontiguousarray(value, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= EPSILON):
        raise D106RDCERuntimeError("support_z_id contains a zero-norm row")
    return np.ascontiguousarray(rows / norms, dtype=np.float64)


def _canonical_rows(rows: np.ndarray) -> np.ndarray:
    order = sorted(range(len(rows)), key=lambda index: rows[index].tobytes(order="C"))
    return np.ascontiguousarray(rows[np.asarray(order, dtype=np.int64)], dtype=np.float64)


def _physical_root(physical_ids: tuple[str, ...]) -> str:
    return _sha256_bytes(_canonical_bytes(sorted(physical_ids)))


def _ordered_physical_root(physical_ids: tuple[str, ...]) -> str:
    return _sha256_bytes(_canonical_bytes(list(physical_ids)))


@dataclass(frozen=True, slots=True)
class _D106ArrayAuthorityReceipt:
    """Exact content receipt for one actual support-side ndarray."""

    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.dtype) is not str
            or not self.dtype
            or type(self.shape) is not tuple
            or not self.shape
            or any(type(value) is not int or value < 0 for value in self.shape)
        ):
            raise D106RDCERuntimeError("row authority array receipt dtype/shape drift")
        object.__setattr__(self, "sha256", _require_sha256(self.sha256, "array receipt sha256"))

    @classmethod
    def from_mapping(
        cls, value: Any, name: str
    ) -> "_D106ArrayAuthorityReceipt":
        if (
            type(value) is not dict
            or set(value) != {"dtype", "shape", "sha256"}
            or type(value["shape"]) is not list
        ):
            raise D106RDCERuntimeError(f"{name} must be an exact array receipt")
        return cls(
            dtype=value["dtype"],
            shape=tuple(value["shape"]),
            sha256=value["sha256"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class _D106RDCERowAuthority:
    """Private, loader-origin authority for a single formal support row."""

    capsule_id: str
    split_id: str
    validator_receipt_sha256: str
    row_id: str
    seed: int
    active_k: int
    registered_classes: tuple[str, ...]
    support_z_id_receipt: _D106ArrayAuthorityReceipt
    support_labels_receipt: _D106ArrayAuthorityReceipt
    support_physical_ids_receipt: _D106ArrayAuthorityReceipt
    ordered_support_physical_ids_sha256: str
    qknn_bank_sha256: str
    support_physical_root_sha256: str
    query_physical_root_sha256: str
    protocol_schema: str
    phase2_data_status: str
    support_query_disjoint: bool
    authority_document_sha256: str
    schema: str = ROW_AUTHORITY_SCHEMA
    _loader_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.schema != ROW_AUTHORITY_SCHEMA
            or self.protocol_schema != "p2_min_v1"
            or self.phase2_data_status != "VALIDATED_ONCE"
            or self.support_query_disjoint is not True
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.active_k) is not int
            or self.active_k not in ALLOWED_K
            or type(self.registered_classes) is not tuple
            or len(self.registered_classes) < 2
            or any(type(value) is not str or not value for value in self.registered_classes)
            or len(set(self.registered_classes)) != len(self.registered_classes)
            or type(self.support_z_id_receipt) is not _D106ArrayAuthorityReceipt
            or type(self.support_labels_receipt) is not _D106ArrayAuthorityReceipt
            or type(self.support_physical_ids_receipt) is not _D106ArrayAuthorityReceipt
        ):
            raise D106RDCERuntimeError("D106 row authority lifecycle/type drift")
        for name in (
            "capsule_id",
            "split_id",
            "validator_receipt_sha256",
            "ordered_support_physical_ids_sha256",
            "qknn_bank_sha256",
            "support_physical_root_sha256",
            "query_physical_root_sha256",
            "authority_document_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        object.__setattr__(self, "row_id", _require_nonempty_text(self.row_id, "row_id"))

    @property
    def is_loader_authorized(self) -> bool:
        return self._loader_token is _ROW_AUTHORITY_LOADER_TOKEN


def load_d106_rdce_row_authority(
    path: str | Path, *, expected_authority_sha256: str
) -> _D106RDCERowAuthority:
    """Load one canonical, SHA-pinned formal row authority document."""

    expected = _require_sha256(expected_authority_sha256, "expected_authority_sha256")
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D106RDCERuntimeError(
            "D106 row authority must be a regular non-symlink file"
        )
    raw = source.read_bytes()
    if _sha256_bytes(raw) != expected:
        raise D106RDCERuntimeError("D106 row authority external SHA256 mismatch")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RDCERuntimeError("D106 row authority must be canonical UTF-8 JSON") from error
    if type(document) is not dict or raw != _canonical_bytes(document):
        raise D106RDCERuntimeError("D106 row authority is not original canonical JSON")
    expected_keys = {
        "schema",
        "capsule_id",
        "split_id",
        "validator_receipt_sha256",
        "row_id",
        "seed",
        "active_k",
        "registered_classes",
        "support_z_id_receipt",
        "support_labels_receipt",
        "support_physical_ids_receipt",
        "ordered_support_physical_ids_sha256",
        "qknn_bank_sha256",
        "support_physical_root_sha256",
        "query_physical_root_sha256",
        "protocol_schema",
        "phase2_data_status",
        "support_query_disjoint",
    }
    if (
        set(document) != expected_keys
        or document["schema"] != ROW_AUTHORITY_SCHEMA
        or type(document["registered_classes"]) is not list
    ):
        raise D106RDCERuntimeError("D106 row authority schema/key drift")
    return _D106RDCERowAuthority(
        capsule_id=document["capsule_id"],
        split_id=document["split_id"],
        validator_receipt_sha256=document["validator_receipt_sha256"],
        row_id=document["row_id"],
        seed=document["seed"],
        active_k=document["active_k"],
        registered_classes=tuple(document["registered_classes"]),
        support_z_id_receipt=_D106ArrayAuthorityReceipt.from_mapping(
            document["support_z_id_receipt"], "support_z_id_receipt"
        ),
        support_labels_receipt=_D106ArrayAuthorityReceipt.from_mapping(
            document["support_labels_receipt"], "support_labels_receipt"
        ),
        support_physical_ids_receipt=_D106ArrayAuthorityReceipt.from_mapping(
            document["support_physical_ids_receipt"], "support_physical_ids_receipt"
        ),
        ordered_support_physical_ids_sha256=document[
            "ordered_support_physical_ids_sha256"
        ],
        qknn_bank_sha256=document["qknn_bank_sha256"],
        support_physical_root_sha256=document["support_physical_root_sha256"],
        query_physical_root_sha256=document["query_physical_root_sha256"],
        protocol_schema=document["protocol_schema"],
        phase2_data_status=document["phase2_data_status"],
        support_query_disjoint=document["support_query_disjoint"],
        authority_document_sha256=expected,
        _loader_token=_ROW_AUTHORITY_LOADER_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class D106RDCESupportRows:
    """Transient target-support row authority for one D106 fit.

    The object is deliberately not persisted by D106RDCERuntimeState.  The
    qKNN bank is recompiled from these actual rows during fit, binding physical
    rows, labels, bank contents, and validated split identity together.
    """

    support_z_id: np.ndarray
    support_labels: np.ndarray
    support_physical_ids: np.ndarray
    qknn_bank: TypedINT8ZIDSupportBank
    split_handle: TypedValidatedOnceP2SplitHandle
    row_id: str
    seed: int
    schema: str = SUPPORT_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SUPPORT_BINDING_SCHEMA:
            raise D106RDCERuntimeError("D106 support-row schema drift")
        if type(self.qknn_bank) is not TypedINT8ZIDSupportBank:
            raise D106RDCERuntimeError("D106 support rows require an exact typed qKNN bank")
        if type(self.split_handle) is not TypedValidatedOnceP2SplitHandle:
            raise D106RDCERuntimeError(
                "D106 support rows require an exact VALIDATED_ONCE split handle"
            )
        _require_nonempty_text(self.row_id, "row_id")
        if type(self.seed) is not int or self.seed < 0:
            raise D106RDCERuntimeError("seed must be a non-negative exact integer")


@dataclass(frozen=True, slots=True)
class D106RDCERuntimeBinding:
    """The typed expected binding required when replaying a runtime wire."""

    asset_binding_sha256: str
    capsule_id: str
    split_id: str
    row_id: str
    seed: int
    qknn_bank_sha256: str
    support_physical_root_sha256: str
    support_binding_sha256: str
    active_k: int
    registered_class_count: int
    row_authority_sha256: str | None = None
    schema: str = RUNTIME_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != RUNTIME_BINDING_SCHEMA
            or type(self.active_k) is not int
            or self.active_k not in ALLOWED_K
            or type(self.registered_class_count) is not int
            or self.registered_class_count < 2
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise D106RDCERuntimeError("D106 runtime binding lifecycle drift")
        for name in (
            "asset_binding_sha256",
            "capsule_id",
            "split_id",
            "qknn_bank_sha256",
            "support_physical_root_sha256",
            "support_binding_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        if self.row_authority_sha256 is not None:
            object.__setattr__(
                self,
                "row_authority_sha256",
                _require_sha256(self.row_authority_sha256, "row_authority_sha256"),
            )
        object.__setattr__(self, "row_id", _require_nonempty_text(self.row_id, "row_id"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_binding_sha256": self.asset_binding_sha256,
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "row_id": self.row_id,
            "seed": self.seed,
            "qknn_bank_sha256": self.qknn_bank_sha256,
            "support_physical_root_sha256": self.support_physical_root_sha256,
            "support_binding_sha256": self.support_binding_sha256,
            "active_k": self.active_k,
            "registered_class_count": self.registered_class_count,
            "row_authority_sha256": self.row_authority_sha256,
        }


def _verify_split_handle(handle: TypedValidatedOnceP2SplitHandle) -> None:
    if type(handle) is not TypedValidatedOnceP2SplitHandle:
        raise D106RDCERuntimeError(
            "D106 runtime requires an exact typed VALIDATED_ONCE split handle"
        )
    if (
        handle.protocol_schema != "p2_min_v1"
        or handle.phase2_data_status != "VALIDATED_ONCE"
        or handle.support_query_disjoint is not True
    ):
        raise D106RDCERuntimeError("D106 validated split handle lifecycle drift")
    for name in (
        "capsule_id",
        "split_id",
        "validator_receipt_sha256",
        "support_physical_root_sha256",
        "query_physical_root_sha256",
    ):
        _require_sha256(getattr(handle, name), f"split handle {name}")


def _group_support(
    rows: np.ndarray, labels: tuple[str, ...], registry: tuple[str, ...]
) -> list[tuple[bytes, np.ndarray]]:
    grouped: dict[str, list[np.ndarray]] = {token: [] for token in registry}
    for row, token in zip(rows, labels, strict=True):
        if token not in grouped:
            raise D106RDCERuntimeError("support_labels contains an unregistered class")
        grouped[token].append(row)
    if any(not value for value in grouped.values()):
        raise D106RDCERuntimeError("every registered class requires support")
    canonical: list[tuple[bytes, np.ndarray]] = []
    for values in grouped.values():
        ordered = _canonical_rows(np.stack(values, axis=0))
        key = _sha256_bytes(ordered.tobytes(order="C")).encode("ascii")
        canonical.append((key, ordered))
    return sorted(canonical, key=lambda item: item[0])


@dataclass(frozen=True, slots=True)
class _VerifiedSupport:
    normalized_rows: np.ndarray
    labels: tuple[str, ...]
    registry: tuple[str, ...]
    groups: tuple[tuple[bytes, np.ndarray], ...]
    binding: D106RDCERuntimeBinding


def _verified_support(
    support: D106RDCESupportRows,
    asset: D106RDCEAsset,
    *,
    row_authority: _D106RDCERowAuthority | None,
) -> _VerifiedSupport:
    if type(support) is not D106RDCESupportRows:
        raise D106RDCERuntimeError("D106 runtime requires exact typed support rows")
    if type(asset) is not D106RDCEAsset:
        raise D106RDCERuntimeError("D106 runtime requires an exact sealed asset")
    if row_authority is not None and (
        type(row_authority) is not _D106RDCERowAuthority
        or not row_authority.is_loader_authorized
    ):
        raise D106RDCERuntimeError(
            "D106 formal runtime requires a loader-origin row authority"
        )
    _verify_split_handle(support.split_handle)
    rows = _normalized_support(support.support_z_id)
    labels = _typed_strings(support.support_labels, "support_labels", len(rows))
    physical_ids = _typed_strings(
        support.support_physical_ids, "support_physical_ids", len(rows)
    )
    if len(set(physical_ids)) != len(physical_ids):
        raise D106RDCERuntimeError("support physical IDs must be unique")
    bank = support.qknn_bank
    registry = tuple(bank.classes)
    if len(registry) < 2 or len(set(registry)) != len(registry):
        raise D106RDCERuntimeError("qKNN bank class registry drift")
    if any(label not in registry for label in labels):
        raise D106RDCERuntimeError("support labels contain an unregistered class")
    if len(rows) != len(registry) * bank.active_k or bank.active_k not in ALLOWED_K:
        raise D106RDCERuntimeError("support rows/bank active K closure drift")
    if tuple(int(np.sum(np.asarray(labels) == name)) for name in registry) != bank.support_counts:
        raise D106RDCERuntimeError("support labels do not bind qKNN bank class counts")
    try:
        reproduced_bank = build_typed_zid_support_bank(
            support.support_z_id,
            labels,
            registry,
            config=bank.config,
        )
    except Exception as error:
        raise D106RDCERuntimeError(
            "support rows cannot reproduce the exact typed qKNN bank"
        ) from error
    if reproduced_bank.bank_receipt_sha256 != bank.bank_receipt_sha256:
        raise D106RDCERuntimeError(
            "support rows do not reproduce the supplied qKNN bank SHA256"
        )
    root = _physical_root(physical_ids)
    if root != support.split_handle.support_physical_root_sha256:
        raise D106RDCERuntimeError(
            "support physical IDs do not match VALIDATED_ONCE split root"
        )
    if row_authority is not None:
        if not asset.is_formal_deployable:
            raise D106RDCERuntimeError(
                "formal runtime cannot consume a non-deployable math asset"
            )
        handle = support.split_handle
        if (
            row_authority.capsule_id != handle.capsule_id
            or row_authority.split_id != handle.split_id
            or row_authority.validator_receipt_sha256
            != handle.validator_receipt_sha256
            or row_authority.support_physical_root_sha256 != root
            or row_authority.query_physical_root_sha256
            != handle.query_physical_root_sha256
            or row_authority.protocol_schema != handle.protocol_schema
            or row_authority.phase2_data_status != handle.phase2_data_status
            or row_authority.support_query_disjoint is not handle.support_query_disjoint
            or row_authority.row_id != support.row_id
            or row_authority.seed != support.seed
            or row_authority.active_k != bank.active_k
            or row_authority.registered_classes != registry
            or row_authority.qknn_bank_sha256 != bank.bank_receipt_sha256
            or row_authority.ordered_support_physical_ids_sha256
            != _ordered_physical_root(physical_ids)
            or row_authority.support_z_id_receipt.as_dict()
            != _array_receipt(support.support_z_id)
            or row_authority.support_labels_receipt.as_dict()
            != _array_receipt(support.support_labels)
            or row_authority.support_physical_ids_receipt.as_dict()
            != _array_receipt(support.support_physical_ids)
        ):
            raise D106RDCERuntimeError(
                "D106 row authority does not bind the actual support row"
            )
    support_payload = {
        "schema": SUPPORT_BINDING_SCHEMA,
        "capsule_id": support.split_handle.capsule_id,
        "split_id": support.split_handle.split_id,
        "validator_receipt_sha256": support.split_handle.validator_receipt_sha256,
        "split_handle_sha256": support.split_handle.handle_digest,
        "support_physical_root_sha256": root,
        "row_id": support.row_id,
        "seed": support.seed,
        "qknn_bank_sha256": bank.bank_receipt_sha256,
        "qknn_config_lock_sha256": bank.config_lock_digest,
        "active_k": bank.active_k,
        "registered_class_count": len(registry),
        "row_authority_sha256": (
            None
            if row_authority is None
            else row_authority.authority_document_sha256
        ),
        "actual_support_z_id": _array_receipt(support.support_z_id),
        "actual_support_labels": _array_receipt(support.support_labels),
        "actual_support_physical_ids": _array_receipt(support.support_physical_ids),
    }
    binding = D106RDCERuntimeBinding(
        asset_binding_sha256=asset.binding_sha256,
        capsule_id=support.split_handle.capsule_id,
        split_id=support.split_handle.split_id,
        row_id=support.row_id,
        seed=support.seed,
        qknn_bank_sha256=bank.bank_receipt_sha256,
        support_physical_root_sha256=root,
        support_binding_sha256=_sha256_bytes(_canonical_bytes(support_payload)),
        active_k=bank.active_k,
        registered_class_count=len(registry),
        row_authority_sha256=(
            None
            if row_authority is None
            else row_authority.authority_document_sha256
        ),
    )
    return _VerifiedSupport(
        normalized_rows=rows,
        labels=labels,
        registry=registry,
        groups=tuple(_group_support(rows, labels, registry)),
        binding=binding,
    )


def _runtime_payload(
    *,
    binding: D106RDCERuntimeBinding,
    attenuation_fp16: np.ndarray,
    deployment_status: str,
) -> dict[str, Any]:
    return {
        "schema": RUNTIME_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        **binding.as_dict(),
        "attenuation_fp16": _array_receipt(attenuation_fp16),
        "deployment_status": deployment_status,
        "support_rows_retained": False,
        "support_labels_retained": False,
        "support_physical_ids_retained": False,
        "query_rows_used_for_fit": 0,
        "target_optimizer_steps": 0,
        "query_state_updates": 0,
    }


@dataclass(frozen=True, slots=True)
class D106RDCERuntimeState:
    """A compact immutable support-only metric state with no FP32 attenuation."""

    asset: D106RDCEAsset
    active_k: int
    registered_class_count: int
    attenuation_fp16: np.ndarray
    capsule_id: str
    split_id: str
    row_id: str
    seed: int
    qknn_bank_sha256: str
    support_physical_root_sha256: str
    support_binding_sha256: str
    runtime_receipt_sha256: str
    row_authority_sha256: str | None = None
    deployment_status: str = NON_DEPLOYABLE_MATH_STATUS
    _row_authority_token: object | None = None
    query_rows_used_for_fit: int = 0
    target_optimizer_steps: int = 0
    query_state_updates: int = 0
    schema: str = RUNTIME_SCHEMA

    def __post_init__(self) -> None:
        if type(self.asset) is not D106RDCEAsset or self.schema != RUNTIME_SCHEMA:
            raise D106RDCERuntimeError("D106 runtime asset/schema drift")
        binding = self.binding
        formal = self.deployment_status == FORMAL_DEPLOYMENT_STATUS
        if self.deployment_status not in {
            FORMAL_DEPLOYMENT_STATUS,
            NON_DEPLOYABLE_MATH_STATUS,
        }:
            raise D106RDCERuntimeError("D106 runtime deployment status drift")
        if formal:
            if (
                not self.asset.is_formal_deployable
                or self.row_authority_sha256 is None
                or self._row_authority_token is not _ROW_AUTHORITY_LOADER_TOKEN
            ):
                raise D106RDCERuntimeError(
                    "formal runtime state requires asset and loader-origin row authority"
                )
        elif self.row_authority_sha256 is not None or self._row_authority_token is not None:
            raise D106RDCERuntimeError(
                "non-deployable runtime math state may not carry row authority"
            )
        if (
            self.query_rows_used_for_fit != 0
            or type(self.query_rows_used_for_fit) is not int
            or self.target_optimizer_steps != 0
            or type(self.target_optimizer_steps) is not int
            or self.query_state_updates != 0
            or type(self.query_state_updates) is not int
        ):
            raise D106RDCERuntimeError("D106 runtime lifecycle invariant drift")
        values = np.asarray(self.attenuation_fp16)
        if (
            values.dtype != np.dtype("<f2")
            or values.shape != (RDCE_RANK,)
            or not np.isfinite(values).all()
            or np.any(values < MIN_ATTENUATION_FP16)
            or np.any(values > MAX_ATTENUATION_FP16)
        ):
            raise D106RDCERuntimeError("D106 runtime FP16 attenuation range/dtype drift")
        if self.active_k == 1 and not np.array_equal(
            values,
            np.full(RDCE_RANK, np.float16(K1_ATTENUATION), dtype=np.float16),
        ):
            raise D106RDCERuntimeError("K=1 must retain fixed non-identity FP16 attenuation")
        receipt = _require_sha256(self.runtime_receipt_sha256, "runtime_receipt_sha256")
        _metric_eigenvalue_minimum(
            decode_d106_rdce_basis(self.asset), values.astype(np.float64)
        )
        payload = _runtime_payload(
            binding=binding,
            attenuation_fp16=values,
            deployment_status=self.deployment_status,
        )
        if _sha256_bytes(_canonical_bytes(payload)) != receipt:
            raise D106RDCERuntimeError("D106 runtime receipt drift")
        object.__setattr__(self, "attenuation_fp16", _readonly(values, np.dtype("<f2")))
        object.__setattr__(self, "runtime_receipt_sha256", receipt)

    @property
    def binding(self) -> D106RDCERuntimeBinding:
        return D106RDCERuntimeBinding(
            asset_binding_sha256=self.asset.binding_sha256,
            capsule_id=self.capsule_id,
            split_id=self.split_id,
            row_id=self.row_id,
            seed=self.seed,
            qknn_bank_sha256=self.qknn_bank_sha256,
            support_physical_root_sha256=self.support_physical_root_sha256,
            support_binding_sha256=self.support_binding_sha256,
            active_k=self.active_k,
            registered_class_count=self.registered_class_count,
            row_authority_sha256=self.row_authority_sha256,
        )

    @property
    def is_formal_deployable(self) -> bool:
        return self.deployment_status == FORMAL_DEPLOYMENT_STATUS

    @property
    def metric_eigenvalue_min(self) -> float:
        """Recompute the low-rank floor; it is not persistent wire metadata."""

        return _metric_eigenvalue_minimum(
            decode_d106_rdce_basis(self.asset),
            self.attenuation_fp16.astype(np.float64),
        )

    @property
    def attenuation(self) -> np.ndarray:
        """Return the sealed three-scalar FP16 attenuation vector."""

        return _readonly(self.attenuation_fp16, np.dtype("<f2"))


def _metric_from_basis_and_attenuation(
    basis: np.ndarray, attenuation: np.ndarray
) -> np.ndarray:
    """Dense audit-only reconstruction of M; deployment never calls this."""

    metric = np.eye(Z_DIM, dtype=np.float64) - basis.T @ (
        attenuation[:, None] * basis
    )
    metric = 0.5 * (metric + metric.T)
    eigenvalues = np.linalg.eigvalsh(metric)
    minimum = float(np.min(eigenvalues))
    if not math.isfinite(minimum) or minimum < MIN_METRIC_EIGENVALUE - 1.0e-10:
        raise _RuntimeScientificReject("metric_eigenvalue_below_0p05")
    return metric


def _metric_eigenvalue_minimum(
    basis: np.ndarray, attenuation: np.ndarray
) -> float:
    """Low-rank floor check used by fit/replay without allocating dense M."""

    directions = np.asarray(basis, dtype=np.float64)
    values = np.asarray(attenuation, dtype=np.float64)
    if (
        directions.shape != (RDCE_RANK, Z_DIM)
        or values.shape != (RDCE_RANK,)
        or not np.isfinite(directions).all()
        or not np.isfinite(values).all()
        or not np.allclose(
            directions @ directions.T,
            np.eye(RDCE_RANK),
            rtol=0.0,
            atol=2.0e-10,
        )
    ):
        raise D106RDCERuntimeError("D106 low-rank metric basis/attenuation drift")
    minimum = float(np.min(1.0 - values))
    if not math.isfinite(minimum) or minimum < MIN_METRIC_EIGENVALUE - 1.0e-10:
        raise _RuntimeScientificReject("metric_eigenvalue_below_0p05")
    return minimum


def _build_runtime_state_core(
    asset: D106RDCEAsset,
    support: D106RDCESupportRows,
    *,
    row_authority: _D106RDCERowAuthority | None,
    formal_loader_token: object | None,
) -> D106RDCERuntimeState:
    if row_authority is not None and formal_loader_token is not _ROW_AUTHORITY_LOADER_TOKEN:
        raise D106RDCERuntimeError("formal runtime core requires the row-loader token")
    verified = _verified_support(
        support, asset, row_authority=row_authority
    )
    active_k = verified.binding.active_k
    if active_k == 1:
        attenuation = np.full(
            RDCE_RANK, np.float16(K1_ATTENUATION), dtype=np.float16
        )
    else:
        basis = decode_d106_rdce_basis(asset)
        tau = decode_d106_rdce_tau(asset)
        class_scatter: list[np.ndarray] = []
        for _, group in verified.groups:
            residual = group - np.mean(group, axis=0, dtype=np.float64)
            projected = residual @ basis.T
            class_scatter.append(
                np.sum(np.square(projected), axis=0) / float(active_k - 1)
            )
        support_scatter = np.mean(
            np.stack(class_scatter, axis=0), axis=0, dtype=np.float64
        )
        if not np.isfinite(support_scatter).all() or np.any(support_scatter < 0.0):
            raise D106RDCERuntimeError("support projected scatter finite/nonnegative drift")
        a0 = min(MAX_ATTENUATION, 1.5 * active_k / float(active_k + 4))
        raw = a0 + GAMMA * np.tanh(
            np.log(
                (support_scatter + SUPPORT_SCATTER_EPSILON)
                / (tau + SUPPORT_SCATTER_EPSILON)
            )
        )
        attenuation = np.asarray(
            np.clip(
                raw,
                float(MIN_ATTENUATION_FP16),
                float(MAX_ATTENUATION_FP16),
            ),
            dtype=np.float16,
        )
    _metric_eigenvalue_minimum(
        decode_d106_rdce_basis(asset), attenuation.astype(np.float64)
    )
    deployment_status = (
        FORMAL_DEPLOYMENT_STATUS
        if row_authority is not None
        else NON_DEPLOYABLE_MATH_STATUS
    )
    payload = _runtime_payload(
        binding=verified.binding,
        attenuation_fp16=attenuation,
        deployment_status=deployment_status,
    )
    return D106RDCERuntimeState(
        asset=asset,
        active_k=verified.binding.active_k,
        registered_class_count=verified.binding.registered_class_count,
        attenuation_fp16=attenuation,
        capsule_id=verified.binding.capsule_id,
        split_id=verified.binding.split_id,
        row_id=verified.binding.row_id,
        seed=verified.binding.seed,
        qknn_bank_sha256=verified.binding.qknn_bank_sha256,
        support_physical_root_sha256=verified.binding.support_physical_root_sha256,
        support_binding_sha256=verified.binding.support_binding_sha256,
        runtime_receipt_sha256=_sha256_bytes(_canonical_bytes(payload)),
        row_authority_sha256=verified.binding.row_authority_sha256,
        deployment_status=deployment_status,
        _row_authority_token=formal_loader_token,
    )


def _build_runtime_state_math(
    asset: D106RDCEAsset,
    support: D106RDCESupportRows,
)-> D106RDCERuntimeState:
    """Private pure-math fit; its state is always NON_DEPLOYABLE."""

    return _build_runtime_state_core(
        asset,
        support,
        row_authority=None,
        formal_loader_token=None,
    )


def _build_runtime_state_from_loaded_authority(
    asset: D106RDCEAsset,
    support: D106RDCESupportRows,
    *,
    row_authority: _D106RDCERowAuthority,
) -> D106RDCERuntimeState:
    """Internal formal path reached only through the row-authority loader."""

    return _build_runtime_state_core(
        asset,
        support,
        row_authority=row_authority,
        formal_loader_token=_ROW_AUTHORITY_LOADER_TOKEN,
    )


def _try_fit_d106_rdce_runtime_math(
    asset: D106RDCEAsset,
    support: D106RDCESupportRows,
) -> D106RDCERuntimeState | D106RDCEScientificRejectReceipt:
    """Private pure-math fit; its output is always NON_DEPLOYABLE."""

    try:
        return _build_runtime_state_math(asset, support)
    except _RuntimeScientificReject as rejection:
        if type(asset) is not D106RDCEAsset:
            raise D106RDCERuntimeError("D106 runtime rejection requires an exact asset")
        return make_d106_rdce_scientific_reject(
            stage="phase2_runtime",
            reason=rejection.reason,
            lineage=asset.lineage,
            source_row_count=asset.source_row_count,
            source_class_count=asset.source_class_count,
        )


def _try_fit_d106_rdce_runtime_from_loaded_authority(
    asset: D106RDCEAsset,
    support: D106RDCESupportRows,
    *,
    row_authority: _D106RDCERowAuthority,
) -> D106RDCERuntimeState | D106RDCEScientificRejectReceipt:
    try:
        return _build_runtime_state_from_loaded_authority(
            asset, support, row_authority=row_authority
        )
    except _RuntimeScientificReject as rejection:
        if type(asset) is not D106RDCEAsset:
            raise D106RDCERuntimeError("D106 runtime rejection requires an exact asset")
        return make_d106_rdce_scientific_reject(
            stage="phase2_runtime",
            reason=rejection.reason,
            lineage=asset.lineage,
            source_row_count=asset.source_row_count,
            source_class_count=asset.source_class_count,
        )


def try_fit_d106_rdce_runtime(
    asset: D106RDCEAsset,
    support: D106RDCESupportRows,
    *,
    row_authority: _D106RDCERowAuthority,
) -> D106RDCERuntimeState | D106RDCEScientificRejectReceipt:
    """Formal fit requiring a canonical externally SHA-bound row authority."""

    return _try_fit_d106_rdce_runtime_from_loaded_authority(
        asset, support, row_authority=row_authority
    )


def fit_d106_rdce_runtime(
    asset: D106RDCEAsset,
    support: D106RDCESupportRows,
    *,
    row_authority: _D106RDCERowAuthority,
) -> D106RDCERuntimeState:
    """Strict formal support-only fit; self-signed receipts cannot substitute."""

    result = try_fit_d106_rdce_runtime(
        asset, support, row_authority=row_authority
    )
    if isinstance(result, D106RDCEScientificRejectReceipt):
        raise D106RDCERuntimeError(f"D106 scientific reject: {result.reason}")
    return result


def d106_rdce_metric_matrix(state: D106RDCERuntimeState) -> np.ndarray:
    """Return dense M only for audit/verification, never for query deployment."""

    if type(state) is not D106RDCERuntimeState:
        raise D106RDCERuntimeError("D106 metric requires an exact runtime state")
    metric = _metric_from_basis_and_attenuation(
        decode_d106_rdce_basis(state.asset),
        state.attenuation_fp16.astype(np.float64),
    )
    dense_minimum = float(np.min(np.linalg.eigvalsh(metric)))
    low_rank_minimum = _metric_eigenvalue_minimum(
        decode_d106_rdce_basis(state.asset),
        state.attenuation_fp16.astype(np.float64),
    )
    if abs(dense_minimum - low_rank_minimum) > 2.0e-8:
        raise D106RDCERuntimeError("D106 dense/low-rank metric audit drift")
    return np.ascontiguousarray(metric, dtype=np.float64)


def d106_rdce_sqrt_metric_matrix(state: D106RDCERuntimeState) -> np.ndarray:
    """Return dense M^1/2 only for audit/verification."""

    if type(state) is not D106RDCERuntimeState:
        raise D106RDCERuntimeError("D106 square-root metric requires an exact runtime state")
    basis = decode_d106_rdce_basis(state.asset)
    attenuation = state.attenuation_fp16.astype(np.float64)
    coefficients = 1.0 - np.sqrt(1.0 - attenuation)
    square_root = np.eye(Z_DIM, dtype=np.float64) - basis.T @ (
        coefficients[:, None] * basis
    )
    square_root = 0.5 * (square_root + square_root.T)
    if not np.allclose(
        square_root @ square_root,
        d106_rdce_metric_matrix(state),
        rtol=0.0,
        atol=2.0e-9,
    ):
        raise D106RDCERuntimeError("D106 square-root metric closure drift")
    return np.ascontiguousarray(square_root, dtype=np.float64)


def _low_rank_sqrt_apply(
    normalized: np.ndarray, basis: np.ndarray, attenuation: np.ndarray
) -> np.ndarray:
    """Apply I-B.T diag(1-sqrt(1-a)) B in exactly 2*3*160 MACs/row."""

    coefficients = 1.0 - np.sqrt(1.0 - attenuation)
    projected = normalized @ basis.T
    return normalized - (projected * coefficients[None, :]) @ basis


class _D106RDCEEphemeralScoringContext:
    """Opaque loader-origin decoded-basis context, never serialized into state."""

    __slots__ = (
        "_runtime_receipt_sha256",
        "_asset_binding_sha256",
        "_basis_bytes",
        "_attenuation_bytes",
        "_basis_content_sha256",
        "_loader_token",
        "_sealed",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        runtime_receipt_sha256: str,
        asset_binding_sha256: str,
        basis: np.ndarray,
        attenuation: np.ndarray,
        basis_content_sha256: str,
        _loader_token: object | None = None,
    ) -> None:
        if _loader_token is not _SCORING_CONTEXT_LOADER_TOKEN:
            raise D106RDCERuntimeError(
                "D106 scoring context requires a loader-origin creation token"
            )
        object.__setattr__(
            self,
            "_runtime_receipt_sha256",
            _require_sha256(runtime_receipt_sha256, "context runtime receipt"),
        )
        object.__setattr__(
            self,
            "_asset_binding_sha256",
            _require_sha256(asset_binding_sha256, "context asset binding"),
        )
        basis = np.asarray(basis)
        attenuation = np.asarray(attenuation)
        if (
            basis.dtype != np.float64
            or basis.shape != (RDCE_RANK, Z_DIM)
            or attenuation.dtype != np.float64
            or attenuation.shape != (RDCE_RANK,)
            or not np.isfinite(basis).all()
            or not np.isfinite(attenuation).all()
        ):
            raise D106RDCERuntimeError("D106 scoring context basis/attenuation drift")
        object.__setattr__(
            self,
            "_basis_content_sha256",
            _require_sha256(basis_content_sha256, "context basis content digest"),
        )
        object.__setattr__(
            self,
            "_basis_bytes",
            np.ascontiguousarray(basis, dtype=np.float64).tobytes(order="C"),
        )
        object.__setattr__(
            self,
            "_attenuation_bytes",
            np.ascontiguousarray(attenuation, dtype=np.float64).tobytes(order="C"),
        )
        object.__setattr__(self, "_loader_token", _loader_token)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise D106RDCERuntimeError("D106 scoring context is immutable")
        object.__setattr__(self, name, value)

    @property
    def runtime_receipt_sha256(self) -> str:
        return self._runtime_receipt_sha256

    @property
    def asset_binding_sha256(self) -> str:
        return self._asset_binding_sha256

    @property
    def basis(self) -> np.ndarray:
        return np.frombuffer(self._basis_bytes, dtype=np.float64).reshape(
            RDCE_RANK, Z_DIM
        )

    @property
    def attenuation(self) -> np.ndarray:
        return np.frombuffer(self._attenuation_bytes, dtype=np.float64)

    @property
    def basis_content_sha256(self) -> str:
        return self._basis_content_sha256


def _basis_content_digest(basis: np.ndarray) -> str:
    """Digest the exact decoded basis shape, dtype, and bytes for one context."""

    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema": "cvs.phase2.d106.rdce_gtsm_ephemeral_basis.v1",
                "basis": _array_receipt(np.asarray(basis)),
            }
        )
    )


_SCORING_CONTEXT_MINT_REGISTRY: weakref.WeakKeyDictionary[
    _D106RDCEEphemeralScoringContext, tuple[str, str, str, str]
] = weakref.WeakKeyDictionary()


def _scoring_context_mint_binding(
    context: _D106RDCEEphemeralScoringContext,
) -> tuple[str, str, str, str]:
    """Independent mint record; context-carried digests are not authority."""

    return (
        context.runtime_receipt_sha256,
        context.asset_binding_sha256,
        _basis_content_digest(context.basis),
        _sha256_bytes(np.ascontiguousarray(context.attenuation).tobytes(order="C")),
    )


def prepare_d106_rdce_scoring_context(
    state: D106RDCERuntimeState,
) -> _D106RDCEEphemeralScoringContext:
    """Decode rank-three basis once before a caller's scoring batch/call."""

    if type(state) is not D106RDCERuntimeState:
        raise D106RDCERuntimeError("D106 context requires an exact runtime state")
    basis = decode_d106_rdce_basis(state.asset)
    context = _D106RDCEEphemeralScoringContext(
        runtime_receipt_sha256=state.runtime_receipt_sha256,
        asset_binding_sha256=state.asset.binding_sha256,
        basis=basis,
        attenuation=state.attenuation_fp16.astype(np.float64),
        basis_content_sha256=_basis_content_digest(basis),
        _loader_token=_SCORING_CONTEXT_LOADER_TOKEN,
    )
    _SCORING_CONTEXT_MINT_REGISTRY[context] = _scoring_context_mint_binding(context)
    return context


def _checked_scoring_context(
    state: D106RDCERuntimeState,
    context: _D106RDCEEphemeralScoringContext | None,
) -> _D106RDCEEphemeralScoringContext:
    if context is None:
        return prepare_d106_rdce_scoring_context(state)
    if type(context) is not _D106RDCEEphemeralScoringContext:
        raise D106RDCERuntimeError("D106 transform requires an exact scoring context")
    if context._loader_token is not _SCORING_CONTEXT_LOADER_TOKEN:
        raise D106RDCERuntimeError(
            "D106 scoring context requires a loader-origin creation token"
        )
    minted_binding = _SCORING_CONTEXT_MINT_REGISTRY.get(context)
    if (
        minted_binding is None
        or _scoring_context_mint_binding(context) != minted_binding
    ):
        raise D106RDCERuntimeError(
            "D106 scoring context does not match its loader mint record"
        )
    if _basis_content_digest(context.basis) != context.basis_content_sha256:
        raise D106RDCERuntimeError("D106 scoring context basis content digest drift")
    if (
        context.runtime_receipt_sha256 != state.runtime_receipt_sha256
        or context.asset_binding_sha256 != state.asset.binding_sha256
        or not np.array_equal(
            context.attenuation, state.attenuation_fp16.astype(np.float64)
        )
    ):
        raise D106RDCERuntimeError("D106 scoring context/state binding drift")
    return context


def transform_d106_rdce_zid(
    state: D106RDCERuntimeState,
    z_id: np.ndarray,
    *,
    context: _D106RDCEEphemeralScoringContext | None = None,
) -> np.ndarray:
    """Apply phi through the low-rank path without reading or mutating state."""

    if type(state) is not D106RDCERuntimeState:
        raise D106RDCERuntimeError("D106 transform requires an exact runtime state")
    normalized = _normalized_support(z_id)
    active_context = _checked_scoring_context(state, context)
    transformed = _low_rank_sqrt_apply(
        normalized,
        active_context.basis,
        active_context.attenuation,
    )
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= EPSILON):
        raise D106RDCERuntimeError("D106 phi produced a zero-norm row")
    return _readonly(transformed / norms, np.dtype(np.float32))


def transform_d106_rdce_query(
    state: D106RDCERuntimeState,
    query_z_id: np.ndarray,
    *,
    context: _D106RDCEEphemeralScoringContext | None = None,
) -> np.ndarray:
    """Explicit query alias; it has no state-update surface."""

    return transform_d106_rdce_zid(state, query_z_id, context=context)


def _runtime_wire_header(state: D106RDCERuntimeState) -> dict[str, Any]:
    return {
        "schema": RUNTIME_WIRE_SCHEMA,
        "runtime": _runtime_payload(
            binding=state.binding,
            attenuation_fp16=state.attenuation_fp16,
            deployment_status=state.deployment_status,
        ),
        "runtime_receipt_sha256": state.runtime_receipt_sha256,
    }


def serialize_d106_rdce_runtime(state: D106RDCERuntimeState) -> bytes:
    """Seal the FP16 support-derived runtime state for replay."""

    if type(state) is not D106RDCERuntimeState or not state.is_formal_deployable:
        raise D106RDCERuntimeError(
            "D106 runtime serialization requires a formal loader-authorized state"
        )
    header = _canonical_bytes(_runtime_wire_header(state))
    return (
        RUNTIME_WIRE_MAGIC
        + struct.pack(">I", len(header))
        + header
        + np.ascontiguousarray(state.attenuation_fp16).tobytes(order="C")
    )


def deserialize_d106_rdce_runtime(
    payload: bytes,
    *,
    asset: D106RDCEAsset,
    expected_wire_sha256: str,
    expected_binding: D106RDCERuntimeBinding,
) -> D106RDCERuntimeState:
    """Replay only a pinned runtime wire for its exact asset/support binding."""

    if not isinstance(payload, bytes) or len(payload) <= len(RUNTIME_WIRE_MAGIC) + 4:
        raise D106RDCERuntimeError("D106 runtime wire must be non-empty bytes")
    expected_wire = _require_sha256(expected_wire_sha256, "expected_wire_sha256")
    if _sha256_bytes(payload) != expected_wire:
        raise D106RDCERuntimeError("D106 runtime wire SHA256 trust anchor mismatch")
    if type(asset) is not D106RDCEAsset or not asset.is_formal_deployable:
        raise D106RDCERuntimeError(
            "D106 runtime replay requires a formal loader-authorized asset"
        )
    if type(expected_binding) is not D106RDCERuntimeBinding:
        raise D106RDCERuntimeError(
            "D106 runtime replay requires exact typed expected binding"
        )
    if asset.binding_sha256 != expected_binding.asset_binding_sha256:
        raise D106RDCERuntimeError("D106 runtime replay asset binding mismatch")
    if expected_binding.row_authority_sha256 is None:
        raise D106RDCERuntimeError(
            "D106 runtime replay requires an expected external row authority"
        )
    if not payload.startswith(RUNTIME_WIRE_MAGIC):
        raise D106RDCERuntimeError("D106 runtime wire magic drift")
    offset = len(RUNTIME_WIRE_MAGIC)
    header_size = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4
    if header_size <= 0 or offset + header_size > len(payload):
        raise D106RDCERuntimeError("D106 runtime wire header length drift")
    raw_header = payload[offset : offset + header_size]
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RDCERuntimeError(
            "D106 runtime wire header must be canonical UTF-8 JSON"
        ) from error
    if type(header) is not dict or raw_header != _canonical_bytes(header):
        raise D106RDCERuntimeError(
            "D106 runtime wire header is not original canonical JSON"
        )
    offset += header_size
    if (
        set(header) != {"schema", "runtime", "runtime_receipt_sha256"}
        or header["schema"] != RUNTIME_WIRE_SCHEMA
        or type(header["runtime"]) is not dict
    ):
        raise D106RDCERuntimeError("D106 runtime wire header schema drift")
    runtime = header["runtime"]
    expected_keys = {
        "schema",
        "candidate_id",
        "asset_binding_sha256",
        "capsule_id",
        "split_id",
        "row_id",
        "seed",
        "qknn_bank_sha256",
        "support_physical_root_sha256",
        "support_binding_sha256",
        "active_k",
        "registered_class_count",
        "row_authority_sha256",
        "attenuation_fp16",
        "deployment_status",
        "support_rows_retained",
        "support_labels_retained",
        "support_physical_ids_retained",
        "query_rows_used_for_fit",
        "target_optimizer_steps",
        "query_state_updates",
    }
    if set(runtime) != expected_keys:
        raise D106RDCERuntimeError("D106 runtime wire payload keys drift")
    if (
        runtime["deployment_status"] != FORMAL_DEPLOYMENT_STATUS
        or runtime["row_authority_sha256"] is None
    ):
        raise D106RDCERuntimeError(
            "D106 runtime wire may not carry a non-deployable math state"
        )
    length = RDCE_RANK * np.dtype("<f2").itemsize
    if offset + length != len(payload):
        raise D106RDCERuntimeError("D106 runtime wire payload length/trailing drift")
    attenuation = np.frombuffer(
        payload[offset : offset + length], dtype=np.dtype("<f2")
    ).copy()
    if _array_receipt(attenuation) != runtime["attenuation_fp16"]:
        raise D106RDCERuntimeError("D106 runtime wire FP16 attenuation receipt drift")
    state = D106RDCERuntimeState(
        asset=asset,
        active_k=runtime["active_k"],
        registered_class_count=runtime["registered_class_count"],
        attenuation_fp16=attenuation,
        capsule_id=runtime["capsule_id"],
        split_id=runtime["split_id"],
        row_id=runtime["row_id"],
        seed=runtime["seed"],
        qknn_bank_sha256=runtime["qknn_bank_sha256"],
        support_physical_root_sha256=runtime["support_physical_root_sha256"],
        support_binding_sha256=runtime["support_binding_sha256"],
        runtime_receipt_sha256=header["runtime_receipt_sha256"],
        row_authority_sha256=runtime["row_authority_sha256"],
        deployment_status=runtime["deployment_status"],
        _row_authority_token=_ROW_AUTHORITY_LOADER_TOKEN,
    )
    if state.binding != expected_binding:
        raise D106RDCERuntimeError("D106 runtime wire expected typed binding mismatch")
    if _runtime_wire_header(state) != header:
        raise D106RDCERuntimeError("D106 runtime wire canonical header drift")
    return state


def audit_d106_rdce_runtime(state: D106RDCERuntimeState) -> dict[str, Any]:
    """Return the state/resource receipt, including the real deployment path."""

    if type(state) is not D106RDCERuntimeState:
        raise D106RDCERuntimeError("D106 audit requires an exact runtime state")
    receipt = {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "runtime_receipt_sha256": state.runtime_receipt_sha256,
        **state.binding.as_dict(),
        "effective_rank": RDCE_RANK,
        "metric_formula": "I-B.T@diag(a)@B",
        "phi_formula": "norm(z-B.T@diag(1-sqrt(1-a))@(B@z))",
        "metric_eigenvalue_min": state.metric_eigenvalue_min,
        "basis_storage": "phase1_int8_plus_fp16_scale",
        "runtime_attenuation_storage": "three_fp16_scalars",
        "runtime_wire_persistent_metric_eigenvalue": False,
        "asset_numeric_payload_bytes": (
            RDCE_RANK * Z_DIM
            + RDCE_RANK * np.dtype(np.float16).itemsize
            + RDCE_RANK
            + RDCE_RANK * np.dtype(np.float16).itemsize
            + RDCE_RANK
            + RDCE_RANK * np.dtype(np.float16).itemsize
        ),
        "runtime_dynamic_numeric_bytes": RDCE_RANK * np.dtype(np.float16).itemsize,
        "persistent_source_rows": 0,
        "persistent_source_ids_or_names": 0,
        "persistent_support_rows": 0,
        "persistent_support_labels": 0,
        "persistent_support_physical_ids": 0,
        "query_rows_used_for_fit": state.query_rows_used_for_fit,
        "target_optimizer_steps": state.target_optimizer_steps,
        "query_state_updates": state.query_state_updates,
        "deployment_phi_path": "two_rank3_projections_no_dense_160x160",
        "dense_metric_path": "audit_only",
        "projection_mac_per_row": 2 * RDCE_RANK * Z_DIM,
        "basis_decode_shape": [RDCE_RANK, Z_DIM],
        "basis_decode_calls_per_scoring_context": 1,
        "basis_decode_calls_per_transform_with_context": 0,
        "basis_decode_calls_per_transform_without_context": 1,
        "basis_decode_scope": "once_per_ephemeral_scoring_context_not_per_row",
        "normalization_scope": "input_l2_per_row_plus_output_l2_per_row",
        "elementwise_scope": "rank3_coefficients_and_projection_rescale_per_row",
        "total_transform_cost_formula": (
            "projection_mac_per_row+input_l2_normalization+"
            "rank3_elementwise+output_l2_normalization"
        ),
        "class_balanced_support_scatter": state.active_k >= 2,
        "k1_fixed_nonidentity": state.active_k == 1,
    }
    receipt["resource_receipt_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return receipt


__all__ = [
    "ALLOWED_K",
    "GAMMA",
    "K1_ATTENUATION",
    "MAX_ATTENUATION",
    "MIN_ATTENUATION",
    "MIN_METRIC_EIGENVALUE",
    "RUNTIME_WIRE_MAGIC",
    "SUPPORT_SCATTER_EPSILON",
    "D106RDCERuntimeBinding",
    "D106RDCERuntimeError",
    "D106RDCERuntimeState",
    "D106RDCESupportRows",
    "ROW_AUTHORITY_SCHEMA",
    "audit_d106_rdce_runtime",
    "d106_rdce_metric_matrix",
    "d106_rdce_sqrt_metric_matrix",
    "deserialize_d106_rdce_runtime",
    "fit_d106_rdce_runtime",
    "load_d106_rdce_row_authority",
    "prepare_d106_rdce_scoring_context",
    "serialize_d106_rdce_runtime",
    "transform_d106_rdce_query",
    "transform_d106_rdce_zid",
    "try_fit_d106_rdce_runtime",
]
