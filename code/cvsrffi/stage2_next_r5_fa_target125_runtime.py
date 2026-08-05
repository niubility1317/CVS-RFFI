"""Truth-free D108-package adapter for ``NEXT-R5 FA-RDCE3 -> qKNN``.

This module deliberately owns only the Target125 *runtime boundary*.  It
reopens the already sealed D108/D92 packages, forwards their SHA-bound
checkpoint to the canonical signed-totalized unit ``z_id160`` representation and
returns the two registration inputs required by the frozen FA/qKNN core.  It
does not load a truth sidecar, infer a query role, choose a row, or score a
prediction.

The core implementation lives in ``stage2_next_r5_fa_target125_core``.  Its
public four-state API is intentionally injected here while that module is
being landed independently, which keeps the sealed-data adapter testable and
prevents a duplicate implementation of FA-RDCE3 or qKNN.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from . import stage2_next_r5_fa_target125_matrix as matrix
from .stage2_d92_pr160_runtime import (
    EXTRACTOR_RUNTIME_SHA256 as PR160_EXTRACTOR_RUNTIME_SHA256,
    SOURCE_RUNTIME_SHA256 as PR160_SOURCE_RUNTIME_SHA256,
    load_pr160_runtime,
)


RUNTIME_ADAPTER_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.runtime_adapter.v1"
MATERIALIZED_STATE_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.zid160_state.v1"
QUERY_ISOLATION_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.query_isolation.v1"
PREPARED_PLAN_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.prepared_plan.v1"
PREPARED_CONTEXT_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.prepared_context.v1"
PREPARE_RECEIPT_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.prepare_receipt.v1"
PREDICTION_SHARD_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.prediction_shard.v1"
METHOD_LOCK_SCHEMA = "cvs.stage2.next_r5.fa_rdce3_qknn.target125.method_lock.v2"
SHARD_COUNT = 8
ZID160_TOTALIZATION_SCHEMA = (
    "cvs.phase2.next_r5.fa_rdce3_qknn.target125.zid160_totalization.v1"
)
ZID160_TOTALIZATION_REVISION = "same_iq_pr160_graph_relu_signed_fallback_r3"
PR160_EXTRACTOR_RUNTIME_SIZE_BYTES = 4_618_957
PR160_RUNTIME_BINDING_SCHEMA = (
    "cvs.phase2.next_r5.fa_rdce3_qknn.target125.pr160_runtime_binding.v1"
)
PR160_GRAPH_RELU_BINDING_MODE = "exact_relu_from_same_pr160_graph_pre"


class NextR5FATarget125RuntimeError(ValueError):
    """Raised when a sealed Target125 package cannot be adapted safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tokens(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise NextR5FATarget125RuntimeError(f"{name} must be a lowercase SHA256")
    return value


def _canonical_sha256(value: Any) -> str:
    return matrix.canonical_sha256(value)


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NextR5FATarget125RuntimeError("unsafe immutable output parent")
    raw = matrix.canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, *, expected_sha256: str, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise NextR5FATarget125RuntimeError(f"{name} must be a regular file")
    if _sha256_file(path) != _sha(expected_sha256, f"{name} SHA256"):
        raise NextR5FATarget125RuntimeError(f"{name} SHA mismatch")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR5FATarget125RuntimeError(f"{name} must be UTF-8 JSON") from error
    if not isinstance(result, dict):
        raise NextR5FATarget125RuntimeError(f"{name} must contain an object")
    return result


def _readonly_float32(value: Any, *, name: str) -> np.ndarray:
    """Validate the frozen signed-totalized R0 unit representation."""

    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[0] < 1
        or rows.shape[1] != matrix.FEATURE_DIM
        or not np.isfinite(rows).all()
    ):
        raise NextR5FATarget125RuntimeError(
            f"{name} must be finite float32 [N,{matrix.FEATURE_DIM}]"
        )
    norms = np.linalg.norm(rows.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise NextR5FATarget125RuntimeError(f"{name} must be canonical unit z_id160")
    result = np.ascontiguousarray(rows, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def _finite_feature_rows(value: Any, *, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[0] < 1
        or rows.shape[1] != matrix.FEATURE_DIM
        or not np.isfinite(rows).all()
    ):
        raise NextR5FATarget125RuntimeError(
            f"{name} must be finite float32 [N,{matrix.FEATURE_DIM}]"
        )
    return np.ascontiguousarray(rows, dtype=np.float32)


def _physical_id_row_root(physical_ids: Sequence[str], rows: np.ndarray) -> str:
    if len(physical_ids) != len(rows):
        raise NextR5FATarget125RuntimeError("physical-ID / z_id160 row binding drift")
    return _canonical_sha256(
        [
            {
                "physical_id": physical_id,
                "row_sha256": hashlib.sha256(
                    np.ascontiguousarray(row).tobytes(order="C")
                ).hexdigest(),
            }
            for physical_id, row in zip(physical_ids, rows, strict=True)
        ]
    )


def _pr160_receipt_binding(value: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "schema",
        "source_runtime_sha256",
        "extractor_runtime_sha256",
        "extractor_runtime_size_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise NextR5FATarget125RuntimeError("PR160 runtime receipt binding closure drift")
    source_runtime_sha256 = _sha(
        value.get("source_runtime_sha256"), "PR160 source runtime SHA256"
    )
    extractor_runtime_sha256 = _sha(
        value.get("extractor_runtime_sha256"), "PR160 extractor runtime SHA256"
    )
    size_bytes = value.get("extractor_runtime_size_bytes")
    if (
        value.get("schema") != PR160_RUNTIME_BINDING_SCHEMA
        or source_runtime_sha256 != PR160_SOURCE_RUNTIME_SHA256
        or extractor_runtime_sha256 != PR160_EXTRACTOR_RUNTIME_SHA256
        or type(size_bytes) is not int
        or size_bytes != PR160_EXTRACTOR_RUNTIME_SIZE_BYTES
    ):
        raise NextR5FATarget125RuntimeError("PR160 runtime receipt identity drift")
    return MappingProxyType(
        {
            "schema": PR160_RUNTIME_BINDING_SCHEMA,
            "source_runtime_sha256": source_runtime_sha256,
            "extractor_runtime_sha256": extractor_runtime_sha256,
            "extractor_runtime_size_bytes": size_bytes,
        }
    )


def _bind_pr160_extractor_runtime(
    *,
    extractor_runtime_path: str | Path,
    expected_extractor_runtime_sha256: str,
    source_runtime_sha256: str,
    expected_size_bytes: int,
) -> Mapping[str, Any]:
    """Verify the graph-derived PR160 artifact before it can enter a plan."""

    source_sha = _sha(source_runtime_sha256, "PR160 source runtime SHA256")
    extractor_sha = _sha(
        expected_extractor_runtime_sha256, "PR160 extractor runtime SHA256"
    )
    path = Path(extractor_runtime_path)
    if (
        source_sha != PR160_SOURCE_RUNTIME_SHA256
        or extractor_sha != PR160_EXTRACTOR_RUNTIME_SHA256
        or type(expected_size_bytes) is not int
        or expected_size_bytes != PR160_EXTRACTOR_RUNTIME_SIZE_BYTES
        or path.is_symlink()
        or not path.is_file()
    ):
        raise NextR5FATarget125RuntimeError("PR160 extractor runtime identity drift")
    source = path.resolve(strict=True)
    if (
        source.stat().st_size != expected_size_bytes
        or _sha256_file(source) != extractor_sha
    ):
        raise NextR5FATarget125RuntimeError("PR160 extractor runtime file binding drift")
    return MappingProxyType(
        {
            "path": str(source),
            "sha256": extractor_sha,
            "size_bytes": expected_size_bytes,
            "source_runtime_sha256": source_sha,
        }
    )


def _prepared_pr160_extractor_runtime(
    *, plan: Mapping[str, Any], source_plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Reload the exact PR160 graph binding from the immutable NEXT-R5 plan."""

    identity = plan.get("identity")
    source_identity = source_plan.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(source_identity, Mapping):
        raise NextR5FATarget125RuntimeError("prepared PR160 runtime identity is missing")
    source_runtime_sha256 = _sha(
        source_identity.get("d92_sealed_runtime_sha256"),
        "prepared D92 source runtime SHA256",
    )
    if (
        source_runtime_sha256 != PR160_SOURCE_RUNTIME_SHA256
        or identity.get("d92_sealed_runtime_sha256") != source_runtime_sha256
    ):
        raise NextR5FATarget125RuntimeError("prepared D92/PR160 source runtime binding drift")
    descriptor = identity.get("pr160_extractor_runtime")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
        "source_runtime_sha256",
    }:
        raise NextR5FATarget125RuntimeError("prepared PR160 extractor descriptor drift")
    if descriptor.get("source_runtime_sha256") != source_runtime_sha256:
        raise NextR5FATarget125RuntimeError("prepared PR160 extractor source binding drift")
    return _bind_pr160_extractor_runtime(
        extractor_runtime_path=descriptor["path"],
        expected_extractor_runtime_sha256=descriptor["sha256"],
        source_runtime_sha256=source_runtime_sha256,
        expected_size_bytes=descriptor["size_bytes"],
    )


def _totalize_same_iq_zid160(
    pre_relu: np.ndarray,
    pr160_graph_post_relu: np.ndarray,
    physical_ids: Sequence[str],
    *,
    scope: str,
    pr160_runtime_binding: Mapping[str, Any],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Preserve the canonical R0 ReLU rule; totalize only exact graph zeros."""

    if scope not in {"REG0_support", "REG0_query", "REG1_support", "REG1_query"}:
        raise NextR5FATarget125RuntimeError("z_id160 totalization scope drift")
    signed = _finite_feature_rows(pre_relu, name="z_id160 pre-ReLU")
    graph_relu = _finite_feature_rows(
        pr160_graph_post_relu, name="same-graph post-ReLU z_id160"
    )
    if signed.shape != graph_relu.shape:
        raise NextR5FATarget125RuntimeError("pre-ReLU / same-graph z_id160 shape drift")
    runtime_binding = _pr160_receipt_binding(pr160_runtime_binding)
    ids = _tokens(physical_ids, name=f"{scope} physical_ids", expected=len(signed))
    if np.any(graph_relu < np.float32(0.0)):
        raise NextR5FATarget125RuntimeError("same-graph z_id160 is not post-ReLU")
    recomputed_relu = np.maximum(signed, np.float32(0.0))
    if not np.array_equal(graph_relu, recomputed_relu):
        raise NextR5FATarget125RuntimeError("same-graph pre-ReLU / ReLU binding drift")
    replace = np.all(graph_relu == np.float32(0.0), axis=1)
    if np.any(replace) and not np.all(
        recomputed_relu[replace] == np.float32(0.0)
    ):
        raise NextR5FATarget125RuntimeError(
            "same-graph zero row is not an exact same-IQ ReLU zero"
        )
    signed_norms = np.linalg.norm(signed.astype(np.float64), axis=1)
    if (
        not np.isfinite(signed_norms).all()
        or np.any(signed_norms[replace] <= 0.0)
    ):
        raise NextR5FATarget125RuntimeError(
            "exact-zero pre-ReLU row cannot be signed-totalized"
        )
    output = np.empty_like(graph_relu)
    try:
        from .stage2_zid_student_t_qknn import normalize_zid_rows

        if np.any(~replace):
            # Graph-derived pre-ReLU uses the original R0 ReLU map for every
            # nonzero row.  Only an exact all-negative graph row is totalized.
            output[~replace] = normalize_zid_rows(graph_relu[~replace])
        if np.any(replace):
            output[replace] = normalize_zid_rows(signed[replace])
    except Exception as error:
        raise NextR5FATarget125RuntimeError(
            "z_id160 totalization normalization failed"
        ) from error
    output = _readonly_float32(output, name=f"{scope} totalized z_id160")
    replaced_ids = [physical_id for physical_id, selected in zip(ids, replace, strict=True) if selected]
    receipt: dict[str, Any] = {
        "schema": ZID160_TOTALIZATION_SCHEMA,
        "revision": ZID160_TOTALIZATION_REVISION,
        "scope": scope,
        "row_count": len(ids),
        "replaced_count": len(replaced_ids),
        "ordered_physical_ids_sha256": _canonical_sha256(list(ids)),
        "replaced_physical_id_root_sha256": _canonical_sha256(sorted(replaced_ids)),
        "pr160_runtime_binding": dict(runtime_binding),
        "input_pr160_pre_relu_array_receipt": _array_receipt(signed),
        "same_iq_pr160_recomputed_relu_array_receipt": _array_receipt(recomputed_relu),
        "same_iq_pr160_graph_relu_array_receipt": _array_receipt(graph_relu),
        "physical_id_to_pr160_pre_relu_row_root_sha256": _physical_id_row_root(ids, signed),
        "physical_id_to_pr160_graph_relu_row_root_sha256": _physical_id_row_root(
            ids, graph_relu
        ),
        "pre_relu_to_pr160_graph_relu_binding": {
            "mode": PR160_GRAPH_RELU_BINDING_MODE,
            "row_count": len(ids),
            "byte_exact": bool(np.array_equal(graph_relu, recomputed_relu)),
            "all_rows_bound": True,
        },
        "output_zid160_array_receipt": _array_receipt(output),
        "nonzero_rows_reuse_original_r0_semantics": True,
        "same_fixed_received_iq": True,
        "query_truth_access": False,
        "query_role_access": False,
        "state_updated": False,
    }
    receipt["totalization_receipt_sha256"] = _canonical_sha256(receipt)
    return output, MappingProxyType(receipt)


def _validate_totalization_receipt(
    value: Mapping[str, Any],
    *,
    rows: np.ndarray,
    physical_ids: Sequence[str],
    scope: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NextR5FATarget125RuntimeError("z_id160 totalization receipt is missing")
    receipt = dict(value)
    claimed = receipt.pop("totalization_receipt_sha256", None)
    binding = receipt.get("pre_relu_to_pr160_graph_relu_binding")
    if not isinstance(binding, Mapping):
        raise NextR5FATarget125RuntimeError("z_id160 totalization ReLU binding is missing")
    try:
        runtime_binding = _pr160_receipt_binding(receipt.get("pr160_runtime_binding"))
    except NextR5FATarget125RuntimeError as error:
        raise NextR5FATarget125RuntimeError(
            "z_id160 totalization PR160 runtime binding drift"
        ) from error
    binding_fields = {
        "mode",
        "row_count",
        "byte_exact",
        "all_rows_bound",
    }
    if (
        receipt.get("schema") != ZID160_TOTALIZATION_SCHEMA
        or receipt.get("revision") != ZID160_TOTALIZATION_REVISION
        or receipt.get("scope") != scope
        or receipt.get("row_count") != len(rows)
        or receipt.get("ordered_physical_ids_sha256")
        != _canonical_sha256(list(physical_ids))
        or receipt.get("output_zid160_array_receipt") != _array_receipt(rows)
        or dict(runtime_binding) != receipt.get("pr160_runtime_binding")
        or not isinstance(receipt.get("input_pr160_pre_relu_array_receipt"), Mapping)
        or not isinstance(receipt.get("same_iq_pr160_recomputed_relu_array_receipt"), Mapping)
        or not isinstance(receipt.get("same_iq_pr160_graph_relu_array_receipt"), Mapping)
        or not isinstance(receipt.get("physical_id_to_pr160_pre_relu_row_root_sha256"), str)
        or not isinstance(
            receipt.get("physical_id_to_pr160_graph_relu_row_root_sha256"), str
        )
        or set(binding) != binding_fields
        or binding.get("mode") != PR160_GRAPH_RELU_BINDING_MODE
        or binding.get("row_count") != len(rows)
        or binding.get("byte_exact") is not True
        or binding.get("all_rows_bound") is not True
        or receipt.get("nonzero_rows_reuse_original_r0_semantics") is not True
        or receipt.get("same_fixed_received_iq") is not True
        or receipt.get("query_truth_access") is not False
        or receipt.get("query_role_access") is not False
        or receipt.get("state_updated") is not False
        or claimed != _canonical_sha256(receipt)
    ):
        raise NextR5FATarget125RuntimeError("z_id160 totalization receipt binding drift")
    _sha(
        receipt["physical_id_to_pr160_pre_relu_row_root_sha256"],
        "PR160 pre-ReLU physical-row root SHA256",
    )
    _sha(
        receipt["physical_id_to_pr160_graph_relu_row_root_sha256"],
        "PR160 graph ReLU physical-row root SHA256",
    )
    receipt["totalization_receipt_sha256"] = claimed
    return MappingProxyType(receipt)


def _tokens(
    value: Sequence[str], *, name: str, expected: int | None = None, unique: bool = True
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise NextR5FATarget125RuntimeError(f"{name} must be a sequence")
    result = tuple(value)
    if (
        (expected is not None and len(result) != expected)
        or not result
        or any(type(item) is not str or not item for item in result)
        or (unique and len(set(result)) != len(result))
    ):
        raise NextR5FATarget125RuntimeError(f"{name} has invalid opaque IDs")
    return result


def _class_indices(value: Sequence[int], *, name: str, expected: int | None = None) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise NextR5FATarget125RuntimeError(f"{name} must be an integer sequence")
    result = tuple(value)
    if (
        (expected is not None and len(result) != expected)
        or not result
        or any(type(item) is not int for item in result)
        or result != tuple(range(len(result)))
    ):
        raise NextR5FATarget125RuntimeError(
            f"{name} must be one continuous zero-based index sequence"
        )
    return result


def _registered_class_records(
    manifest: Mapping[str, Any], *, name: str
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    records = manifest.get("registered_classes")
    if not isinstance(records, list) or not records:
        raise NextR5FATarget125RuntimeError(f"{name} registered class records are missing")
    indices: list[int] = []
    handles: list[str] = []
    for expected_index, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or set(record) != {"class_index", "class_handle"}
            or type(record.get("class_index")) is not int
            or record["class_index"] != expected_index
            or type(record.get("class_handle")) is not str
            or not record["class_handle"]
        ):
            raise NextR5FATarget125RuntimeError(
                f"{name} registered class record/index drift"
            )
        indices.append(record["class_index"])
        handles.append(record["class_handle"])
    return (
        _class_indices(indices, name=f"{name}.registered_class_indices"),
        _tokens(handles, name=f"{name}.registered_class_handles"),
    )


@dataclass(frozen=True, slots=True)
class Target125RegistrationInput:
    """One truth-free canonical R0 registration package.

    ``REG0`` contains only the six old classes and its old-query package;
    ``REG1`` contains the appended registry and its registration query package.
    The type intentionally has no truth, role, quota, score, or metric field.
    """

    registration_phase: str
    registered_classes: Sequence[str]
    registered_class_indices: Sequence[int]
    support_zid160: np.ndarray
    support_labels: Sequence[str]
    support_physical_ids: Sequence[str]
    query_zid160: np.ndarray
    query_physical_ids: Sequence[str]
    support_totalization_receipt: Mapping[str, Any]
    query_totalization_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.registration_phase not in ("REG0", "REG1"):
            raise NextR5FATarget125RuntimeError("registration_phase must be REG0 or REG1")
        classes = _tokens(self.registered_classes, name="registered_classes")
        indices = _class_indices(
            self.registered_class_indices,
            name="registered_class_indices",
            expected=len(classes),
        )
        if (
            (self.registration_phase == "REG0" and indices != tuple(range(matrix.OLD_CLASS_COUNT)))
            or (
                self.registration_phase == "REG1"
                and indices[: matrix.OLD_CLASS_COUNT] != tuple(range(matrix.OLD_CLASS_COUNT))
            )
        ):
            raise NextR5FATarget125RuntimeError("registration class-index phase drift")
        support = _readonly_float32(self.support_zid160, name="support_zid160")
        query = _readonly_float32(self.query_zid160, name="query_zid160")
        labels = _tokens(self.support_labels, name="support_labels", expected=len(support), unique=False)
        support_ids = _tokens(
            self.support_physical_ids, name="support_physical_ids", expected=len(support)
        )
        query_ids = _tokens(
            self.query_physical_ids, name="query_physical_ids", expected=len(query)
        )
        if any(label not in classes for label in labels):
            raise NextR5FATarget125RuntimeError("support labels escape registered classes")
        counts = tuple(labels.count(label) for label in classes)
        if not counts or len(set(counts)) != 1 or counts[0] not in (1, 5, 10):
            raise NextR5FATarget125RuntimeError("support must close a balanced frozen K")
        if set(support_ids).intersection(query_ids):
            raise NextR5FATarget125RuntimeError("support/query physical IDs overlap")
        support_receipt = _validate_totalization_receipt(
            self.support_totalization_receipt,
            rows=support,
            physical_ids=support_ids,
            scope=f"{self.registration_phase}_support",
        )
        query_receipt = _validate_totalization_receipt(
            self.query_totalization_receipt,
            rows=query,
            physical_ids=query_ids,
            scope=f"{self.registration_phase}_query",
        )
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "registered_class_indices", indices)
        object.__setattr__(self, "support_zid160", support)
        object.__setattr__(self, "support_labels", labels)
        object.__setattr__(self, "support_physical_ids", support_ids)
        object.__setattr__(self, "query_zid160", query)
        object.__setattr__(self, "query_physical_ids", query_ids)
        object.__setattr__(self, "support_totalization_receipt", support_receipt)
        object.__setattr__(self, "query_totalization_receipt", query_receipt)

    @property
    def active_k(self) -> int:
        return self.support_labels.count(self.registered_classes[0])

    @property
    def support_physical_root_sha256(self) -> str:
        return _sha256_tokens(self.support_physical_ids)

    @property
    def query_physical_root_sha256(self) -> str:
        return _sha256_tokens(self.query_physical_ids)


class FourStateExecutor(Protocol):
    """Narrow injected bridge to the independently owned FA/qKNN core."""

    def __call__(
        self,
        *,
        outer_row: matrix.Target125OuterRow,
        scene: str,
        reg0: Target125RegistrationInput,
        reg1: Target125RegistrationInput,
        source_row: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Target125ConditionInput:
    """One matched outer/scene condition before FA/qKNN execution."""

    outer_row: matrix.Target125OuterRow
    scene: str
    source_row: Mapping[str, Any]
    reg0: Target125RegistrationInput
    reg1: Target125RegistrationInput

    def __post_init__(self) -> None:
        if type(self.outer_row) is not matrix.Target125OuterRow:
            raise NextR5FATarget125RuntimeError("condition requires an exact frozen outer row")
        if self.scene not in matrix.SCENES:
            raise NextR5FATarget125RuntimeError("condition scene is outside Target125")
        if not isinstance(self.source_row, Mapping):
            raise NextR5FATarget125RuntimeError("condition source_row must be a sealed mapping")
        required = {
            "outer_id", "receiver", "seed", "k_shot", "active_k", "new_count",
            "source_pool_k", "packages", "authority_bundle",
        }
        if not required.issubset(self.source_row):
            raise NextR5FATarget125RuntimeError("sealed D108 source row field closure drift")
        row = self.outer_row
        if (
            self.source_row["receiver"] != row.receiver
            or self.source_row["seed"] != row.seed
            or self.source_row["k_shot"] != row.k_shot
            or self.source_row["active_k"] != row.k_shot
            or self.source_row["new_count"] != row.new_count
            or self.source_row["source_pool_k"] != row.source_pool_k
        ):
            raise NextR5FATarget125RuntimeError("D108 source row / Target125 outer binding drift")
        if (
            self.reg0.registration_phase != "REG0"
            or self.reg1.registration_phase != "REG1"
            or self.reg0.active_k != row.k_shot
            or self.reg1.active_k != row.k_shot
            or len(self.reg0.registered_classes) != matrix.OLD_CLASS_COUNT
            or self.reg0.registered_class_indices != tuple(range(matrix.OLD_CLASS_COUNT))
            or self.reg1.registered_class_indices[: matrix.OLD_CLASS_COUNT]
            != self.reg0.registered_class_indices
            or self.reg1.registered_classes[: matrix.OLD_CLASS_COUNT]
            != self.reg0.registered_classes
            or len(self.reg1.registered_classes) != matrix.OLD_CLASS_COUNT + row.new_count
        ):
            raise NextR5FATarget125RuntimeError("REG0/REG1 registry or K binding drift")
        old_count = matrix.OLD_CLASS_COUNT * row.k_shot
        if (
            self.reg1.support_labels[:old_count] != self.reg0.support_labels
            or self.reg1.support_physical_ids[:old_count] != self.reg0.support_physical_ids
            or not np.array_equal(
                self.reg1.support_zid160[:old_count], self.reg0.support_zid160
            )
        ):
            raise NextR5FATarget125RuntimeError(
                "REG1 must byte-preserve each REG0 old-class support row"
            )


def query_isolation_receipt() -> Mapping[str, Any]:
    """Return the query-use proof attached to every Target125 condition."""

    return MappingProxyType(
        {
            "schema": QUERY_ISOLATION_SCHEMA,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_truth_access": False,
            "query_role_access": False,
            "class_quota_access": False,
            "true_batch_class_count_access": False,
            "query_batch_dependency": False,
            "global_reassignment_calls": 0,
            "source_runtime_access": False,
            "clean_runtime_access": False,
            "phase2_optimizer_steps": 0,
            "phase2_backward_calls": 0,
        }
    )


def execute_target125_condition(
    condition: Target125ConditionInput, *, executor: FourStateExecutor
) -> Mapping[str, Any]:
    """Run one matched condition through an injected frozen four-state core.

    This adapter intentionally does not inspect prediction quality.  It only
    supplies canonical received-IQ-derived inputs and attaches the common
    query-isolation receipt.  K1 aliasing and K5/K10 FA state reuse are
    enforced by the core and later validated by the prediction artifact layer.
    """

    if not callable(executor):
        raise NextR5FATarget125RuntimeError("four-state executor must be callable")
    result = executor(
        outer_row=condition.outer_row,
        scene=condition.scene,
        reg0=condition.reg0,
        reg1=condition.reg1,
        source_row=condition.source_row,
    )
    if not isinstance(result, Mapping):
        raise NextR5FATarget125RuntimeError("four-state core must return a mapping")
    # The result remains opaque to this data adapter; reject only the fields
    # that would prove a forbidden query-side path was opened here.
    forbidden = {"truth", "query_truth", "query_role", "quota", "global_reassignment"}
    if any(str(key).strip().lower().replace("-", "_") in forbidden for key in result):
        raise NextR5FATarget125RuntimeError("four-state core result exposes forbidden query data")
    return MappingProxyType(
        {
            "schema": RUNTIME_ADAPTER_SCHEMA,
            "outer_id": condition.outer_row.outer_id,
            "receiver": condition.outer_row.receiver,
            "seed": condition.outer_row.seed,
            "k_shot": condition.outer_row.k_shot,
            "new_count": condition.outer_row.new_count,
            "scene": condition.scene,
            "core_result": dict(result),
            "query_isolation_receipt": dict(query_isolation_receipt()),
        }
    )


@dataclass(frozen=True, slots=True)
class _SupportFeatureCacheEntry:
    """One support-only PR160 result bound to immutable received-IQ bytes."""

    cache_key: tuple[str, str, tuple[int, ...], str, str]
    pre_relu: np.ndarray
    zid160: np.ndarray


class D108ZID160Materializer:
    """Small adapter from sealed D108 packages to totalized R0 z_id160.

    It deliberately reuses D108's package verifier, package-pair validator,
    support-prefix selector and singleton query forward policy.  Only the
    final 288-D D92 feature concatenation is replaced with the frozen 160-D
    same-IQ PR160 graph-derived signed-totalized unit z_id materialization
    required by NEXT-R5.  No eager checkpoint reconstruction is permitted.
    """

    def __init__(
        self,
        *,
        source_plan: Mapping[str, Any],
        pr160_extractor_runtime: Mapping[str, Any],
        device: str,
        support_batch_size: int = 64,
    ) -> None:
        if type(support_batch_size) is not int or support_batch_size != 64:
            raise NextR5FATarget125RuntimeError("support_batch_size must be exactly 64")
        if not isinstance(source_plan, Mapping):
            raise NextR5FATarget125RuntimeError("source_plan must be a prepared D108 mapping")
        identity = source_plan.get("identity")
        if not isinstance(identity, Mapping):
            raise NextR5FATarget125RuntimeError("prepared D108 identity is missing")
        source_runtime_sha256 = _sha(
            identity.get("d92_sealed_runtime_sha256"), "D108 source runtime SHA256"
        )
        if source_runtime_sha256 != PR160_SOURCE_RUNTIME_SHA256:
            raise NextR5FATarget125RuntimeError("D108/PR160 source runtime binding drift")
        descriptor = pr160_extractor_runtime
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path",
            "sha256",
            "size_bytes",
            "source_runtime_sha256",
        }:
            raise NextR5FATarget125RuntimeError("PR160 extractor descriptor closure drift")
        if descriptor.get("source_runtime_sha256") != source_runtime_sha256:
            raise NextR5FATarget125RuntimeError("PR160 extractor/source runtime binding drift")
        try:
            from . import stage2_d108_target125_runner as d108
            from .stage2_diag_cosine_exploration import _device

            self._d108 = d108
            self._device = _device(device)
            self._pr160_extractor_model = load_pr160_runtime(
                descriptor["path"],
                expected_sha256=descriptor["sha256"],
                device=self._device,
            )
        except Exception as error:
            raise NextR5FATarget125RuntimeError("PR160 graph runtime load failed") from error
        self._source_plan = source_plan
        self._support_batch_size = support_batch_size
        self._package_cache: dict[
            tuple[tuple[str, str], ...], tuple[Any, dict[str, Any]]
        ] = {}
        self._support_feature_cache: dict[
            tuple[str, str, tuple[int, ...], str, str], _SupportFeatureCacheEntry
        ] = {}
        self._pr160_runtime_binding = _pr160_receipt_binding(
            {
                "schema": PR160_RUNTIME_BINDING_SCHEMA,
                "source_runtime_sha256": source_runtime_sha256,
                "extractor_runtime_sha256": descriptor["sha256"],
                "extractor_runtime_size_bytes": descriptor["size_bytes"],
            }
        )

    def _package(self, reference: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        key = tuple(sorted((str(name), str(value)) for name, value in reference.items()))
        if key not in self._package_cache:
            try:
                payloads, manifest, _audit = self._d108._package_payloads(reference)  # type: ignore[attr-defined]
                self._package_cache[key] = (payloads, manifest)
            except Exception as error:
                raise NextR5FATarget125RuntimeError("sealed D108 package verification failed") from error
        return self._package_cache[key]

    def _require_package_source_runtime(self, manifest: Mapping[str, Any]) -> None:
        try:
            from .stage2_diag_cosine_exploration import _descriptor

            descriptor = _descriptor(manifest, "feature_runtime")
            runtime_sha = str(descriptor.get("sha256", ""))
            if runtime_sha != self._pr160_runtime_binding["source_runtime_sha256"]:
                raise NextR5FATarget125RuntimeError("D108 package/source runtime SHA drift")
        except NextR5FATarget125RuntimeError:
            raise
        except Exception as error:
            raise NextR5FATarget125RuntimeError("D108 package runtime descriptor drift") from error

    def _support_cache_package_binding(
        self, *, source_row: Mapping[str, Any], scene: str
    ) -> str:
        """Reuse the sealed split binding already shared with the core."""

        try:
            return _sha(
                _split_id(self._source_plan, source_row, scene),
                "support cache package binding SHA256",
            )
        except NextR5FATarget125RuntimeError:
            raise
        except Exception as error:
            raise NextR5FATarget125RuntimeError(
                "support cache package binding drift"
            ) from error

    def _support_cache_key(
        self,
        *,
        package_binding_sha256: str,
        physical_id: str,
        iq_row: np.ndarray,
    ) -> tuple[str, str, tuple[int, ...], str, str]:
        """Build a label-free key for one immutable support observation."""

        row = np.asarray(iq_row)
        if row.dtype != np.float32 or row.ndim < 1 or not np.isfinite(row).all():
            raise NextR5FATarget125RuntimeError(
                "support cache received-IQ row must be finite float32"
            )
        contiguous = np.ascontiguousarray(row, dtype=np.float32)
        descriptor_sha256 = _canonical_sha256(
            dict(_pr160_receipt_binding(self._pr160_runtime_binding))
        )
        return (
            _sha(package_binding_sha256, "support cache package binding SHA256"),
            physical_id,
            tuple(int(value) for value in contiguous.shape),
            hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            descriptor_sha256,
        )

    def _forward_pr160_pre_relu(
        self, *, iq: np.ndarray, batch_size: int
    ) -> np.ndarray:
        try:
            from .stage2_diag_cosine_exploration import forward_zid160

            return _finite_feature_rows(
                forward_zid160(
                    self._pr160_extractor_model,
                    iq,
                    device=self._device,
                    batch_size=batch_size,
                ),
                name="PR160 pre-ReLU z_id160",
            )
        except NextR5FATarget125RuntimeError:
            raise
        except Exception as error:
            raise NextR5FATarget125RuntimeError(
                "same-graph PR160 pre-ReLU materialization failed"
            ) from error

    @staticmethod
    def _cached_support_row(
        entry: _SupportFeatureCacheEntry,
        *,
        cache_key: tuple[str, str, tuple[int, ...], str, str],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(entry, _SupportFeatureCacheEntry) or entry.cache_key != cache_key:
            raise NextR5FATarget125RuntimeError("support feature cache entry binding drift")
        pre_relu = _finite_feature_rows(
            np.asarray(entry.pre_relu)[None, :], name="cached support PR160 pre-ReLU"
        )[0]
        zid160 = _readonly_float32(
            np.asarray(entry.zid160)[None, :], name="cached support z_id160"
        )[0]
        return pre_relu, zid160

    def _support_zid160(
        self,
        *,
        iq: np.ndarray,
        physical_ids: Sequence[str],
        support_labels: Sequence[str],
        source_row: Mapping[str, Any],
        scene: str,
        registration_phase: str,
        expected_reg0: Target125RegistrationInput | None,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        """Forward only uncached support rows and preserve the REG0 old prefix."""

        if registration_phase not in ("REG0", "REG1"):
            raise NextR5FATarget125RuntimeError("support cache registration phase drift")
        raw_rows = np.asarray(iq)
        if (
            raw_rows.dtype != np.float32
            or raw_rows.ndim < 2
            or raw_rows.shape[0] < 1
            or not np.isfinite(raw_rows).all()
        ):
            raise NextR5FATarget125RuntimeError(
                "support cache received-IQ matrix must be finite float32"
            )
        support_iq = np.ascontiguousarray(raw_rows, dtype=np.float32)
        ids = _tokens(
            physical_ids,
            name=f"{registration_phase} support physical_ids",
            expected=len(support_iq),
        )
        labels = _tokens(
            support_labels,
            name=f"{registration_phase} support labels",
            expected=len(support_iq),
            unique=False,
        )
        package_binding_sha256 = self._support_cache_package_binding(
            source_row=source_row, scene=scene
        )
        keys = tuple(
            self._support_cache_key(
                package_binding_sha256=package_binding_sha256,
                physical_id=physical_id,
                iq_row=row,
            )
            for physical_id, row in zip(ids, support_iq, strict=True)
        )
        cache = getattr(self, "_support_feature_cache", None)
        if not isinstance(cache, dict):
            raise NextR5FATarget125RuntimeError("support feature cache is unavailable")

        old_count = 0
        if registration_phase == "REG0":
            if expected_reg0 is not None:
                raise NextR5FATarget125RuntimeError(
                    "REG0 support cache cannot receive a REG0 prefix"
                )
        else:
            if not isinstance(expected_reg0, Target125RegistrationInput):
                raise NextR5FATarget125RuntimeError(
                    "REG1 support cache requires the materialized REG0 prefix"
                )
            old_count = len(expected_reg0.support_physical_ids)
            if old_count != matrix.OLD_CLASS_COUNT * int(source_row["active_k"]):
                raise NextR5FATarget125RuntimeError("REG0 support cache K binding drift")
            if len(support_iq) <= old_count:
                raise NextR5FATarget125RuntimeError("REG1 support suffix is missing")
            if ids[:old_count] != tuple(expected_reg0.support_physical_ids):
                raise NextR5FATarget125RuntimeError(
                    "REG1 old support physical-ID prefix drift"
                )
            # Labels never select a cache row.  This is only the sealed
            # registration-prefix consistency check before reuse.
            if labels[:old_count] != tuple(expected_reg0.support_labels):
                raise NextR5FATarget125RuntimeError(
                    "REG1 old support label prefix drift"
                )

        pre_relu = np.empty((len(support_iq), matrix.FEATURE_DIM), dtype=np.float32)
        cached_zid160: dict[int, np.ndarray] = {}
        missing_indices: list[int] = []
        for index, cache_key in enumerate(keys):
            entry = cache.get(cache_key)
            if registration_phase == "REG1" and index < old_count and entry is None:
                raise NextR5FATarget125RuntimeError(
                    "REG1 old support received-IQ/cache binding drift"
                )
            if entry is None:
                missing_indices.append(index)
                continue
            cached_pre_relu, cached_zid = self._cached_support_row(
                entry, cache_key=cache_key
            )
            pre_relu[index] = cached_pre_relu
            cached_zid160[index] = cached_zid

        if missing_indices:
            missing_array = np.asarray(missing_indices, dtype=np.int64)
            forwarded = self._forward_pr160_pre_relu(
                iq=np.ascontiguousarray(support_iq[missing_array], dtype=np.float32),
                batch_size=self._support_batch_size,
            )
            if forwarded.shape != (len(missing_indices), matrix.FEATURE_DIM):
                raise NextR5FATarget125RuntimeError(
                    "support cache PR160 forward shape drift"
                )
            pre_relu[missing_array] = forwarded

        zid160, totalization = _totalize_same_iq_zid160(
            pre_relu,
            np.maximum(pre_relu, np.float32(0.0)),
            ids,
            scope=f"{registration_phase}_support",
            pr160_runtime_binding=self._pr160_runtime_binding,
        )
        for index, cached_zid in cached_zid160.items():
            if not np.array_equal(zid160[index], cached_zid):
                raise NextR5FATarget125RuntimeError(
                    "support cache totalized z_id160 binding drift"
                )
        for index in missing_indices:
            cache_key = keys[index]
            frozen_pre_relu = np.array(pre_relu[index], dtype=np.float32, copy=True)
            frozen_zid160 = np.array(zid160[index], dtype=np.float32, copy=True)
            frozen_pre_relu.setflags(write=False)
            frozen_zid160.setflags(write=False)
            cache[cache_key] = _SupportFeatureCacheEntry(
                cache_key=cache_key,
                pre_relu=frozen_pre_relu,
                zid160=frozen_zid160,
            )
        return zid160, totalization

    def _zid160(
        self, *, iq: np.ndarray, physical_ids: Sequence[str], scope: str, batch_size: int
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        try:
            pre_relu = self._forward_pr160_pre_relu(iq=iq, batch_size=batch_size)
            return _totalize_same_iq_zid160(
                pre_relu,
                np.maximum(pre_relu, np.float32(0.0)),
                physical_ids,
                scope=scope,
                pr160_runtime_binding=self._pr160_runtime_binding,
            )
        except NextR5FATarget125RuntimeError:
            raise
        except Exception as error:
            raise NextR5FATarget125RuntimeError("same-graph PR160 z_id160 materialization failed") from error

    def materialize(
        self,
        *,
        source_row: Mapping[str, Any],
        scene: str,
        registration_phase: str,
        expected_reg0: Target125RegistrationInput | None = None,
    ) -> Target125RegistrationInput:
        if scene not in matrix.SCENES or registration_phase not in ("REG0", "REG1"):
            raise NextR5FATarget125RuntimeError("materialization scene/registration drift")
        phase = "before" if registration_phase == "REG0" else "after"
        packages = source_row.get("packages")
        if not isinstance(packages, Mapping):
            raise NextR5FATarget125RuntimeError("sealed source row lacks package map")
        try:
            support_ref = packages[f"{phase}_enrollment"]
            query_ref = packages[f"{phase}_apply"]
            support_payloads, support_manifest = self._package(support_ref)
            query_payloads, query_manifest = self._package(query_ref)
            from .stage2_diag_cosine_exploration import _validate_matched_packages

            _validate_matched_packages(support_manifest, query_manifest)
            self._require_package_source_runtime(support_manifest)
            self._require_package_source_runtime(query_manifest)
            support_indices, registry = _registered_class_records(
                support_manifest,
                name="support package",
            )
            query_indices, query_registry = _registered_class_records(
                query_manifest,
                name="query package",
            )
            if support_indices != query_indices or registry != query_registry:
                raise NextR5FATarget125RuntimeError(
                    "support/query sealed registered-class bridge drift"
                )
            if (
                registration_phase == "REG0"
                and support_indices != tuple(range(matrix.OLD_CLASS_COUNT))
            ):
                raise NextR5FATarget125RuntimeError("REG0 sealed class-index bridge drift")
            for manifest in (support_manifest, query_manifest):
                if (
                    manifest.get("receiver") != source_row["receiver"]
                    or manifest.get("seed") != source_row["seed"]
                    or manifest.get("k_shot") != source_row["source_pool_k"]
                ):
                    raise NextR5FATarget125RuntimeError("sealed package row binding drift")
            support_iq, labels, support_ids = self._d108._support_rows(  # type: ignore[attr-defined]
                support_payloads[scene], registered_classes=registry, active_k=source_row["active_k"]
            )
            query_iq, query_ids = self._d108._query_rows(query_payloads[scene])  # type: ignore[attr-defined]
            if support_iq.shape[1:] != query_iq.shape[1:]:
                raise NextR5FATarget125RuntimeError("support/query received-IQ shape drift")
            support_zid160, support_totalization = self._support_zid160(
                iq=support_iq,
                physical_ids=support_ids,
                support_labels=labels,
                source_row=source_row,
                scene=scene,
                registration_phase=registration_phase,
                expected_reg0=expected_reg0,
            )
            query_zid160, query_totalization = self._zid160(
                iq=query_iq,
                physical_ids=query_ids,
                scope=f"{registration_phase}_query",
                batch_size=1,
            )
            return Target125RegistrationInput(
                registration_phase=registration_phase,
                registered_classes=registry,
                registered_class_indices=support_indices,
                support_zid160=support_zid160,
                support_labels=labels,
                support_physical_ids=support_ids,
                query_zid160=query_zid160,
                query_physical_ids=query_ids,
                support_totalization_receipt=support_totalization,
                query_totalization_receipt=query_totalization,
            )
        except NextR5FATarget125RuntimeError:
            raise
        except Exception as error:
            raise NextR5FATarget125RuntimeError("sealed D108 package materialization failed") from error

    def materialize_condition(
        self, *, outer_row: matrix.Target125OuterRow, source_row: Mapping[str, Any], scene: str
    ) -> Target125ConditionInput:
        reg0 = self.materialize(source_row=source_row, scene=scene, registration_phase="REG0")
        reg1 = self.materialize(
            source_row=source_row,
            scene=scene,
            registration_phase="REG1",
            expected_reg0=reg0,
        )
        return Target125ConditionInput(
            outer_row=outer_row, scene=scene, source_row=source_row, reg0=reg0, reg1=reg1
        )


def _source_row_key(row: Mapping[str, Any]) -> tuple[str, int, int, int, int]:
    try:
        result = (
            str(row["receiver"]),
            int(row["seed"]),
            int(row["k_shot"]),
            int(row["new_count"]),
            int(row["source_pool_k"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise NextR5FATarget125RuntimeError("D108 source-row identity is incomplete") from error
    return result


def _target_row_key(row: matrix.Target125OuterRow) -> tuple[str, int, int, int, int]:
    return (row.receiver, row.seed, row.k_shot, row.new_count, row.source_pool_k)


def _validate_method_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    from . import stage2_next_r5_fa_target125_core as core

    lock = _read_json(path, expected_sha256=expected_sha256, name="NEXT-R5 method lock")
    bridge = lock.get("class_identity_bridge")
    head = lock.get("head")
    representation = lock.get("representation")
    bridge_fields = {
        "source_class_indices",
        "source_asset_old_class_order_sha256",
        "sealed_package_class_index_to_row_local_handle",
        "row_local_handle_scope",
        "cross_row_handle_reuse",
    }
    representation_fields = {
        "r0",
        "r0_totalization_schema",
        "r0_totalization_revision",
        "r0_totalization_rule",
        "r0_post_relu_binding",
        "r0_zero_policy",
        "r0_source_runtime_sha256",
        "r0_pr160_extractor_runtime_sha256",
        "r0_pr160_extractor_runtime_size_bytes",
        "r1",
        "dimension",
        "r1_relu",
        "r1_second_normalization",
    }
    if (
        lock.get("schema") != METHOD_LOCK_SCHEMA
        or lock.get("candidate_id") != matrix.CANDIDATE_ID
        or lock.get("protocol_schema") != matrix.PROTOCOL_SCHEMA
        or lock.get("matrix", {}).get("logical_surface_count")
        != matrix.LOGICAL_STATE_SURFACE_COUNT
        or lock.get("matrix", {}).get("unique_prediction_count")
        != matrix.UNIQUE_PREDICTION_COUNT
        or lock.get("matrix", {}).get("alias_count") != matrix.ALIAS_COUNT
        or type(representation) is not dict
        or set(representation) != representation_fields
        or representation.get("r0") != core.R0_REPRESENTATION
        or representation.get("r0_totalization_schema") != ZID160_TOTALIZATION_SCHEMA
        or representation.get("r0_totalization_revision") != ZID160_TOTALIZATION_REVISION
        or representation.get("r0_totalization_rule")
        != "normalize_relu_pr160_graph_pre_if_nonzero_else_normalize_same_iq_signed_prerelu"
        or representation.get("r0_post_relu_binding")
        != "exact_relu_from_same_pr160_graph_pre"
        or representation.get("r0_zero_policy")
        != "exact_pr160_graph_relu_zero_and_nonzero_finite_prerelu_required"
        or representation.get("r0_source_runtime_sha256")
        != PR160_SOURCE_RUNTIME_SHA256
        or representation.get("r0_pr160_extractor_runtime_sha256")
        != PR160_EXTRACTOR_RUNTIME_SHA256
        or representation.get("r0_pr160_extractor_runtime_size_bytes")
        != PR160_EXTRACTOR_RUNTIME_SIZE_BYTES
        or representation.get("r1") != core.R1_REPRESENTATION
        or representation.get("dimension") != matrix.FEATURE_DIM
        or representation.get("r1_relu") is not False
        or representation.get("r1_second_normalization") is not False
        or type(bridge) is not dict
        or set(bridge) != bridge_fields
        or _class_indices(
            bridge.get("source_class_indices", ()),
            name="method_lock.source_class_indices",
            expected=matrix.OLD_CLASS_COUNT,
        )
        != tuple(range(matrix.OLD_CLASS_COUNT))
        or bridge.get("sealed_package_class_index_to_row_local_handle") is not True
        or bridge.get("row_local_handle_scope") != "per_package_row"
        or bridge.get("cross_row_handle_reuse") is not False
        or not isinstance(head, Mapping)
        or head.get("tie_policy") != core.QKNN_TIE_POLICY
    ):
        raise NextR5FATarget125RuntimeError("NEXT-R5 method-lock identity/count drift")
    _sha(
        bridge.get("source_asset_old_class_order_sha256"),
        "method-lock source old-class order SHA256",
    )
    return lock


def prepare_next_r5_fa_target125_inputs(
    *,
    d108_plan_manifest_path: str | Path,
    expected_d108_plan_file_sha256: str,
    d108_context_manifest_path: str | Path,
    expected_d108_context_file_sha256: str,
    fa_asset_path: str | Path,
    expected_fa_asset_sha256: str,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
    pr160_extractor_runtime_path: str | Path,
    expected_pr160_extractor_runtime_sha256: str,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Pin existing D108 inputs, the Target FA asset and method lock once.

    No data builder runs here.  The function merely verifies the already
    prepared D108 plan/context, maps all 125 frozen Target125 rows to them,
    and writes a small immutable release handoff.
    """

    try:
        from . import stage2_d108_target125_runner as d108

        source_plan, source_context = d108._prepared_inputs(  # type: ignore[attr-defined]
            plan_manifest_path=Path(d108_plan_manifest_path),
            expected_plan_file_sha256=expected_d108_plan_file_sha256,
            context_manifest_path=Path(d108_context_manifest_path),
            expected_context_file_sha256=expected_d108_context_file_sha256,
        )
    except Exception as error:
        raise NextR5FATarget125RuntimeError("prepared D108 Target125 input binding failed") from error
    asset = Path(fa_asset_path)
    if not asset.is_file() or asset.is_symlink() or _sha256_file(asset) != _sha(
        expected_fa_asset_sha256, "FA asset SHA256"
    ):
        raise NextR5FATarget125RuntimeError("Target FA asset binding failed")
    lock_path = Path(method_lock_path)
    lock = _validate_method_lock(lock_path, expected_method_lock_sha256)
    try:
        from . import stage2_next_r5_fa_target125_core as core

        decoded_asset = core.deserialize_target_fa_asset(asset.read_bytes())
    except Exception as error:
        raise NextR5FATarget125RuntimeError("Target FA asset decode failed during prepare") from error
    if decoded_asset.method_lock_sha256 != _sha(
        expected_method_lock_sha256, "method-lock SHA256"
    ):
        raise NextR5FATarget125RuntimeError("Target FA asset / method-lock binding drift")
    bridge = lock["class_identity_bridge"]
    if (
        decoded_asset.source_class_indices != tuple(bridge["source_class_indices"])
        or decoded_asset.source_old_class_order_sha256
        != bridge["source_asset_old_class_order_sha256"]
    ):
        raise NextR5FATarget125RuntimeError("Target FA asset / method-lock class-identity drift")
    identity = source_plan.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("checkpoint"), Mapping):
        raise NextR5FATarget125RuntimeError("D108 prepared checkpoint identity is missing")
    checkpoint_sha = _sha(identity["checkpoint"].get("sha256"), "sealed checkpoint SHA256")
    source_runtime_sha = _sha(
        identity.get("d92_sealed_runtime_sha256"), "sealed D92 source runtime SHA256"
    )
    representation = lock["representation"]
    if (
        source_runtime_sha != PR160_SOURCE_RUNTIME_SHA256
        or representation["r0_source_runtime_sha256"] != source_runtime_sha
        or representation["r0_pr160_extractor_runtime_sha256"]
        != expected_pr160_extractor_runtime_sha256
    ):
        raise NextR5FATarget125RuntimeError("D108/PR160 method-lock runtime binding drift")
    pr160_extractor = _bind_pr160_extractor_runtime(
        extractor_runtime_path=pr160_extractor_runtime_path,
        expected_extractor_runtime_sha256=expected_pr160_extractor_runtime_sha256,
        source_runtime_sha256=source_runtime_sha,
        expected_size_bytes=representation["r0_pr160_extractor_runtime_size_bytes"],
    )
    if decoded_asset.checkpoint_sha256 != checkpoint_sha:
        raise NextR5FATarget125RuntimeError("Target FA asset / checkpoint binding drift")
    source_rows = source_context.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != matrix.OUTER_JOB_COUNT:
        raise NextR5FATarget125RuntimeError("D108 prepared context does not close 125 outer rows")
    by_key = {_source_row_key(row): (index, row) for index, row in enumerate(source_rows)}
    if len(by_key) != matrix.OUTER_JOB_COUNT:
        raise NextR5FATarget125RuntimeError("D108 prepared context outer-row identity is not unique")
    frozen = matrix.freeze_next_r5_fa_target125_matrix()
    rows: list[dict[str, Any]] = []
    for outer in frozen.outer_rows:
        found = by_key.get(_target_row_key(outer))
        if found is None:
            raise NextR5FATarget125RuntimeError("Target125 outer row lacks a matched D108 sealed package")
        index, source_row = found
        rows.append(
            {
                **outer.as_dict(),
                "source_row_index": index,
                "source_outer_id": source_row["outer_id"],
            }
        )
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable prepare output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise NextR5FATarget125RuntimeError("unsafe prepare output parent")
    destination.mkdir()
    target_identity = {
        "d108_plan_manifest": {
            "path": str(Path(d108_plan_manifest_path)),
            "sha256": _sha(expected_d108_plan_file_sha256, "D108 plan SHA256"),
        },
        "d108_context_manifest": {
            "path": str(Path(d108_context_manifest_path)),
            "sha256": _sha(expected_d108_context_file_sha256, "D108 context SHA256"),
        },
        "checkpoint_sha256": checkpoint_sha,
        "d92_sealed_runtime_sha256": source_runtime_sha,
        "pr160_extractor_runtime": dict(pr160_extractor),
        "fa_asset": {"path": str(asset), "sha256": _sha(expected_fa_asset_sha256, "FA asset SHA256")},
        "method_lock": {
            "path": str(lock_path),
            "sha256": _sha(expected_method_lock_sha256, "method-lock SHA256"),
        },
    }
    plan: dict[str, Any] = {
        "schema": PREPARED_PLAN_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "matrix_protocol": frozen.receipt_payload(),
        "identity": target_identity,
        "rows": rows,
    }
    plan["plan_receipt_sha256"] = _canonical_sha256(plan)
    context: dict[str, Any] = {
        "schema": PREPARED_CONTEXT_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "identity": target_identity,
        "rows": rows,
    }
    context["context_receipt_sha256"] = _canonical_sha256(context)
    plan_path = destination / "target125_plan.json"
    context_path = destination / "target125_context.json"
    plan_file_sha = _write_json_new(plan_path, plan)
    context_file_sha = _write_json_new(context_path, context)
    receipt: dict[str, Any] = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "status": "D108_SEALED_INPUTS_AND_TARGET_FA_ASSET_PINNED",
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "context_receipt_sha256": context["context_receipt_sha256"],
        "plan_file_sha256": plan_file_sha,
        "context_file_sha256": context_file_sha,
        "outer_job_count": matrix.OUTER_JOB_COUNT,
        "scene_row_count": matrix.SCENE_ROW_COUNT,
        "logical_state_surface_count": matrix.LOGICAL_STATE_SURFACE_COUNT,
        "unique_prediction_count": matrix.UNIQUE_PREDICTION_COUNT,
        "alias_count": matrix.ALIAS_COUNT,
        "query_truth_access": False,
        "query_role_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
    }
    receipt["prepare_receipt_sha256"] = _canonical_sha256(receipt)
    receipt_path = destination / "prepare_receipt.json"
    receipt_file_sha = _write_json_new(receipt_path, receipt)
    return MappingProxyType(
        {
            **receipt,
            "plan_manifest": str(plan_path),
            "context_manifest": str(context_path),
            "prepare_receipt": str(receipt_path),
            "prepare_receipt_file_sha256": receipt_file_sha,
        }
    )


def _load_prepared_next_r5_inputs(
    *,
    plan_manifest_path: str | Path,
    expected_plan_file_sha256: str,
    context_manifest_path: str | Path,
    expected_context_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    plan = _read_json(Path(plan_manifest_path), expected_sha256=expected_plan_file_sha256, name="NEXT-R5 plan")
    context = _read_json(Path(context_manifest_path), expected_sha256=expected_context_file_sha256, name="NEXT-R5 context")
    required_plan = {"schema", "candidate_id", "protocol_schema", "matrix_protocol", "identity", "rows", "plan_receipt_sha256"}
    required_context = {"schema", "candidate_id", "protocol_schema", "plan_receipt_sha256", "identity", "rows", "context_receipt_sha256"}
    if set(plan) != required_plan or set(context) != required_context:
        raise NextR5FATarget125RuntimeError("NEXT-R5 prepared plan/context field closure drift")
    plan_payload = dict(plan)
    plan_receipt = plan_payload.pop("plan_receipt_sha256")
    if (
        plan["schema"] != PREPARED_PLAN_SCHEMA
        or plan["candidate_id"] != matrix.CANDIDATE_ID
        or plan["protocol_schema"] != matrix.PROTOCOL_SCHEMA
        or _sha(plan_receipt, "NEXT-R5 plan receipt") != _canonical_sha256(plan_payload)
        or context["schema"] != PREPARED_CONTEXT_SCHEMA
        or context["candidate_id"] != matrix.CANDIDATE_ID
        or context["protocol_schema"] != matrix.PROTOCOL_SCHEMA
        or context["plan_receipt_sha256"] != plan_receipt
        or context["identity"] != plan["identity"]
        or context["rows"] != plan["rows"]
    ):
        raise NextR5FATarget125RuntimeError("NEXT-R5 prepared plan/context identity drift")
    context_payload = dict(context)
    context_receipt = context_payload.pop("context_receipt_sha256")
    if _sha(context_receipt, "NEXT-R5 context receipt") != _canonical_sha256(context_payload):
        raise NextR5FATarget125RuntimeError("NEXT-R5 context receipt drift")
    frozen = matrix.freeze_next_r5_fa_target125_matrix()
    if plan["matrix_protocol"] != frozen.receipt_payload() or len(plan["rows"]) != matrix.OUTER_JOB_COUNT:
        raise NextR5FATarget125RuntimeError("NEXT-R5 frozen Target125 matrix drift")
    identity = plan["identity"]
    if not isinstance(identity, Mapping):
        raise NextR5FATarget125RuntimeError("NEXT-R5 prepared identity missing")
    try:
        from . import stage2_d108_target125_runner as d108

        source_plan, source_context = d108._prepared_inputs(  # type: ignore[attr-defined]
            plan_manifest_path=Path(identity["d108_plan_manifest"]["path"]),
            expected_plan_file_sha256=identity["d108_plan_manifest"]["sha256"],
            context_manifest_path=Path(identity["d108_context_manifest"]["path"]),
            expected_context_file_sha256=identity["d108_context_manifest"]["sha256"],
        )
    except Exception as error:
        raise NextR5FATarget125RuntimeError("prepared D108 source-plan reload failed") from error
    return plan, context, source_plan, source_context


def _capsule_id(source_plan: Mapping[str, Any]) -> str:
    identity = source_plan.get("identity")
    if not isinstance(identity, Mapping):
        raise NextR5FATarget125RuntimeError("D108 identity missing for capsule binding")
    return _canonical_sha256(
        {
            "schema": "cvs.phase2.next_r5.fa_rdce3_qknn.target125.capsule_binding.v1",
            "d92_matrix_manifest_sha256": identity["d92_matrix_manifest"]["sha256"],
            "checkpoint_sha256": identity["checkpoint"]["sha256"],
            "d92_sealed_runtime_sha256": identity["d92_sealed_runtime_sha256"],
        }
    )


def _split_id(source_plan: Mapping[str, Any], source_row: Mapping[str, Any], scene: str) -> str:
    return _canonical_sha256(
        {
            "schema": "cvs.phase2.next_r5.fa_rdce3_qknn.target125.split_binding.v1",
            "d108_plan_receipt_sha256": source_plan["plan_receipt_sha256"],
            "source_outer_id": source_row["outer_id"],
            "receiver": source_row["receiver"],
            "seed": source_row["seed"],
            "k_shot": source_row["k_shot"],
            "new_count": source_row["new_count"],
            "source_pool_k": source_row["source_pool_k"],
            "packages": source_row["packages"],
            "scene": scene,
        }
    )


def build_target125_runtime_bindings(
    *,
    source_plan: Mapping[str, Any],
    condition: Target125ConditionInput,
) -> tuple[Any, Any]:
    """Construct the exact core bindings from D108 sealed-plan identities."""

    try:
        from . import stage2_next_r5_fa_target125_core as core

        checkpoint_sha = _sha(source_plan["identity"]["checkpoint"]["sha256"], "checkpoint SHA256")
        capsule = _capsule_id(source_plan)
        split = _split_id(source_plan, condition.source_row, condition.scene)
        common = {
            "checkpoint_sha256": checkpoint_sha,
            "capsule_id": capsule,
            "split_id": split,
            "outer_id": condition.outer_row.outer_id,
            "receiver": condition.outer_row.receiver,
            "seed": condition.outer_row.seed,
            "k_shot": condition.outer_row.k_shot,
            "new_count": condition.outer_row.new_count,
            "source_pool_k": condition.outer_row.source_pool_k,
            "scene": condition.scene,
        }
        reg0 = core.Target125FARuntimeBinding(
            **common,
            registration_phase="REG0",
            registered_classes=tuple(condition.reg0.registered_classes),
            registered_class_indices=tuple(condition.reg0.registered_class_indices),
            support_physical_ids=tuple(condition.reg0.support_physical_ids),
            query_physical_ids=tuple(condition.reg0.query_physical_ids),
        )
        reg1 = core.Target125FARuntimeBinding(
            **common,
            registration_phase="REG1",
            registered_classes=tuple(condition.reg1.registered_classes),
            registered_class_indices=tuple(condition.reg1.registered_class_indices),
            support_physical_ids=tuple(condition.reg1.support_physical_ids),
            query_physical_ids=tuple(condition.reg1.query_physical_ids),
        )
        return reg0, reg1
    except NextR5FATarget125RuntimeError:
        raise
    except Exception as error:
        raise NextR5FATarget125RuntimeError("Target125 FA core runtime-binding construction failed") from error


def _json_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class FAqKNNCoreExecutor:
    """Concrete bridge to the frozen core; it accepts no truth-side input."""

    def __init__(self, *, source_plan: Mapping[str, Any], fa_asset: Any) -> None:
        self._source_plan = source_plan
        self._fa_asset = fa_asset

    def __call__(
        self,
        *,
        outer_row: matrix.Target125OuterRow,
        scene: str,
        reg0: Target125RegistrationInput,
        reg1: Target125RegistrationInput,
        source_row: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            from . import stage2_next_r5_fa_target125_core as core

            condition = Target125ConditionInput(
                outer_row=outer_row,
                scene=scene,
                source_row=source_row,
                reg0=reg0,
                reg1=reg1,
            )
            reg0_binding, reg1_binding = build_target125_runtime_bindings(
                source_plan=self._source_plan, condition=condition
            )
            old_classes = tuple(reg0.registered_classes)
            new_classes = tuple(reg1.registered_classes[matrix.OLD_CLASS_COUNT :])
            new_mask = np.asarray([label in new_classes for label in reg1.support_labels], dtype=bool)
            if int(new_mask.sum()) != len(new_classes) * outer_row.k_shot:
                raise NextR5FATarget125RuntimeError("REG1 new-support partition drift")
            four_state = core.build_fa_qknn_four_state(
                self._fa_asset,
                reg0_binding=reg0_binding,
                reg1_binding=reg1_binding,
                old_support_features=reg0.support_zid160,
                old_support_labels=reg0.support_labels,
                new_support_features=np.ascontiguousarray(reg1.support_zid160[new_mask], dtype=np.float32),
                new_support_labels=tuple(
                    label for label, selected in zip(reg1.support_labels, new_mask, strict=True) if selected
                ),
                new_support_physical_ids=tuple(
                    physical_id for physical_id, selected in zip(reg1.support_physical_ids, new_mask, strict=True) if selected
                ),
            )
            scores = core.score_fa_qknn_four_state(
                four_state,
                reg0_query_features=reg0.query_zid160,
                reg0_query_physical_ids=reg0.query_physical_ids,
                reg1_query_features=reg1.query_zid160,
                reg1_query_physical_ids=reg1.query_physical_ids,
            )
            qstates = {
                "DA0_REG0": four_state.da0_reg0,
                "DA1_REG0": four_state.da1_reg0,
                "DA0_REG1": four_state.da0_reg1,
                "DA1_REG1": four_state.da1_reg1,
            }
            bindings = {"DA0_REG0": reg0_binding, "DA1_REG0": reg0_binding, "DA0_REG1": reg1_binding, "DA1_REG1": reg1_binding}
            state_payloads: dict[str, Any] = {}
            scene_row_id = matrix.make_scene_row_id(outer_row.outer_id, scene)
            for state in matrix.STATES:
                qstate = qstates[state]
                binding = bindings[state]
                source_state = "DA0_REG0" if state == "DA1_REG0" else "DA0_REG1" if state == "DA1_REG1" else None
                receipt: dict[str, Any] = {
                    "schema": "cvs.phase2.next_r5.fa_rdce3_qknn.target125.state_receipt.v2",
                    "state": state,
                    "representation": qstate.representation,
                    "runtime_binding_sha256": binding.binding_sha256,
                    "qknn_state_sha256": qstate.qknn_state_receipt_sha256,
                    "fit_mode": scores.audit["fit_mode"],
                    "tie_policy": scores.audit["tie_policy"],
                    "support_zid160_totalization_receipt_sha256": (
                        reg0.support_totalization_receipt["totalization_receipt_sha256"]
                        if state.endswith("REG0")
                        else reg1.support_totalization_receipt["totalization_receipt_sha256"]
                    ),
                    "query_zid160_totalization_receipt_sha256": (
                        reg0.query_totalization_receipt["totalization_receipt_sha256"]
                        if state.endswith("REG0")
                        else reg1.query_totalization_receipt["totalization_receipt_sha256"]
                    ),
                    "query_rows_used_for_fit": 0,
                    "query_state_updates": 0,
                    "query_selection_count": 0,
                    "query_truth_access": False,
                    "query_role_access": False,
                    "query_batch_dependency": False,
                    "class_quota_access": False,
                    "global_reassignment_access": False,
                }
                if outer_row.k_shot == 1 and source_state is not None:
                    receipt.update(
                        {
                            "exact_prediction_alias": True,
                            "alias_of_surface_id": matrix.make_surface_id(scene_row_id, source_state),
                        }
                    )
                state_payloads[state] = {
                    "registered_classes": list(qstate.classes),
                    "query_physical_ids": list(binding.query_physical_ids),
                    "predictions": list(scores.predictions_by_state[state]),
                    "state_receipt": receipt,
                }
            return {
                "states": state_payloads,
                "fa_reg1_reuse_receipt": _json_plain(four_state.reg1_reuse_receipt),
                "resource_receipt": {
                    "fa": _json_plain(four_state.fa_state.resource_receipt) if four_state.fa_state is not None else {"dynamic_numeric_bytes": 0, "fit_mode": "FA_STRICT_BYPASS"},
                    "qknn": {state: _json_plain(qstates[state].resource_receipt) for state in matrix.STATES},
                },
                "query_isolation_receipt": dict(query_isolation_receipt()),
            }
        except NextR5FATarget125RuntimeError:
            raise
        except Exception as error:
            raise NextR5FATarget125RuntimeError("FA-RDCE3/qKNN four-state execution failed") from error


def _shard_outer_indices(shard_index: int) -> tuple[int, ...]:
    if type(shard_index) is not int or shard_index not in range(SHARD_COUNT):
        raise NextR5FATarget125RuntimeError("shard_index must be an integer in 0..7")
    return tuple(index for index in range(matrix.OUTER_JOB_COUNT) if index % SHARD_COUNT == shard_index)


def _new_output_directory(path: str | Path, *, name: str) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable {name} output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise NextR5FATarget125RuntimeError(f"unsafe {name} output parent")
    destination.mkdir()
    return destination


def _load_target_asset(plan: Mapping[str, Any]) -> Any:
    identity = plan.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("fa_asset"), Mapping):
        raise NextR5FATarget125RuntimeError("prepared Target125 FA asset identity is missing")
    reference = identity["fa_asset"]
    path = Path(str(reference.get("path", "")))
    expected = _sha(reference.get("sha256"), "prepared FA asset SHA256")
    if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
        raise NextR5FATarget125RuntimeError("prepared Target125 FA asset SHA mismatch")
    try:
        from . import stage2_next_r5_fa_target125_core as core

        asset = core.deserialize_target_fa_asset(path.read_bytes())
        if asset.checkpoint_sha256 != _sha(identity.get("checkpoint_sha256"), "prepared checkpoint SHA256"):
            raise NextR5FATarget125RuntimeError("Target125 FA asset checkpoint binding drift")
        method_lock = identity.get("method_lock")
        if not isinstance(method_lock, Mapping):
            raise NextR5FATarget125RuntimeError("Target125 FA asset method-lock binding drift")
        method_lock_sha = _sha(method_lock.get("sha256"), "prepared method-lock SHA256")
        if asset.method_lock_sha256 != method_lock_sha:
            raise NextR5FATarget125RuntimeError("Target125 FA asset method-lock binding drift")
        lock = _validate_method_lock(Path(str(method_lock.get("path", ""))), method_lock_sha)
        bridge = lock["class_identity_bridge"]
        if (
            asset.source_class_indices != tuple(bridge["source_class_indices"])
            or asset.source_old_class_order_sha256
            != bridge["source_asset_old_class_order_sha256"]
        ):
            raise NextR5FATarget125RuntimeError("Target125 FA asset method-lock binding drift")
        return asset
    except NextR5FATarget125RuntimeError:
        raise
    except Exception as error:
        raise NextR5FATarget125RuntimeError("Target125 FA asset decode failed") from error


def predict_next_r5_fa_target125_shard(
    *,
    plan_manifest_path: str | Path,
    expected_plan_file_sha256: str,
    context_manifest_path: str | Path,
    expected_context_file_sha256: str,
    output_dir: str | Path,
    shard_index: int,
    device: str,
) -> Mapping[str, Any]:
    """Run one immutable modulo-8, truth-free Target125 prediction shard."""

    indices = _shard_outer_indices(shard_index)
    plan, context, source_plan, source_context = _load_prepared_next_r5_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    asset = _load_target_asset(plan)
    pr160_extractor = _prepared_pr160_extractor_runtime(
        plan=plan, source_plan=source_plan
    )
    destination = _new_output_directory(output_dir, name=f"shard {shard_index}")
    frozen = matrix.freeze_next_r5_fa_target125_matrix()
    surface_by_key = {
        (surface.outer_id, surface.scene, surface.state): surface for surface in frozen.surfaces
    }
    materializer = D108ZID160Materializer(
        source_plan=source_plan,
        pr160_extractor_runtime=pr160_extractor,
        device=device,
        support_batch_size=64,
    )
    executor = FAqKNNCoreExecutor(source_plan=source_plan, fa_asset=asset)
    shard_records: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    for index in indices:
        target_row = context["rows"][index]
        outer = frozen.outer_rows[index]
        expected_target = {**outer.as_dict(), "source_row_index": target_row["source_row_index"], "source_outer_id": target_row["source_outer_id"]}
        if target_row != expected_target:
            raise NextR5FATarget125RuntimeError("prepared Target125 row/order binding drift")
        source_index = target_row["source_row_index"]
        if type(source_index) is not int or source_index not in range(matrix.OUTER_JOB_COUNT):
            raise NextR5FATarget125RuntimeError("prepared D108 source-row index drift")
        source_row = source_context["rows"][source_index]
        if source_row.get("outer_id") != target_row["source_outer_id"] or _source_row_key(source_row) != _target_row_key(outer):
            raise NextR5FATarget125RuntimeError("prepared D108 source row identity drift")
        registry: tuple[str, ...] | None = None
        for scene in matrix.SCENES:
            condition = materializer.materialize_condition(
                outer_row=outer, source_row=source_row, scene=scene
            )
            core_result = execute_target125_condition(condition, executor=executor)["core_result"]
            states = core_result.get("states") if isinstance(core_result, Mapping) else None
            if not isinstance(states, Mapping) or set(states) != set(matrix.STATES):
                raise NextR5FATarget125RuntimeError("four-state execution surface closure drift")
            for state in matrix.STATES:
                surface = surface_by_key[(outer.outer_id, scene, state)]
                payload = states[state]
                if not isinstance(payload, Mapping):
                    raise NextR5FATarget125RuntimeError("four-state state payload is malformed")
                classes = _tokens(payload.get("registered_classes"), name=f"{state}.registered_classes")
                if registry is None:
                    registry = classes[: matrix.OLD_CLASS_COUNT]
                if classes[: matrix.OLD_CLASS_COUNT] != registry or len(classes) != matrix.OLD_CLASS_COUNT + (0 if surface.registration_phase == "REG0" else outer.new_count):
                    raise NextR5FATarget125RuntimeError("four-state registered-class binding drift")
                query_ids = _tokens(payload.get("query_physical_ids"), name=f"{state}.query_physical_ids")
                labels = _tokens(payload.get("predictions"), name=f"{state}.predictions", expected=len(query_ids), unique=False)
                receipt = payload.get("state_receipt")
                if not isinstance(receipt, Mapping):
                    raise NextR5FATarget125RuntimeError("four-state state receipt is missing")
                if surface.unique_prediction:
                    from . import stage2_next_r5_fa_target125 as artifacts

                    record = artifacts.seal_unique_prediction(
                        output_dir=destination,
                        surface=surface,
                        registered_classes=classes,
                        query_physical_ids=query_ids,
                        predicted_labels=labels,
                        state_receipt=receipt,
                    )
                else:
                    source_id = surface.alias_of_surface_id
                    source = next((item for item in shard_records if item["surface"]["surface_id"] == source_id), None)
                    if source is None:
                        raise NextR5FATarget125RuntimeError("K1 alias source was not sealed in its own condition")
                    from . import stage2_next_r5_fa_target125 as artifacts

                    record = artifacts.seal_k1_alias(
                        surface=surface, source_record=source, state_receipt=receipt
                    )
                shard_records.append(dict(record))
        if registry is None:
            raise NextR5FATarget125RuntimeError("Target125 outer did not materialize a registry")
        outer_rows.append({**outer.as_dict(), "old_classes": list(registry)})
    shard: dict[str, Any] = {
        "schema": PREDICTION_SHARD_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "truth_open": False,
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "matrix_receipt_sha256": frozen.matrix_receipt_sha256,
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "context_receipt_sha256": context["context_receipt_sha256"],
        "outer_indices": list(indices),
        "outer_rows": outer_rows,
        "surface_count": len(shard_records),
        "access_ledger": dict(matrix.ACCESS_LEDGER),
        "surfaces": shard_records,
    }
    shard["shard_receipt_sha256"] = _canonical_sha256(shard)
    shard_path = destination / "prediction_shard_manifest.json"
    shard_file_sha = _write_json_new(shard_path, shard)
    return MappingProxyType(
        {
            "prediction_shard_manifest": str(shard_path),
            "prediction_shard_manifest_file_sha256": shard_file_sha,
            "shard_receipt_sha256": shard["shard_receipt_sha256"],
            "shard_index": shard_index,
            "outer_job_count": len(indices),
            "surface_count": len(shard_records),
        }
    )


def _load_prediction_shard(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise NextR5FATarget125RuntimeError("prediction shard manifest must be a regular file")
    file_sha = _sha256_file(source)
    try:
        shard = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR5FATarget125RuntimeError("prediction shard manifest must be UTF-8 JSON") from error
    required = {
        "schema", "candidate_id", "protocol_schema", "truth_open", "shard_index", "shard_count",
        "matrix_receipt_sha256", "plan_receipt_sha256", "context_receipt_sha256", "outer_indices",
        "outer_rows", "surface_count", "access_ledger", "surfaces", "shard_receipt_sha256",
    }
    if not isinstance(shard, dict) or set(shard) != required or (
        shard["schema"] != PREDICTION_SHARD_SCHEMA
        or shard["candidate_id"] != matrix.CANDIDATE_ID
        or shard["protocol_schema"] != matrix.PROTOCOL_SCHEMA
        or shard["truth_open"] is not False
        or shard["shard_count"] != SHARD_COUNT
        or shard["access_ledger"] != matrix.ACCESS_LEDGER
    ):
        raise NextR5FATarget125RuntimeError("prediction shard identity/access-ledger drift")
    receipt = shard.pop("shard_receipt_sha256")
    if _sha(receipt, "prediction shard receipt") != _canonical_sha256(shard):
        raise NextR5FATarget125RuntimeError("prediction shard receipt drift")
    shard["shard_receipt_sha256"] = receipt
    indices = _shard_outer_indices(shard["shard_index"])
    if shard["outer_indices"] != list(indices) or not isinstance(shard["surfaces"], list):
        raise NextR5FATarget125RuntimeError("prediction shard coverage drift")
    expected_count = sum(
        3 * sum(surface.outer_id == matrix.freeze_next_r5_fa_target125_matrix().outer_rows[index].outer_id for surface in matrix.freeze_next_r5_fa_target125_matrix().surfaces)
        for index in ()
    )
    del expected_count  # cardinality is checked against frozen surfaces at merge.
    return shard, source.parent.resolve(strict=True), file_sha


def merge_next_r5_fa_target125_shards(
    *, shard_manifest_paths: Sequence[str | Path], output_dir: str | Path
) -> Mapping[str, Any]:
    """Copy exact sealed artifacts from eight shards and close one manifest."""

    if isinstance(shard_manifest_paths, (str, bytes)) or len(shard_manifest_paths) != SHARD_COUNT:
        raise NextR5FATarget125RuntimeError("merge requires exactly eight shard manifests")
    loaded = [_load_prediction_shard(path) for path in shard_manifest_paths]
    by_index: dict[int, tuple[dict[str, Any], Path, str]] = {}
    for shard, root, file_sha in loaded:
        index = shard["shard_index"]
        if index in by_index:
            raise NextR5FATarget125RuntimeError("merge has duplicate shard indices")
        by_index[index] = (shard, root, file_sha)
    if set(by_index) != set(range(SHARD_COUNT)):
        raise NextR5FATarget125RuntimeError("merge shard-index coverage drift")
    baseline = by_index[0][0]
    identity_fields = ("matrix_receipt_sha256", "plan_receipt_sha256", "context_receipt_sha256")
    if any(any(shard[field] != baseline[field] for field in identity_fields) for shard, _, _ in by_index.values()):
        raise NextR5FATarget125RuntimeError("merge shard identity drift")
    frozen = matrix.freeze_next_r5_fa_target125_matrix()
    records: dict[str, tuple[dict[str, Any], Path]] = {}
    for index in range(SHARD_COUNT):
        shard, root, _file_sha = by_index[index]
        for record in shard["surfaces"]:
            if not isinstance(record, Mapping) or not isinstance(record.get("surface"), Mapping):
                raise NextR5FATarget125RuntimeError("shard prediction record malformed")
            surface_id = record["surface"].get("surface_id")
            if type(surface_id) is not str or surface_id in records:
                raise NextR5FATarget125RuntimeError("merge duplicate/invalid prediction surface")
            records[surface_id] = (dict(record), root)
    if set(records) != {surface.surface_id for surface in frozen.surfaces}:
        raise NextR5FATarget125RuntimeError("merge logical surface coverage drift")
    destination = _new_output_directory(output_dir, name="merged prediction")
    prediction_root = destination / "predictions"
    prediction_root.mkdir()
    merged_records: list[dict[str, Any]] = []
    for surface in frozen.surfaces:
        record, source_root = records[surface.surface_id]
        if not surface.unique_prediction:
            merged_records.append(record)
            continue
        relative = record.get("prediction_artifact")
        if relative != f"predictions/{surface.surface_id}.json":
            raise NextR5FATarget125RuntimeError("shard artifact relative path drift")
        source = source_root / relative
        if not source.is_file() or source.is_symlink() or _sha256_file(source) != _sha(record.get("prediction_artifact_sha256"), "shard artifact SHA256"):
            raise NextR5FATarget125RuntimeError("shard prediction artifact binding drift")
        raw = source.read_bytes()
        target = prediction_root / f"{surface.surface_id}.json"
        if target.exists():
            raise NextR5FATarget125RuntimeError("merged prediction artifact collision")
        target.write_bytes(raw)
        merged_records.append(record)
    try:
        from . import stage2_next_r5_fa_target125 as artifacts

        result = artifacts.build_prediction_manifest(output_dir=destination, records=merged_records)
        artifacts.validate_prediction_manifest(
            prediction_manifest_path=result["prediction_manifest"],
            expected_prediction_manifest_file_sha256=result["prediction_manifest_file_sha256"],
        )
    except Exception as error:
        raise NextR5FATarget125RuntimeError("merged Target125 prediction manifest validation failed") from error
    return MappingProxyType(
        {
            **dict(result),
            "shard_count": SHARD_COUNT,
            "shard_manifest_file_sha256": [by_index[index][2] for index in range(SHARD_COUNT)],
            "shard_receipts": [by_index[index][0]["shard_receipt_sha256"] for index in range(SHARD_COUNT)],
        }
    )


__all__ = [
    "D108ZID160Materializer",
    "FourStateExecutor",
    "MATERIALIZED_STATE_SCHEMA",
    "NextR5FATarget125RuntimeError",
    "QUERY_ISOLATION_SCHEMA",
    "RUNTIME_ADAPTER_SCHEMA",
    "Target125ConditionInput",
    "Target125RegistrationInput",
    "PREDICTION_SHARD_SCHEMA",
    "PREPARED_CONTEXT_SCHEMA",
    "PREPARED_PLAN_SCHEMA",
    "SHARD_COUNT",
    "execute_target125_condition",
    "FAqKNNCoreExecutor",
    "build_target125_runtime_bindings",
    "merge_next_r5_fa_target125_shards",
    "predict_next_r5_fa_target125_shard",
    "prepare_next_r5_fa_target125_inputs",
    "query_isolation_receipt",
]
