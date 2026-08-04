"""Truth-separated fixed-IQ/model bridge for NEXT-R2 proxy24."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_next_r1_real as next_r1_real
from . import stage2_next_r2_matrix as matrix
from . import stage2_next_r2_runtime as runtime


SCHEMA = "cvs.stage2.next_r2.real_bridge.v2"
CAPSULE_SCHEMA = "cvs.stage2.next_r2.prediction_capsule.v1"
PREDICTION_ROWS_SCHEMA = "cvs.stage2.next_r2.prediction_rows.v1"
VIEW_RULE_ID = "fixed_received_iq_global_complex_rotation_0_plus45_minus45_v1"
VIEW_RULE_SHA256 = hashlib.sha256(VIEW_RULE_ID.encode("utf-8")).hexdigest()
SMOKE_SCHEMA = "cvs.stage2.next_r2.real_smoke.v1"

_CAPSULE_FIELDS = frozenset(
    (
        "schema",
        "capsule_id",
        "split_id",
        "selected_iq_archive_sha256",
        "selected_iq_receipt_sha256",
        "label_join_archive_sha256",
        "physical_id_root_sha256",
        "matrix_sha256",
        "plan",
        "keys",
        "truth_opened_for_capsule_build",
        "query_labels_persisted",
        "capsule_content_sha256",
    )
)
_CAPSULE_KEY_FIELDS = frozenset(
    ("outer_key_id", "held_receiver", "held_class", "active_k", "registrations")
)
_REGISTRATION_NAMES = frozenset(("REG0", "REG1"))
_REGISTRATION_FIELDS = frozenset(
    (
        "registered_classes",
        "support_indices",
        "support_labels",
        "support_physical_ids",
        "query_indices",
        "query_physical_ids",
    )
)


class NextR2RealError(ValueError):
    """The pinned IQ, capsule, model tap, or view binding drifted."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise NextR2RealError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR2RealError(f"{name} must be a lowercase SHA256") from error
    return value


def _registry(values: Sequence[str], *, expected: int, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise NextR2RealError(f"{name} must be an ordered registry")
    result = tuple(values)
    if (
        len(result) != expected
        or len(set(result)) != expected
        or any(not isinstance(item, str) or not item for item in result)
    ):
        raise NextR2RealError(f"{name} registry drift")
    return result


def phase_rotate_received_iq(value: object, quarter_sign: int) -> np.ndarray:
    """Return canonical or +/-pi/4 view of finite float32 [N,2,T] IQ."""

    array = np.asarray(value)
    if (
        array.dtype != np.dtype("<f4")
        or array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] != 2
        or array.shape[2] < 1
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        raise NextR2RealError("received IQ must be finite C-contiguous float32 [N,2,T]")
    if type(quarter_sign) is not int or quarter_sign not in (-1, 0, 1):
        raise NextR2RealError("quarter_sign must be exactly -1, 0, or +1")
    if quarter_sign == 0:
        return np.ascontiguousarray(array, dtype=np.float32)
    cosine = np.float32(math.sqrt(0.5))
    sine = np.float32(quarter_sign) * cosine
    result = np.empty_like(array, dtype=np.float32)
    result[:, 0, :] = cosine * array[:, 0, :] - sine * array[:, 1, :]
    result[:, 1, :] = sine * array[:, 0, :] + cosine * array[:, 1, :]
    if not np.isfinite(result).all():
        raise NextR2RealError("phase-rotated IQ became non-finite")
    return np.ascontiguousarray(result, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class NextR2PredictionRows:
    """Selected received IQ with no class or query-truth field."""

    received_iq: np.ndarray
    receiver_ids: tuple[str, ...]
    physical_ids: tuple[str, ...]
    receiver_registry: tuple[str, ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        iq = np.asarray(self.received_iq)
        if (
            iq.dtype != np.dtype("<f4")
            or iq.ndim != 3
            or iq.shape[0] != next_r1_real.ROW_COUNT
            or iq.shape[1] != 2
            or iq.shape[2] < 1
            or not iq.flags.c_contiguous
            or not np.isfinite(iq).all()
        ):
            raise NextR2RealError("prediction IQ must be finite float32 [588,2,T]")
        receivers = tuple(self.receiver_ids)
        physical = tuple(self.physical_ids)
        registry = _registry(
            self.receiver_registry,
            expected=matrix.SOURCE_RECEIVER_COUNT,
            name="receiver",
        )
        if (
            len(receivers) != next_r1_real.ROW_COUNT
            or any(item not in registry for item in receivers)
            or len(physical) != next_r1_real.ROW_COUNT
            or len(set(physical)) != next_r1_real.ROW_COUNT
        ):
            raise NextR2RealError("prediction row identity drift")
        frozen = np.array(iq, dtype=np.float32, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "received_iq", frozen)
        object.__setattr__(self, "receiver_ids", receivers)
        object.__setattr__(self, "physical_ids", physical)
        object.__setattr__(self, "receiver_registry", registry)
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))


def load_next_r2_prediction_rows(
    *,
    selected_iq_archive: str | Path,
    selected_iq_archive_sha256: str,
    selected_iq_receipt: str | Path,
    selected_iq_receipt_sha256: str,
) -> NextR2PredictionRows:
    """Load pinned received IQ without opening the label-join archive."""

    iq_bytes = next_r1_real._read_pinned(  # noqa: SLF001 - exact pinned reader
        selected_iq_archive, selected_iq_archive_sha256, "selected_iq_archive"
    )
    receipt_bytes = next_r1_real._read_pinned(  # noqa: SLF001
        selected_iq_receipt, selected_iq_receipt_sha256, "selected_iq_receipt"
    )
    try:
        selected_receipt = json.loads(receipt_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR2RealError("selected_iq_receipt must be UTF-8 JSON") from error
    required = {
        "schema": next_r1_real.SELECTED_RECEIPT_SCHEMA,
        "archive_sha256": selected_iq_archive_sha256,
        "row_count": next_r1_real.ROW_COUNT,
        "contains_only_selected_ls_rows": True,
        "source_pool_labels_persisted": False,
        "clean_iq_access": False,
        "target_access": False,
        "formal_query_access": False,
    }
    if not isinstance(selected_receipt, dict) or any(
        selected_receipt.get(key) != expected for key, expected in required.items()
    ):
        raise NextR2RealError("selected IQ receipt legality or binding drift")
    archive = next_r1_real._load_npz(  # noqa: SLF001
        iq_bytes,
        members=next_r1_real.SELECTED_MEMBERS,
        name="selected_iq_archive",
    )
    iq = np.asarray(archive["received_iq"])
    receivers = next_r1_real._strings(  # noqa: SLF001
        archive["receiver_ids"], name="selected.receiver_ids", rows=next_r1_real.ROW_COUNT
    )
    physical = next_r1_real._strings(  # noqa: SLF001
        archive["physical_ids"], name="selected.physical_ids", rows=next_r1_real.ROW_COUNT
    )
    physical_root = _sha("\n".join(physical).encode("utf-8"))
    return NextR2PredictionRows(
        received_iq=np.ascontiguousarray(iq, dtype=np.float32),
        receiver_ids=receivers,
        physical_ids=physical,
        receiver_registry=tuple(sorted(set(receivers))),
        receipt={
            "schema": PREDICTION_ROWS_SCHEMA,
            "selected_iq_archive_sha256": selected_iq_archive_sha256,
            "selected_iq_receipt_sha256": selected_iq_receipt_sha256,
            "physical_id_root_sha256": physical_root,
            "label_join_opened": False,
            "query_labels_present": False,
        },
    )


def load_next_r2_real_rows(**kwargs: Any) -> next_r1_real.NextR1RealRows:
    """Truth-aware builder-only reuse of the exact NEXT-R1 row join."""

    return next_r1_real.load_next_r1_real_rows(**kwargs)


def ordered_cell_indices(
    rows: next_r1_real.NextR1RealRows,
) -> Mapping[tuple[str, str], tuple[int, ...]]:
    """Builder-only truth-aware cell partition; never called by predict."""

    if not isinstance(rows, next_r1_real.NextR1RealRows):
        raise NextR2RealError("cell partition requires exact truth-aware builder rows")
    result: dict[tuple[str, str], tuple[int, ...]] = {}
    for receiver in rows.receiver_registry:
        for class_id in rows.class_registry:
            values = tuple(
                index
                for index, (observed_receiver, observed_class) in enumerate(
                    zip(rows.receiver_ids, rows.tx_labels, strict=True)
                )
                if observed_receiver == receiver and observed_class == class_id
            )
            if len(values) != matrix.PHYSICAL_PER_CELL:
                raise NextR2RealError("every source receiver/class cell must contain 14 rows")
            result[(receiver, class_id)] = tuple(
                sorted(
                    values,
                    key=lambda index: _sha(
                        f"{matrix.CELL_ORDER_SALT}|{receiver}|{class_id}|"
                        f"{rows.physical_ids[index]}".encode("utf-8")
                    ),
                )
            )
    return MappingProxyType(result)


def build_next_r2_prediction_capsule(
    rows: next_r1_real.NextR1RealRows,
    *,
    capsule_id: str,
    split_id: str,
    selected_iq_archive_sha256: str,
    selected_iq_receipt_sha256: str,
    label_join_archive_sha256: str,
) -> Mapping[str, Any]:
    """Builder-only: freeze indices/support labels, never persist query labels."""

    if not isinstance(rows, next_r1_real.NextR1RealRows):
        raise NextR2RealError("capsule builder requires exact truth-aware rows")
    if not capsule_id or not split_id:
        raise NextR2RealError("capsule_id and split_id must be nonempty")
    selected_sha = _require_sha(selected_iq_archive_sha256, name="selected_iq_archive_sha256")
    selected_receipt_sha = _require_sha(
        selected_iq_receipt_sha256, name="selected_iq_receipt_sha256"
    )
    label_sha = _require_sha(label_join_archive_sha256, name="label_join_archive_sha256")
    physical_root = _require_sha(
        rows.receipt.get("physical_id_root_sha256"), name="physical_id_root_sha256"
    )
    plan = matrix.build_next_r2_proxy24_plan(
        rows.receiver_registry,
        rows.class_registry,
        source_identity_sha256=physical_root,
    )
    cells = ordered_cell_indices(rows)
    keys: list[dict[str, Any]] = []
    for key_mapping in plan["keys"]:
        outer_key = matrix.outer_key_from_mapping(key_mapping)
        support_by_class = {
            class_id: cells[(outer_key.held_receiver, class_id)][: outer_key.active_k]
            for class_id in outer_key.all_registered_classes
        }
        query_by_class = {
            class_id: cells[(outer_key.held_receiver, class_id)][matrix.MAX_SUPPORT_K :]
            for class_id in outer_key.all_registered_classes
        }
        registrations: dict[str, Any] = {}
        for registration, classes in (
            ("REG0", outer_key.retained_classes),
            ("REG1", outer_key.all_registered_classes),
        ):
            support_indices = tuple(
                index for class_id in classes for index in support_by_class[class_id]
            )
            query_indices = tuple(
                index for class_id in classes for index in query_by_class[class_id]
            )
            registrations[registration] = {
                "registered_classes": classes,
                "support_indices": support_indices,
                "support_labels": tuple(
                    class_id for class_id in classes for _ in support_by_class[class_id]
                ),
                "support_physical_ids": tuple(rows.physical_ids[index] for index in support_indices),
                "query_indices": query_indices,
                "query_physical_ids": tuple(rows.physical_ids[index] for index in query_indices),
            }
        keys.append(
            {
                "outer_key_id": outer_key.outer_key_id,
                "held_receiver": outer_key.held_receiver,
                "held_class": outer_key.held_class,
                "active_k": outer_key.active_k,
                "registrations": registrations,
            }
        )
    payload: dict[str, Any] = {
        "schema": CAPSULE_SCHEMA,
        "capsule_id": capsule_id,
        "split_id": split_id,
        "selected_iq_archive_sha256": selected_sha,
        "selected_iq_receipt_sha256": selected_receipt_sha,
        "label_join_archive_sha256": label_sha,
        "physical_id_root_sha256": physical_root,
        "matrix_sha256": plan["matrix_sha256"],
        "plan": plan,
        "keys": tuple(keys),
        "truth_opened_for_capsule_build": True,
        "query_labels_persisted": False,
    }
    payload["capsule_content_sha256"] = matrix.canonical_sha256(payload)
    return MappingProxyType(payload)


def capsule_bytes(value: Mapping[str, Any]) -> bytes:
    validated = validate_next_r2_prediction_capsule(value)
    return matrix.canonical_bytes(validated)


def _validate_registration(
    value: Mapping[str, Any],
    *,
    classes: tuple[str, ...],
    active_k: int,
    held_receiver: str,
    rows: NextR2PredictionRows | None,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _REGISTRATION_FIELDS:
        raise NextR2RealError("capsule registration fields drift")
    if tuple(value.get("registered_classes", ())) != classes:
        raise NextR2RealError("capsule registered class drift")
    support_indices = tuple(value.get("support_indices", ()))
    query_indices = tuple(value.get("query_indices", ()))
    support_labels = tuple(value.get("support_labels", ()))
    support_ids = tuple(value.get("support_physical_ids", ()))
    query_ids = tuple(value.get("query_physical_ids", ()))
    if (
        len(support_indices) != len(classes) * active_k
        or len(query_indices) != len(classes) * matrix.QUERY_PER_CLASS
        or len(support_labels) != len(support_indices)
        or len(support_ids) != len(support_indices)
        or len(query_ids) != len(query_indices)
        or len(set(support_indices)) != len(support_indices)
        or len(set(query_indices)) != len(query_indices)
        or len(set(support_ids)) != len(support_ids)
        or len(set(query_ids)) != len(query_ids)
        or set(support_indices) & set(query_indices)
        or any(not isinstance(item, str) or not item for item in support_labels + support_ids + query_ids)
        or set(support_labels) != set(classes)
        or any(support_labels.count(item) != active_k for item in classes)
        or any(type(item) is not int or item < 0 or item >= next_r1_real.ROW_COUNT for item in support_indices + query_indices)
    ):
        raise NextR2RealError("capsule support/query registration drift")
    if rows is not None and (
        tuple(rows.physical_ids[index] for index in support_indices) != support_ids
        or tuple(rows.physical_ids[index] for index in query_indices) != query_ids
        or any(rows.receiver_ids[index] != held_receiver for index in support_indices + query_indices)
    ):
        raise NextR2RealError("capsule indices do not bind prediction physical IDs/receiver")


def validate_next_r2_prediction_capsule(
    value: Mapping[str, Any],
    *,
    rows: NextR2PredictionRows | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NextR2RealError("prediction capsule must be a mapping")
    if set(value) != _CAPSULE_FIELDS:
        raise NextR2RealError("prediction capsule top-level fields drift")
    payload = dict(value)
    observed_content_sha = payload.pop("capsule_content_sha256", None)
    if observed_content_sha != matrix.canonical_sha256(payload):
        raise NextR2RealError("prediction capsule content SHA drift")
    if (
        payload.get("schema") != CAPSULE_SCHEMA
        or not isinstance(payload.get("capsule_id"), str)
        or not payload.get("capsule_id")
        or not isinstance(payload.get("split_id"), str)
        or not payload.get("split_id")
        or payload.get("truth_opened_for_capsule_build") is not True
        or payload.get("query_labels_persisted") is not False
    ):
        raise NextR2RealError("prediction capsule legality fields drift")
    for name in (
        "selected_iq_archive_sha256",
        "selected_iq_receipt_sha256",
        "label_join_archive_sha256",
        "physical_id_root_sha256",
        "matrix_sha256",
    ):
        _require_sha(payload.get(name), name=name)
    plan = matrix.validate_next_r2_proxy24_plan(payload.get("plan", {}))
    if payload["matrix_sha256"] != plan["matrix_sha256"]:
        raise NextR2RealError("capsule matrix binding drift")
    keys = tuple(payload.get("keys", ()))
    if len(keys) != matrix.OUTER_KEY_COUNT:
        raise NextR2RealError("capsule key count drift")
    for key_value, plan_value in zip(keys, plan["keys"], strict=True):
        if not isinstance(key_value, Mapping) or set(key_value) != _CAPSULE_KEY_FIELDS:
            raise NextR2RealError("capsule key fields drift")
        outer_key = matrix.outer_key_from_mapping(plan_value)
        registrations = key_value.get("registrations")
        if (
            key_value.get("outer_key_id") != outer_key.outer_key_id
            or key_value.get("held_receiver") != outer_key.held_receiver
            or key_value.get("held_class") != outer_key.held_class
            or key_value.get("active_k") != outer_key.active_k
            or not isinstance(registrations, Mapping)
            or set(registrations) != _REGISTRATION_NAMES
        ):
            raise NextR2RealError("capsule key/plan drift")
        _validate_registration(
            registrations["REG0"],
            classes=outer_key.retained_classes,
            active_k=outer_key.active_k,
            held_receiver=outer_key.held_receiver,
            rows=rows,
        )
        _validate_registration(
            registrations["REG1"],
            classes=outer_key.all_registered_classes,
            active_k=outer_key.active_k,
            held_receiver=outer_key.held_receiver,
            rows=rows,
        )
        if (
            set(registrations["REG0"]["support_indices"])
            - set(registrations["REG1"]["support_indices"])
            or set(registrations["REG0"]["query_indices"])
            - set(registrations["REG1"]["query_indices"])
        ):
            raise NextR2RealError("capsule REG0 is not a REG1 subset")
    by_pair: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    for key in keys:
        by_pair.setdefault((key["held_receiver"], key["held_class"]), {})[
            int(key["active_k"])
        ] = key
    if len(by_pair) != matrix.SELECTED_RECEIVER_COUNT * matrix.CLASS_COUNT:
        raise NextR2RealError("capsule K-pair coverage drift")
    for pair in by_pair.values():
        if set(pair) != {1, 5}:
            raise NextR2RealError("capsule K1/K5 pair drift")
        for registration in ("REG0", "REG1"):
            k1 = pair[1]["registrations"][registration]
            k5 = pair[5]["registrations"][registration]
            if (
                set(k1["support_indices"]) - set(k5["support_indices"])
                or tuple(k1["query_indices"]) != tuple(k5["query_indices"])
            ):
                raise NextR2RealError("capsule K1/K5 nesting drift")
    if rows is not None and (
        payload["selected_iq_archive_sha256"]
        != rows.receipt.get("selected_iq_archive_sha256")
        or payload["selected_iq_receipt_sha256"]
        != rows.receipt.get("selected_iq_receipt_sha256")
        or payload["physical_id_root_sha256"]
        != rows.receipt.get("physical_id_root_sha256")
        or tuple(plan["receiver_registry"]) != rows.receiver_registry
    ):
        raise NextR2RealError("capsule/prediction IQ binding drift")
    ready = dict(payload)
    ready["capsule_content_sha256"] = observed_content_sha
    return MappingProxyType(ready)


def load_next_r2_prediction_capsule(
    path: str | Path,
    *,
    capsule_sha256: str,
    rows: NextR2PredictionRows,
) -> Mapping[str, Any]:
    value = next_r1_real._read_pinned(path, capsule_sha256, "prediction_capsule")  # noqa: SLF001
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR2RealError("prediction capsule must be UTF-8 JSON") from error
    if not isinstance(decoded, dict) or matrix.canonical_bytes(decoded) != value:
        raise NextR2RealError("prediction capsule must use canonical JSON bytes")
    return validate_next_r2_prediction_capsule(decoded, rows=rows)


class NextR2RealModelBridge:
    """Raw-pre-ReLU wrapper around the exact NEXT-R1 model forward."""

    def __init__(
        self,
        base: next_r1_real.NextR1RealModelBridge,
        rows: NextR2PredictionRows,
    ) -> None:
        if not isinstance(base, next_r1_real.NextR1RealModelBridge):
            raise NextR2RealError("NEXT-R2 bridge requires the exact NEXT-R1 bridge")
        if not isinstance(rows, NextR2PredictionRows) or base.rows is not rows:
            raise NextR2RealError("NEXT-R2 bridge requires redacted prediction rows")
        self._base = base
        self.rows = rows
        self.checkpoint_sha256 = base.checkpoint_sha256

    def _tensor(self, value: np.ndarray) -> Any:
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise NextR2RealError("PyTorch is required for the real bridge") from error
        array = np.ascontiguousarray(value, dtype=np.float32)
        tensor = torch.frombuffer(
            memoryview(array), dtype=torch.float32, count=int(array.size)
        ).reshape(tuple(int(item) for item in array.shape)).clone()
        return tensor.to(self._base.device)

    def forward_indices(self, indices: Sequence[int], *, quarter_sign: int = 0) -> np.ndarray:
        ordered = tuple(int(item) for item in indices)
        if (
            len(ordered) < 1
            or len(set(ordered)) != len(ordered)
            or any(item < 0 or item >= next_r1_real.ROW_COUNT for item in ordered)
        ):
            raise NextR2RealError("forward indices must be unique pinned row indices")
        iq = np.ascontiguousarray(self.rows.received_iq[np.asarray(ordered)], dtype=np.float32)
        viewed = phase_rotate_received_iq(iq, quarter_sign)
        _logits, pre = self._base._forward(  # noqa: SLF001 - exact frozen model path
            self._tensor(viewed), grad=False, parameter_overrides=None
        )
        result = next_r1_real._torch_float32_numpy_copy(  # noqa: SLF001
            pre, name="NEXT-R2 joint_proj.0 pre-ReLU"
        )
        if result.shape != (len(ordered), 160):
            raise NextR2RealError("NEXT-R2 raw pre-ReLU tap shape drift")
        return np.ascontiguousarray(result, dtype=np.float32)


def load_next_r2_real_model(
    rows: NextR2PredictionRows,
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    device: Any,
) -> tuple[NextR2RealModelBridge, Mapping[str, Any]]:
    """Build the exact D105 model while keeping label-free prediction rows."""

    if not isinstance(rows, NextR2PredictionRows):
        raise NextR2RealError("model loader requires redacted prediction rows")
    checkpoint, load_receipt = next_r1_real.load_d105_exact_sha_bound_checkpoint(
        checkpoint_path, checkpoint_sha256
    )
    model, build_receipt = next_r1_real.build_d105_exact_model_from_checkpoint(
        checkpoint, input_len=int(rows.received_iq.shape[2]), device=device
    )
    base = next_r1_real.NextR1RealModelBridge(model, rows, checkpoint_sha256, device)
    wrapped = NextR2RealModelBridge(base, rows)
    return wrapped, MappingProxyType(
        {
            "schema": SCHEMA,
            "view_rule_id": VIEW_RULE_ID,
            "view_rule_sha256": VIEW_RULE_SHA256,
            "prediction_rows_query_labels_present": False,
            "next_r1_exact_model_load_receipt": dict(load_receipt),
            "next_r1_exact_model_build_receipt": dict(build_receipt),
        }
    )


def _capsule_key(
    capsule: Mapping[str, Any], outer_key_id: str
) -> Mapping[str, Any]:
    matched = tuple(item for item in capsule["keys"] if item["outer_key_id"] == outer_key_id)
    if len(matched) != 1:
        raise NextR2RealError("capsule outer-key lookup drift")
    return matched[0]


def _take_rows(
    source: np.ndarray,
    source_indices: tuple[int, ...],
    selected_indices: tuple[int, ...],
) -> np.ndarray:
    positions = {index: position for position, index in enumerate(source_indices)}
    if len(positions) != len(source_indices) or any(item not in positions for item in selected_indices):
        raise NextR2RealError("capsule subset index drift")
    return np.ascontiguousarray(
        source[np.asarray([positions[item] for item in selected_indices])], dtype=np.float32
    )


def build_next_r2_four_state_inputs(
    rows: NextR2PredictionRows,
    bridge: NextR2RealModelBridge,
    outer_key: matrix.NextR2OuterKey,
    *,
    capsule: Mapping[str, Any],
) -> Mapping[str, runtime.NextR2StateInputs]:
    """Materialise one key using only the frozen label-free prediction capsule."""

    if not isinstance(rows, NextR2PredictionRows):
        raise NextR2RealError("four-state builder requires redacted prediction rows")
    if not isinstance(bridge, NextR2RealModelBridge) or bridge.rows is not rows:
        raise NextR2RealError("four-state builder bridge/rows drift")
    frozen = validate_next_r2_prediction_capsule(capsule, rows=rows)
    key = _capsule_key(frozen, outer_key.outer_key_id)
    if (
        key["held_receiver"] != outer_key.held_receiver
        or key["held_class"] != outer_key.held_class
        or key["active_k"] != outer_key.active_k
    ):
        raise NextR2RealError("capsule key identity drift")
    reg1 = key["registrations"]["REG1"]
    support_indices = tuple(reg1["support_indices"])
    query_indices = tuple(reg1["query_indices"])
    canonical_support = bridge.forward_indices(support_indices, quarter_sign=0)
    plus_support = bridge.forward_indices(support_indices, quarter_sign=1)
    minus_support = bridge.forward_indices(support_indices, quarter_sign=-1)
    canonical_query = bridge.forward_indices(query_indices, quarter_sign=0)
    result: dict[str, runtime.NextR2StateInputs] = {}
    for state_id in matrix.STATE_IDS:
        registration = "REG1" if state_id in matrix.REG1_STATES else "REG0"
        bound = key["registrations"][registration]
        selected_support = tuple(bound["support_indices"])
        selected_query = tuple(bound["query_indices"])
        result[state_id] = runtime.NextR2StateInputs(
            outer_key_id=outer_key.outer_key_id,
            state_id=state_id,
            capsule_id=str(frozen["capsule_id"]),
            split_id=str(frozen["split_id"]),
            active_k=outer_key.active_k,
            registered_classes=tuple(bound["registered_classes"]),
            support_canonical=_take_rows(canonical_support, support_indices, selected_support),
            support_phase_plus=_take_rows(plus_support, support_indices, selected_support),
            support_phase_minus=_take_rows(minus_support, support_indices, selected_support),
            support_labels=tuple(bound["support_labels"]),
            support_physical_ids=tuple(bound["support_physical_ids"]),
            query_canonical=_take_rows(canonical_query, query_indices, selected_query),
            query_physical_ids=tuple(bound["query_physical_ids"]),
        )
    return runtime.validate_four_state_inputs(outer_key, result)


def verified_next_r2_real_smoke(
    bridge: NextR2RealModelBridge, indices: Sequence[int]
) -> Mapping[str, Any]:
    ordered = tuple(int(item) for item in indices)
    canonical = bridge.forward_indices(ordered, quarter_sign=0)
    repeated = bridge.forward_indices(ordered, quarter_sign=0)
    plus = bridge.forward_indices(ordered, quarter_sign=1)
    minus = bridge.forward_indices(ordered, quarter_sign=-1)
    if not np.array_equal(canonical, repeated):
        raise NextR2RealError("real canonical forward is not exactly repeatable")
    payload = {
        "schema": SMOKE_SCHEMA,
        "checkpoint_sha256": bridge.checkpoint_sha256,
        "view_rule_sha256": VIEW_RULE_SHA256,
        "physical_row_count": len(ordered),
        "canonical_sha256": _sha(canonical.tobytes(order="C")),
        "plus_sha256": _sha(plus.tobytes(order="C")),
        "minus_sha256": _sha(minus.tobytes(order="C")),
        "canonical_repeat_exact": True,
        "finite_pre_relu160": True,
        "query_truth_access": False,
    }
    payload["smoke_receipt_sha256"] = matrix.canonical_sha256(payload)
    return MappingProxyType(payload)


__all__ = [
    "CAPSULE_SCHEMA",
    "NextR2PredictionRows",
    "NextR2RealError",
    "NextR2RealModelBridge",
    "SCHEMA",
    "SMOKE_SCHEMA",
    "VIEW_RULE_ID",
    "VIEW_RULE_SHA256",
    "build_next_r2_four_state_inputs",
    "build_next_r2_prediction_capsule",
    "capsule_bytes",
    "load_next_r2_prediction_capsule",
    "load_next_r2_prediction_rows",
    "load_next_r2_real_model",
    "load_next_r2_real_rows",
    "ordered_cell_indices",
    "phase_rotate_received_iq",
    "validate_next_r2_prediction_capsule",
    "verified_next_r2_real_smoke",
]
