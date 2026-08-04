"""NEXT-R1 Tail-Safe Lite (TSL) support-only registration head.

This module implements the frozen TSL part of ``STAGE2_RD_GOAL_20260731``.
It is intentionally isolated from FABR, full-D92, the matrix runner, and all
query/truth/role inputs.  A bound :class:`TailSafeLite` owns one immutable
Phase1 prior and exposes the only Phase2 fit surface::

    TailSafeLite(prior, runtime_binding=binding).fit(
        support_z160, support_labels, registered_classes
    )

K1 produces an explicit qKNN alias state.  K5/K10 fit a class-symmetric,
all-class empirical-Bayes diagonal affine head and contract it continuously
towards a spherical reference with the Phase1-sealed ``rho_h``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


Z_DIM = 160
ACTIVE_K_VALUES = (1, 5, 10)
PRIOR_SCHEMA = "cvs.phase2.next_r1.tsl.phase1_prior.v1"
PRIOR_WIRE_SCHEMA = "cvs.phase2.next_r1.tsl.phase1_prior_wire.v1"
AFFINE_SCHEMA = "cvs.phase2.next_r1.tsl.affine.v1"
AFFINE_WIRE_SCHEMA = "cvs.phase2.next_r1.tsl.affine_wire.v1"
ALIAS_SCHEMA = "cvs.phase2.next_r1.tsl.k1_qknn_alias.v1"
FIT_SCHEMA = "cvs.phase2.next_r1.tsl.fit.v1"
RESOURCE_SCHEMA = "cvs.phase2.next_r1.tsl.resource.v1"


class TailSafeLiteError(ValueError):
    """Raised when the frozen TSL contract is not met."""


class TSLTieUnresolvedError(TailSafeLiteError):
    """A final float32 top tie has no legal class decision."""


@dataclass(frozen=True, slots=True)
class TSLRuntimeBinding:
    """Runtime identity shared by the actual checkpoint, representation, and row seal."""

    checkpoint_sha256: str
    representation_rule_sha256: str
    phase1_seal_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "representation_rule_sha256",
            "phase1_seal_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))

    @property
    def binding_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json(
                {
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "phase1_seal_sha256": self.phase1_seal_sha256,
                    "representation_rule_sha256": self.representation_rule_sha256,
                    "schema": "cvs.phase2.next_r1.tsl.runtime_binding.v1",
                }
            )
        )


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise TailSafeLiteError(f"{name} must be a lowercase SHA256 hex digest")
    return text


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TailSafeLiteError("registered_classes must be a sequence")
    result = tuple(str(value) for value in values)
    if len(result) < 2 or any(not value for value in result) or len(set(result)) != len(result):
        raise TailSafeLiteError("registered_classes must contain at least two unique strings")
    return result


def _labels(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TailSafeLiteError(f"{name} must be a sequence")
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise TailSafeLiteError(f"{name} contains an empty label")
    return result


def _raw_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != Z_DIM
        or rows.shape[0] < 1
        or not np.isfinite(rows).all()
    ):
        raise TailSafeLiteError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return np.ascontiguousarray(rows)


def _unit_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = _raw_rows(value, name=name)
    norms = np.sqrt(np.sum(rows.astype(np.float64) ** 2, axis=1))
    if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=2.0e-6, rtol=0.0):
        raise TailSafeLiteError(f"{name} must be canonical L2-normalized z160 rows")
    return _readonly(rows, np.float32)


def normalize_signed_prerelu160(value: np.ndarray) -> np.ndarray:
    """Apply the one frozen signed-totalization rule to pre-ReLU ``p``.

    The positive ReLU view is retained whenever it has positive norm.  An
    all-negative finite pre-ReLU row uses its signed direction; exact-zero and
    non-finite ``p`` rows fail closed.  The accumulation is explicitly float64.
    """

    rows = _raw_rows(value, name="joint_proj.0 pre-ReLU")
    positive = np.maximum(rows, np.float32(0.0))
    positive_norm = np.sqrt(np.sum(positive.astype(np.float64) ** 2, axis=1))
    signed_norm = np.sqrt(np.sum(rows.astype(np.float64) ** 2, axis=1))
    if not np.isfinite(signed_norm).all() or bool(np.any(signed_norm <= 0.0)):
        raise TailSafeLiteError("exact-zero or non-finite pre-ReLU row cannot be totalized")
    result = np.empty_like(rows, dtype=np.float64)
    positive_rows = positive_norm > 0.0
    result[positive_rows] = positive[positive_rows].astype(np.float64) / positive_norm[
        positive_rows, None
    ]
    result[~positive_rows] = rows[~positive_rows].astype(np.float64) / signed_norm[
        ~positive_rows, None
    ]
    return _unit_rows(np.asarray(result, dtype=np.float32), name="signed-totalized z160")


def _scalar_fp16(value: Any, *, name: str, positive: bool = True) -> np.float16:
    array = np.asarray(value)
    if array.shape != ():
        raise TailSafeLiteError(f"{name} must be a scalar")
    try:
        result = np.float16(array.item())
    except (TypeError, ValueError, OverflowError) as error:
        raise TailSafeLiteError(f"{name} cannot be represented as float16") from error
    if not np.isfinite(result) or (positive and not float(result) > 0.0):
        raise TailSafeLiteError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def _fp16_hex(value: np.float16) -> str:
    return np.asarray(value, dtype="<f2").tobytes(order="C").hex()


def _fp16_from_hex(value: Any, *, name: str) -> np.float16:
    if not isinstance(value, str) or len(value) != 4:
        raise TailSafeLiteError(f"{name} must be exactly one little-endian fp16 word")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise TailSafeLiteError(f"{name} is not hexadecimal") from error
    return np.frombuffer(raw, dtype="<f2", count=1)[0]


def _type7_quantile(values: Sequence[float], probability: float) -> float:
    """R type-7 quantile, spelled out to avoid NumPy-version drift."""

    data = np.sort(np.asarray(values, dtype=np.float64))
    if data.ndim != 1 or len(data) < 1 or not np.isfinite(data).all():
        raise TailSafeLiteError("Type-7 quantile requires finite values")
    if not 0.0 <= probability <= 1.0:
        raise TailSafeLiteError("Type-7 probability must be in [0,1]")
    position = (len(data) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(data[lower])
    weight = position - lower
    return float((1.0 - weight) * data[lower] + weight * data[upper])


def _canonical_array_receipt(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(value)
    payload = array.dtype.str.encode("ascii") + b"|" + repr(tuple(array.shape)).encode("ascii") + b"|" + array.tobytes(order="C")
    return _freeze({"dtype": array.dtype.str, "shape": tuple(int(x) for x in array.shape), "sha256": _sha256_bytes(payload)})


@dataclass(frozen=True, slots=True)
class Phase1Cell:
    """One permitted receiver×class Phase1 statistics cell.

    Physical IDs are only used to force a deterministic sample order and form
    the asset binding; they never enter a class score or a tie decision.
    """

    receiver: str
    class_label: str
    physical_ids: tuple[str, ...]
    z160: np.ndarray

    def __post_init__(self) -> None:
        receiver = str(self.receiver)
        class_label = str(self.class_label)
        ids = tuple(str(value) for value in self.physical_ids)
        rows = _unit_rows(self.z160, name="Phase1Cell.z160")
        if not receiver or not class_label or len(ids) != len(rows) or len(ids) < 2:
            raise TailSafeLiteError("Phase1Cell receiver/class/physical-row closure drift")
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise TailSafeLiteError("Phase1Cell physical IDs must be unique nonempty strings")
        order = np.argsort(np.asarray(ids, dtype=object), kind="stable")
        sorted_ids = tuple(ids[int(index)] for index in order)
        sorted_rows = _readonly(rows[order], np.float32)
        object.__setattr__(self, "receiver", receiver)
        object.__setattr__(self, "class_label", class_label)
        object.__setattr__(self, "physical_ids", sorted_ids)
        object.__setattr__(self, "z160", sorted_rows)

    @property
    def degrees_of_freedom(self) -> int:
        return int(len(self.z160) - 1)

    @property
    def physical_id_digest(self) -> str:
        return _sha256_bytes("\n".join(self.physical_ids).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class Phase1PhysicalLOOFold:
    """A preconstructed physical-LOO validation fold for ``rho_h``.

    ``support_physical_ids`` and ``validation_physical_ids`` must be disjoint.
    The caller constructs these folds from Phase1 data before any target access.
    """

    fold_id: str
    support_z160: np.ndarray
    support_labels: tuple[str, ...]
    registered_classes: tuple[str, ...]
    support_physical_ids: tuple[str, ...]
    validation_z160: np.ndarray
    validation_labels: tuple[str, ...]
    validation_physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        fold_id = str(self.fold_id)
        if not fold_id:
            raise TailSafeLiteError("physical-LOO fold_id must be nonempty")
        classes = _registry(self.registered_classes)
        labels = _labels(self.support_labels, name="physical-LOO support labels")
        support = _unit_rows(self.support_z160, name="physical-LOO support")
        support_ids = tuple(str(value) for value in self.support_physical_ids)
        validation = _unit_rows(self.validation_z160, name="physical-LOO validation")
        validation_labels = _labels(self.validation_labels, name="physical-LOO validation labels")
        validation_ids = tuple(str(value) for value in self.validation_physical_ids)
        if (
            len(labels) != len(support)
            or len(support_ids) != len(support)
            or len(validation_labels) != len(validation)
            or len(validation_ids) != len(validation)
            or any(label not in classes for label in labels + validation_labels)
        ):
            raise TailSafeLiteError("physical-LOO labels/IDs must close over their rows/classes")
        if len(set(support_ids)) != len(support_ids) or len(set(validation_ids)) != len(validation_ids):
            raise TailSafeLiteError("physical-LOO physical IDs must be unique within a split")
        if set(support_ids).intersection(validation_ids):
            raise TailSafeLiteError("physical-LOO support and validation IDs must be disjoint")
        order = np.asarray(
            sorted(range(len(support)), key=lambda index: (classes.index(labels[index]), support_ids[index])),
            dtype=np.int64,
        )
        validation_order = np.argsort(np.asarray(validation_ids, dtype=object), kind="stable")
        canonical_support_labels = tuple(labels[int(index)] for index in order)
        canonical_support_ids = tuple(support_ids[int(index)] for index in order)
        canonical_validation_labels = tuple(validation_labels[int(index)] for index in validation_order)
        canonical_validation_ids = tuple(validation_ids[int(index)] for index in validation_order)
        _support_contract(_readonly(support[order], np.float32), canonical_support_labels, classes)
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "support_z160", _readonly(support[order], np.float32))
        object.__setattr__(self, "support_labels", canonical_support_labels)
        object.__setattr__(self, "support_physical_ids", canonical_support_ids)
        object.__setattr__(self, "validation_z160", _readonly(validation[validation_order], np.float32))
        object.__setattr__(self, "validation_labels", canonical_validation_labels)
        object.__setattr__(self, "validation_physical_ids", canonical_validation_ids)


def phase1_cell_physical_id_root(cells: Sequence[Phase1Cell]) -> str:
    """Return the canonical Phase1 cell physical-ID binding for a TSL asset."""

    values = tuple(cells)
    if not values or any(type(cell) is not Phase1Cell for cell in values):
        raise TailSafeLiteError("physical-ID root requires exact nonempty Phase1Cell values")
    payload = {
        "schema": "cvs.phase2.next_r1.tsl.phase1_cell_physical_ids.v1",
        "cells": [
            {
                "receiver": cell.receiver,
                "class_label": cell.class_label,
                "physical_ids": list(cell.physical_ids),
            }
            for cell in sorted(values, key=lambda item: (item.receiver, item.class_label))
        ],
    }
    return _sha256_bytes(_canonical_json(payload))


def _support_contract(
    support_z160: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, int]:
    rows = _unit_rows(support_z160, name="support_z160")
    classes = _registry(registered_classes)
    labels = _labels(support_labels, name="support_labels")
    if len(labels) != len(rows) or any(label not in classes for label in labels):
        raise TailSafeLiteError("support labels must close over support rows and registered classes")
    index = {label: position for position, label in enumerate(classes)}
    class_index = np.asarray([index[label] for label in labels], dtype=np.int64)
    counts = np.bincount(class_index, minlength=len(classes))
    if np.any(counts < 1) or len(set(int(value) for value in counts)) != 1:
        raise TailSafeLiteError("support must be balanced over all registered classes")
    active_k = int(counts[0])
    if active_k not in ACTIVE_K_VALUES or len(rows) != len(classes) * active_k:
        raise TailSafeLiteError("TSL permits balanced K1/K5/K10 support only")
    return rows, classes, class_index, active_k


@dataclass(frozen=True, slots=True)
class TSLPhase1Prior:
    """Immutable INT8/FP16 Phase1-only empirical-Bayes prior."""

    q_logv0: np.ndarray
    scale_logv0: np.float16
    offset_logv0: np.float16
    nu0: np.float16
    rho_h: np.float16
    checkpoint_sha256: str
    cell_physical_id_root_sha256: str
    representation_rule_sha256: str
    schema: str = PRIOR_SCHEMA

    def __post_init__(self) -> None:
        q = np.asarray(self.q_logv0)
        if q.dtype != np.int8 or q.shape != (Z_DIM,) or np.any(q == np.int8(-128)):
            raise TailSafeLiteError("q_logv0 must be int8[160] in [-127,127]")
        scale = _scalar_fp16(self.scale_logv0, name="scale_logv0")
        offset = _scalar_fp16(self.offset_logv0, name="offset_logv0", positive=False)
        nu0 = _scalar_fp16(self.nu0, name="nu0")
        rho_h = _scalar_fp16(self.rho_h, name="rho_h")
        if self.schema != PRIOR_SCHEMA:
            raise TailSafeLiteError("TSL prior schema drift")
        decoded_log = float(offset) + float(scale) * q.astype(np.float64)
        decoded = np.exp(decoded_log)
        if not np.isfinite(decoded).all() or np.any(decoded <= 0.0):
            raise TailSafeLiteError("decoded TSL v0 must be finite and positive")
        object.__setattr__(self, "q_logv0", _readonly(q, np.int8))
        object.__setattr__(self, "scale_logv0", scale)
        object.__setattr__(self, "offset_logv0", offset)
        object.__setattr__(self, "nu0", nu0)
        object.__setattr__(self, "rho_h", rho_h)
        object.__setattr__(self, "checkpoint_sha256", _require_sha256(self.checkpoint_sha256, name="checkpoint_sha256"))
        object.__setattr__(self, "cell_physical_id_root_sha256", _require_sha256(self.cell_physical_id_root_sha256, name="cell_physical_id_root_sha256"))
        object.__setattr__(self, "representation_rule_sha256", _require_sha256(self.representation_rule_sha256, name="representation_rule_sha256"))

    @property
    def decoded_v0(self) -> np.ndarray:
        decoded = np.exp(float(self.offset_logv0) + float(self.scale_logv0) * self.q_logv0.astype(np.float64))
        return _readonly(decoded, np.float64)

    def wire_mapping(self) -> Mapping[str, Any]:
        return _freeze(
            {
                "cell_physical_id_root_sha256": self.cell_physical_id_root_sha256,
                "checkpoint_sha256": self.checkpoint_sha256,
                "nu0_fp16_le_hex": _fp16_hex(self.nu0),
                "offset_logv0_fp16_le_hex": _fp16_hex(self.offset_logv0),
                "q_logv0_int8_b64": base64.b64encode(self.q_logv0.tobytes(order="C")).decode("ascii"),
                "representation_rule_sha256": self.representation_rule_sha256,
                "rho_h_fp16_le_hex": _fp16_hex(self.rho_h),
                "scale_logv0_fp16_le_hex": _fp16_hex(self.scale_logv0),
                "schema": PRIOR_WIRE_SCHEMA,
            }
        )

    @property
    def serialized_bytes(self) -> bytes:
        return _canonical_json(dict(self.wire_mapping()))

    @property
    def prior_sha256(self) -> str:
        return _sha256_bytes(self.serialized_bytes)

    @property
    def validation_receipt(self) -> Mapping[str, Any]:
        v0 = self.decoded_v0
        return _freeze(
            {
                "schema": PRIOR_SCHEMA,
                "prior_sha256": self.prior_sha256,
                "q_logv0": dict(_canonical_array_receipt(self.q_logv0)),
                "scale_logv0_fp16": float(self.scale_logv0),
                "offset_logv0_fp16": float(self.offset_logv0),
                "nu0_fp16": float(self.nu0),
                "rho_h_fp16": float(self.rho_h),
                "decoded_v0_min": float(np.min(v0)),
                "decoded_v0_max": float(np.max(v0)),
                "decoded_v0_all_positive": bool(np.all(v0 > 0.0)),
                "checkpoint_sha256": self.checkpoint_sha256,
                "cell_physical_id_root_sha256": self.cell_physical_id_root_sha256,
                "representation_rule_sha256": self.representation_rule_sha256,
            }
        )


def serialize_phase1_prior(prior: TSLPhase1Prior) -> bytes:
    if type(prior) is not TSLPhase1Prior:
        raise TailSafeLiteError("serialize_phase1_prior requires an exact TSLPhase1Prior")
    return prior.serialized_bytes


def deserialize_phase1_prior(value: bytes) -> TSLPhase1Prior:
    if not isinstance(value, (bytes, bytearray)):
        raise TailSafeLiteError("TSL prior wire must be bytes")
    try:
        document = json.loads(bytes(value).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TailSafeLiteError("TSL prior wire is not canonical UTF-8 JSON") from error
    if not isinstance(document, dict) or document.get("schema") != PRIOR_WIRE_SCHEMA:
        raise TailSafeLiteError("TSL prior wire schema drift")
    expected = {
        "schema", "q_logv0_int8_b64", "scale_logv0_fp16_le_hex", "offset_logv0_fp16_le_hex",
        "nu0_fp16_le_hex", "rho_h_fp16_le_hex", "checkpoint_sha256",
        "cell_physical_id_root_sha256", "representation_rule_sha256",
    }
    if set(document) != expected:
        raise TailSafeLiteError("TSL prior wire fields drift")
    try:
        q_bytes = base64.b64decode(document["q_logv0_int8_b64"].encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as error:
        raise TailSafeLiteError("TSL prior q_logv0 base64 drift") from error
    if len(q_bytes) != Z_DIM:
        raise TailSafeLiteError("TSL prior q_logv0 byte length drift")
    prior = TSLPhase1Prior(
        q_logv0=np.frombuffer(q_bytes, dtype=np.int8, count=Z_DIM).copy(),
        scale_logv0=_fp16_from_hex(document["scale_logv0_fp16_le_hex"], name="scale_logv0"),
        offset_logv0=_fp16_from_hex(document["offset_logv0_fp16_le_hex"], name="offset_logv0"),
        nu0=_fp16_from_hex(document["nu0_fp16_le_hex"], name="nu0"),
        rho_h=_fp16_from_hex(document["rho_h_fp16_le_hex"], name="rho_h"),
        checkpoint_sha256=document["checkpoint_sha256"],
        cell_physical_id_root_sha256=document["cell_physical_id_root_sha256"],
        representation_rule_sha256=document["representation_rule_sha256"],
    )
    if prior.serialized_bytes != bytes(value):
        raise TailSafeLiteError("TSL prior wire is not canonical")
    return prior


@dataclass(frozen=True, slots=True)
class _RawGeometry:
    rows: np.ndarray
    classes: tuple[str, ...]
    indices: np.ndarray
    active_k: int
    means: np.ndarray
    residuals: np.ndarray
    v_post: np.ndarray
    v_sph: float
    w_ref: np.ndarray
    b_ref: np.ndarray
    w_hat: np.ndarray
    b_hat: np.ndarray
    distance: float


def _raw_geometry(
    support_z160: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    prior: TSLPhase1Prior,
) -> _RawGeometry:
    rows, classes, indices, active_k = _support_contract(support_z160, support_labels, registered_classes)
    if active_k == 1:
        raise TailSafeLiteError("raw TSL geometry is not identifiable at K1")
    rows64 = rows.astype(np.float64)
    class_count = len(classes)
    means = np.empty((class_count, Z_DIM), dtype=np.float64)
    residuals = np.empty_like(rows64)
    for class_index in range(class_count):
        member_rows = rows64[indices == class_index]
        if len(member_rows) != active_k:
            raise TailSafeLiteError("TSL class membership/K closure drift")
        mean = np.sum(member_rows, axis=0, dtype=np.float64) / float(active_k)
        means[class_index] = mean
        total = np.sum(member_rows, axis=0, dtype=np.float64)
        leave_one_out = (total[None, :] - member_rows) / float(active_k - 1)
        residuals[indices == class_index] = (float(active_k - 1) / float(active_k)) * (member_rows - leave_one_out)
    v0 = prior.decoded_v0
    v_floor = max(float(np.finfo(np.float32).tiny), 64.0 * float(np.finfo(np.float32).eps) * float(np.mean(v0)))
    denominator = float(prior.nu0) + float(class_count * (active_k - 1))
    v_post = (float(prior.nu0) * v0 + np.sum(residuals * residuals, axis=0, dtype=np.float64)) / denominator
    v_sph = float(np.mean(v_post))
    if (
        not np.isfinite(means).all()
        or not np.isfinite(residuals).all()
        or not np.isfinite(v_post).all()
        or not math.isfinite(v_sph)
        or bool(np.any(v0 < v_floor))
        or bool(np.any(v_post < v_floor))
        or not v_sph >= v_floor
    ):
        raise TailSafeLiteError("TSL variance/numeric floor closure failed")
    w_ref = means / v_sph
    b_ref = -0.5 * np.sum(means * means, axis=1) / v_sph
    w_hat = means / v_post[None, :]
    b_hat = -0.5 * np.sum(means * means / v_post[None, :], axis=1)
    w_ref -= np.mean(w_ref, axis=0, keepdims=True)
    b_ref -= float(np.mean(b_ref))
    w_hat -= np.mean(w_hat, axis=0, keepdims=True)
    b_hat -= float(np.mean(b_hat))
    delta = np.concatenate([(w_hat - w_ref).reshape(-1), b_hat - b_ref])
    distance = float(np.linalg.norm(delta))
    tolerance = 64.0 * float(np.finfo(np.float32).eps) * max(
        1.0,
        float(np.linalg.norm(np.concatenate([w_ref.reshape(-1), b_ref]))),
        float(np.linalg.norm(np.concatenate([w_hat.reshape(-1), b_hat]))),
    )
    if (
        not np.isfinite(w_ref).all()
        or not np.isfinite(b_ref).all()
        or not np.isfinite(w_hat).all()
        or not np.isfinite(b_hat).all()
        or not math.isfinite(distance)
        or distance <= tolerance
    ):
        raise TailSafeLiteError("TSL has no numerically resolved reference-to-EB function")
    return _RawGeometry(
        rows=_readonly(rows, np.float32),
        classes=classes,
        indices=_readonly(indices, np.int64),
        active_k=active_k,
        means=_readonly(means, np.float64),
        residuals=_readonly(residuals, np.float64),
        v_post=_readonly(v_post, np.float64),
        v_sph=v_sph,
        w_ref=_readonly(w_ref, np.float64),
        b_ref=_readonly(b_ref, np.float64),
        w_hat=_readonly(w_hat, np.float64),
        b_hat=_readonly(b_hat, np.float64),
        distance=distance,
    )


def _scores(weights: np.ndarray, intercepts: np.ndarray, query_z160: np.ndarray) -> np.ndarray:
    query = _unit_rows(query_z160, name="query_z160").astype(np.float64)
    logits = query @ weights.T + intercepts[None, :]
    if not np.isfinite(logits).all():
        raise TailSafeLiteError("TSL score became non-finite")
    return np.asarray(logits, dtype=np.float64)


def _rho_cell(fold: Phase1PhysicalLOOFold, *, prior: TSLPhase1Prior) -> tuple[float, Mapping[str, Any]]:
    geometry = _raw_geometry(
        fold.support_z160, fold.support_labels, fold.registered_classes, prior=prior
    )
    reference = _scores(geometry.w_ref, geometry.b_ref, fold.validation_z160)
    empirical = _scores(geometry.w_hat, geometry.b_hat, fold.validation_z160)
    index = {label: position for position, label in enumerate(geometry.classes)}
    ratios: list[float] = []
    accepted_pairs = 0
    for row_index, label in enumerate(fold.validation_labels):
        truth_index = index[label]
        row = reference[row_index]
        maximum = float(np.max(row))
        if int(np.count_nonzero(row == maximum)) != 1 or int(np.argmax(row)) != truth_index:
            continue
        for other in range(len(geometry.classes)):
            if other == truth_index:
                continue
            margin = float(row[truth_index] - row[other])
            if not margin > 0.0:
                raise TailSafeLiteError("reference-correct sample has nonpositive pairwise margin")
            change = float((empirical[row_index, truth_index] - empirical[row_index, other]) - margin)
            accepted_pairs += 1
            if change < 0.0:
                ratios.append(margin / (-change))
    eta_cell = min(1.0, min(ratios, default=1.0))
    rho_cell = eta_cell * geometry.distance
    if not math.isfinite(rho_cell) or not rho_cell > 0.0:
        raise TailSafeLiteError("physical-LOO TSL rho_cell must be finite and positive")
    return rho_cell, _freeze(
        {
            "fold_id": fold.fold_id,
            "active_k": geometry.active_k,
            "D_cell": geometry.distance,
            "reference_correct_pair_count": accepted_pairs,
            "negative_margin_change_pair_count": len(ratios),
            "eta_cell": eta_cell,
            "rho_cell": rho_cell,
            "support_validation_physical_ids_disjoint": True,
        }
    )


@dataclass(frozen=True, slots=True)
class TSLPhase1PriorBuild:
    prior: TSLPhase1Prior
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.prior) is not TSLPhase1Prior:
            raise TailSafeLiteError("TSL prior build requires an exact TSLPhase1Prior")
        object.__setattr__(self, "receipt", _freeze(self.receipt))


def build_phase1_prior(
    cells: Sequence[Phase1Cell],
    validation_folds: Sequence[Phase1PhysicalLOOFold],
    *,
    checkpoint_sha256: str,
    cell_physical_id_root_sha256: str,
    representation_rule_sha256: str,
) -> TSLPhase1PriorBuild:
    """Build the one Phase1-only TSL prior exactly from frozen cell statistics."""

    cell_values = tuple(cells)
    folds = tuple(validation_folds)
    if not cell_values or not folds or any(type(cell) is not Phase1Cell for cell in cell_values) or any(
        type(fold) is not Phase1PhysicalLOOFold for fold in folds
    ):
        raise TailSafeLiteError("TSL prior requires exact nonempty Phase1 cells and physical-LOO folds")
    pairs = tuple((cell.receiver, cell.class_label) for cell in cell_values)
    if len(set(pairs)) != len(pairs):
        raise TailSafeLiteError("Phase1 prior contains duplicate receiver×class cells")
    computed_root = phase1_cell_physical_id_root(cell_values)
    if _require_sha256(cell_physical_id_root_sha256, name="cell_physical_id_root_sha256") != computed_root:
        raise TailSafeLiteError("Phase1 cell physical-ID root does not bind the supplied cells")
    known_physical_ids = {physical_id for cell in cell_values for physical_id in cell.physical_ids}
    if any(
        physical_id not in known_physical_ids
        for fold in folds
        for physical_id in fold.support_physical_ids + fold.validation_physical_ids
    ):
        raise TailSafeLiteError("physical-LOO fold IDs are not bound to the supplied Phase1 cells")
    variances = np.stack(
        [np.var(cell.z160.astype(np.float64), axis=0, ddof=1) for cell in cell_values], axis=0
    )
    positive = variances[variances > 0.0]
    if len(positive) == 0 or not np.isfinite(positive).all():
        raise TailSafeLiteError("Phase1 TSL cells have no positive finite variance")
    v_floor_p1 = max(
        float(np.finfo(np.float32).tiny),
        64.0 * float(np.finfo(np.float32).eps) * float(np.mean(positive, dtype=np.float64)),
    )
    ell = np.mean(np.log(variances + v_floor_p1), axis=0, dtype=np.float64)
    lower = float(np.min(ell))
    upper = float(np.max(ell))
    raw_offset = 0.5 * (upper + lower)
    raw_scale = (upper - lower) / 254.0
    if not math.isfinite(raw_offset) or not math.isfinite(raw_scale) or not raw_scale > 0.0:
        raise TailSafeLiteError("Phase1 TSL log-variance range must be positive and finite")
    offset = _scalar_fp16(raw_offset, name="Phase1 offset_logv0", positive=False)
    scale = _scalar_fp16(raw_scale, name="Phase1 scale_logv0")
    q_float = (ell - raw_offset) / raw_scale
    q = np.clip(np.rint(q_float), -127, 127).astype(np.int8)
    decoded_log = float(offset) + float(scale) * q.astype(np.float64)
    decoded_v0 = np.exp(decoded_log)
    if not np.isfinite(decoded_v0).all() or bool(np.any(decoded_v0 <= 0.0)):
        raise TailSafeLiteError("Phase1 TSL actual fp16 prior decode is invalid")
    degrees = np.asarray([cell.degrees_of_freedom for cell in cell_values], dtype=np.float64)
    raw_nu0 = float(np.exp(np.mean(np.log(degrees))))
    nu0 = _scalar_fp16(raw_nu0, name="Phase1 nu0")
    provisional = TSLPhase1Prior(
        q_logv0=q,
        scale_logv0=scale,
        offset_logv0=offset,
        nu0=nu0,
        rho_h=np.float16(1.0),
        checkpoint_sha256=checkpoint_sha256,
        cell_physical_id_root_sha256=cell_physical_id_root_sha256,
        representation_rule_sha256=representation_rule_sha256,
    )
    rho_cells: list[float] = []
    cell_receipts: list[Mapping[str, Any]] = []
    for fold in folds:
        rho_cell, cell_receipt = _rho_cell(fold, prior=provisional)
        rho_cells.append(rho_cell)
        cell_receipts.append(cell_receipt)
    raw_rho_h = _type7_quantile(rho_cells, 0.05)
    rho_h = _scalar_fp16(raw_rho_h, name="Phase1 rho_h")
    if float(rho_h) > raw_rho_h:
        rho_h = np.nextafter(rho_h, np.float16(0.0), dtype=np.float16)
    rho_h = _scalar_fp16(rho_h, name="Phase1 rho_h after downward rounding")
    prior = TSLPhase1Prior(
        q_logv0=q,
        scale_logv0=scale,
        offset_logv0=offset,
        nu0=nu0,
        rho_h=rho_h,
        checkpoint_sha256=checkpoint_sha256,
        cell_physical_id_root_sha256=cell_physical_id_root_sha256,
        representation_rule_sha256=representation_rule_sha256,
    )
    decoded_error = np.abs(decoded_log - ell)
    v0_error = np.abs(decoded_v0 - np.exp(ell)) / np.maximum(np.exp(ell), float(np.finfo(np.float64).tiny))
    margin_receipts = []
    for entry in cell_receipts:
        payload = dict(entry)
        payload["rho_h_fp16"] = float(prior.rho_h)
        payload["rho_cell_minus_rho_h_fp16"] = float(payload["rho_cell"] - float(prior.rho_h))
        payload["rho_h_not_rounded_up"] = bool(float(prior.rho_h) <= raw_rho_h)
        margin_receipts.append(payload)
    receipt = {
        "schema": PRIOR_SCHEMA,
        "cell_count": len(cell_values),
        "cell_pairs": [[cell.receiver, cell.class_label] for cell in cell_values],
        "cell_variance_unbiased": True,
        "cell_log_variance_equal_weighted": True,
        "cell_physical_ids_sorted_only_for_canonical_order": True,
        "v_floor_p1": v_floor_p1,
        "raw_offset_logv0": raw_offset,
        "raw_scale_logv0": raw_scale,
        "actual_offset_logv0_fp16": float(prior.offset_logv0),
        "actual_scale_logv0_fp16": float(prior.scale_logv0),
        "q_logv0": dict(_canonical_array_receipt(prior.q_logv0)),
        "max_abs_log_decode_error": float(np.max(decoded_error)),
        "max_relative_v0_decode_error": float(np.max(v0_error)),
        "decoded_v0_all_positive": bool(np.all(prior.decoded_v0 > 0.0)),
        "raw_nu0": raw_nu0,
        "actual_nu0_fp16": float(prior.nu0),
        "raw_rho_h_type7_q05": raw_rho_h,
        "actual_rho_h_fp16": float(prior.rho_h),
        "rho_h_not_rounded_up": bool(float(prior.rho_h) <= raw_rho_h),
        "physical_loo_margin_receipts": margin_receipts,
        "checkpoint_sha256": prior.checkpoint_sha256,
        "cell_physical_id_root_sha256": prior.cell_physical_id_root_sha256,
        "representation_rule_sha256": prior.representation_rule_sha256,
        "prior_sha256": prior.prior_sha256,
    }
    return TSLPhase1PriorBuild(prior=prior, receipt=receipt)


@dataclass(frozen=True, slots=True)
class TSLK1AliasState:
    classes: tuple[str, ...]
    prior_sha256: str
    runtime_binding_sha256: str
    active_k: int = 1
    schema: str = ALIAS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ALIAS_SCHEMA or int(self.active_k) != 1:
            raise TailSafeLiteError("TSL K1 alias schema/K drift")
        object.__setattr__(self, "classes", _registry(self.classes))
        object.__setattr__(self, "prior_sha256", _require_sha256(self.prior_sha256, name="prior_sha256"))
        object.__setattr__(self, "runtime_binding_sha256", _require_sha256(self.runtime_binding_sha256, name="runtime_binding_sha256"))

    @property
    def incremental_numeric_state_bytes(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class TSLAffineHeadState:
    classes: tuple[str, ...]
    active_k: int
    weight_qint8: np.ndarray
    scale_fp16: np.ndarray
    intercept_fp16: np.ndarray
    prior_sha256: str
    runtime_binding_sha256: str
    schema: str = AFFINE_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        q = np.asarray(self.weight_qint8)
        scale = np.asarray(self.scale_fp16)
        intercept = np.asarray(self.intercept_fp16)
        if (
            self.schema != AFFINE_SCHEMA
            or int(self.active_k) not in (5, 10)
            or q.dtype != np.int8
            or q.shape != (len(classes), Z_DIM)
            or bool(np.any(q == np.int8(-128)))
            or scale.dtype != np.float16
            or scale.shape != (len(classes),)
            or intercept.dtype != np.float16
            or intercept.shape != (len(classes),)
            or not np.isfinite(scale).all()
            or not np.isfinite(intercept).all()
            or bool(np.any(scale <= 0.0))
        ):
            raise TailSafeLiteError("TSL affine int8/fp16 wire drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "active_k", int(self.active_k))
        object.__setattr__(self, "weight_qint8", _readonly(q, np.int8))
        object.__setattr__(self, "scale_fp16", _readonly(scale, np.float16))
        object.__setattr__(self, "intercept_fp16", _readonly(intercept, np.float16))
        object.__setattr__(self, "prior_sha256", _require_sha256(self.prior_sha256, name="prior_sha256"))
        object.__setattr__(self, "runtime_binding_sha256", _require_sha256(self.runtime_binding_sha256, name="runtime_binding_sha256"))

    @property
    def numeric_state_bytes(self) -> int:
        return int(self.weight_qint8.nbytes + self.scale_fp16.nbytes + self.intercept_fp16.nbytes)

    @property
    def state_sha256(self) -> str:
        return _sha256_bytes(self.serialized_bytes)

    def wire_mapping(self) -> Mapping[str, Any]:
        return _freeze(
            {
                "active_k": self.active_k,
                "classes": list(self.classes),
                "intercept_fp16_b64": base64.b64encode(self.intercept_fp16.astype("<f2").tobytes(order="C")).decode("ascii"),
                "prior_sha256": self.prior_sha256,
                "runtime_binding_sha256": self.runtime_binding_sha256,
                "scale_fp16_b64": base64.b64encode(self.scale_fp16.astype("<f2").tobytes(order="C")).decode("ascii"),
                "schema": AFFINE_WIRE_SCHEMA,
                "weight_qint8_b64": base64.b64encode(self.weight_qint8.tobytes(order="C")).decode("ascii"),
            }
        )

    @property
    def serialized_bytes(self) -> bytes:
        return _canonical_json(dict(self.wire_mapping()))

    def dequantized(self) -> tuple[np.ndarray, np.ndarray]:
        weights = self.weight_qint8.astype(np.float64) * self.scale_fp16.astype(np.float64)[:, None]
        intercepts = self.intercept_fp16.astype(np.float64)
        return _readonly(weights, np.float64), _readonly(intercepts, np.float64)


def serialize_affine_head(state: TSLAffineHeadState) -> bytes:
    if type(state) is not TSLAffineHeadState:
        raise TailSafeLiteError("serialize_affine_head requires an exact TSLAffineHeadState")
    return state.serialized_bytes


def deserialize_affine_head(value: bytes) -> TSLAffineHeadState:
    if not isinstance(value, (bytes, bytearray)):
        raise TailSafeLiteError("TSL affine wire must be bytes")
    try:
        document = json.loads(bytes(value).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TailSafeLiteError("TSL affine wire is not canonical UTF-8 JSON") from error
    expected = {"schema", "classes", "active_k", "weight_qint8_b64", "scale_fp16_b64", "intercept_fp16_b64", "prior_sha256", "runtime_binding_sha256"}
    if not isinstance(document, dict) or document.get("schema") != AFFINE_WIRE_SCHEMA or set(document) != expected:
        raise TailSafeLiteError("TSL affine wire schema/fields drift")
    classes = _registry(document["classes"])
    try:
        q_bytes = base64.b64decode(str(document["weight_qint8_b64"]).encode("ascii"), validate=True)
        scale_bytes = base64.b64decode(str(document["scale_fp16_b64"]).encode("ascii"), validate=True)
        intercept_bytes = base64.b64decode(str(document["intercept_fp16_b64"]).encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as error:
        raise TailSafeLiteError("TSL affine wire base64 drift") from error
    if (
        len(q_bytes) != len(classes) * Z_DIM
        or len(scale_bytes) != len(classes) * 2
        or len(intercept_bytes) != len(classes) * 2
    ):
        raise TailSafeLiteError("TSL affine wire array byte length drift")
    state = TSLAffineHeadState(
        classes=classes,
        active_k=int(document["active_k"]),
        weight_qint8=np.frombuffer(q_bytes, dtype=np.int8).reshape(len(classes), Z_DIM).copy(),
        scale_fp16=np.frombuffer(scale_bytes, dtype="<f2").astype(np.float16, copy=True),
        intercept_fp16=np.frombuffer(intercept_bytes, dtype="<f2").astype(np.float16, copy=True),
        prior_sha256=str(document["prior_sha256"]),
        runtime_binding_sha256=str(document["runtime_binding_sha256"]),
    )
    if state.serialized_bytes != bytes(value):
        raise TailSafeLiteError("TSL affine wire is not canonical")
    return state


def _compile_affine(
    *,
    classes: tuple[str, ...],
    active_k: int,
    weights: np.ndarray,
    intercepts: np.ndarray,
    prior_sha256: str,
    runtime_binding_sha256: str,
) -> tuple[TSLAffineHeadState, Mapping[str, Any], float]:
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(intercepts, dtype=np.float64)
    if w.shape != (len(classes), Z_DIM) or b.shape != (len(classes),) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise TailSafeLiteError("TSL affine compile input drift")
    peak = float(max(np.max(np.abs(w), initial=0.0), np.max(np.abs(b), initial=0.0)))
    if not peak > 0.0:
        raise TailSafeLiteError("TSL affine compile has no function")
    fp16_safe_peak = 60000.0
    exponent = 0 if peak <= fp16_safe_peak else int(math.floor(math.log2(fp16_safe_peak / peak)))
    common_scale = float(math.ldexp(1.0, exponent))
    scaled_w = w * common_scale
    scaled_b = b * common_scale
    if np.max(np.abs(scaled_b), initial=0.0) > float(np.finfo(np.float16).max):
        raise TailSafeLiteError("TSL common power-of-two scaling did not fit fp16 intercepts")
    codes = np.empty_like(scaled_w, dtype=np.int8)
    scales = np.empty((len(classes),), dtype=np.float16)
    for index in range(len(classes)):
        row_peak = float(np.max(np.abs(scaled_w[index]), initial=0.0))
        if row_peak == 0.0:
            scales[index] = np.float16(1.0)
            codes[index] = np.int8(0)
            continue
        scale = np.float16(row_peak / 127.0)
        while float(scale) * 127.0 < row_peak:
            successor = np.nextafter(scale, np.float16(np.inf), dtype=np.float16)
            if not np.isfinite(successor) or successor == scale:
                raise TailSafeLiteError("TSL int8 row scale cannot cover its finite peak")
            scale = successor
        if not float(scale) >= float(np.finfo(np.float16).tiny):
            raise TailSafeLiteError("TSL nonzero row scale underflows fp16 normal range")
        rounded = np.rint(scaled_w[index] / float(scale))
        if np.max(np.abs(rounded), initial=0.0) > 127.0:
            raise TailSafeLiteError("TSL int8 rounding exceeds [-127,127]")
        scales[index] = scale
        codes[index] = rounded.astype(np.int8)
    intercept16 = scaled_b.astype(np.float16)
    if not np.isfinite(intercept16).all():
        raise TailSafeLiteError("TSL intercept fp16 cast became non-finite")
    state = TSLAffineHeadState(
        classes=classes,
        active_k=active_k,
        weight_qint8=codes,
        scale_fp16=scales,
        intercept_fp16=intercept16,
        prior_sha256=prior_sha256,
        runtime_binding_sha256=runtime_binding_sha256,
    )
    audit = _freeze(
        {
            "policy": "all_class_common_positive_power_of_two_before_quantization",
            "common_logit_scale": common_scale,
            "common_logit_scale_exponent_base2": exponent,
            "pre_scale_peak": peak,
            "post_scale_intercept_peak": float(np.max(np.abs(scaled_b), initial=0.0)),
            "int8_row_scale_min": float(np.min(scales)),
            "int8_row_scale_max": float(np.max(scales)),
            "zero_weight_row_count": int(np.count_nonzero(np.max(np.abs(scaled_w), axis=1) == 0.0)),
            "nonzero_intercept_cast_zero_count": int(np.count_nonzero((scaled_b != 0.0) & (intercept16 == 0.0))),
            "nonzero_intercept_subnormal_count": int(
                np.count_nonzero((intercept16 != 0.0) & (np.abs(intercept16.astype(np.float64)) < float(np.finfo(np.float16).tiny)))
            ),
            "state_sha256": state.state_sha256,
        }
    )
    return state, audit, common_scale


@dataclass(frozen=True, slots=True)
class TSLFit:
    state: TSLAffineHeadState | TSLK1AliasState
    fit_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.state) not in (TSLAffineHeadState, TSLK1AliasState):
            raise TailSafeLiteError("TSL fit state must be exact affine or K1 alias")
        object.__setattr__(self, "fit_receipt", _freeze(self.fit_receipt))
        object.__setattr__(self, "resource_receipt", _freeze(self.resource_receipt))


def _resource_receipt(
    *,
    active_k: int,
    class_count: int,
    state: TSLAffineHeadState | TSLK1AliasState,
    fit_wall_clock_ns: int = 0,
) -> Mapping[str, Any]:
    if active_k == 1:
        return _freeze(
            {
                "schema": RESOURCE_SCHEMA,
                "active_k": 1,
                "feature_dim": Z_DIM,
                "class_count": class_count,
                "fit_mode": "exact_qknn_alias",
                "incremental_deployed_numeric_state_bytes": 0,
                "incremental_query_head_macs_per_sample": 0,
                "underlying_qknn_resource_required": True,
                "fabr_forward_cost_included": False,
                "head_fit_wall_clock_ns": 0,
                "head_fit_wall_clock_statistic": "not_measured_for_exact_qknn_alias",
                "query_rows_used_for_fit": 0,
                "query_state_updates": 0,
                "query_selection_count": 0,
            }
        )
    assert type(state) is TSLAffineHeadState
    support_rows = int(active_k * class_count)
    analytic_macs = int(4 * support_rows * Z_DIM + 8 * Z_DIM + 2 * class_count * Z_DIM)
    workspace = int(support_rows * Z_DIM * 8 + class_count * Z_DIM * 8 + 7 * Z_DIM * 8 + class_count * Z_DIM * 8 + class_count * 8)
    return _freeze(
        {
            "schema": RESOURCE_SCHEMA,
            "active_k": active_k,
            "feature_dim": Z_DIM,
            "class_count": class_count,
            "fit_mode": "all_class_empirical_bayes_diagonal_common_trust_region",
            "shared_affine_wire": "int8_W[C,160]+fp16_scale[C]+fp16_intercept[C]",
            "deployed_numeric_state_bytes": state.numeric_state_bytes,
            "deployed_numeric_state_formula": "160C+2C+2C=164C_B",
            "query_head_macs_per_sample": Z_DIM * class_count,
            "query_state_bytes": 0,
            "head_fit_analytic_mac_equivalent": analytic_macs,
            "head_fit_analytic_mac_formula": "4*N*160+8*160+2*C*160",
            "head_fit_wall_clock_ns": int(fit_wall_clock_ns),
            "head_fit_wall_clock_statistic": "raw_single_call_not_threshold_evidence",
            "estimated_peak_explicit_numeric_workspace_bytes": workspace,
            "explicit_dense_matrix_elements_constructed": 0,
            "explicit_spectral_factorization_count": 0,
            "explicit_linear_system_solve_count": 0,
            "fabr_forward_cost_included": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        }
    )


class TailSafeLite:
    """Bind a sealed Phase1 prior and expose the minimal role-free TSL fit API."""

    __slots__ = ("_prior", "_runtime_binding")

    def __init__(self, prior: TSLPhase1Prior, *, runtime_binding: TSLRuntimeBinding) -> None:
        if type(prior) is not TSLPhase1Prior or type(runtime_binding) is not TSLRuntimeBinding:
            raise TailSafeLiteError("TailSafeLite requires an exact sealed TSLPhase1Prior")
        if (
            runtime_binding.checkpoint_sha256 != prior.checkpoint_sha256
            or runtime_binding.representation_rule_sha256 != prior.representation_rule_sha256
        ):
            raise TailSafeLiteError("TSL runtime checkpoint/representation binding mismatch")
        self._prior = prior
        self._runtime_binding = runtime_binding

    @property
    def prior(self) -> TSLPhase1Prior:
        return self._prior

    @property
    def runtime_binding(self) -> TSLRuntimeBinding:
        return self._runtime_binding

    def fit(
        self,
        support_z160: np.ndarray,
        support_labels: Sequence[str],
        registered_classes: Sequence[str],
    ) -> TSLFit:
        """Fit from current support only; no role, F-arm, query, or truth exists here."""

        rows, classes, _indices, active_k = _support_contract(
            support_z160, support_labels, registered_classes
        )
        if active_k == 1:
            state = TSLK1AliasState(
                classes=classes,
                prior_sha256=self._prior.prior_sha256,
                runtime_binding_sha256=self._runtime_binding.binding_sha256,
            )
            return TSLFit(
                state=state,
                fit_receipt={
                    "schema": FIT_SCHEMA,
                    "fit_mode": "exact_same_representation_qknn_logit_alias",
                    "active_k": 1,
                    "class_count": len(classes),
                    "support_rows": len(rows),
                    "prior_sha256": self._prior.prior_sha256,
                    "runtime_binding_sha256": self._runtime_binding.binding_sha256,
                    "phase1_seal_sha256": self._runtime_binding.phase1_seal_sha256,
                    "query_rows_used_for_fit": 0,
                    "query_state_updates": 0,
                    "query_selection_count": 0,
                    "query_role_access": False,
                    "old_new_role_access": False,
                    "full_head_access": False,
                    "support_compactness_policy": "not_identifiable_at_k1_exact_qknn_alias",
                },
                resource_receipt=_resource_receipt(active_k=1, class_count=len(classes), state=state),
            )
        started = time.perf_counter_ns()
        geometry = _raw_geometry(rows, support_labels, classes, prior=self._prior)
        eta = min(1.0, float(self._prior.rho_h) / geometry.distance)
        if not math.isfinite(eta) or not 0.0 < eta <= 1.0:
            raise TailSafeLiteError("TSL common trust-region eta drift")
        weights = geometry.w_ref + eta * (geometry.w_hat - geometry.w_ref)
        intercepts = geometry.b_ref + eta * (geometry.b_hat - geometry.b_ref)
        state, quantization_audit, common_scale = _compile_affine(
            classes=classes,
            active_k=active_k,
            weights=weights,
            intercepts=intercepts,
            prior_sha256=self._prior.prior_sha256,
            runtime_binding_sha256=self._runtime_binding.binding_sha256,
        )
        dequant_w, dequant_b = state.dequantized()
        scaled_ref_w = geometry.w_ref * common_scale
        scaled_ref_b = geometry.b_ref * common_scale
        deployed_delta = float(
            np.linalg.norm(
                np.concatenate([(dequant_w - scaled_ref_w).reshape(-1), dequant_b - scaled_ref_b])
            )
        )
        deployed_tolerance = 64.0 * float(np.finfo(np.float32).eps) * max(
            1.0,
            float(np.linalg.norm(np.concatenate([scaled_ref_w.reshape(-1), scaled_ref_b]))),
            float(np.linalg.norm(np.concatenate([dequant_w.reshape(-1), dequant_b]))),
        )
        if not math.isfinite(deployed_delta) or deployed_delta <= deployed_tolerance:
            raise TailSafeLiteError("TSL quantization leaves only numerical-noise function change")
        elapsed = time.perf_counter_ns() - started
        receipt = {
            "schema": FIT_SCHEMA,
            "fit_mode": "all_class_empirical_bayes_diagonal_common_frobenius_trust_region",
            "active_k": active_k,
            "class_count": len(classes),
            "support_rows": len(rows),
            "prior_sha256": self._prior.prior_sha256,
            "runtime_binding_sha256": self._runtime_binding.binding_sha256,
            "phase1_seal_sha256": self._runtime_binding.phase1_seal_sha256,
            "v_post_min": float(np.min(geometry.v_post)),
            "v_post_max": float(np.max(geometry.v_post)),
            "v_sph": geometry.v_sph,
            "D_prequantized": geometry.distance,
            "rho_h_fp16": float(self._prior.rho_h),
            "eta": eta,
            "all_class_affine_centered": True,
            "support_compactness_policy": "physical_LOO_e_ck=((K-1)/K)*(u_ck-mean_{j_not_k}u_cj)",
            "old_new_role_access": False,
            "full_head_access": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_role_access": False,
            "quantization": dict(quantization_audit),
            "deployed_delta_from_scaled_reference": deployed_delta,
            "deployed_delta_tolerance": deployed_tolerance,
            "state_sha256": state.state_sha256,
        }
        return TSLFit(
            state=state,
            fit_receipt=receipt,
            resource_receipt=_resource_receipt(
                active_k=active_k,
                class_count=len(classes),
                state=state,
                fit_wall_clock_ns=elapsed,
            ),
        )


def alias_qknn_logits(
    state: TSLK1AliasState,
    qknn_logits: np.ndarray,
    *,
    runtime_binding: TSLRuntimeBinding,
) -> np.ndarray:
    """Validate and return the exact caller-owned K1 qKNN logit object."""

    if type(state) is not TSLK1AliasState or type(runtime_binding) is not TSLRuntimeBinding:
        raise TailSafeLiteError("qKNN alias requires exact TSLK1AliasState")
    if state.runtime_binding_sha256 != runtime_binding.binding_sha256:
        raise TailSafeLiteError("qKNN alias runtime binding mismatch")
    logits = np.asarray(qknn_logits)
    if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[1] != len(state.classes) or not np.isfinite(logits).all():
        raise TailSafeLiteError("K1 qKNN alias logits must be finite float32 [N,C]")
    require_unique_float32_top(logits)
    return qknn_logits


@dataclass(frozen=True, slots=True)
class TSLScore:
    logits: np.ndarray
    score_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        logits = np.asarray(self.logits)
        if logits.dtype != np.float32 or logits.ndim != 2 or not np.isfinite(logits).all():
            raise TailSafeLiteError("TSL score logits must be finite float32 [N,C]")
        object.__setattr__(self, "logits", _readonly(logits, np.float32))
        object.__setattr__(self, "score_receipt", _freeze(self.score_receipt))


def score_affine(
    state: TSLAffineHeadState,
    query_z160: np.ndarray,
    *,
    runtime_binding: TSLRuntimeBinding,
) -> TSLScore:
    """Score independently; the state and API contain no query-side update path."""

    if type(state) is not TSLAffineHeadState or type(runtime_binding) is not TSLRuntimeBinding:
        raise TailSafeLiteError("TSL affine scoring requires an exact TSLAffineHeadState")
    if state.runtime_binding_sha256 != runtime_binding.binding_sha256:
        raise TailSafeLiteError("TSL affine score runtime binding mismatch")
    weights, intercepts = state.dequantized()
    logits = np.asarray(_scores(weights, intercepts, query_z160), dtype=np.float32)
    if logits.shape[1] != len(state.classes):
        raise TailSafeLiteError("TSL affine score class-column drift")
    require_unique_float32_top(logits)
    return TSLScore(
        logits=logits,
        score_receipt={
            "schema": "cvs.phase2.next_r1.tsl.score.v1",
            "state_sha256": state.state_sha256,
            "runtime_binding_sha256": runtime_binding.binding_sha256,
            "phase1_seal_sha256": runtime_binding.phase1_seal_sha256,
            "query_rows": len(logits),
            "all_registered_classes_scored": True,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_role_access": False,
            "query_batch_dependency": False,
        },
    )


def require_unique_float32_top(logits: np.ndarray) -> None:
    """Fail closed on an exact final top tie without consulting any tie key."""

    values = np.asarray(logits)
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise TailSafeLiteError("top-tie check requires finite float32 [N,C>=2] logits")
    maximum = np.max(values, axis=1, keepdims=True)
    tied_rows = np.flatnonzero(np.sum(values == maximum, axis=1) > 1)
    if len(tied_rows):
        raise TSLTieUnresolvedError(f"TIE_UNRESOLVED at {len(tied_rows)} independent query rows")


__all__ = [
    "ACTIVE_K_VALUES",
    "AFFINE_WIRE_SCHEMA",
    "AFFINE_SCHEMA",
    "ALIAS_SCHEMA",
    "FIT_SCHEMA",
    "PRIOR_SCHEMA",
    "Phase1Cell",
    "Phase1PhysicalLOOFold",
    "RESOURCE_SCHEMA",
    "TSLAffineHeadState",
    "TSLFit",
    "TSLK1AliasState",
    "TSLPhase1Prior",
    "TSLPhase1PriorBuild",
    "TSLRuntimeBinding",
    "TSLScore",
    "TSLTieUnresolvedError",
    "TailSafeLite",
    "TailSafeLiteError",
    "Z_DIM",
    "alias_qknn_logits",
    "build_phase1_prior",
    "deserialize_affine_head",
    "deserialize_phase1_prior",
    "normalize_signed_prerelu160",
    "phase1_cell_physical_id_root",
    "require_unique_float32_top",
    "score_affine",
    "serialize_affine_head",
    "serialize_phase1_prior",
]
