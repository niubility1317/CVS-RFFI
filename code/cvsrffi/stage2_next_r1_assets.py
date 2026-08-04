"""Phase1-only FABR/TSL asset construction for the frozen NEXT-R1 route.

This module is deliberately a construction-time boundary.  It consumes only
precomputed Phase1 gradients, Phase1 labels, Phase1 representation cells and
sealed Phase1 fold metadata.  Its output contains the deployable FABR and TSL
assets plus compact hash receipts; it does not retain receiver/class strings,
physical IDs, gradient rows, or Phase1 representations.

The real-checkpoint extractor remains external to this module.  It must supply
the row-aligned gradients and implement the narrowly typed validation callback
against the frozen signed-pre-ReLU160 functional path.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import stage2_next_r1_fabr as fabr
from . import stage2_next_r1_tsl as tsl


BUNDLE_SCHEMA = "cvs.phase1.next_r1.asset_bundle.v1"
SELECTION_SCHEMA = "cvs.phase1.next_r1.asset_selection.v1"
PHYSICAL_ROOT_SCHEMA = "cvs.phase1.next_r1.fit_physical_root.v1"
REGISTRY_ROOT_SCHEMA = "cvs.phase1.next_r1.registry_root.v1"

FROZEN_RECEIVER_COUNT = 7
FROZEN_CLASS_COUNT = 6
FROZEN_PHYSICAL_PER_CELL = 14
MIN_SUBSPACE_PRINCIPAL_COSINE = 0.90
EIGENVALUE_TIE_RTOL = 64.0 * float(np.finfo(np.float64).eps)
_F16 = np.dtype("<f2")
_F32 = np.dtype("<f4")


class NextR1AssetError(ValueError):
    """A frozen Phase1 asset construction condition did not close."""


class NextR1AssetSelectionError(NextR1AssetError):
    """No method-locked block passed the Phase1-only selection gates."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise NextR1AssetError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR1AssetError(f"{name} must be a lowercase SHA256") from error
    return value


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise NextR1AssetError("receipt contains an unsupported or non-finite value")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise NextR1AssetError("immutable receipt contains an unsupported or non-finite value")


def _array_receipt(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(value)
    return _freeze(
        {
            "dtype": array.dtype.str,
            "shape": tuple(int(item) for item in array.shape),
            "nbytes": int(array.nbytes),
            "sha256": _sha256(array.tobytes(order="C")),
        }
    )


def _sequence(values: Sequence[str], *, name: str, minimum: int = 1) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise NextR1AssetError(f"{name} must be an ordered sequence of strings")
    try:
        result = tuple(values)
    except TypeError as error:
        raise NextR1AssetError(f"{name} must be an ordered sequence of strings") from error
    if (
        len(result) < minimum
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise NextR1AssetError(f"{name} must contain unique nonempty strings")
    return result


def _label_rows(values: Sequence[str], *, expected_rows: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise NextR1AssetError("phase1_labels must be row-aligned strings")
    try:
        result = tuple(values)
    except TypeError as error:
        raise NextR1AssetError("phase1_labels must be row-aligned strings") from error
    if len(result) != expected_rows or any(not isinstance(item, str) or not item for item in result):
        raise NextR1AssetError("phase1_labels must close over every gradient row")
    return result


def _row_strings(values: Sequence[str], *, name: str, expected_rows: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise NextR1AssetError(f"{name} must be row-aligned strings")
    try:
        result = tuple(values)
    except TypeError as error:
        raise NextR1AssetError(f"{name} must be row-aligned strings") from error
    if len(result) != expected_rows or any(not isinstance(item, str) or not item for item in result):
        raise NextR1AssetError(f"{name} must close over every gradient row")
    return result


def _registry_root(values: Sequence[str], *, kind: str) -> str:
    return _sha256(
        _canonical_bytes({"schema": REGISTRY_ROOT_SCHEMA, "kind": kind, "values": tuple(values)})
    )


def phase1_fit_physical_id_root(
    phase1_receiver_ids: Sequence[str],
    phase1_labels: Sequence[str],
    phase1_physical_ids: Sequence[str],
) -> str:
    """Canonical opaque binding for the fit rows consumed by FABR selection."""

    receivers = tuple(phase1_receiver_ids)
    labels = tuple(phase1_labels)
    physical_ids = tuple(phase1_physical_ids)
    if len(receivers) != len(labels) or len(labels) != len(physical_ids):
        raise NextR1AssetError("fit physical-ID root requires aligned Phase1 rows")
    if any(not isinstance(item, str) or not item for item in receivers + labels + physical_ids):
        raise NextR1AssetError("fit physical-ID root requires nonempty Phase1 strings")
    if len(set(physical_ids)) != len(physical_ids):
        raise NextR1AssetError("Phase1 fit physical IDs must be globally unique")
    rows = sorted(zip(receivers, labels, physical_ids), key=lambda item: item)
    return _sha256(_canonical_bytes({"schema": PHYSICAL_ROOT_SCHEMA, "rows": rows}))


@dataclass(frozen=True, slots=True)
class Phase1GradientBlock:
    """One row-aligned, frozen-TX-loss gradient matrix for a legal FABR block."""

    block_id: str
    gradients: np.ndarray
    phase1_receiver_ids: tuple[str, ...]
    phase1_physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.block_id not in fabr.BLOCK_TIE_ORDER:
            raise NextR1AssetError("gradient block is outside the frozen four-block registry")
        values = np.asarray(self.gradients)
        expected_dimension = fabr.BLOCK_DIMENSIONS[self.block_id]
        if (
            values.dtype != _F32
            or values.ndim != 2
            or values.shape[0] < 2
            or values.shape[1] != expected_dimension
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
        ):
            raise NextR1AssetError(
                f"{self.block_id} gradients must be C-contiguous float32 [N,{expected_dimension}]"
            )
        receiver_ids = _row_strings(
            self.phase1_receiver_ids, name="phase1_receiver_ids", expected_rows=values.shape[0]
        )
        physical_ids = _sequence(
            self.phase1_physical_ids, name="phase1_physical_ids", minimum=2
        )
        if len(physical_ids) != values.shape[0]:
            raise NextR1AssetError("gradient receiver/physical IDs must be row-aligned")
        frozen = np.array(values, dtype=np.float32, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "gradients", frozen)
        object.__setattr__(self, "phase1_receiver_ids", receiver_ids)
        object.__setattr__(self, "phase1_physical_ids", physical_ids)


@dataclass(frozen=True, slots=True)
class Phase1FoldSeal:
    """External Phase1 fold commitment; the deployable bundle retains only its hash."""

    fold_id: str
    held_receiver: str | None
    held_class: str | None
    checkpoint_sha256: str
    representation_rule_sha256: str
    row_phase1_seal_sha256: str
    phase1_fit_physical_id_root_sha256: str
    phase1_cell_physical_id_root_sha256: str

    def __post_init__(self) -> None:
        fold_id = str(self.fold_id)
        if not fold_id:
            raise NextR1AssetError("Phase1 fold_id must be nonempty")
        held_receiver = self.held_receiver
        held_class = self.held_class
        if (held_receiver is None) != (held_class is None):
            raise NextR1AssetError("held receiver/class must be both present or both absent")
        if held_receiver is not None and (not isinstance(held_receiver, str) or not held_receiver):
            raise NextR1AssetError("held_receiver must be a nonempty string when present")
        if held_class is not None and (not isinstance(held_class, str) or not held_class):
            raise NextR1AssetError("held_class must be a nonempty string when present")
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(self, "held_receiver", held_receiver)
        object.__setattr__(self, "held_class", held_class)
        for name in (
            "checkpoint_sha256",
            "representation_rule_sha256",
            "row_phase1_seal_sha256",
            "phase1_fit_physical_id_root_sha256",
            "phase1_cell_physical_id_root_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))

    @property
    def seal_sha256(self) -> str:
        return _sha256(
            _canonical_bytes(
                {
                    "schema": BUNDLE_SCHEMA,
                    "fold_id": self.fold_id,
                    "held_receiver": self.held_receiver,
                    "held_class": self.held_class,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "representation_rule_sha256": self.representation_rule_sha256,
                    "row_phase1_seal_sha256": self.row_phase1_seal_sha256,
                    "phase1_fit_physical_id_root_sha256": self.phase1_fit_physical_id_root_sha256,
                    "phase1_cell_physical_id_root_sha256": self.phase1_cell_physical_id_root_sha256,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class TSLPhysicalLOOBinding:
    """One explicit receiver×class physical-LOO coverage item for the TSL prior."""

    receiver: str
    class_label: str
    fold: tsl.Phase1PhysicalLOOFold

    def __post_init__(self) -> None:
        receiver = str(self.receiver)
        class_label = str(self.class_label)
        if not receiver or not class_label or type(self.fold) is not tsl.Phase1PhysicalLOOFold:
            raise NextR1AssetError("TSL physical-LOO binding requires receiver, class, and exact fold")
        object.__setattr__(self, "receiver", receiver)
        object.__setattr__(self, "class_label", class_label)


@dataclass(frozen=True, slots=True)
class Phase1DirectionalValidation:
    """One externally measured Phase1 directional validation receipt.

    Counts are explicitly Phase1 label-derived technical checks.  They are not
    a selection score and are used only as the frozen pass/fail safety gate.
    """

    basis_sha256: str
    coefficient_sha256: str
    baseline_total_correct: int
    perturbed_total_correct: int
    baseline_per_class_correct: Mapping[str, int]
    perturbed_per_class_correct: Mapping[str, int]
    per_class_total: Mapping[str, int]
    forward_action_max_abs_delta: float
    repeated_forward_jitter_max_abs_delta: float
    validation_seal_sha256: str

    def __post_init__(self) -> None:
        for name in ("basis_sha256", "coefficient_sha256", "validation_seal_sha256"):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))
        for name in ("baseline_total_correct", "perturbed_total_correct"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise NextR1AssetError(f"{name} must be a nonnegative integer")
        maps: dict[str, Mapping[str, int]] = {}
        for name in ("baseline_per_class_correct", "perturbed_per_class_correct", "per_class_total"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise NextR1AssetError(f"{name} must be a nonempty class-count mapping")
            normalized: dict[str, int] = {}
            for label, count in value.items():
                if not isinstance(label, str) or not label or not isinstance(count, int) or count < 0:
                    raise NextR1AssetError(f"{name} must contain nonnegative integer class counts")
                normalized[label] = count
            maps[name] = MappingProxyType(dict(sorted(normalized.items())))
            object.__setattr__(self, name, maps[name])
        keys = tuple(maps["per_class_total"])
        if tuple(maps["baseline_per_class_correct"]) != keys or tuple(maps["perturbed_per_class_correct"]) != keys:
            raise NextR1AssetError("Phase1 directional class-count keys must match exactly")
        if any(maps["per_class_total"][label] <= 0 for label in keys):
            raise NextR1AssetError("Phase1 directional class totals must be positive")
        if any(
            maps[name][label] > maps["per_class_total"][label]
            for name in ("baseline_per_class_correct", "perturbed_per_class_correct")
            for label in keys
        ):
            raise NextR1AssetError("Phase1 directional correct counts exceed their class totals")
        if self.baseline_total_correct != sum(maps["baseline_per_class_correct"].values()):
            raise NextR1AssetError("baseline total correct does not close per-class counts")
        if self.perturbed_total_correct != sum(maps["perturbed_per_class_correct"].values()):
            raise NextR1AssetError("perturbed total correct does not close per-class counts")
        for name in ("forward_action_max_abs_delta", "repeated_forward_jitter_max_abs_delta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise NextR1AssetError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


Phase1ValidationCallback = Callable[
    [str, np.ndarray, np.ndarray, tuple[str, ...]], Phase1DirectionalValidation
]


@dataclass(frozen=True, slots=True)
class NextR1Phase1AssetBundle:
    """Immutable deployable FABR+TSL pair with compact Phase1 closure evidence."""

    fabr_asset: fabr.FABRAsset
    tsl_prior: tsl.TSLPhase1Prior
    fold_seal_sha256: str
    phase1_receiver_registry_sha256: str
    phase1_class_registry_sha256: str
    receipt: Mapping[str, Any]
    schema: str = BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BUNDLE_SCHEMA:
            raise NextR1AssetError("NEXT-R1 asset bundle schema drift")
        if type(self.fabr_asset) is not fabr.FABRAsset or type(self.tsl_prior) is not tsl.TSLPhase1Prior:
            raise NextR1AssetError("asset bundle requires exact FABRAsset and TSLPhase1Prior values")
        for name in (
            "fold_seal_sha256",
            "phase1_receiver_registry_sha256",
            "phase1_class_registry_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))
        if not isinstance(self.receipt, Mapping):
            raise NextR1AssetError("asset bundle receipt must be a mapping")
        receipt = _freeze(self.receipt)
        required = {
            "schema",
            "selection_sha256",
            "fold_seal_sha256",
            "checkpoint_sha256",
            "representation_rule_sha256",
            "row_phase1_seal_sha256",
            "phase1_cell_physical_id_root_sha256",
            "phase1_receiver_registry_sha256",
            "phase1_class_registry_sha256",
            "selected_block_id",
            "tsl_prior_sha256",
        }
        if not required.issubset(receipt):
            raise NextR1AssetError("asset bundle receipt is incomplete")
        if receipt["schema"] != BUNDLE_SCHEMA:
            raise NextR1AssetError("asset bundle receipt schema drift")
        for name in (
            "selection_sha256",
            "fold_seal_sha256",
            "checkpoint_sha256",
            "representation_rule_sha256",
            "row_phase1_seal_sha256",
            "phase1_cell_physical_id_root_sha256",
            "phase1_receiver_registry_sha256",
            "phase1_class_registry_sha256",
            "tsl_prior_sha256",
        ):
            _require_sha256(receipt[name], name=f"receipt.{name}")
        if (
            receipt["fold_seal_sha256"] != self.fold_seal_sha256
            or receipt["phase1_receiver_registry_sha256"] != self.phase1_receiver_registry_sha256
            or receipt["phase1_class_registry_sha256"] != self.phase1_class_registry_sha256
            or receipt["checkpoint_sha256"] != self.fabr_asset.checkpoint_sha256
            or receipt["checkpoint_sha256"] != self.tsl_prior.checkpoint_sha256
            or receipt["representation_rule_sha256"] != self.tsl_prior.representation_rule_sha256
            or receipt["row_phase1_seal_sha256"] != self.fabr_asset.phase1_seal_sha256
            or receipt["phase1_cell_physical_id_root_sha256"]
            != self.tsl_prior.cell_physical_id_root_sha256
            or receipt["selection_sha256"] != self.fabr_asset.phase1_selection_sha256
            or receipt["selected_block_id"] != self.fabr_asset.block_id
            or receipt["tsl_prior_sha256"] != self.tsl_prior.prior_sha256
        ):
            raise NextR1AssetError("FABR/TSL bundle binding drift")
        object.__setattr__(self, "receipt", receipt)

    @property
    def serialized_bytes(self) -> bytes:
        document = {
            "schema": self.schema,
            "fabr_asset_b64": base64.b64encode(fabr.serialize_fabr_asset(self.fabr_asset)).decode("ascii"),
            "tsl_prior_b64": base64.b64encode(tsl.serialize_phase1_prior(self.tsl_prior)).decode("ascii"),
            "fold_seal_sha256": self.fold_seal_sha256,
            "phase1_receiver_registry_sha256": self.phase1_receiver_registry_sha256,
            "phase1_class_registry_sha256": self.phase1_class_registry_sha256,
            "receipt": dict(self.receipt),
        }
        return _canonical_bytes(document)

    @property
    def bundle_sha256(self) -> str:
        return _sha256(self.serialized_bytes)


@dataclass(frozen=True, slots=True)
class _Phase1Input:
    blocks: Mapping[str, Phase1GradientBlock]
    phase1_labels: tuple[str, ...]
    receiver_registry: tuple[str, ...]
    class_registry: tuple[str, ...]
    active_receivers: tuple[str, ...]
    active_classes: tuple[str, ...]
    cell_values: tuple[tsl.Phase1Cell, ...]
    loo_bindings: tuple[TSLPhysicalLOOBinding, ...]
    cell_root_sha256: str
    fit_root_sha256: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    block_id: str
    primary_eigenvalue: float
    secondary_eigenvalue: float
    geometry: fabr.Phase1FisherGeometry
    basis_qint8: np.ndarray
    basis_scale_fp16: np.ndarray
    fisher_k_fp16: np.ndarray
    actual_basis: np.ndarray
    actual_k: np.ndarray
    minimum_principal_cosine: float
    validation_receipts: tuple[Mapping[str, Any], ...]
    jitter_fp16: np.ndarray
    receipt: Mapping[str, Any]


def _basis_sha256(basis: np.ndarray) -> str:
    values = np.ascontiguousarray(basis, dtype=np.float64)
    return _sha256(values.tobytes(order="C"))


def _coefficient_sha256(coefficient: np.ndarray) -> str:
    values = np.ascontiguousarray(coefficient, dtype=np.float32)
    return _sha256(values.tobytes(order="C"))


def _dequantized_basis(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(
        codes.astype(np.float64) * scales.astype(np.float64)[None, :], dtype=np.float64
    )
    result.setflags(write=False)
    return result


def _canonical_sign_columns(value: np.ndarray) -> np.ndarray:
    basis = np.array(value, dtype=np.float64, copy=True, order="C")
    for column in range(basis.shape[1]):
        vector = basis[:, column]
        maximum = float(np.max(np.abs(vector)))
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise NextR1AssetError("generalized eigenvector is zero or non-finite")
        first = int(np.flatnonzero(np.abs(vector) == maximum)[0])
        if vector[first] < 0.0:
            basis[:, column] *= -1.0
    return np.ascontiguousarray(basis, dtype=np.float64)


def _generalized_top2(geometry: fabr.Phase1FisherGeometry) -> tuple[np.ndarray, np.ndarray]:
    fisher = np.asarray(geometry.fisher, dtype=np.float64)
    scatter = np.asarray(geometry.receiver_scatter, dtype=np.float64)
    eigenvalues_f, eigenvectors_f = np.linalg.eigh(fisher)
    if (
        eigenvalues_f.shape[0] < fabr.RANK
        or not np.isfinite(eigenvalues_f).all()
        or float(np.min(eigenvalues_f)) <= 0.0
    ):
        raise NextR1AssetError("Phase1 generalized Fisher metric is not positive definite")
    inverse_square_root = (eigenvectors_f * (1.0 / np.sqrt(eigenvalues_f))[None, :]) @ eigenvectors_f.T
    symmetric = inverse_square_root @ scatter @ inverse_square_root
    symmetric = 0.5 * (symmetric + symmetric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(-eigenvalues, kind="stable")
    selected_values = np.ascontiguousarray(eigenvalues[order[: fabr.RANK]], dtype=np.float64)
    if (
        selected_values.shape != (fabr.RANK,)
        or not np.isfinite(selected_values).all()
        or float(selected_values[0]) <= 0.0
        or float(selected_values[1]) <= 0.0
    ):
        raise NextR1AssetError("Phase1 receiver scatter has no positive rank-two generalized basis")
    basis = inverse_square_root @ eigenvectors[:, order[: fabr.RANK]]
    basis = _canonical_sign_columns(basis)
    if not np.isfinite(basis).all():
        raise NextR1AssetError("Phase1 generalized basis became non-finite")
    return np.ascontiguousarray(basis, dtype=np.float64), selected_values


def _quantize_actual_basis(
    basis: np.ndarray, fisher: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float]]:
    value = np.asarray(basis, dtype=np.float64)
    matrix = np.asarray(fisher, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != fabr.RANK or not np.isfinite(value).all():
        raise NextR1AssetError("basis must be finite [P,2]")
    if matrix.shape != (value.shape[0], value.shape[0]) or not np.isfinite(matrix).all():
        raise NextR1AssetError("Fisher matrix/basis dimension drift")
    raw_scale = np.max(np.abs(value), axis=0) / 127.0
    scale = np.ascontiguousarray(raw_scale.astype(_F16), dtype=_F16)
    if not np.isfinite(scale).all() or np.any(scale <= np.float16(0.0)):
        raise NextR1AssetError("INT8 basis scale is zero, non-finite, or underflowed")
    codes = np.clip(np.rint(value / scale.astype(np.float64)[None, :]), -127, 127).astype(np.int8)
    codes = np.ascontiguousarray(codes, dtype=np.int8)
    if np.any(codes == np.int8(-128)) or not np.all(np.any(codes != 0, axis=0)):
        raise NextR1AssetError("actual INT8 basis lost a rank direction")
    actual_basis = _dequantized_basis(codes, scale)
    singular_values = np.linalg.svd(actual_basis, compute_uv=False)
    if (
        singular_values.shape != (fabr.RANK,)
        or not np.isfinite(singular_values).all()
        or float(singular_values[-1]) <= 64.0 * np.finfo(np.float64).eps * max(1.0, float(singular_values[0]))
    ):
        raise NextR1AssetError("actual INT8 dequantized basis is not rank two")
    raw_k = actual_basis.T @ matrix @ actual_basis
    raw_k = 0.5 * (raw_k + raw_k.T)
    k_fp16 = np.empty((fabr.RANK, fabr.RANK), dtype=_F16)
    k_fp16[0, 0] = np.float16(raw_k[0, 0])
    k_fp16[1, 1] = np.float16(raw_k[1, 1])
    k_fp16[0, 1] = np.float16(raw_k[0, 1])
    k_fp16[1, 0] = k_fp16[0, 1]
    actual_k = np.ascontiguousarray(k_fp16.astype(np.float64), dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(actual_k)
    condition = float(np.linalg.cond(actual_k))
    if (
        not np.isfinite(actual_k).all()
        or not np.isfinite(eigenvalues).all()
        or float(np.min(eigenvalues)) <= 0.0
        or not math.isfinite(condition)
        or condition > fabr.MAX_CONDITION
    ):
        raise NextR1AssetError("actual INT8 Fisher K is non-PD or exceeds the frozen condition cap")
    return (
        codes,
        scale,
        k_fp16,
        actual_basis,
        actual_k,
        condition,
        (float(np.min(eigenvalues)), float(np.max(eigenvalues))),
    )


def _principal_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_q, _ = np.linalg.qr(np.asarray(left, dtype=np.float64), mode="reduced")
    right_q, _ = np.linalg.qr(np.asarray(right, dtype=np.float64), mode="reduced")
    singular_values = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
    if singular_values.shape != (fabr.RANK,) or not np.isfinite(singular_values).all():
        raise NextR1AssetError("principal cosine decomposition drift")
    cosine = float(np.min(np.clip(singular_values, 0.0, 1.0)))
    if not math.isfinite(cosine):
        raise NextR1AssetError("principal cosine became non-finite")
    return cosine


def _ceil_fp16_nonnegative(value: float) -> np.ndarray:
    if not math.isfinite(value) or value < 0.0:
        raise NextR1AssetError("forward jitter must be finite and nonnegative")
    rounded = np.float16(value)
    if not math.isfinite(float(rounded)):
        raise NextR1AssetError("forward jitter overflows FP16")
    if float(rounded) < value:
        rounded = np.nextafter(rounded, np.float16(np.inf), dtype=np.float16)
    if not math.isfinite(float(rounded)):
        raise NextR1AssetError("forward jitter cannot be sealed in FP16")
    return np.asarray([rounded], dtype=_F16)


def _validation_floor(counts: Mapping[str, int], totals: Mapping[str, int]) -> float:
    values = [float(counts[label]) / float(totals[label]) for label in totals]
    return float(min(values))


def _run_directional_validation(
    *,
    block_id: str,
    actual_basis: np.ndarray,
    phase1_labels: tuple[str, ...],
    validation_callback: Phase1ValidationCallback,
) -> tuple[tuple[Mapping[str, Any], ...], float]:
    if not callable(validation_callback):
        raise NextR1AssetError("Phase1 validation callback must be callable")
    basis = np.array(actual_basis, dtype=np.float64, copy=True, order="C")
    basis.setflags(write=False)
    expected_basis_sha256 = _basis_sha256(basis)
    baseline_key: tuple[Any, ...] | None = None
    receipts: list[Mapping[str, Any]] = []
    jitters: list[float] = []
    for direction in range(fabr.RANK):
        for sign in (-1.0, 1.0):
            coefficient = np.zeros(fabr.RANK, dtype=np.float32)
            coefficient[direction] = np.float32(sign * fabr.DELTA)
            coefficient.setflags(write=False)
            try:
                result = validation_callback(block_id, basis, coefficient, phase1_labels)
            except Exception as error:
                raise NextR1AssetError("Phase1 directional validation callback failed") from error
            if type(result) is not Phase1DirectionalValidation:
                raise NextR1AssetError("Phase1 validation callback must return exact Phase1DirectionalValidation")
            if result.basis_sha256 != expected_basis_sha256:
                raise NextR1AssetError("Phase1 validation result is not bound to actual dequantized B")
            if result.coefficient_sha256 != _coefficient_sha256(coefficient):
                raise NextR1AssetError("Phase1 validation result coefficient binding drift")
            baseline_floor = _validation_floor(result.baseline_per_class_correct, result.per_class_total)
            perturbed_floor = _validation_floor(result.perturbed_per_class_correct, result.per_class_total)
            if result.forward_action_max_abs_delta <= result.repeated_forward_jitter_max_abs_delta:
                raise NextR1AssetError("Phase1 functional action does not exceed repeated-forward jitter")
            current_key = (
                result.baseline_total_correct,
                tuple(result.baseline_per_class_correct.items()),
                tuple(result.per_class_total.items()),
                result.validation_seal_sha256,
            )
            if baseline_key is None:
                baseline_key = current_key
            elif current_key != baseline_key:
                raise NextR1AssetError("Phase1 R0 validation baseline drifted across directions")
            jitters.append(result.repeated_forward_jitter_max_abs_delta)
            receipts.append(
                _freeze(
                    {
                        "direction": direction,
                        "signed_delta": float(coefficient[direction]),
                        "basis_sha256": result.basis_sha256,
                        "coefficient_sha256": result.coefficient_sha256,
                        "validation_seal_sha256": result.validation_seal_sha256,
                        "baseline_total_correct": result.baseline_total_correct,
                        "perturbed_total_correct": result.perturbed_total_correct,
                        "total_correct_non_decrease": (
                            result.perturbed_total_correct >= result.baseline_total_correct
                        ),
                        "baseline_per_class_floor": baseline_floor,
                        "perturbed_per_class_floor": perturbed_floor,
                        "per_class_floor_non_decrease": perturbed_floor >= baseline_floor,
                        "forward_action_max_abs_delta": result.forward_action_max_abs_delta,
                        "repeated_forward_jitter_max_abs_delta": result.repeated_forward_jitter_max_abs_delta,
                    }
                )
            )
    return tuple(receipts), max(jitters)


def _cell_row_map(cells: Sequence[tsl.Phase1Cell]) -> Mapping[str, tuple[str, str, np.ndarray]]:
    entries: dict[str, tuple[str, str, np.ndarray]] = {}
    for cell in cells:
        for index, physical_id in enumerate(cell.physical_ids):
            if physical_id in entries:
                raise NextR1AssetError("Phase1 cells reuse a physical ID across receiver/class cells")
            entries[physical_id] = (cell.receiver, cell.class_label, cell.z160[index])
    return MappingProxyType(entries)


def _require_same_f32(left: np.ndarray, right: np.ndarray, *, name: str) -> None:
    if np.asarray(left, dtype=np.float32).tobytes(order="C") != np.asarray(right, dtype=np.float32).tobytes(order="C"):
        raise NextR1AssetError(f"{name} is not bound to the frozen Phase1 cell representation")


def _validate_tsl_loo_bindings(
    bindings: Sequence[TSLPhysicalLOOBinding],
    *,
    active_receivers: tuple[str, ...],
    active_classes: tuple[str, ...],
    cell_rows: Mapping[str, tuple[str, str, np.ndarray]],
) -> tuple[TSLPhysicalLOOBinding, ...]:
    values = tuple(bindings)
    expected_pairs = {(receiver, class_label) for receiver in active_receivers for class_label in active_classes}
    seen_pairs: set[tuple[str, str]] = set()
    validation_ids: set[str] = set()
    for binding in values:
        if type(binding) is not TSLPhysicalLOOBinding:
            raise NextR1AssetError("TSL physical-LOO coverage requires exact binding values")
        pair = (binding.receiver, binding.class_label)
        if pair not in expected_pairs or pair in seen_pairs:
            raise NextR1AssetError("TSL physical-LOO binding pair is missing, outside-grid, or duplicated")
        seen_pairs.add(pair)
        fold = binding.fold
        if not fold.validation_physical_ids or set(fold.validation_labels) != {binding.class_label}:
            raise NextR1AssetError("TSL physical-LOO validation must cover its explicit receiver×class cell")
        for index, physical_id in enumerate(fold.validation_physical_ids):
            if physical_id in validation_ids:
                raise NextR1AssetError("TSL physical-LOO validation physical IDs must be globally unique")
            validation_ids.add(physical_id)
            item = cell_rows.get(physical_id)
            if item is None or item[0] != binding.receiver or item[1] != binding.class_label:
                raise NextR1AssetError("TSL validation physical ID is outside its bound receiver×class cell")
            _require_same_f32(fold.validation_z160[index], item[2], name="TSL validation representation")
        for index, physical_id in enumerate(fold.support_physical_ids):
            item = cell_rows.get(physical_id)
            if item is None or fold.support_labels[index] != item[1]:
                raise NextR1AssetError("TSL support physical ID/label is not bound to a Phase1 cell")
            _require_same_f32(fold.support_z160[index], item[2], name="TSL support representation")
    if seen_pairs != expected_pairs:
        raise NextR1AssetError("TSL physical-LOO coverage does not close the active receiver×class grid")
    return values


def _validate_phase1_input(
    *,
    gradient_blocks: Sequence[Phase1GradientBlock],
    phase1_labels: Sequence[str],
    fold_seal: Phase1FoldSeal,
    phase1_receiver_registry: Sequence[str],
    phase1_class_registry: Sequence[str],
    phase1_cells: Sequence[tsl.Phase1Cell],
    phase1_physical_loo_folds: Sequence[TSLPhysicalLOOBinding],
) -> _Phase1Input:
    if type(fold_seal) is not Phase1FoldSeal:
        raise NextR1AssetError("asset construction requires an exact Phase1FoldSeal")
    blocks = tuple(gradient_blocks)
    if len(blocks) != len(fabr.BLOCK_TIE_ORDER) or {block.block_id for block in blocks if type(block) is Phase1GradientBlock} != set(fabr.BLOCK_TIE_ORDER):
        raise NextR1AssetError("asset construction requires all and only the four frozen gradient blocks")
    if any(type(block) is not Phase1GradientBlock for block in blocks):
        raise NextR1AssetError("asset construction requires exact Phase1GradientBlock values")
    ordered_blocks = {block.block_id: block for block in blocks}
    primary = ordered_blocks[fabr.BLOCK_TIE_ORDER[0]]
    labels = _label_rows(phase1_labels, expected_rows=primary.gradients.shape[0])
    for block_id in fabr.BLOCK_TIE_ORDER[1:]:
        block = ordered_blocks[block_id]
        if (
            block.gradients.shape[0] != primary.gradients.shape[0]
            or block.phase1_receiver_ids != primary.phase1_receiver_ids
            or block.phase1_physical_ids != primary.phase1_physical_ids
        ):
            raise NextR1AssetError("all gradient blocks must share exactly aligned Phase1 row identities")
    receiver_registry = _sequence(
        phase1_receiver_registry, name="phase1_receiver_registry", minimum=FROZEN_RECEIVER_COUNT
    )
    class_registry = _sequence(
        phase1_class_registry, name="phase1_class_registry", minimum=FROZEN_CLASS_COUNT)
    if len(receiver_registry) != FROZEN_RECEIVER_COUNT or len(class_registry) != FROZEN_CLASS_COUNT:
        raise NextR1AssetError("NEXT-R1 requires the frozen seven-receiver by six-class registry")
    if fold_seal.held_receiver is None:
        active_receivers = receiver_registry
        active_classes = class_registry
    else:
        if fold_seal.held_receiver not in receiver_registry or fold_seal.held_class not in class_registry:
            raise NextR1AssetError("fold seal held receiver/class is outside the frozen registries")
        active_receivers = tuple(item for item in receiver_registry if item != fold_seal.held_receiver)
        active_classes = tuple(item for item in class_registry if item != fold_seal.held_class)
    if set(primary.phase1_receiver_ids) != set(active_receivers) or set(labels) != set(active_classes):
        raise NextR1AssetError("fit rows do not exactly realize the sealed active receiver/class registry")
    if any(item not in active_receivers for item in primary.phase1_receiver_ids) or any(item not in active_classes for item in labels):
        raise NextR1AssetError("held receiver/class leaked into the sealed Phase1 fit rows")
    expected_rows = len(active_receivers) * len(active_classes) * FROZEN_PHYSICAL_PER_CELL
    if primary.gradients.shape[0] != expected_rows:
        raise NextR1AssetError("NEXT-R1 fit rows do not match the frozen receiver×class×physical grid")
    fit_root = phase1_fit_physical_id_root(
        primary.phase1_receiver_ids, labels, primary.phase1_physical_ids
    )
    if fit_root != fold_seal.phase1_fit_physical_id_root_sha256:
        raise NextR1AssetError("fold seal does not bind the consumed Phase1 fit physical IDs")
    cells = tuple(phase1_cells)
    if any(type(cell) is not tsl.Phase1Cell for cell in cells):
        raise NextR1AssetError("TSL prior requires exact Phase1Cell values")
    expected_pairs = {(receiver, class_label) for receiver in active_receivers for class_label in active_classes}
    actual_pairs = {(cell.receiver, cell.class_label) for cell in cells}
    if len(cells) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise NextR1AssetError("TSL cells must form the complete active receiver×class grid")
    if any(len(cell.physical_ids) != FROZEN_PHYSICAL_PER_CELL for cell in cells):
        raise NextR1AssetError("every frozen TSL receiver×class cell must contain fourteen physical rows")
    cell_rows = _cell_row_map(cells)
    if set(primary.phase1_physical_ids) != set(cell_rows):
        raise NextR1AssetError("gradient physical IDs and TSL cell physical IDs do not close exactly")
    for index, physical_id in enumerate(primary.phase1_physical_ids):
        receiver, class_label, _row = cell_rows[physical_id]
        if receiver != primary.phase1_receiver_ids[index] or class_label != labels[index]:
            raise NextR1AssetError("gradient receiver/label rows are not bound to their Phase1 cells")
    cell_root = tsl.phase1_cell_physical_id_root(cells)
    if cell_root != fold_seal.phase1_cell_physical_id_root_sha256:
        raise NextR1AssetError("fold seal does not bind the Phase1 TSL cell physical-ID root")
    bindings = _validate_tsl_loo_bindings(
        phase1_physical_loo_folds,
        active_receivers=active_receivers,
        active_classes=active_classes,
        cell_rows=cell_rows,
    )
    return _Phase1Input(
        blocks=MappingProxyType(ordered_blocks),
        phase1_labels=labels,
        receiver_registry=receiver_registry,
        class_registry=class_registry,
        active_receivers=active_receivers,
        active_classes=active_classes,
        cell_values=tuple(sorted(cells, key=lambda item: (item.receiver, item.class_label))),
        loo_bindings=bindings,
        cell_root_sha256=cell_root,
        fit_root_sha256=fit_root,
    )


def _candidate_for_block(
    *,
    block: Phase1GradientBlock,
    phase1_labels: tuple[str, ...],
    validation_callback: Phase1ValidationCallback,
) -> _Candidate:
    geometry = fabr.phase1_fisher_geometry(block.gradients, block.phase1_receiver_ids)
    basis, eigenvalues = _generalized_top2(geometry)
    (
        codes,
        scales,
        fisher_k_fp16,
        actual_basis,
        actual_k,
        condition,
        k_eigenvalues,
    ) = _quantize_actual_basis(basis, geometry.fisher)
    receiver_values = tuple(sorted(set(block.phase1_receiver_ids)))
    loo_values: list[float] = []
    loo_receipts: list[Mapping[str, Any]] = []
    for receiver in receiver_values:
        mask = np.asarray([item != receiver for item in block.phase1_receiver_ids], dtype=bool)
        try:
            loo_geometry = fabr.phase1_fisher_geometry(block.gradients[mask], tuple(np.asarray(block.phase1_receiver_ids)[mask]))
            loo_basis, _loo_eigenvalues = _generalized_top2(loo_geometry)
            _loo_codes, _loo_scales, _loo_k, loo_actual_basis, _loo_actual_k, _loo_condition, _loo_k_eigenvalues = _quantize_actual_basis(
                loo_basis, loo_geometry.fisher
            )
            cosine = _principal_cosine(actual_basis, loo_actual_basis)
        except Exception as error:
            raise NextR1AssetError(f"leave-one-receiver subspace failed for {block.block_id}") from error
        loo_values.append(cosine)
        loo_receipts.append(_freeze({"principal_cosine": cosine}))
    minimum_cosine = float(min(loo_values))
    validation_receipts, max_jitter = _run_directional_validation(
        block_id=block.block_id,
        actual_basis=actual_basis,
        phase1_labels=phase1_labels,
        validation_callback=validation_callback,
    )
    jitter_fp16 = _ceil_fp16_nonnegative(max_jitter)
    receipt = _freeze(
        {
            "block_id": block.block_id,
            "gradient": dict(_array_receipt(block.gradients)),
            "gradient_second_moment": dict(_array_receipt(geometry.gradient_second_moment)),
            "fisher": dict(_array_receipt(geometry.fisher)),
            "receiver_scatter": dict(_array_receipt(geometry.receiver_scatter)),
            "epsilon_f": geometry.epsilon_f,
            "top_generalized_eigenvalues": tuple(float(item) for item in eigenvalues),
            "basis_qint8": dict(_array_receipt(codes)),
            "basis_scale_fp16": dict(_array_receipt(scales)),
            "actual_basis": dict(_array_receipt(actual_basis)),
            "fisher_k_fp16": dict(_array_receipt(fisher_k_fp16)),
            "actual_k_min_eigenvalue": k_eigenvalues[0],
            "actual_k_max_eigenvalue": k_eigenvalues[1],
            "actual_k_condition": condition,
            "leave_one_receiver_count": len(receiver_values),
            "leave_one_receiver_min_principal_cosine": minimum_cosine,
            "leave_one_receiver_cosine_audit_threshold": MIN_SUBSPACE_PRINCIPAL_COSINE,
            "leave_one_receiver_cosine_gate_used": False,
            "leave_one_receiver_cosines": tuple(loo_receipts),
            "directional_validation": tuple(validation_receipts),
            "forward_jitter_tolerance_fp16": float(jitter_fp16[0]),
        }
    )
    return _Candidate(
        block_id=block.block_id,
        primary_eigenvalue=float(eigenvalues[0]),
        secondary_eigenvalue=float(eigenvalues[1]),
        geometry=geometry,
        basis_qint8=codes,
        basis_scale_fp16=scales,
        fisher_k_fp16=fisher_k_fp16,
        actual_basis=actual_basis,
        actual_k=actual_k,
        minimum_principal_cosine=minimum_cosine,
        validation_receipts=validation_receipts,
        jitter_fp16=jitter_fp16,
        receipt=receipt,
    )


def _select_candidate(candidates: Sequence[_Candidate]) -> _Candidate:
    if not candidates:
        raise NextR1AssetSelectionError("no frozen FABR block passed all Phase1 selection gates")
    highest = max(candidate.primary_eigenvalue for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if math.isclose(candidate.primary_eigenvalue, highest, rel_tol=EIGENVALUE_TIE_RTOL, abs_tol=0.0)
    ]
    order = {block_id: index for index, block_id in enumerate(fabr.BLOCK_TIE_ORDER)}
    return sorted(tied, key=lambda candidate: order[candidate.block_id])[0]


def build_next_r1_phase1_assets(
    gradient_blocks: Sequence[Phase1GradientBlock],
    phase1_labels: Sequence[str],
    fold_seal: Phase1FoldSeal,
    phase1_receiver_registry: Sequence[str],
    phase1_class_registry: Sequence[str],
    phase1_cells: Sequence[tsl.Phase1Cell],
    phase1_physical_loo_folds: Sequence[TSLPhysicalLOOBinding],
    validation_callback: Phase1ValidationCallback,
) -> NextR1Phase1AssetBundle:
    """Build one sealed Phase1 NEXT-R1 FABR+TSL bundle.

    ``phase1_labels`` is intentionally explicit: these are the only labels
    accepted by this API.  The callback receives the *actual* dequantized INT8
    basis and each frozen ``±2^-6`` rank direction for its Phase1 technical
    validation.  Phase1 accuracy/floor and LOO cosine are retained as audit
    values, not pre-performance release gates.  Only numerical invalidity,
    binding drift, or a functional action no larger than repeat jitter rejects
    a block; no alternate rank, layer, or numerical fallback is introduced.
    """

    context = _validate_phase1_input(
        gradient_blocks=gradient_blocks,
        phase1_labels=phase1_labels,
        fold_seal=fold_seal,
        phase1_receiver_registry=phase1_receiver_registry,
        phase1_class_registry=phase1_class_registry,
        phase1_cells=phase1_cells,
        phase1_physical_loo_folds=phase1_physical_loo_folds,
    )
    accepted: list[_Candidate] = []
    candidate_receipts: list[Mapping[str, Any]] = []
    for block_id in fabr.BLOCK_TIE_ORDER:
        try:
            candidate = _candidate_for_block(
                block=context.blocks[block_id],
                phase1_labels=context.phase1_labels,
                validation_callback=validation_callback,
            )
        except (NextR1AssetError, fabr.FABRError) as error:
            candidate_receipts.append(
                _freeze({"block_id": block_id, "status": "rejected", "reason": type(error).__name__})
            )
        else:
            accepted.append(candidate)
            candidate_receipts.append(
                _freeze(
                    {
                        "block_id": block_id,
                        "status": "accepted",
                        "primary_generalized_eigenvalue": candidate.primary_eigenvalue,
                        "receipt": dict(candidate.receipt),
                    }
                )
            )
    selected = _select_candidate(accepted)
    selection_payload = {
        "schema": SELECTION_SCHEMA,
        "fold_seal_sha256": fold_seal.seal_sha256,
        "checkpoint_sha256": fold_seal.checkpoint_sha256,
        "representation_rule_sha256": fold_seal.representation_rule_sha256,
        "row_phase1_seal_sha256": fold_seal.row_phase1_seal_sha256,
        "phase1_fit_physical_id_root_sha256": context.fit_root_sha256,
        "phase1_cell_physical_id_root_sha256": context.cell_root_sha256,
        "phase1_receiver_registry_sha256": _registry_root(context.receiver_registry, kind="receiver"),
        "phase1_class_registry_sha256": _registry_root(context.class_registry, kind="class"),
        "phase1_labels_sha256": _sha256("\n".join(context.phase1_labels).encode("utf-8")),
        "active_receiver_count": len(context.active_receivers),
        "active_class_count": len(context.active_classes),
        "selected_block_id": selected.block_id,
        "selected_primary_generalized_eigenvalue": selected.primary_eigenvalue,
        "tie_policy": "relative_float64_64eps_then_t1_to_t2_to_t3_to_joint",
        "phase1_performance_gate_used": False,
        "selection_criterion": "highest_phase1_generalized_eigenvalue_then_frozen_block_order",
        "candidate_receipts": tuple(candidate_receipts),
    }
    selection_sha256 = _sha256(_canonical_bytes(selection_payload))
    fabr_asset = fabr.FABRAsset(
        checkpoint_sha256=fold_seal.checkpoint_sha256,
        phase1_seal_sha256=fold_seal.row_phase1_seal_sha256,
        phase1_selection_sha256=selection_sha256,
        block_id=selected.block_id,
        basis_qint8=np.ascontiguousarray(selected.basis_qint8, dtype=np.int8),
        basis_scale_fp16=np.ascontiguousarray(selected.basis_scale_fp16, dtype=_F16),
        fisher_k_fp16=np.ascontiguousarray(selected.fisher_k_fp16, dtype=_F16),
        forward_jitter_tolerance_fp16=np.ascontiguousarray(selected.jitter_fp16, dtype=_F16),
    )
    prior_build = tsl.build_phase1_prior(
        cells=context.cell_values,
        validation_folds=tuple(binding.fold for binding in context.loo_bindings),
        checkpoint_sha256=fold_seal.checkpoint_sha256,
        cell_physical_id_root_sha256=context.cell_root_sha256,
        representation_rule_sha256=fold_seal.representation_rule_sha256,
    )
    receiver_registry_sha256 = _registry_root(context.receiver_registry, kind="receiver")
    class_registry_sha256 = _registry_root(context.class_registry, kind="class")
    receipt = _freeze(
        {
            "schema": BUNDLE_SCHEMA,
            "selection_sha256": selection_sha256,
            "fold_seal_sha256": fold_seal.seal_sha256,
            "checkpoint_sha256": fold_seal.checkpoint_sha256,
            "representation_rule_sha256": fold_seal.representation_rule_sha256,
            "row_phase1_seal_sha256": fold_seal.row_phase1_seal_sha256,
            "phase1_fit_physical_id_root_sha256": context.fit_root_sha256,
            "phase1_cell_physical_id_root_sha256": context.cell_root_sha256,
            "phase1_receiver_registry_sha256": receiver_registry_sha256,
            "phase1_class_registry_sha256": class_registry_sha256,
            "frozen_receiver_count": len(context.receiver_registry),
            "frozen_class_count": len(context.class_registry),
            "active_receiver_count": len(context.active_receivers),
            "active_class_count": len(context.active_classes),
            "complete_tsl_cell_grid": True,
            "complete_unique_tsl_physical_loo_coverage": True,
            "selected_block_id": selected.block_id,
            "selected_primary_generalized_eigenvalue": selected.primary_eigenvalue,
            "selected_secondary_generalized_eigenvalue": selected.secondary_eigenvalue,
            "selected_minimum_principal_cosine": selected.minimum_principal_cosine,
            "selected_principal_cosine_gate_used": False,
            "selected_phase1_total_non_decrease_all": all(
                bool(value["total_correct_non_decrease"])
                for value in selected.validation_receipts
            ),
            "selected_phase1_floor_non_decrease_all": all(
                bool(value["per_class_floor_non_decrease"])
                for value in selected.validation_receipts
            ),
            "phase1_performance_gate_used": False,
            "selected_actual_basis_sha256": _basis_sha256(selected.actual_basis),
            "selected_actual_fisher_k": dict(_array_receipt(selected.actual_k)),
            "fabr_asset_sha256": fabr.fabr_asset_sha256(fabr_asset),
            "tsl_prior_sha256": prior_build.prior.prior_sha256,
            "tsl_prior_receipt_sha256": _sha256(_canonical_bytes(dict(prior_build.receipt))),
            "tsl_prior_cell_count": len(context.cell_values),
            "tsl_prior_physical_loo_fold_count": len(context.loo_bindings),
        }
    )
    return NextR1Phase1AssetBundle(
        fabr_asset=fabr_asset,
        tsl_prior=prior_build.prior,
        fold_seal_sha256=fold_seal.seal_sha256,
        phase1_receiver_registry_sha256=receiver_registry_sha256,
        phase1_class_registry_sha256=class_registry_sha256,
        receipt=receipt,
    )


__all__ = [
    "BUNDLE_SCHEMA",
    "EIGENVALUE_TIE_RTOL",
    "FROZEN_CLASS_COUNT",
    "FROZEN_PHYSICAL_PER_CELL",
    "FROZEN_RECEIVER_COUNT",
    "MIN_SUBSPACE_PRINCIPAL_COSINE",
    "NextR1AssetError",
    "NextR1AssetSelectionError",
    "NextR1Phase1AssetBundle",
    "PHYSICAL_ROOT_SCHEMA",
    "Phase1DirectionalValidation",
    "Phase1FoldSeal",
    "Phase1GradientBlock",
    "Phase1ValidationCallback",
    "SELECTION_SCHEMA",
    "TSLPhysicalLOOBinding",
    "build_next_r1_phase1_assets",
    "phase1_fit_physical_id_root",
]
