"""Strict Phase1 trainer core for D103-R1-RXID-DUALSPLIT-MB4.

The module is intentionally limited to source training mechanics.  It has no
source-validation arrays, performance selector, deployment writer, Target
reader, or N607 launcher.  Its public data types make the Phase1 permissions
structural:

* ``L_s`` owns TX labels, pre-ReLU rows, TX-null/MMD, MetaBias episodes and the
  class-balanced domain bank.
* ``U_s`` has no TX or pre-ReLU member and may contribute only receiver/day
  self-supervision and VICReg.
* source-validation is represented only by a sealed row-count/content digest;
  no source-validation feature can enter this trainer.

The implementation follows the singleton constants frozen before the D103-R1
formal run.  A single mechanical step is exposed for focused synthetic and
real-checkpoint smoke tests; it computes no performance metric.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

CANDIDATE_ID = "D103-R2-RXID-CROSSRECEIVER-MB4"
SCHEMA = "cvs.d103_r2.rxid_metabias4.phase1_trainer.v1"

FEATURE_DIM = 160
DOMAIN_DIM = 32
CODE_DIM = 4
FINAL_TX_NULL_RANK = 5
K_VALUES = (1, 5, 10)
MMD_GAMMAS = (0.5, 1.0, 2.0)
QUERY_PER_CLASS = 4
SAMPLES_PER_CELL = 2
SEED = 103713

_EPS = 1.0e-12
_LS_KEYS = frozenset(
    {
        "z_dom",
        "pre_relu",
        "receiver_ids",
        "day_ids",
        "tx_labels",
        "physical_ids",
    }
)
_US_KEYS = frozenset({"z_dom", "receiver_ids", "day_ids", "physical_ids"})
_SOURCE_VAL_KEYS = frozenset({"row_count", "content_sha256"})


class D103R1TrainingError(ValueError):
    """Raised when the frozen training or permission contract is violated."""


class SplitRole(str, Enum):
    """Phase1 split roles; these names are part of the permission receipt."""

    LABELED_SOURCE = "L_s"
    UNLABELED_SOURCE = "U_s"
    SOURCE_VALIDATION = "source_val"


class Operation(str, Enum):
    """Audited operations reachable from this source-only trainer."""

    FOLD_MASK = "fold_mask"
    TX_PROJECTOR = "tx_projector"
    TX_MMD = "tx_mmd"
    CLASS_BALANCED_BANK = "class_balanced_bank"
    METABIAS_META = "metabias_meta"
    RX_SELF_SUPERVISION = "receiver_day_self_supervision"
    VICREG = "vicreg"
    FINAL_AGGREGATION = "final_teacher_aggregation"


@dataclass(frozen=True, slots=True)
class D103R1Config:
    """The pre-registered singleton training configuration.

    ``__post_init__`` rejects every drift instead of silently constructing a
    new candidate under the D103-R1 name.
    """

    seed: int = SEED
    learning_rate: float = 1.0e-3
    epochs: int = 20
    meta_steps_per_epoch: int = 20
    k_values: tuple[int, ...] = K_VALUES
    query_per_class: int = QUERY_PER_CLASS
    samples_per_cell: int = SAMPLES_PER_CELL
    mu: float = 0.1
    tau: float = 0.1
    lambda_tx: float = 1.0
    lambda_rx: float = 1.0
    lambda_vicreg: float = 1.0
    lambda_orthogonal: float = 1.0
    mmd_gammas: tuple[float, ...] = MMD_GAMMAS
    qknn_training_temperature: float = 0.2
    bank_temperature: float = 0.25
    lambda0: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    a_max: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    ellipsoid_radius: float = 0.35009765625
    labeled_fraction: float = 0.07
    unlabeled_fraction: float = 0.63
    source_val_fraction: float = 0.30
    performance_selection: bool = False
    early_stopping: bool = False
    outer_results_read: bool = False

    def __post_init__(self) -> None:
        expected: dict[str, Any] = {
            "seed": SEED,
            "learning_rate": 1.0e-3,
            "epochs": 20,
            "meta_steps_per_epoch": 20,
            "k_values": K_VALUES,
            "query_per_class": QUERY_PER_CLASS,
            "samples_per_cell": SAMPLES_PER_CELL,
            "mu": 0.1,
            "tau": 0.1,
            "lambda_tx": 1.0,
            "lambda_rx": 1.0,
            "lambda_vicreg": 1.0,
            "lambda_orthogonal": 1.0,
            "mmd_gammas": MMD_GAMMAS,
            "qknn_training_temperature": 0.2,
            "bank_temperature": 0.25,
            "lambda0": (1.0, 1.0, 1.0, 1.0),
            "a_max": (0.25, 0.25, 0.25, 0.25),
            "ellipsoid_radius": 0.35009765625,
            "labeled_fraction": 0.07,
            "unlabeled_fraction": 0.63,
            "source_val_fraction": 0.30,
            "performance_selection": False,
            "early_stopping": False,
            "outer_results_read": False,
        }
        for name, frozen in expected.items():
            if getattr(self, name) != frozen:
                raise D103R1TrainingError(
                    f"singleton configuration drift: {name}={getattr(self, name)!r}, "
                    f"expected={frozen!r}"
                )
        if not math.isclose(
            self.labeled_fraction
            + self.unlabeled_fraction
            + self.source_val_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise D103R1TrainingError("Phase1 split fractions must sum to one")

    @property
    def total_meta_steps(self) -> int:
        return self.epochs * self.meta_steps_per_epoch


def _readonly_float_rows(value: Any, name: str) -> np.ndarray:
    rows = np.array(value, dtype=np.float32, copy=True, order="C")
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise D103R1TrainingError(f"{name} must be finite [N,{FEATURE_DIM}]")
    rows.setflags(write=False)
    return rows


def _readonly_text_vector(value: Any, rows: int, name: str) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1 or vector.shape[0] != rows or vector.dtype.kind == "O":
        raise D103R1TrainingError(f"{name} must be a non-object text [N] array")
    result = np.array(vector.astype(str), copy=True, order="C")
    if np.any(np.char.str_len(result) == 0):
        raise D103R1TrainingError(f"{name} contains an empty identifier")
    result.setflags(write=False)
    return result


def _require_unique_physical_ids(values: np.ndarray, name: str) -> None:
    if np.unique(values).size != values.size:
        raise D103R1TrainingError(f"{name} physical IDs must be unique")


@dataclass(frozen=True, slots=True)
class LabeledSourceRows:
    """The only Phase1 rows allowed to carry TX labels and pre-ReLU values."""

    z_dom: np.ndarray
    pre_relu: np.ndarray
    receiver_ids: np.ndarray
    day_ids: np.ndarray
    tx_labels: np.ndarray
    physical_ids: np.ndarray

    def __post_init__(self) -> None:
        z_dom = _readonly_float_rows(self.z_dom, "L_s z_dom")
        pre_relu = _readonly_float_rows(self.pre_relu, "L_s pre_relu")
        if pre_relu.shape != z_dom.shape:
            raise D103R1TrainingError("L_s z_dom/pre_relu shape drift")
        row_count = z_dom.shape[0]
        receiver_ids = _readonly_text_vector(
            self.receiver_ids, row_count, "L_s receiver_ids"
        )
        day_ids = _readonly_text_vector(self.day_ids, row_count, "L_s day_ids")
        tx_labels = _readonly_text_vector(self.tx_labels, row_count, "L_s tx_labels")
        physical_ids = _readonly_text_vector(
            self.physical_ids, row_count, "L_s physical_ids"
        )
        _require_unique_physical_ids(physical_ids, "L_s")
        object.__setattr__(self, "z_dom", z_dom)
        object.__setattr__(self, "pre_relu", pre_relu)
        object.__setattr__(self, "receiver_ids", receiver_ids)
        object.__setattr__(self, "day_ids", day_ids)
        object.__setattr__(self, "tx_labels", tx_labels)
        object.__setattr__(self, "physical_ids", physical_ids)

    @property
    def row_count(self) -> int:
        return int(self.z_dom.shape[0])


@dataclass(frozen=True, slots=True)
class UnlabeledSourceRows:
    """TX-blind ``U_s`` rows.

    This type deliberately has no ``tx_labels`` or ``pre_relu`` field, making
    TX-MMD, MetaBias episodes and bank construction structurally unreachable.
    """

    z_dom: np.ndarray
    receiver_ids: np.ndarray
    day_ids: np.ndarray
    physical_ids: np.ndarray

    def __post_init__(self) -> None:
        z_dom = _readonly_float_rows(self.z_dom, "U_s z_dom")
        row_count = z_dom.shape[0]
        receiver_ids = _readonly_text_vector(
            self.receiver_ids, row_count, "U_s receiver_ids"
        )
        day_ids = _readonly_text_vector(self.day_ids, row_count, "U_s day_ids")
        physical_ids = _readonly_text_vector(
            self.physical_ids, row_count, "U_s physical_ids"
        )
        _require_unique_physical_ids(physical_ids, "U_s")
        object.__setattr__(self, "z_dom", z_dom)
        object.__setattr__(self, "receiver_ids", receiver_ids)
        object.__setattr__(self, "day_ids", day_ids)
        object.__setattr__(self, "physical_ids", physical_ids)

    @property
    def row_count(self) -> int:
        return int(self.z_dom.shape[0])


def _require_sha256(value: str, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise D103R1TrainingError(f"{name} must be a lowercase SHA256")
    return digest


@dataclass(frozen=True, slots=True)
class SourceValidationSeal:
    """Non-readable source-validation descriptor.

    The formal arrays stay with the independent falsifier.  The trainer only
    binds their externally produced content identity and row count.
    """

    row_count: int
    content_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.row_count, bool) or int(self.row_count) <= 0:
            raise D103R1TrainingError("source-val row_count must be positive")
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(
            self,
            "content_sha256",
            _require_sha256(self.content_sha256, "source-val content_sha256"),
        )


@dataclass(frozen=True, slots=True)
class D103R1TrainingData:
    labeled: LabeledSourceRows
    unlabeled: UnlabeledSourceRows
    source_val: SourceValidationSeal
    config: D103R1Config = D103R1Config()

    def __post_init__(self) -> None:
        if type(self.labeled) is not LabeledSourceRows:
            raise D103R1TrainingError("training data requires exact L_s type")
        if type(self.unlabeled) is not UnlabeledSourceRows:
            raise D103R1TrainingError("training data requires exact U_s type")
        if type(self.source_val) is not SourceValidationSeal:
            raise D103R1TrainingError("training data requires sealed source-val descriptor")
        if type(self.config) is not D103R1Config:
            raise D103R1TrainingError("training data requires singleton config")
        overlap = np.intersect1d(
            self.labeled.physical_ids, self.unlabeled.physical_ids
        )
        if overlap.size:
            raise D103R1TrainingError("L_s/U_s physical IDs overlap")


def _require_exact_members(
    payload: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    if type(payload) not in (dict, MappingProxyType):
        raise D103R1TrainingError(f"{name} payload must be an exact mapping")
    actual = frozenset(str(key) for key in payload)
    if actual != expected:
        missing = sorted(expected - actual)
        forbidden = sorted(actual - expected)
        raise D103R1TrainingError(
            f"{name} member closure drift: missing={missing}, forbidden={forbidden}"
        )


def build_training_data(
    labeled_payload: Mapping[str, Any],
    unlabeled_payload: Mapping[str, Any],
    source_val_payload: Mapping[str, Any],
    *,
    config: D103R1Config = D103R1Config(),
) -> D103R1TrainingData:
    """Build role-typed data from exact allowlisted mappings."""

    _require_exact_members(labeled_payload, _LS_KEYS, "L_s")
    _require_exact_members(unlabeled_payload, _US_KEYS, "U_s")
    _require_exact_members(source_val_payload, _SOURCE_VAL_KEYS, "source-val")
    return D103R1TrainingData(
        labeled=LabeledSourceRows(**dict(labeled_payload)),
        unlabeled=UnlabeledSourceRows(**dict(unlabeled_payload)),
        source_val=SourceValidationSeal(**dict(source_val_payload)),
        config=config,
    )


@dataclass(frozen=True, slots=True)
class AccessEvent:
    ordinal: int
    role: str
    operation: str
    fields: tuple[str, ...]
    row_count: int


_PERMISSIONS: Mapping[tuple[SplitRole, Operation], frozenset[str]] = (
    MappingProxyType(
        {
            (SplitRole.LABELED_SOURCE, Operation.FOLD_MASK): frozenset(
                {"receiver_ids", "day_ids", "tx_labels"}
            ),
            (SplitRole.UNLABELED_SOURCE, Operation.FOLD_MASK): frozenset(
                {"receiver_ids", "day_ids"}
            ),
            (SplitRole.LABELED_SOURCE, Operation.TX_PROJECTOR): frozenset(
                {"z_dom", "receiver_ids", "day_ids", "tx_labels", "physical_ids"}
            ),
            (SplitRole.LABELED_SOURCE, Operation.TX_MMD): frozenset(
                {"z_dom", "receiver_ids", "day_ids", "tx_labels", "physical_ids"}
            ),
            (SplitRole.LABELED_SOURCE, Operation.CLASS_BALANCED_BANK): frozenset(
                {"z_dom", "receiver_ids", "day_ids", "tx_labels", "physical_ids"}
            ),
            (SplitRole.LABELED_SOURCE, Operation.METABIAS_META): frozenset(
                {
                    "z_dom",
                    "pre_relu",
                    "receiver_ids",
                    "day_ids",
                    "tx_labels",
                    "physical_ids",
                }
            ),
            (SplitRole.LABELED_SOURCE, Operation.RX_SELF_SUPERVISION): frozenset(
                {
                    "z_dom",
                    "receiver_ids",
                    "day_ids",
                    "tx_labels",
                    "physical_ids",
                }
            ),
            (SplitRole.UNLABELED_SOURCE, Operation.RX_SELF_SUPERVISION): frozenset(
                {"z_dom", "receiver_ids", "day_ids", "physical_ids"}
            ),
            (SplitRole.LABELED_SOURCE, Operation.VICREG): frozenset({"z_dom"}),
            (SplitRole.UNLABELED_SOURCE, Operation.VICREG): frozenset({"z_dom"}),
            (SplitRole.LABELED_SOURCE, Operation.FINAL_AGGREGATION): frozenset(
                {
                    "z_dom",
                    "receiver_ids",
                    "day_ids",
                    "tx_labels",
                    "physical_ids",
                }
            ),
        }
    )
)


class PermissionLedger:
    """Fail-closed, append-only in-memory access ledger."""

    def __init__(self) -> None:
        self._events: list[AccessEvent] = []
        self._denied_attempts = 0

    @property
    def events(self) -> tuple[AccessEvent, ...]:
        return tuple(self._events)

    @property
    def denied_attempts(self) -> int:
        return self._denied_attempts

    def authorize(
        self,
        role: SplitRole,
        operation: Operation,
        fields: Sequence[str],
        row_count: int,
    ) -> None:
        if type(role) is not SplitRole or type(operation) is not Operation:
            self._denied_attempts += 1
            raise D103R1TrainingError("permission role/operation must use exact enums")
        allowed = _PERMISSIONS.get((role, operation))
        requested = frozenset(str(field) for field in fields)
        if allowed is None or not requested.issubset(allowed):
            self._denied_attempts += 1
            raise D103R1TrainingError(
                f"permission denied: role={role.value}, operation={operation.value}, "
                f"fields={sorted(requested)}"
            )
        if isinstance(row_count, bool) or int(row_count) < 0:
            self._denied_attempts += 1
            raise D103R1TrainingError("ledger row_count must be non-negative")
        self._events.append(
            AccessEvent(
                ordinal=len(self._events),
                role=role.value,
                operation=operation.value,
                fields=tuple(sorted(requested)),
                row_count=int(row_count),
            )
        )

    def receipt(self) -> Mapping[str, Any]:
        payload = {
            "schema": f"{SCHEMA}.access_ledger",
            "candidate_id": CANDIDATE_ID,
            "events": [
                {
                    "ordinal": event.ordinal,
                    "role": event.role,
                    "operation": event.operation,
                    "fields": list(event.fields),
                    "row_count": event.row_count,
                }
                for event in self._events
            ],
            "denied_attempts": self._denied_attempts,
            "source_val_array_access": False,
            "target_access": False,
            "formal_query_access": False,
            "performance_selection_access": False,
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return MappingProxyType(
            {**payload, "receipt_sha256": hashlib.sha256(canonical).hexdigest()}
        )


@dataclass(frozen=True, slots=True)
class OuterMaskSpec:
    """One final, receiver-held, day-held, or receiver×class LOCO fit."""

    held_receiver: str | None = None
    held_day: str | None = None
    held_class: str | None = None

    def __post_init__(self) -> None:
        for name in ("held_receiver", "held_day", "held_class"):
            value = getattr(self, name)
            if value is not None and not str(value):
                raise D103R1TrainingError(f"{name} must be non-empty or None")


@dataclass(frozen=True, slots=True)
class FoldMasks:
    labeled_train: np.ndarray
    unlabeled_train: np.ndarray
    spec: OuterMaskSpec
    hidden_tx_unlabeled_policy: str

    def __post_init__(self) -> None:
        labeled = np.array(self.labeled_train, dtype=np.bool_, copy=True)
        unlabeled = np.array(self.unlabeled_train, dtype=np.bool_, copy=True)
        if labeled.ndim != 1 or unlabeled.ndim != 1:
            raise D103R1TrainingError("fold masks must be one-dimensional")
        labeled.setflags(write=False)
        unlabeled.setflags(write=False)
        object.__setattr__(self, "labeled_train", labeled)
        object.__setattr__(self, "unlabeled_train", unlabeled)


def build_outer_masks(
    data: D103R1TrainingData,
    spec: OuterMaskSpec,
    ledger: PermissionLedger,
) -> FoldMasks:
    """Build masks without reading source-val or joining TX into ``U_s``.

    In a class-LOCO fit, hidden-TX ``U_s`` cannot be filtered by class without a
    forbidden join.  The only fail-closed implementation is therefore to omit
    all ``U_s`` rows from that fit.
    """

    if type(data) is not D103R1TrainingData or type(spec) is not OuterMaskSpec:
        raise D103R1TrainingError("outer-mask builder requires exact typed inputs")
    if type(ledger) is not PermissionLedger:
        raise D103R1TrainingError("outer-mask builder requires exact permission ledger")
    ledger.authorize(
        SplitRole.LABELED_SOURCE,
        Operation.FOLD_MASK,
        ("receiver_ids", "day_ids", "tx_labels"),
        data.labeled.row_count,
    )
    ledger.authorize(
        SplitRole.UNLABELED_SOURCE,
        Operation.FOLD_MASK,
        ("receiver_ids", "day_ids"),
        data.unlabeled.row_count,
    )
    ls_mask = np.ones(data.labeled.row_count, dtype=np.bool_)
    us_mask = np.ones(data.unlabeled.row_count, dtype=np.bool_)
    if spec.held_receiver is not None:
        if spec.held_receiver not in set(data.labeled.receiver_ids.tolist()):
            raise D103R1TrainingError("held receiver is absent from L_s")
        ls_mask &= data.labeled.receiver_ids != spec.held_receiver
        us_mask &= data.unlabeled.receiver_ids != spec.held_receiver
    if spec.held_day is not None:
        if spec.held_day not in set(data.labeled.day_ids.tolist()):
            raise D103R1TrainingError("held day is absent from L_s")
        ls_mask &= data.labeled.day_ids != spec.held_day
        us_mask &= data.unlabeled.day_ids != spec.held_day
    hidden_policy = "all_eligible_U_s"
    if spec.held_class is not None:
        if spec.held_class not in set(data.labeled.tx_labels.tolist()):
            raise D103R1TrainingError("held class is absent from L_s")
        ls_mask &= data.labeled.tx_labels != spec.held_class
        us_mask[:] = False
        hidden_policy = "exclude_all_U_s_in_class_LOCO_no_TX_join"
    if not np.any(ls_mask):
        raise D103R1TrainingError("outer masks leave no L_s training rows")
    if np.unique(data.labeled.tx_labels[ls_mask]).size < 2:
        raise D103R1TrainingError("outer masks leave fewer than two L_s classes")
    return FoldMasks(ls_mask, us_mask, spec, hidden_policy)


def _stable_order(
    physical_ids: np.ndarray, *, purpose: str, step_index: int
) -> np.ndarray:
    keys = []
    for physical_id in physical_ids.astype(str).tolist():
        value = (
            f"{CANDIDATE_ID}|{SEED}|{step_index}|{purpose}|{physical_id}"
        ).encode("utf-8")
        keys.append(hashlib.sha256(value).digest())
    return np.argsort(np.asarray(keys, dtype="S32"), kind="stable")


def _balanced_labeled_indices(
    rows: LabeledSourceRows,
    mask: np.ndarray,
    *,
    step_index: int,
    samples_per_cell: int,
) -> tuple[np.ndarray, tuple[tuple[str, str], ...], tuple[str, ...]]:
    receivers = tuple(sorted(np.unique(rows.receiver_ids[mask]).tolist()))
    days = tuple(sorted(np.unique(rows.day_ids[mask]).tolist()))
    classes = tuple(sorted(np.unique(rows.tx_labels[mask]).tolist()))
    selected: list[int] = []
    cell_keys: list[tuple[str, str]] = []
    for receiver in receivers:
        for day in days:
            cell_present = False
            for label in classes:
                candidates = np.flatnonzero(
                    mask
                    & (rows.receiver_ids == receiver)
                    & (rows.day_ids == day)
                    & (rows.tx_labels == label)
                )
                if candidates.size == 0:
                    continue
                cell_present = True
                if candidates.size < samples_per_cell:
                    raise D103R1TrainingError(
                        "L_s receiver×day×TX cell lacks frozen sample count: "
                        f"receiver={receiver}, day={day}, class={label}"
                    )
                order = _stable_order(
                    rows.physical_ids[candidates],
                    purpose=f"balanced_L_s|{receiver}|{day}|{label}",
                    step_index=step_index,
                )
                selected.extend(candidates[order[:samples_per_cell]].tolist())
            if cell_present:
                # Every retained bank cell must contain every retained class.
                per_class = [
                    np.any(
                        mask
                        & (rows.receiver_ids == receiver)
                        & (rows.day_ids == day)
                        & (rows.tx_labels == label)
                    )
                    for label in classes
                ]
                if not all(per_class):
                    raise D103R1TrainingError(
                        "class-balanced bank cell is missing a retained class"
                    )
                cell_keys.append((receiver, day))
    result = np.asarray(selected, dtype=np.int64)
    if result.size == 0 or np.unique(rows.physical_ids[result]).size != result.size:
        raise D103R1TrainingError("balanced L_s batch is empty or reuses a physical ID")
    return result, tuple(cell_keys), classes


def _balanced_unlabeled_indices(
    rows: UnlabeledSourceRows,
    mask: np.ndarray,
    *,
    step_index: int,
    samples_per_cell: int,
) -> np.ndarray:
    if not np.any(mask):
        return np.empty(0, dtype=np.int64)
    selected: list[int] = []
    receivers = sorted(np.unique(rows.receiver_ids[mask]).tolist())
    days = sorted(np.unique(rows.day_ids[mask]).tolist())
    for receiver in receivers:
        for day in days:
            candidates = np.flatnonzero(
                mask
                & (rows.receiver_ids == receiver)
                & (rows.day_ids == day)
            )
            if candidates.size == 0:
                continue
            if candidates.size < samples_per_cell:
                raise D103R1TrainingError(
                    "U_s receiver×day cell lacks frozen sample count"
                )
            order = _stable_order(
                rows.physical_ids[candidates],
                purpose=f"balanced_U_s|{receiver}|{day}",
                step_index=step_index,
            )
            selected.extend(candidates[order[:samples_per_cell]].tolist())
    result = np.asarray(selected, dtype=np.int64)
    if result.size and np.unique(rows.physical_ids[result]).size != result.size:
        raise D103R1TrainingError("balanced U_s batch reuses a physical ID")
    return result


def build_tx_projector(
    data: D103R1TrainingData,
    masks: FoldMasks,
    ledger: PermissionLedger,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Build the labeled-only TX mean nullspace for one fit."""

    ledger.authorize(
        SplitRole.LABELED_SOURCE,
        Operation.TX_PROJECTOR,
        ("z_dom", "receiver_ids", "day_ids", "tx_labels", "physical_ids"),
        int(np.sum(masks.labeled_train)),
    )
    indices, _, classes = _balanced_labeled_indices(
        data.labeled,
        masks.labeled_train,
        step_index=0,
        samples_per_cell=data.config.samples_per_cell,
    )
    features = data.labeled.z_dom[indices].astype(np.float64)
    labels = data.labeled.tx_labels[indices]
    receivers = data.labeled.receiver_ids[indices]
    days = data.labeled.day_ids[indices]
    tx_means: list[np.ndarray] = []
    for label in classes:
        cell_means: list[np.ndarray] = []
        for receiver in sorted(np.unique(receivers).tolist()):
            for day in sorted(np.unique(days).tolist()):
                local = (
                    (labels == label)
                    & (receivers == receiver)
                    & (days == day)
                )
                if np.any(local):
                    cell_means.append(features[local].mean(axis=0))
        if not cell_means:
            raise D103R1TrainingError(f"TX {label} has no balanced labeled cells")
        tx_means.append(np.mean(cell_means, axis=0))
    centered = np.stack(tx_means) - np.mean(tx_means, axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(float(singular[0]), 1.0) * 1.0e-10
    rank = int(np.sum(singular > tolerance))
    expected_rank = len(classes) - 1
    if rank != expected_rank:
        raise D103R1TrainingError(
            f"TX mean rank drift: expected={expected_rank}, actual={rank}"
        )
    if masks.spec.held_class is None and rank != FINAL_TX_NULL_RANK:
        raise D103R1TrainingError(
            f"full-class TX null rank must be {FINAL_TX_NULL_RANK}, got {rank}"
        )
    if rank > FINAL_TX_NULL_RANK:
        raise D103R1TrainingError("TX null rank exceeds frozen full-class rank")
    basis = vh[:rank]
    projector = np.eye(FEATURE_DIM, dtype=np.float64) - basis.T @ basis
    residual = float(np.linalg.norm(projector @ basis.T))
    if residual > 1.0e-8 or not np.isfinite(projector).all():
        raise D103R1TrainingError(
            f"TX nullspace residual too large: {residual}"
        )
    result = np.asarray(projector, dtype=np.float32)
    result.setflags(write=False)
    receipt = MappingProxyType(
        {
            "tx_class_count": len(classes),
            "tx_null_rank": rank,
            "expected_rank": expected_rank,
            "null_residual": residual,
            "labeled_rows_used": int(indices.size),
            "unlabeled_rows_used": 0,
            "source_val_rows_used": 0,
        }
    )
    return result, receipt


@dataclass(frozen=True, slots=True)
class EpisodeIndices:
    k_shot: int
    support_receiver: str
    query_receiver: str
    support: np.ndarray
    query: np.ndarray
    classes: tuple[str, ...]


def _build_episode(
    rows: LabeledSourceRows,
    mask: np.ndarray,
    *,
    k_shot: int,
    support_receiver: str,
    query_receiver: str,
    query_per_class: int,
    step_index: int,
) -> EpisodeIndices:
    if support_receiver == query_receiver:
        raise D103R1TrainingError(
            "cross-receiver episode support/query receiver must differ"
        )
    classes = tuple(sorted(np.unique(rows.tx_labels[mask]).tolist()))
    support: list[int] = []
    query: list[int] = []
    for label in classes:
        support_candidates = np.flatnonzero(
            mask
            & (rows.receiver_ids == support_receiver)
            & (rows.tx_labels == label)
        )
        query_candidates = np.flatnonzero(
            mask
            & (rows.receiver_ids == query_receiver)
            & (rows.tx_labels == label)
        )
        if support_candidates.size < k_shot:
            raise D103R1TrainingError(
                f"episode needs K{k_shot} support rows: "
                f"receiver={support_receiver}, class={label}, "
                f"found={support_candidates.size}"
            )
        if query_candidates.size < query_per_class:
            raise D103R1TrainingError(
                f"episode needs {query_per_class} query rows: "
                f"receiver={query_receiver}, class={label}, "
                f"found={query_candidates.size}"
            )
        support_order = _stable_order(
            rows.physical_ids[support_candidates],
            purpose=(
                f"episode|support|{support_receiver}|query|{query_receiver}|"
                f"K{k_shot}|{label}"
            ),
            step_index=step_index,
        )
        query_order = _stable_order(
            rows.physical_ids[query_candidates],
            purpose=(
                f"episode|query|{query_receiver}|support|{support_receiver}|"
                f"K{k_shot}|{label}"
            ),
            step_index=step_index,
        )
        support.extend(support_candidates[support_order[:k_shot]].tolist())
        query.extend(query_candidates[query_order[:query_per_class]].tolist())
    support_array = np.asarray(support, dtype=np.int64)
    query_array = np.asarray(query, dtype=np.int64)
    if np.intersect1d(
        rows.physical_ids[support_array], rows.physical_ids[query_array]
    ).size:
        raise D103R1TrainingError("episode support/query physical IDs overlap")
    support_array.setflags(write=False)
    query_array.setflags(write=False)
    return EpisodeIndices(
        k_shot,
        support_receiver,
        query_receiver,
        support_array,
        query_array,
        classes,
    )


class RXIDMetaBias4Model(torch.nn.Module):
    """Trainable Phase1 state; no source rows or validation state are retained."""

    def __init__(
        self,
        bank_cell_count: int,
        projector: np.ndarray,
        *,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if bank_cell_count < 2:
            raise D103R1TrainingError("class-free bank requires at least two cells")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.raw_w = torch.nn.Parameter(
            (torch.randn(DOMAIN_DIM, FEATURE_DIM, generator=generator) * 0.02).to(
                device
            )
        )
        self.basis = torch.nn.Parameter(
            (torch.randn(FEATURE_DIM, CODE_DIM, generator=generator) * 0.01).to(
                device
            )
        )
        self.bank_t = torch.nn.Parameter(
            (torch.randn(bank_cell_count, CODE_DIM, generator=generator) * 0.01).to(
                device
            )
        )
        self.log_precision = torch.nn.Parameter(
            torch.zeros(bank_cell_count, CODE_DIM, device=device)
        )
        self.log_sigma = torch.nn.Parameter(
            torch.zeros(bank_cell_count, device=device)
        )
        projector_tensor = torch.as_tensor(
            np.array(projector, dtype=np.float32, copy=True), device=device
        )
        self.register_buffer("tx_projector", projector_tensor, persistent=False)

    def encoder(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(
            (self.raw_w @ self.tx_projector).T, mode="reduced"
        )
        return q.T


def _mmd_loss(
    encoded: torch.Tensor, labels: torch.Tensor, class_count: int, gammas: Sequence[float]
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for left in range(class_count):
        x = encoded[labels == left]
        if x.shape[0] < 2:
            raise D103R1TrainingError("TX-MMD requires at least two rows per class")
        for right in range(left + 1, class_count):
            y = encoded[labels == right]
            if y.shape[0] < 2:
                raise D103R1TrainingError("TX-MMD requires at least two rows per class")
            xx = torch.cdist(x, x).square()
            yy = torch.cdist(y, y).square()
            xy = torch.cdist(x, y).square()
            for gamma in gammas:
                losses.append(
                    torch.exp(-gamma * xx).mean()
                    + torch.exp(-gamma * yy).mean()
                    - 2.0 * torch.exp(-gamma * xy).mean()
                )
    if not losses:
        raise D103R1TrainingError("TX-MMD has no class pair")
    return torch.stack(losses).mean()


def _receiver_nce(
    encoded: torch.Tensor,
    receiver: torch.Tensor,
    day: torch.Tensor,
    tx: torch.Tensor | None,
) -> torch.Tensor:
    same_receiver = receiver[:, None] == receiver[None, :]
    cross_day = day[:, None] != day[None, :]
    positive = same_receiver & cross_day
    if tx is not None:
        positive &= tx[:, None] != tx[None, :]
    negative = receiver[:, None] != receiver[None, :]
    allowed = positive | negative
    allowed.fill_diagonal_(False)
    if not torch.all(positive.any(dim=1)):
        raise D103R1TrainingError("receiver self-supervision lacks cross-day positives")
    if not torch.all(negative.any(dim=1)):
        raise D103R1TrainingError("receiver self-supervision lacks receiver negatives")
    logits = encoded @ encoded.T / 0.1
    denominator = torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    loss = (denominator - numerator).mean()
    if not torch.isfinite(loss):
        raise D103R1TrainingError("receiver self-supervision became non-finite")
    return loss


def _vicreg_loss(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.shape[0] < 2:
        raise D103R1TrainingError("VICReg requires at least two rows")
    std = torch.sqrt(encoded.var(dim=0, unbiased=True) + 1.0e-4)
    variance = functional.relu(0.05 - std).mean()
    centered = encoded - encoded.mean(dim=0)
    covariance = centered.T @ centered / (encoded.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return variance + off_diagonal.square().mean()


def _class_index(values: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    try:
        return np.asarray([lookup[str(value)] for value in values], dtype=np.int64)
    except KeyError as error:
        raise D103R1TrainingError("episode label is absent from registry") from error


def _qknn_logits(
    query: torch.Tensor,
    support: torch.Tensor,
    support_y: torch.Tensor,
    class_count: int,
    temperature: float,
) -> torch.Tensor:
    similarity = query @ support.T
    logits: list[torch.Tensor] = []
    for label_index in range(class_count):
        local = similarity[:, support_y == label_index]
        if local.shape[1] == 0:
            raise D103R1TrainingError("qKNN support misses a registered class")
        logits.append(
            torch.logsumexp(local / temperature, dim=1)
            - math.log(local.shape[1])
        )
    return torch.stack(logits, dim=1)


def _class_balanced_meta_loss(
    da_logits: torch.Tensor,
    base_logits: torch.Tensor,
    query_y: torch.Tensor,
    class_count: int,
    config: D103R1Config,
) -> torch.Tensor:
    da_rows = functional.cross_entropy(da_logits, query_y, reduction="none")
    base_rows = functional.cross_entropy(
        base_logits, query_y, reduction="none"
    ).detach()
    da_class: list[torch.Tensor] = []
    base_class: list[torch.Tensor] = []
    for label_index in range(class_count):
        local = query_y == label_index
        if not torch.any(local):
            raise D103R1TrainingError("meta query misses a registered class")
        da_class.append(da_rows[local].mean())
        base_class.append(base_rows[local].mean())
    da = torch.stack(da_class)
    base = torch.stack(base_class)
    log_mean_exp = torch.logsumexp((da - base) / config.tau, dim=0) - math.log(
        class_count
    )
    return da.mean() + config.mu * config.tau * log_mean_exp


@dataclass(frozen=True, slots=True)
class MechanicalStepReceipt:
    schema: str
    candidate_id: str
    step_index: int
    total_loss: float
    meta_loss: float
    tx_mmd_loss: float
    receiver_loss: float
    vicreg_loss: float
    orthogonality_loss: float
    k_values: tuple[int, ...]
    episode_support_receiver: str
    episode_query_receiver: str
    labeled_batch_rows: int
    unlabeled_batch_rows: int
    performance_metrics_computed: bool
    source_val_rows_used: int
    target_access: bool
    formal_query_access: bool
    optimizer_step_completed: bool
    ledger_receipt_sha256: str


def _readonly_export_array(
    value: Any, shape: tuple[int | None, ...], name: str
) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True, order="C")
    if result.ndim != len(shape) or not np.isfinite(result).all():
        raise D103R1TrainingError(f"{name} must be finite with rank {len(shape)}")
    for actual, expected in zip(result.shape, shape):
        if expected is not None and actual != expected:
            raise D103R1TrainingError(
                f"{name} shape drift: actual={result.shape}, expected={shape}"
            )
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class D103R1TeacherState:
    """Final full-precision teacher arrays before the separate INT8 bundle step."""

    u: np.ndarray
    b: np.ndarray
    bank_g: np.ndarray
    bank_t: np.ndarray
    bank_precision: np.ndarray
    bank_sigma: np.ndarray
    aggregation_receipt: Mapping[str, Any]
    access_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        u = _readonly_export_array(self.u, (DOMAIN_DIM, FEATURE_DIM), "teacher U")
        b = _readonly_export_array(self.b, (FEATURE_DIM, CODE_DIM), "teacher B")
        bank_g = _readonly_export_array(
            self.bank_g, (None, DOMAIN_DIM), "teacher bank_g"
        )
        bank_t = _readonly_export_array(
            self.bank_t, (bank_g.shape[0], CODE_DIM), "teacher bank_t"
        )
        precision = _readonly_export_array(
            self.bank_precision,
            (bank_g.shape[0], CODE_DIM),
            "teacher bank_precision",
        )
        sigma = _readonly_export_array(
            self.bank_sigma, (bank_g.shape[0],), "teacher bank_sigma"
        )
        if np.any(precision <= 0.0) or np.any(sigma <= 0.0):
            raise D103R1TrainingError(
                "teacher precision/sigma must remain strictly positive"
            )
        if not isinstance(self.aggregation_receipt, Mapping) or not isinstance(
            self.access_receipt, Mapping
        ):
            raise D103R1TrainingError("teacher receipts must be mappings")
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "bank_g", bank_g)
        object.__setattr__(self, "bank_t", bank_t)
        object.__setattr__(self, "bank_precision", precision)
        object.__setattr__(self, "bank_sigma", sigma)
        object.__setattr__(
            self,
            "aggregation_receipt",
            MappingProxyType(dict(self.aggregation_receipt)),
        )
        object.__setattr__(
            self, "access_receipt", MappingProxyType(dict(self.access_receipt))
        )


def export_teacher_arrays(state: D103R1TeacherState) -> Mapping[str, Any]:
    """Return the exact allowlisted teacher payload.

    This is not the deployment bundle: a separate component must quantize and
    seal these arrays.  No optimizer, row identifiers, split arrays, TX/class
    registry, source-validation content, or ``U_s`` metadata is exported.
    """

    if type(state) is not D103R1TeacherState:
        raise D103R1TrainingError(
            "teacher export requires exact completed D103R1TeacherState"
        )
    return MappingProxyType(
        {
            "U": state.u,
            "B": state.b,
            "bank_g": state.bank_g,
            "bank_t": state.bank_t,
            "bank_precision": state.bank_precision,
            "bank_sigma": state.bank_sigma,
            "aggregation_receipt": state.aggregation_receipt,
            "access_receipt": state.access_receipt,
        }
    )


class D103R1Phase1Trainer:
    """One immutable-fit D103-R1 trainer with an exact 400-step schedule."""

    def __init__(
        self,
        data: D103R1TrainingData,
        spec: OuterMaskSpec = OuterMaskSpec(),
        *,
        device: str | torch.device = "cpu",
        ledger: PermissionLedger | None = None,
    ) -> None:
        if type(data) is not D103R1TrainingData:
            raise D103R1TrainingError("trainer requires exact D103R1TrainingData")
        if type(spec) is not OuterMaskSpec:
            raise D103R1TrainingError("trainer requires exact OuterMaskSpec")
        self.data = data
        self.config = data.config
        self.ledger = ledger if ledger is not None else PermissionLedger()
        if type(self.ledger) is not PermissionLedger:
            raise D103R1TrainingError("trainer requires exact PermissionLedger")
        self.masks = build_outer_masks(data, spec, self.ledger)
        projector, projector_receipt = build_tx_projector(
            data, self.masks, self.ledger
        )
        self.projector_receipt = projector_receipt
        _, cell_keys, classes = _balanced_labeled_indices(
            data.labeled,
            self.masks.labeled_train,
            step_index=0,
            samples_per_cell=self.config.samples_per_cell,
        )
        self.cell_keys = cell_keys
        self.classes = classes
        self.device = torch.device(device)
        self.model = RXIDMetaBias4Model(
            len(cell_keys),
            projector,
            seed=self.config.seed,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.completed_steps = 0
        self._final_state_cache: D103R1TeacherState | None = None

    def _tensors(
        self, values: np.ndarray, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def _bank_g(
        self,
        encoded: torch.Tensor,
        batch_indices: np.ndarray,
    ) -> torch.Tensor:
        rows = self.data.labeled
        labels = _class_index(rows.tx_labels[batch_indices], self.classes)
        label_tensor = self._tensors(labels, dtype=torch.long)
        bank_rows: list[torch.Tensor] = []
        for receiver, day in self.cell_keys:
            cell = (
                (rows.receiver_ids[batch_indices] == receiver)
                & (rows.day_ids[batch_indices] == day)
            )
            cell_tensor = self._tensors(cell, dtype=torch.bool)
            class_means: list[torch.Tensor] = []
            for label_index in range(len(self.classes)):
                local = cell_tensor & (label_tensor == label_index)
                if not torch.any(local):
                    raise D103R1TrainingError("bank cell lost a retained class")
                class_means.append(encoded[local].mean(dim=0))
            bank_rows.append(
                functional.normalize(torch.stack(class_means).mean(dim=0), dim=0)
            )
        return torch.stack(bank_rows)

    def _episode_meta_loss(
        self,
        encoder: torch.Tensor,
        bank_g: torch.Tensor,
        episode: EpisodeIndices,
    ) -> torch.Tensor:
        rows = self.data.labeled
        support_z = self._tensors(rows.z_dom[episode.support])
        support_pre = self._tensors(rows.pre_relu[episode.support])
        support_y = self._tensors(
            _class_index(rows.tx_labels[episode.support], self.classes),
            dtype=torch.long,
        )
        query_pre = self._tensors(rows.pre_relu[episode.query])
        query_y = self._tensors(
            _class_index(rows.tx_labels[episode.query], self.classes),
            dtype=torch.long,
        )

        support_r = functional.normalize(support_z @ encoder.T, dim=1)
        similarity = torch.clamp(support_r @ bank_g.T, -1.0, 1.0)
        weights = torch.softmax(
            similarity / self.config.bank_temperature, dim=1
        )
        precision = functional.softplus(self.model.log_precision) + 0.05
        sigma = functional.softplus(self.model.log_sigma) + 0.05
        coverage = torch.sum(
            weights
            * torch.exp(-(1.0 - similarity) / sigma.square()[None, :]),
            dim=1,
        )
        sample_precision = coverage[:, None] * (weights @ precision)
        sample_mean = weights @ self.model.bank_t

        class_precision: list[torch.Tensor] = []
        class_rhs: list[torch.Tensor] = []
        for label_index in range(len(self.classes)):
            local = support_y == label_index
            if int(local.sum()) != episode.k_shot:
                raise D103R1TrainingError("episode violates balanced K-shot")
            class_precision.append(sample_precision[local].mean(dim=0))
            class_rhs.append(
                (sample_precision[local] * sample_mean[local]).mean(dim=0)
            )
        a_data = torch.stack(class_precision).mean(dim=0)
        b_data = torch.stack(class_rhs).mean(dim=0)
        lambda0 = self._tensors(np.asarray(self.config.lambda0, dtype=np.float32))
        coefficient = b_data / (lambda0 + a_data)
        limits = self._tensors(np.asarray(self.config.a_max, dtype=np.float32))
        coefficient = torch.clamp(coefficient, min=-limits, max=limits)
        quadratic = torch.sum(lambda0 * coefficient.square())
        radius = torch.as_tensor(
            self.config.ellipsoid_radius,
            dtype=coefficient.dtype,
            device=self.device,
        )
        scale = torch.clamp(
            radius / torch.sqrt(torch.clamp(quadratic, min=_EPS)), max=1.0
        )
        coefficient = coefficient * scale

        shift = coefficient @ self.model.basis.T
        support_da = functional.normalize(
            functional.relu(support_pre + shift), dim=1
        )
        query_da = functional.normalize(
            functional.relu(query_pre + shift), dim=1
        )
        support_base = functional.normalize(functional.relu(support_pre), dim=1)
        query_base = functional.normalize(functional.relu(query_pre), dim=1)
        da_logits = _qknn_logits(
            query_da,
            support_da,
            support_y,
            len(self.classes),
            self.config.qknn_training_temperature,
        )
        base_logits = _qknn_logits(
            query_base,
            support_base,
            support_y,
            len(self.classes),
            self.config.qknn_training_temperature,
        )
        return _class_balanced_meta_loss(
            da_logits, base_logits, query_y, len(self.classes), self.config
        )

    def step(self) -> MechanicalStepReceipt:
        """Execute the next pre-registered mechanical step.

        No metric, selector, callback, threshold, or early-stop input is
        accepted.  Steps must be contiguous and cannot exceed the frozen 400.
        """

        step_index = self.completed_steps
        if step_index >= self.config.total_meta_steps:
            raise D103R1TrainingError("frozen 400-step fit is already complete")
        ls = self.data.labeled
        us = self.data.unlabeled
        ls_indices, cell_keys, classes = _balanced_labeled_indices(
            ls,
            self.masks.labeled_train,
            step_index=step_index,
            samples_per_cell=self.config.samples_per_cell,
        )
        if cell_keys != self.cell_keys or classes != self.classes:
            raise D103R1TrainingError("training cell/class registry drift")
        us_indices = _balanced_unlabeled_indices(
            us,
            self.masks.unlabeled_train,
            step_index=step_index,
            samples_per_cell=self.config.samples_per_cell,
        )

        self.optimizer.zero_grad(set_to_none=True)
        encoder = self.model.encoder()
        z_ls = self._tensors(ls.z_dom[ls_indices])
        encoded_ls = functional.normalize(z_ls @ encoder.T, dim=1)
        label_indices = _class_index(ls.tx_labels[ls_indices], self.classes)
        label_tensor = self._tensors(label_indices, dtype=torch.long)
        receiver_values = tuple(sorted(np.unique(ls.receiver_ids[ls_indices]).tolist()))
        day_values = tuple(sorted(np.unique(ls.day_ids[ls_indices]).tolist()))
        receiver_lookup = {value: index for index, value in enumerate(receiver_values)}
        day_lookup = {value: index for index, value in enumerate(day_values)}
        receiver_tensor = self._tensors(
            np.asarray(
                [receiver_lookup[value] for value in ls.receiver_ids[ls_indices]],
                dtype=np.int64,
            ),
            dtype=torch.long,
        )
        day_tensor = self._tensors(
            np.asarray(
                [day_lookup[value] for value in ls.day_ids[ls_indices]],
                dtype=np.int64,
            ),
            dtype=torch.long,
        )

        self.ledger.authorize(
            SplitRole.LABELED_SOURCE,
            Operation.TX_MMD,
            ("z_dom", "receiver_ids", "day_ids", "tx_labels", "physical_ids"),
            int(ls_indices.size),
        )
        tx_loss = _mmd_loss(
            encoded_ls,
            label_tensor,
            len(self.classes),
            self.config.mmd_gammas,
        )
        self.ledger.authorize(
            SplitRole.LABELED_SOURCE,
            Operation.RX_SELF_SUPERVISION,
            ("z_dom", "receiver_ids", "day_ids", "tx_labels", "physical_ids"),
            int(ls_indices.size),
        )
        receiver_losses = [
            _receiver_nce(
                encoded_ls, receiver_tensor, day_tensor, label_tensor
            )
        ]
        self.ledger.authorize(
            SplitRole.LABELED_SOURCE,
            Operation.VICREG,
            ("z_dom",),
            int(ls_indices.size),
        )
        vicreg_losses = [_vicreg_loss(encoded_ls)]

        if us_indices.size:
            z_us = self._tensors(us.z_dom[us_indices])
            encoded_us = functional.normalize(z_us @ encoder.T, dim=1)
            us_receiver_values = tuple(
                sorted(np.unique(us.receiver_ids[us_indices]).tolist())
            )
            us_day_values = tuple(sorted(np.unique(us.day_ids[us_indices]).tolist()))
            us_receiver_lookup = {
                value: index for index, value in enumerate(us_receiver_values)
            }
            us_day_lookup = {
                value: index for index, value in enumerate(us_day_values)
            }
            us_receiver_tensor = self._tensors(
                np.asarray(
                    [
                        us_receiver_lookup[value]
                        for value in us.receiver_ids[us_indices]
                    ],
                    dtype=np.int64,
                ),
                dtype=torch.long,
            )
            us_day_tensor = self._tensors(
                np.asarray(
                    [us_day_lookup[value] for value in us.day_ids[us_indices]],
                    dtype=np.int64,
                ),
                dtype=torch.long,
            )
            self.ledger.authorize(
                SplitRole.UNLABELED_SOURCE,
                Operation.RX_SELF_SUPERVISION,
                ("z_dom", "receiver_ids", "day_ids", "physical_ids"),
                int(us_indices.size),
            )
            receiver_losses.append(
                _receiver_nce(
                    encoded_us, us_receiver_tensor, us_day_tensor, tx=None
                )
            )
            self.ledger.authorize(
                SplitRole.UNLABELED_SOURCE,
                Operation.VICREG,
                ("z_dom",),
                int(us_indices.size),
            )
            vicreg_losses.append(_vicreg_loss(encoded_us))

        self.ledger.authorize(
            SplitRole.LABELED_SOURCE,
            Operation.CLASS_BALANCED_BANK,
            ("z_dom", "receiver_ids", "day_ids", "tx_labels", "physical_ids"),
            int(ls_indices.size),
        )
        bank_g = self._bank_g(encoded_ls, ls_indices)

        eligible_receivers = tuple(
            sorted(np.unique(ls.receiver_ids[self.masks.labeled_train]).tolist())
        )
        if len(eligible_receivers) < 2:
            raise D103R1TrainingError(
                "cross-receiver episode requires at least two eligible receivers"
            )
        episode_support_receiver = eligible_receivers[
            step_index % len(eligible_receivers)
        ]
        episode_query_receiver = eligible_receivers[
            (step_index + 1) % len(eligible_receivers)
        ]
        self.ledger.authorize(
            SplitRole.LABELED_SOURCE,
            Operation.METABIAS_META,
            (
                "z_dom",
                "pre_relu",
                "receiver_ids",
                "day_ids",
                "tx_labels",
                "physical_ids",
            ),
            int(np.sum(self.masks.labeled_train)),
        )
        episodes = [
            _build_episode(
                ls,
                self.masks.labeled_train,
                k_shot=k_shot,
                support_receiver=episode_support_receiver,
                query_receiver=episode_query_receiver,
                query_per_class=self.config.query_per_class,
                step_index=step_index,
            )
            for k_shot in self.config.k_values
        ]
        meta_losses = [
            self._episode_meta_loss(encoder, bank_g, episode)
            for episode in episodes
        ]
        meta_loss = torch.stack(meta_losses).mean()
        receiver_loss = torch.stack(receiver_losses).mean()
        vicreg_loss = torch.stack(vicreg_losses).mean()
        identity = torch.eye(DOMAIN_DIM, device=self.device)
        orthogonality = (encoder @ encoder.T - identity).square().mean()
        loss = (
            meta_loss
            + self.config.lambda_tx * tx_loss
            + self.config.lambda_rx * receiver_loss
            + self.config.lambda_vicreg * vicreg_loss
            + self.config.lambda_orthogonal * orthogonality
        )
        if not torch.isfinite(loss):
            raise D103R1TrainingError("mechanical training loss became non-finite")
        loss.backward()
        self.optimizer.step()
        self.completed_steps += 1
        ledger_receipt = self.ledger.receipt()
        return MechanicalStepReceipt(
            schema=f"{SCHEMA}.mechanical_step",
            candidate_id=CANDIDATE_ID,
            step_index=step_index,
            total_loss=float(loss.detach().cpu()),
            meta_loss=float(meta_loss.detach().cpu()),
            tx_mmd_loss=float(tx_loss.detach().cpu()),
            receiver_loss=float(receiver_loss.detach().cpu()),
            vicreg_loss=float(vicreg_loss.detach().cpu()),
            orthogonality_loss=float(orthogonality.detach().cpu()),
            k_values=tuple(episode.k_shot for episode in episodes),
            episode_support_receiver=episode_support_receiver,
            episode_query_receiver=episode_query_receiver,
            labeled_batch_rows=int(ls_indices.size),
            unlabeled_batch_rows=int(us_indices.size),
            performance_metrics_computed=False,
            source_val_rows_used=0,
            target_access=False,
            formal_query_access=False,
            optimizer_step_completed=True,
            ledger_receipt_sha256=str(ledger_receipt["receipt_sha256"]),
        )

    def final_state(self) -> D103R1TeacherState:
        """Aggregate the legal final teacher state after exactly 400 steps."""

        if self.completed_steps != self.config.total_meta_steps:
            raise D103R1TrainingError(
                "teacher state requires exactly 400 completed meta steps"
            )
        if self.ledger.denied_attempts != 0:
            raise D103R1TrainingError(
                "teacher state refused after a denied permission attempt"
            )
        if self._final_state_cache is not None:
            return self._final_state_cache

        rows = self.data.labeled
        eligible = self.masks.labeled_train
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size == 0:
            raise D103R1TrainingError("teacher aggregation has no eligible L_s rows")
        self.ledger.authorize(
            SplitRole.LABELED_SOURCE,
            Operation.FINAL_AGGREGATION,
            ("z_dom", "receiver_ids", "day_ids", "tx_labels", "physical_ids"),
            int(eligible_indices.size),
        )

        with torch.no_grad():
            encoder_tensor = self.model.encoder()
            all_encoded = functional.normalize(
                self._tensors(rows.z_dom[eligible_indices]) @ encoder_tensor.T,
                dim=1,
            )
            encoded = np.asarray(all_encoded.detach().cpu(), dtype=np.float32)
            encoder = np.asarray(
                encoder_tensor.detach().cpu(), dtype=np.float32
            )
            basis = np.asarray(self.model.basis.detach().cpu(), dtype=np.float32)
            bank_t = np.asarray(
                self.model.bank_t.detach().cpu(), dtype=np.float32
            )
            precision = np.asarray(
                (functional.softplus(self.model.log_precision) + 0.05)
                .detach()
                .cpu(),
                dtype=np.float32,
            )
            sigma = np.asarray(
                (functional.softplus(self.model.log_sigma) + 0.05)
                .detach()
                .cpu(),
                dtype=np.float32,
            )

        eligible_receivers = rows.receiver_ids[eligible_indices]
        eligible_days = rows.day_ids[eligible_indices]
        eligible_labels = rows.tx_labels[eligible_indices]
        eligible_physical = rows.physical_ids[eligible_indices]
        bank_g_rows: list[np.ndarray] = []
        class_cell_counts: list[int] = []
        for receiver, day in self.cell_keys:
            class_means: list[np.ndarray] = []
            for label in self.classes:
                local = (
                    (eligible_receivers == receiver)
                    & (eligible_days == day)
                    & (eligible_labels == label)
                )
                count = int(np.sum(local))
                if count < self.config.samples_per_cell:
                    raise D103R1TrainingError(
                        "final class-cell aggregation requires at least two "
                        "physical samples"
                    )
                if np.unique(eligible_physical[local]).size != count:
                    raise D103R1TrainingError(
                        "final class-cell aggregation reuses a physical sample"
                    )
                class_cell_counts.append(count)
                class_means.append(encoded[local].mean(axis=0))
            # Each class contributes one mean regardless of its physical count.
            class_equal = np.mean(np.stack(class_means), axis=0)
            norm = float(np.linalg.norm(class_equal))
            if not math.isfinite(norm) or norm <= _EPS:
                raise D103R1TrainingError(
                    "final class-equal bank aggregation produced a zero row"
                )
            bank_g_rows.append(np.asarray(class_equal / norm, dtype=np.float32))
        bank_g = np.stack(bank_g_rows)
        if bank_g.shape[0] != bank_t.shape[0]:
            raise D103R1TrainingError("final bank cell count drift")

        array_hashes = {
            "U": _array_sha256(encoder),
            "B": _array_sha256(basis),
            "bank_g": _array_sha256(bank_g),
            "bank_t": _array_sha256(bank_t),
            "bank_precision": _array_sha256(precision),
            "bank_sigma": _array_sha256(sigma),
        }
        aggregation_receipt = MappingProxyType(
            {
                "schema": f"{SCHEMA}.teacher_aggregation_receipt",
                "candidate_id": CANDIDATE_ID,
                "completed_meta_steps": self.completed_steps,
                "eligible_labeled_rows": int(eligible_indices.size),
                "unlabeled_rows_used": 0,
                "source_val_rows_used": 0,
                "bank_cell_count": len(self.cell_keys),
                "registered_class_count": len(self.classes),
                "class_cell_count": len(class_cell_counts),
                "minimum_physical_samples_per_class_cell": min(
                    class_cell_counts
                ),
                "all_eligible_labeled_physical_rows_used": True,
                "aggregation_order": (
                    "physical_mean_within_class_cell_then_equal_mean_over_classes"
                ),
                "class_weight": 1.0 / len(self.classes),
                "array_shapes": {
                    "U": list(encoder.shape),
                    "B": list(basis.shape),
                    "bank_g": list(bank_g.shape),
                    "bank_t": list(bank_t.shape),
                    "bank_precision": list(precision.shape),
                    "bank_sigma": list(sigma.shape),
                },
                "array_sha256": array_hashes,
                "contains_receiver_values": False,
                "contains_day_values": False,
                "contains_class_values": False,
                "contains_physical_ids": False,
                "contains_optimizer": False,
            }
        )
        access_receipt = self.ledger.receipt()
        state = D103R1TeacherState(
            u=encoder,
            b=basis,
            bank_g=bank_g,
            bank_t=bank_t,
            bank_precision=precision,
            bank_sigma=sigma,
            aggregation_receipt=aggregation_receipt,
            access_receipt=access_receipt,
        )
        self._final_state_cache = state
        return state

    def export_teacher_arrays(self) -> Mapping[str, Any]:
        """Export the allowlisted final teacher payload after the 400-step gate."""

        return export_teacher_arrays(self.final_state())


__all__ = [
    "CANDIDATE_ID",
    "CODE_DIM",
    "D103R1Config",
    "D103R1Phase1Trainer",
    "D103R1TeacherState",
    "D103R1TrainingData",
    "D103R1TrainingError",
    "DOMAIN_DIM",
    "EpisodeIndices",
    "FEATURE_DIM",
    "FINAL_TX_NULL_RANK",
    "FoldMasks",
    "K_VALUES",
    "LabeledSourceRows",
    "MMD_GAMMAS",
    "MechanicalStepReceipt",
    "Operation",
    "OuterMaskSpec",
    "PermissionLedger",
    "QUERY_PER_CLASS",
    "RXIDMetaBias4Model",
    "SCHEMA",
    "SAMPLES_PER_CELL",
    "SEED",
    "SourceValidationSeal",
    "SplitRole",
    "UnlabeledSourceRows",
    "build_outer_masks",
    "build_training_data",
    "build_tx_projector",
    "export_teacher_arrays",
]
