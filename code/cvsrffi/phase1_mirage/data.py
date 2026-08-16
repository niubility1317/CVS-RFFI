"""Role-safe source inventory splitting for Phase1 MIRAGE-OWDG."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Collection, Literal, Mapping, Sequence

from .protocol import Phase1DataPolicy, Phase1PolicyError, SampleIdentity, SourcePartition


class SourceProtocolError(Phase1PolicyError):
    """Raised when a source inventory violates the approved role boundary."""


_DATA_POLICY = Phase1DataPolicy()
_PARTITION_ORDER = (
    SourcePartition.L_S,
    SourcePartition.U_S,
    SourcePartition.V_CAL,
    SourcePartition.V_SELECT,
)
_VALIDATION_ROLES = frozenset({"val_cal", "val_select"})
_SPLIT_SCHEMA = "phase1_mirage_source_7_63_15_15_v1"


@dataclass(frozen=True)
class SourceInventoryRow:
    """Source-only metadata visible to the one-time split builder."""

    physical_sample_id: str
    tx_label: int
    receiver_id: str
    day_id: str
    iq_index: int


@dataclass(frozen=True)
class LabeledView:
    """Approved labeled source training record."""

    physical_sample_id: str
    tx_label: int
    receiver_id: str
    day_id: str
    iq_index: int


@dataclass(frozen=True)
class UnlabeledView:
    """Approved unlabeled source training record without TX truth."""

    physical_sample_id: str
    receiver_id: str
    day_id: str
    iq_index: int


@dataclass(frozen=True)
class ValidationView(LabeledView):
    """Approved source validation record with its immutable validation role."""

    split_role: Literal["val_cal", "val_select"]


@dataclass(frozen=True)
class SourceSplitManifest:
    """Immutable source split IDs and non-sensitive construction receipts."""

    l_ids: tuple[str, ...]
    u_ids: tuple[str, ...]
    v_cal_ids: tuple[str, ...]
    v_select_ids: tuple[str, ...]
    id_sha256: Mapping[SourcePartition, str]
    group_counts: Mapping[tuple[int, str, str], tuple[int, int, int, int]]
    receiver_registry: tuple[str, ...]
    tx_registry: tuple[int, ...]
    split_schema: str


def _partition_counts(size: int) -> tuple[int, int, int, int]:
    """Delegate integer split allocation to the Task 1 approved policy."""

    counts = _DATA_POLICY.partition_counts(size)
    return tuple(counts[partition] for partition in _PARTITION_ORDER)


def _id_sha256(identifiers: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()


def _inventory_by_id(rows: Sequence[SourceInventoryRow]) -> dict[str, SourceInventoryRow]:
    inventory: dict[str, SourceInventoryRow] = {}
    for row in rows:
        if not row.physical_sample_id:
            raise SourceProtocolError("physical_sample_id must be non-empty")
        if row.physical_sample_id in inventory:
            raise SourceProtocolError(f"duplicate physical_sample_id: {row.physical_sample_id}")
        inventory[row.physical_sample_id] = row
    return inventory


def _select_rows(
    rows: Sequence[SourceInventoryRow],
    physical_ids: Sequence[str],
    *,
    allowed_ids: Collection[str],
    source_partition: SourcePartition,
) -> tuple[SourceInventoryRow, ...]:
    inventory = _inventory_by_id(rows)
    requested_ids = tuple(physical_ids)
    if len(set(requested_ids)) != len(requested_ids):
        raise SourceProtocolError("duplicate physical_sample_id in materialization request")
    if not set(requested_ids).issubset(allowed_ids):
        raise SourceProtocolError(
            f"physical_sample_id is not authorized for {source_partition.value}"
        )
    try:
        return tuple(inventory[physical_sample_id] for physical_sample_id in requested_ids)
    except KeyError as error:
        raise SourceProtocolError(f"unknown physical_sample_id: {error.args[0]}") from error


def _validate_source_partitions(
    inventory: Mapping[str, SourceInventoryRow], buckets: Sequence[Sequence[str]]
) -> None:
    try:
        _DATA_POLICY.validate_source_partitions(
            l_s=(SampleIdentity(physical_sample_id, str(inventory[physical_sample_id].tx_label)) for physical_sample_id in buckets[0]),
            u_s=(SampleIdentity(physical_sample_id, str(inventory[physical_sample_id].tx_label)) for physical_sample_id in buckets[1]),
            v_cal=(SampleIdentity(physical_sample_id, str(inventory[physical_sample_id].tx_label)) for physical_sample_id in buckets[2]),
            v_select=(SampleIdentity(physical_sample_id, str(inventory[physical_sample_id].tx_label)) for physical_sample_id in buckets[3]),
        )
    except Phase1PolicyError as error:
        raise SourceProtocolError(str(error)) from error


def build_source_split(
    rows: Sequence[SourceInventoryRow],
    *,
    seed: int,
    forbidden_receivers: Collection[str],
) -> SourceSplitManifest:
    """Split source physical IDs by TX, receiver, and day without target access."""

    forbidden = frozenset(forbidden_receivers)
    if any(row.receiver_id in forbidden for row in rows):
        raise SourceProtocolError("target receiver present in source inventory")

    inventory = _inventory_by_id(rows)
    groups: dict[tuple[int, str, str], list[SourceInventoryRow]] = defaultdict(list)
    for row in rows:
        groups[(row.tx_label, row.receiver_id, row.day_id)].append(row)

    buckets: list[list[str]] = [[], [], [], []]
    group_counts: dict[tuple[int, str, str], tuple[int, int, int, int]] = {}
    for group_key, group_rows in groups.items():
        ordered = sorted(
            group_rows,
            key=lambda row: (
                hashlib.sha256(f"{seed}:{row.physical_sample_id}".encode("utf-8")).digest(),
                row.physical_sample_id,
            ),
        )
        counts = _partition_counts(len(ordered))
        group_counts[group_key] = counts
        cursor = 0
        for bucket, count in zip(buckets, counts):
            bucket.extend(row.physical_sample_id for row in ordered[cursor : cursor + count])
            cursor += count

    partition_ids = tuple(tuple(sorted(bucket)) for bucket in buckets)
    _validate_source_partitions(inventory, partition_ids)
    id_sha256 = MappingProxyType(
        {
            partition: _id_sha256(identifiers)
            for partition, identifiers in zip(_PARTITION_ORDER, partition_ids)
        }
    )
    return SourceSplitManifest(
        l_ids=partition_ids[0],
        u_ids=partition_ids[1],
        v_cal_ids=partition_ids[2],
        v_select_ids=partition_ids[3],
        id_sha256=id_sha256,
        group_counts=MappingProxyType(dict(group_counts)),
        receiver_registry=tuple(sorted({row.receiver_id for row in rows})),
        tx_registry=tuple(sorted({row.tx_label for row in rows})),
        split_schema=_SPLIT_SCHEMA,
    )


def materialize_labeled(
    rows: Sequence[SourceInventoryRow],
    physical_ids: Sequence[str],
    *,
    manifest: SourceSplitManifest,
) -> tuple[LabeledView, ...]:
    """Return only the fields approved for labeled source training."""

    return tuple(
        LabeledView(
            physical_sample_id=row.physical_sample_id,
            tx_label=row.tx_label,
            receiver_id=row.receiver_id,
            day_id=row.day_id,
            iq_index=row.iq_index,
        )
        for row in _select_rows(
            rows,
            physical_ids,
            allowed_ids=manifest.l_ids,
            source_partition=SourcePartition.L_S,
        )
    )


def materialize_unlabeled(
    rows: Sequence[SourceInventoryRow],
    physical_ids: Sequence[str],
    *,
    manifest: SourceSplitManifest,
) -> tuple[UnlabeledView, ...]:
    """Return a structural label-free view for unlabeled source training."""

    return tuple(
        UnlabeledView(
            physical_sample_id=row.physical_sample_id,
            receiver_id=row.receiver_id,
            day_id=row.day_id,
            iq_index=row.iq_index,
        )
        for row in _select_rows(
            rows,
            physical_ids,
            allowed_ids=manifest.u_ids,
            source_partition=SourcePartition.U_S,
        )
    )


def materialize_validation(
    rows: Sequence[SourceInventoryRow],
    physical_ids: Sequence[str],
    *,
    split_role: Literal["val_cal", "val_select"],
    manifest: SourceSplitManifest,
) -> tuple[ValidationView, ...]:
    """Return labeled validation data with an explicit approved validation role."""

    if split_role not in _VALIDATION_ROLES:
        raise SourceProtocolError(f"unsupported split_role: {split_role}")
    allowed_ids, source_partition = (
        (manifest.v_cal_ids, SourcePartition.V_CAL)
        if split_role == "val_cal"
        else (manifest.v_select_ids, SourcePartition.V_SELECT)
    )
    return tuple(
        ValidationView(
            physical_sample_id=row.physical_sample_id,
            tx_label=row.tx_label,
            receiver_id=row.receiver_id,
            day_id=row.day_id,
            iq_index=row.iq_index,
            split_role=split_role,
        )
        for row in _select_rows(
            rows,
            physical_ids,
            allowed_ids=allowed_ids,
            source_partition=source_partition,
        )
    )
