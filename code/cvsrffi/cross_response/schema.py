"""Metadata contract: physical receptions, never augmented views, count toward K."""
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class SampleRecord:
    index: int
    tx_id: int
    rx_id: int
    day_id: int
    condition_id: Any
    physical_sample_id: Any
    event_id: Any = None


@dataclass(frozen=True)
class RolePlan:
    donor_tx: Tuple[int, ...]
    query_tx: Tuple[int, ...]
    donor_rx: Tuple[int, ...]
    query_rx: Tuple[int, ...]
    rotation: int = 0
    role_policy_version: str = "legacy_halves_v1"
    plan_id: int = 0

    @property
    def independent_query_rectangle(self):
        return len(self.query_tx) >= 2 and len(self.query_rx) >= 2


@dataclass(frozen=True)
class BlockCandidate:
    tx_ids: tuple
    rx_ids: tuple
    day_id: int
    condition_id: Any
    k: int

    @property
    def size(self):
        return len(self.tx_ids) * len(self.rx_ids) * self.k

    @property
    def key(self):
        return (self.tx_ids, self.rx_ids, self.day_id, self.condition_id, self.k)


@dataclass(frozen=True)
class CrossBlock:
    candidate: BlockCandidate
    records: tuple
    batch_positions: tuple
    roles: RolePlan
    block_id: int
    selection_probability: float

    def validate(self):
        c = self.candidate
        role = self.roles
        for donor, query, axis in ((role.donor_tx, role.query_tx, c.tx_ids),
                                   (role.donor_rx, role.query_rx, c.rx_ids)):
            if not donor or not query or set(donor) & set(query) or set(donor) | set(query) != set(axis):
                raise ValueError("invalid or overlapping donor/query roles")
        if len(set(c.tx_ids)) != len(c.tx_ids) or len(set(c.rx_ids)) != len(c.rx_ids) or c.k < 1:
            raise ValueError("invalid physical axes or K")
        if len(self.records) != c.size or len(self.batch_positions) != c.size:
            raise ValueError("incomplete block")
        if len(set(self.batch_positions)) != c.size:
            raise ValueError("duplicate batch positions")
        if len({r.physical_sample_id for r in self.records}) != c.size:
            raise ValueError("duplicate physical reception")
        expected = [(t, r) for t in c.tx_ids for r in c.rx_ids for _ in range(c.k)]
        for record, cell in zip(self.records, expected):
            if (record.tx_id, record.rx_id) != cell or record.day_id != c.day_id or record.condition_id != c.condition_id:
                raise ValueError("missing cell or incomparable condition")
        query_events = {r.event_id for r in self.records if r.event_id is not None
                        and r.tx_id in role.query_tx and r.rx_id in role.query_rx}
        donor_events = {r.event_id for r in self.records if r.event_id is not None
                        and ((r.tx_id in role.query_tx) != (r.rx_id in role.query_rx))}
        if query_events & donor_events:
            raise ValueError("query/donor physical event overlap")


@dataclass(frozen=True)
class BatchPlan:
    indices: tuple
    blocks: tuple
    skip_reason: str = ""


def records_from_dataset(dataset, *, allowed_split_sources=("ssdg_labeled_tx_visible",)):
    """Read WiSig's legal labeled subset index without loading IQ or hidden labels.

    eq_i is a protocol condition; sig_i is local to a physical TX/RX/day/eq
    cell and is NOT a synchronized event identifier.
    """
    if getattr(dataset, "split_source", None) not in allowed_split_sources:
        raise ValueError("cross blocks require an explicitly allowed source-labeled split")
    if not hasattr(dataset, "index"):
        raise ValueError("dataset must expose physical index metadata")
    records = []
    for i, row in enumerate(dataset.index):
        values = tuple(int(getattr(row, key)) for key in ("tx_i", "rx_i", "day_i", "eq_i", "sig_i"))
        if any(v < 0 for v in values):
            raise ValueError("negative or hidden physical metadata")
        records.append(SampleRecord(i, values[0], values[1], values[2], values[3], values,
                                    getattr(row, "event_id", None)))
    return records
