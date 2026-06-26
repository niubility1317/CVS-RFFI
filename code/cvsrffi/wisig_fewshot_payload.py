"""WiSig feature payload builders for spaceborne few-shot protocols.

This module deliberately works on extracted features plus metadata instead of
raw IQ. It is the audit layer that enforces old/new transmitter separation
before the feature-level CVS-SFE evaluator builds prototypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


UNKNOWN_LABEL = -1


def canonical_tx_id(value: Any) -> str:
    """Return a stable transmitter identity string for cross-PKL comparison."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def parse_tx_id_list(value: str | Sequence[Any] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [canonical_tx_id(item).strip() for item in value]
    return [canonical_tx_id(item) for item in items if str(item).strip() != ""]


def _canonical_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    return np.asarray([canonical_tx_id(v) for v in arr.reshape(-1).tolist()], dtype=object)


def assert_disjoint_tx_sets(*, source_tx_ids: Sequence[Any], new_tx_ids: Sequence[Any], unknown_tx_ids: Sequence[Any] = ()) -> dict:
    source = set(parse_tx_id_list(source_tx_ids))
    new = set(parse_tx_id_list(new_tx_ids))
    unknown = set(parse_tx_id_list(unknown_tx_ids))
    overlaps = {
        "source_new": sorted(source & new),
        "source_unknown": sorted(source & unknown),
        "new_unknown": sorted(new & unknown),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"transmitter identity sets must be disjoint: {bad}")
    return overlaps


@dataclass(frozen=True)
class SfePayload:
    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]


def _indices_for_tx(tx_ids: np.ndarray, requested: Sequence[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for tx in requested:
        idx = np.flatnonzero(tx_ids == tx)
        if idx.size == 0:
            raise ValueError(f"requested transmitter {tx!r} has no samples in feature NPZ")
        out[tx] = idx.astype(np.int64)
    return out


def _indices_for_tx_with_role(
    tx_ids: np.ndarray,
    requested: Sequence[str],
    *,
    dataset_roles: np.ndarray | None,
    role: str | None,
) -> dict[str, np.ndarray]:
    if dataset_roles is None or role is None:
        return _indices_for_tx(tx_ids, requested)
    if dataset_roles.shape[0] != tx_ids.shape[0]:
        raise ValueError("dataset_roles and tx_ids must have equal length")
    role_mask = dataset_roles == str(role)
    out: dict[str, np.ndarray] = {}
    for tx in requested:
        idx = np.flatnonzero((tx_ids == tx) & role_mask)
        if idx.size == 0:
            raise ValueError(f"requested transmitter {tx!r} has no samples for dataset_role={role!r}")
        out[tx] = idx.astype(np.int64)
    return out


def _indices_for_tx_with_any_role(
    tx_ids: np.ndarray,
    requested: Sequence[str],
    *,
    dataset_roles: np.ndarray | None,
    roles: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    if dataset_roles is None or not roles:
        return _indices_for_tx(tx_ids, requested)
    if dataset_roles.shape[0] != tx_ids.shape[0]:
        raise ValueError("dataset_roles and tx_ids must have equal length")
    role_values = [str(role) for role in roles]
    out: dict[str, np.ndarray] = {}
    for tx in requested:
        parts = [np.flatnonzero((tx_ids == tx) & (dataset_roles == role)) for role in role_values]
        idx = np.concatenate([part for part in parts if part.size > 0]) if any(part.size > 0 for part in parts) else np.empty((0,), dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"requested transmitter {tx!r} has no samples for dataset_role in {role_values!r}")
        out[tx] = np.sort(idx.astype(np.int64))
    return out


def _take_for_tx(
    indices_by_tx: Mapping[str, np.ndarray],
    tx: str,
    *,
    start: int,
    count: int,
    role: str,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    idx = np.asarray(indices_by_tx[tx], dtype=np.int64)
    end = int(start) + int(count)
    if idx.size < end:
        raise ValueError(
            f"not enough samples for tx={tx!r} role={role}: need {end}, available {idx.size}"
        )
    return idx[int(start):end]


def _split_overlap_audit(splits: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    names = sorted(splits)
    audit: dict[str, list[int]] = {}
    for i, left in enumerate(names):
        left_set = {int(v) for v in np.asarray(splits[left], dtype=np.int64).tolist()}
        for right in names[i + 1:]:
            right_set = {int(v) for v in np.asarray(splits[right], dtype=np.int64).tolist()}
            audit[f"{left}__{right}"] = sorted(left_set & right_set)
    return audit


def build_sfe_payload_from_feature_arrays(
    *,
    features: np.ndarray,
    tx_ids: Sequence[Any] | np.ndarray,
    dataset_roles: Sequence[Any] | np.ndarray | None = None,
    sample_metadata: Mapping[str, Sequence[Any] | np.ndarray] | None = None,
    source_tx_ids: str | Sequence[Any],
    target_old_tx_ids: str | Sequence[Any] | None = None,
    new_tx_ids: str | Sequence[Any],
    unknown_tx_ids: str | Sequence[Any] | None = None,
    shots: int = 5,
    source_proto_per_tx: int = 20,
    source_query_per_tx: int = 20,
    target_old_support_per_tx: int = 0,
    target_old_query_per_tx: int | None = None,
    query_per_tx: int = 50,
    seed: int = 1337,
    extra_metadata: Mapping[str, Any] | None = None,
) -> SfePayload:
    """Build a CVS-SFE NPZ payload with old/new TX identity de-duplication.

    Input `features` contains all candidate samples. The caller must provide
    explicit transmitter identities; local WiSig integer labels are not trusted
    across compact subsets.
    """

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got shape={features.shape}")
    tx_ids_arr = _canonical_array(tx_ids)
    if tx_ids_arr.shape[0] != features.shape[0]:
        raise ValueError("features and tx_ids must have equal length")

    source_txs = parse_tx_id_list(source_tx_ids)
    target_old_txs = parse_tx_id_list(target_old_tx_ids)
    new_txs = parse_tx_id_list(new_tx_ids)
    unknown_txs = parse_tx_id_list(unknown_tx_ids)
    if not source_txs:
        raise ValueError("source_tx_ids must not be empty")
    old_unknown_only = not new_txs
    if old_unknown_only and not target_old_txs:
        raise ValueError("new_tx_ids may be empty only when target_old_tx_ids are provided")
    if old_unknown_only and int(shots) != 0:
        raise ValueError("new_tx_ids may be empty only for shots=0 old/unknown-only evaluation")
    overlap_audit = assert_disjoint_tx_sets(
        source_tx_ids=source_txs,
        new_tx_ids=new_txs,
        unknown_tx_ids=unknown_txs,
    )

    rng = np.random.default_rng(int(seed))
    role_arr = _canonical_array(dataset_roles) if dataset_roles is not None else None
    metadata_arrays: dict[str, np.ndarray] = {}
    for name, values in (sample_metadata or {}).items():
        arr = _canonical_array(values)
        if arr.shape[0] != features.shape[0]:
            raise ValueError(f"sample_metadata[{name!r}] and features must have equal length")
        metadata_arrays[str(name)] = arr
    source_indices_raw = _indices_for_tx_with_role(
        tx_ids_arr,
        source_txs,
        dataset_roles=role_arr,
        role="source" if role_arr is not None else None,
    )
    old_query_indices_raw = (
        _indices_for_tx_with_role(tx_ids_arr, target_old_txs, dataset_roles=role_arr, role="target_old")
        if target_old_txs
        else source_indices_raw
    )
    new_indices_raw = (
        _indices_for_tx_with_role(
            tx_ids_arr,
            new_txs,
            dataset_roles=role_arr,
            role="target_new" if role_arr is not None else None,
        )
        if new_txs
        else {}
    )
    unknown_indices_raw = (
        _indices_for_tx_with_any_role(
            tx_ids_arr,
            unknown_txs,
            dataset_roles=role_arr,
            roles=("target_unknown", "target_new") if role_arr is not None else None,
        )
        if unknown_txs
        else {}
    )
    source_indices = {
        tx: np.asarray(idx, dtype=np.int64)[rng.permutation(len(idx))]
        for tx, idx in source_indices_raw.items()
    }
    old_query_indices = {
        tx: np.asarray(idx, dtype=np.int64)[rng.permutation(len(idx))]
        for tx, idx in old_query_indices_raw.items()
    }
    new_indices = {
        tx: np.asarray(idx, dtype=np.int64)[rng.permutation(len(idx))]
        for tx, idx in new_indices_raw.items()
    }
    unknown_indices = {
        tx: np.asarray(idx, dtype=np.int64)[rng.permutation(len(idx))]
        for tx, idx in unknown_indices_raw.items()
    }

    source_label_map = {tx: i for i, tx in enumerate(source_txs)}
    new_label_map = {tx: len(source_txs) + i for i, tx in enumerate(new_txs)}

    source_idx_parts: list[np.ndarray] = []
    source_labels: list[int] = []
    target_old_support_idx_parts: list[np.ndarray] = []
    target_old_support_labels: list[int] = []
    source_query_idx_parts: list[np.ndarray] = []
    source_query_labels: list[int] = []
    for tx in source_txs:
        proto_idx = _take_for_tx(
            source_indices,
            tx,
            start=0,
            count=int(source_proto_per_tx),
            role="source_prototype",
        )
        query_source = old_query_indices[tx] if target_old_txs else source_indices[tx]
        old_support_count = int(target_old_support_per_tx) if target_old_txs else 0
        old_support_idx = _take_for_tx(
            {tx: query_source},
            tx,
            start=0,
            count=old_support_count,
            role="target_old_support",
        )
        query_start = old_support_count if target_old_txs else int(source_proto_per_tx)
        query_count = int(target_old_query_per_tx if target_old_query_per_tx is not None else source_query_per_tx)
        query_idx = _take_for_tx(
            {tx: query_source},
            tx,
            start=query_start,
            count=query_count,
            role="target_old_query" if target_old_txs else "source_query",
        )
        source_idx_parts.append(proto_idx)
        source_labels.extend([source_label_map[tx]] * int(proto_idx.size))
        target_old_support_idx_parts.append(old_support_idx)
        target_old_support_labels.extend([source_label_map[tx]] * int(old_support_idx.size))
        source_query_idx_parts.append(query_idx)
        source_query_labels.extend([source_label_map[tx]] * int(query_idx.size))

    support_idx_parts: list[np.ndarray] = list(target_old_support_idx_parts)
    support_labels: list[int] = list(target_old_support_labels)
    new_support_idx_parts: list[np.ndarray] = []
    new_query_idx_parts: list[np.ndarray] = []
    new_query_labels: list[int] = []
    query_roles: list[str] = []
    for tx in new_txs:
        support_idx = _take_for_tx(
            new_indices,
            tx,
            start=0,
            count=int(shots),
            role="new_support",
        )
        query_idx = _take_for_tx(
            new_indices,
            tx,
            start=int(shots),
            count=int(query_per_tx),
            role="new_query",
        )
        support_idx_parts.append(support_idx)
        new_support_idx_parts.append(support_idx)
        support_labels.extend([new_label_map[tx]] * int(support_idx.size))
        new_query_idx_parts.append(query_idx)
        new_query_labels.extend([new_label_map[tx]] * int(query_idx.size))
        query_roles.extend(["new_query"] * int(query_idx.size))

    unknown_query_idx_parts: list[np.ndarray] = []
    unknown_query_labels: list[int] = []
    for tx in unknown_txs:
        query_idx = _take_for_tx(
            unknown_indices,
            tx,
            start=0,
            count=int(query_per_tx),
            role="unknown_query",
        )
        unknown_query_idx_parts.append(query_idx)
        unknown_query_labels.extend([UNKNOWN_LABEL] * int(query_idx.size))
        query_roles.extend(["unknown_query"] * int(query_idx.size))

    source_idx = np.concatenate(source_idx_parts) if source_idx_parts else np.empty((0,), dtype=np.int64)
    support_idx = np.concatenate(support_idx_parts) if support_idx_parts else np.empty((0,), dtype=np.int64)
    target_old_support_idx = (
        np.concatenate(target_old_support_idx_parts) if target_old_support_idx_parts else np.empty((0,), dtype=np.int64)
    )
    new_support_idx = np.concatenate(new_support_idx_parts) if new_support_idx_parts else np.empty((0,), dtype=np.int64)
    query_parts = source_query_idx_parts + new_query_idx_parts + unknown_query_idx_parts
    query_idx = np.concatenate(query_parts) if query_parts else np.empty((0,), dtype=np.int64)
    query_labels = np.asarray(source_query_labels + new_query_labels + unknown_query_labels, dtype=np.int64)
    old_query_role = "target_old_query" if target_old_txs else "source_query"
    query_roles = (
        [old_query_role] * int(sum(len(part) for part in source_query_idx_parts))
        + query_roles
    )
    source_query_split = (
        np.empty((0,), dtype=np.int64)
        if target_old_txs
        else (np.concatenate(source_query_idx_parts) if source_query_idx_parts else np.empty((0,), dtype=np.int64))
    )
    target_old_query_split = (
        np.concatenate(source_query_idx_parts) if target_old_txs and source_query_idx_parts else np.empty((0,), dtype=np.int64)
    )
    split_indices = {
        "source_prototype": source_idx,
        "source_query": source_query_split,
        "target_old_support": target_old_support_idx,
        "target_old_query": target_old_query_split,
        "new_support": new_support_idx,
        "new_query": np.concatenate(new_query_idx_parts) if new_query_idx_parts else np.empty((0,), dtype=np.int64),
        "unknown_query": np.concatenate(unknown_query_idx_parts) if unknown_query_idx_parts else np.empty((0,), dtype=np.int64),
    }
    split_overlap_audit = _split_overlap_audit(split_indices)

    arrays = {
        "source_features": features[source_idx],
        "source_labels": np.asarray(source_labels, dtype=np.int64),
        "support_features": features[support_idx],
        "support_labels": np.asarray(support_labels, dtype=np.int64),
        "query_features": features[query_idx],
        "query_labels": query_labels,
        "source_tx_ids": tx_ids_arr[source_idx].astype(str),
        "support_tx_ids": tx_ids_arr[support_idx].astype(str),
        "query_tx_ids": tx_ids_arr[query_idx].astype(str),
        "query_roles": np.asarray(query_roles, dtype=str),
        "source_sample_indices": source_idx.astype(np.int64),
        "support_sample_indices": support_idx.astype(np.int64),
        "query_sample_indices": query_idx.astype(np.int64),
    }
    if role_arr is not None:
        arrays["query_dataset_roles"] = role_arr[query_idx].astype(str)
    for name, arr in metadata_arrays.items():
        arrays[f"source_{name}"] = arr[source_idx].astype(str)
        arrays[f"support_{name}"] = arr[support_idx].astype(str)
        arrays[f"query_{name}"] = arr[query_idx].astype(str)
    manifest = {
        "protocol": "CVS-SFE",
        "target_visibility": (
            "target_old_support_labeled_unknown_eval_only"
            if old_unknown_only
            else "new_class_support_labeled"
        ),
        "label_set_relation": (
            "Y_T_has_explicit_nonoverlap_tx"
            if old_unknown_only
            else "Y_T_has_unknown_new_tx"
        ),
        "tx_identity_key": "tx_ids",
        "source_tx_ids": source_txs,
        "target_old_tx_ids": target_old_txs,
        "new_tx_ids": new_txs,
        "unknown_tx_ids": unknown_txs,
        "source_label_map": source_label_map,
        "new_label_map": new_label_map,
        "unknown_label": UNKNOWN_LABEL,
        "overlap_audit": overlap_audit,
        "split_overlap_audit": split_overlap_audit,
        "split_indices_by_role": {
            name: [int(v) for v in values.tolist()]
            for name, values in split_indices.items()
        },
        "shots": int(shots),
        "source_proto_per_tx": int(source_proto_per_tx),
        "source_query_per_tx": int(source_query_per_tx),
        "target_old_support_per_tx": int(target_old_support_per_tx),
        "query_per_tx": int(query_per_tx),
        "seed": int(seed),
        "counts": {
            "source_features": int(arrays["source_features"].shape[0]),
            "target_old_support": int(split_indices["target_old_support"].shape[0]),
            "target_old_query": int(split_indices["target_old_query"].shape[0]),
            "support_features": int(arrays["support_features"].shape[0]),
            "query_features": int(arrays["query_features"].shape[0]),
            "query_unknown": int((query_labels == UNKNOWN_LABEL).sum()),
        },
    }
    if extra_metadata:
        manifest["extra_metadata"] = dict(extra_metadata)
    return SfePayload(arrays=arrays, manifest=manifest)
