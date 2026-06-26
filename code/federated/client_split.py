from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset


_ALIASES = {
    "tx_id": ("tx_id", "tx_i", "label", "y", "tx"),
    "rx_id": ("rx_id", "rx_i", "receiver", "rx"),
    "day_id": ("day_id", "day_i", "day", "capture_day"),
    "channel_view": ("channel_view", "channel", "sat_scenario", "scenario", "view"),
}


def _get_attr_or_key(obj: Any, names: Sequence[str]):
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _unwrap_subset_index(dataset, idx: int):
    if hasattr(dataset, "selected") and hasattr(dataset, "base"):
        selected = getattr(dataset, "selected")
        return getattr(dataset, "base"), int(selected[int(idx)])
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return getattr(dataset, "dataset"), int(getattr(dataset, "indices")[int(idx)])
    return dataset, int(idx)


def get_sample_metadata(dataset, idx: int) -> Dict[str, Any]:
    """Read metadata from WiSig-style index entries or sample meta dictionaries."""
    base, local_idx = _unwrap_subset_index(dataset, idx)
    meta: Dict[str, Any] = {}

    index = getattr(base, "index", None)
    if index is not None:
        item = index[int(local_idx)]
        for canonical, aliases in _ALIASES.items():
            value = _get_attr_or_key(item, aliases)
            if value is not None:
                meta[canonical] = value

    if len(meta) < 3:
        sample = dataset[int(idx)]
        if isinstance(sample, Mapping):
            sample_meta = sample.get("meta", sample)
        elif isinstance(sample, (tuple, list)) and len(sample) >= 4 and isinstance(sample[3], Mapping):
            sample_meta = sample[3]
        else:
            sample_meta = {}
        for canonical, aliases in _ALIASES.items():
            value = _get_attr_or_key(sample_meta, aliases)
            if value is not None:
                meta[canonical] = value

    return meta


def _require(meta: Mapping[str, Any], key: str, client_key: str, idx: int):
    if key not in meta:
        raise KeyError(f"Sample {idx} lacks metadata field {key!r} required by fl_client_key={client_key!r}")
    return meta[key]


def infer_client_id(meta: Mapping[str, Any], client_key: str) -> str:
    normalized = str(client_key or "receiver_day").lower()
    rx = _get_attr_or_key(meta, _ALIASES["rx_id"])
    day = _get_attr_or_key(meta, _ALIASES["day_id"])
    channel = _get_attr_or_key(meta, _ALIASES["channel_view"])

    if normalized == "receiver":
        if rx is None:
            raise KeyError("receiver client requires rx_id/rx_i metadata")
        return f"rx{rx}"
    if normalized == "receiver_day":
        if rx is None or day is None:
            raise KeyError("receiver_day client requires rx_id/rx_i and day_id/day_i metadata")
        return f"rx{rx}_day{day}"
    if normalized == "receiver_channel":
        if rx is None or channel is None:
            raise KeyError("receiver_channel client requires rx_id/rx_i and channel_view metadata")
        return f"rx{rx}_ch{channel}"
    if normalized == "receiver_day_channel":
        if rx is None or day is None or channel is None:
            raise KeyError("receiver_day_channel client requires rx_id/rx_i, day_id/day_i, and channel_view metadata")
        return f"rx{rx}_day{day}_ch{channel}"
    raise ValueError("client_key must be one of: receiver, receiver_day, receiver_channel, receiver_day_channel")


def build_client_splits(
    dataset,
    client_key: str,
    *,
    min_samples_per_client: int = 1,
    drop_small: bool = False,
    verbose: bool = False,
) -> Dict[str, list[int]]:
    splits: MutableMapping[str, list[int]] = OrderedDict()
    summaries: Dict[str, Dict[str, Counter]] = {}
    for idx in range(len(dataset)):
        meta = get_sample_metadata(dataset, idx)
        cid = infer_client_id(meta, client_key)
        splits.setdefault(cid, []).append(idx)
        summary = summaries.setdefault(cid, {"tx": Counter(), "rx": Counter(), "day": Counter(), "channel": Counter()})
        for label, aliases in [("tx", _ALIASES["tx_id"]), ("rx", _ALIASES["rx_id"]), ("day", _ALIASES["day_id"]), ("channel", _ALIASES["channel_view"])]:
            value = _get_attr_or_key(meta, aliases)
            if value is not None:
                summary[label][str(value)] += 1

    min_samples = max(1, int(min_samples_per_client))
    small = {cid: ids for cid, ids in splits.items() if len(ids) < min_samples}
    if small and drop_small:
        for cid in list(small.keys()):
            splits.pop(cid, None)
    elif small and verbose:
        print(f"[FED-CLIENT-WARN] keeping {len(small)} clients below min_samples_per_client={min_samples}: {sorted(small)}", flush=True)

    if verbose:
        for cid, ids in splits.items():
            summary = summaries.get(cid, {})
            tx_count = len(summary.get("tx", {}))
            print(
                f"[FED-CLIENT] {cid}: samples={len(ids)} tx_classes={tx_count} "
                f"rx={dict(summary.get('rx', {}))} day={dict(summary.get('day', {}))} "
                f"channel={dict(summary.get('channel', {}))}",
                flush=True,
            )
    return dict(splits)


def build_client_loaders(
    dataset,
    client_splits: Mapping[str, Sequence[int]],
    batch_size: int,
    num_workers: int = 0,
    sampler_cfg: Mapping[str, Any] | None = None,
):
    cfg = dict(sampler_cfg or {})
    shuffle = bool(cfg.get("shuffle", True))
    drop_last = bool(cfg.get("drop_last", False))
    pin_memory = bool(cfg.get("pin_memory", torch.cuda.is_available()))
    loaders = OrderedDict()
    for cid, indices in client_splits.items():
        loaders[cid] = DataLoader(
            Subset(dataset, list(indices)),
            batch_size=int(batch_size),
            shuffle=shuffle,
            num_workers=int(num_workers),
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(int(num_workers) > 0),
        )
    return loaders


def summarize_client_splits(dataset, client_splits: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    return {
        "num_clients": len(client_splits),
        "client_num_samples": {str(cid): len(indices) for cid, indices in client_splits.items()},
        "total_samples": int(sum(len(v) for v in client_splits.values())),
    }
