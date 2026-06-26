from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def train_receiver_indices(split_info: Mapping[str, Any] | None) -> list[int]:
    """Return the source receivers used for training, in paper-domain order."""

    if not split_info:
        return []
    for key in ("train_rxs_idx", "train_rx_idx", "train_receivers_idx"):
        values = split_info.get(key)
        if values is not None:
            return [int(v) for v in values]
    return []


def train_receiver_count(split_info: Mapping[str, Any] | None, fallback: int) -> int:
    indices = train_receiver_indices(split_info)
    return len(indices) if indices else int(fallback)


def compact_receiver_targets(
    raw_receiver: torch.Tensor,
    split_info: Mapping[str, Any] | None,
    *,
    allowed_receivers: Sequence[int] | None = None,
) -> torch.Tensor:
    """Map raw WiSig receiver ids to compact source-domain labels.

    DRIFT/RIEI/DANN/MTL receiver-domain classifiers are trained only on the
    paper's source receivers. Their class space should therefore be
    ``0..n_source_receivers-1`` instead of the global WiSig receiver ids.
    """

    raw = raw_receiver.to(dtype=torch.long)
    source_receivers = [int(v) for v in (allowed_receivers or train_receiver_indices(split_info))]
    if not source_receivers:
        return raw

    mapping = {rx: i for i, rx in enumerate(source_receivers)}
    unique_raw = torch.unique(raw.detach().cpu()).tolist()
    missing = sorted(int(v) for v in unique_raw if int(v) not in mapping)
    if missing:
        raise ValueError(f"Receiver ids {missing} are not in training receivers {source_receivers}.")

    compact = torch.empty_like(raw)
    for src, dst in mapping.items():
        compact = torch.where(raw == int(src), torch.full_like(raw, int(dst)), compact)
    return compact
