"""Balanced source-only proxy episodes for Phase1 MIRAGE-OWDG."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .protocol import (
    Phase1DataPolicy,
    Phase1PolicyError,
    ProxyRole,
    SourcePartition,
)


class ProxyProtocolError(ValueError):
    """Raised when a proxy episode would cross a Phase1 source-role boundary."""


@dataclass(frozen=True)
class ProxyEpisode:
    """One source proxy role assignment with the proxy class removed from registration."""

    proxy_class: int
    registered_class_mask: torch.Tensor
    registered_rows: torch.Tensor
    proxy_rows: torch.Tensor
    schedule_receipt: Mapping[str, int | str]


_DATA_POLICY = Phase1DataPolicy()
_SPLIT_PROXY_ORIGINS: Mapping[str, tuple[ProxyRole, SourcePartition]] = MappingProxyType(
    {
        "train_l": (ProxyRole.PROXY_TRAIN, SourcePartition.L_S),
        "val_cal": (ProxyRole.P_CAL, SourcePartition.V_CAL),
        "val_select": (ProxyRole.P_SELECT, SourcePartition.V_SELECT),
    }
)
_INTEGER_LABEL_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _require_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProxyProtocolError(f"{name} must be an integer")
    return int(value)


def _validated_class_ids(class_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(class_ids, (str, bytes)):
        raise ProxyProtocolError("class_ids must be an integer sequence")
    try:
        raw_class_ids = tuple(class_ids)
    except TypeError as error:
        raise ProxyProtocolError("class_ids must be an integer sequence") from error

    normalized = tuple(_require_integer(class_id, name="class ID") for class_id in raw_class_ids)
    if any(class_id < 0 for class_id in normalized):
        raise ProxyProtocolError("class IDs must be non-negative")
    ordered = tuple(sorted(set(normalized)))
    if tuple(range(len(ordered))) != ordered:
        raise ProxyProtocolError("class IDs must be contiguous starting at 0")
    if len(ordered) < 3:
        raise ProxyProtocolError("proxy episode requires at least three classes")
    return ordered


def _validated_labels(labels: torch.Tensor) -> tuple[int, ...]:
    if not isinstance(labels, torch.Tensor):
        raise ProxyProtocolError("labels must be a torch.Tensor")
    if labels.ndim != 1:
        raise ProxyProtocolError("labels must be one-dimensional")
    if labels.numel() == 0:
        raise ProxyProtocolError("labels must be non-empty")
    if labels.dtype not in _INTEGER_LABEL_DTYPES:
        raise ProxyProtocolError("labels must use an integer dtype")
    return _validated_class_ids(tuple(int(label) for label in labels.detach().cpu().tolist()))


def _require_approved_split_role(split_role: str) -> None:
    try:
        proxy_role, source_partition = _SPLIT_PROXY_ORIGINS[split_role]
    except (KeyError, TypeError) as error:
        allowed = ", ".join(_SPLIT_PROXY_ORIGINS)
        raise ProxyProtocolError(f"split_role must be one of: {allowed}") from error
    try:
        _DATA_POLICY.require_proxy_origin(proxy_role, source_partition)
    except Phase1PolicyError as error:
        raise ProxyProtocolError(str(error)) from error


def proxy_class_for_episode(class_ids: Sequence[int], *, seed: int, episode_index: int) -> int:
    """Choose one proxy class using the approved deterministic balanced cycle."""

    ordered = _validated_class_ids(class_ids)
    normalized_seed = _require_integer(seed, name="seed")
    normalized_episode_index = _require_integer(episode_index, name="episode_index")
    if normalized_episode_index < 0:
        raise ProxyProtocolError("episode_index must be non-negative")
    offset = int(hashlib.sha256(f"{normalized_seed}:proxy".encode("utf-8")).hexdigest(), 16) % len(ordered)
    return ordered[(offset + normalized_episode_index) % len(ordered)]


def build_proxy_episode(
    labels: torch.Tensor,
    *,
    split_role: str,
    seed: int,
    episode_index: int,
) -> ProxyEpisode:
    """Create one role-safe proxy episode from a labeled source batch."""

    _require_approved_split_role(split_role)
    class_ids = _validated_labels(labels)
    proxy_class = proxy_class_for_episode(
        class_ids,
        seed=seed,
        episode_index=episode_index,
    )

    registered_class_mask = torch.ones(
        len(class_ids),
        dtype=torch.bool,
        device=labels.device,
    )
    registered_class_mask[proxy_class] = False
    proxy_rows = torch.nonzero(labels == proxy_class, as_tuple=False).flatten()
    registered_rows = torch.nonzero(labels != proxy_class, as_tuple=False).flatten()
    schedule_receipt = MappingProxyType(
        {
            "class_count": len(class_ids),
            "proxy_row_count": int(proxy_rows.numel()),
            "registered_row_count": int(registered_rows.numel()),
            "split_role": split_role,
            "seed": _require_integer(seed, name="seed"),
            "episode_index": _require_integer(episode_index, name="episode_index"),
        }
    )
    return ProxyEpisode(
        proxy_class=proxy_class,
        registered_class_mask=registered_class_mask,
        registered_rows=registered_rows,
        proxy_rows=proxy_rows,
        schedule_receipt=schedule_receipt,
    )
