"""Validated source-only feature cache for Phase1.5 slow-parameter training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


_ALLOWED_VIEWS = frozenset(
    {"clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}
)


def _sequence(value: Sequence[object], *, name: str, rows: int) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a row-aligned sequence")
    result = tuple(value)
    if len(result) != rows:
        raise ValueError(f"{name} must align with feature rows")
    return result


@dataclass(frozen=True)
class GroundFeatureCache:
    """Ground-only sample rows; this type is never accepted by Phase2 APIs."""

    features: Tensor
    labels: Tensor
    receivers: tuple[object, ...]
    days: tuple[object, ...]
    scenes: tuple[str, ...]
    physical_sample_ids: tuple[str, ...]
    views: tuple[str, ...]
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not torch.is_tensor(self.features)
            or self.features.ndim != 2
            or self.features.shape[0] < 1
            or self.features.shape[1] < 1
            or not self.features.is_floating_point()
            or not bool(torch.isfinite(self.features).all())
        ):
            raise ValueError("features must be a finite nonempty floating matrix")
        rows = int(self.features.shape[0])
        if (
            not torch.is_tensor(self.labels)
            or self.labels.ndim != 1
            or self.labels.numel() != rows
            or self.labels.dtype
            not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        ):
            raise ValueError("labels must be a row-aligned integer vector")
        receivers = _sequence(self.receivers, name="receivers", rows=rows)
        days = _sequence(self.days, name="days", rows=rows)
        scenes = tuple(str(value) for value in _sequence(self.scenes, name="scenes", rows=rows))
        physical_ids = tuple(
            str(value)
            for value in _sequence(
                self.physical_sample_ids, name="physical_sample_ids", rows=rows
            )
        )
        views = tuple(str(value) for value in _sequence(self.views, name="views", rows=rows))
        roles = tuple(str(value) for value in _sequence(self.roles, name="roles", rows=rows))
        if any(role != "L_s" for role in roles):
            raise ValueError("Phase1.5 training cache accepts L_s source rows only")
        if any(not value for value in physical_ids):
            raise ValueError("physical_sample_ids must be nonempty")
        if any(view not in _ALLOWED_VIEWS for view in views):
            raise ValueError("cache view is outside the frozen clean/LEO allowlist")
        row_keys = tuple(zip(physical_ids, views))
        if len(set(row_keys)) != len(row_keys):
            raise ValueError("physical_sample_id and view pairs must be unique")

        object.__setattr__(self, "features", self.features.detach().clone().float())
        object.__setattr__(self, "labels", self.labels.detach().clone().long())
        object.__setattr__(self, "receivers", receivers)
        object.__setattr__(self, "days", days)
        object.__setattr__(self, "scenes", scenes)
        object.__setattr__(self, "physical_sample_ids", physical_ids)
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "roles", roles)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])


__all__ = ["GroundFeatureCache"]
