from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch


def _json_size_bytes(data: Mapping[str, Any]) -> int:
    return len(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


def _to_plain_number(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return [float(v) for v in value.detach().cpu().flatten().tolist()]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [float(v) if isinstance(v, (int, float)) else v for v in value]
    return value


def style_code_from_stats(stats: Mapping[str, Any], dim: int = 8) -> tuple[float, ...]:
    """Build a bounded deterministic low-dimensional code from RF style stats."""
    size = max(1, int(dim))
    vals: list[float] = []
    for key in sorted((stats or {}).keys()):
        raw = stats.get(key, 0.0)
        if isinstance(raw, (list, tuple)):
            raw_values = raw
        else:
            raw_values = (raw,)
        for item in raw_values:
            try:
                value = float(item)
            except (TypeError, ValueError):
                value = 0.0
            if not math.isfinite(value):
                value = 0.0
            vals.append(math.tanh(value / _style_code_scale(str(key))))
    if not vals:
        vals = [0.0]
    while len(vals) < size:
        vals.extend(vals)
    return tuple(float(max(-1.0, min(1.0, v))) for v in vals[:size])


def _style_code_scale(key: str) -> float:
    key = str(key)
    if key.endswith("_hz"):
        return 35000.0
    if key.endswith("_ppm"):
        return 150.0
    if key.endswith("_db"):
        return 8.0
    if key.endswith("_deg"):
        return 3.0
    if "snr" in key:
        return 80.0
    return 4.0


@dataclass(frozen=True)
class StylePacket:
    """Class-marginalized RF style statistics uploaded by one FL client."""

    client_id: str
    round_idx: int
    count: int
    stats: Mapping[str, Any]
    style_id: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    style_code: Optional[Sequence[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": str(self.client_id),
            "round_idx": int(self.round_idx),
            "count": int(self.count),
            "style_id": None if self.style_id is None else int(self.style_id),
            "stats": {str(k): _to_plain_number(v) for k, v in self.stats.items()},
            "metadata": {str(k): _to_plain_number(v) for k, v in self.metadata.items()},
            "style_code": None if self.style_code is None else [float(v) for v in self.style_code],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StylePacket":
        return cls(
            client_id=str(data.get("client_id", "")),
            round_idx=int(data.get("round_idx", 0)),
            count=int(data.get("count", 0)),
            style_id=None if data.get("style_id") is None else int(data.get("style_id")),
            stats=dict(data.get("stats", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
            style_code=None if data.get("style_code") is None else tuple(float(v) for v in data.get("style_code", []) or []),
        )

    def vector(self, keys: Optional[Sequence[str]] = None) -> torch.Tensor:
        if keys is not None and "__style_code__" in set(keys) and self.style_code is not None:
            return torch.tensor([float(v) for v in self.style_code], dtype=torch.float32)
        selected = list(keys) if keys is not None else sorted(self.stats.keys())
        vals: list[float] = []
        for key in selected:
            raw = self.stats.get(key, 0.0)
            if isinstance(raw, (list, tuple)):
                vals.extend(float(v) for v in raw)
            else:
                try:
                    vals.append(float(raw))
                except (TypeError, ValueError):
                    vals.append(0.0)
        if not vals:
            vals = [0.0]
        return torch.tensor(vals, dtype=torch.float32)

    def size_bytes(self) -> int:
        return _json_size_bytes(self.to_dict())


@dataclass(frozen=True)
class StyleDomainBatch:
    """A local batch rebuilt with explicit constructed style-domain labels."""

    x: torch.Tensor
    y: torch.Tensor
    d_raw: Optional[torch.Tensor]
    d_style: torch.Tensor
    sources: tuple[str, ...] = ("clean",)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_style_domains(self) -> int:
        valid = self.d_style.view(-1).long()
        valid = valid[valid >= 0]
        return int(torch.unique(valid).numel()) if valid.numel() else 0

    @property
    def batch_size(self) -> int:
        return int(self.y.numel())
