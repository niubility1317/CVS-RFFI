"""Strict deployment bundle for aggregate slow/fast adapter state."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor

from .slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate


LEGACY_SLOW_FAST_BUNDLE_SCHEMA = "cvs.cached_slow_fast.v1"
SLOW_FAST_BUNDLE_SCHEMA = "cvs.cached_slow_fast.v2"
_METADATA_KEYS = frozenset(
    {
        "base_checkpoint_id",
        "class_ids",
        "prototypes",
        "support_logit_scale",
        "fast_step_size",
        "trust_radius",
    }
)
_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "candidate",
        "slow_u",
        "slow_v",
        "rho",
        "gamma",
        "beta",
        "direction_gate",
        "common_coeff",
        *_METADATA_KEYS,
    }
)


def _metadata(value: Mapping[str, Any], *, feature_dim: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("bundle metadata must be a string-keyed mapping")
    actual = frozenset(value)
    if actual != _METADATA_KEYS:
        raise ValueError(
            "bundle metadata allowlist mismatch: "
            f"missing={sorted(_METADATA_KEYS - actual)} extra={sorted(actual - _METADATA_KEYS)}"
        )
    base_id = value["base_checkpoint_id"]
    if not isinstance(base_id, str) or not base_id.strip():
        raise ValueError("base_checkpoint_id must be nonempty")
    class_ids = value["class_ids"]
    prototypes = value["prototypes"]
    if (
        not torch.is_tensor(class_ids)
        or class_ids.ndim != 1
        or class_ids.numel() < 1
        or class_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    ):
        raise ValueError("class_ids must be a nonempty integer vector")
    if torch.unique(class_ids).numel() != class_ids.numel():
        raise ValueError("class_ids must be unique")
    if (
        not torch.is_tensor(prototypes)
        or prototypes.ndim != 2
        or tuple(prototypes.shape) != (int(class_ids.numel()), int(feature_dim))
        or not prototypes.is_floating_point()
        or not bool(torch.isfinite(prototypes).all())
    ):
        raise ValueError("prototypes must align with class_ids and adapter feature width")
    if bool((torch.linalg.vector_norm(prototypes.float(), dim=1) <= 0).any()):
        raise ValueError("prototype rows must be nonzero")
    numeric: dict[str, float] = {}
    for key in ("support_logit_scale", "fast_step_size", "trust_radius"):
        number = float(value[key])
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
        numeric[key] = number
    return {
        "base_checkpoint_id": base_id,
        "class_ids": class_ids.detach().cpu().clone().long(),
        "prototypes": prototypes.detach().cpu().clone().float(),
        **numeric,
    }


def _cpu(value: Tensor | None) -> Tensor | None:
    return None if value is None else value.detach().cpu().clone()


def save_slow_fast_bundle(
    path: str | Path,
    state: SlowFastAdapterState,
    metadata: Mapping[str, Any],
) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"slow-fast bundle already exists: {output}")
    if not isinstance(state, SlowFastAdapterState):
        raise TypeError("state must be SlowFastAdapterState")
    validated = _metadata(metadata, feature_dim=state.feature_dim)
    payload = {
        "schema": SLOW_FAST_BUNDLE_SCHEMA,
        "candidate": state.candidate.value,
        "slow_u": _cpu(state.slow_u),
        "slow_v": _cpu(state.slow_v),
        "rho": float(state.rho),
        "gamma": _cpu(state.gamma),
        "beta": _cpu(state.beta),
        "direction_gate": _cpu(state.direction_gate),
        "common_coeff": _cpu(state.common_coeff),
        **validated,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(payload, handle)


def load_slow_fast_bundle_strict(
    path: str | Path,
) -> tuple[SlowFastAdapterState, Mapping[str, Any]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"slow-fast bundle is not a regular file: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load slow-fast bundle: {source}") from exc
    if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
        raise ValueError("slow-fast bundle must be a string-keyed mapping")
    actual = frozenset(payload)
    if actual != _PAYLOAD_KEYS:
        raise ValueError(
            "slow-fast bundle field mismatch: "
            f"missing={sorted(_PAYLOAD_KEYS - actual)} extra={sorted(actual - _PAYLOAD_KEYS)}"
        )
    source_schema = payload["schema"]
    if source_schema not in (LEGACY_SLOW_FAST_BUNDLE_SCHEMA, SLOW_FAST_BUNDLE_SCHEMA):
        raise ValueError("slow-fast bundle schema mismatch")
    candidate = SlowFastCandidate(payload["candidate"])
    direction_gate = payload["direction_gate"]
    if (
        source_schema == LEGACY_SLOW_FAST_BUNDLE_SCHEMA
        and candidate is SlowFastCandidate.FAST_LOWRANK_R8
    ):
        legacy_activation = torch.sigmoid(direction_gate.float()).clamp(
            min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6
        )
        direction_gate = torch.atanh(legacy_activation)
    state = SlowFastAdapterState(
        candidate=candidate,
        slow_u=payload["slow_u"],
        slow_v=payload["slow_v"],
        rho=float(payload["rho"]),
        gamma=payload["gamma"],
        beta=payload["beta"],
        direction_gate=direction_gate,
        common_coeff=payload["common_coeff"],
    )
    audit = _metadata(
        {key: payload[key] for key in _METADATA_KEYS},
        feature_dim=state.feature_dim,
    )
    audit["schema"] = SLOW_FAST_BUNDLE_SCHEMA
    audit["source_schema"] = source_schema
    audit["direction_gate_semantics"] = "signed_tanh_zero_centered"
    audit["candidate"] = state.candidate.value
    audit["feature_dim"] = state.feature_dim
    audit["rank"] = state.rank
    audit["fast_parameter_count"] = state.fast_parameter_count
    return state, MappingProxyType(copy.deepcopy(audit))


__all__ = [
    "LEGACY_SLOW_FAST_BUNDLE_SCHEMA",
    "SLOW_FAST_BUNDLE_SCHEMA",
    "load_slow_fast_bundle_strict",
    "save_slow_fast_bundle",
]
