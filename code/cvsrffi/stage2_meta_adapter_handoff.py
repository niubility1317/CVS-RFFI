"""Frozen DA1_REG0 handoff for the existing Stage2-C registration chain."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .meta_adapter import ResidualMetaAdapter


HANDOFF_SCHEMA = "cvs.stage2.meta_adapter.handoff.v1"
_BINDING_KEYS = frozenset(
    {"state", "checkpoint_id", "bundle_id", "capsule_id", "split_id"}
)
_FORBIDDEN_BINDING_TOKENS = frozenset(
    {
        "query",
        "truth",
        "role",
        "optimizer",
        "gradient",
        "classifier",
        "newclass",
    }
)


class MetaAdapterHandoffError(ValueError):
    """Raised when a Stage2-C handoff would carry forbidden state."""


@dataclass(frozen=True)
class FrozenMetaAdapterHandoff:
    """Immutable adapter-only state; no query, truth or trainable head."""

    state: str
    _adapted_state: Mapping[str, Tensor]
    checkpoint_id: str | None
    bundle_id: str | None
    capsule_id: str
    split_id: str
    optimizer_state: None = None
    gradient_state: None = None
    new_class_support_consumed: bool = False
    schema: str = HANDOFF_SCHEMA

    def __post_init__(self) -> None:
        if self.state != "DA1_REG0":
            raise MetaAdapterHandoffError("handoff state must be DA1_REG0")
        if not isinstance(self._adapted_state, Mapping) or not self._adapted_state:
            raise MetaAdapterHandoffError("handoff adapted_state must be nonempty")
        if any(not isinstance(name, str) for name in self._adapted_state):
            raise MetaAdapterHandoffError("handoff adapter names must be strings")
        if any(_contains_forbidden(name) for name in self._adapted_state):
            raise MetaAdapterHandoffError("handoff adapted_state contains forbidden head/query/truth state")
        frozen_state: dict[str, Tensor] = {}
        for name, value in self._adapted_state.items():
            if not torch.is_tensor(value):
                raise MetaAdapterHandoffError(f"handoff state {name!r} must be a tensor")
            frozen_state[name] = value.detach().cpu().clone()
        object.__setattr__(self, "_adapted_state", MappingProxyType(frozen_state))
        if self.optimizer_state is not None or self.gradient_state is not None:
            raise MetaAdapterHandoffError("handoff cannot contain optimizer or gradient state")
        if self.new_class_support_consumed is not False:
            raise MetaAdapterHandoffError("DA1_REG0 handoff cannot consume new-class support")

    def to_payload(self) -> dict[str, Any]:
        """Return only the serializable Stage2-C handoff surface."""

        return {
            "schema": self.schema,
            "state": self.state,
            "checkpoint_id": self.checkpoint_id,
            "bundle_id": self.bundle_id,
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "adapted_state": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "values": value.tolist(),
                }
                for name, value in self._adapted_state.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias used by generic artifact writers."""

        return self.to_payload()

    @property
    def adapted_state(self) -> Mapping[str, Tensor]:
        """Return a deep snapshot so callers cannot mutate stored handoff state."""

        return MappingProxyType(
            {name: value.clone() for name, value in self._adapted_state.items()}
        )

    @property
    def adapter_state(self) -> Mapping[str, Tensor]:
        return self.adapted_state

    def write_json(self, output_path: str | Path) -> None:
        destination = Path(output_path)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"handoff output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_payload(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")


def _contains_forbidden(name: str) -> bool:
    path_segments = [part for part in re.split(r"[./\\]+", name.lower()) if part]
    for segment in path_segments:
        tokens = [part for part in re.split(r"[^a-z0-9]+", segment) if part]
        if any(token in _FORBIDDEN_BINDING_TOKENS for token in tokens):
            return True
        token_pairs = zip(tokens, tokens[1:])
        if any(pair in {("cls", "head"), ("new", "class")} for pair in token_pairs):
            return True
    return False


def _binding_mapping(binding: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(binding, Mapping):
        values = dict(binding)
    elif hasattr(binding, "__dict__"):
        values = dict(vars(binding))
    else:
        raise MetaAdapterHandoffError("binding must be a mapping or attribute object")
    if any(not isinstance(key, str) for key in values):
        raise MetaAdapterHandoffError("handoff binding keys must be strings")
    if set(values) - _BINDING_KEYS:
        forbidden = sorted(set(values) - _BINDING_KEYS)
        raise MetaAdapterHandoffError(
            f"handoff binding allowlist mismatch: unexpected={forbidden}"
        )
    if any(_contains_forbidden(key) for key in values):
        raise MetaAdapterHandoffError("handoff binding contains query/truth or trainable-head state")
    if values.get("state", "DA1_REG0") != "DA1_REG0":
        raise MetaAdapterHandoffError("handoff binding state must be DA1_REG0")
    for key in ("capsule_id", "split_id"):
        if not isinstance(values.get(key), str) or not values[key].strip():
            raise MetaAdapterHandoffError(f"handoff binding {key} must be nonempty")
    for key in ("checkpoint_id", "bundle_id"):
        if key in values and values[key] is not None:
            if not isinstance(values[key], str) or not values[key].strip():
                raise MetaAdapterHandoffError(f"handoff binding {key} must be nonempty")
    if not values.get("checkpoint_id") and not values.get("bundle_id"):
        raise MetaAdapterHandoffError("handoff binding requires checkpoint_id or bundle_id")
    return values


def _adapter_parameter_state(model: nn.Module) -> dict[str, Tensor]:
    prefixes = tuple(
        f"{name}." if name else ""
        for name, module in model.named_modules()
        if isinstance(module, ResidualMetaAdapter)
    )
    if not prefixes:
        raise MetaAdapterHandoffError("model has no ResidualMetaAdapter state")
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    }
    if not state:
        raise MetaAdapterHandoffError("model has no adapter parameters to hand off")
    return state


def freeze_da1_reg0_handoff(
    model: nn.Module,
    binding: Mapping[str, Any] | Any,
) -> FrozenMetaAdapterHandoff:
    """Freeze a model and return only its adapter parameters and frozen handles."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    values = _binding_mapping(binding)
    adapted_state = _adapter_parameter_state(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return FrozenMetaAdapterHandoff(
        state="DA1_REG0",
        _adapted_state=adapted_state,
        checkpoint_id=values.get("checkpoint_id"),
        bundle_id=values.get("bundle_id"),
        capsule_id=values["capsule_id"],
        split_id=values["split_id"],
    )


__all__ = [
    "FrozenMetaAdapterHandoff",
    "HANDOFF_SCHEMA",
    "MetaAdapterHandoffError",
    "freeze_da1_reg0_handoff",
]
