"""Atomic support-conditioned MARC-OT weight initialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .meta_support_set_encoder import SupportDomainState
from .meta_weight_bank import WEIGHT_DELTA_BANK_SCHEMA, WeightDeltaBank, compose_weight_delta, parameter_block_key


@dataclass(frozen=True)
class CalibratedWeightPlan:
    """A complete independent model state or an exact atomic base-state fallback."""

    state_dict: dict[str, Tensor]
    applied: bool
    reason: str
    uncertainty: float
    block_gates: tuple[float, ...]
    block_lrs: tuple[float, ...]


def _clone_state(base_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    copied: dict[str, Tensor] = {}
    for name, value in base_state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise ValueError("base state must map string names to tensors")
        copied[name] = value.clone()
    return copied


def _safe_metadata(state: object) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    if not isinstance(state, SupportDomainState):
        return 1.0, (), ()
    try:
        uncertainty = float(state.uncertainty.detach().cpu().item())
        gates = tuple(float(value) for value in state.block_gates.detach().cpu().reshape(-1))
        lrs = tuple(float(value) for value in state.block_lrs.detach().cpu().reshape(-1))
    except (AttributeError, RuntimeError, ValueError):
        return 1.0, (), ()
    return uncertainty, gates, lrs


def _validate_state_for_bank(state: SupportDomainState, bank: WeightDeltaBank) -> None:
    if not isinstance(state, SupportDomainState):
        raise ValueError("support state has the wrong type")
    values = (state.q, state.uncertainty, state.block_gates, state.block_lrs)
    if any(not isinstance(value, Tensor) or not value.is_floating_point() for value in values):
        raise ValueError("support state must contain floating-point tensors")
    if state.q.ndim != 1 or state.uncertainty.numel() != 1:
        raise ValueError("support-state coefficient or uncertainty geometry drift")
    block_count = len(bank.entries)
    if state.block_gates.shape != (block_count,) or state.block_lrs.shape != (block_count,):
        raise ValueError("support-state block geometry drift")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("non-finite support state is not allowed")
    uncertainty = float(state.uncertainty.detach().cpu().item())
    if not 0.0 <= uncertainty <= 1.0:
        raise ValueError("uncertainty must be in [0, 1]")
    if bool((state.block_gates < 0.0).any()) or bool((state.block_gates > 1.0).any()):
        raise ValueError("block gates must be in [0, 1]")
    if bool((state.block_lrs <= 0.0).any()):
        raise ValueError("block learning rates must be positive")


def _validate_entry_geometry(entry: object, base_state: Mapping[str, Tensor], q: Tensor) -> None:
    spec = entry.spec
    if entry.effective_rank != q.numel():
        raise ValueError("bank coefficient geometry does not match every block rank")
    if not spec.parameter_names or not (
        len(spec.parameter_names) == len(spec.shapes) == len(spec.dtypes)
    ):
        raise ValueError("bank block geometry is malformed")
    for name, shape, dtype_name in zip(spec.parameter_names, spec.shapes, spec.dtypes, strict=True):
        block_name = parameter_block_key(name)
        if block_name is None or block_name != spec.name:
            raise ValueError("bank block contains a non-allowlisted parameter")
        base_value = base_state.get(name)
        if base_value is None or not base_value.is_floating_point():
            raise ValueError("bank parameter is missing from the floating base state")
        if tuple(base_value.shape) != tuple(shape) or str(base_value.dtype) != dtype_name:
            raise ValueError("bank and base parameter geometry differ")


def calibrate_weight_plan(
    base_state: Mapping[str, Tensor],
    base_checkpoint_id: str,
    bank: WeightDeltaBank,
    support_state: SupportDomainState,
) -> CalibratedWeightPlan:
    """Compose `(1-u) * gate * Bq` only after complete bank/state validation.

    Validation or composition failures return an independent, tensor-equal copy of
    ``base_state``.  No partial calibrated state is ever exposed.
    """

    base_copy = _clone_state(base_state)
    uncertainty, gates, lrs = _safe_metadata(support_state)

    def fallback(reason: str) -> CalibratedWeightPlan:
        return CalibratedWeightPlan(base_copy, False, reason, uncertainty, gates, lrs)

    try:
        if not isinstance(base_checkpoint_id, str) or not base_checkpoint_id:
            raise ValueError("base checkpoint identity is invalid")
        if not isinstance(bank, WeightDeltaBank) or bank.schema != WEIGHT_DELTA_BANK_SCHEMA:
            raise ValueError("weight bank schema is invalid")
        if bank.base_checkpoint_id != base_checkpoint_id:
            raise ValueError("base checkpoint identity does not match weight bank")
        if not bank.entries:
            raise ValueError("weight bank has no blocks")
        _validate_state_for_bank(support_state, bank)
        seen_names: set[str] = set()
        for entry in bank.entries:
            _validate_entry_geometry(entry, base_state, support_state.q)
            for name in entry.spec.parameter_names:
                if name in seen_names:
                    raise ValueError("weight bank assigns one parameter to multiple blocks")
                seen_names.add(name)

        candidate = _clone_state(base_state)
        uncertainty_value = float(support_state.uncertainty.detach().cpu().item())
        for index, entry in enumerate(bank.entries):
            composed = compose_weight_delta(entry, support_state.q)
            scale = (1.0 - uncertainty_value) * support_state.block_gates[index]
            if not bool(torch.isfinite(scale).all()):
                raise ValueError("non-finite block combination scale")
            for name, delta in composed.items():
                base_value = base_state[name]
                combined = base_value + scale.to(device=base_value.device, dtype=base_value.dtype) * delta.to(
                    device=base_value.device, dtype=base_value.dtype
                )
                if not bool(torch.isfinite(combined).all()):
                    raise ValueError("non-finite calibrated parameter")
                candidate[name] = combined
        return CalibratedWeightPlan(candidate, True, "applied", uncertainty, gates, lrs)
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        return fallback(str(error))


__all__ = ["CalibratedWeightPlan", "calibrate_weight_plan"]
