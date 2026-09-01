"""Atomic support-conditioned MARC-OT weight initialization."""

from __future__ import annotations

import math
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


def _validate_lr_bounds(lr_min: float, lr_max: float) -> None:
    if (
        not isinstance(lr_min, (int, float))
        or isinstance(lr_min, bool)
        or not isinstance(lr_max, (int, float))
        or isinstance(lr_max, bool)
        or not math.isfinite(lr_min)
        or not math.isfinite(lr_max)
        or lr_min <= 0.0
        or lr_min >= lr_max
    ):
        raise ValueError("learning-rate bounds must be finite, positive and ordered")


def _fallback_metadata(
    bank: object, lr_min: object, lr_max: object
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    block_count = len(bank.entries) if isinstance(bank, WeightDeltaBank) else 0
    safe_lr = 1e-4
    if isinstance(lr_min, (int, float)) and not isinstance(lr_min, bool) and isinstance(lr_max, (int, float)) and not isinstance(lr_max, bool):
        if math.isfinite(lr_min) and math.isfinite(lr_max) and 0.0 < lr_min < lr_max:
            safe_lr = float(lr_min)
    return 1.0, (0.0,) * block_count, (safe_lr,) * block_count


def _validate_base_finite(base_state: Mapping[str, Tensor]) -> None:
    for name, value in base_state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite base tensor: {name!r}")


def _validate_state_for_bank(
    state: SupportDomainState, bank: WeightDeltaBank, *, lr_min: float, lr_max: float
) -> None:
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
    expected_q_size = sum(entry.effective_rank for entry in bank.entries)
    if state.q.numel() != expected_q_size:
        raise ValueError("support-state coefficient geometry does not match bank block ranks")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("non-finite support state is not allowed")
    uncertainty = float(state.uncertainty.detach().cpu().item())
    if not 0.0 <= uncertainty <= 1.0:
        raise ValueError("uncertainty must be in [0, 1]")
    if bool((state.block_gates < 0.0).any()) or bool((state.block_gates > 1.0).any()):
        raise ValueError("block gates must be in [0, 1]")
    if bool((state.block_lrs < lr_min).any()) or bool((state.block_lrs > lr_max).any()):
        raise ValueError("block learning rates are outside frozen bounds")


def _validate_entry_geometry(entry: object, base_state: Mapping[str, Tensor]) -> None:
    spec = entry.spec
    if not isinstance(entry.effective_rank, int) or isinstance(entry.effective_rank, bool) or entry.effective_rank < 0:
        raise ValueError("bank effective rank is invalid")
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
    *,
    lr_min: float,
    lr_max: float,
) -> CalibratedWeightPlan:
    """Compose `(1-u) * gate * Bq` only after complete bank/state validation.

    Validation or composition failures return an independent, tensor-equal copy of
    ``base_state``.  No partial calibrated state is ever exposed.
    """

    base_copy = _clone_state(base_state)
    fallback_uncertainty, fallback_gates, fallback_lrs = _fallback_metadata(bank, lr_min, lr_max)

    def fallback(reason: str) -> CalibratedWeightPlan:
        return CalibratedWeightPlan(
            base_copy,
            False,
            reason,
            fallback_uncertainty,
            fallback_gates,
            fallback_lrs,
        )

    try:
        if not isinstance(base_checkpoint_id, str) or not base_checkpoint_id:
            raise ValueError("base checkpoint identity is invalid")
        _validate_lr_bounds(lr_min, lr_max)
        if not isinstance(bank, WeightDeltaBank) or bank.schema != WEIGHT_DELTA_BANK_SCHEMA:
            raise ValueError("weight bank schema is invalid")
        if bank.base_checkpoint_id != base_checkpoint_id:
            raise ValueError("base checkpoint identity does not match weight bank")
        if not bank.entries:
            raise ValueError("weight bank has no blocks")
        _validate_base_finite(base_state)
        _validate_state_for_bank(support_state, bank, lr_min=lr_min, lr_max=lr_max)
        seen_names: set[str] = set()
        seen_block_names: set[str] = set()
        for entry in bank.entries:
            if entry.spec.name in seen_block_names:
                raise ValueError("weight bank contains a duplicate block name")
            seen_block_names.add(entry.spec.name)
            _validate_entry_geometry(entry, base_state)
            for name in entry.spec.parameter_names:
                if name in seen_names:
                    raise ValueError("weight bank assigns one parameter to multiple blocks")
                seen_names.add(name)

        candidate = _clone_state(base_state)
        uncertainty_value = float(support_state.uncertainty.detach().cpu().item())
        offset = 0
        for index, entry in enumerate(bank.entries):
            next_offset = offset + entry.effective_rank
            q_block = support_state.q[offset:next_offset]
            if q_block.numel() != entry.effective_rank:
                raise ValueError("support-state coefficient segment geometry drift")
            composed = compose_weight_delta(entry, q_block)
            offset = next_offset
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
        uncertainty = uncertainty_value
        gates = tuple(float(value) for value in support_state.block_gates.detach().cpu())
        lrs = tuple(float(value) for value in support_state.block_lrs.detach().cpu())
        return CalibratedWeightPlan(candidate, True, "applied", uncertainty, gates, lrs)
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        return fallback(str(error))


__all__ = ["CalibratedWeightPlan", "calibrate_weight_plan"]
