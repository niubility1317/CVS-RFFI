"""Stable, blockwise low-rank weight deltas for MARC-OT."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


WEIGHT_DELTA_BANK_SCHEMA = "cvs.marc_ot.weight_delta_bank.v1"


@dataclass(frozen=True)
class BlockSpec:
    """Immutable geometry for one independently-composed parameter block."""

    name: str
    parameter_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]


@dataclass(frozen=True)
class DeltaTaskKey:
    """Stable source-domain episode identity for a bank row."""

    receiver: str
    day: str
    scene: str
    k_shot: int
    capture_block: str = "aggregate"


@dataclass(frozen=True)
class DeltaBankEntry:
    """One block's FP32 basis and one coefficient row per task key."""

    spec: BlockSpec
    basis: Tensor
    task_coefficients: Tensor
    effective_rank: int
    relative_error: float


@dataclass(frozen=True)
class WeightDeltaBank:
    """Frozen collection of blockwise source-domain delta bases."""

    schema: str
    base_checkpoint_id: str
    task_keys: tuple[DeltaTaskKey, ...]
    entries: tuple[DeltaBankEntry, ...]


_BLOCK_PREFIXES: tuple[tuple[str, str], ...] = (
    ("id_backbone.t1.", "t1"),
    ("id_backbone.t2.", "t2"),
    ("id_backbone.t3.", "t3"),
    ("id_backbone.f1.", "f1"),
    ("id_backbone.f2.", "f2"),
    ("id_backbone.f3.", "f3"),
    ("id_backbone.time_projection.", "time_projection"),
    ("id_backbone.time_proj.", "time_projection"),
    ("id_backbone.frequency_projection.", "frequency_projection"),
    ("id_backbone.f_proj.", "frequency_projection"),
    ("id_backbone.freq_stats_proj.", "frequency_projection"),
    ("id_backbone.fusion.", "fusion"),
    ("id_backbone.time_fuse.", "fusion"),
    ("id_backbone.freq_gate.", "fusion"),
    ("id_backbone.identity_mapping.", "identity_mapping"),
    ("identity_mapping.", "identity_mapping"),
)


def parameter_block_key(parameter_name: str) -> str | None:
    """Return the canonical MARC-OT block for an allowed parameter name.

    This is the single routing function shared by all MARC-OT components;
    names outside the explicit identity-backbone allowlist return ``None``.
    """

    name = str(parameter_name)
    for prefix, block_name in _BLOCK_PREFIXES:
        if name.startswith(prefix):
            return block_name
    return None


def _validate_explicit_prefixes(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    if not prefixes or any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
        raise ValueError("prefixes must be a non-empty tuple of explicit prefixes")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("prefixes must not contain duplicates")
    if any(parameter_block_key(f"{prefix}__probe__") is None for prefix in prefixes):
        raise ValueError("prefixes must target canonical parameter blocks")
    return prefixes


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.device != right.device:
        return False
    return torch.equal(
        left.detach().contiguous().view(torch.uint8),
        right.detach().contiguous().view(torch.uint8),
    )


def _is_allowlisted(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def extract_block_delta(
    base_state: Mapping[str, Tensor],
    adapted_state: Mapping[str, Tensor],
    *,
    prefixes: tuple[str, ...],
) -> dict[str, Tensor]:
    """Extract only explicitly allowlisted floating deltas from two state dicts.

    Every non-allowlisted tensor, including buffers and the source classifier,
    must be bitwise identical. Geometry and dtype drift are rejected before a
    delta is computed, and non-finite deltas are never repaired silently.
    """

    prefixes = _validate_explicit_prefixes(prefixes)
    base_names = set(base_state)
    adapted_names = set(adapted_state)
    if base_names != adapted_names:
        raise ValueError("base and adapted state dictionaries must have identical keys")

    result: dict[str, Tensor] = {}
    for name in sorted(base_names):
        base_value = base_state[name]
        adapted_value = adapted_state[name]
        if not isinstance(base_value, Tensor) or not isinstance(adapted_value, Tensor):
            raise ValueError(f"state entry {name!r} must be a torch.Tensor")
        if base_value.shape != adapted_value.shape:
            raise ValueError(f"shape drift for {name!r}")
        if base_value.dtype != adapted_value.dtype:
            raise ValueError(f"dtype drift for {name!r}")

        if not _is_allowlisted(name, prefixes):
            if not _bitwise_equal(base_value, adapted_value):
                raise ValueError(f"unallowlisted tensor changed: {name!r}")
            continue
        if parameter_block_key(name) is None:
            raise ValueError(f"allowlisted tensor is outside canonical blocks: {name!r}")
        if not base_value.is_floating_point():
            raise ValueError(f"allowlisted tensor must be floating point: {name!r}")
        delta = adapted_value.detach() - base_value.detach()
        if not bool(torch.isfinite(delta).all()):
            raise ValueError(f"non-finite delta for {name!r}")
        result[name] = delta.clone()
    return result


def _task_sort_key(task_key: DeltaTaskKey) -> tuple[str, str, str, str, int]:
    return (
        task_key.receiver,
        task_key.day,
        task_key.capture_block,
        task_key.scene,
        task_key.k_shot,
    )


def _validate_task_key(task_key: object) -> DeltaTaskKey:
    if not isinstance(task_key, DeltaTaskKey):
        raise ValueError("task key must be a DeltaTaskKey")
    if any(
        not isinstance(value, str) or not value
        for value in (task_key.receiver, task_key.day, task_key.scene, task_key.capture_block)
    ):
        raise ValueError("task key receiver, day, scene and capture_block must be non-empty strings")
    if (
        not isinstance(task_key.k_shot, int)
        or isinstance(task_key.k_shot, bool)
        or task_key.k_shot <= 0
    ):
        raise ValueError("task key k_shot must be a positive integer")
    return task_key


def _validate_delta_mapping(delta: Mapping[str, Tensor]) -> tuple[str, ...]:
    names = tuple(sorted(delta))
    if not names:
        raise ValueError("each task delta must contain at least one parameter")
    for name in names:
        value = delta[name]
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"delta {name!r} must be a floating-point tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite delta for {name!r}")
        if parameter_block_key(name) is None:
            raise ValueError(f"delta parameter is outside canonical blocks: {name!r}")
    return names


def _relative_error(matrix: Tensor, basis: Tensor, coefficients: Tensor) -> float:
    denominator = torch.linalg.vector_norm(matrix)
    if float(denominator) == 0.0:
        return 0.0
    reconstruction = coefficients @ basis.T
    return float(torch.linalg.vector_norm(matrix - reconstruction) / denominator)


def _canonicalize_basis_sign(basis: Tensor, coefficients: Tensor) -> tuple[Tensor, Tensor]:
    basis = basis.clone()
    coefficients = coefficients.clone()
    for index in range(basis.shape[1]):
        column = basis[:, index]
        nonzero = torch.nonzero(column != 0, as_tuple=False)
        if nonzero.numel() and bool(column[int(nonzero[0, 0])] < 0):
            basis[:, index].neg_()
            coefficients[:, index].neg_()
    return basis, coefficients


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    if not dtype_name.startswith("torch."):
        raise ValueError(f"unsupported dtype name: {dtype_name!r}")
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported dtype name: {dtype_name!r}")
    return dtype


def fit_weight_delta_bank(
    base_checkpoint_id: str,
    task_deltas: Mapping[DeltaTaskKey, Mapping[str, Tensor]],
    *,
    max_rank: int = 16,
    max_relative_error: float | None = None,
) -> WeightDeltaBank:
    """Fit deterministic FP32 SVD bases over task deltas grouped by block."""

    if not isinstance(base_checkpoint_id, str) or not base_checkpoint_id:
        raise ValueError("base_checkpoint_id must be a non-empty string")
    if not task_deltas:
        raise ValueError("task_deltas must not be empty")
    if not isinstance(max_rank, int) or isinstance(max_rank, bool) or max_rank < 0:
        raise ValueError("max_rank must be a non-negative integer")
    if max_relative_error is not None and (
        not math.isfinite(max_relative_error) or max_relative_error < 0.0
    ):
        raise ValueError("max_relative_error must be finite and non-negative")

    raw_items = tuple(task_deltas.items())
    for task_key, _ in raw_items:
        _validate_task_key(task_key)
    ordered_items = tuple(sorted(raw_items, key=lambda item: _task_sort_key(item[0])))
    reference_names = _validate_delta_mapping(ordered_items[0][1])
    reference = ordered_items[0][1]
    for task_key, delta in ordered_items[1:]:
        names = _validate_delta_mapping(delta)
        if names != reference_names:
            raise ValueError(f"task delta parameter names differ for {task_key!r}")
        for name in reference_names:
            if delta[name].shape != reference[name].shape:
                raise ValueError(f"shape drift for delta {name!r}")
            if delta[name].dtype != reference[name].dtype:
                raise ValueError(f"dtype drift for delta {name!r}")

    grouped_names: dict[str, list[str]] = {}
    for name in reference_names:
        block_name = parameter_block_key(name)
        assert block_name is not None
        grouped_names.setdefault(block_name, []).append(name)

    entries: list[DeltaBankEntry] = []
    task_count = len(ordered_items)
    for block_name in sorted(grouped_names):
        parameter_names = tuple(grouped_names[block_name])
        shapes = tuple(tuple(reference[name].shape) for name in parameter_names)
        dtypes = tuple(str(reference[name].dtype) for name in parameter_names)
        spec = BlockSpec(block_name, parameter_names, shapes, dtypes)
        matrix = torch.stack(
            [
                torch.cat(
                    [delta[name].detach().reshape(-1).to(device="cpu", dtype=torch.float32) for name in parameter_names]
                )
                for _, delta in ordered_items
            ]
        )
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        singular_rank = int(torch.count_nonzero(singular_values > 0).item())
        nominal_rank = min(max_rank, task_count, singular_rank)

        def compose_at_rank(rank: int) -> tuple[Tensor, Tensor, float]:
            basis = vh[:rank].T.contiguous()
            coefficients = matrix @ basis
            basis, coefficients = _canonicalize_basis_sign(basis, coefficients)
            return basis, coefficients, _relative_error(matrix, basis, coefficients)

        basis, coefficients, relative_error = compose_at_rank(nominal_rank)
        if max_relative_error is not None and relative_error > max_relative_error:
            basis, coefficients, relative_error = compose_at_rank(singular_rank)
        entries.append(
            DeltaBankEntry(
                spec=spec,
                basis=basis,
                task_coefficients=coefficients,
                effective_rank=int(basis.shape[1]),
                relative_error=relative_error,
            )
        )

    return WeightDeltaBank(
        schema=WEIGHT_DELTA_BANK_SCHEMA,
        base_checkpoint_id=base_checkpoint_id,
        task_keys=tuple(task_key for task_key, _ in ordered_items),
        entries=tuple(entries),
    )


def compose_weight_delta(entry: DeltaBankEntry, coefficients: Tensor) -> dict[str, Tensor]:
    """Compose one block's parameter-shaped delta from a finite coefficient row."""

    if not isinstance(coefficients, Tensor):
        raise ValueError("coefficients must be a torch.Tensor")
    if not coefficients.is_floating_point():
        raise ValueError("coefficients must be floating point")
    if coefficients.ndim != 1 or coefficients.shape[0] != entry.effective_rank:
        raise ValueError("coefficient shape does not match the entry rank")
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError("non-finite coefficients are not allowed")
    expected_numel = sum(math.prod(shape) for shape in entry.spec.shapes)
    if entry.basis.ndim != 2 or entry.basis.shape != (expected_numel, entry.effective_rank):
        raise ValueError("entry basis geometry does not match its BlockSpec")
    if entry.basis.dtype != torch.float32:
        raise ValueError("entry basis must be float32")
    if not bool(torch.isfinite(entry.basis).all()):
        raise ValueError("entry basis contains non-finite values")

    flat = entry.basis @ coefficients.to(device=entry.basis.device, dtype=entry.basis.dtype)
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("non-finite composed delta")
    result: dict[str, Tensor] = {}
    offset = 0
    for name, shape, dtype_name in zip(
        entry.spec.parameter_names, entry.spec.shapes, entry.spec.dtypes, strict=True
    ):
        count = math.prod(shape)
        restored = flat[offset : offset + count].reshape(shape).to(_dtype_from_name(dtype_name))
        if not bool(torch.isfinite(restored).all()):
            raise ValueError(f"non-finite restored delta for {name!r}")
        result[name] = restored
        offset += count
    return result


__all__ = [
    "BlockSpec",
    "DeltaBankEntry",
    "DeltaTaskKey",
    "WEIGHT_DELTA_BANK_SCHEMA",
    "WeightDeltaBank",
    "compose_weight_delta",
    "extract_block_delta",
    "fit_weight_delta_bank",
    "parameter_block_key",
]
