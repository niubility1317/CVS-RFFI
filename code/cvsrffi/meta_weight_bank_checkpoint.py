"""Strict deployment checkpoint bundle for the MARC-OT weight bank."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .meta_support_set_encoder import SupportSetEncoder
from .meta_weight_bank import (
    BlockSpec,
    DeltaBankEntry,
    DeltaTaskKey,
    WEIGHT_DELTA_BANK_SCHEMA,
    WeightDeltaBank,
    parameter_block_key,
)


META_WEIGHT_BUNDLE_SCHEMA = "marc_ot_weight_bank_v1"


@dataclass(frozen=True)
class MetaWeightBundle:
    """Validated frozen bank and support encoder bound to one base checkpoint."""

    schema: str
    base_checkpoint_id: str
    bank: WeightDeltaBank
    support_encoder: SupportSetEncoder


def _require_exact_keys(value: object, expected: set[str], context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} members differ: missing={sorted(expected - actual)!r}, "
            f"forbidden={sorted(actual - expected)!r}"
        )
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _finite_tensor(value: object, context: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{context} must be a tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{context} contains non-finite values")
    return value


def _validate_base_state(base_state: Mapping[str, Tensor]) -> None:
    if not isinstance(base_state, Mapping) or not base_state:
        raise ValueError("base_state must be a non-empty tensor mapping")
    for name, value in base_state.items():
        if not isinstance(name, str) or not name or not isinstance(value, Tensor):
            raise ValueError("base_state must map non-empty string names to tensors")
        _finite_tensor(value, f"base_state[{name!r}]")


def _validate_expected_block_specs(
    expected_block_specs: tuple[BlockSpec, ...],
) -> tuple[BlockSpec, ...]:
    if not isinstance(expected_block_specs, tuple) or not expected_block_specs:
        raise ValueError("expected_block_specs must be a non-empty tuple")
    if any(not isinstance(spec, BlockSpec) for spec in expected_block_specs):
        raise ValueError("expected_block_specs must contain BlockSpec values")
    block_names = tuple(spec.name for spec in expected_block_specs)
    if block_names != tuple(sorted(block_names)) or len(set(block_names)) != len(block_names):
        raise ValueError("expected block order must follow Task1 canonical sorted block names")
    for spec in expected_block_specs:
        if spec.parameter_names != tuple(sorted(spec.parameter_names)):
            raise ValueError(
                f"expected parameter order for block {spec.name!r} must follow Task1 canonical sorting"
            )
    return expected_block_specs


def _validate_bank(
    bank: WeightDeltaBank,
    base_state: Mapping[str, Tensor],
    expected_block_specs: tuple[BlockSpec, ...],
) -> None:
    if not isinstance(bank, WeightDeltaBank) or bank.schema != WEIGHT_DELTA_BANK_SCHEMA:
        raise ValueError("weight delta bank schema mismatch")
    if not isinstance(bank.base_checkpoint_id, str) or not bank.base_checkpoint_id:
        raise ValueError("weight delta bank base checkpoint identity is invalid")
    if not isinstance(bank.task_keys, tuple) or not bank.task_keys:
        raise ValueError("weight delta bank task keys must be non-empty")
    if not isinstance(bank.entries, tuple) or not bank.entries:
        raise ValueError("weight delta bank entries must be non-empty")
    expected_block_specs = _validate_expected_block_specs(expected_block_specs)
    actual_block_specs = tuple(entry.spec for entry in bank.entries)
    if actual_block_specs != expected_block_specs:
        raise ValueError("weight delta bank block geometry or order differs from expected geometry")

    seen_blocks: set[str] = set()
    seen_parameters: set[str] = set()
    total_rank = 0
    for task in bank.task_keys:
        if (
            not isinstance(task, DeltaTaskKey)
            or any(not isinstance(value, str) or not value for value in (task.receiver, task.day, task.scene))
            or isinstance(task.k_shot, bool)
            or not isinstance(task.k_shot, int)
            or task.k_shot <= 0
        ):
            raise ValueError("weight delta bank contains an invalid task key")
    for entry in bank.entries:
        if not isinstance(entry, DeltaBankEntry) or not isinstance(entry.spec, BlockSpec):
            raise ValueError("weight delta bank entry type mismatch")
        spec = entry.spec
        if not isinstance(spec.name, str) or not spec.name or spec.name in seen_blocks:
            raise ValueError("weight delta bank block names must be unique and non-empty")
        seen_blocks.add(spec.name)
        if (
            not spec.parameter_names
            or len(spec.parameter_names) != len(spec.shapes)
            or len(spec.parameter_names) != len(spec.dtypes)
        ):
            raise ValueError("weight delta bank block geometry is malformed")
        if (
            isinstance(entry.effective_rank, bool)
            or not isinstance(entry.effective_rank, int)
            or entry.effective_rank < 0
        ):
            raise ValueError("weight delta bank effective rank is invalid")
        total_rank += entry.effective_rank
        if (
            isinstance(entry.relative_error, bool)
            or not isinstance(entry.relative_error, (int, float))
            or not math.isfinite(float(entry.relative_error))
            or float(entry.relative_error) < 0.0
        ):
            raise ValueError("weight delta bank relative error is invalid")
        expected_numel = 0
        for name, shape, dtype_name in zip(
            spec.parameter_names, spec.shapes, spec.dtypes, strict=True
        ):
            if name in seen_parameters:
                raise ValueError("weight delta bank repeats a parameter")
            seen_parameters.add(name)
            if parameter_block_key(name) != spec.name:
                raise ValueError("weight delta bank contains a forbidden parameter member")
            if (
                not isinstance(shape, tuple)
                or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape)
                or not isinstance(dtype_name, str)
            ):
                raise ValueError("weight delta bank parameter geometry is invalid")
            base_value = base_state.get(name)
            if not isinstance(base_value, Tensor) or not base_value.is_floating_point():
                raise ValueError("weight delta bank parameter is absent from floating base state")
            if tuple(base_value.shape) != shape or str(base_value.dtype) != dtype_name:
                raise ValueError("weight delta bank parameter geometry does not match base state")
            expected_numel += base_value.numel()
        basis = _finite_tensor(entry.basis, f"bank basis {spec.name!r}")
        coefficients = _finite_tensor(
            entry.task_coefficients, f"bank task coefficients {spec.name!r}"
        )
        if basis.dtype != torch.float32 or basis.shape != (expected_numel, entry.effective_rank):
            raise ValueError("weight delta bank basis geometry or dtype is invalid")
        if coefficients.dtype != torch.float32 or coefficients.shape != (
            len(bank.task_keys),
            entry.effective_rank,
        ):
            raise ValueError("weight delta bank task coefficient geometry or dtype is invalid")
    if total_rank <= 0:
        raise ValueError("meta weight bundle requires at least one effective bank rank")


def _encoder_config(encoder: SupportSetEncoder) -> dict[str, object]:
    if (
        not isinstance(encoder, SupportSetEncoder)
        or not isinstance(encoder.phi[0], nn.Linear)
        or not isinstance(encoder.rho[-1], nn.Linear)
    ):
        raise ValueError("support encoder architecture is not the reviewed Task2 interface")
    return {
        "feature_dim": encoder.feature_dim,
        "coefficient_dim": encoder.coefficient_dim,
        "block_count": encoder.block_count,
        "hidden_dim": encoder.phi[0].out_features,
        "lr_min": encoder.lr_min,
        "lr_max": encoder.lr_max,
    }


def _validate_encoder_for_bank(encoder: SupportSetEncoder, bank: WeightDeltaBank) -> None:
    config = _encoder_config(encoder)
    if config["coefficient_dim"] != sum(entry.effective_rank for entry in bank.entries):
        raise ValueError("support encoder coefficient geometry does not match bank ranks")
    if config["block_count"] != len(bank.entries):
        raise ValueError("support encoder block geometry does not match bank entries")
    for name, value in encoder.state_dict().items():
        _finite_tensor(value, f"support encoder state {name!r}")


def _bank_payload(bank: WeightDeltaBank) -> dict[str, object]:
    return {
        "schema": bank.schema,
        "task_keys": [
            {
                "receiver": task.receiver,
                "day": task.day,
                "scene": task.scene,
                "k_shot": task.k_shot,
            }
            for task in bank.task_keys
        ],
        "entries": [
            {
                "name": entry.spec.name,
                "parameter_names": list(entry.spec.parameter_names),
                "shapes": [list(shape) for shape in entry.spec.shapes],
                "dtypes": list(entry.spec.dtypes),
                "basis": entry.basis.detach().cpu().clone(),
                "task_coefficients": entry.task_coefficients.detach().cpu().clone(),
                "effective_rank": entry.effective_rank,
                "relative_error": float(entry.relative_error),
            }
            for entry in bank.entries
        ],
    }


def _block_specs_payload(specs: tuple[BlockSpec, ...]) -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "parameter_names": list(spec.parameter_names),
            "shapes": [list(shape) for shape in spec.shapes],
            "dtypes": list(spec.dtypes),
        }
        for spec in specs
    ]


def _encoder_payload(encoder: SupportSetEncoder) -> dict[str, object]:
    return {
        "config": _encoder_config(encoder),
        "state_dict": {
            name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()
        },
    }


def _decode_task_keys(raw: object) -> tuple[DeltaTaskKey, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("bundle task_keys must be a non-empty list")
    tasks: list[DeltaTaskKey] = []
    for index, item in enumerate(raw):
        data = _require_exact_keys(
            item, {"receiver", "day", "scene", "k_shot"}, f"task_keys[{index}]"
        )
        tasks.append(
            DeltaTaskKey(
                receiver=data["receiver"],
                day=data["day"],
                scene=data["scene"],
                k_shot=data["k_shot"],
            )
        )
    return tuple(tasks)


def _decode_bank(raw: object, base_checkpoint_id: str) -> WeightDeltaBank:
    data = _require_exact_keys(raw, {"schema", "task_keys", "entries"}, "bank")
    if data["schema"] != WEIGHT_DELTA_BANK_SCHEMA:
        raise ValueError("embedded weight delta bank schema mismatch")
    task_keys = _decode_task_keys(data["task_keys"])
    raw_entries = data["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("bundle bank entries must be a non-empty list")
    entries: list[DeltaBankEntry] = []
    entry_keys = {
        "name",
        "parameter_names",
        "shapes",
        "dtypes",
        "basis",
        "task_coefficients",
        "effective_rank",
        "relative_error",
    }
    for index, item in enumerate(raw_entries):
        entry = _require_exact_keys(item, entry_keys, f"bank.entries[{index}]")
        if (
            not isinstance(entry["parameter_names"], list)
            or not isinstance(entry["shapes"], list)
            or not isinstance(entry["dtypes"], list)
        ):
            raise ValueError("bundle block geometry members must be lists")
        try:
            spec = BlockSpec(
                name=entry["name"],
                parameter_names=tuple(entry["parameter_names"]),
                shapes=tuple(tuple(shape) for shape in entry["shapes"]),
                dtypes=tuple(entry["dtypes"]),
            )
        except TypeError as error:
            raise ValueError("bundle block geometry is malformed") from error
        basis = _finite_tensor(entry["basis"], f"bank.entries[{index}].basis").detach().clone()
        coefficients = _finite_tensor(
            entry["task_coefficients"], f"bank.entries[{index}].task_coefficients"
        ).detach().clone()
        entries.append(
            DeltaBankEntry(
                spec=spec,
                basis=basis,
                task_coefficients=coefficients,
                effective_rank=entry["effective_rank"],
                relative_error=entry["relative_error"],
            )
        )
    return WeightDeltaBank(
        schema=WEIGHT_DELTA_BANK_SCHEMA,
        base_checkpoint_id=base_checkpoint_id,
        task_keys=task_keys,
        entries=tuple(entries),
    )


def _decode_block_specs(raw: object, context: str) -> tuple[BlockSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} must be a non-empty list")
    specs: list[BlockSpec] = []
    for index, item in enumerate(raw):
        data = _require_exact_keys(
            item,
            {"name", "parameter_names", "shapes", "dtypes"},
            f"{context}[{index}]",
        )
        if (
            not isinstance(data["parameter_names"], list)
            or not isinstance(data["shapes"], list)
            or not isinstance(data["dtypes"], list)
        ):
            raise ValueError(f"{context}[{index}] geometry members must be lists")
        try:
            specs.append(
                BlockSpec(
                    name=data["name"],
                    parameter_names=tuple(data["parameter_names"]),
                    shapes=tuple(tuple(shape) for shape in data["shapes"]),
                    dtypes=tuple(data["dtypes"]),
                )
            )
        except TypeError as error:
            raise ValueError(f"{context}[{index}] geometry is malformed") from error
    return tuple(specs)


def _decode_encoder(raw: object, bank: WeightDeltaBank) -> SupportSetEncoder:
    data = _require_exact_keys(raw, {"config", "state_dict"}, "support_encoder")
    config = _require_exact_keys(
        data["config"],
        {"feature_dim", "coefficient_dim", "block_count", "hidden_dim", "lr_min", "lr_max"},
        "support_encoder.config",
    )
    try:
        encoder = SupportSetEncoder(**dict(config))
    except (TypeError, ValueError) as error:
        raise ValueError("support encoder config is invalid") from error
    raw_state = data["state_dict"]
    if not isinstance(raw_state, Mapping):
        raise ValueError("support encoder state_dict must be a mapping")
    expected_state = encoder.state_dict()
    if set(raw_state) != set(expected_state):
        raise ValueError("support encoder state_dict names differ")
    state: dict[str, Tensor] = {}
    for name, reference in expected_state.items():
        value = _finite_tensor(raw_state[name], f"support_encoder.state_dict[{name!r}]")
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError("support encoder state geometry or dtype differs")
        state[name] = value.detach().clone()
    encoder.load_state_dict(state, strict=True)
    _validate_encoder_for_bank(encoder, bank)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def _decode_bundle(
    raw: object,
    *,
    expected_base_checkpoint_id: str,
    base_state: Mapping[str, Tensor],
    expected_block_specs: tuple[BlockSpec, ...],
) -> MetaWeightBundle:
    data = _require_exact_keys(
        raw,
        {"schema", "base_checkpoint_id", "block_geometry", "bank", "support_encoder"},
        "meta weight bundle",
    )
    if data["schema"] != META_WEIGHT_BUNDLE_SCHEMA:
        raise ValueError("meta weight bundle schema mismatch")
    if (
        not isinstance(expected_base_checkpoint_id, str)
        or not expected_base_checkpoint_id
        or data["base_checkpoint_id"] != expected_base_checkpoint_id
    ):
        raise ValueError("meta weight bundle base checkpoint identity mismatch")
    _validate_base_state(base_state)
    expected_block_specs = _validate_expected_block_specs(expected_block_specs)
    declared_block_specs = _decode_block_specs(data["block_geometry"], "block_geometry")
    if declared_block_specs != expected_block_specs:
        raise ValueError("bundle-declared block geometry or order differs from caller expectation")
    bank = _decode_bank(data["bank"], expected_base_checkpoint_id)
    _validate_bank(bank, base_state, expected_block_specs)
    encoder = _decode_encoder(data["support_encoder"], bank)
    return MetaWeightBundle(
        schema=META_WEIGHT_BUNDLE_SCHEMA,
        base_checkpoint_id=expected_base_checkpoint_id,
        bank=bank,
        support_encoder=encoder,
    )


def _load_raw(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - unsupported runtimes must fail closed.
        raise RuntimeError("this runtime lacks safe weights-only bundle loading") from error


def save_meta_weight_bundle(
    path: str | Path,
    *,
    base_checkpoint_id: str,
    base_state: Mapping[str, Tensor],
    bank: WeightDeltaBank,
    support_encoder: SupportSetEncoder,
    expected_block_specs: tuple[BlockSpec, ...],
) -> Path:
    """Validate and persist the only allowed MARC-OT deployment members."""

    destination = Path(path)
    if not isinstance(base_checkpoint_id, str) or not base_checkpoint_id:
        raise ValueError("base_checkpoint_id must be a non-empty string")
    _validate_base_state(base_state)
    expected_block_specs = _validate_expected_block_specs(expected_block_specs)
    _validate_bank(bank, base_state, expected_block_specs)
    if bank.base_checkpoint_id != base_checkpoint_id:
        raise ValueError("base checkpoint identity does not match weight delta bank")
    _validate_encoder_for_bank(support_encoder, bank)
    payload = {
        "schema": META_WEIGHT_BUNDLE_SCHEMA,
        "base_checkpoint_id": base_checkpoint_id,
        "block_geometry": _block_specs_payload(expected_block_specs),
        "bank": _bank_payload(bank),
        "support_encoder": _encoder_payload(support_encoder),
    }
    torch.save(payload, destination)
    _decode_bundle(
        _load_raw(destination),
        expected_base_checkpoint_id=base_checkpoint_id,
        base_state=base_state,
        expected_block_specs=expected_block_specs,
    )
    return destination


def load_meta_weight_bundle(
    path: str | Path,
    *,
    expected_base_checkpoint_id: str,
    base_state: Mapping[str, Tensor],
    expected_block_specs: tuple[BlockSpec, ...],
) -> MetaWeightBundle:
    """Load a fail-closed bundle and freeze its Phase2 bank/encoder state."""

    return _decode_bundle(
        _load_raw(Path(path)),
        expected_base_checkpoint_id=expected_base_checkpoint_id,
        base_state=base_state,
        expected_block_specs=expected_block_specs,
    )


__all__ = [
    "META_WEIGHT_BUNDLE_SCHEMA",
    "MetaWeightBundle",
    "load_meta_weight_bundle",
    "save_meta_weight_bundle",
]
