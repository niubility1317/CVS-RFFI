"""Head-free deployment bundle for SF-TAPFT ``t3.norm`` deltas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn


SCHEMA = "cvs.sf_t3_norm.delta.v1"
SOURCE_SCHEMA = "cvs.sf_tapft.delta.v3"
T3_DELTA_PARAMETER_NAMES = ("t3.norm.weight", "t3.norm.bias")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"delta bundle is not a regular file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("delta bundle must be a mapping")
    return payload


def write_t3_only_delta_bundle(
    destination_path: str | Path,
    *,
    model_deltas: Mapping[str, Any],
    protocol_schema: str,
    phase2_data_status: str,
    capsule_id: str,
    split_id: str,
    base_checkpoint_path: str,
    candidate_id: str,
    support_count: int,
    d92_method_lock: str,
    adapter_rank: int,
) -> dict[str, Any]:
    """Write a deployment delta whose API has no target-head argument."""

    destination = Path(destination_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"t3-only delta output already exists: {destination}")
    if protocol_schema != "p2_min_v1" or phase2_data_status != "VALIDATED_ONCE":
        raise ValueError("t3-only delta protocol binding mismatch")
    if d92_method_lock != "D92-E0-NORF32":
        raise ValueError("t3-only deployment requires D92-E0-NORF32")
    if any(not isinstance(value, str) or not value.strip() for value in (
        capsule_id, split_id, base_checkpoint_path, candidate_id
    )):
        raise ValueError("t3-only delta identity is empty")
    if isinstance(support_count, bool) or int(support_count) <= 0:
        raise ValueError("t3-only delta support_count must be positive")
    if isinstance(adapter_rank, bool) or int(adapter_rank) <= 0:
        raise ValueError("t3-only delta adapter_rank must be positive")
    if not isinstance(model_deltas, Mapping):
        raise ValueError("t3-only model_deltas must be a mapping")
    normalized: dict[str, torch.Tensor] = {}
    for short_name in T3_DELTA_PARAMETER_NAMES:
        canonical_name = f"model.{short_name}"
        candidates = [name for name in (short_name, canonical_name) if name in model_deltas]
        if len(candidates) != 1:
            raise ValueError(f"t3-only delta key drift: {canonical_name}")
        value = model_deltas[candidates[0]]
        if (
            not torch.is_tensor(value)
            or not value.is_floating_point()
            or value.numel() == 0
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"invalid t3 delta tensor: {canonical_name}")
        normalized[short_name] = value.detach().cpu().to(torch.float16).clone()
    if set(model_deltas) != {
        next(name for name in (short, f"model.{short}") if name in model_deltas)
        for short in T3_DELTA_PARAMETER_NAMES
    }:
        raise ValueError("t3-only delta contains a non-permitted parameter")
    output = {
        "schema": SCHEMA,
        "protocol_schema": protocol_schema,
        "phase2_data_status": phase2_data_status,
        "capsule_id": capsule_id,
        "split_id": split_id,
        "base_checkpoint_path": base_checkpoint_path,
        "adapter_rank": int(adapter_rank),
        "candidate_id": candidate_id,
        "d92_method_lock": d92_method_lock,
        "rf32_used": False,
        "support_count": int(support_count),
        "model_deltas": normalized,
        "updated_parameter_names": list(T3_DELTA_PARAMETER_NAMES),
        "temporary_target_head_persisted": False,
        "query_rows_used": 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        torch.save(output, handle)
    return {
        "schema": SCHEMA,
        "candidate_id": candidate_id,
        "bundle_path": str(destination),
        "bundle_bytes": int(destination.stat().st_size),
        "updated_parameter_names": T3_DELTA_PARAMETER_NAMES,
        "temporary_target_head_persisted": False,
        "d92_method_lock": d92_method_lock,
        "rf32_used": False,
        "query_rows_used": 0,
    }


def convert_sf_tapft_delta_bundle_to_t3_only(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    candidate_id: str,
    d92_method_lock: str,
) -> dict[str, Any]:
    """Strip the temporary target head from one validated SF-TAPFT delta."""

    source = Path(source_path)
    destination = Path(destination_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"t3-only delta output already exists: {destination}")
    payload = _load_mapping(source)
    if payload.get("schema") != SOURCE_SCHEMA:
        raise ValueError("SF-TAPFT source delta schema mismatch")
    if d92_method_lock != "D92-E0-NORF32":
        raise ValueError("t3-only deployment requires D92-E0-NORF32")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be non-empty")
    deltas = payload.get("model_deltas")
    updated = tuple(str(value) for value in payload.get("updated_parameter_names", ()))
    if not isinstance(deltas, Mapping) or set(deltas) != set(T3_DELTA_PARAMETER_NAMES):
        raise ValueError("source delta must contain exactly t3.norm weight and bias")
    if set(updated) != set(T3_DELTA_PARAMETER_NAMES):
        raise ValueError("source updated parameter allowlist drift")
    copied: dict[str, torch.Tensor] = {}
    for name in T3_DELTA_PARAMETER_NAMES:
        value = deltas[name]
        if (
            not torch.is_tensor(value)
            or not value.is_floating_point()
            or value.numel() == 0
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"invalid t3 delta tensor: {name}")
        copied[name] = value.detach().cpu().clone()
    return write_t3_only_delta_bundle(
        destination,
        model_deltas=copied,
        protocol_schema=str(payload["protocol_schema"]),
        phase2_data_status=str(payload["phase2_data_status"]),
        capsule_id=str(payload["capsule_id"]),
        split_id=str(payload["split_id"]),
        base_checkpoint_path=str(payload["base_checkpoint_path"]),
        candidate_id=candidate_id,
        support_count=int(payload["support_count"]),
        d92_method_lock=d92_method_lock,
        adapter_rank=int(payload["adapter_rank"]),
    )


def load_t3_only_delta_bundle_strict(
    path: str | Path,
    *,
    device: str | torch.device,
    expected_target_binding: Mapping[str, Any],
    checkpoint_loader: Callable[..., nn.Module] | None = None,
    adapter_initializer: Callable[..., nn.Module] | None = None,
) -> tuple[nn.Module, None, dict[str, Any]]:
    """Load the base checkpoint and apply exactly two head-free affine deltas."""

    source = Path(path)
    payload = _load_mapping(source)
    expected_keys = {
        "schema",
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "base_checkpoint_path",
        "adapter_rank",
        "candidate_id",
        "d92_method_lock",
        "rf32_used",
        "support_count",
        "model_deltas",
        "updated_parameter_names",
        "temporary_target_head_persisted",
        "query_rows_used",
    }
    if set(payload) != expected_keys or payload.get("schema") != SCHEMA:
        raise ValueError("t3-only delta top-level allowlist mismatch")
    if (
        payload["d92_method_lock"] != "D92-E0-NORF32"
        or payload["rf32_used"] is not False
        or payload["temporary_target_head_persisted"] is not False
        or payload["query_rows_used"] != 0
    ):
        raise ValueError("t3-only delta method boundary mismatch")
    for name in (
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "support_count",
    ):
        if name in expected_target_binding and payload[name] != expected_target_binding[name]:
            raise ValueError("t3-only delta target binding mismatch")
    deltas = payload["model_deltas"]
    updated = tuple(str(value) for value in payload["updated_parameter_names"])
    if (
        not isinstance(deltas, Mapping)
        or tuple(deltas) != T3_DELTA_PARAMETER_NAMES
        or updated != T3_DELTA_PARAMETER_NAMES
    ):
        raise ValueError("t3-only delta parameter allowlist mismatch")
    if checkpoint_loader is None:
        from cvsrffi.target_only_progressive_runner import _default_checkpoint_loader

        checkpoint_loader = _default_checkpoint_loader
    model = checkpoint_loader(payload["base_checkpoint_path"], device=device)
    if adapter_initializer is None:
        from cvsrffi.target_only_progressive_adapt import ensure_time_adapter

        adapter_initializer = ensure_time_adapter
    adapter_initializer(model, rank=int(payload["adapter_rank"]))
    named = dict(model.named_parameters())
    for name in T3_DELTA_PARAMETER_NAMES:
        delta = deltas[name]
        if name not in named or not torch.is_tensor(delta) or delta.shape != named[name].shape:
            raise ValueError(f"t3-only delta parameter mismatch: {name}")
    with torch.no_grad():
        for name in T3_DELTA_PARAMETER_NAMES:
            named[name].add_(deltas[name].to(device=named[name].device, dtype=named[name].dtype))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, None, {
        "schema": SCHEMA,
        "candidate_id": str(payload["candidate_id"]),
        "support_count": int(payload["support_count"]),
        "adapter_rank": int(payload["adapter_rank"]),
        "updated_parameter_names": T3_DELTA_PARAMETER_NAMES,
        "bundle_bytes": int(source.stat().st_size),
        "d92_method_lock": "D92-E0-NORF32",
        "rf32_used": False,
        "temporary_target_head_persisted": False,
        "query_rows_used": 0,
        "capsule_id": str(payload["capsule_id"]),
        "split_id": str(payload["split_id"]),
    }


__all__ = [
    "SCHEMA",
    "T3_DELTA_PARAMETER_NAMES",
    "convert_sf_tapft_delta_bundle_to_t3_only",
    "load_t3_only_delta_bundle_strict",
    "write_t3_only_delta_bundle",
]
