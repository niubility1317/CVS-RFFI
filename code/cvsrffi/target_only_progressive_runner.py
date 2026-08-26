"""No-query runner and strict bundle readback for diagnostic SF-TAPFT V1."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from .target_only_progressive_adapt import (
    SFTAPFTConfig,
    TargetOnlyAdaptationDataset,
    TargetPrototypeHead,
    ensure_time_adapter,
    fit_sf_tapft,
    select_sf_tapft_by_grouped_cv,
)
from .sf_tapft_phase1_binding import (
    SFTAPFTPhase1Binding,
    load_sf_tapft_phase1_binding,
)


SF_TAPFT_BUNDLE_SCHEMA = "cvs.sf_tapft.v1"
_V1_TOP_LEVEL_KEYS = frozenset(
    {
        "candidate_id",
        "method",
        "permission",
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "checkpoint_path",
        "support_path",
        "sf_tapft",
    }
)
_R0_TOP_LEVEL_KEYS = _V1_TOP_LEVEL_KEYS | {"phase1_bundle"}
_BUNDLE_KEYS = frozenset(
    {
        "schema",
        "method",
        "permission",
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "base_checkpoint_path",
        "config",
        "model_state",
        "head_state",
        "class_ids",
        "audit",
    }
)


CheckpointLoader = Callable[..., nn.Module]


def _portable_tensor(array: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
    contiguous = np.ascontiguousarray(array)
    try:
        return torch.from_numpy(contiguous).to(dtype=dtype)
    except TypeError:
        # N607's deployed torch 2.1 / NumPy combination can reject a genuine
        # numpy.ndarray at the C bridge. Support is intentionally small, so a
        # list conversion is a bounded compatibility fallback.
        return torch.tensor(contiguous.tolist(), dtype=dtype)


def _default_checkpoint_loader(path: str | Path, *, device: str | torch.device) -> nn.Module:
    from .stage2_structured_late_block_runner import _load_frozen_checkpoint

    return _load_frozen_checkpoint(path, device=device)


def _parse_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], SFTAPFTConfig]:
    if not isinstance(config, Mapping) or frozenset(config) not in {
        _V1_TOP_LEVEL_KEYS,
        _R0_TOP_LEVEL_KEYS,
    }:
        raise ValueError("SF-TAPFT runner top-level allowlist mismatch")
    values = dict(config)
    if values["method"] != "sf_tapft_v1":
        raise ValueError("method must be sf_tapft_v1")
    if values["permission"] != "DIAGNOSTIC_NON_FORMAL":
        raise ValueError("report-parity SF-TAPFT must use DIAGNOSTIC_NON_FORMAL permission")
    if values["protocol_schema"] != "p2_min_v1":
        raise ValueError("protocol_schema must be p2_min_v1")
    if values["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("phase2_data_status must be VALIDATED_ONCE")
    if not isinstance(values["candidate_id"], str) or not values["candidate_id"].strip():
        raise ValueError("candidate_id must be a non-empty string")
    for name in ("capsule_id", "split_id", "checkpoint_path", "support_path"):
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"{name} must be a non-empty path string")
    raw = values["sf_tapft"]
    if not isinstance(raw, Mapping):
        raise ValueError("sf_tapft must be a mapping")
    allowed = {field.name for field in fields(SFTAPFTConfig)}
    unexpected = set(raw).difference(allowed)
    if unexpected:
        raise ValueError(f"sf_tapft contains unknown fields: {sorted(unexpected)}")
    normalized = dict(raw)
    if "phase_steps" in normalized:
        normalized["phase_steps"] = tuple(normalized["phase_steps"])
    return values, SFTAPFTConfig(**normalized)


def _string_rows(value: np.ndarray, *, label: str, expected: int) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or int(array.size) != int(expected):
        raise ValueError(f"{label} must be a row-aligned string vector")
    rows = tuple(str(item) for item in array.tolist())
    if any(not item.strip() for item in rows):
        raise ValueError(f"{label} must contain non-empty strings")
    return rows


def _load_target_support(path: str | Path) -> TargetOnlyAdaptationDataset:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"target support path is not a regular file: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = set(archive.files)
            required = {"received_iq", "support_labels"}
            allowed = required | {"support_physical_ids", "support_groups"}
            if not required.issubset(keys) or keys.difference(allowed):
                raise ValueError("target support NPZ allowlist mismatch")
            iq_array = np.asarray(archive["received_iq"])
            labels_array = np.asarray(archive["support_labels"])
            if iq_array.ndim < 2 or iq_array.shape[0] <= 0 or not np.issubdtype(iq_array.dtype, np.number):
                raise ValueError("received_iq must be finite numeric target train rows")
            if not np.isfinite(iq_array).all():
                raise ValueError("received_iq must be finite numeric target train rows")
            if labels_array.ndim != 1 or labels_array.size != iq_array.shape[0] or not np.issubdtype(labels_array.dtype, np.integer):
                raise ValueError("support_labels must be a row-aligned integer vector")
            if "support_physical_ids" in keys:
                physical_ids = _string_rows(
                    archive["support_physical_ids"],
                    label="support_physical_ids",
                    expected=int(iq_array.shape[0]),
                )
                physical_id_origin = "provided"
            else:
                physical_ids = tuple(
                    f"validated-support-row-{index:06d}" for index in range(int(iq_array.shape[0]))
                )
                physical_id_origin = "validated_support_row_index"
            groups = (
                _string_rows(
                    archive["support_groups"],
                    label="support_groups",
                    expected=int(iq_array.shape[0]),
                )
                if "support_groups" in keys
                else None
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("target support", "received_iq", "support_")):
            raise
        raise ValueError(f"cannot load target support NPZ: {source}") from exc
    return TargetOnlyAdaptationDataset(
        received_iq=_portable_tensor(iq_array.astype(np.float32, copy=False), dtype=torch.float32),
        labels=_portable_tensor(labels_array.astype(np.int64, copy=False), dtype=torch.long),
        physical_ids=physical_ids,
        groups=groups,
        physical_id_origin=physical_id_origin,
    )


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _audit_payload(audit: Any) -> dict[str, Any]:
    return {
        "method": audit.method,
        "permission": audit.permission,
        "total_steps": int(audit.total_steps),
        "phase_steps": list(audit.phase_steps),
        "trainable_names_by_phase": {
            key: list(value) for key, value in audit.trainable_names_by_phase.items()
        },
        "updated_parameter_names": list(audit.updated_parameter_names),
        "support_losses": list(audit.support_losses),
        "source_loader_opened": bool(audit.source_loader_opened),
        "source_samples_opened": bool(audit.source_samples_opened),
        "source_cache_opened": bool(audit.source_cache_opened),
        "target_eval_opened": bool(audit.target_eval_opened),
        "query_opened": bool(audit.query_opened),
        "bn_running_stats_updated": bool(audit.bn_running_stats_updated),
        "checkpoint_selection_role": str(audit.checkpoint_selection_role),
    }


def _binding_payload(binding: SFTAPFTPhase1Binding) -> dict[str, Any]:
    return {
        "outer_content_root_sha256": binding.outer_content_root_sha256,
        "checkpoint_lineage_sha256": binding.checkpoint_lineage_sha256,
        "runtime_sha256": binding.runtime_sha256,
        "class_handle_binding_sha256": binding.class_handle_binding_sha256,
        "class_handles": list(binding.class_handles),
        "component_pre_sign_content_root_sha256": binding.component_pre_sign_content_root_sha256,
    }


def _validate_support_labels_for_binding(
    support: TargetOnlyAdaptationDataset, binding: SFTAPFTPhase1Binding
) -> None:
    class_count = len(binding.class_handles)
    labels = support.labels
    if class_count <= 0 or int(torch.min(labels).item()) < 0 or int(torch.max(labels).item()) >= class_count:
        raise ValueError("support labels must be indices into the ordered Phase1 class registry")


def _bundle_payload(
    result: Any,
    *,
    resolved: Mapping[str, Any],
    method_config: SFTAPFTConfig,
    binding: SFTAPFTPhase1Binding | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": SF_TAPFT_BUNDLE_SCHEMA,
        "method": result.audit.method,
        "permission": result.audit.permission,
        "protocol_schema": resolved["protocol_schema"],
        "phase2_data_status": resolved["phase2_data_status"],
        "capsule_id": resolved["capsule_id"],
        "split_id": resolved["split_id"],
        "base_checkpoint_path": str(resolved["checkpoint_path"]),
        "config": asdict(method_config),
        "model_state": _cpu_state(result.model),
        "head_state": _cpu_state(result.head),
        "class_ids": list(result.head.class_ids),
        "audit": _audit_payload(result.audit),
    }
    if binding is not None:
        payload["phase1_binding"] = _binding_payload(binding)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def run_sf_tapft_no_query(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: str | torch.device,
    checkpoint_loader: CheckpointLoader | None = None,
) -> dict[str, Any]:
    """Run target-train adaptation without any query input capability."""

    resolved, method_config = _parse_config(config)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"SF-TAPFT output directory already exists: {destination}")
    loader = checkpoint_loader or _default_checkpoint_loader
    binding = (
        load_sf_tapft_phase1_binding(resolved, resolved["checkpoint_path"])
        if "phase1_bundle" in resolved
        else None
    )
    model = loader(resolved["checkpoint_path"], device=device)
    support = _load_target_support(resolved["support_path"])
    if binding is not None:
        _validate_support_labels_for_binding(support, binding)
    result = fit_sf_tapft(model, support, method_config)
    payload = _bundle_payload(
        result,
        resolved=resolved,
        method_config=method_config,
        binding=binding,
    )
    destination.mkdir(parents=True, exist_ok=False)
    bundle_path = destination / "sf_tapft_bundle.pt"
    with bundle_path.open("xb") as handle:
        torch.save(payload, handle)
    receipt = {
        "status": "SMOKE_PASS",
        "candidate_id": resolved["candidate_id"],
        "method": result.audit.method,
        "permission": result.audit.permission,
        "protocol_schema": resolved["protocol_schema"],
        "phase2_data_status": resolved["phase2_data_status"],
        "capsule_id": resolved["capsule_id"],
        "split_id": resolved["split_id"],
        "total_steps": result.audit.total_steps,
        "support_physical_sample_count": len(support.physical_ids),
        "support_physical_id_origin": support.physical_id_origin,
        "updated_parameter_count": len(result.audit.updated_parameter_names),
        "bn_running_stats_updated": result.audit.bn_running_stats_updated,
        "source_opened": False,
        "target_eval_opened": False,
        "query_input_capability": False,
        "query_opened": False,
        "query_truth_opened": False,
        "query_role_opened": False,
        "bundle_path": str(bundle_path),
    }
    if binding is not None:
        receipt["phase1_binding"] = _binding_payload(binding)
    _write_json(destination / "smoke.json", receipt)
    return receipt


def run_sf_tapft_grouped_selection(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: str | torch.device,
    folds: int,
    checkpoint_loader: CheckpointLoader | None = None,
) -> dict[str, Any]:
    """Run grouped target-train OOF selection and one domain-level fallback."""

    resolved, method_config = _parse_config(config)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"SF-TAPFT output directory already exists: {destination}")
    loader = checkpoint_loader or _default_checkpoint_loader
    binding = (
        load_sf_tapft_phase1_binding(resolved, resolved["checkpoint_path"])
        if "phase1_bundle" in resolved
        else None
    )
    model = loader(resolved["checkpoint_path"], device=device)
    support = _load_target_support(resolved["support_path"])
    if binding is not None:
        _validate_support_labels_for_binding(support, binding)
    selection = select_sf_tapft_by_grouped_cv(
        model,
        support,
        method_config,
        folds=int(folds),
    )
    destination.mkdir(parents=True, exist_ok=False)
    bundle_path = None
    if selection.adapted_result is not None:
        bundle_path = destination / "sf_tapft_bundle.pt"
        payload = _bundle_payload(
            selection.adapted_result,
            resolved=resolved,
            method_config=method_config,
            binding=binding,
        )
        with bundle_path.open("xb") as handle:
            torch.save(payload, handle)
    rows = [
        {
            "fold": row.fold,
            "train_groups": sorted(row.train_groups),
            "validation_groups": sorted(row.validation_groups),
            "frozen_balanced_accuracy": row.frozen_balanced_accuracy,
            "adapted_balanced_accuracy": row.adapted_balanced_accuracy,
            "frozen_nll": row.frozen_nll,
            "adapted_nll": row.adapted_nll,
            "frozen_margin": row.frozen_margin,
            "adapted_margin": row.adapted_margin,
            "source_distance": row.source_distance,
            "query_opened": False,
        }
        for row in selection.fold_rows
    ]
    receipt = {
        "status": "SELECTION_COMPLETE",
        "candidate_id": resolved["candidate_id"],
        "method": "sf_tapft_v1",
        "permission": "DIAGNOSTIC_NON_FORMAL",
        "protocol_schema": resolved["protocol_schema"],
        "phase2_data_status": resolved["phase2_data_status"],
        "capsule_id": resolved["capsule_id"],
        "split_id": resolved["split_id"],
        "selected": selection.selected,
        "support_physical_id_origin": support.physical_id_origin,
        "folds": int(folds),
        "frozen_metrics": asdict(selection.frozen_metrics),
        "adapted_metrics": asdict(selection.adapted_metrics),
        "fold_rows": rows,
        "bundle_path": str(bundle_path) if bundle_path is not None else None,
        "source_opened": False,
        "target_eval_opened": False,
        "query_input_capability": False,
        "query_opened": False,
        "query_truth_opened": False,
        "query_role_opened": False,
    }
    if binding is not None:
        receipt["phase1_binding"] = _binding_payload(binding)
    _write_json(destination / "selection.json", receipt)
    return receipt


def load_sf_tapft_bundle_strict(
    path: str | Path,
    *,
    device: str | torch.device,
    checkpoint_loader: CheckpointLoader | None = None,
) -> tuple[nn.Module, TargetPrototypeHead, dict[str, Any]]:
    """Rebuild the base model and strictly read back an SF-TAPFT bundle."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"SF-TAPFT bundle is not a regular file: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot safely load SF-TAPFT bundle: {source}") from exc
    payload_keys = set(payload) if isinstance(payload, Mapping) else set()
    if payload_keys != set(_BUNDLE_KEYS) and payload_keys != set(_BUNDLE_KEYS) | {"phase1_binding"}:
        raise ValueError("SF-TAPFT bundle top-level allowlist mismatch")
    if payload["schema"] != SF_TAPFT_BUNDLE_SCHEMA:
        raise ValueError("SF-TAPFT bundle schema mismatch")
    if payload["method"] != "sf_tapft_v1" or payload["permission"] != "DIAGNOSTIC_NON_FORMAL":
        raise ValueError("SF-TAPFT bundle method or permission mismatch")
    if payload["protocol_schema"] != "p2_min_v1" or payload["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("SF-TAPFT bundle Phase2 data binding mismatch")
    if not str(payload["capsule_id"]).strip() or not str(payload["split_id"]).strip():
        raise ValueError("SF-TAPFT bundle capsule_id or split_id is empty")
    raw_config = dict(payload["config"])
    raw_config["phase_steps"] = tuple(raw_config["phase_steps"])
    config = SFTAPFTConfig(**raw_config)
    loader = checkpoint_loader or _default_checkpoint_loader
    model = loader(payload["base_checkpoint_path"], device=device)
    ensure_time_adapter(model, rank=config.adapter_rank)
    try:
        incompatible = model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as exc:
        raise ValueError("SF-TAPFT adapted model state mismatch") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("SF-TAPFT adapted model state is not strict")
    head_state = payload["head_state"]
    if not isinstance(head_state, Mapping) or set(head_state) != {"weight"}:
        raise ValueError("SF-TAPFT head state allowlist mismatch")
    class_ids = tuple(int(value) for value in payload["class_ids"])
    head = TargetPrototypeHead(
        head_state["weight"],
        class_ids,
        scale=config.prototype_scale,
    )
    head.load_state_dict(head_state, strict=True)
    model.to(torch.device(device)).eval()
    head.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    audit = {
        "schema": SF_TAPFT_BUNDLE_SCHEMA,
        "method": payload["method"],
        "permission": payload["permission"],
        "base_checkpoint_path": payload["base_checkpoint_path"],
        "total_steps": int(payload["audit"]["total_steps"]),
        "checkpoint_selection_role": payload["audit"]["checkpoint_selection_role"],
        "capsule_id": payload["capsule_id"],
        "split_id": payload["split_id"],
        "query_input_capability": False,
    }
    return model, head, audit


__all__ = [
    "SF_TAPFT_BUNDLE_SCHEMA",
    "load_sf_tapft_bundle_strict",
    "run_sf_tapft_grouped_selection",
    "run_sf_tapft_no_query",
]
