"""Strict legacy migration and checkpoint bundles for the V1 meta adapter.

The module deliberately keeps checkpoint handling small: a legacy ADV3B02
state may initialize the non-adapter part of an adapter-enabled model, while a
meta bundle is an exact, self-contained model reconstruction.  Neither path
creates an optimizer or reads any Phase2 data.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from cvsrffi.meta_adapter import iter_inner_adapter_parameters


META_BUNDLE_SCHEMA = "cvs.meta_adapter.tri_r4.v1"
REQUIRED_META_BUNDLE_KEYS = {
    "schema",
    "model_state",
    "model_args",
    "meta_adapter_config",
    "selection",
    "base_checkpoint",
    "class_mapping",
    "prototypes",
}
_META_CONFIG_KEYS = {
    "model_args",
    "meta_adapter_config",
    "base_checkpoint",
    "class_mapping",
    "prototypes",
}
_STATE_CONTAINER_KEYS = ("model", "model_state_dict", "state_dict")
_FORBIDDEN_TRAINABLE_FRAGMENTS = ("cls_head", "classifier", "lda", "cov")


@dataclass(frozen=True)
class CheckpointLoadAudit:
    """Immutable audit for the permissive legacy-to-adapter migration."""

    schema: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    checkpoint_load_strict: bool
    base_checkpoint_id: str
    state_container: str


@dataclass(frozen=True)
class MetaBundleAudit:
    """Immutable audit returned after an exact meta-bundle reconstruction."""

    schema: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    checkpoint_load_strict: bool
    trainable_names: tuple[str, ...]
    trainable_count: int
    total_parameters: int
    trainable_fraction: float
    base_checkpoint_id: str
    class_mapping: Any = field(default=None)
    prototypes: Any = field(default=None)
    selection: Any = field(default=None)


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _checkpoint_identifier(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in (
            "id",
            "checkpoint_id",
            "checkpoint_sha256",
            "sha256",
            "name",
        ):
            candidate = value.get(key)
            if candidate is not None and not isinstance(candidate, (Mapping, list, tuple)):
                return str(candidate)
        return "mapping"
    if value is None:
        return "unknown"
    return str(value)


def _snapshot_for_payload(value: Any) -> Any:
    """Copy nested values and materialize all tensors on CPU."""

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _snapshot_for_payload(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_snapshot_for_payload(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_for_payload(nested) for nested in value)
    if isinstance(value, set):
        return {_snapshot_for_payload(nested) for nested in value}
    return copy.deepcopy(value)


def _freeze_for_audit(value: Any) -> Any:
    """Create a read-only audit view without retaining model-owned objects."""

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {copy.deepcopy(key): _freeze_for_audit(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_for_audit(nested) for nested in value)
    if isinstance(value, tuple):
        return tuple(_freeze_for_audit(nested) for nested in value)
    if isinstance(value, set):
        return frozenset(_freeze_for_audit(nested) for nested in value)
    return copy.deepcopy(value)


def _normalize_state(state: Mapping[Any, Any], *, container: str) -> dict[str, Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"state container {container} must be a non-empty mapping")
    normalized: dict[str, Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"state container {container} has a non-string key")
        if not torch.is_tensor(value):
            raise ValueError(
                f"state container {container} contains non-tensor value for {raw_key}"
            )
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key in normalized:
            if not torch.equal(normalized[key], value):
                raise ValueError(f"state container {container} has duplicate normalized key {key}")
            raise ValueError(f"state container {container} has duplicate normalized key {key}")
        normalized[key] = value
    return normalized


def _states_equal(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        left[key].dtype == right[key].dtype
        and tuple(left[key].shape) == tuple(right[key].shape)
        and torch.equal(left[key], right[key])
        for key in left
    )


def _extract_state_dict(payload: Any) -> tuple[dict[str, Tensor], str]:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")

    candidates: list[tuple[str, dict[str, Tensor]]] = []
    for container in _STATE_CONTAINER_KEYS:
        if container not in payload:
            continue
        candidates.append(
            (container, _normalize_state(payload[container], container=container))
        )
    if candidates:
        first_name, first_state = candidates[0]
        for other_name, other_state in candidates[1:]:
            if not _states_equal(first_state, other_state):
                raise ValueError(
                    "conflicting state dict containers: "
                    f"{first_name} and {other_name}"
                )
        return first_state, first_name

    return _normalize_state(payload, container="raw"), "raw"


def _legacy_error(message: str, audit: CheckpointLoadAudit) -> ValueError:
    return ValueError(
        f"{message}; missing={list(audit.missing_keys)} "
        f"unexpected={list(audit.unexpected_keys)} "
        f"checkpoint_load_strict={audit.checkpoint_load_strict} "
        f"base_checkpoint_id={audit.base_checkpoint_id}"
    )


def _checkpoint_id_from_payload(payload: Mapping[str, Any]) -> str:
    for key in ("base_checkpoint", "checkpoint_id", "checkpoint_sha256", "id"):
        if key in payload:
            return _checkpoint_identifier(payload[key])
    return "legacy"


def load_legacy_base_for_meta(
    model: nn.Module, payload: Mapping[str, Any]
) -> CheckpointLoadAudit:
    """Load a legacy base state while allowing only missing adapter keys.

    This function is intentionally a Phase1 initialization path.  Adapter
    values in the payload are rejected so their Task3 initialization remains
    authoritative; this path is never used for Phase2 loading.
    """

    state, container = _extract_state_dict(payload)
    if any("meta_adapter_" in key for key in state):
        raise ValueError(
            "legacy payload must not contain adapter state; "
            f"keys={sorted(key for key in state if 'meta_adapter_' in key)}"
        )

    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        audit = CheckpointLoadAudit(
            schema="cvs.legacy.adv3b02.v1",
            missing_keys=(),
            unexpected_keys=(),
            checkpoint_load_strict=False,
            base_checkpoint_id=_checkpoint_id_from_payload(payload),
            state_container=container,
        )
        raise _legacy_error(f"legacy checkpoint shape mismatch: {exc}", audit) from exc

    missing = tuple(sorted(str(key) for key in incompatible.missing_keys))
    unexpected = tuple(sorted(str(key) for key in incompatible.unexpected_keys))
    audit = CheckpointLoadAudit(
        schema="cvs.legacy.adv3b02.v1",
        missing_keys=missing,
        unexpected_keys=unexpected,
        checkpoint_load_strict=False,
        base_checkpoint_id=_checkpoint_id_from_payload(payload),
        state_container=container,
    )
    allowed_missing = {
        name for name in model.state_dict() if "meta_adapter_" in name
    }
    non_adapter_missing = tuple(
        name for name in missing if name not in allowed_missing
    )
    if unexpected:
        raise _legacy_error("legacy checkpoint has unexpected keys", audit)
    if non_adapter_missing:
        raise _legacy_error(
            "legacy checkpoint is missing non-adapter keys", audit
        )
    return audit


def _selection_is_target_or_query_derived(value: Any, *, key_context: str = "") -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if (
                "target" in key_text
                or "truth" in key_text
                or "phase2" in key_text
                or ("query" in key_text and "source" not in key_text)
            ):
                return True
            if _selection_is_target_or_query_derived(nested, key_context=key_text):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(
            _selection_is_target_or_query_derived(nested, key_context=key_context)
            for nested in value
        )
    if isinstance(value, str):
        text = value.lower()
        return any(
            fragment in text
            for fragment in ("target query", "query truth", "phase2")
        )
    return False


def _bundle_model_state(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for key, value in model.state_dict().items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("model state must contain only named tensors")
        state[key] = value.detach().cpu().clone()
    return state


def save_meta_bundle(
    path: str | Path,
    model: nn.Module,
    config: Mapping[str, Any] | Any,
    selection: Mapping[str, Any],
) -> None:
    """Save a CPU-materialized, non-overwriting V1 meta bundle."""

    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"meta bundle path already exists: {output}")
    config_mapping = _as_mapping(config, field_name="config")
    missing = _META_CONFIG_KEYS.difference(config_mapping)
    if missing:
        raise ValueError(f"config missing required fields: {sorted(missing)}")
    model_args = _as_mapping(config_mapping["model_args"], field_name="model_args")
    meta_adapter_config = _as_mapping(
        config_mapping["meta_adapter_config"], field_name="meta_adapter_config"
    )
    selection_mapping = _as_mapping(selection, field_name="selection")
    if _selection_is_target_or_query_derived(selection_mapping):
        raise ValueError(
            "selection must contain Phase1 source selection information only; "
            "target/query-derived information is forbidden"
        )

    payload = {
        "schema": META_BUNDLE_SCHEMA,
        "model_state": _bundle_model_state(model),
        "model_args": _snapshot_for_payload(model_args),
        "meta_adapter_config": _snapshot_for_payload(meta_adapter_config),
        "selection": _snapshot_for_payload(selection_mapping),
        "base_checkpoint": _snapshot_for_payload(config_mapping["base_checkpoint"]),
        "class_mapping": _snapshot_for_payload(config_mapping["class_mapping"]),
        "prototypes": _snapshot_for_payload(config_mapping["prototypes"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(payload, handle)


def _adapter_sites_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _builder_args(
    model_args_value: Mapping[str, Any],
    meta_adapter_config_value: Mapping[str, Any],
) -> dict[str, Any]:
    args = dict(model_args_value)
    adapter_config = dict(meta_adapter_config_value)
    rank_config = adapter_config.get(
        "meta_adapter_rank", adapter_config.get("rank")
    )
    sites_config = adapter_config.get(
        "meta_adapter_sites", adapter_config.get("sites")
    )
    if rank_config is not None and "meta_adapter_rank" in args:
        if int(args["meta_adapter_rank"]) != int(rank_config):
            raise ValueError("model_args and meta_adapter_config adapter rank mismatch")
    if sites_config is not None and "meta_adapter_sites" in args:
        if _adapter_sites_value(args["meta_adapter_sites"]) != _adapter_sites_value(sites_config):
            raise ValueError("model_args and meta_adapter_config adapter sites mismatch")
    if rank_config is not None:
        args["meta_adapter_rank"] = int(rank_config)
    if sites_config is not None:
        args["meta_adapter_sites"] = _adapter_sites_value(sites_config)
    return args


def _bundle_state(payload: Mapping[str, Any]) -> dict[str, Tensor]:
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("model_state must be a non-empty mapping")
    normalized: dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError("model_state contains a non-string key")
        if not torch.is_tensor(value):
            raise ValueError(f"model_state contains non-tensor value for {key}")
        normalized[key] = value
    return normalized


def _build_repository_model(model_args: Mapping[str, Any], adapter_config: Mapping[str, Any]) -> nn.Module:
    """Dispatch to the single or dual ADV3B02 builder described by args."""

    args = _builder_args(model_args, adapter_config)
    builder_hint = str(
        args.pop("builder", args.pop("model_builder", args.pop("model_kind", "")))
    ).lower()
    dual_hint = any(
        marker in builder_hint
        for marker in ("dual", "build_dual_model", "dual_cvsincnet")
    )
    is_dual = dual_hint or any(
        key in args for key in ("num_domains", "id_feature_key", "dom_feature_key")
    )
    if builder_hint and not (
        dual_hint
        or builder_hint.endswith("build_model")
        or builder_hint.endswith(".build_model")
    ):
        raise ValueError(f"unsupported model builder hint: {builder_hint}")
    if is_dual:
        from model_dual_cvsincnet import build_dual_model

        return build_dual_model(**args)
    from model import build_model

    return build_model(**args)


def load_meta_bundle_strict(
    path: str | Path, device: str | torch.device
) -> tuple[nn.Module, MetaBundleAudit]:
    """Rebuild the real model and strictly load a V1 bundle."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"meta bundle path is not a regular file: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - torch error wording is version-specific
        raise ValueError(f"cannot load meta bundle: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("meta bundle payload must be a mapping")
    if set(payload) != REQUIRED_META_BUNDLE_KEYS:
        raise ValueError(
            "meta bundle top-level fields must be exactly required fields: "
            f"missing={sorted(REQUIRED_META_BUNDLE_KEYS.difference(payload))} "
            f"unexpected={sorted(set(payload).difference(REQUIRED_META_BUNDLE_KEYS))}"
        )
    if payload.get("schema") != META_BUNDLE_SCHEMA:
        raise ValueError(f"meta bundle schema mismatch: {payload.get('schema')!r}")

    model_args = _as_mapping(payload["model_args"], field_name="model_args")
    meta_adapter_config = _as_mapping(
        payload["meta_adapter_config"], field_name="meta_adapter_config"
    )
    state = _bundle_state(payload)
    try:
        model = _build_repository_model(model_args, meta_adapter_config)
    except Exception as exc:
        raise ValueError(f"cannot rebuild model from model_args: {exc}") from exc

    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "strict meta bundle state load failed "
            "(missing/unexpected/shape mismatch): "
            f"{exc}"
        ) from exc
    missing = tuple(sorted(str(key) for key in incompatible.missing_keys))
    unexpected = tuple(sorted(str(key) for key in incompatible.unexpected_keys))
    if missing or unexpected:
        raise ValueError(
            "strict meta bundle state load failed: "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )

    try:
        model.to(torch.device(device))
    except Exception as exc:
        raise ValueError(f"cannot move meta bundle model to device {device!r}") from exc
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    inner_items = tuple(iter_inner_adapter_parameters(model))
    expected_names = tuple(sorted(name for name, _ in inner_items))
    if not expected_names:
        raise ValueError("strict meta bundle has no Task3 inner adapter parameters")
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("strict meta bundle inner adapter names are not unique")
    for name, parameter in inner_items:
        parameter.requires_grad_(True)
    trainable_names = tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    if trainable_names != expected_names:
        raise ValueError(
            "strict meta bundle trainable names differ from Task3 inner whitelist: "
            f"trainable={list(trainable_names)} expected={list(expected_names)}"
        )
    forbidden = tuple(
        name
        for name in trainable_names
        if any(fragment in name.lower() for fragment in _FORBIDDEN_TRAINABLE_FRAGMENTS)
    )
    if forbidden:
        raise ValueError(f"forbidden trainable state name(s): {list(forbidden)}")

    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    trainable_fraction = float(trainable_count / total_parameters) if total_parameters else 0.0
    if trainable_fraction > 0.01:
        raise ValueError(
            "strict meta bundle trainable parameter fraction exceeds 1%: "
            f"count={trainable_count} total={total_parameters} fraction={trainable_fraction:.6f}"
        )

    model.eval()
    audit = MetaBundleAudit(
        schema=META_BUNDLE_SCHEMA,
        missing_keys=missing,
        unexpected_keys=unexpected,
        checkpoint_load_strict=True,
        trainable_names=trainable_names,
        trainable_count=trainable_count,
        total_parameters=total_parameters,
        trainable_fraction=trainable_fraction,
        base_checkpoint_id=_checkpoint_identifier(payload["base_checkpoint"]),
        class_mapping=_freeze_for_audit(payload["class_mapping"]),
        prototypes=_freeze_for_audit(payload["prototypes"]),
        selection=_freeze_for_audit(payload["selection"]),
    )
    return model, audit


__all__ = [
    "META_BUNDLE_SCHEMA",
    "REQUIRED_META_BUNDLE_KEYS",
    "CheckpointLoadAudit",
    "MetaBundleAudit",
    "load_legacy_base_for_meta",
    "load_meta_bundle_strict",
    "save_meta_bundle",
]
