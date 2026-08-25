"""Strict legacy migration and checkpoint bundles for the V1 meta adapter.

The module deliberately keeps checkpoint handling small: a legacy ADV3B02
state may initialize the non-adapter part of an adapter-enabled model, while a
meta bundle is an exact, self-contained model reconstruction.  Neither path
creates an optimizer or reads any Phase2 data.
"""

from __future__ import annotations

import copy
import math
import re
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
_META_ADAPTER_SITES = ("time", "freq", "fusion")
_REGISTERED_META_ADAPTER_SITE_PROFILES = (
    _META_ADAPTER_SITES,
    ("time", "fusion"),
    ("time",),
    ("fusion",),
)
_REGISTERED_META_ADAPTER_RANK_SITE_PROFILES = (
    (4, _META_ADAPTER_SITES),
    (4, ("time", "fusion")),
    (8, ("time",)),
    (4, ("fusion",)),
    (8, ("fusion",)),
)
_LEGACY_ADAPTER_PREFIXES = tuple(
    f"meta_adapter_{site}." for site in _META_ADAPTER_SITES
)
_MODEL_ARG_KEYS = frozenset(
    {
        "num_classes",
        "num_domains",
        "model_size",
        "dataset",
        "input_len",
        "sample_rate_hz",
        "id_feature_key",
        "dom_feature_key",
        "model_variant",
        "branch_ablation",
        "mixstyle_on",
        "mixstyle_p",
        "mixstyle_alpha",
        "mixstyle_eps",
        "mixstyle_layers",
        "mixstyle_use_domain_label",
        "mixstyle_mix",
        "mixstyle_strength",
        "mixstyle_fallback",
        "domain_branch_ablation",
        "domain_enhancer",
        "domain_enhancer_strength",
        "use_circularity",
        "use_freq_stats",
        "use_pa_stats",
        "use_freq_band_gate",
        "freq_feature_source",
        "pa_feature_source",
        "pa_orders",
        "use_aux_spectral_stats",
        "channel_trim_scale",
        "time_stability_mode",
        "freq_stability_mode",
        "id_time_stability_mode",
        "id_freq_stability_mode",
        "domain_time_stability_mode",
        "domain_freq_stability_mode",
        "time_stability_channels",
        "freq_stability_channels",
        "fast_infer_when_no_aux",
        "use_tx_adv_on_zdom",
        "arch_family",
        "representation_mode",
        "use_crra",
        "crra_rank",
        "crra_alpha_max",
        "crra_shrinkage",
        "crra_condition_dim",
        "crra_nuisance_dim",
        "crra_start_epoch",
        "crra_ramp_epochs",
        "sat_anchor_adapter",
        "sat_anchor_adapter_rank",
        "meta_adapter_rank",
        "meta_adapter_sites",
        "builder",
        "model_builder",
        "model_kind",
    }
)
_MODEL_ARG_INT_KEYS = frozenset(
    {
        "num_classes",
        "num_domains",
        "input_len",
        "time_stability_channels",
        "freq_stability_channels",
        "crra_rank",
        "crra_condition_dim",
        "crra_nuisance_dim",
        "crra_start_epoch",
        "crra_ramp_epochs",
        "sat_anchor_adapter_rank",
        "meta_adapter_rank",
    }
)
_MODEL_ARG_FLOAT_KEYS = frozenset(
    {
        "sample_rate_hz",
        "mixstyle_p",
        "mixstyle_alpha",
        "mixstyle_eps",
        "mixstyle_strength",
        "domain_enhancer_strength",
        "channel_trim_scale",
        "crra_alpha_max",
        "crra_shrinkage",
    }
)
_MODEL_ARG_BOOL_KEYS = frozenset(
    {
        "mixstyle_on",
        "mixstyle_use_domain_label",
        "use_circularity",
        "use_freq_stats",
        "use_pa_stats",
        "use_freq_band_gate",
        "use_aux_spectral_stats",
        "fast_infer_when_no_aux",
        "use_tx_adv_on_zdom",
        "use_crra",
        "sat_anchor_adapter",
    }
)
_MODEL_BUILDER_HINTS = frozenset(
    {
        "build_model",
        "model.build_model",
        "build_dual_model",
        "model_dual_cvsincnet.build_dual_model",
        "dual",
        "single",
    }
)
_META_ADAPTER_CONFIG_KEYS = frozenset(
    {
        "rank",
        "sites",
        "phase2_steps",
        "meta_adapter_rank",
        "meta_adapter_sites",
        "adaptation_objective",
        "support_logit_scale",
    }
)
_ADAPTATION_OBJECTIVES = frozenset(
    {
        "legacy_fixed_head_ce_v1",
        "frozen_prototype_cosine_ce_v1",
        "frozen_prototype_class_floor_ce_v1",
    }
)
_SELECTION_KEYS = frozenset({"source_split", "criterion", "seed"})
_SOURCE_SPLITS = frozenset({"V_cal", "V_select", "source_meta_validation", "L_s"})
_SOURCE_CRITERIA = frozenset(
    {
        "max_min_source_holdout_delta",
        "source_meta_validation",
        "source_holdout_min_delta",
        "source_only",
        "source_selection",
        "source_validation",
    }
)
_BASE_CHECKPOINT_KEYS = frozenset({"id", "role"})
_BASE_CHECKPOINT_ROLES = frozenset({"source_only", "legacy_adv3b02"})
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:+,\-]+$")
_CLASS_KEY_RE = re.compile(r"^[0-9]+$")


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
    adaptation_objective: str = "legacy_fixed_head_ce_v1"
    support_logit_scale: float = 1.0


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _exact_mapping(
    value: Any,
    *,
    field_name: str,
    allowed_keys: frozenset[str],
    required_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    mapping = _as_mapping(value, field_name=field_name)
    keys = set(mapping)
    if any(not isinstance(key, str) for key in keys):
        raise ValueError(f"{field_name} keys must be strings")
    unexpected = keys.difference(allowed_keys)
    missing = required_keys.difference(keys)
    if unexpected or missing:
        raise ValueError(
            f"{field_name} field allowlist mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return {str(key): mapping[key] for key in keys}


def _require_int(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return int(value)


def _require_float(value: Any, *, field_name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _require_token(value: Any, *, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{field_name} must be a non-empty token string")
    if value and _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} contains characters outside its token allowlist")
    return value


def _validate_sites(
    value: Any, *, field_name: str, require_all: bool, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = tuple(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, (list, tuple)):
        if any(not isinstance(item, str) for item in value):
            raise ValueError(f"{field_name} entries must be strings")
        raw = tuple(item.strip() for item in value)
    else:
        raise ValueError(f"{field_name} must be a comma string or string sequence")
    if not raw and allow_empty:
        return ()
    if not raw or len(set(raw)) != len(raw):
        raise ValueError(f"{field_name} must contain unique adapter sites")
    if any(item not in _META_ADAPTER_SITES for item in raw):
        raise ValueError(f"{field_name} contains an unauthorized adapter site")
    if require_all and set(raw) != set(_META_ADAPTER_SITES):
        raise ValueError(f"{field_name} must contain exactly time,freq,fusion")
    return tuple(site for site in _META_ADAPTER_SITES if site in raw)


def _validate_registered_site_profile(value: Any, *, field_name: str) -> tuple[str, ...]:
    sites = _validate_sites(value, field_name=field_name, require_all=False)
    if sites not in _REGISTERED_META_ADAPTER_SITE_PROFILES:
        raise ValueError(
            f"{field_name} must match a registered adapter site profile"
        )
    return sites


def _validate_model_args(value: Any) -> dict[str, Any]:
    args = _exact_mapping(
        value,
        field_name="model_args",
        allowed_keys=_MODEL_ARG_KEYS,
        required_keys=frozenset({"num_classes", "dataset", "input_len"}),
    )
    for key, item in args.items():
        if key in _MODEL_ARG_INT_KEYS:
            minimum = 0
            if key in {"num_classes", "num_domains", "input_len"}:
                minimum = 1
            _require_int(item, field_name=f"model_args.{key}", minimum=minimum)
        elif key in _MODEL_ARG_FLOAT_KEYS:
            _require_float(
                item,
                field_name=f"model_args.{key}",
                positive=key in {"sample_rate_hz", "mixstyle_eps"},
            )
        elif key in _MODEL_ARG_BOOL_KEYS:
            if not isinstance(item, bool):
                raise ValueError(f"model_args.{key} must be a boolean")
        elif key == "pa_orders":
            if item is None:
                continue
            if isinstance(item, str):
                values = tuple(part.strip() for part in item.split(",") if part.strip())
                if not values or any(not part.isdigit() for part in values):
                    raise ValueError("model_args.pa_orders has invalid string shape")
                parsed = tuple(int(part) for part in values)
            elif isinstance(item, (list, tuple)):
                parsed = tuple(
                    _require_int(part, field_name="model_args.pa_orders entry", minimum=1)
                    for part in item
                )
            else:
                raise ValueError("model_args.pa_orders must be a string or integer sequence")
            if any(part % 2 == 0 for part in parsed):
                raise ValueError("model_args.pa_orders must contain odd orders")
        elif key == "meta_adapter_sites":
            _validate_sites(
                item,
                field_name="model_args.meta_adapter_sites",
                require_all=False,
                allow_empty=True,
            )
        elif key in {"builder", "model_builder", "model_kind"}:
            token = _require_token(item, field_name=f"model_args.{key}")
            if key == "model_kind" and token not in {"single", "dual"}:
                raise ValueError("model_args.model_kind must be single or dual")
            if key != "model_kind" and token not in _MODEL_BUILDER_HINTS:
                raise ValueError(f"model_args.{key} is not an allowed builder hint")
        else:
            _require_token(item, field_name=f"model_args.{key}", allow_empty=key == "branch_ablation")

    if "meta_adapter_sites" in args:
        if "meta_adapter_rank" not in args:
            raise ValueError("model_args.meta_adapter_sites requires meta_adapter_rank")
        adapter_rank = int(args["meta_adapter_rank"])
        if adapter_rank == 0:
            _validate_sites(
                args["meta_adapter_sites"],
                field_name="model_args.meta_adapter_sites",
                require_all=False,
                allow_empty=True,
            )
        else:
            adapter_sites = _validate_registered_site_profile(
                args["meta_adapter_sites"],
                field_name="model_args.meta_adapter_sites",
            )
            if (adapter_rank, adapter_sites) not in _REGISTERED_META_ADAPTER_RANK_SITE_PROFILES:
                raise ValueError(
                    "model_args must match a registered rank/site profile"
                )

    dual_hint = any(
        str(args.get(key, "")).lower() in {"dual", "build_dual_model", "model_dual_cvsincnet.build_dual_model"}
        for key in ("builder", "model_builder", "model_kind")
    )
    dual_keys = {"num_domains", "id_feature_key", "dom_feature_key"}.intersection(args)
    if dual_hint or dual_keys:
        required_dual = {"num_domains", "id_feature_key", "dom_feature_key"}
        if not required_dual.issubset(args):
            raise ValueError("dual model_args require num_domains,id_feature_key,dom_feature_key")
    return args


def _validate_meta_adapter_config(value: Any) -> dict[str, Any]:
    config = _exact_mapping(
        value,
        field_name="meta_adapter_config",
        allowed_keys=_META_ADAPTER_CONFIG_KEYS,
    )
    rank_keys = {key for key in ("rank", "meta_adapter_rank") if key in config}
    site_keys = {key for key in ("sites", "meta_adapter_sites") if key in config}
    if len(rank_keys) != 1 or len(site_keys) != 1:
        raise ValueError(
            "meta_adapter_config must contain exactly one rank key and one sites key"
        )
    rank_key = next(iter(rank_keys))
    site_key = next(iter(site_keys))
    rank = _require_int(config[rank_key], field_name=f"meta_adapter_config.{rank_key}", minimum=1)
    sites = _validate_registered_site_profile(
        config[site_key],
        field_name=f"meta_adapter_config.{site_key}",
    )
    if (rank, sites) not in _REGISTERED_META_ADAPTER_RANK_SITE_PROFILES:
        raise ValueError(
            "meta_adapter_config must match a registered rank/site profile"
        )
    if "phase2_steps" in config:
        _require_int(config["phase2_steps"], field_name="meta_adapter_config.phase2_steps", minimum=1)
        if int(config["phase2_steps"]) > 5:
            raise ValueError("meta_adapter_config.phase2_steps must be <= 5")
    alignment_keys = {
        key
        for key in ("adaptation_objective", "support_logit_scale")
        if key in config
    }
    if alignment_keys and alignment_keys != {
        "adaptation_objective",
        "support_logit_scale",
    }:
        raise ValueError(
            "meta_adapter_config adaptation_objective and support_logit_scale must appear together"
        )
    if alignment_keys:
        objective = config["adaptation_objective"]
        if not isinstance(objective, str) or objective not in _ADAPTATION_OBJECTIVES:
            raise ValueError("meta_adapter_config.adaptation_objective is not registered")
        scale = _require_float(
            config["support_logit_scale"],
            field_name="meta_adapter_config.support_logit_scale",
            positive=True,
        )
        if scale > 64.0:
            raise ValueError("meta_adapter_config.support_logit_scale must be <= 64")
        config["adaptation_objective"] = objective
        config["support_logit_scale"] = scale
    return config


def _validate_selection(value: Any) -> dict[str, Any]:
    try:
        selection = _exact_mapping(
            value,
            field_name="selection",
            allowed_keys=_SELECTION_KEYS,
            required_keys=_SELECTION_KEYS,
        )
    except ValueError as exc:
        raise ValueError(
            f"Phase1 source selection must use the exact source-only field allowlist: {exc}"
        ) from exc
    if not isinstance(selection["source_split"], str) or selection["source_split"] not in _SOURCE_SPLITS:
        raise ValueError("selection.source_split must identify a source-only split")
    if not isinstance(selection["criterion"], str) or selection["criterion"] not in _SOURCE_CRITERIA:
        raise ValueError("selection.criterion is not an allowed source-only criterion")
    _require_int(selection["seed"], field_name="selection.seed", minimum=0)
    return selection


def _validate_base_checkpoint(value: Any) -> dict[str, Any]:
    checkpoint = _exact_mapping(
        value,
        field_name="base_checkpoint",
        allowed_keys=_BASE_CHECKPOINT_KEYS,
        required_keys=_BASE_CHECKPOINT_KEYS,
    )
    _require_token(checkpoint["id"], field_name="base_checkpoint.id")
    if not isinstance(checkpoint["role"], str) or checkpoint["role"] not in _BASE_CHECKPOINT_ROLES:
        raise ValueError("base_checkpoint.role must identify source-only legacy provenance")
    return checkpoint


def _validate_class_mapping(value: Any) -> dict[str, str]:
    mapping = _as_mapping(value, field_name="class_mapping")
    if not mapping:
        raise ValueError("class_mapping must be non-empty")
    if any(not isinstance(key, str) or _CLASS_KEY_RE.fullmatch(key) is None for key in mapping):
        raise ValueError("class_mapping keys must be contiguous decimal strings")
    indices = sorted(int(key) for key in mapping)
    if indices != list(range(len(indices))):
        raise ValueError("class_mapping keys must be contiguous from zero")
    result: dict[str, str] = {}
    for key, label in mapping.items():
        result[key] = _require_token(label, field_name=f"class_mapping[{key}]")
    return result


def _validate_prototypes(value: Any, *, class_mapping: Mapping[str, str]) -> Any:
    if torch.is_tensor(value):
        if value.ndim != 2 or value.shape[0] != len(class_mapping):
            raise ValueError("prototypes tensor must have shape [class_count,feature_dim]")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError("prototypes tensor must be finite floating-point data")
        return value
    mapping = _as_mapping(value, field_name="prototypes")
    if set(mapping) != set(class_mapping):
        raise ValueError("prototypes keys must exactly match class_mapping keys")
    shape: tuple[int, ...] | None = None
    result: dict[str, Tensor] = {}
    for key, prototype in mapping.items():
        if not isinstance(key, str):
            raise ValueError("prototypes keys must be strings")
        if not torch.is_tensor(prototype) or prototype.ndim != 1 or prototype.numel() <= 0:
            raise ValueError("each prototype must be a non-empty rank-1 tensor")
        if not torch.is_floating_point(prototype) or not torch.isfinite(prototype).all():
            raise ValueError("each prototype must be finite floating-point data")
        if shape is None:
            shape = tuple(prototype.shape)
        elif tuple(prototype.shape) != shape:
            raise ValueError("all prototypes must have the same shape")
        result[key] = prototype
    return result


def _validate_bundle_metadata(
    config: Mapping[str, Any], selection: Mapping[str, Any], *, bundle: bool = False
) -> dict[str, Any]:
    config_mapping = _as_mapping(config, field_name="config")
    if bundle:
        if set(config_mapping) != REQUIRED_META_BUNDLE_KEYS:
            raise ValueError(
                "meta bundle top-level fields must be exactly schema,model_args,"
                "meta_adapter_config,selection,base_checkpoint,class_mapping,"
                "prototypes,model_state"
            )
        metadata = {key: config_mapping[key] for key in _META_CONFIG_KEYS}
    else:
        if set(config_mapping) != _META_CONFIG_KEYS:
            raise ValueError(
                "config top-level fields must be exactly model_args,meta_adapter_config,"
                "base_checkpoint,class_mapping,prototypes"
            )
        metadata = config_mapping
    model_args = _validate_model_args(metadata["model_args"])
    meta_adapter_config = _validate_meta_adapter_config(metadata["meta_adapter_config"])
    rank_key = "meta_adapter_rank" if "meta_adapter_rank" in meta_adapter_config else "rank"
    site_key = "meta_adapter_sites" if "meta_adapter_sites" in meta_adapter_config else "sites"
    if "meta_adapter_rank" in model_args and model_args["meta_adapter_rank"] != meta_adapter_config[rank_key]:
        raise ValueError("model_args and meta_adapter_config adapter rank mismatch")
    if "meta_adapter_sites" in model_args:
        model_sites = _validate_sites(
            model_args["meta_adapter_sites"],
            field_name="model_args.meta_adapter_sites",
            require_all=False,
        )
        config_sites = _validate_registered_site_profile(
            meta_adapter_config[site_key],
            field_name=f"meta_adapter_config.{site_key}",
        )
        if model_sites != config_sites:
            raise ValueError("model_args and meta_adapter_config adapter sites mismatch")
    class_mapping = _validate_class_mapping(metadata["class_mapping"])
    return {
        "model_args": model_args,
        "meta_adapter_config": meta_adapter_config,
        "selection": _validate_selection(selection),
        "base_checkpoint": _validate_base_checkpoint(metadata["base_checkpoint"]),
        "class_mapping": class_mapping,
        "prototypes": _validate_prototypes(
            metadata["prototypes"], class_mapping=class_mapping
        ),
    }


def _is_legacy_adapter_state_key(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _LEGACY_ADAPTER_PREFIXES)


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
    adapter_state_keys = [key for key in state if _is_legacy_adapter_state_key(key)]
    if adapter_state_keys:
        raise ValueError(
            "legacy payload must not contain adapter state; "
            f"keys={sorted(adapter_state_keys)}"
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
    target_state_keys = set(model.state_dict())
    allowed_missing = {
        name
        for name in target_state_keys
        if _is_legacy_adapter_state_key(name)
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
    validated = _validate_bundle_metadata(config_mapping, selection, bundle=False)

    payload = {
        "schema": META_BUNDLE_SCHEMA,
        "model_state": _bundle_model_state(model),
        "model_args": _snapshot_for_payload(validated["model_args"]),
        "meta_adapter_config": _snapshot_for_payload(validated["meta_adapter_config"]),
        "selection": _snapshot_for_payload(validated["selection"]),
        "base_checkpoint": _snapshot_for_payload(validated["base_checkpoint"]),
        "class_mapping": _snapshot_for_payload(validated["class_mapping"]),
        "prototypes": _snapshot_for_payload(validated["prototypes"]),
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
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ValueError(
            "weights_only=True is required for strict meta bundle loading; "
            "this PyTorch runtime does not support the safe loader"
        ) from exc
    except Exception as exc:  # pragma: no cover - torch error wording is version-specific
        raise ValueError(f"cannot load meta bundle: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("meta bundle payload must be a mapping")
    payload_keys = set(payload)
    if any(not isinstance(key, str) for key in payload_keys):
        raise ValueError("meta bundle top-level keys must be strings")
    if payload_keys != REQUIRED_META_BUNDLE_KEYS:
        raise ValueError(
            "meta bundle top-level fields must be exactly required fields: "
            f"missing={sorted(REQUIRED_META_BUNDLE_KEYS.difference(payload_keys))} "
            f"unexpected={sorted(payload_keys.difference(REQUIRED_META_BUNDLE_KEYS))}"
        )
    schema = payload.get("schema")
    if not isinstance(schema, str) or schema != META_BUNDLE_SCHEMA:
        raise ValueError(f"meta bundle schema mismatch: {schema!r}")

    validated = _validate_bundle_metadata(payload, payload["selection"], bundle=True)
    model_args = validated["model_args"]
    meta_adapter_config = validated["meta_adapter_config"]
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
        base_checkpoint_id=_checkpoint_identifier(validated["base_checkpoint"]),
        class_mapping=_freeze_for_audit(validated["class_mapping"]),
        prototypes=_freeze_for_audit(validated["prototypes"]),
        selection=_freeze_for_audit(validated["selection"]),
        adaptation_objective=str(
            meta_adapter_config.get(
                "adaptation_objective", "legacy_fixed_head_ce_v1"
            )
        ),
        support_logit_scale=float(
            meta_adapter_config.get("support_logit_scale", 1.0)
        ),
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
