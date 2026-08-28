"""Support-only S02 long-horizon ``t3.norm`` delta for D92-E0-NORF32.

The SF-TAPFT target head is an optimization-time helper only.  This boundary
returns no head and persists exactly the two canonical ``t3.norm`` affine
deltas consumed by the unified runner before its fixed D92-E0 head is built.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from .target_only_progressive_adapt import (
    SFTAPFTConfig,
    TargetOnlyAdaptationDataset,
    select_sf_tapft_by_grouped_cv,
)


_PERSISTENT_PARAMETER_NAMES = (
    "model.t3.norm.weight",
    "model.t3.norm.bias",
)
_LONG_HORIZON_STEPS = (4500, 0, 0)


@dataclass(frozen=True)
class S02LongHorizonSpec:
    """Frozen candidate specification shared by local tests and the runner."""

    row_id: str
    candidate_id: str
    method_lock: str
    rf32_used: bool
    persistent_parameter_names: tuple[str, str]
    selection_mode: str
    folds: int
    config: SFTAPFTConfig


@dataclass(frozen=True)
class S02PersistentDelta:
    """The only deployment state produced by S02 support adaptation."""

    parameter_deltas: Mapping[str, Tensor]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        keys = tuple(self.parameter_deltas)
        if keys != _PERSISTENT_PARAMETER_NAMES:
            raise ValueError("S02 delta must contain exactly the two t3.norm affine keys")
        frozen_deltas: dict[str, Tensor] = {}
        for name in keys:
            value = self.parameter_deltas[name]
            if not torch.is_tensor(value) or not value.is_floating_point():
                raise ValueError("S02 persistent deltas must be floating-point tensors")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("S02 persistent deltas must be finite")
            frozen_deltas[name] = value.detach().cpu().clone()
        object.__setattr__(
            self,
            "parameter_deltas",
            MappingProxyType(frozen_deltas),
        )
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def build_s02_long_horizon_spec() -> S02LongHorizonSpec:
    """Return the fixed long-horizon S02 configuration."""

    return S02LongHorizonSpec(
        row_id="S02",
        candidate_id="SF_TAPFT_NORM_T3_LONG_D92_E0_NORF32",
        method_lock="D92-E0-NORF32",
        rf32_used=False,
        persistent_parameter_names=_PERSISTENT_PARAMETER_NAMES,
        selection_mode="grouped_oof_full_support_refit",
        folds=4,
        config=SFTAPFTConfig(
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            phase_steps=_LONG_HORIZON_STEPS,
            scheduler_reference_steps=0,
            validation_steps=(),
            checkpoint_average_top_k=3,
            seed=392002,
        ),
    )


def _audit_value(audit: object, name: str, default: Any = None) -> Any:
    if isinstance(audit, Mapping):
        return audit.get(name, default)
    return getattr(audit, name, default)


def _require_support_only_audit(result: object) -> None:
    audit = getattr(result, "audit", None)
    forbidden_access = (
        "source_loader_opened",
        "source_samples_opened",
        "source_cache_opened",
        "target_eval_opened",
        "query_opened",
    )
    if audit is None or any(bool(_audit_value(audit, name, True)) for name in forbidden_access):
        raise ValueError("S02 adaptation audit must remain query/source-free")
    if tuple(_audit_value(audit, "nonpermitted_changed_names", ())) != ():
        raise ValueError("S02 adaptation changed a non-permitted model parameter")


def _anchor_name_for(
    anchors: Mapping[str, Tensor], canonical_name: str
) -> str:
    suffix = canonical_name.removeprefix("model.")
    matches = tuple(
        name
        for name in anchors
        if name == canonical_name
        or (name.startswith("model.") and name.endswith(suffix))
    )
    if len(matches) != 1:
        raise ValueError(
            f"S02 fitted result must expose exactly one anchor for {canonical_name}"
        )
    return matches[0]


def _persistent_deltas(result: object) -> Mapping[str, Tensor]:
    fitted_model = getattr(result, "model", None)
    anchors = getattr(result, "base_parameter_anchors", None)
    if not isinstance(fitted_model, nn.Module) or not isinstance(anchors, Mapping):
        raise TypeError("S02 fitter must return a model and base_parameter_anchors")
    current = dict(fitted_model.named_parameters())
    deltas: dict[str, Tensor] = {}
    for canonical_name in _PERSISTENT_PARAMETER_NAMES:
        anchor_name = _anchor_name_for(anchors, canonical_name)
        model_name = anchor_name.removeprefix("model.")
        if model_name not in current:
            raise ValueError(f"S02 fitted model is missing {model_name}")
        anchor = anchors[anchor_name]
        value = current[model_name]
        if not torch.is_tensor(anchor) or anchor.shape != value.shape:
            raise ValueError(f"S02 anchor geometry mismatch for {canonical_name}")
        deltas[canonical_name] = (
            value.detach().cpu() - anchor.detach().to(device="cpu", dtype=value.dtype)
        )
    return MappingProxyType(deltas)


def _zero_persistent_deltas(model: nn.Module) -> Mapping[str, Tensor]:
    named = dict(model.named_parameters())
    output: dict[str, Tensor] = {}
    for canonical_name in _PERSISTENT_PARAMETER_NAMES:
        short_name = canonical_name.removeprefix("model.")
        if short_name not in named:
            raise ValueError(f"S02 checkpoint model is missing {short_name}")
        output[canonical_name] = torch.zeros_like(named[short_name], device="cpu")
    return MappingProxyType(output)


def adapt_s02_support_only(
    checkpoint_model: nn.Module,
    support: TargetOnlyAdaptationDataset,
    *,
    grouped_selector: Callable[..., Any] = select_sf_tapft_by_grouped_cv,
) -> S02PersistentDelta:
    """Fit S02 from support only and return no temporary target head state."""

    if not isinstance(checkpoint_model, nn.Module):
        raise TypeError("checkpoint_model must be a torch module")
    if not isinstance(support, TargetOnlyAdaptationDataset):
        raise TypeError("support must be a TargetOnlyAdaptationDataset")
    if not callable(grouped_selector):
        raise TypeError("grouped_selector must be callable")

    spec = build_s02_long_horizon_spec()
    selection = grouped_selector(
        checkpoint_model,
        support,
        spec.config,
        folds=spec.folds,
        full_support_refit=True,
    )
    result = getattr(selection, "full_support_result", None)
    selected = str(getattr(selection, "selected", ""))
    if result is None and selected in {"zero_adapt", "frozen"}:
        return S02PersistentDelta(
            parameter_deltas=_zero_persistent_deltas(checkpoint_model),
            audit={
                "row_id": spec.row_id,
                "candidate_id": spec.candidate_id,
                "training_horizon": "long",
                "phase_steps": spec.config.phase_steps,
                "selection_mode": spec.selection_mode,
                "selection_folds": spec.folds,
                "selection_phase_steps": spec.config.phase_steps,
                "selected_phase_steps": tuple(selection.selected_phase_steps),
                "selected": selected,
                "zero_delta_fallback": True,
                "full_support_refit": False,
                "method_lock": spec.method_lock,
                "d92_variant": "E0",
                "rf32_used": False,
                "support_only": True,
                "query_rows_used": 0,
                "query_opened": False,
                "source_opened": False,
                "temporary_target_head_discarded": True,
                "target_head_persisted": False,
                "persistent_parameter_names": spec.persistent_parameter_names,
            },
        )
    if (
        result is None
        or getattr(selection, "fold0_as_final", True) is not False
        or int(getattr(selection, "final_training_sample_count", -1))
        != len(support.physical_ids)
    ):
        raise ValueError("S02 grouped selection must produce one full-support refit")
    _require_support_only_audit(result)
    deltas = _persistent_deltas(result)
    audit = {
        "row_id": spec.row_id,
        "candidate_id": spec.candidate_id,
        "training_horizon": "long",
        "phase_steps": spec.config.phase_steps,
        "scheduler_reference_steps": spec.config.scheduler_reference_steps,
        "checkpoint_selection_mode": "support_grouped_oof_then_final_step",
        "trainability_profile": spec.config.trainability_profile,
        "norm_rules": spec.config.norm_rules,
        "method_lock": spec.method_lock,
        "d92_variant": "E0",
        "rf32_used": spec.rf32_used,
        "support_only": True,
        "query_rows_used": 0,
        "query_opened": False,
        "source_opened": False,
        "temporary_target_head_discarded": True,
        "target_head_persisted": False,
        "persistent_parameter_names": spec.persistent_parameter_names,
        "selection_mode": spec.selection_mode,
        "selection_folds": spec.folds,
        "selection_phase_steps": spec.config.phase_steps,
        "selected_phase_steps": tuple(selection.selected_phase_steps),
        "selected": selected,
        "zero_delta_fallback": False,
        "full_support_refit": True,
    }
    return S02PersistentDelta(parameter_deltas=deltas, audit=audit)


__all__ = [
    "S02LongHorizonSpec",
    "S02PersistentDelta",
    "adapt_s02_support_only",
    "build_s02_long_horizon_spec",
]
