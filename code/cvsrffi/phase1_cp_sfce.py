"""Frozen conflict-projected class-balanced satellite focal CE for Phase1.

P1-CP-SFCE deliberately preserves the P1-CB-SFCE data, loss and model path.
Only the additional satellite focal-CE gradient is projected against the shared
GeoSat-C base gradient before the one normal optimizer step.  It does not add a
teacher, alignment target, domain/RX input, classifier head or threshold.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from cvsrffi.phase1_cb_sfce import (
    CBSFCEConfig,
    CBSFCEConfigurationError,
    CBSFCERuntimeError,
    FROZEN_CB_SFCE_GAMMA,
    FROZEN_CB_SFCE_LAMBDA,
    FROZEN_CB_SFCE_SCENARIOS,
    cb_sfce_config_receipt,
    cb_sfce_loss,
    remap_cb_sfce_local_labels_to_head_rows,
    resolve_cb_sfce_classifier_weight,
    resolve_cb_sfce_local_head_class_binding,
    strict_cb_sfce_warm_start,
    validate_cb_sfce_args,
    validate_cb_sfce_logit_binding,
)


FROZEN_CP_SFCE_LAMBDA = FROZEN_CB_SFCE_LAMBDA
FROZEN_CP_SFCE_GAMMA = FROZEN_CB_SFCE_GAMMA
FROZEN_CP_SFCE_SCENARIOS = FROZEN_CB_SFCE_SCENARIOS
FROZEN_CP_SFCE_PROJECTED_DOT_TOLERANCE_FACTOR = 1e-6


class CPSFCEConfigurationError(ValueError):
    """Raised when the frozen P1-CP-SFCE C/G contract drifts."""


class CPSFCERuntimeError(RuntimeError):
    """Raised when the CP-SFCE gradient/receipt contract cannot be proved."""


@dataclass(frozen=True)
class CPSFCEConfig:
    """Immutable settings consumed by the CP-SFCE training-loop integration."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float
    gamma: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise CPSFCEConfigurationError(f"{name} must be numeric") from exc


def _translate_configuration(error: BaseException) -> CPSFCEConfigurationError:
    return CPSFCEConfigurationError(str(error).replace("P1-CB-SFCE", "P1-CP-SFCE"))


def _translate_runtime(error: BaseException) -> CPSFCERuntimeError:
    return CPSFCERuntimeError(str(error).replace("P1-CB-SFCE", "P1-CP-SFCE"))


def _compatibility_args(args: Any, config: CPSFCEConfig) -> SimpleNamespace:
    """Map CP flags onto the proven CB-SFCE input/data-contract validator."""

    try:
        values = dict(vars(args))
    except TypeError as exc:
        raise CPSFCEConfigurationError("P1-CP-SFCE requires an argument namespace") from exc
    values["phase1_cb_sfce_frozen_mode"] = bool(config.frozen_mode)
    values["phase1_cb_sfce_enabled"] = bool(config.enabled)
    values["lambda_cb_sfce"] = float(config.loss_weight)
    values["cb_sfce_gamma"] = float(config.gamma)
    return SimpleNamespace(**values)


def validate_cp_sfce_args(args: Any) -> CPSFCEConfig:
    """Validate the frozen CP-SFCE continuation without changing CB-SFCE rules."""

    frozen_mode = _bool_arg(args, "phase1_cp_sfce_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_cp_sfce_enabled", False)
    loss_weight = _float_arg(args, "lambda_cp_sfce", 0.0)
    gamma = _float_arg(args, "cp_sfce_gamma", FROZEN_CP_SFCE_GAMMA)
    if not frozen_mode and not enabled:
        return CPSFCEConfig(False, False, 0.0, FROZEN_CP_SFCE_GAMMA)
    if enabled and not frozen_mode:
        raise CPSFCEConfigurationError(
            "--phase1_cp_sfce_enabled requires --phase1_cp_sfce_frozen_mode true"
        )
    expected_weight = FROZEN_CP_SFCE_LAMBDA if enabled else 0.0
    if not math.isfinite(loss_weight) or abs(loss_weight - expected_weight) > 1e-12:
        raise CPSFCEConfigurationError(
            f"Frozen P1-CP-SFCE requires lambda_cp_sfce={expected_weight:.12g}"
        )
    if not math.isfinite(gamma) or abs(gamma - FROZEN_CP_SFCE_GAMMA) > 1e-12:
        raise CPSFCEConfigurationError(
            f"Frozen P1-CP-SFCE requires cp_sfce_gamma={FROZEN_CP_SFCE_GAMMA:.12g}"
        )
    if _bool_arg(args, "phase1_cb_sfce_frozen_mode", False) or _bool_arg(
        args, "phase1_cb_sfce_enabled", False
    ):
        raise CPSFCEConfigurationError("Frozen P1-CP-SFCE forbids stacking P1-CB-SFCE")
    config = CPSFCEConfig(True, enabled, loss_weight, gamma)
    try:
        validate_cb_sfce_args(_compatibility_args(args, config))
    except CBSFCEConfigurationError as exc:
        raise _translate_configuration(exc) from exc
    return config


def cp_sfce_config_receipt(config: CPSFCEConfig) -> Dict[str, Any]:
    """Create a data-free receipt with CP-only gradient-contract fields."""

    base = cb_sfce_config_receipt(
        CBSFCEConfig(
            frozen_mode=bool(config.frozen_mode),
            enabled=bool(config.enabled),
            loss_weight=float(config.loss_weight),
            gamma=float(config.gamma),
        )
    )
    for key in tuple(base):
        if key.startswith("cb_sfce_"):
            base.pop(key, None)
    base.update(
        {
            "schema": "cvs.phase1.cp_sfce_receipt.v2",
            "method": "P1_CP_SFCE",
            "loss_rule": "PRESENT_TX_EQUAL_MEAN_FOCAL_CE_ON_SINGLE_LEO_TX_LOGITS",
            "gradient_rule": "PROJECT_ADDITIONAL_SFCE_GRADIENT_AGAINST_COMMON_BASE_ON_CONFLICT",
            "projection_scope": "LEO_LOGITS_PATH_TRAINABLE_ID_ENCODER_PLUS_EXACT_CLS_HEAD",
            "uses_gradient_projection": bool(config.enabled),
            "uses_clean_feature": False,
            "uses_clean_logits": False,
            "uses_teacher": False,
            "uses_external_ema_teacher": False,
            "uses_new_head": False,
            "uses_threshold": False,
            "uses_rx_labels": False,
            "uses_domain_labels": False,
            "uses_grl": False,
            "uses_mmd": False,
            "uses_coral": False,
            "uses_explicit_z_alignment": False,
            "uses_proxy_rows": False,
            "uses_held_rows": False,
            "cp_sfce_batches": 0,
            "cp_sfce_rows": 0,
            "cp_sfce_cells": {},
            "cp_sfce_projection_batches": 0,
            "cp_sfce_first_epoch_marker_completed": False,
            "cp_sfce_first_epoch_marker": {},
            "cp_sfce_conflict_batches": {"shared_encoder": 0, "classifier_head": 0},
            "cp_sfce_base_zero_batches": {"shared_encoder": 0, "classifier_head": 0},
            "cp_sfce_outside_aux_zero_or_none_batches": 0,
            "cp_sfce_optimizer_state_step_batches": 0,
            "cp_sfce_optimizer_state_no_step_batches": 0,
            "cp_sfce_base_scaled_overflow_raw_finite_batches": 0,
            "cp_sfce_aux_scaled_overflow_raw_finite_batches": 0,
            "cp_sfce_base_raw_nonfinite_or_disconnected_batches": 0,
            "cp_sfce_aux_raw_nonfinite_or_disconnected_batches": 0,
            "cp_sfce_aux_raw_outside_violation_batches": 0,
            "cp_sfce_amp_skip_scale_decreased_batches": 0,
            "cp_sfce_amp_skip_optimizer_state_unchanged_batches": 0,
            "cp_sfce_last_amp_overflow": {},
            "cp_sfce_last_projection": {},
            "cp_sfce_terminal_contract": "PENDING",
            "cp_sfce_terminal_contract_passed": False,
        }
    )
    return base


def cp_sfce_loss(
    satellite_tx_logits: torch.Tensor,
    source_tx_labels: torch.Tensor,
    *,
    gamma: float = FROZEN_CP_SFCE_GAMMA,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Reuse the frozen CB-SFCE focal loss exactly; CP changes gradients only."""

    try:
        return cb_sfce_loss(satellite_tx_logits, source_tx_labels, gamma=gamma)
    except (CBSFCEConfigurationError, CBSFCERuntimeError) as exc:
        if isinstance(exc, CBSFCEConfigurationError):
            raise _translate_configuration(exc) from exc
        raise _translate_runtime(exc) from exc


def resolve_cp_sfce_local_head_class_binding(**kwargs: Any) -> Dict[str, Any]:
    try:
        return resolve_cb_sfce_local_head_class_binding(**kwargs)
    except CBSFCEConfigurationError as exc:
        raise _translate_configuration(exc) from exc


def remap_cp_sfce_local_labels_to_head_rows(
    local_labels: torch.Tensor,
    local_to_head_class_ids: Sequence[Any],
) -> torch.Tensor:
    try:
        return remap_cb_sfce_local_labels_to_head_rows(local_labels, local_to_head_class_ids)
    except CBSFCERuntimeError as exc:
        raise _translate_runtime(exc) from exc


def resolve_cp_sfce_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    try:
        return resolve_cb_sfce_classifier_weight(model)
    except CBSFCERuntimeError as exc:
        raise _translate_runtime(exc) from exc


def validate_cp_sfce_logit_binding(
    *,
    model: torch.nn.Module,
    tx_logits: torch.Tensor,
    tx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
) -> torch.nn.Parameter:
    try:
        return validate_cb_sfce_logit_binding(
            model=model,
            tx_logits=tx_logits,
            tx_labels=tx_labels,
            expected_class_ids=expected_class_ids,
        )
    except CBSFCERuntimeError as exc:
        raise _translate_runtime(exc) from exc


def strict_cp_sfce_warm_start(
    model: torch.nn.Module,
    checkpoint_model_state: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> Dict[str, Any]:
    try:
        return strict_cb_sfce_warm_start(model, checkpoint_model_state, **kwargs)
    except CBSFCEConfigurationError as exc:
        raise _translate_configuration(exc) from exc


def _unique_parameters(parameters: Iterable[torch.nn.Parameter]) -> Tuple[torch.nn.Parameter, ...]:
    result = []
    seen = set()
    for parameter in parameters:
        if not isinstance(parameter, torch.nn.Parameter):
            raise CPSFCERuntimeError("P1-CP-SFCE parameter scope contains a non-Parameter")
        marker = id(parameter)
        if marker not in seen:
            seen.add(marker)
            result.append(parameter)
    return tuple(result)


def cp_sfce_parameter_scopes(model: torch.nn.Module) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    """Resolve all trainable LEO-logit-path parameters and exact classifier head.

    The VJP itself is requested over every trainable model parameter.  This
    function separately freezes the intended projection scope and makes the
    remaining trainables auditable as auxiliary-only paths.
    """

    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise CPSFCERuntimeError("P1-CP-SFCE requires model.id_backbone")
    head_weight = resolve_cp_sfce_classifier_weight(raw_model)
    try:
        head_module = id_backbone.cls_head.head
    except AttributeError as exc:
        raise CPSFCERuntimeError("P1-CP-SFCE requires exact id_backbone.cls_head.head") from exc
    head = _unique_parameters(parameter for parameter in head_module.parameters() if parameter.requires_grad)
    if not head or id(head_weight) not in {id(parameter) for parameter in head}:
        raise CPSFCERuntimeError("P1-CP-SFCE exact classifier head scope is invalid")
    head_ids = {id(parameter) for parameter in head}
    auxiliary_only_prefixes = (
        "con_proj.",
        "cls_head.imp_merge.",
        "cls_head.dac_head.",
        "cls_head.pa_head.",
    )
    encoder = _unique_parameters(
        parameter
        for name, parameter in id_backbone.named_parameters()
        if parameter.requires_grad
        and id(parameter) not in head_ids
        and not str(name).startswith(auxiliary_only_prefixes)
    )
    if not encoder:
        raise CPSFCERuntimeError("P1-CP-SFCE shared encoder scope is empty")
    all_trainable = _unique_parameters(
        parameter for parameter in raw_model.parameters() if parameter.requires_grad
    )
    scope = _unique_parameters((*encoder, *head))
    scope_ids = {id(parameter) for parameter in scope}
    if not all_trainable or not scope_ids.issubset({id(parameter) for parameter in all_trainable}):
        raise CPSFCERuntimeError("P1-CP-SFCE projection scope is not a live model subset")
    outside = tuple(parameter for parameter in all_trainable if id(parameter) not in scope_ids)
    return {
        "all_trainable": all_trainable,
        "shared_encoder": encoder,
        "classifier_head": head,
        "outside_scope": outside,
    }


def _trainable_parameter_names(
    model: torch.nn.Module, parameters: Sequence[torch.nn.Parameter]
) -> Dict[int, str]:
    """Bind data-free gradient diagnostics to live named parameters."""

    raw_model = getattr(model, "_orig_mod", model)
    required_ids = {id(parameter) for parameter in parameters}
    names = {
        id(parameter): str(name)
        for name, parameter in raw_model.named_parameters()
        if parameter.requires_grad and id(parameter) in required_ids
    }
    if required_ids != set(names):
        raise CPSFCERuntimeError("P1-CP-SFCE trainable parameter-name binding is incomplete")
    return names


def _raw_vjp_overflow_audit(
    *,
    loss: torch.Tensor,
    all_trainable: Sequence[torch.nn.Parameter],
    parameter_scopes: Mapping[str, Sequence[torch.nn.Parameter]],
    names: Mapping[int, str],
    term_name: str,
    require_outside_zero: bool,
) -> Dict[str, Any]:
    """Differentiate an unscaled term once only after its scaled VJP overflow.

    The extra VJP is deliberately exceptional-path work.  It distinguishes a
    finite base gradient whose loss-scale multiplication overflowed from an
    actual raw gradient failure without changing any successful-batch path.
    """

    try:
        gradients = torch.autograd.grad(
            loss,
            tuple(all_trainable),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
    except RuntimeError as error:
        raise CPSFCERuntimeError(f"P1-CP-SFCE raw {term_name} VJP audit failed") from error
    scope_ids = {
        id(parameter)
        for group in ("shared_encoder", "classifier_head")
        for parameter in parameter_scopes[group]
    }
    nonfinite = []
    missing_scope = []
    outside_nonzero = []
    for parameter, gradient in zip(all_trainable, gradients):
        name = names[id(parameter)]
        if gradient is None:
            if id(parameter) in scope_ids:
                missing_scope.append(name)
            continue
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            nonfinite.append(name)
        elif require_outside_zero and id(parameter) not in scope_ids and int(
            torch.count_nonzero(gradient.detach()).item()
        ) != 0:
            outside_nonzero.append(name)
    return {
        "raw_vjp_finite": not nonfinite and not missing_scope and not outside_nonzero,
        "raw_nonfinite_parameter_names": tuple(sorted(nonfinite)),
        "raw_missing_scope_parameter_names": tuple(sorted(missing_scope)),
        "raw_outside_nonzero_parameter_names": tuple(sorted(outside_nonzero)),
    }


def _require_scalar_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor) or tensor.ndim != 0:
        raise CPSFCERuntimeError(f"P1-CP-SFCE requires scalar {name}")
    if not bool(torch.isfinite(tensor.detach()).item()):
        raise CPSFCERuntimeError(f"P1-CP-SFCE {name} is non-finite")


def _require_finite_gradient(name: str, gradient: Optional[torch.Tensor]) -> torch.Tensor:
    if gradient is None:
        raise CPSFCERuntimeError(f"P1-CP-SFCE {name} gradient is missing or disconnected")
    if not bool(torch.isfinite(gradient.detach()).all().item()):
        raise CPSFCERuntimeError(f"P1-CP-SFCE {name} gradient is non-finite")
    return gradient.detach()


def _norm_dot(
    base_grads: Sequence[torch.Tensor], aux_grads: Sequence[torch.Tensor]
) -> Tuple[float, float, float]:
    base_sq = 0.0
    aux_sq = 0.0
    dot = 0.0
    for base_grad, aux_grad in zip(base_grads, aux_grads):
        base_value = base_grad.detach().double()
        aux_value = aux_grad.detach().double()
        base_sq += float(torch.sum(base_value * base_value).item())
        aux_sq += float(torch.sum(aux_value * aux_value).item())
        dot += float(torch.sum(base_value * aux_value).item())
    if not all(math.isfinite(value) for value in (base_sq, aux_sq, dot)):
        raise CPSFCERuntimeError("P1-CP-SFCE gradient norm or dot is non-finite")
    return base_sq, aux_sq, dot


def _projected_dot_tolerance(base_norm: float, aux_norm: float) -> float:
    return float(
        FROZEN_CP_SFCE_PROJECTED_DOT_TOLERANCE_FACTOR
        * max(1.0, float(base_norm) * float(aux_norm))
    )


def _state_step_value(state: Mapping[str, Any]) -> float:
    value = state.get("step", 0)
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CPSFCERuntimeError("P1-CP-SFCE optimizer state step is invalid") from exc
    if not math.isfinite(result) or result < 0.0:
        raise CPSFCERuntimeError("P1-CP-SFCE optimizer state step is non-finite")
    return result


def cp_sfce_capture_optimizer_steps(
    optimizer: torch.optim.Optimizer,
    parameter_scopes: Mapping[str, Sequence[torch.nn.Parameter]],
) -> Dict[str, Tuple[float, ...]]:
    """Capture optimizer state steps; AdamW return values cannot prove an update."""

    result: Dict[str, Tuple[float, ...]] = {}
    for group in ("shared_encoder", "classifier_head"):
        params = tuple(parameter_scopes.get(group, ()))
        if not params:
            raise CPSFCERuntimeError(f"P1-CP-SFCE {group} optimizer scope is empty")
        result[group] = tuple(_state_step_value(optimizer.state.get(parameter, {})) for parameter in params)
    return result


def cp_sfce_capture_optimizer_steps_for_model(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Dict[str, Tuple[float, ...]]:
    """Recapture the same frozen encoder/head scope after ``scaler.step``."""

    return cp_sfce_capture_optimizer_steps(optimizer, cp_sfce_parameter_scopes(model))


def _optimizer_steps_unchanged(
    before: Mapping[str, Sequence[float]], after: Mapping[str, Sequence[float]]
) -> bool:
    for group in ("shared_encoder", "classifier_head"):
        before_values = tuple(float(value) for value in before.get(group, ()))
        after_values = tuple(float(value) for value in after.get(group, ()))
        if not before_values or len(before_values) != len(after_values):
            return False
        if not all(math.isfinite(value) for value in (*before_values, *after_values)):
            return False
        if any(before_value != after_value for before_value, after_value in zip(before_values, after_values)):
            return False
    return True


def finalize_cp_sfce_amp_overflow_skip(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    overflow: Mapping[str, Any],
) -> Dict[str, Any]:
    """Skip one raw-finite scaled overflow through public GradScaler APIs.

    Base-gradient overflow is already known to GradScaler after ``unscale_``;
    its public ``step`` therefore skips AdamW and ``update`` applies the
    configured backoff.  Auxiliary-VJP overflow is not in ``.grad`` buffers,
    so the batch is deliberately zeroed without a step and the public scaler
    backoff is requested explicitly.  Neither path retries the batch.
    """

    if overflow.get("amp_overflow_detected") is not True or overflow.get(
        "amp_overflow_recoverable"
    ) is not True:
        raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow skip requires raw-finite overflow evidence")
    kind = str(overflow.get("amp_overflow_kind", ""))
    if kind not in {"BASE_SCALED_OVERFLOW_RAW_FINITE", "AUX_SCALED_OVERFLOW_RAW_FINITE"}:
        raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow kind is invalid")
    if kind == "BASE_SCALED_OVERFLOW_RAW_FINITE":
        raw_finite = overflow.get("raw_base_vjp_finite") is True
        raw_bad = tuple(overflow.get("raw_base_nonfinite_parameter_names", ())) + tuple(
            overflow.get("raw_base_missing_scope_parameter_names", ())
        )
        scaled_bad = tuple(overflow.get("scaled_base_nonfinite_parameter_names", ()))
    else:
        raw_finite = overflow.get("raw_aux_vjp_finite") is True
        raw_bad = (
            tuple(overflow.get("raw_aux_nonfinite_parameter_names", ()))
            + tuple(overflow.get("raw_aux_missing_scope_parameter_names", ()))
            + tuple(overflow.get("raw_aux_outside_nonzero_parameter_names", ()))
        )
        scaled_bad = tuple(overflow.get("scaled_aux_nonfinite_parameter_names", ()))
    if not raw_finite or raw_bad or not scaled_bad:
        raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow lacks raw-finite gradient evidence")
    try:
        scale_before = float(overflow["captured_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow lacks a captured scale") from error
    if not math.isfinite(scale_before) or scale_before <= 0.0:
        raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow captured scale is invalid")
    before = overflow.get("optimizer_state_before", {})
    if kind == "BASE_SCALED_OVERFLOW_RAW_FINITE":
        scaler.step(optimizer)
        scaler.update()
    else:
        if not hasattr(scaler, "get_backoff_factor"):
            raise CPSFCERuntimeError("P1-CP-SFCE auxiliary AMP overflow requires GradScaler backoff API")
        backoff = float(scaler.get_backoff_factor())
        scale_after_requested = scale_before * backoff
        if not math.isfinite(backoff) or not (0.0 < backoff < 1.0) or not math.isfinite(
            scale_after_requested
        ) or scale_after_requested <= 0.0:
            raise CPSFCERuntimeError("P1-CP-SFCE auxiliary AMP overflow backoff is invalid")
        optimizer.zero_grad(set_to_none=True)
        scaler.update(new_scale=scale_after_requested)
    after = cp_sfce_capture_optimizer_steps_for_model(model, optimizer)
    scale_after = float(scaler.get_scale())
    if not math.isfinite(scale_after) or not (0.0 < scale_after < scale_before):
        raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow skip did not lower GradScaler scale")
    if not _optimizer_steps_unchanged(before, after):
        raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow skip advanced optimizer state")
    optimizer.zero_grad(set_to_none=True)
    return {
        "amp_overflow_kind": kind,
        "captured_scale": scale_before,
        "scale_after": scale_after,
        "optimizer_state_unchanged": True,
        "optimizer_step_applied": False,
        "scaled_parameter_names": tuple(
            str(value)
            for value in overflow.get(
                "scaled_base_nonfinite_parameter_names",
                overflow.get("scaled_aux_nonfinite_parameter_names", ()),
            )
        ),
        "raw_nonfinite_parameter_names": tuple(
            str(value)
            for value in overflow.get(
                "raw_base_nonfinite_parameter_names",
                overflow.get("raw_aux_nonfinite_parameter_names", ()),
            )
        ),
        "raw_missing_scope_parameter_names": tuple(
            str(value)
            for value in overflow.get(
                "raw_base_missing_scope_parameter_names",
                overflow.get("raw_aux_missing_scope_parameter_names", ()),
            )
        ),
        "raw_outside_nonzero_parameter_names": tuple(
            str(value)
            for value in overflow.get("raw_aux_outside_nonzero_parameter_names", ())
        ),
    }


def _projection_group(
    *,
    group_name: str,
    parameters: Sequence[torch.nn.Parameter],
    base_by_id: Mapping[int, torch.Tensor],
    aux_by_id: Mapping[int, torch.Tensor],
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    base = [_require_finite_gradient(f"{group_name} base", base_by_id.get(id(parameter))) for parameter in parameters]
    aux = [_require_finite_gradient(f"{group_name} auxiliary", aux_by_id.get(id(parameter))) for parameter in parameters]
    base_sq, aux_sq, dot = _norm_dot(base, aux)
    base_norm = math.sqrt(base_sq)
    aux_norm = math.sqrt(aux_sq)
    conflict = bool(base_sq > 0.0 and dot < 0.0)
    coefficient: Optional[float] = None
    projected = []
    if conflict:
        coefficient = float(dot / base_sq)
        if not math.isfinite(coefficient):
            raise CPSFCERuntimeError(f"P1-CP-SFCE {group_name} projection coefficient is non-finite")
        for base_grad, aux_grad in zip(base, aux):
            value = aux_grad - aux_grad.new_tensor(coefficient) * base_grad
            projected.append(_require_finite_gradient(f"{group_name} projected auxiliary", value))
    else:
        projected = [gradient.detach() for gradient in aux]
    _, projected_sq, projected_dot = _norm_dot(base, projected)
    projected_norm = math.sqrt(projected_sq)
    tolerance = _projected_dot_tolerance(base_norm, aux_norm)
    if conflict and projected_dot < -tolerance:
        raise CPSFCERuntimeError(
            f"P1-CP-SFCE {group_name} projected dot violates fixed tolerance"
        )
    merged: Dict[int, torch.Tensor] = {}
    for parameter, base_grad, projected_grad in zip(parameters, base, projected):
        merged_value = base_grad + projected_grad
        merged[id(parameter)] = _require_finite_gradient(f"{group_name} combined", merged_value)
    return merged, {
        "parameter_count": int(len(parameters)),
        "base_norm": float(base_norm),
        "base_norm_sq": float(base_sq),
        "aux_norm": float(aux_norm),
        "aux_norm_sq": float(aux_sq),
        "dot": float(dot),
        "base_zero": bool(base_sq == 0.0),
        "conflict": bool(conflict),
        "projection_coefficient": coefficient,
        "projected_aux_norm": float(projected_norm),
        "projected_dot": float(projected_dot),
        "projected_dot_tolerance": float(tolerance),
        "finite": True,
    }


def cp_sfce_scaled_backward_and_project(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    base_loss: torch.Tensor,
    sfce_loss: torch.Tensor,
    loss_weight: float,
) -> Dict[str, Any]:
    """Run exactly one scaled base backward and one scaled auxiliary VJP.

    ``GradScaler`` only unscales parameter ``.grad`` buffers.  The auxiliary
    VJP is therefore explicitly scaled and divided by its captured scale before
    projection, while the base is unscaled exactly once through the optimizer.
    """

    _require_scalar_finite("common base loss", base_loss)
    _require_scalar_finite("satellite focal loss", sfce_loss)
    if not math.isfinite(float(loss_weight)) or abs(float(loss_weight) - FROZEN_CP_SFCE_LAMBDA) > 1e-12:
        raise CPSFCERuntimeError("P1-CP-SFCE loss weight drift")
    parameter_scopes = cp_sfce_parameter_scopes(model)
    all_trainable = tuple(parameter_scopes["all_trainable"])
    captured_scale = float(scaler.get_scale())
    if not math.isfinite(captured_scale) or captured_scale <= 0.0:
        raise CPSFCERuntimeError("P1-CP-SFCE captured GradScaler scale is invalid")
    scaler.scale(base_loss).backward(retain_graph=True)
    scaler.unscale_(optimizer)
    names = _trainable_parameter_names(model, all_trainable)
    scaled_base_nonfinite_parameter_names = tuple(
        sorted(
            names[id(parameter)]
            for parameter in all_trainable
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad.detach()).all().item())
        )
    )
    if scaled_base_nonfinite_parameter_names:
        raw_audit = _raw_vjp_overflow_audit(
            loss=base_loss,
            all_trainable=all_trainable,
            parameter_scopes=parameter_scopes,
            names=names,
            term_name="base",
            require_outside_zero=False,
        )
        optimizer_state_before = cp_sfce_capture_optimizer_steps(optimizer, parameter_scopes)
        return {
            "projection_applied": False,
            "amp_overflow_detected": True,
            "amp_overflow_kind": "BASE_SCALED_OVERFLOW_RAW_FINITE"
            if bool(raw_audit["raw_vjp_finite"])
            else "BASE_RAW_NONFINITE_OR_DISCONNECTED",
            "amp_overflow_recoverable": bool(raw_audit["raw_vjp_finite"]),
            "captured_scale": float(captured_scale),
            "base_scaled_backward_count": 1,
            "optimizer_unscale_count": 1,
            "scaled_aux_vjp_count": 0,
            "unprojected_sfce_backward_count": 0,
            "scaled_base_nonfinite_parameter_names": scaled_base_nonfinite_parameter_names,
            "raw_base_nonfinite_parameter_names": raw_audit["raw_nonfinite_parameter_names"],
            "raw_base_missing_scope_parameter_names": raw_audit["raw_missing_scope_parameter_names"],
            "raw_base_vjp_finite": bool(raw_audit["raw_vjp_finite"]),
            "optimizer_state_before": optimizer_state_before,
        }
    base_by_id = {
        id(parameter): _require_finite_gradient("common base", parameter.grad)
        for parameter in all_trainable
        if parameter.grad is not None
    }
    scaled_aux = torch.autograd.grad(
        scaler.scale(float(loss_weight) * sfce_loss),
        all_trainable,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    scaled_aux_nonfinite_parameter_names = tuple(
        sorted(
            names[id(parameter)]
            for parameter, gradient in zip(all_trainable, scaled_aux)
            if gradient is not None and not bool(torch.isfinite(gradient.detach()).all().item())
        )
    )
    if scaled_aux_nonfinite_parameter_names:
        raw_audit = _raw_vjp_overflow_audit(
            loss=float(loss_weight) * sfce_loss,
            all_trainable=all_trainable,
            parameter_scopes=parameter_scopes,
            names=names,
            term_name="auxiliary",
            require_outside_zero=True,
        )
        optimizer_state_before = cp_sfce_capture_optimizer_steps(optimizer, parameter_scopes)
        raw_valid = bool(raw_audit["raw_vjp_finite"])
        raw_outside = tuple(raw_audit["raw_outside_nonzero_parameter_names"])
        return {
            "projection_applied": False,
            "amp_overflow_detected": True,
            "amp_overflow_kind": "AUX_SCALED_OVERFLOW_RAW_FINITE"
            if raw_valid
            else (
                "AUX_RAW_OUTSIDE_VIOLATION" if raw_outside else "AUX_RAW_NONFINITE_OR_DISCONNECTED"
            ),
            "amp_overflow_recoverable": raw_valid,
            "captured_scale": float(captured_scale),
            "base_scaled_backward_count": 1,
            "optimizer_unscale_count": 1,
            "scaled_aux_vjp_count": 1,
            "unprojected_sfce_backward_count": 0,
            "scaled_aux_nonfinite_parameter_names": scaled_aux_nonfinite_parameter_names,
            "raw_aux_nonfinite_parameter_names": raw_audit["raw_nonfinite_parameter_names"],
            "raw_aux_missing_scope_parameter_names": raw_audit["raw_missing_scope_parameter_names"],
            "raw_aux_outside_nonzero_parameter_names": raw_outside,
            "raw_aux_vjp_finite": raw_valid,
            "optimizer_state_before": optimizer_state_before,
        }
    aux_by_id: Dict[int, torch.Tensor] = {}
    for parameter, scaled_gradient in zip(all_trainable, scaled_aux):
        if scaled_gradient is None:
            continue
        raw_gradient = scaled_gradient.detach() / captured_scale
        aux_by_id[id(parameter)] = _require_finite_gradient("scaled auxiliary", raw_gradient)
    for parameter in parameter_scopes["outside_scope"]:
        auxiliary = aux_by_id.get(id(parameter))
        if auxiliary is not None and int(torch.count_nonzero(auxiliary).item()) != 0:
            raise CPSFCERuntimeError("P1-CP-SFCE auxiliary gradient reaches outside projection scope")
    merged_by_id: Dict[int, torch.Tensor] = {}
    group_info: Dict[str, Any] = {}
    for group in ("shared_encoder", "classifier_head"):
        group_merged, info = _projection_group(
            group_name=group,
            parameters=parameter_scopes[group],
            base_by_id=base_by_id,
            aux_by_id=aux_by_id,
        )
        merged_by_id.update(group_merged)
        group_info[group] = info
    for parameter in (*parameter_scopes["shared_encoder"], *parameter_scopes["classifier_head"]):
        parameter.grad = merged_by_id[id(parameter)].detach().clone()
    optimizer_state_before = cp_sfce_capture_optimizer_steps(optimizer, parameter_scopes)
    return {
        "projection_applied": True,
        "amp_overflow_detected": False,
        "raw_unscaled": True,
        "diagnostic_only": False,
        "captured_scale": float(captured_scale),
        "base_scaled_backward_count": 1,
        "optimizer_unscale_count": 1,
        "scaled_aux_vjp_count": 1,
        "unprojected_sfce_backward_count": 0,
        "all_trainable_parameter_count": int(len(all_trainable)),
        "outside_scope_parameter_count": int(len(parameter_scopes["outside_scope"])),
        "outside_scope_aux_none_or_zero": True,
        "optimizer_state_before": optimizer_state_before,
        "shared_encoder": group_info["shared_encoder"],
        "classifier_head": group_info["classifier_head"],
    }


def update_cp_sfce_coverage_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    scenario: str,
) -> Dict[str, Any]:
    """Accumulate data-free local-TX×scenario coverage with CP field names."""

    result = dict(receipt)
    if str(scenario) not in FROZEN_CP_SFCE_SCENARIOS:
        raise CPSFCERuntimeError("P1-CP-SFCE observed an invalid satellite scenario")
    expected = tuple(int(value) for value in result.get("expected_tx_class_ids", []))
    if expected != (0, 1, 2, 3):
        raise CPSFCERuntimeError("P1-CP-SFCE receipt lacks local4 source TX binding")
    rows = {str(key): int(value) for key, value in dict(batch_info.get("per_tx_rows", {})).items()}
    losses = {str(key): float(value) for key, value in dict(batch_info.get("per_tx_loss", {})).items()}
    finite = {str(key): bool(value) for key, value in dict(batch_info.get("per_tx_finite", {})).items()}
    nonzero = {
        str(key): bool(value)
        for key, value in dict(batch_info.get("per_tx_nonzero_logit_gradient", {})).items()
    }
    if set(rows) != set(losses) or set(rows) != set(finite) or set(rows) != set(nonzero):
        raise CPSFCERuntimeError("P1-CP-SFCE coverage keys do not match")
    cells = {str(key): dict(value) for key, value in dict(result.get("cp_sfce_cells", {})).items()}
    for key, row_count in rows.items():
        if int(key) not in expected or row_count <= 0:
            raise CPSFCERuntimeError("P1-CP-SFCE observed an invalid local TX coverage cell")
        loss_value = float(losses[key])
        if not math.isfinite(loss_value):
            raise CPSFCERuntimeError("P1-CP-SFCE focal loss is non-finite")
        cell_key = f"tx{int(key)}|{str(scenario)}"
        cell = dict(cells.get(cell_key, {}))
        cell["rows"] = int(cell.get("rows", 0)) + row_count
        cell["loss_batches"] = int(cell.get("loss_batches", 0)) + 1
        cell["loss_sum"] = float(cell.get("loss_sum", 0.0)) + loss_value
        cell["finite_batches"] = int(cell.get("finite_batches", 0)) + int(finite[key])
        cell["nonzero_logit_gradient_batches"] = int(
            cell.get("nonzero_logit_gradient_batches", 0)
        ) + int(nonzero[key])
        cell["nonfinite_batches"] = int(cell.get("nonfinite_batches", 0)) + int(not finite[key])
        cells[cell_key] = cell
    result["cp_sfce_cells"] = cells
    result["cp_sfce_batches"] = int(result.get("cp_sfce_batches", 0)) + 1
    result["cp_sfce_rows"] = int(result.get("cp_sfce_rows", 0)) + int(batch_info.get("rows", 0))
    return result


def update_cp_sfce_projection_receipt(
    receipt: Mapping[str, Any],
    projection: Mapping[str, Any],
    *,
    epoch: int,
    batch_index: int,
    scenario: str,
    batch_info: Mapping[str, Any],
) -> Dict[str, Any]:
    """Seal a finite per-batch projection and the one first-epoch marker."""

    result = dict(receipt)
    required_top = (
        "raw_unscaled",
        "diagnostic_only",
        "captured_scale",
        "base_scaled_backward_count",
        "optimizer_unscale_count",
        "scaled_aux_vjp_count",
        "unprojected_sfce_backward_count",
        "outside_scope_aux_none_or_zero",
    )
    if projection.get("raw_unscaled") is not True or projection.get("diagnostic_only") is not False:
        raise CPSFCERuntimeError("P1-CP-SFCE projection is not raw-unscaled operational evidence")
    if str(scenario) not in FROZEN_CP_SFCE_SCENARIOS:
        raise CPSFCERuntimeError("P1-CP-SFCE applied projection has an invalid satellite scenario")
    if any(key not in projection for key in required_top):
        raise CPSFCERuntimeError("P1-CP-SFCE projection receipt lacks required fields")
    if int(projection["base_scaled_backward_count"]) != 1 or int(projection["optimizer_unscale_count"]) != 1:
        raise CPSFCERuntimeError("P1-CP-SFCE must use one base backward and one unscale")
    if int(projection["scaled_aux_vjp_count"]) != 1 or int(projection["unprojected_sfce_backward_count"]) != 0:
        raise CPSFCERuntimeError("P1-CP-SFCE auxiliary gradient path drifted")
    if projection.get("outside_scope_aux_none_or_zero") is not True:
        raise CPSFCERuntimeError("P1-CP-SFCE auxiliary scope audit failed")
    captured_scale = float(projection["captured_scale"])
    if not math.isfinite(captured_scale) or captured_scale <= 0.0:
        raise CPSFCERuntimeError("P1-CP-SFCE captured scale is invalid")
    conflicts = {str(key): int(value) for key, value in dict(result.get("cp_sfce_conflict_batches", {})).items()}
    base_zero = {str(key): int(value) for key, value in dict(result.get("cp_sfce_base_zero_batches", {})).items()}
    for group in ("shared_encoder", "classifier_head"):
        values = projection.get(group)
        if not isinstance(values, Mapping):
            raise CPSFCERuntimeError("P1-CP-SFCE projection lacks a required scope")
        for key in (
            "parameter_count",
            "base_norm",
            "base_norm_sq",
            "aux_norm",
            "aux_norm_sq",
            "dot",
            "projected_aux_norm",
            "projected_dot",
            "projected_dot_tolerance",
        ):
            value = float(values.get(key, float("nan")))
            if not math.isfinite(value):
                raise CPSFCERuntimeError("P1-CP-SFCE projection has non-finite scope evidence")
        if int(values["parameter_count"]) <= 0 or values.get("finite") is not True:
            raise CPSFCERuntimeError("P1-CP-SFCE projection scope is empty or non-finite")
        if bool(values.get("conflict", False)) and float(values["projected_dot"]) < -float(
            values["projected_dot_tolerance"]
        ):
            raise CPSFCERuntimeError("P1-CP-SFCE projected dot violates tolerance")
        conflicts[group] = int(conflicts.get(group, 0)) + int(bool(values.get("conflict", False)))
        base_zero[group] = int(base_zero.get(group, 0)) + int(bool(values.get("base_zero", False)))
    result["cp_sfce_conflict_batches"] = conflicts
    result["cp_sfce_base_zero_batches"] = base_zero
    result["cp_sfce_projection_batches"] = int(result.get("cp_sfce_projection_batches", 0)) + 1
    result["cp_sfce_outside_aux_zero_or_none_batches"] = int(
        result.get("cp_sfce_outside_aux_zero_or_none_batches", 0)
    ) + 1
    rows = {str(key): int(value) for key, value in dict(batch_info.get("per_tx_rows", {})).items()}
    losses = {str(key): float(value) for key, value in dict(batch_info.get("per_tx_loss", {})).items()}
    finite = {str(key): bool(value) for key, value in dict(batch_info.get("per_tx_finite", {})).items()}
    nonzero = {
        str(key): bool(value)
        for key, value in dict(batch_info.get("per_tx_nonzero_logit_gradient", {})).items()
    }
    if set(rows) != set(losses) or set(rows) != set(finite) or set(rows) != set(nonzero):
        raise CPSFCERuntimeError("P1-CP-SFCE applied projection coverage keys do not match")
    cells = {str(key): dict(value) for key, value in dict(result.get("cp_sfce_cells", {})).items()}
    for key, row_count in rows.items():
        if row_count <= 0:
            raise CPSFCERuntimeError("P1-CP-SFCE applied projection has an invalid local TX row count")
        cell_key = f"tx{int(key)}|{str(scenario)}"
        cell = dict(cells.get(cell_key, {}))
        if not cell:
            raise CPSFCERuntimeError("P1-CP-SFCE applied projection has no observed local TX cell")
        cell["applied_rows"] = int(cell.get("applied_rows", 0)) + row_count
        cell["applied_loss_batches"] = int(cell.get("applied_loss_batches", 0)) + 1
        cell["applied_finite_batches"] = int(cell.get("applied_finite_batches", 0)) + int(finite[key])
        cell["applied_nonzero_logit_gradient_batches"] = int(
            cell.get("applied_nonzero_logit_gradient_batches", 0)
        ) + int(nonzero[key])
        cells[cell_key] = cell
    result["cp_sfce_cells"] = cells
    result["cp_sfce_last_projection"] = dict(projection)
    if not bool(result.get("cp_sfce_first_epoch_marker_completed", False)):
        if int(epoch) != 1:
            raise CPSFCERuntimeError("P1-CP-SFCE first technical marker must occur in epoch 1")
        result["cp_sfce_first_epoch_marker_completed"] = True
        result["cp_sfce_first_epoch_marker"] = {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "raw_unscaled": True,
            "technical_only": True,
            "performance_read": False,
            "captured_scale": captured_scale,
            "shared_encoder_conflict": bool(projection["shared_encoder"].get("conflict", False)),
            "classifier_head_conflict": bool(projection["classifier_head"].get("conflict", False)),
        }
    return result


def update_cp_sfce_amp_overflow_receipt(
    receipt: Mapping[str, Any],
    *,
    overflow: Mapping[str, Any],
    finalized_skip: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a data-free AMP overflow classification without retrying a batch."""

    result = dict(receipt)
    if overflow.get("amp_overflow_detected") is not True:
        raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow receipt requires overflow evidence")
    kind = str(overflow.get("amp_overflow_kind", ""))
    base_recoverable = kind == "BASE_SCALED_OVERFLOW_RAW_FINITE"
    aux_recoverable = kind == "AUX_SCALED_OVERFLOW_RAW_FINITE"
    if base_recoverable or aux_recoverable:
        if overflow.get("amp_overflow_recoverable") is not True or finalized_skip is None:
            raise CPSFCERuntimeError("P1-CP-SFCE raw-finite AMP overflow was not finalized as a skip")
        if str(finalized_skip.get("amp_overflow_kind", "")) != kind:
            raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow kind changed before receipt")
        if base_recoverable:
            raw_finite = overflow.get("raw_base_vjp_finite") is True
            raw_bad = tuple(overflow.get("raw_base_nonfinite_parameter_names", ())) + tuple(
                overflow.get("raw_base_missing_scope_parameter_names", ())
            )
            scaled_bad = tuple(overflow.get("scaled_base_nonfinite_parameter_names", ()))
        else:
            raw_finite = overflow.get("raw_aux_vjp_finite") is True
            raw_bad = (
                tuple(overflow.get("raw_aux_nonfinite_parameter_names", ()))
                + tuple(overflow.get("raw_aux_missing_scope_parameter_names", ()))
                + tuple(overflow.get("raw_aux_outside_nonzero_parameter_names", ()))
            )
            scaled_bad = tuple(overflow.get("scaled_aux_nonfinite_parameter_names", ()))
        if not raw_finite or raw_bad or not scaled_bad:
            raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow receipt lacks raw-finite evidence")
        try:
            scale_before = float(finalized_skip["captured_scale"])
            scale_after = float(finalized_skip["scale_after"])
        except (KeyError, TypeError, ValueError) as error:
            raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow receipt lacks scale evidence") from error
        if not math.isfinite(scale_before) or not math.isfinite(scale_after) or not (
            0.0 < scale_after < scale_before
        ):
            raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow receipt scale did not decrease")
        if finalized_skip.get("optimizer_state_unchanged") is not True or finalized_skip.get(
            "optimizer_step_applied"
        ) is not False:
            raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow receipt optimizer state changed")
        if base_recoverable:
            result["cp_sfce_base_scaled_overflow_raw_finite_batches"] = int(
                result.get("cp_sfce_base_scaled_overflow_raw_finite_batches", 0)
            ) + 1
        else:
            result["cp_sfce_aux_scaled_overflow_raw_finite_batches"] = int(
                result.get("cp_sfce_aux_scaled_overflow_raw_finite_batches", 0)
            ) + 1
        result["cp_sfce_amp_skip_scale_decreased_batches"] = int(
            result.get("cp_sfce_amp_skip_scale_decreased_batches", 0)
        ) + 1
        result["cp_sfce_amp_skip_optimizer_state_unchanged_batches"] = int(
            result.get("cp_sfce_amp_skip_optimizer_state_unchanged_batches", 0)
        ) + 1
        result["cp_sfce_last_amp_overflow"] = {
            "kind": kind,
            "pre_scale": scale_before,
            "post_scale": scale_after,
            "scaled_parameter_names": list(finalized_skip.get("scaled_parameter_names", ())),
            "raw_parameter_names": list(finalized_skip.get("raw_nonfinite_parameter_names", ())),
            "count": int(
                result.get("cp_sfce_base_scaled_overflow_raw_finite_batches", 0)
            )
            + int(result.get("cp_sfce_aux_scaled_overflow_raw_finite_batches", 0)),
        }
        return result
    if finalized_skip is not None:
        raise CPSFCERuntimeError("P1-CP-SFCE fatal raw overflow cannot finalize an optimizer skip")
    if kind == "BASE_RAW_NONFINITE_OR_DISCONNECTED":
        result["cp_sfce_base_raw_nonfinite_or_disconnected_batches"] = int(
            result.get("cp_sfce_base_raw_nonfinite_or_disconnected_batches", 0)
        ) + 1
    elif kind == "AUX_RAW_NONFINITE_OR_DISCONNECTED":
        result["cp_sfce_aux_raw_nonfinite_or_disconnected_batches"] = int(
            result.get("cp_sfce_aux_raw_nonfinite_or_disconnected_batches", 0)
        ) + 1
    elif kind == "AUX_RAW_OUTSIDE_VIOLATION":
        result["cp_sfce_aux_raw_outside_violation_batches"] = int(
            result.get("cp_sfce_aux_raw_outside_violation_batches", 0)
        ) + 1
    else:
        raise CPSFCERuntimeError("P1-CP-SFCE AMP overflow receipt kind is invalid")
    result["cp_sfce_last_amp_overflow"] = {
        "kind": kind,
        "pre_scale": float(overflow.get("captured_scale", float("nan"))),
        "scaled_parameter_names": list(
            overflow.get(
                "scaled_base_nonfinite_parameter_names",
                overflow.get("scaled_aux_nonfinite_parameter_names", ()),
            )
        ),
        "raw_parameter_names": list(
            overflow.get(
                "raw_base_nonfinite_parameter_names",
                overflow.get("raw_aux_nonfinite_parameter_names", ()),
            )
        ),
        "raw_missing_scope_parameter_names": list(
            overflow.get(
                "raw_base_missing_scope_parameter_names",
                overflow.get("raw_aux_missing_scope_parameter_names", ()),
            )
        ),
        "raw_outside_nonzero_parameter_names": list(
            overflow.get("raw_aux_outside_nonzero_parameter_names", ())
        ),
    }
    return result


def update_cp_sfce_optimizer_step_receipt(
    receipt: Mapping[str, Any],
    *,
    before: Mapping[str, Sequence[float]],
    after: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    """Require AdamW state-step increments rather than trusting ``scaler.step``."""

    result = dict(receipt)
    summary: Dict[str, Dict[str, float]] = {}
    for group in ("shared_encoder", "classifier_head"):
        before_values = tuple(float(value) for value in before.get(group, ()))
        after_values = tuple(float(value) for value in after.get(group, ()))
        if not before_values or len(before_values) != len(after_values):
            raise CPSFCERuntimeError("P1-CP-SFCE optimizer state scope mismatch")
        deltas = tuple(after_value - before_value for before_value, after_value in zip(before_values, after_values))
        if not all(math.isfinite(value) for value in (*before_values, *after_values, *deltas)):
            raise CPSFCERuntimeError("P1-CP-SFCE optimizer state step is non-finite")
        if min(deltas) <= 0.0:
            result["cp_sfce_optimizer_state_no_step_batches"] = int(
                result.get("cp_sfce_optimizer_state_no_step_batches", 0)
            ) + 1
            raise CPSFCERuntimeError("P1-CP-SFCE optimizer state step did not advance")
        summary[group] = {
            "parameter_count": float(len(deltas)),
            "before_min": float(min(before_values)),
            "after_min": float(min(after_values)),
            "min_delta": float(min(deltas)),
            "max_delta": float(max(deltas)),
        }
    result["cp_sfce_optimizer_state_step_batches"] = int(
        result.get("cp_sfce_optimizer_state_step_batches", 0)
    ) + 1
    result["cp_sfce_last_optimizer_state_step"] = summary
    return result


def validate_cp_sfce_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless every formal CP-SFCE technical contract is met."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    if not bool(result.get("enabled", False)):
        result["cp_sfce_terminal_contract"] = "CONTROL_ARM_NOT_APPLICABLE"
        result["cp_sfce_terminal_contract_passed"] = True
        return result
    expected = tuple(int(value) for value in result.get("expected_tx_class_ids", []))
    if expected != (0, 1, 2, 3):
        raise CPSFCERuntimeError("P1-CP-SFCE terminal receipt lacks local4 TX class order")
    cells = {str(key): dict(value) for key, value in dict(result.get("cp_sfce_cells", {})).items()}
    failed_cells = []
    for class_id in expected:
        for scenario in FROZEN_CP_SFCE_SCENARIOS:
            key = f"tx{class_id}|{scenario}"
            cell = cells.get(key)
            if cell is None or (
                int(cell.get("applied_rows", 0)) <= 0
                or int(cell.get("applied_loss_batches", 0)) <= 0
                or int(cell.get("applied_finite_batches", 0)) <= 0
                or int(cell.get("applied_nonzero_logit_gradient_batches", 0)) <= 0
            ):
                failed_cells.append(key)
    if failed_cells:
        raise CPSFCERuntimeError(
            "P1-CP-SFCE terminal local4×3 coverage failed: " + ",".join(failed_cells)
        )
    marker = result.get("cp_sfce_first_epoch_marker", {})
    if not bool(result.get("cp_sfce_first_epoch_marker_completed", False)) or int(
        dict(marker).get("epoch", -1)
    ) != 1:
        raise CPSFCERuntimeError("P1-CP-SFCE terminal receipt lacks the epoch-1 technical marker")
    try:
        batch_counts = {
            "cp_sfce_batches": int(result.get("cp_sfce_batches", 0)),
            "cp_sfce_projection_batches": int(result.get("cp_sfce_projection_batches", 0)),
            "cp_sfce_outside_aux_zero_or_none_batches": int(
                result.get("cp_sfce_outside_aux_zero_or_none_batches", 0)
            ),
            "cp_sfce_optimizer_state_step_batches": int(
                result.get("cp_sfce_optimizer_state_step_batches", 0)
            ),
            "cp_sfce_optimizer_state_no_step_batches": int(
                result.get("cp_sfce_optimizer_state_no_step_batches", 0)
            ),
            "cp_sfce_base_scaled_overflow_raw_finite_batches": int(
                result.get("cp_sfce_base_scaled_overflow_raw_finite_batches", 0)
            ),
            "cp_sfce_aux_scaled_overflow_raw_finite_batches": int(
                result.get("cp_sfce_aux_scaled_overflow_raw_finite_batches", 0)
            ),
            "cp_sfce_base_raw_nonfinite_or_disconnected_batches": int(
                result.get("cp_sfce_base_raw_nonfinite_or_disconnected_batches", 0)
            ),
            "cp_sfce_aux_raw_nonfinite_or_disconnected_batches": int(
                result.get("cp_sfce_aux_raw_nonfinite_or_disconnected_batches", 0)
            ),
            "cp_sfce_aux_raw_outside_violation_batches": int(
                result.get("cp_sfce_aux_raw_outside_violation_batches", 0)
            ),
            "cp_sfce_amp_skip_scale_decreased_batches": int(
                result.get("cp_sfce_amp_skip_scale_decreased_batches", 0)
            ),
            "cp_sfce_amp_skip_optimizer_state_unchanged_batches": int(
                result.get("cp_sfce_amp_skip_optimizer_state_unchanged_batches", 0)
            ),
        }
    except (TypeError, ValueError) as error:
        raise CPSFCERuntimeError("P1-CP-SFCE terminal receipt has invalid batchwise counts") from error
    batch_total = batch_counts["cp_sfce_batches"]
    applied = batch_counts["cp_sfce_projection_batches"]
    base_skips = batch_counts["cp_sfce_base_scaled_overflow_raw_finite_batches"]
    aux_skips = batch_counts["cp_sfce_aux_scaled_overflow_raw_finite_batches"]
    skip_total = base_skips + aux_skips
    if batch_total <= 0 or applied <= 0 or batch_total != applied + skip_total:
        raise CPSFCERuntimeError("P1-CP-SFCE terminal batchwise projection/step evidence is incomplete")
    if any(
        batch_counts[key] != applied
        for key in (
            "cp_sfce_outside_aux_zero_or_none_batches",
            "cp_sfce_optimizer_state_step_batches",
        )
    ):
        raise CPSFCERuntimeError("P1-CP-SFCE terminal applied projection/step counts are inconsistent")
    if batch_counts["cp_sfce_optimizer_state_no_step_batches"] != 0 or any(
        batch_counts[key] != 0
        for key in (
            "cp_sfce_base_raw_nonfinite_or_disconnected_batches",
            "cp_sfce_aux_raw_nonfinite_or_disconnected_batches",
            "cp_sfce_aux_raw_outside_violation_batches",
        )
    ):
        raise CPSFCERuntimeError("P1-CP-SFCE terminal optimizer-step contract failed")
    if (
        batch_counts["cp_sfce_amp_skip_scale_decreased_batches"] != skip_total
        or batch_counts["cp_sfce_amp_skip_optimizer_state_unchanged_batches"] != skip_total
    ):
        raise CPSFCERuntimeError("P1-CP-SFCE terminal AMP skip evidence is incomplete")
    conflicts = dict(result.get("cp_sfce_conflict_batches", {}))
    base_zero = dict(result.get("cp_sfce_base_zero_batches", {}))
    for group in ("shared_encoder", "classifier_head"):
        try:
            conflict_count = int(conflicts.get(group, 0))
            base_zero_count = int(base_zero.get(group, 0))
        except (TypeError, ValueError) as error:
            raise CPSFCERuntimeError("P1-CP-SFCE terminal scope count is invalid") from error
        if not (0 <= conflict_count <= applied) or not (0 <= base_zero_count <= applied):
            raise CPSFCERuntimeError("P1-CP-SFCE terminal scope count exceeds observed batches")
        if conflict_count < 1:
            raise CPSFCERuntimeError(f"P1-CP-SFCE terminal {group} conflict coverage is inert")
    result["cp_sfce_terminal_contract"] = (
        "FORMAL_LOCAL4_X_SCENARIO3_APPLIED_OR_RAW_FINITE_AMP_SKIP_BATCHWISE_RAW_SCALED_AUX_VJP_AND_SCOPE_CONFLICTS"
    )
    result["cp_sfce_terminal_contract_passed"] = True
    return result


def _cp_sfce_failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "raw gradient audit" in message:
        return "CP_SFCE_RAW_GRADIENT_AUDIT_FAILURE"
    if "amp overflow" in message:
        return "CP_SFCE_AMP_OVERFLOW_FAILURE"
    if "missing or disconnected" in message or "scope audit" in message:
        return "CP_SFCE_GRADIENT_MISSING"
    if "non-finite" in message:
        return "CP_SFCE_NONFINITE"
    if "state step" in message or "optimizer" in message:
        return "CP_SFCE_STATE_NO_STEP"
    if "binding" in message or "head" in message or "class order" in message:
        return "CP_SFCE_BINDING_FAILURE"
    if "coverage" in message or "scenario" in message or "marker" in message:
        return "CP_SFCE_COVERAGE_FAILURE"
    if "project" in message or "conflict" in message or "tolerance" in message:
        return "CP_SFCE_PROJECTION_FAILURE"
    return "CP_SFCE_RUNTIME_FAILURE"


def write_cp_sfce_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free CP-SFCE failure record."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise CPSFCERuntimeError("P1-CP-SFCE failure receipt requires an existing output directory")
    target = target_dir / "cp_sfce_failure_receipt.json"
    payload = {
        "schema": "cvs.phase1.cp_sfce_failure_receipt.v2",
        "status": "FAIL_CLOSED",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage),
        "error_type": type(error).__name__,
        "error_fingerprint": _cp_sfce_failure_fingerprint(error),
        "cp_sfce_receipt": dict(receipt),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = mkstemp(prefix=".cp_sfce_failure_receipt.", suffix=".tmp", dir=str(target_dir))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
