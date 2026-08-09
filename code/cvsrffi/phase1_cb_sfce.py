"""Frozen class-balanced satellite focal CE (P1-CB-SFCE) for Phase1.

The G arm consumes only the existing single satellite-view transmitter logits
and source-known local transmitter labels.  It neither reads clean features or
logits nor introduces a teacher, alignment target, classifier head or rejector.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


FROZEN_CB_SFCE_LAMBDA = 0.10
FROZEN_CB_SFCE_GAMMA = 1.0
FROZEN_CB_SFCE_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
_EPS = 1e-12


class CBSFCEConfigurationError(ValueError):
    """Raised when a frozen P1-CB-SFCE C/G configuration drifts."""


class CBSFCERuntimeError(RuntimeError):
    """Raised when a CB-SFCE batch cannot prove its frozen contract."""


@dataclass(frozen=True)
class CBSFCEConfig:
    """Immutable settings consumed by the Phase1 training-loop integration."""

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
        raise CBSFCEConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _EPS:
        raise CBSFCEConfigurationError(
            f"Frozen P1-CB-SFCE requires {name}={expected:.12g}, got {actual!r}"
        )


def _require_disabled(args: Any, names: Sequence[str]) -> None:
    active = []
    for name in names:
        value = getattr(args, name, False)
        if isinstance(value, bool):
            is_active = bool(value)
        else:
            try:
                is_active = abs(float(value)) > _EPS
            except (TypeError, ValueError):
                is_active = bool(value)
        if is_active:
            active.append(str(name))
    if active:
        raise CBSFCEConfigurationError(
            "Frozen P1-CB-SFCE forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    raw = str(value or "").strip()
    values = (
        tuple(part.strip().lower().replace("-", "_") for part in raw.split(",") if part.strip())
        if raw
        else tuple()
    )
    if values != FROZEN_CB_SFCE_SCENARIOS:
        raise CBSFCEConfigurationError(
            "Frozen P1-CB-SFCE requires --sat_train_scenarios "
            + ",".join(FROZEN_CB_SFCE_SCENARIOS)
        )
    return values


def validate_cb_sfce_args(args: Any) -> CBSFCEConfig:
    """Validate the immutable P1-CB-SFCE C/G continuation contract."""

    frozen_mode = _bool_arg(args, "phase1_cb_sfce_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_cb_sfce_enabled", False)
    loss_weight = _float_arg(args, "lambda_cb_sfce", 0.0)
    gamma = _float_arg(args, "cb_sfce_gamma", FROZEN_CB_SFCE_GAMMA)
    if not frozen_mode and not enabled:
        return CBSFCEConfig(False, False, 0.0, FROZEN_CB_SFCE_GAMMA)
    if enabled and not frozen_mode:
        raise CBSFCEConfigurationError(
            "--phase1_cb_sfce_enabled requires --phase1_cb_sfce_frozen_mode true"
        )
    _require_close("cb_sfce_gamma", gamma, FROZEN_CB_SFCE_GAMMA)
    _require_close(
        "lambda_cb_sfce",
        loss_weight,
        FROZEN_CB_SFCE_LAMBDA if enabled else 0.0,
    )
    if bool(getattr(args, "from_scratch", True)):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE must train the shared encoder and head")
    if int(getattr(args, "epochs", 0)) != 40:
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE requires exactly --epochs 40")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise CBSFCEConfigurationError(
            "Frozen P1-CB-SFCE forbids --sat_view_schedule overrides"
        )
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE requires non-concatenated single LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE forbids unlabeled continuation")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE forbids RX-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE forbids rejection heads")
    _require_disabled(
        args,
        (
            "phase1_ccpc_leo_frozen_mode",
            "phase1_ccpc_leo_enabled",
            "phase1_ccpc_leo_gradient_audit_only",
            "lambda_ccpc_leo",
            "phase1_pamr_frozen_mode",
            "phase1_pamr_enabled",
            "phase1_pamr_audit_only",
            "lambda_pamr",
            "lambda_domain",
            "lambda_adv",
            "lambda_orth",
            "lambda_cons",
            "lambda_group_ce",
            "lambda_fishr",
            "lambda_u",
            "lambda_ent",
            "lambda_u_domain",
            "lambda_u_adv",
            "lambda_u_sat_cons",
            "lambda_u_direct_metric_accept",
            "lambda_u_quarantine_accept",
            "lambda_zid_receiver_invariance",
            "lambda_zid_day_invariance",
            "lambda_zid_channel_invariance",
            "lambda_u_zid_receiver_invariance",
            "lambda_u_zid_day_invariance",
            "lambda_u_zid_channel_invariance",
            "lambda_tx_proto",
            "lambda_rx_proto",
            "lambda_mask_aux",
            "lambda_tx_supcon_masked",
            "lambda_rx_supcon_masked",
            "lambda_txrx_rect",
            "lambda_proto",
            "lambda_open_world_feat",
            "lambda_zid_compact",
            "lambda_proxy_unknown",
            "lambda_manytx_real_oe",
            "lambda_soft_unknown_mixup",
            "lambda_source_episode",
            "lambda_direct_metric_accept",
            "use_phase2_ground_prototypes",
            "use_feature_masks",
            "use_txrx_geometry_losses",
            "use_proto_memory",
            "os_gradient_surgery",
            "os_budget_controller",
            "os_objective_budget_controller",
            "phase1_v2_hard_gates",
            "manytx_real_oe_enabled",
            "manytx_real_oe_protocol_enabled",
            "use_ema_teacher",
            "teacher_ckpt",
            "lambda_teacher_clean_kl",
            "lambda_teacher_sat_kl",
            "lambda_teacher_zid_mse",
        ),
    )
    return CBSFCEConfig(True, enabled, loss_weight, gamma)


def cb_sfce_config_receipt(config: CBSFCEConfig) -> Dict[str, Any]:
    """Create a data-free immutable receipt section for the C/G arms."""

    return {
        "schema": "cvs.phase1.cb_sfce_receipt.v1",
        "method": "P1_CB_SFCE",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "gamma": float(config.gamma),
        "loss_rule": "PRESENT_TX_EQUAL_MEAN_FOCAL_CE_ON_SINGLE_LEO_TX_LOGITS",
        "satellite_scenarios": list(FROZEN_CB_SFCE_SCENARIOS),
        "satellite_schedule": "GLOBAL_BATCH_ROUND_ROBIN_CLEAR_LOW_RAIN",
        "common_lambda_sat_cons": 0.10,
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
        "warm_start_mode": "NOT_APPLICABLE",
        "baseline_path": "",
        "baseline_sha256": "",
        "checkpoint_epoch": -1,
        "checkpoint_role": "",
        "strict_model_keys": False,
        "missing_model_keys": [],
        "unexpected_model_keys": [],
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "source_train_tx": [],
        "source_known_validation_tx": [],
        "source_proxy_unknown_tx": [],
        "dataset_tx_class_order": [],
        "local_tx_class_order": [],
        "checkpoint_train_tx_class_order": [],
        "local_to_dataset_class_ids": [],
        "local_to_head_class_ids": [],
        "expected_tx_class_ids": [],
        "dataset_class_count": 0,
        "local_data_class_count": 0,
        "checkpoint_head_class_count": 0,
        "live_head_class_count": 0,
        "class_order_binding_sha256": "",
        "proxy_rows": 0,
        "held_rows": 0,
        "cb_sfce_batches": 0,
        "cb_sfce_rows": 0,
        "cb_sfce_cells": {},
        "cb_sfce_gradient_relation_attempted": False,
        "cb_sfce_gradient_relation_completed": False,
        "cb_sfce_gradient_relation": {},
        "cb_sfce_terminal_contract": "PENDING",
        "cb_sfce_terminal_contract_passed": False,
    }


def _normalized_tx_class_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CBSFCEConfigurationError(f"P1-CB-SFCE {name} must be a TX class sequence")
    order = tuple(str(value).strip() for value in values)
    if not order or any(not value for value in order) or len(order) != len(set(order)):
        raise CBSFCEConfigurationError(f"P1-CB-SFCE {name} must be non-empty and unique")
    return order


def _positive_class_count(name: str, value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise CBSFCEConfigurationError(f"P1-CB-SFCE {name} must be an integer") from exc
    if count <= 0:
        raise CBSFCEConfigurationError(f"P1-CB-SFCE {name} must be positive")
    return count


def resolve_cb_sfce_local_head_class_binding(
    *,
    local_class_order: Sequence[Any],
    source_train_tx: Sequence[Any],
    checkpoint_train_tx: Sequence[Any],
    dataset_class_order: Sequence[Any],
    local_data_class_count: Any,
    checkpoint_head_class_count: Any,
    live_head_class_count: Any,
) -> Dict[str, Any]:
    """Bind local source-TX labels to strict warm-start head rows."""

    local = _normalized_tx_class_order("local data class order", local_class_order)
    source = _normalized_tx_class_order("source-train TX receipt", source_train_tx)
    checkpoint = _normalized_tx_class_order("checkpoint train TX receipt", checkpoint_train_tx)
    dataset = _normalized_tx_class_order("dataset TX class order", dataset_class_order)
    local_count = _positive_class_count("local data class count", local_data_class_count)
    checkpoint_count = _positive_class_count("checkpoint classifier head row count", checkpoint_head_class_count)
    live_count = _positive_class_count("live classifier head row count", live_head_class_count)
    if local_count != 4 or len(local) != 4:
        raise CBSFCEConfigurationError(
            "P1-CB-SFCE requires exactly four local source-TX classifier rows"
        )
    if local != source:
        raise CBSFCEConfigurationError(
            "P1-CB-SFCE local data TX class order must equal the source-train receipt"
        )
    if checkpoint != source:
        raise CBSFCEConfigurationError(
            "P1-CB-SFCE checkpoint TX class order must equal the source-train receipt"
        )
    if local_count != len(local) or checkpoint_count != live_count or live_count != local_count:
        raise CBSFCEConfigurationError(
            "P1-CB-SFCE local data, checkpoint head and live head class counts must match"
        )
    missing = sorted(set(local).difference(dataset))
    if missing:
        raise CBSFCEConfigurationError("P1-CB-SFCE local TX labels are absent from dataset order")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(range(local_count)),
        "expected_tx_class_ids": list(range(local_count)),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = hashlib.sha256(
        json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return binding


def remap_cb_sfce_local_labels_to_head_rows(
    local_labels: torch.Tensor,
    local_to_head_class_ids: Sequence[Any],
) -> torch.Tensor:
    """Map contiguous local labels through the frozen identity head receipt."""

    if not torch.is_tensor(local_labels):
        raise CBSFCERuntimeError("P1-CB-SFCE local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if not mapping or min(mapping) < 0 or len(mapping) != len(set(mapping)):
        raise CBSFCERuntimeError("P1-CB-SFCE local-to-head mapping is invalid")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= len(mapping):
        raise CBSFCERuntimeError("P1-CB-SFCE local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_cb_sfce_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    """Resolve the existing classifier head without adding or replacing it."""

    raw_model = getattr(model, "_orig_mod", model)
    try:
        weight = raw_model.id_backbone.cls_head.head.weight
    except AttributeError as exc:
        raise CBSFCERuntimeError(
            "P1-CB-SFCE requires model.id_backbone.cls_head.head.weight"
        ) from exc
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise CBSFCERuntimeError("P1-CB-SFCE classifier head weight must be rank-2 Parameter")
    return weight


def validate_cb_sfce_logit_binding(
    *,
    model: torch.nn.Module,
    tx_logits: torch.Tensor,
    tx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
) -> torch.nn.Parameter:
    """Fail closed on a local4/head/logit binding drift before CB-SFCE runs."""

    if not torch.is_tensor(tx_logits) or tx_logits.ndim != 2:
        raise CBSFCERuntimeError("P1-CB-SFCE requires rank-2 satellite tx_logits")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() != tx_logits.size(0) or labels.numel() == 0:
        raise CBSFCERuntimeError("P1-CB-SFCE labels must match satellite tx_logits rows")
    expected = tuple(int(value) for value in expected_class_ids)
    if expected != (0, 1, 2, 3):
        raise CBSFCERuntimeError("P1-CB-SFCE expected local4 TX class order is invalid")
    weight = resolve_cb_sfce_classifier_weight(model)
    if int(weight.size(0)) != len(expected) or int(tx_logits.size(1)) != len(expected):
        raise CBSFCERuntimeError("P1-CB-SFCE live head and tx_logits rows must equal local class order")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= len(expected):
        raise CBSFCERuntimeError("P1-CB-SFCE labels do not bind to local classifier rows")
    if not bool(torch.isfinite(tx_logits.detach()).all().item()):
        raise CBSFCERuntimeError("P1-CB-SFCE satellite tx_logits contain non-finite values")
    return weight


def cb_sfce_loss(
    satellite_tx_logits: torch.Tensor,
    source_tx_labels: torch.Tensor,
    *,
    gamma: float = FROZEN_CB_SFCE_GAMMA,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return present-TX-equal satellite focal CE and data-free coverage counts."""

    _require_close("cb_sfce_gamma", float(gamma), FROZEN_CB_SFCE_GAMMA)
    if not torch.is_tensor(satellite_tx_logits) or satellite_tx_logits.ndim != 2:
        raise CBSFCERuntimeError("P1-CB-SFCE requires rank-2 satellite tx_logits")
    labels = source_tx_labels.reshape(-1).long()
    if labels.numel() != satellite_tx_logits.size(0) or labels.numel() == 0:
        raise CBSFCERuntimeError("P1-CB-SFCE labels must match non-empty satellite logits")
    if satellite_tx_logits.size(1) < 2:
        raise CBSFCERuntimeError("P1-CB-SFCE requires at least two transmitter classes")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= int(satellite_tx_logits.size(1)):
        raise CBSFCERuntimeError("P1-CB-SFCE labels are outside satellite tx_logits class rows")
    if not bool(satellite_tx_logits.requires_grad):
        raise CBSFCERuntimeError("P1-CB-SFCE requires a live satellite tx_logits gradient path")
    logits = satellite_tx_logits.float()
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise CBSFCERuntimeError("P1-CB-SFCE satellite tx_logits contain non-finite values")
    log_prob = F.log_softmax(logits, dim=1)
    true_log_prob = log_prob.gather(1, labels.unsqueeze(1)).squeeze(1)
    true_prob = true_log_prob.exp()
    per_row = (1.0 - true_prob).pow(float(gamma)) * (-true_log_prob)
    if not bool(torch.isfinite(per_row.detach()).all().item()):
        raise CBSFCERuntimeError("P1-CB-SFCE focal per-row loss is non-finite")
    present = torch.unique(labels, sorted=True)
    per_tx_losses = []
    per_tx_rows: Dict[str, int] = {}
    per_tx_loss: Dict[str, float] = {}
    per_tx_finite: Dict[str, bool] = {}
    per_tx_nonzero_logit_gradient: Dict[str, bool] = {}
    for class_id in present.tolist():
        mask = labels.eq(int(class_id))
        values = per_row[mask]
        mean_loss = values.mean()
        key = str(int(class_id))
        per_tx_losses.append(mean_loss)
        per_tx_rows[key] = int(mask.sum().item())
        per_tx_loss[key] = float(mean_loss.detach().item())
        per_tx_finite[key] = bool(torch.isfinite(values.detach()).all().item())
        # For finite multi-class focal CE, a strictly positive per-row loss is
        # equivalent to an analytically reachable nonzero logit derivative.
        # This records all 12 cells without extra autograd work in formal 40E.
        per_tx_nonzero_logit_gradient[key] = bool((values.detach() > 0.0).any().item())
    loss = torch.stack(per_tx_losses).mean()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise CBSFCERuntimeError("P1-CB-SFCE loss is non-finite")
    return loss, {
        "rows": int(labels.numel()),
        "classes": int(present.numel()),
        "per_tx_rows": per_tx_rows,
        "per_tx_loss": per_tx_loss,
        "per_tx_finite": per_tx_finite,
        "per_tx_nonzero_logit_gradient": per_tx_nonzero_logit_gradient,
        "present_tx_equal_aggregation": True,
        "single_satellite_logits_only": True,
    }


def add_cb_sfce_to_loss(
    base_loss: torch.Tensor,
    cb_sfce: Optional[torch.Tensor],
    config: Optional[CBSFCEConfig],
) -> torch.Tensor:
    """Add the sole frozen G-arm term without changing the C-arm tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if cb_sfce is None:
        raise CBSFCERuntimeError("Enabled P1-CB-SFCE requires a satellite focal loss")
    return base_loss + float(config.loss_weight) * cb_sfce


def cb_sfce_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    """Return the trainable logits-path encoder and existing head audit scopes.

    ``PhysicalAwareClassifier`` exposes DAC/PA prediction-only auxiliary heads
    in the same module tree.  They are intentionally excluded because neither
    ordinary TX CE nor CB-SFCE reaches them; including them would turn a valid
    raw-gradient audit into a false ``None`` failure.
    """

    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise CBSFCERuntimeError("P1-CB-SFCE requires model.id_backbone for gradient audit")
    head_weight = resolve_cb_sfce_classifier_weight(raw_model)
    aux_only_prefixes = (
        "con_proj.",
        "cls_head.imp_merge.",
        "cls_head.dac_head.",
        "cls_head.pa_head.",
    )
    encoder = tuple(
        parameter
        for name, parameter in id_backbone.named_parameters()
        if parameter.requires_grad
        and name != "cls_head.head.weight"
        and not str(name).startswith(aux_only_prefixes)
    )
    if not encoder or not head_weight.requires_grad:
        raise CBSFCERuntimeError("P1-CB-SFCE shared encoder/head audit scope is empty")
    return {"shared_encoder": encoder, "classifier_head": (head_weight,)}


def _gradient_group_relation(
    *,
    group_name: str,
    base_loss: torch.Tensor,
    cb_sfce: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    loss_weight: float,
) -> Dict[str, Optional[float]]:
    params = tuple(parameters)
    if not params:
        raise CBSFCERuntimeError(f"P1-CB-SFCE {group_name} gradient scope is empty")
    base_grads = torch.autograd.grad(
        base_loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    cb_grads = torch.autograd.grad(
        float(loss_weight) * cb_sfce,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    base_sq = 0.0
    cb_sq = 0.0
    dot = 0.0
    for base_grad, cb_grad in zip(base_grads, cb_grads):
        if base_grad is None or cb_grad is None:
            raise CBSFCERuntimeError(f"P1-CB-SFCE {group_name} gradient is missing")
        if not bool(torch.isfinite(base_grad.detach()).all().item()) or not bool(
            torch.isfinite(cb_grad.detach()).all().item()
        ):
            raise CBSFCERuntimeError(f"P1-CB-SFCE {group_name} gradient is non-finite")
        base_value = base_grad.detach().double()
        cb_value = cb_grad.detach().double()
        base_sq += float(torch.sum(base_value * base_value).item())
        cb_sq += float(torch.sum(cb_value * cb_value).item())
        dot += float(torch.sum(base_value * cb_value).item())
    base_norm = math.sqrt(base_sq)
    cb_norm = math.sqrt(cb_sq)
    if not math.isfinite(base_norm) or not math.isfinite(cb_norm) or not math.isfinite(dot):
        raise CBSFCERuntimeError(f"P1-CB-SFCE {group_name} gradient relation is non-finite")
    if base_norm <= 0.0 or cb_norm <= 0.0:
        raise CBSFCERuntimeError(f"P1-CB-SFCE {group_name} gradient norm is zero")
    cosine = float(dot / (base_norm * cb_norm))
    if not math.isfinite(cosine):
        raise CBSFCERuntimeError(f"P1-CB-SFCE {group_name} gradient cosine is non-finite")
    return {
        "parameter_count": float(len(params)),
        "base_norm": float(base_norm),
        "cb_sfce_norm": float(cb_norm),
        "cosine": cosine,
        "norm_ratio": float(cb_norm / (base_norm + _EPS)),
    }


def cb_sfce_shared_gradient_relation(
    base_loss: torch.Tensor,
    cb_sfce: torch.Tensor,
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
    *,
    loss_weight: float,
) -> Dict[str, Any]:
    """Measure one raw, pre-GradScaler base/CB relation for encoder and head."""

    if not torch.is_tensor(base_loss) or base_loss.ndim != 0:
        raise CBSFCERuntimeError("P1-CB-SFCE gradient audit requires scalar common base loss")
    if not torch.is_tensor(cb_sfce) or cb_sfce.ndim != 0:
        raise CBSFCERuntimeError("P1-CB-SFCE gradient audit requires scalar focal loss")
    _require_close("lambda_cb_sfce", float(loss_weight), FROZEN_CB_SFCE_LAMBDA)
    expected_groups = ("shared_encoder", "classifier_head")
    if tuple(parameter_groups.keys()) != expected_groups:
        raise CBSFCERuntimeError("P1-CB-SFCE gradient audit requires encoder and head scopes")
    result = {
        group: _gradient_group_relation(
            group_name=group,
            base_loss=base_loss,
            cb_sfce=cb_sfce,
            parameters=parameter_groups[group],
            loss_weight=loss_weight,
        )
        for group in expected_groups
    }
    result["raw_unscaled"] = True
    result["diagnostic_only"] = True
    return result


def update_cb_sfce_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    scenario: str,
) -> Dict[str, Any]:
    """Accumulate data-free local-TX×scenario focal-loss coverage."""

    result = dict(receipt)
    if str(scenario) not in FROZEN_CB_SFCE_SCENARIOS:
        raise CBSFCERuntimeError("P1-CB-SFCE observed a scenario outside frozen clear/low/rain cycle")
    expected = tuple(int(value) for value in result.get("expected_tx_class_ids", []))
    if expected != (0, 1, 2, 3):
        raise CBSFCERuntimeError("P1-CB-SFCE receipt lacks local4 source TX class binding")
    rows = {str(key): int(value) for key, value in dict(batch_info.get("per_tx_rows", {})).items()}
    losses = {str(key): float(value) for key, value in dict(batch_info.get("per_tx_loss", {})).items()}
    finite = {str(key): bool(value) for key, value in dict(batch_info.get("per_tx_finite", {})).items()}
    nonzero = {
        str(key): bool(value)
        for key, value in dict(batch_info.get("per_tx_nonzero_logit_gradient", {})).items()
    }
    if set(rows) != set(losses) or set(rows) != set(finite) or set(rows) != set(nonzero):
        raise CBSFCERuntimeError("P1-CB-SFCE batch coverage keys must match")
    cells = {str(key): dict(value) for key, value in dict(result.get("cb_sfce_cells", {})).items()}
    for key, row_count in rows.items():
        if int(key) not in expected or row_count <= 0:
            raise CBSFCERuntimeError("P1-CB-SFCE observed an invalid local TX cell")
        loss_value = float(losses[key])
        if not math.isfinite(loss_value):
            raise CBSFCERuntimeError("P1-CB-SFCE batch focal loss is non-finite")
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
    result["cb_sfce_cells"] = cells
    result["cb_sfce_batches"] = int(result.get("cb_sfce_batches", 0)) + 1
    result["cb_sfce_rows"] = int(result.get("cb_sfce_rows", 0)) + int(batch_info.get("rows", 0))
    return result


def update_cb_sfce_gradient_relation_receipt(
    receipt: Mapping[str, Any], relation: Mapping[str, Any]
) -> Dict[str, Any]:
    """Record the sole first-effective-batch raw gradient diagnostic."""

    result = dict(receipt)
    if bool(result.get("cb_sfce_gradient_relation_completed", False)):
        raise CBSFCERuntimeError("P1-CB-SFCE raw gradient relation may run only once")
    if relation.get("raw_unscaled") is not True or relation.get("diagnostic_only") is not True:
        raise CBSFCERuntimeError(
            "P1-CB-SFCE gradient relation receipt requires raw_unscaled diagnostic-only evidence"
        )
    for group in ("shared_encoder", "classifier_head"):
        values = relation.get(group)
        if not isinstance(values, Mapping):
            raise CBSFCERuntimeError("P1-CB-SFCE gradient relation lacks required scope")
        for key in ("parameter_count", "base_norm", "cb_sfce_norm", "norm_ratio"):
            value = float(values.get(key, float("nan")))
            if not math.isfinite(value):
                raise CBSFCERuntimeError("P1-CB-SFCE gradient relation is non-finite")
        if float(values["base_norm"]) <= 0.0 or float(values["cb_sfce_norm"]) <= 0.0:
            raise CBSFCERuntimeError("P1-CB-SFCE gradient relation has zero norm")
        cosine = values.get("cosine")
        if cosine is None or not math.isfinite(float(cosine)):
            raise CBSFCERuntimeError("P1-CB-SFCE gradient relation cosine is missing or non-finite")
    result["cb_sfce_gradient_relation_attempted"] = True
    result["cb_sfce_gradient_relation_completed"] = True
    result["cb_sfce_gradient_relation"] = dict(relation)
    return result


def validate_cb_sfce_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless the G arm sealed every local4×scenario cell."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    if not bool(result.get("enabled", False)):
        result["cb_sfce_terminal_contract"] = "CONTROL_ARM_NOT_APPLICABLE"
        result["cb_sfce_terminal_contract_passed"] = True
        return result
    expected = tuple(int(value) for value in result.get("expected_tx_class_ids", []))
    if expected != (0, 1, 2, 3):
        raise CBSFCERuntimeError("P1-CB-SFCE terminal receipt lacks local4 TX class order")
    cells = {str(key): dict(value) for key, value in dict(result.get("cb_sfce_cells", {})).items()}
    missing = []
    invalid = []
    for class_id in expected:
        for scenario in FROZEN_CB_SFCE_SCENARIOS:
            key = f"tx{class_id}|{scenario}"
            cell = cells.get(key)
            if cell is None:
                missing.append(key)
                continue
            if (
                int(cell.get("rows", 0)) <= 0
                or int(cell.get("loss_batches", 0)) <= 0
                or int(cell.get("finite_batches", 0)) <= 0
                or int(cell.get("nonzero_logit_gradient_batches", 0)) <= 0
                or int(cell.get("nonfinite_batches", 0)) != 0
            ):
                invalid.append(key)
    if missing or invalid:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if invalid:
            details.append("invalid=" + ",".join(invalid))
        raise CBSFCERuntimeError(
            "P1-CB-SFCE terminal local4×3 coverage failed: " + "; ".join(details)
        )
    if not bool(result.get("cb_sfce_gradient_relation_completed", False)):
        raise CBSFCERuntimeError("P1-CB-SFCE terminal receipt lacks first-batch raw gradient audit")
    result["cb_sfce_terminal_contract"] = (
        "FORMAL_LOCAL4_X_SCENARIO3_FINITE_NONZERO_LOGIT_GRADIENT_COVERAGE_AND_FIRST_BATCH_RAW_RELATION"
    )
    result["cb_sfce_terminal_contract_passed"] = True
    return result


def _cb_sfce_failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "gradient is missing" in message:
        return "CB_SFCE_GRADIENT_MISSING"
    if "gradient is non-finite" in message or "non-finite" in message:
        return "CB_SFCE_NONFINITE"
    if "binding" in message or "head" in message or "class order" in message:
        return "CB_SFCE_BINDING_FAILURE"
    if "coverage" in message or "scenario" in message:
        return "CB_SFCE_COVERAGE_FAILURE"
    return "CB_SFCE_RUNTIME_FAILURE"


def write_cb_sfce_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free fail-closed record for CB-SFCE."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise CBSFCERuntimeError("P1-CB-SFCE failure receipt requires an existing output directory")
    target = target_dir / "cb_sfce_failure_receipt.json"
    payload = {
        "schema": "cvs.phase1.cb_sfce_failure_receipt.v1",
        "status": "FAIL_CLOSED",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage),
        "error_type": type(error).__name__,
        "error_fingerprint": _cb_sfce_failure_fingerprint(error),
        "cb_sfce_receipt": dict(receipt),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = mkstemp(prefix=".cb_sfce_failure_receipt.", suffix=".tmp", dir=str(target_dir))
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


def strict_cb_sfce_warm_start(
    model: torch.nn.Module,
    checkpoint_model_state: Mapping[str, torch.Tensor],
    *,
    baseline_path: str,
    baseline_sha256: str,
    checkpoint_epoch: int,
    checkpoint_role: str,
) -> Dict[str, Any]:
    """Load only model weights with exact keys; never restore optimizer/RNG state."""

    path = str(baseline_path or "").strip()
    digest = str(baseline_sha256 or "").strip()
    if not path or not digest:
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE warm-start requires baseline path and SHA256")
    if not isinstance(checkpoint_model_state, Mapping):
        raise CBSFCEConfigurationError("Frozen P1-CB-SFCE checkpoint has no model state mapping")
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise CBSFCEConfigurationError(
            f"Frozen P1-CB-SFCE strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise CBSFCEConfigurationError(
            "Frozen P1-CB-SFCE strict baseline model-key mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    return {
        "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP",
        "baseline_path": path,
        "baseline_sha256": digest,
        "checkpoint_epoch": epoch,
        "checkpoint_role": str(checkpoint_role or "UNSPECIFIED"),
        "strict_model_keys": True,
        "missing_model_keys": [],
        "unexpected_model_keys": [],
        "optimizer_state_restored": False,
        "rng_state_restored": False,
    }
