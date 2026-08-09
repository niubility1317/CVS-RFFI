"""Frozen P1-GD-ProtoNLL primitives for Phase1 source-only training.

The trainable G arm adds one lagged-EMA, entropy-regularized class-by-scenario
satellite focal objective to the existing GeoSat-C loss.  The module accepts
only the L-batch satellite identity feature, existing classifier-head rows and
local transmitter labels.  It deliberately has no U/V/proxy input path.

The post-freeze diagonal-Gaussian routines are pure float64 utilities.  They
are not called by the formal training loop; a later final-only scorer may fit
them from labelled L records after the checkpoint has been sealed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


FROZEN_GD_PROTO_NLL_LAMBDA = 0.10
FROZEN_GD_PROTO_NLL_GAMMA = 1.0
FROZEN_GD_PROTO_NLL_BETA = 0.05
FROZEN_GD_PROTO_NLL_ETA = 1.0
FROZEN_GD_PROTO_NLL_LOGIT_SCALE = 16.0
FROZEN_GD_PROTO_NLL_VARIANCE_FLOOR = 1e-6
FROZEN_GD_PROTO_NLL_SHRINKAGE = 0.10
FROZEN_GD_PROTO_NLL_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_GD_PROTO_NLL_CLASS_IDS = (0, 1, 2, 3)
_EPS = 1e-12


class GDProtoNLLConfigurationError(ValueError):
    """Raised when a frozen P1-GD-ProtoNLL C/G configuration drifts."""


class GDProtoNLLRuntimeError(RuntimeError):
    """Raised when a batch cannot prove the frozen P1-GD-ProtoNLL contract."""


@dataclass(frozen=True)
class GDProtoNLLConfig:
    frozen_mode: bool
    enabled: bool
    loss_weight: float
    gamma: float
    beta: float = FROZEN_GD_PROTO_NLL_BETA
    eta: float = FROZEN_GD_PROTO_NLL_ETA
    logit_scale: float = FROZEN_GD_PROTO_NLL_LOGIT_SCALE


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise GDProtoNLLConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _EPS:
        raise GDProtoNLLConfigurationError(
            f"Frozen P1-GD-ProtoNLL requires {name}={expected:.12g}, got {actual!r}"
        )


def _require_disabled(args: Any, names: Sequence[str]) -> None:
    active = []
    for name in names:
        value = getattr(args, name, False)
        if isinstance(value, bool):
            is_active = value
        else:
            try:
                is_active = abs(float(value)) > _EPS
            except (TypeError, ValueError):
                is_active = bool(value)
        if is_active:
            active.append(str(name))
    if active:
        raise GDProtoNLLConfigurationError(
            "Frozen P1-GD-ProtoNLL forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    raw = str(value or "").strip()
    values = tuple(part.strip().lower().replace("-", "_") for part in raw.split(",") if part.strip())
    if values != FROZEN_GD_PROTO_NLL_SCENARIOS:
        raise GDProtoNLLConfigurationError(
            "Frozen P1-GD-ProtoNLL requires --sat_train_scenarios "
            + ",".join(FROZEN_GD_PROTO_NLL_SCENARIOS)
        )
    return values


def validate_gd_proto_nll_args(args: Any) -> GDProtoNLLConfig:
    """Validate the immutable C/G continuation contract."""

    frozen_mode = _bool_arg(args, "phase1_gd_proto_nll_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_gd_proto_nll_enabled", False)
    loss_weight = _float_arg(args, "lambda_gd_proto_nll", 0.0)
    gamma = _float_arg(args, "gd_proto_nll_gamma", FROZEN_GD_PROTO_NLL_GAMMA)
    if not frozen_mode and not enabled:
        return GDProtoNLLConfig(False, False, 0.0, FROZEN_GD_PROTO_NLL_GAMMA)
    if enabled and not frozen_mode:
        raise GDProtoNLLConfigurationError(
            "--phase1_gd_proto_nll_enabled requires --phase1_gd_proto_nll_frozen_mode true"
        )
    _require_close("gd_proto_nll_gamma", gamma, FROZEN_GD_PROTO_NLL_GAMMA)
    _require_close(
        "lambda_gd_proto_nll",
        loss_weight,
        FROZEN_GD_PROTO_NLL_LAMBDA if enabled else 0.0,
    )
    if bool(getattr(args, "from_scratch", True)):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL must train the shared encoder and head")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL requires --id_feature_key feat_joint")
    if int(getattr(args, "epochs", 0)) != 40:
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL requires exactly --epochs 40")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL remains source-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL forbids --sat_view_schedule overrides")
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL requires one non-concatenated LEO forward")
    if bool(getattr(args, "use_unlabeled", False)):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL forbids unlabeled continuation")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL forbids RX-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL forbids rejection heads")
    _require_disabled(
        args,
        (
            "phase1_ccpc_leo_frozen_mode", "phase1_ccpc_leo_enabled", "phase1_ccpc_leo_gradient_audit_only", "lambda_ccpc_leo",
            "phase1_pamr_frozen_mode", "phase1_pamr_enabled", "phase1_pamr_audit_only", "lambda_pamr",
            "phase1_cb_sfce_frozen_mode", "phase1_cb_sfce_enabled", "lambda_cb_sfce",
            "phase1_cp_sfce_frozen_mode", "phase1_cp_sfce_enabled", "lambda_cp_sfce",
            "lambda_domain", "lambda_adv", "lambda_orth", "lambda_cons", "lambda_group_ce",
            "lambda_fishr", "lambda_u", "lambda_ent", "lambda_u_domain", "lambda_u_adv",
            "lambda_u_sat_cons", "lambda_u_direct_metric_accept", "lambda_u_quarantine_accept",
            "lambda_zid_receiver_invariance", "lambda_zid_day_invariance", "lambda_zid_channel_invariance",
            "lambda_u_zid_receiver_invariance", "lambda_u_zid_day_invariance", "lambda_u_zid_channel_invariance",
            "lambda_tx_proto", "lambda_rx_proto", "lambda_mask_aux", "lambda_tx_supcon_masked",
            "lambda_rx_supcon_masked", "lambda_txrx_rect", "lambda_proto", "lambda_open_world_feat",
            "lambda_zid_compact", "lambda_proxy_unknown", "lambda_manytx_real_oe",
            "lambda_soft_unknown_mixup", "lambda_source_episode", "lambda_direct_metric_accept",
            "use_phase2_ground_prototypes", "use_feature_masks", "use_txrx_geometry_losses", "use_proto_memory",
            "os_gradient_surgery", "os_budget_controller", "os_objective_budget_controller", "phase1_v2_hard_gates",
            "manytx_real_oe_enabled", "manytx_real_oe_protocol_enabled", "use_ema_teacher", "teacher_ckpt",
            "lambda_teacher_clean_kl", "lambda_teacher_sat_kl", "lambda_teacher_zid_mse",
        ),
    )
    return GDProtoNLLConfig(True, enabled, loss_weight, gamma)


def gd_proto_nll_config_receipt(config: GDProtoNLLConfig) -> Dict[str, Any]:
    """Create the data-free C/G contract receipt."""

    return {
        "schema": "cvs.phase1.gd_proto_nll_receipt.v1",
        "method": "P1_GD_PROTO_NLL",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "gamma": float(config.gamma),
        "beta": float(config.beta),
        "eta_dro": float(config.eta),
        "prototype_logit_scale": float(config.logit_scale),
        "q0": [1.0 / 12.0] * 12,
        "barl0": [0.0] * 12,
        "loss_rule": "L_ONLY_LOCAL4_REQUIRED_LAGGED_EMA_ENTROPY_REGULARIZED_CLASS_SCENARIO_DRO",
        "prototype_logit_rule": "EXACT_ZERO_FEATURE_FILTER_THEN_ROW_L2_HEAD_WEIGHT_AND_FEAT_JOINT_RAW_COSINE_SCALE16_NO_EPS",
        "feature_zero_filter_rule": "EXACT_NORM_ZERO_ONLY_VALID_LOCAL4_REQUIRED",
        "gradient_witness_rule": "ANALYTIC_NONZERO_D_FOCAL_GAMMA1_D_LOGITS",
        "satellite_scenarios": list(FROZEN_GD_PROTO_NLL_SCENARIOS),
        "satellite_schedule": "BASELINE_EPOCH_BATCH_MODULO_CLEAR_LOW_RAIN",
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
        "gd_proto_nll_batches": 0,
        "gd_proto_nll_total_rows": 0,
        "gd_proto_nll_valid_rows": 0,
        "gd_proto_nll_zero_rows": 0,
        "gd_proto_nll_all_local4_valid_batches": 0,
        "gd_proto_nll_cells": {},
        "gd_proto_nll_state_update_batches": 0,
        "gd_proto_nll_q_final": [],
        "gd_proto_nll_barl_final": [],
        "gd_proto_nll_gradient_relation_attempted": False,
        "gd_proto_nll_gradient_relation_completed": False,
        "gd_proto_nll_gradient_relation": {},
        "gd_proto_nll_terminal_contract": "PENDING",
        "gd_proto_nll_terminal_contract_passed": False,
    }


def _normalized_tx_class_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise GDProtoNLLConfigurationError(f"P1-GD-ProtoNLL {name} must be a TX class sequence")
    order = tuple(str(value).strip() for value in values)
    if not order or any(not value for value in order) or len(order) != len(set(order)):
        raise GDProtoNLLConfigurationError(f"P1-GD-ProtoNLL {name} must be non-empty and unique")
    return order


def _positive_class_count(name: str, value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise GDProtoNLLConfigurationError(f"P1-GD-ProtoNLL {name} must be an integer") from exc
    if count <= 0:
        raise GDProtoNLLConfigurationError(f"P1-GD-ProtoNLL {name} must be positive")
    return count


def resolve_gd_proto_nll_local_head_class_binding(
    *,
    local_class_order: Sequence[Any],
    source_train_tx: Sequence[Any],
    checkpoint_train_tx: Sequence[Any],
    dataset_class_order: Sequence[Any],
    local_data_class_count: Any,
    checkpoint_head_class_count: Any,
    live_head_class_count: Any,
) -> Dict[str, Any]:
    """Bind the local4 L labels to exact strict-warm-start head rows."""

    local = _normalized_tx_class_order("local data class order", local_class_order)
    source = _normalized_tx_class_order("source-train TX receipt", source_train_tx)
    checkpoint = _normalized_tx_class_order("checkpoint train TX receipt", checkpoint_train_tx)
    dataset = _normalized_tx_class_order("dataset TX class order", dataset_class_order)
    local_count = _positive_class_count("local data class count", local_data_class_count)
    checkpoint_count = _positive_class_count("checkpoint classifier head row count", checkpoint_head_class_count)
    live_count = _positive_class_count("live classifier head row count", live_head_class_count)
    if local_count != 4 or len(local) != 4:
        raise GDProtoNLLConfigurationError("P1-GD-ProtoNLL requires exactly four local source-TX rows")
    if local != source or checkpoint != source:
        raise GDProtoNLLConfigurationError(
            "P1-GD-ProtoNLL local/checkpoint TX order must equal source-train receipt"
        )
    if local_count != len(local) or checkpoint_count != live_count or live_count != local_count:
        raise GDProtoNLLConfigurationError("P1-GD-ProtoNLL local/head class counts must match")
    if sorted(set(local).difference(dataset)):
        raise GDProtoNLLConfigurationError("P1-GD-ProtoNLL local TX labels are absent from dataset order")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(FROZEN_GD_PROTO_NLL_CLASS_IDS),
        "expected_tx_class_ids": list(FROZEN_GD_PROTO_NLL_CLASS_IDS),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = hashlib.sha256(
        json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return binding


def remap_gd_proto_nll_local_labels_to_head_rows(
    local_labels: torch.Tensor, local_to_head_class_ids: Sequence[Any]
) -> torch.Tensor:
    if not torch.is_tensor(local_labels):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if mapping != FROZEN_GD_PROTO_NLL_CLASS_IDS:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL local-to-head mapping must be local4 identity")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_gd_proto_nll_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    raw_model = getattr(model, "_orig_mod", model)
    try:
        weight = raw_model.id_backbone.cls_head.head.weight
    except AttributeError as exc:
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL requires model.id_backbone.cls_head.head.weight"
        ) from exc
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL classifier head weight must be rank-2 Parameter")
    return weight


def validate_gd_proto_nll_feature_binding(
    *,
    model: torch.nn.Module,
    satellite_feature: torch.Tensor,
    tx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
    z_id_key: str,
) -> torch.nn.Parameter:
    """Fail closed before raw-cosine prototype logits are formed."""

    if str(z_id_key) != "feat_joint":
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL requires satellite z_id_key=feat_joint")
    if not torch.is_tensor(satellite_feature) or satellite_feature.ndim != 2:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL requires rank-2 satellite feat_joint")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() != satellite_feature.size(0) or labels.numel() == 0:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL labels must match satellite feature rows")
    if tuple(int(value) for value in expected_class_ids) != FROZEN_GD_PROTO_NLL_CLASS_IDS:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL expected local4 class order is invalid")
    weight = resolve_gd_proto_nll_classifier_weight(model)
    if int(weight.size(0)) != 4 or int(weight.size(1)) != int(satellite_feature.size(1)):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL feat_joint/head dimensions or local4 rows drifted")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL labels do not bind to local classifier rows")
    if not bool(torch.isfinite(satellite_feature.detach()).all().item()):
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL feature_nonfinite: satellite feature contains non-finite values"
        )
    return weight


def make_gd_proto_nll_state(device: torch.device | str) -> Dict[str, torch.Tensor]:
    target = torch.device(device)
    return {
        "q": torch.full((12,), 1.0 / 12.0, dtype=torch.float32, device=target),
        "barl": torch.zeros((12,), dtype=torch.float32, device=target),
    }


def _group_index(class_id: int, scenario: str) -> int:
    try:
        scenario_index = FROZEN_GD_PROTO_NLL_SCENARIOS.index(str(scenario))
    except ValueError as exc:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL scenario is outside frozen clear/low/rain") from exc
    if class_id not in FROZEN_GD_PROTO_NLL_CLASS_IDS:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL class is outside local4")
    return int(class_id) * len(FROZEN_GD_PROTO_NLL_SCENARIOS) + scenario_index


def _validated_state(state: Mapping[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL state must be a mapping")
    q = state.get("q")
    barl = state.get("barl")
    if not torch.is_tensor(q) or not torch.is_tensor(barl) or q.numel() != 12 or barl.numel() != 12:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL state must contain 12 q and barl entries")
    q = q.detach().to(device=device, dtype=torch.float32).reshape(12)
    barl = barl.detach().to(device=device, dtype=torch.float32).reshape(12)
    if not bool(torch.isfinite(q).all().item()) or not bool(torch.isfinite(barl).all().item()):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL state is non-finite")
    if bool((q < 0.0).any().item()) or abs(float(q.sum().item()) - 1.0) > 1e-6:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL q must be a finite simplex")
    return q, barl


def gd_proto_nll_loss(
    satellite_feature: torch.Tensor,
    class_weight: torch.Tensor,
    source_tx_labels: torch.Tensor,
    *,
    scenario: str,
    state: Mapping[str, Any],
    gamma: float = FROZEN_GD_PROTO_NLL_GAMMA,
    logit_scale: float = FROZEN_GD_PROTO_NLL_LOGIT_SCALE,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return the old-q L-only local4 DRO focal term and coverage evidence."""

    _require_close("gd_proto_nll_gamma", float(gamma), FROZEN_GD_PROTO_NLL_GAMMA)
    _require_close("prototype_logit_scale", float(logit_scale), FROZEN_GD_PROTO_NLL_LOGIT_SCALE)
    if not torch.is_tensor(satellite_feature) or satellite_feature.ndim != 2 or not satellite_feature.requires_grad:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL requires a live rank-2 satellite feature gradient path")
    if not torch.is_tensor(class_weight) or class_weight.ndim != 2 or not class_weight.requires_grad:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL requires a live classifier head weight path")
    labels = source_tx_labels.reshape(-1).long()
    if labels.numel() != satellite_feature.size(0) or labels.numel() == 0:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL labels must match non-empty satellite feature rows")
    if tuple(torch.unique(labels, sorted=True).tolist()) != FROZEN_GD_PROTO_NLL_CLASS_IDS:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL every G auxiliary batch must contain all local4 classes")
    if int(class_weight.size(0)) != 4 or int(class_weight.size(1)) != int(satellite_feature.size(1)):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL feature/head dimensions drifted")
    if not bool(torch.isfinite(satellite_feature.detach()).all().item()):
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL feature_nonfinite: satellite feature contains non-finite values"
        )
    if not bool(torch.isfinite(class_weight.detach()).all().item()):
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL head_nonfinite_or_zero: classifier head contains non-finite values"
        )
    feature_float = satellite_feature.float()
    weight_float = class_weight.float()
    feature_norms_live = torch.linalg.vector_norm(feature_float, dim=1)
    head_norms_live = torch.linalg.vector_norm(weight_float, dim=1)
    feature_norms = feature_norms_live.detach()
    head_norms = head_norms_live.detach()
    if not bool(torch.isfinite(feature_norms).all().item()):
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL feature_nonfinite: feature L2 norm is non-finite"
        )
    if not bool(torch.isfinite(head_norms).all().item()) or bool((head_norms <= 0.0).any().item()):
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL head_nonfinite_or_zero: classifier head L2 norm is non-finite or zero"
        )
    valid_mask = feature_norms > 0.0
    valid_labels = labels[valid_mask]
    if tuple(torch.unique(valid_labels, sorted=True).tolist()) != FROZEN_GD_PROTO_NLL_CLASS_IDS:
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL feature_zero_filtered: every local4 class must retain at least one valid feature"
        )
    q, _ = _validated_state(state, satellite_feature.device)
    valid_feature = feature_float[valid_mask]
    feature = valid_feature / feature_norms_live[valid_mask].unsqueeze(1)
    weight = weight_float / head_norms_live.unsqueeze(1)
    cosine_logits = float(logit_scale) * F.linear(feature, weight)
    log_prob = F.log_softmax(cosine_logits, dim=1)
    true_log_prob = log_prob.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
    true_prob = true_log_prob.exp()
    per_row = (1.0 - true_prob).pow(float(gamma)) * (-true_log_prob)
    if not bool(torch.isfinite(per_row.detach()).all().item()):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL focal per-row loss is non-finite")
    probability64 = torch.softmax(cosine_logits.detach().double(), dim=1)
    true_probability64 = probability64.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
    cross_entropy64 = -torch.log(true_probability64)
    focal_factor64 = 1.0 - true_probability64 + cross_entropy64 * true_probability64
    one_hot64 = F.one_hot(valid_labels, num_classes=4).double()
    analytic_logit_gradient64 = focal_factor64.unsqueeze(1) * (probability64 - one_hot64)
    analytic_witness_by_row = (
        torch.isfinite(analytic_logit_gradient64).all(dim=1)
        & (torch.linalg.vector_norm(analytic_logit_gradient64, dim=1) > 0.0)
    )
    if not bool(analytic_witness_by_row.all().item()):
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL analytic logit-gradient witness is non-finite or zero"
        )
    class_means = []
    per_tx_total_rows: Dict[str, int] = {}
    per_tx_valid_rows: Dict[str, int] = {}
    per_tx_zero_rows: Dict[str, int] = {}
    per_tx_valid_loss: Dict[str, float] = {}
    per_tx_finite: Dict[str, bool] = {}
    per_tx_analytic_witness: Dict[str, bool] = {}
    q_by_tx: Dict[str, float] = {}
    for class_id in FROZEN_GD_PROTO_NLL_CLASS_IDS:
        total_class_mask = labels.eq(class_id)
        valid_class_mask = valid_labels.eq(class_id)
        values = per_row[valid_class_mask]
        if values.numel() == 0:
            raise GDProtoNLLRuntimeError(
                "P1-GD-ProtoNLL feature_zero_filtered: local4 class has no valid feature"
            )
        mean_loss = values.mean()
        group_index = _group_index(class_id, scenario)
        class_means.append(q[group_index] * mean_loss)
        key = str(class_id)
        total_rows = int(total_class_mask.sum().item())
        valid_rows = int(values.numel())
        per_tx_total_rows[key] = total_rows
        per_tx_valid_rows[key] = valid_rows
        per_tx_zero_rows[key] = total_rows - valid_rows
        per_tx_valid_loss[key] = float(mean_loss.detach().item())
        per_tx_finite[key] = bool(torch.isfinite(values.detach()).all().item())
        per_tx_analytic_witness[key] = bool(analytic_witness_by_row[valid_class_mask].all().item())
        q_by_tx[key] = float(q[group_index].detach().item())
    loss = float(len(FROZEN_GD_PROTO_NLL_SCENARIOS)) * torch.stack(class_means).sum()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL weighted DRO loss is non-finite")
    total_rows = int(labels.numel())
    valid_rows = int(valid_labels.numel())
    zero_rows = total_rows - valid_rows
    return loss, {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "zero_rows": zero_rows,
        "classes": 4,
        "scenario": str(scenario),
        "per_tx_total_rows": per_tx_total_rows,
        "per_tx_valid_rows": per_tx_valid_rows,
        "per_tx_zero_rows": per_tx_zero_rows,
        "per_tx_valid_loss": per_tx_valid_loss,
        "per_tx_finite": per_tx_finite,
        "per_tx_analytic_nonzero_logit_gradient_witness": per_tx_analytic_witness,
        "q_old_by_tx": q_by_tx,
        "fixed_scene_scale": float(len(FROZEN_GD_PROTO_NLL_SCENARIOS)),
        "all_local4_present_before_filter": True,
        "all_local4_valid_after_filter": True,
        "feature_zero_filtered": bool(zero_rows > 0),
        "uses_old_q": True,
    }


def advance_gd_proto_nll_state(
    state: Mapping[str, Any], batch_info: Mapping[str, Any]
) -> Dict[str, torch.Tensor]:
    """Apply the detached four-active-group EMA, then a full-12 softmax update."""

    scenario = str(batch_info.get("scenario", ""))
    state_q = state.get("q") if isinstance(state, Mapping) else None
    if not torch.is_tensor(state_q):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL state lacks q tensor")
    q, barl = _validated_state(state, state_q.device)
    if (
        batch_info.get("all_local4_present_before_filter") is not True
        or batch_info.get("all_local4_valid_after_filter") is not True
    ):
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL feature_zero_filtered: state update requires valid local4 coverage"
        )
    total_rows = int(batch_info.get("total_rows", -1))
    valid_rows = int(batch_info.get("valid_rows", -1))
    zero_rows = int(batch_info.get("zero_rows", -1))
    valid_rows_by_tx = {
        str(key): int(value)
        for key, value in dict(batch_info.get("per_tx_valid_rows", {})).items()
    }
    if (
        total_rows <= 0
        or valid_rows <= 0
        or zero_rows < 0
        or total_rows != valid_rows + zero_rows
        or set(valid_rows_by_tx) != {"0", "1", "2", "3"}
        or any(value <= 0 for value in valid_rows_by_tx.values())
        or valid_rows != sum(valid_rows_by_tx.values())
    ):
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL feature_zero_filtered: state update row coverage does not close"
        )
    losses = {str(key): float(value) for key, value in dict(batch_info.get("per_tx_valid_loss", {})).items()}
    if set(losses) != {"0", "1", "2", "3"}:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL state update requires all four detached class losses")
    updated = barl.clone()
    for class_id in FROZEN_GD_PROTO_NLL_CLASS_IDS:
        value = losses[str(class_id)]
        if not math.isfinite(value):
            raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL detached class loss is non-finite")
        index = _group_index(class_id, scenario)
        updated[index] = (1.0 - FROZEN_GD_PROTO_NLL_BETA) * updated[index] + FROZEN_GD_PROTO_NLL_BETA * value
    q_next = torch.softmax(float(FROZEN_GD_PROTO_NLL_ETA) * updated, dim=0).detach()
    if not bool(torch.isfinite(q_next).all().item()) or bool((q_next < 0.0).any().item()):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL updated q is non-finite")
    return {"q": q_next, "barl": updated.detach()}


def add_gd_proto_nll_to_loss(
    base_loss: torch.Tensor, gd_proto_nll: Optional[torch.Tensor], config: Optional[GDProtoNLLConfig]
) -> torch.Tensor:
    if config is None or not bool(config.enabled):
        return base_loss
    if gd_proto_nll is None:
        raise GDProtoNLLRuntimeError("Enabled P1-GD-ProtoNLL requires a satellite DRO loss")
    return base_loss + float(config.loss_weight) * gd_proto_nll


def gd_proto_nll_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL requires model.id_backbone for gradient audit")
    head_weight = resolve_gd_proto_nll_classifier_weight(raw_model)
    excluded = ("con_proj.", "cls_head.imp_merge.", "cls_head.dac_head.", "cls_head.pa_head.")
    encoder = tuple(
        parameter for name, parameter in id_backbone.named_parameters()
        if parameter.requires_grad and name != "cls_head.head.weight" and not str(name).startswith(excluded)
    )
    if not encoder or not head_weight.requires_grad:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL shared encoder/head audit scope is empty")
    return {"shared_encoder": encoder, "classifier_head": (head_weight,)}


def _gradient_group_relation(
    *, group_name: str, base_loss: torch.Tensor, aux_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter], loss_weight: float,
) -> Dict[str, float]:
    params = tuple(parameters)
    if not params:
        raise GDProtoNLLRuntimeError(f"P1-GD-ProtoNLL {group_name} audit scope is empty")
    base_grads = torch.autograd.grad(base_loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    aux_grads = torch.autograd.grad(float(loss_weight) * aux_loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    base_sq = aux_sq = dot = 0.0
    for base_grad, aux_grad in zip(base_grads, aux_grads):
        if base_grad is None or aux_grad is None:
            raise GDProtoNLLRuntimeError(f"P1-GD-ProtoNLL {group_name} gradient is missing")
        if not bool(torch.isfinite(base_grad.detach()).all().item()) or not bool(torch.isfinite(aux_grad.detach()).all().item()):
            raise GDProtoNLLRuntimeError(f"P1-GD-ProtoNLL {group_name} gradient is non-finite")
        base_value = base_grad.detach().double()
        aux_value = aux_grad.detach().double()
        base_sq += float(torch.sum(base_value * base_value).item())
        aux_sq += float(torch.sum(aux_value * aux_value).item())
        dot += float(torch.sum(base_value * aux_value).item())
    base_norm = math.sqrt(base_sq)
    aux_norm = math.sqrt(aux_sq)
    if not math.isfinite(base_norm) or not math.isfinite(aux_norm) or not math.isfinite(dot) or base_norm <= 0.0 or aux_norm <= 0.0:
        raise GDProtoNLLRuntimeError(f"P1-GD-ProtoNLL {group_name} gradient relation is invalid")
    cosine = float(dot / (base_norm * aux_norm))
    if not math.isfinite(cosine):
        raise GDProtoNLLRuntimeError(f"P1-GD-ProtoNLL {group_name} gradient cosine is non-finite")
    return {"parameter_count": float(len(params)), "base_norm": base_norm, "gd_proto_nll_norm": aux_norm, "cosine": cosine, "norm_ratio": aux_norm / base_norm}


def gd_proto_nll_shared_gradient_relation(
    base_loss: torch.Tensor, aux_loss: torch.Tensor, parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]], *, loss_weight: float
) -> Dict[str, Any]:
    if not torch.is_tensor(base_loss) or base_loss.ndim != 0 or not torch.is_tensor(aux_loss) or aux_loss.ndim != 0:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL gradient audit requires scalar base and auxiliary loss")
    _require_close("lambda_gd_proto_nll", float(loss_weight), FROZEN_GD_PROTO_NLL_LAMBDA)
    expected = ("shared_encoder", "classifier_head")
    if tuple(parameter_groups.keys()) != expected:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL audit requires encoder and exact head scopes")
    result = {group: _gradient_group_relation(group_name=group, base_loss=base_loss, aux_loss=aux_loss, parameters=parameter_groups[group], loss_weight=loss_weight) for group in expected}
    result["raw_unscaled"] = True
    result["diagnostic_only"] = True
    return result


def update_gd_proto_nll_receipt(receipt: Mapping[str, Any], batch_info: Mapping[str, Any], *, scenario: str) -> Dict[str, Any]:
    result = dict(receipt)
    if str(scenario) not in FROZEN_GD_PROTO_NLL_SCENARIOS or str(batch_info.get("scenario", "")) != str(scenario):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL scenario coverage drifted")
    if tuple(int(value) for value in result.get("expected_tx_class_ids", [])) != FROZEN_GD_PROTO_NLL_CLASS_IDS:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL receipt lacks local4 binding")
    total_rows = {str(key): int(value) for key, value in dict(batch_info.get("per_tx_total_rows", {})).items()}
    valid_rows = {str(key): int(value) for key, value in dict(batch_info.get("per_tx_valid_rows", {})).items()}
    zero_rows = {str(key): int(value) for key, value in dict(batch_info.get("per_tx_zero_rows", {})).items()}
    losses = {str(key): float(value) for key, value in dict(batch_info.get("per_tx_valid_loss", {})).items()}
    finite = {str(key): value for key, value in dict(batch_info.get("per_tx_finite", {})).items()}
    analytic_witness = {
        str(key): value
        for key, value in dict(
            batch_info.get("per_tx_analytic_nonzero_logit_gradient_witness", {})
        ).items()
    }
    expected_keys = {"0", "1", "2", "3"}
    maps = (total_rows, valid_rows, zero_rows, losses, finite, analytic_witness)
    if any(set(values) != expected_keys for values in maps):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL every batch must seal all local4 class cells")
    if (
        batch_info.get("all_local4_present_before_filter") is not True
        or batch_info.get("all_local4_valid_after_filter") is not True
    ):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL batch lacks pre/post-filter local4 coverage")
    batch_total_rows = int(batch_info.get("total_rows", -1))
    batch_valid_rows = int(batch_info.get("valid_rows", -1))
    batch_zero_rows = int(batch_info.get("zero_rows", -1))
    if (
        batch_total_rows <= 0
        or batch_valid_rows <= 0
        or batch_zero_rows < 0
        or batch_total_rows != batch_valid_rows + batch_zero_rows
        or batch_total_rows != sum(total_rows.values())
        or batch_valid_rows != sum(valid_rows.values())
        or batch_zero_rows != sum(zero_rows.values())
    ):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL batch total/valid/zero row counts do not close")
    cells = {str(key): dict(value) for key, value in dict(result.get("gd_proto_nll_cells", {})).items()}
    for key in ("0", "1", "2", "3"):
        if (
            total_rows[key] <= 0
            or valid_rows[key] <= 0
            or zero_rows[key] < 0
            or total_rows[key] != valid_rows[key] + zero_rows[key]
            or not math.isfinite(losses[key])
            or finite[key] is not True
            or analytic_witness[key] is not True
        ):
            raise GDProtoNLLRuntimeError(
                "P1-GD-ProtoNLL cell has invalid total/valid/zero rows, loss, or analytic gradient witness"
            )
        cell_key = f"tx{key}|{scenario}"
        cell = dict(cells.get(cell_key, {}))
        cell["total_rows"] = int(cell.get("total_rows", 0)) + total_rows[key]
        cell["valid_rows"] = int(cell.get("valid_rows", 0)) + valid_rows[key]
        cell["zero_rows"] = int(cell.get("zero_rows", 0)) + zero_rows[key]
        cell["valid_loss_batches"] = int(cell.get("valid_loss_batches", 0)) + 1
        cell["valid_loss_sum"] = float(cell.get("valid_loss_sum", 0.0)) + losses[key]
        cell["finite_batches"] = int(cell.get("finite_batches", 0)) + int(finite[key])
        cell["analytic_nonzero_logit_gradient_witness_batches"] = int(
            cell.get("analytic_nonzero_logit_gradient_witness_batches", 0)
        ) + int(analytic_witness[key])
        cell["nonfinite_batches"] = int(cell.get("nonfinite_batches", 0)) + int(not finite[key])
        cells[cell_key] = cell
    result["gd_proto_nll_cells"] = cells
    result["gd_proto_nll_batches"] = int(result.get("gd_proto_nll_batches", 0)) + 1
    result["gd_proto_nll_total_rows"] = int(result.get("gd_proto_nll_total_rows", 0)) + batch_total_rows
    result["gd_proto_nll_valid_rows"] = int(result.get("gd_proto_nll_valid_rows", 0)) + batch_valid_rows
    result["gd_proto_nll_zero_rows"] = int(result.get("gd_proto_nll_zero_rows", 0)) + batch_zero_rows
    result["gd_proto_nll_all_local4_valid_batches"] = int(
        result.get("gd_proto_nll_all_local4_valid_batches", 0)
    ) + 1
    if int(result["gd_proto_nll_total_rows"]) != int(result["gd_proto_nll_valid_rows"]) + int(
        result["gd_proto_nll_zero_rows"]
    ):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL cumulative total/valid/zero row counts do not close")
    return result


def update_gd_proto_nll_state_receipt(receipt: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(receipt)
    state_q = state.get("q") if isinstance(state, Mapping) else None
    if not torch.is_tensor(state_q):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL state receipt lacks q tensor")
    q, barl = _validated_state(state, state_q.device)
    result["gd_proto_nll_state_update_batches"] = int(result.get("gd_proto_nll_state_update_batches", 0)) + 1
    result["gd_proto_nll_q_final"] = [float(value) for value in q.detach().cpu().tolist()]
    result["gd_proto_nll_barl_final"] = [float(value) for value in barl.detach().cpu().tolist()]
    return result


def update_gd_proto_nll_gradient_relation_receipt(receipt: Mapping[str, Any], relation: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(receipt)
    if bool(result.get("gd_proto_nll_gradient_relation_completed", False)):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL raw gradient relation may run only once")
    if relation.get("raw_unscaled") is not True or relation.get("diagnostic_only") is not True:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL audit must be raw-unscaled diagnostic-only")
    for group in ("shared_encoder", "classifier_head"):
        values = relation.get(group)
        if not isinstance(values, Mapping):
            raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL gradient relation lacks scope")
        for key in ("parameter_count", "base_norm", "gd_proto_nll_norm", "norm_ratio", "cosine"):
            value = float(values.get(key, float("nan")))
            if not math.isfinite(value):
                raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL gradient relation is non-finite")
        if float(values["base_norm"]) <= 0.0 or float(values["gd_proto_nll_norm"]) <= 0.0:
            raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL gradient relation has zero norm")
    result["gd_proto_nll_gradient_relation_attempted"] = True
    result["gd_proto_nll_gradient_relation_completed"] = True
    result["gd_proto_nll_gradient_relation"] = dict(relation)
    return result


def validate_gd_proto_nll_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    if str(result.get("checkpoint_role", "") or "").strip() != "training_final_only":
        raise GDProtoNLLRuntimeError(
            "P1-GD-ProtoNLL terminal receipt requires checkpoint_role=training_final_only"
        )
    if not bool(result.get("enabled", False)):
        result["gd_proto_nll_terminal_contract"] = "CONTROL_ARM_NOT_APPLICABLE"
        result["gd_proto_nll_terminal_contract_passed"] = True
        return result
    cells = {str(key): dict(value) for key, value in dict(result.get("gd_proto_nll_cells", {})).items()}
    missing = []
    invalid = []
    cell_total_rows = 0
    cell_valid_rows = 0
    cell_zero_rows = 0
    cell_loss_batches = 0
    for class_id in FROZEN_GD_PROTO_NLL_CLASS_IDS:
        for scenario in FROZEN_GD_PROTO_NLL_SCENARIOS:
            key = f"tx{class_id}|{scenario}"
            cell = cells.get(key)
            if cell is None:
                missing.append(key)
                continue
            total_rows = int(cell.get("total_rows", -1))
            valid_rows = int(cell.get("valid_rows", -1))
            zero_rows = int(cell.get("zero_rows", -1))
            loss_batches = int(cell.get("valid_loss_batches", 0))
            finite_batches = int(cell.get("finite_batches", 0))
            witness_batches = int(cell.get("analytic_nonzero_logit_gradient_witness_batches", 0))
            loss_sum = float(cell.get("valid_loss_sum", float("nan")))
            if (
                total_rows <= 0
                or valid_rows <= 0
                or zero_rows < 0
                or total_rows != valid_rows + zero_rows
                or loss_batches <= 0
                or finite_batches != loss_batches
                or witness_batches != loss_batches
                or int(cell.get("nonfinite_batches", 0)) != 0
                or not math.isfinite(loss_sum)
            ):
                invalid.append(key)
                continue
            cell_total_rows += total_rows
            cell_valid_rows += valid_rows
            cell_zero_rows += zero_rows
            cell_loss_batches += loss_batches
    if missing or invalid:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL terminal local4x3 coverage failed: " + "; ".join((["missing=" + ",".join(missing)] if missing else []) + (["invalid=" + ",".join(invalid)] if invalid else [])))
    batches = int(result.get("gd_proto_nll_batches", 0))
    total_rows = int(result.get("gd_proto_nll_total_rows", -1))
    valid_rows = int(result.get("gd_proto_nll_valid_rows", -1))
    zero_rows = int(result.get("gd_proto_nll_zero_rows", -1))
    if (
        batches <= 0
        or int(result.get("gd_proto_nll_all_local4_valid_batches", 0)) != batches
        or int(result.get("gd_proto_nll_state_update_batches", 0)) != batches
        or total_rows <= 0
        or valid_rows <= 0
        or zero_rows < 0
        or total_rows != valid_rows + zero_rows
        or total_rows != cell_total_rows
        or valid_rows != cell_valid_rows
        or zero_rows != cell_zero_rows
        or cell_loss_batches != 4 * batches
    ):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL DRO state update count does not close")
    q = result.get("gd_proto_nll_q_final", [])
    barl = result.get("gd_proto_nll_barl_final", [])
    if not isinstance(q, Sequence) or isinstance(q, (str, bytes)) or len(q) != 12 or not isinstance(barl, Sequence) or isinstance(barl, (str, bytes)) or len(barl) != 12:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL terminal state is incomplete")
    q_values = [float(value) for value in q]
    bar_values = [float(value) for value in barl]
    if not all(math.isfinite(value) and value >= 0.0 for value in q_values) or abs(sum(q_values) - 1.0) > 1e-6 or not all(math.isfinite(value) for value in bar_values):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL terminal state is invalid")
    if not bool(result.get("gd_proto_nll_gradient_relation_completed", False)):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL terminal receipt lacks first-batch raw gradient audit")
    result["gd_proto_nll_terminal_contract"] = "FORMAL_L_ONLY_LOCAL4_X_SCENARIO3_EXACT_ZERO_FILTER_LAGGED_EMA_DRO_ANALYTIC_LOGIT_GRADIENT_WITNESS_AND_FIRST_BATCH_RAW_RELATION"
    result["gd_proto_nll_terminal_contract_passed"] = True
    return result


def _failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "feature_nonfinite" in message:
        return "GD_PROTO_NLL_FEATURE_NONFINITE"
    if "head_nonfinite_or_zero" in message:
        return "GD_PROTO_NLL_HEAD_NONFINITE_OR_ZERO"
    if "feature_zero_filtered" in message:
        return "GD_PROTO_NLL_FEATURE_ZERO_FILTERED_INVALID_COVERAGE"
    if "missing" in message or "disconnected" in message:
        return "GD_PROTO_NLL_GRADIENT_MISSING"
    if "non-finite" in message or "nonfinite" in message:
        return "GD_PROTO_NLL_NONFINITE"
    if "local4" in message or "class" in message or "head" in message or "binding" in message:
        return "GD_PROTO_NLL_BINDING_OR_COVERAGE"
    return "GD_PROTO_NLL_RUNTIME_FAILURE"


def write_gd_proto_nll_failure_receipt(
    output_dir: str | Path, *, candidate_id: str, run_id: str, receipt: Mapping[str, Any], error: BaseException, failure_stage: str
) -> Path:
    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL failure receipt requires existing output directory")
    target = target_dir / "gd_proto_nll_failure_receipt.json"
    payload = {
        "schema": "cvs.phase1.gd_proto_nll_failure_receipt.v1",
        "status": "FAIL_CLOSED",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage),
        "error_type": type(error).__name__,
        "error_fingerprint": _failure_fingerprint(error),
        "gd_proto_nll_receipt": dict(receipt),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = mkstemp(prefix=".gd_proto_nll_failure_receipt.", suffix=".tmp", dir=str(target_dir))
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


def strict_gd_proto_nll_warm_start(
    model: torch.nn.Module, checkpoint_model_state: Mapping[str, torch.Tensor], *, baseline_path: str,
    baseline_sha256: str, checkpoint_epoch: int, checkpoint_role: str,
) -> Dict[str, Any]:
    path = str(baseline_path or "").strip()
    digest = str(baseline_sha256 or "").strip()
    if not path or not digest or not isinstance(checkpoint_model_state, Mapping):
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL warm-start requires model state, baseline path and SHA256")
    role = str(checkpoint_role or "").strip()
    if role != "training_final_only":
        raise GDProtoNLLConfigurationError(
            "Frozen P1-GD-ProtoNLL warm-start requires checkpoint_role=training_final_only"
        )
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise GDProtoNLLConfigurationError(f"Frozen P1-GD-ProtoNLL strict baseline model-key mismatch: {path}: {exc}") from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise GDProtoNLLConfigurationError("Frozen P1-GD-ProtoNLL strict baseline model-key mismatch")
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    return {
        "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP",
        "baseline_path": path,
        "baseline_sha256": digest,
        "checkpoint_epoch": epoch,
        "checkpoint_role": role,
        "strict_model_keys": True,
        "missing_model_keys": [],
        "unexpected_model_keys": [],
        "optimizer_state_restored": False,
        "rng_state_restored": False,
    }


def _float64_l2(features: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(features) or features.ndim != 2 or features.size(0) == 0:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry requires non-empty rank-2 features")
    values = features.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all().item()):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry feature is non-finite")
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if bool((norms <= 0.0).any().item()):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry feature has zero L2 norm")
    return values / norms


def fit_gd_proto_nll_geometry(l_features: torch.Tensor, l_labels: torch.Tensor) -> Dict[str, Any]:
    """Fit class-equal, shrinkage diagonal Gaussian geometry from labelled L only."""

    normalized = _float64_l2(l_features)
    labels = l_labels.detach().reshape(-1).long()
    if labels.numel() != normalized.size(0) or tuple(torch.unique(labels, sorted=True).tolist()) != FROZEN_GD_PROTO_NLL_CLASS_IDS:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry requires all and only local4 labelled L classes")
    means = []
    raw_variances = []
    counts = []
    for class_id in FROZEN_GD_PROTO_NLL_CLASS_IDS:
        values = normalized[labels.eq(class_id)]
        count = int(values.size(0))
        if count <= 1:
            raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry requires n_c>ddof=1")
        mean = values.mean(dim=0)
        variance = ((values - mean).square()).sum(dim=0) / float(count - 1)
        if not bool(torch.isfinite(mean).all().item()) or not bool(torch.isfinite(variance).all().item()):
            raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry statistics are non-finite")
        means.append(mean)
        raw_variances.append(variance)
        counts.append(count)
    means_tensor = torch.stack(means, dim=0)
    raw_var_tensor = torch.stack(raw_variances, dim=0)
    pooled = raw_var_tensor.mean(dim=0)
    variances = torch.clamp((1.0 - FROZEN_GD_PROTO_NLL_SHRINKAGE) * raw_var_tensor + FROZEN_GD_PROTO_NLL_SHRINKAGE * pooled.unsqueeze(0), min=FROZEN_GD_PROTO_NLL_VARIANCE_FLOOR)
    if not bool(torch.isfinite(variances).all().item()):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL shrunk variance is non-finite")
    return {
        "schema": "cvs.phase1.gd_proto_nll_geometry.v1",
        "feature_normalization": "float64_l2",
        "ddof": 1,
        "variance_floor": FROZEN_GD_PROTO_NLL_VARIANCE_FLOOR,
        "shrinkage": FROZEN_GD_PROTO_NLL_SHRINKAGE,
        "class_ids": list(FROZEN_GD_PROTO_NLL_CLASS_IDS),
        "n_by_class": counts,
        "means": means_tensor.detach().cpu(),
        "raw_variances": raw_var_tensor.detach().cpu(),
        "pooled_variance": pooled.detach().cpu(),
        "variances": variances.detach().cpu(),
    }


def gd_proto_nll_score(features: torch.Tensor, geometry: Mapping[str, Any]) -> torch.Tensor:
    """Return float64 equal-class negative log mixture density without a threshold."""

    normalized = _float64_l2(features)
    required = ("means", "variances", "class_ids", "ddof", "variance_floor", "shrinkage")
    if any(key not in geometry for key in required):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry is incomplete")
    if tuple(int(value) for value in geometry["class_ids"]) != FROZEN_GD_PROTO_NLL_CLASS_IDS or int(geometry["ddof"]) != 1:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry class/DDOF contract drifted")
    if abs(float(geometry["variance_floor"]) - FROZEN_GD_PROTO_NLL_VARIANCE_FLOOR) > _EPS or abs(float(geometry["shrinkage"]) - FROZEN_GD_PROTO_NLL_SHRINKAGE) > _EPS:
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry variance contract drifted")
    means = torch.as_tensor(geometry["means"], dtype=torch.float64, device=normalized.device)
    variances = torch.as_tensor(geometry["variances"], dtype=torch.float64, device=normalized.device)
    if means.ndim != 2 or variances.shape != means.shape or means.size(0) != 4 or means.size(1) != normalized.size(1):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry shape drifted")
    if not bool(torch.isfinite(means).all().item()) or not bool(torch.isfinite(variances).all().item()) or bool((variances < FROZEN_GD_PROTO_NLL_VARIANCE_FLOOR).any().item()):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL geometry values are invalid")
    difference = normalized.unsqueeze(1) - means.unsqueeze(0)
    nll = 0.5 * torch.sum(difference.square() / variances.unsqueeze(0) + torch.log(2.0 * math.pi * variances).unsqueeze(0), dim=2)
    score = math.log(4.0) - torch.logsumexp(-nll, dim=1)
    if not bool(torch.isfinite(nll).all().item()) or not bool(torch.isfinite(score).all().item()):
        raise GDProtoNLLRuntimeError("P1-GD-ProtoNLL NLL score is non-finite")
    return score
