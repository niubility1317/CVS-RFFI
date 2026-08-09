"""Frozen P1-ICMT continuation contract for Phase1 source-only DG.

P1-ICMT preserves the GeoSat-C clean and single-LEO forwards.  Its G arm
adds only a classwise low-margin tail penalty on the existing raw pre-softmax
TX logits.  It is intentionally not a DRO, quantile, EMA, paired-alignment,
or gradient-projection route.
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


FROZEN_ICMT_LAMBDA = 0.05
FROZEN_ICMT_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_ICMT_CLASS_IDS = (0, 1, 2, 3)
FROZEN_ICMT_VIEW_CLASS_DIVISOR = 8
_TOLERANCE = 1e-12


class ICMTConfigurationError(ValueError):
    """Raised when a frozen P1-ICMT C/G configuration drifts."""


class ICMTRuntimeError(RuntimeError):
    """Raised when a P1-ICMT runtime or receipt contract cannot be proved."""


@dataclass(frozen=True)
class ICMTConfig:
    """Immutable P1-ICMT controls consumed by the common training loop."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise ICMTConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
        raise ICMTConfigurationError(
            f"Frozen P1-ICMT requires {name}={expected:.12g}, got {actual!r}"
        )


def _require_disabled(args: Any, names: Sequence[str]) -> None:
    active = []
    for name in names:
        value = getattr(args, name, False)
        if isinstance(value, bool):
            is_active = bool(value)
        else:
            try:
                is_active = abs(float(value)) > _TOLERANCE
            except (TypeError, ValueError):
                is_active = bool(value)
        if is_active:
            active.append(str(name))
    if active:
        raise ICMTConfigurationError(
            "Frozen P1-ICMT forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    raw = str(value or "").strip()
    scenarios = tuple(
        item.strip().lower().replace("-", "_")
        for item in raw.split(",")
        if item.strip()
    )
    if scenarios != FROZEN_ICMT_SCENARIOS:
        raise ICMTConfigurationError(
            "Frozen P1-ICMT requires --sat_train_scenarios "
            + ",".join(FROZEN_ICMT_SCENARIOS)
        )
    return scenarios


def validate_icmt_args(args: Any) -> ICMTConfig:
    """Validate the frozen C/G common-base contract before data are loaded."""

    frozen_mode = _bool_arg(args, "phase1_icmt_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_icmt_enabled", False)
    loss_weight = _float_arg(args, "lambda_icmt", 0.0)
    if not frozen_mode and not enabled:
        return ICMTConfig(False, False, 0.0)
    if enabled and not frozen_mode:
        raise ICMTConfigurationError(
            "--phase1_icmt_enabled requires --phase1_icmt_frozen_mode true"
        )
    _require_close(
        "lambda_icmt",
        loss_weight,
        FROZEN_ICMT_LAMBDA if enabled else 0.0,
    )
    if bool(getattr(args, "from_scratch", True)):
        raise ICMTConfigurationError("Frozen P1-ICMT requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise ICMTConfigurationError("Frozen P1-ICMT requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise ICMTConfigurationError("Frozen P1-ICMT must train the shared z_id encoder and head")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise ICMTConfigurationError("Frozen P1-ICMT requires --id_feature_key feat_joint")
    if int(getattr(args, "epochs", 0)) != 40 or int(getattr(args, "label_epochs", 0)) != 40:
        raise ICMTConfigurationError("Frozen P1-ICMT requires exactly 40 labeled epochs")
    if int(getattr(args, "pseudo_epochs", 0)) != 0:
        raise ICMTConfigurationError("Frozen P1-ICMT forbids pseudo epochs")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise ICMTConfigurationError("Frozen P1-ICMT requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise ICMTConfigurationError("Frozen P1-ICMT remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise ICMTConfigurationError("Frozen P1-ICMT requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise ICMTConfigurationError("Frozen P1-ICMT requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise ICMTConfigurationError("Frozen P1-ICMT forbids --sat_view_schedule overrides")
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise ICMTConfigurationError("Frozen P1-ICMT requires non-concatenated single-LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise ICMTConfigurationError("Frozen P1-ICMT permits only source_known_train L rows")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise ICMTConfigurationError("Frozen P1-ICMT forbids RX/day-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise ICMTConfigurationError("Frozen P1-ICMT permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise ICMTConfigurationError("Frozen P1-ICMT forbids a reject head")
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
            "phase1_cb_sfce_frozen_mode",
            "phase1_cb_sfce_enabled",
            "lambda_cb_sfce",
            "phase1_gd_proto_nll_frozen_mode",
            "phase1_gd_proto_nll_enabled",
            "lambda_gd_proto_nll",
            "phase1_cp_sfce_frozen_mode",
            "phase1_cp_sfce_enabled",
            "lambda_cp_sfce",
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
    return ICMTConfig(True, enabled, loss_weight)


def icmt_config_receipt(config: ICMTConfig) -> Dict[str, Any]:
    """Create a data-free receipt skeleton for either frozen arm."""

    return {
        "schema": "cvs.phase1.icmt_receipt.v1",
        "method": "P1_ICMT",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "loss_rule": "RAW_PRE_SOFTMAX_LOGIT_CLASSWISE_STRICT_LOW_MARGIN_TAIL_TIGHTENING",
        "loss_divisor": FROZEN_ICMT_VIEW_CLASS_DIVISOR,
        "mean_denominator": "ALL_N_C_ROWS",
        "active_rule": "STRICT_MARGIN_LT_STOPGRAD_CLASS_MEAN_TIE_ZERO",
        "logit_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "z_id_key": "feat_joint",
        "postfreeze_nll_path": "PENDING_ONLY: L_ONLY_TOTALIZED_L2_ON_SAME_feat_joint",
        "satellite_scenarios": list(FROZEN_ICMT_SCENARIOS),
        "satellite_schedule": "GLOBAL_BATCH_INDEX_(EPOCH_PLUS_BATCH_MINUS_2)_MOD_3",
        "common_lambda_sat_cons": 0.10,
        "uses_new_forward": False,
        "uses_resampling": False,
        "uses_rx_labels": False,
        "uses_day_labels": False,
        "uses_domain_labels": False,
        "uses_proxy_rows": False,
        "uses_held_rows": False,
        "uses_unlabeled_rows": False,
        "uses_margin_target": False,
        "uses_temperature": False,
        "uses_epsilon": False,
        "uses_topk_or_quantile": False,
        "uses_ema_or_q": False,
        "uses_gradient_projection": False,
        "uses_cross_view_alignment": False,
        "warm_start_mode": "NOT_APPLICABLE",
        "baseline_path": "",
        "baseline_sha256": "",
        "initial_checkpoint_sha256": "",
        "checkpoint_epoch": -1,
        "checkpoint_role": "",
        "strict_model_keys": False,
        "missing_model_keys": [],
        "unexpected_model_keys": [],
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "optimizer_initial_state_sha256": "",
        "optimizer_initial_state_empty": False,
        "source_train_tx": [],
        "source_known_validation_tx": [],
        "source_proxy_unknown_tx": [],
        "source_partition_sha256": "",
        "source_labeled_indices_sha256": "",
        "source_split_manifest_sha256": "",
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
        "common_batch_sequence_sha256": "",
        "common_batch_sequence_batches": 0,
        "common_batch_sequence_rows": 0,
        "common_scenario_batches": {scenario: 0 for scenario in FROZEN_ICMT_SCENARIOS},
        "icmt_batches": 0,
        "icmt_clean_rows": 0,
        "icmt_leo_rows": 0,
        "icmt_clean_cells": {},
        "icmt_leo_cells": {},
        "icmt_gradient_relation_attempted": False,
        "icmt_gradient_relation_completed": False,
        "icmt_gradient_relation": {},
        "icmt_terminal_contract": "PENDING",
        "icmt_terminal_contract_passed": False,
        "proxy_rows": 0,
        "held_rows": 0,
    }


def _normalized_tx_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ICMTConfigurationError(f"P1-ICMT {name} must be a TX class sequence")
    order = tuple(str(value).strip() for value in values)
    if not order or len(order) != len(set(order)) or any(not value for value in order):
        raise ICMTConfigurationError(f"P1-ICMT {name} must be non-empty and unique")
    return order


def _positive_count(name: str, value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ICMTConfigurationError(f"P1-ICMT {name} must be an integer") from exc
    if count <= 0:
        raise ICMTConfigurationError(f"P1-ICMT {name} must be positive")
    return count


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_icmt_local_head_class_binding(
    *,
    local_class_order: Sequence[Any],
    source_train_tx: Sequence[Any],
    checkpoint_train_tx: Sequence[Any],
    dataset_class_order: Sequence[Any],
    local_data_class_count: Any,
    checkpoint_head_class_count: Any,
    live_head_class_count: Any,
) -> Dict[str, Any]:
    """Bind local L labels to the strict warm-start local4 head rows."""

    local = _normalized_tx_order("local data class order", local_class_order)
    source = _normalized_tx_order("source-train TX receipt", source_train_tx)
    checkpoint = _normalized_tx_order("checkpoint train TX receipt", checkpoint_train_tx)
    dataset = _normalized_tx_order("dataset TX class order", dataset_class_order)
    local_count = _positive_count("local data class count", local_data_class_count)
    checkpoint_count = _positive_count("checkpoint classifier head row count", checkpoint_head_class_count)
    live_count = _positive_count("live classifier head row count", live_head_class_count)
    if local_count != 4 or len(local) != 4:
        raise ICMTConfigurationError("P1-ICMT requires exactly four local source-TX rows")
    if local != source or checkpoint != source:
        raise ICMTConfigurationError(
            "P1-ICMT local/checkpoint TX order must equal the source-train receipt"
        )
    if local_count != len(local) or checkpoint_count != live_count or live_count != local_count:
        raise ICMTConfigurationError("P1-ICMT local/head class counts must match")
    if set(local).difference(dataset):
        raise ICMTConfigurationError("P1-ICMT local TX labels are absent from dataset order")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(FROZEN_ICMT_CLASS_IDS),
        "expected_tx_class_ids": list(FROZEN_ICMT_CLASS_IDS),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = _canonical_sha256(binding)
    return binding


def remap_icmt_local_labels_to_head_rows(
    local_labels: torch.Tensor,
    local_to_head_class_ids: Sequence[Any],
) -> torch.Tensor:
    """Map contiguous local source labels through the sealed identity mapping."""

    if not torch.is_tensor(local_labels):
        raise ICMTRuntimeError("P1-ICMT local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if mapping != FROZEN_ICMT_CLASS_IDS:
        raise ICMTRuntimeError("P1-ICMT local-to-head mapping must be local4 identity")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise ICMTRuntimeError("P1-ICMT local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_icmt_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    """Resolve the existing exact classifier head; no G-only head is created."""

    raw_model = getattr(model, "_orig_mod", model)
    try:
        head = raw_model.id_backbone.cls_head.head
    except AttributeError as exc:
        raise ICMTRuntimeError("P1-ICMT requires model.id_backbone.cls_head.head") from exc
    if not isinstance(head, torch.nn.Module):
        raise ICMTRuntimeError("P1-ICMT exact classifier head is not a module")
    parameters = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
    if not parameters:
        raise ICMTRuntimeError("P1-ICMT exact classifier head has no trainable parameter")
    return head


def resolve_icmt_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    """Resolve the exact head's local4 class weight for binding checks."""

    head = resolve_icmt_classifier_head(model)
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise ICMTRuntimeError("P1-ICMT classifier head weight must be a rank-2 Parameter")
    return weight


def _validate_view_binding(
    *,
    view_name: str,
    output: Mapping[str, Any],
    labels: torch.Tensor,
    head_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if str(output.get("z_id_key", "")) != "feat_joint":
        raise ICMTRuntimeError(f"P1-ICMT {view_name} z_id_key must be feat_joint")
    z_id = output.get("z_id")
    logits = output.get("tx_logits")
    if not torch.is_tensor(z_id) or z_id.ndim != 2:
        raise ICMTRuntimeError(f"P1-ICMT {view_name} z_id must be rank-2 feat_joint")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ICMTRuntimeError(f"P1-ICMT {view_name} tx_logits must be rank-2 raw pre-softmax logits")
    if z_id.size(0) != labels.numel() or logits.size(0) != labels.numel():
        raise ICMTRuntimeError(f"P1-ICMT {view_name} rows must align with source L labels")
    if int(head_weight.size(0)) != 4 or int(logits.size(1)) != 4:
        raise ICMTRuntimeError(f"P1-ICMT {view_name} head/logit class rows must be local4")
    if int(head_weight.size(1)) != int(z_id.size(1)):
        raise ICMTRuntimeError(f"P1-ICMT {view_name} feat_joint/head dimension binding drifted")
    if not bool(z_id.requires_grad) or not bool(logits.requires_grad):
        raise ICMTRuntimeError(f"P1-ICMT {view_name} requires a live z_id/head gradient path")
    if not bool(torch.isfinite(z_id.detach()).all().item()):
        raise ICMTRuntimeError(f"P1-ICMT {view_name} z_id contains non-finite values")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise ICMTRuntimeError(f"P1-ICMT {view_name} raw logits contain non-finite values")
    return z_id, logits


def validate_icmt_binding(
    *,
    model: torch.nn.Module,
    out_clean: Mapping[str, Any],
    out_leo: Mapping[str, Any],
    tx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
) -> torch.nn.Parameter:
    """Fail closed unless both standard forwards use one ``feat_joint`` head path."""

    if not isinstance(out_clean, Mapping) or not isinstance(out_leo, Mapping):
        raise ICMTRuntimeError("P1-ICMT requires clean and LEO mapping outputs")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() == 0:
        raise ICMTRuntimeError("P1-ICMT requires a non-empty source L batch")
    if tuple(int(value) for value in expected_class_ids) != FROZEN_ICMT_CLASS_IDS:
        raise ICMTRuntimeError("P1-ICMT expected local4 class order is invalid")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise ICMTRuntimeError("P1-ICMT source labels do not bind to local4 head rows")
    head_weight = resolve_icmt_classifier_weight(model)
    if not bool(torch.isfinite(head_weight.detach()).all().item()):
        raise ICMTRuntimeError("P1-ICMT exact classifier head is non-finite")
    _validate_view_binding(
        view_name="clean", output=out_clean, labels=labels, head_weight=head_weight
    )
    _validate_view_binding(
        view_name="leo", output=out_leo, labels=labels, head_weight=head_weight
    )
    return head_weight


def _validate_aux_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    view_name: str,
) -> None:
    if not torch.is_tensor(logits) or logits.ndim != 2 or int(logits.size(1)) != 4:
        raise ICMTRuntimeError(f"P1-ICMT {view_name} requires local4 rank-2 logits")
    if not bool(logits.requires_grad):
        raise ICMTRuntimeError(f"P1-ICMT {view_name} logits must retain a live gradient path")
    if logits.size(0) != labels.numel() or labels.numel() == 0:
        raise ICMTRuntimeError(f"P1-ICMT {view_name} logits and labels do not align")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise ICMTRuntimeError(f"P1-ICMT {view_name} logits are non-finite")


def _view_icmt_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    view_name: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    _validate_aux_logits(logits, labels, view_name=view_name)
    if tuple(torch.unique(labels, sorted=True).tolist()) != FROZEN_ICMT_CLASS_IDS:
        raise ICMTRuntimeError(
            f"P1-ICMT {view_name} requires all local4 classes in every G auxiliary batch"
        )
    row_ids = torch.arange(labels.numel(), device=labels.device)
    true_logits = logits[row_ids, labels]
    other_mask = F.one_hot(labels, num_classes=4).to(dtype=torch.bool)
    other_logits = logits.masked_fill(other_mask, float("-inf"))
    margins = true_logits - torch.logsumexp(other_logits, dim=1)
    if not bool(torch.isfinite(margins.detach()).all().item()):
        raise ICMTRuntimeError(f"P1-ICMT {view_name} margins are non-finite")
    per_class = []
    rows: Dict[str, int] = {}
    active_rows: Dict[str, int] = {}
    finite_rows: Dict[str, int] = {}
    class_losses: Dict[str, float] = {}
    for class_id in FROZEN_ICMT_CLASS_IDS:
        class_mask = labels.eq(class_id)
        class_margin = margins[class_mask]
        count = int(class_margin.numel())
        if count < 2:
            raise ICMTRuntimeError(
                f"P1-ICMT {view_name} every local4 class requires n_c>=2 before backward"
            )
        mean_margin = class_margin.mean()
        detached_mean = mean_margin.detach()
        strict_low = class_margin < detached_mean
        # The stop-gradient class mean fixes the threshold.  ReLU makes every
        # equality tie contribute exactly zero without a target or epsilon.
        per_row = torch.relu(detached_mean - class_margin).square()
        class_loss = per_row.mean()
        if not bool(torch.isfinite(class_loss.detach()).item()):
            raise ICMTRuntimeError(f"P1-ICMT {view_name} class loss is non-finite")
        key = str(class_id)
        rows[key] = count
        active_rows[key] = int(strict_low.sum().item())
        finite_rows[key] = int(torch.isfinite(class_margin.detach()).sum().item())
        class_losses[key] = float(class_loss.detach().item())
        if active_rows[key] > rows[key] or finite_rows[key] != rows[key]:
            raise ICMTRuntimeError(f"P1-ICMT {view_name} rows/active/finite contract drifted")
        per_class.append(class_loss)
    return torch.stack(per_class).sum(), {
        "rows": int(labels.numel()),
        "classes": 4,
        "per_tx_rows": rows,
        "per_tx_active_rows": active_rows,
        "per_tx_finite_rows": finite_rows,
        "per_tx_loss": class_losses,
        "all_local4_n_ge_2": True,
        "strict_tie_zero": True,
        "all_n_c_mean_denominator": True,
    }


def icmt_loss(
    clean_tx_logits: torch.Tensor,
    leo_tx_logits: torch.Tensor,
    source_tx_labels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute the exact fixed ``1/8`` clean+LEO classwise ICMT objective."""

    labels = source_tx_labels.reshape(-1).long()
    clean_sum, clean_info = _view_icmt_loss(
        clean_tx_logits, labels, view_name="clean"
    )
    leo_sum, leo_info = _view_icmt_loss(leo_tx_logits, labels, view_name="leo")
    loss = (clean_sum + leo_sum) / float(FROZEN_ICMT_VIEW_CLASS_DIVISOR)
    if not bool(torch.isfinite(loss.detach()).item()):
        raise ICMTRuntimeError("P1-ICMT loss is non-finite")
    return loss, {
        "rows": int(labels.numel()),
        "classes": 4,
        "views": {"clean": clean_info, "leo": leo_info},
        "loss_divisor": FROZEN_ICMT_VIEW_CLASS_DIVISOR,
        "fixed_lambda": FROZEN_ICMT_LAMBDA,
        "strict_tie_zero": True,
        "all_n_c_mean_denominator": True,
        "no_active_renormalization": True,
    }


def add_icmt_to_loss(
    base_loss: torch.Tensor,
    icmt: Optional[torch.Tensor],
    config: Optional[ICMTConfig],
) -> torch.Tensor:
    """Add the sole G-arm term; C receives the exact common base tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if icmt is None:
        raise ICMTRuntimeError("Enabled P1-ICMT requires its auxiliary loss")
    return base_loss + float(config.loss_weight) * icmt


def icmt_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    """Return only the common ``feat_joint`` encoder and exact head scopes."""

    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise ICMTRuntimeError("P1-ICMT requires model.id_backbone for VJP audit")
    head = resolve_icmt_classifier_head(raw_model)
    head_parameters = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
    excluded = (
        "cls_head.head.",
        "con_proj.",
        "cls_head.imp_merge.",
        "cls_head.dac_head.",
        "cls_head.pa_head.",
    )
    encoder = tuple(
        parameter
        for name, parameter in id_backbone.named_parameters()
        if parameter.requires_grad and not str(name).startswith(excluded)
    )
    if not encoder or not head_parameters:
        raise ICMTRuntimeError("P1-ICMT shared z_id encoder or exact head scope is empty")
    return {"shared_encoder": encoder, "classifier_head": head_parameters}


def _gradient_group_relation(
    *,
    group_name: str,
    base_loss: torch.Tensor,
    icmt: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    loss_weight: float,
) -> Dict[str, float]:
    params = tuple(parameters)
    if not params:
        raise ICMTRuntimeError(f"P1-ICMT {group_name} VJP scope is empty")
    base_grads = torch.autograd.grad(
        base_loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    icmt_grads = torch.autograd.grad(
        float(loss_weight) * icmt,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    base_sq = 0.0
    icmt_sq = 0.0
    dot = 0.0
    for base_grad, icmt_grad in zip(base_grads, icmt_grads):
        if base_grad is None or icmt_grad is None:
            raise ICMTRuntimeError(f"P1-ICMT {group_name} VJP is None or detached")
        if not bool(torch.isfinite(base_grad.detach()).all().item()) or not bool(
            torch.isfinite(icmt_grad.detach()).all().item()
        ):
            raise ICMTRuntimeError(f"P1-ICMT {group_name} VJP is non-finite")
        base_value = base_grad.detach().double()
        icmt_value = icmt_grad.detach().double()
        base_sq += float(torch.sum(base_value * base_value).item())
        icmt_sq += float(torch.sum(icmt_value * icmt_value).item())
        dot += float(torch.sum(base_value * icmt_value).item())
    base_norm = math.sqrt(base_sq)
    icmt_norm = math.sqrt(icmt_sq)
    if (
        not math.isfinite(base_norm)
        or not math.isfinite(icmt_norm)
        or not math.isfinite(dot)
        or base_norm <= 0.0
        or icmt_norm <= 0.0
    ):
        raise ICMTRuntimeError(f"P1-ICMT {group_name} VJP norm is zero or non-finite")
    cosine = float(dot / (base_norm * icmt_norm))
    if not math.isfinite(cosine):
        raise ICMTRuntimeError(f"P1-ICMT {group_name} VJP cosine is non-finite")
    return {
        "parameter_count": float(len(params)),
        "base_norm": float(base_norm),
        "icmt_norm": float(icmt_norm),
        "cosine": cosine,
        "norm_ratio": float(icmt_norm / base_norm),
    }


def icmt_shared_gradient_relation(
    base_loss: torch.Tensor,
    icmt: torch.Tensor,
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
    *,
    loss_weight: float,
) -> Dict[str, Any]:
    """Measure one raw, unscaled VJP relation without changing an update."""

    if not torch.is_tensor(base_loss) or base_loss.ndim != 0:
        raise ICMTRuntimeError("P1-ICMT VJP audit requires scalar common base loss")
    if not torch.is_tensor(icmt) or icmt.ndim != 0:
        raise ICMTRuntimeError("P1-ICMT VJP audit requires scalar ICMT loss")
    _require_close("lambda_icmt", float(loss_weight), FROZEN_ICMT_LAMBDA)
    expected_groups = ("shared_encoder", "classifier_head")
    if tuple(parameter_groups.keys()) != expected_groups:
        raise ICMTRuntimeError("P1-ICMT VJP audit requires shared encoder and exact head scopes")
    result = {
        name: _gradient_group_relation(
            group_name=name,
            base_loss=base_loss,
            icmt=icmt,
            parameters=parameter_groups[name],
            loss_weight=loss_weight,
        )
        for name in expected_groups
    }
    result["raw_unscaled"] = True
    result["diagnostic_only"] = True
    return result


def update_icmt_gradient_relation_receipt(
    receipt: Mapping[str, Any], relation: Mapping[str, Any]
) -> Dict[str, Any]:
    """Seal the required first-valid-batch VJP without influencing training."""

    result = dict(receipt)
    if bool(result.get("icmt_gradient_relation_completed", False)):
        raise ICMTRuntimeError("P1-ICMT VJP audit may run only once")
    if relation.get("raw_unscaled") is not True or relation.get("diagnostic_only") is not True:
        raise ICMTRuntimeError("P1-ICMT VJP audit must be raw-unscaled diagnostic-only")
    for group in ("shared_encoder", "classifier_head"):
        values = relation.get(group)
        if not isinstance(values, Mapping):
            raise ICMTRuntimeError("P1-ICMT VJP audit lacks a required scope")
        for key in ("parameter_count", "base_norm", "icmt_norm", "norm_ratio", "cosine"):
            value = float(values.get(key, float("nan")))
            if not math.isfinite(value):
                raise ICMTRuntimeError("P1-ICMT VJP audit is non-finite")
        if float(values["parameter_count"]) <= 0.0 or float(values["base_norm"]) <= 0.0 or float(values["icmt_norm"]) <= 0.0:
            raise ICMTRuntimeError("P1-ICMT VJP audit is head-only, detached, or zero")
    result["icmt_gradient_relation_attempted"] = True
    result["icmt_gradient_relation_completed"] = True
    result["icmt_gradient_relation"] = dict(relation)
    return result


def _view_maps(view_info: Mapping[str, Any], view_name: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    rows = {str(key): int(value) for key, value in dict(view_info.get("per_tx_rows", {})).items()}
    active = {
        str(key): int(value)
        for key, value in dict(view_info.get("per_tx_active_rows", {})).items()
    }
    finite = {
        str(key): int(value)
        for key, value in dict(view_info.get("per_tx_finite_rows", {})).items()
    }
    expected = {str(value) for value in FROZEN_ICMT_CLASS_IDS}
    if set(rows) != expected or set(active) != expected or set(finite) != expected:
        raise ICMTRuntimeError(f"P1-ICMT {view_name} receipt lacks all local4 cells")
    for key in expected:
        if rows[key] < 2 or active[key] < 0 or active[key] > rows[key] or finite[key] != rows[key]:
            raise ICMTRuntimeError(f"P1-ICMT {view_name} rows/active/finite contract drifted")
    return rows, active, finite


def _accumulate_cell(
    cells: Dict[str, Dict[str, int]],
    *,
    key: str,
    rows: int,
    active: int,
    finite: int,
) -> None:
    cell = dict(cells.get(key, {}))
    cell["rows"] = int(cell.get("rows", 0)) + int(rows)
    cell["active_rows"] = int(cell.get("active_rows", 0)) + int(active)
    cell["finite_rows"] = int(cell.get("finite_rows", 0)) + int(finite)
    cell["batches"] = int(cell.get("batches", 0)) + 1
    cell["finite_batches"] = int(cell.get("finite_batches", 0)) + 1
    cell["nonfinite_rows"] = int(cell.get("nonfinite_rows", 0))
    if int(cell["active_rows"]) > int(cell["rows"]) or int(cell["finite_rows"]) != int(cell["rows"]):
        raise ICMTRuntimeError("P1-ICMT cumulative rows/active/finite receipt does not close")
    cells[key] = cell


def update_icmt_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    scenario: str,
) -> Dict[str, Any]:
    """Accumulate the required clean×4 and LEO×4×3 G-arm receipt cells."""

    result = dict(receipt)
    if str(scenario) not in FROZEN_ICMT_SCENARIOS:
        raise ICMTRuntimeError("P1-ICMT scenario is outside frozen clear/low/rain cycle")
    if tuple(int(value) for value in result.get("expected_tx_class_ids", [])) != FROZEN_ICMT_CLASS_IDS:
        raise ICMTRuntimeError("P1-ICMT receipt lacks local4 class binding")
    if int(batch_info.get("loss_divisor", -1)) != FROZEN_ICMT_VIEW_CLASS_DIVISOR:
        raise ICMTRuntimeError("P1-ICMT receipt loss divisor drifted")
    if batch_info.get("strict_tie_zero") is not True or batch_info.get("all_n_c_mean_denominator") is not True:
        raise ICMTRuntimeError("P1-ICMT receipt strict tie or all-n_c contract drifted")
    views = batch_info.get("views")
    if not isinstance(views, Mapping) or set(views) != {"clean", "leo"}:
        raise ICMTRuntimeError("P1-ICMT receipt requires clean and LEO view evidence")
    clean_rows, clean_active, clean_finite = _view_maps(views["clean"], "clean")
    leo_rows, leo_active, leo_finite = _view_maps(views["leo"], "leo")
    clean_cells = {
        str(key): dict(value)
        for key, value in dict(result.get("icmt_clean_cells", {})).items()
    }
    leo_cells = {
        str(key): dict(value)
        for key, value in dict(result.get("icmt_leo_cells", {})).items()
    }
    for class_id in FROZEN_ICMT_CLASS_IDS:
        key = str(class_id)
        _accumulate_cell(
            clean_cells,
            key=f"tx{key}",
            rows=clean_rows[key],
            active=clean_active[key],
            finite=clean_finite[key],
        )
        _accumulate_cell(
            leo_cells,
            key=f"tx{key}|{scenario}",
            rows=leo_rows[key],
            active=leo_active[key],
            finite=leo_finite[key],
        )
    result["icmt_clean_cells"] = clean_cells
    result["icmt_leo_cells"] = leo_cells
    result["icmt_batches"] = int(result.get("icmt_batches", 0)) + 1
    result["icmt_clean_rows"] = int(result.get("icmt_clean_rows", 0)) + int(
        batch_info.get("rows", 0)
    )
    result["icmt_leo_rows"] = int(result.get("icmt_leo_rows", 0)) + int(
        batch_info.get("rows", 0)
    )
    return result


def bind_icmt_source_data_order(
    receipt: Mapping[str, Any], source_split_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind the frozen labeled physical-index source before any training batch."""

    result = dict(receipt)
    source = dict(source_split_receipt or {})
    labeled_sha = str(source.get("labeled_indices_sha256", "") or "")
    manifest_sha = str(source.get("split_manifest_sha256", "") or "")
    if len(labeled_sha) != 64 or len(manifest_sha) != 64:
        raise ICMTConfigurationError("P1-ICMT requires labeled-index and source-split SHA256 receipts")
    result["source_labeled_indices_sha256"] = labeled_sha
    result["source_split_manifest_sha256"] = manifest_sha
    return result


def _as_plain_list(values: Any) -> list[Any]:
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return []


def update_icmt_common_batch_sequence_receipt(
    receipt: Mapping[str, Any],
    *,
    epoch: int,
    batch_index: int,
    scenario: str,
    source_tx_labels: torch.Tensor,
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Chain an opaque physical/data-order digest for both C and G arms.

    This is receipt-only: it neither enters the loss nor changes sampling.
    ``base_index`` is an opaque source index supplied by the existing loader;
    ``sig_i`` is accepted only as a legacy fallback.
    """

    result = dict(receipt)
    expected = FROZEN_ICMT_SCENARIOS[(int(epoch) + int(batch_index) - 2) % 3]
    if str(scenario) != expected:
        raise ICMTRuntimeError("P1-ICMT common LEO scenario sequence drifted")
    labels = source_tx_labels.detach().reshape(-1).long()
    if labels.numel() == 0:
        raise ICMTRuntimeError("P1-ICMT common batch sequence requires source L rows")
    if metadata is None:
        raise ICMTRuntimeError("P1-ICMT common batch sequence requires opaque physical metadata")
    opaque_ids = _as_plain_list(metadata.get("base_index"))
    if len(opaque_ids) != int(labels.numel()):
        opaque_ids = _as_plain_list(metadata.get("sig_i"))
    if len(opaque_ids) != int(labels.numel()):
        raise ICMTRuntimeError("P1-ICMT physical batch sequence metadata is incomplete")
    event = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "scenario": str(scenario),
        "rows": [[str(opaque), int(label)] for opaque, label in zip(opaque_ids, labels.cpu().tolist())],
    }
    prior = str(result.get("common_batch_sequence_sha256", "") or "")
    if not prior:
        prior = str(result.get("source_labeled_indices_sha256", "") or "")
    if len(prior) != 64:
        raise ICMTRuntimeError("P1-ICMT common batch sequence lacks source data-order SHA256")
    result["common_batch_sequence_sha256"] = hashlib.sha256(
        (prior + "\n" + json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode(
            "utf-8"
        )
    ).hexdigest()
    result["common_batch_sequence_batches"] = int(
        result.get("common_batch_sequence_batches", 0)
    ) + 1
    result["common_batch_sequence_rows"] = int(
        result.get("common_batch_sequence_rows", 0)
    ) + int(labels.numel())
    scenario_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if set(scenario_batches) != set(FROZEN_ICMT_SCENARIOS):
        raise ICMTRuntimeError("P1-ICMT common scenario receipt is malformed")
    scenario_batches[str(scenario)] = int(scenario_batches[str(scenario)]) + 1
    result["common_scenario_batches"] = scenario_batches
    return result


def bind_icmt_optimizer_initial_state(
    receipt: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Seal the newly created AdamW state before the first backward call."""

    result = dict(receipt)
    state = optimizer.state_dict()
    if dict(state.get("state", {})):
        raise ICMTConfigurationError("P1-ICMT requires a new AdamW state, not a restored optimizer")
    groups = []
    for group in state.get("param_groups", []):
        normalized = {
            str(key): value
            for key, value in dict(group).items()
            if str(key) != "params"
        }
        normalized["parameter_count"] = len(list(dict(group).get("params", [])))
        groups.append(normalized)
    payload = {
        "optimizer_type": type(optimizer).__name__,
        "state_empty": True,
        "param_groups": groups,
    }
    result["optimizer_initial_state_sha256"] = _canonical_sha256(payload)
    result["optimizer_initial_state_empty"] = True
    return result


def _validate_common_terminal_contract(result: Mapping[str, Any]) -> None:
    for key in (
        "baseline_sha256",
        "initial_checkpoint_sha256",
        "class_order_binding_sha256",
        "source_labeled_indices_sha256",
        "source_split_manifest_sha256",
        "optimizer_initial_state_sha256",
        "common_batch_sequence_sha256",
    ):
        if len(str(result.get(key, "") or "")) != 64:
            raise ICMTRuntimeError(f"P1-ICMT terminal receipt lacks {key}")
    if str(result.get("checkpoint_role", "") or "") != "training_final_only":
        raise ICMTRuntimeError("P1-ICMT requires a training_final_only warm-start checkpoint")
    if result.get("optimizer_state_restored") is not False or result.get("rng_state_restored") is not False:
        raise ICMTRuntimeError("P1-ICMT optimizer/RNG restoration is forbidden")
    if result.get("optimizer_initial_state_empty") is not True:
        raise ICMTRuntimeError("P1-ICMT missing new AdamW initial-state receipt")
    batches = int(result.get("common_batch_sequence_batches", 0))
    rows = int(result.get("common_batch_sequence_rows", 0))
    scenarios = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if batches <= 0 or rows <= 0 or set(scenarios) != set(FROZEN_ICMT_SCENARIOS) or any(
        value <= 0 for value in scenarios.values()
    ):
        raise ICMTRuntimeError("P1-ICMT common batch/scenario receipt is incomplete")


def _validate_cell(cell: Mapping[str, Any]) -> bool:
    rows = int(cell.get("rows", -1))
    active = int(cell.get("active_rows", -1))
    finite = int(cell.get("finite_rows", -1))
    batches = int(cell.get("batches", 0))
    finite_batches = int(cell.get("finite_batches", 0))
    return (
        rows > 0
        and 0 <= active <= rows
        and finite == rows
        and batches > 0
        and finite_batches == batches
        and int(cell.get("nonfinite_rows", 0)) == 0
    )


def validate_icmt_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless common bindings and all 16 G cells close exactly."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    _validate_common_terminal_contract(result)
    if not bool(result.get("enabled", False)):
        if (
            int(result.get("icmt_batches", 0)) != 0
            or int(result.get("icmt_clean_rows", 0)) != 0
            or int(result.get("icmt_leo_rows", 0)) != 0
            or dict(result.get("icmt_clean_cells", {}))
            or dict(result.get("icmt_leo_cells", {}))
        ):
            raise ICMTRuntimeError("P1-ICMT C arm must retain N/A-or-zero ICMT fields")
        result["icmt_terminal_contract"] = "CONTROL_ARM_NOT_APPLICABLE_COMMON_SEQUENCE_BOUND"
        result["icmt_terminal_contract_passed"] = True
        return result
    clean_cells = {
        str(key): dict(value)
        for key, value in dict(result.get("icmt_clean_cells", {})).items()
    }
    leo_cells = {
        str(key): dict(value)
        for key, value in dict(result.get("icmt_leo_cells", {})).items()
    }
    missing = []
    invalid = []
    total_clean_rows = 0
    total_leo_rows = 0
    for class_id in FROZEN_ICMT_CLASS_IDS:
        clean_key = f"tx{class_id}"
        clean = clean_cells.get(clean_key)
        if clean is None:
            missing.append(clean_key)
            continue
        if not _validate_cell(clean):
            invalid.append(clean_key)
            continue
        class_leo_rows = 0
        for scenario in FROZEN_ICMT_SCENARIOS:
            leo_key = f"tx{class_id}|{scenario}"
            leo = leo_cells.get(leo_key)
            if leo is None:
                missing.append(leo_key)
                continue
            if not _validate_cell(leo):
                invalid.append(leo_key)
                continue
            class_leo_rows += int(leo["rows"])
            total_leo_rows += int(leo["rows"])
        if int(clean["rows"]) != class_leo_rows:
            invalid.append(clean_key + "_ROWS_NEQ_SUM_THREE_LEO")
        total_clean_rows += int(clean["rows"])
    if missing or invalid:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if invalid:
            details.append("invalid=" + ",".join(invalid))
        raise ICMTRuntimeError("P1-ICMT terminal 16-cell coverage failed: " + "; ".join(details))
    batches = int(result.get("icmt_batches", 0))
    if (
        batches <= 0
        or int(result.get("icmt_clean_rows", -1)) != total_clean_rows
        or int(result.get("icmt_leo_rows", -1)) != total_leo_rows
        or total_clean_rows != total_leo_rows
        or not bool(result.get("icmt_gradient_relation_completed", False))
    ):
        raise ICMTRuntimeError("P1-ICMT terminal batch, row, or VJP receipt does not close")
    result["icmt_terminal_contract"] = (
        "FORMAL_COMMON_WARM_START_DATA_ORDER_NEW_ADAMW_AND_CLEAN4_LEO4X3_"
        "ROWS_ACTIVE_FINITE_WITH_FIRST_VALID_RAW_UNSCALED_VJP"
    )
    result["icmt_terminal_contract_passed"] = True
    return result


def _failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "vjp" in message or "gradient" in message or "head-only" in message:
        return "ICMT_VJP_PATH_FAILURE"
    if "non-finite" in message or "nonfinite" in message:
        return "ICMT_NONFINITE"
    if "n_c" in message or "local4" in message or "class" in message:
        return "ICMT_LOCAL4_COVERAGE_FAILURE"
    if "sequence" in message or "receipt" in message or "rows/active/finite" in message:
        return "ICMT_RECEIPT_CLOSURE_FAILURE"
    if "head" in message or "binding" in message or "feat_joint" in message:
        return "ICMT_BINDING_FAILURE"
    return "ICMT_RUNTIME_FAILURE"


def write_icmt_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free fail-closed record for the ICMT arm."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise ICMTRuntimeError("P1-ICMT failure receipt requires an existing output directory")
    target = target_dir / "icmt_failure_receipt.json"
    payload = {
        "schema": "cvs.phase1.icmt_failure_receipt.v1",
        "status": "FAIL_CLOSED",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage),
        "error_type": type(error).__name__,
        "error_fingerprint": _failure_fingerprint(error),
        "icmt_receipt": dict(receipt),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = mkstemp(prefix=".icmt_failure_receipt.", suffix=".tmp", dir=str(target_dir))
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


def strict_icmt_warm_start(
    model: torch.nn.Module,
    checkpoint_model_state: Mapping[str, torch.Tensor],
    *,
    baseline_path: str,
    baseline_sha256: str,
    checkpoint_epoch: int,
    checkpoint_role: str,
) -> Dict[str, Any]:
    """Load model weights only; optimizer and RNG state are deliberately new."""

    path = str(baseline_path or "").strip()
    digest = str(baseline_sha256 or "").strip()
    if not path or len(digest) != 64 or not isinstance(checkpoint_model_state, Mapping):
        raise ICMTConfigurationError(
            "Frozen P1-ICMT warm-start requires model state, path, and SHA256"
        )
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise ICMTConfigurationError(
            f"Frozen P1-ICMT strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise ICMTConfigurationError(
            "Frozen P1-ICMT strict baseline model-key mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    role = str(checkpoint_role or "")
    if role != "training_final_only":
        raise ICMTConfigurationError(
            "Frozen P1-ICMT requires baseline checkpoint_role=training_final_only"
        )
    return {
        "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP",
        "baseline_path": path,
        "baseline_sha256": digest,
        "initial_checkpoint_sha256": digest,
        "checkpoint_epoch": epoch,
        "checkpoint_role": role,
        "strict_model_keys": True,
        "missing_model_keys": [],
        "unexpected_model_keys": [],
        "optimizer_state_restored": False,
        "rng_state_restored": False,
    }
