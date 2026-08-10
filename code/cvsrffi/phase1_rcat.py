"""Frozen P1-RCAT continuation contract for Phase1 source-only DG.

P1-RCAT (Receiver-Conditioned Angular Transport) leaves the common F1C/
GeoSat-C clean and single-LEO forwards untouched.  Its G arm adds exactly one
source-L-only term on the existing ``feat_joint`` tensor.  The clean angular
anchor is detached; the LEO feature and its shared encoder remain live.  No
head loss, state, cache, threshold, resampling, RX/day model input, or extra
forward is introduced here.
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


FROZEN_RCAT_LAMBDA = 0.02
FROZEN_RCAT_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_RCAT_CLASS_IDS = (0, 1, 2, 3)
FROZEN_RCAT_SOURCE_RECEIVER_IDS = (0, 1, 2, 3, 4, 5, 6)
FROZEN_RCAT_TERM_DIVISOR = 28
FROZEN_RCAT_OPTIMIZER_TYPE = "AdamW"
FROZEN_RCAT_FLOAT32_LEDGER_REL_TOL = 32.0 * float(torch.finfo(torch.float32).eps)
RCAT_RECEIPT_SCHEMA = "cvs.phase1.rcat_receipt.v1"
_TOLERANCE = 1e-12


class RCATConfigurationError(ValueError):
    """Raised when a frozen P1-RCAT C/G configuration drifts."""


class RCATRuntimeError(RuntimeError):
    """Raised when a P1-RCAT runtime or receipt contract cannot be proved."""


@dataclass(frozen=True)
class RCATConfig:
    """Immutable P1-RCAT controls consumed by the common training loop."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise RCATConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
        raise RCATConfigurationError(
            f"Frozen P1-RCAT requires {name}={expected:.12g}, got {actual!r}"
        )


def _float32_ledger_close(actual: float, expected: float) -> bool:
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    return abs(actual - expected) <= FROZEN_RCAT_FLOAT32_LEDGER_REL_TOL * max(
        1.0, abs(actual), abs(expected)
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
        raise RCATConfigurationError(
            "Frozen P1-RCAT forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    scenarios = tuple(
        part.strip().lower().replace("-", "_")
        for part in str(value or "").split(",")
        if part.strip()
    )
    if scenarios != FROZEN_RCAT_SCENARIOS:
        raise RCATConfigurationError(
            "Frozen P1-RCAT requires --sat_train_scenarios "
            + ",".join(FROZEN_RCAT_SCENARIOS)
        )
    return scenarios


def validate_rcat_args(args: Any) -> RCATConfig:
    """Validate the frozen common-base C/G contract before data are loaded."""

    frozen_mode = _bool_arg(args, "phase1_rcat_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_rcat_enabled", False)
    loss_weight = _float_arg(args, "lambda_rcat", 0.0)
    if not frozen_mode and not enabled:
        return RCATConfig(False, False, 0.0)
    if enabled and not frozen_mode:
        raise RCATConfigurationError(
            "--phase1_rcat_enabled requires --phase1_rcat_frozen_mode true"
        )
    _require_close("lambda_rcat", loss_weight, FROZEN_RCAT_LAMBDA if enabled else 0.0)
    if bool(getattr(args, "from_scratch", True)):
        raise RCATConfigurationError("Frozen P1-RCAT requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise RCATConfigurationError("Frozen P1-RCAT requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise RCATConfigurationError("Frozen P1-RCAT must train the shared feat_joint encoder")
    if not bool(getattr(args, "amp", True)):
        raise RCATConfigurationError("Frozen P1-RCAT requires the common AMP training path")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise RCATConfigurationError("Frozen P1-RCAT requires --id_feature_key feat_joint")
    if int(getattr(args, "epochs", 0)) != 40 or int(getattr(args, "label_epochs", 0)) != 40:
        raise RCATConfigurationError("Frozen P1-RCAT requires exactly 40 labeled epochs")
    if int(getattr(args, "pseudo_epochs", 0)) != 0:
        raise RCATConfigurationError("Frozen P1-RCAT forbids pseudo epochs")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise RCATConfigurationError("Frozen P1-RCAT requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise RCATConfigurationError("Frozen P1-RCAT remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise RCATConfigurationError("Frozen P1-RCAT requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise RCATConfigurationError("Frozen P1-RCAT requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise RCATConfigurationError("Frozen P1-RCAT forbids --sat_view_schedule overrides")
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise RCATConfigurationError("Frozen P1-RCAT requires non-concatenated single-LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise RCATConfigurationError("Frozen P1-RCAT permits only source_known_train L updates")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise RCATConfigurationError("Frozen P1-RCAT forbids RX/day-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise RCATConfigurationError("Frozen P1-RCAT permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise RCATConfigurationError("Frozen P1-RCAT forbids a reject head")
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
            "phase1_icmt_frozen_mode",
            "phase1_icmt_enabled",
            "lambda_icmt",
            "phase1_cagm_frozen_mode",
            "phase1_cagm_enabled",
            "lambda_cagm",
            "phase1_rcrmd_frozen_mode",
            "phase1_rcrmd_enabled",
            "lambda_rcrmd",
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
    return RCATConfig(True, enabled, loss_weight)


def rcat_config_receipt(config: RCATConfig) -> Dict[str, Any]:
    """Create the data-free receipt skeleton for either frozen C/G arm."""

    return {
        "schema": RCAT_RECEIPT_SCHEMA,
        "method": "P1_RCAT",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "loss_rule": "SOURCE_L_RX_BY_LOCAL4_EQUAL_WEIGHT_STOPGRAD_CLEAN_TO_LEO_TOTALIZED_L2_feat_joint",
        "loss_formula": "T(z)=z/||z||2_if_norm_gt_0_else_0;q=||T(z_leo)-sg(T(z_clean))||2;g_rc=0_if_n_rc=0_else_mean_Irc(q);L=sum_rc(g_rc)/28",
        "loss_global_denominator": FROZEN_RCAT_TERM_DIVISOR,
        "local_class_ids": list(FROZEN_RCAT_CLASS_IDS),
        "frozen_source_receiver_ids": list(FROZEN_RCAT_SOURCE_RECEIVER_IDS),
        "frozen_cells_per_scene": FROZEN_RCAT_TERM_DIVISOR,
        "z_id_key": "feat_joint",
        "feature_dimension_contract": "RAW_ENCODER_feat_joint_EXACT_HEAD_INPUT_DIMENSION_BOUND",
        "totalized_l2_rule": "T(z)=z/||z||2_IF_norm_gt_0_ELSE_0",
        "training_accumulation_dtype": "float32",
        "postfreeze_totalized_l2_dtype": "float64_SAME_PIECEWISE_RULE_NOT_BYTE_IDENTICAL",
        "clean_feature_detached": True,
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD",
        "common_lambda_sat_cons": 0.10,
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "common_l_base_head_input_path_verified": False,
        "aux_gradient_scope": "LEO_feat_joint_AND_SHARED_ENCODER_FINITE_NONZERO;EXACT_HEAD_AUX_VJP_NA_NONE_OR_ZERO",
        "uses_new_forward": False,
        "uses_resampling": False,
        "uses_rx_labels": True,
        "rx_permission": "SOURCE_KNOWN_TRAIN_L_PHYSICAL_ID_BOUND_rx_i_ONLY",
        "rx_metadata_allowlist": ["rx_i"],
        "no_day_assertion": "day_i_NOT_READ_BY_RCAT",
        "uses_day_labels": False,
        "uses_domain_labels": False,
        "uses_target_rows": False,
        "uses_proxy_rows": False,
        "uses_held_rows": False,
        "uses_unlabeled_rows": False,
        "uses_ema_or_state": False,
        "uses_threshold": False,
        "uses_gradient_projection": False,
        "uses_cross_sample_pairing": False,
        "uses_cross_receiver_pairing": False,
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
        "optimizer_type": "",
        "optimizer_initial_state_sha256": "",
        "optimizer_initial_state_empty": False,
        "amp_contract": "COMMON_TRAINER_AMP_ENABLED",
        "source_train_tx": [],
        "source_known_validation_tx": [],
        "source_proxy_unknown_tx": [],
        "source_partition_sha256": "",
        "source_labeled_indices_sha256": "",
        "source_split_manifest_sha256": "",
        "source_receiver_ids": [],
        "source_receiver_count": 0,
        "source_receiver_ids_sha256": "",
        "source_receiver_provenance": "PENDING_SOURCE_L_ONLY_BINDING",
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
        "common_scenario_batches": {scenario: 0 for scenario in FROZEN_RCAT_SCENARIOS},
        "rcat_common_cells": {},
        "rcat_common_batch_cells": [],
        "rcat_batches": 0,
        "rcat_total_rows": 0,
        "rcat_positive_q": 0,
        "rcat_loss_sum": 0.0,
        "rcat_float32_ledger_rel_tolerance": FROZEN_RCAT_FLOAT32_LEDGER_REL_TOL,
        "rcat_scenes": {},
        "rcat_g_batch_aux": [],
        "rcat_gradient_audit_attempted": False,
        "rcat_gradient_audit_completed": False,
        "rcat_gradient_audit": {},
        "rcat_terminal_contract": "PENDING",
        "rcat_terminal_contract_passed": False,
        "proxy_rows": 0,
        "held_rows": 0,
    }


def _normalized_tx_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RCATConfigurationError(f"P1-RCAT {name} must be a TX class sequence")
    order = tuple(str(value).strip() for value in values)
    if not order or len(order) != len(set(order)) or any(not value for value in order):
        raise RCATConfigurationError(f"P1-RCAT {name} must be non-empty and unique")
    return order


def _positive_count(name: str, value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise RCATConfigurationError(f"P1-RCAT {name} must be an integer") from exc
    if count <= 0:
        raise RCATConfigurationError(f"P1-RCAT {name} must be positive")
    return count


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _receiver_key(receiver_id: int, class_id: int) -> str:
    return f"rx{int(receiver_id)}|tx{int(class_id)}"


def _source_receiver_ids(values: Any) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RCATConfigurationError("P1-RCAT source_receivers must be a source-only sequence")
    parsed = []
    for value in values:
        text = str(value).strip()
        if not text:
            raise RCATConfigurationError("P1-RCAT source receiver id may not be empty")
        try:
            receiver_id = int(text)
        except (TypeError, ValueError) as exc:
            raise RCATConfigurationError("P1-RCAT source receiver id must be an integer") from exc
        if str(receiver_id) != text:
            raise RCATConfigurationError("P1-RCAT source receiver id is not canonical")
        parsed.append(receiver_id)
    canonical = tuple(sorted(parsed))
    if not canonical or len(canonical) != len(set(canonical)):
        raise RCATConfigurationError("P1-RCAT source receiver ids must be non-empty and unique")
    return canonical


def _require_frozen_source_receivers(receivers: Sequence[Any]) -> Tuple[int, ...]:
    parsed = _source_receiver_ids(receivers)
    if parsed != FROZEN_RCAT_SOURCE_RECEIVER_IDS:
        raise RCATConfigurationError(
            "P1-RCAT requires frozen F1C source receivers 0..6; got " + str(list(parsed))
        )
    return parsed


def resolve_rcat_local_head_class_binding(
    *,
    local_class_order: Sequence[Any],
    source_train_tx: Sequence[Any],
    checkpoint_train_tx: Sequence[Any],
    dataset_class_order: Sequence[Any],
    local_data_class_count: Any,
    checkpoint_head_class_count: Any,
    live_head_class_count: Any,
) -> Dict[str, Any]:
    """Bind contiguous local L labels to the strict warm-start local4 head rows."""

    local = _normalized_tx_order("local data class order", local_class_order)
    source = _normalized_tx_order("source-train TX receipt", source_train_tx)
    checkpoint = _normalized_tx_order("checkpoint train TX receipt", checkpoint_train_tx)
    dataset = _normalized_tx_order("dataset TX class order", dataset_class_order)
    local_count = _positive_count("local data class count", local_data_class_count)
    checkpoint_count = _positive_count("checkpoint classifier head row count", checkpoint_head_class_count)
    live_count = _positive_count("live classifier head row count", live_head_class_count)
    if local_count != 4 or len(local) != 4:
        raise RCATConfigurationError("P1-RCAT requires exactly four local source-TX rows")
    if local != source or checkpoint != source:
        raise RCATConfigurationError(
            "P1-RCAT local/checkpoint TX order must equal the source-train receipt"
        )
    if local_count != len(local) or checkpoint_count != live_count or live_count != local_count:
        raise RCATConfigurationError("P1-RCAT local/head class counts must match")
    if set(local).difference(dataset):
        raise RCATConfigurationError("P1-RCAT local TX labels are absent from dataset order")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(FROZEN_RCAT_CLASS_IDS),
        "expected_tx_class_ids": list(FROZEN_RCAT_CLASS_IDS),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = _canonical_sha256(binding)
    return binding


def remap_rcat_local_labels_to_head_rows(
    local_labels: torch.Tensor, local_to_head_class_ids: Sequence[Any]
) -> torch.Tensor:
    """Map contiguous local source labels through the sealed identity mapping."""

    if not torch.is_tensor(local_labels):
        raise RCATRuntimeError("P1-RCAT local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if mapping != FROZEN_RCAT_CLASS_IDS:
        raise RCATRuntimeError("P1-RCAT local-to-head mapping must be local4 identity")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCATRuntimeError("P1-RCAT local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_rcat_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    """Resolve the exact common head only to bind the common L_base path."""

    raw_model = getattr(model, "_orig_mod", model)
    try:
        head = raw_model.id_backbone.cls_head.head
    except AttributeError as exc:
        raise RCATRuntimeError("P1-RCAT requires model.id_backbone.cls_head.head") from exc
    if not isinstance(head, torch.nn.Module):
        raise RCATRuntimeError("P1-RCAT exact classifier head is not a module")
    if not tuple(parameter for parameter in head.parameters() if parameter.requires_grad):
        raise RCATRuntimeError("P1-RCAT exact classifier head has no trainable parameter")
    return head


def resolve_rcat_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    weight = getattr(resolve_rcat_classifier_head(model), "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise RCATRuntimeError("P1-RCAT classifier head weight must be a rank-2 Parameter")
    return weight


def _validated_receiver_labels(
    receiver_labels: torch.Tensor, *, rows: int, expected_receiver_ids: Sequence[Any]
) -> torch.Tensor:
    if not torch.is_tensor(receiver_labels):
        raise RCATRuntimeError("P1-RCAT requires source-L physical rx_i labels")
    values = receiver_labels.reshape(-1).long()
    expected = _require_frozen_source_receivers(expected_receiver_ids)
    if values.numel() != int(rows) or values.numel() == 0:
        raise RCATRuntimeError("P1-RCAT source-L rx_i rows do not align")
    observed = {int(value) for value in values.detach().cpu().tolist()}
    if observed.difference(set(expected)):
        raise RCATRuntimeError("P1-RCAT rx_i contains a receiver outside frozen source R_s")
    return values


def _validate_view_binding(
    *,
    view_name: str,
    output: Mapping[str, Any],
    labels: torch.Tensor,
    head_weight: torch.Tensor,
) -> torch.Tensor:
    if str(output.get("z_id_key", "")) != "feat_joint":
        raise RCATRuntimeError(f"P1-RCAT {view_name} z_id_key must be feat_joint")
    z_id = output.get("z_id")
    logits = output.get("tx_logits")
    if not torch.is_tensor(z_id) or z_id.ndim != 2:
        raise RCATRuntimeError(f"P1-RCAT {view_name} z_id must be rank-2 feat_joint")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise RCATRuntimeError(f"P1-RCAT {view_name} tx_logits must be rank-2 raw logits")
    if z_id.size(0) != labels.numel() or logits.size(0) != labels.numel():
        raise RCATRuntimeError(f"P1-RCAT {view_name} rows must align with source L labels")
    if int(head_weight.size(0)) != 4 or int(logits.size(1)) != 4:
        raise RCATRuntimeError(f"P1-RCAT {view_name} head/logit class rows must be local4")
    if int(head_weight.size(1)) != int(z_id.size(1)):
        raise RCATRuntimeError(f"P1-RCAT {view_name} feat_joint/head dimension binding drifted")
    if not bool(z_id.requires_grad) or not bool(logits.requires_grad):
        raise RCATRuntimeError(f"P1-RCAT {view_name} requires a live feat_joint/head path")
    if not bool(torch.isfinite(z_id.detach()).all().item()):
        raise RCATRuntimeError(f"P1-RCAT {view_name} feat_joint is non-finite")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise RCATRuntimeError(f"P1-RCAT {view_name} raw logits are non-finite")
    return z_id


def validate_rcat_binding(
    *,
    model: torch.nn.Module,
    out_clean: Mapping[str, Any],
    out_leo: Mapping[str, Any],
    tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
    expected_receiver_ids: Sequence[Any],
) -> torch.nn.Parameter:
    """Fail closed unless common forwards expose feat_joint and live head inputs."""

    if not isinstance(out_clean, Mapping) or not isinstance(out_leo, Mapping):
        raise RCATRuntimeError("P1-RCAT requires clean and LEO mapping outputs")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCATRuntimeError("P1-RCAT source labels must bind to local4 head rows")
    if tuple(int(value) for value in expected_class_ids) != FROZEN_RCAT_CLASS_IDS:
        raise RCATRuntimeError("P1-RCAT expected local4 class order is invalid")
    _validated_receiver_labels(
        source_rx_labels, rows=int(labels.numel()), expected_receiver_ids=expected_receiver_ids
    )
    head_weight = resolve_rcat_classifier_weight(model)
    if not bool(torch.isfinite(head_weight.detach()).all().item()):
        raise RCATRuntimeError("P1-RCAT exact classifier head is non-finite")
    _validate_view_binding(view_name="clean", output=out_clean, labels=labels, head_weight=head_weight)
    _validate_view_binding(view_name="leo", output=out_leo, labels=labels, head_weight=head_weight)
    return head_weight


def _totalized_l2_with_zeros(features: torch.Tensor, *, view_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(features) or features.ndim != 2 or features.size(0) <= 0 or features.size(1) <= 0:
        raise RCATRuntimeError(f"P1-RCAT {view_name} feat_joint must be non-empty rank-2")
    if not bool(torch.isfinite(features.detach()).all().item()):
        raise RCATRuntimeError(f"P1-RCAT {view_name} feat_joint is non-finite")
    values = features.float()
    norms = torch.linalg.vector_norm(values, ord=2, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms.detach()).all().item()):
        raise RCATRuntimeError(f"P1-RCAT {view_name} feat_joint norm is non-finite")
    nonzero = norms.gt(0.0)
    safe_norms = torch.where(nonzero, norms, torch.ones_like(norms))
    normalized = values / safe_norms
    totalized = torch.where(nonzero, normalized, torch.zeros_like(values))
    if not bool(torch.isfinite(totalized.detach()).all().item()):
        raise RCATRuntimeError(f"P1-RCAT {view_name} totalized-L2 output is non-finite")
    return totalized, ~nonzero.reshape(-1)


def totalized_l2(features: torch.Tensor) -> torch.Tensor:
    """Return float32 ``T(z)`` with the exact zero-to-zero totalization rule."""

    return _totalized_l2_with_zeros(features, view_name="input")[0]


def _cell_template(receiver_ids: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    receivers = _source_receiver_ids(receiver_ids)
    return {
        _receiver_key(receiver_id, class_id): {
            "rows": 0,
            "positive_q": 0,
            "finite_q": 0,
            "clean_zero_rows": 0,
            "leo_zero_rows": 0,
            "union_zero_rows": 0,
            "both_zero_rows": 0,
            "nonfinite_rows": 0,
            "q_sum": 0.0,
            "g_sum": 0.0,
            "loss_sum": 0.0,
            "batches": 0,
            "nonempty_batches": 0,
            "finite_batches": 0,
        }
        for receiver_id in receivers
        for class_id in FROZEN_RCAT_CLASS_IDS
    }


def _batch_cell_weights(
    receiver_ids: Sequence[Any], counts: Mapping[str, int]
) -> Dict[str, Dict[str, float]]:
    receivers = _source_receiver_ids(receiver_ids)
    expected = {
        _receiver_key(receiver_id, class_id)
        for receiver_id in receivers
        for class_id in FROZEN_RCAT_CLASS_IDS
    }
    if set(counts) != expected:
        raise RCATRuntimeError("P1-RCAT batch cell count coverage drifted")
    scale = 1.0 / float(FROZEN_RCAT_TERM_DIVISOR)
    return {
        key: {
            "cell_weight": scale,
            "row_weight": (scale / float(int(counts[key]))) if int(counts[key]) > 0 else 0.0,
        }
        for key in sorted(expected)
    }


def rcat_loss(
    clean_feat_joint: torch.Tensor,
    leo_feat_joint: torch.Tensor,
    source_tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    source_receiver_ids: Sequence[Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute the one fixed-scale receiver×class RCAT auxiliary loss.

    Only the same source-L physical row's existing clean and one LEO feature are
    paired.  Each of the fixed 7×4 cells contributes its mean, or a
    differentiable zero if empty; no occupied-cell renormalization is allowed.
    """

    labels = source_tx_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCATRuntimeError("P1-RCAT source labels are outside local4")
    receivers = _require_frozen_source_receivers(source_receiver_ids)
    rx_labels = _validated_receiver_labels(
        source_rx_labels, rows=int(labels.numel()), expected_receiver_ids=receivers
    )
    clean_t, clean_zero = _totalized_l2_with_zeros(clean_feat_joint, view_name="clean")
    leo_t, leo_zero = _totalized_l2_with_zeros(leo_feat_joint, view_name="leo")
    if clean_t.shape != leo_t.shape or clean_t.size(0) != labels.numel():
        raise RCATRuntimeError("P1-RCAT clean/LEO feat_joint rows or dimensions do not align")
    delta = leo_t - clean_t.detach()
    q = delta.square().sum(dim=1)
    if not bool(torch.isfinite(q.detach()).all().item()):
        raise RCATRuntimeError("P1-RCAT q contains non-finite values")
    cells: Dict[str, Dict[str, Any]] = {}
    terms = []
    total_rows = 0
    total_positive = 0
    total_finite = 0
    clean_zero_total = 0
    leo_zero_total = 0
    union_zero_total = 0
    both_zero_total = 0
    scale = 1.0 / float(FROZEN_RCAT_TERM_DIVISOR)
    for receiver_id in receivers:
        for class_id in FROZEN_RCAT_CLASS_IDS:
            key = _receiver_key(receiver_id, class_id)
            mask = rx_labels.eq(receiver_id) & labels.eq(class_id)
            count = int(mask.sum().item())
            group_q = q[mask]
            if count == 0:
                g_rc = group_q.sum()
                positive = finite = clean_zero_count = leo_zero_count = union_zero_count = both_zero_count = 0
                q_sum = 0.0
            else:
                if not bool(torch.isfinite(group_q.detach()).all().item()):
                    raise RCATRuntimeError("P1-RCAT receiver/class q is non-finite")
                group_clean_zero = clean_zero[mask]
                group_leo_zero = leo_zero[mask]
                g_rc = group_q.mean()
                positive = int(group_q.detach().gt(0.0).sum().item())
                finite = int(torch.isfinite(group_q.detach()).sum().item())
                clean_zero_count = int(group_clean_zero.sum().item())
                leo_zero_count = int(group_leo_zero.sum().item())
                both_zero_count = int((group_clean_zero & group_leo_zero).sum().item())
                union_zero_count = int((group_clean_zero | group_leo_zero).sum().item())
                q_sum = float(group_q.detach().sum().item())
            if not bool(torch.isfinite(g_rc.detach()).item()):
                raise RCATRuntimeError("P1-RCAT g_rc is non-finite")
            if (
                positive < 0
                or positive > count
                or finite != count
                or union_zero_count != clean_zero_count + leo_zero_count - both_zero_count
            ):
                raise RCATRuntimeError("P1-RCAT receiver/class counters do not close")
            cells[key] = {
                "n_rc": count,
                "positive_q": positive,
                "finite_q": finite,
                "clean_zero_rows": clean_zero_count,
                "leo_zero_rows": leo_zero_count,
                "union_zero_rows": union_zero_count,
                "both_zero_rows": both_zero_count,
                "nonfinite_rows": 0,
                "q_sum": q_sum,
                "g_rc": float(g_rc.detach().item()),
                "loss_contribution": scale * float(g_rc.detach().item()),
            }
            terms.append(g_rc)
            total_rows += count
            total_positive += positive
            total_finite += finite
            clean_zero_total += clean_zero_count
            leo_zero_total += leo_zero_count
            union_zero_total += union_zero_count
            both_zero_total += both_zero_count
    if total_rows != int(labels.numel()) or total_finite != total_rows:
        raise RCATRuntimeError("P1-RCAT batch receiver/class coverage does not close")
    loss = torch.stack(terms).sum() * scale
    if not bool(torch.isfinite(loss.detach()).item()):
        raise RCATRuntimeError("P1-RCAT loss is non-finite")
    counts = {key: int(value["n_rc"]) for key, value in cells.items()}
    weights = _batch_cell_weights(receivers, counts)
    for key in cells:
        cells[key].update(weights[key])
    return loss, {
        "rows": int(labels.numel()),
        "positive_q": total_positive,
        "finite_q": total_finite,
        "clean_zero_rows": clean_zero_total,
        "leo_zero_rows": leo_zero_total,
        "union_zero_rows": union_zero_total,
        "both_zero_rows": both_zero_total,
        "nonfinite_rows": 0,
        "loss_sum": float(loss.detach().item()),
        "global_denominator": FROZEN_RCAT_TERM_DIVISOR,
        "fixed_scale": scale,
        "source_receiver_ids": list(receivers),
        "cells": cells,
        "finite": True,
        "clean_feature_detached": True,
        "totalized_l2_rule": "NORM_GT_0_NORMALIZE_ELSE_ZERO",
        "training_accumulation_dtype": "float32",
        "empty_cell_zero": True,
        "no_active_renormalization": True,
        "no_cross_sample_or_cross_receiver_pairing": True,
    }


def add_rcat_to_loss(
    base_loss: torch.Tensor, rcat: Optional[torch.Tensor], config: Optional[RCATConfig]
) -> torch.Tensor:
    """Add the sole G-arm term; C returns the untouched common base tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if rcat is None:
        raise RCATRuntimeError("Enabled P1-RCAT requires its auxiliary loss")
    return base_loss + float(config.loss_weight) * rcat


def rcat_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    """Return shared feat_joint encoder and exact-head diagnostic scopes."""

    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise RCATRuntimeError("P1-RCAT requires model.id_backbone for VJP audit")
    head = resolve_rcat_classifier_head(raw_model)
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
        raise RCATRuntimeError("P1-RCAT shared encoder or exact head audit scope is empty")
    return {"shared_encoder": encoder, "classifier_head": head_parameters}


def _finite_nonzero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.Tensor], *, group_name: str
) -> Dict[str, float]:
    params = tuple(parameters)
    if not params:
        raise RCATRuntimeError(f"P1-RCAT {group_name} VJP scope is empty")
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    squared_norm = 0.0
    for gradient in gradients:
        if gradient is None:
            raise RCATRuntimeError(f"P1-RCAT {group_name} VJP is None or detached")
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise RCATRuntimeError(f"P1-RCAT {group_name} VJP is non-finite")
        value = gradient.detach().double()
        squared_norm += float(torch.sum(value * value).item())
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm) or norm <= 0.0:
        raise RCATRuntimeError(f"P1-RCAT {group_name} VJP norm is zero or non-finite")
    return {"parameter_count": float(len(params)), "norm": float(norm)}


def _head_none_or_zero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter]
) -> Dict[str, Any]:
    params = tuple(parameters)
    if not params:
        raise RCATRuntimeError("P1-RCAT classifier head VJP scope is empty")
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    none_count = 0
    zero_count = 0
    for gradient in gradients:
        if gradient is None:
            none_count += 1
            continue
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise RCATRuntimeError("P1-RCAT classifier head auxiliary VJP is non-finite")
        if int(torch.count_nonzero(gradient.detach()).item()) != 0:
            raise RCATRuntimeError("P1-RCAT classifier head must have no auxiliary gradient")
        zero_count += 1
    return {
        "parameter_count": float(len(params)),
        "none_parameters": float(none_count),
        "zero_parameters": float(zero_count),
        "nonzero_parameters": 0.0,
        "none_or_zero_expected": True,
    }


def rcat_aux_gradient_audit(
    rcat: torch.Tensor,
    feat_joint_leo: torch.Tensor,
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
) -> Dict[str, Any]:
    """Audit first-positive raw RCAT VJPs without touching AMP or optimizer state."""

    if not torch.is_tensor(rcat) or rcat.ndim != 0:
        raise RCATRuntimeError("P1-RCAT VJP audit requires a scalar auxiliary loss")
    if not torch.is_tensor(feat_joint_leo) or feat_joint_leo.ndim != 2:
        raise RCATRuntimeError("P1-RCAT VJP audit requires LEO feat_joint")
    if tuple(parameter_groups.keys()) != ("shared_encoder", "classifier_head"):
        raise RCATRuntimeError("P1-RCAT VJP audit requires encoder and exact-head scopes")
    result = {
        "feat_joint_leo": _finite_nonzero_vjp(
            rcat, (feat_joint_leo,), group_name="feat_joint_leo"
        ),
        "shared_encoder": _finite_nonzero_vjp(
            rcat, parameter_groups["shared_encoder"], group_name="shared_encoder"
        ),
        "classifier_head": _head_none_or_zero_vjp(rcat, parameter_groups["classifier_head"]),
        "raw_unscaled": True,
        "diagnostic_only": True,
        "touches_amp_optimizer_rng": False,
        "exact_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        "common_l_base_head_input_path": "LIVE_AND_BOUND_SEPARATELY",
    }
    return result


def update_rcat_gradient_audit_receipt(
    receipt: Mapping[str, Any], audit: Mapping[str, Any]
) -> Dict[str, Any]:
    """Seal the first positive-q VJP audit exactly once."""

    result = dict(receipt)
    if bool(result.get("rcat_gradient_audit_completed", False)):
        raise RCATRuntimeError("P1-RCAT VJP audit may run only once")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
        or audit.get("exact_head_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
    ):
        raise RCATRuntimeError("P1-RCAT VJP audit semantics drifted")
    for group_name in ("feat_joint_leo", "shared_encoder"):
        values = audit.get(group_name)
        if not isinstance(values, Mapping):
            raise RCATRuntimeError("P1-RCAT VJP audit lacks a required nonzero scope")
        count = float(values.get("parameter_count", 0.0))
        norm = float(values.get("norm", float("nan")))
        if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
            raise RCATRuntimeError("P1-RCAT required auxiliary VJP is zero or non-finite")
    head = audit.get("classifier_head")
    if not isinstance(head, Mapping):
        raise RCATRuntimeError("P1-RCAT VJP audit lacks exact-head N/A scope")
    head_count = float(head.get("parameter_count", 0.0))
    none_count = float(head.get("none_parameters", float("nan")))
    zero_count = float(head.get("zero_parameters", float("nan")))
    nonzero_count = float(head.get("nonzero_parameters", float("nan")))
    if (
        head_count <= 0.0
        or not all(math.isfinite(value) and value >= 0.0 for value in (none_count, zero_count, nonzero_count))
        or none_count + zero_count != head_count
        or nonzero_count != 0.0
        or head.get("none_or_zero_expected") is not True
    ):
        raise RCATRuntimeError("P1-RCAT exact-head auxiliary VJP contract failed")
    result["rcat_gradient_audit_attempted"] = True
    result["rcat_gradient_audit_completed"] = True
    result["rcat_gradient_audit"] = dict(audit)
    return result


def bind_rcat_source_data_order(
    receipt: Mapping[str, Any], source_split_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind source-L physical order and its source-only RX allowlist."""

    result = dict(receipt)
    source = dict(source_split_receipt or {})
    labeled_sha = str(source.get("labeled_indices_sha256", "") or "")
    manifest_sha = str(source.get("split_manifest_sha256", "") or "")
    if len(labeled_sha) != 64 or len(manifest_sha) != 64:
        raise RCATConfigurationError(
            "P1-RCAT requires labeled-index and source-split SHA256 receipts"
        )
    receivers = _require_frozen_source_receivers(source.get("source_receivers", ()))
    result["source_labeled_indices_sha256"] = labeled_sha
    result["source_split_manifest_sha256"] = manifest_sha
    result["source_receiver_ids"] = list(receivers)
    result["source_receiver_count"] = len(receivers)
    result["source_receiver_ids_sha256"] = _canonical_sha256(list(receivers))
    result["source_receiver_provenance"] = "SOURCE_SPLIT_RECEIPT_source_receivers_PHYSICAL_ID_BOUND_L_ONLY"
    return result


def _as_plain_list(values: Any) -> list[Any]:
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return []


def _common_cell_event(
    *, receiver_ids: Sequence[Any], labels: torch.Tensor, rx_labels: torch.Tensor
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    receivers = _source_receiver_ids(receiver_ids)
    counts = {
        _receiver_key(receiver_id, class_id): int(
            (rx_labels.eq(receiver_id) & labels.eq(class_id)).sum().item()
        )
        for receiver_id in receivers
        for class_id in FROZEN_RCAT_CLASS_IDS
    }
    if sum(counts.values()) != int(labels.numel()):
        raise RCATRuntimeError("P1-RCAT common n_rc counters do not close")
    return counts, _batch_cell_weights(receivers, counts)


def update_rcat_common_batch_sequence_receipt(
    receipt: Mapping[str, Any],
    *,
    epoch: int,
    batch_index: int,
    scenario: str,
    source_tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Chain C/G-identical physical/RX/class/scene coverage and n_rc receipts."""

    result = dict(receipt)
    expected = FROZEN_RCAT_SCENARIOS[(int(epoch) + int(batch_index) - 2) % 3]
    if str(scenario) != expected:
        raise RCATRuntimeError("P1-RCAT common LEO scenario sequence drifted")
    labels = source_tx_labels.detach().reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCATRuntimeError("P1-RCAT common sequence requires local4 source L labels")
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    rx_labels = _validated_receiver_labels(
        source_rx_labels, rows=int(labels.numel()), expected_receiver_ids=receivers
    ).detach()
    if metadata is None:
        raise RCATRuntimeError("P1-RCAT common sequence requires opaque physical metadata")
    opaque_ids = _as_plain_list(metadata.get("base_index"))
    if len(opaque_ids) != int(labels.numel()):
        opaque_ids = _as_plain_list(metadata.get("sig_i"))
    if len(opaque_ids) != int(labels.numel()):
        raise RCATRuntimeError("P1-RCAT physical batch sequence metadata is incomplete")
    counts, weights = _common_cell_event(
        receiver_ids=receivers, labels=labels, rx_labels=rx_labels
    )
    event = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "scenario": str(scenario),
        "same_physical_clean_leo": True,
        "rows": [
            [str(opaque), int(label), int(receiver_id)]
            for opaque, label, receiver_id in zip(
                opaque_ids, labels.cpu().tolist(), rx_labels.cpu().tolist()
            )
        ],
        "n_rc": counts,
        "effective_weights": weights,
    }
    prior = str(result.get("common_batch_sequence_sha256", "") or "")
    if not prior:
        prior = str(result.get("source_labeled_indices_sha256", "") or "")
    if len(prior) != 64:
        raise RCATRuntimeError("P1-RCAT common batch sequence lacks source data-order SHA256")
    result["common_batch_sequence_sha256"] = hashlib.sha256(
        (prior + "\n" + json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()
    result["common_batch_sequence_batches"] = int(result.get("common_batch_sequence_batches", 0)) + 1
    result["common_batch_sequence_rows"] = int(result.get("common_batch_sequence_rows", 0)) + int(labels.numel())
    scenario_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if set(scenario_batches) != set(FROZEN_RCAT_SCENARIOS):
        raise RCATRuntimeError("P1-RCAT common scenario receipt is malformed")
    scenario_batches[str(scenario)] += 1
    result["common_scenario_batches"] = scenario_batches
    common_cells = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("rcat_common_cells", {})).items()
    }
    scene_cells = common_cells.get(str(scenario), _cell_template(receivers))
    expected_keys = set(_cell_template(receivers))
    if set(scene_cells) != expected_keys:
        raise RCATRuntimeError("P1-RCAT common receiver/class cells are malformed")
    for key in sorted(expected_keys):
        count = int(counts[key])
        cell = dict(scene_cells[key])
        cell["rows"] = int(cell.get("rows", 0)) + count
        cell["batches"] = int(cell.get("batches", 0)) + 1
        cell["nonempty_batches"] = int(cell.get("nonempty_batches", 0)) + int(count > 0)
        scene_cells[key] = cell
    common_cells[str(scenario)] = scene_cells
    result["rcat_common_cells"] = common_cells
    batch_cells = list(result.get("rcat_common_batch_cells", []))
    batch_cells.append({key: event[key] for key in ("epoch", "batch_index", "scenario", "same_physical_clean_leo", "n_rc", "effective_weights")})
    result["rcat_common_batch_cells"] = batch_cells
    return result


def bind_rcat_optimizer_initial_state(
    receipt: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Seal the new AdamW state before the first backward call."""

    result = dict(receipt)
    optimizer_type = type(optimizer).__name__
    if optimizer_type != FROZEN_RCAT_OPTIMIZER_TYPE:
        raise RCATConfigurationError(
            "P1-RCAT requires optimizer_type=AdamW, got " + (optimizer_type or "<empty>")
        )
    state = optimizer.state_dict()
    if dict(state.get("state", {})):
        raise RCATConfigurationError("P1-RCAT requires a new AdamW state")
    groups = []
    for group in state.get("param_groups", []):
        normalized = {str(key): value for key, value in dict(group).items() if str(key) != "params"}
        normalized["parameter_count"] = len(list(dict(group).get("params", [])))
        groups.append(normalized)
    result["optimizer_initial_state_sha256"] = _canonical_sha256(
        {"optimizer_type": optimizer_type, "state_empty": True, "param_groups": groups}
    )
    result["optimizer_type"] = optimizer_type
    result["optimizer_initial_state_empty"] = True
    return result


def _accumulate_aux_cell(cells: Dict[str, Dict[str, Any]], *, key: str, info: Mapping[str, Any]) -> None:
    cell = dict(cells[key])
    count = int(info.get("n_rc", -1))
    positive = int(info.get("positive_q", -1))
    finite = int(info.get("finite_q", -1))
    clean_zero = int(info.get("clean_zero_rows", -1))
    leo_zero = int(info.get("leo_zero_rows", -1))
    union_zero = int(info.get("union_zero_rows", -1))
    both_zero = int(info.get("both_zero_rows", -1))
    nonfinite = int(info.get("nonfinite_rows", -1))
    q_sum = float(info.get("q_sum", float("nan")))
    g_value = float(info.get("g_rc", float("nan")))
    loss_value = float(info.get("loss_contribution", float("nan")))
    if (
        count < 0
        or positive < 0
        or positive > count
        or finite != count
        or min(clean_zero, leo_zero, union_zero, both_zero, nonfinite) < 0
        or union_zero != clean_zero + leo_zero - both_zero
        or nonfinite != 0
        or not all(math.isfinite(value) for value in (q_sum, g_value, loss_value))
    ):
        raise RCATRuntimeError("P1-RCAT cumulative G cell info is malformed")
    for field, value in (
        ("rows", count),
        ("positive_q", positive),
        ("finite_q", finite),
        ("clean_zero_rows", clean_zero),
        ("leo_zero_rows", leo_zero),
        ("union_zero_rows", union_zero),
        ("both_zero_rows", both_zero),
        ("nonfinite_rows", nonfinite),
    ):
        cell[field] = int(cell.get(field, 0)) + value
    for field, value in (("q_sum", q_sum), ("g_sum", g_value), ("loss_sum", loss_value)):
        cell[field] = float(cell.get(field, 0.0)) + value
    cell["batches"] = int(cell.get("batches", 0)) + 1
    cell["nonempty_batches"] = int(cell.get("nonempty_batches", 0)) + int(count > 0)
    cell["finite_batches"] = int(cell.get("finite_batches", 0)) + 1
    if (
        int(cell["positive_q"]) > int(cell["rows"])
        or int(cell["finite_q"]) != int(cell["rows"])
        or int(cell["nonfinite_rows"]) != 0
    ):
        raise RCATRuntimeError("P1-RCAT cumulative G cell counters do not close")
    cells[key] = cell


def update_rcat_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    scenario: str,
    epoch: int,
    batch_index: int,
) -> Dict[str, Any]:
    """Accumulate G-only q/loss evidence after common C/G coverage is sealed."""

    result = dict(receipt)
    if str(result.get("schema", "")) != RCAT_RECEIPT_SCHEMA:
        raise RCATRuntimeError("P1-RCAT receipt schema is invalid")
    if result.get("enabled") is not True:
        raise RCATRuntimeError("P1-RCAT auxiliary receipt update is G-arm only")
    if str(scenario) not in FROZEN_RCAT_SCENARIOS:
        raise RCATRuntimeError("P1-RCAT scenario is outside frozen clear/low/rain cycle")
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    expected_keys = set(_cell_template(receivers))
    if tuple(int(value) for value in result.get("expected_tx_class_ids", [])) != FROZEN_RCAT_CLASS_IDS:
        raise RCATRuntimeError("P1-RCAT receipt lacks local4 class binding")
    if (
        batch_info.get("finite") is not True
        or batch_info.get("clean_feature_detached") is not True
        or batch_info.get("empty_cell_zero") is not True
        or batch_info.get("no_active_renormalization") is not True
        or batch_info.get("no_cross_sample_or_cross_receiver_pairing") is not True
        or batch_info.get("training_accumulation_dtype") != "float32"
    ):
        raise RCATRuntimeError("P1-RCAT batch semantic receipt drifted")
    if int(batch_info.get("global_denominator", -1)) != FROZEN_RCAT_TERM_DIVISOR:
        raise RCATRuntimeError("P1-RCAT global denominator drifted")
    scale = float(batch_info.get("fixed_scale", float("nan")))
    if not math.isfinite(scale) or abs(scale - 1.0 / FROZEN_RCAT_TERM_DIVISOR) > _TOLERANCE:
        raise RCATRuntimeError("P1-RCAT fixed scale drifted")
    cells = {str(key): dict(value) for key, value in dict(batch_info.get("cells", {})).items()}
    if set(cells) != expected_keys:
        raise RCATRuntimeError("P1-RCAT G receipt lacks all receiver×class cells")
    total_rows = int(batch_info.get("rows", -1))
    positive_q = int(batch_info.get("positive_q", -1))
    finite_q = int(batch_info.get("finite_q", -1))
    nonfinite_rows = int(batch_info.get("nonfinite_rows", -1))
    loss_sum = float(batch_info.get("loss_sum", float("nan")))
    if (
        total_rows <= 0
        or positive_q < 0
        or positive_q > total_rows
        or finite_q != total_rows
        or nonfinite_rows != 0
        or not math.isfinite(loss_sum)
    ):
        raise RCATRuntimeError("P1-RCAT G batch rows/q/nonfinite/loss do not close")
    common_events = list(result.get("rcat_common_batch_cells", []))
    if not common_events:
        raise RCATRuntimeError("P1-RCAT G batch lacks its common C/G coverage receipt")
    common_event = dict(common_events[-1])
    if (
        int(common_event.get("epoch", -1)) != int(epoch)
        or int(common_event.get("batch_index", -1)) != int(batch_index)
        or str(common_event.get("scenario", "")) != str(scenario)
        or common_event.get("same_physical_clean_leo") is not True
    ):
        raise RCATRuntimeError("P1-RCAT G/common same-physical receipt alignment drifted")
    common_counts = {str(key): int(value) for key, value in dict(common_event.get("n_rc", {})).items()}
    if common_counts != {key: int(value.get("n_rc", -1)) for key, value in cells.items()}:
        raise RCATRuntimeError("P1-RCAT G/common n_rc receipt mismatch")
    common_weights = {
        str(key): {str(field): float(field_value) for field, field_value in dict(value).items()}
        for key, value in dict(common_event.get("effective_weights", {})).items()
    }
    for key in expected_keys:
        if (
            abs(float(cells[key].get("cell_weight", float("nan"))) - float(common_weights[key]["cell_weight"])) > _TOLERANCE
            or abs(float(cells[key].get("row_weight", float("nan"))) - float(common_weights[key]["row_weight"])) > _TOLERANCE
        ):
            raise RCATRuntimeError("P1-RCAT G/common effective weight mismatch")
    scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(scene_cells).items()}
        for scene, scene_cells in dict(result.get("rcat_scenes", {})).items()
    }
    scene_cells = scenes.get(str(scenario), _cell_template(receivers))
    if set(scene_cells) != expected_keys:
        raise RCATRuntimeError("P1-RCAT G scene cells are malformed")
    for key in sorted(expected_keys):
        _accumulate_aux_cell(scene_cells, key=key, info=cells[key])
    scenes[str(scenario)] = scene_cells
    result["rcat_scenes"] = scenes
    aux_events = list(result.get("rcat_g_batch_aux", []))
    aux_events.append(
        {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "scenario": str(scenario),
            "positive_q": positive_q,
            "loss_sum": loss_sum,
            "cell_positive_q": {key: int(cells[key]["positive_q"]) for key in sorted(expected_keys)},
            "cell_loss_sum": {key: float(cells[key]["loss_contribution"]) for key in sorted(expected_keys)},
        }
    )
    result["rcat_g_batch_aux"] = aux_events
    result["rcat_batches"] = int(result.get("rcat_batches", 0)) + 1
    result["rcat_total_rows"] = int(result.get("rcat_total_rows", 0)) + total_rows
    result["rcat_positive_q"] = int(result.get("rcat_positive_q", 0)) + positive_q
    result["rcat_loss_sum"] = float(result.get("rcat_loss_sum", 0.0)) + loss_sum
    return result


def _validate_common_terminal_contract(result: Mapping[str, Any]) -> None:
    if str(result.get("schema", "")) != RCAT_RECEIPT_SCHEMA:
        raise RCATRuntimeError("P1-RCAT terminal receipt schema is invalid")
    for key in (
        "baseline_sha256",
        "initial_checkpoint_sha256",
        "class_order_binding_sha256",
        "source_labeled_indices_sha256",
        "source_split_manifest_sha256",
        "source_receiver_ids_sha256",
        "optimizer_initial_state_sha256",
        "common_batch_sequence_sha256",
    ):
        if len(str(result.get(key, "") or "")) != 64:
            raise RCATRuntimeError(f"P1-RCAT terminal receipt lacks {key}")
    if str(result.get("checkpoint_role", "") or "") != "training_final_only":
        raise RCATRuntimeError("P1-RCAT requires training_final_only warm start")
    if result.get("optimizer_state_restored") is not False or result.get("rng_state_restored") is not False:
        raise RCATRuntimeError("P1-RCAT optimizer/RNG restoration is forbidden")
    if str(result.get("optimizer_type", "")) != FROZEN_RCAT_OPTIMIZER_TYPE:
        raise RCATRuntimeError("P1-RCAT terminal optimizer_type must be AdamW")
    if result.get("optimizer_initial_state_empty") is not True:
        raise RCATRuntimeError("P1-RCAT missing new AdamW initial-state receipt")
    if result.get("amp_contract") != "COMMON_TRAINER_AMP_ENABLED":
        raise RCATRuntimeError("P1-RCAT terminal AMP contract drifted")
    if result.get("common_l_base_head_input_path_verified") is not True:
        raise RCATRuntimeError("P1-RCAT common L_base exact head-input path is not verified")
    batches = int(result.get("common_batch_sequence_batches", 0))
    rows = int(result.get("common_batch_sequence_rows", 0))
    scenario_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if (
        batches <= 0
        or rows <= 0
        or set(scenario_batches) != set(FROZEN_RCAT_SCENARIOS)
        or any(value <= 0 for value in scenario_batches.values())
    ):
        raise RCATRuntimeError("P1-RCAT common batch/scenario receipt is incomplete")


def _validate_common_cells(result: Mapping[str, Any]) -> None:
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    expected_keys = set(_cell_template(receivers))
    common_scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("rcat_common_cells", {})).items()
    }
    if set(common_scenes) != set(FROZEN_RCAT_SCENARIOS):
        raise RCATRuntimeError("P1-RCAT terminal common 84-cell coverage is incomplete")
    common_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    observed_rows = 0
    for scenario in FROZEN_RCAT_SCENARIOS:
        cells = common_scenes[scenario]
        if set(cells) != expected_keys:
            raise RCATRuntimeError("P1-RCAT terminal common receiver/class cells drifted")
        for key in expected_keys:
            cell = cells[key]
            if (
                int(cell.get("rows", 0)) <= 0
                or int(cell.get("batches", 0)) != common_batches[scenario]
                or int(cell.get("nonempty_batches", 0)) <= 0
            ):
                raise RCATRuntimeError("P1-RCAT terminal common 84-cell receipt has an uncovered cell")
            observed_rows += int(cell.get("rows", 0))
    if observed_rows != int(result.get("common_batch_sequence_rows", 0)):
        raise RCATRuntimeError("P1-RCAT terminal common 84-cell row total does not close")
    events = list(result.get("rcat_common_batch_cells", []))
    if len(events) != int(result.get("common_batch_sequence_batches", 0)):
        raise RCATRuntimeError("P1-RCAT terminal common per-batch cell receipt is incomplete")
    event_rows = 0
    for event in events:
        if event.get("same_physical_clean_leo") is not True:
            raise RCATRuntimeError("P1-RCAT terminal same-physical receipt drifted")
        counts = {str(key): int(value) for key, value in dict(event.get("n_rc", {})).items()}
        if set(counts) != expected_keys or any(value < 0 for value in counts.values()):
            raise RCATRuntimeError("P1-RCAT terminal per-batch n_rc is malformed")
        weights = {
            str(key): {str(field): float(field_value) for field, field_value in dict(value).items()}
            for key, value in dict(event.get("effective_weights", {})).items()
        }
        expected_weights = _batch_cell_weights(receivers, counts)
        if set(weights) != expected_keys:
            raise RCATRuntimeError("P1-RCAT terminal per-batch effective weights are incomplete")
        for key in expected_keys:
            for field in ("cell_weight", "row_weight"):
                actual = float(weights[key].get(field, float("nan")))
                expected = float(expected_weights[key][field])
                if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
                    raise RCATRuntimeError("P1-RCAT terminal fixed effective weight drifted")
        event_rows += sum(counts.values())
    if event_rows != int(result.get("common_batch_sequence_rows", 0)):
        raise RCATRuntimeError("P1-RCAT terminal per-batch n_rc rows do not close")


def validate_rcat_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless common 84 cells and G-only q/VJP evidence close."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    _validate_common_terminal_contract(result)
    _validate_common_cells(result)
    enabled = result.get("enabled")
    if enabled is not True and enabled is not False:
        raise RCATRuntimeError("P1-RCAT terminal enabled flag must be strict bool")
    if enabled is False:
        forbidden_nonzero = ("rcat_batches", "rcat_total_rows", "rcat_positive_q")
        if any(int(result.get(key, 0)) != 0 for key in forbidden_nonzero) or abs(
            float(result.get("rcat_loss_sum", 0.0))
        ) > _TOLERANCE:
            raise RCATRuntimeError("P1-RCAT C arm must retain zero auxiliary counters")
        if any(bool(result.get(key)) for key in ("rcat_scenes", "rcat_g_batch_aux", "rcat_gradient_audit")):
            raise RCATRuntimeError("P1-RCAT C arm must retain N/A-or-zero auxiliary fields")
        if bool(result.get("rcat_gradient_audit_attempted", False)) or bool(
            result.get("rcat_gradient_audit_completed", False)
        ):
            raise RCATRuntimeError("P1-RCAT C arm may not run an auxiliary VJP audit")
        result["rcat_terminal_contract"] = "CONTROL_ARM_COMMON_SAME_PHYSICAL_RX_CLASS_SCENE_84_CELL_COVERAGE_AUX_NA_OR_ZERO"
        result["rcat_terminal_contract_passed"] = True
        return result
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    expected_keys = set(_cell_template(receivers))
    common_scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("rcat_common_cells", {})).items()
    }
    scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("rcat_scenes", {})).items()
    }
    if set(scenes) != set(FROZEN_RCAT_SCENARIOS):
        raise RCATRuntimeError("P1-RCAT terminal G 84-cell coverage is incomplete")
    total_rows = total_positive = 0
    total_loss = 0.0
    for scenario in FROZEN_RCAT_SCENARIOS:
        cells = scenes[scenario]
        if set(cells) != expected_keys:
            raise RCATRuntimeError("P1-RCAT terminal G receiver/class cells drifted")
        for key in expected_keys:
            cell = cells[key]
            common = common_scenes[scenario][key]
            rows = int(cell.get("rows", -1))
            positive = int(cell.get("positive_q", -1))
            finite = int(cell.get("finite_q", -1))
            nonfinite = int(cell.get("nonfinite_rows", -1))
            batches = int(cell.get("batches", -1))
            finite_batches = int(cell.get("finite_batches", -1))
            nonempty = int(cell.get("nonempty_batches", -1))
            loss = float(cell.get("loss_sum", float("nan")))
            if (
                rows <= 0
                or rows != int(common.get("rows", -2))
                or positive < 0
                or positive > rows
                or finite != rows
                or nonfinite != 0
                or batches != int(common.get("batches", -2))
                or finite_batches != batches
                or nonempty != int(common.get("nonempty_batches", -2))
                or not math.isfinite(loss)
            ):
                raise RCATRuntimeError("P1-RCAT terminal G r×c×scene receipt does not close")
            total_rows += rows
            total_positive += positive
            total_loss += loss
    common_rows = int(result.get("common_batch_sequence_rows", 0))
    if (
        int(result.get("rcat_batches", -1)) != int(result.get("common_batch_sequence_batches", -2))
        or int(result.get("rcat_total_rows", -1)) != common_rows
        or total_rows != common_rows
        or int(result.get("rcat_positive_q", -1)) != total_positive
        or total_positive <= 0
        or not _float32_ledger_close(float(result.get("rcat_loss_sum", float("nan"))), total_loss)
    ):
        raise RCATRuntimeError("P1-RCAT terminal G batch/positive/loss counters do not close")
    events = list(result.get("rcat_g_batch_aux", []))
    if len(events) != int(result.get("rcat_batches", 0)):
        raise RCATRuntimeError("P1-RCAT terminal G per-batch auxiliary receipt is incomplete")
    if sum(int(event.get("positive_q", -1)) for event in events) != total_positive:
        raise RCATRuntimeError("P1-RCAT terminal G per-batch positive_q does not close")
    if not bool(result.get("rcat_gradient_audit_completed", False)):
        raise RCATRuntimeError("P1-RCAT terminal first-positive auxiliary VJP audit is incomplete")
    result["rcat_terminal_contract"] = "FORMAL_COMMON_SAME_PHYSICAL_RX_CLASS_SCENE_84_CELL_FIXED_SCALE_WITH_G_ONLY_TOTALIZED_L2_Q_AND_FIRST_feat_joint_SHARED_ENCODER_VJP_EXACT_HEAD_AUX_NA"
    result["rcat_terminal_contract_passed"] = True
    return result


def _failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "vjp" in message or "gradient" in message or "head" in message:
        return "RCAT_AUX_GRADIENT_PATH_FAILURE"
    if "non-finite" in message or "nonfinite" in message:
        return "RCAT_NONFINITE"
    if "receiver" in message or "rx_i" in message or "84-cell" in message or "r×c" in message:
        return "RCAT_SOURCE_RX_OR_CELL_COVERAGE_FAILURE"
    if "sequence" in message or "receipt" in message or "coverage" in message:
        return "RCAT_RECEIPT_CLOSURE_FAILURE"
    if "binding" in message or "feat_joint" in message or "totalized" in message:
        return "RCAT_BINDING_FAILURE"
    return "RCAT_RUNTIME_FAILURE"


def write_rcat_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free fail-closed record for the RCAT arm."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise RCATRuntimeError(f"P1-RCAT failure receipt output directory is absent: {target_dir}")
    payload = {
        "schema": "cvs.phase1.rcat_failure_receipt.v1",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage or ""),
        "exception_type": type(error).__name__,
        "exception_fingerprint": _failure_fingerprint(error),
        "message": str(error),
        "receipt": dict(receipt),
    }
    target = target_dir / "phase1_rcat_failure_receipt.json"
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = mkstemp(prefix=".rcat_failure_receipt.", suffix=".tmp", dir=str(target_dir))
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


def strict_rcat_warm_start(
    model: torch.nn.Module,
    checkpoint_model_state: Mapping[str, torch.Tensor],
    *,
    baseline_path: str,
    baseline_sha256: str,
    checkpoint_epoch: int,
    checkpoint_role: str,
) -> Dict[str, Any]:
    """Load strict model weights only; optimizer and RNG remain new."""

    path = str(baseline_path or "").strip()
    digest = str(baseline_sha256 or "").strip()
    if not path or len(digest) != 64 or not isinstance(checkpoint_model_state, Mapping):
        raise RCATConfigurationError(
            "Frozen P1-RCAT warm-start requires model state, path, and SHA256"
        )
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise RCATConfigurationError(
            f"Frozen P1-RCAT strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RCATConfigurationError(
            "Frozen P1-RCAT strict baseline model-key mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    if str(checkpoint_role or "") != "training_final_only":
        raise RCATConfigurationError(
            "Frozen P1-RCAT requires baseline checkpoint_role=training_final_only"
        )
    return {
        "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP",
        "baseline_path": path,
        "baseline_sha256": digest,
        "initial_checkpoint_sha256": digest,
        "checkpoint_epoch": epoch,
        "checkpoint_role": "training_final_only",
        "strict_model_keys": True,
        "missing_model_keys": [],
        "unexpected_model_keys": [],
        "optimizer_state_restored": False,
        "rng_state_restored": False,
    }
