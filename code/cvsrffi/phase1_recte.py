"""Frozen P1-RECTE continuation contract for Phase1 source-only DG.

P1-RECTE (Receiver-Equivariant Cell-Tail Equalization) keeps the common
F1C/GeoSat-C clean and single-LEO forwards unchanged.  Its G arm reads only
source-known-train L labels and physical receiver IDs, then compares the
relative clean-to-LEO margin drops of the fixed 7 x 4 receiver/class cells.
The clean raw logits and the exact current head parameters are stop-gradient
inside the auxiliary term.  Therefore RECTE can update the existing LEO
feat_joint/shared encoder path but cannot obtain an auxiliary gradient through
the classifier head.  It introduces no target/proxy/held/U/V feedback,
resampling, model state, cache, threshold, or extra model forward.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from torch.func import functional_call as _functional_call
except (ImportError, AttributeError):
    _functional_call = None


FROZEN_RECTE_LAMBDA = 0.02
FROZEN_RECTE_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_RECTE_CLASS_IDS = (0, 1, 2, 3)
FROZEN_RECTE_SOURCE_RECEIVER_IDS = (0, 1, 2, 3, 4, 5, 6)
FROZEN_RECTE_SOURCE_RECEIVER_COUNT = len(FROZEN_RECTE_SOURCE_RECEIVER_IDS)
FROZEN_RECTE_CELL_COUNT = len(FROZEN_RECTE_CLASS_IDS) * len(
    FROZEN_RECTE_SOURCE_RECEIVER_IDS
)
FROZEN_RECTE_PAIR_DENOMINATOR = FROZEN_RECTE_CELL_COUNT * (
    FROZEN_RECTE_CELL_COUNT - 1
) // 2
FROZEN_RECTE_UNORDERED_PAIR_DENOMINATOR = FROZEN_RECTE_PAIR_DENOMINATOR
FROZEN_RECTE_TERM_DIVISOR = FROZEN_RECTE_PAIR_DENOMINATOR
FROZEN_RECTE_OPTIMIZER_TYPE = "AdamW"
FROZEN_RECTE_FLOAT32_LEDGER_REL_TOL = 32.0 * float(torch.finfo(torch.float32).eps)
RECTE_RECEIPT_SCHEMA = "cvs.phase1.recte_receipt.v1"
_TOLERANCE = 1e-12


class RECTEConfigurationError(ValueError):
    """Raised when a frozen P1-RECTE C/G configuration drifts."""


class RECTERuntimeError(RuntimeError):
    """Raised when a P1-RECTE runtime or receipt contract cannot be proved."""


@dataclass(frozen=True)
class RECTEConfig:
    """Immutable P1-RECTE controls consumed by the common training loop."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise RECTEConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
        raise RECTEConfigurationError(
            f"Frozen P1-RECTE requires {name}={expected:.12g}, got {actual!r}"
        )


def _float32_ledger_close(actual: float, expected: float) -> bool:
    return (
        math.isfinite(actual)
        and math.isfinite(expected)
        and abs(actual - expected)
        <= FROZEN_RECTE_FLOAT32_LEDGER_REL_TOL * max(1.0, abs(actual), abs(expected))
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
        raise RECTEConfigurationError(
            "Frozen P1-RECTE forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    scenarios = tuple(
        part.strip().lower().replace("-", "_")
        for part in str(value or "").split(",")
        if part.strip()
    )
    if scenarios != FROZEN_RECTE_SCENARIOS:
        raise RECTEConfigurationError(
            "Frozen P1-RECTE requires --sat_train_scenarios "
            + ",".join(FROZEN_RECTE_SCENARIOS)
        )
    return scenarios


def validate_recte_args(args: Any) -> RECTEConfig:
    """Validate the frozen common-base C/G contract before data are loaded."""

    frozen_mode = _bool_arg(args, "phase1_recte_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_recte_enabled", False)
    loss_weight = _float_arg(args, "lambda_recte", 0.0)
    if not frozen_mode and not enabled:
        return RECTEConfig(False, False, 0.0)
    if enabled and not frozen_mode:
        raise RECTEConfigurationError(
            "--phase1_recte_enabled requires --phase1_recte_frozen_mode true"
        )
    _require_close(
        "lambda_recte", loss_weight, FROZEN_RECTE_LAMBDA if enabled else 0.0
    )
    if bool(getattr(args, "from_scratch", True)):
        raise RECTEConfigurationError("Frozen P1-RECTE requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise RECTEConfigurationError("Frozen P1-RECTE requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise RECTEConfigurationError("Frozen P1-RECTE must train the shared feat_joint encoder")
    if not bool(getattr(args, "amp", True)):
        raise RECTEConfigurationError("Frozen P1-RECTE requires the common AMP training path")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise RECTEConfigurationError("Frozen P1-RECTE requires --id_feature_key feat_joint")
    if int(getattr(args, "epochs", 0)) != 40 or int(getattr(args, "label_epochs", 0)) != 40:
        raise RECTEConfigurationError("Frozen P1-RECTE requires exactly 40 labeled epochs")
    if int(getattr(args, "pseudo_epochs", 0)) != 0:
        raise RECTEConfigurationError("Frozen P1-RECTE forbids pseudo epochs")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise RECTEConfigurationError("Frozen P1-RECTE requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise RECTEConfigurationError("Frozen P1-RECTE remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise RECTEConfigurationError("Frozen P1-RECTE requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise RECTEConfigurationError("Frozen P1-RECTE requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise RECTEConfigurationError("Frozen P1-RECTE forbids --sat_view_schedule overrides")
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise RECTEConfigurationError("Frozen P1-RECTE requires non-concatenated single-LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise RECTEConfigurationError("Frozen P1-RECTE permits only source_known_train L updates")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise RECTEConfigurationError("Frozen P1-RECTE forbids RX/day-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise RECTEConfigurationError("Frozen P1-RECTE permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise RECTEConfigurationError("Frozen P1-RECTE forbids a reject head")
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
            "phase1_rcat_frozen_mode",
            "phase1_rcat_enabled",
            "lambda_rcat",
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
    return RECTEConfig(True, enabled, loss_weight)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _receiver_key(receiver_id: int, class_id: int) -> str:
    return f"rx{int(receiver_id)}|tx{int(class_id)}"


def _source_receiver_ids(values: Any) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RECTEConfigurationError("P1-RECTE source_receivers must be a source-only sequence")
    parsed = []
    for value in values:
        text = str(value).strip()
        if not text:
            raise RECTEConfigurationError("P1-RECTE source receiver id may not be empty")
        try:
            receiver_id = int(text)
        except (TypeError, ValueError) as exc:
            raise RECTEConfigurationError("P1-RECTE source receiver id must be an integer") from exc
        if str(receiver_id) != text:
            raise RECTEConfigurationError("P1-RECTE source receiver id is not canonical")
        parsed.append(receiver_id)
    canonical = tuple(sorted(parsed))
    if not canonical or len(canonical) != len(set(canonical)):
        raise RECTEConfigurationError("P1-RECTE source receiver ids must be non-empty and unique")
    return canonical


def _require_frozen_source_receivers(receivers: Sequence[Any]) -> Tuple[int, ...]:
    parsed = _source_receiver_ids(receivers)
    if parsed != FROZEN_RECTE_SOURCE_RECEIVER_IDS:
        raise RECTEConfigurationError(
            "P1-RECTE requires frozen F1C source receivers 0..6; got "
            + str(list(parsed))
        )
    return parsed


def _normalized_tx_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RECTEConfigurationError(f"P1-RECTE {name} must be a TX class sequence")
    order = tuple(str(value).strip() for value in values)
    if not order or len(order) != len(set(order)) or any(not value for value in order):
        raise RECTEConfigurationError(f"P1-RECTE {name} must be non-empty and unique")
    return order


def _positive_count(name: str, value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise RECTEConfigurationError(f"P1-RECTE {name} must be an integer") from exc
    if count <= 0:
        raise RECTEConfigurationError(f"P1-RECTE {name} must be positive")
    return count


def recte_config_receipt(config: RECTEConfig) -> Dict[str, Any]:
    """Create the data-free receipt skeleton for either frozen C/G arm."""

    return {
        "schema": RECTE_RECEIPT_SCHEMA,
        "method": "P1_RECTE",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "loss_rule": "SOURCE_L_RX_BY_LOCAL4_FIXED_28_CELL_UNORDERED_OCCUPIED_PAIR_TAIL_ONLY_RELATIVE_MARGIN_EQUALIZATION",
        "loss_formula": "m=t_y-logsumexp(t_not_y);delta_rc=mean_Irc(m_leo)-mean_Irc(sg(m_clean));q_ab=[sg(delta_b)-delta_a]_+^2+[sg(delta_a)-delta_b]_+^2;L=sum_{a<b}A_aA_b*q_ab/378",
        "loss_global_denominator": FROZEN_RECTE_PAIR_DENOMINATOR,
        "fixed_unordered_pair_count": FROZEN_RECTE_PAIR_DENOMINATOR,
        "frozen_cells_per_scene": FROZEN_RECTE_CELL_COUNT,
        "local_class_ids": list(FROZEN_RECTE_CLASS_IDS),
        "frozen_source_receiver_ids": list(FROZEN_RECTE_SOURCE_RECEIVER_IDS),
        "z_id_key": "feat_joint",
        "feature_dimension_contract": "RAW_ENCODER_feat_joint_EXACT_HEAD_INPUT_DIMENSION_BOUND",
        "pair_orientation": "LEXICOGRAPHIC_UNORDERED_a_lt_b;ONLY_LOWER_DELTA_GETS_TAIL_GRADIENT",
        "pair_permutation_equivariance": "RX_AND_LOCAL_CLASS_PERMUTATION_EQUIVARIANT_NO_ID_SPECIFIC_WEIGHT",
        "clean_raw_logits_detached": True,
        "functional_exact_head_readout": "torch.func.functional_call(current_exact_head,DETACHED_CURRENT_PARAMETERS_AND_CLONED_DETACHED_BUFFERS,(feat_joint_leo,),labels=source_L_y_if_exact_head_accepts_labels)",
        "functional_head_parameters_stopgrad": True,
        "functional_head_readout_resource": "ONE_EXACT_HEAD_ONLY_FUNCTIONAL_READOUT_PER_G_BATCH_O(HEAD)_NOT_EXTRA_MODEL_OR_CLEAN_OR_LEO_FORWARD",
        "functional_logits_live_equality_required": True,
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "common_l_base_head_input_path_verified": False,
        "aux_gradient_scope": "LEO_feat_joint_AND_SHARED_ENCODER_FINITE_NONZERO;EXACT_HEAD_AUX_VJP_NA_NONE_OR_ZERO",
        "uses_new_forward": False,
        "uses_resampling": False,
        "uses_rx_labels": True,
        "rx_permission": "SOURCE_KNOWN_TRAIN_L_PHYSICAL_ID_BOUND_rx_i_ONLY",
        "rx_metadata_allowlist": ["rx_i"],
        "zero_feature_rows_preserved": True,
        "empty_pair_zero_contribution": True,
        "no_active_pair_renormalization": True,
        "no_day_assertion": "day_i_NOT_READ_BY_RECTE",
        "uses_day_labels": False,
        "uses_fold_labels": False,
        "uses_domain_labels": False,
        "uses_target_rows": False,
        "uses_proxy_rows": False,
        "uses_held_rows": False,
        "uses_unlabeled_rows": False,
        "u_loader_common_trainer_boundary": "MAY_BE_CONSTRUCTED_BY_COMMON_TRAINER_BUT_RECTE_ZERO_ITERATE_ZERO_FORWARD_ZERO_LOSS_ZERO_BACKWARD_ZERO_OPTIMIZER",
        "v_common_trainer_boundary": "COMMON_READ_ONLY_DIAGNOSTIC_ONLY_RECTE_ZERO_LOSS_ZERO_BACKWARD_ZERO_OPTIMIZER_ZERO_CALIBRATION_ZERO_MODEL_SELECTION_FEEDBACK",
        "uses_ema_or_state": False,
        "uses_threshold": False,
        "uses_gradient_projection": False,
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
        "common_scenario_batches": {scene: 0 for scene in FROZEN_RECTE_SCENARIOS},
        "recte_common_cells": {},
        "recte_common_batch_cells": [],
        "recte_batches": 0,
        "recte_total_rows": 0,
        "recte_occupied_unordered_pair_count": 0,
        "recte_positive_tail_pair_count": 0,
        "recte_loss_sum": 0.0,
        "recte_functional_head_readout_count": 0,
        "recte_float32_ledger_rel_tolerance": FROZEN_RECTE_FLOAT32_LEDGER_REL_TOL,
        "recte_scenes": {},
        "recte_g_batch_aux": [],
        "recte_gradient_audit_attempted": False,
        "recte_gradient_audit_completed": False,
        "recte_gradient_audit_scenes": {},
        "recte_terminal_contract": "PENDING",
        "recte_terminal_contract_passed": False,
        "proxy_rows": 0,
        "held_rows": 0,
    }


def resolve_recte_local_head_class_binding(
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
    checkpoint_count = _positive_count(
        "checkpoint classifier head row count", checkpoint_head_class_count
    )
    live_count = _positive_count("live classifier head row count", live_head_class_count)
    if local_count != 4 or len(local) != 4:
        raise RECTEConfigurationError("P1-RECTE requires exactly four local source-TX rows")
    if local != source or checkpoint != source:
        raise RECTEConfigurationError(
            "P1-RECTE local/checkpoint TX order must equal the source-train receipt"
        )
    if local_count != len(local) or checkpoint_count != live_count or live_count != local_count:
        raise RECTEConfigurationError("P1-RECTE local/head class counts must match")
    if set(local).difference(dataset):
        raise RECTEConfigurationError("P1-RECTE local TX labels are absent from dataset order")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(FROZEN_RECTE_CLASS_IDS),
        "expected_tx_class_ids": list(FROZEN_RECTE_CLASS_IDS),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = _canonical_sha256(binding)
    return binding


def remap_recte_local_labels_to_head_rows(
    local_labels: torch.Tensor, local_to_head_class_ids: Sequence[Any]
) -> torch.Tensor:
    """Map contiguous local source labels through the sealed identity mapping."""

    if not torch.is_tensor(local_labels):
        raise RECTERuntimeError("P1-RECTE local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if mapping != FROZEN_RECTE_CLASS_IDS:
        raise RECTERuntimeError("P1-RECTE local-to-head mapping must be local4 identity")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RECTERuntimeError("P1-RECTE local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_recte_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    """Resolve the exact common classifier head used by the live base loss."""

    raw_model = getattr(model, "_orig_mod", model)
    try:
        head = raw_model.id_backbone.cls_head.head
    except AttributeError as exc:
        raise RECTERuntimeError("P1-RECTE requires model.id_backbone.cls_head.head") from exc
    if not isinstance(head, torch.nn.Module):
        raise RECTERuntimeError("P1-RECTE exact classifier head must be a torch module")
    if not isinstance(head.weight, torch.nn.Parameter) or head.weight.ndim != 2:
        raise RECTERuntimeError("P1-RECTE classifier head weight must be a rank-2 Parameter")
    if not bool(head.weight.requires_grad):
        raise RECTERuntimeError("P1-RECTE exact classifier head weight must be live for common L_base")
    if not tuple(parameter for parameter in head.parameters() if parameter.requires_grad):
        raise RECTERuntimeError("P1-RECTE exact classifier head has no live trainable parameter")
    return head


def resolve_recte_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    return resolve_recte_classifier_head(model).weight


def _validated_receiver_labels(
    receiver_labels: torch.Tensor, *, rows: int, expected_receiver_ids: Sequence[Any]
) -> torch.Tensor:
    if not torch.is_tensor(receiver_labels):
        raise RECTERuntimeError("P1-RECTE requires source-L physical rx_i labels")
    values = receiver_labels.reshape(-1).long()
    expected = _require_frozen_source_receivers(expected_receiver_ids)
    if values.numel() != int(rows) or values.numel() == 0:
        raise RECTERuntimeError("P1-RECTE source-L rx_i rows do not align")
    observed = {int(value) for value in values.detach().cpu().tolist()}
    if observed.difference(set(expected)):
        raise RECTERuntimeError("P1-RECTE rx_i contains a receiver outside frozen source R_s")
    return values


def _validate_view_binding(
    *,
    view_name: str,
    output: Mapping[str, Any],
    labels: torch.Tensor,
    head: torch.nn.Module,
) -> None:
    if str(output.get("z_id_key", "")) != "feat_joint":
        raise RECTERuntimeError(f"P1-RECTE {view_name} z_id_key must be feat_joint")
    z_id = output.get("z_id")
    logits = output.get("tx_logits")
    if not torch.is_tensor(z_id) or z_id.ndim != 2:
        raise RECTERuntimeError(f"P1-RECTE {view_name} z_id must be rank-2 feat_joint")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise RECTERuntimeError(f"P1-RECTE {view_name} tx_logits must be rank-2 raw logits")
    if z_id.size(0) != labels.numel() or logits.size(0) != labels.numel():
        raise RECTERuntimeError(f"P1-RECTE {view_name} rows must align with source L labels")
    if int(head.weight.size(0)) != 4 or int(logits.size(1)) != 4:
        raise RECTERuntimeError(f"P1-RECTE {view_name} head/logit class rows must be local4")
    if int(head.weight.size(1)) != int(z_id.size(1)):
        raise RECTERuntimeError(f"P1-RECTE {view_name} feat_joint/head dimension binding drifted")
    if not bool(z_id.requires_grad) or not bool(logits.requires_grad):
        raise RECTERuntimeError(f"P1-RECTE {view_name} requires a live common feat_joint/head path")
    if not bool(torch.isfinite(z_id.detach()).all().item()):
        raise RECTERuntimeError(f"P1-RECTE {view_name} feat_joint is non-finite")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise RECTERuntimeError(f"P1-RECTE {view_name} raw logits are non-finite")


def validate_recte_binding(
    *,
    model: torch.nn.Module,
    out_clean: Mapping[str, Any],
    out_leo: Mapping[str, Any],
    tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
    expected_receiver_ids: Sequence[Any],
) -> torch.nn.Module:
    """Fail closed unless common forwards expose the live exact-head path."""

    if not isinstance(out_clean, Mapping) or not isinstance(out_leo, Mapping):
        raise RECTERuntimeError("P1-RECTE requires clean and LEO mapping outputs")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RECTERuntimeError("P1-RECTE source labels must bind to local4 head rows")
    if tuple(int(value) for value in expected_class_ids) != FROZEN_RECTE_CLASS_IDS:
        raise RECTERuntimeError("P1-RECTE expected local4 class order is invalid")
    _validated_receiver_labels(
        source_rx_labels, rows=int(labels.numel()), expected_receiver_ids=expected_receiver_ids
    )
    head = resolve_recte_classifier_head(model)
    if not bool(torch.isfinite(head.weight.detach()).all().item()):
        raise RECTERuntimeError("P1-RECTE exact classifier head weight is non-finite")
    for name, parameter in head.named_parameters():
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise RECTERuntimeError(f"P1-RECTE exact classifier head parameter is non-finite: {name}")
    for name, buffer in head.named_buffers():
        if not bool(torch.isfinite(buffer.detach()).all().item()):
            raise RECTERuntimeError(f"P1-RECTE exact classifier head buffer is non-finite: {name}")
    _validate_view_binding(view_name="clean", output=out_clean, labels=labels, head=head)
    _validate_view_binding(view_name="leo", output=out_leo, labels=labels, head=head)
    return head


def _raw_local4_margins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    view_name: str,
    require_live_path: bool,
) -> torch.Tensor:
    if not torch.is_tensor(logits) or logits.ndim != 2 or int(logits.size(1)) != 4:
        raise RECTERuntimeError(f"P1-RECTE {view_name} requires local4 raw logits")
    if logits.size(0) != labels.numel() or labels.numel() == 0:
        raise RECTERuntimeError(f"P1-RECTE {view_name} logits and labels do not align")
    if require_live_path and not bool(logits.requires_grad):
        raise RECTERuntimeError(f"P1-RECTE {view_name} logits must retain a live gradient path")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise RECTERuntimeError(f"P1-RECTE {view_name} logits are non-finite")
    stable_logits = logits.float()
    rows = torch.arange(labels.numel(), device=labels.device)
    true_logits = stable_logits[rows, labels]
    true_mask = F.one_hot(labels, num_classes=4).to(dtype=torch.bool)
    margins = true_logits - torch.logsumexp(
        stable_logits.masked_fill(true_mask, float("-inf")), dim=1
    )
    if not bool(torch.isfinite(margins.detach()).all().item()):
        raise RECTERuntimeError(f"P1-RECTE {view_name} raw local4 margins are non-finite")
    return margins


def _cell_order(receiver_ids: Sequence[Any]) -> Tuple[str, ...]:
    receivers = _require_frozen_source_receivers(receiver_ids)
    return tuple(
        _receiver_key(receiver_id, class_id)
        for receiver_id in receivers
        for class_id in FROZEN_RECTE_CLASS_IDS
    )


def _cell_template(receiver_ids: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    return {
        key: {"rows": 0, "batches": 0, "nonempty_batches": 0}
        for key in _cell_order(receiver_ids)
    }


def _functional_exact_head_logits(
    leo_feat_joint: torch.Tensor,
    live_leo_tx_logits: torch.Tensor,
    source_tx_labels: torch.Tensor,
    *,
    classifier_head: torch.nn.Module,
) -> torch.Tensor:
    """Run the exact current head functionally with detached state only.

    This is deliberately not a linear shortcut: production GeoSat-C uses a
    CosFace head whose normalization, scale, and label-margin behavior are all
    part of the live readout.  Parameters and registered buffers are copied
    into a detached functional state, so gradients can reach only z_L and the
    call cannot mutate live head state.  A known CosFace cache is additionally
    checked for accidental writes.
    """

    if _functional_call is None:
        raise RECTERuntimeError("P1-RECTE requires torch.func.functional_call")
    if not torch.is_tensor(leo_feat_joint) or leo_feat_joint.ndim != 2:
        raise RECTERuntimeError("P1-RECTE LEO feat_joint must be rank-2")
    if not torch.is_tensor(live_leo_tx_logits) or live_leo_tx_logits.ndim != 2:
        raise RECTERuntimeError("P1-RECTE live LEO logits must be rank-2")
    if not bool(leo_feat_joint.requires_grad):
        raise RECTERuntimeError("P1-RECTE LEO feat_joint must retain a live gradient path")
    if not bool(torch.isfinite(leo_feat_joint.detach()).all().item()):
        raise RECTERuntimeError("P1-RECTE LEO feat_joint is non-finite")
    if leo_feat_joint.size(0) != live_leo_tx_logits.size(0):
        raise RECTERuntimeError("P1-RECTE LEO feat_joint/live logits row binding drifted")
    if not isinstance(classifier_head, torch.nn.Module):
        raise RECTERuntimeError("P1-RECTE exact head must be a module for functional readout")
    weight = getattr(classifier_head, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise RECTERuntimeError("P1-RECTE exact head functional weight must be rank-2")
    if int(weight.size(0)) != 4 or int(weight.size(1)) != int(leo_feat_joint.size(1)):
        raise RECTERuntimeError("P1-RECTE current exact head/feat_joint binding drifted")
    labels = source_tx_labels.reshape(-1).long()
    if labels.numel() != leo_feat_joint.size(0):
        raise RECTERuntimeError("P1-RECTE functional exact-head labels do not align with LEO rows")
    functional_state: Dict[str, torch.Tensor] = {
        name: parameter.detach().clone()
        for name, parameter in classifier_head.named_parameters()
    }
    functional_state.update(
        {
            name: buffer.detach().clone()
            for name, buffer in classifier_head.named_buffers()
        }
    )
    if not functional_state:
        raise RECTERuntimeError("P1-RECTE exact head functional state is empty")
    try:
        signature = inspect.signature(classifier_head.forward)
    except (TypeError, ValueError) as exc:
        raise RECTERuntimeError("P1-RECTE exact head forward signature is unavailable") from exc
    parameters = signature.parameters
    labels_parameter = parameters.get("labels")
    accepts_labels = (
        labels_parameter is not None
        and labels_parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ) or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = {"labels": labels} if accepts_labels else {}
    training_before = bool(classifier_head.training)
    cache_key_before = getattr(classifier_head, "_norm_weight_cache_key", None)
    cache_before = getattr(classifier_head, "_norm_weight_cache", None)
    try:
        functional_logits = _functional_call(
            classifier_head,
            functional_state,
            (leo_feat_joint,),
            kwargs=kwargs,
            strict=True,
        )
    except Exception as exc:
        raise RECTERuntimeError(
            "P1-RECTE exact-head functional readout failed without a fallback path"
        ) from exc
    if bool(classifier_head.training) != training_before:
        raise RECTERuntimeError("P1-RECTE functional head call changed training mode")
    if (
        getattr(classifier_head, "_norm_weight_cache_key", None) != cache_key_before
        or getattr(classifier_head, "_norm_weight_cache", None) is not cache_before
    ):
        raise RECTERuntimeError("P1-RECTE functional exact-head readout wrote live cache/state")
    if not torch.is_tensor(functional_logits) or functional_logits.ndim != 2:
        raise RECTERuntimeError("P1-RECTE functional exact head did not return rank-2 logits")
    if not bool(functional_logits.requires_grad):
        raise RECTERuntimeError("P1-RECTE frozen-head readout detached feat_joint")
    if not bool(torch.isfinite(functional_logits.detach()).all().item()):
        raise RECTERuntimeError("P1-RECTE functional LEO logits are non-finite")
    if not torch.equal(functional_logits.detach(), live_leo_tx_logits.detach()):
        raise RECTERuntimeError(
            "P1-RECTE functional current-head logits are not numerically equal to live LEO logits"
        )
    return functional_logits


def recte_loss(
    clean_tx_logits: torch.Tensor,
    leo_feat_joint: torch.Tensor,
    live_leo_tx_logits: torch.Tensor,
    source_tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    source_receiver_ids: Sequence[Any],
    *,
    classifier_head: torch.nn.Module,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute fixed-378 unordered occupied-cell tail equalization.

    Each fixed receiver/class cell supplies its mean clean-to-LEO margin drop
    if occupied.  Only occupied unordered pairs contribute, empty pairs are
    exactly zero, and the denominator remains 378 regardless of occupancy.
    Within a non-tied pair only the lower delta receives auxiliary gradient.
    """

    labels = source_tx_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RECTERuntimeError("P1-RECTE source labels are outside local4")
    receivers = _require_frozen_source_receivers(source_receiver_ids)
    rx_labels = _validated_receiver_labels(
        source_rx_labels, rows=int(labels.numel()), expected_receiver_ids=receivers
    )
    if not torch.is_tensor(clean_tx_logits) or clean_tx_logits.ndim != 2:
        raise RECTERuntimeError("P1-RECTE clean logits must be rank-2")
    if clean_tx_logits.size(0) != labels.numel():
        raise RECTERuntimeError("P1-RECTE clean logits/labels rows do not align")
    clean_margins = _raw_local4_margins(
        clean_tx_logits.detach(),
        labels,
        view_name="clean",
        require_live_path=False,
    )
    functional_leo_logits = _functional_exact_head_logits(
        leo_feat_joint,
        live_leo_tx_logits,
        labels,
        classifier_head=classifier_head,
    )
    leo_margins = _raw_local4_margins(
        functional_leo_logits,
        labels,
        view_name="functional_leo",
        require_live_path=True,
    )
    if clean_margins.shape != leo_margins.shape:
        raise RECTERuntimeError("P1-RECTE clean/LEO margin shapes do not align")
    order = _cell_order(receivers)
    cells: Dict[str, Dict[str, Any]] = {}
    deltas: Dict[str, torch.Tensor] = {}
    occupied: Dict[str, bool] = {}
    total_rows = 0
    for receiver_id in receivers:
        for class_id in FROZEN_RECTE_CLASS_IDS:
            key = _receiver_key(receiver_id, class_id)
            mask = rx_labels.eq(receiver_id) & labels.eq(class_id)
            count = int(mask.sum().item())
            total_rows += count
            occupied[key] = count > 0
            if count:
                clean_cell = clean_margins[mask]
                leo_cell = leo_margins[mask]
                if not bool(torch.isfinite(clean_cell.detach()).all().item()) or not bool(
                    torch.isfinite(leo_cell.detach()).all().item()
                ):
                    raise RECTERuntimeError("P1-RECTE occupied cell margins are non-finite")
                delta = leo_cell.mean() - clean_cell.mean().detach()
                if not bool(torch.isfinite(delta.detach()).item()):
                    raise RECTERuntimeError("P1-RECTE occupied cell delta is non-finite")
                delta_value = float(delta.detach().item())
            else:
                delta = leo_margins[mask].sum()
                delta_value = 0.0
            deltas[key] = delta
            cells[key] = {
                "n_rc": count,
                "occupied": bool(count > 0),
                "delta_defined": bool(count > 0),
                "delta": delta_value,
                "empty_cell_zero_contribution": bool(count == 0),
            }
    if total_rows != int(labels.numel()) or tuple(cells) != order:
        raise RECTERuntimeError("P1-RECTE receiver/class cell coverage or order drifted")
    terms = []
    occupied_unordered_pair_count = 0
    positive_tail_pair_count = 0
    for left_index, left_key in enumerate(order):
        for right_key in order[left_index + 1 :]:
            if not (occupied[left_key] and occupied[right_key]):
                continue
            occupied_unordered_pair_count += 1
            left_delta = deltas[left_key]
            right_delta = deltas[right_key]
            pair_q = torch.relu(right_delta.detach() - left_delta).square()
            pair_q = pair_q + torch.relu(left_delta.detach() - right_delta).square()
            if not bool(torch.isfinite(pair_q.detach()).item()):
                raise RECTERuntimeError("P1-RECTE unordered tail pair is non-finite")
            if bool(pair_q.detach().gt(0.0).item()):
                positive_tail_pair_count += 1
            terms.append(pair_q)
    zero = leo_margins.sum() * 0.0
    pair_sum = torch.stack(terms).sum() if terms else zero
    loss = pair_sum / float(FROZEN_RECTE_PAIR_DENOMINATOR)
    if not bool(torch.isfinite(loss.detach()).item()):
        raise RECTERuntimeError("P1-RECTE loss is non-finite")
    if occupied_unordered_pair_count < 0 or positive_tail_pair_count < 0:
        raise RECTERuntimeError("P1-RECTE pair counters are invalid")
    if positive_tail_pair_count > occupied_unordered_pair_count:
        raise RECTERuntimeError("P1-RECTE positive tail pairs exceed occupied unordered pairs")
    return loss, {
        "rows": int(labels.numel()),
        "loss_sum": float(loss.detach().item()),
        "global_denominator": FROZEN_RECTE_PAIR_DENOMINATOR,
        "fixed_scale": 1.0 / float(FROZEN_RECTE_PAIR_DENOMINATOR),
        "source_receiver_ids": list(receivers),
        "cell_order": list(order),
        "cells": cells,
        "occupied_unordered_pair_count": int(occupied_unordered_pair_count),
        "positive_tail_pair_count": int(positive_tail_pair_count),
        "functional_logits_equal_live": True,
        "functional_head_readout_count": 1,
        "finite": True,
        "clean_raw_logits_detached": True,
        "functional_head_parameters_stopgrad": True,
        "tail_only_lower_delta_gradient": True,
        "empty_pair_zero_contribution": True,
        "no_active_pair_renormalization": True,
        "zero_feature_rows_preserved": True,
        "training_accumulation_dtype": "float32",
    }


def add_recte_to_loss(
    base_loss: torch.Tensor, recte: Optional[torch.Tensor], config: Optional[RECTEConfig]
) -> torch.Tensor:
    """Add the sole G-arm term; C returns the untouched common base tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if recte is None:
        raise RECTERuntimeError("Enabled P1-RECTE requires its auxiliary loss")
    return base_loss + float(config.loss_weight) * recte


def recte_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    """Return shared feat_joint encoder and exact-head diagnostic scopes."""

    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise RECTERuntimeError("P1-RECTE requires model.id_backbone for VJP audit")
    head = resolve_recte_classifier_head(raw_model)
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
        raise RECTERuntimeError("P1-RECTE shared encoder or exact head audit scope is empty")
    return {"shared_encoder": encoder, "classifier_head": head_parameters}


def _finite_nonzero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.Tensor], *, group_name: str
) -> Dict[str, float]:
    params = tuple(parameters)
    if not params:
        raise RECTERuntimeError(f"P1-RECTE {group_name} VJP scope is empty")
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    squared_norm = 0.0
    for gradient in gradients:
        if gradient is None:
            raise RECTERuntimeError(f"P1-RECTE {group_name} VJP is None or detached")
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise RECTERuntimeError(f"P1-RECTE {group_name} VJP is non-finite")
        value = gradient.detach().double()
        squared_norm += float(torch.sum(value * value).item())
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm) or norm <= 0.0:
        raise RECTERuntimeError(f"P1-RECTE {group_name} VJP norm is zero or non-finite")
    return {"parameter_count": float(len(params)), "norm": float(norm)}


def _head_none_or_zero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter]
) -> Dict[str, Any]:
    params = tuple(parameters)
    if not params:
        raise RECTERuntimeError("P1-RECTE classifier head VJP scope is empty")
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
            raise RECTERuntimeError("P1-RECTE classifier head auxiliary VJP is non-finite")
        if int(torch.count_nonzero(gradient.detach()).item()) != 0:
            raise RECTERuntimeError("P1-RECTE classifier head must have no auxiliary gradient")
        zero_count += 1
    return {
        "parameter_count": float(len(params)),
        "none_parameters": float(none_count),
        "zero_parameters": float(zero_count),
        "nonzero_parameters": 0.0,
        "none_or_zero_expected": True,
    }


def recte_aux_gradient_audit(
    recte: torch.Tensor,
    feat_joint_leo: torch.Tensor,
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
) -> Dict[str, Any]:
    """Audit first-positive raw RECTE VJPs without touching AMP/optimizer/RNG."""

    if not torch.is_tensor(recte) or recte.ndim != 0:
        raise RECTERuntimeError("P1-RECTE VJP audit requires a scalar auxiliary loss")
    if not torch.is_tensor(feat_joint_leo) or feat_joint_leo.ndim != 2:
        raise RECTERuntimeError("P1-RECTE VJP audit requires LEO feat_joint")
    if tuple(parameter_groups.keys()) != ("shared_encoder", "classifier_head"):
        raise RECTERuntimeError("P1-RECTE VJP audit requires encoder and exact-head scopes")
    return {
        "feat_joint_leo": _finite_nonzero_vjp(
            recte, (feat_joint_leo,), group_name="feat_joint_leo"
        ),
        "shared_encoder": _finite_nonzero_vjp(
            recte, parameter_groups["shared_encoder"], group_name="shared_encoder"
        ),
        "classifier_head": _head_none_or_zero_vjp(
            recte, parameter_groups["classifier_head"]
        ),
        "raw_unscaled": True,
        "diagnostic_only": True,
        "touches_amp_optimizer_rng": False,
        "exact_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        "common_l_base_head_input_path": "LIVE_AND_BOUND_SEPARATELY",
    }


def update_recte_gradient_audit_receipt(
    receipt: Mapping[str, Any], audit: Mapping[str, Any], *, scenario: str
) -> Dict[str, Any]:
    """Seal the first positive-tail-pair raw VJP audit once per LEO scene."""

    result = dict(receipt)
    scenario_name = str(scenario)
    if scenario_name not in FROZEN_RECTE_SCENARIOS:
        raise RECTERuntimeError("P1-RECTE VJP audit scenario is outside the frozen set")
    scene_audits = {
        str(key): dict(value)
        for key, value in dict(result.get("recte_gradient_audit_scenes", {})).items()
    }
    if scenario_name in scene_audits:
        raise RECTERuntimeError("P1-RECTE VJP audit may run only once per scene")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
        or audit.get("exact_head_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
    ):
        raise RECTERuntimeError("P1-RECTE VJP audit semantics drifted")
    for group_name in ("feat_joint_leo", "shared_encoder"):
        values = audit.get(group_name)
        if not isinstance(values, Mapping):
            raise RECTERuntimeError("P1-RECTE VJP audit lacks a required nonzero scope")
        count = float(values.get("parameter_count", 0.0))
        norm = float(values.get("norm", float("nan")))
        if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
            raise RECTERuntimeError("P1-RECTE required auxiliary VJP is zero or non-finite")
    head = audit.get("classifier_head")
    if not isinstance(head, Mapping):
        raise RECTERuntimeError("P1-RECTE VJP audit lacks exact-head N/A scope")
    head_count = float(head.get("parameter_count", 0.0))
    none_count = float(head.get("none_parameters", float("nan")))
    zero_count = float(head.get("zero_parameters", float("nan")))
    nonzero_count = float(head.get("nonzero_parameters", float("nan")))
    if (
        head_count <= 0.0
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in (none_count, zero_count, nonzero_count)
        )
        or none_count + zero_count != head_count
        or nonzero_count != 0.0
        or head.get("none_or_zero_expected") is not True
    ):
        raise RECTERuntimeError("P1-RECTE exact-head auxiliary VJP contract failed")
    scene_audits[scenario_name] = dict(audit)
    result["recte_gradient_audit_attempted"] = True
    result["recte_gradient_audit_completed"] = set(scene_audits) == set(
        FROZEN_RECTE_SCENARIOS
    )
    result["recte_gradient_audit_scenes"] = scene_audits
    return result


def bind_recte_source_data_order(
    receipt: Mapping[str, Any], source_split_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind source-L physical order and its source-only RX allowlist."""

    result = dict(receipt)
    source = dict(source_split_receipt or {})
    labeled_sha = str(source.get("labeled_indices_sha256", "") or "")
    manifest_sha = str(source.get("split_manifest_sha256", "") or "")
    if len(labeled_sha) != 64 or len(manifest_sha) != 64:
        raise RECTEConfigurationError(
            "P1-RECTE requires labeled-index and source-split SHA256 receipts"
        )
    receivers = _require_frozen_source_receivers(source.get("source_receivers", ()))
    result["source_labeled_indices_sha256"] = labeled_sha
    result["source_split_manifest_sha256"] = manifest_sha
    result["source_receiver_ids"] = list(receivers)
    result["source_receiver_count"] = len(receivers)
    result["source_receiver_ids_sha256"] = _canonical_sha256(list(receivers))
    result["source_receiver_provenance"] = (
        "SOURCE_SPLIT_RECEIPT_source_receivers_PHYSICAL_ID_BOUND_L_ONLY"
    )
    return result


def _as_plain_list(values: Any) -> list[Any]:
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return []


def _common_cell_event(
    *, receiver_ids: Sequence[Any], labels: torch.Tensor, rx_labels: torch.Tensor
) -> Tuple[Dict[str, int], Dict[str, bool], Tuple[str, ...]]:
    order = _cell_order(receiver_ids)
    counts = {
        _receiver_key(receiver_id, class_id): int(
            (rx_labels.eq(receiver_id) & labels.eq(class_id)).sum().item()
        )
        for receiver_id in _require_frozen_source_receivers(receiver_ids)
        for class_id in FROZEN_RECTE_CLASS_IDS
    }
    if tuple(counts) != order or sum(counts.values()) != int(labels.numel()):
        raise RECTERuntimeError("P1-RECTE common n_rc counters do not close")
    return counts, {key: bool(counts[key] > 0) for key in order}, order


def update_recte_common_batch_sequence_receipt(
    receipt: Mapping[str, Any],
    *,
    epoch: int,
    batch_index: int,
    scenario: str,
    source_tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Chain C/G-identical physical/RX/class/scene coverage and cell order."""

    result = dict(receipt)
    expected_scene = FROZEN_RECTE_SCENARIOS[(int(epoch) + int(batch_index) - 2) % 3]
    if str(scenario) != expected_scene:
        raise RECTERuntimeError("P1-RECTE common LEO scenario sequence drifted")
    labels = source_tx_labels.detach().reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RECTERuntimeError("P1-RECTE common sequence requires local4 source L labels")
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    rx_labels = _validated_receiver_labels(
        source_rx_labels, rows=int(labels.numel()), expected_receiver_ids=receivers
    ).detach()
    if metadata is None:
        raise RECTERuntimeError("P1-RECTE common sequence requires opaque physical metadata")
    opaque_ids = _as_plain_list(metadata.get("base_index"))
    if len(opaque_ids) != int(labels.numel()):
        opaque_ids = _as_plain_list(metadata.get("sig_i"))
    if len(opaque_ids) != int(labels.numel()):
        raise RECTERuntimeError("P1-RECTE physical batch sequence metadata is incomplete")
    counts, occupancy, order = _common_cell_event(
        receiver_ids=receivers, labels=labels, rx_labels=rx_labels
    )
    event = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "scenario": str(scenario),
        "same_physical_clean_leo": True,
        "cell_order": list(order),
        "rows": [
            [str(opaque), int(label), int(receiver_id)]
            for opaque, label, receiver_id in zip(
                opaque_ids, labels.cpu().tolist(), rx_labels.cpu().tolist()
            )
        ],
        "n_rc": counts,
        "occupied": occupancy,
    }
    prior = str(result.get("common_batch_sequence_sha256", "") or "")
    if not prior:
        prior = str(result.get("source_labeled_indices_sha256", "") or "")
    if len(prior) != 64:
        raise RECTERuntimeError("P1-RECTE common batch sequence lacks source data-order SHA256")
    result["common_batch_sequence_sha256"] = hashlib.sha256(
        (
            prior
            + "\n"
            + json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ).encode("utf-8")
    ).hexdigest()
    result["common_batch_sequence_batches"] = int(
        result.get("common_batch_sequence_batches", 0)
    ) + 1
    result["common_batch_sequence_rows"] = int(
        result.get("common_batch_sequence_rows", 0)
    ) + int(labels.numel())
    scene_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if set(scene_batches) != set(FROZEN_RECTE_SCENARIOS):
        raise RECTERuntimeError("P1-RECTE common scenario receipt is malformed")
    scene_batches[str(scenario)] += 1
    result["common_scenario_batches"] = scene_batches
    common_cells = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("recte_common_cells", {})).items()
    }
    scene_cells = common_cells.get(str(scenario), _cell_template(receivers))
    if tuple(scene_cells) != order:
        raise RECTERuntimeError("P1-RECTE common receiver/class cells are malformed")
    for key in order:
        cell = dict(scene_cells[key])
        count = int(counts[key])
        cell["rows"] = int(cell.get("rows", 0)) + count
        cell["batches"] = int(cell.get("batches", 0)) + 1
        cell["nonempty_batches"] = int(cell.get("nonempty_batches", 0)) + int(count > 0)
        scene_cells[key] = cell
    common_cells[str(scenario)] = scene_cells
    result["recte_common_cells"] = common_cells
    batch_events = list(result.get("recte_common_batch_cells", []))
    batch_events.append(
        {
            key: event[key]
            for key in (
                "epoch",
                "batch_index",
                "scenario",
                "same_physical_clean_leo",
                "cell_order",
                "n_rc",
                "occupied",
            )
        }
    )
    result["recte_common_batch_cells"] = batch_events
    return result


def bind_recte_optimizer_initial_state(
    receipt: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Seal the new AdamW state before the first backward call."""

    result = dict(receipt)
    optimizer_type = type(optimizer).__name__
    if optimizer_type != FROZEN_RECTE_OPTIMIZER_TYPE:
        raise RECTEConfigurationError(
            "P1-RECTE requires optimizer_type=AdamW, got " + (optimizer_type or "<empty>")
        )
    state = optimizer.state_dict()
    if dict(state.get("state", {})):
        raise RECTEConfigurationError("P1-RECTE requires a new AdamW state")
    groups = []
    for group in state.get("param_groups", []):
        normalized = {
            str(key): value for key, value in dict(group).items() if str(key) != "params"
        }
        normalized["parameter_count"] = len(list(dict(group).get("params", [])))
        groups.append(normalized)
    result["optimizer_initial_state_sha256"] = _canonical_sha256(
        {"optimizer_type": optimizer_type, "state_empty": True, "param_groups": groups}
    )
    result["optimizer_type"] = optimizer_type
    result["optimizer_initial_state_empty"] = True
    return result


def update_recte_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    scenario: str,
    epoch: int,
    batch_index: int,
) -> Dict[str, Any]:
    """Accumulate G-only tail-pair evidence after common coverage is sealed."""

    result = dict(receipt)
    if str(result.get("schema", "")) != RECTE_RECEIPT_SCHEMA:
        raise RECTERuntimeError("P1-RECTE receipt schema is invalid")
    if result.get("enabled") is not True:
        raise RECTERuntimeError("P1-RECTE auxiliary receipt update is G-arm only")
    if str(scenario) not in FROZEN_RECTE_SCENARIOS:
        raise RECTERuntimeError("P1-RECTE scenario is outside frozen clear/low/rain cycle")
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    order = _cell_order(receivers)
    if tuple(int(value) for value in batch_info.get("source_receiver_ids", ())) != receivers:
        raise RECTERuntimeError("P1-RECTE G batch source receiver allowlist drifted")
    if tuple(int(value) for value in result.get("expected_tx_class_ids", [])) != FROZEN_RECTE_CLASS_IDS:
        raise RECTERuntimeError("P1-RECTE receipt lacks local4 class binding")
    for key in (
        "finite",
        "clean_raw_logits_detached",
        "functional_head_parameters_stopgrad",
        "tail_only_lower_delta_gradient",
        "empty_pair_zero_contribution",
        "no_active_pair_renormalization",
        "zero_feature_rows_preserved",
        "functional_logits_equal_live",
    ):
        if batch_info.get(key) is not True:
            raise RECTERuntimeError(f"P1-RECTE batch semantic receipt drifted: {key}")
    if batch_info.get("training_accumulation_dtype") != "float32":
        raise RECTERuntimeError("P1-RECTE batch accumulation dtype drifted")
    if int(batch_info.get("global_denominator", -1)) != FROZEN_RECTE_PAIR_DENOMINATOR:
        raise RECTERuntimeError("P1-RECTE unordered pair denominator drifted")
    if int(batch_info.get("functional_head_readout_count", -1)) != 1:
        raise RECTERuntimeError("P1-RECTE requires exactly one G-only functional head readout")
    scale = float(batch_info.get("fixed_scale", float("nan")))
    if not math.isfinite(scale) or abs(
        scale - 1.0 / float(FROZEN_RECTE_PAIR_DENOMINATOR)
    ) > _TOLERANCE:
        raise RECTERuntimeError("P1-RECTE fixed scale drifted")
    cells = {str(key): dict(value) for key, value in dict(batch_info.get("cells", {})).items()}
    if tuple(batch_info.get("cell_order", ())) != order or tuple(cells) != order:
        raise RECTERuntimeError("P1-RECTE G receiver/class canonical order drifted")
    rows = int(batch_info.get("rows", -1))
    occupied_pairs = int(batch_info.get("occupied_unordered_pair_count", -1))
    positive_pairs = int(batch_info.get("positive_tail_pair_count", -1))
    loss_sum = float(batch_info.get("loss_sum", float("nan")))
    if (
        rows <= 0
        or occupied_pairs < 0
        or occupied_pairs > FROZEN_RECTE_PAIR_DENOMINATOR
        or positive_pairs < 0
        or positive_pairs > occupied_pairs
        or not math.isfinite(loss_sum)
    ):
        raise RECTERuntimeError("P1-RECTE G batch rows/pair/loss counters do not close")
    common_events = list(result.get("recte_common_batch_cells", []))
    if not common_events:
        raise RECTERuntimeError("P1-RECTE G batch lacks its common C/G coverage receipt")
    common_event = dict(common_events[-1])
    if (
        int(common_event.get("epoch", -1)) != int(epoch)
        or int(common_event.get("batch_index", -1)) != int(batch_index)
        or str(common_event.get("scenario", "")) != str(scenario)
        or common_event.get("same_physical_clean_leo") is not True
        or tuple(common_event.get("cell_order", ())) != order
    ):
        raise RECTERuntimeError("P1-RECTE G/common same-physical cell binding drifted")
    common_counts = {
        str(key): int(value) for key, value in dict(common_event.get("n_rc", {})).items()
    }
    common_occupancy = {
        str(key): bool(value) for key, value in dict(common_event.get("occupied", {})).items()
    }
    if tuple(common_counts) != order or tuple(common_occupancy) != order:
        raise RECTERuntimeError("P1-RECTE G/common cell fields are malformed")
    if common_counts != {key: int(cells[key].get("n_rc", -1)) for key in order}:
        raise RECTERuntimeError("P1-RECTE G/common n_rc receipt mismatch")
    if common_occupancy != {key: bool(cells[key].get("occupied", False)) for key in order}:
        raise RECTERuntimeError("P1-RECTE G/common occupied-cell receipt mismatch")
    if sum(common_counts.values()) != rows:
        raise RECTERuntimeError("P1-RECTE G/common rows do not close")
    scenes = {str(key): dict(value) for key, value in dict(result.get("recte_scenes", {})).items()}
    scene = dict(
        scenes.get(
            str(scenario),
            {
                "batches": 0,
                "rows": 0,
                "occupied_unordered_pair_count": 0,
                "positive_tail_pair_count": 0,
                "loss_sum": 0.0,
                "functional_logits_equal_live_batches": 0,
            },
        )
    )
    scene["batches"] = int(scene.get("batches", 0)) + 1
    scene["rows"] = int(scene.get("rows", 0)) + rows
    scene["occupied_unordered_pair_count"] = int(
        scene.get("occupied_unordered_pair_count", 0)
    ) + occupied_pairs
    scene["positive_tail_pair_count"] = int(
        scene.get("positive_tail_pair_count", 0)
    ) + positive_pairs
    scene["loss_sum"] = float(scene.get("loss_sum", 0.0)) + loss_sum
    scene["functional_logits_equal_live_batches"] = int(
        scene.get("functional_logits_equal_live_batches", 0)
    ) + 1
    scenes[str(scenario)] = scene
    result["recte_scenes"] = scenes
    events = list(result.get("recte_g_batch_aux", []))
    events.append(
        {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "scenario": str(scenario),
            "rows": rows,
            "occupied_unordered_pair_count": occupied_pairs,
            "positive_tail_pair_count": positive_pairs,
            "loss_sum": loss_sum,
            "functional_logits_equal_live": True,
            "functional_head_readout_count": 1,
            "cell_order": list(order),
            "n_rc": common_counts,
            "occupied": common_occupancy,
        }
    )
    result["recte_g_batch_aux"] = events
    result["recte_batches"] = int(result.get("recte_batches", 0)) + 1
    result["recte_total_rows"] = int(result.get("recte_total_rows", 0)) + rows
    result["recte_occupied_unordered_pair_count"] = int(
        result.get("recte_occupied_unordered_pair_count", 0)
    ) + occupied_pairs
    result["recte_positive_tail_pair_count"] = int(
        result.get("recte_positive_tail_pair_count", 0)
    ) + positive_pairs
    result["recte_loss_sum"] = float(result.get("recte_loss_sum", 0.0)) + loss_sum
    result["recte_functional_head_readout_count"] = int(
        result.get("recte_functional_head_readout_count", 0)
    ) + 1
    return result


def _validate_common_terminal_contract(result: Mapping[str, Any]) -> None:
    if str(result.get("schema", "")) != RECTE_RECEIPT_SCHEMA:
        raise RECTERuntimeError("P1-RECTE terminal receipt schema is invalid")
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
            raise RECTERuntimeError(f"P1-RECTE terminal receipt lacks {key}")
    if str(result.get("checkpoint_role", "") or "") != "training_final_only":
        raise RECTERuntimeError("P1-RECTE requires training_final_only warm start")
    if result.get("optimizer_state_restored") is not False or result.get("rng_state_restored") is not False:
        raise RECTERuntimeError("P1-RECTE optimizer/RNG restoration is forbidden")
    if str(result.get("optimizer_type", "")) != FROZEN_RECTE_OPTIMIZER_TYPE:
        raise RECTERuntimeError("P1-RECTE terminal optimizer_type must be AdamW")
    if result.get("optimizer_initial_state_empty") is not True:
        raise RECTERuntimeError("P1-RECTE missing new AdamW initial-state receipt")
    if result.get("amp_contract") != "COMMON_TRAINER_AMP_ENABLED":
        raise RECTERuntimeError("P1-RECTE terminal AMP contract drifted")
    if result.get("common_l_base_head_input_path_verified") is not True:
        raise RECTERuntimeError("P1-RECTE common L_base exact head-input path is not verified")
    if int(result.get("loss_global_denominator", -1)) != FROZEN_RECTE_PAIR_DENOMINATOR:
        raise RECTERuntimeError("P1-RECTE terminal unordered pair denominator drifted")
    if int(result.get("frozen_cells_per_scene", -1)) != FROZEN_RECTE_CELL_COUNT:
        raise RECTERuntimeError("P1-RECTE terminal fixed cell count drifted")
    if int(result.get("fixed_unordered_pair_count", -1)) != FROZEN_RECTE_PAIR_DENOMINATOR:
        raise RECTERuntimeError("P1-RECTE terminal fixed unordered-pair count drifted")
    batches = int(result.get("common_batch_sequence_batches", 0))
    rows = int(result.get("common_batch_sequence_rows", 0))
    scene_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if (
        batches <= 0
        or rows <= 0
        or set(scene_batches) != set(FROZEN_RECTE_SCENARIOS)
        or any(value <= 0 for value in scene_batches.values())
    ):
        raise RECTERuntimeError("P1-RECTE common batch/scenario receipt is incomplete")


def _validate_common_cells(result: Mapping[str, Any]) -> None:
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    order = _cell_order(receivers)
    scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("recte_common_cells", {})).items()
    }
    if set(scenes) != set(FROZEN_RECTE_SCENARIOS):
        raise RECTERuntimeError("P1-RECTE terminal common receiver/class/scene coverage is incomplete")
    scene_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    total_rows = 0
    for scene_name in FROZEN_RECTE_SCENARIOS:
        cells = scenes[scene_name]
        if tuple(cells) != order:
            raise RECTERuntimeError("P1-RECTE terminal common receiver/class order drifted")
        for key in order:
            cell = cells[key]
            if (
                int(cell.get("rows", 0)) <= 0
                or int(cell.get("batches", -1)) != scene_batches[scene_name]
                or int(cell.get("nonempty_batches", 0)) <= 0
            ):
                raise RECTERuntimeError("P1-RECTE terminal common cell coverage is incomplete")
            total_rows += int(cell.get("rows", 0))
    if total_rows != int(result.get("common_batch_sequence_rows", 0)):
        raise RECTERuntimeError("P1-RECTE terminal common cell rows do not close")
    events = list(result.get("recte_common_batch_cells", []))
    if len(events) != int(result.get("common_batch_sequence_batches", 0)):
        raise RECTERuntimeError("P1-RECTE terminal common batch receipt is incomplete")
    event_rows = 0
    for event in events:
        if (
            event.get("same_physical_clean_leo") is not True
            or tuple(event.get("cell_order", ())) != order
        ):
            raise RECTERuntimeError("P1-RECTE terminal common same-physical/order drifted")
        counts = {str(key): int(value) for key, value in dict(event.get("n_rc", {})).items()}
        occupied = {str(key): bool(value) for key, value in dict(event.get("occupied", {})).items()}
        if tuple(counts) != order or tuple(occupied) != order or any(value < 0 for value in counts.values()):
            raise RECTERuntimeError("P1-RECTE terminal common n_rc/occupied fields are malformed")
        if occupied != {key: bool(counts[key] > 0) for key in order}:
            raise RECTERuntimeError("P1-RECTE terminal common occupancy fields drifted")
        event_rows += sum(counts.values())
    if event_rows != int(result.get("common_batch_sequence_rows", 0)):
        raise RECTERuntimeError("P1-RECTE terminal common n_rc rows do not close")


def validate_recte_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless common coverage and per-scene positive-tail/VJP close."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    _validate_common_terminal_contract(result)
    _validate_common_cells(result)
    enabled = result.get("enabled")
    if enabled is not True and enabled is not False:
        raise RECTERuntimeError("P1-RECTE terminal enabled flag must be strict bool")
    if enabled is False:
        zero_keys = (
            "recte_batches",
            "recte_total_rows",
            "recte_occupied_unordered_pair_count",
            "recte_positive_tail_pair_count",
            "recte_functional_head_readout_count",
        )
        if any(int(result.get(key, 0)) != 0 for key in zero_keys) or abs(
            float(result.get("recte_loss_sum", 0.0))
        ) > _TOLERANCE:
            raise RECTERuntimeError("P1-RECTE C arm must retain zero auxiliary counters")
        if any(
            bool(result.get(key))
            for key in (
                "recte_scenes",
                "recte_g_batch_aux",
                "recte_gradient_audit_scenes",
            )
        ) or bool(result.get("recte_gradient_audit_attempted", False)) or bool(
            result.get("recte_gradient_audit_completed", False)
        ):
            raise RECTERuntimeError("P1-RECTE C arm must retain N/A-or-zero auxiliary fields")
        result["recte_terminal_contract"] = (
            "CONTROL_ARM_COMMON_SAME_PHYSICAL_RX_CLASS_SCENE_ORDER_COVERAGE_AUX_NA_OR_ZERO"
        )
        result["recte_terminal_contract_passed"] = True
        return result
    scenes = {str(key): dict(value) for key, value in dict(result.get("recte_scenes", {})).items()}
    if set(scenes) != set(FROZEN_RECTE_SCENARIOS):
        raise RECTERuntimeError("P1-RECTE terminal G scene coverage is incomplete")
    total_rows = total_occupied = total_positive = total_readouts = 0
    total_loss = 0.0
    for scene_name in FROZEN_RECTE_SCENARIOS:
        scene = scenes[scene_name]
        batches = int(scene.get("batches", -1))
        rows = int(scene.get("rows", -1))
        occupied = int(scene.get("occupied_unordered_pair_count", -1))
        positive = int(scene.get("positive_tail_pair_count", -1))
        readouts = int(scene.get("functional_logits_equal_live_batches", -1))
        loss = float(scene.get("loss_sum", float("nan")))
        if (
            batches <= 0
            or rows <= 0
            or occupied < 0
            or occupied > batches * FROZEN_RECTE_PAIR_DENOMINATOR
            or positive <= 0
            or positive > occupied
            or readouts != batches
            or not math.isfinite(loss)
        ):
            raise RECTERuntimeError(
                "P1-RECTE terminal each scene requires positive tail pair and functional-logit equality"
            )
        total_rows += rows
        total_occupied += occupied
        total_positive += positive
        total_readouts += readouts
        total_loss += loss
    events = list(result.get("recte_g_batch_aux", []))
    if len(events) != int(result.get("recte_batches", -1)):
        raise RECTERuntimeError("P1-RECTE terminal G batch auxiliary receipt is incomplete")
    if any(
        event.get("functional_logits_equal_live") is not True
        or int(event.get("functional_head_readout_count", -1)) != 1
        for event in events
    ):
        raise RECTERuntimeError("P1-RECTE terminal functional head receipt drifted")
    if (
        int(result.get("recte_batches", -1)) != int(result.get("common_batch_sequence_batches", -2))
        or int(result.get("recte_total_rows", -1)) != int(result.get("common_batch_sequence_rows", -2))
        or total_rows != int(result.get("common_batch_sequence_rows", -2))
        or int(result.get("recte_occupied_unordered_pair_count", -1)) != total_occupied
        or int(result.get("recte_positive_tail_pair_count", -1)) != total_positive
        or int(result.get("recte_functional_head_readout_count", -1)) != total_readouts
        or total_positive <= 0
        or not _float32_ledger_close(float(result.get("recte_loss_sum", float("nan"))), total_loss)
    ):
        raise RECTERuntimeError("P1-RECTE terminal G rows/pair/loss counters do not close")
    scene_audits = {
        str(key): dict(value)
        for key, value in dict(result.get("recte_gradient_audit_scenes", {})).items()
    }
    if (
        not bool(result.get("recte_gradient_audit_completed", False))
        or set(scene_audits) != set(FROZEN_RECTE_SCENARIOS)
    ):
        raise RECTERuntimeError(
            "P1-RECTE terminal per-scene first-positive-tail raw VJP audit is incomplete"
        )
    for scene_name in FROZEN_RECTE_SCENARIOS:
        audit = scene_audits[scene_name]
        if (
            audit.get("raw_unscaled") is not True
            or audit.get("diagnostic_only") is not True
            or audit.get("touches_amp_optimizer_rng") is not False
            or audit.get("exact_head_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        ):
            raise RECTERuntimeError(
                "P1-RECTE terminal per-scene VJP audit semantics drifted"
            )
        for group_name in ("feat_joint_leo", "shared_encoder"):
            values = audit.get(group_name)
            if not isinstance(values, Mapping):
                raise RECTERuntimeError(
                    "P1-RECTE terminal per-scene VJP audit lacks a required scope"
                )
            count = float(values.get("parameter_count", 0.0))
            norm = float(values.get("norm", float("nan")))
            if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
                raise RECTERuntimeError(
                    "P1-RECTE terminal per-scene required VJP is zero or non-finite"
                )
        head = audit.get("classifier_head")
        if not isinstance(head, Mapping):
            raise RECTERuntimeError(
                "P1-RECTE terminal per-scene VJP audit lacks exact-head scope"
            )
        head_count = float(head.get("parameter_count", 0.0))
        none_count = float(head.get("none_parameters", float("nan")))
        zero_count = float(head.get("zero_parameters", float("nan")))
        nonzero_count = float(head.get("nonzero_parameters", float("nan")))
        if (
            head_count <= 0.0
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in (none_count, zero_count, nonzero_count)
            )
            or none_count + zero_count != head_count
            or nonzero_count != 0.0
            or head.get("none_or_zero_expected") is not True
        ):
            raise RECTERuntimeError(
                "P1-RECTE terminal per-scene exact-head auxiliary VJP contract failed"
            )
    result["recte_terminal_contract"] = (
        "FORMAL_COMMON_SAME_PHYSICAL_RX_CLASS_SCENE_ORDER_FIXED_28_CELL_378_UNORDERED_PAIR_"
        "WITH_G_ONLY_EACH_SCENE_POSITIVE_TAIL_AND_RAW_feat_joint_SHARED_ENCODER_VJP_"
        "EXACT_HEAD_AUX_NA"
    )
    result["recte_terminal_contract_passed"] = True
    return result


def _failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "vjp" in message or "gradient" in message or "head" in message:
        return "RECTE_AUX_GRADIENT_OR_HEAD_PATH_FAILURE"
    if "non-finite" in message or "nonfinite" in message:
        return "RECTE_NONFINITE"
    if "receiver" in message or "rx_i" in message or "cell" in message:
        return "RECTE_SOURCE_RX_OR_CELL_COVERAGE_FAILURE"
    if "pair" in message or "tail" in message or "denominator" in message:
        return "RECTE_FIXED_PAIR_CONTRACT_FAILURE"
    if "sequence" in message or "receipt" in message or "coverage" in message:
        return "RECTE_RECEIPT_CLOSURE_FAILURE"
    return "RECTE_RUNTIME_FAILURE"


def write_recte_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free fail-closed record for the RECTE arm."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise RECTERuntimeError(f"P1-RECTE failure receipt output directory is absent: {target_dir}")
    payload = {
        "schema": "cvs.phase1.recte_failure_receipt.v1",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage or ""),
        "exception_type": type(error).__name__,
        "exception_fingerprint": _failure_fingerprint(error),
        "message": str(error),
        "receipt": dict(receipt),
    }
    target = target_dir / "phase1_recte_failure_receipt.json"
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd, temporary_name = mkstemp(
        prefix=".recte_failure_receipt.", suffix=".tmp", dir=str(target_dir)
    )
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


def strict_recte_warm_start(
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
        raise RECTEConfigurationError(
            "Frozen P1-RECTE warm-start requires model state, path, and SHA256"
        )
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise RECTEConfigurationError(
            f"Frozen P1-RECTE strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RECTEConfigurationError(
            "Frozen P1-RECTE strict baseline model-key mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    if str(checkpoint_role or "") != "training_final_only":
        raise RECTEConfigurationError(
            "Frozen P1-RECTE requires baseline checkpoint_role=training_final_only"
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
