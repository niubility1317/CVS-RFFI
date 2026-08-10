"""Frozen P1-HSCF continuation contract for Phase1 source-only DG.

P1-HSCF (Head-Subspace Configuration Fidelity) preserves the common
GeoSat-C clean and one-LEO forward exactly.  Its G arm reads the already
computed raw local4 classifier logits for the same source-L physical rows.  It
removes the per-row all-class component and the batch mean, anchors the clean
configuration with stop-gradient, and applies one fixed 1/512 squared error to
the live LEO configuration.  It therefore preserves relative head geometry
without introducing a second view, model, state, cache, RX/day input, or
post-freeze feedback channel.
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


FROZEN_HSCF_LAMBDA = 0.02
FROZEN_HSCF_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_HSCF_CLASS_IDS = (0, 1, 2, 3)
FROZEN_HSCF_BATCH_SIZE = 128
FROZEN_HSCF_CLASS_COUNT = len(FROZEN_HSCF_CLASS_IDS)
FROZEN_HSCF_GLOBAL_DENOMINATOR = FROZEN_HSCF_BATCH_SIZE * FROZEN_HSCF_CLASS_COUNT
FROZEN_HSCF_OPTIMIZER_TYPE = "AdamW"
FROZEN_HSCF_FLOAT32_LEDGER_REL_TOL = 32.0 * float(torch.finfo(torch.float32).eps)
FROZEN_HSCF_BIAS_VJP_NUMERICAL_ZERO_REL_TOL = 64.0 * float(torch.finfo(torch.float32).eps)
HSCF_RECEIPT_SCHEMA = "cvs.phase1.hscf_receipt.v1"
_TOLERANCE = 1e-12


class HSCFConfigurationError(ValueError):
    """Raised when a frozen P1-HSCF C/G configuration drifts."""


class HSCFRuntimeError(RuntimeError):
    """Raised when a P1-HSCF runtime or receipt contract cannot be proved."""


@dataclass(frozen=True)
class HSCFConfig:
    """Immutable P1-HSCF controls consumed by the common training loop."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise HSCFConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
        raise HSCFConfigurationError(
            f"Frozen P1-HSCF requires {name}={expected:.12g}, got {actual!r}"
        )


def _float32_ledger_close(actual: float, expected: float) -> bool:
    return (
        math.isfinite(actual)
        and math.isfinite(expected)
        and abs(actual - expected)
        <= FROZEN_HSCF_FLOAT32_LEDGER_REL_TOL * max(1.0, abs(actual), abs(expected))
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
        raise HSCFConfigurationError(
            "Frozen P1-HSCF forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    scenarios = tuple(
        part.strip().lower().replace("-", "_")
        for part in str(value or "").split(",")
        if part.strip()
    )
    if scenarios != FROZEN_HSCF_SCENARIOS:
        raise HSCFConfigurationError(
            "Frozen P1-HSCF requires --sat_train_scenarios "
            + ",".join(FROZEN_HSCF_SCENARIOS)
        )
    return scenarios


def validate_hscf_args(args: Any) -> HSCFConfig:
    """Validate the frozen common-base C/G contract before data are loaded."""

    frozen_mode = _bool_arg(args, "phase1_hscf_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_hscf_enabled", False)
    loss_weight = _float_arg(args, "lambda_hscf", 0.0)
    if not frozen_mode and not enabled:
        return HSCFConfig(False, False, 0.0)
    if enabled and not frozen_mode:
        raise HSCFConfigurationError(
            "--phase1_hscf_enabled requires --phase1_hscf_frozen_mode true"
        )
    _require_close("lambda_hscf", loss_weight, FROZEN_HSCF_LAMBDA if enabled else 0.0)
    if int(getattr(args, "batch_size", 0)) != FROZEN_HSCF_BATCH_SIZE:
        raise HSCFConfigurationError("Frozen P1-HSCF requires --batch_size 128")
    if bool(getattr(args, "from_scratch", True)):
        raise HSCFConfigurationError("Frozen P1-HSCF requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise HSCFConfigurationError("Frozen P1-HSCF requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise HSCFConfigurationError("Frozen P1-HSCF must train the shared feat_joint encoder")
    if not bool(getattr(args, "amp", True)):
        raise HSCFConfigurationError("Frozen P1-HSCF requires the common AMP training path")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise HSCFConfigurationError("Frozen P1-HSCF requires --id_feature_key feat_joint")
    if int(getattr(args, "epochs", 0)) != 40 or int(getattr(args, "label_epochs", 0)) != 40:
        raise HSCFConfigurationError("Frozen P1-HSCF requires exactly 40 labeled epochs")
    if int(getattr(args, "pseudo_epochs", 0)) != 0:
        raise HSCFConfigurationError("Frozen P1-HSCF forbids pseudo epochs")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise HSCFConfigurationError("Frozen P1-HSCF requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise HSCFConfigurationError("Frozen P1-HSCF remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise HSCFConfigurationError("Frozen P1-HSCF requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise HSCFConfigurationError("Frozen P1-HSCF requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise HSCFConfigurationError("Frozen P1-HSCF forbids --sat_view_schedule overrides")
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise HSCFConfigurationError("Frozen P1-HSCF requires non-concatenated single-LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise HSCFConfigurationError("Frozen P1-HSCF permits only source_known_train L updates")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise HSCFConfigurationError("Frozen P1-HSCF requires the common non-RX-conditioned batch order")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise HSCFConfigurationError("Frozen P1-HSCF permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise HSCFConfigurationError("Frozen P1-HSCF forbids a reject head")
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
            "phase1_recte_frozen_mode",
            "phase1_recte_enabled",
            "lambda_recte",
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
    return HSCFConfig(True, enabled, loss_weight)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def hscf_config_receipt(config: HSCFConfig) -> Dict[str, Any]:
    """Create the data-free receipt skeleton for either frozen C/G arm."""

    return {
        "schema": HSCF_RECEIPT_SCHEMA,
        "method": "P1_HSCF",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "loss_rule": "SOURCE_L_SAME_PHYSICAL_CLEAN_SINGLE_LEO_HEAD_SUBSPACE_CONFIGURATION_FIDELITY",
        "loss_formula": "a_i^v=ell_i^v-mean_c(ell_i^v);r_i^v=a_i^v-mean_j(a_j^v);L_HSCF=sum_i||r_i^L-sg(r_i^C)||_2^2/512",
        "loss_global_denominator": FROZEN_HSCF_GLOBAL_DENOMINATOR,
        "fixed_batch_size": FROZEN_HSCF_BATCH_SIZE,
        "fixed_local_class_count": FROZEN_HSCF_CLASS_COUNT,
        "fixed_scale": 1.0 / float(FROZEN_HSCF_GLOBAL_DENOMINATOR),
        "local_class_ids": list(FROZEN_HSCF_CLASS_IDS),
        "z_id_key": "feat_joint",
        "feature_dimension_contract": "RAW_ENCODER_feat_joint_EXACT_HEAD_INPUT_DIMENSION_BOUND",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "common_l_base_head_input_path_verified": False,
        "logit_centering": "CLASS_PROJECTOR_P=I4-11T/4_THEN_EXACT_BATCH_MEAN_OVER_128_NO_RENORM",
        "clean_logits_detached": True,
        "leo_logits_live": True,
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD_SAME_ORDER",
        "common_lambda_sat_cons": 0.10,
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "common_batch_size": FROZEN_HSCF_BATCH_SIZE,
        "common_loader_drop_last": True,
        "common_order_contract": "C_G_IDENTICAL_SEED_SAMPLER_PHYSICAL_IDS_AND_CLEAR_LOW_RAIN_SEQUENCE",
        "aux_gradient_scope": "LEO_RAW_LOGITS_SHARED_ENCODER_AND_EXACT_HEAD_WEIGHT_FINITE_NONZERO;HEAD_BIAS_AND_CLEAN_LOGITS_NA_NONE_OR_ZERO",
        "uses_new_forward": False,
        "uses_resampling": False,
        "uses_rx_labels": False,
        "uses_day_labels": False,
        "uses_fold_labels": False,
        "uses_domain_labels": False,
        "uses_target_rows": False,
        "uses_proxy_rows": False,
        "uses_held_rows": False,
        "uses_unlabeled_rows": False,
        "u_loader_common_trainer_boundary": "MAY_BE_CONSTRUCTED_BY_COMMON_TRAINER_BUT_HSCF_ZERO_ITERATE_ZERO_FORWARD_ZERO_LOSS_ZERO_BACKWARD_ZERO_OPTIMIZER",
        "v_common_trainer_boundary": "COMMON_READ_ONLY_DIAGNOSTIC_ONLY_HSCF_ZERO_LOSS_ZERO_BACKWARD_ZERO_OPTIMIZER_ZERO_CALIBRATION_ZERO_MODEL_SELECTION_FEEDBACK",
        "uses_ema_or_state": False,
        "uses_threshold": False,
        "uses_gradient_projection": False,
        "uses_margin_or_hinge": False,
        "uses_centroid_radius_or_gram": False,
        "uses_tail_or_cell_equalization": False,
        "uses_independent_view": False,
        "uses_postfreeze_selection": False,
        "zero_logits_legal": True,
        "nonfinite_fail_closed": True,
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
        "source_labeled_provenance": "PENDING_SOURCE_L_ONLY_BINDING",
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
        "common_scenario_batches": {scene: 0 for scene in FROZEN_HSCF_SCENARIOS},
        "hscf_common_scenes": {},
        "hscf_common_batches": [],
        "hscf_batches": 0,
        "hscf_total_rows": 0,
        "hscf_positive_batches": 0,
        "hscf_positive_components": 0,
        "hscf_loss_sum": 0.0,
        "hscf_float32_ledger_rel_tolerance": FROZEN_HSCF_FLOAT32_LEDGER_REL_TOL,
        "hscf_bias_vjp_numerical_zero_rel_tolerance": FROZEN_HSCF_BIAS_VJP_NUMERICAL_ZERO_REL_TOL,
        "hscf_scenes": {},
        "hscf_g_batch_aux": [],
        "hscf_gradient_audit_attempted": False,
        "hscf_gradient_audit_completed": False,
        "hscf_gradient_audit_scenes": {},
        "hscf_terminal_contract": "PENDING",
        "hscf_terminal_contract_passed": False,
        "proxy_rows": 0,
        "held_rows": 0,
    }


def _normalized_tx_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise HSCFConfigurationError(f"P1-HSCF {name} must be a TX class sequence")
    order = tuple(str(value).strip() for value in values)
    if not order or len(order) != len(set(order)) or any(not value for value in order):
        raise HSCFConfigurationError(f"P1-HSCF {name} must be non-empty and unique")
    return order


def _positive_count(name: str, value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise HSCFConfigurationError(f"P1-HSCF {name} must be an integer") from exc
    if count <= 0:
        raise HSCFConfigurationError(f"P1-HSCF {name} must be positive")
    return count


def resolve_hscf_local_head_class_binding(
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
    if local_count != FROZEN_HSCF_CLASS_COUNT or len(local) != FROZEN_HSCF_CLASS_COUNT:
        raise HSCFConfigurationError("P1-HSCF requires exactly four local source-TX rows")
    if local != source or checkpoint != source:
        raise HSCFConfigurationError(
            "P1-HSCF local/checkpoint TX order must equal the source-train receipt"
        )
    if local_count != len(local) or checkpoint_count != live_count or live_count != local_count:
        raise HSCFConfigurationError("P1-HSCF local/head class counts must match")
    if set(local).difference(dataset):
        raise HSCFConfigurationError("P1-HSCF local TX labels are absent from dataset order")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(FROZEN_HSCF_CLASS_IDS),
        "expected_tx_class_ids": list(FROZEN_HSCF_CLASS_IDS),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = _canonical_sha256(binding)
    return binding


def remap_hscf_local_labels_to_head_rows(
    local_labels: torch.Tensor, local_to_head_class_ids: Sequence[Any]
) -> torch.Tensor:
    """Map contiguous local source labels through the sealed identity mapping."""

    if not torch.is_tensor(local_labels):
        raise HSCFRuntimeError("P1-HSCF local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if mapping != FROZEN_HSCF_CLASS_IDS:
        raise HSCFRuntimeError("P1-HSCF local-to-head mapping must be local4 identity")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise HSCFRuntimeError("P1-HSCF local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_hscf_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    """Resolve the exact live common classifier head without creating a forward."""

    raw_model = getattr(model, "_orig_mod", model)
    try:
        head = raw_model.id_backbone.cls_head.head
    except AttributeError as exc:
        raise HSCFRuntimeError("P1-HSCF requires model.id_backbone.cls_head.head") from exc
    if not isinstance(head, torch.nn.Module):
        raise HSCFRuntimeError("P1-HSCF exact classifier head is not a module")
    return head


def resolve_hscf_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    weight = getattr(resolve_hscf_classifier_head(model), "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2 or not weight.requires_grad:
        raise HSCFRuntimeError("P1-HSCF classifier head weight must be a trainable rank-2 Parameter")
    return weight


def _validated_labels(labels: torch.Tensor, *, rows: int) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise HSCFRuntimeError("P1-HSCF requires source-L local TX labels")
    values = labels.reshape(-1).long()
    if values.numel() != int(rows) or values.numel() != FROZEN_HSCF_BATCH_SIZE:
        raise HSCFRuntimeError("P1-HSCF requires exactly one drop_last source-L batch of 128 rows")
    if int(values.min().item()) < 0 or int(values.max().item()) >= FROZEN_HSCF_CLASS_COUNT:
        raise HSCFRuntimeError("P1-HSCF source-L labels are outside frozen local4")
    return values


def _validate_view_logits(
    *, view_name: str, output: Mapping[str, Any], labels: torch.Tensor, head_weight: torch.Tensor
) -> torch.Tensor:
    if str(output.get("z_id_key", "")) != "feat_joint":
        raise HSCFRuntimeError(f"P1-HSCF {view_name} z_id_key must be feat_joint")
    logits = output.get("tx_logits")
    z_id = output.get("z_id")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise HSCFRuntimeError(f"P1-HSCF {view_name} tx_logits must be rank-2 raw logits")
    if not torch.is_tensor(z_id) or z_id.ndim != 2:
        raise HSCFRuntimeError(f"P1-HSCF {view_name} z_id must be rank-2 feat_joint")
    if logits.size(0) != labels.numel() or z_id.size(0) != labels.numel():
        raise HSCFRuntimeError(f"P1-HSCF {view_name} rows must align with source L labels")
    if int(logits.size(1)) != FROZEN_HSCF_CLASS_COUNT or int(head_weight.size(0)) != FROZEN_HSCF_CLASS_COUNT:
        raise HSCFRuntimeError(f"P1-HSCF {view_name} head/logit class rows must be local4")
    if int(z_id.size(1)) != int(head_weight.size(1)):
        raise HSCFRuntimeError(f"P1-HSCF {view_name} feat_joint/head dimension binding drifted")
    if not bool(logits.requires_grad) or not bool(z_id.requires_grad):
        raise HSCFRuntimeError(f"P1-HSCF {view_name} requires a live common feat_joint/head path")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise HSCFRuntimeError(f"P1-HSCF {view_name} raw logits are non-finite")
    if not bool(torch.isfinite(z_id.detach()).all().item()):
        raise HSCFRuntimeError(f"P1-HSCF {view_name} feat_joint is non-finite")
    return logits


def validate_hscf_binding(
    *,
    model: torch.nn.Module,
    out_clean: Mapping[str, Any],
    out_leo: Mapping[str, Any],
    tx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
) -> torch.nn.Parameter:
    """Fail closed unless existing clean/LEO forwards bind the exact local4 head."""

    if not isinstance(out_clean, Mapping) or not isinstance(out_leo, Mapping):
        raise HSCFRuntimeError("P1-HSCF requires clean and LEO mapping outputs")
    labels = _validated_labels(tx_labels, rows=int(tx_labels.numel()))
    if tuple(int(value) for value in expected_class_ids) != FROZEN_HSCF_CLASS_IDS:
        raise HSCFRuntimeError("P1-HSCF expected local4 class order is invalid")
    weight = resolve_hscf_classifier_weight(model)
    if not bool(torch.isfinite(weight.detach()).all().item()):
        raise HSCFRuntimeError("P1-HSCF exact classifier head weight is non-finite")
    _validate_view_logits(view_name="clean", output=out_clean, labels=labels, head_weight=weight)
    _validate_view_logits(view_name="leo", output=out_leo, labels=labels, head_weight=weight)
    return weight


def _center_logits(logits: torch.Tensor, *, view_name: str, detach: bool) -> torch.Tensor:
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise HSCFRuntimeError(f"P1-HSCF {view_name} logits must be rank-2")
    if tuple(logits.shape) != (FROZEN_HSCF_BATCH_SIZE, FROZEN_HSCF_CLASS_COUNT):
        raise HSCFRuntimeError(
            "P1-HSCF requires exact [128,4] logits; active-size rescaling is forbidden"
        )
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise HSCFRuntimeError(f"P1-HSCF {view_name} logits are non-finite")
    values = logits.detach().float() if detach else logits.float()
    class_centered = values - values.mean(dim=1, keepdim=True)
    residual = class_centered - class_centered.mean(dim=0, keepdim=True)
    if not bool(torch.isfinite(class_centered.detach()).all().item()):
        raise HSCFRuntimeError(f"P1-HSCF {view_name} class-centered logits are non-finite")
    if not bool(torch.isfinite(residual.detach()).all().item()):
        raise HSCFRuntimeError(f"P1-HSCF {view_name} batch-centered logits are non-finite")
    return residual


def hscf_loss(clean_tx_logits: torch.Tensor, leo_tx_logits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute the one fixed-scale HSCF term from existing raw local4 logits.

    ``clean_tx_logits`` is fully detached before either centering operation.
    ``leo_tx_logits`` remains live through class and batch centering, the exact
    head, and shared encoder.  No active-row or active-class renormalization is
    permitted, so B=128 and K=4 are checked before every calculation.
    """

    clean_residual = _center_logits(clean_tx_logits, view_name="clean", detach=True)
    leo_residual = _center_logits(leo_tx_logits, view_name="leo", detach=False)
    if clean_residual.shape != leo_residual.shape:
        raise HSCFRuntimeError("P1-HSCF clean/LEO residual shapes differ")
    delta = leo_residual - clean_residual
    squared = delta.square()
    if not bool(torch.isfinite(squared.detach()).all().item()):
        raise HSCFRuntimeError("P1-HSCF configuration squared residual is non-finite")
    loss = squared.sum() / float(FROZEN_HSCF_GLOBAL_DENOMINATOR)
    if not bool(torch.isfinite(loss.detach()).item()):
        raise HSCFRuntimeError("P1-HSCF loss is non-finite")
    positive_components = int(squared.detach().gt(0.0).sum().item())
    loss_value = float(loss.detach().item())
    return loss, {
        "rows": FROZEN_HSCF_BATCH_SIZE,
        "class_count": FROZEN_HSCF_CLASS_COUNT,
        "configuration_components": FROZEN_HSCF_GLOBAL_DENOMINATOR,
        "positive_components": positive_components,
        "positive_batch": bool(loss_value > 0.0),
        "loss_sum": loss_value,
        "global_denominator": FROZEN_HSCF_GLOBAL_DENOMINATOR,
        "fixed_scale": 1.0 / float(FROZEN_HSCF_GLOBAL_DENOMINATOR),
        "finite": True,
        "clean_logits_detached": True,
        "leo_logits_live": True,
        "class_projector": "I4_MINUS_11T_OVER_4",
        "batch_centering_rows": FROZEN_HSCF_BATCH_SIZE,
        "training_accumulation_dtype": "float32",
        "zero_logits_legal": True,
        "no_active_renormalization": True,
        "no_rx_day_fold_read": True,
        "uses_new_forward": False,
    }


def add_hscf_to_loss(
    base_loss: torch.Tensor, hscf: Optional[torch.Tensor], config: Optional[HSCFConfig]
) -> torch.Tensor:
    """Add the sole G-arm term; C returns the untouched common base tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if hscf is None:
        raise HSCFRuntimeError("Enabled P1-HSCF requires its auxiliary loss")
    return base_loss + float(config.loss_weight) * hscf


def hscf_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    """Return exact HSCF VJP scopes without changing optimizer state."""

    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise HSCFRuntimeError("P1-HSCF requires model.id_backbone for VJP audit")
    head = resolve_hscf_classifier_head(raw_model)
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or not weight.requires_grad:
        raise HSCFRuntimeError("P1-HSCF exact head weight VJP scope is absent")
    bias = getattr(head, "bias", None)
    if bias is not None and (not isinstance(bias, torch.nn.Parameter) or not bias.requires_grad):
        raise HSCFRuntimeError("P1-HSCF exact head bias must be a trainable Parameter when present")
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
    if not encoder:
        raise HSCFRuntimeError("P1-HSCF shared encoder VJP scope is empty")
    result: Dict[str, Tuple[torch.nn.Parameter, ...]] = {
        "shared_encoder": encoder,
        "head_weight": (weight,),
    }
    if isinstance(bias, torch.nn.Parameter):
        result["head_bias"] = (bias,)
    return result


def _finite_nonzero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.Tensor], *, group_name: str
) -> Dict[str, float]:
    params = tuple(parameters)
    if not params:
        raise HSCFRuntimeError(f"P1-HSCF {group_name} VJP scope is empty")
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    squared_norm = 0.0
    for gradient in gradients:
        if gradient is None:
            raise HSCFRuntimeError(f"P1-HSCF {group_name} VJP is None or detached")
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise HSCFRuntimeError(f"P1-HSCF {group_name} VJP is non-finite")
        value = gradient.detach().double()
        squared_norm += float(torch.sum(value * value).item())
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm) or norm <= 0.0:
        raise HSCFRuntimeError(f"P1-HSCF {group_name} VJP norm is zero or non-finite")
    return {"parameter_count": float(len(params)), "norm": float(norm)}


def _none_or_zero_vjp(
    loss: torch.Tensor,
    parameters: Iterable[torch.Tensor],
    *,
    group_name: str,
    numerical_zero_rel_tolerance: float = 0.0,
) -> Dict[str, Any]:
    params = tuple(parameters)
    if not params:
        return {
            "parameter_count": 0.0,
            "none_parameters": 0.0,
            "zero_parameters": 0.0,
            "nonzero_parameters": 0.0,
            "none_or_zero_expected": True,
        }
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    none_count = 0
    zero_count = 0
    numerical_zero_count = 0
    tolerance = float(numerical_zero_rel_tolerance) * max(1.0, abs(float(loss.detach().item())))
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise HSCFRuntimeError(f"P1-HSCF {group_name} numerical-zero tolerance is invalid")
    for gradient in gradients:
        if gradient is None:
            none_count += 1
            continue
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise HSCFRuntimeError(f"P1-HSCF {group_name} VJP is non-finite")
        if int(torch.count_nonzero(gradient.detach()).item()) != 0:
            max_abs = float(gradient.detach().abs().max().item())
            if not math.isfinite(max_abs) or max_abs > tolerance:
                raise HSCFRuntimeError(f"P1-HSCF {group_name} must have no auxiliary gradient")
            numerical_zero_count += 1
        zero_count += 1
    return {
        "parameter_count": float(len(params)),
        "none_parameters": float(none_count),
        "zero_parameters": float(zero_count),
        "numerical_zero_parameters": float(numerical_zero_count),
        "numerical_zero_tolerance": float(tolerance),
        "nonzero_parameters": 0.0,
        "none_or_zero_expected": True,
    }


def hscf_aux_gradient_audit(
    hscf: torch.Tensor,
    clean_tx_logits: torch.Tensor,
    leo_tx_logits: torch.Tensor,
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
) -> Dict[str, Any]:
    """Audit first-positive raw HSCF VJPs without touching AMP/optimizer/RNG."""

    if not torch.is_tensor(hscf) or hscf.ndim != 0:
        raise HSCFRuntimeError("P1-HSCF VJP audit requires a scalar auxiliary loss")
    if not torch.is_tensor(clean_tx_logits) or not torch.is_tensor(leo_tx_logits):
        raise HSCFRuntimeError("P1-HSCF VJP audit requires raw clean and LEO logits")
    if tuple(clean_tx_logits.shape) != (FROZEN_HSCF_BATCH_SIZE, FROZEN_HSCF_CLASS_COUNT):
        raise HSCFRuntimeError("P1-HSCF VJP clean logits shape drifted")
    if tuple(leo_tx_logits.shape) != (FROZEN_HSCF_BATCH_SIZE, FROZEN_HSCF_CLASS_COUNT):
        raise HSCFRuntimeError("P1-HSCF VJP LEO logits shape drifted")
    required = {"shared_encoder", "head_weight"}
    if not required.issubset(set(parameter_groups)):
        raise HSCFRuntimeError("P1-HSCF VJP audit requires shared_encoder and head_weight scopes")
    result = {
        "leo_raw_logits": _finite_nonzero_vjp(hscf, (leo_tx_logits,), group_name="leo_raw_logits"),
        "shared_encoder": _finite_nonzero_vjp(
            hscf, parameter_groups["shared_encoder"], group_name="shared_encoder"
        ),
        "head_weight": _finite_nonzero_vjp(
            hscf, parameter_groups["head_weight"], group_name="head_weight"
        ),
        "head_bias": _none_or_zero_vjp(
            hscf,
            parameter_groups.get("head_bias", ()),
            group_name="head_bias",
            numerical_zero_rel_tolerance=FROZEN_HSCF_BIAS_VJP_NUMERICAL_ZERO_REL_TOL,
        ),
        "clean_raw_logits": _none_or_zero_vjp(
            hscf, (clean_tx_logits,), group_name="clean_raw_logits"
        ),
        "raw_unscaled": True,
        "diagnostic_only": True,
        "touches_amp_optimizer_rng": False,
        "clean_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        "head_bias_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        "common_l_base_head_input_path": "LIVE_AND_BOUND_SEPARATELY",
    }
    return result


def _validate_none_or_zero_audit(values: Any, *, group_name: str, allow_absent: bool) -> None:
    if not isinstance(values, Mapping):
        raise HSCFRuntimeError(f"P1-HSCF VJP audit lacks {group_name}")
    count = float(values.get("parameter_count", float("nan")))
    none_count = float(values.get("none_parameters", float("nan")))
    zero_count = float(values.get("zero_parameters", float("nan")))
    nonzero_count = float(values.get("nonzero_parameters", float("nan")))
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in (count, none_count, zero_count, nonzero_count))
        or none_count + zero_count != count
        or nonzero_count != 0.0
        or values.get("none_or_zero_expected") is not True
        or (not allow_absent and count <= 0.0)
    ):
        raise HSCFRuntimeError(f"P1-HSCF {group_name} none-or-zero VJP contract failed")


def update_hscf_gradient_audit_receipt(
    receipt: Mapping[str, Any], audit: Mapping[str, Any], *, scenario: str
) -> Dict[str, Any]:
    """Seal one first-positive raw VJP audit for each frozen LEO scene."""

    result = dict(receipt)
    if str(scenario) not in FROZEN_HSCF_SCENARIOS:
        raise HSCFRuntimeError("P1-HSCF VJP audit scenario is outside clear/low/rain")
    prior = {str(key): dict(value) for key, value in dict(result.get("hscf_gradient_audit_scenes", {})).items()}
    if str(scenario) in prior:
        raise HSCFRuntimeError("P1-HSCF per-scene VJP audit may run only once")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
        or audit.get("clean_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        or audit.get("head_bias_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
    ):
        raise HSCFRuntimeError("P1-HSCF VJP audit semantics drifted")
    for group_name in ("leo_raw_logits", "shared_encoder", "head_weight"):
        values = audit.get(group_name)
        if not isinstance(values, Mapping):
            raise HSCFRuntimeError(f"P1-HSCF VJP audit lacks {group_name}")
        count = float(values.get("parameter_count", 0.0))
        norm = float(values.get("norm", float("nan")))
        if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
            raise HSCFRuntimeError(f"P1-HSCF {group_name} VJP is zero or non-finite")
    _validate_none_or_zero_audit(audit.get("clean_raw_logits"), group_name="clean_raw_logits", allow_absent=False)
    _validate_none_or_zero_audit(audit.get("head_bias"), group_name="head_bias", allow_absent=True)
    prior[str(scenario)] = dict(audit)
    result["hscf_gradient_audit_attempted"] = True
    result["hscf_gradient_audit_scenes"] = prior
    result["hscf_gradient_audit_completed"] = set(prior) == set(FROZEN_HSCF_SCENARIOS)
    return result


def bind_hscf_source_data_order(
    receipt: Mapping[str, Any], source_split_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind only the source-L physical-order receipts; RX/day are not read."""

    result = dict(receipt)
    source = dict(source_split_receipt or {})
    labeled_sha = str(source.get("labeled_indices_sha256", "") or "")
    manifest_sha = str(source.get("split_manifest_sha256", "") or "")
    if len(labeled_sha) != 64 or len(manifest_sha) != 64:
        raise HSCFConfigurationError(
            "P1-HSCF requires labeled-index and source-split SHA256 receipts"
        )
    result["source_labeled_indices_sha256"] = labeled_sha
    result["source_split_manifest_sha256"] = manifest_sha
    result["source_labeled_provenance"] = "SOURCE_SPLIT_RECEIPT_L_PHYSICAL_ORDER_ONLY_RX_DAY_NOT_READ"
    return result


def _as_plain_list(values: Any) -> list[Any]:
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return []


def update_hscf_common_batch_sequence_receipt(
    receipt: Mapping[str, Any],
    *,
    epoch: int,
    batch_index: int,
    scenario: str,
    source_tx_labels: torch.Tensor,
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Chain C/G-identical source-L physical order and scene receipts."""

    result = dict(receipt)
    expected = FROZEN_HSCF_SCENARIOS[(int(epoch) + int(batch_index) - 2) % 3]
    if str(scenario) != expected:
        raise HSCFRuntimeError("P1-HSCF common LEO scenario sequence drifted")
    labels = _validated_labels(source_tx_labels, rows=int(source_tx_labels.numel())).detach()
    if metadata is None:
        raise HSCFRuntimeError("P1-HSCF common sequence requires opaque physical metadata")
    opaque_ids = _as_plain_list(metadata.get("base_index"))
    if len(opaque_ids) != FROZEN_HSCF_BATCH_SIZE:
        opaque_ids = _as_plain_list(metadata.get("sig_i"))
    if len(opaque_ids) != FROZEN_HSCF_BATCH_SIZE:
        raise HSCFRuntimeError("P1-HSCF physical batch sequence metadata is incomplete")
    event = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "scenario": str(scenario),
        "same_physical_clean_leo": True,
        "same_order_clean_leo": True,
        "rows": [[str(opaque), int(label)] for opaque, label in zip(opaque_ids, labels.cpu().tolist())],
        "fixed_batch_size": FROZEN_HSCF_BATCH_SIZE,
        "fixed_local_class_count": FROZEN_HSCF_CLASS_COUNT,
        "global_denominator": FROZEN_HSCF_GLOBAL_DENOMINATOR,
    }
    prior = str(result.get("common_batch_sequence_sha256", "") or "")
    if not prior:
        prior = str(result.get("source_labeled_indices_sha256", "") or "")
    if len(prior) != 64:
        raise HSCFRuntimeError("P1-HSCF common batch sequence lacks source data-order SHA256")
    result["common_batch_sequence_sha256"] = hashlib.sha256(
        (prior + "\n" + json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()
    result["common_batch_sequence_batches"] = int(result.get("common_batch_sequence_batches", 0)) + 1
    result["common_batch_sequence_rows"] = int(result.get("common_batch_sequence_rows", 0)) + FROZEN_HSCF_BATCH_SIZE
    scenario_batches = {
        str(key): int(value) for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if set(scenario_batches) != set(FROZEN_HSCF_SCENARIOS):
        raise HSCFRuntimeError("P1-HSCF common scenario receipt is malformed")
    scenario_batches[str(scenario)] += 1
    result["common_scenario_batches"] = scenario_batches
    common_scenes = {
        str(key): dict(value) for key, value in dict(result.get("hscf_common_scenes", {})).items()
    }
    scene = dict(common_scenes.get(str(scenario), {"batches": 0, "rows": 0}))
    scene["batches"] = int(scene.get("batches", 0)) + 1
    scene["rows"] = int(scene.get("rows", 0)) + FROZEN_HSCF_BATCH_SIZE
    common_scenes[str(scenario)] = scene
    result["hscf_common_scenes"] = common_scenes
    events = list(result.get("hscf_common_batches", []))
    events.append({key: event[key] for key in ("epoch", "batch_index", "scenario", "same_physical_clean_leo", "same_order_clean_leo", "fixed_batch_size", "fixed_local_class_count", "global_denominator")})
    result["hscf_common_batches"] = events
    return result


def bind_hscf_optimizer_initial_state(
    receipt: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Seal the new AdamW state before the first backward call."""

    result = dict(receipt)
    optimizer_type = type(optimizer).__name__
    if optimizer_type != FROZEN_HSCF_OPTIMIZER_TYPE:
        raise HSCFConfigurationError(
            "P1-HSCF requires optimizer_type=AdamW, got " + (optimizer_type or "<empty>")
        )
    state = optimizer.state_dict()
    if dict(state.get("state", {})):
        raise HSCFConfigurationError("P1-HSCF requires a new AdamW state")
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


def update_hscf_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    scenario: str,
    epoch: int,
    batch_index: int,
) -> Dict[str, Any]:
    """Accumulate G-only HSCF evidence after same-physical common receipt sealing."""

    result = dict(receipt)
    if str(result.get("schema", "")) != HSCF_RECEIPT_SCHEMA:
        raise HSCFRuntimeError("P1-HSCF receipt schema is invalid")
    if result.get("enabled") is not True:
        raise HSCFRuntimeError("P1-HSCF auxiliary receipt update is G-arm only")
    if str(scenario) not in FROZEN_HSCF_SCENARIOS:
        raise HSCFRuntimeError("P1-HSCF scenario is outside frozen clear/low/rain cycle")
    for key in (
        "finite",
        "clean_logits_detached",
        "leo_logits_live",
        "zero_logits_legal",
        "no_active_renormalization",
        "no_rx_day_fold_read",
    ):
        if batch_info.get(key) is not True:
            raise HSCFRuntimeError(f"P1-HSCF batch semantic receipt drifted: {key}")
    if batch_info.get("training_accumulation_dtype") != "float32":
        raise HSCFRuntimeError("P1-HSCF batch accumulation dtype drifted")
    if int(batch_info.get("rows", -1)) != FROZEN_HSCF_BATCH_SIZE or int(batch_info.get("class_count", -1)) != FROZEN_HSCF_CLASS_COUNT:
        raise HSCFRuntimeError("P1-HSCF G batch shape receipt drifted")
    if int(batch_info.get("global_denominator", -1)) != FROZEN_HSCF_GLOBAL_DENOMINATOR:
        raise HSCFRuntimeError("P1-HSCF global denominator drifted")
    scale = float(batch_info.get("fixed_scale", float("nan")))
    if not math.isfinite(scale) or abs(scale - 1.0 / float(FROZEN_HSCF_GLOBAL_DENOMINATOR)) > _TOLERANCE:
        raise HSCFRuntimeError("P1-HSCF fixed scale drifted")
    loss_sum = float(batch_info.get("loss_sum", float("nan")))
    positive_components = int(batch_info.get("positive_components", -1))
    positive_batch = bool(batch_info.get("positive_batch", False))
    if (
        not math.isfinite(loss_sum)
        or positive_components < 0
        or positive_components > FROZEN_HSCF_GLOBAL_DENOMINATOR
        or positive_batch != bool(loss_sum > 0.0)
    ):
        raise HSCFRuntimeError("P1-HSCF G batch loss counters do not close")
    common_events = list(result.get("hscf_common_batches", []))
    if not common_events:
        raise HSCFRuntimeError("P1-HSCF G batch lacks its common C/G coverage receipt")
    common_event = dict(common_events[-1])
    if (
        int(common_event.get("epoch", -1)) != int(epoch)
        or int(common_event.get("batch_index", -1)) != int(batch_index)
        or str(common_event.get("scenario", "")) != str(scenario)
        or common_event.get("same_physical_clean_leo") is not True
        or common_event.get("same_order_clean_leo") is not True
        or int(common_event.get("fixed_batch_size", -1)) != FROZEN_HSCF_BATCH_SIZE
        or int(common_event.get("fixed_local_class_count", -1)) != FROZEN_HSCF_CLASS_COUNT
        or int(common_event.get("global_denominator", -1)) != FROZEN_HSCF_GLOBAL_DENOMINATOR
    ):
        raise HSCFRuntimeError("P1-HSCF G/common physical/order/scale binding drifted")
    scenes = {str(key): dict(value) for key, value in dict(result.get("hscf_scenes", {})).items()}
    scene = dict(
        scenes.get(
            str(scenario),
            {"batches": 0, "rows": 0, "positive_batches": 0, "positive_components": 0, "loss_sum": 0.0},
        )
    )
    scene["batches"] = int(scene.get("batches", 0)) + 1
    scene["rows"] = int(scene.get("rows", 0)) + FROZEN_HSCF_BATCH_SIZE
    scene["positive_batches"] = int(scene.get("positive_batches", 0)) + int(positive_batch)
    scene["positive_components"] = int(scene.get("positive_components", 0)) + positive_components
    scene["loss_sum"] = float(scene.get("loss_sum", 0.0)) + loss_sum
    scenes[str(scenario)] = scene
    result["hscf_scenes"] = scenes
    events = list(result.get("hscf_g_batch_aux", []))
    events.append(
        {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "scenario": str(scenario),
            "rows": FROZEN_HSCF_BATCH_SIZE,
            "positive_batch": positive_batch,
            "positive_components": positive_components,
            "loss_sum": loss_sum,
            "global_denominator": FROZEN_HSCF_GLOBAL_DENOMINATOR,
        }
    )
    result["hscf_g_batch_aux"] = events
    result["hscf_batches"] = int(result.get("hscf_batches", 0)) + 1
    result["hscf_total_rows"] = int(result.get("hscf_total_rows", 0)) + FROZEN_HSCF_BATCH_SIZE
    result["hscf_positive_batches"] = int(result.get("hscf_positive_batches", 0)) + int(positive_batch)
    result["hscf_positive_components"] = int(result.get("hscf_positive_components", 0)) + positive_components
    result["hscf_loss_sum"] = float(result.get("hscf_loss_sum", 0.0)) + loss_sum
    return result


def _validate_common_terminal_contract(result: Mapping[str, Any]) -> None:
    if str(result.get("schema", "")) != HSCF_RECEIPT_SCHEMA:
        raise HSCFRuntimeError("P1-HSCF terminal receipt schema is invalid")
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
            raise HSCFRuntimeError(f"P1-HSCF terminal receipt lacks {key}")
    if str(result.get("checkpoint_role", "") or "") != "training_final_only":
        raise HSCFRuntimeError("P1-HSCF requires training_final_only warm start")
    if result.get("optimizer_state_restored") is not False or result.get("rng_state_restored") is not False:
        raise HSCFRuntimeError("P1-HSCF optimizer/RNG restoration is forbidden")
    if str(result.get("optimizer_type", "")) != FROZEN_HSCF_OPTIMIZER_TYPE:
        raise HSCFRuntimeError("P1-HSCF terminal optimizer_type must be AdamW")
    if result.get("optimizer_initial_state_empty") is not True:
        raise HSCFRuntimeError("P1-HSCF missing new AdamW initial-state receipt")
    if result.get("amp_contract") != "COMMON_TRAINER_AMP_ENABLED":
        raise HSCFRuntimeError("P1-HSCF terminal AMP contract drifted")
    if result.get("common_l_base_head_input_path_verified") is not True:
        raise HSCFRuntimeError("P1-HSCF common L_base exact head-input path is not verified")
    if (
        int(result.get("fixed_batch_size", -1)) != FROZEN_HSCF_BATCH_SIZE
        or int(result.get("fixed_local_class_count", -1)) != FROZEN_HSCF_CLASS_COUNT
        or int(result.get("loss_global_denominator", -1)) != FROZEN_HSCF_GLOBAL_DENOMINATOR
        or result.get("common_loader_drop_last") is not True
    ):
        raise HSCFRuntimeError("P1-HSCF terminal fixed B/local4/denominator/drop_last drifted")
    batches = int(result.get("common_batch_sequence_batches", 0))
    rows = int(result.get("common_batch_sequence_rows", 0))
    scene_batches = {str(key): int(value) for key, value in dict(result.get("common_scenario_batches", {})).items()}
    common_scenes = {str(key): dict(value) for key, value in dict(result.get("hscf_common_scenes", {})).items()}
    if (
        batches <= 0
        or rows != batches * FROZEN_HSCF_BATCH_SIZE
        or set(scene_batches) != set(FROZEN_HSCF_SCENARIOS)
        or set(common_scenes) != set(FROZEN_HSCF_SCENARIOS)
        or any(value <= 0 for value in scene_batches.values())
    ):
        raise HSCFRuntimeError("P1-HSCF common batch/scenario receipt is incomplete")
    for scenario in FROZEN_HSCF_SCENARIOS:
        scene = common_scenes[scenario]
        if (
            int(scene.get("batches", -1)) != scene_batches[scenario]
            or int(scene.get("rows", -1)) != scene_batches[scenario] * FROZEN_HSCF_BATCH_SIZE
        ):
            raise HSCFRuntimeError("P1-HSCF common scene rows do not close")
    events = list(result.get("hscf_common_batches", []))
    if len(events) != batches:
        raise HSCFRuntimeError("P1-HSCF terminal common batch receipt is incomplete")
    for event in events:
        if (
            event.get("same_physical_clean_leo") is not True
            or event.get("same_order_clean_leo") is not True
            or int(event.get("fixed_batch_size", -1)) != FROZEN_HSCF_BATCH_SIZE
            or int(event.get("fixed_local_class_count", -1)) != FROZEN_HSCF_CLASS_COUNT
            or int(event.get("global_denominator", -1)) != FROZEN_HSCF_GLOBAL_DENOMINATOR
        ):
            raise HSCFRuntimeError("P1-HSCF terminal common same-physical/order receipt drifted")


def _validate_nonzero_audit(values: Any, *, group_name: str) -> None:
    if not isinstance(values, Mapping):
        raise HSCFRuntimeError(f"P1-HSCF terminal VJP lacks {group_name}")
    count = float(values.get("parameter_count", 0.0))
    norm = float(values.get("norm", float("nan")))
    if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
        raise HSCFRuntimeError(f"P1-HSCF terminal {group_name} VJP is zero or non-finite")


def validate_hscf_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless common C/G and G-only per-scene HSCF evidence close."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    _validate_common_terminal_contract(result)
    enabled = result.get("enabled")
    if enabled is not True and enabled is not False:
        raise HSCFRuntimeError("P1-HSCF terminal enabled flag must be strict bool")
    if enabled is False:
        zero_keys = ("hscf_batches", "hscf_total_rows", "hscf_positive_batches", "hscf_positive_components")
        if any(int(result.get(key, 0)) != 0 for key in zero_keys) or abs(float(result.get("hscf_loss_sum", 0.0))) > _TOLERANCE:
            raise HSCFRuntimeError("P1-HSCF C arm must retain zero auxiliary counters")
        if any(bool(result.get(key)) for key in ("hscf_scenes", "hscf_g_batch_aux", "hscf_gradient_audit_scenes")) or bool(result.get("hscf_gradient_audit_attempted", False)) or bool(result.get("hscf_gradient_audit_completed", False)):
            raise HSCFRuntimeError("P1-HSCF C arm must retain N/A-or-zero auxiliary fields")
        result["hscf_terminal_contract"] = "CONTROL_ARM_COMMON_SAME_PHYSICAL_ORDER_B128_LOCAL4_DENOM512_SCENE_COVERAGE_AUX_NA_OR_ZERO"
        result["hscf_terminal_contract_passed"] = True
        return result
    scenes = {str(key): dict(value) for key, value in dict(result.get("hscf_scenes", {})).items()}
    common_scenes = {str(key): dict(value) for key, value in dict(result.get("hscf_common_scenes", {})).items()}
    if set(scenes) != set(FROZEN_HSCF_SCENARIOS):
        raise HSCFRuntimeError("P1-HSCF terminal G scene coverage is incomplete")
    total_rows = total_positive_batches = total_positive_components = 0
    total_loss = 0.0
    for scenario in FROZEN_HSCF_SCENARIOS:
        scene = scenes[scenario]
        common = common_scenes[scenario]
        batches = int(scene.get("batches", -1))
        rows = int(scene.get("rows", -1))
        positive_batches = int(scene.get("positive_batches", -1))
        positive_components = int(scene.get("positive_components", -1))
        loss_sum = float(scene.get("loss_sum", float("nan")))
        if (
            batches <= 0
            or rows != batches * FROZEN_HSCF_BATCH_SIZE
            or batches != int(common.get("batches", -2))
            or rows != int(common.get("rows", -2))
            or positive_batches <= 0
            or positive_batches > batches
            or positive_components <= 0
            or positive_components > batches * FROZEN_HSCF_GLOBAL_DENOMINATOR
            or not math.isfinite(loss_sum)
        ):
            raise HSCFRuntimeError(
                "P1-HSCF terminal each scene requires common C/G closure and a positive HSCF batch"
            )
        total_rows += rows
        total_positive_batches += positive_batches
        total_positive_components += positive_components
        total_loss += loss_sum
    events = list(result.get("hscf_g_batch_aux", []))
    if len(events) != int(result.get("hscf_batches", -1)):
        raise HSCFRuntimeError("P1-HSCF terminal G auxiliary batch receipt is incomplete")
    if (
        int(result.get("hscf_batches", -1)) != int(result.get("common_batch_sequence_batches", -2))
        or int(result.get("hscf_total_rows", -1)) != int(result.get("common_batch_sequence_rows", -2))
        or total_rows != int(result.get("common_batch_sequence_rows", -2))
        or int(result.get("hscf_positive_batches", -1)) != total_positive_batches
        or int(result.get("hscf_positive_components", -1)) != total_positive_components
        or total_positive_batches <= 0
        or total_positive_components <= 0
        or not _float32_ledger_close(float(result.get("hscf_loss_sum", float("nan"))), total_loss)
    ):
        raise HSCFRuntimeError("P1-HSCF terminal G batch/positive/loss counters do not close")
    audits = {str(key): dict(value) for key, value in dict(result.get("hscf_gradient_audit_scenes", {})).items()}
    if not bool(result.get("hscf_gradient_audit_completed", False)) or set(audits) != set(FROZEN_HSCF_SCENARIOS):
        raise HSCFRuntimeError("P1-HSCF terminal per-scene first-positive raw VJP audit is incomplete")
    for scenario in FROZEN_HSCF_SCENARIOS:
        audit = audits[scenario]
        if (
            audit.get("raw_unscaled") is not True
            or audit.get("diagnostic_only") is not True
            or audit.get("touches_amp_optimizer_rng") is not False
            or audit.get("clean_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
            or audit.get("head_bias_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        ):
            raise HSCFRuntimeError("P1-HSCF terminal per-scene VJP semantics drifted")
        for group_name in ("leo_raw_logits", "shared_encoder", "head_weight"):
            _validate_nonzero_audit(audit.get(group_name), group_name=group_name)
        _validate_none_or_zero_audit(audit.get("clean_raw_logits"), group_name="clean_raw_logits", allow_absent=False)
        _validate_none_or_zero_audit(audit.get("head_bias"), group_name="head_bias", allow_absent=True)
    result["hscf_terminal_contract"] = "FORMAL_COMMON_C_G_SAME_PHYSICAL_ORDER_B128_LOCAL4_DENOM512_CLEAR_LOW_RAIN_WITH_G_ONLY_HSCF_AND_PER_SCENE_RAW_LOGIT_ENCODER_HEAD_WEIGHT_VJP"
    result["hscf_terminal_contract_passed"] = True
    return result


def _failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "vjp" in message or "gradient" in message or "head" in message:
        return "HSCF_AUX_GRADIENT_OR_HEAD_PATH_FAILURE"
    if "non-finite" in message or "nonfinite" in message:
        return "HSCF_NONFINITE"
    if "128" in message or "512" in message or "local4" in message or "batch" in message:
        return "HSCF_FIXED_SCALE_OR_BATCH_FAILURE"
    if "sequence" in message or "receipt" in message or "coverage" in message:
        return "HSCF_RECEIPT_CLOSURE_FAILURE"
    if "binding" in message or "logit" in message or "physical" in message:
        return "HSCF_BINDING_FAILURE"
    return "HSCF_RUNTIME_FAILURE"


def write_hscf_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free fail-closed record for the HSCF arm."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise HSCFRuntimeError(f"P1-HSCF failure receipt output directory is absent: {target_dir}")
    payload = {
        "schema": "cvs.phase1.hscf_failure_receipt.v1",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage or ""),
        "exception_type": type(error).__name__,
        "exception_fingerprint": _failure_fingerprint(error),
        "message": str(error),
        "receipt": dict(receipt),
    }
    target = target_dir / "phase1_hscf_failure_receipt.json"
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = mkstemp(prefix=".hscf_failure_receipt.", suffix=".tmp", dir=str(target_dir))
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


def strict_hscf_warm_start(
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
        raise HSCFConfigurationError("Frozen P1-HSCF warm-start requires model state, path, and SHA256")
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise HSCFConfigurationError(
            f"Frozen P1-HSCF strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise HSCFConfigurationError(
            "Frozen P1-HSCF strict baseline model-key mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    if str(checkpoint_role or "") != "training_final_only":
        raise HSCFConfigurationError("Frozen P1-HSCF requires baseline checkpoint_role=training_final_only")
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
