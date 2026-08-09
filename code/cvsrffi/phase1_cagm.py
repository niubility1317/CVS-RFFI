"""Frozen P1-CAGM continuation contract for Phase1 source-only DG.

P1-CAGM (Clean-Anchored Class Geometry Matching) keeps the common GeoSat-C
clean and one-LEO forwards intact.  Its G arm adds only a detached-clean,
LEO-gradient class-geometry match
on the existing ``feat_joint`` feature.  It deliberately contains no state,
threshold, head loss, extra view, RX/day/domain input, or postfreeze action.
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


FROZEN_CAGM_LAMBDA = 0.02
FROZEN_CAGM_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_CAGM_CLASS_IDS = (0, 1, 2, 3)
FROZEN_CAGM_TERM_DIVISOR = 10
FROZEN_CAGM_OPTIMIZER_TYPE = "AdamW"
CAGM_RECEIPT_SCHEMA = "cvs.phase1.cagm_receipt.v2"
_TOLERANCE = 1e-12


class CAGMConfigurationError(ValueError):
    """Raised when a frozen P1-CAGM C/G configuration drifts."""


class CAGMRuntimeError(RuntimeError):
    """Raised when a P1-CAGM runtime or receipt contract cannot be proved."""


@dataclass(frozen=True)
class CAGMConfig:
    """Immutable P1-CAGM controls consumed by the common training loop."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise CAGMConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
        raise CAGMConfigurationError(
            f"Frozen P1-CAGM requires {name}={expected:.12g}, got {actual!r}"
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
        raise CAGMConfigurationError(
            "Frozen P1-CAGM forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    raw = str(value or "").strip()
    scenarios = tuple(
        item.strip().lower().replace("-", "_")
        for item in raw.split(",")
        if item.strip()
    )
    if scenarios != FROZEN_CAGM_SCENARIOS:
        raise CAGMConfigurationError(
            "Frozen P1-CAGM requires --sat_train_scenarios "
            + ",".join(FROZEN_CAGM_SCENARIOS)
        )
    return scenarios


def validate_cagm_args(args: Any) -> CAGMConfig:
    """Validate the frozen common-base C/G contract before data are loaded."""

    frozen_mode = _bool_arg(args, "phase1_cagm_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_cagm_enabled", False)
    loss_weight = _float_arg(args, "lambda_cagm", 0.0)
    if not frozen_mode and not enabled:
        return CAGMConfig(False, False, 0.0)
    if enabled and not frozen_mode:
        raise CAGMConfigurationError(
            "--phase1_cagm_enabled requires --phase1_cagm_frozen_mode true"
        )
    _require_close("lambda_cagm", loss_weight, FROZEN_CAGM_LAMBDA if enabled else 0.0)
    if bool(getattr(args, "from_scratch", True)):
        raise CAGMConfigurationError("Frozen P1-CAGM requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise CAGMConfigurationError("Frozen P1-CAGM requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise CAGMConfigurationError("Frozen P1-CAGM must train the shared feat_joint encoder")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise CAGMConfigurationError("Frozen P1-CAGM requires --id_feature_key feat_joint")
    if int(getattr(args, "epochs", 0)) != 40 or int(getattr(args, "label_epochs", 0)) != 40:
        raise CAGMConfigurationError("Frozen P1-CAGM requires exactly 40 labeled epochs")
    if int(getattr(args, "pseudo_epochs", 0)) != 0:
        raise CAGMConfigurationError("Frozen P1-CAGM forbids pseudo epochs")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise CAGMConfigurationError("Frozen P1-CAGM requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise CAGMConfigurationError("Frozen P1-CAGM remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise CAGMConfigurationError("Frozen P1-CAGM requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise CAGMConfigurationError("Frozen P1-CAGM requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise CAGMConfigurationError("Frozen P1-CAGM forbids --sat_view_schedule overrides")
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise CAGMConfigurationError("Frozen P1-CAGM requires non-concatenated single-LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise CAGMConfigurationError("Frozen P1-CAGM permits only source_known_train L updates")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise CAGMConfigurationError("Frozen P1-CAGM forbids RX/day-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise CAGMConfigurationError("Frozen P1-CAGM permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise CAGMConfigurationError("Frozen P1-CAGM forbids a reject head")
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
    return CAGMConfig(True, enabled, loss_weight)


def cagm_config_receipt(config: CAGMConfig) -> Dict[str, Any]:
    """Create a data-free receipt skeleton for either frozen arm."""

    return {
        "schema": CAGM_RECEIPT_SCHEMA,
        "method": "P1_CAGM",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "loss_rule": "DETACHED_CLEAN_CLASSWISE_ANGULAR_RADIUS_AND_GRAM_MATCH_ON_LEO_feat_joint",
        "loss_divisor": FROZEN_CAGM_TERM_DIVISOR,
        "z_id_key": "feat_joint",
        "clean_statistics_detached": True,
        "joint_zero_mask_aux_only": bool(config.enabled),
        "joint_zero_mask_aux_only_semantics": (
            "G_AUXILIARY_ONLY_BASE_RETAINS_FULL_BATCH"
            if bool(config.enabled)
            else "C_CONTROL_NOT_APPLICABLE"
        ),
        "aux_gradient_scope": "LEO_SHARED_ENCODER_ONLY_HEAD_NONE_OR_ZERO_EXPECTED",
        "satellite_scenarios": list(FROZEN_CAGM_SCENARIOS),
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
        "uses_ema_or_state": False,
        "uses_threshold": False,
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
        "optimizer_type": "",
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
        "common_scenario_batches": {scenario: 0 for scenario in FROZEN_CAGM_SCENARIOS},
        "cagm_batches": 0,
        "cagm_total_rows": 0,
        "cagm_valid_rows": 0,
        "cagm_clean_zero_rows": 0,
        "cagm_leo_zero_rows": 0,
        "cagm_union_zero_rows": 0,
        "cagm_both_zero_rows": 0,
        "cagm_scenes": {},
        "cagm_radius_terms": {},
        "cagm_gram_terms": {},
        "cagm_gradient_audit_attempted": False,
        "cagm_gradient_audit_completed": False,
        "cagm_gradient_audit": {},
        "cagm_terminal_contract": "PENDING",
        "cagm_terminal_contract_passed": False,
        "proxy_rows": 0,
        "held_rows": 0,
    }


def _normalized_tx_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CAGMConfigurationError(f"P1-CAGM {name} must be a TX class sequence")
    order = tuple(str(value).strip() for value in values)
    if not order or len(order) != len(set(order)) or any(not value for value in order):
        raise CAGMConfigurationError(f"P1-CAGM {name} must be non-empty and unique")
    return order


def _positive_count(name: str, value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise CAGMConfigurationError(f"P1-CAGM {name} must be an integer") from exc
    if count <= 0:
        raise CAGMConfigurationError(f"P1-CAGM {name} must be positive")
    return count


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_cagm_local_head_class_binding(
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
        raise CAGMConfigurationError("P1-CAGM requires exactly four local source-TX rows")
    if local != source or checkpoint != source:
        raise CAGMConfigurationError(
            "P1-CAGM local/checkpoint TX order must equal the source-train receipt"
        )
    if local_count != len(local) or checkpoint_count != live_count or live_count != local_count:
        raise CAGMConfigurationError("P1-CAGM local/head class counts must match")
    if set(local).difference(dataset):
        raise CAGMConfigurationError("P1-CAGM local TX labels are absent from dataset order")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(FROZEN_CAGM_CLASS_IDS),
        "expected_tx_class_ids": list(FROZEN_CAGM_CLASS_IDS),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = _canonical_sha256(binding)
    return binding


def remap_cagm_local_labels_to_head_rows(
    local_labels: torch.Tensor,
    local_to_head_class_ids: Sequence[Any],
) -> torch.Tensor:
    """Map contiguous local source labels through the sealed identity mapping."""

    if not torch.is_tensor(local_labels):
        raise CAGMRuntimeError("P1-CAGM local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if mapping != FROZEN_CAGM_CLASS_IDS:
        raise CAGMRuntimeError("P1-CAGM local-to-head mapping must be local4 identity")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise CAGMRuntimeError("P1-CAGM local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_cagm_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    """Resolve the exact common head solely for the expected zero-gradient audit."""

    raw_model = getattr(model, "_orig_mod", model)
    try:
        head = raw_model.id_backbone.cls_head.head
    except AttributeError as exc:
        raise CAGMRuntimeError("P1-CAGM requires model.id_backbone.cls_head.head") from exc
    if not isinstance(head, torch.nn.Module):
        raise CAGMRuntimeError("P1-CAGM exact classifier head is not a module")
    if not tuple(parameter for parameter in head.parameters() if parameter.requires_grad):
        raise CAGMRuntimeError("P1-CAGM exact classifier head has no trainable parameter")
    return head


def resolve_cagm_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    """Resolve the exact local4 head weight for strict binding checks."""

    weight = getattr(resolve_cagm_classifier_head(model), "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise CAGMRuntimeError("P1-CAGM classifier head weight must be a rank-2 Parameter")
    return weight


def _validate_feature_view(
    *, view_name: str, output: Mapping[str, Any], labels: torch.Tensor
) -> torch.Tensor:
    if str(output.get("z_id_key", "")) != "feat_joint":
        raise CAGMRuntimeError(f"P1-CAGM {view_name} z_id_key must be feat_joint")
    z_id = output.get("z_id")
    if not torch.is_tensor(z_id) or z_id.ndim != 2:
        raise CAGMRuntimeError(f"P1-CAGM {view_name} z_id must be rank-2 feat_joint")
    if z_id.size(0) != labels.numel() or z_id.size(1) <= 0:
        raise CAGMRuntimeError(f"P1-CAGM {view_name} rows must align with source L labels")
    if not bool(z_id.requires_grad):
        raise CAGMRuntimeError(f"P1-CAGM {view_name} z_id must retain a live gradient path")
    return z_id


def validate_cagm_binding(
    *,
    model: torch.nn.Module,
    out_clean: Mapping[str, Any],
    out_leo: Mapping[str, Any],
    tx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fail closed unless both existing forwards expose one live ``feat_joint``."""

    if not isinstance(out_clean, Mapping) or not isinstance(out_leo, Mapping):
        raise CAGMRuntimeError("P1-CAGM requires clean and LEO mapping outputs")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() == 0:
        raise CAGMRuntimeError("P1-CAGM requires a non-empty source L batch")
    if tuple(int(value) for value in expected_class_ids) != FROZEN_CAGM_CLASS_IDS:
        raise CAGMRuntimeError("P1-CAGM expected local4 class order is invalid")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise CAGMRuntimeError("P1-CAGM source labels do not bind to local4 head rows")
    weight = resolve_cagm_classifier_weight(model)
    if int(weight.size(0)) != 4 or not bool(torch.isfinite(weight.detach()).all().item()):
        raise CAGMRuntimeError("P1-CAGM exact classifier head binding is invalid")
    clean = _validate_feature_view(view_name="clean", output=out_clean, labels=labels)
    leo = _validate_feature_view(view_name="leo", output=out_leo, labels=labels)
    if clean.shape != leo.shape:
        raise CAGMRuntimeError("P1-CAGM clean and LEO feat_joint shapes must match")
    return clean, leo


def _float32_row_norms(z: torch.Tensor, *, view_name: str) -> torch.Tensor:
    if not bool(torch.isfinite(z.detach()).all().item()):
        raise CAGMRuntimeError(f"P1-CAGM {view_name} z contains non-finite values")
    norms = torch.linalg.vector_norm(z.float(), ord=2, dim=1)
    if not bool(torch.isfinite(norms.detach()).all().item()):
        raise CAGMRuntimeError(f"P1-CAGM {view_name} float32 row norms are non-finite")
    return norms


def _normalised_class_anchor(
    h: torch.Tensor, *, class_id: int, view_name: str
) -> torch.Tensor:
    centroid = h.mean(dim=0)
    if not bool(torch.isfinite(centroid.detach()).all().item()):
        raise CAGMRuntimeError(f"P1-CAGM {view_name} class {class_id} centroid is non-finite")
    centroid_norm = torch.linalg.vector_norm(centroid.float(), ord=2)
    if not bool(torch.isfinite(centroid_norm.detach()).item()) or float(centroid_norm.detach()) <= 0.0:
        raise CAGMRuntimeError(
            f"P1-CAGM {view_name} class {class_id} centroid norm is zero or non-finite"
        )
    anchor = centroid / centroid_norm
    if not bool(torch.isfinite(anchor.detach()).all().item()):
        raise CAGMRuntimeError(f"P1-CAGM {view_name} class {class_id} anchor is non-finite")
    return anchor


def _term_key_pair(left: int, right: int) -> str:
    return f"tx{left}|tx{right}"


def cagm_loss(
    clean_z: torch.Tensor,
    leo_z: torch.Tensor,
    source_tx_labels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute detached-clean local4 angular radius and Gram matching exactly."""

    if not torch.is_tensor(clean_z) or not torch.is_tensor(leo_z):
        raise CAGMRuntimeError("P1-CAGM requires tensor clean and LEO feat_joint inputs")
    if clean_z.ndim != 2 or leo_z.ndim != 2 or clean_z.shape != leo_z.shape:
        raise CAGMRuntimeError("P1-CAGM clean and LEO feat_joint inputs must be equal rank-2 shapes")
    labels = source_tx_labels.reshape(-1).long()
    if labels.numel() != clean_z.size(0) or labels.numel() == 0:
        raise CAGMRuntimeError("P1-CAGM feature rows and source labels do not align")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise CAGMRuntimeError("P1-CAGM source labels are outside local4")
    clean_norm = _float32_row_norms(clean_z, view_name="clean")
    leo_norm = _float32_row_norms(leo_z, view_name="leo")
    clean_nonzero = clean_norm > 0.0
    leo_nonzero = leo_norm > 0.0
    valid = clean_nonzero & leo_nonzero
    clean_zero = ~clean_nonzero
    leo_zero = ~leo_nonzero
    union_zero = ~valid
    both_zero = clean_zero & leo_zero
    if int(valid.sum().item()) <= 0:
        raise CAGMRuntimeError("P1-CAGM joint zero mask leaves no auxiliary-valid rows")
    clean_h = (clean_z.float()[valid] / clean_norm[valid].unsqueeze(1)).detach()
    leo_h = leo_z.float()[valid] / leo_norm[valid].unsqueeze(1)
    valid_labels = labels[valid]
    if not bool(torch.isfinite(clean_h.detach()).all().item()) or not bool(
        torch.isfinite(leo_h.detach()).all().item()
    ):
        raise CAGMRuntimeError("P1-CAGM normalised auxiliary rows are non-finite")
    radii_clean: Dict[str, torch.Tensor] = {}
    radii_leo: Dict[str, torch.Tensor] = {}
    anchors_clean: Dict[int, torch.Tensor] = {}
    anchors_leo: Dict[int, torch.Tensor] = {}
    per_tx_valid: Dict[str, int] = {}
    for class_id in FROZEN_CAGM_CLASS_IDS:
        class_mask = valid_labels.eq(class_id)
        count = int(class_mask.sum().item())
        if count < 2:
            raise CAGMRuntimeError(
                "P1-CAGM every local4 class requires auxiliary-valid n_c>=2 before backward"
            )
        per_tx_valid[str(class_id)] = count
        clean_class = clean_h[class_mask]
        leo_class = leo_h[class_mask]
        anchor_clean = _normalised_class_anchor(
            clean_class, class_id=class_id, view_name="clean"
        )
        anchor_leo = _normalised_class_anchor(leo_class, class_id=class_id, view_name="leo")
        anchors_clean[class_id] = anchor_clean
        anchors_leo[class_id] = anchor_leo
        radius_clean = (1.0 - (clean_class * anchor_clean).sum(dim=1)).mean()
        radius_leo = (1.0 - (leo_class * anchor_leo).sum(dim=1)).mean()
        if not bool(torch.isfinite(radius_clean.detach()).item()) or not bool(
            torch.isfinite(radius_leo.detach()).item()
        ):
            raise CAGMRuntimeError("P1-CAGM class radius is non-finite")
        key = f"tx{class_id}"
        radii_clean[key] = radius_clean
        radii_leo[key] = radius_leo
    radius_delta = {
        key: radii_leo[key] - radii_clean[key].detach() for key in radii_clean
    }
    gram_delta: Dict[str, torch.Tensor] = {}
    for left_index, left in enumerate(FROZEN_CAGM_CLASS_IDS):
        for right in FROZEN_CAGM_CLASS_IDS[left_index + 1 :]:
            clean_gram = (anchors_clean[left] * anchors_clean[right]).sum()
            leo_gram = (anchors_leo[left] * anchors_leo[right]).sum()
            if not bool(torch.isfinite(clean_gram.detach()).item()) or not bool(
                torch.isfinite(leo_gram.detach()).item()
            ):
                raise CAGMRuntimeError("P1-CAGM inter-class Gram value is non-finite")
            gram_delta[_term_key_pair(left, right)] = leo_gram - clean_gram.detach()
    if len(radius_delta) != 4 or len(gram_delta) != 6:
        raise CAGMRuntimeError("P1-CAGM fixed four-radius/six-Gram term coverage drifted")
    term_sum = sum(term.square() for term in radius_delta.values()) + sum(
        term.square() for term in gram_delta.values()
    )
    loss = term_sum / float(FROZEN_CAGM_TERM_DIVISOR)
    if not bool(torch.isfinite(loss.detach()).item()):
        raise CAGMRuntimeError("P1-CAGM loss is non-finite")
    radius_values = {key: float(value.detach().item()) for key, value in radius_delta.items()}
    gram_values = {key: float(value.detach().item()) for key, value in gram_delta.items()}
    if not all(math.isfinite(value) for value in (*radius_values.values(), *gram_values.values())):
        raise CAGMRuntimeError("P1-CAGM receipt delta is non-finite")
    total_rows = int(labels.numel())
    valid_rows = int(valid.sum().item())
    clean_zero_rows = int(clean_zero.sum().item())
    leo_zero_rows = int(leo_zero.sum().item())
    union_zero_rows = int(union_zero.sum().item())
    both_zero_rows = int(both_zero.sum().item())
    if (
        total_rows != valid_rows + union_zero_rows
        or union_zero_rows != clean_zero_rows + leo_zero_rows - both_zero_rows
    ):
        raise CAGMRuntimeError("P1-CAGM joint zero-mask counters do not close")
    return loss, {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "clean_zero_rows": clean_zero_rows,
        "leo_zero_rows": leo_zero_rows,
        "union_zero_rows": union_zero_rows,
        "both_zero_rows": both_zero_rows,
        "classes": 4,
        "per_tx_valid_rows": per_tx_valid,
        "radius_delta": radius_values,
        "gram_delta": gram_values,
        "finite": True,
        "clean_statistics_detached": True,
        "joint_zero_mask_aux_only": True,
        "all_local4_n_ge_2": True,
        "loss_divisor": FROZEN_CAGM_TERM_DIVISOR,
        "no_valid_or_term_renormalization": True,
    }


def add_cagm_to_loss(
    base_loss: torch.Tensor, cagm: Optional[torch.Tensor], config: Optional[CAGMConfig]
) -> torch.Tensor:
    """Add the single G-arm term; C receives the exact common base tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if cagm is None:
        raise CAGMRuntimeError("Enabled P1-CAGM requires its auxiliary loss")
    return base_loss + float(config.loss_weight) * cagm


def cagm_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    """Return the LEO ``feat_joint`` encoder and exact head audit scopes."""

    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise CAGMRuntimeError("P1-CAGM requires model.id_backbone for VJP audit")
    head = resolve_cagm_classifier_head(raw_model)
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
        raise CAGMRuntimeError("P1-CAGM shared encoder or exact head audit scope is empty")
    return {"shared_encoder": encoder, "classifier_head": head_parameters}


def _finite_nonzero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter], *, group_name: str
) -> Dict[str, float]:
    params = tuple(parameters)
    if not params:
        raise CAGMRuntimeError(f"P1-CAGM {group_name} VJP scope is empty")
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    squared_norm = 0.0
    for gradient in gradients:
        if gradient is None:
            raise CAGMRuntimeError(f"P1-CAGM {group_name} VJP is None or detached")
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise CAGMRuntimeError(f"P1-CAGM {group_name} VJP is non-finite")
        value = gradient.detach().double()
        squared_norm += float(torch.sum(value * value).item())
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm) or norm <= 0.0:
        raise CAGMRuntimeError(f"P1-CAGM {group_name} VJP norm is zero or non-finite")
    return {"parameter_count": float(len(params)), "norm": float(norm)}


def _head_none_or_zero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter]
) -> Dict[str, Any]:
    params = tuple(parameters)
    if not params:
        raise CAGMRuntimeError("P1-CAGM classifier head VJP scope is empty")
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
            raise CAGMRuntimeError("P1-CAGM classifier head auxiliary VJP is non-finite")
        if not bool(torch.count_nonzero(gradient.detach()).item() == 0):
            raise CAGMRuntimeError("P1-CAGM classifier head must have no auxiliary gradient")
        zero_count += 1
    return {
        "parameter_count": float(len(params)),
        "none_parameters": float(none_count),
        "zero_parameters": float(zero_count),
        "nonzero_parameters": 0.0,
        "none_or_zero_expected": True,
    }


def cagm_aux_gradient_audit(
    cagm: torch.Tensor,
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
) -> Dict[str, Any]:
    """Audit raw CAGM gradients without changing the actual AMP update."""

    if not torch.is_tensor(cagm) or cagm.ndim != 0:
        raise CAGMRuntimeError("P1-CAGM VJP audit requires a scalar auxiliary loss")
    if tuple(parameter_groups.keys()) != ("shared_encoder", "classifier_head"):
        raise CAGMRuntimeError("P1-CAGM VJP audit requires encoder and exact head scopes")
    result = {
        "shared_encoder": _finite_nonzero_vjp(
            cagm, parameter_groups["shared_encoder"], group_name="shared_encoder"
        ),
        "classifier_head": _head_none_or_zero_vjp(cagm, parameter_groups["classifier_head"]),
        "raw_unscaled": True,
        "diagnostic_only": True,
    }
    return result


def update_cagm_gradient_audit_receipt(
    receipt: Mapping[str, Any], audit: Mapping[str, Any]
) -> Dict[str, Any]:
    """Seal the required first-valid-batch raw auxiliary VJP audit once."""

    result = dict(receipt)
    if bool(result.get("cagm_gradient_audit_completed", False)):
        raise CAGMRuntimeError("P1-CAGM VJP audit may run only once")
    if audit.get("raw_unscaled") is not True or audit.get("diagnostic_only") is not True:
        raise CAGMRuntimeError("P1-CAGM VJP audit must be raw-unscaled diagnostic-only")
    encoder = audit.get("shared_encoder")
    head = audit.get("classifier_head")
    if not isinstance(encoder, Mapping) or not isinstance(head, Mapping):
        raise CAGMRuntimeError("P1-CAGM VJP audit lacks a required scope")
    if (
        float(encoder.get("parameter_count", 0.0)) <= 0.0
        or not math.isfinite(float(encoder.get("norm", float("nan"))))
        or float(encoder["norm"]) <= 0.0
    ):
        raise CAGMRuntimeError("P1-CAGM shared encoder VJP is zero or non-finite")
    head_count = float(head.get("parameter_count", 0.0))
    none_count = float(head.get("none_parameters", float("nan")))
    zero_count = float(head.get("zero_parameters", float("nan")))
    nonzero_count = float(head.get("nonzero_parameters", float("nan")))
    if (
        head_count <= 0.0
        or not all(math.isfinite(value) and value >= 0.0 for value in (none_count, zero_count, nonzero_count))
        or none_count + zero_count != head_count
        or head.get("none_or_zero_expected") is not True
        or nonzero_count != 0.0
    ):
        raise CAGMRuntimeError("P1-CAGM classifier head auxiliary-gradient contract failed")
    result["cagm_gradient_audit_attempted"] = True
    result["cagm_gradient_audit_completed"] = True
    result["cagm_gradient_audit"] = dict(audit)
    return result


def bind_cagm_source_data_order(
    receipt: Mapping[str, Any], source_split_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind the frozen labeled physical-index source before any training batch."""

    result = dict(receipt)
    source = dict(source_split_receipt or {})
    labeled_sha = str(source.get("labeled_indices_sha256", "") or "")
    manifest_sha = str(source.get("split_manifest_sha256", "") or "")
    if len(labeled_sha) != 64 or len(manifest_sha) != 64:
        raise CAGMConfigurationError(
            "P1-CAGM requires labeled-index and source-split SHA256 receipts"
        )
    result["source_labeled_indices_sha256"] = labeled_sha
    result["source_split_manifest_sha256"] = manifest_sha
    return result


def _as_plain_list(values: Any) -> list[Any]:
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return []


def update_cagm_common_batch_sequence_receipt(
    receipt: Mapping[str, Any],
    *,
    epoch: int,
    batch_index: int,
    scenario: str,
    source_tx_labels: torch.Tensor,
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Chain the opaque common physical/data-order receipt for C and G arms."""

    result = dict(receipt)
    expected = FROZEN_CAGM_SCENARIOS[(int(epoch) + int(batch_index) - 2) % 3]
    if str(scenario) != expected:
        raise CAGMRuntimeError("P1-CAGM common LEO scenario sequence drifted")
    labels = source_tx_labels.detach().reshape(-1).long()
    if labels.numel() == 0:
        raise CAGMRuntimeError("P1-CAGM common batch sequence requires source L rows")
    if metadata is None:
        raise CAGMRuntimeError("P1-CAGM common batch sequence requires opaque physical metadata")
    opaque_ids = _as_plain_list(metadata.get("base_index"))
    if len(opaque_ids) != int(labels.numel()):
        opaque_ids = _as_plain_list(metadata.get("sig_i"))
    if len(opaque_ids) != int(labels.numel()):
        raise CAGMRuntimeError("P1-CAGM physical batch sequence metadata is incomplete")
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
        raise CAGMRuntimeError("P1-CAGM common batch sequence lacks source data-order SHA256")
    result["common_batch_sequence_sha256"] = hashlib.sha256(
        (prior + "\n" + json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()
    result["common_batch_sequence_batches"] = int(result.get("common_batch_sequence_batches", 0)) + 1
    result["common_batch_sequence_rows"] = int(result.get("common_batch_sequence_rows", 0)) + int(labels.numel())
    scenario_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if set(scenario_batches) != set(FROZEN_CAGM_SCENARIOS):
        raise CAGMRuntimeError("P1-CAGM common scenario receipt is malformed")
    scenario_batches[str(scenario)] += 1
    result["common_scenario_batches"] = scenario_batches
    return result


def bind_cagm_optimizer_initial_state(
    receipt: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Seal the newly created AdamW state before the first backward call."""

    result = dict(receipt)
    optimizer_type = type(optimizer).__name__
    if optimizer_type != FROZEN_CAGM_OPTIMIZER_TYPE:
        raise CAGMConfigurationError(
            "P1-CAGM requires optimizer_type=AdamW, "
            f"got {optimizer_type or '<empty>'}"
        )
    state = optimizer.state_dict()
    if dict(state.get("state", {})):
        raise CAGMConfigurationError("P1-CAGM requires a new AdamW state, not a restored optimizer")
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


def _term_receipt() -> Dict[str, Any]:
    return {"batches": 0, "finite_batches": 0, "sum_delta": 0.0, "sum_sq_delta": 0.0}


def _scene_receipt() -> Dict[str, Any]:
    return {
        "batches": 0,
        "total_rows": 0,
        "valid_rows": 0,
        "clean_zero_rows": 0,
        "leo_zero_rows": 0,
        "union_zero_rows": 0,
        "both_zero_rows": 0,
        "per_tx_valid_rows": {str(class_id): 0 for class_id in FROZEN_CAGM_CLASS_IDS},
        "radius_terms": {f"tx{class_id}": _term_receipt() for class_id in FROZEN_CAGM_CLASS_IDS},
        "gram_terms": {
            _term_key_pair(left, right): _term_receipt()
            for left_index, left in enumerate(FROZEN_CAGM_CLASS_IDS)
            for right in FROZEN_CAGM_CLASS_IDS[left_index + 1 :]
        },
    }


def _update_terms(
    terms: Mapping[str, Any], values: Mapping[str, Any], *, expected: Sequence[str]
) -> Dict[str, Any]:
    result = {str(key): dict(value) for key, value in dict(terms).items()}
    if set(values) != set(expected):
        raise CAGMRuntimeError("P1-CAGM fixed term coverage is malformed")
    for key in expected:
        value = float(values[key])
        if not math.isfinite(value):
            raise CAGMRuntimeError("P1-CAGM receipt term is non-finite")
        term = dict(result.get(key, _term_receipt()))
        term["batches"] = int(term.get("batches", 0)) + 1
        term["finite_batches"] = int(term.get("finite_batches", 0)) + 1
        term["sum_delta"] = float(term.get("sum_delta", 0.0)) + value
        term["sum_sq_delta"] = float(term.get("sum_sq_delta", 0.0)) + value * value
        result[key] = term
    return result


def update_cagm_receipt(
    receipt: Mapping[str, Any], batch_info: Mapping[str, Any], *, scenario: str
) -> Dict[str, Any]:
    """Accumulate every G batch/scene's joint-mask and ten-term CAGM evidence."""

    result = dict(receipt)
    if str(result.get("schema", "")) != CAGM_RECEIPT_SCHEMA:
        raise CAGMRuntimeError("P1-CAGM receipt schema must be strict v2")
    if result.get("enabled") is not True:
        raise CAGMRuntimeError("P1-CAGM batch receipt update is G-arm only")
    if result.get("joint_zero_mask_aux_only") is not True:
        raise CAGMRuntimeError(
            "P1-CAGM G receipt joint_zero_mask_aux_only must remain strictly True"
        )
    if batch_info.get("joint_zero_mask_aux_only") is not True:
        raise CAGMRuntimeError(
            "P1-CAGM every G batch requires joint_zero_mask_aux_only is True"
        )
    if str(scenario) not in FROZEN_CAGM_SCENARIOS:
        raise CAGMRuntimeError("P1-CAGM scenario is outside frozen clear/low/rain cycle")
    if tuple(int(value) for value in result.get("expected_tx_class_ids", [])) != FROZEN_CAGM_CLASS_IDS:
        raise CAGMRuntimeError("P1-CAGM receipt lacks local4 class binding")
    if int(batch_info.get("loss_divisor", -1)) != FROZEN_CAGM_TERM_DIVISOR:
        raise CAGMRuntimeError("P1-CAGM receipt loss divisor drifted")
    if batch_info.get("finite") is not True or batch_info.get("clean_statistics_detached") is not True:
        raise CAGMRuntimeError("P1-CAGM finite/detached-clean receipt contract drifted")
    total = int(batch_info.get("total_rows", -1))
    valid = int(batch_info.get("valid_rows", -1))
    clean_zero = int(batch_info.get("clean_zero_rows", -1))
    leo_zero = int(batch_info.get("leo_zero_rows", -1))
    union_zero = int(batch_info.get("union_zero_rows", -1))
    both_zero = int(batch_info.get("both_zero_rows", -1))
    if (
        total <= 0
        or valid <= 0
        or total != valid + union_zero
        or union_zero != clean_zero + leo_zero - both_zero
        or min(clean_zero, leo_zero, union_zero, both_zero) < 0
    ):
        raise CAGMRuntimeError("P1-CAGM batch joint-zero counters do not close")
    per_tx_valid = {str(key): int(value) for key, value in dict(batch_info.get("per_tx_valid_rows", {})).items()}
    expected_tx = {str(value) for value in FROZEN_CAGM_CLASS_IDS}
    if set(per_tx_valid) != expected_tx or any(value < 2 for value in per_tx_valid.values()):
        raise CAGMRuntimeError("P1-CAGM receipt requires auxiliary-valid n_c>=2 for every local4 class")
    radius_keys = tuple(f"tx{class_id}" for class_id in FROZEN_CAGM_CLASS_IDS)
    gram_keys = tuple(
        _term_key_pair(left, right)
        for left_index, left in enumerate(FROZEN_CAGM_CLASS_IDS)
        for right in FROZEN_CAGM_CLASS_IDS[left_index + 1 :]
    )
    scenes = {str(key): dict(value) for key, value in dict(result.get("cagm_scenes", {})).items()}
    scene = dict(scenes.get(str(scenario), _scene_receipt()))
    scene["batches"] = int(scene.get("batches", 0)) + 1
    for receipt_key, batch_value in (
        ("total_rows", total),
        ("valid_rows", valid),
        ("clean_zero_rows", clean_zero),
        ("leo_zero_rows", leo_zero),
        ("union_zero_rows", union_zero),
        ("both_zero_rows", both_zero),
    ):
        scene[receipt_key] = int(scene.get(receipt_key, 0)) + batch_value
    scene_per_tx = {str(key): int(value) for key, value in dict(scene.get("per_tx_valid_rows", {})).items()}
    if set(scene_per_tx) != expected_tx:
        raise CAGMRuntimeError("P1-CAGM scene per-class receipt is malformed")
    for key, value in per_tx_valid.items():
        scene_per_tx[key] += value
    scene["per_tx_valid_rows"] = scene_per_tx
    scene["radius_terms"] = _update_terms(
        scene.get("radius_terms", {}), batch_info.get("radius_delta", {}), expected=radius_keys
    )
    scene["gram_terms"] = _update_terms(
        scene.get("gram_terms", {}), batch_info.get("gram_delta", {}), expected=gram_keys
    )
    scenes[str(scenario)] = scene
    result["cagm_scenes"] = scenes
    result["cagm_radius_terms"] = _update_terms(
        result.get("cagm_radius_terms", {}), batch_info.get("radius_delta", {}), expected=radius_keys
    )
    result["cagm_gram_terms"] = _update_terms(
        result.get("cagm_gram_terms", {}), batch_info.get("gram_delta", {}), expected=gram_keys
    )
    result["cagm_batches"] = int(result.get("cagm_batches", 0)) + 1
    result["cagm_total_rows"] = int(result.get("cagm_total_rows", 0)) + total
    result["cagm_valid_rows"] = int(result.get("cagm_valid_rows", 0)) + valid
    result["cagm_clean_zero_rows"] = int(result.get("cagm_clean_zero_rows", 0)) + clean_zero
    result["cagm_leo_zero_rows"] = int(result.get("cagm_leo_zero_rows", 0)) + leo_zero
    result["cagm_union_zero_rows"] = int(result.get("cagm_union_zero_rows", 0)) + union_zero
    result["cagm_both_zero_rows"] = int(result.get("cagm_both_zero_rows", 0)) + both_zero
    return result


def _validate_common_terminal_contract(result: Mapping[str, Any]) -> None:
    if str(result.get("schema", "")) != CAGM_RECEIPT_SCHEMA:
        raise CAGMRuntimeError("P1-CAGM terminal receipt schema must be strict v2")
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
            raise CAGMRuntimeError(f"P1-CAGM terminal receipt lacks {key}")
    if str(result.get("checkpoint_role", "") or "") != "training_final_only":
        raise CAGMRuntimeError("P1-CAGM requires a training_final_only warm-start checkpoint")
    if result.get("optimizer_state_restored") is not False or result.get("rng_state_restored") is not False:
        raise CAGMRuntimeError("P1-CAGM optimizer/RNG restoration is forbidden")
    if str(result.get("optimizer_type", "")) != FROZEN_CAGM_OPTIMIZER_TYPE:
        raise CAGMRuntimeError("P1-CAGM terminal optimizer_type must be AdamW")
    if result.get("optimizer_initial_state_empty") is not True:
        raise CAGMRuntimeError("P1-CAGM missing new AdamW initial-state receipt")
    batches = int(result.get("common_batch_sequence_batches", 0))
    rows = int(result.get("common_batch_sequence_rows", 0))
    scenarios = {str(key): int(value) for key, value in dict(result.get("common_scenario_batches", {})).items()}
    if batches <= 0 or rows <= 0 or set(scenarios) != set(FROZEN_CAGM_SCENARIOS) or any(value <= 0 for value in scenarios.values()):
        raise CAGMRuntimeError("P1-CAGM common batch/scenario receipt is incomplete")


def _validate_term_map(terms: Mapping[str, Any], expected: Sequence[str], *, batches: int) -> None:
    if set(terms) != set(expected):
        raise CAGMRuntimeError("P1-CAGM terminal ten-term coverage is incomplete")
    for key in expected:
        term = dict(terms[key])
        if (
            int(term.get("batches", -1)) != batches
            or int(term.get("finite_batches", -1)) != batches
            or not math.isfinite(float(term.get("sum_delta", float("nan"))))
            or not math.isfinite(float(term.get("sum_sq_delta", float("nan"))))
        ):
            raise CAGMRuntimeError(f"P1-CAGM terminal term coverage failed for {key}")


def validate_cagm_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless common bindings and all scene/ten-term counters close."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    _validate_common_terminal_contract(result)
    enabled = result.get("enabled")
    if enabled is not True and enabled is not False:
        raise CAGMRuntimeError("P1-CAGM terminal enabled flag must be strict bool")
    if enabled is True:
        if result.get("joint_zero_mask_aux_only") is not True:
            raise CAGMRuntimeError(
                "P1-CAGM terminal G joint_zero_mask_aux_only must be strictly True"
            )
        if str(result.get("joint_zero_mask_aux_only_semantics", "")) != (
            "G_AUXILIARY_ONLY_BASE_RETAINS_FULL_BATCH"
        ):
            raise CAGMRuntimeError("P1-CAGM terminal G joint-zero-mask semantics drifted")
    else:
        if result.get("joint_zero_mask_aux_only") is not False:
            raise CAGMRuntimeError(
                "P1-CAGM terminal C joint_zero_mask_aux_only must be explicitly False"
            )
        if str(result.get("joint_zero_mask_aux_only_semantics", "")) != (
            "C_CONTROL_NOT_APPLICABLE"
        ):
            raise CAGMRuntimeError("P1-CAGM terminal C joint-zero-mask semantics drifted")
    if enabled is False:
        forbidden = (
            "cagm_batches",
            "cagm_total_rows",
            "cagm_valid_rows",
            "cagm_clean_zero_rows",
            "cagm_leo_zero_rows",
            "cagm_union_zero_rows",
            "cagm_both_zero_rows",
        )
        if any(int(result.get(key, 0)) != 0 for key in forbidden) or any(
            dict(result.get(key, {})) for key in ("cagm_scenes", "cagm_radius_terms", "cagm_gram_terms")
        ):
            raise CAGMRuntimeError("P1-CAGM C arm must retain N/A-or-zero CAGM fields")
        result["cagm_terminal_contract"] = "CONTROL_ARM_NOT_APPLICABLE_COMMON_SEQUENCE_BOUND"
        result["cagm_terminal_contract_passed"] = True
        return result
    radius_keys = tuple(f"tx{class_id}" for class_id in FROZEN_CAGM_CLASS_IDS)
    gram_keys = tuple(
        _term_key_pair(left, right)
        for left_index, left in enumerate(FROZEN_CAGM_CLASS_IDS)
        for right in FROZEN_CAGM_CLASS_IDS[left_index + 1 :]
    )
    scenes = {str(key): dict(value) for key, value in dict(result.get("cagm_scenes", {})).items()}
    if set(scenes) != set(FROZEN_CAGM_SCENARIOS):
        raise CAGMRuntimeError("P1-CAGM terminal scene coverage is incomplete")
    common_scenes = {str(key): int(value) for key, value in dict(result.get("common_scenario_batches", {})).items()}
    totals = {key: 0 for key in ("batches", "total_rows", "valid_rows", "clean_zero_rows", "leo_zero_rows", "union_zero_rows", "both_zero_rows")}
    expected_tx = {str(value) for value in FROZEN_CAGM_CLASS_IDS}
    for scenario in FROZEN_CAGM_SCENARIOS:
        scene = scenes[scenario]
        batches = int(scene.get("batches", 0))
        if batches <= 0 or batches != common_scenes[scenario]:
            raise CAGMRuntimeError("P1-CAGM terminal scene batch coverage drifted")
        for key in totals:
            totals[key] += int(scene.get(key, 0))
        if (
            int(scene.get("total_rows", 0)) <= 0
            or int(scene.get("valid_rows", 0)) <= 0
            or int(scene.get("total_rows", 0)) != int(scene.get("valid_rows", 0)) + int(scene.get("union_zero_rows", 0))
            or int(scene.get("union_zero_rows", 0)) != int(scene.get("clean_zero_rows", 0)) + int(scene.get("leo_zero_rows", 0)) - int(scene.get("both_zero_rows", 0))
        ):
            raise CAGMRuntimeError("P1-CAGM terminal scene zero-mask closure failed")
        per_tx = {str(key): int(value) for key, value in dict(scene.get("per_tx_valid_rows", {})).items()}
        if set(per_tx) != expected_tx or any(value < 2 for value in per_tx.values()):
            raise CAGMRuntimeError("P1-CAGM terminal per-class valid coverage failed")
        _validate_term_map(scene.get("radius_terms", {}), radius_keys, batches=batches)
        _validate_term_map(scene.get("gram_terms", {}), gram_keys, batches=batches)
    if any(int(result.get(f"cagm_{key}", -1)) != value for key, value in totals.items()):
        raise CAGMRuntimeError("P1-CAGM terminal aggregate counters do not close")
    if totals["batches"] != int(result.get("common_batch_sequence_batches", 0)) or totals["total_rows"] != int(result.get("common_batch_sequence_rows", 0)):
        raise CAGMRuntimeError("P1-CAGM terminal C/G common-sequence counters do not close")
    _validate_term_map(result.get("cagm_radius_terms", {}), radius_keys, batches=totals["batches"])
    _validate_term_map(result.get("cagm_gram_terms", {}), gram_keys, batches=totals["batches"])
    if not bool(result.get("cagm_gradient_audit_completed", False)):
        raise CAGMRuntimeError("P1-CAGM terminal first-valid auxiliary VJP audit is incomplete")
    result["cagm_terminal_contract"] = (
        "FORMAL_COMMON_WARM_START_DATA_ORDER_NEW_ADAMW_AND_SCENEWISE_JOINT_ZERO_MASK_"
        "FOUR_RADIUS_SIX_GRAM_FINITE_WITH_FIRST_RAW_ENCODER_VJP_AND_HEAD_NO_AUX_GRAD"
    )
    result["cagm_terminal_contract_passed"] = True
    return result


def _failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "vjp" in message or "gradient" in message or "head" in message:
        return "CAGM_AUX_GRADIENT_PATH_FAILURE"
    if "non-finite" in message or "nonfinite" in message:
        return "CAGM_NONFINITE"
    if "zero" in message or "n_c" in message or "class" in message:
        return "CAGM_JOINT_MASK_OR_CLASS_COVERAGE_FAILURE"
    if "sequence" in message or "receipt" in message or "coverage" in message:
        return "CAGM_RECEIPT_CLOSURE_FAILURE"
    if "binding" in message or "feat_joint" in message:
        return "CAGM_BINDING_FAILURE"
    return "CAGM_RUNTIME_FAILURE"


def write_cagm_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free fail-closed record for the CAGM arm."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise CAGMRuntimeError("P1-CAGM failure receipt requires an existing output directory")
    target = target_dir / "cagm_failure_receipt.json"
    payload = {
        "schema": "cvs.phase1.cagm_failure_receipt.v1",
        "status": "FAIL_CLOSED",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage),
        "error_type": type(error).__name__,
        "error_fingerprint": _failure_fingerprint(error),
        "cagm_receipt": dict(receipt),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = mkstemp(prefix=".cagm_failure_receipt.", suffix=".tmp", dir=str(target_dir))
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


def strict_cagm_warm_start(
    model: torch.nn.Module,
    checkpoint_model_state: Mapping[str, torch.Tensor],
    *,
    baseline_path: str,
    baseline_sha256: str,
    checkpoint_epoch: int,
    checkpoint_role: str,
) -> Dict[str, Any]:
    """Load model weights only; optimizer and RNG state deliberately stay new."""

    path = str(baseline_path or "").strip()
    digest = str(baseline_sha256 or "").strip()
    if not path or len(digest) != 64 or not isinstance(checkpoint_model_state, Mapping):
        raise CAGMConfigurationError(
            "Frozen P1-CAGM warm-start requires model state, path, and SHA256"
        )
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise CAGMConfigurationError(
            f"Frozen P1-CAGM strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise CAGMConfigurationError(
            "Frozen P1-CAGM strict baseline model-key mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    if str(checkpoint_role or "") != "training_final_only":
        raise CAGMConfigurationError(
            "Frozen P1-CAGM requires baseline checkpoint_role=training_final_only"
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
