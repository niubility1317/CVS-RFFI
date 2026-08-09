"""Frozen P1-RCRMD continuation contract for Phase1 source-only DG.

P1-RCRMD (Receiver-Conditioned Relative Margin Drop) preserves the common
GeoSat-C clean and one-LEO forwards.  Its G arm adds a fixed, source-L-only
receiver-by-class equal-weight loss on the existing local4 raw TX logits:

``q_i = [stopgrad(m_clean_i) - m_leo_i]_+^2`` and
``L = sum_{r,c} mean_{i in I_rc}(q_i) / (4 |R_s|)``.

An empty receiver/class batch cell contributes exactly zero without changing
the frozen denominator.  The module is deliberately not a quantile/DRO,
does not construct batches, and has no day, target, proxy, state, threshold,
or additional-forward input path.
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


FROZEN_RCRMD_LAMBDA = 0.02
FROZEN_RCRMD_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_RCRMD_CLASS_IDS = (0, 1, 2, 3)
FROZEN_RCRMD_SOURCE_RECEIVER_IDS = (0, 1, 2, 3, 4, 5, 6)
FROZEN_RCRMD_SOURCE_RECEIVER_COUNT = len(FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
FROZEN_RCRMD_OPTIMIZER_TYPE = "AdamW"
FROZEN_RCRMD_FLOAT32_LEDGER_REL_TOL = 32.0 * float(torch.finfo(torch.float32).eps)
RCRMD_RECEIPT_SCHEMA = "cvs.phase1.rcrmd_receipt.v1"
_TOLERANCE = 1e-12


class RCRMDConfigurationError(ValueError):
    """Raised when a frozen P1-RCRMD C/G configuration drifts."""


class RCRMDRuntimeError(RuntimeError):
    """Raised when a P1-RCRMD runtime or receipt contract cannot be proved."""


@dataclass(frozen=True)
class RCRMDConfig:
    """Immutable P1-RCRMD controls consumed by the common training loop."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise RCRMDConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
        raise RCRMDConfigurationError(
            f"Frozen P1-RCRMD requires {name}={expected:.12g}, got {actual!r}"
        )


def _float32_ledger_close(actual: float, expected: float) -> bool:
    """Reconcile the float32 optimization scalar with its float64 cell ledger.

    The forward loss sums 28 non-negative float32 cell means before applying
    the fixed scale, whereas the receipt re-sums those exact cell scalars as
    Python floats.  Thirty-two float32 epsilons cover that frozen reduction
    depth without turning a material ledger mismatch into an accepted receipt.
    """

    return (
        math.isfinite(actual)
        and math.isfinite(expected)
        and abs(actual - expected)
        <= FROZEN_RCRMD_FLOAT32_LEDGER_REL_TOL
        * max(1.0, abs(actual), abs(expected))
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
        raise RCRMDConfigurationError(
            "Frozen P1-RCRMD forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    raw = str(value or "").strip()
    scenarios = tuple(
        item.strip().lower().replace("-", "_")
        for item in raw.split(",")
        if item.strip()
    )
    if scenarios != FROZEN_RCRMD_SCENARIOS:
        raise RCRMDConfigurationError(
            "Frozen P1-RCRMD requires --sat_train_scenarios "
            + ",".join(FROZEN_RCRMD_SCENARIOS)
        )
    return scenarios


def validate_rcrmd_args(args: Any) -> RCRMDConfig:
    """Validate the frozen common-base C/G contract before data are loaded."""

    frozen_mode = _bool_arg(args, "phase1_rcrmd_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_rcrmd_enabled", False)
    loss_weight = _float_arg(args, "lambda_rcrmd", 0.0)
    if not frozen_mode and not enabled:
        return RCRMDConfig(False, False, 0.0)
    if enabled and not frozen_mode:
        raise RCRMDConfigurationError(
            "--phase1_rcrmd_enabled requires --phase1_rcrmd_frozen_mode true"
        )
    _require_close("lambda_rcrmd", loss_weight, FROZEN_RCRMD_LAMBDA if enabled else 0.0)
    if bool(getattr(args, "from_scratch", True)):
        raise RCRMDConfigurationError("Frozen P1-RCRMD requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise RCRMDConfigurationError("Frozen P1-RCRMD requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise RCRMDConfigurationError("Frozen P1-RCRMD must train the shared feat_joint encoder")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise RCRMDConfigurationError("Frozen P1-RCRMD requires --id_feature_key feat_joint")
    if int(getattr(args, "epochs", 0)) != 40 or int(getattr(args, "label_epochs", 0)) != 40:
        raise RCRMDConfigurationError("Frozen P1-RCRMD requires exactly 40 labeled epochs")
    if int(getattr(args, "pseudo_epochs", 0)) != 0:
        raise RCRMDConfigurationError("Frozen P1-RCRMD forbids pseudo epochs")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise RCRMDConfigurationError("Frozen P1-RCRMD requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise RCRMDConfigurationError("Frozen P1-RCRMD remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise RCRMDConfigurationError("Frozen P1-RCRMD requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise RCRMDConfigurationError("Frozen P1-RCRMD requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise RCRMDConfigurationError("Frozen P1-RCRMD forbids --sat_view_schedule overrides")
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise RCRMDConfigurationError("Frozen P1-RCRMD requires non-concatenated single-LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise RCRMDConfigurationError("Frozen P1-RCRMD permits only source_known_train L updates")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise RCRMDConfigurationError("Frozen P1-RCRMD forbids RX-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise RCRMDConfigurationError("Frozen P1-RCRMD permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise RCRMDConfigurationError("Frozen P1-RCRMD forbids a reject head")
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
    return RCRMDConfig(True, enabled, loss_weight)


def rcrmd_config_receipt(config: RCRMDConfig) -> Dict[str, Any]:
    """Create the data-free receipt skeleton for either frozen arm."""

    return {
        "schema": RCRMD_RECEIPT_SCHEMA,
        "method": "P1_RCRMD",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "loss_rule": "SOURCE_L_RX_BY_LOCAL4_EQUAL_WEIGHT_STOPGRAD_CLEAN_TO_LEO_POSITIVE_RAW_MARGIN_DROP_SQUARED",
        "loss_formula": "q=[sg(m_clean)-m_leo]_+^2;g_rc=0_if_n_rc=0_else_mean_Irc(q);L=sum_rc(g_rc)/(4*|R_s|)",
        "loss_global_denominator": "4_TIMES_FIXED_SOURCE_RECEIVER_COUNT",
        "local_class_ids": list(FROZEN_RCRMD_CLASS_IDS),
        "frozen_source_receiver_ids": list(FROZEN_RCRMD_SOURCE_RECEIVER_IDS),
        "frozen_source_receiver_count": FROZEN_RCRMD_SOURCE_RECEIVER_COUNT,
        "frozen_cells_per_scene": (
            len(FROZEN_RCRMD_CLASS_IDS) * FROZEN_RCRMD_SOURCE_RECEIVER_COUNT
        ),
        "logit_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "clean_margin_detached": True,
        "uses_new_forward": False,
        "uses_resampling": False,
        "uses_rx_labels": True,
        "rx_permission": "SOURCE_KNOWN_TRAIN_L_PHYSICAL_ID_BOUND_rx_i_ONLY",
        "rx_metadata_allowlist": ["rx_i"],
        "no_day_assertion": "day_i_NOT_READ_BY_RCRMD",
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
        "common_scenario_batches": {scenario: 0 for scenario in FROZEN_RCRMD_SCENARIOS},
        "rcrmd_common_cells": {},
        "rcrmd_common_batch_cells": [],
        "rcrmd_batches": 0,
        "rcrmd_total_rows": 0,
        "rcrmd_active_q": 0,
        "rcrmd_loss_sum": 0.0,
        "rcrmd_float32_ledger_rel_tolerance": FROZEN_RCRMD_FLOAT32_LEDGER_REL_TOL,
        "rcrmd_scenes": {},
        "rcrmd_g_batch_aux": [],
        "rcrmd_gradient_audit_attempted": False,
        "rcrmd_gradient_audit_completed": False,
        "rcrmd_gradient_audit": {},
        "rcrmd_terminal_contract": "PENDING",
        "rcrmd_terminal_contract_passed": False,
        "proxy_rows": 0,
        "held_rows": 0,
    }


def _normalized_tx_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RCRMDConfigurationError(f"P1-RCRMD {name} must be a TX class sequence")
    order = tuple(str(value).strip() for value in values)
    if not order or len(order) != len(set(order)) or any(not value for value in order):
        raise RCRMDConfigurationError(f"P1-RCRMD {name} must be non-empty and unique")
    return order


def _positive_count(name: str, value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise RCRMDConfigurationError(f"P1-RCRMD {name} must be an integer") from exc
    if count <= 0:
        raise RCRMDConfigurationError(f"P1-RCRMD {name} must be positive")
    return count


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _receiver_key(receiver_id: int, class_id: int) -> str:
    return f"rx{int(receiver_id)}|tx{int(class_id)}"


def _source_receiver_ids(values: Any) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RCRMDConfigurationError("P1-RCRMD source_receivers must be a non-empty source-only sequence")
    parsed = []
    for value in values:
        text = str(value).strip()
        if not text:
            raise RCRMDConfigurationError("P1-RCRMD source receiver id may not be empty")
        try:
            receiver_id = int(text)
        except (TypeError, ValueError) as exc:
            raise RCRMDConfigurationError(
                "P1-RCRMD source receiver id must be an integer physical receiver index"
            ) from exc
        if str(receiver_id) != text:
            raise RCRMDConfigurationError("P1-RCRMD source receiver id is not canonical")
        parsed.append(receiver_id)
    canonical = tuple(sorted(parsed))
    if not canonical or len(canonical) != len(set(canonical)):
        raise RCRMDConfigurationError("P1-RCRMD source receiver ids must be non-empty and unique")
    return canonical


def _require_frozen_source_receivers(receivers: Sequence[Any]) -> Tuple[int, ...]:
    """Reject any allowlist other than the frozen F1C source-RX set 0..6."""

    parsed = _source_receiver_ids(receivers)
    if parsed != FROZEN_RCRMD_SOURCE_RECEIVER_IDS:
        raise RCRMDConfigurationError(
            "P1-RCRMD requires frozen F1C source receivers 0..6 "
            f"({FROZEN_RCRMD_SOURCE_RECEIVER_COUNT} receivers); got {list(parsed)}"
        )
    return parsed


def resolve_rcrmd_local_head_class_binding(
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
        raise RCRMDConfigurationError("P1-RCRMD requires exactly four local source-TX rows")
    if local != source or checkpoint != source:
        raise RCRMDConfigurationError(
            "P1-RCRMD local/checkpoint TX order must equal the source-train receipt"
        )
    if local_count != len(local) or checkpoint_count != live_count or live_count != local_count:
        raise RCRMDConfigurationError("P1-RCRMD local/head class counts must match")
    if set(local).difference(dataset):
        raise RCRMDConfigurationError("P1-RCRMD local TX labels are absent from dataset order")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(FROZEN_RCRMD_CLASS_IDS),
        "expected_tx_class_ids": list(FROZEN_RCRMD_CLASS_IDS),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = _canonical_sha256(binding)
    return binding


def remap_rcrmd_local_labels_to_head_rows(
    local_labels: torch.Tensor,
    local_to_head_class_ids: Sequence[Any],
) -> torch.Tensor:
    """Map contiguous local source labels through the sealed identity mapping."""

    if not torch.is_tensor(local_labels):
        raise RCRMDRuntimeError("P1-RCRMD local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if mapping != FROZEN_RCRMD_CLASS_IDS:
        raise RCRMDRuntimeError("P1-RCRMD local-to-head mapping must be local4 identity")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCRMDRuntimeError("P1-RCRMD local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_rcrmd_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    """Resolve the existing exact classifier head; no G-only head is created."""

    raw_model = getattr(model, "_orig_mod", model)
    try:
        head = raw_model.id_backbone.cls_head.head
    except AttributeError as exc:
        raise RCRMDRuntimeError("P1-RCRMD requires model.id_backbone.cls_head.head") from exc
    if not isinstance(head, torch.nn.Module):
        raise RCRMDRuntimeError("P1-RCRMD exact classifier head is not a module")
    if not tuple(parameter for parameter in head.parameters() if parameter.requires_grad):
        raise RCRMDRuntimeError("P1-RCRMD exact classifier head has no trainable parameter")
    return head


def resolve_rcrmd_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    """Resolve the exact local4 head weight for strict binding checks."""

    weight = getattr(resolve_rcrmd_classifier_head(model), "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise RCRMDRuntimeError("P1-RCRMD classifier head weight must be a rank-2 Parameter")
    return weight


def _validate_view_binding(
    *,
    view_name: str,
    output: Mapping[str, Any],
    labels: torch.Tensor,
    head_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if str(output.get("z_id_key", "")) != "feat_joint":
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} z_id_key must be feat_joint")
    z_id = output.get("z_id")
    logits = output.get("tx_logits")
    if not torch.is_tensor(z_id) or z_id.ndim != 2:
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} z_id must be rank-2 feat_joint")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise RCRMDRuntimeError(
            f"P1-RCRMD {view_name} tx_logits must be rank-2 raw pre-softmax logits"
        )
    if z_id.size(0) != labels.numel() or logits.size(0) != labels.numel():
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} rows must align with source L labels")
    if int(head_weight.size(0)) != 4 or int(logits.size(1)) != 4:
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} head/logit class rows must be local4")
    if int(head_weight.size(1)) != int(z_id.size(1)):
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} feat_joint/head dimension binding drifted")
    if not bool(z_id.requires_grad) or not bool(logits.requires_grad):
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} requires a live z_id/head gradient path")
    if not bool(torch.isfinite(z_id.detach()).all().item()):
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} z_id contains non-finite values")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} raw logits contain non-finite values")
    return z_id, logits


def _validated_receiver_labels(
    receiver_labels: torch.Tensor,
    *,
    rows: int,
    expected_receiver_ids: Sequence[Any],
) -> torch.Tensor:
    if not torch.is_tensor(receiver_labels):
        raise RCRMDRuntimeError("P1-RCRMD requires source-L physical rx_i labels")
    values = receiver_labels.reshape(-1).long()
    expected = _require_frozen_source_receivers(expected_receiver_ids)
    if values.numel() != int(rows) or values.numel() == 0:
        raise RCRMDRuntimeError("P1-RCRMD source-L rx_i rows do not align")
    observed = {int(value) for value in values.detach().cpu().tolist()}
    if observed.difference(set(expected)):
        raise RCRMDRuntimeError("P1-RCRMD rx_i contains a receiver outside frozen source R_s")
    return values


def validate_rcrmd_binding(
    *,
    model: torch.nn.Module,
    out_clean: Mapping[str, Any],
    out_leo: Mapping[str, Any],
    tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
    expected_receiver_ids: Sequence[Any],
) -> torch.nn.Parameter:
    """Fail closed unless both common forwards use the exact local4 head path."""

    if not isinstance(out_clean, Mapping) or not isinstance(out_leo, Mapping):
        raise RCRMDRuntimeError("P1-RCRMD requires clean and LEO mapping outputs")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() == 0:
        raise RCRMDRuntimeError("P1-RCRMD requires a non-empty source L batch")
    if tuple(int(value) for value in expected_class_ids) != FROZEN_RCRMD_CLASS_IDS:
        raise RCRMDRuntimeError("P1-RCRMD expected local4 class order is invalid")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCRMDRuntimeError("P1-RCRMD source labels do not bind to local4 head rows")
    _validated_receiver_labels(
        source_rx_labels, rows=int(labels.numel()), expected_receiver_ids=expected_receiver_ids
    )
    head_weight = resolve_rcrmd_classifier_weight(model)
    if not bool(torch.isfinite(head_weight.detach()).all().item()):
        raise RCRMDRuntimeError("P1-RCRMD exact classifier head is non-finite")
    _validate_view_binding(view_name="clean", output=out_clean, labels=labels, head_weight=head_weight)
    _validate_view_binding(view_name="leo", output=out_leo, labels=labels, head_weight=head_weight)
    return head_weight


def _raw_local4_margins(logits: torch.Tensor, labels: torch.Tensor, *, view_name: str) -> torch.Tensor:
    if not torch.is_tensor(logits) or logits.ndim != 2 or int(logits.size(1)) != 4:
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} requires local4 raw logits")
    if logits.size(0) != labels.numel() or labels.numel() == 0:
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} logits and labels do not align")
    if not bool(logits.requires_grad):
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} logits must retain a live gradient path")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} logits are non-finite")
    stable_logits = logits.float()
    row_ids = torch.arange(labels.numel(), device=labels.device)
    true_logits = stable_logits[row_ids, labels]
    other_mask = F.one_hot(labels, num_classes=4).to(dtype=torch.bool)
    margins = true_logits - torch.logsumexp(stable_logits.masked_fill(other_mask, float("-inf")), dim=1)
    if not bool(torch.isfinite(margins.detach()).all().item()):
        raise RCRMDRuntimeError(f"P1-RCRMD {view_name} raw local4 margins are non-finite")
    return margins


def _cell_template(receiver_ids: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    receivers = _source_receiver_ids(receiver_ids)
    return {
        _receiver_key(receiver_id, class_id): {
            "rows": 0,
            "active_q": 0,
            "finite_q": 0,
            "q_sum": 0.0,
            "g_sum": 0.0,
            "loss_sum": 0.0,
            "batches": 0,
            "nonempty_batches": 0,
            "finite_batches": 0,
        }
        for receiver_id in receivers
        for class_id in FROZEN_RCRMD_CLASS_IDS
    }


def _batch_cell_weights(receiver_ids: Sequence[Any], counts: Mapping[str, int]) -> Dict[str, Dict[str, float]]:
    receivers = _source_receiver_ids(receiver_ids)
    scale = 1.0 / float(4 * len(receivers))
    expected = {_receiver_key(receiver_id, class_id) for receiver_id in receivers for class_id in FROZEN_RCRMD_CLASS_IDS}
    if set(counts) != expected:
        raise RCRMDRuntimeError("P1-RCRMD batch cell count coverage drifted")
    weights: Dict[str, Dict[str, float]] = {}
    for key in sorted(expected):
        count = int(counts[key])
        if count < 0:
            raise RCRMDRuntimeError("P1-RCRMD n_rc may not be negative")
        weights[key] = {
            "cell_weight": scale,
            "row_weight": (scale / float(count)) if count > 0 else 0.0,
        }
    return weights


def rcrmd_loss(
    clean_tx_logits: torch.Tensor,
    leo_tx_logits: torch.Tensor,
    source_tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    source_receiver_ids: Sequence[Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute the exact fixed-scale source-L receiver×class RCRMD loss.

    The only pair is the same physical source-L row's existing clean and one
    LEO observation.  Missing ``(r,c)`` cells contribute a differentiable
    zero and never alter either denominator or sampling.
    """

    labels = source_tx_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCRMDRuntimeError("P1-RCRMD source labels are outside local4")
    receivers = _require_frozen_source_receivers(source_receiver_ids)
    rx_labels = _validated_receiver_labels(
        source_rx_labels, rows=int(labels.numel()), expected_receiver_ids=receivers
    )
    clean_margin = _raw_local4_margins(clean_tx_logits, labels, view_name="clean")
    leo_margin = _raw_local4_margins(leo_tx_logits, labels, view_name="leo")
    q = torch.relu(clean_margin.detach() - leo_margin).square()
    if not bool(torch.isfinite(q.detach()).all().item()):
        raise RCRMDRuntimeError("P1-RCRMD q contains non-finite values")
    cells: Dict[str, Dict[str, Any]] = {}
    terms = []
    total_rows = 0
    total_active = 0
    total_finite = 0
    scale = 1.0 / float(4 * len(receivers))
    for receiver_id in receivers:
        for class_id in FROZEN_RCRMD_CLASS_IDS:
            key = _receiver_key(receiver_id, class_id)
            mask = rx_labels.eq(receiver_id) & labels.eq(class_id)
            count = int(mask.sum().item())
            total_rows += count
            if count == 0:
                group_q = q[mask]
                g_rc = group_q.sum()
                active = 0
                finite = 0
                q_sum = 0.0
            else:
                group_q = q[mask]
                if not bool(torch.isfinite(group_q.detach()).all().item()):
                    raise RCRMDRuntimeError("P1-RCRMD receiver/class q is non-finite")
                g_rc = group_q.mean()
                active = int((group_q.detach() > 0.0).sum().item())
                finite = int(torch.isfinite(group_q.detach()).sum().item())
                q_sum = float(group_q.detach().sum().item())
            if not bool(torch.isfinite(g_rc.detach()).item()):
                raise RCRMDRuntimeError("P1-RCRMD g_rc is non-finite")
            if active < 0 or active > count or finite != count:
                raise RCRMDRuntimeError("P1-RCRMD receiver/class q counters do not close")
            g_value = float(g_rc.detach().item())
            cells[key] = {
                "n_rc": count,
                "active_q": active,
                "finite_q": finite,
                "q_sum": q_sum,
                "g_rc": g_value,
                "loss_contribution": scale * g_value,
            }
            terms.append(g_rc)
            total_active += active
            total_finite += finite
    if total_rows != int(labels.numel()) or total_finite != total_rows:
        raise RCRMDRuntimeError("P1-RCRMD batch receiver/class coverage does not close")
    loss = torch.stack(terms).sum() * scale
    if not bool(torch.isfinite(loss.detach()).item()):
        raise RCRMDRuntimeError("P1-RCRMD loss is non-finite")
    counts = {key: int(value["n_rc"]) for key, value in cells.items()}
    weights = _batch_cell_weights(receivers, counts)
    for key, weight in weights.items():
        if abs(float(weight["cell_weight"]) - scale) > _TOLERANCE:
            raise RCRMDRuntimeError("P1-RCRMD fixed cell scale drifted")
        cells[key].update(weight)
    return loss, {
        "rows": int(labels.numel()),
        "active_q": total_active,
        "finite_q": total_finite,
        "loss_sum": float(loss.detach().item()),
        "global_denominator": int(4 * len(receivers)),
        "fixed_scale": scale,
        "source_receiver_ids": list(receivers),
        "cells": cells,
        "finite": True,
        "clean_margin_detached": True,
        "empty_cell_zero": True,
        "no_active_renormalization": True,
        "no_cross_sample_or_cross_receiver_pairing": True,
    }


def add_rcrmd_to_loss(
    base_loss: torch.Tensor,
    rcrmd: Optional[torch.Tensor],
    config: Optional[RCRMDConfig],
) -> torch.Tensor:
    """Add the sole G-arm term; C receives the exact common base tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if rcrmd is None:
        raise RCRMDRuntimeError("Enabled P1-RCRMD requires its auxiliary loss")
    return base_loss + float(config.loss_weight) * rcrmd


def rcrmd_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    """Return the common feat_joint encoder and exact head audit scopes."""

    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise RCRMDRuntimeError("P1-RCRMD requires model.id_backbone for VJP audit")
    head = resolve_rcrmd_classifier_head(raw_model)
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
        raise RCRMDRuntimeError("P1-RCRMD shared encoder or exact head audit scope is empty")
    return {"shared_encoder": encoder, "classifier_head": head_parameters}


def _finite_nonzero_vjp(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    group_name: str,
) -> Dict[str, float]:
    params = tuple(parameters)
    if not params:
        raise RCRMDRuntimeError(f"P1-RCRMD {group_name} VJP scope is empty")
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    squared_norm = 0.0
    for gradient in gradients:
        if gradient is None:
            raise RCRMDRuntimeError(f"P1-RCRMD {group_name} VJP is None or detached")
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise RCRMDRuntimeError(f"P1-RCRMD {group_name} VJP is non-finite")
        value = gradient.detach().double()
        squared_norm += float(torch.sum(value * value).item())
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm) or norm <= 0.0:
        raise RCRMDRuntimeError(f"P1-RCRMD {group_name} VJP norm is zero or non-finite")
    return {"parameter_count": float(len(params)), "norm": float(norm)}


def rcrmd_aux_gradient_audit(
    rcrmd: torch.Tensor,
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
) -> Dict[str, Any]:
    """Audit raw RCRMD VJPs without changing AMP, optimizer, or RNG state."""

    if not torch.is_tensor(rcrmd) or rcrmd.ndim != 0:
        raise RCRMDRuntimeError("P1-RCRMD VJP audit requires a scalar auxiliary loss")
    expected_groups = ("shared_encoder", "classifier_head")
    if tuple(parameter_groups.keys()) != expected_groups:
        raise RCRMDRuntimeError("P1-RCRMD VJP audit requires encoder and exact-head scopes")
    return {
        name: _finite_nonzero_vjp(rcrmd, parameter_groups[name], group_name=name)
        for name in expected_groups
    } | {"raw_unscaled": True, "diagnostic_only": True, "touches_amp_optimizer_rng": False}


def update_rcrmd_gradient_audit_receipt(
    receipt: Mapping[str, Any], audit: Mapping[str, Any]
) -> Dict[str, Any]:
    """Seal the first active-q raw auxiliary VJP audit exactly once."""

    result = dict(receipt)
    if bool(result.get("rcrmd_gradient_audit_completed", False)):
        raise RCRMDRuntimeError("P1-RCRMD VJP audit may run only once")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
    ):
        raise RCRMDRuntimeError("P1-RCRMD VJP audit must be raw, diagnostic-only, and state-free")
    for group_name in ("shared_encoder", "classifier_head"):
        values = audit.get(group_name)
        if not isinstance(values, Mapping):
            raise RCRMDRuntimeError("P1-RCRMD VJP audit lacks a required scope")
        count = float(values.get("parameter_count", 0.0))
        norm = float(values.get("norm", float("nan")))
        if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
            raise RCRMDRuntimeError("P1-RCRMD encoder/head raw VJP is zero or non-finite")
    result["rcrmd_gradient_audit_attempted"] = True
    result["rcrmd_gradient_audit_completed"] = True
    result["rcrmd_gradient_audit"] = dict(audit)
    return result


def bind_rcrmd_source_data_order(
    receipt: Mapping[str, Any], source_split_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind source-L physical order and its source-only physical RX allowlist."""

    result = dict(receipt)
    source = dict(source_split_receipt or {})
    labeled_sha = str(source.get("labeled_indices_sha256", "") or "")
    manifest_sha = str(source.get("split_manifest_sha256", "") or "")
    if len(labeled_sha) != 64 or len(manifest_sha) != 64:
        raise RCRMDConfigurationError(
            "P1-RCRMD requires labeled-index and source-split SHA256 receipts"
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
    *,
    receiver_ids: Sequence[Any],
    labels: torch.Tensor,
    rx_labels: torch.Tensor,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    receivers = _source_receiver_ids(receiver_ids)
    counts = {
        _receiver_key(receiver_id, class_id): int(
            (rx_labels.eq(receiver_id) & labels.eq(class_id)).sum().item()
        )
        for receiver_id in receivers
        for class_id in FROZEN_RCRMD_CLASS_IDS
    }
    if sum(counts.values()) != int(labels.numel()):
        raise RCRMDRuntimeError("P1-RCRMD common n_rc counters do not close")
    return counts, _batch_cell_weights(receivers, counts)


def update_rcrmd_common_batch_sequence_receipt(
    receipt: Mapping[str, Any],
    *,
    epoch: int,
    batch_index: int,
    scenario: str,
    source_tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Chain C/G-identical physical/RX/class/scene coverage and n_rc receipts.

    This is receipt-only.  It neither enters the loss nor changes input order,
    sampler state, or model execution.  ``base_index`` is opaque physical
    identity metadata; ``sig_i`` is accepted only as the existing fallback.
    """

    result = dict(receipt)
    expected = FROZEN_RCRMD_SCENARIOS[(int(epoch) + int(batch_index) - 2) % 3]
    if str(scenario) != expected:
        raise RCRMDRuntimeError("P1-RCRMD common LEO scenario sequence drifted")
    labels = source_tx_labels.detach().reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCRMDRuntimeError("P1-RCRMD common sequence requires local4 source L labels")
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    rx_labels = _validated_receiver_labels(
        source_rx_labels, rows=int(labels.numel()), expected_receiver_ids=receivers
    ).detach()
    if metadata is None:
        raise RCRMDRuntimeError("P1-RCRMD common sequence requires opaque physical metadata")
    opaque_ids = _as_plain_list(metadata.get("base_index"))
    if len(opaque_ids) != int(labels.numel()):
        opaque_ids = _as_plain_list(metadata.get("sig_i"))
    if len(opaque_ids) != int(labels.numel()):
        raise RCRMDRuntimeError("P1-RCRMD physical batch sequence metadata is incomplete")
    counts, weights = _common_cell_event(
        receiver_ids=receivers, labels=labels, rx_labels=rx_labels
    )
    event = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "scenario": str(scenario),
        "rows": [
            [str(opaque), int(label), int(receiver_id)]
            for opaque, label, receiver_id in zip(
                opaque_ids, labels.cpu().tolist(), rx_labels.cpu().tolist()
            )
        ],
        "n_rc": counts,
        "effective_weights": weights,
    }
    cell_event = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "scenario": str(scenario),
        "n_rc": counts,
        "effective_weights": weights,
    }
    prior = str(result.get("common_batch_sequence_sha256", "") or "")
    if not prior:
        prior = str(result.get("source_labeled_indices_sha256", "") or "")
    if len(prior) != 64:
        raise RCRMDRuntimeError("P1-RCRMD common batch sequence lacks source data-order SHA256")
    result["common_batch_sequence_sha256"] = hashlib.sha256(
        (prior + "\n" + json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()
    result["common_batch_sequence_batches"] = int(result.get("common_batch_sequence_batches", 0)) + 1
    result["common_batch_sequence_rows"] = int(result.get("common_batch_sequence_rows", 0)) + int(labels.numel())
    scenario_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if set(scenario_batches) != set(FROZEN_RCRMD_SCENARIOS):
        raise RCRMDRuntimeError("P1-RCRMD common scenario receipt is malformed")
    scenario_batches[str(scenario)] += 1
    result["common_scenario_batches"] = scenario_batches
    common_cells = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("rcrmd_common_cells", {})).items()
    }
    scene_cells = common_cells.get(str(scenario), _cell_template(receivers))
    expected_keys = set(_cell_template(receivers))
    if set(scene_cells) != expected_keys:
        raise RCRMDRuntimeError("P1-RCRMD common receiver/class cells are malformed")
    for key in sorted(expected_keys):
        count = int(counts[key])
        weight = dict(weights[key])
        if (
            not math.isfinite(float(weight.get("cell_weight", float("nan"))))
            or not math.isfinite(float(weight.get("row_weight", float("nan"))))
        ):
            raise RCRMDRuntimeError("P1-RCRMD common effective weights are non-finite")
        cell = dict(scene_cells[key])
        cell["rows"] = int(cell.get("rows", 0)) + count
        cell["batches"] = int(cell.get("batches", 0)) + 1
        cell["nonempty_batches"] = int(cell.get("nonempty_batches", 0)) + int(count > 0)
        scene_cells[key] = cell
    common_cells[str(scenario)] = scene_cells
    result["rcrmd_common_cells"] = common_cells
    batch_cells = list(result.get("rcrmd_common_batch_cells", []))
    batch_cells.append(cell_event)
    result["rcrmd_common_batch_cells"] = batch_cells
    return result


def bind_rcrmd_optimizer_initial_state(
    receipt: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Seal the newly created AdamW state before the first backward call."""

    result = dict(receipt)
    optimizer_type = type(optimizer).__name__
    if optimizer_type != FROZEN_RCRMD_OPTIMIZER_TYPE:
        raise RCRMDConfigurationError(
            "P1-RCRMD requires optimizer_type=AdamW, "
            f"got {optimizer_type or '<empty>'}"
        )
    state = optimizer.state_dict()
    if dict(state.get("state", {})):
        raise RCRMDConfigurationError("P1-RCRMD requires a new AdamW state, not a restored optimizer")
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


def _accumulate_aux_cell(
    cells: Dict[str, Dict[str, Any]],
    *,
    key: str,
    info: Mapping[str, Any],
) -> None:
    count = int(info.get("n_rc", -1))
    active = int(info.get("active_q", -1))
    finite = int(info.get("finite_q", -1))
    q_sum = float(info.get("q_sum", float("nan")))
    g_value = float(info.get("g_rc", float("nan")))
    loss_value = float(info.get("loss_contribution", float("nan")))
    if (
        count < 0
        or active < 0
        or active > count
        or finite != count
        or not all(math.isfinite(value) for value in (q_sum, g_value, loss_value))
    ):
        raise RCRMDRuntimeError("P1-RCRMD G cell counters or values are malformed")
    cell = dict(
        cells.get(
            key,
            {
                "rows": 0,
                "active_q": 0,
                "finite_q": 0,
                "q_sum": 0.0,
                "g_sum": 0.0,
                "loss_sum": 0.0,
                "batches": 0,
                "nonempty_batches": 0,
                "finite_batches": 0,
            },
        )
    )
    cell["rows"] = int(cell.get("rows", 0)) + count
    cell["active_q"] = int(cell.get("active_q", 0)) + active
    cell["finite_q"] = int(cell.get("finite_q", 0)) + finite
    cell["q_sum"] = float(cell.get("q_sum", 0.0)) + q_sum
    cell["g_sum"] = float(cell.get("g_sum", 0.0)) + g_value
    cell["loss_sum"] = float(cell.get("loss_sum", 0.0)) + loss_value
    cell["batches"] = int(cell.get("batches", 0)) + 1
    cell["nonempty_batches"] = int(cell.get("nonempty_batches", 0)) + int(count > 0)
    cell["finite_batches"] = int(cell.get("finite_batches", 0)) + 1
    if int(cell["active_q"]) > int(cell["rows"]) or int(cell["finite_q"]) != int(cell["rows"]):
        raise RCRMDRuntimeError("P1-RCRMD cumulative G cell counters do not close")
    cells[key] = cell


def update_rcrmd_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    scenario: str,
    epoch: int,
    batch_index: int,
) -> Dict[str, Any]:
    """Accumulate G-only q/loss evidence after common C/G coverage is sealed."""

    result = dict(receipt)
    if str(result.get("schema", "")) != RCRMD_RECEIPT_SCHEMA:
        raise RCRMDRuntimeError("P1-RCRMD receipt schema is invalid")
    if result.get("enabled") is not True:
        raise RCRMDRuntimeError("P1-RCRMD auxiliary receipt update is G-arm only")
    if str(scenario) not in FROZEN_RCRMD_SCENARIOS:
        raise RCRMDRuntimeError("P1-RCRMD scenario is outside frozen clear/low/rain cycle")
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    if tuple(int(value) for value in batch_info.get("source_receiver_ids", ())) != receivers:
        raise RCRMDRuntimeError("P1-RCRMD G batch source receiver allowlist drifted")
    if tuple(int(value) for value in result.get("expected_tx_class_ids", [])) != FROZEN_RCRMD_CLASS_IDS:
        raise RCRMDRuntimeError("P1-RCRMD receipt lacks local4 class binding")
    if batch_info.get("finite") is not True or batch_info.get("clean_margin_detached") is not True:
        raise RCRMDRuntimeError("P1-RCRMD finite/detached-clean receipt contract drifted")
    if (
        batch_info.get("empty_cell_zero") is not True
        or batch_info.get("no_active_renormalization") is not True
        or batch_info.get("no_cross_sample_or_cross_receiver_pairing") is not True
    ):
        raise RCRMDRuntimeError("P1-RCRMD empty-cell or fixed-scale contract drifted")
    if int(batch_info.get("global_denominator", -1)) != 4 * len(receivers):
        raise RCRMDRuntimeError("P1-RCRMD global denominator drifted")
    scale = float(batch_info.get("fixed_scale", float("nan")))
    if not math.isfinite(scale) or abs(scale - 1.0 / float(4 * len(receivers))) > _TOLERANCE:
        raise RCRMDRuntimeError("P1-RCRMD fixed scale drifted")
    cells = {str(key): dict(value) for key, value in dict(batch_info.get("cells", {})).items()}
    expected_keys = set(_cell_template(receivers))
    if set(cells) != expected_keys:
        raise RCRMDRuntimeError("P1-RCRMD G receipt lacks all receiver×class cells")
    total_rows = int(batch_info.get("rows", -1))
    active_q = int(batch_info.get("active_q", -1))
    finite_q = int(batch_info.get("finite_q", -1))
    loss_sum = float(batch_info.get("loss_sum", float("nan")))
    if total_rows <= 0 or active_q < 0 or active_q > total_rows or finite_q != total_rows or not math.isfinite(loss_sum):
        raise RCRMDRuntimeError("P1-RCRMD G batch rows/active/finite/loss do not close")
    common_events = list(result.get("rcrmd_common_batch_cells", []))
    if not common_events:
        raise RCRMDRuntimeError("P1-RCRMD G batch lacks its common C/G coverage receipt")
    common_event = dict(common_events[-1])
    if (
        int(common_event.get("epoch", -1)) != int(epoch)
        or int(common_event.get("batch_index", -1)) != int(batch_index)
        or str(common_event.get("scenario", "")) != str(scenario)
    ):
        raise RCRMDRuntimeError("P1-RCRMD G/common receipt batch alignment drifted")
    common_counts = {str(key): int(value) for key, value in dict(common_event.get("n_rc", {})).items()}
    if common_counts != {key: int(value.get("n_rc", -1)) for key, value in cells.items()}:
        raise RCRMDRuntimeError("P1-RCRMD G/common n_rc receipt mismatch")
    common_weights = {
        str(key): {str(field): float(field_value) for field, field_value in dict(value).items()}
        for key, value in dict(common_event.get("effective_weights", {})).items()
    }
    for key in expected_keys:
        cell = cells[key]
        weight = common_weights.get(key, {})
        if (
            abs(float(cell.get("cell_weight", float("nan"))) - float(weight.get("cell_weight", float("nan")))) > _TOLERANCE
            or abs(float(cell.get("row_weight", float("nan"))) - float(weight.get("row_weight", float("nan")))) > _TOLERANCE
        ):
            raise RCRMDRuntimeError("P1-RCRMD G/common effective weight mismatch")
    scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(scene_cells).items()}
        for scene, scene_cells in dict(result.get("rcrmd_scenes", {})).items()
    }
    scene_cells = scenes.get(str(scenario), _cell_template(receivers))
    if set(scene_cells) != expected_keys:
        raise RCRMDRuntimeError("P1-RCRMD G scene cells are malformed")
    for key in sorted(expected_keys):
        _accumulate_aux_cell(scene_cells, key=key, info=cells[key])
    scenes[str(scenario)] = scene_cells
    result["rcrmd_scenes"] = scenes
    aux_events = list(result.get("rcrmd_g_batch_aux", []))
    aux_events.append(
        {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "scenario": str(scenario),
            "active_q": active_q,
            "loss_sum": loss_sum,
            "cell_active_q": {key: int(cells[key]["active_q"]) for key in sorted(expected_keys)},
            "cell_loss_sum": {key: float(cells[key]["loss_contribution"]) for key in sorted(expected_keys)},
        }
    )
    result["rcrmd_g_batch_aux"] = aux_events
    result["rcrmd_batches"] = int(result.get("rcrmd_batches", 0)) + 1
    result["rcrmd_total_rows"] = int(result.get("rcrmd_total_rows", 0)) + total_rows
    result["rcrmd_active_q"] = int(result.get("rcrmd_active_q", 0)) + active_q
    result["rcrmd_loss_sum"] = float(result.get("rcrmd_loss_sum", 0.0)) + loss_sum
    return result


def _validate_common_terminal_contract(result: Mapping[str, Any]) -> None:
    if str(result.get("schema", "")) != RCRMD_RECEIPT_SCHEMA:
        raise RCRMDRuntimeError("P1-RCRMD terminal receipt schema is invalid")
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
            raise RCRMDRuntimeError(f"P1-RCRMD terminal receipt lacks {key}")
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    if int(result.get("source_receiver_count", 0)) != FROZEN_RCRMD_SOURCE_RECEIVER_COUNT:
        raise RCRMDRuntimeError("P1-RCRMD terminal frozen source receiver count drifted")
    if tuple(int(value) for value in result.get("frozen_source_receiver_ids", ())) != FROZEN_RCRMD_SOURCE_RECEIVER_IDS:
        raise RCRMDRuntimeError("P1-RCRMD terminal frozen source receiver-id receipt drifted")
    if int(result.get("frozen_source_receiver_count", 0)) != FROZEN_RCRMD_SOURCE_RECEIVER_COUNT:
        raise RCRMDRuntimeError("P1-RCRMD terminal frozen source receiver-count receipt drifted")
    if int(result.get("frozen_cells_per_scene", 0)) != (
        len(FROZEN_RCRMD_CLASS_IDS) * FROZEN_RCRMD_SOURCE_RECEIVER_COUNT
    ):
        raise RCRMDRuntimeError("P1-RCRMD terminal frozen 28-cell-per-scene receipt drifted")
    ledger_tolerance = float(result.get("rcrmd_float32_ledger_rel_tolerance", float("nan")))
    if (
        not math.isfinite(ledger_tolerance)
        or abs(ledger_tolerance - FROZEN_RCRMD_FLOAT32_LEDGER_REL_TOL) > _TOLERANCE
    ):
        raise RCRMDRuntimeError("P1-RCRMD terminal float32 ledger tolerance drifted")
    if str(result.get("checkpoint_role", "") or "") != "training_final_only":
        raise RCRMDRuntimeError("P1-RCRMD requires a training_final_only warm-start checkpoint")
    if result.get("optimizer_state_restored") is not False or result.get("rng_state_restored") is not False:
        raise RCRMDRuntimeError("P1-RCRMD optimizer/RNG restoration is forbidden")
    if str(result.get("optimizer_type", "")) != FROZEN_RCRMD_OPTIMIZER_TYPE:
        raise RCRMDRuntimeError("P1-RCRMD terminal optimizer_type must be AdamW")
    if result.get("optimizer_initial_state_empty") is not True:
        raise RCRMDRuntimeError("P1-RCRMD missing new AdamW initial-state receipt")
    batches = int(result.get("common_batch_sequence_batches", 0))
    rows = int(result.get("common_batch_sequence_rows", 0))
    scenarios = {str(key): int(value) for key, value in dict(result.get("common_scenario_batches", {})).items()}
    if batches <= 0 or rows <= 0 or set(scenarios) != set(FROZEN_RCRMD_SCENARIOS) or any(value <= 0 for value in scenarios.values()):
        raise RCRMDRuntimeError("P1-RCRMD common batch/scenario receipt is incomplete")
    expected_keys = set(_cell_template(receivers))
    if len(expected_keys) != len(FROZEN_RCRMD_CLASS_IDS) * FROZEN_RCRMD_SOURCE_RECEIVER_COUNT:
        raise RCRMDRuntimeError("P1-RCRMD terminal expected receiver/class cells are not frozen 28")
    common_scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(scene_cells).items()}
        for scene, scene_cells in dict(result.get("rcrmd_common_cells", {})).items()
    }
    if set(common_scenes) != set(FROZEN_RCRMD_SCENARIOS):
        raise RCRMDRuntimeError("P1-RCRMD terminal common receiver/class/scene coverage is incomplete")
    total_cell_rows = 0
    for scenario in FROZEN_RCRMD_SCENARIOS:
        scene_cells = common_scenes[scenario]
        if set(scene_cells) != expected_keys:
            raise RCRMDRuntimeError("P1-RCRMD terminal common receiver/class cells drifted")
        scene_rows = 0
        for key in expected_keys:
            cell = scene_cells[key]
            cell_rows = int(cell.get("rows", -1))
            cell_batches = int(cell.get("batches", -1))
            nonempty = int(cell.get("nonempty_batches", -1))
            if cell_rows <= 0 or cell_batches != scenarios[scenario] or nonempty <= 0 or nonempty > cell_batches:
                raise RCRMDRuntimeError("P1-RCRMD terminal r×c×scene common coverage failed")
            scene_rows += cell_rows
        total_cell_rows += scene_rows
    if total_cell_rows != rows:
        raise RCRMDRuntimeError("P1-RCRMD terminal common n_rc rows do not close")
    batch_cells = list(result.get("rcrmd_common_batch_cells", []))
    if len(batch_cells) != batches:
        raise RCRMDRuntimeError("P1-RCRMD terminal per-batch n_rc receipt is incomplete")
    observed_rows = 0
    for event in batch_cells:
        event_counts = {str(key): int(value) for key, value in dict(event.get("n_rc", {})).items()}
        if set(event_counts) != expected_keys or any(value < 0 for value in event_counts.values()):
            raise RCRMDRuntimeError("P1-RCRMD terminal batch n_rc keys are invalid")
        weights = {
            str(key): {str(field): float(value) for field, value in dict(mapping).items()}
            for key, mapping in dict(event.get("effective_weights", {})).items()
        }
        expected_weights = _batch_cell_weights(receivers, event_counts)
        if set(weights) != expected_keys:
            raise RCRMDRuntimeError("P1-RCRMD terminal batch effective weights are incomplete")
        for key in expected_keys:
            for field in ("cell_weight", "row_weight"):
                actual = float(weights[key].get(field, float("nan")))
                expected_weight = float(expected_weights[key][field])
                if not math.isfinite(actual) or abs(actual - expected_weight) > _TOLERANCE:
                    raise RCRMDRuntimeError("P1-RCRMD terminal fixed effective weight drifted")
        observed_rows += sum(event_counts.values())
    if observed_rows != rows:
        raise RCRMDRuntimeError("P1-RCRMD terminal per-batch n_rc rows do not close")


def validate_rcrmd_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless common coverage and G-only q/VJP evidence close."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    _validate_common_terminal_contract(result)
    enabled = result.get("enabled")
    if enabled is not True and enabled is not False:
        raise RCRMDRuntimeError("P1-RCRMD terminal enabled flag must be strict bool")
    if enabled is False:
        forbidden_nonzero = ("rcrmd_batches", "rcrmd_total_rows", "rcrmd_active_q")
        if any(int(result.get(key, 0)) != 0 for key in forbidden_nonzero) or abs(float(result.get("rcrmd_loss_sum", 0.0))) > _TOLERANCE:
            raise RCRMDRuntimeError("P1-RCRMD C arm must retain zero auxiliary counters")
        if any(
            bool(result.get(key))
            for key in ("rcrmd_scenes", "rcrmd_g_batch_aux", "rcrmd_gradient_audit")
        ) or bool(result.get("rcrmd_gradient_audit_attempted", False)) or bool(result.get("rcrmd_gradient_audit_completed", False)):
            raise RCRMDRuntimeError("P1-RCRMD C arm must retain N/A-or-zero auxiliary fields")
        result["rcrmd_terminal_contract"] = "CONTROL_ARM_COMMON_PHYSICAL_RX_CLASS_SCENE_COVERAGE_AND_N_RC_BOUND_AUX_NA_OR_ZERO"
        result["rcrmd_terminal_contract_passed"] = True
        return result
    receivers = _require_frozen_source_receivers(result.get("source_receiver_ids", ()))
    expected_keys = set(_cell_template(receivers))
    common_scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(scene_cells).items()}
        for scene, scene_cells in dict(result.get("rcrmd_common_cells", {})).items()
    }
    scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(scene_cells).items()}
        for scene, scene_cells in dict(result.get("rcrmd_scenes", {})).items()
    }
    if set(scenes) != set(FROZEN_RCRMD_SCENARIOS):
        raise RCRMDRuntimeError("P1-RCRMD terminal G receiver/class/scene coverage is incomplete")
    total_rows = 0
    total_active = 0
    total_loss = 0.0
    for scenario in FROZEN_RCRMD_SCENARIOS:
        scene_cells = scenes[scenario]
        if set(scene_cells) != expected_keys:
            raise RCRMDRuntimeError("P1-RCRMD terminal G receiver/class cells drifted")
        for key in expected_keys:
            cell = scene_cells[key]
            common_cell = common_scenes[scenario][key]
            rows = int(cell.get("rows", -1))
            active = int(cell.get("active_q", -1))
            finite = int(cell.get("finite_q", -1))
            batches = int(cell.get("batches", -1))
            finite_batches = int(cell.get("finite_batches", -1))
            nonempty = int(cell.get("nonempty_batches", -1))
            loss = float(cell.get("loss_sum", float("nan")))
            if (
                rows <= 0
                or rows != int(common_cell.get("rows", -2))
                or active < 0
                or active > rows
                or finite != rows
                or batches != int(common_cell.get("batches", -2))
                or finite_batches != batches
                or nonempty != int(common_cell.get("nonempty_batches", -2))
                or not math.isfinite(loss)
            ):
                raise RCRMDRuntimeError("P1-RCRMD terminal G r×c×scene receipt does not close")
            total_rows += rows
            total_active += active
            total_loss += loss
    common_rows = int(result.get("common_batch_sequence_rows", 0))
    if (
        int(result.get("rcrmd_batches", -1)) != int(result.get("common_batch_sequence_batches", -2))
        or int(result.get("rcrmd_total_rows", -1)) != common_rows
        or total_rows != common_rows
        or int(result.get("rcrmd_active_q", -1)) != total_active
        or total_active <= 0
        or not _float32_ledger_close(
            float(result.get("rcrmd_loss_sum", float("nan"))),
            total_loss,
        )
    ):
        raise RCRMDRuntimeError("P1-RCRMD terminal G batch/active/loss counters do not close")
    aux_events = list(result.get("rcrmd_g_batch_aux", []))
    if len(aux_events) != int(result.get("rcrmd_batches", 0)):
        raise RCRMDRuntimeError("P1-RCRMD terminal G per-batch auxiliary receipt is incomplete")
    if sum(int(event.get("active_q", -1)) for event in aux_events) != total_active:
        raise RCRMDRuntimeError("P1-RCRMD terminal G per-batch active_q does not close")
    if not bool(result.get("rcrmd_gradient_audit_completed", False)):
        raise RCRMDRuntimeError("P1-RCRMD terminal first-active raw encoder/head VJP audit is incomplete")
    result["rcrmd_terminal_contract"] = (
        "FORMAL_COMMON_PHYSICAL_RX_CLASS_SCENE_N_RC_FIXED_SCALE_WITH_G_ONLY_ACTIVE_Q_LOSS_"
        "AND_FIRST_ACTIVE_RAW_ENCODER_AND_EXACT_HEAD_VJP"
    )
    result["rcrmd_terminal_contract_passed"] = True
    return result


def _failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "vjp" in message or "gradient" in message or "head" in message:
        return "RCRMD_AUX_GRADIENT_PATH_FAILURE"
    if "non-finite" in message or "nonfinite" in message:
        return "RCRMD_NONFINITE"
    if "receiver" in message or "rx_i" in message or "r×c" in message:
        return "RCRMD_SOURCE_RX_OR_CELL_COVERAGE_FAILURE"
    if "sequence" in message or "receipt" in message or "coverage" in message:
        return "RCRMD_RECEIPT_CLOSURE_FAILURE"
    if "binding" in message or "logit" in message or "margin" in message:
        return "RCRMD_BINDING_FAILURE"
    return "RCRMD_RUNTIME_FAILURE"


def write_rcrmd_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free fail-closed record for the RCRMD arm."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise RCRMDRuntimeError("P1-RCRMD failure receipt requires an existing output directory")
    target = target_dir / "rcrmd_failure_receipt.json"
    payload = {
        "schema": "cvs.phase1.rcrmd_failure_receipt.v1",
        "status": "FAIL_CLOSED",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage),
        "error_type": type(error).__name__,
        "error_fingerprint": _failure_fingerprint(error),
        "rcrmd_receipt": dict(receipt),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = mkstemp(prefix=".rcrmd_failure_receipt.", suffix=".tmp", dir=str(target_dir))
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


def strict_rcrmd_warm_start(
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
        raise RCRMDConfigurationError(
            "Frozen P1-RCRMD warm-start requires model state, path, and SHA256"
        )
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise RCRMDConfigurationError(
            f"Frozen P1-RCRMD strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RCRMDConfigurationError(
            "Frozen P1-RCRMD strict baseline model-key mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    if str(checkpoint_role or "") != "training_final_only":
        raise RCRMDConfigurationError(
            "Frozen P1-RCRMD requires baseline checkpoint_role=training_final_only"
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
