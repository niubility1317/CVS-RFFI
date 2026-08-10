"""Frozen P1-RCMMC continuation contract for Phase1 source-only DG.

P1-RCMMC (Receiver-Conditional Moment-Matrix Congruence) adds one
source-L-only auxiliary term to the existing same-physical clean/single-LEO
``feat_joint`` path.  It never creates a view, model, parameter, cache, or
cross-batch state.  The clean statistics are stop-gradient anchors; only the
LEO feature and shared encoder receive auxiliary gradients.

The term intentionally records no IQ, feature, moment matrix, physical key,
or receiver token in its persisted receipt.  Source receiver tokens are read
only from the sealed split receipt at run setup, retained in the trainer as
ephemeral routing metadata, and represented in the receipt by an ordered SHA.

RCAT is a strict rowwise refinement of this moment constraint: RCAT loss zero
implies RCMMC loss zero.  RCMMC must not be described as bidirectionally
zero-set incomparable with RCAT.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


FROZEN_RCMMC_LAMBDA = 0.02
FROZEN_RCMMC_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_RCMMC_CLASS_IDS = (0, 1, 2, 3)
FROZEN_RCMMC_BATCH_SIZE = 128
FROZEN_RCMMC_FEATURE_DIM = 160
FROZEN_RCMMC_SOURCE_RECEIVER_COUNT = 7
FROZEN_RCMMC_CELL_COUNT = FROZEN_RCMMC_SOURCE_RECEIVER_COUNT * len(FROZEN_RCMMC_CLASS_IDS)
FROZEN_RCMMC_TERM_DIVISOR = FROZEN_RCMMC_CELL_COUNT
FROZEN_RCMMC_OPTIMIZER_TYPE = "AdamW"
FROZEN_RCMMC_FLOAT32_LEDGER_REL_TOL = 32.0 * float(torch.finfo(torch.float32).eps)
FROZEN_RCMMC_AUX_TENSOR_BOUND_FORMULA = "4[32Bd+4*28(d+d^2+(d+1)^2)]"
RCMMC_RECEIPT_SCHEMA = "cvs.phase1.rcmmc_receipt.v1"
_TOLERANCE = 1e-12


class RCMMCConfigurationError(ValueError):
    """Raised when the frozen P1-RCMMC C/G contract drifts."""


class RCMMCRuntimeError(RuntimeError):
    """Raised when P1-RCMMC data, gradients, or receipts cannot be proved."""


@dataclass(frozen=True)
class RCMMCConfig:
    """Immutable P1-RCMMC controls consumed by the common training loop."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise RCMMCConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
        raise RCMMCConfigurationError(
            f"Frozen P1-RCMMC requires {name}={expected:.12g}, got {actual!r}"
        )


def _float32_ledger_close(actual: float, expected: float) -> bool:
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    return abs(actual - expected) <= FROZEN_RCMMC_FLOAT32_LEDGER_REL_TOL * max(
        1.0, abs(actual), abs(expected)
    )


def _require_disabled(args: Any, names: Sequence[str]) -> None:
    active = []
    for name in names:
        value = getattr(args, name, False)
        if isinstance(value, bool):
            enabled = bool(value)
        else:
            try:
                enabled = abs(float(value)) > _TOLERANCE
            except (TypeError, ValueError):
                enabled = bool(value)
        if enabled:
            active.append(str(name))
    if active:
        raise RCMMCConfigurationError(
            "Frozen P1-RCMMC forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    scenarios = tuple(
        part.strip().lower().replace("-", "_")
        for part in str(value or "").split(",")
        if part.strip()
    )
    if scenarios != FROZEN_RCMMC_SCENARIOS:
        raise RCMMCConfigurationError(
            "Frozen P1-RCMMC requires --sat_train_scenarios "
            + ",".join(FROZEN_RCMMC_SCENARIOS)
        )
    return scenarios


def validate_rcmmc_args(args: Any) -> RCMMCConfig:
    """Validate the frozen common-base C/G contract before data are loaded."""

    frozen_mode = _bool_arg(args, "phase1_rcmmc_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_rcmmc_enabled", False)
    loss_weight = _float_arg(args, "lambda_rcmmc", 0.0)
    if not frozen_mode and not enabled:
        return RCMMCConfig(False, False, 0.0)
    if enabled and not frozen_mode:
        raise RCMMCConfigurationError(
            "--phase1_rcmmc_enabled requires --phase1_rcmmc_frozen_mode true"
        )
    _require_close("lambda_rcmmc", loss_weight, FROZEN_RCMMC_LAMBDA if enabled else 0.0)
    if bool(getattr(args, "from_scratch", True)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC must train the shared feat_joint encoder")
    if not bool(getattr(args, "amp", True)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires the common AMP training path")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires --id_feature_key feat_joint")
    if int(getattr(args, "batch_size", 0)) != FROZEN_RCMMC_BATCH_SIZE:
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires --batch_size 128")
    if int(getattr(args, "epochs", 0)) != 40 or int(getattr(args, "label_epochs", 0)) != 40:
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires exactly 40 labeled epochs")
    if int(getattr(args, "pseudo_epochs", 0)) != 0:
        raise RCMMCConfigurationError("Frozen P1-RCMMC forbids pseudo epochs")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise RCMMCConfigurationError("Frozen P1-RCMMC forbids --sat_view_schedule overrides")
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires non-concatenated single-LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC permits only source_known_train L updates")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC forbids RX/day-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise RCMMCConfigurationError("Frozen P1-RCMMC forbids a reject head")
    _require_disabled(
        args,
        (
            "phase1_ccpc_leo_frozen_mode", "phase1_ccpc_leo_enabled", "phase1_ccpc_leo_gradient_audit_only", "lambda_ccpc_leo",
            "phase1_pamr_frozen_mode", "phase1_pamr_enabled", "phase1_pamr_audit_only", "lambda_pamr",
            "phase1_cb_sfce_frozen_mode", "phase1_cb_sfce_enabled", "lambda_cb_sfce",
            "phase1_gd_proto_nll_frozen_mode", "phase1_gd_proto_nll_enabled", "lambda_gd_proto_nll",
            "phase1_icmt_frozen_mode", "phase1_icmt_enabled", "lambda_icmt",
            "phase1_cagm_frozen_mode", "phase1_cagm_enabled", "lambda_cagm",
            "phase1_rcrmd_frozen_mode", "phase1_rcrmd_enabled", "lambda_rcrmd",
            "phase1_rcat_frozen_mode", "phase1_rcat_enabled", "lambda_rcat",
            "phase1_hscf_frozen_mode", "phase1_hscf_enabled", "lambda_hscf",
            "phase1_recte_frozen_mode", "phase1_recte_enabled", "lambda_recte",
            "phase1_cp_sfce_frozen_mode", "phase1_cp_sfce_enabled", "lambda_cp_sfce",
            "lambda_domain", "lambda_adv", "lambda_orth", "lambda_cons", "lambda_group_ce", "lambda_fishr",
            "lambda_u", "lambda_ent", "lambda_u_domain", "lambda_u_adv", "lambda_u_sat_cons",
            "lambda_u_direct_metric_accept", "lambda_u_quarantine_accept", "lambda_zid_receiver_invariance",
            "lambda_zid_day_invariance", "lambda_zid_channel_invariance", "lambda_u_zid_receiver_invariance",
            "lambda_u_zid_day_invariance", "lambda_u_zid_channel_invariance", "lambda_tx_proto",
            "lambda_rx_proto", "lambda_mask_aux", "lambda_tx_supcon_masked", "lambda_rx_supcon_masked",
            "lambda_txrx_rect", "lambda_proto", "lambda_open_world_feat", "lambda_zid_compact",
            "lambda_proxy_unknown", "lambda_manytx_real_oe", "lambda_soft_unknown_mixup", "lambda_source_episode",
            "lambda_direct_metric_accept", "use_phase2_ground_prototypes", "use_feature_masks",
            "use_txrx_geometry_losses", "use_proto_memory", "os_gradient_surgery", "os_budget_controller",
            "os_objective_budget_controller", "phase1_v2_hard_gates", "manytx_real_oe_enabled",
            "manytx_real_oe_protocol_enabled", "use_ema_teacher", "teacher_ckpt", "lambda_teacher_clean_kl",
            "lambda_teacher_sat_kl", "lambda_teacher_zid_mse",
        ),
    )
    return RCMMCConfig(True, enabled, loss_weight)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def rcmmc_shape_ledger(
    *,
    batch_size: int = FROZEN_RCMMC_BATCH_SIZE,
    feature_dim: int = FROZEN_RCMMC_FEATURE_DIM,
    cell_count: int = FROZEN_RCMMC_CELL_COUNT,
) -> Dict[str, Any]:
    """Return the conservative FP32 live-tensor accounting contract.

    The bound includes clean/LEO ``mu/Q/M`` shape envelopes for all 28 cells
    and one backward-prepared autograd-saved envelope.  RCMMC itself streams
    ``X.T @ X`` cell by cell and never materializes a ``B×d²`` or
    ``B×28×d²`` tensor.  CUDA allocator deltas are deliberately measured by
    the focused resource test, rather than asserted here as a fake exact peak.
    """

    if int(batch_size) != FROZEN_RCMMC_BATCH_SIZE:
        raise RCMMCConfigurationError("P1-RCMMC shape ledger requires B=128")
    if int(feature_dim) != FROZEN_RCMMC_FEATURE_DIM:
        raise RCMMCConfigurationError("P1-RCMMC shape ledger requires d=160")
    if int(cell_count) != FROZEN_RCMMC_CELL_COUNT:
        raise RCMMCConfigurationError("P1-RCMMC shape ledger requires 28 cells")
    b = int(batch_size)
    d = int(feature_dim)
    cells = int(cell_count)
    elements = 32 * b * d + 4 * cells * (d + d * d + (d + 1) * (d + 1))
    return {
        "aux_tensor_bound_formula": FROZEN_RCMMC_AUX_TENSOR_BOUND_FORMULA,
        "batch_size": b,
        "feature_dim": d,
        "cell_count": cells,
        "fp32_element_bytes": 4,
        "conservative_live_tensor_upper_bound_bytes": int(4 * elements),
        "includes_clean_leo_mu_q_m_and_one_saved_autograd_envelope": True,
        "forbids_batch_d2_materialization": True,
        "forbids_batch_cell_d2_materialization": True,
        "cross_batch_cache": False,
        "cuda_peak_delta_required_from_focused_test": True,
    }


def rcmmc_config_receipt(config: RCMMCConfig) -> Dict[str, Any]:
    """Create the data-free receipt skeleton for either frozen C/G arm."""

    return {
        "schema": RCMMC_RECEIPT_SCHEMA,
        "method": "P1_RCMMC",
        "candidate_pattern": "F{1..6}{C|G}_RCMMC12",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "loss_rule": "SOURCE_L_ORDERED_RECEIVER_SLOT_BY_LOCAL4_MOMENT_MATRIX_CONGRUENCE_STOPGRAD_CLEAN_TO_LEO_TOTALIZED_L2_feat_joint",
        "loss_formula": "D_rc=2||mu_L-sg(mu_C)||2^2+||Q_L-sg(Q_C)||F^2;L=sum_rc(A_rc*D_rc)/28",
        "loss_global_denominator": FROZEN_RCMMC_TERM_DIVISOR,
        "local_class_count": len(FROZEN_RCMMC_CLASS_IDS),
        "frozen_batch_size": FROZEN_RCMMC_BATCH_SIZE,
        "frozen_feature_dim": FROZEN_RCMMC_FEATURE_DIM,
        "frozen_source_receiver_count": FROZEN_RCMMC_SOURCE_RECEIVER_COUNT,
        "z_id_key": "feat_joint",
        "training_accumulation_dtype": "float32_OUTSIDE_AMP",
        "clean_feature_detached": True,
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD",
        "receipt_payload": "SCALARS_COUNTS_AND_SHA_ONLY_NO_IQ_FEATURE_MOMENT_MATRIX_OR_RECEIVER_TOKEN",
        "rcat_relation": "RCAT_ZERO_IMPLIES_RCMMC_ZERO_STRICT_RELAXATION_NOT_BIDIRECTIONALLY_INCOMPARABLE",
        "shape_ledger": rcmmc_shape_ledger(),
        "common_lambda_sat_cons": 0.10,
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "common_l_base_head_input_path_verified": False,
        "aux_gradient_scope": "LEO_feat_joint_AND_SHARED_ENCODER_FINITE_NONZERO;EXACT_HEAD_AUX_VJP_NA_NONE_OR_ZERO",
        "uses_new_forward": False,
        "uses_resampling": False,
        "uses_rx_labels": True,
        "rx_permission": "SOURCE_KNOWN_TRAIN_L_PHYSICAL_ID_BOUND_SOURCE_SPLIT_RECEIPT_ORDERED_TOKEN_ONLY",
        "uses_day_labels": False,
        "uses_domain_labels": False,
        "uses_target_rows": False,
        "uses_proxy_rows": False,
        "uses_held_rows": False,
        "uses_unlabeled_rows": False,
        "uses_ema_or_state": False,
        "uses_threshold": False,
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
        "source_train_tx_count": 0,
        "source_known_validation_tx_count": 0,
        "source_proxy_unknown_tx_count": 0,
        "source_partition_sha256": "",
        "source_labeled_indices_sha256": "",
        "source_split_manifest_sha256": "",
        "source_receiver_count": 0,
        "source_receiver_order_sha256": "",
        "source_receiver_ids_sha256": "",
        "source_receiver_provenance": "PENDING_SOURCE_L_ONLY_BINDING",
        "dataset_class_count": 0,
        "local_data_class_count": 0,
        "checkpoint_head_class_count": 0,
        "live_head_class_count": 0,
        "class_order_binding_sha256": "",
        "common_batch_sequence_sha256": "",
        "common_batch_sequence_batches": 0,
        "common_batch_sequence_rows": 0,
        "common_scenario_batches": {scenario: 0 for scenario in FROZEN_RCMMC_SCENARIOS},
        "rcmmc_common_cells": {},
        "rcmmc_common_batch_cells": [],
        "rcmmc_batches": 0,
        "rcmmc_total_rows": 0,
        "rcmmc_positive_d_cells": 0,
        "rcmmc_positive_d_batches": 0,
        "rcmmc_sum_d": 0.0,
        "rcmmc_loss_sum": 0.0,
        "rcmmc_scene_positive_batches": {scenario: 0 for scenario in FROZEN_RCMMC_SCENARIOS},
        "rcmmc_scenes": {},
        "rcmmc_g_batch_aux": [],
        "rcmmc_gradient_audit_attempted": False,
        "rcmmc_gradient_audit_completed": False,
        "rcmmc_gradient_audit": {},
        "rcmmc_terminal_contract": "PENDING",
        "rcmmc_terminal_contract_passed": False,
        "proxy_rows": 0,
        "held_rows": 0,
    }


def _normalized_tx_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RCMMCConfigurationError(f"P1-RCMMC {name} must be a TX class sequence")
    order = tuple(str(value).strip() for value in values)
    if not order or len(order) != len(set(order)) or any(not value for value in order):
        raise RCMMCConfigurationError(f"P1-RCMMC {name} must be non-empty and unique")
    return order


def _positive_count(name: str, value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise RCMMCConfigurationError(f"P1-RCMMC {name} must be an integer") from exc
    if count <= 0:
        raise RCMMCConfigurationError(f"P1-RCMMC {name} must be positive")
    return count


def _source_receiver_tokens(values: Any) -> Tuple[int, ...]:
    """Parse the sealed ordered source receiver list without hard-coded IDs."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RCMMCConfigurationError("P1-RCMMC source_receivers must be a source-split sequence")
    parsed = []
    for value in values:
        text = str(value).strip()
        if not text:
            raise RCMMCConfigurationError("P1-RCMMC source receiver token may not be empty")
        try:
            token = int(text)
        except (TypeError, ValueError) as exc:
            raise RCMMCConfigurationError("P1-RCMMC source receiver token must be an integer") from exc
        if str(token) != text:
            raise RCMMCConfigurationError("P1-RCMMC source receiver token is not canonical")
        parsed.append(token)
    tokens = tuple(parsed)
    if len(tokens) != FROZEN_RCMMC_SOURCE_RECEIVER_COUNT or len(set(tokens)) != len(tokens):
        raise RCMMCConfigurationError("P1-RCMMC requires exactly seven ordered unique source receiver tokens")
    return tokens


def resolve_rcmmc_source_receiver_tokens(source_split_receipt: Mapping[str, Any]) -> Tuple[int, ...]:
    """Read the runtime-only ordered source receiver tokens from the split receipt."""

    source = dict(source_split_receipt or {})
    schema = str(source.get("schema", "") or "")
    if schema and schema != "cvs.phase1.source_split_receipt.v1":
        raise RCMMCConfigurationError("P1-RCMMC requires a source split receipt v1")
    return _source_receiver_tokens(source.get("source_receivers", ()))


def _receiver_key(receiver_slot: int, class_id: int) -> str:
    return f"r{int(receiver_slot)}|c{int(class_id)}"


def _cell_template() -> Dict[str, Dict[str, Any]]:
    return {
        _receiver_key(receiver_slot, class_id): {
            "rows": 0,
            "batches": 0,
            "nonempty_batches": 0,
            "finite_batches": 0,
            "positive_d_cells": 0,
            "clean_zero_rows": 0,
            "leo_zero_rows": 0,
            "both_zero_rows": 0,
            "sum_d": 0.0,
            "loss_sum": 0.0,
        }
        for receiver_slot in range(FROZEN_RCMMC_SOURCE_RECEIVER_COUNT)
        for class_id in FROZEN_RCMMC_CLASS_IDS
    }


def _receiver_positions(receiver_labels: torch.Tensor, receiver_tokens: Sequence[Any], *, rows: int) -> torch.Tensor:
    if not torch.is_tensor(receiver_labels):
        raise RCMMCRuntimeError("P1-RCMMC requires source-L physical rx_i labels")
    values = receiver_labels.reshape(-1).long()
    tokens = _source_receiver_tokens(receiver_tokens)
    if values.numel() != int(rows) or values.numel() == 0:
        raise RCMMCRuntimeError("P1-RCMMC source-L rx_i rows do not align")
    positions = torch.full_like(values, -1)
    for slot, token in enumerate(tokens):
        positions = torch.where(values.eq(int(token)), torch.full_like(values, int(slot)), positions)
    if bool(positions.lt(0).any().item()):
        raise RCMMCRuntimeError("P1-RCMMC rx_i contains a receiver outside source-split R_s")
    return positions


def resolve_rcmmc_local_head_class_binding(
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
    if local_count != len(FROZEN_RCMMC_CLASS_IDS) or len(local) != len(FROZEN_RCMMC_CLASS_IDS):
        raise RCMMCConfigurationError("P1-RCMMC requires exactly four local source-TX rows")
    if local != source or checkpoint != source:
        raise RCMMCConfigurationError("P1-RCMMC local/checkpoint TX order must equal source-train receipt")
    if local_count != len(local) or checkpoint_count != live_count or live_count != local_count:
        raise RCMMCConfigurationError("P1-RCMMC local/head class counts must match")
    if set(local).difference(dataset):
        raise RCMMCConfigurationError("P1-RCMMC local TX labels are absent from dataset order")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(FROZEN_RCMMC_CLASS_IDS),
        "expected_tx_class_ids": list(FROZEN_RCMMC_CLASS_IDS),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = _canonical_sha256(binding)
    return binding


def remap_rcmmc_local_labels_to_head_rows(
    local_labels: torch.Tensor, local_to_head_class_ids: Sequence[Any]
) -> torch.Tensor:
    if not torch.is_tensor(local_labels):
        raise RCMMCRuntimeError("P1-RCMMC local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if mapping != FROZEN_RCMMC_CLASS_IDS:
        raise RCMMCRuntimeError("P1-RCMMC local-to-head mapping must be local4 identity")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCMMCRuntimeError("P1-RCMMC local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_rcmmc_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    raw_model = getattr(model, "_orig_mod", model)
    try:
        head = raw_model.id_backbone.cls_head.head
    except AttributeError as exc:
        raise RCMMCRuntimeError("P1-RCMMC requires model.id_backbone.cls_head.head") from exc
    if not isinstance(head, torch.nn.Module):
        raise RCMMCRuntimeError("P1-RCMMC exact classifier head is not a module")
    if not tuple(parameter for parameter in head.parameters() if parameter.requires_grad):
        raise RCMMCRuntimeError("P1-RCMMC exact classifier head has no trainable parameter")
    return head


def resolve_rcmmc_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    weight = getattr(resolve_rcmmc_classifier_head(model), "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise RCMMCRuntimeError("P1-RCMMC classifier head weight must be a rank-2 Parameter")
    return weight


def _validate_view_binding(
    *, view_name: str, output: Mapping[str, Any], labels: torch.Tensor, head_weight: torch.Tensor
) -> torch.Tensor:
    if str(output.get("z_id_key", "")) != "feat_joint":
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} z_id_key must be feat_joint")
    z_id = output.get("z_id")
    logits = output.get("tx_logits")
    if not torch.is_tensor(z_id) or z_id.ndim != 2:
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} z_id must be rank-2 feat_joint")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} tx_logits must be rank-2 raw logits")
    if z_id.size(0) != labels.numel() or logits.size(0) != labels.numel():
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} rows must align with source L labels")
    if int(head_weight.size(0)) != 4 or int(logits.size(1)) != 4:
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} head/logit class rows must be local4")
    if int(head_weight.size(1)) != int(z_id.size(1)):
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} feat_joint/head dimension binding drifted")
    if not bool(z_id.requires_grad) or not bool(logits.requires_grad):
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} requires a live feat_joint/head path")
    if not bool(torch.isfinite(z_id.detach()).all().item()):
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} feat_joint is non-finite")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} raw logits are non-finite")
    return z_id


def validate_rcmmc_binding(
    *,
    model: torch.nn.Module,
    out_clean: Mapping[str, Any],
    out_leo: Mapping[str, Any],
    tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    expected_class_ids: Sequence[Any],
    source_receiver_tokens: Optional[Sequence[Any]] = None,
    expected_receiver_ids: Optional[Sequence[Any]] = None,
    enforce_frozen_shape: bool = False,
) -> torch.nn.Parameter:
    """Fail closed unless existing forwards expose the frozen 128×160 path."""

    if not isinstance(out_clean, Mapping) or not isinstance(out_leo, Mapping):
        raise RCMMCRuntimeError("P1-RCMMC requires clean and LEO mapping outputs")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() <= 0:
        raise RCMMCRuntimeError("P1-RCMMC source L labels must be non-empty")
    if bool(enforce_frozen_shape) and labels.numel() != FROZEN_RCMMC_BATCH_SIZE:
        raise RCMMCRuntimeError("P1-RCMMC requires exactly B=128 source-L rows")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCMMCRuntimeError("P1-RCMMC source labels must bind to local4 head rows")
    if tuple(int(value) for value in expected_class_ids) != FROZEN_RCMMC_CLASS_IDS:
        raise RCMMCRuntimeError("P1-RCMMC expected local4 class order is invalid")
    tokens = source_receiver_tokens if source_receiver_tokens is not None else expected_receiver_ids
    if tokens is None:
        raise RCMMCRuntimeError("P1-RCMMC requires source-split receiver tokens")
    _receiver_positions(source_rx_labels, tokens, rows=int(labels.numel()))
    head_weight = resolve_rcmmc_classifier_weight(model)
    if not bool(torch.isfinite(head_weight.detach()).all().item()):
        raise RCMMCRuntimeError("P1-RCMMC exact classifier head is non-finite")
    clean_z = _validate_view_binding(view_name="clean", output=out_clean, labels=labels, head_weight=head_weight)
    leo_z = _validate_view_binding(view_name="leo", output=out_leo, labels=labels, head_weight=head_weight)
    if bool(enforce_frozen_shape) and (
        int(clean_z.size(1)) != FROZEN_RCMMC_FEATURE_DIM
        or int(leo_z.size(1)) != FROZEN_RCMMC_FEATURE_DIM
    ):
        raise RCMMCRuntimeError("P1-RCMMC requires feat_joint d=160")
    return head_weight


def _autocast_disabled(features: torch.Tensor):
    if str(features.device.type) in {"cuda", "cpu"}:
        return torch.autocast(device_type=str(features.device.type), enabled=False)
    return nullcontext()


def _totalized_l2_with_zeros(features: torch.Tensor, *, view_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(features) or features.ndim != 2 or features.size(0) <= 0 or features.size(1) <= 0:
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} feat_joint must be non-empty rank-2")
    if not bool(torch.isfinite(features.detach()).all().item()):
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} feat_joint is non-finite")
    values = features.float()
    norms = torch.linalg.vector_norm(values, ord=2, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms.detach()).all().item()):
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} feat_joint norm is non-finite")
    nonzero = norms.gt(0.0)
    # Index only nonzero rows before division: no eager 0/0 intermediate is created.
    totalized = torch.zeros_like(values) + values * 0.0
    if bool(nonzero.any().item()):
        row_mask = nonzero.reshape(-1)
        totalized[row_mask] = values[row_mask] / norms[row_mask]
    if not bool(torch.isfinite(totalized.detach()).all().item()):
        raise RCMMCRuntimeError(f"P1-RCMMC {view_name} totalized-L2 output is non-finite")
    return totalized, ~nonzero.reshape(-1)


def totalized_l2(features: torch.Tensor) -> torch.Tensor:
    """Return float32 ``T(z)`` using the exact safe zero-to-zero rule."""

    with _autocast_disabled(features):
        return _totalized_l2_with_zeros(features, view_name="input")[0]


def _batch_cell_weights(counts: Mapping[str, int]) -> Dict[str, Dict[str, float]]:
    expected = set(_cell_template())
    if set(counts) != expected:
        raise RCMMCRuntimeError("P1-RCMMC batch cell count coverage drifted")
    scale = 1.0 / float(FROZEN_RCMMC_TERM_DIVISOR)
    return {
        key: {
            "cell_weight": scale,
            "row_weight": scale / float(int(counts[key])) if int(counts[key]) > 0 else 0.0,
        }
        for key in sorted(expected)
    }


def rcmmc_loss(
    clean_feat_joint: torch.Tensor,
    leo_feat_joint: torch.Tensor,
    source_tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    source_receiver_tokens: Sequence[Any],
    *,
    require_frozen_shape: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute streamed 28-cell RCMMC without materializing batch×d² tensors."""

    labels = source_tx_labels.reshape(-1).long()
    if labels.numel() <= 0:
        raise RCMMCRuntimeError("P1-RCMMC source-L rows must be non-empty")
    if bool(require_frozen_shape) and labels.numel() != FROZEN_RCMMC_BATCH_SIZE:
        raise RCMMCRuntimeError("P1-RCMMC requires exactly B=128 source-L rows")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCMMCRuntimeError("P1-RCMMC source labels are outside local4")
    positions = _receiver_positions(source_rx_labels, source_receiver_tokens, rows=int(labels.numel()))
    if clean_feat_joint.shape != leo_feat_joint.shape or clean_feat_joint.size(0) != labels.numel():
        raise RCMMCRuntimeError("P1-RCMMC clean/LEO feat_joint rows or dimensions do not align")
    if bool(require_frozen_shape) and int(clean_feat_joint.size(1)) != FROZEN_RCMMC_FEATURE_DIM:
        raise RCMMCRuntimeError("P1-RCMMC requires feat_joint d=160")
    with _autocast_disabled(leo_feat_joint):
        clean_t, clean_zero = _totalized_l2_with_zeros(clean_feat_joint, view_name="clean")
        leo_t, leo_zero = _totalized_l2_with_zeros(leo_feat_joint, view_name="leo")
        clean_t = clean_t.detach()
        zero_scalar = leo_t.sum() * 0.0
        cells: Dict[str, Dict[str, Any]] = {}
        terms = []
        total_rows = total_positive = total_finite = 0
        clean_zero_total = leo_zero_total = both_zero_total = 0
        sum_d = 0.0
        for receiver_slot in range(FROZEN_RCMMC_SOURCE_RECEIVER_COUNT):
            for class_id in FROZEN_RCMMC_CLASS_IDS:
                key = _receiver_key(receiver_slot, class_id)
                mask = positions.eq(receiver_slot) & labels.eq(class_id)
                count = int(mask.sum().item())
                group_clean_zero = clean_zero[mask]
                group_leo_zero = leo_zero[mask]
                clean_zero_count = int(group_clean_zero.sum().item())
                leo_zero_count = int(group_leo_zero.sum().item())
                both_zero_count = int((group_clean_zero & group_leo_zero).sum().item())
                if count == 0:
                    d_rc = zero_scalar
                    positive = 0
                else:
                    clean_group = clean_t[mask]
                    leo_group = leo_t[mask]
                    # Each cell is processed independently in FP32.  ``X.T @ X`` yields
                    # one d×d matrix; no B×d² or B×28×d² tensor is formed.
                    mu_clean = clean_group.mean(dim=0)
                    mu_leo = leo_group.mean(dim=0)
                    q_clean = clean_group.transpose(0, 1).matmul(clean_group) / float(count)
                    q_leo = leo_group.transpose(0, 1).matmul(leo_group) / float(count)
                    q_clean = 0.5 * (q_clean + q_clean.transpose(0, 1))
                    q_leo = 0.5 * (q_leo + q_leo.transpose(0, 1))
                    delta_mu = mu_leo - mu_clean.detach()
                    delta_q = q_leo - q_clean.detach()
                    d_rc = 2.0 * delta_mu.square().sum() + delta_q.square().sum()
                    positive = int(d_rc.detach().gt(0.0).item())
                if not bool(torch.isfinite(d_rc.detach()).item()):
                    raise RCMMCRuntimeError("P1-RCMMC receiver/class D is non-finite")
                d_value = float(d_rc.detach().item())
                if d_value < -_TOLERANCE or d_value > 12.0 + 1e-4:
                    raise RCMMCRuntimeError("P1-RCMMC D violates the totalized-L2 finite bound")
                cells[key] = {
                    "n_rc": count,
                    "occupied": bool(count > 0),
                    "positive_d": positive,
                    "finite_d": 1,
                    "clean_zero_rows": clean_zero_count,
                    "leo_zero_rows": leo_zero_count,
                    "both_zero_rows": both_zero_count,
                    "sum_d": d_value,
                    "loss_contribution": d_value / float(FROZEN_RCMMC_TERM_DIVISOR),
                }
                terms.append(d_rc)
                total_rows += count
                total_positive += positive
                total_finite += 1
                clean_zero_total += clean_zero_count
                leo_zero_total += leo_zero_count
                both_zero_total += both_zero_count
                sum_d += d_value
        if total_rows != int(labels.numel()) or total_finite != FROZEN_RCMMC_CELL_COUNT:
            raise RCMMCRuntimeError("P1-RCMMC batch receiver/class coverage does not close")
        loss = torch.stack(terms).sum() / float(FROZEN_RCMMC_TERM_DIVISOR)
        if not bool(torch.isfinite(loss.detach()).item()):
            raise RCMMCRuntimeError("P1-RCMMC loss is non-finite")
    counts = {key: int(value["n_rc"]) for key, value in cells.items()}
    weights = _batch_cell_weights(counts)
    for key in cells:
        cells[key].update(weights[key])
    return loss, {
        "rows": int(labels.numel()),
        "positive_d_cells": total_positive,
        "finite_d_cells": total_finite,
        "clean_zero_rows": clean_zero_total,
        "leo_zero_rows": leo_zero_total,
        "both_zero_rows": both_zero_total,
        "sum_d": float(sum_d),
        "loss_sum": float(loss.detach().item()),
        "global_denominator": FROZEN_RCMMC_TERM_DIVISOR,
        "fixed_scale": 1.0 / float(FROZEN_RCMMC_TERM_DIVISOR),
        "cells": cells,
        "finite": True,
        "clean_feature_detached": True,
        "totalized_l2_rule": "MASK_NORM_GT_0_THEN_DIVIDE_ELSE_ZERO",
        "training_accumulation_dtype": "float32_OUTSIDE_AMP",
        "empty_cell_zero": True,
        "no_active_renormalization": True,
        "streamed_cell_xtx": True,
        "forbids_batch_d2_materialization": True,
        "forbids_batch_cell_d2_materialization": True,
        "shape_ledger": rcmmc_shape_ledger(),
    }


def add_rcmmc_to_loss(
    base_loss: torch.Tensor, rcmmc: Optional[torch.Tensor], config: Optional[RCMMCConfig]
) -> torch.Tensor:
    """Add the sole G-arm term; C returns the untouched common base tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if rcmmc is None:
        raise RCMMCRuntimeError("Enabled P1-RCMMC requires its auxiliary loss")
    return base_loss + float(config.loss_weight) * rcmmc


def rcmmc_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise RCMMCRuntimeError("P1-RCMMC requires model.id_backbone for VJP audit")
    head = resolve_rcmmc_classifier_head(raw_model)
    head_parameters = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
    excluded = ("cls_head.head.", "con_proj.", "cls_head.imp_merge.", "cls_head.dac_head.", "cls_head.pa_head.")
    encoder = tuple(
        parameter
        for name, parameter in id_backbone.named_parameters()
        if parameter.requires_grad and not str(name).startswith(excluded)
    )
    if not encoder or not head_parameters:
        raise RCMMCRuntimeError("P1-RCMMC shared encoder or exact head audit scope is empty")
    return {"shared_encoder": encoder, "classifier_head": head_parameters}


def _finite_nonzero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.Tensor], *, group_name: str
) -> Dict[str, float]:
    params = tuple(parameters)
    if not params:
        raise RCMMCRuntimeError(f"P1-RCMMC {group_name} VJP scope is empty")
    gradients = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    squared_norm = 0.0
    for gradient in gradients:
        if gradient is None:
            raise RCMMCRuntimeError(f"P1-RCMMC {group_name} VJP is None or detached")
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise RCMMCRuntimeError(f"P1-RCMMC {group_name} VJP is non-finite")
        value = gradient.detach().double()
        squared_norm += float(torch.sum(value * value).item())
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm) or norm <= 0.0:
        raise RCMMCRuntimeError(f"P1-RCMMC {group_name} VJP norm is zero or non-finite")
    return {"parameter_count": float(len(params)), "norm": float(norm)}


def _none_or_zero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.Tensor], *, group_name: str
) -> Dict[str, Any]:
    params = tuple(parameters)
    if not params:
        raise RCMMCRuntimeError(f"P1-RCMMC {group_name} VJP scope is empty")
    gradients = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    none_count = zero_count = 0
    for gradient in gradients:
        if gradient is None:
            none_count += 1
            continue
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise RCMMCRuntimeError(f"P1-RCMMC {group_name} auxiliary VJP is non-finite")
        if int(torch.count_nonzero(gradient.detach()).item()) != 0:
            raise RCMMCRuntimeError(f"P1-RCMMC {group_name} must have no auxiliary gradient")
        zero_count += 1
    return {
        "parameter_count": float(len(params)),
        "none_parameters": float(none_count),
        "zero_parameters": float(zero_count),
        "nonzero_parameters": 0.0,
        "none_or_zero_expected": True,
    }


def _head_none_or_zero_vjp(loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter]) -> Dict[str, Any]:
    return _none_or_zero_vjp(loss, parameters, group_name="classifier head")


def rcmmc_aux_gradient_audit(
    rcmmc: torch.Tensor,
    clean_feat_joint: torch.Tensor,
    feat_joint_leo: torch.Tensor,
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
) -> Dict[str, Any]:
    """Audit the one canonical raw RCMMC VJP before scaled backward.

    The four arguments are mandatory: the clean feature must be present so
    the receipt proves its stop-gradient VJP is None-or-zero, rather than
    merely inferring that fact from the loss construction.
    """

    if not torch.is_tensor(rcmmc) or rcmmc.ndim != 0:
        raise RCMMCRuntimeError("P1-RCMMC VJP audit requires a scalar auxiliary loss")
    if not torch.is_tensor(clean_feat_joint) or clean_feat_joint.ndim != 2:
        raise RCMMCRuntimeError("P1-RCMMC VJP audit requires clean feat_joint")
    if not torch.is_tensor(feat_joint_leo) or feat_joint_leo.ndim != 2:
        raise RCMMCRuntimeError("P1-RCMMC VJP audit requires LEO feat_joint")
    if not isinstance(parameter_groups, Mapping) or tuple(parameter_groups.keys()) != ("shared_encoder", "classifier_head"):
        raise RCMMCRuntimeError("P1-RCMMC VJP audit requires encoder and exact-head scopes")
    return {
        "feat_joint_leo": _finite_nonzero_vjp(rcmmc, (feat_joint_leo,), group_name="feat_joint_leo"),
        "shared_encoder": _finite_nonzero_vjp(
            rcmmc, parameter_groups["shared_encoder"], group_name="shared_encoder"
        ),
        "classifier_head": _head_none_or_zero_vjp(rcmmc, parameter_groups["classifier_head"]),
        "clean_feat_joint": _none_or_zero_vjp(
            rcmmc, (clean_feat_joint,), group_name="clean feat_joint"
        ),
        "raw_unscaled": True,
        "diagnostic_only": True,
        "touches_amp_optimizer_rng": False,
        "exact_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        "clean_feat_joint_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
    }


def _validate_rcmmc_gradient_audit_payload(audit: Mapping[str, Any]) -> None:
    """Revalidate every persisted raw-VJP field without touching autograd."""

    if not isinstance(audit, Mapping):
        raise RCMMCRuntimeError("P1-RCMMC VJP audit receipt is malformed")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
        or audit.get("exact_head_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        or audit.get("clean_feat_joint_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
    ):
        raise RCMMCRuntimeError("P1-RCMMC VJP audit semantics drifted")
    for group_name in ("feat_joint_leo", "shared_encoder"):
        values = audit.get(group_name)
        if not isinstance(values, Mapping):
            raise RCMMCRuntimeError("P1-RCMMC VJP audit lacks required nonzero scope")
        count = float(values.get("parameter_count", 0.0))
        norm = float(values.get("norm", float("nan")))
        if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
            raise RCMMCRuntimeError("P1-RCMMC required auxiliary VJP is zero or non-finite")
    for field_name, label, expected_count in (
        ("classifier_head", "exact-head", None),
        ("clean_feat_joint", "clean", 1.0),
    ):
        values = audit.get(field_name)
        if not isinstance(values, Mapping):
            raise RCMMCRuntimeError(f"P1-RCMMC VJP audit lacks {label} None-or-zero scope")
        count = float(values.get("parameter_count", 0.0))
        none_count = float(values.get("none_parameters", float("nan")))
        zero_count = float(values.get("zero_parameters", float("nan")))
        nonzero_count = float(values.get("nonzero_parameters", float("nan")))
        if (
            count <= 0.0
            or (expected_count is not None and count != expected_count)
            or not all(math.isfinite(value) and value >= 0.0 for value in (none_count, zero_count, nonzero_count))
            or none_count + zero_count != count
            or nonzero_count != 0.0
            or values.get("none_or_zero_expected") is not True
        ):
            raise RCMMCRuntimeError(f"P1-RCMMC {label} auxiliary VJP contract failed")


def update_rcmmc_gradient_audit_receipt(
    receipt: Mapping[str, Any], audit: Mapping[str, Any]
) -> Dict[str, Any]:
    result = dict(receipt)
    if bool(result.get("rcmmc_gradient_audit_completed", False)):
        raise RCMMCRuntimeError("P1-RCMMC VJP audit may run only once")
    _validate_rcmmc_gradient_audit_payload(audit)
    result["rcmmc_gradient_audit_attempted"] = True
    result["rcmmc_gradient_audit_completed"] = True
    result["rcmmc_gradient_audit"] = dict(audit)
    return result


def bind_rcmmc_source_data_order(
    receipt: Mapping[str, Any], source_split_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind source-L ordering and an ordered-token SHA without persisting tokens."""

    result = dict(receipt)
    source = dict(source_split_receipt or {})
    labeled_sha = str(source.get("labeled_indices_sha256", "") or "")
    manifest_sha = str(source.get("split_manifest_sha256", "") or "")
    if len(labeled_sha) != 64 or len(manifest_sha) != 64:
        raise RCMMCConfigurationError("P1-RCMMC requires labeled-index and source-split SHA256 receipts")
    tokens = resolve_rcmmc_source_receiver_tokens(source)
    token_sha = _canonical_sha256([int(value) for value in tokens])
    result["source_labeled_indices_sha256"] = labeled_sha
    result["source_split_manifest_sha256"] = manifest_sha
    result["source_receiver_count"] = len(tokens)
    result["source_receiver_order_sha256"] = token_sha
    result["source_receiver_ids_sha256"] = token_sha
    result["source_receiver_provenance"] = "SOURCE_SPLIT_RECEIPT_ORDERED_SOURCE_RECEIVERS_PHYSICAL_ID_BOUND_L_ONLY"
    return result


def _as_plain_list(values: Any) -> list[Any]:
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return []


def _common_cell_event(
    *, labels: torch.Tensor, receiver_positions: torch.Tensor
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    counts = {
        _receiver_key(receiver_slot, class_id): int(
            (receiver_positions.eq(receiver_slot) & labels.eq(class_id)).sum().item()
        )
        for receiver_slot in range(FROZEN_RCMMC_SOURCE_RECEIVER_COUNT)
        for class_id in FROZEN_RCMMC_CLASS_IDS
    }
    if sum(counts.values()) != int(labels.numel()):
        raise RCMMCRuntimeError("P1-RCMMC common n_rc counters do not close")
    return counts, _batch_cell_weights(counts)


def update_rcmmc_common_batch_sequence_receipt(
    receipt: Mapping[str, Any],
    *,
    epoch: int,
    batch_index: int,
    scenario: str,
    source_tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    source_receiver_tokens: Optional[Sequence[Any]] = None,
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Chain C/G common physical order only through SHA and scalar cell counts."""

    result = dict(receipt)
    expected = FROZEN_RCMMC_SCENARIOS[(int(epoch) + int(batch_index) - 2) % 3]
    if str(scenario) != expected:
        raise RCMMCRuntimeError("P1-RCMMC common LEO scenario sequence drifted")
    labels = source_tx_labels.detach().reshape(-1).long()
    if labels.numel() <= 0:
        raise RCMMCRuntimeError("P1-RCMMC common sequence requires non-empty source L labels")
    if int(labels.numel()) != FROZEN_RCMMC_BATCH_SIZE:
        raise RCMMCRuntimeError("P1-RCMMC common sequence requires B=128 source L labels")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise RCMMCRuntimeError("P1-RCMMC common sequence requires local4 source L labels")
    runtime_tokens = source_receiver_tokens
    if runtime_tokens is None:
        runtime_tokens = result.get("source_receiver_ids", ())
    positions = _receiver_positions(source_rx_labels, runtime_tokens, rows=int(labels.numel())).detach()
    if metadata is None:
        raise RCMMCRuntimeError("P1-RCMMC common sequence requires opaque physical metadata")
    base_indices = _as_plain_list(metadata.get("base_index"))
    signal_indices = _as_plain_list(metadata.get("sig_i"))
    rows = int(labels.numel())
    if len(base_indices) == rows and len(signal_indices) == rows:
        opaque_rows = [[str(base), str(signal)] for base, signal in zip(base_indices, signal_indices)]
    elif len(base_indices) == rows:
        opaque_rows = [[str(base)] for base in base_indices]
    elif len(signal_indices) == rows:
        opaque_rows = [[str(signal)] for signal in signal_indices]
    else:
        raise RCMMCRuntimeError("P1-RCMMC physical batch sequence metadata is incomplete")
    counts, weights = _common_cell_event(labels=labels, receiver_positions=positions)
    row_order_sha = _canonical_sha256(
        [[opaque, int(label), int(slot)] for opaque, label, slot in zip(
            opaque_rows, labels.cpu().tolist(), positions.cpu().tolist()
        )]
    )
    event = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "scenario": str(scenario),
        "same_physical_clean_leo": True,
        "row_order_sha256": row_order_sha,
        "n_rc": counts,
        "effective_weights": weights,
    }
    prior = str(result.get("common_batch_sequence_sha256", "") or "") or str(
        result.get("source_labeled_indices_sha256", "") or ""
    )
    if len(prior) != 64:
        raise RCMMCRuntimeError("P1-RCMMC common batch sequence lacks source data-order SHA256")
    result["common_batch_sequence_sha256"] = hashlib.sha256(
        (prior + "\n" + json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()
    result["common_batch_sequence_batches"] = int(result.get("common_batch_sequence_batches", 0)) + 1
    result["common_batch_sequence_rows"] = int(result.get("common_batch_sequence_rows", 0)) + int(labels.numel())
    scenario_batches = {str(key): int(value) for key, value in dict(result.get("common_scenario_batches", {})).items()}
    if set(scenario_batches) != set(FROZEN_RCMMC_SCENARIOS):
        raise RCMMCRuntimeError("P1-RCMMC common scenario receipt is malformed")
    scenario_batches[str(scenario)] += 1
    result["common_scenario_batches"] = scenario_batches
    common_cells = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("rcmmc_common_cells", {})).items()
    }
    scene_cells = common_cells.get(str(scenario), _cell_template())
    expected_keys = set(_cell_template())
    if set(scene_cells) != expected_keys:
        raise RCMMCRuntimeError("P1-RCMMC common receiver/class cells are malformed")
    for key in sorted(expected_keys):
        count = int(counts[key])
        cell = dict(scene_cells[key])
        cell["rows"] = int(cell.get("rows", 0)) + count
        cell["batches"] = int(cell.get("batches", 0)) + 1
        cell["nonempty_batches"] = int(cell.get("nonempty_batches", 0)) + int(count > 0)
        scene_cells[key] = cell
    common_cells[str(scenario)] = scene_cells
    result["rcmmc_common_cells"] = common_cells
    events = list(result.get("rcmmc_common_batch_cells", []))
    events.append(event)
    result["rcmmc_common_batch_cells"] = events
    return result


def bind_rcmmc_optimizer_initial_state(
    receipt: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    result = dict(receipt)
    optimizer_type = type(optimizer).__name__
    if optimizer_type != FROZEN_RCMMC_OPTIMIZER_TYPE:
        raise RCMMCConfigurationError("P1-RCMMC requires optimizer_type=AdamW, got " + (optimizer_type or "<empty>"))
    state = optimizer.state_dict()
    if dict(state.get("state", {})):
        raise RCMMCConfigurationError("P1-RCMMC requires a new AdamW state")
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
    positive = int(info.get("positive_d", -1))
    finite = int(info.get("finite_d", -1))
    clean_zero = int(info.get("clean_zero_rows", -1))
    leo_zero = int(info.get("leo_zero_rows", -1))
    both_zero = int(info.get("both_zero_rows", -1))
    sum_d = float(info.get("sum_d", float("nan")))
    loss_value = float(info.get("loss_contribution", float("nan")))
    if (
        count < 0 or positive not in (0, 1) or finite != 1 or min(clean_zero, leo_zero, both_zero) < 0
        or both_zero > min(clean_zero, leo_zero) or not math.isfinite(sum_d) or not math.isfinite(loss_value)
    ):
        raise RCMMCRuntimeError("P1-RCMMC cumulative G cell info is malformed")
    for field, value in (("rows", count), ("positive_d_cells", positive), ("clean_zero_rows", clean_zero),
                         ("leo_zero_rows", leo_zero), ("both_zero_rows", both_zero)):
        cell[field] = int(cell.get(field, 0)) + int(value)
    cell["sum_d"] = float(cell.get("sum_d", 0.0)) + sum_d
    cell["loss_sum"] = float(cell.get("loss_sum", 0.0)) + loss_value
    cell["batches"] = int(cell.get("batches", 0)) + 1
    cell["nonempty_batches"] = int(cell.get("nonempty_batches", 0)) + int(count > 0)
    cell["finite_batches"] = int(cell.get("finite_batches", 0)) + 1
    cells[key] = cell


def update_rcmmc_receipt(
    receipt: Mapping[str, Any], batch_info: Mapping[str, Any], *, scenario: str, epoch: int, batch_index: int
) -> Dict[str, Any]:
    """Accumulate G-only scalar/count RCMMC evidence after common coverage seals."""

    result = dict(receipt)
    if str(result.get("schema", "")) != RCMMC_RECEIPT_SCHEMA or result.get("enabled") is not True:
        raise RCMMCRuntimeError("P1-RCMMC auxiliary receipt update is G-arm only")
    if str(scenario) not in FROZEN_RCMMC_SCENARIOS:
        raise RCMMCRuntimeError("P1-RCMMC scenario is outside frozen clear/low/rain cycle")
    expected_keys = set(_cell_template())
    if (
        batch_info.get("finite") is not True or batch_info.get("clean_feature_detached") is not True
        or batch_info.get("empty_cell_zero") is not True or batch_info.get("no_active_renormalization") is not True
        or batch_info.get("streamed_cell_xtx") is not True
        or batch_info.get("training_accumulation_dtype") != "float32_OUTSIDE_AMP"
        or batch_info.get("forbids_batch_d2_materialization") is not True
        or batch_info.get("forbids_batch_cell_d2_materialization") is not True
    ):
        raise RCMMCRuntimeError("P1-RCMMC batch semantic receipt drifted")
    if int(batch_info.get("global_denominator", -1)) != FROZEN_RCMMC_TERM_DIVISOR:
        raise RCMMCRuntimeError("P1-RCMMC global denominator drifted")
    cells = {str(key): dict(value) for key, value in dict(batch_info.get("cells", {})).items()}
    if set(cells) != expected_keys:
        raise RCMMCRuntimeError("P1-RCMMC G receipt lacks all receiver×class cells")
    rows = int(batch_info.get("rows", -1))
    positive = int(batch_info.get("positive_d_cells", -1))
    finite = int(batch_info.get("finite_d_cells", -1))
    sum_d = float(batch_info.get("sum_d", float("nan")))
    loss_sum = float(batch_info.get("loss_sum", float("nan")))
    if rows != FROZEN_RCMMC_BATCH_SIZE or positive < 0 or positive > FROZEN_RCMMC_CELL_COUNT or finite != FROZEN_RCMMC_CELL_COUNT or not math.isfinite(sum_d) or not math.isfinite(loss_sum):
        raise RCMMCRuntimeError("P1-RCMMC G batch rows/D/nonfinite/loss do not close")
    events = list(result.get("rcmmc_common_batch_cells", []))
    if not events:
        raise RCMMCRuntimeError("P1-RCMMC G batch lacks common C/G coverage receipt")
    common_event = dict(events[-1])
    if (
        int(common_event.get("epoch", -1)) != int(epoch)
        or int(common_event.get("batch_index", -1)) != int(batch_index)
        or str(common_event.get("scenario", "")) != str(scenario)
        or common_event.get("same_physical_clean_leo") is not True
        or len(str(common_event.get("row_order_sha256", ""))) != 64
    ):
        raise RCMMCRuntimeError("P1-RCMMC G/common same-physical receipt alignment drifted")
    common_counts = {str(key): int(value) for key, value in dict(common_event.get("n_rc", {})).items()}
    if common_counts != {key: int(value.get("n_rc", -1)) for key, value in cells.items()}:
        raise RCMMCRuntimeError("P1-RCMMC G/common n_rc receipt mismatch")
    scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(scene_cells).items()}
        for scene, scene_cells in dict(result.get("rcmmc_scenes", {})).items()
    }
    scene_cells = scenes.get(str(scenario), _cell_template())
    if set(scene_cells) != expected_keys:
        raise RCMMCRuntimeError("P1-RCMMC G scene cells are malformed")
    for key in sorted(expected_keys):
        _accumulate_aux_cell(scene_cells, key=key, info=cells[key])
    scenes[str(scenario)] = scene_cells
    result["rcmmc_scenes"] = scenes
    aux_events = list(result.get("rcmmc_g_batch_aux", []))
    aux_events.append({
        "epoch": int(epoch), "batch_index": int(batch_index), "scenario": str(scenario),
        "row_order_sha256": str(common_event["row_order_sha256"]), "positive_d_cells": positive,
        "sum_d": sum_d, "loss_sum": loss_sum,
    })
    result["rcmmc_g_batch_aux"] = aux_events
    result["rcmmc_batches"] = int(result.get("rcmmc_batches", 0)) + 1
    result["rcmmc_total_rows"] = int(result.get("rcmmc_total_rows", 0)) + rows
    result["rcmmc_positive_d_cells"] = int(result.get("rcmmc_positive_d_cells", 0)) + positive
    result["rcmmc_positive_d_batches"] = int(result.get("rcmmc_positive_d_batches", 0)) + int(positive > 0)
    result["rcmmc_sum_d"] = float(result.get("rcmmc_sum_d", 0.0)) + sum_d
    result["rcmmc_loss_sum"] = float(result.get("rcmmc_loss_sum", 0.0)) + loss_sum
    scene_positive = {str(key): int(value) for key, value in dict(result.get("rcmmc_scene_positive_batches", {})).items()}
    if set(scene_positive) != set(FROZEN_RCMMC_SCENARIOS):
        raise RCMMCRuntimeError("P1-RCMMC scene positive-D receipt is malformed")
    scene_positive[str(scenario)] += int(positive > 0)
    result["rcmmc_scene_positive_batches"] = scene_positive
    return result


def _validate_common_terminal_contract(result: Mapping[str, Any]) -> None:
    if str(result.get("schema", "")) != RCMMC_RECEIPT_SCHEMA:
        raise RCMMCRuntimeError("P1-RCMMC terminal receipt schema is invalid")
    for key in (
        "baseline_sha256", "initial_checkpoint_sha256", "class_order_binding_sha256",
        "source_labeled_indices_sha256", "source_split_manifest_sha256", "source_receiver_ids_sha256",
        "optimizer_initial_state_sha256", "common_batch_sequence_sha256",
    ):
        if len(str(result.get(key, "") or "")) != 64:
            raise RCMMCRuntimeError(f"P1-RCMMC terminal receipt lacks {key}")
    source_receiver_order_sha = str(
        result.get("source_receiver_order_sha256", "") or result.get("source_receiver_ids_sha256", "") or ""
    )
    if len(source_receiver_order_sha) != 64:
        raise RCMMCRuntimeError("P1-RCMMC terminal receipt lacks source_receiver_order_sha256")
    if int(result.get("source_receiver_count", 0)) != FROZEN_RCMMC_SOURCE_RECEIVER_COUNT:
        raise RCMMCRuntimeError("P1-RCMMC terminal source receiver count drifted")
    if int(result.get("frozen_batch_size", 0)) != FROZEN_RCMMC_BATCH_SIZE or int(result.get("frozen_feature_dim", 0)) != FROZEN_RCMMC_FEATURE_DIM:
        raise RCMMCRuntimeError("P1-RCMMC terminal B/d contract drifted")
    if str(result.get("checkpoint_role", "") or "") != "training_final_only":
        raise RCMMCRuntimeError("P1-RCMMC requires training_final_only warm start")
    if result.get("optimizer_state_restored") is not False or result.get("rng_state_restored") is not False:
        raise RCMMCRuntimeError("P1-RCMMC optimizer/RNG restoration is forbidden")
    if str(result.get("optimizer_type", "")) != FROZEN_RCMMC_OPTIMIZER_TYPE or result.get("optimizer_initial_state_empty") is not True:
        raise RCMMCRuntimeError("P1-RCMMC requires a new AdamW receipt")
    if result.get("amp_contract") != "COMMON_TRAINER_AMP_ENABLED" or result.get("common_l_base_head_input_path_verified") is not True:
        raise RCMMCRuntimeError("P1-RCMMC terminal common AMP/head path contract drifted")
    scenario_batches = {str(key): int(value) for key, value in dict(result.get("common_scenario_batches", {})).items()}
    if int(result.get("common_batch_sequence_batches", 0)) <= 0 or int(result.get("common_batch_sequence_rows", 0)) <= 0 or set(scenario_batches) != set(FROZEN_RCMMC_SCENARIOS) or any(value <= 0 for value in scenario_batches.values()):
        raise RCMMCRuntimeError("P1-RCMMC common batch/scenario receipt is incomplete")


def _validate_common_cells(result: Mapping[str, Any]) -> None:
    expected_keys = set(_cell_template())
    common_scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("rcmmc_common_cells", {})).items()
    }
    if set(common_scenes) != set(FROZEN_RCMMC_SCENARIOS):
        raise RCMMCRuntimeError("P1-RCMMC terminal common 84-cell coverage is incomplete")
    scenario_batches = {str(key): int(value) for key, value in dict(result.get("common_scenario_batches", {})).items()}
    observed_rows = 0
    for scenario in FROZEN_RCMMC_SCENARIOS:
        cells = common_scenes[scenario]
        if set(cells) != expected_keys:
            raise RCMMCRuntimeError("P1-RCMMC terminal common receiver/class cells drifted")
        for key in expected_keys:
            cell = cells[key]
            if int(cell.get("rows", 0)) <= 0 or int(cell.get("batches", 0)) != scenario_batches[scenario] or int(cell.get("nonempty_batches", 0)) <= 0:
                raise RCMMCRuntimeError("P1-RCMMC terminal common 84-cell receipt has an uncovered cell")
            observed_rows += int(cell.get("rows", 0))
    if observed_rows != int(result.get("common_batch_sequence_rows", 0)):
        raise RCMMCRuntimeError("P1-RCMMC terminal common 84-cell row total does not close")
    events = list(result.get("rcmmc_common_batch_cells", []))
    if len(events) != int(result.get("common_batch_sequence_batches", 0)):
        raise RCMMCRuntimeError("P1-RCMMC terminal common per-batch cell receipt is incomplete")
    event_rows = 0
    for event in events:
        counts = {str(key): int(value) for key, value in dict(event.get("n_rc", {})).items()}
        if event.get("same_physical_clean_leo") is not True or len(str(event.get("row_order_sha256", ""))) != 64 or set(counts) != expected_keys or any(value < 0 for value in counts.values()):
            raise RCMMCRuntimeError("P1-RCMMC terminal per-batch order/count receipt drifted")
        event_rows += sum(counts.values())
    if event_rows != int(result.get("common_batch_sequence_rows", 0)):
        raise RCMMCRuntimeError("P1-RCMMC terminal per-batch n_rc rows do not close")


def validate_rcmmc_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless common 84 cells, scalar-only receipts, and VJP close."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    _validate_common_terminal_contract(result)
    _validate_common_cells(result)
    enabled = result.get("enabled")
    if enabled is not True and enabled is not False:
        raise RCMMCRuntimeError("P1-RCMMC terminal enabled flag must be strict bool")
    if enabled is False:
        forbidden_nonzero = ("rcmmc_batches", "rcmmc_total_rows", "rcmmc_positive_d_cells", "rcmmc_positive_d_batches")
        if any(int(result.get(key, 0)) != 0 for key in forbidden_nonzero) or abs(float(result.get("rcmmc_sum_d", 0.0))) > _TOLERANCE or abs(float(result.get("rcmmc_loss_sum", 0.0))) > _TOLERANCE:
            raise RCMMCRuntimeError("P1-RCMMC C arm must retain zero auxiliary counters")
        if any(bool(result.get(key)) for key in ("rcmmc_scenes", "rcmmc_g_batch_aux", "rcmmc_gradient_audit")) or bool(result.get("rcmmc_gradient_audit_attempted", False)) or bool(result.get("rcmmc_gradient_audit_completed", False)):
            raise RCMMCRuntimeError("P1-RCMMC C arm must retain N/A-or-zero auxiliary fields")
        result["rcmmc_terminal_contract"] = "CONTROL_ARM_COMMON_SAME_PHYSICAL_ORDERED_RX_SLOT_LOCAL4_84_CELL_COVERAGE_AUX_NA_OR_ZERO"
        result["rcmmc_terminal_contract_passed"] = True
        return result
    expected_keys = set(_cell_template())
    scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("rcmmc_scenes", {})).items()
    }
    if set(scenes) != set(FROZEN_RCMMC_SCENARIOS):
        raise RCMMCRuntimeError("P1-RCMMC terminal G 84-cell coverage is incomplete")
    total_rows = total_positive = 0
    total_sum_d = total_loss = 0.0
    common_scenes = {str(scene): {str(key): dict(value) for key, value in dict(cells).items()} for scene, cells in dict(result.get("rcmmc_common_cells", {})).items()}
    for scenario in FROZEN_RCMMC_SCENARIOS:
        if int(dict(result.get("rcmmc_scene_positive_batches", {})).get(scenario, 0)) <= 0:
            raise RCMMCRuntimeError("P1-RCMMC terminal scene lacks a positive-D batch")
        cells = scenes[scenario]
        if set(cells) != expected_keys:
            raise RCMMCRuntimeError("P1-RCMMC terminal G receiver/class cells drifted")
        for key in expected_keys:
            cell = cells[key]
            common = common_scenes[scenario][key]
            rows = int(cell.get("rows", -1))
            positive = int(cell.get("positive_d_cells", -1))
            batches = int(cell.get("batches", -1))
            finite_batches = int(cell.get("finite_batches", -1))
            sum_d = float(cell.get("sum_d", float("nan")))
            loss_sum = float(cell.get("loss_sum", float("nan")))
            if rows <= 0 or rows != int(common.get("rows", -2)) or positive < 0 or positive > batches or batches != int(common.get("batches", -2)) or finite_batches != batches or not math.isfinite(sum_d) or not math.isfinite(loss_sum):
                raise RCMMCRuntimeError("P1-RCMMC terminal G r×c×scene receipt does not close")
            total_rows += rows
            total_positive += positive
            total_sum_d += sum_d
            total_loss += loss_sum
    common_rows = int(result.get("common_batch_sequence_rows", 0))
    if (
        int(result.get("rcmmc_batches", -1)) != int(result.get("common_batch_sequence_batches", -2))
        or int(result.get("rcmmc_total_rows", -1)) != common_rows
        or total_rows != common_rows
        or int(result.get("rcmmc_positive_d_cells", -1)) != total_positive
        or int(result.get("rcmmc_positive_d_batches", -1)) <= 0
        or not _float32_ledger_close(float(result.get("rcmmc_sum_d", float("nan"))), total_sum_d)
        or not _float32_ledger_close(float(result.get("rcmmc_loss_sum", float("nan"))), total_loss)
    ):
        raise RCMMCRuntimeError("P1-RCMMC terminal G batch/D/loss counters do not close")
    events = list(result.get("rcmmc_g_batch_aux", []))
    if len(events) != int(result.get("rcmmc_batches", 0)) or not bool(result.get("rcmmc_gradient_audit_completed", False)):
        raise RCMMCRuntimeError("P1-RCMMC terminal G per-batch or VJP receipt is incomplete")
    _validate_rcmmc_gradient_audit_payload(result.get("rcmmc_gradient_audit", {}))
    result["rcmmc_terminal_contract"] = "FORMAL_COMMON_SAME_PHYSICAL_ORDERED_RX_SLOT_LOCAL4_84_CELL_FIXED28_STREAMED_MOMENT_CONGRUENCE_WITH_G_ONLY_FIRST_feat_joint_SHARED_ENCODER_VJP_CLEAN_AND_EXACT_HEAD_AUX_NA"
    result["rcmmc_terminal_contract_passed"] = True
    return result


def _failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "vjp" in message or "gradient" in message or "head" in message:
        return "RCMMC_AUX_GRADIENT_PATH_FAILURE"
    if "non-finite" in message or "nonfinite" in message:
        return "RCMMC_NONFINITE"
    if "receiver" in message or "rx_i" in message or "84-cell" in message or "r×c" in message:
        return "RCMMC_SOURCE_RX_OR_CELL_COVERAGE_FAILURE"
    if "sequence" in message or "receipt" in message or "coverage" in message:
        return "RCMMC_RECEIPT_CLOSURE_FAILURE"
    if "binding" in message or "feat_joint" in message or "totalized" in message or "dimension" in message:
        return "RCMMC_BINDING_FAILURE"
    return "RCMMC_RUNTIME_FAILURE"


def write_rcmmc_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free fail-closed RCMMC failure receipt."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise RCMMCRuntimeError(f"P1-RCMMC failure receipt output directory is absent: {target_dir}")
    payload = {
        "schema": "cvs.phase1.rcmmc_failure_receipt.v1",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage or ""),
        "exception_type": type(error).__name__,
        "exception_fingerprint": _failure_fingerprint(error),
        "message": str(error),
        "receipt": dict(receipt),
    }
    target = target_dir / "phase1_rcmmc_failure_receipt.json"
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = mkstemp(prefix=".rcmmc_failure_receipt.", suffix=".tmp", dir=str(target_dir))
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


def strict_rcmmc_warm_start(
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
        raise RCMMCConfigurationError("Frozen P1-RCMMC warm-start requires model state, path, and SHA256")
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise RCMMCConfigurationError(f"Frozen P1-RCMMC strict baseline model-key mismatch: {path}: {exc}") from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RCMMCConfigurationError("Frozen P1-RCMMC strict baseline model-key mismatch: " f"missing={missing} unexpected={unexpected}")
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    if str(checkpoint_role or "") != "training_final_only":
        raise RCMMCConfigurationError("Frozen P1-RCMMC requires baseline checkpoint_role=training_final_only")
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
