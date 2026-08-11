"""Frozen P1-HNCCD core contract for Phase1 source-only DG.

The only G-arm auxiliary term is a same-batch, source-L, LEO-only
head/null cross-covariance penalty. It uses the exact classifier weight and
never creates a new view, cache, model, parameter, or cross-batch state.
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


FROZEN_HNCCD_LAMBDA = 0.02
FROZEN_HNCCD_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_HNCCD_CLASS_IDS = (0, 1, 2, 3)
FROZEN_HNCCD_BATCH_SIZE = 128
FROZEN_HNCCD_FEATURE_DIM = 160
FROZEN_HNCCD_HEAD_ROWS = 4
FROZEN_HNCCD_SOURCE_RECEIVER_COUNT = 7
FROZEN_HNCCD_CELL_COUNT = 28
FROZEN_HNCCD_TERM_DIVISOR = 28
FROZEN_HNCCD_OPTIMIZER_TYPE = "AdamW"
FROZEN_HNCCD_FLOAT32_LEDGER_REL_TOL = 32.0 * float(torch.finfo(torch.float32).eps)
FROZEN_HNCCD_AUX_TENSOR_BOUND_FORMULA = "4[8Bd+28(4+d+4d)+d4+44]"
HNCCD_RECEIPT_SCHEMA = "cvs.phase1.hnccd_receipt.v1"
_TOLERANCE = 1e-12


class HNCCDConfigurationError(ValueError):
    """Raised when the frozen P1-HNCCD C/G contract drifts."""


class HNCCDRuntimeError(RuntimeError):
    """Raised when P1-HNCCD data, gradients, or receipts cannot be proved."""


@dataclass(frozen=True)
class HNCCDConfig:
    """Immutable P1-HNCCD controls consumed by the common training loop."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise HNCCDConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
        raise HNCCDConfigurationError(
            f"Frozen P1-HNCCD requires {name}={expected:.12g}, got {actual!r}"
        )


def _require_disabled(args: Any, names: Sequence[str]) -> None:
    active = []
    for name in names:
        value = getattr(args, name, False)
        if isinstance(value, bool):
            enabled = value
        else:
            try:
                enabled = abs(float(value)) > _TOLERANCE
            except (TypeError, ValueError):
                enabled = bool(value)
        if enabled:
            active.append(str(name))
    if active:
        raise HNCCDConfigurationError(
            "Frozen P1-HNCCD forbids stacked routes: " + ", ".join(active)
        )


def _normalized_scenarios(value: Any) -> Tuple[str, ...]:
    scenarios = tuple(
        item.strip().lower().replace("-", "_")
        for item in str(value or "").split(",")
        if item.strip()
    )
    if scenarios != FROZEN_HNCCD_SCENARIOS:
        raise HNCCDConfigurationError(
            "Frozen P1-HNCCD requires --sat_train_scenarios "
            + ",".join(FROZEN_HNCCD_SCENARIOS)
        )
    return scenarios


def validate_hnccd_args(args: Any) -> HNCCDConfig:
    """Validate the frozen source-L-only C/G contract before data are loaded."""

    frozen_mode = _bool_arg(args, "phase1_hnccd_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_hnccd_enabled", False)
    loss_weight = _float_arg(args, "lambda_hnccd", 0.0)
    if not frozen_mode and not enabled:
        return HNCCDConfig(False, False, 0.0)
    if enabled and not frozen_mode:
        raise HNCCDConfigurationError(
            "--phase1_hnccd_enabled requires --phase1_hnccd_frozen_mode true"
        )
    _require_close(
        "lambda_hnccd",
        loss_weight,
        FROZEN_HNCCD_LAMBDA if enabled else 0.0,
    )
    if bool(getattr(args, "from_scratch", True)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD must train the shared feat_joint encoder")
    if not bool(getattr(args, "amp", True)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires the common AMP training path")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires --id_feature_key feat_joint")
    if int(getattr(args, "batch_size", 0)) != FROZEN_HNCCD_BATCH_SIZE:
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires --batch_size 128")
    if int(getattr(args, "epochs", 0)) != 40 or int(getattr(args, "label_epochs", 0)) != 40:
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires exactly 40 labeled epochs")
    if int(getattr(args, "pseudo_epochs", 0)) != 0:
        raise HNCCDConfigurationError("Frozen P1-HNCCD forbids pseudo epochs")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires the existing single LEO forward")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    _require_close("sat_view_prob", _float_arg(args, "sat_view_prob", 1.0), 1.0)
    if int(getattr(args, "sat_cons_start_epoch", 1)) != 1:
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires --sat_cons_start_epoch 1")
    _normalized_scenarios(getattr(args, "sat_train_scenarios", ""))
    if str(getattr(args, "sat_view_schedule", "") or "").strip():
        raise HNCCDConfigurationError("Frozen P1-HNCCD forbids --sat_view_schedule overrides")
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires non-concatenated single-LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD permits only source_known_train L updates")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD forbids RX/day-conditioned batches")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD permits no extra training views")
    if bool(getattr(args, "reject_head", False)):
        raise HNCCDConfigurationError("Frozen P1-HNCCD forbids a reject head")
    _require_disabled(
        args,
        (
            "phase1_ccpc_leo_frozen_mode", "phase1_ccpc_leo_enabled",
            "phase1_pamr_frozen_mode", "phase1_pamr_enabled",
            "phase1_cb_sfce_frozen_mode", "phase1_cb_sfce_enabled",
            "phase1_gd_proto_nll_frozen_mode", "phase1_gd_proto_nll_enabled",
            "phase1_icmt_frozen_mode", "phase1_icmt_enabled",
            "phase1_cagm_frozen_mode", "phase1_cagm_enabled",
            "phase1_rcrmd_frozen_mode", "phase1_rcrmd_enabled",
            "phase1_rcat_frozen_mode", "phase1_rcat_enabled",
            "phase1_rcmmc_frozen_mode", "phase1_rcmmc_enabled",
            "phase1_hscf_frozen_mode", "phase1_hscf_enabled",
            "phase1_recte_frozen_mode", "phase1_recte_enabled",
            "phase1_cp_sfce_frozen_mode", "phase1_cp_sfce_enabled",
            "lambda_ccpc_leo", "lambda_pamr", "lambda_cb_sfce",
            "lambda_gd_proto_nll", "lambda_icmt", "lambda_cagm",
            "lambda_rcrmd", "lambda_rcat", "lambda_rcmmc", "lambda_hscf",
            "lambda_recte", "lambda_cp_sfce", "lambda_domain", "lambda_adv",
            "lambda_orth", "lambda_cons", "lambda_group_ce", "lambda_fishr",
            "lambda_u", "lambda_ent", "lambda_u_domain", "lambda_u_adv",
            "lambda_u_sat_cons", "lambda_u_direct_metric_accept",
            "lambda_u_quarantine_accept", "lambda_zid_receiver_invariance",
            "lambda_zid_day_invariance", "lambda_zid_channel_invariance",
            "lambda_u_zid_receiver_invariance", "lambda_u_zid_day_invariance",
            "lambda_u_zid_channel_invariance", "lambda_tx_proto",
            "lambda_rx_proto", "lambda_mask_aux", "lambda_tx_supcon_masked",
            "lambda_rx_supcon_masked", "lambda_txrx_rect", "lambda_proto",
            "lambda_open_world_feat", "lambda_zid_compact",
            "lambda_proxy_unknown", "lambda_manytx_real_oe",
            "lambda_soft_unknown_mixup", "lambda_source_episode",
            "lambda_direct_metric_accept", "use_phase2_ground_prototypes",
            "use_feature_masks", "use_txrx_geometry_losses", "use_proto_memory",
            "os_gradient_surgery", "os_budget_controller",
            "os_objective_budget_controller", "phase1_v2_hard_gates",
            "manytx_real_oe_enabled", "manytx_real_oe_protocol_enabled",
            "use_ema_teacher", "teacher_ckpt", "lambda_teacher_clean_kl",
            "lambda_teacher_sat_kl", "lambda_teacher_zid_mse",
        ),
    )
    return HNCCDConfig(True, enabled, loss_weight)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def hnccd_shape_ledger(
    *,
    batch_size: int = FROZEN_HNCCD_BATCH_SIZE,
    feature_dim: int = FROZEN_HNCCD_FEATURE_DIM,
    cell_count: int = FROZEN_HNCCD_CELL_COUNT,
) -> Dict[str, Any]:
    """Return the current-batch FP32 tensor accounting contract."""

    if int(batch_size) != FROZEN_HNCCD_BATCH_SIZE:
        raise HNCCDConfigurationError("P1-HNCCD shape ledger requires B=128")
    if int(feature_dim) != FROZEN_HNCCD_FEATURE_DIM:
        raise HNCCDConfigurationError("P1-HNCCD shape ledger requires d=160")
    if int(cell_count) != FROZEN_HNCCD_CELL_COUNT:
        raise HNCCDConfigurationError("P1-HNCCD shape ledger requires 28 cells")
    b, d, cells = int(batch_size), int(feature_dim), int(cell_count)
    elements = 8 * b * d + cells * (4 + d + 4 * d) + d * 4 + 16
    return {
        "aux_tensor_bound_formula": FROZEN_HNCCD_AUX_TENSOR_BOUND_FORMULA,
        "batch_size": b,
        "feature_dim": d,
        "cell_count": cells,
        "fp32_element_bytes": 4,
        "conservative_live_tensor_upper_bound_bytes": int(4 * elements),
        "current_batch_layout": "Q[d,4],h[B,4],b[B,d],per_cell_H_B_C[4,d]",
        "uses_current_batch_only": True,
        "cross_batch_cache": False,
        "forbids_batch_d2_materialization": True,
        "forbids_batch_cell_d2_materialization": True,
        "resource_observation_only": True,
    }


def _source_receiver_tokens(values: Any) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise HNCCDConfigurationError("P1-HNCCD source_receivers must be a source-split sequence")
    tokens = []
    for value in values:
        text = str(value).strip()
        if not text:
            raise HNCCDConfigurationError("P1-HNCCD source receiver token may not be empty")
        try:
            token = int(text)
        except (TypeError, ValueError) as exc:
            raise HNCCDConfigurationError("P1-HNCCD source receiver token must be an integer") from exc
        if str(token) != text:
            raise HNCCDConfigurationError("P1-HNCCD source receiver token is not canonical")
        tokens.append(token)
    result = tuple(tokens)
    if len(result) != FROZEN_HNCCD_SOURCE_RECEIVER_COUNT or len(set(result)) != len(result):
        raise HNCCDConfigurationError("P1-HNCCD requires exactly seven ordered unique source receiver tokens")
    return result


def resolve_hnccd_source_receiver_tokens(source_split_receipt: Mapping[str, Any]) -> Tuple[int, ...]:
    source = dict(source_split_receipt or {})
    schema = str(source.get("schema", "") or "")
    if schema and schema != "cvs.phase1.source_split_receipt.v1":
        raise HNCCDConfigurationError("P1-HNCCD requires a source split receipt v1")
    return _source_receiver_tokens(source.get("source_receivers", ()))


def _receiver_key(receiver_slot: int, class_id: int) -> str:
    return f"r{int(receiver_slot)}|c{int(class_id)}"


def _receiver_positions(
    receiver_labels: torch.Tensor, receiver_tokens: Sequence[Any], *, rows: int
) -> torch.Tensor:
    if not torch.is_tensor(receiver_labels):
        raise HNCCDRuntimeError("P1-HNCCD requires source-L physical rx_i labels")
    values = receiver_labels.reshape(-1).long()
    tokens = _source_receiver_tokens(receiver_tokens)
    if values.numel() != int(rows) or values.numel() == 0:
        raise HNCCDRuntimeError("P1-HNCCD source-L rx_i rows do not align")
    positions = torch.full_like(values, -1)
    for slot, token in enumerate(tokens):
        positions = torch.where(values.eq(int(token)), torch.full_like(values, int(slot)), positions)
    if bool(positions.lt(0).any().item()):
        raise HNCCDRuntimeError("P1-HNCCD rx_i contains a receiver outside source-split R_s")
    return positions


def _normalized_tx_order(name: str, values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise HNCCDConfigurationError(f"P1-HNCCD {name} must be a TX class sequence")
    result = tuple(str(value).strip() for value in values)
    if not result or len(result) != len(set(result)) or any(not value for value in result):
        raise HNCCDConfigurationError(f"P1-HNCCD {name} must be non-empty and unique")
    return result


def resolve_hnccd_local_head_class_binding(
    *,
    local_class_order: Sequence[Any],
    source_train_tx: Sequence[Any],
    checkpoint_train_tx: Sequence[Any],
    dataset_class_order: Sequence[Any],
    local_data_class_count: Any,
    checkpoint_head_class_count: Any,
    live_head_class_count: Any,
) -> Dict[str, Any]:
    """Bind contiguous local labels to strict local4 warm-start head rows."""

    local = _normalized_tx_order("local data class order", local_class_order)
    source = _normalized_tx_order("source-train TX receipt", source_train_tx)
    checkpoint = _normalized_tx_order("checkpoint train TX receipt", checkpoint_train_tx)
    dataset = _normalized_tx_order("dataset TX class order", dataset_class_order)
    try:
        local_count = int(local_data_class_count)
        checkpoint_count = int(checkpoint_head_class_count)
        live_count = int(live_head_class_count)
    except (TypeError, ValueError) as exc:
        raise HNCCDConfigurationError("P1-HNCCD local/head counts must be integers") from exc
    if (
        local_count != FROZEN_HNCCD_HEAD_ROWS
        or len(local) != FROZEN_HNCCD_HEAD_ROWS
        or checkpoint_count != FROZEN_HNCCD_HEAD_ROWS
        or live_count != FROZEN_HNCCD_HEAD_ROWS
    ):
        raise HNCCDConfigurationError("P1-HNCCD requires exactly four aligned local head rows")
    if local != source or checkpoint != source or set(local).difference(dataset):
        raise HNCCDConfigurationError("P1-HNCCD TX class-order binding drifted")
    binding = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": list(dataset),
        "local_tx_class_order": list(local),
        "checkpoint_train_tx_class_order": list(checkpoint),
        "local_to_dataset_class_ids": [int(dataset.index(tx)) for tx in local],
        "local_to_head_class_ids": list(FROZEN_HNCCD_CLASS_IDS),
        "expected_tx_class_ids": list(FROZEN_HNCCD_CLASS_IDS),
        "dataset_class_count": len(dataset),
        "local_data_class_count": local_count,
        "checkpoint_head_class_count": checkpoint_count,
        "live_head_class_count": live_count,
    }
    binding["class_order_binding_sha256"] = _canonical_sha256(binding)
    return binding


def remap_hnccd_local_labels_to_head_rows(
    local_labels: torch.Tensor, local_to_head_class_ids: Sequence[Any]
) -> torch.Tensor:
    if not torch.is_tensor(local_labels):
        raise HNCCDRuntimeError("P1-HNCCD local TX labels must be a tensor")
    mapping = tuple(int(value) for value in local_to_head_class_ids)
    if mapping != FROZEN_HNCCD_CLASS_IDS:
        raise HNCCDRuntimeError("P1-HNCCD local-to-head mapping must be local4 identity")
    labels = local_labels.reshape(-1).long()
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise HNCCDRuntimeError("P1-HNCCD local TX labels are outside frozen class order")
    lookup = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
    return lookup.index_select(0, labels).reshape(local_labels.shape)


def resolve_hnccd_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    raw_model = getattr(model, "_orig_mod", model)
    try:
        head = raw_model.id_backbone.cls_head.head
    except AttributeError as exc:
        raise HNCCDRuntimeError("P1-HNCCD requires model.id_backbone.cls_head.head") from exc
    if not isinstance(head, torch.nn.Module):
        raise HNCCDRuntimeError("P1-HNCCD exact classifier head is not a module")
    return head


def resolve_hnccd_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    weight = getattr(resolve_hnccd_classifier_head(model), "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2 or not weight.requires_grad:
        raise HNCCDRuntimeError("P1-HNCCD classifier head weight must be a trainable rank-2 Parameter")
    return weight


def resolve_hnccd_classifier_bias(model: torch.nn.Module) -> Tuple[torch.nn.Parameter, ...]:
    bias = getattr(resolve_hnccd_classifier_head(model), "bias", None)
    if bias is None:
        return tuple()
    if not isinstance(bias, torch.nn.Parameter):
        raise HNCCDRuntimeError("P1-HNCCD classifier head bias must be a Parameter or None")
    return (bias,) if bias.requires_grad else tuple()


def _autocast_disabled(value: torch.Tensor):
    if str(value.device.type) in {"cuda", "cpu"}:
        return torch.autocast(device_type=str(value.device.type), enabled=False)
    return nullcontext()


def _validate_exact_head_weight(
    weight: torch.Tensor, *, enforce_frozen_shape: bool, with_autograd: bool
) -> None:
    if not torch.is_tensor(weight) or weight.ndim != 2 or int(weight.size(0)) != 4:
        raise HNCCDRuntimeError("P1-HNCCD exact head weight must have W[4,d]")
    if bool(enforce_frozen_shape) and int(weight.size(1)) != FROZEN_HNCCD_FEATURE_DIM:
        raise HNCCDRuntimeError("P1-HNCCD exact head weight must have d=160")
    values = weight.float() if with_autograd else weight.detach().float()
    if not bool(torch.isfinite(values.detach()).all().item()):
        raise HNCCDRuntimeError("P1-HNCCD exact head weight is non-finite")
    gram = values.matmul(values.transpose(0, 1))
    if not bool(torch.isfinite(gram.detach()).all().item()):
        raise HNCCDRuntimeError("P1-HNCCD exact head Gram matrix is non-finite")
    try:
        factor = torch.linalg.cholesky(gram)
    except RuntimeError as exc:
        raise HNCCDRuntimeError(
            "P1-HNCCD exact head W must be full-row-rank for Cholesky"
        ) from exc
    if not bool(torch.isfinite(factor.detach()).all().item()):
        raise HNCCDRuntimeError("P1-HNCCD exact head Cholesky factor is non-finite")


def hnccd_head_null_basis(head_weight: torch.Tensor) -> torch.Tensor:
    """Return live FP32 Q equal to W transpose times inverse L transpose."""

    with _autocast_disabled(head_weight):
        _validate_exact_head_weight(
            head_weight, enforce_frozen_shape=False, with_autograd=True
        )
        values = head_weight.float()
        factor = torch.linalg.cholesky(values.matmul(values.transpose(0, 1)))
        try:
            left_solve = torch.linalg.solve_triangular(factor, values, upper=False)
        except RuntimeError as exc:
            raise HNCCDRuntimeError("P1-HNCCD exact head triangular solve failed") from exc
        q = left_solve.transpose(0, 1)
        if q.shape != (values.size(1), FROZEN_HNCCD_HEAD_ROWS):
            raise HNCCDRuntimeError("P1-HNCCD Q shape drifted from [d,4]")
        if not bool(torch.isfinite(q.detach()).all().item()):
            raise HNCCDRuntimeError("P1-HNCCD Q is non-finite")
        return q


def _totalized_l2_with_zeros(
    features: torch.Tensor, *, view_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(features) or features.ndim != 2 or features.size(0) <= 0:
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} feat_joint must be non-empty rank-2")
    if not bool(torch.isfinite(features.detach()).all().item()):
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} feat_joint is non-finite")
    values = features.float()
    norms = torch.linalg.vector_norm(values, ord=2, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms.detach()).all().item()):
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} feat_joint norm is non-finite")
    nonzero = norms.gt(0.0)
    totalized = torch.zeros_like(values) + values * 0.0
    if bool(nonzero.any().item()):
        mask = nonzero.reshape(-1)
        totalized[mask] = values[mask] / norms[mask]
    if not bool(torch.isfinite(totalized.detach()).all().item()):
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} totalized-L2 output is non-finite")
    return totalized, ~nonzero.reshape(-1)


def totalized_l2(features: torch.Tensor) -> torch.Tensor:
    """Return safe FP32 totalized-L2 values with exact zero rows preserved."""

    with _autocast_disabled(features):
        return _totalized_l2_with_zeros(features, view_name="input")[0]


def _cell_template() -> Dict[str, Dict[str, Any]]:
    return {
        _receiver_key(receiver_slot, class_id): {
            "rows": 0,
            "batches": 0,
            "nonempty_batches": 0,
            "finite_batches": 0,
            "positive_c_cells": 0,
            "insufficient_cells": 0,
            "leo_zero_rows": 0,
            "sum_c": 0.0,
            "loss_sum": 0.0,
        }
        for receiver_slot in range(FROZEN_HNCCD_SOURCE_RECEIVER_COUNT)
        for class_id in FROZEN_HNCCD_CLASS_IDS
    }


def hnccd_loss(
    leo_feat_joint: torch.Tensor,
    head_weight: torch.Tensor,
    source_tx_labels: torch.Tensor,
    source_rx_labels: torch.Tensor,
    source_receiver_tokens: Sequence[Any],
    *,
    require_frozen_shape: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute the fixed-28 LEO head/null cross-covariance loss."""

    labels = source_tx_labels.reshape(-1).long()
    if labels.numel() <= 0:
        raise HNCCDRuntimeError("P1-HNCCD source-L rows must be non-empty")
    if bool(require_frozen_shape) and labels.numel() != FROZEN_HNCCD_BATCH_SIZE:
        raise HNCCDRuntimeError("P1-HNCCD requires exactly B=128 source-L rows")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise HNCCDRuntimeError("P1-HNCCD source labels are outside local4")
    positions = _receiver_positions(
        source_rx_labels, source_receiver_tokens, rows=int(labels.numel())
    )
    if leo_feat_joint.size(0) != labels.numel():
        raise HNCCDRuntimeError("P1-HNCCD LEO feat_joint rows do not align with source labels")
    if bool(require_frozen_shape) and int(leo_feat_joint.size(1)) != FROZEN_HNCCD_FEATURE_DIM:
        raise HNCCDRuntimeError("P1-HNCCD requires feat_joint d=160")
    with _autocast_disabled(leo_feat_joint):
        u, leo_zero = _totalized_l2_with_zeros(leo_feat_joint, view_name="leo")
        q = hnccd_head_null_basis(head_weight)
        if int(q.size(0)) != int(u.size(1)):
            raise HNCCDRuntimeError("P1-HNCCD Q/feat_joint dimension binding drifted")
        h = u.matmul(q)
        b = u - h.matmul(q.transpose(0, 1))
        if not bool(torch.isfinite(h.detach()).all().item()) or not bool(
            torch.isfinite(b.detach()).all().item()
        ):
            raise HNCCDRuntimeError("P1-HNCCD head/null decomposition is non-finite")
        zero_scalar = h.sum() * 0.0 + b.sum() * 0.0 + q.sum() * 0.0
        cells: Dict[str, Dict[str, Any]] = {}
        terms = []
        total_rows = total_positive = total_finite = total_insufficient = 0
        leo_zero_total = 0
        sum_c = 0.0
        for receiver_slot in range(FROZEN_HNCCD_SOURCE_RECEIVER_COUNT):
            for class_id in FROZEN_HNCCD_CLASS_IDS:
                key = _receiver_key(receiver_slot, class_id)
                mask = positions.eq(receiver_slot) & labels.eq(class_id)
                count = int(mask.sum().item())
                zero_rows = int(leo_zero[mask].sum().item())
                insufficient = int(count < 2)
                if insufficient:
                    c_norm = zero_scalar
                    positive = 0
                else:
                    h_group = h[mask]
                    b_group = b[mask]
                    h_centered = h_group - h_group.mean(dim=0, keepdim=True)
                    b_centered = b_group - b_group.mean(dim=0, keepdim=True)
                    cross = h_centered.transpose(0, 1).matmul(b_centered) / float(count)
                    if cross.shape != (FROZEN_HNCCD_HEAD_ROWS, int(u.size(1))):
                        raise HNCCDRuntimeError("P1-HNCCD cross covariance shape drifted")
                    c_norm = cross.square().sum()
                    positive = int(c_norm.detach().gt(0.0).item())
                if not bool(torch.isfinite(c_norm.detach()).item()):
                    raise HNCCDRuntimeError("P1-HNCCD receiver/class cross covariance is non-finite")
                c_value = float(c_norm.detach().item())
                cells[key] = {
                    "n_rc": count,
                    "occupied": bool(count > 0),
                    "insufficient_n_lt_2": bool(insufficient),
                    "positive_c": positive,
                    "finite_c": 1,
                    "leo_zero_rows": zero_rows,
                    "sum_c": c_value,
                    "loss_contribution": c_value / float(FROZEN_HNCCD_TERM_DIVISOR),
                }
                terms.append(c_norm)
                total_rows += count
                total_positive += positive
                total_finite += 1
                total_insufficient += insufficient
                leo_zero_total += zero_rows
                sum_c += c_value
        if total_rows != int(labels.numel()) or total_finite != FROZEN_HNCCD_CELL_COUNT:
            raise HNCCDRuntimeError("P1-HNCCD batch receiver/class coverage does not close")
        loss = torch.stack(terms).sum() / float(FROZEN_HNCCD_TERM_DIVISOR)
        if not bool(torch.isfinite(loss.detach()).item()):
            raise HNCCDRuntimeError("P1-HNCCD loss is non-finite")
    return loss, {
        "rows": int(labels.numel()),
        "positive_c_cells": total_positive,
        "finite_c_cells": total_finite,
        "insufficient_cells": total_insufficient,
        "leo_zero_rows": leo_zero_total,
        "sum_c": float(sum_c),
        "loss_sum": float(loss.detach().item()),
        "global_denominator": FROZEN_HNCCD_TERM_DIVISOR,
        "fixed_scale": 1.0 / float(FROZEN_HNCCD_TERM_DIVISOR),
        "cells": cells,
        "finite": True,
        "totalized_l2_rule": "MASK_NORM_GT_0_THEN_DIVIDE_ELSE_ZERO",
        "training_accumulation_dtype": "float32_OUTSIDE_AMP",
        "n_lt_2_differentiable_zero": True,
        "no_active_renormalization": True,
        "streamed_cell_cross_covariance": True,
        "head_null_basis": "FP32_CHOLESKY_TRIANGULAR_SOLVE_Q_d_by_4",
        "forbids_detach_on_leo_or_head": True,
        "forbids_batch_d2_materialization": True,
        "forbids_batch_cell_d2_materialization": True,
        "shape_ledger": hnccd_shape_ledger(),
    }


def add_hnccd_to_loss(
    base_loss: torch.Tensor, hnccd: Optional[torch.Tensor], config: Optional[HNCCDConfig]
) -> torch.Tensor:
    """Add the sole G-arm term and preserve the common C base tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if hnccd is None:
        raise HNCCDRuntimeError("Enabled P1-HNCCD requires its auxiliary loss")
    return base_loss + float(config.loss_weight) * hnccd


def hnccd_shared_encoder_and_head_parameters(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Parameter, ...]]:
    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise HNCCDRuntimeError("P1-HNCCD requires model.id_backbone for VJP audit")
    weight = resolve_hnccd_classifier_weight(raw_model)
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
        raise HNCCDRuntimeError("P1-HNCCD shared encoder audit scope is empty")
    return {
        "shared_encoder": encoder,
        "head_weight": (weight,),
        "head_bias": resolve_hnccd_classifier_bias(raw_model),
    }


def _finite_nonzero_vjp(
    loss: torch.Tensor, parameters: Iterable[torch.Tensor], *, group_name: str
) -> Dict[str, float]:
    params = tuple(parameters)
    if not params:
        raise HNCCDRuntimeError(f"P1-HNCCD {group_name} VJP scope is empty")
    gradients = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    squared_norm = 0.0
    for gradient in gradients:
        if gradient is None:
            raise HNCCDRuntimeError(f"P1-HNCCD {group_name} VJP is None or detached")
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise HNCCDRuntimeError(f"P1-HNCCD {group_name} VJP is non-finite")
        value = gradient.detach().double()
        squared_norm += float(torch.sum(value * value).item())
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm) or norm <= 0.0:
        raise HNCCDRuntimeError(f"P1-HNCCD {group_name} VJP norm is zero or non-finite")
    return {"parameter_count": float(len(params)), "norm": float(norm)}


def _none_or_zero_vjp(
    loss: torch.Tensor,
    parameters: Iterable[torch.Tensor],
    *,
    group_name: str,
    allow_absent: bool = False,
) -> Dict[str, Any]:
    params = tuple(parameters)
    if not params:
        if allow_absent:
            return {
                "parameter_count": 0.0,
                "none_parameters": 0.0,
                "zero_parameters": 0.0,
                "nonzero_parameters": 0.0,
                "none_or_zero_expected": True,
                "parameter_absent": True,
            }
        raise HNCCDRuntimeError(f"P1-HNCCD {group_name} VJP scope is empty")
    gradients = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    none_count = zero_count = 0
    for gradient in gradients:
        if gradient is None:
            none_count += 1
            continue
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise HNCCDRuntimeError(f"P1-HNCCD {group_name} auxiliary VJP is non-finite")
        if int(torch.count_nonzero(gradient.detach()).item()) != 0:
            raise HNCCDRuntimeError(f"P1-HNCCD {group_name} must have no auxiliary gradient")
        zero_count += 1
    return {
        "parameter_count": float(len(params)),
        "none_parameters": float(none_count),
        "zero_parameters": float(zero_count),
        "nonzero_parameters": 0.0,
        "none_or_zero_expected": True,
        "parameter_absent": False,
    }


def hnccd_aux_gradient_audit(
    hnccd: torch.Tensor,
    clean_feat_joint: torch.Tensor,
    feat_joint_leo: torch.Tensor,
    parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
) -> Dict[str, Any]:
    """Audit one raw, unscaled HNCCD VJP without changing AMP or optimizer."""

    if not torch.is_tensor(hnccd) or hnccd.ndim != 0:
        raise HNCCDRuntimeError("P1-HNCCD VJP audit requires a scalar auxiliary loss")
    if not torch.is_tensor(clean_feat_joint) or clean_feat_joint.ndim != 2:
        raise HNCCDRuntimeError("P1-HNCCD VJP audit requires clean feat_joint")
    if not torch.is_tensor(feat_joint_leo) or feat_joint_leo.ndim != 2:
        raise HNCCDRuntimeError("P1-HNCCD VJP audit requires LEO feat_joint")
    if tuple(parameter_groups.keys()) != ("shared_encoder", "head_weight", "head_bias"):
        raise HNCCDRuntimeError("P1-HNCCD VJP audit parameter groups drifted")
    return {
        "feat_joint_leo": _finite_nonzero_vjp(
            hnccd, (feat_joint_leo,), group_name="feat_joint_leo"
        ),
        "shared_encoder": _finite_nonzero_vjp(
            hnccd, parameter_groups["shared_encoder"], group_name="shared_encoder"
        ),
        "head_weight": _finite_nonzero_vjp(
            hnccd, parameter_groups["head_weight"], group_name="exact head weight"
        ),
        "head_bias": _none_or_zero_vjp(
            hnccd, parameter_groups["head_bias"], group_name="head bias", allow_absent=True
        ),
        "clean_feat_joint": _none_or_zero_vjp(
            hnccd, (clean_feat_joint,), group_name="clean feat_joint"
        ),
        "raw_unscaled": True,
        "diagnostic_only": True,
        "touches_amp_optimizer_rng": False,
        "clean_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        "head_bias_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
        "exact_head_weight_source": "model.id_backbone.cls_head.head.weight",
    }


def hnccd_config_receipt(config: HNCCDConfig) -> Dict[str, Any]:
    """Create the scalar-only receipt skeleton for either frozen C/G arm."""

    return {
        "schema": HNCCD_RECEIPT_SCHEMA,
        "method": "P1_HNCCD",
        "candidate_pattern": "F{1..6}{C|G}_HNCCD12",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "loss_rule": "SOURCE_L_ORDERED_RECEIVER_SLOT_BY_LOCAL4_LEO_HEAD_NULL_CROSS_COVARIANCE_DECORRELATION_TOTALIZED_L2_feat_joint",
        "loss_formula": "Q=W^T chol(WW^T)^(-T);h=Q^T u;b=u-Qh;C_rc=(H-Hbar)^T(B-Bbar)/n_rc;if_n_lt_2_C=0;L=sum_rc||C_rc||F^2/28",
        "loss_global_denominator": FROZEN_HNCCD_TERM_DIVISOR,
        "fixed_batch_size": FROZEN_HNCCD_BATCH_SIZE,
        "fixed_local_class_count": FROZEN_HNCCD_HEAD_ROWS,
        "local_class_count": FROZEN_HNCCD_HEAD_ROWS,
        "local_class_ids": list(FROZEN_HNCCD_CLASS_IDS),
        "frozen_batch_size": FROZEN_HNCCD_BATCH_SIZE,
        "frozen_feature_dim": FROZEN_HNCCD_FEATURE_DIM,
        "frozen_source_receiver_count": FROZEN_HNCCD_SOURCE_RECEIVER_COUNT,
        "exact_head_weight_path": "model.id_backbone.cls_head.head.weight",
        "exact_head_weight_shape": [4, FROZEN_HNCCD_FEATURE_DIM],
        "head_null_basis_rule": "FP32_DIFFERENTIABLE_CHOLESKY_WWT_AND_TRIANGULAR_SOLVE_Q_EQ_WT_LINVTRANSPOSE_NO_PINV_EPSILON_OR_FALLBACK",
        "head_full_row_rank_required": True,
        "z_id_key": "feat_joint",
        "training_accumulation_dtype": "float32_OUTSIDE_AMP",
        "clean_feature_detached": "NOT_READ_BY_HNCCD_AUXILIARY",
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD",
        "common_batch_size": FROZEN_HNCCD_BATCH_SIZE,
        "common_loader_drop_last": True,
        "common_order_contract": "C_G_IDENTICAL_SEED_SAMPLER_PHYSICAL_IDS_AND_CLEAR_LOW_RAIN_SEQUENCE",
        "receipt_payload": "SCALARS_COUNTS_AND_SHA_ONLY_NO_IQ_FEATURE_COVARIANCE_OR_RECEIVER_TOKEN",
        "shape_ledger": hnccd_shape_ledger(),
        "common_lambda_sat_cons": 0.10,
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "common_l_base_head_input_path_verified": False,
        "aux_gradient_scope": "LEO_feat_joint_SHARED_ENCODER_EXACT_HEAD_WEIGHT_FINITE_NONZERO;CLEAN_AND_HEAD_BIAS_NONE_OR_ZERO",
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
        "u_loader_common_trainer_boundary": "MAY_BE_CONSTRUCTED_BY_COMMON_TRAINER_BUT_HNCCD_ZERO_ITERATE_ZERO_FORWARD_ZERO_LOSS_ZERO_BACKWARD_ZERO_OPTIMIZER",
        "v_common_trainer_boundary": "COMMON_READ_ONLY_DIAGNOSTIC_ONLY_HNCCD_ZERO_LOSS_ZERO_BACKWARD_ZERO_OPTIMIZER_ZERO_CALIBRATION_ZERO_MODEL_SELECTION_FEEDBACK",
        "uses_ema_or_state": False,
        "uses_threshold": False,
        "uses_cross_sample_pairing": False,
        "uses_cross_receiver_pairing": False,
        "resource_selection_feedback": False,
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
        "common_scenario_batches": {scene: 0 for scene in FROZEN_HNCCD_SCENARIOS},
        "hnccd_common_cells": {},
        "hnccd_common_batch_cells": [],
        "hnccd_batches": 0,
        "hnccd_total_rows": 0,
        "hnccd_positive_c_cells": 0,
        "hnccd_positive_c_batches": 0,
        "hnccd_insufficient_cells": 0,
        "hnccd_sum_c": 0.0,
        "hnccd_loss_sum": 0.0,
        "hnccd_scene_positive_batches": {scene: 0 for scene in FROZEN_HNCCD_SCENARIOS},
        "hnccd_scenes": {},
        "hnccd_g_batch_aux": [],
        "hnccd_gradient_audit_attempted": False,
        "hnccd_gradient_audit_completed": False,
        "hnccd_gradient_audit_scenes": {},
        "hnccd_amp_overflow_raw_finite_batches": 0,
        "hnccd_amp_overflow_raw_nonfinite_batches": 0,
        "hnccd_amp_overflow_material_nonfinite_batches": 0,
        "hnccd_amp_skip_scale_decreased_batches": 0,
        "hnccd_amp_skip_optimizer_state_unchanged_batches": 0,
        "hnccd_effective_optimizer_steps": 0,
        "hnccd_optimizer_step_attempts": 0,
        "hnccd_consecutive_amp_overflow_skips": 0,
        "hnccd_max_consecutive_amp_overflow_skips": 0,
        "hnccd_persistent_amp_overflow": False,
        "hnccd_last_amp_overflow": {},
        "hnccd_last_material_nonfinite": {},
        "hnccd_last_optimizer_step": {},
        "hnccd_optimizer_events": [],
        "hnccd_resource_observations": [],
        "hnccd_terminal_contract": "PENDING",
        "hnccd_terminal_contract_passed": False,
        "proxy_rows": 0,
        "held_rows": 0,
    }


def bind_hnccd_source_data_order(
    receipt: Mapping[str, Any], source_split_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind source-L order and only the SHA of the sealed receiver order."""

    result = dict(receipt)
    source = dict(source_split_receipt or {})
    labeled_sha = str(source.get("labeled_indices_sha256", "") or "")
    manifest_sha = str(source.get("split_manifest_sha256", "") or "")
    if len(labeled_sha) != 64 or len(manifest_sha) != 64:
        raise HNCCDConfigurationError("P1-HNCCD requires labeled-index and source-split SHA256 receipts")
    tokens = resolve_hnccd_source_receiver_tokens(source)
    token_sha = _canonical_sha256([int(value) for value in tokens])
    result["source_labeled_indices_sha256"] = labeled_sha
    result["source_split_manifest_sha256"] = manifest_sha
    result["source_receiver_count"] = len(tokens)
    result["source_receiver_order_sha256"] = token_sha
    result["source_receiver_ids_sha256"] = token_sha
    result["source_receiver_provenance"] = "SOURCE_SPLIT_RECEIPT_ORDERED_SOURCE_RECEIVERS_PHYSICAL_ID_BOUND_L_ONLY"
    return result


def _validate_view_binding(
    *, view_name: str, output: Mapping[str, Any], labels: torch.Tensor, head_weight: torch.Tensor
) -> torch.Tensor:
    if str(output.get("z_id_key", "")) != "feat_joint":
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} z_id_key must be feat_joint")
    z_id, logits = output.get("z_id"), output.get("tx_logits")
    if not torch.is_tensor(z_id) or z_id.ndim != 2:
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} z_id must be rank-2 feat_joint")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} tx_logits must be rank-2 raw logits")
    if z_id.size(0) != labels.numel() or logits.size(0) != labels.numel():
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} rows must align with source L labels")
    if int(head_weight.size(0)) != 4 or int(logits.size(1)) != 4:
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} head/logit class rows must be local4")
    if int(head_weight.size(1)) != int(z_id.size(1)):
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} feat_joint/head dimension binding drifted")
    if not bool(z_id.requires_grad) or not bool(logits.requires_grad):
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} requires a live feat_joint/head path")
    if not bool(torch.isfinite(z_id.detach()).all().item()):
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} feat_joint is non-finite")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise HNCCDRuntimeError(f"P1-HNCCD {view_name} raw logits are non-finite")
    return z_id


def validate_hnccd_binding(
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
    """Fail closed unless the common forwards expose the live exact head path."""

    if not isinstance(out_clean, Mapping) or not isinstance(out_leo, Mapping):
        raise HNCCDRuntimeError("P1-HNCCD requires clean and LEO mapping outputs")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() <= 0:
        raise HNCCDRuntimeError("P1-HNCCD source L labels must be non-empty")
    if bool(enforce_frozen_shape) and labels.numel() != FROZEN_HNCCD_BATCH_SIZE:
        raise HNCCDRuntimeError("P1-HNCCD requires exactly B=128 source-L rows")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise HNCCDRuntimeError("P1-HNCCD source labels must bind to local4 head rows")
    if tuple(int(value) for value in expected_class_ids) != FROZEN_HNCCD_CLASS_IDS:
        raise HNCCDRuntimeError("P1-HNCCD expected local4 class order is invalid")
    tokens = source_receiver_tokens if source_receiver_tokens is not None else expected_receiver_ids
    if tokens is None:
        raise HNCCDRuntimeError("P1-HNCCD requires source-split receiver tokens")
    _receiver_positions(source_rx_labels, tokens, rows=int(labels.numel()))
    weight = resolve_hnccd_classifier_weight(model)
    with _autocast_disabled(weight):
        _validate_exact_head_weight(
            weight, enforce_frozen_shape=bool(enforce_frozen_shape), with_autograd=False
        )
    clean_z = _validate_view_binding(
        view_name="clean", output=out_clean, labels=labels, head_weight=weight
    )
    leo_z = _validate_view_binding(
        view_name="leo", output=out_leo, labels=labels, head_weight=weight
    )
    if bool(enforce_frozen_shape) and (
        int(clean_z.size(1)) != FROZEN_HNCCD_FEATURE_DIM
        or int(leo_z.size(1)) != FROZEN_HNCCD_FEATURE_DIM
    ):
        raise HNCCDRuntimeError("P1-HNCCD requires feat_joint d=160")
    return weight


def _as_plain_list(values: Any) -> list[Any]:
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return []


def _common_cell_counts(
    labels: torch.Tensor, receiver_positions: torch.Tensor
) -> Dict[str, int]:
    counts = {
        _receiver_key(receiver_slot, class_id): int(
            (receiver_positions.eq(receiver_slot) & labels.eq(class_id)).sum().item()
        )
        for receiver_slot in range(FROZEN_HNCCD_SOURCE_RECEIVER_COUNT)
        for class_id in FROZEN_HNCCD_CLASS_IDS
    }
    if sum(counts.values()) != int(labels.numel()):
        raise HNCCDRuntimeError("P1-HNCCD common n_rc counters do not close")
    return counts


def update_hnccd_common_batch_sequence_receipt(
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
    """Chain only same-physical/order SHA and scalar cell counts for C and G."""

    result = dict(receipt)
    expected_scene = FROZEN_HNCCD_SCENARIOS[(int(epoch) + int(batch_index) - 2) % 3]
    if str(scenario) != expected_scene:
        raise HNCCDRuntimeError("P1-HNCCD common LEO scenario sequence drifted")
    labels = source_tx_labels.detach().reshape(-1).long()
    if labels.numel() != FROZEN_HNCCD_BATCH_SIZE:
        raise HNCCDRuntimeError("P1-HNCCD common sequence requires B=128 source L labels")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 4:
        raise HNCCDRuntimeError("P1-HNCCD common sequence requires local4 source L labels")
    if source_receiver_tokens is None:
        raise HNCCDRuntimeError("P1-HNCCD common sequence requires runtime source receiver tokens")
    positions = _receiver_positions(
        source_rx_labels, source_receiver_tokens, rows=int(labels.numel())
    ).detach()
    if metadata is None:
        raise HNCCDRuntimeError("P1-HNCCD common sequence requires opaque physical metadata")
    base_indices = _as_plain_list(metadata.get("base_index"))
    signal_indices = _as_plain_list(metadata.get("sig_i"))
    if len(base_indices) == labels.numel() and len(signal_indices) == labels.numel():
        opaque_rows = [[str(a), str(b)] for a, b in zip(base_indices, signal_indices)]
    elif len(base_indices) == labels.numel():
        opaque_rows = [[str(a)] for a in base_indices]
    elif len(signal_indices) == labels.numel():
        opaque_rows = [[str(a)] for a in signal_indices]
    else:
        raise HNCCDRuntimeError("P1-HNCCD physical batch sequence metadata is incomplete")
    counts = _common_cell_counts(labels, positions)
    row_order_sha = _canonical_sha256(
        [
            [opaque, int(label), int(slot)]
            for opaque, label, slot in zip(
                opaque_rows, labels.cpu().tolist(), positions.cpu().tolist()
            )
        ]
    )
    event = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "scenario": str(scenario),
        "same_physical_clean_leo": True,
        "row_order_sha256": row_order_sha,
        "n_rc": counts,
        "fixed_denominator": FROZEN_HNCCD_TERM_DIVISOR,
    }
    prior = str(result.get("common_batch_sequence_sha256", "") or "") or str(
        result.get("source_labeled_indices_sha256", "") or ""
    )
    if len(prior) != 64:
        raise HNCCDRuntimeError("P1-HNCCD common sequence lacks source data-order SHA256")
    result["common_batch_sequence_sha256"] = hashlib.sha256(
        (prior + "\n" + json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()
    result["common_batch_sequence_batches"] = int(result.get("common_batch_sequence_batches", 0)) + 1
    result["common_batch_sequence_rows"] = int(result.get("common_batch_sequence_rows", 0)) + int(labels.numel())
    scene_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if set(scene_batches) != set(FROZEN_HNCCD_SCENARIOS):
        raise HNCCDRuntimeError("P1-HNCCD common scenario receipt is malformed")
    scene_batches[str(scenario)] += 1
    result["common_scenario_batches"] = scene_batches
    all_scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("hnccd_common_cells", {})).items()
    }
    scene_cells = all_scenes.get(str(scenario), _cell_template())
    if set(scene_cells) != set(_cell_template()):
        raise HNCCDRuntimeError("P1-HNCCD common receiver/class cells are malformed")
    for key, count in counts.items():
        cell = dict(scene_cells[key])
        cell["rows"] = int(cell["rows"]) + int(count)
        cell["batches"] = int(cell["batches"]) + 1
        cell["nonempty_batches"] = int(cell["nonempty_batches"]) + int(count > 0)
        scene_cells[key] = cell
    all_scenes[str(scenario)] = scene_cells
    result["hnccd_common_cells"] = all_scenes
    events = list(result.get("hnccd_common_batch_cells", []))
    events.append(event)
    result["hnccd_common_batch_cells"] = events
    return result


def bind_hnccd_optimizer_initial_state(
    receipt: Mapping[str, Any], optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Seal a new AdamW state before the first backward."""

    result = dict(receipt)
    optimizer_type = type(optimizer).__name__
    if optimizer_type != FROZEN_HNCCD_OPTIMIZER_TYPE:
        raise HNCCDConfigurationError(
            "P1-HNCCD requires optimizer_type=AdamW, got " + (optimizer_type or "<empty>")
        )
    state = optimizer.state_dict()
    if dict(state.get("state", {})):
        raise HNCCDConfigurationError("P1-HNCCD requires a new AdamW state")
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


def strict_hnccd_warm_start(
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
        raise HNCCDConfigurationError("Frozen P1-HNCCD warm-start requires model state, path, and SHA256")
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise HNCCDConfigurationError(
            f"Frozen P1-HNCCD strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise HNCCDConfigurationError(
            "Frozen P1-HNCCD strict baseline model-key mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    try:
        epoch = int(checkpoint_epoch)
    except (TypeError, ValueError):
        epoch = -1
    if str(checkpoint_role or "") != "training_final_only":
        raise HNCCDConfigurationError("Frozen P1-HNCCD requires baseline checkpoint_role=training_final_only")
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


def _float32_ledger_close(actual: float, expected: float) -> bool:
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    return abs(actual - expected) <= FROZEN_HNCCD_FLOAT32_LEDGER_REL_TOL * max(
        1.0, abs(actual), abs(expected)
    )


def _validate_nonzero_audit(values: Any, *, group_name: str) -> None:
    if not isinstance(values, Mapping):
        raise HNCCDRuntimeError(f"P1-HNCCD VJP audit lacks {group_name}")
    count = float(values.get("parameter_count", 0.0))
    norm = float(values.get("norm", float("nan")))
    if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
        raise HNCCDRuntimeError(f"P1-HNCCD {group_name} VJP is zero or non-finite")


def _validate_none_or_zero_audit(
    values: Any, *, group_name: str, allow_absent: bool
) -> None:
    if not isinstance(values, Mapping):
        raise HNCCDRuntimeError(f"P1-HNCCD VJP audit lacks {group_name}")
    count = float(values.get("parameter_count", float("nan")))
    none_count = float(values.get("none_parameters", float("nan")))
    zero_count = float(values.get("zero_parameters", float("nan")))
    nonzero_count = float(values.get("nonzero_parameters", float("nan")))
    absent = values.get("parameter_absent") is True
    if (
        not all(
            math.isfinite(value) and value >= 0.0
            for value in (count, none_count, zero_count, nonzero_count)
        )
        or none_count + zero_count != count
        or nonzero_count != 0.0
        or values.get("none_or_zero_expected") is not True
        or (count == 0.0 and not (allow_absent and absent))
        or (count > 0.0 and absent)
    ):
        raise HNCCDRuntimeError(
            f"P1-HNCCD {group_name} None-or-zero VJP contract failed"
        )


def _validate_hnccd_gradient_audit_payload(audit: Mapping[str, Any]) -> None:
    if not isinstance(audit, Mapping):
        raise HNCCDRuntimeError("P1-HNCCD VJP audit receipt is malformed")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
        or audit.get("clean_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        or audit.get("head_bias_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        or audit.get("exact_head_weight_source")
        != "model.id_backbone.cls_head.head.weight"
    ):
        raise HNCCDRuntimeError("P1-HNCCD VJP audit semantics drifted")
    for group_name in ("feat_joint_leo", "shared_encoder", "head_weight"):
        _validate_nonzero_audit(audit.get(group_name), group_name=group_name)
    _validate_none_or_zero_audit(
        audit.get("clean_feat_joint"),
        group_name="clean_feat_joint",
        allow_absent=False,
    )
    _validate_none_or_zero_audit(
        audit.get("head_bias"), group_name="head_bias", allow_absent=True
    )


def update_hnccd_gradient_audit_receipt(
    receipt: Mapping[str, Any], audit: Mapping[str, Any], *, scenario: str
) -> Dict[str, Any]:
    """Seal exactly one raw unscaled positive VJP for each LEO scene."""

    result = dict(receipt)
    if str(scenario) not in FROZEN_HNCCD_SCENARIOS:
        raise HNCCDRuntimeError("P1-HNCCD VJP scenario is outside clear/low/rain")
    audits = {
        str(key): dict(value)
        for key, value in dict(result.get("hnccd_gradient_audit_scenes", {})).items()
    }
    if str(scenario) in audits:
        raise HNCCDRuntimeError("P1-HNCCD VJP audit may run only once per scene")
    _validate_hnccd_gradient_audit_payload(audit)
    audits[str(scenario)] = dict(audit)
    result["hnccd_gradient_audit_attempted"] = True
    result["hnccd_gradient_audit_completed"] = set(audits) == set(
        FROZEN_HNCCD_SCENARIOS
    )
    result["hnccd_gradient_audit_scenes"] = audits
    return result


def _accumulate_aux_cell(
    cells: Dict[str, Dict[str, Any]], *, key: str, info: Mapping[str, Any]
) -> None:
    cell = dict(cells[key])
    count = int(info.get("n_rc", -1))
    positive = int(info.get("positive_c", -1))
    finite = int(info.get("finite_c", -1))
    insufficient = bool(info.get("insufficient_n_lt_2", False))
    zero_rows = int(info.get("leo_zero_rows", -1))
    sum_c = float(info.get("sum_c", float("nan")))
    loss_value = float(info.get("loss_contribution", float("nan")))
    if (
        count < 0
        or positive not in (0, 1)
        or finite != 1
        or zero_rows < 0
        or insufficient != (count < 2)
        or not math.isfinite(sum_c)
        or not math.isfinite(loss_value)
    ):
        raise HNCCDRuntimeError("P1-HNCCD cumulative G cell info is malformed")
    for field, value in (
        ("rows", count),
        ("positive_c_cells", positive),
        ("insufficient_cells", int(insufficient)),
        ("leo_zero_rows", zero_rows),
    ):
        cell[field] = int(cell.get(field, 0)) + int(value)
    cell["sum_c"] = float(cell.get("sum_c", 0.0)) + sum_c
    cell["loss_sum"] = float(cell.get("loss_sum", 0.0)) + loss_value
    cell["batches"] = int(cell.get("batches", 0)) + 1
    cell["nonempty_batches"] = int(cell.get("nonempty_batches", 0)) + int(count > 0)
    cell["finite_batches"] = int(cell.get("finite_batches", 0)) + 1
    cells[key] = cell


def update_hnccd_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    scenario: str,
    epoch: int,
    batch_index: int,
) -> Dict[str, Any]:
    """Accumulate G-only HNCCD scalars after the common C/G order receipt."""

    result = dict(receipt)
    if str(result.get("schema", "")) != HNCCD_RECEIPT_SCHEMA or result.get(
        "enabled"
    ) is not True:
        raise HNCCDRuntimeError("P1-HNCCD auxiliary receipt update is G-arm only")
    if str(scenario) not in FROZEN_HNCCD_SCENARIOS:
        raise HNCCDRuntimeError("P1-HNCCD scenario is outside frozen clear/low/rain")
    expected_keys = set(_cell_template())
    if (
        batch_info.get("finite") is not True
        or batch_info.get("n_lt_2_differentiable_zero") is not True
        or batch_info.get("no_active_renormalization") is not True
        or batch_info.get("streamed_cell_cross_covariance") is not True
        or batch_info.get("head_null_basis")
        != "FP32_CHOLESKY_TRIANGULAR_SOLVE_Q_d_by_4"
        or batch_info.get("training_accumulation_dtype") != "float32_OUTSIDE_AMP"
        or batch_info.get("forbids_detach_on_leo_or_head") is not True
        or batch_info.get("forbids_batch_d2_materialization") is not True
        or batch_info.get("forbids_batch_cell_d2_materialization") is not True
    ):
        raise HNCCDRuntimeError("P1-HNCCD batch semantic receipt drifted")
    cells = {str(key): dict(value) for key, value in dict(batch_info.get("cells", {})).items()}
    if set(cells) != expected_keys:
        raise HNCCDRuntimeError("P1-HNCCD G receipt lacks all receiver/class cells")
    rows = int(batch_info.get("rows", -1))
    positive = int(batch_info.get("positive_c_cells", -1))
    finite = int(batch_info.get("finite_c_cells", -1))
    insufficient = int(batch_info.get("insufficient_cells", -1))
    sum_c = float(batch_info.get("sum_c", float("nan")))
    loss_sum = float(batch_info.get("loss_sum", float("nan")))
    if (
        rows != FROZEN_HNCCD_BATCH_SIZE
        or positive < 0
        or positive > FROZEN_HNCCD_CELL_COUNT
        or finite != FROZEN_HNCCD_CELL_COUNT
        or insufficient < 0
        or insufficient > FROZEN_HNCCD_CELL_COUNT
        or int(batch_info.get("global_denominator", -1))
        != FROZEN_HNCCD_TERM_DIVISOR
        or not math.isfinite(sum_c)
        or not math.isfinite(loss_sum)
    ):
        raise HNCCDRuntimeError("P1-HNCCD G batch rows/C/loss do not close")
    common_events = list(result.get("hnccd_common_batch_cells", []))
    if not common_events:
        raise HNCCDRuntimeError("P1-HNCCD G batch lacks common C/G coverage receipt")
    common = dict(common_events[-1])
    if (
        int(common.get("epoch", -1)) != int(epoch)
        or int(common.get("batch_index", -1)) != int(batch_index)
        or str(common.get("scenario", "")) != str(scenario)
        or common.get("same_physical_clean_leo") is not True
        or len(str(common.get("row_order_sha256", ""))) != 64
    ):
        raise HNCCDRuntimeError("P1-HNCCD G/common same-physical receipt alignment drifted")
    common_counts = {str(key): int(value) for key, value in dict(common.get("n_rc", {})).items()}
    if common_counts != {key: int(value.get("n_rc", -1)) for key, value in cells.items()}:
        raise HNCCDRuntimeError("P1-HNCCD G/common n_rc receipt mismatch")
    scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(value_map).items()}
        for scene, value_map in dict(result.get("hnccd_scenes", {})).items()
    }
    scene_cells = scenes.get(str(scenario), _cell_template())
    if set(scene_cells) != expected_keys:
        raise HNCCDRuntimeError("P1-HNCCD G scene cells are malformed")
    for key in sorted(expected_keys):
        _accumulate_aux_cell(scene_cells, key=key, info=cells[key])
    scenes[str(scenario)] = scene_cells
    result["hnccd_scenes"] = scenes
    aux_events = list(result.get("hnccd_g_batch_aux", []))
    aux_events.append(
        {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "scenario": str(scenario),
            "row_order_sha256": str(common["row_order_sha256"]),
            "positive_c_cells": positive,
            "insufficient_cells": insufficient,
            "sum_c": sum_c,
            "loss_sum": loss_sum,
        }
    )
    result["hnccd_g_batch_aux"] = aux_events
    result["hnccd_batches"] = int(result.get("hnccd_batches", 0)) + 1
    result["hnccd_total_rows"] = int(result.get("hnccd_total_rows", 0)) + rows
    result["hnccd_positive_c_cells"] = int(result.get("hnccd_positive_c_cells", 0)) + positive
    result["hnccd_positive_c_batches"] = int(
        result.get("hnccd_positive_c_batches", 0)
    ) + int(positive > 0)
    result["hnccd_insufficient_cells"] = int(
        result.get("hnccd_insufficient_cells", 0)
    ) + insufficient
    result["hnccd_sum_c"] = float(result.get("hnccd_sum_c", 0.0)) + sum_c
    result["hnccd_loss_sum"] = float(result.get("hnccd_loss_sum", 0.0)) + loss_sum
    positive_scenes = {
        str(key): int(value)
        for key, value in dict(result.get("hnccd_scene_positive_batches", {})).items()
    }
    if set(positive_scenes) != set(FROZEN_HNCCD_SCENARIOS):
        raise HNCCDRuntimeError("P1-HNCCD scene-positive receipt is malformed")
    positive_scenes[str(scenario)] += int(positive > 0)
    result["hnccd_scene_positive_batches"] = positive_scenes
    return result


def update_hnccd_resource_receipt(
    receipt: Mapping[str, Any], *, peak_memory_bytes: Any, step_time_seconds: Any
) -> Dict[str, Any]:
    """Record scalar resource observations only; they never select a candidate."""

    result = dict(receipt)
    try:
        peak = int(peak_memory_bytes)
        duration = float(step_time_seconds)
    except (TypeError, ValueError) as exc:
        raise HNCCDRuntimeError("P1-HNCCD resource observation is not numeric") from exc
    if peak < 0 or not math.isfinite(duration) or duration < 0.0:
        raise HNCCDRuntimeError("P1-HNCCD resource observation is invalid")
    observations = [dict(value) for value in list(result.get("hnccd_resource_observations", []))]
    observations.append(
        {
            "peak_memory_bytes": peak,
            "step_time_seconds": duration,
            "selection_feedback": False,
        }
    )
    result["hnccd_resource_observations"] = observations
    result["resource_selection_feedback"] = False
    return result


def _require_hnccd_material_loss_finite(loss: torch.Tensor) -> None:
    """Reject a non-finite combined loss before the single AMP backward."""

    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise HNCCDRuntimeError("P1-HNCCD material loss must be a scalar tensor")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise HNCCDRuntimeError("P1-HNCCD material loss is non-finite")


def _hnccd_trainable_parameter_binding(
    model: torch.nn.Module,
) -> Tuple[Tuple[torch.nn.Parameter, ...], Dict[int, str]]:
    """Bind an exceptional raw-material VJP to the live optimizer parameters."""

    raw_model = getattr(model, "_orig_mod", model)
    parameters = tuple(parameter for parameter in raw_model.parameters() if parameter.requires_grad)
    if not parameters:
        raise HNCCDRuntimeError("P1-HNCCD raw material audit has no trainable parameters")
    names = {
        id(parameter): str(name)
        for name, parameter in raw_model.named_parameters()
        if parameter.requires_grad
    }
    if {id(parameter) for parameter in parameters} != set(names):
        raise HNCCDRuntimeError("P1-HNCCD raw material parameter-name binding is incomplete")
    return parameters, names


def _hnccd_state_step_value(state: Mapping[str, Any]) -> float:
    value = state.get("step", 0)
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise HNCCDRuntimeError("P1-HNCCD optimizer state step is invalid") from error
    if not math.isfinite(result) or result < 0.0:
        raise HNCCDRuntimeError("P1-HNCCD optimizer state step is non-finite")
    return result


def _hnccd_optimizer_state_steps(optimizer: torch.optim.Optimizer) -> Tuple[float, ...]:
    """Capture AdamW state steps to prove an AMP skip did not update it."""

    parameters = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group.get("params", ()):
            if not isinstance(parameter, torch.nn.Parameter) or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
    if not parameters:
        raise HNCCDRuntimeError("P1-HNCCD optimizer has no parameters")
    return tuple(_hnccd_state_step_value(optimizer.state.get(parameter, {})) for parameter in parameters)


def _hnccd_optimizer_steps_unchanged(before: Sequence[float], after: Sequence[float]) -> bool:
    before_values = tuple(float(value) for value in before)
    after_values = tuple(float(value) for value in after)
    return bool(
        before_values
        and len(before_values) == len(after_values)
        and all(math.isfinite(value) and value >= 0.0 for value in (*before_values, *after_values))
        and all(left == right for left, right in zip(before_values, after_values))
    )


def _hnccd_pending_optimizer_event(
    receipt: Mapping[str, Any],
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Bind exactly one step-or-skip event to the latest sealed HNCCD batch."""

    batches = list(receipt.get("hnccd_g_batch_aux", []))
    if not batches:
        raise HNCCDRuntimeError("P1-HNCCD optimizer event lacks a sealed G batch")
    batch = dict(batches[-1])
    try:
        epoch = int(batch["epoch"])
        batch_index = int(batch["batch_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise HNCCDRuntimeError("P1-HNCCD optimizer event batch identity is invalid") from error
    scenario = str(batch.get("scenario", ""))
    if epoch <= 0 or batch_index < 0 or scenario not in FROZEN_HNCCD_SCENARIOS:
        raise HNCCDRuntimeError("P1-HNCCD optimizer event batch identity drifted")
    history = [dict(event) for event in list(receipt.get("hnccd_optimizer_events", []))]
    identity = (epoch, batch_index, scenario)
    if any(
        (int(event.get("epoch", -1)), int(event.get("batch_index", -1)), str(event.get("scenario", "")))
        == identity
        for event in history
    ):
        raise HNCCDRuntimeError("P1-HNCCD optimizer event duplicates a G batch")
    return {"epoch": epoch, "batch_index": batch_index, "scenario": scenario}, history


def _hnccd_raw_material_gradient_audit(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    names: Mapping[int, str],
    scaled_nonfinite_parameter_ids: Sequence[int],
) -> Dict[str, Any]:
    """Use the retained existing graph once to distinguish an AMP-only overflow.

    This does not perform a second forward, an optimizer operation, or another
    scaled backward. It is only invoked after the single backward and unscale
    showed a non-finite parameter buffer.
    """

    try:
        gradients = torch.autograd.grad(
            loss,
            tuple(parameters),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
    except RuntimeError as error:
        raise HNCCDRuntimeError("P1-HNCCD raw material gradient audit failed") from error
    affected = {int(value) for value in scaled_nonfinite_parameter_ids}
    if not affected:
        raise HNCCDRuntimeError("P1-HNCCD raw material audit lacks scaled non-finite parameters")
    raw_nonfinite = []
    raw_missing = []
    for parameter, gradient in zip(parameters, gradients):
        if id(parameter) not in affected:
            continue
        name = names[id(parameter)]
        if gradient is None:
            raw_missing.append(name)
        elif not bool(torch.isfinite(gradient.detach()).all().item()):
            raw_nonfinite.append(name)
    return {
        "raw_material_vjp_finite": not raw_nonfinite and not raw_missing,
        "raw_material_nonfinite_parameter_names": tuple(sorted(raw_nonfinite)),
        "raw_material_missing_parameter_names": tuple(sorted(raw_missing)),
    }


def release_hnccd_retained_graph_roots(roots: Dict[str, Any]) -> None:
    """Clear every caller-owned retained-graph root before the next forward.

    The trainer moves output, loss, VJP, and logging tensor aliases into this
    single dictionary after the one permitted backward path. Clearing it is the
    explicit graph-release action; no second backward, unscale, forward, gc,
    or allocator-cache action is permitted here.
    """

    if not isinstance(roots, dict) or not roots:
        raise HNCCDRuntimeError("P1-HNCCD retained-graph root table is missing")
    if any(not isinstance(name, str) or not name for name in roots):
        raise HNCCDRuntimeError("P1-HNCCD retained-graph root name is invalid")
    roots.clear()
    if roots:
        raise HNCCDRuntimeError("P1-HNCCD retained-graph roots were not released")


def hnccd_scaled_backward_and_classify(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loss: torch.Tensor,
) -> Dict[str, Any]:
    """Run the one normal AMP backward/unscale and classify only overflow.

    A finite batch returns to the caller for the ordinary clip/step/update.
    An exceptional non-finite scaled buffer gets one raw combined-loss VJP on
    this already-retained graph, with no retry or second unscale.
    """

    _require_hnccd_material_loss_finite(loss)
    try:
        captured_scale = float(scaler.get_scale())
    except (AttributeError, TypeError, ValueError) as error:
        raise HNCCDRuntimeError("P1-HNCCD GradScaler scale is unavailable") from error
    if not math.isfinite(captured_scale) or captured_scale <= 0.0:
        raise HNCCDRuntimeError("P1-HNCCD GradScaler scale is invalid")
    parameters, names = _hnccd_trainable_parameter_binding(model)
    scaler.scale(loss).backward(retain_graph=True)
    scaler.unscale_(optimizer)
    scaled_nonfinite = tuple(
        parameter
        for parameter in parameters
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad.detach()).all().item())
    )
    if not scaled_nonfinite:
        return {
            "amp_overflow_detected": False,
            "captured_scale": captured_scale,
            "scaled_backward_count": 1,
            "optimizer_unscale_count": 1,
        }
    raw_audit = _hnccd_raw_material_gradient_audit(
        loss,
        parameters,
        names,
        tuple(id(parameter) for parameter in scaled_nonfinite),
    )
    raw_finite = bool(raw_audit["raw_material_vjp_finite"])
    return {
        "amp_overflow_detected": True,
        "amp_overflow_kind": (
            "COMBINED_SCALED_OVERFLOW_RAW_FINITE"
            if raw_finite
            else "COMBINED_RAW_NONFINITE_OR_DISCONNECTED"
        ),
        "amp_overflow_recoverable": raw_finite,
        "captured_scale": captured_scale,
        "scaled_backward_count": 1,
        "optimizer_unscale_count": 1,
        "scaled_nonfinite_parameter_names": tuple(sorted(names[id(parameter)] for parameter in scaled_nonfinite)),
        "raw_material_vjp_finite": raw_finite,
        "raw_material_nonfinite_parameter_names": raw_audit["raw_material_nonfinite_parameter_names"],
        "raw_material_missing_parameter_names": raw_audit["raw_material_missing_parameter_names"],
        "optimizer_state_before": _hnccd_optimizer_state_steps(optimizer),
    }


def finalize_hnccd_amp_overflow_skip(
    *,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    overflow: Mapping[str, Any],
) -> Dict[str, Any]:
    """Use public GradScaler APIs to back off exactly one raw-finite overflow."""

    if overflow.get("amp_overflow_detected") is not True or overflow.get(
        "amp_overflow_recoverable"
    ) is not True:
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow skip requires raw-finite evidence")
    if str(overflow.get("amp_overflow_kind", "")) != "COMBINED_SCALED_OVERFLOW_RAW_FINITE":
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow kind is invalid")
    if overflow.get("raw_material_vjp_finite") is not True or tuple(
        overflow.get("raw_material_nonfinite_parameter_names", ())
    ) or tuple(overflow.get("raw_material_missing_parameter_names", ())):
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow lacks raw-finite material evidence")
    scaled_names = tuple(str(value) for value in overflow.get("scaled_nonfinite_parameter_names", ()))
    if not scaled_names:
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow lacks scaled non-finite parameter evidence")
    try:
        scale_before = float(overflow["captured_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow lacks a captured scale") from error
    if not math.isfinite(scale_before) or scale_before <= 0.0:
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow captured scale is invalid")
    before = tuple(float(value) for value in overflow.get("optimizer_state_before", ()))
    if not before:
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow lacks optimizer-state evidence")
    scaler.step(optimizer)
    scaler.update()
    after = _hnccd_optimizer_state_steps(optimizer)
    try:
        scale_after = float(scaler.get_scale())
    except (AttributeError, TypeError, ValueError) as error:
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow post-scale is unavailable") from error
    if not math.isfinite(scale_after) or not (0.0 < scale_after < scale_before):
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow skip did not lower GradScaler scale")
    if not _hnccd_optimizer_steps_unchanged(before, after):
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow skip advanced optimizer state")
    optimizer.zero_grad(set_to_none=True)
    return {
        "amp_overflow_kind": "COMBINED_SCALED_OVERFLOW_RAW_FINITE",
        "pre_scale": scale_before,
        "post_scale": scale_after,
        "optimizer_state_unchanged": True,
        "optimizer_step_applied": False,
        "scaled_parameter_names": scaled_names,
    }


def update_hnccd_amp_overflow_receipt(
    receipt: Mapping[str, Any],
    *,
    overflow: Mapping[str, Any],
    finalized_skip: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a data-free HNCCD AMP classification without retrying a batch."""

    result = dict(receipt)
    if result.get("enabled") is not True:
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow receipt is G-arm only")
    if overflow.get("amp_overflow_detected") is not True:
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow receipt requires overflow evidence")
    kind = str(overflow.get("amp_overflow_kind", ""))
    recoverable = kind == "COMBINED_SCALED_OVERFLOW_RAW_FINITE"
    raw_failure = kind == "COMBINED_RAW_NONFINITE_OR_DISCONNECTED"
    if not (recoverable or raw_failure):
        raise HNCCDRuntimeError("P1-HNCCD AMP overflow receipt kind is invalid")
    if recoverable:
        if overflow.get("amp_overflow_recoverable") is not True or finalized_skip is None:
            raise HNCCDRuntimeError("P1-HNCCD raw-finite AMP overflow was not finalized as a skip")
        if str(finalized_skip.get("amp_overflow_kind", "")) != kind:
            raise HNCCDRuntimeError("P1-HNCCD AMP overflow kind changed before receipt")
        if overflow.get("raw_material_vjp_finite") is not True or tuple(
            overflow.get("raw_material_nonfinite_parameter_names", ())
        ) or tuple(overflow.get("raw_material_missing_parameter_names", ())):
            raise HNCCDRuntimeError("P1-HNCCD AMP overflow receipt lacks raw-finite evidence")
        scaled_names = tuple(str(value) for value in overflow.get("scaled_nonfinite_parameter_names", ()))
        if not scaled_names:
            raise HNCCDRuntimeError("P1-HNCCD AMP overflow receipt lacks scaled non-finite evidence")
        try:
            pre_scale = float(finalized_skip["pre_scale"])
            post_scale = float(finalized_skip["post_scale"])
        except (KeyError, TypeError, ValueError) as error:
            raise HNCCDRuntimeError("P1-HNCCD AMP overflow receipt lacks scale evidence") from error
        if not math.isfinite(pre_scale) or not math.isfinite(post_scale) or not (
            0.0 < post_scale < pre_scale
        ):
            raise HNCCDRuntimeError("P1-HNCCD AMP overflow receipt scale did not decrease")
        if finalized_skip.get("optimizer_state_unchanged") is not True or finalized_skip.get(
            "optimizer_step_applied"
        ) is not False:
            raise HNCCDRuntimeError("P1-HNCCD AMP overflow receipt optimizer state changed")
        result["hnccd_amp_overflow_raw_finite_batches"] = int(
            result.get("hnccd_amp_overflow_raw_finite_batches", 0)
        ) + 1
        result["hnccd_amp_skip_scale_decreased_batches"] = int(
            result.get("hnccd_amp_skip_scale_decreased_batches", 0)
        ) + 1
        result["hnccd_amp_skip_optimizer_state_unchanged_batches"] = int(
            result.get("hnccd_amp_skip_optimizer_state_unchanged_batches", 0)
        ) + 1
        result["hnccd_optimizer_step_attempts"] = int(
            result.get("hnccd_optimizer_step_attempts", 0)
        ) + 1
        consecutive = int(result.get("hnccd_consecutive_amp_overflow_skips", 0)) + 1
        result["hnccd_consecutive_amp_overflow_skips"] = consecutive
        result["hnccd_max_consecutive_amp_overflow_skips"] = max(
            int(result.get("hnccd_max_consecutive_amp_overflow_skips", 0)), consecutive
        )
        result["hnccd_persistent_amp_overflow"] = True
        batch_event, history = _hnccd_pending_optimizer_event(result)
        history.append(
            {
                **batch_event,
                "action": "AMP_OVERFLOW_SKIP",
                "pre_scale": pre_scale,
                "post_scale": post_scale,
                "optimizer_step_skipped": True,
                "effective_optimizer_steps": int(result.get("hnccd_effective_optimizer_steps", 0)),
            }
        )
        result["hnccd_optimizer_events"] = history
        result["hnccd_last_amp_overflow"] = {
            "kind": kind,
            "pre_scale": pre_scale,
            "post_scale": post_scale,
            "optimizer_step_skipped": True,
            "effective_optimizer_steps": int(result.get("hnccd_effective_optimizer_steps", 0)),
            "scaled_parameter_names": list(scaled_names),
        }
        return result
    if finalized_skip is not None:
        raise HNCCDRuntimeError("P1-HNCCD raw gradient failure cannot finalize a skip")
    result["hnccd_amp_overflow_raw_nonfinite_batches"] = int(
        result.get("hnccd_amp_overflow_raw_nonfinite_batches", 0)
    ) + 1
    result["hnccd_last_amp_overflow"] = {
        "kind": kind,
        "pre_scale": float(overflow.get("captured_scale", float("nan"))),
        "post_scale": None,
        "optimizer_step_skipped": False,
        "effective_optimizer_steps": int(result.get("hnccd_effective_optimizer_steps", 0)),
        "scaled_parameter_names": list(overflow.get("scaled_nonfinite_parameter_names", ())),
        "raw_nonfinite_parameter_names": list(
            overflow.get("raw_material_nonfinite_parameter_names", ())
        ),
        "raw_missing_parameter_names": list(
            overflow.get("raw_material_missing_parameter_names", ())
        ),
    }
    return result


def record_hnccd_material_nonfinite_receipt(
    receipt: Mapping[str, Any], *, reason: str
) -> Dict[str, Any]:
    """Record a fail-closed material non-finite without training data."""

    result = dict(receipt)
    if result.get("enabled") is not True:
        raise HNCCDRuntimeError("P1-HNCCD material non-finite receipt is G-arm only")
    if str(reason) not in {"total_loss_nonfinite", "post_clip_combined_gradient_nonfinite"}:
        raise HNCCDRuntimeError("P1-HNCCD material non-finite reason is invalid")
    result["hnccd_amp_overflow_material_nonfinite_batches"] = int(
        result.get("hnccd_amp_overflow_material_nonfinite_batches", 0)
    ) + 1
    result["hnccd_last_material_nonfinite"] = {
        "reason": str(reason),
        "effective_optimizer_steps": int(result.get("hnccd_effective_optimizer_steps", 0)),
    }
    return result


def update_hnccd_optimizer_step_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Record one effective optimizer step after finite unscaled gradients."""

    result = dict(receipt)
    if result.get("enabled") is not True:
        raise HNCCDRuntimeError("P1-HNCCD optimizer-step receipt is G-arm only")
    attempts = int(result.get("hnccd_optimizer_step_attempts", 0)) + 1
    effective = int(result.get("hnccd_effective_optimizer_steps", 0)) + 1
    if attempts <= 0 or effective <= 0 or effective > attempts:
        raise HNCCDRuntimeError("P1-HNCCD optimizer-step receipt counters are invalid")
    result["hnccd_optimizer_step_attempts"] = attempts
    result["hnccd_effective_optimizer_steps"] = effective
    result["hnccd_consecutive_amp_overflow_skips"] = 0
    result["hnccd_persistent_amp_overflow"] = False
    batch_event, history = _hnccd_pending_optimizer_event(result)
    history.append(
        {
            **batch_event,
            "action": "EFFECTIVE_OPTIMIZER_STEP",
            "optimizer_step_applied": True,
            "effective_optimizer_steps": effective,
        }
    )
    result["hnccd_optimizer_events"] = history
    result["hnccd_last_optimizer_step"] = {
        "optimizer_step_applied": True,
        "effective_optimizer_steps": effective,
    }
    return result


def _validate_hnccd_common_terminal_contract(result: Mapping[str, Any]) -> None:
    """Fail closed on warm-start, exact-head, and common C/G receipt drift."""

    if str(result.get("schema", "")) != HNCCD_RECEIPT_SCHEMA:
        raise HNCCDRuntimeError("P1-HNCCD terminal receipt schema is invalid")
    for key in (
        "baseline_sha256",
        "initial_checkpoint_sha256",
        "class_order_binding_sha256",
        "source_labeled_indices_sha256",
        "source_split_manifest_sha256",
        "source_receiver_order_sha256",
        "optimizer_initial_state_sha256",
        "common_batch_sequence_sha256",
    ):
        if len(str(result.get(key, "") or "")) != 64:
            raise HNCCDRuntimeError(f"P1-HNCCD terminal receipt lacks {key}")
    if str(result.get("checkpoint_role", "") or "") != "training_final_only":
        raise HNCCDRuntimeError("P1-HNCCD requires training_final_only warm start")
    if result.get("strict_model_keys") is not True or result.get("optimizer_state_restored") is not False or result.get(
        "rng_state_restored"
    ) is not False:
        raise HNCCDRuntimeError("P1-HNCCD warm-start must restore exact model weights only")
    if str(result.get("optimizer_type", "")) != FROZEN_HNCCD_OPTIMIZER_TYPE:
        raise HNCCDRuntimeError("P1-HNCCD terminal optimizer_type must be AdamW")
    if result.get("optimizer_initial_state_empty") is not True:
        raise HNCCDRuntimeError("P1-HNCCD requires a new AdamW initial-state receipt")
    if result.get("amp_contract") != "COMMON_TRAINER_AMP_ENABLED":
        raise HNCCDRuntimeError("P1-HNCCD terminal AMP contract drifted")
    if result.get("common_l_base_head_input_path_verified") is not True:
        raise HNCCDRuntimeError("P1-HNCCD common L_base exact head-input path is not verified")
    if (
        int(result.get("fixed_batch_size", -1)) != FROZEN_HNCCD_BATCH_SIZE
        or int(result.get("fixed_local_class_count", -1)) != FROZEN_HNCCD_HEAD_ROWS
        or int(result.get("frozen_feature_dim", -1)) != FROZEN_HNCCD_FEATURE_DIM
        or int(result.get("frozen_source_receiver_count", -1)) != FROZEN_HNCCD_SOURCE_RECEIVER_COUNT
        or int(result.get("loss_global_denominator", -1)) != FROZEN_HNCCD_TERM_DIVISOR
        or result.get("common_loader_drop_last") is not True
        or result.get("head_full_row_rank_required") is not True
        or result.get("head_null_basis_rule")
        != "FP32_DIFFERENTIABLE_CHOLESKY_WWT_AND_TRIANGULAR_SOLVE_Q_EQ_WT_LINVTRANSPOSE_NO_PINV_EPSILON_OR_FALLBACK"
        or result.get("training_accumulation_dtype") != "float32_OUTSIDE_AMP"
    ):
        raise HNCCDRuntimeError("P1-HNCCD terminal frozen shape/head-null contract drifted")
    if (
        str(result.get("exact_head_weight_path", ""))
        != "model.id_backbone.cls_head.head.weight"
        or list(result.get("exact_head_weight_shape", []))
        != [FROZEN_HNCCD_HEAD_ROWS, FROZEN_HNCCD_FEATURE_DIM]
        or tuple(int(value) for value in result.get("local_class_ids", ()))
        != FROZEN_HNCCD_CLASS_IDS
        or int(result.get("source_receiver_count", -1)) != FROZEN_HNCCD_SOURCE_RECEIVER_COUNT
    ):
        raise HNCCDRuntimeError("P1-HNCCD terminal exact head/local4/receiver7 binding drifted")
    source_provenance = str(result.get("source_receiver_provenance", ""))
    if source_provenance != "SOURCE_SPLIT_RECEIPT_ORDERED_SOURCE_RECEIVERS_PHYSICAL_ID_BOUND_L_ONLY":
        raise HNCCDRuntimeError("P1-HNCCD terminal source-L receiver provenance drifted")
    for forbidden_key in (
        "uses_target_rows",
        "uses_proxy_rows",
        "uses_held_rows",
        "uses_unlabeled_rows",
        "uses_ema_or_state",
        "uses_threshold",
        "uses_cross_sample_pairing",
        "uses_cross_receiver_pairing",
        "resource_selection_feedback",
    ):
        if result.get(forbidden_key) is not False:
            raise HNCCDRuntimeError(f"P1-HNCCD terminal forbidden route drifted: {forbidden_key}")
    if int(result.get("proxy_rows", -1)) != 0 or int(result.get("held_rows", -1)) != 0:
        raise HNCCDRuntimeError("P1-HNCCD target/proxy/held rows must remain outside training")
    batches = int(result.get("common_batch_sequence_batches", 0))
    rows = int(result.get("common_batch_sequence_rows", 0))
    scenario_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    common_scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("hnccd_common_cells", {})).items()
    }
    expected_keys = set(_cell_template())
    if (
        batches <= 0
        or rows != batches * FROZEN_HNCCD_BATCH_SIZE
        or set(scenario_batches) != set(FROZEN_HNCCD_SCENARIOS)
        or set(common_scenes) != set(FROZEN_HNCCD_SCENARIOS)
        or any(value <= 0 for value in scenario_batches.values())
    ):
        raise HNCCDRuntimeError("P1-HNCCD common batch/scenario receipt is incomplete")
    for scenario in FROZEN_HNCCD_SCENARIOS:
        cells = common_scenes[scenario]
        if set(cells) != expected_keys:
            raise HNCCDRuntimeError("P1-HNCCD terminal common cells are incomplete")
        if sum(int(cell.get("rows", -1)) for cell in cells.values()) != scenario_batches[scenario] * FROZEN_HNCCD_BATCH_SIZE:
            raise HNCCDRuntimeError("P1-HNCCD terminal common cell rows do not close")
        for cell in cells.values():
            if (
                int(cell.get("batches", -1)) != scenario_batches[scenario]
                or int(cell.get("rows", -1)) < 0
                or int(cell.get("nonempty_batches", -1)) < 0
            ):
                raise HNCCDRuntimeError("P1-HNCCD terminal common cell counter drifted")
    events = list(result.get("hnccd_common_batch_cells", []))
    if len(events) != batches:
        raise HNCCDRuntimeError("P1-HNCCD terminal common batch receipt is incomplete")
    for event in events:
        counts = {str(key): int(value) for key, value in dict(event.get("n_rc", {})).items()}
        if (
            event.get("same_physical_clean_leo") is not True
            or len(str(event.get("row_order_sha256", ""))) != 64
            or int(event.get("fixed_denominator", -1)) != FROZEN_HNCCD_TERM_DIVISOR
            or set(counts) != expected_keys
            or sum(counts.values()) != FROZEN_HNCCD_BATCH_SIZE
        ):
            raise HNCCDRuntimeError("P1-HNCCD terminal common same-physical/order receipt drifted")
    observations = result.get("hnccd_resource_observations")
    if type(observations) is not list or len(observations) != batches:
        raise HNCCDRuntimeError("P1-HNCCD terminal resource observations must close one-to-one with common C/G batches")
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise HNCCDRuntimeError("P1-HNCCD terminal resource observation must be a mapping")
        peak_memory_bytes = observation.get("peak_memory_bytes")
        step_time_seconds = observation.get("step_time_seconds")
        if type(peak_memory_bytes) is not int or peak_memory_bytes < 0:
            raise HNCCDRuntimeError("P1-HNCCD terminal peak_memory_bytes must be a nonnegative strict int")
        if (
            isinstance(step_time_seconds, bool)
            or not isinstance(step_time_seconds, (int, float))
            or not math.isfinite(float(step_time_seconds))
            or float(step_time_seconds) < 0.0
        ):
            raise HNCCDRuntimeError("P1-HNCCD terminal step_time_seconds must be finite and nonnegative")
        if observation.get("selection_feedback") is not False:
            raise HNCCDRuntimeError("P1-HNCCD resource telemetry must not select a model")


def _validate_hnccd_optimizer_terminal(result: Mapping[str, Any], *, batch_total: int) -> None:
    try:
        counts = {
            "attempts": int(result.get("hnccd_optimizer_step_attempts", 0)),
            "effective": int(result.get("hnccd_effective_optimizer_steps", 0)),
            "raw_finite_skips": int(result.get("hnccd_amp_overflow_raw_finite_batches", 0)),
            "raw_nonfinite": int(result.get("hnccd_amp_overflow_raw_nonfinite_batches", 0)),
            "material_nonfinite": int(result.get("hnccd_amp_overflow_material_nonfinite_batches", 0)),
            "scale_decreased": int(result.get("hnccd_amp_skip_scale_decreased_batches", 0)),
            "state_unchanged": int(result.get("hnccd_amp_skip_optimizer_state_unchanged_batches", 0)),
            "consecutive": int(result.get("hnccd_consecutive_amp_overflow_skips", 0)),
            "maximum": int(result.get("hnccd_max_consecutive_amp_overflow_skips", 0)),
        }
    except (TypeError, ValueError) as error:
        raise HNCCDRuntimeError("P1-HNCCD terminal optimizer/AMP counters are invalid") from error
    if any(value < 0 for value in counts.values()):
        raise HNCCDRuntimeError("P1-HNCCD terminal optimizer/AMP counters are negative")
    if (
        counts["effective"] <= 0
        or counts["attempts"] != batch_total
        or counts["attempts"] != counts["effective"] + counts["raw_finite_skips"]
        or counts["raw_nonfinite"] != 0
        or counts["material_nonfinite"] != 0
        or counts["scale_decreased"] != counts["raw_finite_skips"]
        or counts["state_unchanged"] != counts["raw_finite_skips"]
        or counts["maximum"] < counts["consecutive"]
        or bool(result.get("hnccd_persistent_amp_overflow", False)) != bool(counts["consecutive"] > 0)
        or counts["consecutive"] > 0
    ):
        raise HNCCDRuntimeError("P1-HNCCD terminal AMP/effective-step closure failed")
    aux_events = [dict(event) for event in list(result.get("hnccd_g_batch_aux", []))]
    optimizer_events = [dict(event) for event in list(result.get("hnccd_optimizer_events", []))]
    expected_identity = [
        (int(event.get("epoch", -1)), int(event.get("batch_index", -1)), str(event.get("scenario", "")))
        for event in aux_events
    ]
    actual_identity = [
        (int(event.get("epoch", -1)), int(event.get("batch_index", -1)), str(event.get("scenario", "")))
        for event in optimizer_events
    ]
    if len(optimizer_events) != batch_total or actual_identity != expected_identity:
        raise HNCCDRuntimeError("P1-HNCCD terminal optimizer events do not close per G batch")
    effective_seen = 0
    skip_seen = 0
    for event in optimizer_events:
        action = str(event.get("action", ""))
        if action == "EFFECTIVE_OPTIMIZER_STEP":
            effective_seen += 1
            if event.get("optimizer_step_applied") is not True or int(
                event.get("effective_optimizer_steps", -1)
            ) != effective_seen:
                raise HNCCDRuntimeError("P1-HNCCD terminal effective optimizer event is invalid")
        elif action == "AMP_OVERFLOW_SKIP":
            skip_seen += 1
            try:
                pre_scale = float(event["pre_scale"])
                post_scale = float(event["post_scale"])
            except (KeyError, TypeError, ValueError) as error:
                raise HNCCDRuntimeError("P1-HNCCD terminal AMP skip lacks scale receipt") from error
            if (
                event.get("optimizer_step_skipped") is not True
                or int(event.get("effective_optimizer_steps", -1)) != effective_seen
                or not math.isfinite(pre_scale)
                or not math.isfinite(post_scale)
                or not (0.0 < post_scale < pre_scale)
            ):
                raise HNCCDRuntimeError("P1-HNCCD terminal AMP skip event is invalid")
        else:
            raise HNCCDRuntimeError("P1-HNCCD terminal optimizer event action is invalid")
    if effective_seen != counts["effective"] or skip_seen != counts["raw_finite_skips"]:
        raise HNCCDRuntimeError("P1-HNCCD terminal optimizer events do not match counters")
    last_step = dict(result.get("hnccd_last_optimizer_step", {}))
    if last_step.get("optimizer_step_applied") is not True or int(
        last_step.get("effective_optimizer_steps", -1)
    ) != counts["effective"]:
        raise HNCCDRuntimeError("P1-HNCCD terminal effective optimizer-step receipt is invalid")


def validate_hnccd_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless actual common and per-scene HNCCD evidence closes."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    _validate_hnccd_common_terminal_contract(result)
    enabled = result.get("enabled")
    if enabled is not True and enabled is not False:
        raise HNCCDRuntimeError("P1-HNCCD terminal enabled flag must be strict bool")
    zero_keys = (
        "hnccd_batches",
        "hnccd_total_rows",
        "hnccd_positive_c_cells",
        "hnccd_positive_c_batches",
        "hnccd_insufficient_cells",
        "hnccd_amp_overflow_raw_finite_batches",
        "hnccd_amp_overflow_raw_nonfinite_batches",
        "hnccd_amp_overflow_material_nonfinite_batches",
        "hnccd_amp_skip_scale_decreased_batches",
        "hnccd_amp_skip_optimizer_state_unchanged_batches",
        "hnccd_effective_optimizer_steps",
        "hnccd_optimizer_step_attempts",
        "hnccd_consecutive_amp_overflow_skips",
        "hnccd_max_consecutive_amp_overflow_skips",
    )
    if enabled is False:
        if any(int(result.get(key, 0)) != 0 for key in zero_keys) or abs(
            float(result.get("hnccd_sum_c", 0.0))
        ) > _TOLERANCE or abs(float(result.get("hnccd_loss_sum", 0.0))) > _TOLERANCE:
            raise HNCCDRuntimeError("P1-HNCCD C arm must retain zero auxiliary counters")
        if (
            bool(result.get("hnccd_scenes"))
            or bool(result.get("hnccd_g_batch_aux"))
            or bool(result.get("hnccd_gradient_audit_scenes"))
            or bool(result.get("hnccd_last_amp_overflow"))
            or bool(result.get("hnccd_last_material_nonfinite"))
            or bool(result.get("hnccd_last_optimizer_step"))
            or bool(result.get("hnccd_optimizer_events"))
            or bool(result.get("hnccd_gradient_audit_attempted", False))
            or bool(result.get("hnccd_gradient_audit_completed", False))
            or bool(result.get("hnccd_persistent_amp_overflow", False))
        ):
            raise HNCCDRuntimeError("P1-HNCCD C arm must retain N/A-or-zero auxiliary fields")
        result["hnccd_terminal_contract"] = "CONTROL_ARM_COMMON_SAME_PHYSICAL_ORDER_B128_RX7_LOCAL4_DENOM28_AUX_NA_OR_ZERO"
        result["hnccd_terminal_contract_passed"] = True
        return result
    expected_keys = set(_cell_template())
    scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("hnccd_scenes", {})).items()
    }
    common_scenes = {
        str(scene): {str(key): dict(value) for key, value in dict(cells).items()}
        for scene, cells in dict(result.get("hnccd_common_cells", {})).items()
    }
    scene_batches = {
        str(key): int(value)
        for key, value in dict(result.get("common_scenario_batches", {})).items()
    }
    if set(scenes) != set(FROZEN_HNCCD_SCENARIOS):
        raise HNCCDRuntimeError("P1-HNCCD terminal G scene coverage is incomplete")
    total_rows = 0
    total_positive_cells = 0
    total_positive_batches = 0
    total_insufficient = 0
    total_sum_c = 0.0
    total_loss = 0.0
    for scenario in FROZEN_HNCCD_SCENARIOS:
        cells = scenes[scenario]
        common_cells = common_scenes[scenario]
        if set(cells) != expected_keys or set(common_cells) != expected_keys:
            raise HNCCDRuntimeError("P1-HNCCD terminal G/common cell coverage is incomplete")
        scene_rows = scene_positive_cells = scene_insufficient = 0
        scene_sum_c = scene_loss = 0.0
        for key in sorted(expected_keys):
            cell = cells[key]
            common_cell = common_cells[key]
            rows = int(cell.get("rows", -1))
            positive = int(cell.get("positive_c_cells", -1))
            insufficient = int(cell.get("insufficient_cells", -1))
            batches = int(cell.get("batches", -1))
            finite_batches = int(cell.get("finite_batches", -1))
            sum_c = float(cell.get("sum_c", float("nan")))
            loss_sum = float(cell.get("loss_sum", float("nan")))
            if (
                rows < 0
                or positive < 0
                or insufficient < 0
                or batches != scene_batches[scenario]
                or finite_batches != scene_batches[scenario]
                or rows != int(common_cell.get("rows", -2))
                or not math.isfinite(sum_c)
                or not math.isfinite(loss_sum)
            ):
                raise HNCCDRuntimeError("P1-HNCCD terminal per-cell receipt drifted")
            scene_rows += rows
            scene_positive_cells += positive
            scene_insufficient += insufficient
            scene_sum_c += sum_c
            scene_loss += loss_sum
        if (
            scene_rows != scene_batches[scenario] * FROZEN_HNCCD_BATCH_SIZE
            or scene_positive_cells <= 0
            or not math.isfinite(scene_sum_c)
            or not math.isfinite(scene_loss)
        ):
            raise HNCCDRuntimeError("P1-HNCCD terminal each scene needs positive HNCCD evidence")
        total_rows += scene_rows
        total_positive_cells += scene_positive_cells
        total_insufficient += scene_insufficient
        total_sum_c += scene_sum_c
        total_loss += scene_loss
        total_positive_batches += int(result.get("hnccd_scene_positive_batches", {}).get(scenario, 0))
    aux_events = list(result.get("hnccd_g_batch_aux", []))
    if (
        len(aux_events) != int(result.get("hnccd_batches", -1))
        or int(result.get("hnccd_batches", -1)) != int(result.get("common_batch_sequence_batches", -2))
        or int(result.get("hnccd_total_rows", -1)) != total_rows
        or int(result.get("hnccd_positive_c_cells", -1)) != total_positive_cells
        or int(result.get("hnccd_positive_c_batches", -1)) != total_positive_batches
        or int(result.get("hnccd_insufficient_cells", -1)) != total_insufficient
        or total_positive_cells <= 0
        or total_positive_batches <= 0
        or not _float32_ledger_close(float(result.get("hnccd_sum_c", float("nan"))), total_sum_c)
        or not _float32_ledger_close(float(result.get("hnccd_loss_sum", float("nan"))), total_loss)
    ):
        raise HNCCDRuntimeError("P1-HNCCD terminal G cell/positive/loss counters do not close")
    audits = {
        str(key): dict(value)
        for key, value in dict(result.get("hnccd_gradient_audit_scenes", {})).items()
    }
    if result.get("hnccd_gradient_audit_completed") is not True or set(audits) != set(
        FROZEN_HNCCD_SCENARIOS
    ):
        raise HNCCDRuntimeError("P1-HNCCD terminal per-scene first-positive raw VJP audit is incomplete")
    for scenario in FROZEN_HNCCD_SCENARIOS:
        _validate_hnccd_gradient_audit_payload(audits[scenario])
    _validate_hnccd_optimizer_terminal(result, batch_total=int(result.get("hnccd_batches", 0)))
    result["hnccd_terminal_contract"] = "FORMAL_COMMON_C_G_SAME_PHYSICAL_ORDER_B128_RX7_LOCAL4_DENOM28_FP32_CHOLESKY_HEAD_NULL_CROSS_COVARIANCE_PER_SCENE_RAW_VJP_AND_EFFECTIVE_ADAMW_STEPS"
    result["hnccd_terminal_contract_passed"] = True
    return result


def _hnccd_failure_fingerprint(error: BaseException) -> str:
    message = str(error).lower()
    if "cholesky" in message or "rank" in message or "head" in message:
        return "HNCCD_EXACT_HEAD_CHOLESKY_FAILURE"
    if "vjp" in message or "gradient" in message:
        return "HNCCD_AUXILIARY_VJP_FAILURE"
    if "non-finite" in message or "nonfinite" in message:
        return "HNCCD_NONFINITE"
    if "128" in message or "28" in message or "receiver" in message or "local4" in message:
        return "HNCCD_FIXED_SHAPE_OR_CELL_FAILURE"
    if "sequence" in message or "receipt" in message or "coverage" in message:
        return "HNCCD_RECEIPT_CLOSURE_FAILURE"
    if "binding" in message or "physical" in message or "source" in message:
        return "HNCCD_SOURCE_BINDING_FAILURE"
    return "HNCCD_RUNTIME_FAILURE"


def write_hnccd_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist a data-free fail-closed HNCCD failure record."""

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise HNCCDRuntimeError(
            f"P1-HNCCD failure receipt output directory is absent: {target_dir}"
        )
    payload = {
        "schema": "cvs.phase1.hnccd_failure_receipt.v1",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage or ""),
        "exception_type": type(error).__name__,
        "exception_fingerprint": _hnccd_failure_fingerprint(error),
        "message": str(error),
        "receipt": dict(receipt),
    }
    target = target_dir / "phase1_hnccd_failure_receipt.json"
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    fd, temporary_name = mkstemp(
        prefix=".hnccd_failure_receipt.", suffix=".tmp", dir=str(target_dir)
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
