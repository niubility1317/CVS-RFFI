"""Frozen paired angular-margin restoration (P1-PAMR) for Phase1.

PAMR deliberately consumes only paired clean/LEO identity features, transmitter
labels and the already-present classifier weight.  The clean feature and class
weight form a detached scalar angular-boundary target; gradients therefore flow
only through the LEO feature branch and the shared encoder that produced it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


FROZEN_PAMR_LAMBDA = 0.05
_EPS = 1e-8


class PAMRConfigurationError(ValueError):
    """Raised when a frozen P1-PAMR continuation configuration drifts."""


class PAMRRuntimeError(RuntimeError):
    """Raised when a PAMR batch cannot prove its frozen geometry contract."""


@dataclass(frozen=True)
class PAMRConfig:
    """Frozen settings consumed by the Phase1 training-loop integration."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float
    audit_only: bool = False


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise PAMRConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > 1e-12:
        raise PAMRConfigurationError(
            f"Frozen P1-PAMR requires {name}={expected:.12g}, got {actual!r}"
        )


def _require_disabled(args: Any, names: Tuple[str, ...]) -> None:
    active = []
    for name in names:
        value = getattr(args, name, False)
        if isinstance(value, bool):
            is_active = bool(value)
        else:
            try:
                is_active = abs(float(value)) > 1e-12
            except (TypeError, ValueError):
                is_active = bool(value)
        if is_active:
            active.append(name)
    if active:
        raise PAMRConfigurationError(
            "Frozen P1-PAMR forbids stacked routes: " + ", ".join(active)
        )


def validate_pamr_args(args: Any) -> PAMRConfig:
    """Validate the immutable C/G P1-PAMR continuation contract."""

    frozen_mode = _bool_arg(args, "phase1_pamr_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_pamr_enabled", False)
    audit_only = _bool_arg(args, "phase1_pamr_audit_only", False)
    loss_weight = _float_arg(args, "lambda_pamr", 0.0)
    if audit_only and not (frozen_mode and enabled):
        raise PAMRConfigurationError("P1-PAMR audit mode requires the frozen enabled G arm")
    if not frozen_mode and not enabled:
        return PAMRConfig(False, False, 0.0, False)
    if enabled and not frozen_mode:
        raise PAMRConfigurationError(
            "--phase1_pamr_enabled requires --phase1_pamr_frozen_mode true"
        )
    if enabled:
        _require_close("lambda_pamr", loss_weight, FROZEN_PAMR_LAMBDA)
    else:
        _require_close("lambda_pamr", loss_weight, 0.0)
    if bool(getattr(args, "from_scratch", True)):
        raise PAMRConfigurationError("Frozen P1-PAMR requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise PAMRConfigurationError("Frozen P1-PAMR requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise PAMRConfigurationError("Frozen P1-PAMR must train the shared encoder")
    required_epochs = 1 if audit_only else 40
    if int(getattr(args, "epochs", 0)) != required_epochs:
        raise PAMRConfigurationError(
            f"Frozen P1-PAMR requires exactly --epochs {required_epochs}"
        )
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise PAMRConfigurationError("Frozen P1-PAMR requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise PAMRConfigurationError("Frozen P1-PAMR remains source-validation-only")
    if str(getattr(args, "id_feature_key", "")) != "feat_joint":
        raise PAMRConfigurationError("Frozen P1-PAMR requires --id_feature_key feat_joint")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise PAMRConfigurationError("Frozen P1-PAMR requires the GeoSat-C paired LEO path")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise PAMRConfigurationError("Frozen P1-PAMR requires non-concatenated paired LEO rows")
    if bool(getattr(args, "use_unlabeled", False)):
        raise PAMRConfigurationError("Frozen P1-PAMR forbids unlabeled/domain-gated continuation")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise PAMRConfigurationError("Frozen P1-PAMR forbids RX-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise PAMRConfigurationError("Frozen P1-PAMR permits no extra views beyond paired clean/LEO")
    if bool(getattr(args, "reject_head", False)):
        raise PAMRConfigurationError("Frozen P1-PAMR forbids rejection heads")
    _require_disabled(
        args,
        (
            "phase1_ccpc_leo_frozen_mode",
            "phase1_ccpc_leo_enabled",
            "phase1_ccpc_leo_gradient_audit_only",
            "lambda_ccpc_leo",
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
    return PAMRConfig(True, enabled, loss_weight, audit_only)


def pamr_config_receipt(config: PAMRConfig) -> Dict[str, Any]:
    """Create the data-free, immutable section of a PAMR receipt."""

    return {
        "schema": "cvs.phase1.pamr_receipt.v1",
        "method": "P1_PAMR",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "audit_only": bool(config.audit_only),
        "technical_only": bool(config.audit_only),
        "technical_only_claim": "NO_PERFORMANCE_RESULT" if config.audit_only else "",
        "performance_result_available": False,
        "loss_rule": "TX_EQUAL_RELU_DETACHED_CLEAN_RAW_COSINE_MARGIN_MINUS_LEO_MARGIN",
        "clean_gate_rule": "RAW_COSINE_ARGMAX_EQUALS_TX_LABEL",
        "id_feature_key": "feat_joint",
        "class_weight_path": "id_backbone.cls_head.head.weight",
        "class_order_contract": "DATASET_TX_LABEL_INDEX_EQUALS_HEAD_WEIGHT_ROW_INDEX",
        "clean_margin_detached": bool(config.enabled),
        "class_weight_detached": bool(config.enabled),
        "uses_detached_clean_boundary_self_distillation": bool(config.enabled),
        "leo_only_gradient": bool(config.enabled),
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
        "proxy_rows": 0,
        "held_rows": 0,
        "expected_tx_class_ids": [],
        "class_count": 0,
        "pamr_batches": 0,
        "pamr_rows": 0,
        "pamr_valid_anchors": 0,
        "pamr_active_hinges": 0,
        "pamr_valid_anchors_by_tx": {},
        "pamr_active_hinges_by_tx": {},
        "pamr_grad_nonzero_batches": 0,
        "pamr_grad_zero_batches": 0,
        "pamr_grad_nonfinite_batches": 0,
        "pamr_gradient_audit_attempted_batches": 0,
        "pamr_gradient_audit_completed": False,
        "pamr_gradient_audit_mode": (
            "FIRST_EFFECTIVE_BATCH_ONLY" if config.audit_only else "NOT_REQUIRED_FORMAL"
        ),
        "pamr_shared_encoder_parameter_scope": "id_backbone.* excluding id_backbone.cls_head.head.weight",
        "pamr_shared_gradient_relation_batches": 0,
        "pamr_shared_gradient_cosine_count": 0,
        "pamr_shared_gradient_cosine_sum": 0.0,
        "pamr_shared_gradient_cosine_min": None,
        "pamr_shared_gradient_cosine_max": None,
        "pamr_shared_gradient_norm_ratio_count": 0,
        "pamr_shared_gradient_norm_ratio_sum": 0.0,
        "pamr_shared_gradient_norm_ratio_min": None,
        "pamr_shared_gradient_norm_ratio_max": None,
        "pamr_terminal_gradient_contract": "PENDING",
        "pamr_terminal_gradient_contract_passed": False,
    }


def _validate_loss_inputs(
    z_leo: torch.Tensor,
    z_clean: torch.Tensor,
    tx_labels: torch.Tensor,
    class_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not all(torch.is_tensor(value) for value in (z_leo, z_clean, tx_labels, class_weight)):
        raise PAMRRuntimeError("P1-PAMR requires tensor paired features, labels and class weight")
    if z_leo.ndim != 2 or z_clean.ndim != 2 or class_weight.ndim != 2:
        raise PAMRRuntimeError("P1-PAMR expects rank-2 features and classifier weight")
    if tuple(z_leo.shape) != tuple(z_clean.shape):
        raise PAMRRuntimeError("P1-PAMR clean and LEO features must have identical shape")
    if int(z_leo.size(1)) != int(class_weight.size(1)):
        raise PAMRRuntimeError("P1-PAMR feat_joint dimension must equal classifier head weight dimension")
    labels = tx_labels.reshape(-1).long()
    if int(labels.numel()) != int(z_leo.size(0)):
        raise PAMRRuntimeError("P1-PAMR transmitter labels must match paired feature rows")
    if int(labels.numel()) == 0:
        raise PAMRRuntimeError("P1-PAMR requires at least one paired row")
    if int(class_weight.size(0)) < 2:
        raise PAMRRuntimeError("P1-PAMR requires at least two classifier classes")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= int(class_weight.size(0)):
        raise PAMRRuntimeError("P1-PAMR labels must index the classifier weight rows")
    if not bool(z_leo.requires_grad):
        raise PAMRRuntimeError("P1-PAMR requires an LEO feature with a live gradient path")
    for name, value in (
        ("z_leo", z_leo),
        ("z_clean", z_clean),
        ("class_weight", class_weight),
    ):
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise PAMRRuntimeError(f"P1-PAMR {name} contains non-finite values")
    return z_leo, z_clean, labels, class_weight


def _raw_cosine_and_margin(
    feature: torch.Tensor,
    labels: torch.Tensor,
    normalized_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cosine = F.linear(F.normalize(feature.float(), dim=1, eps=_EPS), normalized_weight)
    true_cosine = cosine.gather(1, labels.unsqueeze(1)).squeeze(1)
    alternatives = cosine.clone()
    alternatives.scatter_(1, labels.unsqueeze(1), float("-inf"))
    margin = true_cosine - alternatives.max(dim=1).values
    return cosine, margin, true_cosine


def pamr_loss(
    z_leo: torch.Tensor,
    z_clean: torch.Tensor,
    tx_labels: torch.Tensor,
    class_weight: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return an equal-TX angular-margin restoration loss and data-free counts.

    The clean raw cosine gate and clean margin are detached.  It is therefore
    deliberately *not* an L2/cosine feature alignment loss: only an insufficient
    LEO class margin is penalised.
    """

    z_leo, z_clean, labels, class_weight = _validate_loss_inputs(
        z_leo, z_clean, tx_labels, class_weight
    )
    normalized_weight = F.normalize(class_weight.detach().float(), dim=1, eps=_EPS)
    clean_cosine, clean_margin, _ = _raw_cosine_and_margin(
        z_clean.detach(), labels, normalized_weight
    )
    _, leo_margin, _ = _raw_cosine_and_margin(z_leo, labels, normalized_weight)
    clean_correct = clean_cosine.argmax(dim=1).eq(labels)
    hinge = F.relu(clean_margin.detach() - leo_margin)

    present_classes = torch.unique(labels, sorted=True)
    per_tx_losses = []
    valid_by_tx: Dict[str, int] = {}
    hinge_by_tx: Dict[str, int] = {}
    for class_id in present_classes.tolist():
        class_mask = labels.eq(int(class_id))
        valid_mask = class_mask & clean_correct
        valid_count = int(valid_mask.sum().item())
        active_count = int((hinge[valid_mask] > 0.0).sum().item()) if valid_count else 0
        key = str(int(class_id))
        valid_by_tx[key] = valid_count
        hinge_by_tx[key] = active_count
        if valid_count:
            per_tx_losses.append(hinge[valid_mask].mean())
    if per_tx_losses:
        loss = torch.stack(per_tx_losses).mean()
    else:
        # Preserve a legal graph for a finite stationary/empty-gate batch.  The
        # terminal receipt, rather than a single batch, establishes coverage.
        loss = z_leo.sum() * 0.0
    if not bool(torch.isfinite(loss.detach()).item()):
        raise PAMRRuntimeError("P1-PAMR loss is non-finite")
    return loss, {
        "rows": int(labels.numel()),
        "classes": int(present_classes.numel()),
        "valid_anchors": int(sum(valid_by_tx.values())),
        "active_hinges": int(sum(hinge_by_tx.values())),
        "valid_anchors_by_tx": valid_by_tx,
        "active_hinges_by_tx": hinge_by_tx,
        "clean_margin_detached": True,
        "class_weight_detached": True,
        "clean_gate_raw_cosine": True,
        "tx_equal_aggregation": True,
    }


def add_pamr_to_loss(
    base_loss: torch.Tensor,
    pamr: Optional[torch.Tensor],
    config: Optional[PAMRConfig],
) -> torch.Tensor:
    """Add the sole frozen G-arm term without changing the C-arm tensor."""

    if config is None or not bool(config.enabled):
        return base_loss
    if pamr is None:
        raise PAMRRuntimeError("Enabled P1-PAMR requires a paired angular-margin loss")
    return base_loss + float(config.loss_weight) * pamr


def pamr_unscaled_gradient(
    pamr: torch.Tensor,
    leo_feature: torch.Tensor,
    *,
    loss_weight: float,
) -> Optional[torch.Tensor]:
    """Obtain the raw, pre-GradScaler P1-only feature gradient."""

    if not torch.is_tensor(pamr) or pamr.ndim != 0:
        raise PAMRRuntimeError("P1-PAMR gradient audit requires a scalar PAMR loss")
    if not torch.is_tensor(leo_feature):
        raise PAMRRuntimeError("P1-PAMR gradient audit requires the LEO feature tensor")
    gradient = torch.autograd.grad(
        float(loss_weight) * pamr,
        leo_feature,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    return gradient


def pamr_gradient_status(gradient: Optional[torch.Tensor]) -> Dict[str, bool]:
    """Classify a raw PAMR feature gradient; None/non-finite fail closed."""

    if gradient is None:
        raise PAMRRuntimeError("P1-PAMR has no LEO feature gradient")
    finite = bool(torch.isfinite(gradient.detach()).all().item())
    nonzero = finite and bool(torch.count_nonzero(gradient.detach()).item() > 0)
    return {
        "finite": finite,
        "nonzero": nonzero,
        "zero": bool(finite and not nonzero),
        "nonfinite": bool(not finite),
    }


def require_finite_pamr_gradient(status: Mapping[str, bool]) -> None:
    if bool(status.get("nonfinite", False)):
        raise PAMRRuntimeError("P1-PAMR LEO feature gradient is non-finite")
    if not bool(status.get("finite", False)):
        raise PAMRRuntimeError("P1-PAMR LEO feature gradient is not finite")


def _pamr_failure_fingerprint(error: BaseException) -> str:
    """Return a stable, data-free code for an audited PAMR failure."""

    message = str(error)
    if "no LEO feature gradient" in message:
        return "PAMR_LEO_GRADIENT_MISSING"
    if "LEO feature gradient is non-finite" in message:
        return "PAMR_LEO_GRADIENT_NONFINITE"
    if "shared encoder" in message or "shared-gradient" in message:
        return "PAMR_SHARED_GRADIENT_FAILURE"
    if "binding" in message or "feat_joint" in message or "class-order" in message:
        return "PAMR_BINDING_FAILURE"
    return "PAMR_RUNTIME_FAILURE"


def write_pamr_failure_receipt(
    output_dir: str | Path,
    *,
    candidate_id: str,
    run_id: str,
    receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Path:
    """Atomically persist the minimal fail-closed record for one PAMR run.

    The receipt intentionally contains only run identifiers, the aggregate PAMR
    receipt and a stable error fingerprint.  It never serializes samples,
    tensors, physical IDs, receiver/domain metadata or error text.
    """

    target_dir = Path(output_dir)
    if not target_dir.is_dir():
        raise PAMRRuntimeError(
            "P1-PAMR failure receipt requires an existing candidate output directory"
        )
    target = target_dir / "pamr_failure_receipt.json"
    payload = {
        "schema": "cvs.phase1.pamr_failure_receipt.v1",
        "status": "FAIL_CLOSED",
        "candidate_id": str(candidate_id or ""),
        "run_id": str(run_id or ""),
        "failure_stage": str(failure_stage),
        "error_type": type(error).__name__,
        "error_fingerprint": _pamr_failure_fingerprint(error),
        "pamr_receipt": dict(receipt),
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd, temporary_name = mkstemp(
        prefix=".pamr_failure_receipt.",
        suffix=".tmp",
        dir=str(target_dir),
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


def resolve_pamr_classifier_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    """Resolve exactly the existing raw-cosine classifier-head weight path."""

    raw_model = getattr(model, "_orig_mod", model)
    try:
        weight = raw_model.id_backbone.cls_head.head.weight
    except AttributeError as exc:
        raise PAMRRuntimeError(
            "P1-PAMR requires model.id_backbone.cls_head.head.weight"
        ) from exc
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise PAMRRuntimeError("P1-PAMR classifier head weight must be a rank-2 Parameter")
    return weight


def validate_pamr_binding(
    *,
    model: torch.nn.Module,
    out_clean: Mapping[str, Any],
    out_leo: Mapping[str, Any],
    tx_labels: torch.Tensor,
) -> torch.nn.Parameter:
    """Fail closed unless the actual feat_joint/head/class-order path matches."""

    if str(out_clean.get("z_id_key", "")) != "feat_joint" or str(
        out_leo.get("z_id_key", "")
    ) != "feat_joint":
        raise PAMRRuntimeError("P1-PAMR requires z_id=feat_joint on both clean and LEO paths")
    z_clean = out_clean.get("z_id")
    z_leo = out_leo.get("z_id")
    logits = out_clean.get("tx_logits")
    if not torch.is_tensor(z_clean) or not torch.is_tensor(z_leo) or not torch.is_tensor(logits):
        raise PAMRRuntimeError("P1-PAMR binding requires z_id and tx_logits tensors")
    weight = resolve_pamr_classifier_weight(model)
    if int(weight.size(1)) != int(z_clean.size(1)) or int(z_clean.size(1)) != int(z_leo.size(1)):
        raise PAMRRuntimeError("P1-PAMR feat_joint/head dimension binding mismatch")
    if int(weight.size(0)) != int(logits.size(1)):
        raise PAMRRuntimeError("P1-PAMR classifier head row count and tx logit class order mismatch")
    labels = tx_labels.reshape(-1).long()
    if labels.numel() != z_clean.size(0) or labels.numel() != z_leo.size(0):
        raise PAMRRuntimeError("P1-PAMR paired z_id rows and labels are not aligned")
    if labels.numel() == 0 or int(labels.min().item()) < 0 or int(labels.max().item()) >= int(weight.size(0)):
        raise PAMRRuntimeError("P1-PAMR labels do not bind to classifier weight row order")
    return weight


def pamr_shared_encoder_parameters(model: torch.nn.Module) -> Sequence[torch.nn.Parameter]:
    """Return the trainable shared-ID path, excluding detached class weight."""

    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", None)
    if id_backbone is None:
        raise PAMRRuntimeError("P1-PAMR requires model.id_backbone for shared-gradient audit")
    parameters = [
        parameter
        for name, parameter in id_backbone.named_parameters()
        if parameter.requires_grad and name != "cls_head.head.weight"
    ]
    if not parameters:
        raise PAMRRuntimeError("P1-PAMR shared encoder parameter scope is empty")
    return parameters


def pamr_shared_gradient_relation(
    base_loss: torch.Tensor,
    pamr: torch.Tensor,
    shared_parameters: Iterable[torch.nn.Parameter],
    *,
    loss_weight: float,
) -> Dict[str, Optional[float]]:
    """Measure raw base/PAMR gradients on the same shared encoder parameters."""

    parameters = tuple(shared_parameters)
    if not parameters:
        raise PAMRRuntimeError("P1-PAMR shared-gradient audit received no parameters")
    base_grads = torch.autograd.grad(
        base_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    pamr_grads = torch.autograd.grad(
        float(loss_weight) * pamr,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    base_norm_sq = 0.0
    pamr_norm_sq = 0.0
    dot = 0.0
    active = 0
    for base_grad, pamr_grad in zip(base_grads, pamr_grads):
        for gradient in (base_grad, pamr_grad):
            if gradient is not None and not bool(torch.isfinite(gradient.detach()).all().item()):
                raise PAMRRuntimeError("P1-PAMR shared encoder gradient is non-finite")
        if base_grad is None and pamr_grad is None:
            continue
        active += 1
        if base_grad is not None:
            base_value = base_grad.detach().double()
            base_norm_sq += float(torch.sum(base_value * base_value).item())
        if pamr_grad is not None:
            pamr_value = pamr_grad.detach().double()
            pamr_norm_sq += float(torch.sum(pamr_value * pamr_value).item())
        if base_grad is not None and pamr_grad is not None:
            dot += float(torch.sum(base_grad.detach().double() * pamr_grad.detach().double()).item())
    if active == 0:
        raise PAMRRuntimeError("P1-PAMR shared encoder has no reachable base/PAMR gradient")
    base_norm = math.sqrt(base_norm_sq)
    pamr_norm = math.sqrt(pamr_norm_sq)
    if not math.isfinite(base_norm) or not math.isfinite(pamr_norm) or not math.isfinite(dot):
        raise PAMRRuntimeError("P1-PAMR shared gradient relation is non-finite")
    cosine: Optional[float]
    if base_norm > 0.0 and pamr_norm > 0.0:
        cosine = float(dot / (base_norm * pamr_norm))
    else:
        cosine = None
    return {
        "shared_parameter_count": float(active),
        "base_norm": float(base_norm),
        "pamr_norm": float(pamr_norm),
        "cosine": cosine,
        "norm_ratio": float(pamr_norm / (base_norm + _EPS)),
    }


def _add_counts(target: Dict[str, int], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise PAMRRuntimeError(f"P1-PAMR receipt count for TX {key!r} is invalid") from exc
        if count < 0:
            raise PAMRRuntimeError(f"P1-PAMR receipt count for TX {key!r} is negative")
        target[str(key)] = int(target.get(str(key), 0)) + count


def update_pamr_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    leo_grad_nonzero: Optional[bool] = None,
    leo_grad_zero: Optional[bool] = None,
    leo_grad_nonfinite: Optional[bool] = None,
) -> Dict[str, Any]:
    """Accumulate one P1-PAMR forward batch without data/raw tensors.

    Formal training records only coverage.  The optional raw-gradient outcome is
    retained for the one-epoch technical audit and preserves the previous public
    call form used by focused tests.
    """

    result = dict(receipt)
    expected = [int(value) for value in result.get("expected_tx_class_ids", [])]
    valid_by_tx = {str(key): int(value) for key, value in dict(batch_info.get("valid_anchors_by_tx", {})).items()}
    hinge_by_tx = {str(key): int(value) for key, value in dict(batch_info.get("active_hinges_by_tx", {})).items()}
    if set(valid_by_tx) != set(hinge_by_tx):
        raise PAMRRuntimeError("P1-PAMR valid-anchor and hinge TX keys must match")
    for key in valid_by_tx:
        if expected and int(key) not in expected:
            raise PAMRRuntimeError("P1-PAMR observed an out-of-contract TX class index")
    result["pamr_batches"] = int(result.get("pamr_batches", 0)) + 1
    result["pamr_rows"] = int(result.get("pamr_rows", 0)) + int(batch_info.get("rows", 0))
    result["pamr_valid_anchors"] = int(result.get("pamr_valid_anchors", 0)) + int(
        batch_info.get("valid_anchors", 0)
    )
    result["pamr_active_hinges"] = int(result.get("pamr_active_hinges", 0)) + int(
        batch_info.get("active_hinges", 0)
    )
    valid_total = dict(result.get("pamr_valid_anchors_by_tx", {}))
    hinge_total = dict(result.get("pamr_active_hinges_by_tx", {}))
    _add_counts(valid_total, valid_by_tx)
    _add_counts(hinge_total, hinge_by_tx)
    result["pamr_valid_anchors_by_tx"] = valid_total
    result["pamr_active_hinges_by_tx"] = hinge_total
    gradient_values = (leo_grad_nonzero, leo_grad_zero, leo_grad_nonfinite)
    if any(value is not None for value in gradient_values):
        if not all(value is not None for value in gradient_values):
            raise PAMRRuntimeError("P1-PAMR gradient receipt requires a complete status")
        result = update_pamr_gradient_receipt(
            result,
            leo_grad_nonzero=bool(leo_grad_nonzero),
            leo_grad_zero=bool(leo_grad_zero),
            leo_grad_nonfinite=bool(leo_grad_nonfinite),
        )
    return result


def update_pamr_gradient_receipt(
    receipt: Mapping[str, Any],
    *,
    leo_grad_nonzero: bool,
    leo_grad_zero: bool,
    leo_grad_nonfinite: bool,
) -> Dict[str, Any]:
    """Record one raw PAMR audit outcome without double-counting coverage."""

    outcomes = int(bool(leo_grad_nonzero)) + int(bool(leo_grad_zero)) + int(bool(leo_grad_nonfinite))
    if outcomes != 1:
        raise PAMRRuntimeError("P1-PAMR gradient receipt requires exactly one status")
    result = dict(receipt)
    result["pamr_gradient_audit_attempted_batches"] = int(
        result.get("pamr_gradient_audit_attempted_batches", 0)
    ) + 1
    result["pamr_grad_nonzero_batches"] = int(result.get("pamr_grad_nonzero_batches", 0)) + int(
        bool(leo_grad_nonzero)
    )
    result["pamr_grad_zero_batches"] = int(result.get("pamr_grad_zero_batches", 0)) + int(
        bool(leo_grad_zero)
    )
    result["pamr_grad_nonfinite_batches"] = int(result.get("pamr_grad_nonfinite_batches", 0)) + int(
        bool(leo_grad_nonfinite)
    )
    return result


def _accumulate_float_summary(result: Dict[str, Any], prefix: str, value: Optional[float]) -> None:
    if value is None:
        return
    if not math.isfinite(float(value)):
        raise PAMRRuntimeError(f"P1-PAMR {prefix} is non-finite")
    count_key = f"{prefix}_count"
    sum_key = f"{prefix}_sum"
    min_key = f"{prefix}_min"
    max_key = f"{prefix}_max"
    result[count_key] = int(result.get(count_key, 0)) + 1
    result[sum_key] = float(result.get(sum_key, 0.0)) + float(value)
    result[min_key] = float(value) if result.get(min_key) is None else min(float(result[min_key]), float(value))
    result[max_key] = float(value) if result.get(max_key) is None else max(float(result[max_key]), float(value))


def update_pamr_gradient_relation_receipt(
    receipt: Mapping[str, Any], relation: Mapping[str, Optional[float]]
) -> Dict[str, Any]:
    """Accumulate raw base/PAMR shared-encoder relation diagnostics."""

    result = dict(receipt)
    if float(relation.get("shared_parameter_count", 0.0) or 0.0) <= 0.0:
        raise PAMRRuntimeError("P1-PAMR shared-gradient relation has no active parameters")
    result["pamr_shared_gradient_relation_batches"] = int(
        result.get("pamr_shared_gradient_relation_batches", 0)
    ) + 1
    _accumulate_float_summary(result, "pamr_shared_gradient_cosine", relation.get("cosine"))
    _accumulate_float_summary(result, "pamr_shared_gradient_norm_ratio", relation.get("norm_ratio"))
    return result


def validate_pamr_terminal_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed unless all frozen PAMR health and TX-coverage evidence exists."""

    result = dict(receipt)
    if not bool(result.get("frozen_mode", False)):
        return result
    if not bool(result.get("enabled", False)):
        result["pamr_terminal_gradient_contract"] = "CONTROL_ARM_NOT_APPLICABLE"
        result["pamr_terminal_gradient_contract_passed"] = True
        return result
    expected = [str(int(value)) for value in result.get("expected_tx_class_ids", [])]
    if not expected:
        raise PAMRRuntimeError("P1-PAMR terminal receipt lacks expected source TX class indices")
    valid_by_tx = dict(result.get("pamr_valid_anchors_by_tx", {}))
    hinge_by_tx = dict(result.get("pamr_active_hinges_by_tx", {}))
    missing_valid = [key for key in expected if int(valid_by_tx.get(key, 0)) <= 0]
    missing_hinge = [key for key in expected if int(hinge_by_tx.get(key, 0)) <= 0]
    if missing_valid:
        raise PAMRRuntimeError(
            "P1-PAMR terminal receipt has zero valid clean-correct anchors for TX rows: "
            + ", ".join(missing_valid)
        )
    if missing_hinge:
        raise PAMRRuntimeError(
            "P1-PAMR terminal receipt has zero active hinge coverage for TX rows: "
            + ", ".join(missing_hinge)
        )
    if bool(result.get("audit_only", False)):
        nonzero = int(result.get("pamr_grad_nonzero_batches", 0))
        nonfinite = int(result.get("pamr_grad_nonfinite_batches", 0))
        if nonzero < 1:
            raise PAMRRuntimeError(
                "P1-PAMR audit terminal receipt requires at least one nonzero raw feature gradient"
            )
        if nonfinite != 0:
            raise PAMRRuntimeError(
                "P1-PAMR audit terminal receipt rejects non-finite raw feature gradients"
            )
        if not bool(result.get("pamr_gradient_audit_completed", False)):
            raise PAMRRuntimeError("P1-PAMR audit terminal receipt lacks a completed raw gradient audit")
        if int(result.get("pamr_shared_gradient_relation_batches", 0)) < 1:
            raise PAMRRuntimeError("P1-PAMR audit terminal receipt lacks shared encoder gradient relation")
        if int(result.get("pamr_shared_gradient_norm_ratio_count", 0)) < 1:
            raise PAMRRuntimeError("P1-PAMR audit terminal receipt lacks finite shared gradient norm ratio")
        result["pamr_terminal_gradient_contract"] = (
            "AUDIT_RAW_NONZERO_GRADIENT_AND_PER_TX_ANCHOR_HINGE_COVERAGE"
        )
    else:
        result["pamr_terminal_gradient_contract"] = "FORMAL_PER_TX_ANCHOR_HINGE_COVERAGE"
    result["pamr_terminal_gradient_contract_passed"] = True
    return result


def strict_pamr_warm_start(
    model: torch.nn.Module,
    checkpoint_model_state: Mapping[str, torch.Tensor],
    *,
    baseline_path: str,
    baseline_sha256: str,
    checkpoint_epoch: int,
    checkpoint_role: str,
) -> Dict[str, Any]:
    """Load only model weights, with strict key equality and no resume state."""

    path = str(baseline_path or "").strip()
    digest = str(baseline_sha256 or "").strip()
    if not path:
        raise PAMRConfigurationError("Frozen P1-PAMR warm-start requires a baseline path")
    if not digest:
        raise PAMRConfigurationError("Frozen P1-PAMR warm-start requires a baseline SHA256")
    if not isinstance(checkpoint_model_state, Mapping):
        raise PAMRConfigurationError("Frozen P1-PAMR baseline checkpoint has no model state mapping")
    raw_model = getattr(model, "_orig_mod", model)
    try:
        incompatible = raw_model.load_state_dict(dict(checkpoint_model_state), strict=True)
    except Exception as exc:
        raise PAMRConfigurationError(
            f"Frozen P1-PAMR strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise PAMRConfigurationError(
            "Frozen P1-PAMR strict baseline model-key mismatch: "
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
