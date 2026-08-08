"""Frozen source-only CCPC-LEO objective for Phase1 continuation runs.

The implementation deliberately receives only paired clean/LEO identity
features and transmitter labels.  Receiver, day, domain, proxy and held-role
metadata are not accepted by this module, so they cannot affect its positive
set or denominator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F


FROZEN_CCPC_LAMBDA = 0.02
FROZEN_CCPC_TEMPERATURE = 0.12


class CCPCLEOConfigurationError(ValueError):
    """Raised when a frozen CCPC-LEO continuation configuration drifts."""


class CCPCLEORuntimeError(RuntimeError):
    """Raised when a CCPC-LEO batch cannot prove its required pairing."""


@dataclass(frozen=True)
class CCPCLEOConfig:
    """Frozen settings consumed by the train-loop integration."""

    frozen_mode: bool
    enabled: bool
    loss_weight: float
    temperature: float


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError) as exc:
        raise CCPCLEOConfigurationError(f"{name} must be numeric") from exc


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > 1e-12:
        raise CCPCLEOConfigurationError(
            f"Frozen CCPC-LEO requires {name}={expected:.12g}, got {actual!r}"
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
        raise CCPCLEOConfigurationError(
            "Frozen CCPC-LEO forbids stacked routes: " + ", ".join(active)
        )


def validate_ccpc_leo_args(args: Any) -> CCPCLEOConfig:
    """Validate the C/G frozen continuation contract.

    Legacy training invocations retain their behaviour when
    ``phase1_ccpc_leo_frozen_mode`` and ``phase1_ccpc_leo_enabled`` are both
    false.  The control C arm sets frozen mode but leaves the CCPC loss off;
    the G arm sets frozen mode and enables the one fixed loss.
    """

    frozen_mode = _bool_arg(args, "phase1_ccpc_leo_frozen_mode", False)
    enabled = _bool_arg(args, "phase1_ccpc_leo_enabled", False)
    loss_weight = _float_arg(args, "lambda_ccpc_leo", 0.0)
    temperature = _float_arg(args, "ccpc_leo_temperature", FROZEN_CCPC_TEMPERATURE)
    if not frozen_mode and not enabled:
        return CCPCLEOConfig(
            frozen_mode=False,
            enabled=False,
            loss_weight=0.0,
            temperature=temperature,
        )
    if enabled and not frozen_mode:
        raise CCPCLEOConfigurationError(
            "--phase1_ccpc_leo_enabled requires --phase1_ccpc_leo_frozen_mode true"
        )
    _require_close("ccpc_leo_temperature", temperature, FROZEN_CCPC_TEMPERATURE)
    if enabled:
        _require_close("lambda_ccpc_leo", loss_weight, FROZEN_CCPC_LAMBDA)
    else:
        _require_close("lambda_ccpc_leo", loss_weight, 0.0)
    if bool(getattr(args, "from_scratch", True)):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO requires a GeoSat-C baseline checkpoint")
    if not str(getattr(args, "baseline_ckpt", "") or "").strip():
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO requires --baseline_ckpt")
    if bool(getattr(args, "freeze_backbone", False)):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO must train the shared encoder, not a head")
    if int(getattr(args, "epochs", 0)) != 40:
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO requires exactly --epochs 40")
    if str(getattr(args, "checkpoint_selection", "")) != "final_only":
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO requires --checkpoint_selection final_only")
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO remains source-validation-only")
    if not bool(getattr(args, "use_sat_consistency", False)):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO requires the GeoSat-C paired LEO path")
    _require_close("lambda_sat_cons", _float_arg(args, "lambda_sat_cons", 0.0), 0.10)
    _require_close("lambda_sat_cls", _float_arg(args, "lambda_sat_cls", 0.0), 0.0)
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO requires the non-concatenated paired LEO path")
    if bool(getattr(args, "use_unlabeled", False)):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO forbids unlabeled/domain-gated continuation")
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO forbids RX-conditioned batch construction")
    if bool(getattr(args, "use_aug", False)) or bool(getattr(args, "use_mixstyle", False)):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO requires paired clean/LEO rows without extra views")
    if bool(getattr(args, "reject_head", False)):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO forbids reject heads")
    _require_disabled(
        args,
        (
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
    return CCPCLEOConfig(
        frozen_mode=True,
        enabled=enabled,
        loss_weight=loss_weight,
        temperature=temperature,
    )


def ccpc_config_receipt(config: CCPCLEOConfig) -> Dict[str, Any]:
    """Create the immutable, data-free part of the CCPC run receipt."""

    return {
        "schema": "cvs.phase1.ccpc_leo_receipt.v1",
        "frozen_mode": bool(config.frozen_mode),
        "enabled": bool(config.enabled),
        "lambda": float(config.loss_weight),
        "temperature": float(config.temperature),
        "positive_rule": "same_tx_clean",
        "denominator_rule": "batch_all_tx_clean",
        "clean_detached": bool(config.enabled),
        "leo_only_gradient": bool(config.enabled),
        "uses_rx_labels": False,
        "uses_domain_labels": False,
        "uses_grl": False,
        "uses_mmd": False,
        "uses_coral": False,
        "uses_proxy_rows": False,
        "uses_held_rows": False,
        "uses_threshold": False,
        "uses_reject_head": False,
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
        "rows": 0,
        "classes": 0,
        "positive_pairs": 0,
        "leo_grad_nonzero": False,
        "proxy_rows": 0,
        "held_rows": 0,
    }


def strict_ccpc_warm_start(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    baseline_path: str,
    baseline_sha256: str,
    checkpoint_epoch: Any,
    checkpoint_role: Any,
) -> Dict[str, Any]:
    """Load only frozen baseline model weights and prove exact key compatibility.

    Optimizer, GradScaler and RNG state are deliberately never consumed here:
    CCPC C/G continuation starts with a newly-created AdamW/AMP state while
    sharing the same frozen GeoSat-C model-weight origin.
    """

    path = str(baseline_path or "").strip()
    digest = str(baseline_sha256 or "").strip()
    if not path:
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO warm-start requires a baseline path")
    if not digest:
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO warm-start requires a baseline SHA256")
    if not isinstance(state_dict, Mapping):
        raise CCPCLEOConfigurationError("Frozen CCPC-LEO baseline checkpoint has no model state mapping")
    target = getattr(model, "_orig_mod", model)
    try:
        incompatible = target.load_state_dict(dict(state_dict), strict=True)
    except Exception as exc:
        raise CCPCLEOConfigurationError(
            f"Frozen CCPC-LEO strict baseline model-key mismatch: {path}: {exc}"
        ) from exc
    missing_keys = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing_keys or unexpected_keys:
        raise CCPCLEOConfigurationError(
            "Frozen CCPC-LEO strict baseline model-key mismatch: "
            f"missing={missing_keys} unexpected={unexpected_keys}"
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


def _require_pair_shapes(z_leo: torch.Tensor, z_clean: torch.Tensor, tx_labels: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(z_leo) or not torch.is_tensor(z_clean) or not torch.is_tensor(tx_labels):
        raise CCPCLEORuntimeError("CCPC-LEO requires tensor z_leo, z_clean and tx_labels")
    if z_leo.ndim != 2 or z_clean.ndim != 2:
        raise CCPCLEORuntimeError("CCPC-LEO features must be rank-2 [rows, dim]")
    if tuple(z_leo.shape) != tuple(z_clean.shape):
        raise CCPCLEORuntimeError(
            "CCPC-LEO physical row binding failed: clean/LEO feature shapes differ"
        )
    labels = tx_labels.reshape(-1).to(device=z_leo.device, dtype=torch.long)
    if labels.numel() != z_leo.size(0):
        raise CCPCLEORuntimeError(
            "CCPC-LEO physical row binding failed: label rows do not match clean/LEO rows"
        )
    if z_leo.size(0) <= 0 or z_leo.size(1) <= 0:
        raise CCPCLEORuntimeError("CCPC-LEO requires a non-empty feature bank")
    if not bool(torch.isfinite(z_leo.detach()).all().item()) or not bool(torch.isfinite(z_clean.detach()).all().item()):
        raise CCPCLEORuntimeError("CCPC-LEO rejects non-finite clean or LEO features")
    if int(torch.unique(labels).numel()) < 2:
        raise CCPCLEORuntimeError("CCPC-LEO requires at least two TX classes per batch")
    return labels


def _contrastive_loss_from_positive_mask(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute supervised paired contrastive loss after validating positives."""

    if logits.ndim != 2 or tuple(logits.shape) != tuple(positive_mask.shape):
        raise CCPCLEORuntimeError("CCPC-LEO positive mask must match the contrastive-logit matrix")
    positives_per_anchor = positive_mask.sum(dim=1)
    if not bool((positives_per_anchor > 0).all().item()):
        raise CCPCLEORuntimeError("CCPC-LEO requires at least one same-TX clean positive per LEO anchor")
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_per_anchor = -(
        (log_probs * positive_mask.to(dtype=log_probs.dtype)).sum(dim=1)
        / positives_per_anchor
    )
    loss = loss_per_anchor.mean()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise CCPCLEORuntimeError("CCPC-LEO rejects a non-finite loss")
    return loss, positives_per_anchor


def ccpc_leo_loss(
    z_leo: torch.Tensor,
    z_clean: torch.Tensor,
    tx_labels: torch.Tensor,
    *,
    temperature: float = FROZEN_CCPC_TEMPERATURE,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return the asymmetric batch-local CCPC loss and auditable counts.

    ``z_clean`` is always detached before it becomes the clean bank.  The
    function has no receiver/domain argument; positive membership depends only
    on equality of transmitter labels, which makes it label-permutation
    equivariant.
    """

    actual_temperature = float(temperature)
    _require_close("ccpc_leo_temperature", actual_temperature, FROZEN_CCPC_TEMPERATURE)
    labels = _require_pair_shapes(z_leo, z_clean, tx_labels)
    if not bool(z_leo.requires_grad):
        raise CCPCLEORuntimeError("CCPC-LEO requires gradient-enabled LEO features")
    leo = F.normalize(z_leo.float(), dim=1, eps=1e-8)
    clean = F.normalize(z_clean.detach().float(), dim=1, eps=1e-8)
    logits = torch.matmul(leo, clean.transpose(0, 1)) / actual_temperature
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise CCPCLEORuntimeError("CCPC-LEO rejects non-finite contrastive logits")
    positive_mask = labels[:, None].eq(labels[None, :])
    loss, positives_per_anchor = _contrastive_loss_from_positive_mask(logits, positive_mask)
    info: Dict[str, Any] = {
        "rows": int(labels.numel()),
        "classes": int(torch.unique(labels).numel()),
        "positive_pairs": int(positive_mask.sum().detach().item()),
        "positive_anchors": int((positives_per_anchor > 0).sum().detach().item()),
        "clean_detached": True,
        "leo_grad_required": True,
        "leo_grad_nonzero": False,
        "temperature": actual_temperature,
    }
    return loss, info


def add_ccpc_to_loss(
    base_loss: torch.Tensor,
    ccpc_loss: torch.Tensor | None,
    config: CCPCLEOConfig,
) -> torch.Tensor:
    """Add CCPC only for G; return the exact original tensor for C/off mode."""

    if not bool(config.enabled):
        return base_loss
    if ccpc_loss is None:
        raise CCPCLEORuntimeError("Enabled CCPC-LEO requires a computed CCPC loss")
    if not bool(torch.isfinite(ccpc_loss.detach()).item()):
        raise CCPCLEORuntimeError("Enabled CCPC-LEO received a non-finite loss")
    return base_loss + float(config.loss_weight) * ccpc_loss


def update_ccpc_receipt(
    receipt: Mapping[str, Any],
    batch_info: Mapping[str, Any],
    *,
    leo_grad_nonzero: bool,
) -> Dict[str, Any]:
    """Accumulate one CCPC batch without introducing a persistent feature bank."""

    out = dict(receipt)
    if not bool(out.get("enabled", False)):
        return out
    out["rows"] = int(out.get("rows", 0)) + int(batch_info.get("rows", 0))
    out["positive_pairs"] = int(out.get("positive_pairs", 0)) + int(batch_info.get("positive_pairs", 0))
    out["classes"] = max(int(out.get("classes", 0)), int(batch_info.get("classes", 0)))
    out["clean_detached"] = bool(out.get("clean_detached", False)) and bool(batch_info.get("clean_detached", False))
    out["leo_grad_nonzero"] = bool(out.get("leo_grad_nonzero", False)) or bool(leo_grad_nonzero)
    out["ccpc_batches"] = int(out.get("ccpc_batches", 0)) + 1
    return out
