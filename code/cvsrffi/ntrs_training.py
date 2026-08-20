"""Phase1-safe training controls for ADVB02 NTRS.

The helpers in this module deliberately keep the first NTRS candidate narrow:
source-only state updates, the exact three-scenario LEO_WEAK family, bounded
identity correction, and no target-time adaptation or unknown rescue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

from training_controls import LEO_WEAK_SCENARIOS
from cvsrffi.losses import (
    ntrs_class_attraction_loss,
    ntrs_class_conditional_alignment_loss,
    ntrs_conditional_decorrelation_loss,
    ntrs_correctability_loss,
    ntrs_margin_preservation_loss,
    ntrs_relation_distillation_loss,
    ntrs_score_stability_loss,
    ntrs_shared_receiver_offset_loss,
)


@dataclass(frozen=True)
class NTRSTrainingStage:
    name: str
    core_lr_scale: float
    ntrs_lr_scale: float
    nuisance_scale: float
    geometry_scale: float
    safety_scale: float


def ntrs_training_stage(epoch: int, *, variant: str = "v1") -> NTRSTrainingStage:
    """Map an epoch to the historical v1 or recovery-v2 schedule."""

    epoch = max(1, int(epoch))
    variant = str(variant or "v1").lower().strip()
    if variant in {"v3_adapter", "v4_operator"}:
        return NTRSTrainingStage("ADAPTER", 0.0, 1.0, 0.0, 1.0, 0.0)
    if variant == "v2_min":
        if epoch <= 90:
            return NTRSTrainingStage("V2-S0", 1.0, 0.0, 0.0, 0.0, 0.0)
        if epoch <= 130:
            ramp = float(epoch - 90) / 40.0
            return NTRSTrainingStage("V2-RAMP", 1.0, 1.0, 0.0, ramp, 0.0)
        return NTRSTrainingStage("V2-FULL", 1.0, 1.0, 0.0, 1.0, 0.0)
    if variant != "v1":
        raise ValueError(f"unsupported NTRS variant: {variant}")
    if epoch <= 16:
        return NTRSTrainingStage("S1", 1.0, 0.0, 0.0, 0.0, 0.0)
    if epoch <= 40:
        return NTRSTrainingStage("S2-a", 0.20, 1.0, 1.0, 0.0, 0.0)
    if epoch <= 68:
        return NTRSTrainingStage("S2-b", 0.20, 1.0, 1.0, 1.0, 0.0)
    return NTRSTrainingStage("S3", 0.10, 0.50, 1.0, 1.0, 1.0)


def ntrs_stage_code(stage: NTRSTrainingStage) -> int:
    codes = {"S1": 1, "S2-a": 2, "S2-b": 3, "S3": 4, "V2-S0": 5, "V2-RAMP": 6, "V2-FULL": 7, "ADAPTER": 8}
    if stage.name not in codes:
        raise ValueError(f"unsupported NTRS stage name: {stage.name}")
    return codes[stage.name]


def ntrs_relative_correction_loss(
    anchor: torch.Tensor,
    correction: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if anchor.shape != correction.shape or anchor.dim() != 2:
        raise ValueError("NTRS relative correction requires aligned [B, D] tensors")
    numerator = correction.float().square().sum(dim=1)
    denominator = anchor.detach().float().square().sum(dim=1).add(float(eps))
    return (numerator / denominator).mean()


def is_ntrs_parameter_name(name: str) -> bool:
    return any(part.startswith("ntrs_") for part in str(name).split("."))


def set_ntrs_optimizer_learning_rates(
    optimizer: Any,
    *,
    epoch: int,
    base_lr: float,
    variant: str = "v1",
    core_lr_mode: str = "v1",
    core_lr_ratio: float = 0.02,
) -> dict[str, float]:
    """Apply the frozen backbone/robustifier rates and return their values."""

    if float(base_lr) <= 0.0:
        raise ValueError("NTRS base_lr must be positive")
    stage = ntrs_training_stage(epoch, variant=variant)
    core_lr_mode = str(core_lr_mode or "v1").lower().strip()
    if core_lr_mode not in {"v1", "baseline", "adapter_joint"}:
        raise ValueError(f"unsupported NTRS core_lr_mode: {core_lr_mode}")
    if core_lr_mode == "v1":
        core_scale = float(ntrs_training_stage(epoch, variant="v1").core_lr_scale)
    elif core_lr_mode == "adapter_joint":
        core_scale = float(core_lr_ratio)
        if core_scale < 0.01 or core_scale > 0.05:
            raise ValueError("NTRS adapter joint core LR ratio must be in [0.01, 0.05]")
    else:
        core_scale = 1.0
    rates = {
        "core": float(base_lr) * core_scale,
        "ntrs": float(base_lr) * float(stage.ntrs_lr_scale),
    }
    for group in optimizer.param_groups:
        name = str(group.get("group_name", "core"))
        group["lr"] = rates["ntrs"] if name == "ntrs" else rates["core"]
    return rates


def ntrs_source_update_mask(
    *,
    batch_size: int,
    clean_count: int,
    concat_expanded: bool,
    device: Any = None,
) -> torch.Tensor:
    """Select source-clean rows for mutable slow/support state updates."""

    batch_size = max(0, int(batch_size))
    mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    if bool(concat_expanded):
        clean_count = int(clean_count)
        if clean_count <= 0 or batch_size < 2 * clean_count:
            raise ValueError("NTRS concat source mask requires aligned clean/satellite pairs")
        mask[clean_count:] = False
    return mask


def _connected_zero(output: Mapping[str, Any]) -> torch.Tensor:
    for key in ("tx_logits", "z_id", "ntrs_z_anchor"):
        value = output.get(key)
        if torch.is_tensor(value):
            return value.sum() * 0.0
    return torch.tensor(0.0)


def _factor_cross_entropy(
    outputs: list[Mapping[str, Any]],
    labels: list[Optional[torch.Tensor]],
    key: str,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits_parts = []
    label_parts = []
    for output, target in zip(outputs, labels):
        logits = output.get(key)
        if not torch.is_tensor(logits) or not torch.is_tensor(target):
            continue
        target = target.to(device=logits.device).view(-1).long()
        if logits.dim() != 2 or int(target.numel()) != int(logits.size(0)):
            raise ValueError(f"NTRS factor labels for {key} must align with logits")
        valid = (target >= 0) & (target < int(logits.size(1)))
        if bool(valid.any()):
            logits_parts.append(logits[valid].float())
            label_parts.append(target[valid])
    if not logits_parts:
        return zero, {"valid_count": 0.0, "accuracy": 0.0}
    logits = torch.cat(logits_parts, dim=0)
    target = torch.cat(label_parts, dim=0)
    loss = F.cross_entropy(logits, target)
    accuracy = (logits.detach().argmax(dim=1) == target).float().mean()
    return loss, {
        "valid_count": float(target.numel()),
        "accuracy": float(accuracy.item()),
    }


def _concatenate_output_tensor(
    outputs: list[Mapping[str, Any]],
    key: str,
) -> Optional[torch.Tensor]:
    parts = [output.get(key) for output in outputs]
    if not parts or not all(torch.is_tensor(value) for value in parts):
        return None
    return torch.cat(parts, dim=0)


def _concatenate_optional_labels(
    values: list[Optional[torch.Tensor]],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not values or not all(torch.is_tensor(value) for value in values):
        return None
    return torch.cat([value.to(device=device).view(-1).long() for value in values], dim=0)


def compute_ntrs_loss_bundle(
    clean_output: Mapping[str, Any],
    satellite_output: Optional[Mapping[str, Any]],
    *,
    clean_labels: torch.Tensor,
    satellite_labels: Optional[torch.Tensor],
    clean_receivers: Optional[torch.Tensor],
    satellite_receivers: Optional[torch.Tensor],
    clean_days: Optional[torch.Tensor],
    satellite_days: Optional[torch.Tensor],
    clean_channels: Optional[torch.Tensor],
    satellite_channels: Optional[torch.Tensor],
    prototypes: torch.Tensor,
    margin_epsilon: float,
    correctability_epsilon: float,
    energy_threshold: float,
    class_attraction_max_cosine: float,
    variant: str = "v1",
) -> dict[str, Any]:
    """Compute the four report loss groups from paired source views.

    The function is deliberately weight-free. The caller applies the frozen
    S1/S2-a/S2-b/S3 stage scales and registered coefficients exactly once.
    """

    if not torch.is_tensor(clean_labels) or not torch.is_tensor(prototypes):
        raise ValueError("NTRS loss bundle requires source labels and raw prototypes")
    variant = str(variant or "v1").lower().strip()
    outputs = [clean_output]
    label_values: list[Optional[torch.Tensor]] = [clean_labels]
    receiver_values = [clean_receivers]
    day_values = [clean_days]
    channel_values = [clean_channels]
    if satellite_output is not None:
        if not torch.is_tensor(satellite_labels):
            raise ValueError("NTRS paired satellite output requires satellite labels")
        outputs.append(satellite_output)
        label_values.append(satellite_labels)
        receiver_values.append(satellite_receivers)
        day_values.append(satellite_days)
        channel_values.append(satellite_channels)

    zero = _connected_zero(clean_output)
    losses: dict[str, torch.Tensor] = {}
    info: dict[str, Any] = {}
    if variant in {"v3_adapter", "v4_operator"} and satellite_output is not None:
        losses["robust_ce"], info["robust_ce"] = _factor_cross_entropy(
            [satellite_output], [satellite_labels], "ntrs_robust_logits", zero
        )
    else:
        losses["robust_ce"], info["robust_ce"] = _factor_cross_entropy(
            outputs, label_values, "ntrs_robust_logits", zero
        )
    losses["sat_kl"] = zero
    losses["margin"] = zero
    losses["relation"] = zero
    losses["class_conditional"] = zero
    losses["clean_zero"] = zero
    losses["satellite_relative"] = zero
    losses["q_distill"] = zero
    losses["pair_shift"] = zero
    losses["pair_cosine"] = zero
    losses["harm"] = zero
    losses["rescue"] = zero
    losses["clean_tail"] = zero
    if satellite_output is not None:
        clean_anchor = clean_output.get("ntrs_z_anchor")
        satellite_robust = satellite_output.get("ntrs_z_rob")
        clean_logits = clean_output.get("ntrs_raw_logits")
        satellite_logits = satellite_output.get("ntrs_robust_logits")
        if torch.is_tensor(clean_logits) and torch.is_tensor(satellite_logits):
            if clean_logits.shape != satellite_logits.shape:
                raise ValueError("NTRS paired KL logits must share shape")
            losses["sat_kl"] = F.kl_div(
                F.log_softmax(satellite_logits.float(), dim=1),
                clean_logits.detach().float().softmax(dim=1),
                reduction="batchmean",
            )
        losses["margin"], info["margin"] = ntrs_margin_preservation_loss(
            clean_anchor,
            satellite_robust,
            satellite_labels,
            prototypes.detach(),
            epsilon=float(margin_epsilon),
        )
        losses["relation"], info["relation"] = ntrs_relation_distillation_loss(
            clean_anchor,
            satellite_robust,
        )
        losses["class_conditional"], info["class_conditional"] = (
            ntrs_class_conditional_alignment_loss(
                clean_anchor,
                satellite_robust,
                satellite_labels,
            )
        )

    losses["receiver"], info["receiver"] = _factor_cross_entropy(
        outputs, receiver_values, "ntrs_receiver_logits", zero
    )
    losses["day"], info["day"] = _factor_cross_entropy(
        outputs, day_values, "ntrs_day_logits", zero
    )
    losses["channel"], info["channel"] = _factor_cross_entropy(
        outputs, channel_values, "ntrs_channel_logits", zero
    )
    losses["context_tx_adv"], info["context_tx_adv"] = _factor_cross_entropy(
        outputs, label_values, "ntrs_context_tx_adv_logits", zero
    )

    all_z_id = _concatenate_output_tensor(outputs, "z_id")
    all_z_dom = _concatenate_output_tensor(outputs, "z_dom")
    all_labels = _concatenate_optional_labels(
        label_values,
        device=clean_labels.device,
    )
    losses["conditional_decorrelation"], info["conditional_decorrelation"] = (
        ntrs_conditional_decorrelation_loss(all_z_id, all_z_dom, all_labels)
    )
    all_receivers = _concatenate_optional_labels(
        receiver_values,
        device=clean_labels.device,
    )
    losses["shared_receiver"], info["shared_receiver"] = ntrs_shared_receiver_offset_loss(
        all_z_id,
        all_labels,
        all_receivers,
    )

    anchor = _concatenate_output_tensor(outputs, "ntrs_z_anchor")
    robust = _concatenate_output_tensor(outputs, "ntrs_z_rob")
    alpha = _concatenate_output_tensor(outputs, "ntrs_alpha")
    residual = _concatenate_output_tensor(outputs, "ntrs_subspace_residual")
    correction = _concatenate_output_tensor(outputs, "ntrs_correction")
    candidate_correction = _concatenate_output_tensor(outputs, "ntrs_correction")
    if variant in {"v3_adapter", "v4_operator"}:
        clean_correction = clean_output.get("ntrs_correction")
        satellite_anchor = satellite_output.get("ntrs_z_anchor") if satellite_output is not None else None
        satellite_correction = satellite_output.get("ntrs_correction") if satellite_output is not None else None
        if torch.is_tensor(clean_correction):
            losses["clean_zero"] = clean_correction.float().square().sum(dim=1).mean()
        if torch.is_tensor(satellite_anchor) and torch.is_tensor(satellite_correction):
            losses["satellite_relative"] = ntrs_relative_correction_loss(
                satellite_anchor, satellite_correction
            )
        losses["minimum_correction"] = losses["satellite_relative"]
        if variant == "v4_operator" and satellite_output is not None:
            clean_anchor_v4 = clean_output.get("ntrs_z_anchor")
            satellite_anchor_v4 = satellite_output.get("ntrs_z_anchor")
            satellite_correction_v4 = satellite_output.get("ntrs_correction")
            satellite_robust_v4 = satellite_output.get("ntrs_z_rob")
            if all(
                torch.is_tensor(value)
                for value in (
                    clean_anchor_v4,
                    satellite_anchor_v4,
                    satellite_correction_v4,
                    satellite_robust_v4,
                )
            ):
                if clean_anchor_v4.shape != satellite_anchor_v4.shape:
                    raise ValueError("NTRS V4 paired anchors must share shape")
                target_shift = (satellite_anchor_v4 - clean_anchor_v4).detach()
                losses["pair_shift"] = F.smooth_l1_loss(
                    satellite_correction_v4.float(), target_shift.float()
                )
                losses["pair_cosine"] = (
                    1.0
                    - F.cosine_similarity(
                        satellite_robust_v4.float(),
                        clean_anchor_v4.detach().float(),
                        dim=1,
                        eps=1e-6,
                    )
                ).mean()

            q_iq = satellite_output.get("ntrs_q_iq")
            q_meta = satellite_output.get("ntrs_q_meta")
            metadata_valid = satellite_output.get("ntrs_metadata_valid")
            if torch.is_tensor(q_iq) and torch.is_tensor(q_meta):
                if q_iq.shape != q_meta.shape:
                    raise ValueError("NTRS V4 q teacher and student must share shape")
                valid = (
                    metadata_valid.to(device=q_iq.device).view(-1).bool()
                    if torch.is_tensor(metadata_valid)
                    else torch.ones(q_iq.size(0), dtype=torch.bool, device=q_iq.device)
                )
                if int(valid.numel()) != int(q_iq.size(0)):
                    raise ValueError("NTRS V4 metadata validity must align with q")
                if bool(valid.any()):
                    losses["q_distill"] = F.mse_loss(
                        q_iq[valid].float(), q_meta[valid].detach().float()
                    )

            clean_relative = None
            if torch.is_tensor(clean_anchor_v4) and torch.is_tensor(clean_correction):
                clean_relative = clean_correction.float().norm(dim=1) / clean_anchor_v4.detach().float().norm(dim=1).clamp_min(1e-6)
                losses["clean_tail"] = F.relu(clean_relative - 0.002).mean()

            raw_sat_logits = satellite_output.get("ntrs_raw_logits")
            robust_sat_logits = satellite_output.get("ntrs_robust_logits")
            clean_raw_logits = clean_output.get("ntrs_raw_logits")
            if all(torch.is_tensor(value) for value in (raw_sat_logits, robust_sat_logits, clean_raw_logits)):
                labels_v4 = satellite_labels.to(device=raw_sat_logits.device).view(-1).long()
                if int(labels_v4.numel()) != int(raw_sat_logits.size(0)):
                    raise ValueError("NTRS V4 labels must align with paired logits")

                def _true_margin(logits: torch.Tensor) -> torch.Tensor:
                    values = logits.float()
                    true_value = values.gather(1, labels_v4[:, None]).squeeze(1)
                    masked = values.clone()
                    masked.scatter_(1, labels_v4[:, None], float("-inf"))
                    return true_value - masked.max(dim=1).values

                raw_margin_v4 = _true_margin(raw_sat_logits)
                robust_margin_v4 = _true_margin(robust_sat_logits)
                clean_margin_v4 = _true_margin(clean_raw_logits)
                protect = raw_margin_v4.detach() > 0.0
                rescue = (clean_margin_v4.detach() > 0.0) & (raw_margin_v4.detach() < 0.0)
                if bool(protect.any()):
                    losses["harm"] = F.relu(
                        raw_margin_v4.detach()[protect]
                        - robust_margin_v4[protect]
                        - float(margin_epsilon)
                    ).mean()
                if bool(rescue.any()):
                    losses["rescue"] = F.softplus(
                        float(margin_epsilon) - robust_margin_v4[rescue]
                    ).mean()
    elif variant == "v2_min":
        losses["minimum_correction"] = (
            ntrs_relative_correction_loss(anchor, candidate_correction)
            if torch.is_tensor(anchor) and torch.is_tensor(candidate_correction)
            else zero
        )
    elif variant == "v1":
        losses["minimum_correction"] = (
            (robust - anchor).square().sum(dim=1).mean()
            if torch.is_tensor(anchor) and torch.is_tensor(robust)
            else zero
        )
    else:
        raise ValueError(f"unsupported NTRS loss variant: {variant}")
    losses["alpha"] = alpha.abs().mean() if torch.is_tensor(alpha) else zero
    losses["subspace"] = residual.square().mean() if torch.is_tensor(residual) else zero

    raw_logits = _concatenate_output_tensor(outputs, "ntrs_raw_logits")
    robust_logits = _concatenate_output_tensor(outputs, "ntrs_robust_logits")
    correctability = _concatenate_output_tensor(outputs, "ntrs_correctability")
    correction_energy = _concatenate_output_tensor(outputs, "ntrs_correction_energy")
    losses["correctability"], _target, info["correctability"] = ntrs_correctability_loss(
        raw_logits,
        robust_logits,
        all_labels,
        correctability,
        improvement_epsilon=float(correctability_epsilon),
    )
    losses["score_stability"], info["score_stability"] = ntrs_score_stability_loss(
        raw_logits,
        robust_logits,
        correction_energy=correction_energy,
        energy_threshold=float(energy_threshold),
    )
    losses["class_attraction"], info["class_attraction"] = ntrs_class_attraction_loss(
        anchor,
        correction,
        raw_logits=raw_logits,
        prototypes=prototypes.detach(),
        max_cosine=float(class_attraction_max_cosine),
    )
    return {"losses": losses, "info": info}


def _normalize_scenarios(scenarios: Any) -> list[str]:
    if isinstance(scenarios, str):
        values: Iterable[Any] = scenarios.replace(";", ",").split(",")
    else:
        values = list(scenarios or [])
    return [
        str(value).strip().lower().replace("-", "_")
        for value in values
        if str(value).strip()
    ]


def validate_ntrs_phase1_scenarios(scenarios: Any) -> None:
    """Require exactly one copy of every approved LEO_WEAK scenario."""

    normalized = _normalize_scenarios(scenarios)
    required = list(LEO_WEAK_SCENARIOS)
    if len(normalized) != len(required) or set(normalized) != set(required):
        raise ValueError(
            "ADVB02 NTRS Phase1 training requires exactly "
            f"{required}; got {normalized}"
        )


def validate_ntrs_phase1_config(args: Any) -> None:
    """Reject settings outside the frozen source-only NTRS first version."""

    if not bool(getattr(args, "use_ntrs", False)):
        return
    if bool(getattr(args, "use_crra", False)):
        raise ValueError("ADVB02 CRRA and NTRS are independent candidates and cannot be enabled together")
    if bool(getattr(args, "ntrs_target_adapter", False)):
        raise ValueError("NTRS target adapter is not allowed in the Phase1 source-only training path")
    if bool(getattr(args, "ntrs_unknown_rescue", False)):
        raise ValueError("NTRS unknown rescue must remain disabled in the first Phase1 candidate")
    if int(getattr(args, "ntrs_rank", 0)) <= 0:
        raise ValueError("ntrs_rank must be positive")
    variant = str(getattr(args, "ntrs_variant", "v1") or "v1").lower().strip()
    alpha_max = float(getattr(args, "ntrs_alpha_max", 0.20))
    alpha_cap = 0.05 if variant in {"v3_adapter", "v4_operator"} else 0.20
    if alpha_max < 0.0 or alpha_max > alpha_cap:
        raise ValueError(f"ntrs_alpha_max must remain in [0, {alpha_cap:.2f}] for {variant}")
    if variant in {"v3_adapter", "v4_operator"}:
        if not str(getattr(args, "baseline_ckpt", "")).strip() or bool(getattr(args, "from_scratch", True)):
            raise ValueError("NTRS adapter-only requires a mature baseline_ckpt and from_scratch=false")
        if bool(getattr(args, "ntrs_adapter_only", False)) and str(
            getattr(args, "ntrs_core_lr_mode", "baseline")
        ) == "adapter_joint":
            raise ValueError("adapter-only mode cannot enable joint core learning rate")
    if variant == "v4_operator":
        context_mode = str(getattr(args, "ntrs_context_mode", "normalized") or "normalized").lower().strip()
        operator_mode = str(getattr(args, "ntrs_operator_mode", "operator") or "operator").lower().strip()
        if context_mode not in {"normalized", "metadata_teacher", "constant", "shuffled"}:
            raise ValueError("unsupported NTRS V4 context mode")
        if operator_mode not in {"additive", "operator", "pca_additive"}:
            raise ValueError("unsupported NTRS V4 operator mode")
        if operator_mode == "pca_additive" and not str(
            getattr(args, "ntrs_pca_artifact", "")
        ).strip():
            raise ValueError("NTRS V4 PCA additive mode requires ntrs_pca_artifact")
        if float(getattr(args, "lambda_ntrs_harm", 0.0)) > 0.0 and float(
            getattr(args, "lambda_ntrs_harm", 0.0)
        ) <= float(getattr(args, "lambda_ntrs_rescue", 0.0)):
            raise ValueError("NTRS V4 harm weight must exceed rescue weight")
    if float(getattr(args, "ntrs_support_tau", 1.0)) <= 0.0:
        raise ValueError("ntrs_support_tau must be positive")
    if float(getattr(args, "ntrs_energy_threshold", 0.10)) <= 0.0:
        raise ValueError("ntrs_energy_threshold must be positive")
    slow_decay = float(getattr(args, "ntrs_slow_ema_decay", 0.95))
    if not 0.0 <= slow_decay < 1.0:
        raise ValueError("ntrs_slow_ema_decay must be in [0, 1)")
    for name, value in vars(args).items():
        if str(name).startswith("lambda_ntrs_") and float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")


__all__ = [
    "compute_ntrs_loss_bundle",
    "NTRSTrainingStage",
    "is_ntrs_parameter_name",
    "ntrs_source_update_mask",
    "ntrs_stage_code",
    "ntrs_relative_correction_loss",
    "ntrs_training_stage",
    "set_ntrs_optimizer_learning_rates",
    "validate_ntrs_phase1_config",
    "validate_ntrs_phase1_scenarios",
]
