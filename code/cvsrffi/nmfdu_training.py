from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NMFDUStage:
    index: int
    name: str
    gate_mode: str


class NMFDUStageController:
    """Apply the report-defined E1-80/E81-120/E121-200 schedule."""

    def __init__(self, boundaries: Sequence[int] = (80, 120, 200)) -> None:
        self.boundaries = tuple(int(value) for value in boundaries)
        if len(self.boundaries) != 3 or not (
            0 < self.boundaries[0] < self.boundaries[1] <= self.boundaries[2]
        ):
            raise ValueError("boundaries must be three increasing positive epochs")

    def stage_for_epoch(self, epoch: int) -> int:
        epoch = int(epoch)
        if epoch < 1 or epoch > self.boundaries[2]:
            raise ValueError("epoch is outside the registered NMFDU schedule")
        if epoch <= self.boundaries[0]:
            return 1
        if epoch <= self.boundaries[1]:
            return 2
        return 3

    @staticmethod
    def _resolve_backbone_and_gate(model):
        backbone = model.id_backbone if hasattr(model, "id_backbone") else model
        gate = getattr(backbone, "nmfdu_gate", None)
        if gate is None:
            raise ValueError("model does not contain an enabled NMFDU identity gate")
        return backbone, gate

    def apply(self, model, epoch: int) -> NMFDUStage:
        stage_index = self.stage_for_epoch(epoch)
        backbone, gate = self._resolve_backbone_and_gate(model)
        if getattr(gate, "ablation_mode", "full") == "equal":
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            for parameter in gate.sample_gate.parameters():
                parameter.requires_grad_(False)
            gate.evidence_state.freeze_discriminability(False)
            if hasattr(backbone, "set_nmfdu_stage"):
                backbone.set_nmfdu_stage(1)
            else:
                gate.set_stage(1)
            return NMFDUStage(1, "equal_capacity_control", "equal_non_null")
        if stage_index == 1:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            for parameter in gate.sample_gate.parameters():
                parameter.requires_grad_(False)
            gate.evidence_state.freeze_discriminability(False)
            stage = NMFDUStage(1, "branch_pretraining", "equal_non_null")
        elif stage_index == 2:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for parameter in gate.sample_gate.parameters():
                parameter.requires_grad_(True)
            gate.evidence_state.freeze_discriminability(True)
            stage = NMFDUStage(2, "gate_training", "physical_null_aware")
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            gate.evidence_state.freeze_discriminability(True)
            stage = NMFDUStage(3, "joint_finetuning", "physical_null_aware")
        if hasattr(backbone, "set_nmfdu_stage"):
            backbone.set_nmfdu_stage(stage_index)
        else:
            gate.set_stage(stage_index)
        return stage

    def parameter_groups(
        self,
        model,
        epoch: int,
        base_lr: float,
        gate_lr_scale: float = 0.5,
        joint_backbone_lr_scale: float = 0.1,
    ):
        stage = self.apply(model, epoch)
        _, gate = self._resolve_backbone_and_gate(model)
        if stage.index == 2:
            return [
                {
                    "name": "nmfdu_gate",
                    "params": [p for p in gate.sample_gate.parameters() if p.requires_grad],
                    "lr": float(base_lr) * float(gate_lr_scale),
                }
            ]
        gate_ids = {id(p) for p in gate.parameters()}
        gate_parameters = [
            p for p in gate.parameters() if p.requires_grad and id(p) in gate_ids
        ]
        backbone_parameters = [
            p for p in model.parameters() if p.requires_grad and id(p) not in gate_ids
        ]
        backbone_scale = 1.0 if stage.index == 1 else float(joint_backbone_lr_scale)
        return [
            {
                "name": "backbone",
                "params": backbone_parameters,
                "lr": float(base_lr) * backbone_scale,
            },
            {
                "name": "nmfdu",
                "params": gate_parameters,
                "lr": float(base_lr) * float(gate_lr_scale),
            },
        ]


def nmfdu_optimizer_groups(
    model,
    parameters: Iterable[torch.nn.Parameter],
    *,
    base_lr: float,
):
    """Keep every parameter in AdamW while exposing stable NMFDU LR roles."""

    _, gate = NMFDUStageController._resolve_backbone_and_gate(model)
    sample_ids = {id(parameter) for parameter in gate.sample_gate.parameters()}
    gate_ids = {id(parameter) for parameter in gate.parameters()}
    groups = {
        "nmfdu_backbone": [],
        "nmfdu_branch": [],
        "nmfdu_sample_gate": [],
    }
    seen = set()
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if id(parameter) in sample_ids:
            groups["nmfdu_sample_gate"].append(parameter)
        elif id(parameter) in gate_ids:
            groups["nmfdu_branch"].append(parameter)
        else:
            groups["nmfdu_backbone"].append(parameter)
    result = [
        {"params": values, "nmfdu_role": role, "lr": float(base_lr)}
        for role, values in groups.items()
        if values
    ]
    if not result:
        raise ValueError("optimizer has no NMFDU parameters")
    return result


def apply_nmfdu_optimizer_lr(
    optimizer,
    *,
    stage: int,
    base_lr: float,
    gate_lr_scale: float = 0.5,
    joint_backbone_lr_scale: float = 0.1,
) -> None:
    """Apply report-defined frozen/low-rate families without rebuilding AdamW."""

    stage = int(stage)
    if stage not in (1, 2, 3):
        raise ValueError("NMFDU stage must be 1, 2 or 3")
    if float(gate_lr_scale) <= 0.0 or float(joint_backbone_lr_scale) <= 0.0:
        raise ValueError("NMFDU learning-rate scales must be positive")
    for group in optimizer.param_groups:
        role = group.get("nmfdu_role")
        if role is None:
            continue
        if stage == 1:
            scale = 0.0 if role == "nmfdu_sample_gate" else 1.0
        elif stage == 2:
            scale = float(gate_lr_scale) if role == "nmfdu_sample_gate" else 0.0
        else:
            scale = (
                float(joint_backbone_lr_scale)
                if role == "nmfdu_backbone"
                else float(gate_lr_scale)
            )
        group["lr"] = float(base_lr) * scale


def branch_auxiliary_loss(
    branch_logits: Mapping[str, torch.Tensor], labels: torch.Tensor
) -> torch.Tensor:
    if not branch_logits:
        raise ValueError("branch_logits cannot be empty")
    return torch.stack(
        [F.cross_entropy(logits, labels) for logits in branch_logits.values()]
    ).mean()


def oracle_margin_distribution(
    branch_logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    uncertainty: torch.Tensor,
    temperature: float = 1.0,
    uncertainty_scale: float = 1.0,
) -> torch.Tensor:
    names = tuple(branch_logits)
    if uncertainty.shape != (labels.numel(), len(names)):
        raise ValueError("uncertainty must have shape [B,branch_count]")
    margins = []
    for name in names:
        logits = branch_logits[name]
        if logits.size(0) != labels.numel():
            raise ValueError("branch logit batch does not match labels")
        target = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        competitor = logits.masked_fill(
            F.one_hot(labels, num_classes=logits.size(1)).bool(),
            float("-inf"),
        ).max(dim=1).values
        margins.append(target - competitor)
    margin_tensor = torch.stack(margins, dim=-1)
    adjusted = margin_tensor - float(uncertainty_scale) * uncertainty.detach()
    return torch.softmax(adjusted.detach() / max(float(temperature), 1e-4), dim=-1)


def route_kl_loss(
    gate_weights: torch.Tensor, target_distribution: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if gate_weights.shape != target_distribution.shape:
        raise ValueError("gate weights and route target must share shape")
    normalized_gate = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target_distribution.detach().clamp_min(eps)
    return (target * (target.log() - normalized_gate.clamp_min(eps).log())).sum(dim=-1).mean()


def quality_weighted_mean(
    per_sample_loss: torch.Tensor, quality: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if per_sample_loss.shape != quality.shape:
        raise ValueError("loss and quality must share shape")
    quality = torch.nan_to_num(quality.float(), nan=0.0).clamp(0.0, 1.0)
    return (per_sample_loss * quality).mean()


def fused_pair_loss(
    clean_fused: torch.Tensor,
    leo_fused: torch.Tensor,
    *,
    clean_quality: torch.Tensor,
    leo_quality: torch.Tensor,
) -> torch.Tensor:
    if clean_fused.shape != leo_fused.shape:
        raise ValueError("paired fused embeddings must share shape")
    reliability = torch.minimum(clean_quality, leo_quality).detach()
    per_sample = (clean_fused - leo_fused).square().mean(dim=-1)
    return quality_weighted_mean(per_sample, reliability)


def reliable_branch_pair_loss(
    clean_embeddings: Mapping[str, torch.Tensor],
    leo_embeddings: Mapping[str, torch.Tensor],
    *,
    branch_names: Sequence[str],
    clean_reliability: torch.Tensor,
    leo_reliability: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    branch_names = tuple(branch_names)
    expected = (clean_reliability.size(0), len(branch_names))
    if clean_reliability.shape != expected or leo_reliability.shape != expected:
        raise ValueError("branch reliability must have shape [B,branch_count]")
    losses = []
    for name in branch_names:
        clean = clean_embeddings[name]
        leo = leo_embeddings[name]
        losses.append(1.0 - F.cosine_similarity(clean, leo, dim=-1, eps=eps))
    per_branch = torch.stack(losses, dim=-1)
    reliability = torch.minimum(clean_reliability, leo_reliability).detach().clamp(0.0, 1.0)
    return (per_branch * reliability).mean()


def nmfdu_labeled_objective(
    aux_id: Mapping[str, object],
    labels: torch.Tensor,
    *,
    stage: int,
    clean_count: int,
    lambda_branch_aux: float,
    lambda_route: float,
    lambda_phys: float,
    lambda_fused_pair: float,
    lambda_branch_pair: float,
    lambda_null_cal: float,
    lambda_balance: float,
    oracle_temperature: float = 0.5,
    ablation_mode: str = "full",
) -> Mapping[str, torch.Tensor]:
    """Compose only the NMFDU additions for an L_s clean/LEO batch."""

    required = (
        "nmfdu_branch_logits",
        "nmfdu_branch_embeddings",
        "physical_gate_diag",
        "feat_joint",
    )
    missing = [key for key in required if key not in aux_id]
    if missing:
        raise KeyError(f"missing NMFDU training outputs: {missing}")
    diagnostics = aux_id["physical_gate_diag"]["per_sample"]
    ablation_mode = str(ablation_mode or "full").lower().strip()
    if ablation_mode not in {
        "equal",
        "i_only",
        "i_d",
        "i_d_s",
        "physical_fixed",
        "physical_full",
        "full_no_null",
        "full",
    }:
        raise ValueError("unknown NMFDU ablation mode")
    branch_logits = aux_id["nmfdu_branch_logits"]
    branch_embeddings = aux_id["nmfdu_branch_embeddings"]
    zero = aux_id["feat_joint"].sum() * 0.0
    branch_aux = branch_auxiliary_loss(branch_logits, labels) if stage in (1, 3) else zero
    route = phys = null_cal = balance = zero
    if stage in (2, 3):
        uses_stability = ablation_mode not in {"i_only", "i_d"}
        uses_uncertainty = ablation_mode in {
            "physical_fixed",
            "physical_full",
            "full_no_null",
            "full",
        }
        route_uncertainty = (
            diagnostics["U"]
            if uses_uncertainty
            else torch.zeros_like(diagnostics["U"])
        )
        oracle = oracle_margin_distribution(
            branch_logits,
            labels,
            uncertainty=route_uncertainty,
            temperature=oracle_temperature,
        )
        route = route_kl_loss(diagnostics["weights"], oracle)
        physical_prior = torch.softmax(diagnostics["physical_logits"].detach(), dim=-1)
        phys = route_kl_loss(diagnostics["weights"], physical_prior)
        reliability = diagnostics["I"]
        if ablation_mode != "i_only":
            reliability = reliability * diagnostics["D"]
        if uses_stability:
            reliability = reliability * diagnostics["S"]
        if uses_uncertainty:
            reliability = reliability * (1.0 - diagnostics["U"])
        physical_quality = reliability.max(dim=-1).values.detach()
        null_target = 1.0 - physical_quality
        if ablation_mode != "full_no_null":
            # null_weight is already a softmax probability.  Keep the report's
            # probability-space calibration semantics, but evaluate BCE in
            # FP32 because probability BCE is intentionally unsafe under AMP.
            with torch.autocast(
                device_type=diagnostics["null_weight"].device.type,
                enabled=False,
            ):
                null_probability = diagnostics["null_weight"].float().clamp(
                    1e-6, 1.0 - 1e-6
                )
                null_cal = F.binary_cross_entropy(
                    null_probability,
                    null_target.float(),
                )
        usage = diagnostics["weights"].mean(dim=0)
        balance = (usage - usage.new_full(usage.shape, 1.0 / usage.numel())).square().mean()

    fused_pair = branch_pair = zero
    clean_count = int(clean_count)
    total = int(labels.numel())
    if stage == 3 and clean_count > 0 and total == 2 * clean_count:
        clean_slice = slice(0, clean_count)
        leo_slice = slice(clean_count, total)
        fused_pair = fused_pair_loss(
            aux_id["feat_joint"][clean_slice],
            aux_id["feat_joint"][leo_slice],
            clean_quality=diagnostics["q_sample"][clean_slice],
            leo_quality=diagnostics["q_sample"][leo_slice],
        )
        reliability = diagnostics["I"]
        if ablation_mode != "i_only":
            reliability = reliability * diagnostics["D"]
        if ablation_mode not in {"i_only", "i_d"}:
            reliability = reliability * diagnostics["S"]
        if ablation_mode in {
            "physical_fixed",
            "physical_full",
            "full_no_null",
            "full",
        }:
            reliability = reliability * (1.0 - diagnostics["U"])
        clean_embeddings = {
            name: value[clean_slice] for name, value in branch_embeddings.items()
        }
        leo_embeddings = {
            name: value[leo_slice] for name, value in branch_embeddings.items()
        }
        branch_pair = reliable_branch_pair_loss(
            clean_embeddings,
            leo_embeddings,
            branch_names=tuple(branch_embeddings),
            clean_reliability=reliability[clean_slice],
            leo_reliability=reliability[leo_slice],
        )

    total_loss = (
        float(lambda_branch_aux) * branch_aux
        + float(lambda_route) * route
        + float(lambda_phys) * phys
        + float(lambda_fused_pair) * fused_pair
        + float(lambda_branch_pair) * branch_pair
        + float(lambda_null_cal) * null_cal
        + float(lambda_balance) * balance
    )
    return {
        "total": total_loss,
        "branch_aux": branch_aux,
        "route": route,
        "phys": phys,
        "fused_pair": fused_pair,
        "branch_pair": branch_pair,
        "null_cal": null_cal,
        "balance": balance,
        "q_sample_mean": diagnostics["q_sample"].mean().detach(),
        "null_mean": diagnostics["null_weight"].mean().detach(),
        "entropy_mean": diagnostics["entropy"].mean().detach(),
    }
