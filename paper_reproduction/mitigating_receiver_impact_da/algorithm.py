from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from paper_reproduction.mitigating_receiver_impact_da.losses import (
    adaptive_pseudo_labels,
    class_balance_weights,
    curriculum_thresholds,
    dv_kl_domain_alignment,
    gada_minimax_objective,
)
from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet


@dataclass
class PseudoLabelState:
    """Batch-to-batch CPL and class-weighting state from Algorithm 1."""

    num_classes: int
    pseudo_counts: torch.Tensor = field(init=False)
    predicted_counts: torch.Tensor = field(init=False)
    total_seen: int = 0

    def __post_init__(self) -> None:
        self.pseudo_counts = torch.zeros(int(self.num_classes), dtype=torch.float32)
        self.predicted_counts = torch.zeros(int(self.num_classes), dtype=torch.float32)

    def thresholds(self, *, base_tau: float, device: torch.device) -> torch.Tensor:
        return curriculum_thresholds(self.pseudo_counts.to(device), base_tau=base_tau)

    def class_weights(self, *, prior: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if self.total_seen <= 0:
            # Eq. (9) depends on previous target predictions; the first batch has no denominator.
            return torch.ones(int(self.num_classes), dtype=torch.float32, device=device)
        prior_on_device = None if prior is None else prior.to(device)
        return class_balance_weights(
            self.predicted_counts.to(device),
            total_seen=self.total_seen,
            prior=prior_on_device,
        )

    def update(self, predicted_labels: torch.Tensor, selected_labels: torch.Tensor) -> None:
        predicted = torch.bincount(predicted_labels.detach().cpu(), minlength=int(self.num_classes)).float()
        selected = torch.bincount(selected_labels.detach().cpu(), minlength=int(self.num_classes)).float()
        self.predicted_counts += predicted
        self.pseudo_counts += selected
        self.total_seen += int(predicted_labels.numel())


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> list[bool]:
    previous: list[bool] = []
    for parameter in module.parameters():
        previous.append(bool(parameter.requires_grad))
        parameter.requires_grad_(enabled)
    return previous


def _restore_requires_grad(module: torch.nn.Module, previous: list[bool]) -> None:
    for parameter, enabled in zip(module.parameters(), previous):
        parameter.requires_grad_(enabled)


def _estimate_outputs(model: ReceiverImpactGADNet, source_x: torch.Tensor, target_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = model.feature_extractor.training
    model.feature_extractor.eval()
    try:
        with torch.no_grad():
            source_features, _ = model.feature_extractor(source_x, return_activations=False)
            target_features, _ = model.feature_extractor(target_x, return_activations=False)
    finally:
        model.feature_extractor.train(was_training)
    return model.estimate_network(source_features.detach()), model.estimate_network(target_features.detach())


def gada_batch_step(
    model: ReceiverImpactGADNet,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
    *,
    target_y_audit: torch.Tensor | None = None,
    state: PseudoLabelState,
    optimizer_t: Any,
    optimizer_ec: Any,
    estimate_steps: int = 7,
    base_tau: float = 0.7,
    mu: float = 0.5,
    kl_weight: float = 0.005,
    class_prior: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int]:
    """Run one Algorithm 1 batch: T ascent, pseudo-labeling, then E/C descent."""
    if estimate_steps <= 0:
        raise ValueError("estimate_steps must be positive")

    model.train()
    last_estimate_loss: torch.Tensor | None = None
    for _ in range(int(estimate_steps)):
        optimizer_t.zero_grad()
        source_estimates, target_estimates = _estimate_outputs(model, source_x, target_x)
        zeta = dv_kl_domain_alignment(source_estimates, target_estimates)
        estimate_loss = -zeta
        estimate_loss.backward()
        optimizer_t.step()
        last_estimate_loss = estimate_loss.detach()

    optimizer_ec.zero_grad()
    previous_t_grad = _set_requires_grad(model.estimate_network, False)
    try:
        source_outputs = model(source_x)
        target_outputs = model(target_x)
        target_probs = torch.softmax(target_outputs["tx_logits"].detach(), dim=1)
        target_confidence = target_probs.max(dim=1).values
        thresholds = state.thresholds(base_tau=base_tau, device=target_probs.device)
        pseudo_labels, target_mask = adaptive_pseudo_labels(target_probs, thresholds)
        class_weights = state.class_weights(prior=class_prior, device=target_probs.device)
        terms = gada_minimax_objective(
            source_outputs,
            target_outputs,
            source_labels=source_y,
            target_pseudo_labels=pseudo_labels,
            target_mask=target_mask,
            class_weights=class_weights,
            mu=mu,
            kl_weight=kl_weight,
        )
        terms["loss"].backward()
        optimizer_ec.step()
    finally:
        _restore_requires_grad(model.estimate_network, previous_t_grad)

    state.update(pseudo_labels, pseudo_labels[target_mask])
    result = {
        "loss": terms["loss"].detach(),
        "loss_weighted_ce": terms["loss_weighted_ce"].detach(),
        "loss_source": terms["loss_source"].detach(),
        "loss_target": terms["loss_target"].detach(),
        "loss_kl": terms["loss_kl"].detach(),
        "target_selected": target_mask.sum().detach(),
        "target_conf_mean": target_confidence.mean().detach(),
        "class_weight_min": class_weights.min().detach(),
        "class_weight_max": class_weights.max().detach(),
        "estimate_steps": int(estimate_steps),
        "estimate_loss": torch.tensor(0.0) if last_estimate_loss is None else last_estimate_loss,
    }
    if target_y_audit is not None:
        audit_labels = target_y_audit.to(pseudo_labels.device).long()
        if audit_labels.shape[0] != pseudo_labels.shape[0]:
            raise ValueError("target_y_audit must have one label per target sample")
        selected_correct = ((pseudo_labels == audit_labels) & target_mask).sum()
        pred_correct = (pseudo_labels == audit_labels).sum()
        result.update(
            {
                "target_selected_correct": selected_correct.detach(),
                "target_audit_total": torch.as_tensor(
                    int(audit_labels.numel()), dtype=torch.long, device=pseudo_labels.device
                ),
                "target_pred_correct": pred_correct.detach(),
            }
        )
    return result
