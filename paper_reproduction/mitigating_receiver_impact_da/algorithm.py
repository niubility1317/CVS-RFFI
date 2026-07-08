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
    mine_kl_stabilized_objective,
)
from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet


def _snapshot_batch_norm_buffers(module: torch.nn.Module) -> list[tuple[torch.nn.Module, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
    snapshots: list[tuple[torch.nn.Module, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]] = []
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            running_mean = None if child.running_mean is None else child.running_mean.detach().clone()
            running_var = None if child.running_var is None else child.running_var.detach().clone()
            batches = None if child.num_batches_tracked is None else child.num_batches_tracked.detach().clone()
            snapshots.append((child, running_mean, running_var, batches))
    return snapshots


def _restore_batch_norm_buffers(
    snapshots: list[tuple[torch.nn.Module, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]
) -> None:
    for child, running_mean, running_var, batches in snapshots:
        if running_mean is not None and child.running_mean is not None:
            child.running_mean.copy_(running_mean)
        if running_var is not None and child.running_var is not None:
            child.running_var.copy_(running_var)
        if batches is not None and child.num_batches_tracked is not None:
            child.num_batches_tracked.copy_(batches)


@dataclass
class PseudoLabelState:
    """Batch-to-batch CPL and class-weighting state from Algorithm 1."""

    num_classes: int
    target_size: int | None = None
    target_batches: int | None = None
    pseudo_counts: torch.Tensor = field(init=False)
    predicted_counts: torch.Tensor = field(init=False)
    pseudo_labels_by_index: torch.Tensor | None = field(init=False, default=None)
    predicted_labels_by_index: torch.Tensor | None = field(init=False, default=None)
    total_seen: int = 0

    def __post_init__(self) -> None:
        self.pseudo_counts = torch.zeros(int(self.num_classes), dtype=torch.float32)
        self.predicted_counts = torch.zeros(int(self.num_classes), dtype=torch.float32)
        if self.target_size is not None:
            target_size = int(self.target_size)
            if target_size <= 0:
                raise ValueError("target_size must be positive when provided")
            self.pseudo_labels_by_index = torch.full((target_size,), -1, dtype=torch.long)
            self.predicted_labels_by_index = torch.full((target_size,), -1, dtype=torch.long)
        if self.target_batches is not None and int(self.target_batches) <= 0:
            raise ValueError("target_batches must be positive when provided")

    def reset_epoch(self) -> None:
        self.pseudo_counts.zero_()
        self.predicted_counts.zero_()
        self.total_seen = 0
        if self.pseudo_labels_by_index is not None:
            self.pseudo_labels_by_index.fill_(-1)
        if self.predicted_labels_by_index is not None:
            self.predicted_labels_by_index.fill_(-1)

    def thresholds(self, *, base_tau: float, device: torch.device) -> torch.Tensor:
        return curriculum_thresholds(self.pseudo_counts.to(device), base_tau=base_tau)

    def official_threshold_mask(
        self,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        *,
        base_tau: float,
    ) -> torch.Tensor:
        if self.pseudo_labels_by_index is None or self.target_batches is None:
            thresholds = self.thresholds(base_tau=base_tau, device=confidence.device)
            return confidence > thresholds[labels]
        pseudo_state = self.pseudo_labels_by_index
        selected_state = pseudo_state[pseudo_state >= 0]
        class_counts = torch.bincount(selected_state, minlength=int(self.num_classes)).float()
        negative_count = torch.as_tensor(int((pseudo_state < 0).sum().item()), dtype=torch.float32)
        max_counter = torch.cat([class_counts, negative_count.reshape(1)]).max()
        if float(max_counter.item()) < float(self.target_batches):
            max_class_count = class_counts.max()
            if float(max_class_count.item()) > 0.0:
                classwise_acc = (class_counts / max_class_count.clamp_min(1.0)).to(confidence.device)
                thresholds = float(base_tau) * (classwise_acc / (2.0 - classwise_acc).clamp_min(1e-8))
                return confidence > thresholds[labels]
        return confidence > float(base_tau)

    def _weights_from_counts(
        self,
        counts: torch.Tensor,
        *,
        total_seen: int,
        prior: torch.Tensor | None,
        device: torch.device,
        smoothing: float = 0.0,
        clip_min: float | None = None,
        clip_max: float | None = None,
        mean_normalize: bool = False,
    ) -> torch.Tensor:
        if total_seen <= 0:
            return torch.ones(int(self.num_classes), dtype=torch.float32, device=device)
        if smoothing == 0.0 and clip_min is None and clip_max is None and not mean_normalize:
            counts = counts.to(device)
            prior_on_device = None if prior is None else prior.to(device)
            if prior_on_device is None:
                prior_probs = torch.full_like(counts, 1.0 / max(counts.numel(), 1))
            else:
                prior_probs = prior_on_device.float()
                prior_probs = prior_probs / prior_probs.sum().clamp_min(1e-8)
            weights = torch.ones_like(counts)
            observed = counts > 0
            if observed.any():
                estimated = counts[observed] / counts.sum().clamp_min(1.0)
                weights[observed] = prior_probs[observed] / estimated.clamp_min(1e-8)
            return weights
        prior_on_device = None if prior is None else prior.to(device)
        return class_balance_weights(
            counts.to(device),
            total_seen=total_seen,
            prior=prior_on_device,
            smoothing=smoothing,
            clip_min=clip_min,
            clip_max=clip_max,
            mean_normalize=mean_normalize,
        )

    def class_weights(
        self,
        *,
        prior: torch.Tensor | None,
        device: torch.device,
        smoothing: float = 0.0,
        clip_min: float | None = None,
        clip_max: float | None = None,
        mean_normalize: bool = False,
    ) -> torch.Tensor:
        return self._weights_from_counts(
            self.predicted_counts,
            total_seen=self.total_seen,
            prior=prior,
            device=device,
            smoothing=smoothing,
            clip_min=clip_min,
            clip_max=clip_max,
            mean_normalize=mean_normalize,
        )

    def class_weights_after_predictions(
        self,
        predicted_labels: torch.Tensor,
        *,
        target_indices: torch.Tensor | None,
        prior: torch.Tensor | None,
        device: torch.device,
        smoothing: float = 0.0,
        clip_min: float | None = None,
        clip_max: float | None = None,
        mean_normalize: bool = False,
    ) -> torch.Tensor:
        if target_indices is not None and self.predicted_labels_by_index is not None:
            predicted_state = self.predicted_labels_by_index.clone()
            indices = target_indices.detach().cpu().long().flatten()
            if indices.numel() != predicted_labels.numel():
                raise ValueError("target_indices must have one index per target prediction")
            predicted_state[indices] = predicted_labels.detach().cpu().long().flatten()
            valid = predicted_state[predicted_state >= 0]
            counts = torch.bincount(valid, minlength=int(self.num_classes)).float()
            total_seen = int(valid.numel())
        else:
            current = torch.bincount(predicted_labels.detach().cpu(), minlength=int(self.num_classes)).float()
            counts = self.predicted_counts + current
            total_seen = int(self.total_seen + int(predicted_labels.numel()))
        return self._weights_from_counts(
            counts,
            total_seen=total_seen,
            prior=prior,
            device=device,
            smoothing=smoothing,
            clip_min=clip_min,
            clip_max=clip_max,
            mean_normalize=mean_normalize,
        )

    def update(
        self,
        predicted_labels: torch.Tensor,
        selected_labels: torch.Tensor,
        *,
        target_indices: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> None:
        if target_indices is not None and self.predicted_labels_by_index is not None and self.pseudo_labels_by_index is not None:
            indices = target_indices.detach().cpu().long().flatten()
            if indices.numel() != predicted_labels.numel():
                raise ValueError("target_indices must have one index per target prediction")
            self.predicted_labels_by_index[indices] = predicted_labels.detach().cpu().long().flatten()
            if target_mask is None:
                raise ValueError("target_mask is required when target_indices are provided")
            selected_indices = indices[target_mask.detach().cpu().bool().flatten()]
            if selected_indices.numel() != selected_labels.numel():
                raise ValueError("selected_labels must match selected target indices")
            self.pseudo_labels_by_index[selected_indices] = selected_labels.detach().cpu().long().flatten()
            predicted_state = self.predicted_labels_by_index[self.predicted_labels_by_index >= 0]
            pseudo_state = self.pseudo_labels_by_index[self.pseudo_labels_by_index >= 0]
            self.predicted_counts = torch.bincount(predicted_state, minlength=int(self.num_classes)).float()
            self.pseudo_counts = torch.bincount(pseudo_state, minlength=int(self.num_classes)).float()
            self.total_seen = int(predicted_state.numel())
            return
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


def _estimate_outputs(
    model: ReceiverImpactGADNet,
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    *,
    detach_features: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    bn_snapshot = _snapshot_batch_norm_buffers(model.feature_extractor) if detach_features else []
    try:
        if detach_features:
            with torch.no_grad():
                source_features, _ = model.feature_extractor(source_x, return_activations=False)
                target_features, _ = model.feature_extractor(target_x, return_activations=False)
            source_features = source_features.detach()
            target_features = target_features.detach()
        else:
            source_features, _ = model.feature_extractor(source_x, return_activations=False)
            target_features, _ = model.feature_extractor(target_x, return_activations=False)
    finally:
        if detach_features:
            _restore_batch_norm_buffers(bn_snapshot)
    return model.estimate_network(source_features), model.estimate_network(target_features)


def _select_pseudo_labels(
    target_logits: torch.Tensor,
    state: PseudoLabelState,
    *,
    base_tau: float,
    threshold_mode: str,
    score_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_score_mode = str(score_mode).strip().lower()
    if normalized_score_mode == "probability":
        scores = torch.softmax(target_logits.detach(), dim=1)
    elif normalized_score_mode == "logit":
        scores = target_logits.detach()
    else:
        raise ValueError("pseudo_score_mode must be one of: probability, logit")
    confidence, labels = scores.max(dim=1)
    normalized_threshold_mode = str(threshold_mode).strip().lower()
    if normalized_threshold_mode == "paper":
        thresholds = state.thresholds(base_tau=base_tau, device=scores.device)
        mask = confidence > thresholds[labels]
    elif normalized_threshold_mode == "official":
        mask = state.official_threshold_mask(labels, confidence, base_tau=base_tau)
    else:
        raise ValueError("pseudo_threshold_mode must be one of: paper, official")
    return labels, mask, confidence


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
    class_weight_smoothing: float = 0.0,
    class_weight_clip_min: float | None = None,
    class_weight_clip_max: float | None = None,
    class_weight_mean_normalize: bool = False,
    kl_estimator_mode: str = "dvkl",
    mine_ma_rate: float = 0.01,
    mine_update_scale: float = 0.5,
    pseudo_threshold_mode: str = "paper",
    pseudo_score_mode: str = "probability",
    class_weight_timing: str = "previous",
    target_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int]:
    """Run one Algorithm 1 batch: T ascent, pseudo-labeling, then E/C descent."""
    if estimate_steps <= 0:
        raise ValueError("estimate_steps must be positive")

    model.train()
    last_estimate_loss: torch.Tensor | None = None
    last_estimate_zeta: torch.Tensor | None = None
    ma_et: torch.Tensor | float = 1.0
    normalized_kl_mode = str(kl_estimator_mode).strip().lower()
    for _ in range(int(estimate_steps)):
        optimizer_t.zero_grad()
        source_estimates, target_estimates = _estimate_outputs(model, source_x, target_x)
        if normalized_kl_mode == "dvkl":
            zeta = dv_kl_domain_alignment(source_estimates, target_estimates)
            estimate_loss = -zeta
        elif normalized_kl_mode == "mine_ma":
            mine_terms = mine_kl_stabilized_objective(
                source_estimates,
                target_estimates,
                ma_et=ma_et,
                ma_rate=mine_ma_rate,
            )
            ma_et = mine_terms["ma_et"]
            zeta = mine_terms["kl"]
            estimate_loss = -float(mine_update_scale) * mine_terms["loss"]
        else:
            raise ValueError("kl_estimator_mode must be one of: dvkl, mine_ma")
        estimate_loss.backward()
        optimizer_t.step()
        last_estimate_loss = estimate_loss.detach()
        last_estimate_zeta = zeta.detach()

    optimizer_ec.zero_grad()
    previous_t_grad = _set_requires_grad(model.estimate_network, False)
    try:
        source_estimate_logits, target_estimate_logits = _estimate_outputs(
            model,
            source_x,
            target_x,
            detach_features=False,
        )
        source_outputs = model(source_x)
        target_outputs = model(target_x)
        source_outputs = dict(source_outputs)
        target_outputs = dict(target_outputs)
        source_outputs["estimate_logits"] = source_estimate_logits
        target_outputs["estimate_logits"] = target_estimate_logits
        pseudo_labels, target_mask, target_confidence = _select_pseudo_labels(
            target_outputs["tx_logits"],
            state,
            base_tau=base_tau,
            threshold_mode=pseudo_threshold_mode,
            score_mode=pseudo_score_mode,
        )
        if str(class_weight_timing).strip().lower() == "current":
            class_weights = state.class_weights_after_predictions(
                pseudo_labels,
                target_indices=target_indices,
                prior=class_prior,
                device=target_outputs["tx_logits"].device,
                smoothing=class_weight_smoothing,
                clip_min=class_weight_clip_min,
                clip_max=class_weight_clip_max,
                mean_normalize=class_weight_mean_normalize,
            )
        elif str(class_weight_timing).strip().lower() == "previous":
            class_weights = state.class_weights(
                prior=class_prior,
                device=target_outputs["tx_logits"].device,
                smoothing=class_weight_smoothing,
                clip_min=class_weight_clip_min,
                clip_max=class_weight_clip_max,
                mean_normalize=class_weight_mean_normalize,
            )
        else:
            raise ValueError("class_weight_timing must be one of: previous, current")
        kl_loss_override = None
        if normalized_kl_mode == "mine_ma":
            mine_terms = mine_kl_stabilized_objective(
                source_estimate_logits,
                target_estimate_logits,
                ma_et=ma_et,
                ma_rate=mine_ma_rate,
            )
            kl_loss_override = mine_terms["loss"]
        terms = gada_minimax_objective(
            source_outputs,
            target_outputs,
            source_labels=source_y,
            target_pseudo_labels=pseudo_labels,
            target_mask=target_mask,
            class_weights=class_weights,
            mu=mu,
            kl_weight=kl_weight,
            kl_loss_override=kl_loss_override,
        )
        terms["loss"].backward()
        optimizer_ec.step()
    finally:
        _restore_requires_grad(model.estimate_network, previous_t_grad)

    state.update(
        pseudo_labels,
        pseudo_labels[target_mask],
        target_indices=target_indices,
        target_mask=target_mask,
    )
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
        "estimate_zeta": torch.tensor(0.0) if last_estimate_zeta is None else last_estimate_zeta,
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
