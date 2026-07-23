"""Paper-mechanism class-incremental training on a trainable ADV3B02 backbone.

The original paper encoders are intentionally not claimed as reproduced here.
This module transplants the complete incremental mechanisms onto ADV3B02:

* CSIL: zero-bias cosine fingerprints, separated added channels, old
  fingerprint masks, KD, and support-estimated diagonal Fisher/EWC.
* MoPC-HR: non-exemplar prototype augmentation, hierarchical regularization,
  paper-cosine prototype correction, and corrected-prototype inference.

All functions consume support tensors only. Query tensors are accepted only by
the separate prediction functions after the enrolled state is hash-locked.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from paper_reproduction.mopc_hr_non_exemplar_cil_sei import (
    compute_class_prototypes,
    correct_old_prototypes,
    prototype_augmentation,
)


METHODS = ("csil_paper_full", "mopc_hr_paper_full")
FeatureFn = Callable[[nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def zero_bias_logits(
    features: torch.Tensor,
    fingerprints: torch.Tensor,
    *,
    scale: float = 5.0,
    offset: float = 5.0,
) -> torch.Tensor:
    """CSIL zero-bias fingerprint response, ``scale*cosine+offset``."""
    return scale * (F.normalize(features, dim=1) @ F.normalize(fingerprints, dim=1).t()) + offset


def _class_means(features: torch.Tensor, labels: torch.Tensor, class_count: int) -> torch.Tensor:
    values = []
    for class_id in range(int(class_count)):
        rows = features[labels == class_id]
        if rows.numel() == 0:
            raise ValueError(f"class {class_id} is absent from support")
        values.append(rows.mean(0))
    return torch.stack(values)


def _batches(
    rows: int,
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device).manual_seed(int(seed))
    iteration = 0
    for epoch in range(int(epochs)):
        order = torch.randperm(int(rows), generator=generator, device=device)
        for start in range(0, int(rows), int(batch_size)):
            iteration += 1
            yield epoch + 1, iteration, order[start : start + int(batch_size)]


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite {name}")


def _stratified_train_indices(
    labels: torch.Tensor,
    *,
    fraction: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=labels.device).manual_seed(int(seed))
    selected = []
    for class_id in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        order = indices[
            torch.randperm(len(indices), generator=generator, device=labels.device)
        ]
        count = max(1, int(math.floor(float(fraction) * len(indices))))
        selected.append(order[:count])
    return torch.cat(selected)


class CSILADV3B02(nn.Module):
    """ADV3B02 plus the CSIL channel-separated zero-bias fingerprint head."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_fn: FeatureFn,
        old_fingerprints: torch.Tensor,
        new_support_features: torch.Tensor,
        new_support_labels: torch.Tensor,
        old_count: int,
        total_count: int,
        added_dim: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_fn = feature_fn
        feature_dim = int(old_fingerprints.shape[1])
        self.added_projection = nn.Linear(
            feature_dim, int(added_dim), bias=True
        ).to(device=old_fingerprints.device, dtype=old_fingerprints.dtype)
        nn.init.kaiming_uniform_(self.added_projection.weight, a=math.sqrt(5))
        nn.init.zeros_(self.added_projection.bias)
        with torch.no_grad():
            added = self.added_projection(new_support_features)
            new_ids = new_support_labels - int(old_count)
            new_added = _class_means(added, new_ids, int(total_count) - int(old_count))
            fingerprints = old_fingerprints.new_zeros((int(total_count), feature_dim + int(added_dim)))
            fingerprints[: int(old_count), :feature_dim] = old_fingerprints
            fingerprints[int(old_count) :, feature_dim:] = new_added
        self.fingerprints = nn.Parameter(fingerprints)
        mask = torch.zeros_like(fingerprints)
        mask[int(old_count) :, feature_dim:] = 1.0
        self.register_buffer("fingerprint_gradient_mask", mask)
        self.old_count = int(old_count)
        self.feature_dim = feature_dim

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base, direct = self.feature_fn(self.backbone, x)
        added = self.added_projection(base)
        return torch.cat([base, added], dim=1), direct

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, direct = self.features(x)
        return zero_bias_logits(features, self.fingerprints), direct

    def apply_fingerprint_mask(self) -> None:
        if self.fingerprints.grad is not None:
            self.fingerprints.grad.mul_(self.fingerprint_gradient_mask)


class MoPCADV3B02(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_fn: FeatureFn,
        initial_fingerprints: torch.Tensor,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_fn = feature_fn
        self.classifier = nn.Linear(
            int(initial_fingerprints.shape[1]),
            int(initial_fingerprints.shape[0]),
            bias=False,
        ).to(
            device=initial_fingerprints.device,
            dtype=initial_fingerprints.dtype,
        )
        with torch.no_grad():
            self.classifier.weight.copy_(F.normalize(initial_fingerprints, dim=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, direct = self.feature_fn(self.backbone, x)
        return self.classifier(features), features, direct


def _semantic_layer_hierarchical_regularization(
    current_parameters: Mapping[str, torch.Tensor],
    previous_parameters: Mapping[str, torch.Tensor],
    *,
    lambda_max: float,
) -> torch.Tensor:
    """Paper HR with one decaying coefficient per semantic module layer."""
    if list(current_parameters) != list(previous_parameters):
        raise ValueError("current/previous parameter order drift")
    layer_names = []
    for name in current_parameters:
        layer = name.rsplit(".", 1)[0] if "." in name else name
        if layer not in layer_names:
            layer_names.append(layer)
    if not layer_names:
        raise ValueError("hierarchical regularization requires parameters")
    layer_index = {name: index for index, name in enumerate(layer_names)}
    first = next(iter(current_parameters.values()))
    total = first.new_zeros(())
    for name, current in current_parameters.items():
        layer = name.rsplit(".", 1)[0] if "." in name else name
        coefficient = float(lambda_max) * (
            1.0 - float(layer_index[layer]) / float(len(layer_names))
        )
        total = total + coefficient * torch.sum(
            (current - previous_parameters[name].detach()) ** 2
        )
    return total


@dataclass
class PaperFullState:
    method: str
    teacher_backbone: nn.Module
    current_model: nn.Module
    feature_fn: FeatureFn
    old_count: int
    total_count: int
    before_old_prototypes: torch.Tensor
    after_prototypes: torch.Tensor
    loss_trace: list[dict[str, Any]]
    resource: dict[str, Any]

    def serializable_state(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "old_count": self.old_count,
            "total_count": self.total_count,
            "teacher_backbone": {
                key: value.detach().cpu() for key, value in self.teacher_backbone.state_dict().items()
            },
            "current_model": {
                key: value.detach().cpu() for key, value in self.current_model.state_dict().items()
            },
            "before_old_prototypes": self.before_old_prototypes.detach().cpu(),
            "after_prototypes": self.after_prototypes.detach().cpu(),
            "resource": self.resource,
        }


def _estimate_diagonal_fisher(
    backbone: nn.Module,
    *,
    feature_fn: FeatureFn,
    old_x: torch.Tensor,
    old_y: torch.Tensor,
    old_fingerprints: torch.Tensor,
) -> dict[str, torch.Tensor]:
    backbone.zero_grad(set_to_none=True)
    features, _ = feature_fn(backbone, old_x)
    loss = F.cross_entropy(zero_bias_logits(features, old_fingerprints), old_y)
    loss.backward()
    fisher = {}
    for name, parameter in backbone.named_parameters():
        grad = parameter.grad
        fisher[name] = (
            torch.zeros_like(parameter)
            if grad is None
            else torch.exp(grad.detach().pow(2).clamp(max=20.0))
        )
    backbone.zero_grad(set_to_none=True)
    return fisher


def _ewc(
    current: Mapping[str, torch.Tensor],
    previous: Mapping[str, torch.Tensor],
    fisher: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    first = next(iter(current.values()))
    total = first.new_zeros(())
    for name, value in current.items():
        total = total + (fisher[name] * (value - previous[name]) ** 2).sum()
    return total


def fit_csil_paper_full(
    backbone: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    *,
    feature_fn: FeatureFn,
    old_count: int,
    seed: int,
    epochs: int = 3,
    batch_size: int = 20,
    added_dim: int = 32,
    kd_weight: float = 0.2,
    ewc_weight: float = 1.0,
    l2_factor: float = 0.05,
    momentum: float = 0.9,
    base_old_fingerprints: torch.Tensor | None = None,
    base_fisher: Mapping[str, torch.Tensor] | None = None,
) -> PaperFullState:
    total_count = int(torch.unique(support_y).numel())
    if total_count <= int(old_count):
        raise ValueError("CSIL requires at least one new class")
    teacher = copy.deepcopy(backbone).eval()
    old_mask = support_y < int(old_count)
    new_mask = ~old_mask
    with torch.no_grad():
        new_features, _ = feature_fn(teacher, support_x[new_mask])
        if base_old_fingerprints is None:
            old_features, _ = feature_fn(teacher, support_x[old_mask])
            old_fingerprints = _class_means(
                old_features, support_y[old_mask], int(old_count)
            )
        else:
            old_fingerprints = base_old_fingerprints.to(
                device=support_x.device, dtype=new_features.dtype
            )
    if base_fisher is None:
        fisher_backbone = copy.deepcopy(backbone)
        fisher_backbone.train()
        fisher = _estimate_diagonal_fisher(
            fisher_backbone,
            feature_fn=feature_fn,
            old_x=support_x[old_mask],
            old_y=support_y[old_mask],
            old_fingerprints=old_fingerprints,
        )
        fisher_source = "target_old_support_fallback"
    else:
        fisher = {
            name: base_fisher[name].to(device=value.device, dtype=value.dtype)
            for name, value in backbone.named_parameters()
        }
        fisher_source = "original_base_source_training_state"
    model = CSILADV3B02(
        copy.deepcopy(backbone),
        feature_fn=feature_fn,
        old_fingerprints=old_fingerprints,
        new_support_features=new_features,
        new_support_labels=support_y[new_mask],
        old_count=int(old_count),
        total_count=total_count,
        added_dim=int(added_dim),
    ).to(support_x.device).train()
    previous = {name: value.detach().clone() for name, value in model.backbone.named_parameters()}
    velocity = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
    trace: list[dict[str, Any]] = []
    all_new_x = support_x[new_mask]
    all_new_y = support_y[new_mask]
    train_indices = _stratified_train_indices(
        all_new_y, fraction=0.60, seed=int(seed) + 17
    )
    new_x = all_new_x[train_indices]
    new_y = all_new_y[train_indices]
    updated_names: set[str] = set()
    for epoch, iteration, indices in _batches(
        len(new_x),
        batch_size=int(batch_size),
        epochs=int(epochs),
        device=new_x.device,
        seed=int(seed),
    ):
        model.zero_grad(set_to_none=True)
        logits, _ = model(new_x[indices])
        with torch.no_grad():
            teacher_features, _ = feature_fn(teacher, new_x[indices])
            teacher_old = zero_bias_logits(teacher_features, old_fingerprints)
        ce = F.cross_entropy(logits, new_y[indices])
        kd = F.mse_loss(logits[:, : int(old_count)], teacher_old)
        current = dict(model.backbone.named_parameters())
        ewc = _ewc(current, previous, fisher)
        loss = ce + float(kd_weight) * kd + float(ewc_weight) * ewc
        _finite(loss, "CSIL loss")
        loss.backward()
        model.apply_fingerprint_mask()
        learning_rate = 0.01 / (1.0 + 0.01 * float(iteration))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                update = parameter.grad + 2.0 * float(l2_factor) * parameter
                if name == "fingerprints":
                    update.mul_(model.fingerprint_gradient_mask)
                velocity[name].mul_(float(momentum)).add_(update)
                parameter.add_(velocity[name], alpha=-learning_rate)
                if bool(torch.count_nonzero(update)):
                    updated_names.add(name)
        trace.append(
            {
                "method": "csil_paper_full",
                "epoch": epoch,
                "iteration": iteration,
                "learning_rate": learning_rate,
                "cross_entropy": float(ce.detach()),
                "knowledge_distillation": float(kd.detach()),
                "ewc": float(ewc.detach()),
                "loss": float(loss.detach()),
            }
        )
    model.eval()
    with torch.no_grad():
        current_features, _ = model.features(support_x)
        after = _class_means(current_features, support_y, total_count)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active_fisher = [value for value in fisher.values() if bool(torch.count_nonzero(value))]
    return PaperFullState(
        method="csil_paper_full",
        teacher_backbone=teacher,
        current_model=model,
        feature_fn=feature_fn,
        old_count=int(old_count),
        total_count=total_count,
        before_old_prototypes=old_fingerprints.detach(),
        after_prototypes=after.detach(),
        loss_trace=trace,
        resource={
            "backbone_frozen": False,
            "backbone_trainable_parameters": sum(p.numel() for p in model.backbone.parameters()),
            "trainable_parameters": int(trainable),
            "optimizer_steps": len(trace),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "optimizer": "masked_sgd",
            "kd_weight": float(kd_weight),
            "ewc_weight": float(ewc_weight),
            "l2_factor": float(l2_factor),
            "momentum": float(momentum),
            "added_embedding_dim": int(added_dim),
            "new_support_rows_total": int(len(all_new_x)),
            "new_support_rows_train": int(len(new_x)),
            "new_support_split": "stratified_60_percent_min_one_no_tail_drop",
            "optimizer_updated_parameter_tensors": len(updated_names),
            "optimizer_updated_parameters": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name in updated_names
            ),
            "fisher_min_active": min(float(v.min()) for v in active_fisher),
            "fisher_max": max(float(v.max()) for v in active_fisher),
            "fisher_zero_tensor_count": sum(
                not bool(torch.count_nonzero(value)) for value in fisher.values()
            ),
            "fisher_source": fisher_source,
            "old_fingerprint_source": (
                "original_base_source_training_state"
                if base_old_fingerprints is not None
                else "target_old_support_fallback"
            ),
        },
    )


def fit_mopc_hr_paper_full(
    backbone: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    *,
    feature_fn: FeatureFn,
    old_count: int,
    seed: int,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 2e-4,
    prototype_noise_std: float = 0.05,
    alpha: float = 0.97,
    beta: float = 1.0,
    lambda_max: float = 1.0,
    base_old_prototypes: torch.Tensor | None = None,
) -> PaperFullState:
    total_count = int(torch.unique(support_y).numel())
    teacher = copy.deepcopy(backbone).eval()
    old_mask = support_y < int(old_count)
    new_mask = ~old_mask
    with torch.no_grad():
        all_teacher_features, _ = feature_fn(teacher, support_x)
        if base_old_prototypes is None:
            old_prototypes, _ = compute_class_prototypes(
                all_teacher_features[old_mask], support_y[old_mask]
            )
            old_prototype_source = "target_old_support_fallback"
        else:
            old_prototypes = base_old_prototypes.to(
                device=support_x.device, dtype=all_teacher_features.dtype
            )
            old_prototype_source = "original_base_source_training_state"
        initial = _class_means(all_teacher_features, support_y, total_count)
        # Paper/public trainer allocate the final classifier width up front. Only
        # registered logits are visible to each incremental CE.
        init_generator = torch.Generator(device=support_x.device).manual_seed(
            int(seed) + 313
        )
        initial[int(old_count) :] = F.normalize(
            torch.randn(
                initial[int(old_count) :].shape,
                dtype=initial.dtype,
                device=initial.device,
                generator=init_generator,
            ),
            dim=1,
        )
    model = MoPCADV3B02(
        copy.deepcopy(backbone),
        feature_fn=feature_fn,
        initial_fingerprints=initial,
    ).to(support_x.device).train()
    generator = torch.Generator(device=support_x.device).manual_seed(int(seed) + 991)
    trace: list[dict[str, Any]] = []
    updated_names: set[str] = set()
    historical_prototypes = old_prototypes
    registered_count = int(old_count)
    stage_sizes = []
    remaining = total_count - int(old_count)
    while remaining > 0:
        stage_size = min(5, remaining)
        stage_sizes.append(stage_size)
        remaining -= stage_size
    for stage_index, stage_size in enumerate(stage_sizes, start=1):
        stage_start = registered_count
        stage_end = registered_count + stage_size
        stage_mask = (support_y >= stage_start) & (support_y < stage_end)
        stage_x = support_x[stage_mask]
        stage_y = support_y[stage_mask]
        stage_teacher = copy.deepcopy(model).eval()
        previous_parameters = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(learning_rate),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        for epoch, iteration, indices in _batches(
            len(stage_x),
            batch_size=int(batch_size),
            epochs=int(epochs),
            device=stage_x.device,
            seed=int(seed) + stage_index * 100_003,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(stage_x[indices])
            augmented, augmented_labels = prototype_augmentation(
                historical_prototypes,
                torch.arange(
                    registered_count,
                    device=support_x.device,
                    dtype=support_y.dtype,
                ),
                num_samples=int(batch_size),
                noise_std=float(prototype_noise_std),
                generator=generator,
            )
            prototype_logits = model.classifier(augmented)
            ce = F.cross_entropy(logits[:, :stage_end], stage_y[indices])
            proto = F.cross_entropy(
                prototype_logits[:, :stage_end], augmented_labels
            )
            hr = _semantic_layer_hierarchical_regularization(
                dict(model.named_parameters()),
                previous_parameters,
                lambda_max=float(lambda_max),
            )
            loss = ce + proto + float(beta) * hr
            _finite(loss, "MoPC-HR loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)):
                    updated_names.add(name)
            optimizer.step()
            trace.append(
                {
                    "method": "mopc_hr_paper_full",
                    "stage": stage_index,
                    "registered_class_count": stage_end,
                    "epoch": epoch,
                    "iteration": iteration,
                    "cross_entropy": float(ce.detach()),
                    "prototype_augmentation": float(proto.detach()),
                    "hierarchical_regularization": float(hr.detach()),
                    "loss": float(loss.detach()),
                }
            )
        model.eval()
        with torch.no_grad():
            _, previous_stage_features, _ = stage_teacher(stage_x)
            _, current_stage_features, _ = model(stage_x)
            previous_new, _ = compute_class_prototypes(
                previous_stage_features, stage_y
            )
            current_new, _ = compute_class_prototypes(
                current_stage_features, stage_y
            )
            corrected_historical = correct_old_prototypes(
                historical_prototypes,
                previous_new,
                current_new,
                alpha=float(alpha),
                similarity_mode="paper_cosine",
            )
            historical_prototypes = torch.cat(
                [corrected_historical, current_new], dim=0
            )
        registered_count = stage_end
        model.train()
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return PaperFullState(
        method="mopc_hr_paper_full",
        teacher_backbone=teacher,
        current_model=model,
        feature_fn=feature_fn,
        old_count=int(old_count),
        total_count=total_count,
        before_old_prototypes=old_prototypes.detach(),
        after_prototypes=historical_prototypes.detach(),
        loss_trace=trace,
        resource={
            "backbone_frozen": False,
            "backbone_trainable_parameters": sum(p.numel() for p in model.backbone.parameters()),
            "trainable_parameters": int(trainable),
            "optimizer_steps": len(trace),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "optimizer": "SGD",
            "learning_rate": float(learning_rate),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "prototype_noise_std": float(prototype_noise_std),
            "prototype_momentum_alpha": float(alpha),
            "prototype_similarity": "paper_cosine",
            "hierarchical_regularization": "layer_decayed_squared_l2",
            "incremental_stage_sizes": stage_sizes,
            "query_decision": "current_model_all_registered_classifier_logits",
            "optimizer_updated_parameter_tensors": len(updated_names),
            "optimizer_updated_parameters": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name in updated_names
            ),
            "beta": float(beta),
            "lambda_max": float(lambda_max),
            "old_prototype_source": old_prototype_source,
        },
    )


@torch.no_grad()
def predict_before(state: PaperFullState, query_x: torch.Tensor) -> torch.Tensor:
    features, _ = state.feature_fn(state.teacher_backbone, query_x)
    scores = F.normalize(features, dim=1) @ F.normalize(
        state.before_old_prototypes, dim=1
    ).t()
    return scores.argmax(1)


@torch.no_grad()
def predict_after(state: PaperFullState, query_x: torch.Tensor) -> torch.Tensor:
    if state.method == "csil_paper_full":
        logits, _ = state.current_model(query_x)
        return logits.argmax(1)
    if state.method == "mopc_hr_paper_full":
        logits, _, _ = state.current_model(query_x)
        return logits[:, : state.total_count].argmax(1)
    raise ValueError(state.method)


def fit_paper_full(
    method: str,
    backbone: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    *,
    feature_fn: FeatureFn,
    old_count: int,
    seed: int,
    base_state: Mapping[str, Any] | None = None,
) -> PaperFullState:
    if method == "csil_paper_full":
        return fit_csil_paper_full(
            backbone,
            support_x,
            support_y,
            feature_fn=feature_fn,
            old_count=old_count,
            seed=seed,
            base_old_fingerprints=(
                None if base_state is None else base_state["old_fingerprints"]
            ),
            base_fisher=None if base_state is None else base_state["fisher"],
        )
    if method == "mopc_hr_paper_full":
        return fit_mopc_hr_paper_full(
            backbone,
            support_x,
            support_y,
            feature_fn=feature_fn,
            old_count=old_count,
            seed=seed,
            base_old_prototypes=(
                None if base_state is None else base_state["old_prototypes"]
            ),
        )
    raise ValueError(f"method must be one of {METHODS}")
