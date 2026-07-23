"""Official-repository execution semantics on the ADV3B02 interface.

This module keeps the public trainers' numerically active behavior. The only
method adaptations are the ADV3B02 feature interface and CVS class cardinality.
The public fixed-batch/drop-last behavior is retained, including zero optimizer
steps when a K-shot increment is smaller than one complete batch.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn


FeatureFn = Callable[[nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
METHODS = ("csil_official_repo", "mopc_hr_official_repo")


def zero_bias_logits(
    features: torch.Tensor,
    fingerprints: torch.Tensor,
    *,
    norm_mag: float = 5.0,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    feature_norm = torch.sqrt(
        torch.sum(features.square(), dim=1, keepdim=True) + float(epsilon) ** 2
    )
    fingerprint_norm = torch.sqrt(
        torch.sum(fingerprints.square(), dim=1, keepdim=True)
        + float(epsilon) ** 2
    )
    return (
        (features / feature_norm)
        @ (fingerprints / fingerprint_norm).t()
        * float(norm_mag)
        + float(norm_mag)
    )


def _class_means(
    features: torch.Tensor, labels: torch.Tensor, class_ids: torch.Tensor
) -> torch.Tensor:
    result = []
    for class_id in class_ids.tolist():
        selected = features[labels == int(class_id)]
        if selected.numel() == 0:
            raise ValueError(f"class {class_id} is absent")
        result.append(selected.mean(0))
    return torch.stack(result)


def _drop_last_batches(
    rows: int,
    *,
    requested_batch: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[list[tuple[int, int, torch.Tensor]], int, bool]:
    if rows < 0:
        raise ValueError("training rows must be non-negative")
    effective_batch = int(requested_batch)
    small_k_adaptation = False
    generator = torch.Generator(device=device).manual_seed(int(seed))
    result = []
    iteration = 0
    for epoch in range(int(epochs)):
        order = torch.randperm(int(rows), generator=generator, device=device)
        complete = int(rows) // effective_batch
        for batch_index in range(complete):
            iteration += 1
            start = batch_index * effective_batch
            result.append(
                (epoch + 1, iteration, order[start : start + effective_batch])
            )
    return result, effective_batch, small_k_adaptation


def csil_official_fisher_objective(probabilities: torch.Tensor) -> torch.Tensor:
    shifted = probabilities - probabilities.min() + 1e-5
    return torch.log(shifted).mean()


def csil_fisher_from_model(
    model: nn.Module, source_x: torch.Tensor, *, batch_size: int = 128
) -> dict[str, torch.Tensor]:
    global_min = None
    with torch.no_grad():
        for start in range(0, len(source_x), int(batch_size)):
            probabilities = torch.softmax(
                model(source_x[start : start + int(batch_size)]), dim=1
            )
            value = probabilities.min()
            global_min = value if global_min is None else torch.minimum(global_min, value)
    if global_min is None:
        raise ValueError("Fisher source is empty")
    model.zero_grad(set_to_none=True)
    class_count = int(model.fingerprints.shape[0])
    denominator = float(len(source_x) * class_count)
    for start in range(0, len(source_x), int(batch_size)):
        probabilities = torch.softmax(
            model(source_x[start : start + int(batch_size)]), dim=1
        )
        objective = torch.log(probabilities - global_min + 1e-5).sum()
        (objective / denominator).backward()
    fisher = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        fisher[name] = (
            torch.zeros_like(parameter)
            if gradient is None
            else torch.exp(gradient.detach().square())
        )
    model.zero_grad(set_to_none=True)
    return fisher


def csil_ewc(
    current: Mapping[str, torch.Tensor],
    previous: Mapping[str, torch.Tensor],
    fisher: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    first = next(iter(current.values()))
    total = first.new_zeros(())
    for name, old_value in previous.items():
        new_value = current[name]
        overlap = tuple(slice(0, size) for size in old_value.shape)
        total = total + (
            fisher[name] * (new_value[overlap] - old_value).square()
        ).sum()
    return total / 2.0


def csil_distillation(
    previous_old_response: torch.Tensor, current_old_response: torch.Tensor
) -> torch.Tensor:
    return (previous_old_response - current_old_response).square().sum() / 32.0


class CSILModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_fn: FeatureFn,
        fc_weight: torch.Tensor,
        fc_bias: torch.Tensor,
        fingerprints: torch.Tensor,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_fn = feature_fn
        self.fc_bf_fp = nn.Linear(
            int(fc_weight.shape[1]), int(fc_weight.shape[0]), bias=True
        ).to(device=fc_weight.device, dtype=fc_weight.dtype)
        self.fingerprints = nn.Parameter(fingerprints.detach().clone())
        with torch.no_grad():
            self.fc_bf_fp.weight.copy_(fc_weight)
            self.fc_bf_fp.bias.copy_(fc_bias)

    def pre_fingerprint(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.feature_fn(self.backbone, x)
        return self.fc_bf_fp(features)

    def response(self, x: torch.Tensor) -> torch.Tensor:
        return zero_bias_logits(self.pre_fingerprint(x), self.fingerprints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.response(x)


class MoPCModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_fn: FeatureFn,
        classifier_weight: torch.Tensor,
        classifier_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_fn = feature_fn
        self.fc = nn.Linear(
            int(classifier_weight.shape[1]),
            int(classifier_weight.shape[0]),
            bias=True,
        ).to(device=classifier_weight.device, dtype=classifier_weight.dtype)
        with torch.no_grad():
            self.fc.weight.copy_(classifier_weight)
            self.fc.bias.copy_(classifier_bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, _ = self.feature_fn(self.backbone, x)
        return self.fc(features), features


def _manual_sgdm_step(
    model: nn.Module,
    velocity: dict[str, torch.Tensor],
    *,
    learning_rate: float,
    momentum: float,
    l2_factor: float,
    masks: Mapping[str, torch.Tensor] | None = None,
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad + 2.0 * float(l2_factor) * parameter
            velocity[name].mul_(float(momentum)).add_(
                gradient, alpha=float(learning_rate)
            )
            update = velocity[name]
            if masks is not None and name in masks:
                update = update * masks[name]
            parameter.sub_(update)


def build_csil_base_state(
    backbone: nn.Module,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    *,
    feature_fn: FeatureFn,
    old_count: int,
    seed: int,
    epochs: int = 20,
    batch_size: int = 128,
    fisher_x: torch.Tensor | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    with torch.no_grad():
        feature, _ = feature_fn(backbone, source_x[:1])
    feature_dim = int(feature.shape[1])
    fc_weight = torch.empty(
        int(old_count), feature_dim, device=source_x.device, dtype=feature.dtype
    )
    nn.init.xavier_uniform_(fc_weight)
    fc_bias = torch.zeros(int(old_count), device=source_x.device, dtype=feature.dtype)
    scale = math.sqrt(1.0 / (2.0 * int(old_count)))
    fingerprints = torch.rand(
        int(old_count),
        int(old_count),
        device=source_x.device,
        dtype=feature.dtype,
    ) * scale
    model = CSILModel(
        copy.deepcopy(backbone),
        feature_fn=feature_fn,
        fc_weight=fc_weight,
        fc_bias=fc_bias,
        fingerprints=fingerprints,
    ).train()
    generator = torch.Generator(device=source_x.device).manual_seed(int(seed) + 11)
    # MATLAB trainNetwork's default Shuffle="once": one permutation is reused
    # across epochs. Its final incomplete mini-batch is retained.
    order = torch.randperm(len(source_x), generator=generator, device=source_x.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01
    )
    optimizer_steps = 0
    for _epoch in range(int(epochs)):
        for start in range(0, len(source_x), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(source_x[indices]), source_y[indices])
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
    model.eval()
    fisher_rows = source_x if fisher_x is None else fisher_x
    fisher = csil_fisher_from_model(model, fisher_rows)
    return {
        "backbone_state": {
            key: value.detach().cpu()
            for key, value in model.backbone.state_dict().items()
        },
        "fc_weight": model.fc_bf_fp.weight.detach().cpu(),
        "fc_bias": model.fc_bf_fp.bias.detach().cpu(),
        "fingerprints": model.fingerprints.detach().cpu(),
        "fisher": {key: value.detach().cpu() for key, value in fisher.items()},
        "optimizer_steps": optimizer_steps,
        "effective_batch": int(batch_size),
        "tail_batch_retained": len(source_x) % int(batch_size) != 0,
        "shuffle": "once",
        "fisher_sample_count": int(len(fisher_rows)),
        "fisher_split": (
            "source_train_fallback"
            if fisher_x is None
            else "disjoint_source_validation"
        ),
        "small_k_adaptation": False,
    }


def build_mopc_base_state(
    backbone: nn.Module,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    *,
    feature_fn: FeatureFn,
    old_count: int,
    total_capacity: int,
    seed: int,
    epochs: int = 20,
    batch_size: int = 16,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    with torch.no_grad():
        feature, _ = feature_fn(backbone, source_x[:1])
    classifier = nn.Linear(int(feature.shape[1]), int(total_capacity), bias=True).to(
        device=source_x.device, dtype=feature.dtype
    )
    model = MoPCModel(
        copy.deepcopy(backbone),
        feature_fn=feature_fn,
        classifier_weight=classifier.weight,
        classifier_bias=classifier.bias,
    ).train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-4
    )
    batches, effective_batch, adapted = _drop_last_batches(
        len(source_x),
        requested_batch=int(batch_size),
        epochs=int(epochs),
        device=source_x.device,
        seed=int(seed) + 23,
    )
    for _epoch, _iteration, indices in batches:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(source_x[indices])
        loss = F.cross_entropy(logits[:, : int(old_count)], source_y[indices])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, features = model(source_x)
        old_ids = torch.arange(int(old_count), device=source_x.device)
        prototypes = _class_means(features, source_y, old_ids)
    return {
        "backbone_state": {
            key: value.detach().cpu()
            for key, value in model.backbone.state_dict().items()
        },
        "classifier_weight": model.fc.weight.detach().cpu(),
        "classifier_bias": model.fc.bias.detach().cpu(),
        "old_prototypes": prototypes.detach().cpu(),
        "optimizer_steps": len(batches),
        "effective_batch": effective_batch,
        "small_k_adaptation": adapted,
    }


def mopc_parameter_hr(
    current: list[torch.Tensor], previous: list[torch.Tensor]
) -> torch.Tensor:
    if len(current) != len(previous) or not current:
        raise ValueError("MoPC parameter surface drift")
    total = current[0].new_zeros(())
    count = len(current)
    for index, (current_value, previous_value) in enumerate(
        zip(current, previous), start=1
    ):
        coefficient = 1.0 - float(index - 1) / float(count)
        total = total + coefficient * torch.norm(
            previous_value.detach() - current_value, p=2
        )
    return total


def mopc_correct_prototypes(
    old: torch.Tensor,
    new_previous: torch.Tensor,
    new_current: torch.Tensor,
    *,
    alpha: float = 0.97,
) -> torch.Tensor:
    weights = torch.softmax(old @ new_previous.t(), dim=1)
    delta = new_current - new_previous
    return float(alpha) * old + (1.0 - float(alpha)) * (weights @ delta)


@dataclass
class OfficialState:
    method: str
    before_model: nn.Module
    current_model: nn.Module
    old_count: int
    total_count: int
    loss_trace: list[dict[str, Any]]
    resource: dict[str, Any]
    corrected_prototypes: torch.Tensor | None = None

    @property
    def teacher_backbone(self) -> nn.Module:
        return self.before_model.backbone

    def serializable_state(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "old_count": self.old_count,
            "total_count": self.total_count,
            "before_model": {
                key: value.detach().cpu()
                for key, value in self.before_model.state_dict().items()
            },
            "current_model": {
                key: value.detach().cpu()
                for key, value in self.current_model.state_dict().items()
            },
            "corrected_prototypes": (
                None
                if self.corrected_prototypes is None
                else self.corrected_prototypes.detach().cpu()
            ),
            "resource": self.resource,
        }


def fit_csil_official_repo(
    backbone: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    *,
    feature_fn: FeatureFn,
    old_count: int,
    seed: int,
    base_state: Mapping[str, Any],
) -> OfficialState:
    base = base_state["csil"]
    backbone = copy.deepcopy(backbone)
    backbone.load_state_dict(base["backbone_state"], strict=True)
    old_fc_weight = base["fc_weight"].to(support_x.device)
    old_fc_bias = base["fc_bias"].to(support_x.device)
    old_fingerprints = base["fingerprints"].to(support_x.device)
    before = CSILModel(
        copy.deepcopy(backbone),
        feature_fn=feature_fn,
        fc_weight=old_fc_weight,
        fc_bias=old_fc_bias,
        fingerprints=old_fingerprints,
    ).eval()
    total_count = int(torch.unique(support_y).numel())
    new_count = total_count - int(old_count)
    if new_count <= 0:
        raise ValueError("CSIL requires new classes")
    new_fc_weight = old_fc_weight.new_empty((total_count, old_fc_weight.shape[1]))
    new_fc_bias = old_fc_bias.new_zeros(total_count)
    new_fingerprints = old_fingerprints.new_zeros((total_count, total_count))
    with torch.no_grad():
        new_fc_weight[:old_count] = old_fc_weight
        new_fc_bias[:old_count] = old_fc_bias
        generator = torch.Generator(device=support_x.device).manual_seed(int(seed) + 71)
        new_fc_weight[old_count:] = 1e-4 * torch.rand(
            new_fc_weight[old_count:].shape,
            device=support_x.device,
            dtype=new_fc_weight.dtype,
            generator=generator,
        )
        new_fingerprints[:old_count, :old_count] = old_fingerprints
    model = CSILModel(
        backbone,
        feature_fn=feature_fn,
        fc_weight=new_fc_weight,
        fc_bias=new_fc_bias,
        fingerprints=new_fingerprints,
    ).to(support_x.device)
    new_mask = support_y >= int(old_count)
    new_x_all = support_x[new_mask]
    new_y_all = support_y[new_mask]
    with torch.no_grad():
        ids = torch.arange(int(old_count), total_count, device=support_x.device)
        if new_count <= int(old_count):
            # CSIL.m computes new fingerprints from the pre-expansion
            # fc_bf_fp response and takes its final newClassesNum coordinates.
            prior_response = before.pre_fingerprint(new_x_all)
            new_means = _class_means(prior_response, new_y_all, ids)
            new_block = new_means[:, -new_count:]
            initialization = "official_pre_expansion_tail_coordinates"
            cardinality_adaptation = False
        else:
            # The public code has no defined slice when newClassesNum exceeds
            # the previous output width. Use the newly allocated coordinates.
            expanded = model.pre_fingerprint(new_x_all)
            new_means = _class_means(expanded, new_y_all, ids)
            new_block = new_means[:, old_count:]
            initialization = "expanded_new_coordinates"
            cardinality_adaptation = True
        model.fingerprints[old_count:, old_count:] = F.normalize(new_block, dim=1)
    generator = torch.Generator(device=support_x.device).manual_seed(int(seed) + 79)
    order = torch.randperm(len(new_x_all), generator=generator, device=support_x.device)
    official_cut = int(math.floor(0.6 * len(order)))
    train_count = official_cut - 1
    small_k_split_adaptation = False
    train_indices = order[:train_count]
    train_x = new_x_all[train_indices]
    train_y = new_y_all[train_indices]
    batches, effective_batch, small_k_batch_adaptation = _drop_last_batches(
        len(train_x),
        requested_batch=20,
        epochs=3,
        device=support_x.device,
        seed=int(seed) + 83,
    )
    fisher = {
        key: value.to(support_x.device) for key, value in base["fisher"].items()
    }
    previous = {
        key: value.detach().clone() for key, value in before.named_parameters()
    }
    velocity = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
    }
    masks = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
    }
    masks["fc_bf_fp.weight"][old_count:] = 1
    masks["fc_bf_fp.bias"][old_count:] = 1
    masks["fingerprints"][old_count:, old_count:] = 1
    trace = []
    model.train()
    for epoch, iteration, indices in batches:
        model.zero_grad(set_to_none=True)
        current_response = model(train_x[indices])
        with torch.no_grad():
            previous_response = before(train_x[indices])
        ce = F.cross_entropy(current_response, train_y[indices])
        ewc = csil_ewc(dict(model.named_parameters()), previous, fisher)
        kd = csil_distillation(
            previous_response, current_response[:, : int(old_count)]
        )
        loss = ce + ewc + 0.2 * kd
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CSIL loss")
        loss.backward()
        learning_rate = 0.01 / (1.0 + 0.01 * float(iteration))
        _manual_sgdm_step(
            model,
            velocity,
            learning_rate=learning_rate,
            momentum=0.9,
            l2_factor=0.05,
            masks=masks,
        )
        trace.append(
            {
                "epoch": epoch,
                "iteration": iteration,
                "learning_rate": learning_rate,
                "cross_entropy": float(ce.detach()),
                "ewc": float(ewc.detach()),
                "knowledge_distillation": float(kd.detach()),
                "loss": float(loss.detach()),
            }
        )
    model.eval()
    return OfficialState(
        method="csil_official_repo",
        before_model=before,
        current_model=model,
        old_count=int(old_count),
        total_count=total_count,
        loss_trace=trace,
        resource={
            "official_repo_commit": "8ce8637daf4dc60eeb1c56bff64c050c5b2353e9",
            "official_entry": "ContinualLearning/WorkStage/CSIL.m",
            "backbone_frozen_incremental": True,
            "optimizer_steps": len(trace),
            "epochs": 3,
            "requested_batch_size": 20,
            "effective_batch_size": effective_batch,
            "small_k_execution_adaptation": bool(
                small_k_split_adaptation or small_k_batch_adaptation
            ),
            "official_zero_step_due_to_drop_last": len(trace) == 0,
            "new_dimension": new_count,
            "new_fingerprint_initialization": initialization,
            "class_cardinality_initialization_adaptation": cardinality_adaptation,
            "query_decision": "zero_bias_all_registered_argmax",
        },
    )


def fit_mopc_hr_official_repo(
    backbone: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    *,
    feature_fn: FeatureFn,
    old_count: int,
    seed: int,
    base_state: Mapping[str, Any],
) -> OfficialState:
    base = base_state["mopc_hr"]
    backbone = copy.deepcopy(backbone)
    backbone.load_state_dict(base["backbone_state"], strict=True)
    model = MoPCModel(
        backbone,
        feature_fn=feature_fn,
        classifier_weight=base["classifier_weight"].to(support_x.device),
        classifier_bias=base["classifier_bias"].to(support_x.device),
    ).to(support_x.device)
    before = copy.deepcopy(model).eval()
    total_count = int(torch.unique(support_y).numel())
    increment_size = total_count - int(old_count)
    new_mask = support_y >= int(old_count)
    stage_x = support_x[new_mask]
    stage_y = support_y[new_mask]
    batches, effective_batch, small_k_adaptation = _drop_last_batches(
        len(stage_x),
        requested_batch=16,
        epochs=20,
        device=support_x.device,
        seed=int(seed) + 101,
    )
    historical = base["old_prototypes"].to(support_x.device)
    reference = copy.deepcopy(model).eval()
    previous_parameters = [
        parameter.detach().clone() for parameter in reference.parameters()
    ]
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-4
    )
    generator = torch.Generator(device=support_x.device).manual_seed(int(seed) + 107)
    trace = []
    model.train()
    for epoch, iteration, indices in batches:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(stage_x[indices])
        with torch.no_grad():
            teacher_logits, _ = reference(stage_x[indices])
            previous_prob = torch.softmax(
                teacher_logits[:, : int(old_count)] / 2.0, dim=1
            )
        current_log_prob = torch.log_softmax(
            logits[:, : int(old_count)] / 2.0, dim=1
        )
        kd = -(previous_prob * current_log_prob).sum(dim=1).mean()
        ce = F.cross_entropy(logits[:, :total_count], stage_y[indices])
        sample_count = len(indices)
        sampled_classes = torch.randint(
            0,
            int(old_count),
            (sample_count,),
            device=support_x.device,
            generator=generator,
        )
        noise = torch.randn(
            (sample_count, historical.shape[1]),
            device=support_x.device,
            dtype=historical.dtype,
            generator=generator,
        ) * 0.05
        augmented = historical[sampled_classes] + noise
        proto = F.cross_entropy(
            model.fc(augmented)[:, :total_count] / 2.0, sampled_classes
        )
        hr = mopc_parameter_hr(
            list(model.parameters()), previous_parameters
        )
        loss = ce + proto + hr
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite MoPC-HR loss")
        loss.backward()
        optimizer.step()
        trace.append(
            {
                "epoch": epoch,
                "iteration": iteration,
                "cross_entropy": float(ce.detach()),
                "knowledge_distillation_not_in_total": float(kd.detach()),
                "prototype_augmentation": float(proto.detach()),
                "hierarchical_regularization": float(hr.detach()),
                "loss": float(loss.detach()),
            }
        )
    model.eval()
    with torch.no_grad():
        _, previous_features = reference(stage_x)
        _, current_features = model(stage_x)
        new_ids = torch.arange(
            int(old_count), total_count, device=support_x.device
        )
        previous_new = _class_means(previous_features, stage_y, new_ids)
        current_new = _class_means(current_features, stage_y, new_ids)
        corrected_old = mopc_correct_prototypes(
            historical, previous_new, current_new, alpha=0.97
        )
        corrected = torch.cat([corrected_old, current_new], dim=0)
    return OfficialState(
        method="mopc_hr_official_repo",
        before_model=before,
        current_model=model,
        old_count=int(old_count),
        total_count=total_count,
        loss_trace=trace,
        corrected_prototypes=corrected,
        resource={
            "official_repo_commit": "ae6554316ad1a2175920e330133a2f103408bf78",
            "official_entry": "MoPC_HR_trainer.py",
            "backbone_frozen_incremental": False,
            "optimizer_steps": len(trace),
            "epochs": 20,
            "requested_batch_size": 16,
            "effective_batch_size": effective_batch,
            "small_k_execution_adaptation": small_k_adaptation,
            "official_zero_step_due_to_drop_last": len(trace) == 0,
            "increment_size": increment_size,
            "class_schedule_adaptation": increment_size not in (25, 10, 5, 3),
            "hierarchical_regularization": "per_parameter_unsquared_l2",
            "prototype_similarity": "raw_dot_then_softmax",
            "prototype_logit_temperature": 2.0,
            "kd_in_total_loss": False,
            "query_decision": "current_model_all_registered_classifier_logits",
        },
    )


def fit_official_repo(
    method: str,
    backbone: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    *,
    feature_fn: FeatureFn,
    old_count: int,
    seed: int,
    base_state: Mapping[str, Any],
) -> OfficialState:
    if method == "csil_official_repo":
        return fit_csil_official_repo(
            backbone,
            support_x,
            support_y,
            feature_fn=feature_fn,
            old_count=old_count,
            seed=seed,
            base_state=base_state,
        )
    if method == "mopc_hr_official_repo":
        return fit_mopc_hr_official_repo(
            backbone,
            support_x,
            support_y,
            feature_fn=feature_fn,
            old_count=old_count,
            seed=seed,
            base_state=base_state,
        )
    raise ValueError(f"method must be one of {METHODS}")


@torch.no_grad()
def predict_before(state: OfficialState, query_x: torch.Tensor) -> torch.Tensor:
    if state.method == "csil_official_repo":
        return state.before_model(query_x).argmax(1)
    logits, _ = state.before_model(query_x)
    return logits[:, : state.old_count].argmax(1)


@torch.no_grad()
def predict_after(state: OfficialState, query_x: torch.Tensor) -> torch.Tensor:
    if state.method == "csil_official_repo":
        return state.current_model(query_x).argmax(1)
    logits, _ = state.current_model(query_x)
    return logits[:, : state.total_count].argmax(1)
