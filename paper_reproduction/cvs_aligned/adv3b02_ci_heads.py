"""Support-only class-incremental heads over a frozen ADV3B02 feature runtime.

These are CVS-aligned feature-head extensions. They reuse the audited method
losses/mechanisms but do not retrain the paper-specific feature encoders.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from paper_reproduction.CSIL.losses import compute_csil_loss
from paper_reproduction.CSIL.model import CSILClassifier, csil_masked_sgd_step
from paper_reproduction.mopc_hr_non_exemplar_cil_sei import (
    compute_class_prototypes,
    correct_old_prototypes,
    mopc_hr_incremental_objective,
    prototype_augmentation,
)
from paper_reproduction.orthogonal_incremental_sei.losses import (
    base_training_loss,
    incremental_calibration_loss,
)
from paper_reproduction.orthogonal_incremental_sei.pseudo_targets import (
    assign_base_targets,
    make_simplex_pseudo_targets,
    perturb_pseudo_targets,
)


METHODS = ("csil", "mopc_hr", "orthogonal_incremental")


@dataclass
class FittedIncrementalHead:
    method: str
    before_state: dict[str, torch.Tensor]
    after_state: dict[str, torch.Tensor]
    loss_trace: list[dict[str, float | int | str]]
    resource: dict[str, Any]


def _require_support(features: torch.Tensor, labels: torch.Tensor, old_count: int) -> int:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("support features/labels shape drift")
    ids = torch.unique(labels, sorted=True)
    if ids.tolist() != list(range(int(ids.numel()))):
        raise ValueError("registered support labels must be compact and complete")
    if not 0 < int(old_count) < int(ids.numel()):
        raise ValueError("incremental head requires old and seen-new classes")
    return int(ids.numel())


def _finite_trace(trace: list[dict[str, Any]]) -> None:
    for row in trace:
        for key, value in row.items():
            if key in {"method", "phase"}:
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise FloatingPointError(f"non-finite loss trace: {row}")


def _state_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(int(value.numel() * value.element_size()) for value in state.values())


def _state_params(state: Mapping[str, torch.Tensor]) -> int:
    return sum(int(value.numel()) for value in state.values())


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _cosine_predict(features: torch.Tensor, weights: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    return (float(scale) * F.normalize(features.float(), dim=1) @ F.normalize(weights.float(), dim=1).t()).argmax(1)


def _fit_csil(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_count: int,
    seed: int,
    steps: int,
) -> FittedIncrementalHead:
    torch.manual_seed(int(seed))
    device = features.device
    model = CSILClassifier(
        input_dim=int(features.shape[1]), embedding_dim=64, num_classes=int(old_count)
    ).to(device)
    trace: list[dict[str, Any]] = []
    old_mask = labels < int(old_count)
    old_x, old_y = features[old_mask], labels[old_mask]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(old_x)
        loss = F.cross_entropy(logits, old_y)
        loss.backward()
        optimizer.step()
        trace.append({"method": "csil", "phase": "base_old", "step": step, "loss": float(loss.detach())})
    before_model = copy.deepcopy(model).eval()
    before_state = _clone_state(before_model)
    teacher = copy.deepcopy(before_model).eval()
    previous_params = {name: value.detach().clone() for name, value in before_model.named_parameters()}
    fisher = {name: torch.ones_like(value) for name, value in previous_params.items()}
    model.expand_for_stage(
        new_classes=int(torch.unique(labels).numel()) - int(old_count),
        added_embedding_dim=32,
        stage_id=1,
    )
    velocity: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        teacher_old = teacher(old_x)
    for step in range(1, int(steps) + 1):
        model.zero_grad(set_to_none=True)
        logits = model(features)
        current_old = model(old_x)[:, :old_count]
        losses = compute_csil_loss(
            logits=logits,
            labels=labels,
            current_old_response=current_old,
            previous_old_response=teacher_old,
            params=dict(model.named_parameters()),
            previous_params=previous_params,
            fisher=fisher,
            kd_weight=1.0,
            ewc_weight=1e-3,
        )
        losses.total.backward()
        velocity = csil_masked_sgd_step(
            model, lr=0.03, momentum=0.9, weight_decay=1e-4, state=velocity
        )
        trace.append({
            "method": "csil", "phase": "increment", "step": step,
            "loss": float(losses.total.detach()),
            "cross_entropy": float(losses.cross_entropy.detach()),
            "knowledge_distillation": float(losses.knowledge_distillation.detach()),
            "ewc": float(losses.ewc.detach()),
        })
    after_state = _clone_state(model.eval())
    resource_state = {f"after.{k}": v for k, v in after_state.items()}
    return FittedIncrementalHead(
        "csil", before_state, after_state, trace,
        {
            "trainable_parameters": _state_params(after_state),
            "optimizer_steps": 2 * int(steps),
            "persistent_state_bytes": _state_bytes(resource_state),
            "head_kind": "zero_bias_cosine_channel_separation",
        },
    )


def _fit_mopc(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_count: int,
    seed: int,
    steps: int,
) -> FittedIncrementalHead:
    torch.manual_seed(int(seed))
    device = features.device
    dim = int(features.shape[1])
    projection = nn.Linear(dim, 128, bias=False).to(device)
    nn.init.orthogonal_(projection.weight)
    old_mask = labels < int(old_count)
    old_x, old_y = features[old_mask], labels[old_mask]
    trace: list[dict[str, Any]] = []
    optimizer = torch.optim.Adam(projection.parameters(), lr=2e-3)
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        z = projection(old_x)
        protos, ids = compute_class_prototypes(z, old_y, range(old_count))
        logits = 5.0 * F.normalize(z, dim=1) @ F.normalize(protos, dim=1).t()
        loss = F.cross_entropy(logits, old_y)
        loss.backward()
        optimizer.step()
        trace.append({"method": "mopc_hr", "phase": "base_old", "step": step, "loss": float(loss.detach())})
    before_projection = copy.deepcopy(projection).eval()
    with torch.no_grad():
        old_before, _ = compute_class_prototypes(
            before_projection(old_x), old_y, range(old_count)
        )
    previous = {name: value.detach().clone() for name, value in projection.named_parameters()}
    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(int(seed) + 17)
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        z = projection(features)
        protos, ids = compute_class_prototypes(z, labels, range(int(torch.unique(labels).numel())))
        logits = 5.0 * F.normalize(z, dim=1) @ F.normalize(protos, dim=1).t()
        replay_x, replay_y = prototype_augmentation(
            protos[:old_count], ids[:old_count], num_samples=max(16, old_count * 4),
            generator=generator,
        )
        replay_logits = 5.0 * F.normalize(replay_x, dim=1) @ F.normalize(protos, dim=1).t()
        losses = mopc_hr_incremental_objective(
            logits, labels, replay_logits, replay_y,
            dict(projection.named_parameters()), previous, beta=1e-4, lambda_max=1.0,
        )
        losses.total.backward()
        optimizer.step()
        trace.append({
            "method": "mopc_hr", "phase": "increment", "step": step,
            "loss": float(losses.total.detach()),
            "cross_entropy": float(losses.cross_entropy.detach()),
            "prototype_augmentation": float(losses.prototype_augmentation.detach()),
            "hierarchical_regularization": float(losses.hierarchical_regularization.detach()),
        })
    with torch.no_grad():
        all_after, ids = compute_class_prototypes(
            projection(features), labels, range(int(torch.unique(labels).numel()))
        )
        new_previous, _ = compute_class_prototypes(
            before_projection(features[~old_mask]), labels[~old_mask],
            range(old_count, int(torch.unique(labels).numel())),
        )
        corrected_old = correct_old_prototypes(
            old_before, new_previous, all_after[old_count:], alpha=0.97,
            similarity_mode="paper_cosine",
        )
        all_after = torch.cat([corrected_old, all_after[old_count:]], dim=0)
    before_state = {
        "projection.weight": before_projection.weight.detach().cpu(),
        "class_weights": old_before.detach().cpu(),
    }
    after_state = {
        "projection.weight": projection.weight.detach().cpu(),
        "class_weights": all_after.detach().cpu(),
    }
    return FittedIncrementalHead(
        "mopc_hr", before_state, after_state, trace,
        {
            "trainable_parameters": int(projection.weight.numel()),
            "optimizer_steps": 2 * int(steps),
            "persistent_state_bytes": _state_bytes(after_state),
            "head_kind": "prototype_augmentation_hr_momentum_correction",
        },
    )


def _fit_orthogonal(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_count: int,
    seed: int,
    steps: int,
) -> FittedIncrementalHead:
    torch.manual_seed(int(seed))
    device = features.device
    dim = int(features.shape[1])
    projection = nn.Linear(dim, 128, bias=False).to(device)
    nn.init.orthogonal_(projection.weight)
    class_count = int(torch.unique(labels).numel())
    simplex = make_simplex_pseudo_targets(
        num_targets=class_count, feature_dim=128, total_classes=class_count,
        device=device,
    )
    perturbed = perturb_pseudo_targets(simplex, noise_range=0.01, seed=int(seed))
    assigned_old = assign_base_targets(range(old_count), simplex)
    old_mask = labels < int(old_count)
    old_x, old_y = features[old_mask], labels[old_mask]
    trace: list[dict[str, Any]] = []
    optimizer = torch.optim.Adam(projection.parameters(), lr=2e-3)
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = base_training_loss(
            projection(old_x), old_y, assigned_old, simplex, perturbed,
            contrast_temperature=0.1, center_temperature=0.1,
        )
        loss.backward()
        optimizer.step()
        trace.append({
            "method": "orthogonal_incremental", "phase": "base_old", "step": step,
            "loss": float(loss.detach()), "ce": float(terms["ce"]),
            "contrastive": float(terms["contrastive"]), "center": float(terms["center"]),
        })
    before_projection = copy.deepcopy(projection).eval()
    new_ids = torch.arange(old_count, class_count, device=device)
    with torch.no_grad():
        new_prototypes, _ = compute_class_prototypes(
            projection(features[~old_mask]), labels[~old_mask], new_ids
        )
    new_weights = nn.Parameter(new_prototypes.detach().clone())
    optimizer = torch.optim.Adam([new_weights], lr=1e-2)
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        new_z = projection(features[~old_mask]).detach()
        loss, terms = incremental_calibration_loss(
            new_z, labels[~old_mask], simplex[:old_count], new_weights,
            new_class_ids=new_ids, prototypes=new_prototypes,
            top_k=min(60, class_count - 1), margin=0.2, tau_fuse=0.01,
            lambda_align=1.6,
        )
        loss.backward()
        optimizer.step()
        trace.append({
            "method": "orthogonal_incremental", "phase": "increment", "step": step,
            "loss": float(loss.detach()), "margin": float(terms["margin"]),
            "align": float(terms["align"]), "hard_count": float(terms["hard_count"]),
        })
    before_state = {
        "projection.weight": before_projection.weight.detach().cpu(),
        "class_weights": simplex[:old_count].detach().cpu(),
    }
    after_state = {
        "projection.weight": projection.weight.detach().cpu(),
        "class_weights": torch.cat([simplex[:old_count], new_weights.detach()], dim=0).cpu(),
    }
    return FittedIncrementalHead(
        "orthogonal_incremental", before_state, after_state, trace,
        {
            "trainable_parameters": int(projection.weight.numel() + new_weights.numel()),
            "optimizer_steps": 2 * int(steps),
            "persistent_state_bytes": _state_bytes(after_state),
            "head_kind": "simplex_pseudo_target_incremental_calibration",
        },
    )


def fit_incremental_head(
    method: str,
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    *,
    old_count: int,
    seed: int,
    steps: int = 10,
) -> FittedIncrementalHead:
    """Fit one locked support-only head; query tensors are intentionally absent."""
    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f"unknown class-incremental method: {method}")
    _require_support(support_features, support_labels, old_count)
    if not 0 < int(steps) <= 20:
        raise ValueError("head steps must be in [1, 20]")
    started = time.perf_counter()
    if method == "csil":
        fitted = _fit_csil(support_features, support_labels, old_count=old_count, seed=seed, steps=steps)
    elif method == "mopc_hr":
        fitted = _fit_mopc(support_features, support_labels, old_count=old_count, seed=seed, steps=steps)
    else:
        fitted = _fit_orthogonal(support_features, support_labels, old_count=old_count, seed=seed, steps=steps)
    fitted.resource["adaptation_wall_seconds"] = time.perf_counter() - started
    fitted.resource["backbone_frozen"] = True
    fitted.resource["query_rows_used_for_training"] = 0
    fitted.resource["dense_query_graph_used"] = False
    _finite_trace(fitted.loss_trace)
    if fitted.resource["trainable_parameters"] > 50_000:
        raise ValueError("class-incremental head exceeds preferred parameter cap")
    if fitted.resource["persistent_state_bytes"] > 256 * 1024:
        raise ValueError("class-incremental head exceeds persistent-state cap")
    return fitted


def predict_incremental_head(
    fitted: FittedIncrementalHead,
    query_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return before/after class indices without role-specific query branching."""
    device = query_features.device
    if fitted.method == "csil":
        before = CSILClassifier(
            input_dim=int(query_features.shape[1]), embedding_dim=64,
            num_classes=int(fitted.before_state["classifier.weight"].shape[0]),
        ).to(device)
        before.load_state_dict({key: value.to(device) for key, value in fitted.before_state.items()})
        after_classes, after_dim = fitted.after_state["classifier.weight"].shape
        after = CSILClassifier(
            input_dim=int(query_features.shape[1]), embedding_dim=int(after_dim),
            num_classes=int(after_classes), stage_id=1,
        ).to(device)
        after.load_state_dict({key: value.to(device) for key, value in fitted.after_state.items()})
        with torch.no_grad():
            return before(query_features).argmax(1), after(query_features).argmax(1)
    before_projection = fitted.before_state["projection.weight"].to(device)
    after_projection = fitted.after_state["projection.weight"].to(device)
    before_weights = fitted.before_state["class_weights"].to(device)
    after_weights = fitted.after_state["class_weights"].to(device)
    before_z = F.linear(query_features, before_projection)
    after_z = F.linear(query_features, after_projection)
    return _cosine_predict(before_z, before_weights), _cosine_predict(after_z, after_weights)


def prototype_baseline(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    query_features: torch.Tensor,
    *,
    class_count: int,
) -> torch.Tensor:
    prototypes, _ = compute_class_prototypes(
        support_features, support_labels, range(int(class_count))
    )
    return _cosine_predict(query_features, prototypes)
