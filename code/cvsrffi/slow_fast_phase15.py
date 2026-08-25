"""Source-only Phase1.5 fitting for aggregate slow adapter parameters."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate
from .slow_fast_cache import GroundFeatureCache
from .slow_fast_objectives import (
    frozen_prototype_losses,
    smooth_class_floor_loss,
    trust_region_loss,
)


def _validate_prototypes(cache: GroundFeatureCache, prototypes: Tensor) -> Tensor:
    if (
        not torch.is_tensor(prototypes)
        or prototypes.ndim != 2
        or prototypes.shape[1] != cache.feature_dim
        or not prototypes.is_floating_point()
        or not bool(torch.isfinite(prototypes).all())
    ):
        raise ValueError("frozen prototypes must match the cache feature width")
    if bool((cache.labels < 0).any()) or bool((cache.labels >= prototypes.shape[0]).any()):
        raise ValueError("cache labels must index frozen prototype rows")
    return prototypes.detach().clone().float()


def _domain_vectors(cache: GroundFeatureCache, prototypes: Tensor) -> Tensor:
    grouped: dict[tuple[object, object, str], dict[int, list[Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in range(cache.features.shape[0]):
        key = (cache.receivers[index], cache.days[index], cache.scenes[index])
        grouped[key][int(cache.labels[index].item())].append(cache.features[index])
    expected_classes = set(int(value) for value in torch.unique(cache.labels).tolist())
    vectors: list[Tensor] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        per_class = grouped[key]
        if set(per_class) != expected_classes:
            raise ValueError("every Phase1.5 pseudo-domain must cover the same classes")
        residuals = [
            torch.stack(per_class[class_id]).mean(dim=0) - prototypes[class_id]
            for class_id in sorted(expected_classes)
        ]
        vectors.append(torch.stack(residuals).mean(dim=0))
    if not vectors:
        raise ValueError("Phase1.5 cache contains no pseudo-domain vectors")
    return torch.stack(vectors)


def fit_common_shift_basis(
    cache: GroundFeatureCache,
    prototypes: Tensor,
    *,
    rank: int = 4,
) -> Tensor:
    frozen = _validate_prototypes(cache, prototypes)
    requested = int(rank)
    if requested <= 0 or requested > cache.feature_dim:
        raise ValueError("rank must be in the feature dimension")
    vectors = _domain_vectors(cache, frozen)
    if vectors.shape[0] < requested:
        raise ValueError("not enough pseudo-domains for the requested rank")
    _left, _singular, right = torch.linalg.svd(vectors.float(), full_matrices=False)
    return right[:requested].transpose(0, 1).contiguous().detach()


def _pair_indices(cache: GroundFeatureCache) -> tuple[Tensor, Tensor]:
    groups: dict[str, dict[str, int]] = defaultdict(dict)
    for index, (physical_id, view) in enumerate(
        zip(cache.physical_sample_ids, cache.views)
    ):
        groups[physical_id][view] = index
    clean_indices: list[int] = []
    leo_indices: list[int] = []
    for views in groups.values():
        clean = views.get("clean")
        if clean is None:
            continue
        for name, index in views.items():
            if name != "clean":
                clean_indices.append(clean)
                leo_indices.append(index)
    device = cache.features.device
    return (
        torch.tensor(clean_indices, dtype=torch.long, device=device),
        torch.tensor(leo_indices, dtype=torch.long, device=device),
    )


def _apply_trainable(
    features: Tensor,
    *,
    candidate: SlowFastCandidate,
    u: Tensor,
    v: Tensor,
    gamma: Tensor,
    beta: Tensor,
    direction_gate: Tensor | None,
    rho: float | Tensor,
) -> Tensor:
    normalized = F.layer_norm(features, (features.shape[1],))
    latent = (1.0 + gamma) * (normalized @ v) + beta
    if candidate is SlowFastCandidate.FAST_LOWRANK_R8:
        latent = torch.tanh(direction_gate) * latent
    residual_strength = torch.as_tensor(rho, device=features.device, dtype=features.dtype)
    return F.normalize(
        features + residual_strength * (latent @ u.transpose(0, 1)),
        dim=1,
        eps=1.0e-8,
    )


def _episode_indices(
    cache: GroundFeatureCache,
    rng: random.Random,
    *,
    query_per_class: int,
) -> tuple[Tensor, Tensor, int]:
    grouped: dict[tuple[object, object, str], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in range(cache.features.shape[0]):
        domain = (cache.receivers[index], cache.days[index], cache.scenes[index])
        grouped[domain][int(cache.labels[index].item())].append(index)
    classes = sorted(int(value) for value in torch.unique(cache.labels).tolist())
    eligible = [
        domain
        for domain, rows in grouped.items()
        if set(rows) == set(classes) and min(len(rows[class_id]) for class_id in classes) >= 2
    ]
    if not eligible:
        raise ValueError("Phase1.5 has no pseudo-domain with disjoint support/query rows")
    domain = rng.choice(eligible)
    max_k = min(len(grouped[domain][class_id]) - 1 for class_id in classes)
    choices = [(1, 0.1), (2, 0.2), (5, 0.5), (10, 0.2)]
    feasible = [(k, weight) for k, weight in choices if k <= max_k]
    total = sum(weight for _k, weight in feasible)
    k_shot = rng.choices(
        [k for k, _weight in feasible],
        weights=[weight / total for _k, weight in feasible],
        k=1,
    )[0]
    support: list[int] = []
    query: list[int] = []
    for class_id in classes:
        shuffled = list(grouped[domain][class_id])
        rng.shuffle(shuffled)
        support.extend(shuffled[:k_shot])
        query.extend(shuffled[k_shot : k_shot + max(1, int(query_per_class))])
    target_device = cache.features.device
    return (
        torch.tensor(support, dtype=torch.long, device=target_device),
        torch.tensor(query, dtype=torch.long, device=target_device),
        int(k_shot),
    )


def train_slow_fast_basis(
    cache: GroundFeatureCache,
    prototypes: Tensor,
    *,
    candidate: SlowFastCandidate,
    steps: int = 200,
    learning_rate: float = 1.0e-2,
    seed: int = 392002,
    rho: float = 0.1,
    floor_weight: float = 0.2,
    pair_weight: float = 0.2,
    trust_radius: float = 0.15,
    device: str | torch.device = "cpu",
    meta_steps: int | None = None,
    inner_steps: int = 3,
    fast_step_size: float = 0.02,
    support_logit_scale: float = 8.0,
    query_per_class: int = 5,
) -> tuple[SlowFastAdapterState, dict[str, Any]]:
    candidate = SlowFastCandidate(candidate)
    target_device = torch.device(device)
    if target_device.type != "cpu":
        cache = GroundFeatureCache(
            features=cache.features.to(target_device),
            labels=cache.labels.to(target_device),
            receivers=cache.receivers,
            days=cache.days,
            scenes=cache.scenes,
            physical_sample_ids=cache.physical_sample_ids,
            views=cache.views,
            roles=cache.roles,
        )
        prototypes = prototypes.to(target_device)
    resolved_logit_scale = float(support_logit_scale)
    if not math.isfinite(resolved_logit_scale) or resolved_logit_scale <= 0.0:
        raise ValueError("support_logit_scale must be finite and positive")
    if candidate is SlowFastCandidate.COMMON_SHIFT_R4:
        basis = fit_common_shift_basis(cache, prototypes, rank=4)
        state = SlowFastAdapterState(
            candidate=candidate,
            slow_u=basis,
            common_coeff=torch.zeros(4),
        )
        return state, {
            "steps": 0,
            "support_logit_scale": resolved_logit_scale,
            "initial_loss": math.nan,
            "final_loss": math.nan,
        }
    if int(steps) < 1:
        raise ValueError("steps must be positive for learned slow-fast candidates")
    resolved_meta_steps = int(steps if meta_steps is None else meta_steps)
    if resolved_meta_steps < 1 or int(inner_steps) != 3:
        raise ValueError("meta_steps must be positive and inner_steps must equal 3")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    frozen = _validate_prototypes(cache, prototypes)
    features = cache.features.float()
    labels = cache.labels.long()
    torch.manual_seed(int(seed))
    basis = fit_common_shift_basis(cache, frozen, rank=8)
    u = torch.nn.Parameter(-0.05 * basis.clone())
    v = torch.nn.Parameter(basis.clone())
    gamma = torch.nn.Parameter(torch.zeros(8, device=features.device))
    beta = torch.nn.Parameter(torch.zeros(8, device=features.device))
    direction_gate = (
        torch.nn.Parameter(torch.zeros(8, device=features.device))
        if candidate is SlowFastCandidate.FAST_LOWRANK_R8
        else None
    )
    rho_max = 0.25
    rho_ratio = min(max(float(rho) / rho_max, 1.0e-4), 1.0 - 1.0e-4)
    raw_rho = torch.nn.Parameter(
        torch.tensor(math.log(rho_ratio / (1.0 - rho_ratio)), device=features.device)
    )
    initial_step = float(fast_step_size)
    raw_step = torch.nn.Parameter(
        torch.tensor(math.log(math.expm1(initial_step)), device=features.device)
    )
    parameters = [u, v, gamma, beta, raw_rho, raw_step]
    if direction_gate is not None:
        parameters.append(direction_gate)
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    clean_indices, leo_indices = _pair_indices(cache)

    def objective() -> Tensor:
        current_rho = rho_max * torch.sigmoid(raw_rho)
        adapted = _apply_trainable(
            features,
            candidate=candidate,
            u=u,
            v=v,
            gamma=gamma,
            beta=beta,
            direction_gate=direction_gate,
            rho=current_rho,
        )
        per_sample = frozen_prototype_losses(
            adapted, labels, frozen, scale=resolved_logit_scale
        )
        floor = smooth_class_floor_loss(per_sample, labels, temperature=0.1)
        trust = trust_region_loss(adapted, features, max_relative_move=trust_radius)
        if clean_indices.numel():
            pair = (
                1.0
                - F.cosine_similarity(
                    adapted.index_select(0, leo_indices),
                    features.index_select(0, clean_indices),
                    dim=1,
                )
            ).mean()
        else:
            pair = adapted.sum() * 0.0
        return per_sample.mean() + float(floor_weight) * floor + float(pair_weight) * pair + trust

    with torch.no_grad():
        initial_loss = float(objective().item())
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Phase1.5 objective became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
    with torch.no_grad():
        stage1_final_loss = float(objective().item())

    rng = random.Random(int(seed) + 15)
    k_counts = {"1": 0, "2": 0, "5": 0, "10": 0}
    for _meta_step in range(resolved_meta_steps):
        support_indices, query_indices, k_shot = _episode_indices(
            cache, rng, query_per_class=int(query_per_class)
        )
        k_counts[str(k_shot)] += 1
        fast_gamma = gamma
        fast_beta = beta
        fast_gate = direction_gate
        for _inner in range(int(inner_steps)):
            current_rho = rho_max * torch.sigmoid(raw_rho)
            current_fast_step = F.softplus(raw_step)
            adapted_support = _apply_trainable(
                features.index_select(0, support_indices),
                candidate=candidate,
                u=u,
                v=v,
                gamma=fast_gamma,
                beta=fast_beta,
                direction_gate=fast_gate,
                rho=current_rho,
            )
            support_loss = frozen_prototype_losses(
                adapted_support,
                labels.index_select(0, support_indices),
                frozen,
                scale=resolved_logit_scale,
            ).mean()
            fast_values = [fast_gamma, fast_beta]
            if fast_gate is not None:
                fast_values.append(fast_gate)
            gradients = torch.autograd.grad(
                support_loss, fast_values, create_graph=True
            )
            updated = [
                value - current_fast_step * gradient
                for value, gradient in zip(fast_values, gradients)
            ]
            fast_gamma, fast_beta = updated[:2]
            fast_gate = updated[2] if len(updated) == 3 else None
        query_base = features.index_select(0, query_indices)
        query_labels = labels.index_select(0, query_indices)
        adapted_query = _apply_trainable(
            query_base,
            candidate=candidate,
            u=u,
            v=v,
            gamma=fast_gamma,
            beta=fast_beta,
            direction_gate=fast_gate,
            rho=rho_max * torch.sigmoid(raw_rho),
        )
        query_losses = frozen_prototype_losses(
            adapted_query, query_labels, frozen, scale=resolved_logit_scale
        )
        outer_loss = (
            query_losses.mean()
            + float(floor_weight)
            * smooth_class_floor_loss(query_losses, query_labels, temperature=0.1)
            + trust_region_loss(
                adapted_query, query_base, max_relative_move=trust_radius
            )
        )
        optimizer.zero_grad(set_to_none=True)
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
    with torch.no_grad():
        final_loss = float(objective().item())
    state = SlowFastAdapterState(
        candidate=candidate,
        slow_u=u,
        slow_v=v,
        rho=float((rho_max * torch.sigmoid(raw_rho)).detach().cpu().item()),
        gamma=gamma,
        beta=beta,
        direction_gate=direction_gate,
    )
    return state, {
        "steps": int(steps),
        "meta_steps": resolved_meta_steps,
        "inner_steps": int(inner_steps),
        "episode_k_counts": k_counts,
        "learned_rho": state.rho,
        "learned_fast_step_size": float(F.softplus(raw_step).detach().cpu().item()),
        "initial_loss": initial_loss,
        "stage1_final_loss": stage1_final_loss,
        "final_loss": final_loss,
        "feature_dim": cache.feature_dim,
        "rank": state.rank,
        "fast_parameter_count": state.fast_parameter_count,
        "support_logit_scale": resolved_logit_scale,
    }


__all__ = ["fit_common_shift_basis", "train_slow_fast_basis"]
