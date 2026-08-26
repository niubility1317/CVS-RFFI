"""Nested source evaluation for the factored Slow-Fast analytic adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .factored_slow_fast import (
    FactoredSlowFastState,
    apply_factored_context,
    basis_scene_diagnostics,
    fit_factored_state,
    solve_factored_context,
    support_safety_diagnostics,
)
from .slow_fast_cache import GroundFeatureCache


_SCENES = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


@dataclass(frozen=True)
class NestedFeatureEpisode:
    receiver: object
    scene: str
    draw_index: int
    support_indices: Tensor
    query_indices: Tensor
    support_ids: frozenset[str]
    query_ids: frozenset[str]


def build_nested_draws(
    cache: GroundFeatureCache,
    *,
    k_shot: int,
    draws: int,
    query_per_class: int,
    seed: int,
    receivers: Sequence[object] | None = None,
    scenes: Sequence[str] = _SCENES,
) -> tuple[NestedFeatureEpisode, ...]:
    if int(k_shot) < 1 or int(draws) < 1 or int(query_per_class) < 1:
        raise ValueError("K, draws and query_per_class must be positive")
    selected_receivers = tuple(
        sorted(set(cache.receivers), key=str) if receivers is None else receivers
    )
    classes = tuple(sorted(int(value) for value in torch.unique(cache.labels).tolist()))
    episodes: list[NestedFeatureEpisode] = []
    for receiver in selected_receivers:
        for scene in tuple(str(value) for value in scenes):
            pools = {
                class_id: [
                    index
                    for index, (row_receiver, row_scene, label) in enumerate(
                        zip(cache.receivers, cache.scenes, cache.labels.tolist())
                    )
                    if row_receiver == receiver and row_scene == scene and int(label) == class_id
                ]
                for class_id in classes
            }
            if any(len(pool) < int(k_shot) + int(query_per_class) for pool in pools.values()):
                raise ValueError(f"receiver={receiver} scene={scene} lacks K plus query rows")
            signatures: set[tuple[str, ...]] = set()
            for draw_index in range(int(draws)):
                chosen_support: list[int] = []
                chosen_query: list[int] = []
                for attempt in range(1000):
                    chosen_support.clear()
                    chosen_query.clear()
                    for class_id in classes:
                        pool = list(pools[class_id])
                        rng = random.Random(
                            f"{int(seed)}|{receiver!s}|{scene}|{draw_index}|{attempt}|{class_id}"
                        )
                        rng.shuffle(pool)
                        chosen_support.extend(pool[: int(k_shot)])
                        chosen_query.extend(
                            pool[int(k_shot) : int(k_shot) + int(query_per_class)]
                        )
                    signature = tuple(
                        sorted(cache.physical_sample_ids[index] for index in chosen_support)
                    )
                    if signature not in signatures or len(signatures) >= math.prod(
                        math.comb(len(pools[class_id]), int(k_shot)) for class_id in classes
                    ):
                        signatures.add(signature)
                        break
                support_ids = frozenset(cache.physical_sample_ids[index] for index in chosen_support)
                query_ids = frozenset(cache.physical_sample_ids[index] for index in chosen_query)
                if len(support_ids) != len(chosen_support) or len(query_ids) != len(chosen_query) or not support_ids.isdisjoint(query_ids):
                    raise ValueError("nested draw physical IDs are duplicated or overlapping")
                episodes.append(
                    NestedFeatureEpisode(
                        receiver=receiver,
                        scene=scene,
                        draw_index=draw_index,
                        support_indices=torch.tensor(chosen_support, dtype=torch.long),
                        query_indices=torch.tensor(chosen_query, dtype=torch.long),
                        support_ids=support_ids,
                        query_ids=query_ids,
                    )
                )
    return tuple(episodes)


def _orthogonal_bases(receiver: Tensor, leo: Tensor) -> tuple[Tensor, Tensor]:
    rx, _ = torch.linalg.qr(receiver, mode="reduced")
    leo_residual = leo - rx @ (rx.transpose(0, 1) @ leo)
    leo_orthogonal, _ = torch.linalg.qr(leo_residual, mode="reduced")
    return rx, leo_orthogonal


def _analytic_context_mean(
    support: Tensor,
    labels: Tensor,
    state: FactoredSlowFastState,
    receiver_basis: Tensor,
    leo_basis: Tensor,
) -> Tensor:
    basis = torch.cat((receiver_basis, leo_basis), dim=1)
    ridge = torch.cat(
        (
            torch.full((receiver_basis.shape[1],), state.ridge_receiver, device=support.device),
            torch.full((leo_basis.shape[1],), state.ridge_leo, device=support.device),
        )
    )
    system = basis.transpose(0, 1) @ basis + torch.diag(ridge)
    row_by_id = {int(value): row for row, value in enumerate(state.class_ids.tolist())}
    codes: list[Tensor] = []
    for class_id in sorted(int(value) for value in torch.unique(labels).tolist()):
        residual = support[labels == class_id].mean(dim=0) - state.geometric_centers[row_by_id[class_id]].to(support)
        codes.append(torch.linalg.solve(system, basis.transpose(0, 1) @ residual))
    return torch.stack(codes).mean(dim=0)


def meta_refine_factored_state(
    cache: GroundFeatureCache,
    state: FactoredSlowFastState,
    *,
    excluded_receivers: Iterable[object],
    steps: int,
    k_shot: int,
    query_per_class: int,
    seed: int,
    learning_rate: float = 5.0e-3,
) -> tuple[FactoredSlowFastState, dict[str, Any]]:
    excluded = {str(value) for value in excluded_receivers}
    fit_receivers = sorted(
        {str(value) for value in cache.receivers if str(value) not in excluded}
    )
    if not fit_receivers or int(steps) < 0:
        raise ValueError("meta refinement requires fit receivers and nonnegative steps")
    if int(steps) == 0:
        return state, {
            "fit_receivers": fit_receivers,
            "excluded_receivers": sorted(excluded),
            "steps": 0,
            "outer_query_used": False,
        }
    episodes = build_nested_draws(
        cache,
        k_shot=int(k_shot),
        draws=1,
        query_per_class=int(query_per_class),
        seed=int(seed),
        receivers=tuple(value for value in sorted(set(cache.receivers), key=str) if str(value) in fit_receivers),
    )
    receiver_basis = torch.nn.Parameter(state.receiver_basis.clone())
    leo_basis = torch.nn.Parameter(state.leo_basis.clone())
    optimizer = torch.optim.Adam((receiver_basis, leo_basis), lr=float(learning_rate))
    generator = random.Random(int(seed))
    losses: list[float] = []
    for _step in range(int(steps)):
        episode = episodes[generator.randrange(len(episodes))]
        support = cache.features.index_select(0, episode.support_indices).float()
        support_label_rows = cache.labels.index_select(0, episode.support_indices).long()
        support_labels = state.class_ids.index_select(0, support_label_rows)
        query = cache.features.index_select(0, episode.query_indices).float()
        query_labels = cache.labels.index_select(0, episode.query_indices).long()
        rx, leo = _orthogonal_bases(receiver_basis, leo_basis)
        context = _analytic_context_mean(support, support_labels, state, rx, leo)
        basis = torch.cat((rx, leo), dim=1)
        adapted = F.normalize(query - (basis @ context).unsqueeze(0), dim=1)
        scores = adapted @ state.decision_prototypes.to(adapted).transpose(0, 1)
        ce = F.cross_entropy(8.0 * scores, query_labels)
        class_losses = torch.stack(
            [F.cross_entropy(8.0 * scores[query_labels == class_id], query_labels[query_labels == class_id]) for class_id in torch.unique(query_labels)]
        )
        margins = scores.gather(1, query_labels[:, None]).squeeze(1) - scores.masked_fill(
            F.one_hot(query_labels, num_classes=scores.shape[1]).bool(), float("-inf")
        ).max(dim=1).values
        heldout_row = int(_step % state.class_ids.numel())
        heldout_external = int(state.class_ids[heldout_row])
        old_support = support_labels != heldout_external
        pseudo_new_penalty = scores.sum() * 0.0
        if bool(old_support.any()) and bool((query_labels == heldout_row).any()):
            old_context = _analytic_context_mean(
                support[old_support], support_labels[old_support], state, rx, leo
            )
            heldout_query = query[query_labels == heldout_row]
            old_prototypes = state.decision_prototypes[
                torch.arange(state.class_ids.numel()) != heldout_row
            ].to(heldout_query)
            baseline_intrusion = (F.normalize(heldout_query, dim=1) @ old_prototypes.T).max(dim=1).values
            adapted_heldout = F.normalize(
                heldout_query - (basis @ old_context).unsqueeze(0), dim=1
            )
            adapted_intrusion = (adapted_heldout @ old_prototypes.T).max(dim=1).values
            pseudo_new_penalty = F.relu(adapted_intrusion - baseline_intrusion).mean()
        outer_loss = (
            ce
            + 0.2 * class_losses.max()
            + 0.1 * F.relu(0.1 - margins).mean()
            + 0.05 * pseudo_new_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_((receiver_basis, leo_basis), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            normalized_rx, normalized_leo = _orthogonal_bases(receiver_basis, leo_basis)
            receiver_basis.copy_(normalized_rx)
            leo_basis.copy_(normalized_leo)
        losses.append(float(outer_loss.detach()))
    refined = replace(
        state,
        receiver_basis=receiver_basis.detach(),
        leo_basis=leo_basis.detach(),
    )
    return refined, {
        "schema": "cvs.factored_slow_fast.meta_fit.v1",
        "fit_receivers": fit_receivers,
        "excluded_receivers": sorted(excluded),
        "steps": int(steps),
        "initial_outer_loss": losses[0],
        "final_outer_loss": losses[-1],
        "outer_query_used": False,
    }


def _inner_select_ridge(
    cache: GroundFeatureCache,
    prototypes: Tensor,
    class_ids: Tensor,
    *,
    outer_receiver: object,
    rank_rx: int,
    rank_leo: int,
    ridge_grid: Sequence[float],
    k_shot: int,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    grid = tuple(float(value) for value in ridge_grid)
    if not grid or any(value <= 0.0 or not math.isfinite(value) for value in grid):
        raise ValueError("inner ridge grid must be finite and positive")
    if len(grid) == 1:
        return grid[0], {"protocol": "single_preregistered_ridge", "ridge": grid[0]}
    training_receivers = [value for value in sorted(set(cache.receivers), key=str) if value != outer_receiver]
    gains = {ridge: [] for ridge in grid}
    for inner_receiver in training_receivers:
        state, _ = fit_factored_state(
            cache,
            prototypes,
            class_ids,
            excluded_receivers=(outer_receiver, inner_receiver),
            rank_rx=rank_rx,
            rank_leo=rank_leo,
            ridge_receiver=grid[0],
            ridge_leo=grid[0],
        )
        episodes = build_nested_draws(
            cache,
            k_shot=k_shot,
            draws=1,
            query_per_class=1,
            seed=seed,
            receivers=(inner_receiver,),
        )
        for ridge in grid:
            candidate = replace(state, ridge_receiver=ridge, ridge_leo=ridge)
            for episode in episodes:
                support = cache.features.index_select(0, episode.support_indices)
                label_rows = cache.labels.index_select(0, episode.support_indices)
                labels = state.class_ids.index_select(0, label_rows)
                query = cache.features.index_select(0, episode.query_indices)
                query_labels = cache.labels.index_select(0, episode.query_indices)
                context, _ = solve_factored_context(support, labels, candidate)
                baseline = (F.normalize(query, dim=1) @ candidate.decision_prototypes.T).argmax(dim=1)
                adapted = (apply_factored_context(query, candidate, context) @ candidate.decision_prototypes.T).argmax(dim=1)
                gains[ridge].append(float((adapted == query_labels).float().mean() - (baseline == query_labels).float().mean()))
    selected = max(grid, key=lambda value: (sum(gains[value]) / len(gains[value]), -value))
    return selected, {
        "protocol": "nested_inner_receiver_loro",
        "ridge": selected,
        "mean_gain_by_ridge": {str(value): sum(gains[value]) / len(gains[value]) for value in grid},
    }


def _safe_context(
    support: Tensor,
    labels: Tensor,
    state: FactoredSlowFastState,
) -> tuple[Tensor, dict[str, Any]]:
    context, context_audit = solve_factored_context(support, labels, state)
    row_by_id = {int(value): row for row, value in enumerate(state.class_ids.tolist())}
    label_rows = torch.tensor([row_by_id[int(value)] for value in labels.tolist()])
    last: dict[str, Any] | None = None
    scales = (0.0,) if float(context_audit["support_shift_norm"]) <= 0.01 else (1.0, 0.75, 0.5, 0.25, 0.0)
    for scale in scales:
        scaled = context * scale
        adapted = apply_factored_context(support, state, scaled)
        last = support_safety_diagnostics(
            F.normalize(support, dim=1),
            adapted,
            label_rows,
            state.decision_prototypes,
            coverage=float(context_audit["coverage"]) * scale * scale,
            disagreement=float(context_audit["class_code_disagreement"]) * scale,
            min_coverage=0.05 if scale > 0.0 else 0.0,
            max_disagreement=1.0,
            min_correct_margin_q10=0.5,
            min_wrong_margin_median=0.0,
            min_class_margin_cvar=-0.05,
        )
        if scale == 0.0 or last["safe_to_commit"]:
            return scaled, {**context_audit, **last, "selected_context_scale": scale}
    raise AssertionError("zero context must be a safe fallback")


def generate_nested_predictions(
    cache: GroundFeatureCache,
    decision_prototypes: Tensor,
    class_ids: Tensor,
    *,
    k_shot: int,
    draws: int,
    query_per_class: int,
    seed: int,
    outer_receivers: Sequence[object] | None = None,
    rank_rx: int = 4,
    rank_leo: int = 4,
    meta_steps: int = 50,
    inner_ridge_grid: Sequence[float] = (0.03, 0.1, 0.3),
) -> dict[str, Any]:
    receivers = tuple(sorted(set(cache.receivers), key=str) if outer_receivers is None else outer_receivers)
    episodes = build_nested_draws(
        cache,
        k_shot=k_shot,
        draws=draws,
        query_per_class=query_per_class,
        seed=seed,
        receivers=receivers,
    )
    rows: list[dict[str, Any]] = []
    outer_audits: dict[str, Any] = {}
    for outer in receivers:
        ridge, ridge_audit = _inner_select_ridge(
            cache,
            decision_prototypes,
            class_ids,
            outer_receiver=outer,
            rank_rx=rank_rx,
            rank_leo=rank_leo,
            ridge_grid=inner_ridge_grid,
            k_shot=k_shot,
            seed=seed,
        )
        b3, fit_audit = fit_factored_state(
            cache,
            decision_prototypes,
            class_ids,
            excluded_receiver=outer,
            rank_rx=rank_rx,
            rank_leo=rank_leo,
            ridge_receiver=ridge,
            ridge_leo=ridge,
        )
        b5, meta_audit = meta_refine_factored_state(
            cache,
            b3,
            excluded_receivers=(outer,),
            steps=meta_steps,
            k_shot=k_shot,
            query_per_class=query_per_class,
            seed=seed + 1009,
        )
        outer_audits[str(outer)] = {
            "ridge_selection": ridge_audit,
            "fit": fit_audit,
            "meta": meta_audit,
            "basis_diagnostics": basis_scene_diagnostics(cache, b3, excluded_receiver=outer),
        }
        for episode in (value for value in episodes if value.receiver == outer):
            support = cache.features.index_select(0, episode.support_indices)
            support_label_rows = cache.labels.index_select(0, episode.support_indices)
            query = cache.features.index_select(0, episode.query_indices)
            baseline = F.normalize(query, dim=1)
            score_map: dict[str, list[list[float]]] = {
                "A0": (baseline @ F.normalize(decision_prototypes, dim=1).T).tolist()
            }
            support_audits: dict[str, Any] = {}
            pseudo_new: dict[str, Any] = {}
            for name, state in (("B3", b3), ("B5", b5)):
                support_labels = state.class_ids.index_select(0, support_label_rows)
                context, context_audit = _safe_context(support, support_labels, state)
                score_map[name] = (
                    apply_factored_context(query, state, context) @ state.decision_prototypes.T
                ).tolist()
                support_audits[name] = {
                    key: value
                    for key, value in context_audit.items()
                    if key != "per_class_codes"
                }
                per_class_intrusion: dict[str, Any] = {}
                for heldout_row in sorted(int(value) for value in torch.unique(support_label_rows).tolist()):
                    heldout_class = int(state.class_ids[heldout_row])
                    old_mask = support_label_rows != heldout_row
                    if not bool(old_mask.any()):
                        continue
                    old_state_rows = state.class_ids != heldout_class
                    old_state = replace(
                        state,
                        geometric_centers=state.geometric_centers[old_state_rows],
                        decision_prototypes=state.decision_prototypes[old_state_rows],
                        class_ids=state.class_ids[old_state_rows],
                    )
                    old_context, _ = solve_factored_context(
                        support[old_mask], support_labels[old_mask], old_state
                    )
                    heldout_query = query
                    da0_max = (
                        F.normalize(heldout_query, dim=1) @ old_state.decision_prototypes.T
                    ).max(dim=1).values
                    da1_max = (
                        apply_factored_context(heldout_query, old_state, old_context)
                        @ old_state.decision_prototypes.T
                    ).max(dim=1).values
                    per_class_intrusion[str(heldout_class)] = {
                        "heldout_class_id": heldout_class,
                        "query_ids": [
                            cache.physical_sample_ids[index]
                            for index in episode.query_indices.tolist()
                        ],
                        "da0_max": da0_max.tolist(),
                        "da1_max": da1_max.tolist(),
                    }
                pseudo_new[name] = per_class_intrusion
            rows.append(
                {
                    "receiver": str(outer),
                    "scene": episode.scene,
                    "draw_index": episode.draw_index,
                    "support_ids": sorted(episode.support_ids),
                    "query_ids": [cache.physical_sample_ids[index] for index in episode.query_indices.tolist()],
                    "scores": score_map,
                    "support_audits": support_audits,
                    "pseudo_new_scores": pseudo_new,
                }
            )
    return {
        "schema": "cvs.factored_slow_fast.predictions.v1",
        "states": ["A0", "B3", "B5"],
        "registered_class_ids": [int(value) for value in class_ids.tolist()],
        "outer_receivers": [str(value) for value in receivers],
        "k_shot": int(k_shot),
        "draws": int(draws),
        "query_per_class": int(query_per_class),
        "seed": int(seed),
        "query_truth_opened": False,
        "target_support_used": False,
        "target_query_used": False,
        "query_state_update_count": 0,
        "outer_audits": outer_audits,
        "rows": rows,
    }


def _macro_metrics(predictions: Tensor, labels: Tensor, class_ids: Tensor) -> tuple[float, float, dict[int, float]]:
    per_class = {
        int(class_id): float((predictions[labels == int(class_id)] == labels[labels == int(class_id)]).float().mean())
        for class_id in class_ids.tolist()
        if bool((labels == int(class_id)).any())
    }
    return sum(per_class.values()) / len(per_class), min(per_class.values()), per_class


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0

    def ranks(values: Sequence[float]) -> Tensor:
        result = torch.empty(len(values), dtype=torch.float64)
        ordered = sorted(range(len(values)), key=lambda index: values[index])
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
                end += 1
            rank = 0.5 * (start + end - 1)
            for position in range(start, end):
                result[ordered[position]] = rank
            start = end
        return result

    left_rank = ranks(left)
    right_rank = ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denominator) <= 1.0e-12:
        return 0.0
    return float(torch.dot(left_centered, right_centered) / denominator)


def score_nested_predictions(
    predictions: Mapping[str, Any],
    cache: GroundFeatureCache,
    *,
    worst_receiver_epsilon_pp: float = 0.5,
    worst_floor_epsilon_pp: float = 5.0,
    worst_class_epsilon_pp: float = 5.0,
    pseudo_new_intrusion_epsilon: float = 0.0,
) -> dict[str, Any]:
    if predictions.get("schema") != "cvs.factored_slow_fast.predictions.v1" or predictions.get("query_truth_opened") is not False:
        raise ValueError("prediction artifact schema/truth boundary mismatch")
    states = tuple(predictions.get("states", ()))
    if states != ("A0", "B3", "B5"):
        raise ValueError("prediction states must be A0/B3/B5")
    registered_class_ids = tuple(int(value) for value in predictions.get("registered_class_ids", ()))
    if not registered_class_ids or len(set(registered_class_ids)) != len(registered_class_ids):
        raise ValueError("prediction registered class IDs are invalid")
    row_by_external_id = {class_id: row for row, class_id in enumerate(registered_class_ids)}
    label_by_id: dict[str, int] = {}
    for physical_id, label in zip(cache.physical_sample_ids, cache.labels.tolist()):
        prior = label_by_id.setdefault(physical_id, int(label))
        if prior != int(label):
            raise ValueError("physical sample ID has inconsistent source truth")
    validated_rows: list[tuple[Mapping[str, Any], Tensor, dict[str, Tensor]]] = []
    for row in predictions.get("rows", []):
        query_ids = row.get("query_ids")
        if not isinstance(query_ids, list) or not query_ids or any(value not in label_by_id for value in query_ids):
            raise ValueError("prediction query IDs do not close against source truth")
        labels = torch.tensor([label_by_id[value] for value in query_ids], dtype=torch.long)
        score_tensors = {name: torch.tensor(row["scores"][name], dtype=torch.float32) for name in states}
        if any(value.ndim != 2 or value.shape[0] != labels.numel() or not bool(torch.isfinite(value).all()) for value in score_tensors.values()):
            raise ValueError("prediction score rows are invalid")
        validated_rows.append((row, labels, score_tensors))
    if not validated_rows:
        raise ValueError("prediction artifact contains no rows")
    class_ids = torch.unique(cache.labels).sort().values
    summaries: dict[str, Any] = {}
    for state_name in states:
        episode_rows: list[dict[str, Any]] = []
        intrusion_values: list[float] = []
        receiver_deltas: dict[str, list[float]] = {}
        support_utilities: list[float] = []
        query_utilities: list[float] = []
        for row, labels, score_tensors in validated_rows:
            base_predictions = score_tensors["A0"].argmax(dim=1)
            state_predictions = score_tensors[state_name].argmax(dim=1)
            base_mean, base_floor, base_classes = _macro_metrics(base_predictions, labels, class_ids)
            mean, floor, classes = _macro_metrics(state_predictions, labels, class_ids)
            class_deltas = {str(key): (classes[key] - base_classes[key]) * 100.0 for key in classes}
            mean_delta = (mean - base_mean) * 100.0
            receiver_deltas.setdefault(str(row["receiver"]), []).append(mean_delta)
            if state_name != "A0":
                support_utilities.append(float(row["support_audits"][state_name]["support_utility"]))
                query_utilities.append(mean_delta)
                for pseudo in row["pseudo_new_scores"][state_name].values():
                    heldout_class_id = int(pseudo["heldout_class_id"])
                    if heldout_class_id not in row_by_external_id:
                        raise ValueError("pseudo-new heldout class is not registered")
                    pseudo_query_ids = pseudo["query_ids"]
                    if pseudo_query_ids != row["query_ids"]:
                        raise ValueError("pseudo-new prediction rows must cover the full truth-blind query")
                    heldout_row = row_by_external_id[heldout_class_id]
                    intrusion_values.extend(
                        float(after - before)
                        for query_id, before, after in zip(
                            pseudo_query_ids, pseudo["da0_max"], pseudo["da1_max"]
                        )
                        if label_by_id[query_id] == heldout_row
                    )
            episode_rows.append(
                {
                    "receiver": str(row["receiver"]),
                    "scene": row["scene"],
                    "draw_index": int(row["draw_index"]),
                    "mean_delta_pp": mean_delta,
                    "floor_delta_pp": (floor - base_floor) * 100.0,
                    "class_delta_pp": class_deltas,
                }
            )
        receiver_means = {key: sum(values) / len(values) for key, values in receiver_deltas.items()}
        receiver_values = torch.tensor(list(receiver_means.values()), dtype=torch.float32)
        lcb = float(receiver_values.mean())
        if receiver_values.numel() > 1:
            lcb -= 1.2815515655446004 * float(receiver_values.std(unbiased=True)) / math.sqrt(receiver_values.numel())
        max_intrusion = max(intrusion_values) if intrusion_values else 0.0
        support_query_spearman = _spearman(support_utilities, query_utilities) if state_name != "A0" else 0.0
        worst_floor = min(value["floor_delta_pp"] for value in episode_rows)
        worst_class = min(min(value["class_delta_pp"].values()) for value in episode_rows)
        worst_receiver = min(receiver_means.values())
        feasible = (
            worst_receiver >= -float(worst_receiver_epsilon_pp)
            and worst_floor >= -float(worst_floor_epsilon_pp)
            and worst_class >= -float(worst_class_epsilon_pp)
            and max_intrusion <= float(pseudo_new_intrusion_epsilon) + 1.0e-8
            and (state_name == "A0" or support_query_spearman >= 0.2)
        )
        summaries[state_name] = {
            "mean_delta_pp": sum(value["mean_delta_pp"] for value in episode_rows) / len(episode_rows),
            "mean_floor_delta_pp": sum(value["floor_delta_pp"] for value in episode_rows) / len(episode_rows),
            "worst_receiver_mean_delta_pp": worst_receiver,
            "worst_episode_floor_delta_pp": worst_floor,
            "worst_episode_class_delta_pp": worst_class,
            "max_pseudo_new_intrusion_delta": max_intrusion,
            "receiver_mean_lcb90_pp": lcb,
            "support_query_spearman": support_query_spearman,
            "feasible": bool(feasible),
            "episode_summaries": episode_rows,
            "receiver_summaries": receiver_means,
            "deployment_fast_parameter_count": 0 if state_name == "A0" else int(
                next(iter(predictions["outer_audits"].values()))["fit"]["fast_parameter_count"]
            ),
            "query_updates": 0,
        }
    feasible_names = [name for name in states if summaries[name]["feasible"]]
    selected = max(
        feasible_names,
        key=lambda name: (
            summaries[name]["receiver_mean_lcb90_pp"] + 0.2 * summaries[name]["mean_floor_delta_pp"],
            -summaries[name]["max_pseudo_new_intrusion_delta"],
            -summaries[name]["deployment_fast_parameter_count"],
            name,
        ),
    )
    return {
        "schema": "cvs.factored_slow_fast.score.v1",
        "truth_opened_after_predictions_validated": True,
        "receiver_count": len(predictions["outer_receivers"]),
        "episode_count": len(validated_rows),
        "strategy_summaries": summaries,
        "selected_strategy": selected,
        "status": "SOURCE_NESTED_ADAPTATION_SELECTED" if selected != "A0" else "SOURCE_NESTED_CALIBRATED_TO_ABSTAIN",
        "target_performance_status": "UNKNOWN_MISSING_INDEPENDENT_TARGET_CAPSULE",
    }


__all__ = [
    "NestedFeatureEpisode",
    "build_nested_draws",
    "generate_nested_predictions",
    "meta_refine_factored_state",
    "score_nested_predictions",
]
