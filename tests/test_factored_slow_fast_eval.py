from __future__ import annotations

import torch
from torch.nn import functional as F

from cvsrffi.factored_slow_fast_eval import (
    build_nested_draws,
    generate_nested_predictions,
    meta_refine_factored_state,
    score_nested_predictions,
)
from cvsrffi.factored_slow_fast import fit_factored_state
from cvsrffi.slow_fast_cache import GroundFeatureCache


def _cache() -> GroundFeatureCache:
    rows: list[torch.Tensor] = []
    labels: list[int] = []
    receivers: list[str] = []
    days: list[str] = []
    scenes: list[str] = []
    physical_ids: list[str] = []
    views: list[str] = []
    roles: list[str] = []
    centers = F.normalize(torch.tensor([[0.0, 0.0, 1.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.1, 1.0, 0.2, 0.0]]), dim=1)
    for receiver_index, receiver in enumerate(("r0", "r1", "r2")):
        rx = torch.tensor([0.08 * (receiver_index - 1), 0.04 * receiver_index, 0.0, 0.0, 0.0, 0.0])
        for class_id in (0, 1):
            for sample in range(4):
                pid = f"{receiver}-{class_id}-{sample}"
                clean = F.normalize(centers[class_id] + rx + torch.tensor([0.0, 0.0, 0.0, 0.0, sample * 0.001, 0.0]), dim=0)
                shifts = {
                    "clean": torch.zeros(6),
                    "leo_clear_weak": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.05, 0.0]),
                    "leo_low_elev_weak": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.02, 0.07]),
                    "leo_rain_weak": torch.tensor([0.0, 0.0, 0.0, 0.0, -0.04, 0.08]),
                }
                for view, shift in shifts.items():
                    rows.append(F.normalize(clean + shift, dim=0))
                    labels.append(class_id)
                    receivers.append(receiver)
                    days.append("d0")
                    scenes.append(view)
                    physical_ids.append(pid)
                    views.append(view)
                    roles.append("L_s")
    return GroundFeatureCache(
        features=torch.stack(rows), labels=torch.tensor(labels), receivers=tuple(receivers), days=tuple(days),
        scenes=tuple(scenes), physical_sample_ids=tuple(physical_ids), views=tuple(views), roles=tuple(roles),
    )


def _prototypes() -> torch.Tensor:
    return F.normalize(torch.tensor([[0.0, 0.0, 1.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.1, 1.0, 0.2, 0.0]]), dim=1)


def test_multiple_draws_are_unique_and_physical_id_disjoint() -> None:
    episodes = build_nested_draws(_cache(), k_shot=1, draws=3, query_per_class=1, seed=392002)

    assert len(episodes) == 3 * 4 * 3
    assert {(episode.receiver, episode.scene, episode.draw_index) for episode in episodes} == {
        (receiver, scene, draw) for receiver in ("r0", "r1", "r2")
        for scene in ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak") for draw in range(3)
    }
    assert all(episode.support_ids.isdisjoint(episode.query_ids) for episode in episodes)


def test_meta_refinement_never_reads_excluded_receiver() -> None:
    cache = _cache()
    state, _ = fit_factored_state(cache, _prototypes(), torch.tensor([0, 1]), excluded_receiver="r2", rank_rx=2, rank_leo=2)
    refined, audit = meta_refine_factored_state(
        cache, state, excluded_receivers=("r2",), steps=2, k_shot=1, query_per_class=1, seed=7
    )

    assert audit["fit_receivers"] == ["r0", "r1"]
    assert audit["excluded_receivers"] == ["r2"]
    assert audit["outer_query_used"] is False
    assert torch.isfinite(refined.receiver_basis).all()
    assert torch.isfinite(refined.leo_basis).all()


def test_prediction_generation_is_truth_blind_and_scorer_joins_truth_last() -> None:
    cache = _cache()
    predictions = generate_nested_predictions(
        cache, _prototypes(), torch.tensor([10, 20]), k_shot=1, draws=2, query_per_class=1,
        seed=19, outer_receivers=("r2",), rank_rx=2, rank_leo=2, meta_steps=1,
        inner_ridge_grid=(0.1,),
    )

    assert predictions["schema"] == "cvs.factored_slow_fast.predictions.v1"
    assert predictions["query_truth_opened"] is False
    assert predictions["outer_receivers"] == ["r2"]
    assert predictions["states"] == ["A0", "B3", "B5"]
    assert all("true_class_id" not in row for row in predictions["rows"])
    assert all(set(row["scores"]) == {"A0", "B3", "B5"} for row in predictions["rows"])
    assert all(set(row["pseudo_new_scores"]) == {"B3", "B5"} for row in predictions["rows"])
    for row in predictions["rows"]:
        for strategy in ("B3", "B5"):
            for pseudo_new in row["pseudo_new_scores"][strategy].values():
                assert pseudo_new["query_ids"] == row["query_ids"]
                assert len(pseudo_new["da0_max"]) == len(row["query_ids"])
                assert len(pseudo_new["da1_max"]) == len(row["query_ids"])

    score = score_nested_predictions(predictions, cache)

    assert score["truth_opened_after_predictions_validated"] is True
    assert score["receiver_count"] == 1
    assert score["episode_count"] == 8
    assert set(score["strategy_summaries"]) == {"A0", "B3", "B5"}
    assert "max_pseudo_new_intrusion_delta" in score["strategy_summaries"]["B3"]
    assert score["selected_strategy"] in {"A0", "B3", "B5"}
