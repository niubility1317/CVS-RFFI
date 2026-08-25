from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cvsrffi.slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate
from cvsrffi.slow_fast_cache import GroundFeatureCache
from cvsrffi.slow_fast_calibration import (
    CALIBRATION_SCHEMA_KEYS,
    CalibrationCandidate,
    build_receiver_heldout_episodes,
    calibrate_p05_gate,
    save_calibration_json,
)
from cvsrffi.slow_fast_selection import SupportTrustPolicy


def _cache(*, samples_per_class: int = 12) -> tuple[GroundFeatureCache, torch.Tensor]:
    prototypes = torch.eye(8)[:2]
    rows = []
    labels = []
    receivers = []
    physical_ids = []
    for receiver in (0, 1, 2):
        receiver_shift = torch.zeros(8)
        receiver_shift[2 + receiver] = 0.2
        for class_id in range(2):
            for sample in range(samples_per_class):
                rows.append(prototypes[class_id] + receiver_shift)
                labels.append(class_id)
                receivers.append(receiver)
                physical_ids.append(f"rx{receiver}-c{class_id}-s{sample}")
    count = len(rows)
    return (
        GroundFeatureCache(
            features=torch.stack(rows),
            labels=torch.tensor(labels),
            receivers=tuple(receivers),
            days=tuple(0 for _ in range(count)),
            scenes=tuple("leo_clear_weak" for _ in range(count)),
            physical_sample_ids=tuple(physical_ids),
            views=tuple("leo_clear_weak" for _ in range(count)),
            roles=tuple("L_s" for _ in range(count)),
        ),
        prototypes,
    )


def _film_state() -> SlowFastAdapterState:
    return SlowFastAdapterState(
        candidate=SlowFastCandidate.FAST_FILM_R8,
        slow_u=torch.eye(8),
        slow_v=torch.flip(torch.eye(8), dims=(1,)),
        rho=0.1,
        gamma=torch.zeros(8),
        beta=torch.zeros(8),
    )


def _candidate(name: str, q90_move: float) -> CalibrationCandidate:
    return CalibrationCandidate(
        name=name,
        policy=SupportTrustPolicy(
            q90_move=q90_move,
            hard_move=0.5,
            q90_relative_move=10.0,
            minimum_positive_folds=1,
        ),
        lambda_grid=(0.0, 0.5, 1.0),
        repeats=1,
        steps=1,
        step_size=0.01,
    )


def test_receiver_heldout_episode_keeps_k10_and_physical_ids_disjoint() -> None:
    cache, _prototypes = _cache(samples_per_class=12)

    result = build_receiver_heldout_episodes(
        cache, k_shot=10, seed=17, scenes=("leo_clear_weak",)
    )

    assert len(result.episodes) == 3
    assert result.skipped == ()
    for episode in result:
        assert episode.support_ids.isdisjoint(episode.query_ids)
        assert torch.bincount(episode.support_labels, minlength=2).tolist() == [10, 10]
        assert episode.heldout_receiver not in episode.fit_receivers
        assert torch.bincount(episode.query_labels, minlength=2).tolist() == [2, 2]


def test_episode_builder_records_insufficient_receiver_without_lowering_k() -> None:
    cache, _prototypes = _cache(samples_per_class=10)

    result = build_receiver_heldout_episodes(
        cache, k_shot=10, seed=17, scenes=("leo_clear_weak",)
    )

    assert result.episodes == ()
    assert len(result.skipped) == 3
    assert all(item["reason"] == "INSUFFICIENT_K10_PLUS_QUERY" for item in result.skipped)


def test_source_only_calibration_freezes_one_film_policy_without_sample_rows() -> None:
    cache, prototypes = _cache(samples_per_class=4)
    episodes = build_receiver_heldout_episodes(
        cache, k_shot=2, seed=19, scenes=("leo_clear_weak",)
    )

    calibration = calibrate_p05_gate(
        episodes,
        (_candidate("wide", 0.20), _candidate("narrow", 0.10)),
        prototypes=prototypes,
        initial_state=_film_state(),
        logit_scale=8.0,
        seed=19,
    )

    assert set(calibration) == CALIBRATION_SCHEMA_KEYS
    assert calibration["candidate_id"] == "FAST_FILM_R8"
    assert calibration["target_support_used"] is False
    assert calibration["target_query_used"] is False
    assert calibration["selected_config"]["name"] in {"wide", "narrow"}
    assert "features" not in str(calibration)
    assert "physical_sample_ids" not in str(calibration)


def test_calibration_json_is_non_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    payload = {key: None for key in CALIBRATION_SCHEMA_KEYS}
    payload["schema"] = "cvs.slow_fast.p05.calibration.v1"

    save_calibration_json(path, payload)

    with pytest.raises(FileExistsError, match="already exists"):
        save_calibration_json(path, payload)
