"""Source-only receiver-held-out calibration for the Slow-Fast P0.5 gate."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate, apply_slow_fast
from .slow_fast_cache import GroundFeatureCache
from .slow_fast_selection import SupportTrustPolicy, select_support_only_state


CALIBRATION_SCHEMA = "cvs.slow_fast.p05.calibration.v1"
CALIBRATION_SCHEMA_KEYS = frozenset(
    {
        "schema",
        "status",
        "candidate_id",
        "source_role",
        "calibration_protocol",
        "k_shot",
        "seed",
        "source_receiver_count",
        "episode_count",
        "skipped_episodes",
        "target_support_used",
        "target_query_used",
        "selected_config",
        "config_summaries",
        "deployment_fields",
    }
)


@dataclass(frozen=True)
class ReceiverHeldoutEpisode:
    heldout_receiver: object
    fit_receivers: tuple[object, ...]
    scene: str
    support_features: Tensor
    support_labels: Tensor
    support_ids: frozenset[str]
    query_features: Tensor
    query_labels: Tensor
    query_ids: frozenset[str]


@dataclass(frozen=True)
class ReceiverHeldoutEpisodeSet:
    episodes: tuple[ReceiverHeldoutEpisode, ...]
    skipped: tuple[Mapping[str, Any], ...]

    def __iter__(self) -> Iterator[ReceiverHeldoutEpisode]:
        return iter(self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)


@dataclass(frozen=True)
class CalibrationCandidate:
    name: str
    policy: SupportTrustPolicy
    lambda_grid: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    repeats: int = 3
    steps: int = 3
    step_size: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("calibration candidate name must be nonempty")
        grid = tuple(float(value) for value in self.lambda_grid)
        if not grid or tuple(sorted(set(grid))) != grid or grid[0] != 0.0 or grid[-1] > 1.0:
            raise ValueError("calibration lambda_grid must be sorted, unique and include zero")
        if int(self.repeats) < 1 or int(self.steps) < 1:
            raise ValueError("calibration repeats and steps must be positive")
        if not math.isfinite(float(self.step_size)) or float(self.step_size) <= 0.0:
            raise ValueError("calibration step_size must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy": {
                "q90_move": float(self.policy.q90_move),
                "hard_move": float(self.policy.hard_move),
                "q90_relative_move": float(self.policy.q90_relative_move),
                "minimum_positive_folds": int(self.policy.minimum_positive_folds),
                "lcb_z": float(self.policy.lcb_z),
                "require_fold_lcb": bool(self.policy.require_fold_lcb),
            },
            "lambda_grid": [float(value) for value in self.lambda_grid],
            "repeats": int(self.repeats),
            "steps": int(self.steps),
            "step_size": float(self.step_size),
        }


def default_p05_candidates(trust_radius: float) -> tuple[CalibrationCandidate, ...]:
    """Return the preregistered P0.5 rule comparison for source calibration."""

    hard = float(trust_radius)
    if not math.isfinite(hard) or hard <= 0.0 or hard >= 2.0:
        raise ValueError("trust_radius must lie in (0, 2)")
    q90 = min(hard, 0.75 * hard)
    common = {
        "q90_move": q90,
        "hard_move": hard,
    }
    return (
        CalibrationCandidate(
            name="P05_Q90_HARD",
            policy=SupportTrustPolicy(
                **common,
                q90_relative_move=1.0e6,
                minimum_positive_folds=0,
                require_fold_lcb=False,
            ),
        ),
        CalibrationCandidate(
            name="P05_RELATIVE_K12",
            policy=SupportTrustPolicy(
                **common,
                q90_relative_move=1.2,
                minimum_positive_folds=0,
                require_fold_lcb=False,
            ),
        ),
        CalibrationCandidate(
            name="P05_RELATIVE_K08",
            policy=SupportTrustPolicy(
                **common,
                q90_relative_move=0.8,
                minimum_positive_folds=0,
                require_fold_lcb=False,
            ),
        ),
        CalibrationCandidate(
            name="P05_RELATIVE_K08_FOLD_LCB",
            policy=SupportTrustPolicy(
                **common,
                q90_relative_move=0.8,
                minimum_positive_folds=5,
                require_fold_lcb=True,
            ),
        ),
    )


def build_receiver_heldout_episodes(
    cache: GroundFeatureCache,
    *,
    k_shot: int,
    seed: int,
    scenes: Sequence[str] = (
        "clean",
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    ),
) -> ReceiverHeldoutEpisodeSet:
    """Create source receiver-held-out K-shot/query episodes without lowering K."""

    if not isinstance(cache, GroundFeatureCache):
        raise TypeError("cache must be GroundFeatureCache")
    if int(k_shot) < 1:
        raise ValueError("k_shot must be positive")
    requested_scenes = tuple(str(value) for value in scenes)
    if not requested_scenes or len(set(requested_scenes)) != len(requested_scenes):
        raise ValueError("scenes must be a nonempty unique sequence")
    classes = tuple(sorted(int(value) for value in torch.unique(cache.labels).tolist()))
    if classes != tuple(range(len(classes))):
        raise ValueError("ground cache labels must be contiguous prototype rows")
    receivers = tuple(sorted(set(cache.receivers), key=str))
    episodes: list[ReceiverHeldoutEpisode] = []
    skipped: list[Mapping[str, Any]] = []
    for heldout in receivers:
        fit_receivers = tuple(value for value in receivers if value != heldout)
        for scene in requested_scenes:
            selected_support: list[int] = []
            selected_query: list[int] = []
            insufficient = False
            for class_id in classes:
                pool = [
                    index
                    for index, (receiver, row_scene, label) in enumerate(
                        zip(cache.receivers, cache.scenes, cache.labels.tolist())
                    )
                    if receiver == heldout and row_scene == scene and int(label) == class_id
                ]
                if len(pool) < int(k_shot) + 1:
                    insufficient = True
                    break
                rng = random.Random(f"{int(seed)}|{heldout!s}|{scene}|{class_id}")
                rng.shuffle(pool)
                selected_support.extend(pool[: int(k_shot)])
                selected_query.extend(pool[int(k_shot) :])
            if insufficient:
                skipped.append(
                    {
                        "heldout_receiver": heldout,
                        "scene": scene,
                        "reason": f"INSUFFICIENT_K{int(k_shot)}_PLUS_QUERY",
                    }
                )
                continue
            support_ids = frozenset(cache.physical_sample_ids[index] for index in selected_support)
            query_ids = frozenset(cache.physical_sample_ids[index] for index in selected_query)
            if len(support_ids) != len(selected_support) or len(query_ids) != len(selected_query):
                raise ValueError("episode contains duplicate physical sample IDs")
            if not support_ids.isdisjoint(query_ids):
                raise ValueError("episode support/query physical sample IDs overlap")
            support_index = torch.tensor(selected_support, dtype=torch.long, device=cache.features.device)
            query_index = torch.tensor(selected_query, dtype=torch.long, device=cache.features.device)
            episodes.append(
                ReceiverHeldoutEpisode(
                    heldout_receiver=heldout,
                    fit_receivers=fit_receivers,
                    scene=scene,
                    support_features=cache.features.index_select(0, support_index),
                    support_labels=cache.labels.index_select(0, support_index),
                    support_ids=support_ids,
                    query_features=cache.features.index_select(0, query_index),
                    query_labels=cache.labels.index_select(0, query_index),
                    query_ids=query_ids,
                )
            )
    return ReceiverHeldoutEpisodeSet(tuple(episodes), tuple(skipped))


def _query_metrics(features: Tensor, labels: Tensor, prototypes: Tensor) -> tuple[float, float, Tensor]:
    scores = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
    predictions = scores.argmax(dim=1)
    per_class = [
        float((predictions[labels == class_id] == labels[labels == class_id]).float().mean())
        for class_id in range(int(prototypes.shape[0]))
    ]
    return float(sum(per_class) / len(per_class)), float(min(per_class)), scores


def calibrate_p05_gate(
    episodes: ReceiverHeldoutEpisodeSet | Sequence[ReceiverHeldoutEpisode],
    candidates: Sequence[CalibrationCandidate],
    *,
    prototypes: Tensor,
    initial_state: SlowFastAdapterState,
    logit_scale: float,
    seed: int,
) -> dict[str, Any]:
    """Freeze one FILM P0.5 rule using source held-out query only."""

    episode_rows = tuple(episodes.episodes if isinstance(episodes, ReceiverHeldoutEpisodeSet) else episodes)
    skipped = tuple(episodes.skipped if isinstance(episodes, ReceiverHeldoutEpisodeSet) else ())
    if not episode_rows:
        raise ValueError("source calibration requires at least one receiver-held-out episode")
    if not candidates or len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("calibration candidates must be nonempty and uniquely named")
    if initial_state.candidate is not SlowFastCandidate.FAST_FILM_R8:
        raise ValueError("P0.5 calibration freezes FAST_FILM_R8 only")
    prototypes = F.normalize(prototypes.detach().float(), dim=1)
    if prototypes.shape[1] != initial_state.feature_dim:
        raise ValueError("calibration prototypes and FILM state width mismatch")

    config_summaries: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        evaluations: list[dict[str, Any]] = []
        for episode_index, episode in enumerate(episode_rows):
            selected, audit = select_support_only_state(
                episode.support_features,
                episode.support_labels,
                prototypes,
                initial_state,
                k_shot=int(torch.bincount(episode.support_labels)[0]),
                logit_scale=float(logit_scale),
                trust_radius=float(candidate.policy.hard_move),
                steps=int(candidate.steps),
                step_size=float(candidate.step_size),
                lambda_grid=candidate.lambda_grid,
                crossfit_seed=int(seed) + candidate_index * 1009 + episode_index,
                repeats=int(candidate.repeats),
                physical_ids=tuple(sorted(episode.support_ids)),
                trust_policy=candidate.policy,
            )
            baseline_features = F.normalize(episode.query_features, dim=1)
            adapted_features = (
                baseline_features
                if float(selected.rho) == 0.0
                else apply_slow_fast(episode.query_features, selected)
            )
            baseline_mean, baseline_floor, baseline_scores = _query_metrics(
                baseline_features, episode.query_labels, prototypes
            )
            adapted_mean, adapted_floor, adapted_scores = _query_metrics(
                adapted_features, episode.query_labels, prototypes
            )
            evaluations.append(
                {
                    "heldout_receiver": str(episode.heldout_receiver),
                    "scene": episode.scene,
                    "mean_delta_pp": float((adapted_mean - baseline_mean) * 100.0),
                    "floor_delta_pp": float((adapted_floor - baseline_floor) * 100.0),
                    "confidence_intrusion_proxy": float(
                        (adapted_scores.max(dim=1).values - baseline_scores.max(dim=1).values).mean()
                    ),
                    "selected_lambda": float(audit["selected_lambda"]),
                    "selected_effective_lambda": float(audit["selected_effective_lambda"]),
                    "crossfit_fit_count": int(audit["crossfit_fit_count"]),
                    "gradient_updates": int(audit["gradient_updates"]),
                }
            )
        receiver_means: dict[str, list[float]] = {}
        for evaluation in evaluations:
            receiver_means.setdefault(evaluation["heldout_receiver"], []).append(
                evaluation["mean_delta_pp"]
            )
        worst_receiver_mean = min(
            sum(values) / len(values) for values in receiver_means.values()
        )
        config_summaries.append(
            {
                "config": candidate.to_dict(),
                "worst_receiver_mean_delta_pp": float(worst_receiver_mean),
                "worst_episode_floor_delta_pp": float(
                    min(item["floor_delta_pp"] for item in evaluations)
                ),
                "max_confidence_intrusion_proxy": float(
                    max(item["confidence_intrusion_proxy"] for item in evaluations)
                ),
                "mean_crossfit_fit_count": float(
                    sum(item["crossfit_fit_count"] for item in evaluations) / len(evaluations)
                ),
                "mean_gradient_updates": float(
                    sum(item["gradient_updates"] for item in evaluations) / len(evaluations)
                ),
                "episode_summaries": evaluations,
            }
        )
    reference = candidates[0].to_dict()
    always_da0 = {
        **reference,
        "name": "P05_ALWAYS_DA0",
        "lambda_grid": [0.0],
    }
    config_summaries.append(
        {
            "config": always_da0,
            "worst_receiver_mean_delta_pp": 0.0,
            "worst_episode_floor_delta_pp": 0.0,
            "max_confidence_intrusion_proxy": 0.0,
            "mean_crossfit_fit_count": 0.0,
            "mean_gradient_updates": 0.0,
            "episode_summaries": [
                {
                    "heldout_receiver": str(episode.heldout_receiver),
                    "scene": episode.scene,
                    "mean_delta_pp": 0.0,
                    "floor_delta_pp": 0.0,
                    "confidence_intrusion_proxy": 0.0,
                    "selected_lambda": 0.0,
                    "selected_effective_lambda": 0.0,
                    "crossfit_fit_count": 0,
                    "gradient_updates": 0,
                }
                for episode in episode_rows
            ],
        }
    )
    selected_summary = max(
        config_summaries,
        key=lambda item: (
            item["worst_receiver_mean_delta_pp"],
            item["worst_episode_floor_delta_pp"],
            -item["max_confidence_intrusion_proxy"],
            -item["mean_crossfit_fit_count"],
            item["config"]["name"],
        ),
    )
    receivers = {str(episode.heldout_receiver) for episode in episode_rows}
    k_values = {int(torch.bincount(episode.support_labels)[0]) for episode in episode_rows}
    if len(k_values) != 1:
        raise ValueError("calibration episodes do not share one K-shot")
    selected_config = selected_summary["config"]
    return {
        "schema": CALIBRATION_SCHEMA,
        "status": (
            "CALIBRATED_TO_ABSTAIN"
            if selected_config["name"] == "P05_ALWAYS_DA0"
            else "CALIBRATED_SOURCE_ONLY"
        ),
        "candidate_id": SlowFastCandidate.FAST_FILM_R8.value,
        "source_role": "L_s",
        "calibration_protocol": "source_receiver_heldout_support_query_v1",
        "k_shot": next(iter(k_values)),
        "seed": int(seed),
        "source_receiver_count": len(receivers),
        "episode_count": len(episode_rows),
        "skipped_episodes": [dict(value) for value in skipped],
        "target_support_used": False,
        "target_query_used": False,
        "selected_config": selected_config,
        "config_summaries": config_summaries,
        "deployment_fields": {
            "policy": selected_config["policy"],
            "lambda_grid": selected_config["lambda_grid"],
            "crossfit_repeats": selected_config["repeats"],
            "steps": selected_config["steps"],
            "step_size": selected_config["step_size"],
        },
    }


def save_calibration_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"calibration JSON already exists: {output}")
    if set(payload) != set(CALIBRATION_SCHEMA_KEYS) or payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("calibration JSON schema fields mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_calibration_strict(
    path: str | Path,
) -> tuple[SupportTrustPolicy, dict[str, Any]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"calibration JSON is not a regular file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"calibration JSON cannot be loaded: {source}") from error
    if not isinstance(payload, Mapping) or set(payload) != set(CALIBRATION_SCHEMA_KEYS):
        raise ValueError("calibration JSON schema fields mismatch")
    if payload.get("schema") != CALIBRATION_SCHEMA or payload.get("status") not in {
        "CALIBRATED_SOURCE_ONLY",
        "CALIBRATED_TO_ABSTAIN",
    }:
        raise ValueError("calibration JSON schema/status mismatch")
    if payload.get("candidate_id") != SlowFastCandidate.FAST_FILM_R8.value:
        raise ValueError("calibration must freeze FAST_FILM_R8")
    if payload.get("source_role") != "L_s" or payload.get("target_support_used") is not False or payload.get("target_query_used") is not False:
        raise ValueError("calibration must be source-only with no target access")
    if not isinstance(payload.get("k_shot"), int) or int(payload["k_shot"]) < 1:
        raise ValueError("calibration k_shot must be positive")
    deployment = payload.get("deployment_fields")
    expected_deployment = {"policy", "lambda_grid", "crossfit_repeats", "steps", "step_size"}
    if not isinstance(deployment, Mapping) or set(deployment) != expected_deployment:
        raise ValueError("calibration deployment_fields allowlist mismatch")
    policy_payload = deployment.get("policy")
    expected_policy = {
        "q90_move",
        "hard_move",
        "q90_relative_move",
        "minimum_positive_folds",
        "lcb_z",
        "require_fold_lcb",
    }
    if not isinstance(policy_payload, Mapping) or set(policy_payload) != expected_policy:
        raise ValueError("calibration policy allowlist mismatch")
    policy = SupportTrustPolicy(**policy_payload)
    lambda_grid = tuple(float(value) for value in deployment["lambda_grid"])
    if not lambda_grid or tuple(sorted(set(lambda_grid))) != lambda_grid or lambda_grid[0] != 0.0 or lambda_grid[-1] > 1.0:
        raise ValueError("calibration lambda_grid is invalid")
    repeats = int(deployment["crossfit_repeats"])
    steps = int(deployment["steps"])
    step_size = float(deployment["step_size"])
    if repeats < 1 or steps < 1 or not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("calibration deployment update settings are invalid")
    selected = payload.get("selected_config")
    if not isinstance(selected, Mapping) or any(selected.get(key) != deployment[key] for key in expected_deployment):
        raise ValueError("selected_config disagrees with deployment_fields")
    return policy, {
        "schema": CALIBRATION_SCHEMA,
        "candidate_id": SlowFastCandidate.FAST_FILM_R8.value,
        "k_shot": int(payload["k_shot"]),
        "seed": int(payload["seed"]),
        "selected_name": str(selected.get("name", "")),
        "lambda_grid": lambda_grid,
        "crossfit_repeats": repeats,
        "steps": steps,
        "step_size": step_size,
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "CALIBRATION_SCHEMA_KEYS",
    "CalibrationCandidate",
    "ReceiverHeldoutEpisode",
    "ReceiverHeldoutEpisodeSet",
    "build_receiver_heldout_episodes",
    "calibrate_p05_gate",
    "default_p05_candidates",
    "load_calibration_strict",
    "save_calibration_json",
]
