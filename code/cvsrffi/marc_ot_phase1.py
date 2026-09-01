"""Explicit Phase1 MARC-OT schedule, bank-step, and strict-bundle closure."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .meta_bank_trainer import (
    MetaBankStepResult,
    MetaBankTrainerConfig,
    run_meta_bank_step,
)
from .meta_episodes import (
    HierarchicalMetaEpisodeSampler,
    MetaEpisode,
    audit_marc_ot_episode_coverage,
    sample_marc_ot_coverage_schedule,
    validate_episode_semantics,
)
from .meta_support_set_encoder import SupportSetEncoder
from .meta_trainer import MetaEpisodeBatch
from .meta_weight_bank import BlockSpec, WeightDeltaBank
from .meta_weight_bank_checkpoint import (
    MetaWeightBundle,
    load_meta_weight_bundle,
    save_meta_weight_bundle,
)


@dataclass(frozen=True)
class MARCOTPhase1Closure:
    """Software coverage, actual training coverage, and strict bundle readback."""

    entrypoint: str
    bundle_path: Path
    software_coverage: Mapping[str, Any]
    training_coverage: Mapping[str, Any]
    step_results: tuple[MetaBankStepResult, ...]
    loaded_bundle: MetaWeightBundle
    pilot_executed: bool = False


def _actual_training_coverage(episodes: Sequence[MetaEpisode]) -> Mapping[str, Any]:
    rows = tuple(episodes)
    transitions = tuple(
        sorted(
            {
                (
                    str(episode.kind.value),
                    str(episode.support[0].view),
                    str((episode.query_adapt + episode.query_guard)[0].view),
                )
                for episode in rows
            }
        )
    )
    return {
        "trained_episode_count": len(rows),
        "k_shot": tuple(sorted({int(episode.k_shot) for episode in rows})),
        "episode_kinds": tuple(sorted({str(episode.kind.value) for episode in rows})),
        "scene_transitions": transitions,
        "training_step_executed": True,
        "input_provenance": "CALLER_SUPPLIED_UNCLAIMED",
        "pilot_executed": False,
    }


def run_marc_ot_phase1_bank_training(
    *,
    sampler: HierarchicalMetaEpisodeSampler,
    batch_builder: Callable[[MetaEpisode], MetaEpisodeBatch],
    functional_forward: Callable[[Mapping[str, Tensor], Tensor], object],
    base_state: Mapping[str, Tensor],
    base_checkpoint_id: str,
    bank: WeightDeltaBank,
    support_encoder: SupportSetEncoder,
    support_feature_model: nn.Module,
    trainer_config: MetaBankTrainerConfig,
    optimizer: torch.optim.Optimizer,
    expected_block_specs: tuple[BlockSpec, ...],
    bundle_path: str | Path,
    training_episode_selector: Callable[
        [tuple[MetaEpisode, ...]], Sequence[MetaEpisode]
    ],
    schedule_seed: int = 0,
) -> MARCOTPhase1Closure:
    """Run the real MARC-OT bank trainer after complete schedule validation.

    The full schedule proves software support. Only episodes returned by the
    explicit selector are trained and recorded as actual Phase1 coverage.
    """

    if not isinstance(trainer_config, MetaBankTrainerConfig):
        raise TypeError("trainer_config must be MetaBankTrainerConfig")
    if not callable(batch_builder) or not callable(training_episode_selector):
        raise TypeError("batch_builder and training_episode_selector must be callable")
    destination = Path(bundle_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable MARC-OT bundle exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("MARC-OT bundle parent directory must already exist")

    scheduled = sample_marc_ot_coverage_schedule(sampler, seed=int(schedule_seed))
    software_coverage = audit_marc_ot_episode_coverage(
        scheduled,
        source_receiver_ids=trainer_config.source_receiver_ids,
        require_complete=True,
    )
    selected = tuple(training_episode_selector(scheduled))
    if not selected:
        raise ValueError("at least one scheduled MARC-OT episode must be trained")
    scheduled_set = set(scheduled)
    if any(episode not in scheduled_set for episode in selected):
        raise ValueError("training selector returned an episode outside the frozen schedule")

    results: list[MetaBankStepResult] = []
    for episode in selected:
        validate_episode_semantics(
            episode,
            source_receiver_ids=trainer_config.source_receiver_ids,
        )
        batch = batch_builder(episode)
        if not isinstance(batch, MetaEpisodeBatch) or batch.episode != episode:
            raise ValueError("batch builder must preserve the exact scheduled episode")
        results.append(
            run_meta_bank_step(
                functional_forward,
                base_state=base_state,
                base_checkpoint_id=base_checkpoint_id,
                bank=bank,
                support_encoder=support_encoder,
                support_feature_model=support_feature_model,
                batch=batch,
                config=trainer_config,
                optimizer=optimizer,
            )
        )

    saved = save_meta_weight_bundle(
        destination,
        base_checkpoint_id=base_checkpoint_id,
        base_state=base_state,
        bank=bank,
        support_encoder=support_encoder,
        expected_block_specs=expected_block_specs,
    )
    loaded = load_meta_weight_bundle(
        saved,
        expected_base_checkpoint_id=base_checkpoint_id,
        base_state=base_state,
        expected_block_specs=expected_block_specs,
    )
    return MARCOTPhase1Closure(
        entrypoint="run_marc_ot_phase1_bank_training",
        bundle_path=saved,
        software_coverage=dict(software_coverage),
        training_coverage=dict(_actual_training_coverage(selected)),
        step_results=tuple(results),
        loaded_bundle=loaded,
        pilot_executed=False,
    )


__all__ = [
    "MARCOTPhase1Closure",
    "run_marc_ot_phase1_bank_training",
]
