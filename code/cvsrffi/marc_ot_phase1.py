"""Explicit Phase1 MARC-OT schedule, bank-step, and strict-bundle closure."""

from __future__ import annotations

import math
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
    marc_ot_episode_semantic_key,
    sample_marc_ot_coverage_schedule,
    validate_episode_semantics,
)
from .meta_support_set_encoder import SupportSetEncoder
from .marc_ot_support_features import build_marc_ot_support_features
from .meta_trainer import MetaEpisodeBatch
from .meta_weight_bank import BlockSpec, DeltaTaskKey, WeightDeltaBank
from .meta_weight_bank_checkpoint import (
    MetaWeightBundle,
    TaskDomainDescriptor,
    load_meta_weight_bundle,
    save_meta_weight_bundle,
)


@dataclass(frozen=True)
class MARCOTTaskDomainSelection:
    """Explicit task key and legal Phase1 partition used for one aggregate."""

    task_key: DeltaTaskKey
    partition: str


def canonical_episode_task_domain_selection(
    episode: MetaEpisode,
) -> MARCOTTaskDomainSelection:
    """Bind the default pseudo-target domain from explicit query-adapt facts."""

    if not isinstance(episode, MetaEpisode) or not episode.query_adapt:
        raise ValueError("episode query_adapt facts are required for task domain binding")
    facts = {(ref.rx_i, ref.day_i, ref.view) for ref in episode.query_adapt}
    if len(facts) != 1:
        raise ValueError("episode query_adapt facts do not identify one task domain")
    receiver, day, scene = next(iter(facts))
    return MARCOTTaskDomainSelection(
        task_key=DeltaTaskKey(str(int(receiver)), str(int(day)), str(scene), episode.k_shot),
        partition="query_adapt",
    )


def _validated_task_domain_selection(
    episode: MetaEpisode,
    selection: MARCOTTaskDomainSelection,
) -> tuple[tuple[object, ...], DeltaTaskKey]:
    if not isinstance(selection, MARCOTTaskDomainSelection):
        raise ValueError("task domain selector must return MARCOTTaskDomainSelection")
    if selection.partition not in {"support", "query_adapt"}:
        raise ValueError("task domain partition must be support or query_adapt; query_guard is forbidden")
    key = selection.task_key
    if not isinstance(key, DeltaTaskKey):
        raise ValueError("task domain selector returned an invalid task key")
    refs = tuple(getattr(episode, selection.partition))
    if len(refs) < 2:
        raise ValueError("task domain descriptor requires multiple physical samples")
    facts = {(ref.rx_i, ref.day_i, ref.view) for ref in refs}
    if len(facts) != 1:
        raise ValueError("selected episode partition does not identify one task domain")
    receiver, day, scene = next(iter(facts))
    explicit_key = DeltaTaskKey(
        str(int(receiver)), str(int(day)), str(scene), int(episode.k_shot)
    )
    if key != explicit_key:
        raise ValueError("task domain key differs from explicit episode partition facts")
    physical_ids = tuple(ref.physical_sample_id for ref in refs)
    if len(set(physical_ids)) != len(physical_ids):
        raise ValueError("task domain partition repeats a physical sample")
    return refs, key


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


def _actual_training_coverage(
    episodes: Sequence[MetaEpisode],
    *,
    updated_required_tensor_count: int,
) -> Mapping[str, Any]:
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
        "updated_required_tensor_count": int(updated_required_tensor_count),
        "input_provenance": "CALLER_SUPPLIED_UNCLAIMED",
        "pilot_executed": False,
    }


def _snapshot_required_training_state(
    bank: WeightDeltaBank,
    support_encoder: SupportSetEncoder,
) -> dict[str, Tensor]:
    state = {
        **{
            f"bank_basis.{entry.spec.name}": entry.basis.detach().cpu().clone()
            for entry in bank.entries
        },
        **{
            f"support_encoder.{name}": value.detach().cpu().clone()
            for name, value in support_encoder.state_dict().items()
        },
    }
    if not state or any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("required Phase1 bank/encoder state must be finite and nonempty")
    return state


def _loaded_required_training_state(bundle: MetaWeightBundle) -> dict[str, Tensor]:
    return {
        **{
            f"bank_basis.{entry.spec.name}": entry.basis.detach().cpu().clone()
            for entry in bundle.bank.entries
        },
        **{
            f"support_encoder.{name}": value.detach().cpu().clone()
            for name, value in bundle.support_encoder.state_dict().items()
        },
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
    task_domain_selector: Callable[
        [MetaEpisode], MARCOTTaskDomainSelection
    ] = canonical_episode_task_domain_selection,
) -> MARCOTPhase1Closure:
    """Run the real MARC-OT bank trainer after complete schedule validation.

    The full schedule proves software support. Only episodes returned by the
    explicit selector are trained and recorded as actual Phase1 coverage.
    """

    if not isinstance(trainer_config, MetaBankTrainerConfig):
        raise TypeError("trainer_config must be MetaBankTrainerConfig")
    if (
        not callable(batch_builder)
        or not callable(training_episode_selector)
        or not callable(task_domain_selector)
    ):
        raise TypeError("batch_builder, training selector, and task domain selector must be callable")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if any(
        isinstance(group.get("lr"), bool)
        or not isinstance(group.get("lr"), (int, float))
        or not math.isfinite(float(group["lr"]))
        or float(group["lr"]) <= 0.0
        for group in optimizer.param_groups
    ):
        raise ValueError("optimizer learning rate must be finite and positive")
    destination = Path(bundle_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable MARC-OT bundle exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("MARC-OT bundle parent directory must already exist")

    pre_step_state = _snapshot_required_training_state(bank, support_encoder)

    scheduled = sample_marc_ot_coverage_schedule(sampler, seed=int(schedule_seed))
    software_coverage = audit_marc_ot_episode_coverage(
        scheduled,
        source_receiver_ids=trainer_config.source_receiver_ids,
        require_complete=True,
    )
    selected = tuple(training_episode_selector(scheduled))
    if not selected:
        raise ValueError("at least one scheduled MARC-OT episode must be trained")
    if len({id(episode) for episode in selected}) != len(selected):
        raise ValueError("training selector returned a duplicate episode")
    semantic_keys = tuple(
        marc_ot_episode_semantic_key(
            episode,
            source_receiver_ids=trainer_config.source_receiver_ids,
        )
        for episode in selected
    )
    if len(set(semantic_keys)) != len(semantic_keys):
        raise ValueError("training selector returned a duplicate semantic key")
    scheduled_set = set(scheduled)
    if any(episode not in scheduled_set for episode in selected):
        raise ValueError("training selector returned an episode outside the frozen schedule")

    selections: list[tuple[tuple[object, ...], DeltaTaskKey, str]] = []
    semantic_to_task: dict[object, DeltaTaskKey] = {}
    selected_task_keys: list[DeltaTaskKey] = []
    for episode, semantic_key in zip(selected, semantic_keys, strict=True):
        selection = task_domain_selector(episode)
        refs, task_key = _validated_task_domain_selection(episode, selection)
        prior = semantic_to_task.setdefault(semantic_key, task_key)
        if prior != task_key:
            raise ValueError("one semantic episode cell maps to multiple task keys")
        selections.append((refs, task_key, selection.partition))
        selected_task_keys.append(task_key)
    if set(selected_task_keys) != set(bank.task_keys):
        missing = set(bank.task_keys) - set(selected_task_keys)
        extra = set(selected_task_keys) - set(bank.task_keys)
        raise ValueError(
            f"task domain coverage must exactly match bank task keys: missing={missing!r}, extra={extra!r}"
        )

    results: list[MetaBankStepResult] = []
    task_row_sums: dict[DeltaTaskKey, Tensor] = {}
    task_row_counts: dict[DeltaTaskKey, int] = {}
    task_physical_ids: dict[DeltaTaskKey, set[str]] = {}
    for episode, (refs, task_key, partition) in zip(selected, selections, strict=True):
        validate_episode_semantics(
            episode,
            source_receiver_ids=trainer_config.source_receiver_ids,
        )
        batch = batch_builder(episode)
        if not isinstance(batch, MetaEpisodeBatch) or batch.episode != episode:
            raise ValueError("batch builder must preserve the exact scheduled episode")
        if partition == "query_adapt":
            row_mask = batch.adapt_mask
            descriptor_iq = batch.query_x[row_mask]
            descriptor_labels = batch.query_y[row_mask]
            nominal_k = episode.query_per_class
        else:
            descriptor_iq = batch.support_x
            descriptor_labels = batch.support_y
            nominal_k = episode.k_shot
        physical_ids = tuple(str(ref.physical_sample_id) for ref in refs)
        if len(physical_ids) != len(descriptor_iq):
            raise ValueError("task domain descriptor rows differ from explicit partition facts")
        seen_ids = task_physical_ids.setdefault(task_key, set())
        if seen_ids.intersection(physical_ids):
            raise ValueError("task domain aggregation repeats a physical sample")
        seen_ids.update(physical_ids)
        with torch.no_grad():
            feature_batch = build_marc_ot_support_features(
                support_feature_model,
                descriptor_iq,
                descriptor_labels,
                physical_ids,
                nominal_k=nominal_k,
                validated_unpadded=True,
                scope="phase1_source",
                fit_scope="full_episode",
            )
        row_sum = feature_batch.rows.detach().to(device="cpu", dtype=torch.float32).sum(dim=0)
        task_row_sums[task_key] = task_row_sums.get(task_key, torch.zeros_like(row_sum)) + row_sum
        task_row_counts[task_key] = task_row_counts.get(task_key, 0) + len(feature_batch.rows)
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

    post_step_state = _snapshot_required_training_state(bank, support_encoder)
    if set(post_step_state) != set(pre_step_state):
        raise RuntimeError("required Phase1 training state registry changed")
    updated_names = tuple(
        name
        for name in pre_step_state
        if not torch.equal(pre_step_state[name], post_step_state[name])
    )
    if not updated_names:
        raise RuntimeError("Phase1 bank step did not update any required bank/encoder tensor")

    if set(task_row_sums) != set(bank.task_keys):
        raise RuntimeError("task domain aggregate coverage changed during Phase1 training")
    task_domain_descriptors = {
        task_key: TaskDomainDescriptor(
            values=task_row_sums[task_key] / float(task_row_counts[task_key]),
            aggregation_count=task_row_counts[task_key],
        )
        for task_key in bank.task_keys
    }

    saved = save_meta_weight_bundle(
        destination,
        base_checkpoint_id=base_checkpoint_id,
        base_state=base_state,
        bank=bank,
        support_encoder=support_encoder,
        expected_block_specs=expected_block_specs,
        task_domain_descriptors=task_domain_descriptors,
    )
    loaded = load_meta_weight_bundle(
        saved,
        expected_base_checkpoint_id=base_checkpoint_id,
        base_state=base_state,
        expected_block_specs=expected_block_specs,
    )
    loaded_state = _loaded_required_training_state(loaded)
    if set(loaded_state) != set(post_step_state) or any(
        not torch.equal(loaded_state[name], post_step_state[name])
        for name in post_step_state
    ):
        raise RuntimeError("strict MARC-OT bundle readback differs from post-step state")
    return MARCOTPhase1Closure(
        entrypoint="run_marc_ot_phase1_bank_training",
        bundle_path=saved,
        software_coverage=dict(software_coverage),
        training_coverage=dict(
            _actual_training_coverage(
                selected,
                updated_required_tensor_count=len(updated_names),
            )
        ),
        step_results=tuple(results),
        loaded_bundle=loaded,
        pilot_executed=False,
    )


__all__ = [
    "MARCOTPhase1Closure",
    "MARCOTTaskDomainSelection",
    "canonical_episode_task_domain_selection",
    "run_marc_ot_phase1_bank_training",
]
