"""Frozen experiment identities for the 2026-07-28 CVS full ablation design.

This module is intentionally import-light.  It defines the experiment rows and
their invariants, but it does not claim that every listed arm already has a
reachable executor.  Release readiness is a separate, fail-closed field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_SCHEMA = "p2_min_v1"
DESIGN_ID = "cvs_full_ablation_phase1_phase2_20260728"
DESIGN_SCHEMA = "cvs.full_ablation.design.v1"
TARGET_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
LEO_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
SCREENING_SLICES = ((1, 20), (2, 20), (5, 20), (10, 5), (10, 20))
CONFIRMATION_K = (1, 2, 5, 10)
CONFIRMATION_NEW_CLASS_COUNTS = (5, 10, 20)
OLD_CLASS_COUNT = 6
GPU_COUNT = 8
SLOTS_PER_GPU = 2

# The design explicitly records these Stage2 seeds as already observed.  They
# may be used for code regression only, never for a fresh screening or
# confirmation claim.
OBSERVED_STAGE2_SEEDS = frozenset(range(713101, 713107))

REQUIRED_RUN_ARTIFACT_FIELDS = (
    "run_id",
    "ablation_id",
    "evidence_level",
    "git_commit",
    "config_hash",
    "phase1_bundle_hash",
    "protocol_schema",
    "capsule_id",
    "split_id",
    "phase2_data_status",
    "receiver_id",
    "train_seed",
    "support_seed",
    "query_seed",
    "new_class_draw_seed",
    "channel_assignment_hash",
    "k_shot",
    "old_class_ids_hash",
    "new_class_ids_hash",
    "support_physical_ids_hash",
    "query_physical_ids_hash",
    "predictions_hash",
    "score_artifact_hash",
    "scorer_receipt",
    "all_primary_metrics",
    "per_class_metrics",
    "fallback_counts",
    "fisher_gate_accept_counts",
    "atomic_rollback_counts",
    "quantization_error",
    "state_bytes",
    "registration_time",
    "peak_memory",
    "query_latency",
    "exit_status",
)


class FullAblationSpecError(ValueError):
    """Raised when a full-ablation plan violates its preregistration."""


@dataclass(frozen=True)
class ArmSpec:
    ablation_id: str
    phase: str
    evidence_level: str
    mechanism_family: str
    comparison_target: str
    executor_status: str = "UNMAPPED_FAIL_CLOSED"
    physical_config_id: str | None = None


@dataclass(frozen=True)
class SeedBundle:
    train_seed: int
    support_seed: int
    query_seed: int

    def validate(self, *, require_fresh_stage2: bool) -> None:
        values = tuple(int(value) for value in asdict(self).values())
        if any(value <= 0 for value in values):
            raise FullAblationSpecError("all seed fields must be positive")
        if len(set(values)) != len(values):
            raise FullAblationSpecError(
                "train/support/query seeds must be recorded separately"
            )
        if require_fresh_stage2 and any(
            value in OBSERVED_STAGE2_SEEDS for value in values
        ):
            raise FullAblationSpecError(
                "fresh Stage2 plan reuses an observed 713101-713106 seed"
            )


@dataclass(frozen=True)
class WorkerSlot:
    gpu: int
    slot: int

    @property
    def key(self) -> str:
        return f"gpu{self.gpu}_slot{self.slot}"


PHASE1_T1_ARMS = (
    ArmSpec("P1-FULL", "phase1", "M", "reference", "current full Phase1"),
    ArmSpec("P1-SUP", "phase1", "M", "supervision", "P1-FULL"),
    ArmSpec("P1-A0", "phase1", "M", "dual_representation", "P1-FULL"),
    ArmSpec("P1-B0", "phase1", "M", "pseudo_label", "P1-FULL"),
    ArmSpec("P1-C0", "phase1", "M", "angular_tail_geometry", "P1-FULL"),
    ArmSpec("P1-D0", "phase1", "M", "counterfactual_extrapolation", "P1-FULL"),
)

PHASE2_BASELINE_ARMS = (
    ArmSpec("P2-BASE-COSINE", "stage2c", "M", "baseline", "P2-FULL"),
    ArmSpec("P2-BASE-EUCLIDEAN", "stage2c", "M", "baseline", "P2-FULL"),
    ArmSpec("P2-BASE-QKNN", "stage2c", "M", "baseline", "P2-FULL"),
    ArmSpec("P2-BASE-DIAG-LDA", "stage2c", "M", "baseline", "P2-FULL"),
    ArmSpec("P2-BASE-POOLED-LW-LDA", "stage2c", "M", "baseline", "P2-FULL"),
    ArmSpec("P2-BASE-FULL-BLOCK-LDA", "stage2c", "M", "baseline", "P2-FULL"),
    ArmSpec("P2-BASE-ADAPTER-HEAD", "stage2c", "M", "baseline", "P2-FULL"),
)

PHASE2_T1_ARMS = (
    ArmSpec("P2-FULL", "stage2c", "M", "reference", "current full Phase2"),
    *PHASE2_BASELINE_ARMS,
    ArmSpec("P2-A0", "stage2c", "M", "joint_feature", "P2-FULL"),
    ArmSpec("P2-B0", "stage2c", "M", "robust_center", "P2-FULL"),
    ArmSpec("P2-C3", "stage2c", "M", "task_covariance", "P2-FULL"),
    ArmSpec("P2-D0", "stage2c", "M", "dual_geometry", "P2-FULL"),
    ArmSpec("P2-D1", "stage2c", "M", "dual_geometry", "P2-FULL"),
    ArmSpec("P2-D2", "stage2c", "M", "crossfit_fusion", "P2-FULL"),
    ArmSpec("P2-E0", "stage2c", "M", "fisher_safety", "P2-FULL"),
    ArmSpec("P2-F0", "stage2c", "M", "quantization", "P2-FULL"),
    ArmSpec("P2-F1", "stage2c", "M", "quantization", "P2-FULL"),
    ArmSpec("P2-F2", "stage2c", "M", "quantization", "P2-FULL"),
    ArmSpec(
        "P2-F3",
        "stage2c",
        "M",
        "quantization",
        "P2-FULL",
        physical_config_id="P2-FULL",
    ),
)


def _unique_positive(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise FullAblationSpecError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise FullAblationSpecError(f"{name} contains duplicate values")
    return result


def assign_worker_slots(row_count: int) -> tuple[WorkerSlot, ...]:
    """Assign rows round-robin to exactly sixteen persistent slot queues."""

    count = int(row_count)
    if count <= 0:
        raise FullAblationSpecError("row_count must be positive")
    slots = tuple(
        WorkerSlot(gpu=gpu, slot=slot)
        for gpu in range(GPU_COUNT)
        for slot in range(SLOTS_PER_GPU)
    )
    return tuple(slots[index % len(slots)] for index in range(count))


def build_phase1_t1_rows(
    train_seeds: Sequence[int],
    *,
    git_commit: str,
) -> list[dict[str, Any]]:
    seeds = _unique_positive(train_seeds, name="Phase1 train seeds")
    if len(seeds) != 5:
        raise FullAblationSpecError("Phase1 T1 requires exactly five paired seeds")
    if len(str(git_commit).strip()) < 7:
        raise FullAblationSpecError("a concrete Git commit is required")
    rows: list[dict[str, Any]] = []
    for arm in PHASE1_T1_ARMS:
        for seed in seeds:
            rows.append(
                {
                    "design_id": DESIGN_ID,
                    "design_schema": DESIGN_SCHEMA,
                    "phase": "phase1",
                    "ablation_id": arm.ablation_id,
                    "evidence_level": arm.evidence_level,
                    "mechanism_family": arm.mechanism_family,
                    "comparison_target": arm.comparison_target,
                    "git_commit": str(git_commit),
                    "train_seed": seed,
                    "split_fractions": {
                        "labeled": 0.07,
                        "unlabeled": 0.63,
                        "source_validation": 0.30,
                    },
                    "epochs": 200,
                    "checkpoint_selection": "source_validation_only",
                    "executor_status": arm.executor_status,
                }
            )
    slots = assign_worker_slots(len(rows))
    for row, slot in zip(rows, slots):
        row["worker"] = asdict(slot)
        row["row_key"] = (
            f"{row['ablation_id']}__train_seed_{row['train_seed']}"
        )
    validate_plan_rows(rows)
    return rows


def build_phase2_rows(
    *,
    stage: str,
    arms: Sequence[ArmSpec],
    seed_bundles: Sequence[SeedBundle],
    class_draw_seeds: Sequence[int],
    git_commit: str,
) -> list[dict[str, Any]]:
    """Build screening or confirmation rows without multiplying scenarios."""

    if stage not in {"screening", "confirmation"}:
        raise FullAblationSpecError("stage must be screening or confirmation")
    if not arms:
        raise FullAblationSpecError("at least one Phase2 arm is required")
    if len({arm.ablation_id for arm in arms}) != len(arms):
        raise FullAblationSpecError("Phase2 arm IDs must be unique")
    expected_seed_count = 3 if stage == "screening" else 5
    if len(seed_bundles) != expected_seed_count:
        raise FullAblationSpecError(
            f"{stage} requires exactly {expected_seed_count} seed bundles"
        )
    for bundle in seed_bundles:
        bundle.validate(require_fresh_stage2=True)
    all_seed_values = [
        value
        for bundle in seed_bundles
        for value in asdict(bundle).values()
    ]
    if len(set(all_seed_values)) != len(all_seed_values):
        raise FullAblationSpecError("seed bundles must be globally independent")
    draws = _unique_positive(class_draw_seeds, name="new-class draw seeds")
    expected_draw_count = 1 if stage == "screening" else 3
    if len(draws) != expected_draw_count:
        raise FullAblationSpecError(
            f"{stage} requires exactly {expected_draw_count} class draws"
        )
    if any(draw in OBSERVED_STAGE2_SEEDS for draw in draws):
        raise FullAblationSpecError("new-class draw seed is not fresh")
    if set(draws) & set(all_seed_values):
        raise FullAblationSpecError(
            "new-class draw seeds must be independent of row seeds"
        )
    if len(str(git_commit).strip()) < 7:
        raise FullAblationSpecError("a concrete Git commit is required")
    slices = (
        SCREENING_SLICES
        if stage == "screening"
        else tuple(
            (k_shot, new_count)
            for k_shot in CONFIRMATION_K
            for new_count in CONFIRMATION_NEW_CLASS_COUNTS
        )
    )
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for receiver in TARGET_RECEIVERS:
            for k_shot, new_count in slices:
                for bundle in seed_bundles:
                    for draw_seed in draws:
                        rows.append(
                            {
                                "design_id": DESIGN_ID,
                                "design_schema": DESIGN_SCHEMA,
                                "phase": "stage2c",
                                "stage": stage,
                                "ablation_id": arm.ablation_id,
                                "evidence_level": arm.evidence_level,
                                "mechanism_family": arm.mechanism_family,
                                "comparison_target": arm.comparison_target,
                                "physical_config_id": (
                                    arm.physical_config_id or arm.ablation_id
                                ),
                                "git_commit": str(git_commit),
                                "protocol_schema": PROTOCOL_SCHEMA,
                                "receiver_id": receiver,
                                "k_shot": int(k_shot),
                                "old_class_count": OLD_CLASS_COUNT,
                                "new_class_count": int(new_count),
                                "scenarios": list(LEO_SCENARIOS),
                                **asdict(bundle),
                                "new_class_draw_seed": int(draw_seed),
                                "executor_status": arm.executor_status,
                                "formal_launch_authority": False,
                            }
                        )
    slots = assign_worker_slots(len(rows))
    for row, slot in zip(rows, slots):
        row["worker"] = asdict(slot)
        row["row_key"] = (
            f"{row['ablation_id']}__rx_{row['receiver_id'].replace('-', '_')}"
            f"__k_{row['k_shot']}__new_{row['new_class_count']}"
            f"__support_{row['support_seed']}__query_{row['query_seed']}"
            f"__draw_{row['new_class_draw_seed']}"
        )
    validate_plan_rows(rows)
    return rows


def validate_plan_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise FullAblationSpecError("plan has no rows")
    keys = [str(row.get("row_key", "")) for row in rows]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise FullAblationSpecError("row keys must be non-empty and unique")
    for row in rows:
        worker = row.get("worker")
        if not isinstance(worker, Mapping):
            raise FullAblationSpecError("row is missing worker assignment")
        gpu, slot = int(worker.get("gpu", -1)), int(worker.get("slot", -1))
        if not (0 <= gpu < GPU_COUNT and 0 <= slot < SLOTS_PER_GPU):
            raise FullAblationSpecError("worker assignment exceeds 8x2 bounds")
        if row.get("phase") == "stage2c":
            if row.get("protocol_schema") != PROTOCOL_SCHEMA:
                raise FullAblationSpecError("Stage2 protocol schema drift")
            if tuple(row.get("scenarios", ())) != LEO_SCENARIOS:
                raise FullAblationSpecError("Stage2 scenario list drift")
            if int(row.get("k_shot", 0)) <= 0:
                raise FullAblationSpecError("Stage2 K must be positive")


def validate_artifact_record(record: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_RUN_ARTIFACT_FIELDS if field not in record]
    if missing:
        raise FullAblationSpecError(
            "run artifact is incomplete: " + ",".join(missing)
        )
    if record.get("protocol_schema") != PROTOCOL_SCHEMA:
        raise FullAblationSpecError("run artifact protocol schema drift")
    if record.get("phase2_data_status") != "VALIDATED_ONCE":
        raise FullAblationSpecError("run artifact lacks VALIDATED_ONCE data status")
    if not str(record["support_physical_ids_hash"]).strip():
        raise FullAblationSpecError("support physical ID hash is empty")
    if record["support_physical_ids_hash"] == record["query_physical_ids_hash"]:
        raise FullAblationSpecError("support/query physical ID hashes must differ")


__all__ = [
    "ArmSpec",
    "CONFIRMATION_K",
    "CONFIRMATION_NEW_CLASS_COUNTS",
    "DESIGN_ID",
    "DESIGN_SCHEMA",
    "FullAblationSpecError",
    "GPU_COUNT",
    "LEO_SCENARIOS",
    "OBSERVED_STAGE2_SEEDS",
    "PHASE1_T1_ARMS",
    "PHASE2_T1_ARMS",
    "PROTOCOL_SCHEMA",
    "REQUIRED_RUN_ARTIFACT_FIELDS",
    "SCREENING_SLICES",
    "SLOTS_PER_GPU",
    "SeedBundle",
    "TARGET_RECEIVERS",
    "assign_worker_slots",
    "build_phase1_t1_rows",
    "build_phase2_rows",
    "validate_artifact_record",
    "validate_plan_rows",
]
