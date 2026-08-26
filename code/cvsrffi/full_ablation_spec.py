"""Frozen experiment identities for the 2026-07-28 CVS full ablation design.

This module is intentionally import-light.  It defines the experiment rows and
their invariants, but it does not claim that every listed arm already has a
reachable executor.  Release readiness is a separate, fail-closed field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from cvsrffi.phase1_ablation_factory import (
    PHASE1_LABEL_ABLATION_IDS,
    PHASE1_LABEL_RHOS_BY_ID,
    phase1_ablation_config_hash,
)

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
    "support_query_overlap_count",
    "support_query_disjoint_receipt_sha256",
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

    @property
    def method_seed(self) -> int:
        """Stage2 algorithm seed; never the Phase1 bundle-training seed."""

        return int(self.train_seed)

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


@dataclass(frozen=True)
class Stage2InputBinding:
    phase1_bundle_hash: str
    phase1_bundle_training_seed: int
    capsule_id: str
    split_id: str
    channel_assignment_hash: str
    old_class_ids_hash: str
    new_class_ids_hash: str
    support_physical_ids_hash: str
    query_physical_ids_hash: str
    support_query_disjoint_receipt_sha256: str
    support_prefix_receipt_sha256: str
    new_class_prefix_receipt_sha256: str
    query_fixed_receipt_sha256: str
    support_query_overlap_count: int = 0
    phase2_data_status: str = "VALIDATED_ONCE"

    def validate(self) -> None:
        if self.phase2_data_status != "VALIDATED_ONCE":
            raise FullAblationSpecError(
                "Stage2 binding lacks VALIDATED_ONCE data status"
            )
        if int(self.phase1_bundle_training_seed) <= 0:
            raise FullAblationSpecError(
                "Phase1 bundle-training seed must be positive"
            )
        if not str(self.capsule_id).strip() or not str(self.split_id).strip():
            raise FullAblationSpecError("capsule_id and split_id are required")
        hash_fields = (
            "phase1_bundle_hash",
            "channel_assignment_hash",
            "old_class_ids_hash",
            "new_class_ids_hash",
            "support_physical_ids_hash",
            "query_physical_ids_hash",
            "support_query_disjoint_receipt_sha256",
            "support_prefix_receipt_sha256",
            "new_class_prefix_receipt_sha256",
            "query_fixed_receipt_sha256",
        )
        for field in hash_fields:
            if not _is_sha256(getattr(self, field)):
                raise FullAblationSpecError(
                    f"Stage2 binding {field} must be SHA256"
                )
        if (
            self.support_physical_ids_hash
            == self.query_physical_ids_hash
            or int(self.support_query_overlap_count) != 0
        ):
            raise FullAblationSpecError(
                "Stage2 support/query physical IDs are not disjoint"
            )


PHASE1_T1_ARMS = (
    ArmSpec(
        "P1-FULL",
        "phase1",
        "M",
        "reference",
        "current full Phase1",
        "LOCAL_IMPLEMENTED_PENDING_REVIEW",
    ),
    ArmSpec(
        "P1-SUP",
        "phase1",
        "M",
        "supervision",
        "P1-FULL",
        "LOCAL_IMPLEMENTED_PENDING_REVIEW",
    ),
    ArmSpec(
        "P1-A0",
        "phase1",
        "M",
        "dual_representation",
        "P1-FULL",
        "LOCAL_IMPLEMENTED_PENDING_REVIEW",
    ),
    ArmSpec(
        "P1-B0",
        "phase1",
        "M",
        "pseudo_label",
        "P1-FULL",
        "LOCAL_IMPLEMENTED_PENDING_REVIEW",
    ),
    ArmSpec(
        "P1-C0",
        "phase1",
        "M",
        "angular_tail_geometry",
        "P1-FULL",
        "LOCAL_IMPLEMENTED_PENDING_REVIEW",
    ),
    ArmSpec(
        "P1-D0",
        "phase1",
        "M",
        "counterfactual_extrapolation",
        "P1-FULL",
        "LOCAL_IMPLEMENTED_PENDING_REVIEW",
    ),
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

PHASE2_STATE_T1_ARMS = (
    ArmSpec(
        "P2-S2A",
        "stage2a",
        "M",
        "zero_label_deployment",
        "P1-FULL",
    ),
    ArmSpec(
        "P2-S2B-PROTO",
        "stage2b",
        "M",
        "old_class_adaptation",
        "P2-S2B-FULL",
    ),
    ArmSpec(
        "P2-S2B-DIAGOFF",
        "stage2b",
        "M",
        "old_class_metric",
        "P2-S2B-FULL",
    ),
    ArmSpec(
        "P2-S2B-FULL",
        "stage2b",
        "M",
        "old_class_adaptation",
        "reference",
    ),
)

PHASE2_T1_ARMS = (
    ArmSpec("P2-FULL", "stage2c", "M", "reference", "current full Phase2"),
    *PHASE2_BASELINE_ARMS,
    ArmSpec("P2-A0", "stage2c", "M", "joint_feature", "P2-FULL"),
    ArmSpec("P2-A1", "stage2c", "M", "joint_feature", "P2-FULL"),
    ArmSpec("P2-A2", "stage2c", "M", "joint_feature", "P2-FULL"),
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


# Approved current-method ablation screen.  This intentionally lives beside,
# rather than inside, the historical full T1 matrix: every arm starts from the
# current identity160+FFT96 method state and the FP32 F0 comparison is absent.
PHASE2_E0_256_ABLATION_ARMS = (
    ArmSpec(
        "P2-256-FULL",
        "stage2c",
        "S_SCREENING",
        "reference",
        "current_256d_reference",
        executor_status="LOCAL_IMPLEMENTED",
    ),
    ArmSpec(
        "P2-256-A0",
        "stage2c",
        "S_SCREENING",
        "joint_feature",
        "P2-256-FULL",
        executor_status="LOCAL_IMPLEMENTED",
    ),
    ArmSpec(
        "P2-256-B0",
        "stage2c",
        "S_SCREENING",
        "robust_center",
        "P2-256-FULL",
        executor_status="LOCAL_IMPLEMENTED",
    ),
    ArmSpec(
        "P2-256-S0",
        "stage2c",
        "S_SCREENING",
        "auto_shrinkage",
        "P2-256-FULL",
        executor_status="LOCAL_IMPLEMENTED",
    ),
    ArmSpec(
        "P2-256-C3",
        "stage2c",
        "S_SCREENING",
        "task_covariance",
        "P2-256-FULL",
        executor_status="LOCAL_IMPLEMENTED",
    ),
    ArmSpec(
        "P2-256-D0",
        "stage2c",
        "S_SCREENING",
        "dual_geometry",
        "P2-256-FULL",
        executor_status="LOCAL_IMPLEMENTED",
    ),
    ArmSpec(
        "P2-256-D2",
        "stage2c",
        "S_SCREENING",
        "crossfit_fusion",
        "P2-256-FULL",
        executor_status="LOCAL_IMPLEMENTED",
    ),
)


def _unique_positive(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise FullAblationSpecError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise FullAblationSpecError(f"{name} contains duplicate values")
    return result


def _require_git_commit(value: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 40 or any(ch not in "0123456789abcdef" for ch in result):
        raise FullAblationSpecError("a full 40-character Git commit is required")
    return result


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


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
    concrete_commit = _require_git_commit(git_commit)
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
                    "git_commit": concrete_commit,
                    "method_config_hash": phase1_ablation_config_hash(
                        arm.ablation_id
                    ),
                    "train_seed": seed,
                    "split_fractions": {
                        "labeled": 0.07,
                        "unlabeled": 0.63,
                        "source_validation": 0.30,
                    },
                    "epochs": 200,
                    "checkpoint_selection": "final_only",
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


def build_phase1_label_rows(
    train_seeds: Sequence[int],
    *,
    git_commit: str,
) -> list[dict[str, Any]]:
    """Build the 14 new P1-LABEL rows; rho=0.10 reuses P1-FULL T1."""

    seeds = _unique_positive(train_seeds, name="Phase1 label train seeds")
    if len(seeds) != 5:
        raise FullAblationSpecError(
            "Phase1 label matrix requires five registered paired seeds"
        )
    concrete_commit = _require_git_commit(git_commit)
    rows: list[dict[str, Any]] = []
    for ablation_id in PHASE1_LABEL_ABLATION_IDS:
        rho = float(PHASE1_LABEL_RHOS_BY_ID[ablation_id])
        selected_seeds = seeds if abs(rho - 0.01) <= 1e-12 else seeds[:3]
        split_fractions = {
            "labeled": 0.70 * rho,
            "unlabeled": 0.70 * (1.0 - rho),
            "source_validation": 0.30,
        }
        for seed in selected_seeds:
            rows.append(
                {
                    "design_id": DESIGN_ID,
                    "design_schema": DESIGN_SCHEMA,
                    "phase": "phase1",
                    "ablation_id": ablation_id,
                    "evidence_level": (
                        "M_CONFIRMATION"
                        if abs(rho - 0.01) <= 1e-12
                        else "S_SCREENING"
                    ),
                    "mechanism_family": "label_rate_sensitivity",
                    "comparison_target": "P1-FULL@rho0.10",
                    "git_commit": concrete_commit,
                    "method_config_hash": phase1_ablation_config_hash(
                        ablation_id
                    ),
                    "rho_label": rho,
                    "split_fractions": split_fractions,
                    "train_seed": seed,
                    "epochs": 200,
                    "checkpoint_selection": "final_only",
                    "executor_status": "LOCAL_IMPLEMENTED_PENDING_REVIEW",
                }
            )
    # P1-LABEL has fourteen rows. Spread the first wave across all eight GPUs
    # before allocating the second slot so the bounded matrix uses the full
    # server instead of leaving GPU7 idle.
    balanced_slots = tuple(
        WorkerSlot(gpu=gpu, slot=slot)
        for slot in range(SLOTS_PER_GPU)
        for gpu in range(GPU_COUNT)
    )
    slots = balanced_slots[: len(rows)]
    for row, slot in zip(rows, slots):
        row["worker"] = asdict(slot)
        row["row_key"] = (
            f"{row['ablation_id']}__train_seed_{row['train_seed']}"
        )
    if len(rows) != 14:
        raise FullAblationSpecError(
            "Phase1 label matrix must contain exactly 14 new-training rows"
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
    concrete_commit = _require_git_commit(git_commit)
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
                                "git_commit": concrete_commit,
                                "protocol_schema": PROTOCOL_SCHEMA,
                                "receiver_id": receiver,
                                "k_shot": int(k_shot),
                                "old_class_count": OLD_CLASS_COUNT,
                                "new_class_count": int(new_count),
                                "scenarios": list(LEO_SCENARIOS),
                                **asdict(bundle),
                                "method_seed": int(bundle.method_seed),
                                "phase1_bundle_training_seed": None,
                                "new_class_draw_seed": int(draw_seed),
                                "data_binding_status": "UNBOUND_FAIL_CLOSED",
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


def build_phase2_e0_256_screen_rows(
    *,
    arms: Sequence[ArmSpec],
    seed_bundle: SeedBundle,
    class_draw_seed: int,
    receiver_id: str,
    k_shot: int,
    new_class_count: int,
    git_commit: str,
) -> list[dict[str, Any]]:
    """Build only the approved seven-arm current-256D same-row screen."""

    expected_ids = tuple(
        arm.ablation_id for arm in PHASE2_E0_256_ABLATION_ARMS
    )
    actual_ids = tuple(arm.ablation_id for arm in arms)
    if actual_ids != expected_ids:
        raise FullAblationSpecError(
            "current-256D screen must contain its exact seven approved arms"
        )
    if "P2-256-F0" in actual_ids or any("F0" in value for value in actual_ids):
        raise FullAblationSpecError(
            "current-256D screen must not include an FP32 F0 arm"
        )
    if str(receiver_id) != "3-19":
        raise FullAblationSpecError(
            "current-256D screen is fixed to receiver 3-19"
        )
    if (int(k_shot), int(new_class_count)) != (10, 5):
        raise FullAblationSpecError(
            "current-256D screen is fixed to K10/new5"
        )
    seed_bundle.validate(require_fresh_stage2=True)
    draw = int(class_draw_seed)
    if draw <= 0 or draw in OBSERVED_STAGE2_SEEDS:
        raise FullAblationSpecError("current-256D draw seed is not fresh")
    seed_values = tuple(int(value) for value in asdict(seed_bundle).values())
    if draw in seed_values:
        raise FullAblationSpecError(
            "current-256D draw seed must differ from all row seeds"
        )
    concrete_commit = _require_git_commit(git_commit)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        rows.append(
            {
                "design_id": DESIGN_ID,
                "design_schema": DESIGN_SCHEMA,
                "phase": "stage2c",
                "stage": "screening",
                "ablation_id": arm.ablation_id,
                "evidence_level": arm.evidence_level,
                "mechanism_family": arm.mechanism_family,
                "comparison_target": arm.comparison_target,
                "physical_config_id": (
                    arm.physical_config_id or arm.ablation_id
                ),
                "git_commit": concrete_commit,
                "protocol_schema": PROTOCOL_SCHEMA,
                "receiver_id": "3-19",
                "k_shot": 10,
                "old_class_count": OLD_CLASS_COUNT,
                "new_class_count": 5,
                "scenarios": list(LEO_SCENARIOS),
                **asdict(seed_bundle),
                "method_seed": int(seed_bundle.method_seed),
                "phase1_bundle_training_seed": None,
                "new_class_draw_seed": draw,
                "data_binding_status": "UNBOUND_FAIL_CLOSED",
                "executor_status": arm.executor_status,
                "formal_launch_authority": False,
            }
        )
    slots = assign_worker_slots(len(rows))
    for row, slot in zip(rows, slots):
        row["worker"] = asdict(slot)
        row["row_key"] = (
            f"{row['ablation_id']}__rx_3_19__k_10__new_5"
            f"__support_{row['support_seed']}__query_{row['query_seed']}"
            f"__draw_{row['new_class_draw_seed']}"
        )
    validate_plan_rows(rows)
    return rows


def build_phase2_state_rows(
    *,
    arms: Sequence[ArmSpec],
    seed_bundles: Sequence[SeedBundle],
    git_commit: str,
) -> list[dict[str, Any]]:
    """Build the independent Stage2-A/B confirmation tables.

    Stage2-A has no target support and therefore no K dimension. Stage2-B
    uses old-class support at K={1,2,5,10}, but it has neither a new-class
    count nor a new-class draw. Both tables use the five fresh confirmation
    seed bundles and keep the three LEO scenarios inside each row.
    """

    if not arms:
        raise FullAblationSpecError("at least one Stage2 state arm is required")
    if len({arm.ablation_id for arm in arms}) != len(arms):
        raise FullAblationSpecError("Stage2 state arm IDs must be unique")
    if any(arm.phase not in {"stage2a", "stage2b"} for arm in arms):
        raise FullAblationSpecError(
            "Stage2 state rows accept only Stage2-A/B arms"
        )
    if len(seed_bundles) != 5:
        raise FullAblationSpecError(
            "Stage2 state confirmation requires five seed bundles"
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
    concrete_commit = _require_git_commit(git_commit)

    rows: list[dict[str, Any]] = []
    for arm in arms:
        k_values: tuple[int | None, ...] = (
            (None,) if arm.phase == "stage2a" else CONFIRMATION_K
        )
        for receiver in TARGET_RECEIVERS:
            for k_shot in k_values:
                for bundle in seed_bundles:
                    row = {
                        "design_id": DESIGN_ID,
                        "design_schema": DESIGN_SCHEMA,
                        "phase": arm.phase,
                        "stage": "confirmation",
                        "ablation_id": arm.ablation_id,
                        "evidence_level": arm.evidence_level,
                        "mechanism_family": arm.mechanism_family,
                        "comparison_target": arm.comparison_target,
                        "physical_config_id": (
                            arm.physical_config_id or arm.ablation_id
                        ),
                        "git_commit": concrete_commit,
                        "protocol_schema": PROTOCOL_SCHEMA,
                        "receiver_id": receiver,
                        "k_shot": k_shot,
                        "old_class_count": OLD_CLASS_COUNT,
                        "new_class_count": 0,
                        "scenarios": list(LEO_SCENARIOS),
                        **asdict(bundle),
                        "support_seed": (
                            None
                            if arm.phase == "stage2a"
                            else int(bundle.support_seed)
                        ),
                        "reserved_support_seed": (
                            int(bundle.support_seed)
                            if arm.phase == "stage2a"
                            else None
                        ),
                        "target_support_access": arm.phase != "stage2a",
                        "method_seed": int(bundle.method_seed),
                        "phase1_bundle_training_seed": None,
                        "new_class_draw_seed": None,
                        "data_binding_status": "UNBOUND_FAIL_CLOSED",
                        "executor_status": arm.executor_status,
                        "formal_launch_authority": False,
                    }
                    rows.append(row)
    slots = assign_worker_slots(len(rows))
    for row, slot in zip(rows, slots):
        row["worker"] = asdict(slot)
        k_suffix = (
            "zero_support"
            if row["phase"] == "stage2a"
            else f"k_{row['k_shot']}"
        )
        row["row_key"] = (
            f"{row['ablation_id']}__rx_{row['receiver_id'].replace('-', '_')}"
            f"__{k_suffix}__method_{row['method_seed']}"
            f"__query_{row['query_seed']}"
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
        if row.get("phase") in {"stage2a", "stage2b", "stage2c"}:
            if row.get("protocol_schema") != PROTOCOL_SCHEMA:
                raise FullAblationSpecError("Stage2 protocol schema drift")
            if tuple(row.get("scenarios", ())) != LEO_SCENARIOS:
                raise FullAblationSpecError("Stage2 scenario list drift")
            if row.get("phase") == "stage2a":
                if row.get("k_shot") is not None:
                    raise FullAblationSpecError(
                        "Stage2-A cannot declare target-support K"
                    )
            elif int(row.get("k_shot", 0)) <= 0:
                raise FullAblationSpecError("Stage2-B/C K must be positive")
            if int(row.get("method_seed", -1)) != int(
                row.get("train_seed", -2)
            ):
                raise FullAblationSpecError(
                    "Stage2 method seed identity drift"
                )


def bind_stage2_row(
    row: Mapping[str, Any],
    binding: Stage2InputBinding,
) -> dict[str, Any]:
    """Attach immutable bundle/data evidence without granting launch authority."""

    if row.get("phase") != "stage2c":
        raise FullAblationSpecError("only a Stage2-C row can be data-bound")
    binding.validate()
    bound = dict(row)
    bound.update(asdict(binding))
    bound["data_binding_status"] = "BOUND_VALIDATED_ONCE"
    bound["formal_launch_authority"] = False
    validate_plan_rows([bound])
    return bound


def stage2_physical_execution_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Dynamic physical identity after effective config and data are frozen."""

    required = (
        "effective_config_hash",
        "phase1_bundle_hash",
        "capsule_id",
        "split_id",
        "receiver_id",
        "support_physical_ids_hash",
        "query_physical_ids_hash",
        "k_shot",
        "new_class_count",
        "method_seed",
        "support_seed",
        "query_seed",
        "new_class_draw_seed",
    )
    missing = [
        field
        for field in required
        if row.get(field) in (None, "")
    ]
    if missing:
        raise FullAblationSpecError(
            "physical execution key is unbound: " + ",".join(missing)
        )
    for field in (
        "effective_config_hash",
        "phase1_bundle_hash",
        "support_physical_ids_hash",
        "query_physical_ids_hash",
    ):
        if not _is_sha256(row[field]):
            raise FullAblationSpecError(
                f"physical execution key {field} must be SHA256"
            )
    return tuple(row[field] for field in required)


def validate_stage2_registry_disjointness(
    screening_bundles: Sequence[SeedBundle],
    screening_draws: Sequence[int],
    confirmation_bundles: Sequence[SeedBundle],
    confirmation_draws: Sequence[int],
) -> None:
    screening = {
        int(value)
        for bundle in screening_bundles
        for value in asdict(bundle).values()
    } | {int(value) for value in screening_draws}
    confirmation = {
        int(value)
        for bundle in confirmation_bundles
        for value in asdict(bundle).values()
    } | {int(value) for value in confirmation_draws}
    if screening & confirmation:
        raise FullAblationSpecError(
            "screening and confirmation seeds/draws overlap"
        )


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
    hash_fields = (
        "config_hash",
        "phase1_bundle_hash",
        "channel_assignment_hash",
        "old_class_ids_hash",
        "new_class_ids_hash",
        "support_physical_ids_hash",
        "query_physical_ids_hash",
        "support_query_disjoint_receipt_sha256",
        "predictions_hash",
        "score_artifact_hash",
    )
    _require_git_commit(str(record["git_commit"]))
    for field in hash_fields:
        if not _is_sha256(record[field]):
            raise FullAblationSpecError(f"{field} must be a SHA256 hex digest")
    if record["support_physical_ids_hash"] == record["query_physical_ids_hash"]:
        raise FullAblationSpecError("support/query physical ID hashes must differ")
    if int(record["support_query_overlap_count"]) != 0:
        raise FullAblationSpecError("support/query physical IDs overlap")


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
    "PHASE2_E0_256_ABLATION_ARMS",
    "PHASE2_STATE_T1_ARMS",
    "PROTOCOL_SCHEMA",
    "REQUIRED_RUN_ARTIFACT_FIELDS",
    "SCREENING_SLICES",
    "SLOTS_PER_GPU",
    "SeedBundle",
    "Stage2InputBinding",
    "TARGET_RECEIVERS",
    "assign_worker_slots",
    "build_phase1_t1_rows",
    "build_phase1_label_rows",
    "build_phase2_rows",
    "build_phase2_e0_256_screen_rows",
    "build_phase2_state_rows",
    "bind_stage2_row",
    "stage2_physical_execution_key",
    "validate_artifact_record",
    "validate_plan_rows",
    "validate_stage2_registry_disjointness",
]
