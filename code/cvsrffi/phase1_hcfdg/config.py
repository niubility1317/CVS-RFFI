"""Frozen Phase1 HCF-DG configuration and screening matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SELECTION_SEEDS: tuple[int, int, int] = (392001, 392002, 392003)

V1_OPTIMIZER_UPDATES = 4000
V2_OPTIMIZER_UPDATES = 6300
HCFDG_BATCH_SIZE = 96
V1_BATCH_SIZE = HCFDG_BATCH_SIZE
V1_BACKBONE_LR = 1e-4
V1_HEAD_LR = 3e-4
V1_WARMUP_FRACTION = 0.05
V1_COSINE_MIN_LR = 1e-6
V1_COSFACE_MARGIN_RAMP_FRACTION = 0.20
V1_COSFACE_FINAL_MARGIN = 0.30
V1_AMP_ENABLED = True


@dataclass(frozen=True)
class HCFDGConfig:
    candidate_id: str
    optimizer_updates: int
    use_dual_control: bool = False
    use_environment_encoder: bool = False
    use_rectangular_batch: bool = False
    use_lodo: bool = False
    use_csd: bool = False
    counterfactual_mode: str = "off"
    use_hdro: bool = False
    use_content_conditioning: bool = False
    residual_mode: str = "off"


@dataclass(frozen=True)
class StageBudget:
    stage0: int = 700
    stage1: int = 1200
    stage2: int = 2100
    stage3: int = 1700
    stage4: int = 600
    freeze_progress: float = 0.50
    environment_update_interval: int = 4
    environment_updates_per_interval: int = 1

    @property
    def total_updates(self) -> int:
        return self.stage0 + self.stage1 + self.stage2 + self.stage3 + self.stage4

    @property
    def stage0_updates(self) -> int:
        return self.stage0

    @property
    def stage1_updates(self) -> int:
        return self.stage1

    @property
    def stage2_updates(self) -> int:
        return self.stage2

    @property
    def stage3_updates(self) -> int:
        return self.stage3

    @property
    def stage4_updates(self) -> int:
        return self.stage4

    @property
    def freeze_point(self) -> float:
        return self.freeze_progress

    @property
    def environment_updates_per_four_main_updates(self) -> int:
        if self.environment_update_interval != 4:
            return 0
        return self.environment_updates_per_interval


V2_STAGE_BUDGET = StageBudget()


_CANDIDATE_CONFIGS: dict[str, HCFDGConfig] = {
    "A0": HCFDGConfig(
        candidate_id="A0",
        optimizer_updates=V1_OPTIMIZER_UPDATES,
        use_dual_control=True,
    ),
    "A1": HCFDGConfig(
        candidate_id="A1",
        optimizer_updates=V1_OPTIMIZER_UPDATES,
    ),
    "A2": HCFDGConfig(
        candidate_id="A2",
        optimizer_updates=V1_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
    ),
    "A3": HCFDGConfig(
        candidate_id="A3",
        optimizer_updates=V1_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
    ),
    "A4": HCFDGConfig(
        candidate_id="A4",
        optimizer_updates=V1_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
        use_lodo=True,
    ),
    "A5": HCFDGConfig(
        candidate_id="A5",
        optimizer_updates=V1_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
        use_lodo=True,
        use_csd=True,
    ),
    "A6": HCFDGConfig(
        candidate_id="A6",
        optimizer_updates=V2_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
        use_lodo=True,
        use_csd=True,
        counterfactual_mode="receiver_swap",
    ),
    "A7": HCFDGConfig(
        candidate_id="A7",
        optimizer_updates=V2_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
        use_lodo=True,
        use_csd=True,
        counterfactual_mode="receiver_day_channel_joint_curriculum",
    ),
    "A8": HCFDGConfig(
        candidate_id="A8",
        optimizer_updates=V2_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
        use_lodo=True,
        use_csd=True,
        counterfactual_mode="receiver_day_channel_joint_curriculum",
        use_hdro=True,
    ),
    "A9": HCFDGConfig(
        candidate_id="A9",
        optimizer_updates=V2_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
        use_lodo=True,
        use_csd=True,
        counterfactual_mode="receiver_day_channel_joint_curriculum",
        use_hdro=True,
        use_content_conditioning=True,
    ),
    "A10": HCFDGConfig(
        candidate_id="A10",
        optimizer_updates=V2_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
        use_lodo=True,
        use_csd=True,
        counterfactual_mode="receiver_day_channel_joint_curriculum",
        use_hdro=True,
        use_content_conditioning=True,
        residual_mode="phasedelta",
    ),
    "A11": HCFDGConfig(
        candidate_id="A11",
        optimizer_updates=V2_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
        use_lodo=True,
        use_csd=True,
        counterfactual_mode="receiver_day_channel_joint_curriculum",
        use_hdro=True,
        use_content_conditioning=True,
        residual_mode="dsq",
    ),
    "A12": HCFDGConfig(
        candidate_id="A12",
        optimizer_updates=V2_OPTIMIZER_UPDATES,
        use_environment_encoder=True,
        use_rectangular_batch=True,
        use_lodo=True,
        use_csd=True,
        counterfactual_mode="receiver_day_channel_joint_curriculum",
        use_hdro=True,
        use_content_conditioning=True,
        residual_mode="phasedelta_dsq",
    ),
}

_QUICK_CANDIDATES = tuple(f"A{i}" for i in range(6))
_DEEP_CANDIDATES = tuple(f"A{i}" for i in range(6, 10))
_RESIDUAL_CANDIDATES = ("A10", "A11", "A12")


def _candidate_key(candidate_id: str) -> str:
    if not isinstance(candidate_id, str):
        raise ValueError(f"unknown candidate: {candidate_id!r}")
    key = candidate_id.strip().upper()
    if key not in _CANDIDATE_CONFIGS:
        raise ValueError(f"unknown candidate: {candidate_id}")
    return key


def candidate_config(candidate_id: str, v2_passed: bool | None = None) -> HCFDGConfig:
    """Return the frozen definition for one explicitly named candidate.

    Matrix construction performs the V2 authorization check.  An explicit
    negative value is still rejected here so callers cannot opt into a
    residual definition while declaring that V2 did not pass.
    """

    key = _candidate_key(candidate_id)
    if key in _RESIDUAL_CANDIDATES and v2_passed is False:
        raise ValueError("A10-A12 require v2_passed=True")
    return _CANDIDATE_CONFIGS[key]


def _validated_folds(folds: Iterable[int]) -> tuple[int, int]:
    if isinstance(folds, (str, bytes)):
        raise ValueError("folds must contain exactly two folds")
    try:
        resolved = tuple(folds)
    except TypeError as exc:
        raise ValueError("folds must contain exactly two folds") from exc
    if len(resolved) != 2:
        raise ValueError("folds must contain exactly two folds")
    if len(set(resolved)) != 2:
        raise ValueError("duplicate folds are not allowed")
    if any(not isinstance(fold, int) or isinstance(fold, bool) for fold in resolved):
        raise ValueError("folds must contain integer fold IDs")
    return resolved


def _matrix_rows(
    candidate_ids: Iterable[str],
    folds: Iterable[int],
    *,
    optimizer_updates: int,
    v2_parent_candidate_id: str | None = None,
) -> tuple["MatrixRow", ...]:
    validated = _validated_folds(folds)
    return tuple(
        MatrixRow(
            candidate_id=candidate_id,
            heldout_receiver=fold,
            seed=seed,
            optimizer_updates=optimizer_updates,
            v2_parent_candidate_id=v2_parent_candidate_id,
        )
        for candidate_id in candidate_ids
        for fold in validated
        for seed in SELECTION_SEEDS
    )


@dataclass(frozen=True)
class MatrixRow:
    candidate_id: str
    heldout_receiver: int
    seed: int
    optimizer_updates: int
    gpu: int | None = None
    v2_parent_candidate_id: str | None = None


def quick_screen_rows(folds: Iterable[int]) -> tuple[MatrixRow, ...]:
    """Return the report-ordered A0-A5 two-fold, three-seed matrix."""

    return _matrix_rows(
        _QUICK_CANDIDATES,
        folds,
        optimizer_updates=V1_OPTIMIZER_UPDATES,
    )


def deep_screen_rows(folds: Iterable[int]) -> tuple[MatrixRow, ...]:
    """Return the report-ordered A6-A9 two-fold, three-seed matrix."""

    return _matrix_rows(
        _DEEP_CANDIDATES,
        folds,
        optimizer_updates=V2_OPTIMIZER_UPDATES,
    )


def residual_rows(
    folds: Iterable[int],
    v2_passed: bool = False,
    *,
    v2_parent_candidate_id: str = "A9",
) -> tuple[MatrixRow, ...]:
    """Return residual rows only after a passed A8/A9 V2 parent is selected."""

    if not v2_passed:
        raise ValueError("residual rows require v2_passed=True")
    parent = _candidate_key(v2_parent_candidate_id)
    if parent not in {"A8", "A9"}:
        raise ValueError("v2_parent_candidate_id must be A8 or A9")
    return _matrix_rows(
        _RESIDUAL_CANDIDATES,
        folds,
        optimizer_updates=V2_OPTIMIZER_UPDATES,
        v2_parent_candidate_id=parent,
    )


__all__ = [
    "HCFDG_BATCH_SIZE",
    "HCFDGConfig",
    "MatrixRow",
    "SELECTION_SEEDS",
    "StageBudget",
    "V1_AMP_ENABLED",
    "V1_BACKBONE_LR",
    "V1_BATCH_SIZE",
    "V1_COSFACE_FINAL_MARGIN",
    "V1_COSFACE_MARGIN_RAMP_FRACTION",
    "V1_COSINE_MIN_LR",
    "V1_HEAD_LR",
    "V1_OPTIMIZER_UPDATES",
    "V1_WARMUP_FRACTION",
    "V2_OPTIMIZER_UPDATES",
    "V2_STAGE_BUDGET",
    "candidate_config",
    "deep_screen_rows",
    "quick_screen_rows",
    "residual_rows",
]
