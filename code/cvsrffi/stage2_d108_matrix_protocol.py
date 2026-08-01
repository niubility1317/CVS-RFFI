"""Frozen, truth-free Target125 topology for D108-CB-RRC-SMME.

This module models only the legal execution topology.  It never opens query
labels, roles, metrics, or an arm-selection surface.  The resulting matrix is
the shared identity for the D108 input binder, runner, and independent
truth-side scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


MATRIX_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target125.matrix.v1"
PROTOCOL_SCHEMA = "p2_min_v1"
CANDIDATE_ID = "D108-CB-RRC-SMME/r1"

RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713102, 713103, 713104, 713105, 713106)
TARGET125_SLICES = ((10, 5), (10, 10), (10, 20), (5, 20), (1, 20))
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
PHASES = ("before", "after")

OLD_CLASS_COUNT = 6
FEATURE_WIDTH = 288
OUTER_JOB_COUNT = 125
SCENE_ROW_COUNT = 375
ARM_PAIR_COUNT = 1500
SURFACE_COUNT = 3000

MATCHED_ARM_PAIR_COUNT = ARM_PAIR_COUNT
STATE_SURFACE_COUNT = SURFACE_COUNT

ACCESS_LEDGER = {
    "clean_source_runtime_access": False,
    "query_fit_access": False,
    "query_update_access": False,
    "query_truth_access": False,
    "query_role_access": False,
    "query_selection_access": False,
}

TRACEABILITY: tuple[Mapping[str, str], ...] = (
    {
        "id": "D108-T125-01",
        "source_section": "D108-CB-RRC-SMME/r1 frozen execution request",
        "requirement": "freeze 5 receiver x 5 seed x 5 slice topology",
        "target": "freeze_d108_target125_matrix",
        "status": "implemented",
        "verification": "test_target125_matrix_counts_and_ids",
    },
    {
        "id": "D108-T125-02",
        "source_section": "p2_min_v1",
        "requirement": "four fixed causal arms and no routed/K selector",
        "target": "ARMS and D108ArmPair",
        "status": "implemented",
        "verification": "test_four_fixed_arms_without_routing",
    },
    {
        "id": "D108-T125-03",
        "source_section": "truth-after-seal handoff",
        "requirement": "stable D108 IDs and exact 125/375/1500/3000 coverage",
        "target": "D108MatrixPlan and audit_surface_coverage",
        "status": "implemented",
        "verification": "test_missing_surface_and_tamper_fail_closed",
    },
)


class D108MatrixProtocolError(ValueError):
    """Raised when the frozen D108 execution topology drifts."""


def canonical_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 JSON representation used for receipts."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D108MatrixProtocolError("canonical JSON payload is invalid") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise D108MatrixProtocolError(f"{name} must be a non-empty exact string")
    return value


def _require_member(value: Any, allowed: Sequence[str], name: str) -> str:
    text = _require_text(value, name)
    if text not in allowed:
        raise D108MatrixProtocolError(f"{name} is outside the frozen allowlist")
    return text


def _require_slice(k_shot: Any, new_count: Any, name: str) -> tuple[int, int]:
    if type(k_shot) is not int or type(new_count) is not int:
        raise D108MatrixProtocolError(f"{name} values must be exact integers")
    pair = (k_shot, new_count)
    if pair not in TARGET125_SLICES:
        raise D108MatrixProtocolError(f"{name} is outside the frozen Target125 slices")
    return pair


def make_outer_id(receiver: str, seed: int, k_shot: int, new_count: int) -> str:
    receiver = _require_member(receiver, RECEIVERS, "receiver")
    if type(seed) is not int or seed not in SEEDS:
        raise D108MatrixProtocolError("seed is outside the frozen Target125 grid")
    k_shot, new_count = _require_slice(k_shot, new_count, "outer slice")
    return f"d108-rx-{receiver}__seed-{seed}__k-{k_shot}__new-{new_count}"


def make_scene_row_id(outer_id: str, scene: str) -> str:
    _require_text(outer_id, "outer_id")
    return f"{outer_id}__scene-{_require_member(scene, SCENES, 'scene')}"


def make_arm_pair_id(scene_row_id: str, arm: str) -> str:
    _require_text(scene_row_id, "scene_row_id")
    return f"{scene_row_id}__arm-{_require_member(arm, ARMS, 'arm')}"


def make_surface_id(arm_pair_id: str, phase: str) -> str:
    _require_text(arm_pair_id, "arm_pair_id")
    return f"{arm_pair_id}__phase-{_require_member(phase, PHASES, 'phase')}"


@dataclass(frozen=True, slots=True)
class D108OuterRow:
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int

    def __post_init__(self) -> None:
        receiver = _require_member(self.receiver, RECEIVERS, "outer receiver")
        if type(self.seed) is not int or self.seed not in SEEDS:
            raise D108MatrixProtocolError("outer seed drift")
        k_shot, new_count = _require_slice(self.k_shot, self.new_count, "outer slice")
        if self.outer_id != make_outer_id(receiver, self.seed, k_shot, new_count):
            raise D108MatrixProtocolError("outer_id drift")

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_id": self.outer_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
        }


@dataclass(frozen=True, slots=True)
class D108SceneRow:
    scene_row_id: str
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    scene: str

    def __post_init__(self) -> None:
        outer = D108OuterRow(
            self.outer_id, self.receiver, self.seed, self.k_shot, self.new_count
        )
        if self.scene_row_id != make_scene_row_id(outer.outer_id, self.scene):
            raise D108MatrixProtocolError("scene_row_id drift")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_row_id": self.scene_row_id,
            "outer_id": self.outer_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "scene": self.scene,
        }


@dataclass(frozen=True, slots=True)
class D108ArmPair:
    arm_pair_id: str
    scene_row_id: str
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    scene: str
    arm: str

    def __post_init__(self) -> None:
        row = D108SceneRow(
            self.scene_row_id,
            self.outer_id,
            self.receiver,
            self.seed,
            self.k_shot,
            self.new_count,
            self.scene,
        )
        if self.arm_pair_id != make_arm_pair_id(row.scene_row_id, self.arm):
            raise D108MatrixProtocolError("arm_pair_id drift")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_pair_id": self.arm_pair_id,
            "scene_row_id": self.scene_row_id,
            "outer_id": self.outer_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "scene": self.scene,
            "arm": self.arm,
        }


@dataclass(frozen=True, slots=True)
class D108Surface:
    surface_id: str
    arm_pair_id: str
    scene_row_id: str
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    scene: str
    arm: str
    phase: str

    def __post_init__(self) -> None:
        pair = D108ArmPair(
            self.arm_pair_id,
            self.scene_row_id,
            self.outer_id,
            self.receiver,
            self.seed,
            self.k_shot,
            self.new_count,
            self.scene,
            self.arm,
        )
        if self.surface_id != make_surface_id(pair.arm_pair_id, self.phase):
            raise D108MatrixProtocolError("surface_id drift")

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "arm_pair_id": self.arm_pair_id,
            "scene_row_id": self.scene_row_id,
            "outer_id": self.outer_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "scene": self.scene,
            "arm": self.arm,
            "phase": self.phase,
        }


@dataclass(frozen=True, slots=True)
class D108MatrixPlan:
    outer_rows: tuple[D108OuterRow, ...]
    scene_rows: tuple[D108SceneRow, ...]
    arm_pairs: tuple[D108ArmPair, ...]
    surfaces: tuple[D108Surface, ...]
    matrix_receipt_sha256: str

    def receipt_payload(self) -> dict[str, Any]:
        payload = _matrix_payload(
            self.outer_rows, self.scene_rows, self.arm_pairs, self.surfaces
        )
        return {**payload, "matrix_receipt_sha256": self.matrix_receipt_sha256}


def _matrix_payload(
    outer_rows: Sequence[D108OuterRow],
    scene_rows: Sequence[D108SceneRow],
    arm_pairs: Sequence[D108ArmPair],
    surfaces: Sequence[D108Surface],
) -> dict[str, Any]:
    return {
        "schema": MATRIX_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT,
        "surface_count": SURFACE_COUNT,
        "receivers": list(RECEIVERS),
        "seeds": list(SEEDS),
        "slices": [
            {"k_shot": k_shot, "new_count": new_count}
            for k_shot, new_count in TARGET125_SLICES
        ],
        "scenes": list(SCENES),
        "arms": list(ARMS),
        "phases": list(PHASES),
        "access_ledger": dict(ACCESS_LEDGER),
        "outer_rows": [row.as_dict() for row in outer_rows],
        "scene_rows": [row.as_dict() for row in scene_rows],
        "arm_pairs": [pair.as_dict() for pair in arm_pairs],
        "surfaces": [surface.as_dict() for surface in surfaces],
    }


def _enumerate_matrix() -> tuple[
    tuple[D108OuterRow, ...],
    tuple[D108SceneRow, ...],
    tuple[D108ArmPair, ...],
    tuple[D108Surface, ...],
]:
    outer_rows: list[D108OuterRow] = []
    scene_rows: list[D108SceneRow] = []
    arm_pairs: list[D108ArmPair] = []
    surfaces: list[D108Surface] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for k_shot, new_count in TARGET125_SLICES:
                outer_id = make_outer_id(receiver, seed, k_shot, new_count)
                outer = D108OuterRow(outer_id, receiver, seed, k_shot, new_count)
                outer_rows.append(outer)
                for scene in SCENES:
                    scene_row_id = make_scene_row_id(outer_id, scene)
                    scene_row = D108SceneRow(
                        scene_row_id,
                        outer_id,
                        receiver,
                        seed,
                        k_shot,
                        new_count,
                        scene,
                    )
                    scene_rows.append(scene_row)
                    for arm in ARMS:
                        arm_pair_id = make_arm_pair_id(scene_row_id, arm)
                        pair = D108ArmPair(
                            arm_pair_id,
                            scene_row_id,
                            outer_id,
                            receiver,
                            seed,
                            k_shot,
                            new_count,
                            scene,
                            arm,
                        )
                        arm_pairs.append(pair)
                        for phase in PHASES:
                            surfaces.append(
                                D108Surface(
                                    make_surface_id(arm_pair_id, phase),
                                    arm_pair_id,
                                    scene_row_id,
                                    outer_id,
                                    receiver,
                                    seed,
                                    k_shot,
                                    new_count,
                                    scene,
                                    arm,
                                    phase,
                                )
                            )
    return tuple(outer_rows), tuple(scene_rows), tuple(arm_pairs), tuple(surfaces)


def validate_d108_target125_matrix(plan: D108MatrixPlan) -> None:
    """Fail closed unless every frozen identifier and count is exact."""

    if type(plan) is not D108MatrixPlan:
        raise D108MatrixProtocolError("matrix plan must use the exact D108 type")
    expected = _enumerate_matrix()
    actual = (plan.outer_rows, plan.scene_rows, plan.arm_pairs, plan.surfaces)
    if actual != expected:
        raise D108MatrixProtocolError("Target125 matrix row/order/ID coverage drift")
    if (
        len(plan.outer_rows) != OUTER_JOB_COUNT
        or len(plan.scene_rows) != SCENE_ROW_COUNT
        or len(plan.arm_pairs) != ARM_PAIR_COUNT
        or len(plan.surfaces) != SURFACE_COUNT
    ):
        raise D108MatrixProtocolError("Target125 count closure drift")
    if plan.matrix_receipt_sha256 != canonical_sha256(_matrix_payload(*expected)):
        raise D108MatrixProtocolError("Target125 matrix receipt drift")


def freeze_d108_target125_matrix() -> D108MatrixPlan:
    """Construct the one and only truth-free D108 Target125 matrix."""

    outer_rows, scene_rows, arm_pairs, surfaces = _enumerate_matrix()
    plan = D108MatrixPlan(
        outer_rows=outer_rows,
        scene_rows=scene_rows,
        arm_pairs=arm_pairs,
        surfaces=surfaces,
        matrix_receipt_sha256=canonical_sha256(
            _matrix_payload(outer_rows, scene_rows, arm_pairs, surfaces)
        ),
    )
    validate_d108_target125_matrix(plan)
    return plan


def audit_surface_coverage(surface_ids: Iterable[str]) -> None:
    """Reject missing, duplicate, or extra Target125 prediction surfaces."""

    values = list(surface_ids)
    expected = [surface.surface_id for surface in freeze_d108_target125_matrix().surfaces]
    if len(values) != SURFACE_COUNT or len(set(values)) != SURFACE_COUNT:
        raise D108MatrixProtocolError("surface coverage is incomplete or duplicated")
    if set(values) != set(expected):
        raise D108MatrixProtocolError("surface coverage has missing or extra IDs")


__all__ = [
    "ACCESS_LEDGER",
    "ARMS",
    "ARM_PAIR_COUNT",
    "CANDIDATE_ID",
    "D108ArmPair",
    "D108MatrixPlan",
    "D108MatrixProtocolError",
    "D108OuterRow",
    "D108SceneRow",
    "D108Surface",
    "FEATURE_WIDTH",
    "MATCHED_ARM_PAIR_COUNT",
    "MATRIX_SCHEMA",
    "OLD_CLASS_COUNT",
    "OUTER_JOB_COUNT",
    "PHASES",
    "PROTOCOL_SCHEMA",
    "RECEIVERS",
    "SCENES",
    "SCENE_ROW_COUNT",
    "SEEDS",
    "STATE_SURFACE_COUNT",
    "SURFACE_COUNT",
    "TARGET125_SLICES",
    "TRACEABILITY",
    "audit_surface_coverage",
    "canonical_bytes",
    "canonical_sha256",
    "freeze_d108_target125_matrix",
    "make_arm_pair_id",
    "make_outer_id",
    "make_scene_row_id",
    "make_surface_id",
    "validate_d108_target125_matrix",
]
