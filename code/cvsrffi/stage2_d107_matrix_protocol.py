"""Frozen, truth-free Target125 topology for D107-SCMKRR.

This module deliberately models only the legal execution topology.  It does
not open query labels, roles, metrics, or any routing surface.  The resulting
plan is the shared identity used by the D107 input builder, predictor, and the
independent truth-side scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


MATRIX_SCHEMA = "cvs.phase2.d107.scmkrr.target125.matrix.v1"
PROTOCOL_SCHEMA = "p2_min_v1"
CANDIDATE_ID = "D107-SCMKRR/r1"

RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713102, 713103, 713104, 713105, 713106)
TARGET125_SLICES = ((10, 5), (10, 10), (10, 20), (5, 20), (1, 20))
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
PHASES = ("before", "after")

OLD_CLASS_COUNT = 6
FEATURE_WIDTH = 160
OUTER_JOB_COUNT = 125
SCENE_ROW_COUNT = 375
ARM_PAIR_COUNT = 1500
SURFACE_COUNT = 3000

# Explicit aliases make the 125/375/1500/3000 closure readable to callers.
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

# The traceability record is intentionally local to this owned implementation
# surface: the task grants no separate report file to this sub-agent.
TRACEABILITY: tuple[Mapping[str, str], ...] = (
    {
        "id": "D107-T125-01",
        "source_section": "Target125 execution request",
        "requirement": "freeze 5 receiver x 5 seed x 5 slice topology",
        "target": "freeze_d107_target125_matrix",
        "status": "implemented",
        "verification": "test_target125_matrix_counts_and_ids",
    },
    {
        "id": "D107-T125-02",
        "source_section": "p2_min_v1",
        "requirement": "four fixed arms and no routed/K selector",
        "target": "ARMS and D107ArmPair",
        "status": "implemented",
        "verification": "test_four_fixed_arms_without_routing",
    },
    {
        "id": "D107-T125-03",
        "source_section": "truth-side handoff contract",
        "requirement": "stable IDs and exact 125/375/1500/3000 coverage",
        "target": "D107MatrixPlan and audit_surface_coverage",
        "status": "implemented",
        "verification": "test_missing_surface_and_tamper_fail_closed",
    },
)


class D107MatrixProtocolError(ValueError):
    """Raised when the frozen D107 execution topology drifts."""


def canonical_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 JSON representation used for all receipts."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D107MatrixProtocolError("canonical JSON payload is invalid") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise D107MatrixProtocolError(f"{name} must be a non-empty exact string")
    return value


def _require_exact_int(value: Any, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        raise D107MatrixProtocolError(f"{name} must equal frozen value {expected}")
    return value


def _require_member(value: Any, allowed: Sequence[str], name: str) -> str:
    text = _require_text(value, name)
    if text not in allowed:
        raise D107MatrixProtocolError(f"{name} is outside the frozen allowlist")
    return text


def _require_slice(k_shot: Any, new_count: Any, name: str) -> tuple[int, int]:
    if type(k_shot) is not int or type(new_count) is not int:
        raise D107MatrixProtocolError(f"{name} values must be exact integers")
    pair = (k_shot, new_count)
    if pair not in TARGET125_SLICES:
        raise D107MatrixProtocolError(f"{name} is outside the frozen Target125 slices")
    return pair


def make_outer_id(receiver: str, seed: int, k_shot: int, new_count: int) -> str:
    """Return the scorer-stable Target125 outer identifier.

    Receiver names are deliberately retained verbatim here.  The allowed
    receiver tokens contain no path separators and are part of the manifest
    contract, rather than a cosmetic file-name transform.
    """

    receiver = _require_member(receiver, RECEIVERS, "receiver")
    if type(seed) is not int or seed not in SEEDS:
        raise D107MatrixProtocolError("seed is outside the frozen Target125 grid")
    k_shot, new_count = _require_slice(k_shot, new_count, "outer slice")
    return f"d107-rx-{receiver}__seed-{seed}__k-{k_shot}__new-{new_count}"


def make_scene_row_id(outer_id: str, scene: str) -> str:
    _require_text(outer_id, "outer_id")
    scene = _require_member(scene, SCENES, "scene")
    return f"{outer_id}__scene-{scene}"


def make_arm_pair_id(scene_row_id: str, arm: str) -> str:
    _require_text(scene_row_id, "scene_row_id")
    arm = _require_member(arm, ARMS, "arm")
    return f"{scene_row_id}__arm-{arm}"


def make_surface_id(arm_pair_id: str, phase: str) -> str:
    _require_text(arm_pair_id, "arm_pair_id")
    phase = _require_member(phase, PHASES, "phase")
    return f"{arm_pair_id}__phase-{phase}"


@dataclass(frozen=True, slots=True)
class D107OuterRow:
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int

    def __post_init__(self) -> None:
        receiver = _require_member(self.receiver, RECEIVERS, "outer receiver")
        if type(self.seed) is not int or self.seed not in SEEDS:
            raise D107MatrixProtocolError("outer seed drift")
        k_shot, new_count = _require_slice(
            self.k_shot, self.new_count, "outer slice"
        )
        expected = make_outer_id(receiver, self.seed, k_shot, new_count)
        if self.outer_id != expected:
            raise D107MatrixProtocolError("outer_id drift")

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_id": self.outer_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
        }


@dataclass(frozen=True, slots=True)
class D107SceneRow:
    scene_row_id: str
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    scene: str

    def __post_init__(self) -> None:
        outer = D107OuterRow(
            self.outer_id, self.receiver, self.seed, self.k_shot, self.new_count
        )
        scene = _require_member(self.scene, SCENES, "scene")
        if self.scene_row_id != make_scene_row_id(outer.outer_id, scene):
            raise D107MatrixProtocolError("scene_row_id drift")

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
class D107ArmPair:
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
        row = D107SceneRow(
            self.scene_row_id,
            self.outer_id,
            self.receiver,
            self.seed,
            self.k_shot,
            self.new_count,
            self.scene,
        )
        arm = _require_member(self.arm, ARMS, "arm")
        if self.arm_pair_id != make_arm_pair_id(row.scene_row_id, arm):
            raise D107MatrixProtocolError("arm_pair_id drift")

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
class D107Surface:
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
        pair = D107ArmPair(
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
        phase = _require_member(self.phase, PHASES, "phase")
        if self.surface_id != make_surface_id(pair.arm_pair_id, phase):
            raise D107MatrixProtocolError("surface_id drift")

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
class D107MatrixPlan:
    outer_rows: tuple[D107OuterRow, ...]
    scene_rows: tuple[D107SceneRow, ...]
    arm_pairs: tuple[D107ArmPair, ...]
    surfaces: tuple[D107Surface, ...]
    matrix_receipt_sha256: str

    def receipt_payload(self) -> dict[str, Any]:
        payload = _matrix_payload(
            self.outer_rows, self.scene_rows, self.arm_pairs, self.surfaces
        )
        return {**payload, "matrix_receipt_sha256": self.matrix_receipt_sha256}


def _matrix_payload(
    outer_rows: Sequence[D107OuterRow],
    scene_rows: Sequence[D107SceneRow],
    arm_pairs: Sequence[D107ArmPair],
    surfaces: Sequence[D107Surface],
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
    tuple[D107OuterRow, ...],
    tuple[D107SceneRow, ...],
    tuple[D107ArmPair, ...],
    tuple[D107Surface, ...],
]:
    outer_rows: list[D107OuterRow] = []
    scene_rows: list[D107SceneRow] = []
    arm_pairs: list[D107ArmPair] = []
    surfaces: list[D107Surface] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for k_shot, new_count in TARGET125_SLICES:
                outer_id = make_outer_id(receiver, seed, k_shot, new_count)
                outer = D107OuterRow(
                    outer_id, receiver, seed, k_shot, new_count
                )
                outer_rows.append(outer)
                for scene in SCENES:
                    scene_row_id = make_scene_row_id(outer_id, scene)
                    scene_row = D107SceneRow(
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
                        pair = D107ArmPair(
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
                                D107Surface(
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
    return (
        tuple(outer_rows),
        tuple(scene_rows),
        tuple(arm_pairs),
        tuple(surfaces),
    )


def validate_d107_target125_matrix(plan: D107MatrixPlan) -> None:
    """Fail closed unless every frozen identifier and count is exact."""

    if type(plan) is not D107MatrixPlan:
        raise D107MatrixProtocolError("matrix plan must use the exact D107 type")
    expected = _enumerate_matrix()
    actual = (plan.outer_rows, plan.scene_rows, plan.arm_pairs, plan.surfaces)
    if actual != expected:
        raise D107MatrixProtocolError("Target125 matrix row/order/ID coverage drift")
    if (
        len(plan.outer_rows) != OUTER_JOB_COUNT
        or len(plan.scene_rows) != SCENE_ROW_COUNT
        or len(plan.arm_pairs) != ARM_PAIR_COUNT
        or len(plan.surfaces) != SURFACE_COUNT
    ):
        raise D107MatrixProtocolError("Target125 count closure drift")
    payload = _matrix_payload(*expected)
    if plan.matrix_receipt_sha256 != canonical_sha256(payload):
        raise D107MatrixProtocolError("Target125 matrix receipt drift")


def freeze_d107_target125_matrix() -> D107MatrixPlan:
    """Construct the one and only truth-free D107 Target125 plan."""

    outer_rows, scene_rows, arm_pairs, surfaces = _enumerate_matrix()
    receipt = canonical_sha256(
        _matrix_payload(outer_rows, scene_rows, arm_pairs, surfaces)
    )
    plan = D107MatrixPlan(
        outer_rows=outer_rows,
        scene_rows=scene_rows,
        arm_pairs=arm_pairs,
        surfaces=surfaces,
        matrix_receipt_sha256=receipt,
    )
    validate_d107_target125_matrix(plan)
    return plan


def audit_surface_coverage(surface_ids: Iterable[str]) -> None:
    """Reject missing, duplicate, or extra Target125 prediction surfaces."""

    values = list(surface_ids)
    expected = [surface.surface_id for surface in freeze_d107_target125_matrix().surfaces]
    if len(values) != SURFACE_COUNT or len(set(values)) != SURFACE_COUNT:
        raise D107MatrixProtocolError("surface coverage is incomplete or duplicated")
    if set(values) != set(expected):
        raise D107MatrixProtocolError("surface coverage has missing or extra IDs")


__all__ = [
    "ACCESS_LEDGER",
    "ARMS",
    "ARM_PAIR_COUNT",
    "CANDIDATE_ID",
    "D107ArmPair",
    "D107MatrixPlan",
    "D107MatrixProtocolError",
    "D107OuterRow",
    "D107SceneRow",
    "D107Surface",
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
    "freeze_d107_target125_matrix",
    "make_arm_pair_id",
    "make_outer_id",
    "make_scene_row_id",
    "make_surface_id",
    "validate_d107_target125_matrix",
]
