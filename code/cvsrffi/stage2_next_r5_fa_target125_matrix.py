"""Frozen truth-free topology for NEXT-R5 FA-RDCE3 -> qKNN Target125.

This module intentionally models execution identity only.  It contains no
query labels, class roles, scores, candidate routing, or performance values.
The 125 outer cells reuse the sealed Target125 receiver/seed/slice topology,
but expose the four explicit DA/registration states required by the current
Stage2 reporting convention.

K=1 has no FA state.  Its two DA1 surfaces are therefore logical aliases of
the matching DA0 qKNN predictions.  K=5 and K=10 have four distinct
predictions per scene.  This gives exactly 1500 logical surfaces, 1350 unique
prediction artifacts, and 150 aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


MATRIX_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.matrix.v1"
PROTOCOL_SCHEMA = "p2_min_v1"
CANDIDATE_ID = "NEXT-R5-FA-RDCE3-Q-TARGET125"

RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713102, 713103, 713104, 713105, 713106)
TARGET125_SLICES = ((10, 5), (10, 10), (10, 20), (5, 20), (1, 20))
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
STATES = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
REG0_STATES = frozenset(("DA0_REG0", "DA1_REG0"))
REG1_STATES = frozenset(("DA0_REG1", "DA1_REG1"))
DA1_STATES = frozenset(("DA1_REG0", "DA1_REG1"))

STATE_NAMES_ZH = {
    "DA0_REG0": "域适应前/新类注册前",
    "DA1_REG0": "域适应后/新类注册前",
    "DA0_REG1": "域适应前/新类注册后",
    "DA1_REG1": "域适应后/新类注册后",
}
METRIC_AVAILABILITY = {
    "DA0_REG0": {"seen_new_acc": "N/A", "H_old_new": "N/A"},
    "DA1_REG0": {"seen_new_acc": "N/A", "H_old_new": "N/A"},
    "DA0_REG1": {"seen_new_acc": "REQUIRED", "H_old_new": "REQUIRED"},
    "DA1_REG1": {"seen_new_acc": "REQUIRED", "H_old_new": "REQUIRED"},
}

FEATURE_DIM = 160
OLD_CLASS_COUNT = 6
OUTER_JOB_COUNT = 125
SCENE_ROW_COUNT = 375
LOGICAL_STATE_SURFACE_COUNT = 1500
UNIQUE_PREDICTION_COUNT = 1350
ALIAS_COUNT = 150

ACCESS_LEDGER = {
    "clean_source_runtime_access": False,
    "query_fit_access": False,
    "query_update_access": False,
    "query_truth_access": False,
    "query_role_access": False,
    "query_selection_access": False,
    "query_batch_dependency": False,
    "class_quota_access": False,
    "global_reassignment_access": False,
}


class NextR5FATarget125MatrixError(ValueError):
    """Raised when the frozen FA-RDCE3 Target125 topology drifts."""


def canonical_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes used for immutable receipts."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NextR5FATarget125MatrixError("canonical matrix payload is invalid") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise NextR5FATarget125MatrixError(f"{name} must be a non-empty exact string")
    return value


def _member(value: Any, allowed: Sequence[str], name: str) -> str:
    text = _text(value, name)
    if text not in allowed:
        raise NextR5FATarget125MatrixError(f"{name} is outside the frozen allowlist")
    return text


def _slice(k_shot: Any, new_count: Any, name: str) -> tuple[int, int]:
    if type(k_shot) is not int or type(new_count) is not int:
        raise NextR5FATarget125MatrixError(f"{name} must use exact integer values")
    value = (k_shot, new_count)
    if value not in TARGET125_SLICES:
        raise NextR5FATarget125MatrixError(f"{name} is outside frozen Target125 slices")
    return value


def source_pool_k_for(k_shot: int, new_count: int) -> int:
    """Return the sealed support-package K used to materialize one logical K.

    K5/new20 is intentionally the prefix of the matched K10/new20 package.
    This is an input-selection fact, not a candidate-specific support search.
    """

    k_shot, new_count = _slice(k_shot, new_count, "source-pool slice")
    return 10 if (k_shot, new_count) == (5, 20) else k_shot


def make_outer_id(receiver: str, seed: int, k_shot: int, new_count: int) -> str:
    receiver = _member(receiver, RECEIVERS, "receiver")
    if type(seed) is not int or seed not in SEEDS:
        raise NextR5FATarget125MatrixError("seed is outside frozen Target125 grid")
    k_shot, new_count = _slice(k_shot, new_count, "outer slice")
    return (
        f"next-r5-rx-{receiver}__seed-{seed}__k-{k_shot}"
        f"__new-{new_count}"
    )


def make_scene_row_id(outer_id: str, scene: str) -> str:
    return f"{_text(outer_id, 'outer_id')}__scene-{_member(scene, SCENES, 'scene')}"


def make_surface_id(scene_row_id: str, state: str) -> str:
    return f"{_text(scene_row_id, 'scene_row_id')}__state-{_member(state, STATES, 'state')}"


def _alias_target(scene_row_id: str, state: str, k_shot: int) -> str | None:
    if k_shot != 1 or state not in DA1_STATES:
        return None
    source_state = "DA0_REG0" if state == "DA1_REG0" else "DA0_REG1"
    return make_surface_id(scene_row_id, source_state)


@dataclass(frozen=True, slots=True)
class Target125OuterRow:
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    source_pool_k: int

    def __post_init__(self) -> None:
        receiver = _member(self.receiver, RECEIVERS, "outer receiver")
        if type(self.seed) is not int or self.seed not in SEEDS:
            raise NextR5FATarget125MatrixError("outer seed drift")
        k_shot, new_count = _slice(self.k_shot, self.new_count, "outer slice")
        if self.outer_id != make_outer_id(receiver, self.seed, k_shot, new_count):
            raise NextR5FATarget125MatrixError("outer_id drift")
        if self.source_pool_k != source_pool_k_for(k_shot, new_count):
            raise NextR5FATarget125MatrixError("outer source_pool_k drift")

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_id": self.outer_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "source_pool_k": self.source_pool_k,
        }


@dataclass(frozen=True, slots=True)
class Target125SceneRow:
    scene_row_id: str
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    source_pool_k: int
    scene: str

    def __post_init__(self) -> None:
        outer = Target125OuterRow(
            self.outer_id,
            self.receiver,
            self.seed,
            self.k_shot,
            self.new_count,
            self.source_pool_k,
        )
        if self.scene_row_id != make_scene_row_id(outer.outer_id, self.scene):
            raise NextR5FATarget125MatrixError("scene_row_id drift")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_row_id": self.scene_row_id,
            "outer_id": self.outer_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "source_pool_k": self.source_pool_k,
            "scene": self.scene,
        }


@dataclass(frozen=True, slots=True)
class Target125StateSurface:
    surface_id: str
    scene_row_id: str
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    source_pool_k: int
    scene: str
    state: str
    unique_prediction: bool
    alias_of_surface_id: str | None

    def __post_init__(self) -> None:
        scene_row = Target125SceneRow(
            self.scene_row_id,
            self.outer_id,
            self.receiver,
            self.seed,
            self.k_shot,
            self.new_count,
            self.source_pool_k,
            self.scene,
        )
        state = _member(self.state, STATES, "state surface state")
        if self.surface_id != make_surface_id(scene_row.scene_row_id, state):
            raise NextR5FATarget125MatrixError("surface_id drift")
        expected_alias = _alias_target(scene_row.scene_row_id, state, self.k_shot)
        if self.alias_of_surface_id != expected_alias:
            raise NextR5FATarget125MatrixError("K1 alias target drift")
        if self.unique_prediction is not (expected_alias is None):
            raise NextR5FATarget125MatrixError("unique prediction flag drift")

    @property
    def registration_phase(self) -> str:
        return "REG0" if self.state in REG0_STATES else "REG1"

    @property
    def da_phase(self) -> str:
        return "DA1" if self.state in DA1_STATES else "DA0"

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "scene_row_id": self.scene_row_id,
            "outer_id": self.outer_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "source_pool_k": self.source_pool_k,
            "scene": self.scene,
            "state": self.state,
            "state_name_zh": STATE_NAMES_ZH[self.state],
            "registration_phase": self.registration_phase,
            "da_phase": self.da_phase,
            "metric_availability": dict(METRIC_AVAILABILITY[self.state]),
            "unique_prediction": self.unique_prediction,
            "alias_of_surface_id": self.alias_of_surface_id,
        }


@dataclass(frozen=True, slots=True)
class Target125MatrixPlan:
    outer_rows: tuple[Target125OuterRow, ...]
    scene_rows: tuple[Target125SceneRow, ...]
    surfaces: tuple[Target125StateSurface, ...]
    matrix_receipt_sha256: str

    def receipt_payload(self) -> dict[str, Any]:
        return {
            **_matrix_payload(self.outer_rows, self.scene_rows, self.surfaces),
            "matrix_receipt_sha256": self.matrix_receipt_sha256,
        }


def _matrix_payload(
    outer_rows: Sequence[Target125OuterRow],
    scene_rows: Sequence[Target125SceneRow],
    surfaces: Sequence[Target125StateSurface],
) -> dict[str, Any]:
    return {
        "schema": MATRIX_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "old_class_count": OLD_CLASS_COUNT,
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "logical_state_surface_count": LOGICAL_STATE_SURFACE_COUNT,
        "unique_prediction_count": UNIQUE_PREDICTION_COUNT,
        "alias_count": ALIAS_COUNT,
        "receivers": list(RECEIVERS),
        "seeds": list(SEEDS),
        "slices": [
            {"k_shot": k_shot, "new_count": new_count}
            for k_shot, new_count in TARGET125_SLICES
        ],
        "scenes": list(SCENES),
        "states": list(STATES),
        "state_names_zh": dict(STATE_NAMES_ZH),
        "metric_availability": {key: dict(value) for key, value in METRIC_AVAILABILITY.items()},
        "access_ledger": dict(ACCESS_LEDGER),
        "k1_alias_policy": {
            "fit_mode": "FA_STRICT_BYPASS",
            "DA1_REG0": "DA0_REG0_exact_alias",
            "DA1_REG1": "DA0_REG1_exact_alias",
        },
        "k5_k10_policy": {
            "fit_mode": "FISHER_CLOSED_FORM",
            "fit_support": "REG0_old_class_support_only",
            "reg1_state": "same_object_reuse_from_DA1_REG0",
        },
        "outer_rows": [row.as_dict() for row in outer_rows],
        "scene_rows": [row.as_dict() for row in scene_rows],
        "surfaces": [surface.as_dict() for surface in surfaces],
    }


def _enumerate_matrix() -> tuple[
    tuple[Target125OuterRow, ...],
    tuple[Target125SceneRow, ...],
    tuple[Target125StateSurface, ...],
]:
    outer_rows: list[Target125OuterRow] = []
    scene_rows: list[Target125SceneRow] = []
    surfaces: list[Target125StateSurface] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for k_shot, new_count in TARGET125_SLICES:
                outer_id = make_outer_id(receiver, seed, k_shot, new_count)
                outer = Target125OuterRow(
                    outer_id,
                    receiver,
                    seed,
                    k_shot,
                    new_count,
                    source_pool_k_for(k_shot, new_count),
                )
                outer_rows.append(outer)
                for scene in SCENES:
                    scene_row_id = make_scene_row_id(outer_id, scene)
                    scene_row = Target125SceneRow(
                        scene_row_id,
                        outer_id,
                        receiver,
                        seed,
                        k_shot,
                        new_count,
                        outer.source_pool_k,
                        scene,
                    )
                    scene_rows.append(scene_row)
                    for state in STATES:
                        alias = _alias_target(scene_row_id, state, k_shot)
                        surfaces.append(
                            Target125StateSurface(
                                make_surface_id(scene_row_id, state),
                                scene_row_id,
                                outer_id,
                                receiver,
                                seed,
                                k_shot,
                                new_count,
                                outer.source_pool_k,
                                scene,
                                state,
                                alias is None,
                                alias,
                            )
                        )
    return tuple(outer_rows), tuple(scene_rows), tuple(surfaces)


def validate_next_r5_fa_target125_matrix(plan: Target125MatrixPlan) -> None:
    """Fail closed unless IDs, order, aliases, and cardinalities are exact."""

    if type(plan) is not Target125MatrixPlan:
        raise NextR5FATarget125MatrixError("matrix plan must use the exact Target125 type")
    expected = _enumerate_matrix()
    if (plan.outer_rows, plan.scene_rows, plan.surfaces) != expected:
        raise NextR5FATarget125MatrixError("Target125 matrix row/order/alias coverage drift")
    unique_count = sum(surface.unique_prediction for surface in plan.surfaces)
    alias_count = sum(not surface.unique_prediction for surface in plan.surfaces)
    if (
        len(plan.outer_rows) != OUTER_JOB_COUNT
        or len(plan.scene_rows) != SCENE_ROW_COUNT
        or len(plan.surfaces) != LOGICAL_STATE_SURFACE_COUNT
        or unique_count != UNIQUE_PREDICTION_COUNT
        or alias_count != ALIAS_COUNT
    ):
        raise NextR5FATarget125MatrixError("Target125 count closure drift")
    if plan.matrix_receipt_sha256 != canonical_sha256(_matrix_payload(*expected)):
        raise NextR5FATarget125MatrixError("Target125 matrix receipt drift")


def freeze_next_r5_fa_target125_matrix() -> Target125MatrixPlan:
    """Return the only permitted NEXT-R5 FA-RDCE3 Target125 topology."""

    outer_rows, scene_rows, surfaces = _enumerate_matrix()
    plan = Target125MatrixPlan(
        outer_rows=outer_rows,
        scene_rows=scene_rows,
        surfaces=surfaces,
        matrix_receipt_sha256=canonical_sha256(
            _matrix_payload(outer_rows, scene_rows, surfaces)
        ),
    )
    validate_next_r5_fa_target125_matrix(plan)
    return plan


def audit_logical_surface_coverage(surface_ids: Iterable[str]) -> None:
    """Reject any missing, duplicate, or extra logical state surface."""

    received = tuple(surface_ids)
    expected = tuple(
        surface.surface_id for surface in freeze_next_r5_fa_target125_matrix().surfaces
    )
    if len(received) != len(expected) or len(set(received)) != len(received):
        raise NextR5FATarget125MatrixError("logical surface coverage cardinality drift")
    if received != expected:
        raise NextR5FATarget125MatrixError("logical surface coverage/order drift")


def audit_unique_prediction_coverage(prediction_surface_ids: Iterable[str]) -> None:
    """Reject a prediction manifest that omits an artifact or materializes K1 aliases."""

    received = tuple(prediction_surface_ids)
    expected = tuple(
        surface.surface_id
        for surface in freeze_next_r5_fa_target125_matrix().surfaces
        if surface.unique_prediction
    )
    if len(received) != UNIQUE_PREDICTION_COUNT or len(set(received)) != len(received):
        raise NextR5FATarget125MatrixError("unique prediction coverage cardinality drift")
    if received != expected:
        raise NextR5FATarget125MatrixError("unique prediction coverage/order drift")


def surface_by_id(surface_id: str) -> Target125StateSurface:
    """Resolve one frozen state surface without accepting a partial matrix."""

    text = _text(surface_id, "surface_id")
    for surface in freeze_next_r5_fa_target125_matrix().surfaces:
        if surface.surface_id == text:
            return surface
    raise NextR5FATarget125MatrixError("surface_id is outside frozen Target125 matrix")


__all__ = [
    "ALIAS_COUNT",
    "CANDIDATE_ID",
    "FEATURE_DIM",
    "LOGICAL_STATE_SURFACE_COUNT",
    "METRIC_AVAILABILITY",
    "NextR5FATarget125MatrixError",
    "OLD_CLASS_COUNT",
    "OUTER_JOB_COUNT",
    "PROTOCOL_SCHEMA",
    "RECEIVERS",
    "REG0_STATES",
    "REG1_STATES",
    "SCENE_ROW_COUNT",
    "SCENES",
    "SEEDS",
    "STATES",
    "STATE_NAMES_ZH",
    "TARGET125_SLICES",
    "Target125MatrixPlan",
    "Target125OuterRow",
    "Target125SceneRow",
    "Target125StateSurface",
    "UNIQUE_PREDICTION_COUNT",
    "audit_logical_surface_coverage",
    "audit_unique_prediction_coverage",
    "canonical_bytes",
    "canonical_sha256",
    "freeze_next_r5_fa_target125_matrix",
    "make_outer_id",
    "make_scene_row_id",
    "make_surface_id",
    "source_pool_k_for",
    "surface_by_id",
    "validate_next_r5_fa_target125_matrix",
]
