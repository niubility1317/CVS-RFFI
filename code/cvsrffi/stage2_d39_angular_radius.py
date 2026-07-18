"""D39 support-only angular-radius calibration over the frozen D38-B route.

D39 changes no D38 optimizer, loss, step, or prototype trajectory.  It wraps
the compiled D38 state with one FP16 angular radius per registered class and
uses one label-permutation-equivariant angular Gaussian score for every class.
Old radii and the old-only prior are frozen before new-class registration;
new radii are then computed independently and appended.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from cvsrffi.stage2_d38_strong_b3_quantized import (
    D38_SCORE_TEMPERATURE,
    D38StrongB3Config,
    D38StrongB3State,
    fit_d38_strong_b3_quantized,
    old_prefix_bitwise_unchanged_d38,
    score_d38_strong_b3,
)


SCHEMA_INT8 = "cvs.phase2.d39.angular_radius_int8.v1"
SCHEMA_FP32 = "cvs.phase2.d39.angular_radius_fp32_ablation.v1"
RADIUS_NU = 4.0
RADIUS_EPSILON = 0.001
R0_FLOOR = 0.05


class D39AngularRadiusError(ValueError):
    pass


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class D39AngularRadiusConfig:
    """Locked D39 mechanism; values may not be tuned per class or row."""

    nu: float = RADIUS_NU
    epsilon: float = RADIUS_EPSILON
    r0_floor: float = R0_FLOOR

    def __post_init__(self) -> None:
        if (
            float(self.nu) != RADIUS_NU
            or float(self.epsilon) != RADIUS_EPSILON
            or float(self.r0_floor) != R0_FLOOR
        ):
            raise D39AngularRadiusError("D39 radius mechanism lock drift")


@dataclass(frozen=True)
class D39AngularRadiusState:
    schema: str
    base_state: D38StrongB3State
    radius_fp16: np.ndarray
    r0_fp16: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.base_state.classes)
        is_int8 = self.schema == SCHEMA_INT8
        is_fp32 = self.schema == SCHEMA_FP32
        if (
            not (is_int8 or is_fp32)
            or is_int8 != self.base_state.is_int8
            or self.radius_fp16.dtype != np.float16
            or self.radius_fp16.shape != (count,)
            or self.r0_fp16.dtype != np.float16
            or self.r0_fp16.shape != (1,)
            or not np.isfinite(self.radius_fp16).all()
            or not np.isfinite(self.r0_fp16).all()
            or not bool(np.all(self.radius_fp16 > 0))
            or not bool(np.all(self.r0_fp16 > 0))
            or self.base_state.arm != "B"
        ):
            raise D39AngularRadiusError("D39 state drift")
        if is_int8 and self.base_state.fp32_weights.shape != (0, 288):
            raise D39AngularRadiusError("formal D39 state contains FP32 prototype")
        object.__setattr__(
            self, "radius_fp16", _readonly(self.radius_fp16, np.float16)
        )
        object.__setattr__(self, "r0_fp16", _readonly(self.r0_fp16, np.float16))

    @property
    def is_int8(self) -> bool:
        return self.schema == SCHEMA_INT8

    @property
    def classes(self) -> tuple[str, ...]:
        return self.base_state.classes

    @property
    def old_class_count(self) -> int:
        return int(self.base_state.old_class_count)

    @property
    def radius_state_bytes(self) -> int:
        return int(self.radius_fp16.nbytes + self.r0_fp16.nbytes)

    @property
    def wrapper_metadata_bytes(self) -> int:
        metadata = {
            "epsilon": RADIUS_EPSILON,
            "nu": RADIUS_NU,
            "r0_floor": R0_FLOOR,
            "schema": self.schema,
        }
        return len(
            json.dumps(
                metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )

    @property
    def persistent_state_bytes(self) -> int:
        return int(
            self.base_state.persistent_state_bytes
            + self.radius_state_bytes
            + self.wrapper_metadata_bytes
        )


@dataclass(frozen=True)
class D39AngularRadiusResult:
    before_state: D39AngularRadiusState
    state: D39AngularRadiusState
    matched_fp32_state: D39AngularRadiusState
    training_trace: tuple[dict[str, Any], ...]
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


def _support_labels(
    labels: Sequence[str],
    classes: Sequence[str],
    row_count: int,
    name: str,
) -> tuple[np.ndarray, tuple[str, ...], int]:
    y = np.asarray(tuple(str(value) for value in labels))
    registry = tuple(str(value) for value in classes)
    if (
        len(y) != row_count
        or not registry
        or len(set(registry)) != len(registry)
        or set(y.tolist()) != set(registry)
    ):
        raise D39AngularRadiusError(f"{name} registry drift")
    counts = [int(np.sum(y == label)) for label in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D39AngularRadiusError(f"{name} must be symmetric K-shot")
    return y, registry, counts[0]


def _class_second_moments(
    state: D38StrongB3State,
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    *,
    name: str,
) -> tuple[np.ndarray, int]:
    scores = score_d38_strong_b3(state, features)
    y, registry, k_shot = _support_labels(labels, classes, len(scores), name)
    columns = {label: state.classes.index(label) for label in registry}
    result: list[float] = []
    for label in registry:
        cosine = np.asarray(
            scores[y == label, columns[label]] / np.float32(D38_SCORE_TEMPERATURE),
            dtype=np.float32,
        )
        theta = np.arccos(np.clip(cosine, -1.0, 1.0)).astype(np.float32)
        second_moment = float(np.mean(theta * theta))
        if not math.isfinite(second_moment) or second_moment < 0.0:
            raise D39AngularRadiusError("non-finite angular second moment")
        result.append(second_moment)
    return np.asarray(result, dtype=np.float32), k_shot


def _quantized_r0(old_second_moments: np.ndarray) -> np.ndarray:
    value = max(float(np.sqrt(np.mean(old_second_moments))), R0_FLOOR)
    quantized = np.asarray([value], dtype=np.float16)
    if not np.isfinite(quantized).all() or not bool(np.all(quantized > 0)):
        raise D39AngularRadiusError("D39 r0 FP16 closure drift")
    return quantized


def _shrunken_radius(
    second_moments: np.ndarray, k_shot: int, r0_fp16: np.ndarray
) -> np.ndarray:
    r0 = float(np.asarray(r0_fp16, dtype=np.float16)[0])
    if k_shot == 1:
        # Exact numerical realization of the locked formula at K-1 == 0.
        return np.full(len(second_moments), np.float16(r0), dtype=np.float16)
    numerator = RADIUS_NU * r0 * r0 + (k_shot - 1) * np.asarray(
        second_moments, dtype=np.float64
    )
    radius = np.sqrt(numerator / (RADIUS_NU + k_shot - 1))
    quantized = np.asarray(radius, dtype=np.float16)
    if not np.isfinite(quantized).all() or not bool(np.all(quantized > 0)):
        raise D39AngularRadiusError("D39 radius FP16 closure drift")
    return quantized


def fit_d39_angular_radius(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_classes: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_classes: Sequence[str],
    *,
    seed: int,
    device: torch.device | str = "cpu",
    config: D39AngularRadiusConfig | None = None,
) -> D39AngularRadiusResult:
    locked = config or D39AngularRadiusConfig()
    del locked  # Values are schema-locked; no per-row tuning surface exists.
    old_lifecycle: dict[str, Any] = {"hook_call_count": 0}

    def materialize_old_radius_before_stage2c(
        before_base_state: D38StrongB3State,
        stage2b_trace: tuple[dict[str, Any], ...],
    ) -> None:
        if old_lifecycle["hook_call_count"] != 0:
            raise D39AngularRadiusError("D39 old-radius hook called more than once")
        if (
            len(stage2b_trace) != 20
            or any(
                row.get("phase") != "stage2b_fullbatch_old_adaptation"
                for row in stage2b_trace
            )
            or int(stage2b_trace[-1].get("optimizer_step", -1)) != 20
        ):
            raise D39AngularRadiusError("D39 old-radius lifecycle trace drift")
        old_m2, old_k = _class_second_moments(
            before_base_state,
            old_support_features,
            old_support_labels,
            old_classes,
            name="old radius support",
        )
        r0_fp16 = _quantized_r0(old_m2)
        old_radius = _shrunken_radius(old_m2, old_k, r0_fp16)
        old_lifecycle.update(
            {
                "hook_call_count": 1,
                "old_k": old_k,
                "old_m2": old_m2,
                "r0_fp16": r0_fp16,
                "old_radius": old_radius,
                "stage2b_trace_length": len(stage2b_trace),
                "last_optimizer_step": int(stage2b_trace[-1]["optimizer_step"]),
            }
        )

    base = fit_d38_strong_b3_quantized(
        old_support_features,
        old_support_labels,
        old_classes,
        new_support_features,
        new_support_labels,
        new_classes,
        seed=int(seed),
        device=device,
        config=D38StrongB3Config(arm="B"),
        before_stage2c_hook=materialize_old_radius_before_stage2c,
    )
    if old_lifecycle.get("hook_call_count") != 1:
        raise D39AngularRadiusError("D39 old-radius hook did not close")
    old_k = int(old_lifecycle["old_k"])
    r0_fp16 = np.asarray(old_lifecycle["r0_fp16"], dtype=np.float16)
    old_radius = np.asarray(old_lifecycle["old_radius"], dtype=np.float16)
    new_m2, new_k = _class_second_moments(
        base.state,
        new_support_features,
        new_support_labels,
        new_classes,
        name="new radius support",
    )
    if old_k != new_k:
        raise D39AngularRadiusError("D39 old/new K closure drift")
    new_radius = _shrunken_radius(new_m2, new_k, r0_fp16)
    all_radius = np.concatenate([old_radius, new_radius]).astype(np.float16)
    before_state = D39AngularRadiusState(
        schema=SCHEMA_INT8,
        base_state=base.before_state,
        radius_fp16=old_radius,
        r0_fp16=r0_fp16,
    )
    state = D39AngularRadiusState(
        schema=SCHEMA_INT8,
        base_state=base.state,
        radius_fp16=all_radius,
        r0_fp16=r0_fp16,
    )
    matched_fp32 = D39AngularRadiusState(
        schema=SCHEMA_FP32,
        base_state=base.matched_fp32_state,
        radius_fp16=all_radius,
        r0_fp16=r0_fp16,
    )
    prefix_unchanged = old_prefix_bitwise_unchanged_d39(before_state, state)
    if not prefix_unchanged:
        raise D39AngularRadiusError("D39 old base/radius prefix changed")
    if not np.array_equal(state.radius_fp16, matched_fp32.radius_fp16):
        raise D39AngularRadiusError("D39 int8/FP32 radius identity drift")

    old_rows = len(np.asarray(old_support_labels))
    new_rows = len(np.asarray(new_support_labels))
    class_count = len(state.classes)
    geometry = {
        **dict(base.geometry_audit),
        "schema": "cvs.phase2.d39.angular_radius_geometry.v1",
        "base_route": "D38-B_training_trajectory_unchanged",
        "radius_formula": "(nu*r0^2+(K-1)*m2)/(nu+K-1)",
        "radius_nu": RADIUS_NU,
        "radius_epsilon": RADIUS_EPSILON,
        "radius_r0_floor": R0_FLOOR,
        "angular_gaussian_score": True,
        "all_registered_classes_same_score_formula": True,
        "label_permutation_equivariant": True,
        "r0_old_support_only": True,
        "old_radius_materialized_before_stage2c": True,
        "old_radius_materialization_hook_call_count": int(
            old_lifecycle["hook_call_count"]
        ),
        "old_radius_materialization_stage2b_trace_length": int(
            old_lifecycle["stage2b_trace_length"]
        ),
        "old_radius_materialization_last_optimizer_step": int(
            old_lifecycle["last_optimizer_step"]
        ),
        "old_radius_source_state": "registration_preceding_int8_before_state",
        "new_radius_source_state": "final_int8_append_state",
        "old_radius_new_support_row_count": 0,
        "new_radius_old_support_row_count": 0,
        "held_radius_fit_row_count": 0,
        "query_rows_used": 0,
        "old_base_prefix_bitwise_unchanged": old_prefix_bitwise_unchanged_d38(
            before_state.base_state, state.base_state
        ),
        "old_radius_prefix_bitwise_unchanged": bool(
            np.array_equal(
                before_state.radius_fp16,
                state.radius_fp16[: before_state.old_class_count],
            )
        ),
        "r0_bitwise_unchanged": bool(
            np.array_equal(before_state.r0_fp16, state.r0_fp16)
        ),
        "k1_radius_equals_r0": bool(
            old_k != 1
            or (
                np.array_equal(
                    before_state.radius_fp16,
                    np.full_like(before_state.radius_fp16, before_state.r0_fp16[0]),
                )
                and np.array_equal(
                    new_radius, np.full_like(new_radius, before_state.r0_fp16[0])
                )
            )
        ),
        "matched_fp32_reuses_exact_fp16_radius": True,
        "fp32_target_prototype_stored_in_formal_state": False,
    }
    resource = dict(base.resource_audit)
    radius_fit_acos = old_rows + new_rows
    resource.update(
        {
            "schema": "cvs.phase2.d39.angular_radius_resource.v1",
            "base_persistent_state_bytes": base.state.persistent_state_bytes,
            "radius_state_bytes": state.radius_state_bytes,
            "wrapper_metadata_bytes": state.wrapper_metadata_bytes,
            "persistent_state_bytes": state.persistent_state_bytes,
            "persistent_state_cap_pass": state.persistent_state_bytes <= 256 * 1024,
            "resident_fp32_target_prototype_count": 0,
            "formal_state_int8_only": True,
            "radius_storage_dtype": "float16",
            "r0_storage_dtype": "float16",
            "radius_fit_trainable_parameters": 0,
            "radius_fit_optimizer_steps": 0,
            "radius_fit_epochs": 0,
            "radius_fit_acos_scalar_operations": radius_fit_acos,
            "radius_fit_scalar_operation_count": int(6 * radius_fit_acos + 8 * class_count),
            "per_query_acos_scalar_operations": class_count,
            "per_query_log_scalar_operations": class_count,
            "angular_radius_scalar_operations_per_query": int(8 * class_count),
            "base_estimated_macs_per_query": int(resource["estimated_macs_per_query"]),
            "old_radius_support_row_count": old_rows,
            "new_radius_support_row_count": new_rows,
            "old_radius_materialized_before_stage2c": True,
            "old_radius_materialization_hook_call_count": int(
                old_lifecycle["hook_call_count"]
            ),
            "old_radius_materialization_stage2b_trace_length": int(
                old_lifecycle["stage2b_trace_length"]
            ),
            "old_radius_new_support_row_count": 0,
            "new_radius_old_support_row_count": 0,
            "held_radius_fit_row_count": 0,
            "query_rows_used_for_fit": 0,
            "query_labels_used_for_fit": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "dense_query_graph_bytes": 0,
            "class_id_specific_branch": False,
            "radius_label_permutation_equivariant": True,
            "registered_class_count": class_count,
            "old_k_shot": old_k,
            "new_k_shot": new_k,
            "r0_fp16": float(state.r0_fp16[0]),
            "radius_min_fp16": float(np.min(state.radius_fp16)),
            "radius_max_fp16": float(np.max(state.radius_fp16)),
        }
    )
    return D39AngularRadiusResult(
        before_state=before_state,
        state=state,
        matched_fp32_state=matched_fp32,
        training_trace=base.training_trace,
        geometry_audit=geometry,
        resource_audit=resource,
    )


def score_d39_angular_radius(
    state: D39AngularRadiusState, features: np.ndarray
) -> np.ndarray:
    raw = score_d38_strong_b3(state.base_state, features)
    cosine = np.asarray(raw / np.float32(D38_SCORE_TEMPERATURE), dtype=np.float32)
    theta = np.arccos(np.clip(cosine, -1.0, 1.0)).astype(np.float32)
    radius = state.radius_fp16.astype(np.float32) + np.float32(RADIUS_EPSILON)
    scores = np.asarray(
        -0.5 * (theta / radius[None, :]) ** 2 - np.log(radius[None, :]),
        dtype=np.float32,
    )
    if not np.isfinite(scores).all():
        raise D39AngularRadiusError("non-finite D39 angular Gaussian score")
    return _readonly(scores, np.float32)


def predict_d39_angular_radius(
    state: D39AngularRadiusState, features: np.ndarray
) -> np.ndarray:
    return np.asarray(state.classes)[
        np.argmax(score_d39_angular_radius(state, features), axis=1)
    ]


def old_prefix_bitwise_unchanged_d39(
    before: D39AngularRadiusState, after: D39AngularRadiusState
) -> bool:
    count = before.old_class_count
    return bool(
        before.is_int8
        and after.is_int8
        and old_prefix_bitwise_unchanged_d38(before.base_state, after.base_state)
        and np.array_equal(before.radius_fp16, after.radius_fp16[:count])
        and np.array_equal(before.r0_fp16, after.r0_fp16)
    )


def pairwise_support_diagnostics_d39(
    state: D39AngularRadiusState,
    held_new_features: np.ndarray,
    held_new_labels: Sequence[str],
    physical_ids: Sequence[str],
    *,
    scenario: str,
    outer_fold: int,
    physical_ranks: Sequence[int],
) -> list[dict[str, Any]]:
    labels = tuple(str(value) for value in held_new_labels)
    ids = tuple(str(value) for value in physical_ids)
    ranks = tuple(int(value) for value in physical_ranks)
    scores = score_d39_angular_radius(state, held_new_features)
    old_count = state.old_class_count
    if (
        len(state.classes) == old_count
        or len(labels) != len(scores)
        or len(ids) != len(scores)
        or len(ranks) != len(scores)
        or len(set(ids)) != len(ids)
        or any(label not in state.classes[old_count:] for label in labels)
        or not str(scenario)
    ):
        raise D39AngularRadiusError("D39 pairwise diagnostic closure drift")
    output: list[dict[str, Any]] = []
    for row, truth, physical_id, rank in zip(scores, labels, ids, ranks, strict=True):
        truth_index = state.classes.index(truth)
        competing_new = np.array(row[old_count:], copy=True)
        competing_new[truth_index - old_count] = -np.inf
        competitor_index = old_count + int(np.argmax(competing_new))
        top_old_index = int(np.argmax(row[:old_count]))
        output.append(
            {
                "scenario": str(scenario),
                "outer_fold": int(outer_fold),
                "physical_rank": rank,
                "physical_sample_id": physical_id,
                "true_new_handle": truth,
                "top_competing_new_handle": state.classes[competitor_index],
                "true_new_score": float(row[truth_index]),
                "top_competing_new_score": float(row[competitor_index]),
                "new_new_margin": float(row[truth_index] - row[competitor_index]),
                "top_old_handle": state.classes[top_old_index],
                "top_old_score": float(row[top_old_index]),
                "new_old_margin": float(row[truth_index] - row[top_old_index]),
                "query_rows_used": 0,
            }
        )
    return output


__all__ = [
    "D39AngularRadiusConfig",
    "D39AngularRadiusError",
    "D39AngularRadiusResult",
    "D39AngularRadiusState",
    "pairwise_support_diagnostics_d39",
    "fit_d39_angular_radius",
    "old_prefix_bitwise_unchanged_d39",
    "predict_d39_angular_radius",
    "score_d39_angular_radius",
]
