"""D40 synchronous hard-negative barycentric residualization (HNBR).

D40 reuses D38's twenty-step Stage2-B metric, then applies one fixed,
closed-form spherical projection.  Old directions are residualized
synchronously and compiled before registration.  New directions are likewise
residualized synchronously against the frozen old final references and all
other new base directions, then appended without rewriting the old prefix.

The module accepts support rows only.  It exposes no query truth, role, quota,
batch assignment, clean-data, or source-data fitting surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from cvsrffi.stage2_d38_strong_b3_quantized import (
    D38_SCORE_TEMPERATURE,
    D38StrongB3Config,
    D38StrongB3QuantizedError,
    D38StrongB3State,
    append_d38_state,
    compile_d38_state,
    decode_d38_state_weights,
    fit_d38_strong_b3_quantized,
    old_prefix_bitwise_unchanged_d38,
    score_d38_strong_b3,
    transform_d38_features,
)


FEATURE_DIM = 288
HNBR_TEMPERATURE = D38_SCORE_TEMPERATURE
NORM_EPSILON = 1.0e-12
ALLOWED_NEW_CLASS_COUNTS = (2, 5, 10, 20)
SCHEMA_INT8 = "cvs.phase2.d40.hnbr_residual_int8.v1"
SCHEMA_FP32 = "cvs.phase2.d40.hnbr_fp32_ablation.v1"


class D40HNBRError(ValueError):
    pass


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _unit_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D40HNBRError(f"{name} must be finite float32 [N,{FEATURE_DIM}]")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or bool(np.any(norms <= NORM_EPSILON)):
        raise D40HNBRError(f"{name} contains a near-zero direction")
    return np.asarray(rows / norms, dtype=np.float32)


def _support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = np.asarray(features)
    registry = tuple(str(value) for value in classes)
    y = np.asarray(tuple(str(value) for value in labels))
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
        or len(y) != len(rows)
        or not registry
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
        or set(y.tolist()) != set(registry)
    ):
        raise D40HNBRError(f"{name} support contract drift")
    counts = [int(np.sum(y == label)) for label in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D40HNBRError(f"{name} support must be symmetric K-shot")
    targets = np.asarray([registry.index(value) for value in y], dtype=np.int64)
    return np.ascontiguousarray(rows), targets, registry, counts[0]


def _class_centroids(
    transformed_rows: np.ndarray,
    targets: np.ndarray,
    class_count: int,
) -> np.ndarray:
    centroids = np.stack(
        [
            np.asarray(
                transformed_rows[targets == index].mean(axis=0), dtype=np.float32
            )
            for index in range(class_count)
        ]
    ).astype(np.float32)
    return _unit_rows(centroids, "D40 class centroid")


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _int8_prefix_sha256(state: D38StrongB3State) -> str:
    digest = hashlib.sha256()
    for value in (
        state.log_diag_fp32,
        state.code1_qint8,
        state.code2_qint8,
        state.scale1_fp16,
        state.scale2_fp16,
        state.inverse_norm_fp16,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class D40HNBRConfig:
    """Schema-locked HNBR configuration with no tunable coefficient."""

    temperature: float = HNBR_TEMPERATURE
    norm_epsilon: float = NORM_EPSILON

    def __post_init__(self) -> None:
        if (
            float(self.temperature) != HNBR_TEMPERATURE
            or float(self.norm_epsilon) != NORM_EPSILON
        ):
            raise D40HNBRError("D40 HNBR mechanism lock drift")


@dataclass(frozen=True)
class D40HNBRState:
    schema: str
    base_state: D38StrongB3State

    def __post_init__(self) -> None:
        is_int8 = self.schema == SCHEMA_INT8
        is_fp32 = self.schema == SCHEMA_FP32
        if (
            not (is_int8 or is_fp32)
            or is_int8 != self.base_state.is_int8
            or self.base_state.arm != "A"
        ):
            raise D40HNBRError("D40 state drift")
        if is_int8 and self.base_state.fp32_weights.shape != (0, FEATURE_DIM):
            raise D40HNBRError("formal D40 state contains FP32 target direction")

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
    def wrapper_metadata_bytes(self) -> int:
        metadata = {
            "norm_epsilon": NORM_EPSILON,
            "schema": self.schema,
            "temperature": HNBR_TEMPERATURE,
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
            self.base_state.persistent_state_bytes + self.wrapper_metadata_bytes
        )


@dataclass(frozen=True)
class D40HNBRResult:
    before_state: D40HNBRState
    state: D40HNBRState
    matched_fp32_before_state: D40HNBRState
    matched_fp32_state: D40HNBRState
    training_trace: tuple[dict[str, Any], ...]
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


def hnbr_residualize_directions(
    base_directions: np.ndarray,
    *,
    frozen_negative_directions: np.ndarray | None = None,
) -> np.ndarray:
    """Synchronously remove each direction's positive hard-negative projection."""

    base = _unit_rows(base_directions, "D40 HNBR base")
    frozen = None
    if frozen_negative_directions is not None:
        frozen = _unit_rows(
            frozen_negative_directions, "D40 HNBR frozen negative"
        )
    if frozen is None and len(base) < 2:
        raise D40HNBRError("D40 HNBR requires at least two directions")
    output: list[np.ndarray] = []
    for class_index, direction in enumerate(base):
        other = np.delete(base, class_index, axis=0)
        negatives = other if frozen is None else np.concatenate([frozen, other], axis=0)
        if len(negatives) < 1:
            raise D40HNBRError("D40 HNBR negative set is empty")
        logits = np.asarray(
            np.float32(HNBR_TEMPERATURE) * (negatives @ direction),
            dtype=np.float32,
        )
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted).astype(np.float32)
        denominator = float(np.sum(exponentials, dtype=np.float64))
        if not math.isfinite(denominator) or denominator <= NORM_EPSILON:
            raise D40HNBRError("D40 HNBR softmax normalization drift")
        attention = np.asarray(exponentials / np.float32(denominator), dtype=np.float32)
        negative_raw = np.asarray(attention @ negatives, dtype=np.float32)
        negative = _unit_rows(
            negative_raw[None, :], "D40 HNBR negative centroid"
        )[0]
        projection = np.float32(max(0.0, float(direction @ negative)))
        residual = np.asarray(direction - projection * negative, dtype=np.float32)
        output.append(
            _unit_rows(residual[None, :], "D40 HNBR residual direction")[0]
        )
    return _readonly(np.stack(output).astype(np.float32), np.float32)


def fit_d40_hnbr(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_classes: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_classes: Sequence[str],
    *,
    seed: int,
    device: torch.device | str = "cpu",
    config: D40HNBRConfig | None = None,
) -> D40HNBRResult:
    locked = config or D40HNBRConfig()
    del locked
    old_rows, _, old_registry, old_k = _support(
        old_support_features, old_support_labels, old_classes, "old"
    )
    new_rows, new_targets, new_registry, new_k = _support(
        new_support_features, new_support_labels, new_classes, "new"
    )
    if (
        not 2 <= len(old_registry) <= 20
        or len(new_registry) not in ALLOWED_NEW_CLASS_COUNTS
        or set(old_registry) & set(new_registry)
        or old_k != new_k
    ):
        raise D40HNBRError("D40 class/K closure drift")

    try:
        d38 = fit_d38_strong_b3_quantized(
            old_rows,
            old_support_labels,
            old_registry,
            new_rows,
            new_support_labels,
            new_registry,
            seed=int(seed),
            device=device,
            config=D38StrongB3Config(arm="A"),
        )
    except D38StrongB3QuantizedError as exc:
        raise D40HNBRError(f"D40 D38 Stage2-B closure failed: {exc}") from exc
    if (
        len(d38.training_trace) != 20
        or any(
            row.get("phase") != "stage2b_fullbatch_old_adaptation"
            for row in d38.training_trace
        )
    ):
        raise D40HNBRError("D40 Stage2-B trace drift")

    d38_fp32_directions = decode_d38_state_weights(d38.matched_fp32_state)
    old_base = np.asarray(
        d38_fp32_directions[: len(old_registry)], dtype=np.float32
    )
    old_final_reference = hnbr_residualize_directions(old_base)
    log_diag = np.asarray(d38.before_state.log_diag_fp32, dtype=np.float32)
    before_int8_base = compile_d38_state(
        old_registry,
        len(old_registry),
        log_diag,
        old_final_reference,
        arm="A",
        precision="int8",
    )
    before_fp32_base = compile_d38_state(
        old_registry,
        len(old_registry),
        log_diag,
        old_final_reference,
        arm="A",
        precision="fp32",
    )

    # Stage2-C must consume the actual frozen old deployment identity.  The
    # same new FP32 reference produced from these decoded int8 directions is
    # then compiled for both the formal and matched-FP32 deployment states.
    old_final_deployed = decode_d38_state_weights(before_int8_base)

    transformed_new = transform_d38_features(before_fp32_base, new_rows)
    new_base = _class_centroids(
        transformed_new, new_targets, len(new_registry)
    )
    new_final_reference = hnbr_residualize_directions(
        new_base,
        frozen_negative_directions=old_final_deployed,
    )
    final_int8_base = append_d38_state(
        before_int8_base, new_registry, new_final_reference
    )
    final_fp32_base = append_d38_state(
        before_fp32_base, new_registry, new_final_reference
    )

    before_state = D40HNBRState(SCHEMA_INT8, before_int8_base)
    state = D40HNBRState(SCHEMA_INT8, final_int8_base)
    matched_before = D40HNBRState(SCHEMA_FP32, before_fp32_base)
    matched_state = D40HNBRState(SCHEMA_FP32, final_fp32_base)
    if not old_prefix_bitwise_unchanged_d40(before_state, state):
        raise D40HNBRError("D40 old prefix changed during append")

    support_rows = np.concatenate([old_rows, new_rows], axis=0).astype(np.float32)
    int8_scores = score_d40_hnbr(state, support_rows)
    fp32_scores = score_d40_hnbr(matched_state, support_rows)
    support_argmax_changes = int(
        np.sum(np.argmax(int8_scores, axis=1) != np.argmax(fp32_scores, axis=1))
    )
    before_old_argmax_changes = int(
        np.sum(
            np.argmax(score_d40_hnbr(before_state, old_rows), axis=1)
            != np.argmax(score_d40_hnbr(matched_before, old_rows), axis=1)
        )
    )
    formal_directions = decode_d38_state_weights(state.base_state)
    reference_directions = decode_d38_state_weights(matched_state.base_state)
    quantization_error = np.abs(formal_directions - reference_directions)

    old_count = len(old_registry)
    new_count = len(new_registry)
    old_pair_count = old_count * (old_count - 1)
    new_negative_count = new_count * (old_count + new_count - 1)
    hnbr_direction_macs = int(
        2 * FEATURE_DIM * (old_pair_count + new_negative_count)
        + 5 * FEATURE_DIM * (old_count + new_count)
    )
    new_transform_macs = int(FEATURE_DIM * len(new_rows))
    centroid_macs = int(FEATURE_DIM * len(new_rows))
    hnbr_support_macs = hnbr_direction_macs + new_transform_macs + centroid_macs
    peak_parameters = int(d38.resource_audit["trainable_parameters"])
    base_adaptation_macs = int(d38.resource_audit["estimated_adaptation_macs"])
    class_count = old_count + new_count

    geometry = {
        "schema": "cvs.phase2.d40.hnbr_geometry.v1",
        "feature_geometry": d38.geometry_audit["feature_geometry"],
        "base_route": "D38_arm_A_stage2b20_metric_only",
        "stage2b_solver": "fullbatch_adamw20_then_synchronous_old_hnbr",
        "stage2c_solver": "zero_step_synchronous_new_hnbr",
        "hnbr_temperature": HNBR_TEMPERATURE,
        "hnbr_norm_epsilon": NORM_EPSILON,
        "stable_softmax_subtracts_row_max": True,
        "positive_projection_only": True,
        "old_hnbr_synchronous": True,
        "new_hnbr_synchronous": True,
        "old_hnbr_negative_set": "all_other_old_base_directions",
        "new_hnbr_negative_set": (
            "all_old_final_decoded_int8_directions_plus_all_other_new_base_directions"
        ),
        "new_hnbr_old_negative_precision": "int8_decoded",
        "new_hnbr_old_negative_source": "before_int8_base_frozen_old_prefix",
        "new_hnbr_old_negative_source_sha256": _array_sha256(
            old_final_deployed
        ),
        "before_int8_decoded_old_direction_sha256": _array_sha256(
            decode_d38_state_weights(before_int8_base)
        ),
        "before_int8_old_prefix_sha256": _int8_prefix_sha256(before_int8_base),
        "new_hnbr_old_negative_matches_before_int8_decode": bool(
            np.array_equal(
                old_final_deployed,
                decode_d38_state_weights(before_int8_base),
            )
        ),
        "old_fp32_reference_used_as_new_hnbr_negative": False,
        "new_hnbr_uses_residualized_new_direction_as_negative": False,
        "old_compiled_before_stage2c": True,
        "old_prefix_bitwise_unchanged": True,
        "matched_fp32_shared_reference_directions": True,
        "target_old_int8_used_for_prediction": True,
        "target_new_int8_used_for_prediction": True,
        "fp32_target_direction_stored_in_formal_state": False,
        "class_id_specific_branch": False,
        "label_permutation_equivariant": True,
        "query_rows_used": 0,
        "quantization_error_mean": float(np.mean(quantization_error)),
        "quantization_error_max": float(np.max(quantization_error)),
        "int8_vs_fp32_support_argmax_change_count": support_argmax_changes,
        "int8_vs_fp32_before_old_argmax_change_count": before_old_argmax_changes,
    }
    resource = {
        "schema": "cvs.phase2.d40.hnbr_resource.v1",
        "support_only": True,
        "trainable_parameters": peak_parameters,
        "trainable_parameter_cap": 2016,
        "trainable_parameter_cap_pass": peak_parameters <= 2016,
        "adaptation_epochs": 20,
        "optimizer_steps": 20,
        "stage2c_optimizer_steps": 0,
        "adaptation_epoch_cap": 20,
        "optimizer_step_cap": 20,
        "adaptation_epoch_cap_pass": True,
        "optimizer_step_cap_pass": True,
        "persistent_state_bytes": state.persistent_state_bytes,
        "base_persistent_state_bytes": state.base_state.persistent_state_bytes,
        "wrapper_metadata_bytes": state.wrapper_metadata_bytes,
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_cap_pass": state.persistent_state_bytes <= 256 * 1024,
        "estimated_d38_stage2b_macs": base_adaptation_macs,
        "estimated_hnbr_support_macs": hnbr_support_macs,
        "estimated_adaptation_macs": base_adaptation_macs + hnbr_support_macs,
        "estimated_macs_per_query": int(
            FEATURE_DIM + 2 * FEATURE_DIM * class_count
        ),
        "dense_query_graph_bytes": 0,
        "query_dependent_batch_optimization": False,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
        "resident_fp32_target_prototype_count": 0,
        "formal_state_int8_only": True,
        "old_k_shot": old_k,
        "new_k_shot": new_k,
        "registered_class_count": class_count,
        "runtime_device": str(device),
        "peak_cuda_memory_bytes": int(
            d38.resource_audit["peak_cuda_memory_bytes"]
        ),
        "int8_vs_fp32_support_argmax_change_count": support_argmax_changes,
    }
    return D40HNBRResult(
        before_state=before_state,
        state=state,
        matched_fp32_before_state=matched_before,
        matched_fp32_state=matched_state,
        training_trace=tuple(dict(row) for row in d38.training_trace),
        geometry_audit=geometry,
        resource_audit=resource,
    )


def score_d40_hnbr(state: D40HNBRState, features: np.ndarray) -> np.ndarray:
    if not isinstance(state, D40HNBRState):
        raise D40HNBRError("D40 score state drift")
    return score_d38_strong_b3(state.base_state, features)


def predict_d40_hnbr(state: D40HNBRState, features: np.ndarray) -> np.ndarray:
    return np.asarray(state.classes)[np.argmax(score_d40_hnbr(state, features), axis=1)]


def old_prefix_bitwise_unchanged_d40(
    before: D40HNBRState, after: D40HNBRState
) -> bool:
    return bool(
        before.is_int8
        and after.is_int8
        and old_prefix_bitwise_unchanged_d38(before.base_state, after.base_state)
    )


def pairwise_support_diagnostics_d40(
    state: D40HNBRState,
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
    scores = score_d40_hnbr(state, held_new_features)
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
        raise D40HNBRError("D40 pairwise diagnostic closure drift")
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
    "D40HNBRConfig",
    "D40HNBRError",
    "D40HNBRResult",
    "D40HNBRState",
    "HNBR_TEMPERATURE",
    "fit_d40_hnbr",
    "hnbr_residualize_directions",
    "old_prefix_bitwise_unchanged_d40",
    "pairwise_support_diagnostics_d40",
    "predict_d40_hnbr",
    "score_d40_hnbr",
]
