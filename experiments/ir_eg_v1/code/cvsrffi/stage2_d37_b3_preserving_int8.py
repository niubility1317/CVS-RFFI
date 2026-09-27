"""D37 B3-preserving residual-int8 registration head.

Only admitted LEO_weak support may construct or calibrate this state.  The
target-old rows are compiled directly from the final B3 weights and remain an
append-only byte prefix when target-new rows are registered.  A single shared
new-class offset is admitted only when support-OOF scores expose a non-empty
old-safety/new-reachability interval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from cvsrffi.stage2_b3_fisher_closed_form import (
    B3FisherClosedFormState,
    FEATURE_DIM,
    TEMPERATURE,
)


SCHEMA = "cvs.phase2.d37_b3_preserving_residual_int8.v1"
ALLOWED_NEW_CLASS_COUNTS = (2, 5, 10, 20)
BLOCK_SLICES = (slice(0, 160), slice(160, 256), slice(256, 288))
ARM_MARGINS = {"A": 0.0, "B": 0.05, "C": 0.10}
NEW_STRICT_EPSILON = 1.0e-4
OOF_SOURCE = "support_physical_rank_pair_crossfit"


class D37B3PreservingInt8Error(ValueError):
    pass


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D37B3PreservingInt8Error(
            f"{name} must be finite float32 [N,{FEATURE_DIM}]"
        )
    return np.ascontiguousarray(rows)


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12) or not np.isfinite(norms).all():
        raise D37B3PreservingInt8Error("zero or non-finite feature norm")
    return np.asarray(values / norms, dtype=np.float32)


def _support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _rows(features, f"{name} features")
    y = np.asarray(tuple(str(value) for value in labels))
    registry = tuple(str(value) for value in classes)
    if (
        len(y) != len(rows)
        or not registry
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
        or set(y.tolist()) != set(registry)
    ):
        raise D37B3PreservingInt8Error(f"{name} registry drift")
    counts = [int(np.sum(y == value)) for value in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D37B3PreservingInt8Error(f"{name} must be symmetric K-shot")
    targets = np.asarray([registry.index(str(value)) for value in y], dtype=np.int64)
    return rows, targets, registry, counts[0]


def _transform(rows: np.ndarray, log_diag: np.ndarray) -> np.ndarray:
    return _normalize(
        np.asarray(rows, dtype=np.float32)
        * np.exp(np.asarray(log_diag, dtype=np.float32), dtype=np.float32)[None, :]
    )


def _positive_fp16(value: float) -> np.float16:
    smallest = np.nextafter(np.float16(0), np.float16(1))
    return np.float16(max(float(value), float(smallest)))


def _residual_quantize(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = _normalize(np.asarray(weights, dtype=np.float32))
    count = len(rows)
    code1 = np.zeros((count, FEATURE_DIM), dtype=np.int8)
    code2 = np.zeros((count, FEATURE_DIM), dtype=np.int8)
    scale1 = np.empty((count, len(BLOCK_SLICES)), dtype=np.float16)
    scale2 = np.empty((count, len(BLOCK_SLICES)), dtype=np.float16)
    for row_index, row in enumerate(rows):
        for block_index, block in enumerate(BLOCK_SLICES):
            values = row[block]
            s1 = _positive_fp16(np.max(np.abs(values)) / 127.0)
            q1 = np.clip(
                np.rint(values / np.float32(s1)), -127, 127
            ).astype(np.int8)
            residual = values - np.float32(s1) * q1.astype(np.float32)
            s2 = _positive_fp16(np.max(np.abs(residual)) / 127.0)
            q2 = np.clip(
                np.rint(residual / np.float32(s2)), -127, 127
            ).astype(np.int8)
            code1[row_index, block] = q1
            code2[row_index, block] = q2
            scale1[row_index, block_index] = s1
            scale2[row_index, block_index] = s2
    decoded = _decode(code1, code2, scale1, scale2)
    return code1, code2, scale1, scale2, decoded


def _decode(
    code1: np.ndarray,
    code2: np.ndarray,
    scale1: np.ndarray,
    scale2: np.ndarray,
) -> np.ndarray:
    count = len(code1)
    decoded = np.empty((count, FEATURE_DIM), dtype=np.float32)
    for block_index, block in enumerate(BLOCK_SLICES):
        decoded[:, block] = (
            code1[:, block].astype(np.float32)
            * scale1[:, block_index].astype(np.float32)[:, None]
            + code2[:, block].astype(np.float32)
            * scale2[:, block_index].astype(np.float32)[:, None]
        )
    return decoded


@dataclass(frozen=True)
class D37B3PreservingInt8Config:
    arm: str = "A"

    def __post_init__(self) -> None:
        arm = str(self.arm).upper()
        if arm.startswith("D37-"):
            arm = arm[4:]
        if arm not in ARM_MARGINS:
            raise D37B3PreservingInt8Error("D37 arm lock drift")
        object.__setattr__(self, "arm", arm)

    @property
    def margin(self) -> float:
        return ARM_MARGINS[self.arm]


@dataclass(frozen=True)
class D37B3PreservingInt8State:
    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    log_diag_fp32: np.ndarray
    code1_qint8: np.ndarray
    code2_qint8: np.ndarray
    scale1_fp16: np.ndarray
    scale2_fp16: np.ndarray
    new_offset_fp16: np.ndarray
    margin_fp16: np.ndarray
    arm: str

    def __post_init__(self) -> None:
        count = len(self.classes)
        new_count = count - int(self.old_class_count)
        if (
            self.schema != SCHEMA
            or not 2 <= int(self.old_class_count) <= 20
            or new_count not in (0,) + ALLOWED_NEW_CLASS_COUNTS
            or len(set(self.classes)) != count
            or any(not str(value) for value in self.classes)
            or self.arm not in ARM_MARGINS
            or self.log_diag_fp32.dtype != np.float32
            or self.log_diag_fp32.shape != (FEATURE_DIM,)
            or self.code1_qint8.dtype != np.int8
            or self.code1_qint8.shape != (count, FEATURE_DIM)
            or self.code2_qint8.dtype != np.int8
            or self.code2_qint8.shape != (count, FEATURE_DIM)
            or self.scale1_fp16.dtype != np.float16
            or self.scale1_fp16.shape != (count, len(BLOCK_SLICES))
            or self.scale2_fp16.dtype != np.float16
            or self.scale2_fp16.shape != (count, len(BLOCK_SLICES))
            or self.new_offset_fp16.dtype != np.float16
            or self.new_offset_fp16.shape not in ((0,), (1,))
            or self.margin_fp16.dtype != np.float16
            or self.margin_fp16.shape not in ((0,), (1,))
            or self.new_offset_fp16.shape != self.margin_fp16.shape
            or new_count == 0
            and self.new_offset_fp16.shape != (0,)
            or not np.isfinite(self.log_diag_fp32).all()
            or not np.isfinite(self.scale1_fp16).all()
            or not np.isfinite(self.scale2_fp16).all()
            or not np.isfinite(self.new_offset_fp16).all()
            or not np.isfinite(self.margin_fp16).all()
            or bool(np.any(self.scale1_fp16 <= 0))
            or bool(np.any(self.scale2_fp16 <= 0))
        ):
            raise D37B3PreservingInt8Error("D37 state drift")
        for field, dtype in (
            ("log_diag_fp32", np.float32),
            ("code1_qint8", np.int8),
            ("code2_qint8", np.int8),
            ("scale1_fp16", np.float16),
            ("scale2_fp16", np.float16),
            ("new_offset_fp16", np.float16),
            ("margin_fp16", np.float16),
        ):
            object.__setattr__(self, field, _readonly(getattr(self, field), dtype))

    @property
    def persistent_state_bytes(self) -> int:
        return int(
            self.log_diag_fp32.nbytes
            + self.code1_qint8.nbytes
            + self.code2_qint8.nbytes
            + self.scale1_fp16.nbytes
            + self.scale2_fp16.nbytes
            + self.new_offset_fp16.nbytes
            + self.margin_fp16.nbytes
        )

    @property
    def calibration_ready(self) -> bool:
        return self.new_offset_fp16.shape == (1,)


@dataclass(frozen=True)
class D37B3PreservingInt8Result:
    before_state: D37B3PreservingInt8State
    state_no_offset: D37B3PreservingInt8State
    training_trace: tuple[dict[str, Any], ...]
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


@dataclass(frozen=True)
class D37FeasibleOffsetResult:
    state: D37B3PreservingInt8State
    lower_bound: float
    upper_bound: float
    offset: float
    old_oof_count: int
    new_oof_count: int
    fold_count: int
    physical_id_sha256: str
    source: str


def fit_d37_b3_preserving_int8(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_classes: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_classes: Sequence[str],
    b3_state: B3FisherClosedFormState,
    *,
    config: D37B3PreservingInt8Config | None = None,
) -> D37B3PreservingInt8Result:
    """Compile B3 old weights and append same-geometry new int8 weights."""

    locked = config or D37B3PreservingInt8Config()
    old_rows, _, old_registry, old_k = _support(
        old_support_features, old_support_labels, old_classes, "old support"
    )
    new_rows, new_targets, new_registry, new_k = _support(
        new_support_features, new_support_labels, new_classes, "new support"
    )
    if (
        old_registry != tuple(b3_state.classes)
        or len(new_registry) not in ALLOWED_NEW_CLASS_COUNTS
        or set(old_registry) & set(new_registry)
    ):
        raise D37B3PreservingInt8Error("B3/registration class closure drift")

    transformed_new = _transform(new_rows, b3_state.log_diag)
    new_weights = np.stack(
        [
            _normalize(
                np.mean(transformed_new[new_targets == index], axis=0, keepdims=True)
            )[0]
            for index in range(len(new_registry))
        ]
    ).astype(np.float32)
    old_weights = np.asarray(b3_state.weights, dtype=np.float32)
    # Compile Stage2-B independently, then append separately compiled Stage2-C
    # rows.  This prevents a joint compiler from silently rewriting old bytes.
    old_code1, old_code2, old_scale1, old_scale2, old_decoded = (
        _residual_quantize(old_weights)
    )
    new_code1, new_code2, new_scale1, new_scale2, new_decoded = (
        _residual_quantize(new_weights)
    )
    all_weights = np.concatenate([old_weights, new_weights], axis=0)
    code1 = np.concatenate([old_code1, new_code1], axis=0)
    code2 = np.concatenate([old_code2, new_code2], axis=0)
    scale1 = np.concatenate([old_scale1, new_scale1], axis=0)
    scale2 = np.concatenate([old_scale2, new_scale2], axis=0)
    decoded = np.concatenate([old_decoded, new_decoded], axis=0)
    old_count = len(old_registry)
    empty = np.zeros(0, dtype=np.float16)
    before_state = D37B3PreservingInt8State(
        schema=SCHEMA,
        classes=old_registry,
        old_class_count=old_count,
        log_diag_fp32=np.asarray(b3_state.log_diag, dtype=np.float32),
        code1_qint8=old_code1,
        code2_qint8=old_code2,
        scale1_fp16=old_scale1,
        scale2_fp16=old_scale2,
        new_offset_fp16=empty,
        margin_fp16=empty,
        arm=locked.arm,
    )
    state_no_offset = D37B3PreservingInt8State(
        schema=SCHEMA,
        classes=old_registry + new_registry,
        old_class_count=old_count,
        log_diag_fp32=np.asarray(b3_state.log_diag, dtype=np.float32),
        code1_qint8=code1,
        code2_qint8=code2,
        scale1_fp16=scale1,
        scale2_fp16=scale2,
        new_offset_fp16=empty,
        margin_fp16=empty,
        arm=locked.arm,
    )
    if not old_prefix_bitwise_unchanged_d37(before_state, state_no_offset):
        raise D37B3PreservingInt8Error("old int8 prefix changed during append")

    transformed_old = _transform(old_rows, b3_state.log_diag)
    reference_old_scores = np.float32(TEMPERATURE) * (
        transformed_old @ old_weights.T
    )
    compiled_old_scores = np.float32(TEMPERATURE) * (
        transformed_old @ old_decoded.T
    )
    old_support_decision_equivalent = np.array_equal(
        np.argmax(reference_old_scores, axis=1),
        np.argmax(compiled_old_scores, axis=1),
    )
    if not old_support_decision_equivalent:
        raise D37B3PreservingInt8Error(
            "residual-int8 old head changed a B3 support decision"
        )

    error = np.abs(decoded - all_weights)
    single_code, _, single_scale, _, single_decoded = _residual_quantize(
        all_weights
    )
    # The first stage from the same compiler is the matched single-int8 control.
    single_decoded.fill(0.0)
    for block_index, block in enumerate(BLOCK_SLICES):
        single_decoded[:, block] = (
            single_code[:, block].astype(np.float32)
            * single_scale[:, block_index].astype(np.float32)[:, None]
        )
    single_error = np.abs(single_decoded - all_weights)
    geometry = {
        "schema": "cvs.phase2.d37_b3_preserving_int8_geometry.v1",
        "feature_geometry": "b3_log_diag_then_unit_row",
        "fixed_quantization_blocks": [160, 96, 32],
        "two_level_residual_int8": True,
        "old_weight_source": "final_b3_target_support_weight",
        "new_weight_source": "same_b3_space_target_new_support_mean",
        "old_prefix_bitwise_unchanged": True,
        "old_support_decision_equivalent_to_fp32_b3": True,
        "old_support_score_error_max": float(
            np.max(np.abs(reference_old_scores - compiled_old_scores))
        ),
        "target_old_int8_used_for_prediction": True,
        "target_new_int8_used_for_prediction": True,
        "fp32_target_prototype_stored": False,
        "quantization_error_mean": float(np.mean(error)),
        "quantization_error_max": float(np.max(error)),
        "single_level_error_mean": float(np.mean(single_error)),
        "residual_error_reduction_fraction": float(
            1.0 - np.mean(error) / max(float(np.mean(single_error)), 1.0e-12)
        ),
        "all_new_classes_globally_finite": True,
        "class_id_specific_branch": False,
        "label_permutation_equivariant": True,
    }
    row_count = len(old_rows) + len(new_rows)
    class_count = len(all_weights)
    quantize_macs = int(4 * class_count * FEATURE_DIM)
    new_mean_macs = int(2 * len(new_rows) * FEATURE_DIM)
    resource = {
        "schema": "cvs.phase2.d37_b3_preserving_int8_resource.v1",
        "active_adapter_parameters": 0,
        "trainable_parameters": 0,
        "adaptation_epochs": 0,
        "optimizer_steps": 0,
        "estimated_registration_macs": new_mean_macs + quantize_macs,
        "estimated_macs_per_query": int(
            FEATURE_DIM + 2 * class_count * FEATURE_DIM
        ),
        "persistent_state_bytes": state_no_offset.persistent_state_bytes + 4,
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_cap_pass": state_no_offset.persistent_state_bytes + 4
        <= 256 * 1024,
        "dense_query_graph_bytes": 0,
        "query_dependent_batch_optimization": False,
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "single_received_iq_row_per_support_sample": True,
        "old_k_shot": old_k,
        "new_k_shot": new_k,
        "support_row_count": row_count,
    }
    trace = (
        {
            "solver": "direct_b3_weight_residual_int8_compile",
            "scope": "target_old",
            "class_count": old_count,
            "optimizer_steps": 0,
            "query_rows_used": 0,
        },
        {
            "solver": "same_geometry_support_mean_residual_int8_compile",
            "scope": "target_new",
            "class_count": len(new_registry),
            "optimizer_steps": 0,
            "query_rows_used": 0,
        },
    )
    return D37B3PreservingInt8Result(
        before_state=before_state,
        state_no_offset=state_no_offset,
        training_trace=trace,
        geometry_audit=geometry,
        resource_audit=resource,
    )


def base_score_d37_b3_preserving_int8(
    state: D37B3PreservingInt8State, features: np.ndarray
) -> np.ndarray:
    rows = _rows(features, "D37 scoring features")
    transformed = _transform(rows, state.log_diag_fp32)
    decoded = _decode(
        state.code1_qint8,
        state.code2_qint8,
        state.scale1_fp16,
        state.scale2_fp16,
    )
    # Explicit row-local construction: no query-query edge or batch assignment.
    scores = np.stack(
        [
            np.float32(TEMPERATURE)
            * np.asarray(row @ decoded.T, dtype=np.float32)
            for row in transformed
        ]
    )
    if not np.isfinite(scores).all():
        raise D37B3PreservingInt8Error("non-finite D37 base score")
    return _readonly(scores, np.float32)


def fit_oof_feasible_offset_d37(
    state_no_offset: D37B3PreservingInt8State,
    oof_base_scores: np.ndarray,
    oof_labels: Sequence[str],
    *,
    oof_fold_ids: Sequence[int],
    oof_physical_ids: Sequence[str],
    source: str,
) -> D37FeasibleOffsetResult:
    """Fit one shared new offset from physical support-OOF scores only."""

    if state_no_offset.old_class_count == len(state_no_offset.classes):
        raise D37B3PreservingInt8Error("D37 offset requires registered new classes")
    if state_no_offset.calibration_ready:
        raise D37B3PreservingInt8Error("D37 offset must start from uncalibrated state")
    scores = np.asarray(oof_base_scores)
    labels = np.asarray(tuple(str(value) for value in oof_labels))
    folds = np.asarray(tuple(int(value) for value in oof_fold_ids), dtype=np.int64)
    physical_ids = tuple(str(value) for value in oof_physical_ids)
    classes = state_no_offset.classes
    if (
        source != OOF_SOURCE
        or
        scores.dtype != np.float32
        or scores.ndim != 2
        or scores.shape != (len(labels), len(classes))
        or folds.shape != (len(labels),)
        or len(physical_ids) != len(labels)
        or len(set(physical_ids)) != len(physical_ids)
        or any(not value for value in physical_ids)
        or len(set(folds.tolist())) < 2
        or len(labels) < len(classes)
        or set(labels.tolist()) != set(classes)
        or not np.isfinite(scores).all()
    ):
        raise D37B3PreservingInt8Error("OOF score/label closure drift")
    old_count = state_no_offset.old_class_count
    if any(len(set(folds[labels == name].tolist())) < 2 for name in classes):
        raise D37B3PreservingInt8Error(
            "each class must have support-OOF evidence from at least two folds"
        )
    old_mask = np.isin(labels, np.asarray(classes[:old_count]))
    new_mask = ~old_mask
    if not np.any(old_mask) or not np.any(new_mask):
        raise D37B3PreservingInt8Error("OOF rows must cover old and new support")
    margin = float(ARM_MARGINS[state_no_offset.arm])
    old_scores = scores[old_mask]
    upper = float(
        np.min(
            np.max(old_scores[:, :old_count], axis=1)
            - np.max(old_scores[:, old_count:], axis=1)
            - margin
        )
    )
    lower_rows: list[float] = []
    for row, truth in zip(scores[new_mask], labels[new_mask], strict=True):
        truth_index = classes.index(str(truth))
        competing_new = row[old_count:].copy()
        competing_new[truth_index - old_count] = -np.inf
        other_new = float(np.max(competing_new))
        true_new = float(row[truth_index])
        # A shared offset cancels between new classes and therefore cannot
        # repair a wrong new-vs-new ordering.  Such a fold is infeasible.
        if true_new <= other_new + margin + NEW_STRICT_EPSILON:
            raise D37B3PreservingInt8Error(
                "empty OOF feasible interval: true new class does not strictly beat other new classes"
            )
        lower_rows.append(
            float(np.max(row[:old_count]))
            - true_new
            + margin
            + NEW_STRICT_EPSILON
        )
    lower = float(max(lower_rows))
    if lower > upper + 1.0e-7:
        raise D37B3PreservingInt8Error(
            f"empty OOF feasible interval: lower={lower:.8f}, upper={upper:.8f}"
        )
    midpoint = np.float16(0.5 * (lower + upper))
    if float(midpoint) < lower:
        midpoint = np.nextafter(midpoint, np.float16(np.inf))
    if float(midpoint) > upper:
        midpoint = np.nextafter(midpoint, np.float16(-np.inf))
    offset = float(midpoint)
    if offset < lower or offset > upper:
        raise D37B3PreservingInt8Error(
            "OOF feasible interval contains no deployable FP16 offset"
        )
    calibrated = D37B3PreservingInt8State(
        schema=state_no_offset.schema,
        classes=state_no_offset.classes,
        old_class_count=state_no_offset.old_class_count,
        log_diag_fp32=state_no_offset.log_diag_fp32,
        code1_qint8=state_no_offset.code1_qint8,
        code2_qint8=state_no_offset.code2_qint8,
        scale1_fp16=state_no_offset.scale1_fp16,
        scale2_fp16=state_no_offset.scale2_fp16,
        new_offset_fp16=np.asarray([midpoint], dtype=np.float16),
        margin_fp16=np.asarray([margin], dtype=np.float16),
        arm=state_no_offset.arm,
    )
    return D37FeasibleOffsetResult(
        state=calibrated,
        lower_bound=lower,
        upper_bound=upper,
        offset=float(calibrated.new_offset_fp16[0]),
        old_oof_count=int(np.sum(old_mask)),
        new_oof_count=int(np.sum(new_mask)),
        fold_count=len(set(folds.tolist())),
        physical_id_sha256=hashlib.sha256(
            "\n".join(sorted(physical_ids)).encode("utf-8")
        ).hexdigest(),
        source=source,
    )


def score_d37_b3_preserving_int8(
    state: D37B3PreservingInt8State, features: np.ndarray
) -> np.ndarray:
    if (
        len(state.classes) > state.old_class_count
        and not state.calibration_ready
    ):
        raise D37B3PreservingInt8Error(
            "registered D37 state is not OOF-calibrated; use base_score only for calibration diagnostics"
        )
    scores = np.array(base_score_d37_b3_preserving_int8(state, features), copy=True)
    if state.calibration_ready:
        scores[:, state.old_class_count :] += float(state.new_offset_fp16[0])
    return _readonly(scores, np.float32)


def predict_d37_b3_preserving_int8(
    state: D37B3PreservingInt8State, features: np.ndarray
) -> np.ndarray:
    scores = score_d37_b3_preserving_int8(state, features)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)]


def old_prefix_bitwise_unchanged_d37(
    before: D37B3PreservingInt8State,
    after: D37B3PreservingInt8State,
) -> bool:
    count = before.old_class_count
    return bool(
        before.classes == after.classes[:count]
        and np.array_equal(before.log_diag_fp32, after.log_diag_fp32)
        and np.array_equal(before.code1_qint8, after.code1_qint8[:count])
        and np.array_equal(before.code2_qint8, after.code2_qint8[:count])
        and np.array_equal(before.scale1_fp16, after.scale1_fp16[:count])
        and np.array_equal(before.scale2_fp16, after.scale2_fp16[:count])
    )


__all__ = [
    "D37B3PreservingInt8Config",
    "D37B3PreservingInt8Error",
    "D37B3PreservingInt8Result",
    "D37B3PreservingInt8State",
    "D37FeasibleOffsetResult",
    "base_score_d37_b3_preserving_int8",
    "fit_d37_b3_preserving_int8",
    "fit_oof_feasible_offset_d37",
    "old_prefix_bitwise_unchanged_d37",
    "predict_d37_b3_preserving_int8",
    "score_d37_b3_preserving_int8",
]
