from __future__ import annotations

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_d42_allclass_fold_consensus_plane as afcp
from cvsrffi.stage2_d92_registration_balanced_covariance import OLD_CLASS_COUNT


def _state(class_count: int = 11) -> d42.D42UnifiedShrinkageLDAState:
    code1 = np.zeros((class_count, d42.FEATURE_DIM), dtype=np.int8)
    for class_index in range(class_count):
        code1[class_index, class_index] = 1
        code1[class_index, 160 + class_index] = 1
        code1[class_index, 256 + class_index] = 1
    return d42.D42UnifiedShrinkageLDAState(
        schema=d42.SCHEMA_INT8,
        classes=tuple(f"tx_{index}" for index in range(class_count)),
        old_class_count=OLD_CLASS_COUNT,
        log_diag_fp32=np.zeros(d42.FEATURE_DIM, dtype=np.float32),
        coef1_qint8=code1,
        coef2_qint8=np.zeros_like(code1),
        scale1_fp16=np.ones(
            (class_count, len(d42.BLOCK_SLICES)), dtype=np.float16
        ),
        scale2_fp16=np.full(
            (class_count, len(d42.BLOCK_SLICES)), 0.25, dtype=np.float16
        ),
        intercept_fp16=np.zeros(class_count, dtype=np.float16),
        coef_fp32=np.zeros((0, d42.FEATURE_DIM), dtype=np.float32),
        intercept_fp32=np.zeros(0, dtype=np.float32),
        covariance_policy="sklearn_lsqr_auto_shrinkage_equal_prior",
    )


def _active_state(class_count: int = 11) -> d42.D42UnifiedShrinkageLDAState:
    code1 = np.zeros((class_count, d42.FEATURE_DIM), dtype=np.int8)
    for class_index in range(class_count):
        code1[class_index, 20 + class_index] = 1
    return d42.D42UnifiedShrinkageLDAState(
        schema=d42.SCHEMA_INT8,
        classes=tuple(f"tx_{index}" for index in range(class_count)),
        old_class_count=OLD_CLASS_COUNT,
        log_diag_fp32=np.zeros(d42.FEATURE_DIM, dtype=np.float32),
        coef1_qint8=code1,
        coef2_qint8=np.zeros_like(code1),
        scale1_fp16=np.ones(
            (class_count, len(d42.BLOCK_SLICES)), dtype=np.float16
        ),
        scale2_fp16=np.full(
            (class_count, len(d42.BLOCK_SLICES)), 0.25, dtype=np.float16
        ),
        intercept_fp16=np.zeros(class_count, dtype=np.float16),
        coef_fp32=np.zeros((0, d42.FEATURE_DIM), dtype=np.float32),
        intercept_fp32=np.zeros(0, dtype=np.float32),
        covariance_policy="sklearn_lsqr_auto_shrinkage_equal_prior",
    )


def _active_support(class_count: int = 11, k_shot: int = 4):
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for class_index in range(class_count):
        for sample_index in range(k_shot):
            row = np.zeros(d42.FEATURE_DIM, dtype=np.float32)
            row[20 + class_index] = np.float32(0.1)
            row[100 + class_index] = np.float32(1e-3 * (sample_index + 1))
            row[[0, 160, 256]] = np.float32(class_index - 5)
            rows.append(row)
            targets.append(class_index)
    raw = np.stack(rows)
    filler_coordinates = np.asarray(
        [*range(120, 150), *range(200, 255)], dtype=np.int64
    )
    squared_norms = np.sum(np.square(raw, dtype=np.float64), axis=1)
    raw[:, filler_coordinates] = np.sqrt(
        (100.0 - squared_norms) / float(len(filler_coordinates))
    ).astype(np.float32)[:, None]
    return (
        d42._transform(raw, np.zeros(d42.FEATURE_DIM, dtype=np.float32)),
        np.asarray(targets, dtype=np.int64),
    )


def test_afcp_keeps_k2_as_byte_exact_d92_full_alias():
    state = _state()
    rows = np.zeros((len(state.classes) * 2, d42.FEATURE_DIM), dtype=np.float32)
    targets = np.repeat(np.arange(len(state.classes), dtype=np.int64), 2)

    candidate, receipt = afcp.apply_d42_allclass_fold_consensus_plane(
        state,
        rows,
        targets,
        old_class_count=OLD_CLASS_COUNT,
    )

    assert candidate is state
    assert receipt["d92_afcp_active"] is False
    assert receipt["d92_afcp_fallback_active"] is False
    assert receipt["d92_afcp_fallback_reason"] == "K1_K2_EXACT_D92_FULL_ALIAS"
    assert receipt["d92_afcp_final_state_sha256"] == receipt["d92_afcp_e0_state_sha256"]
    assert receipt["d92_afcp_changed_code2_count"] == 0


def test_afcp_publishes_one_guarded_allclass_three_block_code_plane():
    state = _active_state()
    rows, targets = _active_support()

    candidate, receipt = afcp.apply_d42_allclass_fold_consensus_plane(
        state,
        rows,
        targets,
        old_class_count=OLD_CLASS_COUNT,
    )

    assert receipt["d92_afcp_active"] is True
    assert receipt["d92_afcp_fallback_active"] is False
    assert receipt["d92_afcp_modified_state_field_names"] == ["coef2_qint8"]
    assert receipt["d92_afcp_all_three_blocks_changed"] is True
    assert all(count > 0 for count in receipt["d92_afcp_block_changed_code2_counts"])
    assert 0 < receipt["d92_afcp_changed_code2_count"] <= 3 * len(state.classes)
    assert receipt["d92_afcp_support_guard_pass"] is True
    assert receipt["d92_afcp_final_state_sha256"] != receipt["d92_afcp_e0_state_sha256"]
    assert not np.array_equal(candidate.coef2_qint8, state.coef2_qint8)
