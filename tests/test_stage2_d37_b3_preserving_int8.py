from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_b3_fisher_closed_form import (
    B3FisherClosedFormState,
    SCHEMA as B3_SCHEMA,
    score_b3_fisher_closed_form,
)
from cvsrffi.stage2_d37_b3_preserving_int8 import (
    D37B3PreservingInt8Config,
    D37B3PreservingInt8Error,
    base_score_d37_b3_preserving_int8,
    fit_d37_b3_preserving_int8,
    fit_oof_feasible_offset_d37,
    old_prefix_bitwise_unchanged_d37,
    score_d37_b3_preserving_int8,
)


DIM = 288


def _unit(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _fixture(
    *, old_count: int = 3, new_count: int = 2, k_shot: int = 4
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    B3FisherClosedFormState,
]:
    rng = np.random.default_rng(3701 + old_count * 10 + new_count)
    anchors = _unit(rng.normal(size=(old_count + new_count, DIM)).astype(np.float32))
    old_classes = tuple(f"old-{index}" for index in range(old_count))
    new_classes = tuple(f"new-{index}" for index in range(new_count))

    def support(start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        labels = []
        names = (old_classes + new_classes)[start : start + count]
        for local_index, name in enumerate(names):
            anchor = anchors[start + local_index]
            for _ in range(k_shot):
                rows.append(anchor + 0.01 * rng.normal(size=DIM).astype(np.float32))
                labels.append(name)
        return np.asarray(rows, dtype=np.float32), np.asarray(labels)

    old_rows, old_labels = support(0, old_count)
    new_rows, new_labels = support(old_count, new_count)
    log_diag = np.linspace(-0.15, 0.15, DIM, dtype=np.float32)
    transformed_old = _unit(old_rows * np.exp(log_diag)[None, :])
    weights = np.stack(
        [
            _unit(
                np.mean(
                    transformed_old[old_labels == name], axis=0, keepdims=True
                )
            )[0]
            for name in old_classes
        ]
    ).astype(np.float32)
    b3 = B3FisherClosedFormState(
        schema=B3_SCHEMA,
        classes=old_classes,
        log_diag=log_diag,
        weights=weights,
        support_count_by_class=np.full(old_count, k_shot, dtype=np.uint16),
        selected_strength=1.0,
        active_scalars=DIM * (1 + old_count),
        optimizer_steps=0,
    )
    return (
        old_rows,
        old_labels,
        old_classes,
        new_rows,
        new_labels,
        new_classes,
        b3,
    )


def _fit(arm: str = "A", **kwargs):
    values = _fixture(**kwargs)
    result = fit_d37_b3_preserving_int8(
        *values[:6], values[6], config=D37B3PreservingInt8Config(arm)
    )
    return values, result


def _oof_metadata(labels: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    folds = np.tile(np.asarray([0, 1], dtype=np.int64), len(labels) // 2)
    physical_ids = tuple(f"physical-{index}" for index in range(len(labels)))
    return folds, physical_ids


def test_direct_b3_compile_preserves_old_decisions_and_byte_prefix() -> None:
    values, result = _fit()
    old_rows, _, _, _, _, _, b3 = values
    reference = score_b3_fisher_closed_form(b3, old_rows)
    compiled = score_d37_b3_preserving_int8(result.before_state, old_rows)
    assert np.array_equal(np.argmax(reference, axis=1), np.argmax(compiled, axis=1))
    assert old_prefix_bitwise_unchanged_d37(
        result.before_state, result.state_no_offset
    )
    assert result.geometry_audit["target_old_int8_used_for_prediction"] is True
    assert result.geometry_audit["target_new_int8_used_for_prediction"] is True
    assert result.geometry_audit["fp32_target_prototype_stored"] is False


def test_residual_int8_is_readonly_and_improves_single_level_error() -> None:
    _, result = _fit()
    state = result.state_no_offset
    assert state.code1_qint8.dtype == np.int8
    assert state.code2_qint8.dtype == np.int8
    assert state.scale1_fp16.dtype == np.float16
    assert state.scale2_fp16.dtype == np.float16
    assert not state.code1_qint8.flags.writeable
    assert not state.scale1_fp16.flags.writeable
    assert result.geometry_audit["quantization_error_mean"] < result.geometry_audit[
        "single_level_error_mean"
    ]
    assert result.geometry_audit["residual_error_reduction_fraction"] > 0.5


@pytest.mark.parametrize("arm,margin", [("A", 0.0), ("B", 0.05), ("C", 0.10)])
def test_oof_feasible_interval_enforces_old_safety_and_new_reachability(
    arm: str, margin: float
) -> None:
    values, result = _fit(arm)
    classes = result.state_no_offset.classes
    labels = np.repeat(np.asarray(classes), 2)
    scores = np.zeros((len(classes), len(classes)), dtype=np.float32)
    scores = np.repeat(scores, 2, axis=0)
    for row_index, name in enumerate(labels):
        scores[row_index, classes.index(str(name))] = 4.0
    folds, physical_ids = _oof_metadata(labels)
    calibrated = fit_oof_feasible_offset_d37(
        result.state_no_offset,
        scores,
        labels,
        oof_fold_ids=folds,
        oof_physical_ids=physical_ids,
        source="support_physical_rank_pair_crossfit",
    )
    assert calibrated.lower_bound <= calibrated.upper_bound
    assert calibrated.lower_bound <= calibrated.offset <= calibrated.upper_bound
    assert calibrated.state.calibration_ready
    assert float(calibrated.state.margin_fp16[0]) == pytest.approx(
        margin, abs=3.0e-5
    )
    shifted = scores.copy()
    shifted[:, result.state_no_offset.old_class_count :] += calibrated.offset
    assert np.array_equal(
        np.argmax(shifted, axis=1),
        np.asarray([classes.index(str(name)) for name in labels]),
    )
    assert old_prefix_bitwise_unchanged_d37(
        result.before_state, calibrated.state
    )


def test_empty_oof_interval_fails_closed() -> None:
    _, result = _fit("A")
    classes = result.state_no_offset.classes
    old_count = result.state_no_offset.old_class_count
    labels = np.repeat(np.asarray(classes), 2)
    scores = np.zeros((len(labels), len(classes)), dtype=np.float32)
    # Every old row prefers a new class; every new row prefers an old class.
    old_mask = np.isin(labels, np.asarray(classes[:old_count]))
    scores[old_mask, old_count] = 2.0
    scores[~old_mask, 0] = 2.0
    folds, physical_ids = _oof_metadata(labels)
    with pytest.raises(D37B3PreservingInt8Error, match="empty OOF feasible interval"):
        fit_oof_feasible_offset_d37(
            result.state_no_offset,
            scores,
            labels,
            oof_fold_ids=folds,
            oof_physical_ids=physical_ids,
            source="support_physical_rank_pair_crossfit",
        )


def test_shared_offset_cannot_repair_wrong_new_to_new_order() -> None:
    _, result = _fit("A")
    classes = result.state_no_offset.classes
    old_count = result.state_no_offset.old_class_count
    labels = np.repeat(np.asarray(classes), 2)
    scores = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for row_index, name in enumerate(labels):
        truth_index = classes.index(str(name))
        scores[row_index, truth_index] = 4.0
    first_new_rows = labels == classes[old_count]
    scores[first_new_rows, old_count] = 0.0
    scores[first_new_rows, old_count + 1] = 2.0
    folds, physical_ids = _oof_metadata(labels)
    with pytest.raises(
        D37B3PreservingInt8Error, match="true new class does not strictly beat"
    ):
        fit_oof_feasible_offset_d37(
            result.state_no_offset,
            scores,
            labels,
            oof_fold_ids=folds,
            oof_physical_ids=physical_ids,
            source="support_physical_rank_pair_crossfit",
        )


def test_oof_calibration_requires_crossfit_provenance() -> None:
    _, result = _fit("A")
    classes = result.state_no_offset.classes
    labels = np.repeat(np.asarray(classes), 2)
    scores = np.zeros((len(labels), len(classes)), dtype=np.float32)
    folds, physical_ids = _oof_metadata(labels)
    with pytest.raises(D37B3PreservingInt8Error, match="OOF score/label closure"):
        fit_oof_feasible_offset_d37(
            result.state_no_offset,
            scores,
            labels,
            oof_fold_ids=folds,
            oof_physical_ids=physical_ids,
            source="query_or_in_sample",
        )


def test_row_scoring_is_batch_split_and_order_invariant() -> None:
    values, result = _fit("A")
    query = np.concatenate([values[0][:3], values[3][:3]], axis=0)
    together = base_score_d37_b3_preserving_int8(result.state_no_offset, query)
    separate = np.concatenate(
        [
            base_score_d37_b3_preserving_int8(
                result.state_no_offset, row[None, :]
            )
            for row in query
        ],
        axis=0,
    )
    assert np.allclose(together, separate, atol=1.0e-6)
    order = np.asarray([5, 2, 0, 4, 1, 3])
    reordered = base_score_d37_b3_preserving_int8(
        result.state_no_offset, query[order]
    )
    assert np.allclose(reordered, together[order], atol=1.0e-6)
    assert np.isfinite(together).all()


@pytest.mark.parametrize("new_count", [2, 5, 10, 20])
def test_resource_caps_hold_for_registered_scale(new_count: int) -> None:
    _, result = _fit(old_count=6, new_count=new_count, k_shot=1)
    resource = result.resource_audit
    assert resource["active_adapter_parameters"] == 0
    assert resource["optimizer_steps"] == 0
    assert resource["adaptation_epochs"] == 0
    assert resource["persistent_state_cap_pass"] is True
    assert resource["persistent_state_bytes"] <= 256 * 1024
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["query_dependent_batch_optimization"] is False


def test_scoring_never_fits_or_opens_query_state() -> None:
    values, result = _fit()
    before_bytes = result.before_state.code1_qint8.tobytes()
    _ = score_d37_b3_preserving_int8(result.before_state, values[0])
    assert result.before_state.code1_qint8.tobytes() == before_bytes
    with pytest.raises(D37B3PreservingInt8Error, match="not OOF-calibrated"):
        score_d37_b3_preserving_int8(result.state_no_offset, values[3])
    assert result.resource_audit["query_rows_used_for_fit"] == 0
    assert result.resource_audit["query_labels_used_for_fit"] is False
    assert result.resource_audit["query_role_oracle_access"] is False
