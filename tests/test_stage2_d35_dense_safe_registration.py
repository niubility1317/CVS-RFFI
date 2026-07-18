from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d35_dense_safe_registration import (
    D35DenseSafeConfig,
    D35DenseSafeRegistrationError,
    fit_d35_dense_safe_registration,
    score_d35_dense_safe_registration,
)


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.asarray(
        value / np.linalg.norm(value, axis=1, keepdims=True), dtype=np.float32
    )


def _fixture(k_shot: int = 3):
    rng = np.random.default_rng(3500 + k_shot)
    old_classes = ("old-a", "old-b")
    new_classes = ("new-a", "new-b")
    old_centers = np.zeros((2, 288), dtype=np.float32)
    old_centers[0, 0] = 1.0
    old_centers[1, 1] = 1.0
    new_centers = np.zeros((2, 288), dtype=np.float32)
    new_centers[0, [0, 2]] = (0.55, 0.84)
    new_centers[1, [1, 3]] = (0.50, 0.87)
    new_centers = _unit(new_centers)
    old_rows = []
    new_rows = []
    old_labels = []
    new_labels = []
    for index, name in enumerate(old_classes):
        for _ in range(k_shot):
            old_rows.append(
                old_centers[index] + 0.01 * rng.normal(size=288).astype(np.float32)
            )
            old_labels.append(name)
    for index, name in enumerate(new_classes):
        for shot in range(k_shot):
            direction = 1.0 if shot % 2 == 0 else -1.0
            perturbation = np.zeros(288, dtype=np.float32)
            perturbation[4 + index] = 0.12 * direction
            new_rows.append(
                new_centers[index]
                + perturbation
                + 0.01 * rng.normal(size=288).astype(np.float32)
            )
            new_labels.append(name)
    old_x = _unit(np.stack(old_rows))
    new_x = _unit(np.stack(new_rows))
    old_prefix = np.asarray(old_x @ old_centers.T, dtype=np.float32)
    new_old_prefix = np.asarray(new_x @ old_centers.T, dtype=np.float32)
    return (
        old_x,
        np.asarray(old_labels),
        old_classes,
        old_prefix,
        new_x,
        np.asarray(new_labels),
        new_classes,
        new_old_prefix,
    )


def _fit(arm: str, k_shot: int = 3):
    values = _fixture(k_shot)
    result = fit_d35_dense_safe_registration(
        *values, config=D35DenseSafeConfig(arm=arm)
    )
    return result, values


@pytest.mark.parametrize(
    ("arm", "prototype_count", "buffer_lambda"),
    [("A", 1, 0.0), ("B", 2, 0.25), ("C", 2, 0.5)],
)
def test_fixed_arms_build_int8_dense_safe_state(
    arm: str, prototype_count: int, buffer_lambda: float
) -> None:
    result, _ = _fit(arm)
    state = result.state

    assert state.new_prototypes_qint8.shape == (2, prototype_count, 288)
    assert state.new_prototypes_qint8.dtype == np.int8
    assert state.prototype_mask.all()
    assert state.new_prototype_scales.dtype == np.float32
    assert state.new_prototype_inverse_norms.dtype == np.float32
    assert state.safety_thresholds.shape == (2, 2)
    assert state.prototype_selector.shape == (2, 2)
    assert state.prototype_selector.dtype == np.uint8
    assert np.isfinite(state.new_prototype_scales).all()
    assert np.isfinite(state.new_prototype_inverse_norms).all()
    assert np.isfinite(state.safety_thresholds).all()
    assert all(
        row["old_buffer_lambda"] == buffer_lambda
        for row in result.geometry_audit["threshold_trace"]
    )
    assert state.optimizer_steps == 0


def test_old_prefix_is_bitwise_preserved_and_fit_old_correct_rows_do_not_degrade() -> None:
    result, values = _fit("C")
    old_x, _, _, old_prefix = values[:4]
    scores = score_d35_dense_safe_registration(result.state, old_x, old_prefix)

    assert scores[:, :2].tobytes() == old_prefix.tobytes()
    assert result.geometry_audit["old_score_prefix_bitwise_preserved"] is True
    assert result.geometry_audit["old_support_intrusion_count"] == 0
    assert result.geometry_audit["old_support_non_degradation_pass"] is True
    before = np.argmax(old_prefix, axis=1)
    np.testing.assert_array_equal(np.argmax(scores, axis=1), before)


def test_all_new_classes_are_globally_visible_without_nonedge_minus_two_gate() -> None:
    result, values = _fit("B")
    new_x = values[4]
    new_old_prefix = values[7]
    state = result.state
    scores = score_d35_dense_safe_registration(state, new_x, new_old_prefix)

    q = state.new_prototypes_qint8.reshape(-1, 288).astype(np.float32)
    inverse = state.new_prototype_inverse_norms.reshape(-1)
    raw = (new_x @ q.T) * inverse[None, :] * np.float32(18.0)
    raw = raw.reshape(len(new_x), 2, 2)
    winner = np.argmax(new_old_prefix, axis=1)
    manual_new = np.empty((len(new_x), 2), dtype=np.float32)
    for row in range(len(new_x)):
        for new_class in range(2):
            selected = state.prototype_selector[winner[row], new_class]
            manual_new[row, new_class] = (
                raw[row, new_class, selected]
                - state.safety_thresholds[winner[row], new_class]
            )
    np.testing.assert_allclose(scores[:, 2:], manual_new, atol=1.0e-6, rtol=0.0)
    assert result.geometry_audit["all_new_classes_global_visible"] is True
    assert result.geometry_audit["global_visible_not_guaranteed_reachable"] is True
    assert result.geometry_audit["visibility_gate"] is False
    assert result.geometry_audit["nonedge_fallback"] is False
    assert not np.allclose(
        scores[:, 2:], np.max(new_old_prefix, axis=1, keepdims=True) - 2.0
    )


def test_development_old_and_new_physical_loso_are_separate() -> None:
    result, values = _fit("B", k_shot=3)
    geometry = result.geometry_audit

    assert geometry["k1_loso_status"] == "EVALUATED"
    assert len(geometry["old_leave_one_out"]) == len(values[0])
    assert len(geometry["new_physical_leave_one_out"]) == len(values[4])
    assert all("held_index" in row for row in geometry["old_leave_one_out"])
    assert all("margin" in row for row in geometry["new_physical_leave_one_out"])
    resource = result.resource_audit
    assert resource["estimated_deploy_refit_macs"] > 0
    assert resource["estimated_development_old_loso_macs"] > 0
    assert resource["estimated_development_new_loso_macs"] > 0


def test_k1_does_not_fabricate_loso() -> None:
    result, _ = _fit("C", k_shot=1)

    assert result.geometry_audit["k1_loso_status"] == "NOT_EVALUATED"
    assert result.geometry_audit["old_leave_one_out"] == [
        {"mode": "K1_NOT_EVALUATED", "status": "NOT_EVALUATED"}
    ]
    assert result.geometry_audit["new_physical_leave_one_out"] == [
        {"mode": "K1_NOT_EVALUATED", "status": "NOT_EVALUATED"}
    ]
    assert result.resource_audit["estimated_development_total_loso_macs"] == 0
    assert result.state.active_prototype_count == 2
    np.testing.assert_array_equal(
        result.state.prototype_selector, np.zeros((2, 2), dtype=np.uint8)
    )
    assert result.resource_audit["query_selected_prototype_count"] == 2
    assert result.resource_audit["estimated_macs_per_query"] == 2 * 288


def test_resource_counts_values_bytes_and_incremental_query_macs_separately() -> None:
    result, _ = _fit("B")
    state = result.state
    resource = result.resource_audit
    expected_active = (
        state.active_prototype_count * 288
        + state.active_prototype_count
        + state.active_prototype_count
        + state.prototype_selector.size
        + state.safety_thresholds.size
        + state.support_count_by_new_class.size
    )
    expected_query = 2 * 288

    assert resource["active_parameters"] == expected_active
    assert resource["persistent_state_bytes"] == state.persistent_state_bytes
    assert resource["active_parameters"] != resource["persistent_state_bytes"]
    assert resource["estimated_macs_per_query"] == expected_query
    assert resource["estimated_registration_macs_per_unit_query"] == expected_query
    assert resource["query_prototype_dot_macs"] == 2 * 288
    assert resource["query_inverse_temperature_scalar_ops"] == 4
    assert resource["query_threshold_subtraction_scalar_ops"] == 2
    assert resource["query_prototype_max_comparisons"] == 0
    assert resource["query_old_winner_argmax_comparisons"] == 1
    assert resource["estimated_scalar_ops_per_query"] == 7
    assert resource["prototype_selector_uint8_bytes"] == 4
    assert resource["dense_query_graph_bytes"] == 0


def test_winner_conditioned_selector_can_choose_different_stored_prototypes() -> None:
    old_classes = ("old-a", "old-b")
    new_classes = ("new-a", "new-b")
    old_centers = np.zeros((2, 288), dtype=np.float32)
    old_centers[0, 0] = 1.0
    old_centers[1, 1] = 1.0
    old_rows = []
    for old_index, side_dim in ((0, 4), (1, 5)):
        for offset in (0.02, -0.02, 0.01, -0.01):
            old_rows.append(
                old_centers[old_index]
                + offset * np.eye(288, dtype=np.float32)[side_dim]
            )
    old_x = _unit(np.stack(old_rows))
    old_y = np.asarray(["old-a"] * 4 + ["old-b"] * 4)
    new_rows = []
    for old_index, side_dim in ((0, 2), (1, 3)):
        for sign in (1.0, -1.0):
            row = 0.92 * old_centers[old_index]
            row = row + sign * 0.25 * np.eye(288, dtype=np.float32)[side_dim]
            new_rows.append(row)
    for sign in (1.0, -1.0, 0.7, -0.7):
        row = 0.80 * old_centers[0]
        row = row + sign * 0.20 * np.eye(288, dtype=np.float32)[6]
        new_rows.append(row)
    new_x = _unit(np.stack(new_rows))
    new_y = np.asarray(["new-a"] * 4 + ["new-b"] * 4)
    old_prefix = np.asarray(old_x @ old_centers.T, dtype=np.float32)
    new_old_prefix = np.asarray(new_x @ old_centers.T, dtype=np.float32)

    result = fit_d35_dense_safe_registration(
        old_x,
        old_y,
        old_classes,
        old_prefix,
        new_x,
        new_y,
        new_classes,
        new_old_prefix,
        config=D35DenseSafeConfig(arm="B"),
    )

    assert result.state.prototype_selector[0, 0] != result.state.prototype_selector[1, 0]
    assert result.resource_audit["estimated_macs_per_query"] == 2 * 288
    assert result.resource_audit["query_selected_prototype_count"] == 2


def test_rejects_nonunit_or_nonfloat32_fast_rows() -> None:
    values = list(_fixture())
    values[0] = values[0] * np.float32(1.01)
    with pytest.raises(D35DenseSafeRegistrationError, match="FAST-adapted unit"):
        fit_d35_dense_safe_registration(*values)

    values = list(_fixture())
    values[4] = values[4].astype(np.float64)
    with pytest.raises(D35DenseSafeRegistrationError, match="finite"):
        fit_d35_dense_safe_registration(*values)
