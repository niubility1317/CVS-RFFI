from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cvsrffi.stage2_d38_strong_b3_quantized import (
    D38_SCORE_TEMPERATURE,
    D38StrongB3Config,
    fit_d38_strong_b3_quantized,
    score_d38_strong_b3,
)
from cvsrffi.stage2_d39_angular_radius import (
    D39AngularRadiusConfig,
    D39AngularRadiusError,
    D39AngularRadiusState,
    fit_d39_angular_radius,
    old_prefix_bitwise_unchanged_d39,
    pairwise_support_diagnostics_d39,
    predict_d39_angular_radius,
    score_d39_angular_radius,
)


def _support(
    class_names: tuple[str, ...], *, k: int, offset: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(class_names):
        center = np.zeros(288, dtype=np.float32)
        center[(offset + class_index * 17) % 288] = 1.0
        center[(offset + class_index * 17 + 43) % 288] = 0.3
        for _ in range(k):
            rows.append(
                (center + rng.normal(0.0, 0.018, 288)).astype(np.float32)
            )
            labels.append(label)
    return np.stack(rows).astype(np.float32), np.asarray(labels)


def _inputs(k: int = 3, new_count: int = 2):
    old_classes = ("old_a", "old_b")
    new_classes = tuple(f"new_{index}" for index in range(new_count))
    old_x, old_y = _support(old_classes, k=k, offset=0, seed=10 + k)
    new_x, new_y = _support(new_classes, k=k, offset=101, seed=20 + k)
    return old_classes, new_classes, old_x, old_y, new_x, new_y


def _fit(k: int = 3, new_count: int = 2):
    old_classes, new_classes, old_x, old_y, new_x, new_y = _inputs(
        k, new_count
    )
    result = fit_d39_angular_radius(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=713101,
        device="cpu",
    )
    return result, old_x, old_y, new_x, new_y


def test_d39_preserves_exact_d38_b_training_trace() -> None:
    old_classes, new_classes, old_x, old_y, new_x, new_y = _inputs()
    d38 = fit_d38_strong_b3_quantized(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=713101,
        device="cpu",
        config=D38StrongB3Config("B"),
    )
    d39 = fit_d39_angular_radius(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=713101,
        device="cpu",
    )

    assert d39.training_trace == d38.training_trace
    assert len(d39.training_trace) == 30
    assert d39.resource_audit["optimizer_steps"] == 30
    assert d39.resource_audit["radius_fit_optimizer_steps"] == 0
    assert d39.geometry_audit["base_route"] == "D38-B_training_trajectory_unchanged"
    assert d39.geometry_audit["old_radius_materialized_before_stage2c"] is True
    assert d39.geometry_audit["old_radius_materialization_hook_call_count"] == 1
    assert d39.geometry_audit["old_radius_materialization_stage2b_trace_length"] == 20
    assert d39.geometry_audit["old_radius_materialization_last_optimizer_step"] == 20


def test_formula_golden_uses_public_d38_cosine_and_fp16_radius() -> None:
    result, old_x, _, new_x, _ = _fit()
    rows = np.concatenate([old_x[:2], new_x[:2]], axis=0).astype(np.float32)
    raw = score_d38_strong_b3(result.state.base_state, rows)
    cosine = np.clip(raw / np.float32(D38_SCORE_TEMPERATURE), -1.0, 1.0)
    theta = np.arccos(cosine).astype(np.float32)
    radius = result.state.radius_fp16.astype(np.float32) + np.float32(0.001)
    expected = np.asarray(
        -0.5 * (theta / radius[None, :]) ** 2 - np.log(radius[None, :]),
        dtype=np.float32,
    )

    assert np.array_equal(score_d39_angular_radius(result.state, rows), expected)
    assert np.isfinite(expected).all()


def test_formal_state_is_int8_only_and_old_base_radius_r0_are_append_only() -> None:
    result, _, _, _, _ = _fit()
    before = result.before_state
    after = result.state
    ablation = result.matched_fp32_state

    assert before.is_int8 and after.is_int8
    assert before.base_state.fp32_weights.shape == (0, 288)
    assert after.base_state.fp32_weights.shape == (0, 288)
    assert before.radius_fp16.dtype == np.float16
    assert after.radius_fp16.dtype == np.float16
    assert after.r0_fp16.dtype == np.float16
    assert old_prefix_bitwise_unchanged_d39(before, after)
    assert np.array_equal(before.radius_fp16, after.radius_fp16[:2])
    assert np.array_equal(before.r0_fp16, after.r0_fp16)
    assert not ablation.is_int8
    assert np.array_equal(after.radius_fp16, ablation.radius_fp16)
    assert np.array_equal(after.r0_fp16, ablation.r0_fp16)
    assert result.resource_audit["resident_fp32_target_prototype_count"] == 0
    assert result.geometry_audit["matched_fp32_reuses_exact_fp16_radius"] is True


def test_k1_strictly_degenerates_every_radius_to_frozen_r0() -> None:
    result, _, _, _, _ = _fit(k=1)

    expected = np.full_like(result.state.radius_fp16, result.state.r0_fp16[0])
    assert np.array_equal(result.state.radius_fp16, expected)
    assert result.geometry_audit["k1_radius_equals_r0"] is True
    assert result.resource_audit["old_k_shot"] == 1
    assert result.resource_audit["new_k_shot"] == 1
    assert result.resource_audit["held_radius_fit_row_count"] == 0


def test_k_greater_than_one_radius_matches_locked_shrinkage_formula() -> None:
    k_shot = 5
    result, old_x, old_y, new_x, new_y = _fit(k=k_shot)
    support_parts = (
        (
            result.before_state.base_state,
            old_x,
            old_y,
            result.before_state.classes,
            result.before_state.radius_fp16,
        ),
        (
            result.state.base_state,
            new_x,
            new_y,
            result.state.classes[result.state.old_class_count :],
            result.state.radius_fp16[result.state.old_class_count :],
        ),
    )
    r0 = float(result.state.r0_fp16[0])
    for base_state, features, labels, classes, actual_radius in support_parts:
        raw = score_d38_strong_b3(base_state, features)
        expected: list[float] = []
        for label in classes:
            column = base_state.classes.index(label)
            cosine = np.clip(
                raw[np.asarray(labels) == label, column]
                / np.float32(D38_SCORE_TEMPERATURE),
                -1.0,
                1.0,
            )
            theta = np.arccos(cosine).astype(np.float32)
            m2 = float(np.mean(theta * theta))
            expected.append(
                np.sqrt(
                    (4.0 * r0 * r0 + (k_shot - 1) * m2)
                    / (4.0 + k_shot - 1)
                )
            )
        assert np.array_equal(actual_radius, np.asarray(expected, dtype=np.float16))


def test_new_support_cannot_change_old_r0_or_old_radius_prefix() -> None:
    old_classes, new_classes, old_x, old_y, new_x, new_y = _inputs()
    first = fit_d39_angular_radius(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=91,
    )
    perturbed_new = np.asarray(new_x[:, ::-1], dtype=np.float32)
    second = fit_d39_angular_radius(
        old_x,
        old_y,
        old_classes,
        perturbed_new,
        new_y,
        new_classes,
        seed=91,
    )

    assert np.array_equal(first.before_state.r0_fp16, second.before_state.r0_fp16)
    assert np.array_equal(
        first.before_state.radius_fp16, second.before_state.radius_fp16
    )
    assert np.array_equal(
        first.before_state.base_state.code1_qint8,
        second.before_state.base_state.code1_qint8,
    )
    assert first.geometry_audit["old_radius_new_support_row_count"] == 0
    assert first.geometry_audit["new_radius_old_support_row_count"] == 0


def test_scoring_is_row_local_split_order_invariant_and_all_class() -> None:
    result, old_x, _, new_x, _ = _fit()
    rows = np.concatenate([old_x, new_x], axis=0).astype(np.float32)
    whole = score_d39_angular_radius(result.state, rows)
    split = np.concatenate(
        [
            score_d39_angular_radius(result.state, rows[:5]),
            score_d39_angular_radius(result.state, rows[5:]),
        ],
        axis=0,
    )
    permutation = np.asarray([7, 0, 9, 2, 11, 4, 1, 10, 3, 8, 5, 6])

    assert whole.shape == (len(rows), len(result.state.classes))
    assert np.array_equal(whole, split)
    assert np.array_equal(
        score_d39_angular_radius(result.state, rows[permutation]),
        whole[permutation],
    )
    assert np.array_equal(
        predict_d39_angular_radius(result.state, rows),
        np.asarray(result.state.classes)[np.argmax(whole, axis=1)],
    )


def test_pairwise_diagnostic_records_calibrated_new_new_and_new_old() -> None:
    result, _, _, new_x, new_y = _fit()
    rows = pairwise_support_diagnostics_d39(
        result.state,
        new_x,
        new_y,
        [f"physical-{index}" for index in range(len(new_x))],
        scenario="leo_clear_weak",
        outer_fold=3,
        physical_ranks=[0, 1, 2, 0, 1, 2],
    )

    assert len(rows) == len(new_x)
    assert all(row["query_rows_used"] == 0 for row in rows)
    assert all(row["true_new_handle"] != row["top_competing_new_handle"] for row in rows)
    assert all(np.isfinite(row["new_new_margin"]) for row in rows)
    assert all(np.isfinite(row["new_old_margin"]) for row in rows)
    assert {row["scenario"] for row in rows} == {"leo_clear_weak"}


def test_label_permutation_keeps_numeric_geometry_and_radius() -> None:
    old_classes, new_classes, old_x, old_y, new_x, new_y = _inputs(k=2)
    first = fit_d39_angular_radius(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=99,
    )
    old_map = {"old_a": "x", "old_b": "y"}
    new_map = {"new_0": "u", "new_1": "v"}
    second = fit_d39_angular_radius(
        old_x,
        np.asarray([old_map[value] for value in old_y]),
        ("x", "y"),
        new_x,
        np.asarray([new_map[value] for value in new_y]),
        ("u", "v"),
        seed=99,
    )
    rows = np.concatenate([old_x, new_x], axis=0).astype(np.float32)

    assert np.array_equal(first.state.radius_fp16, second.state.radius_fp16)
    assert np.array_equal(first.state.r0_fp16, second.state.r0_fp16)
    assert np.array_equal(
        score_d39_angular_radius(first.state, rows),
        score_d39_angular_radius(second.state, rows),
    )


@pytest.mark.parametrize("new_count", [2, 5, 10, 20])
def test_resource_state_and_scalar_operation_caps_hold_by_new_scale(
    new_count: int,
) -> None:
    result, _, _, _, _ = _fit(k=1, new_count=new_count)
    resource = result.resource_audit

    assert resource["trainable_parameter_cap_pass"] is True
    assert resource["adaptation_epoch_cap_pass"] is True
    assert resource["optimizer_step_cap_pass"] is True
    assert resource["persistent_state_cap_pass"] is True
    assert result.state.persistent_state_bytes < 256 * 1024
    assert resource["radius_state_bytes"] == 2 * (2 + new_count + 1)
    assert resource["wrapper_metadata_bytes"] > 0
    assert resource["per_query_acos_scalar_operations"] == 2 + new_count
    assert resource["per_query_log_scalar_operations"] == 2 + new_count
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["dense_query_graph_bytes"] == 0


@pytest.mark.parametrize("k_shot", [1, 5, 10, 20])
def test_locked_budget_and_radius_formula_hold_by_k(k_shot: int) -> None:
    result, _, _, _, _ = _fit(k=k_shot)

    assert len(result.training_trace) == 30
    assert result.resource_audit["optimizer_steps"] == 30
    assert result.resource_audit["old_k_shot"] == k_shot
    assert result.resource_audit["new_k_shot"] == k_shot
    assert np.isfinite(result.state.radius_fp16).all()
    assert np.all(result.state.radius_fp16 > 0)


def test_invalid_radius_state_and_config_fail_closed() -> None:
    result, _, _, _, _ = _fit()
    bad_radius = np.array(result.state.radius_fp16, copy=True)
    bad_radius[0] = np.float16(0.0)

    with pytest.raises(D39AngularRadiusError):
        D39AngularRadiusState(
            schema=result.state.schema,
            base_state=result.state.base_state,
            radius_fp16=bad_radius,
            r0_fp16=result.state.r0_fp16,
        )
    with pytest.raises(D39AngularRadiusError):
        D39AngularRadiusConfig(nu=5.0)


def test_formal_state_rejects_non_b_d38_base() -> None:
    result, _, _, _, _ = _fit()
    with pytest.raises(D39AngularRadiusError):
        D39AngularRadiusState(
            schema=result.state.schema,
            base_state=replace(result.state.base_state, arm="A"),
            radius_fp16=result.state.radius_fp16,
            r0_fp16=result.state.r0_fp16,
        )
