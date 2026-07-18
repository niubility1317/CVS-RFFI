from __future__ import annotations

import numpy as np
import pytest
import torch

from cvsrffi.stage2_d38_strong_b3_quantized import (
    D38_SCORE_TEMPERATURE,
    D38StrongB3Config,
    D38StrongB3QuantizedError,
    fit_d38_strong_b3_quantized,
    old_prefix_bitwise_unchanged_d38,
    pairwise_support_diagnostics_d38,
    predict_d38_strong_b3,
    score_d38_strong_b3,
)


def test_public_score_temperature_seam_is_locked() -> None:
    assert D38_SCORE_TEMPERATURE == 18.0


def _support(
    class_names: tuple[str, ...], *, k: int, offset: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(class_names):
        center = np.zeros(288, dtype=np.float32)
        center[(offset + class_index * 13) % 288] = 1.0
        center[(offset + class_index * 13 + 37) % 288] = 0.35
        for _ in range(k):
            row = center + rng.normal(0.0, 0.015, 288).astype(np.float32)
            rows.append(row.astype(np.float32))
            labels.append(label)
    return np.stack(rows).astype(np.float32), np.asarray(labels)


def _fit(arm: str = "B"):
    old_classes = ("old_a", "old_b")
    new_classes = ("new_a", "new_b")
    old_x, old_y = _support(old_classes, k=3, offset=0, seed=10)
    new_x, new_y = _support(new_classes, k=3, offset=101, seed=20)
    result = fit_d38_strong_b3_quantized(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=713101,
        device=torch.device("cpu"),
        config=D38StrongB3Config(arm),
    )
    return result, old_x, old_y, new_x, new_y


def test_arm_b_is_20_plus_10_fullbatch_and_formal_resource_passes() -> None:
    result, _, _, _, _ = _fit("B")

    assert len(result.training_trace) == 30
    assert [row["phase"] for row in result.training_trace[:20]] == [
        "stage2b_fullbatch_old_adaptation"
    ] * 20
    assert [row["phase"] for row in result.training_trace[20:]] == [
        "stage2c_all_support_new_weight_only"
    ] * 10
    assert [row["optimizer_step"] for row in result.training_trace] == list(
        range(1, 31)
    )
    assert result.resource_audit["adaptation_epochs"] == 30
    assert result.resource_audit["optimizer_steps"] == 30
    assert result.resource_audit["trainable_parameter_cap_pass"] is True
    assert result.resource_audit["adaptation_epoch_cap_pass"] is True
    assert result.resource_audit["optimizer_step_cap_pass"] is True
    assert result.resource_audit["persistent_state_cap_pass"] is True
    assert result.resource_audit["query_rows_used_for_fit"] == 0


def test_before_stage2c_hook_sees_only_completed_stage2b_trace() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_a", "new_b")
    old_x, old_y = _support(old_classes, k=2, offset=0, seed=501)
    new_x, new_y = _support(new_classes, k=2, offset=90, seed=502)
    captured: list[tuple[object, tuple[dict[str, object], ...]]] = []

    def hook(state, trace) -> None:
        captured.append((state, trace))

    result = fit_d38_strong_b3_quantized(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=713101,
        config=D38StrongB3Config("B"),
        before_stage2c_hook=hook,
    )

    assert len(captured) == 1
    before_state, stage2b_trace = captured[0]
    assert before_state is result.before_state
    assert len(stage2b_trace) == 20
    assert all(row["phase"] == "stage2b_fullbatch_old_adaptation" for row in stage2b_trace)
    assert stage2b_trace[-1]["optimizer_step"] == 20
    assert len(result.training_trace) == 30
    assert result.training_trace[20]["phase"] == "stage2c_all_support_new_weight_only"


def test_arm_a_is_centroid_only_after_20_stage2b_steps() -> None:
    result, _, _, _, _ = _fit("A")

    assert len(result.training_trace) == 20
    assert result.resource_audit["adaptation_epochs"] == 20
    assert result.resource_audit["optimizer_steps"] == 20
    assert result.geometry_audit["stage2c_solver"] == "centroid_only"


def test_formal_state_is_int8_only_and_old_prefix_is_append_only() -> None:
    result, _, _, _, _ = _fit("B")
    before = result.before_state
    after = result.state

    assert before.is_int8 and after.is_int8
    assert before.fp32_weights.shape == (0, 288)
    assert after.fp32_weights.shape == (0, 288)
    assert before.code1_qint8.dtype == np.int8
    assert after.code2_qint8.dtype == np.int8
    assert after.scale1_fp16.dtype == np.float16
    assert after.inverse_norm_fp16.dtype == np.float16
    assert old_prefix_bitwise_unchanged_d38(before, after)
    assert result.resource_audit["resident_fp32_target_prototype_count"] == 0
    array_bytes = sum(
        value.nbytes
        for value in (
            after.log_diag_fp32,
            after.code1_qint8,
            after.code2_qint8,
            after.scale1_fp16,
            after.scale2_fp16,
            after.inverse_norm_fp16,
            after.fp32_weights,
        )
    )
    assert after.registry_state_bytes > 0
    assert after.persistent_state_bytes == array_bytes + after.registry_state_bytes
    assert (
        result.resource_audit["registry_state_bytes"]
        == after.registry_state_bytes
    )
    assert result.geometry_audit["old_compiled_before_stage2c"] is True
    assert result.geometry_audit["stage2c_uses_decoded_int8_old_head"] is True


def test_fp32_ablation_is_explicit_and_not_the_formal_state() -> None:
    result, old_x, _, new_x, _ = _fit("B")
    formal = result.state
    ablation = result.matched_fp32_state

    assert formal.is_int8 is True
    assert ablation.is_int8 is False
    assert ablation.fp32_weights.shape == (4, 288)
    assert ablation.code1_qint8.shape == (0, 288)
    rows = np.concatenate([old_x, new_x], axis=0).astype(np.float32)
    assert score_d38_strong_b3(formal, rows).shape == (12, 4)
    assert score_d38_strong_b3(ablation, rows).shape == (12, 4)


def test_scoring_is_row_local_split_and_order_invariant() -> None:
    result, old_x, _, new_x, _ = _fit("B")
    rows = np.concatenate([old_x, new_x], axis=0).astype(np.float32)
    whole = score_d38_strong_b3(result.state, rows)
    split = np.concatenate(
        [
            score_d38_strong_b3(result.state, rows[:4]),
            score_d38_strong_b3(result.state, rows[4:]),
        ],
        axis=0,
    )
    permutation = np.asarray([7, 0, 9, 2, 11, 4, 1, 10, 3, 8, 5, 6])
    permuted = score_d38_strong_b3(result.state, rows[permutation])

    assert np.array_equal(whole, split)
    assert np.array_equal(permuted, whole[permutation])
    assert np.array_equal(
        predict_d38_strong_b3(result.state, rows),
        np.asarray(result.state.classes)[np.argmax(whole, axis=1)],
    )


def test_pairwise_diagnostic_records_new_new_and_new_old_margins() -> None:
    result, _, _, new_x, new_y = _fit("B")
    rows = pairwise_support_diagnostics_d38(
        result.state,
        new_x,
        new_y,
        [f"physical-{index}" for index in range(len(new_x))],
        scenario="leo_clear_weak",
        outer_fold=2,
        physical_ranks=[0, 1, 2, 0, 1, 2],
    )

    assert len(rows) == len(new_x)
    assert all(row["query_rows_used"] == 0 for row in rows)
    assert all(row["true_new_handle"] != row["top_competing_new_handle"] for row in rows)
    assert all(np.isfinite(row["new_new_margin"]) for row in rows)
    assert all(np.isfinite(row["new_old_margin"]) for row in rows)
    assert {row["scenario"] for row in rows} == {"leo_clear_weak"}


def test_label_permutation_keeps_same_numeric_geometry() -> None:
    old_x, old_y = _support(("old_a", "old_b"), k=2, offset=0, seed=31)
    new_x, new_y = _support(("new_a", "new_b"), k=2, offset=91, seed=32)
    first = fit_d38_strong_b3_quantized(
        old_x,
        old_y,
        ("old_a", "old_b"),
        new_x,
        new_y,
        ("new_a", "new_b"),
        seed=99,
        config=D38StrongB3Config("B"),
    )
    renamed_old = np.asarray([{"old_a": "x", "old_b": "y"}[v] for v in old_y])
    renamed_new = np.asarray([{"new_a": "u", "new_b": "v"}[v] for v in new_y])
    second = fit_d38_strong_b3_quantized(
        old_x,
        renamed_old,
        ("x", "y"),
        new_x,
        renamed_new,
        ("u", "v"),
        seed=99,
        config=D38StrongB3Config("B"),
    )
    rows = np.concatenate([old_x, new_x], axis=0).astype(np.float32)

    assert np.array_equal(
        score_d38_strong_b3(first.state, rows),
        score_d38_strong_b3(second.state, rows),
    )


def test_invalid_registry_and_nonfinite_rows_fail_closed() -> None:
    old_x, old_y = _support(("a", "b"), k=2, offset=0, seed=3)
    new_x, new_y = _support(("c", "d"), k=2, offset=70, seed=4)
    new_x[0, 0] = np.nan

    with pytest.raises(D38StrongB3QuantizedError):
        fit_d38_strong_b3_quantized(
            old_x,
            old_y,
            ("a", "b"),
            new_x,
            new_y,
            ("c", "d"),
            seed=1,
        )


@pytest.mark.parametrize("new_count", [2, 5, 10, 20])
def test_formal_resource_caps_hold_for_registered_scales(new_count: int) -> None:
    old_classes = ("old_a", "old_b")
    new_classes = tuple(f"new_{index}" for index in range(new_count))
    old_x, old_y = _support(old_classes, k=1, offset=0, seed=200)
    new_x, new_y = _support(new_classes, k=1, offset=40, seed=300)
    result = fit_d38_strong_b3_quantized(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=7,
        config=D38StrongB3Config("B"),
    )

    assert result.resource_audit["trainable_parameter_cap_pass"] is True
    assert result.resource_audit["adaptation_epoch_cap_pass"] is True
    assert result.resource_audit["optimizer_step_cap_pass"] is True
    assert result.resource_audit["persistent_state_cap_pass"] is True
    assert result.state.persistent_state_bytes < 256 * 1024


@pytest.mark.parametrize("k_shot", [1, 5, 10, 20])
def test_locked_step_budget_is_independent_of_k(k_shot: int) -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_a", "new_b")
    old_x, old_y = _support(old_classes, k=k_shot, offset=0, seed=410)
    new_x, new_y = _support(new_classes, k=k_shot, offset=70, seed=420)
    result = fit_d38_strong_b3_quantized(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=713101,
        config=D38StrongB3Config("B"),
    )

    assert len(result.training_trace) == 30
    assert result.resource_audit["old_k_shot"] == k_shot
    assert result.resource_audit["new_k_shot"] == k_shot
    assert result.resource_audit["optimizer_steps"] == 30
