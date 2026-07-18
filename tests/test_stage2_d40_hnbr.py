from __future__ import annotations

import numpy as np
import pytest

import cvsrffi.stage2_d40_hnbr as d40
from cvsrffi.stage2_d38_strong_b3_quantized import (
    decode_d38_state_weights,
)
from cvsrffi.stage2_d40_hnbr import (
    D40HNBRConfig,
    D40HNBRError,
    fit_d40_hnbr,
    hnbr_residualize_directions,
    old_prefix_bitwise_unchanged_d40,
    pairwise_support_diagnostics_d40,
    predict_d40_hnbr,
    score_d40_hnbr,
)


def _support(
    class_names: tuple[str, ...], *, k: int, offset: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(class_names):
        center = np.zeros(288, dtype=np.float32)
        center[(offset + class_index * 11) % 288] = 1.0
        center[(offset + class_index * 11 + 41) % 288] = 0.4
        for _ in range(k):
            rows.append(
                (center + rng.normal(0.0, 0.01, 288)).astype(np.float32)
            )
            labels.append(label)
    return np.stack(rows).astype(np.float32), np.asarray(labels)


def _fit(*, k: int = 3, new_count: int = 2, seed: int = 713101):
    old_classes = ("old_a", "old_b")
    new_classes = tuple(f"new_{index}" for index in range(new_count))
    old_x, old_y = _support(old_classes, k=k, offset=0, seed=seed + 1)
    new_x, new_y = _support(new_classes, k=k, offset=90, seed=seed + 2)
    result = fit_d40_hnbr(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=seed,
        config=D40HNBRConfig(),
    )
    return result, old_x, old_y, new_x, new_y


def _reference_hnbr(base: np.ndarray, frozen: np.ndarray | None = None) -> np.ndarray:
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    if frozen is not None:
        frozen = frozen / np.linalg.norm(frozen, axis=1, keepdims=True)
    output = []
    for index, direction in enumerate(base):
        other = np.delete(base, index, axis=0)
        negatives = other if frozen is None else np.concatenate([frozen, other])
        logits = 18.0 * (negatives @ direction)
        weights = np.exp(logits - logits.max())
        weights = weights / weights.sum()
        negative = weights @ negatives
        negative = negative / np.linalg.norm(negative)
        rho = max(0.0, float(direction @ negative))
        residual = direction - rho * negative
        output.append(residual / np.linalg.norm(residual))
    return np.stack(output).astype(np.float32)


def test_hnbr_formula_matches_independent_golden() -> None:
    rng = np.random.default_rng(40)
    base = rng.normal(size=(4, 288)).astype(np.float32)
    frozen = rng.normal(size=(3, 288)).astype(np.float32)

    actual = hnbr_residualize_directions(
        base, frozen_negative_directions=frozen
    )
    expected = _reference_hnbr(base, frozen)

    assert np.allclose(actual, expected, atol=2e-6)
    assert actual.flags.writeable is False
    assert np.allclose(np.linalg.norm(actual, axis=1), 1.0, atol=1e-6)


def test_softmax_is_stable_for_large_near_tied_logits() -> None:
    base = np.zeros((3, 288), dtype=np.float32)
    base[:, 0] = 1.0
    base[0, 1] = 0.2
    base[1, 2] = 0.2
    base[2, 3] = 0.2

    result = hnbr_residualize_directions(base)

    assert np.isfinite(result).all()
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)


def test_near_zero_negative_centroid_and_residual_fail_closed() -> None:
    opposite = np.zeros((2, 288), dtype=np.float32)
    opposite[0, 0] = 1.0
    opposite[1, 0] = -1.0
    base = np.zeros((1, 288), dtype=np.float32)
    base[0, 1] = 1.0
    with pytest.raises(D40HNBRError, match="negative centroid"):
        hnbr_residualize_directions(
            base, frozen_negative_directions=opposite
        )

    duplicate = np.zeros((2, 288), dtype=np.float32)
    duplicate[:, 0] = 1.0
    with pytest.raises(D40HNBRError, match="residual direction"):
        hnbr_residualize_directions(duplicate)


def test_new_hnbr_is_synchronous_and_order_equivariant() -> None:
    rng = np.random.default_rng(41)
    old_final = rng.normal(size=(3, 288)).astype(np.float32)
    new_base = rng.normal(size=(5, 288)).astype(np.float32)
    permutation = np.asarray([3, 0, 4, 1, 2])

    old_first = hnbr_residualize_directions(old_final)
    old_permutation = np.asarray([2, 0, 1])
    old_second = hnbr_residualize_directions(old_final[old_permutation])

    first = hnbr_residualize_directions(
        new_base, frozen_negative_directions=old_final
    )
    second = hnbr_residualize_directions(
        new_base[permutation], frozen_negative_directions=old_final
    )

    assert np.allclose(old_second, old_first[old_permutation], atol=2e-6)
    assert np.allclose(second, first[permutation], atol=2e-6)


def test_fit_is_stage2b20_then_zero_step_hnbr_and_formal_is_int8_only() -> None:
    result, _, _, _, _ = _fit()

    assert len(result.training_trace) == 20
    assert all(
        row["phase"] == "stage2b_fullbatch_old_adaptation"
        for row in result.training_trace
    )
    assert result.resource_audit["adaptation_epochs"] == 20
    assert result.resource_audit["optimizer_steps"] == 20
    assert result.resource_audit["stage2c_optimizer_steps"] == 0
    assert result.geometry_audit["old_hnbr_synchronous"] is True
    assert result.geometry_audit["new_hnbr_synchronous"] is True
    assert result.geometry_audit[
        "new_hnbr_uses_residualized_new_direction_as_negative"
    ] is False
    assert result.before_state.is_int8 and result.state.is_int8
    assert result.state.base_state.fp32_weights.shape == (0, 288)
    assert result.matched_fp32_state.is_int8 is False
    assert result.resource_audit["resident_fp32_target_prototype_count"] == 0
    assert result.resource_audit["formal_state_int8_only"] is True


def test_current_six_old_class_scale_hits_locked_parameter_ceiling() -> None:
    old_classes = tuple(f"old_{index}" for index in range(6))
    new_classes = tuple(f"new_{index}" for index in range(5))
    old_x, old_y = _support(old_classes, k=1, offset=0, seed=905)
    new_x, new_y = _support(new_classes, k=1, offset=120, seed=906)

    result = fit_d40_hnbr(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=713101,
    )

    assert result.resource_audit["trainable_parameters"] == 2016
    assert result.resource_audit["trainable_parameter_cap"] == 2016
    assert result.resource_audit["trainable_parameter_cap_pass"] is True


def test_old_hnbr_before_is_compiled_and_prefix_is_bitwise_frozen() -> None:
    result, _, _, _, _ = _fit()
    before = result.before_state.base_state
    after = result.state.base_state

    assert old_prefix_bitwise_unchanged_d40(result.before_state, result.state)
    assert np.array_equal(before.log_diag_fp32, after.log_diag_fp32)
    assert np.array_equal(before.code1_qint8, after.code1_qint8[:2])
    assert np.array_equal(before.code2_qint8, after.code2_qint8[:2])
    assert np.array_equal(before.scale1_fp16, after.scale1_fp16[:2])
    assert np.array_equal(before.scale2_fp16, after.scale2_fp16[:2])
    assert np.array_equal(before.inverse_norm_fp16, after.inverse_norm_fp16[:2])


def test_new_hnbr_reads_actual_decoded_int8_old_prefix(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray | None]] = []
    original = d40.hnbr_residualize_directions

    def spy(base_directions, *, frozen_negative_directions=None):
        calls.append(
            (
                np.array(base_directions, copy=True),
                None
                if frozen_negative_directions is None
                else np.array(frozen_negative_directions, copy=True),
            )
        )
        return original(
            base_directions,
            frozen_negative_directions=frozen_negative_directions,
        )

    monkeypatch.setattr(d40, "hnbr_residualize_directions", spy)
    result, _, _, _, _ = _fit()
    decoded_before = decode_d38_state_weights(result.before_state.base_state)

    assert len(calls) == 2
    assert calls[0][1] is None
    assert np.array_equal(calls[1][1], decoded_before)
    assert result.geometry_audit["new_hnbr_old_negative_precision"] == "int8_decoded"
    assert result.geometry_audit[
        "new_hnbr_old_negative_matches_before_int8_decode"
    ] is True
    assert result.geometry_audit[
        "new_hnbr_old_negative_source_sha256"
    ] == result.geometry_audit["before_int8_decoded_old_direction_sha256"]
    assert result.geometry_audit[
        "old_fp32_reference_used_as_new_hnbr_negative"
    ] is False


def test_different_fp32_old_ablation_cannot_change_new_reference(monkeypatch) -> None:
    baseline, _, _, _, _ = _fit(seed=880)
    original_compile = d40.compile_d38_state

    def compile_with_different_fp32_old(
        classes,
        old_class_count,
        log_diag_fp32,
        weights_fp32,
        *,
        arm,
        precision,
    ):
        weights = np.asarray(weights_fp32, dtype=np.float32)
        if str(precision).lower() == "fp32" and len(classes) == old_class_count:
            replacement = np.zeros_like(weights)
            for index in range(len(replacement)):
                replacement[index, 200 + index] = 1.0
            weights = replacement
        return original_compile(
            classes,
            old_class_count,
            log_diag_fp32,
            weights,
            arm=arm,
            precision=precision,
        )

    monkeypatch.setattr(d40, "compile_d38_state", compile_with_different_fp32_old)
    changed, _, _, _, _ = _fit(seed=880)
    count = changed.state.old_class_count
    baseline_new = decode_d38_state_weights(
        baseline.matched_fp32_state.base_state
    )[count:]
    changed_new = decode_d38_state_weights(
        changed.matched_fp32_state.base_state
    )[count:]

    assert not np.array_equal(
        decode_d38_state_weights(baseline.matched_fp32_before_state.base_state),
        decode_d38_state_weights(changed.matched_fp32_before_state.base_state),
    )
    assert np.array_equal(baseline_new, changed_new)


def test_int8_and_fp32_states_share_reference_directions() -> None:
    result, old_x, _, new_x, _ = _fit()
    formal = decode_d38_state_weights(result.state.base_state)
    reference = decode_d38_state_weights(result.matched_fp32_state.base_state)
    rows = np.concatenate([old_x, new_x]).astype(np.float32)

    assert formal.shape == reference.shape == (4, 288)
    assert np.max(np.abs(formal - reference)) < 2e-4
    assert score_d40_hnbr(result.state, rows).shape == (12, 4)
    assert result.geometry_audit["matched_fp32_shared_reference_directions"] is True


def test_scoring_is_row_local_split_and_order_invariant() -> None:
    result, old_x, _, new_x, _ = _fit()
    rows = np.concatenate([old_x, new_x]).astype(np.float32)
    whole = score_d40_hnbr(result.state, rows)
    split = np.concatenate(
        [
            score_d40_hnbr(result.state, rows[:5]),
            score_d40_hnbr(result.state, rows[5:]),
        ]
    )
    permutation = np.asarray([9, 0, 5, 2, 11, 4, 1, 10, 3, 8, 7, 6])

    assert np.array_equal(whole, split)
    assert np.array_equal(score_d40_hnbr(result.state, rows[permutation]), whole[permutation])
    assert np.array_equal(
        predict_d40_hnbr(result.state, rows),
        np.asarray(result.state.classes)[np.argmax(whole, axis=1)],
    )


def test_label_renaming_keeps_numeric_geometry() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_a", "new_b")
    old_x, old_y = _support(old_classes, k=2, offset=0, seed=71)
    new_x, new_y = _support(new_classes, k=2, offset=90, seed=72)
    first = fit_d40_hnbr(
        old_x, old_y, old_classes, new_x, new_y, new_classes, seed=77
    )
    renamed_old = np.asarray([{"old_a": "x", "old_b": "y"}[v] for v in old_y])
    renamed_new = np.asarray([{"new_a": "u", "new_b": "v"}[v] for v in new_y])
    second = fit_d40_hnbr(
        old_x,
        renamed_old,
        ("x", "y"),
        new_x,
        renamed_new,
        ("u", "v"),
        seed=77,
    )
    rows = np.concatenate([old_x, new_x]).astype(np.float32)

    assert np.array_equal(
        score_d40_hnbr(first.state, rows),
        score_d40_hnbr(second.state, rows),
    )


@pytest.mark.parametrize("k_shot", [1, 5, 10, 20])
def test_k_shot_uses_same_zero_step_rule(k_shot: int) -> None:
    result, _, _, _, _ = _fit(k=k_shot)

    assert len(result.training_trace) == 20
    assert result.resource_audit["old_k_shot"] == k_shot
    assert result.resource_audit["new_k_shot"] == k_shot
    assert result.resource_audit["stage2c_optimizer_steps"] == 0


@pytest.mark.parametrize("new_count", [2, 5, 10, 20])
def test_registered_new_scales_pass_state_and_resource_caps(new_count: int) -> None:
    result, _, _, _, _ = _fit(k=1, new_count=new_count)

    assert len(result.state.classes) == 2 + new_count
    assert result.state.persistent_state_bytes < 256 * 1024
    assert result.resource_audit["persistent_state_cap_pass"] is True
    assert result.resource_audit["trainable_parameter_cap_pass"] is True
    assert result.resource_audit["adaptation_epoch_cap_pass"] is True
    assert result.resource_audit["optimizer_step_cap_pass"] is True
    assert result.resource_audit["estimated_hnbr_support_macs"] > 0


def test_pairwise_diagnostic_records_new_new_and_new_old_margins() -> None:
    result, _, _, new_x, new_y = _fit()
    rows = pairwise_support_diagnostics_d40(
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


def test_invalid_mechanism_lock_and_nonfinite_support_fail_closed() -> None:
    with pytest.raises(D40HNBRError):
        D40HNBRConfig(temperature=17.0)
    old_x, old_y = _support(("a", "b"), k=1, offset=0, seed=81)
    new_x, new_y = _support(("c", "d"), k=1, offset=80, seed=82)
    new_x[0, 0] = np.nan

    with pytest.raises(D40HNBRError):
        fit_d40_hnbr(old_x, old_y, ("a", "b"), new_x, new_y, ("c", "d"), seed=1)
