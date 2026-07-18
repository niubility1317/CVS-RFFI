from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cvsrffi.stage2_d38_strong_b3_quantized import decode_d38_state_weights
from cvsrffi.stage2_d41_bec import (
    D41BECConfig,
    D41BECError,
    VIEW_NAMES,
    d41_bec_loss,
    fit_d41_bec,
    make_d41_views,
    pairwise_support_diagnostics_d41,
    predict_d41_bec,
    score_d41_bec,
)


def _support(
    class_names: tuple[str, ...], *, k: int, offset: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(class_names):
        center = np.zeros(288, dtype=np.float32)
        center[(offset + 17 * class_index) % 160] = 1.0
        center[160 + (offset + 13 * class_index) % 96] = 0.5
        center[256 + (offset + 7 * class_index) % 32] = 0.25
        for _ in range(k):
            row = center + rng.normal(0.0, 0.01, 288).astype(np.float32)
            row = row / np.linalg.norm(row)
            rows.append(row.astype(np.float32))
            labels.append(label)
    return np.stack(rows).astype(np.float32), np.asarray(labels)


def _fit(*, k: int = 2, new_count: int = 2, seed: int = 713101):
    old_classes = ("old_a", "old_b")
    new_classes = tuple(f"new_{index}" for index in range(new_count))
    old_x, old_y = _support(old_classes, k=k, offset=1, seed=seed + 1)
    new_x, new_y = _support(new_classes, k=k, offset=67, seed=seed + 2)
    result = fit_d41_bec(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=seed,
        config=D41BECConfig(),
    )
    return result, old_x, old_y, new_x, new_y


def _state_bytes(state) -> tuple[bytes, ...]:
    base = state.base_state
    return tuple(
        np.ascontiguousarray(value).tobytes()
        for value in (
            base.log_diag_fp32,
            base.code1_qint8,
            base.code2_qint8,
            base.scale1_fp16,
            base.scale2_fp16,
            base.inverse_norm_fp16,
            base.fp32_weights,
        )
    )


def test_make_views_matches_block_erasure_golden_and_preserves_full() -> None:
    rows, _ = _support(("a", "b"), k=1, offset=3, seed=40)
    views = make_d41_views(rows)

    assert tuple(views) == VIEW_NAMES
    assert np.array_equal(views["full"], rows)
    assert np.count_nonzero(views["minus_z"][:, :160]) == 0
    assert np.count_nonzero(views["minus_fft"][:, 160:256]) == 0
    assert np.count_nonzero(views["minus_rf"][:, 256:288]) == 0
    for value in views.values():
        assert value.flags.writeable is False
        assert np.allclose(np.linalg.norm(value, axis=1), 1.0, atol=1e-6)


def test_masked_near_zero_view_fails_closed() -> None:
    rows = np.zeros((1, 288), dtype=np.float32)
    rows[0, :160] = 1.0
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)

    with pytest.raises(D41BECError, match="near-zero"):
        make_d41_views(rows)


def test_bec_loss_matches_independent_macro_ce_and_js_golden() -> None:
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    logits = (
        torch.tensor([[3.0, -1.0], [2.0, 0.0], [-1.0, 3.0], [0.0, 2.0]]),
        torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]]),
        torch.tensor([[2.5, -0.5], [1.5, 0.0], [-0.5, 2.5], [0.0, 1.5]]),
        torch.tensor([[1.5, 0.0], [1.0, 0.5], [0.0, 1.5], [0.5, 1.0]]),
    )
    total, macro_ce, mean_js = d41_bec_loss(logits, targets, 2)
    ce_terms = []
    for value in logits:
        sample = F.cross_entropy(value, targets, reduction="none")
        ce_terms.append(torch.stack([sample[targets == c].mean() for c in range(2)]).mean())
    expected_ce = torch.stack(ce_terms).mean()
    full_prob = torch.softmax(logits[0], dim=1)
    expected_js = []
    for value in logits[1:]:
        masked_prob = torch.softmax(value, dim=1)
        mixture = 0.5 * (full_prob + masked_prob)
        js = 0.5 * torch.sum(full_prob * (torch.log(full_prob) - torch.log(mixture)), dim=1)
        js += 0.5 * torch.sum(masked_prob * (torch.log(masked_prob) - torch.log(mixture)), dim=1)
        expected_js.append(js.mean())
    expected_js_mean = torch.stack(expected_js).mean()

    assert torch.allclose(macro_ce, expected_ce, atol=1e-7)
    assert torch.allclose(mean_js, expected_js_mean, atol=1e-7)
    assert torch.allclose(total, expected_ce + expected_js_mean, atol=1e-7)


def test_logaddexp_js_is_finite_for_extreme_logits() -> None:
    targets = torch.tensor([0, 1], dtype=torch.long)
    full = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    masked = torch.tensor([[-1000.0, 1000.0], [1000.0, -1000.0]])

    total, macro_ce, mean_js = d41_bec_loss(
        (full, masked, full.clone(), masked.clone()), targets, 2
    )

    assert torch.isfinite(total)
    assert torch.isfinite(macro_ce)
    assert torch.isfinite(mean_js)


def test_fit_has_b20_c10_trace_and_joint_c_updates() -> None:
    result, _, _, _, _ = _fit()

    assert len(result.training_trace) == 30
    assert [row["phase"] for row in result.training_trace[:20]] == [
        "stage2b_four_view_bec_old"
    ] * 20
    assert [row["phase"] for row in result.training_trace[20:]] == [
        "stage2c_four_view_bec_all_registry"
    ] * 10
    assert [row["optimizer_step"] for row in result.training_trace] == list(range(1, 31))
    assert result.geometry_audit["c_logdiag_update_norm"] > 0
    assert result.geometry_audit["c_old_weight_update_norm"] > 0
    assert result.geometry_audit["c_new_weight_update_norm"] > 0
    assert result.geometry_audit["stage2c_old_weights_trainable"] is True
    assert result.geometry_audit["stage2c_new_weights_trainable"] is True
    assert result.resource_audit["stage2c_optimizer_steps"] == 10


def test_before_state_is_independent_and_immutable_during_c() -> None:
    result, _, _, _, _ = _fit()
    formal_snapshot = _state_bytes(result.before_state)
    fp32_snapshot = _state_bytes(result.matched_fp32_before_state)

    assert result.before_state.classes == ("old_a", "old_b")
    assert result.state.classes == ("old_a", "old_b", "new_0", "new_1")
    assert _state_bytes(result.before_state) == formal_snapshot
    assert _state_bytes(result.matched_fp32_before_state) == fp32_snapshot
    assert result.geometry_audit["before_state_immutable_during_stage2c"] is True
    assert not np.array_equal(
        result.before_state.base_state.log_diag_fp32,
        result.state.base_state.log_diag_fp32,
    )


def test_formal_states_are_int8_only_and_fp32_is_matched_ablation() -> None:
    result, old_x, _, new_x, _ = _fit()
    rows = np.concatenate([old_x, new_x]).astype(np.float32)

    assert result.before_state.is_int8 and result.state.is_int8
    assert result.state.base_state.fp32_weights.shape == (0, 288)
    assert result.matched_fp32_before_state.is_int8 is False
    assert result.matched_fp32_state.is_int8 is False
    assert result.matched_fp32_state.base_state.fp32_weights.shape == (4, 288)
    assert result.resource_audit["resident_fp32_target_prototype_count"] == 0
    assert score_d41_bec(result.state, rows).shape == (len(rows), 4)
    assert np.max(
        np.abs(
            decode_d38_state_weights(result.state.base_state)
            - decode_d38_state_weights(result.matched_fp32_state.base_state)
        )
    ) < 2e-4


def test_scoring_uses_full_view_and_is_row_local_order_invariant() -> None:
    result, old_x, _, new_x, _ = _fit()
    rows = np.concatenate([old_x, new_x]).astype(np.float32)
    whole = score_d41_bec(result.state, rows)
    split = np.concatenate(
        [score_d41_bec(result.state, rows[:3]), score_d41_bec(result.state, rows[3:])]
    )
    permutation = np.asarray([5, 0, 7, 2, 1, 6, 3, 4])

    assert np.array_equal(whole, split)
    assert np.array_equal(score_d41_bec(result.state, rows[permutation]), whole[permutation])
    assert np.array_equal(
        predict_d41_bec(result.state, rows),
        np.asarray(result.state.classes)[np.argmax(whole, axis=1)],
    )
    assert result.geometry_audit["query_view"] == "full_only"
    assert result.resource_audit["query_view_count"] == 1


def test_label_renaming_and_registry_permutation_are_equivariant() -> None:
    old_x, old_y = _support(("old_a", "old_b"), k=1, offset=1, seed=60)
    new_x, new_y = _support(("new_a", "new_b"), k=1, offset=67, seed=61)
    first = fit_d41_bec(
        old_x, old_y, ("old_a", "old_b"), new_x, new_y, ("new_a", "new_b"), seed=9
    )
    handle_map = {"old_a": "x", "old_b": "y", "new_a": "u", "new_b": "v"}
    renamed_old = np.asarray([handle_map[v] for v in old_y])
    renamed_new = np.asarray([handle_map[v] for v in new_y])
    second = fit_d41_bec(
        old_x, renamed_old, ("y", "x"), new_x, renamed_new, ("v", "u"), seed=9
    )
    rows = np.concatenate([old_x, new_x]).astype(np.float32)
    first_scores = score_d41_bec(first.state, rows)
    second_scores = score_d41_bec(second.state, rows)
    inverse_permutation = [
        second.state.classes.index(handle_map[class_id])
        for class_id in first.state.classes
    ]
    aligned_second_scores = second_scores[:, inverse_permutation]
    expected_second_predictions = np.asarray(
        [handle_map[value] for value in predict_d41_bec(first.state, rows)]
    )

    assert second.state.classes == ("y", "x", "v", "u")
    assert np.allclose(
        first_scores, aligned_second_scores, rtol=0.0, atol=1.0e-6
    )
    assert np.array_equal(
        predict_d41_bec(second.state, rows), expected_second_predictions
    )


@pytest.mark.parametrize("k_shot", [1, 5, 10, 20])
def test_k_shot_keeps_locked_b20_c10(k_shot: int) -> None:
    result, _, _, _, _ = _fit(k=k_shot)

    assert len(result.training_trace) == 30
    assert result.resource_audit["old_k_shot"] == k_shot
    assert result.resource_audit["new_k_shot"] == k_shot
    assert result.resource_audit["stage2c_optimizer_steps"] == 10


@pytest.mark.parametrize("new_count", [2, 5, 10, 20])
def test_new_count_scales_pass_formal_resource_caps(new_count: int) -> None:
    result, _, _, _, _ = _fit(k=1, new_count=new_count)

    assert len(result.state.classes) == 2 + new_count
    assert result.state.persistent_state_bytes < 256 * 1024
    assert result.resource_audit["persistent_state_cap_pass"] is True
    assert result.resource_audit["trainable_parameter_cap_pass"] is True
    assert result.resource_audit[
        "four_view_plus_js_exceeds_single_view_lower_bound"
    ] is True
    assert result.resource_audit["estimated_adaptation_macs"] > result.resource_audit[
        "estimated_single_view_adaptation_macs_lower_bound"
    ]


def test_current_old6_new5_peak_parameter_count_is_3456() -> None:
    old_classes = tuple(f"old_{index}" for index in range(6))
    new_classes = tuple(f"new_{index}" for index in range(5))
    old_x, old_y = _support(old_classes, k=1, offset=1, seed=70)
    new_x, new_y = _support(new_classes, k=1, offset=67, seed=71)

    result = fit_d41_bec(
        old_x, old_y, old_classes, new_x, new_y, new_classes, seed=713101
    )

    assert result.resource_audit["stage2b_trainable_parameters"] == 2016
    assert result.resource_audit["stage2c_trainable_parameters"] == 3456
    assert result.resource_audit["trainable_parameters"] == 3456


def test_pairwise_diagnostic_records_new_new_and_new_old_margins() -> None:
    result, _, _, new_x, new_y = _fit()
    rows = pairwise_support_diagnostics_d41(
        result.state,
        new_x,
        new_y,
        [f"physical-{index}" for index in range(len(new_x))],
        scenario="leo_clear_weak",
        outer_fold=2,
        physical_ranks=[0, 1, 0, 1],
    )

    assert len(rows) == len(new_x)
    assert all(row["query_rows_used"] == 0 for row in rows)
    assert all(row["true_new_handle"] != row["top_competing_new_handle"] for row in rows)
    assert all(np.isfinite(row["new_new_margin"]) for row in rows)
    assert all(np.isfinite(row["new_old_margin"]) for row in rows)


def test_locked_mechanism_has_no_forbidden_components_or_ground_update() -> None:
    result, _, _, _, _ = _fit()

    for key in (
        "feature_noise_used",
        "prototype_anchor_used",
        "worst_class_surrogate_used",
        "new_anchor_used",
        "hnbr_used",
        "bias_used",
        "radius_used",
        "ground_int8_update_access",
    ):
        assert result.geometry_audit[key] is False
    assert result.geometry_audit["ground_int8_component_input_count"] == 0


def test_nonfinite_support_and_config_drift_fail_closed() -> None:
    with pytest.raises(D41BECError):
        D41BECConfig(stage2c_steps=9)
    old_x, old_y = _support(("a", "b"), k=1, offset=1, seed=80)
    new_x, new_y = _support(("c", "d"), k=1, offset=67, seed=81)
    new_x[0, 0] = np.nan

    with pytest.raises(D41BECError):
        fit_d41_bec(old_x, old_y, ("a", "b"), new_x, new_y, ("c", "d"), seed=1)
