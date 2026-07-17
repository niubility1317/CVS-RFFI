from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from cvsrffi.stage2_inloop_safe_cap_suffix import (
    BIAS_RECOVERY_TARGET,
    D32_BIAS_RECOVERY_CAP,
    D32_GROUP_BALANCED_CAP,
    D32_NEW_CVAR_CAP,
    D32Stage2CConfig,
    MAX_PEAK_TRAINABLE_PARAMETERS,
    append_stage2c_inloop_safe_cap_suffix,
    predict_all_registered,
    score_all_registered,
)
from cvsrffi.stage2_multimodal_compact_diag import (
    D26CompactDiagConfig,
    fit_stage2b_compact_diag,
    score_all_registered as score_d26,
)


DIM = 288


def _support(
    classes: tuple[str, ...],
    k: int,
    *,
    seed: int,
    starts: tuple[int, ...] | None = None,
    noise: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    offsets = starts or tuple(range(len(classes)))
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(classes):
        center = np.zeros(DIM, dtype=np.float32)
        center[offsets[class_index]] = 1.0
        center[(offsets[class_index] + 31) % DIM] = 0.25
        for _ in range(k):
            rows.append(center + rng.normal(0.0, noise, size=DIM).astype(np.float32))
            labels.append(label)
    return np.stack(rows).astype(np.float32), np.asarray(labels)


def _base(
    old_classes: tuple[str, ...] = ("o0", "o1", "o2"),
    k: int = 5,
) -> tuple[object, np.ndarray, np.ndarray]:
    old_rows, old_labels = _support(old_classes, k, seed=11)
    fit = fit_stage2b_compact_diag(
        old_rows,
        old_labels,
        old_classes,
        config=D26CompactDiagConfig(stage2b_steps=2, stage2c_steps=0),
    )
    return fit.state, old_rows, old_labels


def _append(
    method: str = D32_GROUP_BALANCED_CAP,
    *,
    new_count: int = 5,
    new_k: int = 5,
):
    base, old_rows, old_labels = _base()
    new_classes = tuple(f"n{index}" for index in range(new_count))
    # First new class deliberately conflicts with o0 and exercises the cap.
    starts = (0,) + tuple(range(7, 7 + new_count - 1))
    new_rows, new_labels = _support(
        new_classes, new_k, seed=29, starts=starts, noise=0.04
    )
    result = append_stage2c_inloop_safe_cap_suffix(
        base,
        new_rows,
        new_labels,
        new_classes,
        old_rows,
        old_labels,
        config=D32Stage2CConfig(method),
    )
    return base, old_rows, old_labels, new_rows, new_labels, result


def test_inloop_cap_freezes_old_prefix_and_preserves_pre_correct_old_rows() -> None:
    base, old_rows, old_labels, _, _, result = _append(D32_GROUP_BALANCED_CAP)
    state = result.state
    assert state.log_diag.tobytes() == base.log_diag.tobytes()
    assert state.weights[: len(base.classes)].tobytes() == base.weights.tobytes()
    assert score_all_registered(state, old_rows)[:, : len(base.classes)].tobytes() == (
        score_d26(base, old_rows).tobytes()
    )
    before = np.argmax(score_d26(base, old_rows), axis=1)
    truth = np.asarray([base.classes.index(str(label)) for label in old_labels])
    after = np.argmax(score_all_registered(state, old_rows), axis=1)
    assert np.all((before != truth) | (after == truth))
    gate = json.loads(state.support_gate_json)
    assert gate["support_only_checkpoint_gate_pass"] is True
    assert gate["pre_registration_correct_old_rows_preserved"] is True
    assert gate["per_old_class_non_degradation"] is True
    assert gate["inloop_and_deployment_biases_identical"] is True
    assert np.asarray(gate["new_class_biases"], dtype=np.float32).tobytes() == (
        state.new_class_biases.tobytes()
    )


@pytest.mark.parametrize("new_count", [2, 5, 20])
def test_two_five_twenty_new_classes_have_complete_cap_trace(new_count: int) -> None:
    _, _, _, _, _, result = _append(new_count=new_count, new_k=2)
    assert len(result.loss_trace) == 11
    assert [row["step"] for row in result.loss_trace] == list(range(11))
    assert sum(bool(row["selected_checkpoint"]) for row in result.loss_trace) == 1
    for row in result.loss_trace:
        assert 0.0 <= row["precap_old_support_accuracy"] <= 1.0
        assert 0.0 <= row["precap_new_support_accuracy"] <= 1.0
        assert 0.0 <= row["postcap_old_support_accuracy"] <= 1.0
        assert 0.0 <= row["postcap_new_support_accuracy"] <= 1.0
        assert row["bias_min"] <= row["bias_mean"] <= row["bias_max"] <= 0.0
        assert isinstance(row["cap_active_old_class_histogram"], dict)
        assert row["all_registered_support_rows_in_loss"] > 0
        assert row["old_weight_update_count"] == 0
        assert row["shared_diagonal_update_count"] == 0
        assert row["rollback_applied"] is (not row["candidate_safety_pass"])


def test_training_selected_checkpoint_and_deployment_use_identical_score_surface() -> None:
    _, old_rows, _, new_rows, _, result = _append(D32_NEW_CVAR_CAP)
    state = result.state
    selected = result.loss_trace[state.selected_checkpoint_step]
    support = np.concatenate((old_rows, new_rows), axis=0)
    predictions = np.argmax(score_all_registered(state, support), axis=1)
    old_truth = np.repeat(np.arange(state.old_class_count), 5)
    new_truth = np.repeat(np.arange(5) + state.old_class_count, 5)
    truth = np.concatenate((old_truth, new_truth))
    old_acc = float(np.mean(predictions[: len(old_rows)] == truth[: len(old_rows)]))
    new_acc = float(np.mean(predictions[len(old_rows) :] == truth[len(old_rows) :]))
    assert old_acc == pytest.approx(selected["postcap_old_support_accuracy"], abs=1e-7)
    assert new_acc == pytest.approx(selected["postcap_new_support_accuracy"], abs=1e-7)
    audit = state.resource_audit()
    assert audit["training_and_deployment_score_surface_identical"] is True
    assert audit["in_loop_bias_applied_at_step0"] is True
    assert audit["safety_cap_recomputed_each_optimizer_step"] is True
    assert audit["safety_bias_trainable_parameters"] == 0


def test_method_locks_are_group_balanced_cvar_and_finite_bias_recovery() -> None:
    _, _, _, _, _, plain = _append(D32_GROUP_BALANCED_CAP)
    _, _, _, _, _, cvar = _append(D32_NEW_CVAR_CAP)
    _, _, _, _, _, recovery = _append(D32_BIAS_RECOVERY_CAP)
    assert len(plain.loss_trace) == 11
    assert len(cvar.loss_trace) == 11
    assert len(recovery.loss_trace) == 16
    assert plain.state.config.new_cvar_weight == 0.0
    assert cvar.state.config.new_cvar_weight == 0.35
    assert recovery.state.config.new_cvar_weight == 0.35
    assert recovery.state.config.bias_recovery_weight == 0.15
    assert plain.state.config.learning_rate == 0.03
    assert cvar.state.config.learning_rate == 0.03
    assert recovery.state.config.learning_rate == 0.025
    assert plain.state.config.centroid_anchor_weight == 0.02
    assert cvar.state.config.centroid_anchor_weight == 0.02
    assert recovery.state.config.centroid_anchor_weight == 0.03
    assert plain.state.config.safety_delta == 1.0e-4
    assert cvar.state.config.safety_delta == 0.10
    assert recovery.state.config.safety_delta == 0.10
    assert recovery.state.config.optimizer_steps == 15
    for row in recovery.loss_trace:
        assert row["bias_recovery_target"] == BIAS_RECOVERY_TARGET
        assert row["bias_recovery_loss"] >= 0.0
        assert row["centroid_anchor_loss"] >= 0.0
        assert row["centroid_anchor_weight"] == 0.03
        assert row["inloop_safety_delta"] == 0.10
        assert row["new_class_cvar_loss"] >= 0.0
        assert row["group_balanced_old_new_ce"] == pytest.approx(
            0.5 * (row["old_group_ce"] + row["new_group_ce"]), rel=1e-6
        )
    plain_lock = plain.state.resource_audit()["method_lock"]
    cvar_lock = cvar.state.resource_audit()["method_lock"]
    recovery_lock = recovery.state.resource_audit()["method_lock"]
    assert plain_lock["learning_rate"] == 0.03
    assert plain_lock["inloop_safety_delta"] == 1.0e-4
    assert cvar_lock["inloop_safety_delta"] == 0.10
    assert recovery_lock["bias_recovery_normalization"] == 4.0
    assert recovery_lock["centroid_anchor_weight"] == 0.03
    selected = recovery.loss_trace[recovery.state.selected_checkpoint_step]
    biases = recovery.state.new_class_biases.astype(np.float64)
    expected_recovery = float(np.mean((np.maximum(-biases - 4.0, 0.0) / 4.0) ** 2))
    assert selected["bias_recovery_loss"] == pytest.approx(expected_recovery, rel=1e-5)


def test_k1_centroid_plus_cap_has_zero_updates_and_valid_predictions() -> None:
    base, _, _, new_rows, _, result = _append(new_count=2, new_k=1)
    assert result.state.stage2c_optimizer_steps == 0
    assert result.state.selected_checkpoint_step == 0
    assert result.state.rollback_count == 0
    assert len(result.loss_trace) == 1
    assert result.loss_trace[0]["selected_checkpoint"] is True
    assert result.loss_trace[0]["gradient_norm"] == 0.0
    gate = json.loads(result.state.support_gate_json)
    assert gate["k1_centroid_cap_zero_update_bypass"] is True
    assert len(predict_all_registered(result.state, new_rows)) == len(new_rows)
    assert result.state.weights[: len(base.classes)].tobytes() == base.weights.tobytes()


def test_twenty_new_resource_step_mac_state_and_protocol_caps() -> None:
    old_classes = tuple(f"o{index}" for index in range(6))
    base, old_rows, old_labels = _base(old_classes=old_classes, k=2)
    new_classes = tuple(f"n{index:02d}" for index in range(20))
    new_rows, new_labels = _support(
        new_classes, 2, seed=101, starts=tuple(range(20, 40)), noise=0.02
    )
    result = append_stage2c_inloop_safe_cap_suffix(
        base,
        new_rows,
        new_labels,
        new_classes,
        old_rows,
        old_labels,
        config=D32Stage2CConfig(D32_BIAS_RECOVERY_CAP),
    )
    audit = result.state.resource_audit()
    assert audit["peak_trainable_parameters"] == MAX_PEAK_TRAINABLE_PARAMETERS
    assert audit["stage2c_active_trainable_parameters_per_step_max"] == 7 * DIM
    assert audit["stage2c_total_new_weight_state_scalars"] == 20 * DIM
    assert audit["stage2c_optimizer_steps"] == 15
    assert audit["formal_total_optimizer_step_cap_pass"] is True
    assert audit["estimated_adaptation_macs"] == (
        audit["estimated_stage2b_adaptation_macs"]
        + audit["estimated_stage2c_adaptation_macs"]
    )
    assert audit["inloop_cap_reuses_registered_score_matrix"] is True
    assert audit["persistent_state_cap_pass"] is True
    assert audit["support_gate_external_evidence_excluded_from_deployment_state"] is True
    assert audit["dense_query_graph_bytes"] == 0
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["query_labels_used_for_fit"] is False
    assert audit["query_role_oracle_access"] is False
    assert audit["query_class_quota_access"] is False
    assert audit["query_batch_global_assignment"] is False
    assert audit["clean_sample_access"] is False
    assert audit["source_sample_access"] is False
    names = set(inspect.signature(append_stage2c_inloop_safe_cap_suffix).parameters)
    assert not any(
        forbidden in name
        for name in names
        for forbidden in ("query", "clean", "source", "quota", "role")
    )


def test_row_permutation_is_bitwise_invariant() -> None:
    base, old_rows, old_labels = _base()
    new_classes = ("n0", "n1", "n2", "n3", "n4")
    new_rows, new_labels = _support(
        new_classes, 5, seed=67, starts=(0, 6, 7, 8, 9)
    )
    config = D32Stage2CConfig(D32_BIAS_RECOVERY_CAP)
    first = append_stage2c_inloop_safe_cap_suffix(
        base, new_rows, new_labels, new_classes, old_rows, old_labels, config=config
    )
    old_order = np.asarray([9, 1, 13, 4, 7, 0, 14, 2, 11, 5, 8, 3, 10, 6, 12])
    new_order = np.asarray(
        [24, 0, 6, 12, 18, 3, 9, 15, 21, 1, 7, 13, 19, 4, 10, 16, 22, 2, 8, 14, 20, 5, 11, 17, 23]
    )
    second = append_stage2c_inloop_safe_cap_suffix(
        base,
        new_rows[new_order],
        new_labels[new_order],
        new_classes,
        old_rows[old_order],
        old_labels[old_order],
        config=config,
    )
    assert second.state.weights.tobytes() == first.state.weights.tobytes()
    assert second.state.new_class_biases.tobytes() == first.state.new_class_biases.tobytes()
    assert second.state.support_gate_json == first.state.support_gate_json
    assert second.loss_trace == first.loss_trace
