from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_multimodal_compact_diag import (
    D26CompactDiagConfig,
    D26CompactDiagError,
    MAX_PERSISTENT_STATE_BYTES,
    MAX_TRAINABLE_PARAMETERS,
    append_stage2c_new_suffix,
    fit_stage2b_compact_diag,
    predict_all_registered,
    score_all_registered,
)


def _support(
    classes: tuple[str, ...],
    k_shot: int,
    *,
    seed: int,
    centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = rng.normal(size=(len(classes), 288)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, class_name in enumerate(classes):
        for _ in range(k_shot):
            row = centers[index] + 0.03 * rng.normal(size=288).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row.astype(np.float32))
            labels.append(class_name)
    return np.stack(rows), np.asarray(labels)


def _old_new_collision_support(
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic case where registration bias zero forgets old rows."""

    old_train_centers = np.zeros((2, 288), dtype=np.float32)
    old_train_centers[0, 0] = 1.0
    old_train_centers[1, 1] = 1.0
    collision_centers = np.zeros((2, 288), dtype=np.float32)
    collision_centers[0, 0] = 0.8
    collision_centers[0, 2] = 0.6
    collision_centers[1, 1] = 0.8
    collision_centers[1, 3] = 0.6

    def repeat(centers: np.ndarray, classes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        rows = np.repeat(centers, k_shot, axis=0).astype(np.float32)
        labels = np.repeat(np.asarray(classes), k_shot)
        return rows, labels

    old_x, old_y = repeat(old_train_centers, ("old_a", "old_b"))
    old_guard_x, old_guard_y = repeat(collision_centers, ("old_a", "old_b"))
    new_x, new_y = repeat(collision_centers, ("new_c", "new_d"))
    return old_x, old_y, old_guard_x, old_guard_y, new_x, new_y


def test_stage2b_default_is_tiny_full_batch_b3_compression() -> None:
    classes = ("old_a", "old_b", "old_c")
    x, y = _support(classes, 3, seed=1)
    result = fit_stage2b_compact_diag(x, y, classes)

    assert result.state.stage2b_optimizer_steps == 15
    assert result.state.stage2c_optimizer_steps == 0
    assert result.state.old_class_count == len(classes)
    assert result.state.weights.shape == (3, 288)
    assert result.state.log_diag.shape == (288,)
    assert result.state.log_diag.dtype == np.float32
    assert result.state.weights.dtype == np.float32
    assert not result.state.log_diag.flags.writeable
    assert not result.state.weights.flags.writeable
    assert len(result.loss_trace) == 16
    assert [row["step"] for row in result.loss_trace] == list(range(16))
    assert all(row["phase"] == "stage2b_old_support_full_batch" for row in result.loss_trace)
    assert all(row["runtime_dtype"] == "float32" for row in result.loss_trace)
    assert result.state.trainable_parameters == 288 + 3 * 288
    audit = result.state.resource_audit()
    assert audit["trainable_parameters"] < MAX_TRAINABLE_PARAMETERS
    assert audit["persistent_state_bytes"] < MAX_PERSISTENT_STATE_BYTES
    assert audit["trainable_parameter_cap_pass"] is True
    assert audit["persistent_state_cap_pass"] is True
    assert audit["optimizer_step_cap_pass"] is True
    assert audit["formal_adaptation_epoch_cap_pass"] is True
    assert audit["estimated_query_temporary_bytes"] >= (
        3 * 288 + len(classes) * 288
    ) * 4
    assert audit["dense_query_graph_bytes"] == 0
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["clean_sample_access"] is False
    assert audit["source_sample_access"] is False


def test_stage2c_trains_only_new_suffix_and_freezes_old_raw_scores() -> None:
    old_classes = ("old_a", "old_b", "old_c")
    new_classes = ("new_d", "new_e")
    old_x, old_y = _support(old_classes, 3, seed=2)
    config = D26CompactDiagConfig(stage2b_steps=15, stage2c_steps=10)
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    new_x, new_y = _support(new_classes, 3, seed=3)

    old_raw_before = score_all_registered(before, old_x).copy()
    result = append_stage2c_new_suffix(
        before,
        new_x,
        new_y,
        new_classes,
        old_x,
        old_y,
    )
    after = result.state
    old_raw_after = score_all_registered(after, old_x)[:, : len(old_classes)]

    assert len(result.loss_trace) == 11
    assert after.stage2b_optimizer_steps + after.stage2c_optimizer_steps == 25
    assert after.classes == old_classes + new_classes
    assert after.log_diag.tobytes() == before.log_diag.tobytes()
    assert after.weights[: len(old_classes)].tobytes() == before.weights.tobytes()
    assert after.old_lock_sha256 == before.old_lock_sha256
    assert np.array_equal(old_raw_before, old_raw_after)
    assert all(row["old_weight_update_count"] == 0 for row in result.loss_trace)
    assert all(row["shared_diagonal_update_count"] == 0 for row in result.loss_trace)
    assert after.trainable_parameters < MAX_TRAINABLE_PARAMETERS
    assert after.persistent_state_bytes < MAX_PERSISTENT_STATE_BYTES


def test_new_group_bias_strictly_guards_pre_registration_old_only_rows() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y, old_guard_x, old_guard_y, new_x, new_y = (
        _old_new_collision_support(2)
    )
    config = D26CompactDiagConfig(stage2b_steps=0, stage2c_steps=0)
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_guard_x, old_guard_y
    ).state
    audit = json.loads(after.bias_audit_json)

    assert config.bias_guard_mode == "pre_registration_old_only"
    assert config.new_group_bias_grid == (
        -12.0,
        -8.0,
        -6.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
    )
    assert audit["selection_policy"] == (
        "new_support_leave_one_out_with_old_support_floor_guard"
    )
    assert audit["bias_guard_mode"] == "pre_registration_old_only"
    assert audit["guard_baseline_semantics"] == (
        "stage2b_pre_registration_old_only_head"
    )
    assert audit["query_rows_used"] == 0
    assert audit["old_guard_pass"] is True
    assert audit["old_correct_rows_preserved"] is True
    assert audit["old_guard_support_non_degradation_guaranteed"] is True
    assert audit["bias0_is_not_stage2b_old_only_baseline"] is True
    assert audit["registration_non_forgetting_guaranteed"] is False
    assert audit["terminal_old_support_non_degradation_gate_required"] is True
    assert all(value == 1.0 for value in audit["per_old_class_old_only_accuracy"].values())
    assert all(value == 0.0 for value in audit["per_old_class_bias0_accuracy"].values())
    for class_name in old_classes:
        assert (
            audit["per_old_class_selected_accuracy"][class_name]
            >= audit["per_old_class_old_only_accuracy"][class_name]
        )
    assert after.new_group_bias == -4.0
    assert predict_all_registered(after, old_guard_x).tolist() == old_guard_y.tolist()
    bias0_evidence = next(
        row for row in audit["candidate_evidence"] if row["bias"] == 0.0
    )
    assert bias0_evidence["old_guard_pass"] is False
    assert bias0_evidence["guard_baseline_correct_row_count"] == len(old_guard_y)


def test_k1_registration_uses_closest_safe_bias_without_fake_loo() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y, old_guard_x, old_guard_y, new_x, new_y = (
        _old_new_collision_support(1)
    )
    config = D26CompactDiagConfig(stage2b_steps=15, stage2c_steps=15)
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_guard_x, old_guard_y
    ).state
    audit = json.loads(after.bias_audit_json)

    feasible = [row for row in audit["candidate_evidence"] if row["old_guard_pass"]]
    expected = min(feasible, key=lambda row: (abs(row["bias"]), row["bias"]))
    assert after.new_group_bias == expected["bias"] == -4.0
    assert audit["selection_policy"] == (
        "k1_closest_to_zero_with_pre_registration_old_only_guard_no_loo"
    )
    assert audit["new_support_loo_evaluated"] is False
    assert audit["new_support_selection_rows"] == 0
    assert all(row["new_support_loo_evaluated"] is False for row in feasible)
    assert all("per_new_class" not in row for row in feasible)
    assert audit["old_guard_support_non_degradation_guaranteed"] is True
    assert audit["bias0_is_not_stage2b_old_only_baseline"] is True
    assert audit["registration_non_forgetting_guaranteed"] is False
    assert audit["terminal_old_support_non_degradation_gate_required"] is True
    assert predict_all_registered(after, old_guard_x).tolist() == old_guard_y.tolist()
    assert after.stage2b_optimizer_steps + after.stage2c_optimizer_steps == 30


def test_historical_joint_bias0_guard_mode_remains_constructible() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y, old_guard_x, old_guard_y, new_x, new_y = (
        _old_new_collision_support(2)
    )
    config = D26CompactDiagConfig(
        stage2b_steps=0,
        stage2c_steps=0,
        bias_guard_mode="joint_bias0",
        new_group_bias_grid=(-2.0, -1.0, -0.5, 0.0, 0.5),
    )
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_guard_x, old_guard_y
    ).state
    audit = json.loads(after.bias_audit_json)

    assert audit["bias_guard_mode"] == "joint_bias0"
    assert audit["guard_baseline_semantics"] == (
        "post_registration_combined_old_plus_new_head_with_new_group_bias_zero"
    )
    assert audit["old_guard_pass"] is True
    assert audit["old_guard_support_non_degradation_guaranteed"] is False
    assert after.new_group_bias in config.new_group_bias_grid


def test_strict_k1_bias_selection_fails_closed_when_grid_has_no_safe_value() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y, old_guard_x, old_guard_y, new_x, new_y = (
        _old_new_collision_support(1)
    )
    config = D26CompactDiagConfig(
        stage2b_steps=0,
        stage2c_steps=0,
        bias_guard_mode="pre_registration_old_only",
        new_group_bias_grid=(0.0,),
    )
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    with pytest.raises(D26CompactDiagError, match="no pre-registration-old-only-safe"):
        append_stage2c_new_suffix(
            before, new_x, new_y, new_classes, old_guard_x, old_guard_y
        )


def test_bias_guard_mode_changes_no_trainable_or_compute_resource_contract() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y, old_guard_x, old_guard_y, new_x, new_y = (
        _old_new_collision_support(2)
    )
    audits = []
    for mode in ("joint_bias0", "pre_registration_old_only"):
        config = D26CompactDiagConfig(
            stage2b_steps=0,
            stage2c_steps=0,
            bias_guard_mode=mode,
        )
        before = fit_stage2b_compact_diag(
            old_x, old_y, old_classes, config=config
        ).state
        state = append_stage2c_new_suffix(
            before, new_x, new_y, new_classes, old_guard_x, old_guard_y
        ).state
        audits.append(state.resource_audit())

    invariant_keys = (
        "trainable_parameters",
        "stage2b_trainable_parameters",
        "stage2c_trainable_parameters",
        "estimated_macs_per_query",
        "estimated_adaptation_macs",
        "dense_query_graph_bytes",
        "new_group_bias_scalar_count",
        "query_rows_used_for_fit",
        "query_role_oracle_access",
        "query_class_quota_access",
    )
    assert {key: audits[0][key] for key in invariant_keys} == {
        key: audits[1][key] for key in invariant_keys
    }
    assert all(audit["persistent_state_cap_pass"] is True for audit in audits)


def test_prediction_is_one_argmax_over_all_registered_classes() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y = _support(old_classes, 2, seed=8)
    new_x, new_y = _support(new_classes, 2, seed=9)
    config = D26CompactDiagConfig(stage2c_steps=0)
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    state = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_x, old_y
    ).state
    rows = np.concatenate((old_x[:1], new_x[:1]))
    scores = score_all_registered(state, rows)
    predicted = predict_all_registered(state, rows)

    assert scores.shape == (2, 4)
    assert scores.dtype == np.float32
    assert not scores.flags.writeable
    assert predicted.tolist() == np.asarray(state.classes)[np.argmax(scores, axis=1)].tolist()


def test_protocol_surface_has_no_query_truth_role_quota_or_source_loader() -> None:
    import cvsrffi.stage2_multimodal_compact_diag as module

    public_fit_signatures = "\n".join(
        (
            str(inspect.signature(module.fit_stage2b_compact_diag)),
            str(inspect.signature(module.append_stage2c_new_suffix)),
        )
    ).lower()
    forbidden = ("query", "truth", "role", "quota", "clean", "source")
    assert all(token not in public_fit_signatures for token in forbidden)
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "torch.from_numpy" not in source
    assert ".numpy()" not in source


def test_configuration_and_atomic_append_fail_closed() -> None:
    with pytest.raises(D26CompactDiagError, match="0/10/15"):
        D26CompactDiagConfig(stage2c_steps=5)
    with pytest.raises(D26CompactDiagError, match="at most 15"):
        D26CompactDiagConfig(stage2b_steps=16, stage2c_steps=15)
    with pytest.raises(D26CompactDiagError, match="bias guard mode"):
        D26CompactDiagConfig(bias_guard_mode="query_selected")

    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y = _support(old_classes, 2, seed=10)
    new_x, new_y = _support(new_classes, 2, seed=11)
    config = D26CompactDiagConfig(stage2c_steps=0)
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_x, old_y
    ).state
    more_x, more_y = _support(("new_e",), 2, seed=12)
    with pytest.raises(D26CompactDiagError, match="one atomic"):
        append_stage2c_new_suffix(
            after, more_x, more_y, ("new_e",), old_x, old_y
        )


def test_support_schema_rejects_non_symmetric_k_and_wrong_dimension() -> None:
    x, y = _support(("old_a", "old_b"), 2, seed=13)
    with pytest.raises(D26CompactDiagError, match="K-shot"):
        fit_stage2b_compact_diag(x[:-1], y[:-1], ("old_a", "old_b"))
    with pytest.raises(D26CompactDiagError, match=r"\[N,288\]"):
        fit_stage2b_compact_diag(x[:, :-1], y, ("old_a", "old_b"))
