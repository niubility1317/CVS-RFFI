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
    assert after.new_class_biases.shape == (0,)
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
    assert after.new_class_biases.shape == (0,)


def test_per_new_class_caps_mathematically_preserve_old_only_correct_rows() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y, old_guard_x, old_guard_y, new_x, new_y = (
        _old_new_collision_support(2)
    )
    config = D26CompactDiagConfig(
        stage2b_steps=0,
        stage2c_steps=0,
        bias_guard_mode="per_new_class_pre_registration_old_only",
    )
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    old_only_scores = score_all_registered(before, old_guard_x)
    old_only_predictions = np.argmax(old_only_scores, axis=1)
    old_truth = np.asarray([old_classes.index(value) for value in old_guard_y])
    old_only_correct = old_only_predictions == old_truth
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_guard_x, old_guard_y
    ).state
    audit = json.loads(after.bias_audit_json)
    combined_scores = score_all_registered(after, old_guard_x)

    assert old_only_correct.all()
    assert after.new_group_bias == 0.0
    assert after.new_class_biases.shape == (len(new_classes),)
    assert after.new_class_biases.dtype == np.float32
    assert not after.new_class_biases.flags.writeable
    assert after.new_class_biases.tolist() == pytest.approx(audit["selected_biases"])
    assert np.array_equal(combined_scores[:, : len(old_classes)], old_only_scores)
    winning_old = old_only_scores[np.arange(len(old_truth)), old_truth]
    assert np.all(
        combined_scores[old_only_correct, len(old_classes) :]
        < winning_old[old_only_correct, None]
    )
    assert predict_all_registered(after, old_guard_x).tolist() == old_guard_y.tolist()
    assert audit["bias_guard_mode"] == (
        "per_new_class_pre_registration_old_only"
    )
    assert audit["old_correct_rows_preserved"] is True
    assert audit["old_guard_support_non_degradation_guaranteed"] is True
    assert audit["new_support_loo_evaluated"] is True
    assert audit["coordinate_pass_count"] == 1
    assert audit["bias_candidate_evaluation_count"] == (
        len(new_classes) * len(config.new_class_bias_offsets)
    )
    for candidate in audit["candidate_evidence"]:
        assert candidate["cap_pass"] is True
        assert candidate["old_guard_pass"] is True
        assert np.all(
            np.asarray(candidate["biases"]) <= np.asarray(audit["bias_caps"]) + 1e-7
        )
    for class_name in old_classes:
        assert (
            audit["per_old_class_selected_accuracy"][class_name]
            >= audit["per_old_class_old_only_accuracy"][class_name]
        )


def test_per_new_class_k1_uses_caps_directly_without_pseudo_loo() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y, old_guard_x, old_guard_y, new_x, new_y = (
        _old_new_collision_support(1)
    )
    config = D26CompactDiagConfig(
        stage2b_steps=0,
        stage2c_steps=0,
        bias_guard_mode="per_new_class_pre_registration_old_only",
    )
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_guard_x, old_guard_y
    ).state
    audit = json.loads(after.bias_audit_json)

    assert after.new_class_biases.tolist() == pytest.approx(audit["bias_caps"])
    assert audit["selection_policy"] == (
        "k1_direct_per_new_class_safety_cap_no_pseudo_loo"
    )
    assert audit["new_support_loo_evaluated"] is False
    assert audit["new_support_selection_rows"] == 0
    assert audit["coordinate_pass_count"] == 0
    assert audit["bias_candidate_evaluation_count"] == 0
    assert audit["candidate_evidence"] == []
    assert "per_new_class" not in audit
    assert predict_all_registered(after, old_guard_x).tolist() == old_guard_y.tolist()


def test_per_new_class_bias_resource_audit_counts_only_closed_form_state() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y, old_guard_x, old_guard_y, new_x, new_y = (
        _old_new_collision_support(2)
    )
    config = D26CompactDiagConfig(
        stage2b_steps=0,
        stage2c_steps=0,
        bias_guard_mode="per_new_class_pre_registration_old_only",
    )
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_guard_x, old_guard_y
    ).state
    resource = after.resource_audit()

    assert resource["bias_trainable_parameters"] == 0
    assert resource["new_group_bias_scalar_count"] == 0
    assert resource["new_class_bias_scalar_count"] == len(new_classes)
    assert resource["new_class_bias_vector_bytes"] == len(new_classes) * 4
    assert resource["registered_bias_additions_per_query"] == len(new_classes)
    assert resource["bias_additions_counted_as_macs"] is False
    assert resource["estimated_bias_selection_macs"] > 0
    assert resource["bias_candidate_evaluation_count"] == (
        len(new_classes) * len(config.new_class_bias_offsets)
    )
    assert resource["trainable_parameters"] == before.trainable_parameters
    assert resource["estimated_macs_per_query"] == 288 + 4 * 288
    assert resource["persistent_state_cap_pass"] is True
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["query_role_oracle_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["clean_sample_access"] is False
    assert resource["source_sample_access"] is False


def test_per_new_class_bias_twenty_new_classes_stays_within_state_cap() -> None:
    old_classes = ("old_a", "old_b", "old_c")
    new_classes = tuple(f"new_{index:02d}" for index in range(20))
    old_x, old_y = _support(old_classes, 2, seed=140)
    new_x, new_y = _support(new_classes, 2, seed=141)
    config = D26CompactDiagConfig(
        stage2b_steps=0,
        stage2c_steps=0,
        bias_guard_mode="per_new_class_pre_registration_old_only",
    )
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_x, old_y
    ).state
    resource = after.resource_audit()

    assert after.new_class_biases.shape == (20,)
    assert resource["new_class_bias_vector_bytes"] == 80
    assert resource["bias_candidate_evaluation_count"] == 100
    assert resource["persistent_state_bytes"] < MAX_PERSISTENT_STATE_BYTES
    assert resource["persistent_state_cap_pass"] is True


def test_per_new_class_bias_fails_closed_without_old_only_correct_guard_row() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d")
    old_x, old_y, _, _, new_x, new_y = _old_new_collision_support(1)
    wrong_old_guard_x = old_x[::-1].copy()
    config = D26CompactDiagConfig(
        stage2b_steps=0,
        stage2c_steps=0,
        bias_guard_mode="per_new_class_pre_registration_old_only",
    )
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    with pytest.raises(D26CompactDiagError, match="at least one old-only-correct"):
        append_stage2c_new_suffix(
            before,
            new_x,
            new_y,
            new_classes,
            wrong_old_guard_x,
            old_y,
        )


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
    with pytest.raises(D26CompactDiagError, match="bias offsets"):
        D26CompactDiagConfig(new_class_bias_offsets=(0.0, 0.5))

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
