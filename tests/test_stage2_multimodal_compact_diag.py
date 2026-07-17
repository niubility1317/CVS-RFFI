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


def test_new_group_bias_has_old_per_class_floor_guard() -> None:
    old_classes = ("old_a", "old_b")
    old_x, old_y = _support(old_classes, 4, seed=4)
    # Deliberately place new support close to old support so positive bias is risky.
    old_centers = np.stack(
        [old_x[old_y == class_name].mean(axis=0) for class_name in old_classes]
    )
    old_centers /= np.linalg.norm(old_centers, axis=1, keepdims=True)
    new_classes = ("new_c", "new_d")
    new_x, new_y = _support(new_classes, 4, seed=5, centers=old_centers)
    config = D26CompactDiagConfig(stage2c_steps=0)
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_x, old_y
    ).state
    audit = json.loads(after.bias_audit_json)

    assert audit["selection_policy"] == (
        "new_support_leave_one_out_with_old_support_floor_guard"
    )
    assert audit["query_rows_used"] == 0
    assert audit["old_guard_pass"] is True
    assert audit["old_correct_rows_preserved"] is True
    assert audit["bias0_is_not_stage2b_old_only_baseline"] is True
    assert audit["registration_non_forgetting_guaranteed"] is False
    assert audit["terminal_old_support_non_degradation_gate_required"] is True
    for class_name in old_classes:
        assert (
            audit["per_old_class_selected_accuracy"][class_name]
            >= audit["per_old_class_bias0_accuracy"][class_name]
        )
    assert after.new_group_bias in config.new_group_bias_grid


def test_k1_registration_safely_uses_zero_bias_without_fake_loo() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_c", "new_d", "new_e")
    old_x, old_y = _support(old_classes, 1, seed=6)
    new_x, new_y = _support(new_classes, 1, seed=7)
    config = D26CompactDiagConfig(stage2c_steps=15)
    before = fit_stage2b_compact_diag(old_x, old_y, old_classes, config=config).state
    after = append_stage2c_new_suffix(
        before, new_x, new_y, new_classes, old_x, old_y
    ).state
    audit = json.loads(after.bias_audit_json)

    assert after.new_group_bias == 0.0
    assert audit["selection_policy"] == "k1_safe_zero_no_pseudo_loo"
    assert audit["new_support_selection_rows"] == 0
    assert audit["bias0_is_not_stage2b_old_only_baseline"] is True
    assert audit["registration_non_forgetting_guaranteed"] is False
    assert audit["terminal_old_support_non_degradation_gate_required"] is True
    assert after.stage2b_optimizer_steps + after.stage2c_optimizer_steps == 30


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
