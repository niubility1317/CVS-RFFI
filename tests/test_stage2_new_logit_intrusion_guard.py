from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

import cvsrffi.stage2_new_logit_intrusion_guard as d13
from cvsrffi.stage2_joint_residual_logit_head import (
    _build_runtime_authorized_feature_artifact_internal,
)
from cvsrffi.stage2_new_logit_intrusion_guard import (
    IntrusionGuardHyperparameters,
    NewLogitIntrusionGuardError,
    _calibrate_new_penalties,
    _prototypes,
    _score_numpy,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    predict_all_registered,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _support(
    classes: tuple[str, ...], k: int, *, dim: int = 16, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat(np.asarray(classes), k)
    ranks = np.tile(np.arange(k, dtype=np.int64), len(classes))
    rows = np.concatenate(
        [
            centers[index][None, :] + 0.04 * rng.normal(size=(k, dim))
            for index in range(len(classes))
        ],
        axis=0,
    ).astype(np.float32)
    return rows, labels, ranks


def _artifact(
    rows: np.ndarray,
    *,
    seed: int = 101,
    runtime_sha256: str = HASH_A,
    operator_id: str = "base",
):
    rng = np.random.default_rng(seed)
    iq = rng.normal(size=(len(rows), 2, 16)).astype(np.float32)
    hashes = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in iq
    ]
    cursor = 0

    def extract(value: np.ndarray) -> np.ndarray:
        nonlocal cursor
        assert value.shape[0] == 1
        result = rows[cursor : cursor + 1]
        cursor += 1
        return result

    return _build_runtime_authorized_feature_artifact_internal(
        iq,
        physical_sample_ids=[f"sid_{seed}_{index}" for index in range(len(iq))],
        parent_received_iq_sha256=hashes,
        sealed_runtime_sha256=runtime_sha256,
        feature_code_sha256=HASH_B,
        sealed_phase1_checkpoint_sha256=HASH_C,
        extract_single_received_iq=extract,
        operator_id=operator_id,
        view_seed=0,
    )


def _before_after(k: int = 10):
    old_rows, old_labels, old_ranks = _support(("old0", "old1"), k)
    new_rows, new_labels, new_ranks = _support(("new0", "new1"), k, seed=13)
    joint_rows = np.concatenate([old_rows, new_rows])
    joint_labels = np.concatenate([old_labels, new_labels])
    joint_ranks = np.concatenate([old_ranks, new_ranks])
    return (
        _artifact(old_rows),
        old_labels,
        old_ranks,
        _artifact(joint_rows),
        joint_labels,
        joint_ranks,
    )


def _hp(mode: str = "constant") -> IntrusionGuardHyperparameters:
    return IntrusionGuardHyperparameters(
        candidate_id=f"d13_test_{mode}",
        mode=mode,
        old_risk_quantile=0.9,
        new_room_quantile=0.25,
        safety=0.0,
        cap=0.4,
        new_floor_margin=0.0,
        hinge_strength=0.5 if mode == "hinge_margin" else 0.0,
    )


@pytest.mark.parametrize("mode", ["constant", "hinge_margin"])
def test_old_prototypes_and_score_columns_are_bitwise_frozen(mode: str) -> None:
    fitted = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp(mode)
    )
    np.testing.assert_array_equal(
        fitted.before_state.prototypes,
        fitted.after_state.prototypes[: fitted.after_state.old_class_count],
    )
    rng = np.random.default_rng(9)
    query = rng.normal(size=(5, fitted.after_state.feature_dim)).astype(np.float32)
    base_state = fit_before_after_locked(
        *_before_after(),
        k_shot=10,
        hyperparameters=IntrusionGuardHyperparameters(
            candidate_id="zero",
            mode=mode,
            old_risk_quantile=0.9,
            new_room_quantile=0.25,
            safety=0.0,
            cap=0.0,
            new_floor_margin=0.0,
            hinge_strength=0.0,
            force_zero=True,
        ),
    ).after_state
    guarded = _score_numpy(query, fitted.after_state)
    base = _score_numpy(query, base_state)
    np.testing.assert_array_equal(
        guarded[:, : fitted.after_state.old_class_count],
        base[:, : fitted.after_state.old_class_count],
    )


def test_constant_formula_records_feasibility_and_shortfall() -> None:
    rows = np.asarray(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
            [0.75, 0.25],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(["old0", "old0", "old1", "old1", "new0", "new0"])
    classes = ("old0", "old1", "new0")
    prototypes = _prototypes(rows, labels, classes)
    penalties, thresholds, strengths, diagnostics = _calibrate_new_penalties(
        rows,
        labels,
        classes,
        prototypes,
        old_class_count=2,
        hyperparameters=IntrusionGuardHyperparameters(
            candidate_id="shortfall",
            mode="constant",
            old_risk_quantile=1.0,
            new_room_quantile=0.0,
            safety=0.2,
            cap=0.01,
            new_floor_margin=0.0,
        ),
    )
    assert penalties[2] <= 0.01 + 1.0e-7
    assert thresholds[2] == 0.0
    assert strengths[2] == 0.0
    assert diagnostics[0]["quantile_method"] == "linear"
    assert diagnostics[0]["before_correct_old_margin_only"] is True
    assert diagnostics[0]["protection_shortfall"] >= 0.0
    assert diagnostics[0]["protection_feasible"] is (
        diagnostics[0]["protection_shortfall"] <= 1.0e-12
    )


@pytest.mark.parametrize("mode", ["constant", "hinge_margin"])
def test_joint_l2o_holds_old_and_new_and_reports_required_metrics(mode: str) -> None:
    audit, trace = evaluate_joint_leave_two_out(
        *_before_after(), hyperparameters=_hp(mode)
    )
    assert len(audit["folds"]) == 5
    assert all(row["old_train_rows_per_class"] == 8 for row in audit["folds"])
    assert all(row["new_held_rows_per_class"] == 2 for row in audit["folds"])
    assert set(audit["before_old"]["per_class_accuracy"]) == {"old0", "old1"}
    assert set(audit["after_new"]["per_class_accuracy"]) == {"new0", "new1"}
    assert audit["old_score_columns_bitwise_unchanged"] is True
    assert "old_forgetting" in audit
    assert "h_old_new" in audit
    assert "joint_accuracy" in audit
    assert "all_new_class_calibration_feasible" in audit
    assert len(trace) == 15


def test_held_two_mutation_does_not_change_that_fold_calibration() -> None:
    args = list(_before_after())
    original, original_trace = evaluate_joint_leave_two_out(
        *args, hyperparameters=_hp("hinge_margin")
    )
    changed_rows = np.array(args[3].features, copy=True)
    changed_labels = np.asarray(args[4]).astype(str)
    changed_ranks = np.asarray(args[5], dtype=np.int64)
    held_new = (changed_labels == "new0") & np.isin(changed_ranks, (0, 1))
    changed_rows[held_new] *= -7.0
    args[3] = _artifact(changed_rows)
    changed, changed_trace = evaluate_joint_leave_two_out(
        *args, hyperparameters=_hp("hinge_margin")
    )
    assert (
        original["folds"][0]["new_class_penalties"]
        == changed["folds"][0]["new_class_penalties"]
    )
    assert (
        original["folds"][0]["calibration_diagnostics"]
        == changed["folds"][0]["calibration_diagnostics"]
    )
    original_after = next(
        row
        for row in original_trace
        if row.get("fold") == 0
        and row.get("phase") == "joint_l2o_after_closed_form"
    )
    changed_after = next(
        row
        for row in changed_trace
        if row.get("fold") == 0
        and row.get("phase") == "joint_l2o_after_closed_form"
    )
    assert (
        original_after["support_selection_sha256"]
        == changed_after["support_selection_sha256"]
    )


def test_mapping_and_exact_k_drift_fail_closed() -> None:
    args = list(_before_after())
    args[0] = {"base": np.zeros((20, 16), dtype=np.float32)}
    with pytest.raises(NewLogitIntrusionGuardError, match="authorized artifact"):
        fit_before_after_locked(
            *args, k_shot=10, hyperparameters=_hp()
        )
    with pytest.raises(NewLogitIntrusionGuardError, match="strict physical K-shot"):
        fit_before_after_locked(
            *_before_after(), k_shot=5, hyperparameters=_hp()
        )


def test_state_is_readonly_content_hashed_and_increment_is_tiny() -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp("hinge_margin")
    ).after_state
    assert len(state.state_content_sha256) == 64
    assert state.resource["trainable_parameters"] == 0
    assert state.resource["adapt_epochs"] == 0
    assert state.resource["incremental_guard_state_bytes"] == 24
    assert state.resource["persistent_state_bytes"] <= 256 * 1024
    with pytest.raises(ValueError):
        state.hinge_thresholds[2] += 1.0
    with pytest.raises(ValueError):
        state.hinge_thresholds.setflags(write=True)


def test_prediction_is_one_sample_all_registered_and_runtime_bound() -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    ).after_state
    rows, _, _ = _support(("probe", "unused"), 1, seed=90)
    prediction, scores = predict_all_registered(state, _artifact(rows[:1], seed=303))
    assert prediction.shape == (1,)
    assert scores.shape == (1, 4)
    with pytest.raises(NewLogitIntrusionGuardError, match="exactly one"):
        predict_all_registered(state, _artifact(rows, seed=304))
    with pytest.raises(NewLogitIntrusionGuardError, match="binding mismatch"):
        predict_all_registered(
            state, _artifact(rows[:1], seed=305, runtime_sha256="d" * 64)
        )


def test_d13_does_not_expose_or_reference_artifact_token_subset_factory() -> None:
    source = inspect.getsource(d13)
    assert "_select_artifact" not in source
    assert "_ARTIFACT_TOKEN" not in source
    assert not hasattr(d13, "build_runtime_authorized_feature_artifact")
