from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest

import cvsrffi.stage2_fcar as fcar
from cvsrffi.stage2_fcar import (
    FcarError,
    FcarHyperparameters,
    _base_scores,
    _build_oof,
    _baseline_floor,
    _delta,
    _evaluate,
    _deployment_consistency_records,
    _passes,
    _rollback_key,
    _score_numpy,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    predict_all_registered,
)
from cvsrffi.stage2_joint_residual_logit_head import (
    _build_runtime_authorized_feature_artifact_internal,
)


def _support(classes, k, *, dim=24, seed=7, noise=0.10):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat(np.asarray(classes), k)
    ranks = np.tile(np.arange(k, dtype=np.int64), len(classes))
    rows = np.concatenate([
        centers[index][None, :] + noise * rng.normal(size=(k, dim))
        for index in range(len(classes))
    ]).astype(np.float32)
    return rows, labels, ranks


def _artifact(rows, *, seed=101, runtime="a" * 64):
    rng = np.random.default_rng(seed)
    iq = rng.normal(size=(len(rows), 2, 12)).astype(np.float32)
    parent = [
        hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for value in iq
    ]
    cursor = 0

    def extract(_):
        nonlocal cursor
        result = rows[cursor:cursor + 1]
        cursor += 1
        return result

    return _build_runtime_authorized_feature_artifact_internal(
        iq,
        physical_sample_ids=[f"sid_{seed}_{index}" for index in range(len(rows))],
        parent_received_iq_sha256=parent,
        sealed_runtime_sha256=runtime,
        feature_code_sha256="b" * 64,
        sealed_phase1_checkpoint_sha256="c" * 64,
        extract_single_received_iq=extract,
        operator_id="base",
        view_seed=0,
    )


def _before_after(k=10, noise=0.10):
    old = _support(("old0", "old1", "old2", "old3"), k, noise=noise)
    new = _support(("new0", "new1"), k, seed=13, noise=noise)
    joint_rows = np.concatenate([old[0], new[0]])
    return (
        _artifact(old[0]), old[1], old[2],
        _artifact(joint_rows), np.concatenate([old[1], new[1]]),
        np.concatenate([old[2], new[2]]),
    )


def _hp():
    return FcarHyperparameters(
        candidate_id="d16_test", rank=8, shrink=0.5, ridge=0.01,
        margin_band=0.30,
    )


def test_delta_is_asymmetric_piecewise_linear_on_fixed_grid() -> None:
    assert _delta(0.4, 0.02, 0.01, 0.5) == 0.0
    assert _delta(0.75, 0.02, 0.01, 0.5) == pytest.approx(0.01)
    assert _delta(-0.75, 0.02, 0.01, 0.5) == pytest.approx(-0.005)
    assert _delta(1.0, 0.02, 0.01, 0.5) == pytest.approx(0.02)


def test_two_fold_oof_threshold_models_exclude_each_record() -> None:
    rows, labels, ranks = _support(("c0", "c1", "c2"), 10, seed=61)
    classes = tuple(sorted(np.unique(labels).tolist()))
    records, _ = _build_oof(rows, labels, ranks, classes, 0, _hp())
    assert len(records) == 30
    for record in records:
        assert record["threshold_models_exclude_self"] is True
        assert record["row_index"] not in record["threshold_peer_row_indices"]
        assert record["row_index"] not in record["model_train_row_indices"]
        assert all(
            ranks[index] % 2 != record["eval_fold"]
            for index in record["model_train_row_indices"]
        )


def test_mutating_record_does_not_change_its_threshold() -> None:
    rows, labels, ranks = _support(("c0", "c1", "c2"), 10, seed=91)
    classes = tuple(sorted(np.unique(labels).tolist()))
    original, _ = _build_oof(rows, labels, ranks, classes, 0, _hp())
    target = 0
    changed = np.array(rows, copy=True)
    changed[target] *= -1000.0
    attacked, _ = _build_oof(changed, labels, ranks, classes, 0, _hp())
    for key in ("q_pos", "q_neg", "mid", "half_gap", "threshold_peer_row_indices"):
        assert original[target][key] == attacked[target][key]


def test_deployment_consistency_veto_uses_full_prototype_and_envelope() -> None:
    rows = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    reference = (
        {
            "row_index": 0,
            "truth": 0,
            "base_scores": np.asarray([0.0, 1.0], dtype=np.float32),
            "h": -1.0,
        },
        {
            "row_index": 1,
            "truth": 1,
            "base_scores": np.asarray([1.0, 0.0], dtype=np.float32),
            "h": 1.0,
        },
    )
    full_prototypes = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32
    )
    built = _deployment_consistency_records(
        rows,
        reference,
        full_prototypes,
        targets=(0,),
        active=[0],
        dims=np.asarray([[0]], dtype=np.int64),
        cm=np.asarray([[1.0]], dtype=np.float32),
        cv=np.asarray([[0.1]], dtype=np.float32),
        rm=np.asarray([[-1.0]], dtype=np.float32),
        rv=np.asarray([[0.1]], dtype=np.float32),
        mid=np.asarray([0.0], dtype=np.float32),
        half_gap=np.asarray([1.0], dtype=np.float32),
    )
    assert np.argmax(reference[0]["base_scores"]) == 1
    assert np.argmax(built[0]["base_scores"]) == 0
    assert np.argmax(reference[1]["base_scores"]) == 0
    assert np.argmax(built[1]["base_scores"]) == 1
    assert built[0]["deployment_state_base_and_h"] is True
    metrics = _evaluate(
        built, 0, 0.0, 0.0, _hp(), combined={0: (0.005, 0.0)}
    )
    assert metrics["base_correct"][0] == 1


def test_before_old_state_and_scores_are_bitwise_locked() -> None:
    fitted = fit_before_after_locked(
        *_before_after(noise=0.20), k_shot=10, hyperparameters=_hp()
    )
    old = fitted.after_state.old_class_count
    assert np.any(fitted.before_state.enabled)
    assert np.any(fitted.after_state.enabled)
    for name in (
        "selected_dims", "class_mean", "class_var", "rest_mean", "rest_var",
        "enabled", "llr_mid", "llr_half_gap", "a_plus", "a_minus",
    ):
        np.testing.assert_array_equal(
            getattr(fitted.before_state, name),
            getattr(fitted.after_state, name)[:old],
        )
    probes = np.random.default_rng(123).normal(
        size=(257, fitted.after_state.feature_dim)
    ).astype(np.float32)
    np.testing.assert_array_equal(
        _score_numpy(probes, fitted.after_state)[:, :old],
        _score_numpy(probes, fitted.before_state),
    )
    assert all(
        row.get("deployment_state_consistency_veto") == "pass"
        for phase in fitted.trace
        for row in phase["diagnostics"]
        if row["enabled"]
    )


def test_k1_is_canonical_z0_and_k2_to_k4_fail_closed() -> None:
    fitted = fit_before_after_locked(
        *_before_after(k=1), k_shot=1, hyperparameters=_hp()
    )
    state = fitted.after_state
    assert state.hyperparameters.force_zero is True
    assert state.hyperparameters.rank == 0
    assert state.selected_dims.shape == (6, 0)
    assert not np.any(state.enabled)
    assert not np.any(state.a_plus)
    assert not np.any(state.a_minus)
    probes = np.random.default_rng(5).normal(
        size=(11, state.feature_dim)
    ).astype(np.float32)
    _, expected = _base_scores(probes, state)
    np.testing.assert_array_equal(_score_numpy(probes, state), expected)
    for k in (2, 3, 4):
        with pytest.raises(FcarError, match="K2-K4 unsupported"):
            fit_before_after_locked(
                *_before_after(k=k), k_shot=k, hyperparameters=_hp()
            )


def test_k5_has_valid_two_fold_class_balanced_oof() -> None:
    fitted = fit_before_after_locked(
        *_before_after(k=5, noise=0.12), k_shot=5,
        hyperparameters=_hp(),
    )
    assert fitted.before_state.k_shot == 5
    assert fitted.after_state.k_shot == 5
    assert fitted.after_state.resource["two_fold_model_fits_per_class"] == 2
    rows, labels, ranks = _support(("c0", "c1", "c2"), 5, seed=202)
    classes = tuple(sorted(np.unique(labels).tolist()))
    records, _ = _build_oof(rows, labels, ranks, classes, 0, _hp())
    assert len(records) == 15
    assert {record["eval_fold"] for record in records} == {0, 1}
    assert all(record["threshold_models_exclude_self"] for record in records)


def test_strict_k10_joint_leave_two_out_has_five_train_k8_folds() -> None:
    audit, trace = evaluate_joint_leave_two_out(
        *_before_after(noise=0.20), hyperparameters=_hp()
    )
    assert len(audit["folds"]) == 5
    assert audit["old_score_bitwise_locked"] is True
    assert all(row["train_rows_per_class"] == 8 for row in audit["folds"])
    assert all(row["held_rows_per_class"] == 2 for row in audit["folds"])
    assert all(row["held_disjoint_from_selection"] for row in audit["folds"])
    assert "joint" in audit
    assert "overall_accuracy" in audit["joint"]
    assert "min_class_accuracy" in audit["joint"]
    assert set(audit["joint"]["per_class_accuracy"]) == {
        "old0", "old1", "old2", "old3", "new0", "new1"
    }
    assert 0.0 <= audit["H_old_new"] <= 1.0
    assert set(audit["per_class_old_forgetting"]) == {
        "old0", "old1", "old2", "old3"
    }
    assert set(
        audit["candidate_vs_z0_per_class_non_degraded"]["after_new"]
    ) == {"new0", "new1"}
    assert any(row["phase"] == "joint_l2o_fold" for row in trace)
    with pytest.raises(FcarError, match="strict physical K-shot"):
        evaluate_joint_leave_two_out(
            *_before_after(k=5), hyperparameters=_hp()
        )


def test_outer_held2_extreme_mutation_cannot_change_fold0_training_decision() -> None:
    args = list(_before_after(noise=0.20))
    original, _ = evaluate_joint_leave_two_out(
        *args, hyperparameters=_hp()
    )
    changed_old = np.array(args[0].features, copy=True)
    changed_joint = np.array(args[3].features, copy=True)
    old_held = np.isin(np.asarray(args[2]), (0, 1))
    joint_held = np.isin(np.asarray(args[5]), (0, 1))
    changed_old[old_held] *= -1000.0
    changed_joint[joint_held] *= -1000.0
    args[0] = _artifact(changed_old)
    args[3] = _artifact(changed_joint)
    attacked, _ = evaluate_joint_leave_two_out(
        *args, hyperparameters=_hp()
    )
    original_fold = original["folds"][0]
    attacked_fold = attacked["folds"][0]
    for key in (
        "before_selection_sha",
        "after_selection_sha",
        "before_decision_tensor_sha",
        "after_decision_tensor_sha",
        "floor_handles",
        "enabled",
        "before_train_physical_id_sha",
        "after_train_physical_id_sha",
    ):
        assert original_fold[key] == attacked_fold[key]
    assert original_fold["held_disjoint_from_selection"] is True
    assert attacked_fold["held_disjoint_from_selection"] is True


def test_floor_is_automatic_bottom_quartile_and_rollback_order_is_deterministic() -> None:
    records = []
    classes = ("opaque-z", "opaque-a", "opaque-q", "opaque-b", "opaque-y")
    for truth in range(len(classes)):
        for rank in range(4):
            scores = np.full(len(classes), -1.0, dtype=np.float32)
            scores[truth] = 1.0
            if truth == 2 and rank > 0:
                scores[0] = 2.0
            records.append({"truth": truth, "base_scores": scores})
    floor, accuracy = _baseline_floor(records, classes)
    assert floor == {2, 1}
    assert accuracy[2] < accuracy[1]
    targets = (0, 1, 2)
    benefit = {0: (0, 0.01), 1: (0, 0.02), 2: (1, 0.10)}
    floor_classes = {2}
    order = sorted(
        range(3),
        key=lambda index: _rollback_key(
            index, targets, floor_classes, benefit, classes
        ),
    )
    assert order == [0, 1, 2]


def test_nonfloor_nonzero_requires_strict_benefit_and_zero_added_capture() -> None:
    base = np.asarray([2, 2, 2], dtype=np.int64)
    q20 = np.asarray([0.1, 0.2, 0.3])
    neutral = {
        "base_correct": base,
        "candidate_correct": base.copy(),
        "new_capture": np.zeros(3, dtype=np.int64),
        "lost_baseline_correct": np.zeros(3, dtype=np.int64),
        "base_q20": q20,
        "candidate_q20": q20.copy(),
    }
    assert _passes(neutral, 1, {0}, strict_floor=True) is False
    improved = {**neutral, "candidate_q20": q20 + np.asarray([0.0, 0.0, 0.01])}
    assert _passes(improved, 1, {0}, strict_floor=True) is True
    captured = {
        **improved,
        "new_capture": np.asarray([1, 0, 0], dtype=np.int64),
    }
    assert _passes(captured, 1, {0}, strict_floor=True) is False


def test_state_grid_hash_resource_and_single_query_contract() -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    ).after_state
    grid = np.asarray(fcar.AMPLITUDE_GRID, dtype=np.float32)
    assert np.all(np.isin(state.a_plus, grid))
    assert np.all(np.isin(state.a_minus, grid))
    assert state.resource["trainable_parameters"] == 0
    assert state.resource["adapt_epochs"] == 0
    assert state.resource["two_fold_model_fits_per_class"] == 2
    assert state.resource["backbone_forwards_per_physical_sample"] == 1
    assert state.resource["fft_branches_per_physical_sample"] == 0
    assert state.resource["head_scalar_ops_per_sample_upper_bound"] > 0
    assert state.resource["persistent_array_state_bytes"] < 80 * 1024
    changed = np.array(state.a_plus, copy=True)
    changed[0] = 0.02 if changed[0] != 0.02 else 0.01
    with pytest.raises(FcarError, match="state content SHA mismatch"):
        replace(state, a_plus=changed)
    disabled = np.flatnonzero(~state.enabled)
    if len(disabled):
        broken_dims = np.array(state.selected_dims, copy=True)
        broken_dims[disabled[0], 0] = 0
        with pytest.raises(FcarError, match="state content SHA mismatch"):
            replace(state, selected_dims=broken_dims)
    rows = _support(("probe", "unused"), 1)[0]
    prediction, scores = predict_all_registered(state, _artifact(rows[:1]))
    assert prediction.shape == (1,)
    assert scores.shape == (1, 6)
    with pytest.raises(FcarError, match="exactly one"):
        predict_all_registered(state, _artifact(rows))


def test_malicious_self_sealed_state_cannot_bypass_semantic_guards() -> None:
    state = fit_before_after_locked(
        *_before_after(noise=0.20), k_shot=10, hyperparameters=_hp()
    ).after_state
    assert np.any(state.enabled)
    with pytest.raises(FcarError, match="state drift"):
        replace(
            state,
            candidate_id="self-sealed-evil",
            state_content_sha256="",
        )
    with pytest.raises(FcarError, match="state drift"):
        replace(state, k_shot=2, state_content_sha256="")
    bad_resource = dict(state.resource)
    bad_resource["enabled_class_count"] += 1
    with pytest.raises(FcarError, match="state drift"):
        replace(
            state, resource=bad_resource, state_content_sha256=""
        )
    disabled = int(np.flatnonzero(~state.enabled)[0])
    bad_plus = np.array(state.a_plus, copy=True)
    bad_plus[disabled] = np.float32(0.005)
    with pytest.raises(FcarError, match="state drift"):
        replace(state, a_plus=bad_plus, state_content_sha256="")
    k1 = fit_before_after_locked(
        *_before_after(k=1), k_shot=1, hyperparameters=_hp()
    ).after_state
    noncanonical_resource = dict(k1.resource)
    noncanonical_resource["two_fold_model_fits_per_class"] = 2
    with pytest.raises(FcarError, match="state drift"):
        replace(
            k1,
            resource=noncanonical_resource,
            state_content_sha256="",
        )


def test_module_has_no_role_quota_query_oracle_or_hardcoded_class_ids() -> None:
    source = inspect.getsource(fcar)
    assert "query_role" not in source
    assert "query_quota" not in source
    assert "floor_class_ids" not in source
    assert "old0" not in source
    assert "new0" not in source
