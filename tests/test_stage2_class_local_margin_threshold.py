from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

import cvsrffi.stage2_class_local_margin_threshold as d15
from cvsrffi.stage2_class_local_margin_threshold import (
    ClassLocalMarginThresholdError,
    MarginThresholdHyperparameters,
    _choose_binary_threshold,
    _maximum_weight_matching,
    _normalize,
    _old_scores,
    _score_numpy,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    predict_all_registered,
)
from cvsrffi.stage2_joint_residual_logit_head import (
    _build_runtime_authorized_feature_artifact_internal,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _support(
    classes: tuple[str, ...], k: int, *, dim: int = 20, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat(np.asarray(classes), k)
    ranks = np.tile(np.arange(k, dtype=np.int64), len(classes))
    rows = np.concatenate(
        [
            centers[index][None, :] + 0.16 * rng.normal(size=(k, dim))
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
):
    rng = np.random.default_rng(seed)
    iq = rng.normal(size=(len(rows), 2, 12)).astype(np.float32)
    parents = [
        hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for value in iq
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
        parent_received_iq_sha256=parents,
        sealed_runtime_sha256=runtime_sha256,
        feature_code_sha256=HASH_B,
        sealed_phase1_checkpoint_sha256=HASH_C,
        extract_single_received_iq=extract,
        operator_id="base",
        view_seed=0,
    )


def _before_after(k: int = 10):
    old_rows, old_labels, old_ranks = _support(
        ("old0", "old1", "old2", "old3"), k
    )
    new_rows, new_labels, new_ranks = _support(("new0", "new1"), k, seed=13)
    return (
        _artifact(old_rows),
        old_labels,
        old_ranks,
        _artifact(np.concatenate([old_rows, new_rows])),
        np.concatenate([old_labels, new_labels]),
        np.concatenate([old_ranks, new_ranks]),
    )


def _hp(cap: float = 0.05) -> MarginThresholdHyperparameters:
    return MarginThresholdHyperparameters(
        candidate_id=f"d15_cap_{cap}",
        cap=cap,
        select_band_old=0.25,
        max_old_pairs=3,
        force_zero=False,
    )


def _zero() -> MarginThresholdHyperparameters:
    return MarginThresholdHyperparameters(
        candidate_id="d15_z0",
        cap=0.0,
        select_band_old=0.0,
        max_old_pairs=0,
        force_zero=True,
    )


def test_matching_is_max_weight_and_endpoint_disjoint() -> None:
    result = _maximum_weight_matching(
        {(0, 1): 10.0, (0, 2): 6.0, (1, 3): 6.0},
        ("a", "b", "c", "d"),
        max_edges=3,
    )
    assert result == ((0, 2), (1, 3))


def test_binary_threshold_can_be_positive_negative_or_identity() -> None:
    positive, _ = _choose_binary_threshold(
        [0.02, 0.03], [0.01, 0.015], cap=0.05
    )
    negative, _ = _choose_binary_threshold(
        [-0.01, 0.0], [-0.04, -0.03], cap=0.05
    )
    identity, _ = _choose_binary_threshold(
        [0.10, 0.20], [-0.20, -0.10], cap=0.0
    )
    assert positive > 0.0
    assert negative < 0.0
    assert identity == 0.0


def test_old_threshold_directly_shifts_cosine_margin() -> None:
    rows = np.asarray([[1.0, 1.0]], dtype=np.float32)
    prototypes = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = _old_scores(
        rows,
        prototypes,
        np.asarray([[0, 1]], dtype=np.int64),
        np.asarray([0.04], dtype=np.float32),
    )
    base = _normalize(rows) @ prototypes.T
    assert np.isclose(
        result[0, 0] - result[0, 1],
        base[0, 0] - base[0, 1] - 0.04,
    )


def test_before_pair_is_sparse_and_after_old_scores_are_bitwise_locked() -> None:
    fitted = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    )
    assert len(fitted.before_state.old_pairs) <= 3
    assert len(fitted.before_state.old_pairs.reshape(-1)) == len(
        set(fitted.before_state.old_pairs.reshape(-1).tolist())
    )
    np.testing.assert_array_equal(
        fitted.before_state.old_pairs, fitted.after_state.old_pairs
    )
    np.testing.assert_array_equal(
        fitted.before_state.old_thresholds,
        fitted.after_state.old_thresholds,
    )
    rng = np.random.default_rng(9)
    probes = rng.normal(size=(13, fitted.after_state.feature_dim)).astype(np.float32)
    before = _score_numpy(probes, fitted.before_state)
    after = _score_numpy(probes, fitted.after_state)
    np.testing.assert_array_equal(
        after[:, : fitted.after_state.old_class_count], before
    )


def test_joint_l2o_reports_joint_floor_and_threshold_signs() -> None:
    audit, trace = evaluate_joint_leave_two_out(
        *_before_after(), hyperparameters=_hp()
    )
    assert len(audit["folds"]) == 5
    assert audit["old_score_bitwise_locked"] is True
    assert "before_old_per_class_non_degraded_vs_base" in audit
    assert "after_old_per_class_non_degraded_vs_before" in audit
    assert "after_new_per_class_non_degraded_vs_base" in audit
    assert "h_old_new" in audit
    assert "joint_accuracy" in audit
    assert all(row["old_train_rows_per_class"] == 8 for row in audit["folds"])
    assert any(
        item["phase"] == "after_margin_threshold_fit"
        and "new_thresholds" in item
        for item in trace
    )


def test_held_old_and_new_mutation_does_not_change_fold0_train_state() -> None:
    args = list(_before_after())
    original, _ = evaluate_joint_leave_two_out(*args, hyperparameters=_hp())
    changed_old_rows = np.array(args[0].features, copy=True)
    changed_rows = np.array(args[3].features, copy=True)
    labels = np.asarray(args[4]).astype(str)
    ranks = np.asarray(args[5], dtype=np.int64)
    held = np.isin(ranks, (0, 1))
    changed_old_rows[np.isin(np.asarray(args[2]), (0, 1))] *= -25.0
    changed_rows[held] *= -25.0
    args[0] = _artifact(changed_old_rows)
    args[3] = _artifact(changed_rows)
    changed, _ = evaluate_joint_leave_two_out(*args, hyperparameters=_hp())
    for key in (
        "before_selection_sha",
        "after_selection_sha",
        "old_pairs",
        "old_thresholds",
        "new_thresholds",
        "calibration_tensor_sha",
    ):
        assert original["folds"][0][key] == changed["folds"][0][key]


def test_true_zero_is_alpha0_and_under_80kib() -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_zero()
    ).after_state
    assert len(state.old_pairs) == 0
    assert not np.any(state.new_thresholds)
    rng = np.random.default_rng(10)
    probes = rng.normal(size=(9, state.feature_dim)).astype(np.float32)
    expected = _normalize(probes) @ state.prototypes.T
    actual = _score_numpy(probes, state)
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    assert state.resource["persistent_array_state_bytes"] < 80 * 1024
    assert state.resource["trainable_parameters"] == 0
    assert state.resource["adapt_epochs"] == 0


def test_mapping_exact_k_and_runtime_drift_fail_closed() -> None:
    args = list(_before_after())
    args[0] = {"base": np.zeros((40, 20), dtype=np.float32)}
    with pytest.raises(ClassLocalMarginThresholdError, match="authorized artifact"):
        fit_before_after_locked(*args, k_shot=10, hyperparameters=_hp())
    with pytest.raises(ClassLocalMarginThresholdError, match="strict physical K-shot"):
        fit_before_after_locked(*_before_after(), k_shot=5, hyperparameters=_hp())
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    ).after_state
    rows, _, _ = _support(("probe", "unused"), 1, dim=20)
    with pytest.raises(ClassLocalMarginThresholdError, match="query binding"):
        predict_all_registered(
            state, _artifact(rows[:1], runtime_sha256="d" * 64)
        )


def test_prediction_is_exactly_one_all_registered_and_state_readonly() -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    ).after_state
    rows, _, _ = _support(("probe", "unused"), 1, dim=20)
    prediction, scores = predict_all_registered(state, _artifact(rows[:1]))
    assert prediction.shape == (1,)
    assert scores.shape == (1, 6)
    with pytest.raises(ClassLocalMarginThresholdError, match="exactly one"):
        predict_all_registered(state, _artifact(rows))
    with pytest.raises(ValueError):
        state.new_thresholds[0] += 1.0


def test_no_public_artifact_factory_or_query_oracle_surface() -> None:
    source = inspect.getsource(d15)
    assert "_ARTIFACT_TOKEN" not in source
    assert "_select_artifact" not in source
    assert "query_role" not in source
    assert "query_quota" not in source
    assert not hasattr(d15, "build_runtime_authorized_feature_artifact")
