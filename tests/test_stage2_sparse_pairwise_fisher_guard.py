from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

import cvsrffi.stage2_sparse_pairwise_fisher_guard as d14
from cvsrffi.stage2_joint_residual_logit_head import (
    _build_runtime_authorized_feature_artifact_internal,
)
from cvsrffi.stage2_sparse_pairwise_fisher_guard import (
    SparsePairwiseFisherGuardError,
    SparsePairwiseFisherHyperparameters,
    _maximum_weight_matching,
    _normalize,
    _score_numpy,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    predict_all_registered,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _support(
    classes: tuple[str, ...], k: int, *, dim: int = 24, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat(np.asarray(classes), k)
    ranks = np.tile(np.arange(k, dtype=np.int64), len(classes))
    rows = np.concatenate(
        [
            centers[index][None, :] + 0.15 * rng.normal(size=(k, dim))
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
    old_rows, old_labels, old_ranks = _support(
        ("old0", "old1", "old2", "old3"), k
    )
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


def _hp() -> SparsePairwiseFisherHyperparameters:
    return SparsePairwiseFisherHyperparameters(
        candidate_id="d14_test",
        ridge=0.05,
        gamma_old=0.08,
        gamma_new=0.08,
        band_old=0.30,
        band_new=0.30,
        max_old_edges=3,
    )


def _zero_hp() -> SparsePairwiseFisherHyperparameters:
    return SparsePairwiseFisherHyperparameters(
        candidate_id="d14_true_zero_base",
        ridge=0.05,
        gamma_old=0.0,
        gamma_new=0.0,
        band_old=0.0,
        band_new=0.0,
        max_old_edges=0,
        force_zero=True,
    )


def test_maximum_weight_matching_is_not_greedy_and_is_endpoint_disjoint() -> None:
    result = _maximum_weight_matching(
        {(0, 1): 10.0, (0, 2): 6.0, (1, 3): 6.0},
        ("a", "b", "c", "d"),
        max_edges=3,
    )
    assert result == ((0, 2), (1, 3))
    assert len(sum((list(pair) for pair in result), [])) == len(
        set(sum((list(pair) for pair in result), []))
    )


def test_before_edges_are_sparse_and_after_old_scores_are_bitwise_locked() -> None:
    fitted = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    )
    pairs = fitted.before_state.old_edge_pairs
    assert len(pairs) <= 3
    assert len(pairs.reshape(-1)) == len(set(pairs.reshape(-1).tolist()))
    np.testing.assert_array_equal(
        fitted.before_state.old_edge_pairs, fitted.after_state.old_edge_pairs
    )
    np.testing.assert_array_equal(
        fitted.before_state.old_edge_directions,
        fitted.after_state.old_edge_directions,
    )
    assert np.sum(fitted.after_state.new_rivals >= 0) <= 2
    rng = np.random.default_rng(22)
    probes = rng.normal(size=(15, fitted.after_state.feature_dim)).astype(np.float32)
    before = _score_numpy(probes, fitted.before_state)
    after = _score_numpy(probes, fitted.after_state)
    np.testing.assert_array_equal(
        after[:, : fitted.after_state.old_class_count], before
    )


def test_joint_l2o_is_train_k8_only_and_reports_floor_gates() -> None:
    audit, trace = evaluate_joint_leave_two_out(
        *_before_after(), hyperparameters=_hp()
    )
    assert len(audit["folds"]) == 5
    assert audit["max_old_edge_count"] <= 3
    assert audit["all_old_edges_endpoint_disjoint"] is True
    assert audit["max_new_rivals_per_class"] == 1
    assert audit["old_score_columns_bitwise_equal_before_after"] is True
    assert "before_old_per_class_non_degraded_vs_base" in audit
    assert "after_old_per_class_non_degraded_vs_before" in audit
    assert "after_new_per_class_non_degraded_vs_base" in audit
    assert all(row["old_train_rows_per_class"] == 8 for row in audit["folds"])
    assert all(row["new_held_rows_per_class"] == 2 for row in audit["folds"])
    assert any(
        item["phase"] == "after_sparse_pairwise_closed_form"
        and "selection_diagnostics" in item
        for item in trace
    )


def test_extreme_held_two_mutation_cannot_change_fold_train_state() -> None:
    args = list(_before_after())
    original, _ = evaluate_joint_leave_two_out(*args, hyperparameters=_hp())
    changed_rows = np.array(args[3].features, copy=True)
    labels = np.asarray(args[4]).astype(str)
    ranks = np.asarray(args[5], dtype=np.int64)
    held = (labels == "new0") & np.isin(ranks, (0, 1))
    changed_rows[held] *= -100.0
    args[3] = _artifact(changed_rows)
    changed, _ = evaluate_joint_leave_two_out(*args, hyperparameters=_hp())
    for key in (
        "before_support_selection_sha256",
        "after_support_selection_sha256",
        "before_selection_tensor_sha256",
        "after_selection_tensor_sha256",
        "old_edge_pairs",
        "new_rivals",
    ):
        assert original["folds"][0][key] == changed["folds"][0][key]


def test_true_zero_has_empty_edges_and_bitwise_alpha0_scores() -> None:
    fitted = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_zero_hp()
    )
    assert fitted.before_state.operator_id == "base"
    assert len(fitted.before_state.old_edge_pairs) == 0
    assert len(fitted.after_state.old_edge_pairs) == 0
    assert not np.any(fitted.after_state.new_rivals >= 0)
    rng = np.random.default_rng(33)
    probes = rng.normal(size=(11, fitted.after_state.feature_dim)).astype(np.float32)
    expected = _normalize(probes) @ fitted.after_state.prototypes.T
    actual = _score_numpy(probes, fitted.after_state)
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    np.testing.assert_array_equal(
        np.argmax(actual, axis=1), np.argmax(expected, axis=1)
    )


def test_k1_closes_internal_loo_edges_without_nan_or_random_direction() -> None:
    fitted = fit_before_after_locked(
        *_before_after(k=1), k_shot=1, hyperparameters=_hp()
    )
    assert len(fitted.before_state.old_edge_pairs) == 0
    assert not np.any(fitted.after_state.new_rivals >= 0)
    assert np.isfinite(fitted.after_state.prototypes).all()
    assert np.isfinite(fitted.after_state.new_edge_directions).all()


def test_mapping_exact_k_and_operator_drift_fail_closed() -> None:
    args = list(_before_after())
    args[0] = {"base": np.zeros((40, 24), dtype=np.float32)}
    with pytest.raises(SparsePairwiseFisherGuardError, match="authorized artifact"):
        fit_before_after_locked(*args, k_shot=10, hyperparameters=_hp())
    with pytest.raises(SparsePairwiseFisherGuardError, match="strict physical K-shot"):
        fit_before_after_locked(
            *_before_after(), k_shot=5, hyperparameters=_hp()
        )
    operator_args = list(_before_after())
    operator_args[3] = _artifact(operator_args[3].features, operator_id="fft_eq")
    with pytest.raises(SparsePairwiseFisherGuardError, match="binding drift"):
        fit_before_after_locked(
            *operator_args, k_shot=10, hyperparameters=_hp()
        )


def test_state_is_readonly_hashed_zero_parameter_and_under_budget() -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    ).after_state
    assert len(state.state_content_sha256) == 64
    assert state.resource["trainable_parameters"] == 0
    assert state.resource["adapt_epochs"] == 0
    assert state.resource["persistent_array_state_bytes"] <= 256 * 1024
    assert state.resource["dense_query_graph"] is False
    with pytest.raises(ValueError):
        state.prototypes[0, 0] += 1.0
    with pytest.raises(ValueError):
        state.old_edge_pairs.setflags(write=True)


def test_prediction_is_one_sample_all_registered_and_runtime_bound() -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    ).after_state
    rows, _, _ = _support(("probe", "unused"), 1, dim=24, seed=90)
    prediction, scores = predict_all_registered(
        state, _artifact(rows[:1], seed=303)
    )
    assert prediction.shape == (1,)
    assert scores.shape == (1, 6)
    with pytest.raises(SparsePairwiseFisherGuardError, match="exactly one"):
        predict_all_registered(state, _artifact(rows, seed=304))
    with pytest.raises(SparsePairwiseFisherGuardError, match="binding mismatch"):
        predict_all_registered(
            state,
            _artifact(rows[:1], seed=305, runtime_sha256="d" * 64),
        )


def test_d14_has_no_public_artifact_factory_or_private_token_reference() -> None:
    source = inspect.getsource(d14)
    assert "_select_artifact" not in source
    assert "_ARTIFACT_TOKEN" not in source
    assert "query_role" not in source
    assert "query_quota" not in source
    assert not hasattr(d14, "build_runtime_authorized_feature_artifact")
