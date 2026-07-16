from __future__ import annotations

import hashlib

import numpy as np
import pytest

import cvsrffi.stage2_joint_residual_logit_head as d12
from cvsrffi.stage2_joint_residual_logit_head import (
    JointResidualLogitHeadError,
    JointResidualLogitHeadState,
    ResidualHeadHyperparameters,
    _build_runtime_authorized_feature_artifact_internal,
    _parameter_count,
    _score_numpy,
    _support_selection_sha256,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    predict_all_registered,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _support(
    classes: tuple[str, ...], k: int, *, dim: int = 32, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat(np.asarray(classes), k)
    ranks = np.tile(np.arange(k, dtype=np.int64), len(classes))
    rows = np.concatenate(
        [
            centers[index][None, :] + 0.035 * rng.normal(size=(k, dim))
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
    view_seed: int = 0,
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
        view_seed=view_seed,
    )


def _hp(*, alpha: float = 0.10) -> ResidualHeadHyperparameters:
    return ResidualHeadHyperparameters(
        candidate_id=f"d12_test_a{alpha}",
        rank=8,
        epochs=2,
        learning_rate=0.01,
        alpha=alpha,
        seed=19,
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
        _artifact(joint_rows, seed=101),
        joint_labels,
        joint_ranks,
    )


def test_parameter_count_is_under_three_thousand_for_d288_c11() -> None:
    assert _parameter_count(288, 8, 11) == 2392
    assert _parameter_count(288, 8, 11) < 3000


def test_alpha_zero_preserves_base_cosine_exactly() -> None:
    rng = np.random.default_rng(4)
    rows = rng.normal(size=(3, 8)).astype(np.float32)
    prototypes = rng.normal(size=(2, 8)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    w1 = rng.normal(size=(8, 4)).astype(np.float32)
    w2 = rng.normal(size=(4, 2)).astype(np.float32)
    actual = _score_numpy(rows, prototypes, w1, w2, 0.0)
    normalized = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    np.testing.assert_allclose(actual, normalized @ prototypes.T, atol=0.0)


def test_fit_before_after_joint_head_and_resources() -> None:
    args = _before_after()
    result = fit_before_after_locked(
        *args, k_shot=10, hyperparameters=_hp(), device="cpu"
    )
    assert result.before_state.classes == ("old0", "old1")
    assert result.after_state.classes == ("old0", "old1", "new0", "new1")
    assert result.after_state.registration_generation == 1
    assert result.after_state.resource["trainable_parameters"] == 288
    assert result.after_state.resource["adapt_epochs"] <= 15
    assert result.after_state.resource["persistent_state_bytes"] <= 256 * 1024
    assert len(result.trace) == 4


def test_alpha0_is_true_zero_parameter_base_fallback() -> None:
    hp = ResidualHeadHyperparameters(
        candidate_id="base_fallback",
        epochs=0,
        learning_rate=0.0,
        alpha=0.0,
        old_logit_distillation_weight=0.0,
        residual_identity_weight=0.0,
        factor_weight=0.0,
    )
    result = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=hp, device="cpu"
    )
    assert result.after_state.resource["trainable_parameters"] == 0
    assert result.after_state.resource["residual_head_mac_per_sample"] == 0
    assert result.after_state.w1.shape == (32, 0)
    assert result.after_state.w2.shape == (0, 4)
    assert all(row["residual_training_skipped"] for row in result.trace)


def test_joint_l2o_holds_two_old_and_new_and_reports_full_metrics() -> None:
    audit, trace = evaluate_joint_leave_two_out(
        *_before_after(), hyperparameters=_hp(), device="cpu"
    )
    assert len(audit["folds"]) == 5
    assert all(row["old_train_rows_per_class"] == 8 for row in audit["folds"])
    assert all(row["new_held_rows_per_class"] == 2 for row in audit["folds"])
    assert set(audit["before_old"]["per_class_accuracy"]) == {"old0", "old1"}
    assert set(audit["after_new"]["per_class_accuracy"]) == {"new0", "new1"}
    assert "base_after_old" in audit
    assert "base_after_new" in audit
    assert "old_per_class_non_degraded_vs_base_cosine" in audit
    assert 0.0 <= audit["joint_accuracy"] <= 1.0
    assert 0.0 <= audit["h_old_new"] <= 1.0
    assert len(trace) == 25


def test_ordinary_mapping_or_array_cannot_enter_formal_fit() -> None:
    args = list(_before_after())
    args[0] = {"base": np.zeros((20, 32), dtype=np.float32)}
    with pytest.raises(JointResidualLogitHeadError, match="authorized artifact"):
        fit_before_after_locked(
            *args, k_shot=10, hyperparameters=_hp(), device="cpu"
        )


def test_actual_received_iq_sha_is_required() -> None:
    rows, _, _ = _support(("a", "b"), 5)
    iq = np.zeros((10, 2, 16), dtype=np.float32)
    with pytest.raises(JointResidualLogitHeadError, match="SHA binding"):
        _build_runtime_authorized_feature_artifact_internal(
            iq,
            physical_sample_ids=[f"x{i}" for i in range(10)],
            parent_received_iq_sha256=["f" * 64] * 10,
            sealed_runtime_sha256=HASH_A,
            feature_code_sha256=HASH_B,
            sealed_phase1_checkpoint_sha256=HASH_C,
            extract_single_received_iq=lambda _: rows[:1],
        )


def test_k5_cannot_reach_k10_rows() -> None:
    with pytest.raises(JointResidualLogitHeadError, match="strict physical K-shot"):
        fit_before_after_locked(
            *_before_after(),
            k_shot=5,
            hyperparameters=_hp(),
            device="cpu",
        )


def test_after_old_lineage_replacement_fails_inside_module() -> None:
    args = list(_before_after())
    joint_rows = args[3].features.copy()
    args[3] = _artifact(joint_rows, seed=909)
    with pytest.raises(JointResidualLogitHeadError, match="lineage exact-reuse"):
        fit_before_after_locked(
            *args,
            k_shot=10,
            hyperparameters=_hp(),
            device="cpu",
        )


def test_prediction_is_single_sample_all_registered_and_bound() -> None:
    result = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp(), device="cpu"
    )
    rows, _, _ = _support(("probe", "unused"), 1, seed=90)
    one = _artifact(rows[:1], seed=303)
    predictions, scores = predict_all_registered(result.after_state, one)
    assert predictions.shape == (1,)
    assert scores.shape == (1, 4)
    with pytest.raises(JointResidualLogitHeadError, match="exactly one"):
        predict_all_registered(result.after_state, _artifact(rows, seed=304))
    with pytest.raises(JointResidualLogitHeadError, match="binding mismatch"):
        predict_all_registered(
            result.after_state,
            _artifact(rows[:1], seed=305, runtime_sha256="d" * 64),
        )
    with pytest.raises(JointResidualLogitHeadError, match="operator binding"):
        predict_all_registered(
            result.after_state,
            _artifact(rows[:1], seed=306, operator_id="different_operator"),
        )


def test_state_arrays_are_readonly_and_content_hashed() -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp(), device="cpu"
    ).after_state
    assert len(state.state_content_sha256) == 64
    with pytest.raises(ValueError):
        state.w2[0, 0] += 1.0
    with pytest.raises(ValueError):
        state.w2.setflags(write=True)
    with pytest.raises(JointResidualLogitHeadError, match="content SHA mismatch"):
        JointResidualLogitHeadState(
            schema=state.schema,
            candidate_id=state.candidate_id,
            classes=state.classes,
            prototypes=state.prototypes,
            w1=state.w1,
            w2=state.w2,
            hyperparameters=state.hyperparameters,
            feature_dim=state.feature_dim,
            k_shot=state.k_shot,
            old_class_count=state.old_class_count,
            registration_generation=state.registration_generation,
            resource=state.resource,
            support_feature_artifact_sha256=(
                state.support_feature_artifact_sha256
            ),
            support_selection_sha256=state.support_selection_sha256,
            sealed_runtime_sha256=state.sealed_runtime_sha256,
            feature_code_sha256=state.feature_code_sha256,
            sealed_phase1_checkpoint_sha256=(
                state.sealed_phase1_checkpoint_sha256
            ),
            operator_id=state.operator_id,
            view_seed=state.view_seed,
            state_content_sha256="0" * 64,
        )


def test_state_hash_covers_alpha_resource_and_fold_selection() -> None:
    args = _before_after()
    state = fit_before_after_locked(
        *args, k_shot=10, hyperparameters=_hp(), device="cpu"
    ).after_state
    changed_hp = ResidualHeadHyperparameters(
        **{
            **state.hyperparameters.__dict__,
            "alpha": state.hyperparameters.alpha + 0.01,
        }
    )
    with pytest.raises(JointResidualLogitHeadError, match="content SHA mismatch"):
        JointResidualLogitHeadState(
            schema=state.schema,
            candidate_id=state.candidate_id,
            classes=state.classes,
            prototypes=state.prototypes,
            w1=state.w1,
            w2=state.w2,
            hyperparameters=changed_hp,
            feature_dim=state.feature_dim,
            k_shot=state.k_shot,
            old_class_count=state.old_class_count,
            registration_generation=state.registration_generation,
            resource=state.resource,
            support_feature_artifact_sha256=state.support_feature_artifact_sha256,
            support_selection_sha256=state.support_selection_sha256,
            sealed_runtime_sha256=state.sealed_runtime_sha256,
            feature_code_sha256=state.feature_code_sha256,
            sealed_phase1_checkpoint_sha256=(
                state.sealed_phase1_checkpoint_sha256
            ),
            operator_id=state.operator_id,
            view_seed=state.view_seed,
            state_content_sha256=state.state_content_sha256,
        )
    after_artifact, labels, ranks = args[3], args[4], args[5]
    full = _support_selection_sha256(after_artifact, labels, ranks)
    mask = ~np.isin(ranks, (0, 1))
    fold = _support_selection_sha256(after_artifact, labels, ranks, mask)
    assert full != fold


def test_no_public_callback_or_raw_feature_factory() -> None:
    assert not hasattr(d12, "build_runtime_authorized_feature_artifact")
    assert not hasattr(d12, "wrap_features")
