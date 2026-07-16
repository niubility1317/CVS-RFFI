from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest

import cvsrffi.stage2_sprtdr as d17
from cvsrffi.stage2_joint_residual_logit_head import (
    _build_runtime_authorized_feature_artifact_internal,
)
from cvsrffi.stage2_sprtdr import (
    ALPHA_GRID,
    SprtdrError,
    SprtdrHyperparameters,
    _decision_sha,
    _empty_pair_arrays,
    _llr_h,
    _make_state,
    _maximum_weight_matching,
    _normalize,
    _pair_model,
    _score_numpy,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    predict_all_registered,
)


def _support(classes, k, *, dim=24, seed=7, noise=0.12):
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
    parents = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in iq
    ]
    cursor = 0

    def extract(_):
        nonlocal cursor
        result = rows[cursor:cursor + 1]
        cursor += 1
        return result

    return _build_runtime_authorized_feature_artifact_internal(
        iq,
        physical_sample_ids=[f"pid_{seed}_{i}" for i in range(len(rows))],
        parent_received_iq_sha256=parents,
        sealed_runtime_sha256=runtime,
        feature_code_sha256="b" * 64,
        sealed_phase1_checkpoint_sha256="c" * 64,
        extract_single_received_iq=extract,
        operator_id="base",
        view_seed=0,
    )


def _before_after(k=10, *, noise=0.12):
    old = _support(("oa", "ob", "oc", "od"), k, noise=noise)
    new = _support(("nx", "ny"), k, seed=23, noise=noise)
    joint_rows = np.concatenate([old[0], new[0]])
    return (
        _artifact(old[0]), old[1], old[2],
        _artifact(joint_rows), np.concatenate([old[1], new[1]]),
        np.concatenate([old[2], new[2]]),
    )


def _before_after_separable(k):
    classes = ("oa", "ob", "oc", "od", "nx", "ny")
    rng = np.random.default_rng(811)
    centers = np.eye(len(classes), 12, dtype=np.float32)
    rows = np.concatenate([
        centers[index][None, :] + 0.001 * rng.normal(size=(k, 12))
        for index in range(len(classes))
    ]).astype(np.float32)
    labels = np.repeat(np.asarray(classes), k)
    ranks = np.tile(np.arange(k, dtype=np.int64), len(classes))
    old_count = 4 * k
    return (
        _artifact(rows[:old_count]), labels[:old_count], ranks[:old_count],
        _artifact(rows), labels, ranks,
    )


def _hp(band=0.04):
    return SprtdrHyperparameters(
        candidate_id="d17_test",
        rank=8,
        margin_band=band,
        max_old_edges=3,
        max_new_rivals=2,
    )


def test_student_t_pair_direction_and_endpoint_swap_are_canonical() -> None:
    rng = np.random.default_rng(4)
    a = np.asarray([1.0, 0.0, 0.0]) + 0.03 * rng.normal(size=(10, 3))
    b = np.asarray([-1.0, 0.0, 0.0]) + 0.03 * rng.normal(size=(10, 3))
    rows = np.concatenate([a, b]).astype(np.float32)
    labels = np.asarray(["a"] * 10 + ["b"] * 10)
    ab = _pair_model(rows, labels, "a", "b", 3)
    ba = _pair_model(rows, labels, "b", "a", 3)
    assert ab is not None and ba is not None
    probes = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    hab = _llr_h(probes, *ab)
    hba = _llr_h(probes, *ba)
    assert hab[0] > 0.5 and hab[1] < -0.5
    np.testing.assert_allclose(hba, -hab, atol=1e-5)


def test_pair_dimensions_and_statistics_use_only_the_two_endpoints() -> None:
    rows, labels, _ = _support(("a", "b", "irrelevant"), 5, dim=8, seed=18)
    original = _pair_model(rows, labels, "a", "b", 8)
    attacked = np.array(rows, copy=True)
    attacked[labels == "irrelevant"] *= -1000.0
    changed = _pair_model(attacked, labels, "a", "b", 8)
    assert original is not None and changed is not None
    for left, right in zip(original, changed):
        np.testing.assert_array_equal(left, right)


def test_matching_is_endpoint_disjoint_and_not_greedy() -> None:
    selected = _maximum_weight_matching(
        {(0, 1): 10.0, (0, 2): 6.0, (1, 3): 6.0},
        ("a", "b", "c", "d"),
        3,
    )
    assert selected == ((0, 2), (1, 3))
    assert len(sum((list(pair) for pair in selected), [])) == len(
        set(sum((list(pair) for pair in selected), []))
    )


def _manual_states():
    classes = ("a", "b", "c", "n")
    rows, labels, ranks = _support(classes, 5, dim=3, seed=41, noise=0.03)
    artifact = _artifact(rows, seed=411)
    prototypes = np.asarray([
        [1.0, 0.0, 0.0],
        [0.98, 0.20, 0.0],
        [0.0, 1.0, 0.0],
        [0.99, -0.10, 0.0],
    ], dtype=np.float32)
    prototypes = _normalize(prototypes)
    hp = SprtdrHyperparameters(
        candidate_id="manual", rank=1, margin_band=0.04,
        max_old_edges=3, max_new_rivals=2,
    )
    before_arrays = _empty_pair_arrays(1, 0, 2)
    before_arrays.update({
        "old_pairs": np.asarray([[0, 1]], dtype=np.int64),
        "old_dims": np.asarray([[0]], dtype=np.int64),
        "old_mu_a": np.asarray([[1.0]], dtype=np.float32),
        "old_var_a": np.asarray([[0.05]], dtype=np.float32),
        "old_mu_b": np.asarray([[0.75]], dtype=np.float32),
        "old_var_b": np.asarray([[0.05]], dtype=np.float32),
        "old_mid": np.asarray([0.0], dtype=np.float32),
        "old_gap": np.asarray([1.0], dtype=np.float32),
        "old_alpha_pos": np.asarray([0.01], dtype=np.float32),
        "old_alpha_neg": np.asarray([0.01], dtype=np.float32),
    })
    old_selection = np.isin(labels, classes[:3])
    before = _make_state(
        artifact, labels, ranks, old_selection, classes[:3], prototypes[:3],
        hp, 3, 0, before_arrays, np.asarray([True, False, False]),
        np.empty((0,), dtype=np.bool_),
    )
    after_arrays = _empty_pair_arrays(1, 1, 2)
    for name in (
        "old_pairs", "old_dims", "old_mu_a", "old_var_a", "old_mu_b",
        "old_var_b", "old_mid", "old_gap", "old_alpha_pos", "old_alpha_neg",
    ):
        after_arrays[name] = np.array(before_arrays[name], copy=True)
    after_arrays["new_rivals"][0, 0] = 0
    after_arrays["new_dims"][0, 0] = [0]
    after_arrays["new_mu"][0, 0] = [1.0]
    after_arrays["new_var"][0, 0] = [0.05]
    after_arrays["rival_mu"][0, 0] = [0.75]
    after_arrays["rival_var"][0, 0] = [0.05]
    after_arrays["new_mid"][0, 0] = 0.0
    after_arrays["new_gap"][0, 0] = 1.0
    after_arrays["new_alpha_pos"][0, 0] = 0.01
    after_arrays["new_alpha_neg"][0, 0] = 0.01
    after = _make_state(
        artifact, labels, ranks, np.ones(len(labels), dtype=bool), classes,
        prototypes, hp, 3, 1, after_arrays,
        np.asarray([True, False, False]), np.asarray([True]),
    )
    return before, after


def test_old_old_is_zero_sum_and_after_old_scores_are_bitwise_locked() -> None:
    before, after = _manual_states()
    probes = np.asarray([
        [1.0, 0.05, 0.0],
        [0.99, 0.10, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    base = _normalize(probes) @ before.prototypes.T
    scored = _score_numpy(probes, before)
    np.testing.assert_array_equal(scored[:, 2], base[:, 2])
    np.testing.assert_allclose(
        scored[:, 0] + scored[:, 1],
        base[:, 0] + base[:, 1],
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        _score_numpy(probes, after)[:, :3],
        scored,
    )


def test_new_old_requires_immutable_global_top2_and_changes_only_new() -> None:
    before, after = _manual_states()
    probes = np.asarray([
        [1.0, -0.04, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    before_old = _score_numpy(probes, before)
    result = _score_numpy(probes, after)
    np.testing.assert_array_equal(result[:, :3], before_old)
    base_new = _normalize(probes) @ after.prototypes[3:].T
    assert result[0, 3] != base_new[0, 0]
    assert result[1, 3] == base_new[1, 0]


def test_generation1_reuses_one_normalize_and_one_registered_prototype_pass(
    monkeypatch,
) -> None:
    _, after = _manual_states()
    probes = np.asarray([
        [1.0, -0.04, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    # Split-class reference proves appending columns cannot perturb old dots.
    normalized = d17._normalize(probes)
    old_base = d17._prototype_scores(
        normalized, after.prototypes[:after.old_class_count]
    )
    old_scores = d17._score_old_from_base(normalized, old_base, after)
    new_base = d17._prototype_scores(
        normalized, after.prototypes[after.old_class_count:]
    )
    reference = np.concatenate([old_scores, new_base], axis=1).astype(np.float32)
    immutable = np.concatenate([old_base, new_base], axis=1)
    for row_index in range(len(probes)):
        new_local = int(np.argmax(new_base[row_index]))
        old_index = int(np.argmax(old_base[row_index]))
        slots = np.flatnonzero(after.new_rivals[new_local] == old_index)
        if not len(slots):
            continue
        new_index = after.old_class_count + new_local
        if set(np.argsort(immutable[row_index], kind="stable")[-2:].tolist()) != {
            new_index, old_index,
        }:
            continue
        if abs(float(immutable[row_index, new_index] - immutable[row_index, old_index])) > after.hyperparameters.margin_band:
            continue
        slot = int(slots[0])
        h = float(d17._llr_h_normalized(
            normalized[row_index:row_index + 1],
            after.new_dims[new_local, slot],
            after.new_mu[new_local, slot], after.new_var[new_local, slot],
            after.rival_mu[new_local, slot], after.rival_var[new_local, slot],
            after.new_mid[new_local, slot], after.new_gap[new_local, slot],
        )[0])
        reference[row_index, new_index] += np.float32(
            after.new_alpha_pos[new_local, slot] * d17._phi_positive(h)
            - after.new_alpha_neg[new_local, slot] * d17._phi_negative(h)
        )

    counts = {"normalize": 0, "prototype": 0, "prototype_columns": []}
    original_normalize = d17._normalize
    original_prototype = d17._prototype_scores

    def counted_normalize(rows):
        counts["normalize"] += 1
        return original_normalize(rows)

    def counted_prototype(rows, prototypes):
        counts["prototype"] += 1
        counts["prototype_columns"].append(len(prototypes))
        return original_prototype(rows, prototypes)

    monkeypatch.setattr(d17, "_normalize", counted_normalize)
    monkeypatch.setattr(d17, "_prototype_scores", counted_prototype)
    actual = _score_numpy(probes, after)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), np.argmax(reference, axis=1))
    assert counts == {
        "normalize": 1,
        "prototype": 1,
        "prototype_columns": [len(after.classes)],
    }
    assert after.resource["prototype_scorer_passes_per_query"] == 1
    assert after.resource["prototype_scorer"] == (
        "class_column_independent_einsum_optimize_false"
    )
    assert after.resource["normalized_feature_passes_per_query"] == 1
    assert after.resource["prototype_mac_per_query"] == (
        len(after.classes) * after.feature_dim
    )


def test_k1_canonical_z0_k2_to_k4_closed_and_k5_exact_runs() -> None:
    k1 = fit_before_after_locked(
        *_before_after(k=1), k_shot=1, hyperparameters=_hp()
    )
    assert k1.after_state.hyperparameters.force_zero is True
    assert k1.after_state.hyperparameters.rank == 0
    assert not len(k1.before_state.old_pairs)
    assert not np.any(k1.after_state.new_rivals >= 0)
    for k in (2, 3, 4):
        with pytest.raises(SprtdrError, match="K2-K4"):
            fit_before_after_locked(
                *_before_after(k=k), k_shot=k, hyperparameters=_hp()
            )
    k5 = fit_before_after_locked(
        *_before_after(k=5), k_shot=5, hyperparameters=_hp()
    )
    assert k5.after_state.k_shot == 5


@pytest.mark.parametrize("k", (5, 10))
def test_no_edge_non_k1_fit_is_canonical_true_z0(k) -> None:
    fitted = fit_before_after_locked(
        *_before_after_separable(k), k_shot=k, hyperparameters=_hp()
    )
    before = fitted.before_state
    after = fitted.after_state
    assert before.hyperparameters.force_zero is True
    assert after.hyperparameters.force_zero is True
    assert before.hyperparameters.rank == after.hyperparameters.rank == 0
    assert before.hyperparameters == after.hyperparameters
    assert not len(before.old_pairs)
    assert not np.any(after.new_rivals >= 0)
    assert before.old_dims.shape == (0, 0)
    assert after.new_dims.shape == (2, 0, 0)
    trace = fitted.trace[0]
    assert trace["returned_z0"] is True
    assert trace["canonical_true_z0"] is True
    assert trace["true_z0_reason"] == (
        "no_active_pair_after_self_excluded_selection"
    )


def test_support_inclusive_veto_failure_returns_whole_route_true_z0(
    monkeypatch,
) -> None:
    def forced_arrays(
        rows, labels, classes, old_class_count, hp,
        before_arrays=None, **kwargs,
    ):
        new_count = len(classes) - old_class_count
        arrays = _empty_pair_arrays(hp.rank, new_count, hp.max_new_rivals)
        arrays["old_pairs"] = np.asarray([[0, 1]], dtype=np.int64)
        arrays["old_dims"] = np.arange(hp.rank, dtype=np.int64)[None, :]
        arrays["old_mu_a"] = np.ones((1, hp.rank), dtype=np.float32)
        arrays["old_var_a"] = np.ones((1, hp.rank), dtype=np.float32)
        arrays["old_mu_b"] = -np.ones((1, hp.rank), dtype=np.float32)
        arrays["old_var_b"] = np.ones((1, hp.rank), dtype=np.float32)
        arrays["old_mid"] = np.zeros(1, dtype=np.float32)
        arrays["old_gap"] = np.ones(1, dtype=np.float32)
        arrays["old_alpha_pos"] = np.asarray([0.01], dtype=np.float32)
        arrays["old_alpha_neg"] = np.asarray([0.01], dtype=np.float32)
        return arrays, ()

    def forced_bad_oof(rows, labels, classes, state):
        scores = np.zeros((len(rows), len(classes)), dtype=np.float32)
        scores[:, 0] = 1.0
        return scores

    monkeypatch.setattr(d17, "_fit_arrays", forced_arrays)
    monkeypatch.setattr(d17, "_oof_replay_scores", forced_bad_oof)
    fitted = fit_before_after_locked(
        *_before_after(k=5), k_shot=5, hyperparameters=_hp()
    )
    assert fitted.before_state.hyperparameters.force_zero is True
    assert fitted.after_state.hyperparameters.force_zero is True
    assert not len(fitted.before_state.old_pairs)
    assert not np.any(fitted.after_state.new_rivals >= 0)
    trace = fitted.trace[0]
    assert trace["returned_z0"] is True
    assert trace["canonical_true_z0"] is True
    assert trace["true_z0_reason"] == "support_inclusive_veto_failed"


def test_state_is_hashed_readonly_semantic_and_under_50k() -> None:
    state = fit_before_after_locked(
        *_before_after(k=5), k_shot=5, hyperparameters=_hp()
    ).after_state
    assert state.resource["persistent_array_state_bytes"] < 50 * 1024
    assert state.resource["estimated_serialized_state_bytes"] < 50 * 1024
    assert state.resource["trainable_parameters"] == 0
    assert state.resource["adapt_epochs"] == 0
    assert state.resource["dense_query_graph"] is False
    assert state.resource["student_t_nu"] == 3
    with pytest.raises(ValueError):
        state.prototypes[0, 0] += 1.0
    bad = dict(state.resource)
    bad["student_t_nu"] = 5
    with pytest.raises(SprtdrError, match="state drift"):
        replace(state, resource=bad, state_content_sha256="")


def test_malicious_self_sealed_state_cannot_bypass_semantic_guards() -> None:
    before, after = _manual_states()
    with pytest.raises(SprtdrError, match="state drift"):
        replace(
            before,
            old_pairs=np.asarray([[-1, 1]], dtype=np.int64),
            state_content_sha256="",
        )
    with pytest.raises(SprtdrError, match="state drift"):
        replace(
            before,
            old_pairs=np.asarray([[0, 3]], dtype=np.int64),
            state_content_sha256="",
        )
    with pytest.raises(SprtdrError, match="state drift"):
        replace(
            before,
            classes=("a", "a", "c"),
            state_content_sha256="",
        )
    with pytest.raises(SprtdrError, match="state drift"):
        replace(
            before,
            old_alpha_pos=np.zeros(1, dtype=np.float32),
            old_alpha_neg=np.zeros(1, dtype=np.float32),
            state_content_sha256="",
        )
    bad_rivals = np.array(after.new_rivals, copy=True)
    bad_rivals[0, 1] = -2
    with pytest.raises(SprtdrError, match="state drift"):
        replace(after, new_rivals=bad_rivals, state_content_sha256="")
    bad_resource = dict(after.resource)
    bad_resource["backbone_forwards_per_physical_sample"] = 0
    with pytest.raises(SprtdrError, match="state drift"):
        replace(after, resource=bad_resource, state_content_sha256="")
    bad_resource = dict(after.resource)
    bad_resource["estimated_serialized_state_bytes"] -= 1
    with pytest.raises(SprtdrError, match="state drift"):
        replace(after, resource=bad_resource, state_content_sha256="")


def test_worst_case_13_old_20_new_d96_rank8_state_is_under_50k() -> None:
    classes = tuple(f"opaque_{index:02d}" for index in range(33))
    rows, labels, ranks = _support(classes, 5, dim=96, seed=303)
    artifact = _artifact(rows, seed=3303)
    prototypes = _normalize(
        np.random.default_rng(77).normal(size=(33, 96)).astype(np.float32)
    )
    hp = SprtdrHyperparameters(
        candidate_id="d17_worst_case",
        rank=8,
        margin_band=0.04,
        max_old_edges=3,
        max_new_rivals=2,
    )
    arrays = _empty_pair_arrays(8, 20, 2)
    state = _make_state(
        artifact, labels, ranks, np.ones(len(labels), dtype=bool),
        classes, prototypes, hp, 13, 1, arrays,
        np.zeros(13, dtype=np.bool_), np.zeros(20, dtype=np.bool_),
    )
    assert state.resource["persistent_array_state_bytes"] < 50 * 1024
    assert state.resource["estimated_serialized_state_bytes"] < 50 * 1024
    assert state.resource["head_mac_upper_bound_per_query"] < (
        state.resource["identity_qknn_mac_per_query"]
    )
    assert state.resource["student_log1p_ops_upper_bound_per_query"] == 32


def test_d96_13old_to_33registered_n2_old_scores_are_bitwise_locked() -> None:
    old_classes = tuple(f"old_{index:02d}" for index in range(13))
    new_classes = tuple(f"new_{index:02d}" for index in range(20))
    classes = old_classes + new_classes
    rows, labels, ranks = _support(classes, 5, dim=96, seed=1801)
    old_rows = 13 * 5
    before_artifact = _artifact(rows[:old_rows], seed=1802)
    after_artifact = _artifact(rows, seed=1802)
    prototypes = _normalize(
        np.random.default_rng(1803).normal(size=(33, 96)).astype(np.float32)
    )
    hp = SprtdrHyperparameters(
        candidate_id="d17_d96_oldlock",
        rank=0,
        margin_band=0.0,
        max_old_edges=0,
        max_new_rivals=0,
        force_zero=True,
    )
    before = _make_state(
        before_artifact,
        labels[:old_rows], ranks[:old_rows],
        np.ones(old_rows, dtype=bool), old_classes, prototypes[:13], hp,
        13, 0, _empty_pair_arrays(0, 0, 0),
        np.zeros(13, dtype=np.bool_), np.empty(0, dtype=np.bool_),
    )
    after = _make_state(
        after_artifact,
        labels, ranks, np.ones(len(labels), dtype=bool), classes, prototypes,
        hp, 13, 1, _empty_pair_arrays(0, 20, 0),
        np.zeros(13, dtype=np.bool_), np.zeros(20, dtype=np.bool_),
    )
    # N=2 is the real outer held2 shape for one class in each fold.
    held2 = np.random.default_rng(1804).normal(size=(2, 96)).astype(np.float32)
    before_scores = _score_numpy(held2, before)
    after_scores = _score_numpy(held2, after)
    np.testing.assert_array_equal(after_scores[:, :13], before_scores)
    normalized = _normalize(held2)
    per_column = np.stack([
        np.einsum("nd,d->n", normalized, prototype, optimize=False)
        for prototype in prototypes
    ], axis=1).astype(np.float32)
    np.testing.assert_array_equal(after_scores, per_column)


def test_strict_k10_outer_l2o_reports_z0_joint_and_old_lock() -> None:
    result, trace = evaluate_joint_leave_two_out(
        *_before_after(k=10), hyperparameters=_hp()
    )
    assert len(result["folds"]) == 5
    assert result["old_score_bitwise_locked"] is True
    assert result["max_old_edge_count"] <= 3
    assert result["max_new_rivals_per_class"] <= 2
    assert "base_before_old" in result
    assert "base_after_old" in result
    assert "base_after_new" in result
    assert "joint" in result and "base_joint" in result
    assert "H_old_new" in result and "base_H_old_new" in result
    assert "candidate_vs_z0_per_class_non_degraded" in result
    assert all(row["train_rows_per_class"] == 8 for row in result["folds"])
    assert any(row["phase"] == "sprtdr_outer_l2o_fold" for row in trace)


def test_outer_held_mutation_does_not_change_fold0_train_decision() -> None:
    args = list(_before_after(k=10))
    original, _ = evaluate_joint_leave_two_out(
        *args, hyperparameters=_hp()
    )
    changed_old = np.array(args[0].features, copy=True)
    changed_joint = np.array(args[3].features, copy=True)
    held_old = np.isin(np.asarray(args[2]), (0, 1))
    held_joint = np.isin(np.asarray(args[5]), (0, 1))
    changed_old[held_old] *= -1000.0
    changed_joint[held_joint] *= -1000.0
    args[0] = _artifact(changed_old)
    args[3] = _artifact(changed_joint)
    attacked, _ = evaluate_joint_leave_two_out(
        *args, hyperparameters=_hp()
    )
    first = original["folds"][0]
    second = attacked["folds"][0]
    for key in (
        "old_pairs", "new_rivals", "old_dims", "new_dims", "old_stats_sha",
        "new_stats_sha", "amplitude_sha", "floor_handles",
        "before_decision_sha", "after_decision_sha",
        "before_selection_sha", "after_selection_sha",
    ):
        assert first[key] == second[key]


def test_single_query_all_registered_and_no_oracle_surface() -> None:
    state = fit_before_after_locked(
        *_before_after(k=5), k_shot=5, hyperparameters=_hp()
    ).after_state
    rows, _, _ = _support(("probe", "unused"), 1, dim=24, seed=99)
    prediction, scores = predict_all_registered(
        state, _artifact(rows[:1], seed=909)
    )
    assert prediction.shape == (1,)
    assert scores.shape == (1, 6)
    with pytest.raises(SprtdrError, match="exactly one"):
        predict_all_registered(state, _artifact(rows, seed=910))
    source = inspect.getsource(d17)
    assert "query_role" not in source
    assert "query_quota" not in source
    assert "global_assignment" not in source
    assert "clean_sample" not in source
    assert "_ARTIFACT_TOKEN" not in source
    assert set(ALPHA_GRID) == {0.0, 0.005, 0.01}
