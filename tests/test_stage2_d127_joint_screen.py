from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi import stage2_d127_d92_lite as d92_lite
from cvsrffi import stage2_d127_joint_screen as joint
from cvsrffi import stage2_zid_student_t_qknn as qknn


def _lock(k_shot: int) -> qknn.Phase1ZIDStudentTLock:
    return qknn.Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=4.0,
        kernel_effective_dim=160,
        kernel_volume_gamma=1.0,
        shared_h0=0.45,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _row(
    k_shot: int, *, seed: int = 127
) -> tuple[dict[str, object], tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    classes = ("tx_a", "tx_b", "tx_c")
    labels = tuple(class_id for class_id in classes for _ in range(k_shot))
    centers = rng.normal(size=(len(classes), joint.Z_DIM)).astype(np.float32)
    base_support = np.vstack(
        [
            centers[index]
            + np.float32(0.06)
            * rng.normal(size=(k_shot, joint.Z_DIM)).astype(np.float32)
            for index in range(len(classes))
        ]
    ).astype(np.float32)
    base_query = np.vstack(
        [
            centers[index]
            + np.float32(0.08)
            * rng.normal(size=(2, joint.Z_DIM)).astype(np.float32)
            for index in range(len(classes))
        ]
    ).astype(np.float32)
    direction = rng.normal(size=(joint.Z_DIM,)).astype(np.float32)
    direction /= np.sqrt(np.sum(direction * direction, dtype=np.float64)).astype(np.float32)
    adapted_support = (base_support + np.float32(0.035) * direction).astype(np.float32)
    adapted_query = (base_query + np.float32(0.035) * direction).astype(np.float32)
    opaque_ids = tuple(f"opaque_{index:02d}" for index in range(len(base_query)))
    return (
        {
            "base_support_zid": base_support,
            "adapted_support_zid": adapted_support,
            "base_query_zid": base_query,
            "adapted_query_zid": adapted_query,
            "support_labels": labels,
            "registered_classes": classes,
            "opaque_query_ids": opaque_ids,
            "qknn_lock": _lock(k_shot),
        },
        classes,
    )


def test_k1_strictly_reuses_typed_m0_and_mda_qknn_logits() -> None:
    kwargs, classes = _row(1)
    result = joint.run_d127_joint_four_arm(**kwargs)

    assert tuple(arm.arm_id for arm in result.arms) == (
        joint.M0,
        joint.M_DA,
        joint.M_L92,
        joint.M_JOINT,
    )
    assert result.m_l92.logits is result.m0.logits
    assert result.m_joint.logits is result.m_da.logits
    assert type(result.m_l92.qknn_alias_receipt) is d92_lite.D92LiteQKNNAliasReceipt
    assert type(result.m_joint.qknn_alias_receipt) is d92_lite.D92LiteQKNNAliasReceipt
    assert result.m_l92.receipt["underlying_qknn_arm"] == joint.M0
    assert result.m_joint.receipt["underlying_qknn_arm"] == joint.M_DA
    assert result.m_l92.receipt["underlying_qknn_logit_object_reused"] is True
    assert result.m_joint.receipt["underlying_qknn_logit_object_reused"] is True
    assert result.m_l92.predictions == result.m0.predictions
    assert result.m_joint.predictions == result.m_da.predictions
    assert result.m0.classes == classes


@pytest.mark.parametrize("k_shot", [1, 5])
def test_split_common_and_adapted_pairs_match_four_arm(k_shot: int) -> None:
    kwargs, _classes = _row(k_shot, seed=912 + k_shot)
    four = joint.run_d127_joint_four_arm(**kwargs)
    shared = {
        "support_labels": kwargs["support_labels"],
        "registered_classes": kwargs["registered_classes"],
        "opaque_query_ids": kwargs["opaque_query_ids"],
        "qknn_lock": kwargs["qknn_lock"],
    }
    common = joint.run_d127_common_two_arm(
        base_support_zid=kwargs["base_support_zid"],
        base_query_zid=kwargs["base_query_zid"],
        **shared,
    )
    adapted = joint.run_d127_adapted_two_arm(
        adapted_support_zid=kwargs["adapted_support_zid"],
        adapted_query_zid=kwargs["adapted_query_zid"],
        **shared,
    )
    assert tuple(arm.arm_id for arm in common.arms) == (joint.M0, joint.M_L92)
    assert tuple(arm.arm_id for arm in adapted.arms) == (joint.M_DA, joint.M_JOINT)
    for split, original in zip(
        common.arms + adapted.arms,
        (four.m0, four.m_l92, four.m_da, four.m_joint),
    ):
        np.testing.assert_array_equal(split.logits, original.logits)
        assert split.predictions == original.predictions
    if k_shot == 1:
        assert common.lite_arm.logits is common.qknn_arm.logits
        assert adapted.lite_arm.logits is adapted.qknn_arm.logits


@pytest.mark.parametrize("k_shot", [5, 10])
def test_active_lite_heads_preserve_four_arm_closure(k_shot: int) -> None:
    kwargs, classes = _row(k_shot)
    result = joint.run_d127_joint_four_arm(**kwargs)

    assert result.m_l92.logits is not result.m0.logits
    assert result.m_joint.logits is not result.m_da.logits
    assert result.m_l92.qknn_alias_receipt is None
    assert result.m_joint.qknn_alias_receipt is None
    for arm in result.arms:
        assert arm.logits.dtype == np.float32
        assert not arm.logits.flags.writeable
        assert arm.logits.shape == (6, len(classes))
        assert len(arm.predictions) == 6
        assert set(arm.predictions).issubset(set(classes))
        assert arm.receipt["all_registered_classes_scored"] is True
        assert arm.receipt["query_rows_used_for_fit"] == 0
        assert arm.receipt["query_state_updates"] == 0
        assert arm.receipt["query_selection_count"] == 0
        assert arm.receipt["query_batch_dependency"] is False
    assert result.m_l92.receipt["fit_mode"] == "diagonal_oas_form"
    assert result.m_joint.receipt["fit_mode"] == "diagonal_oas_form"
    assert result.receipt["same_row_four_arm_binding"] is True
    assert result.receipt["adapted_cache_reused_for_m_da_and_m_joint"] is True
    assert result.receipt["adaptation_calls_in_joint_interface"] == 0


def test_same_caller_provided_adapted_cache_objects_feed_mda_and_mjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _classes = _row(5, seed=302)
    base_support = kwargs["base_support_zid"]
    adapted_support = kwargs["adapted_support_zid"]
    base_query = kwargs["base_query_zid"]
    adapted_query = kwargs["adapted_query_zid"]
    assert isinstance(base_support, np.ndarray)
    assert isinstance(adapted_support, np.ndarray)
    assert isinstance(base_query, np.ndarray)
    assert isinstance(adapted_query, np.ndarray)

    seen: dict[str, list[int]] = {"qknn_support": [], "qknn_query": [], "lite_support": [], "lite_query": []}
    original_build = joint.qknn.build_typed_zid_support_bank
    original_qknn_score = joint.qknn.score_zid_student_t_logits
    original_lite_fit = joint.d92_lite.fit_d92_lite
    original_lite_score = joint.d92_lite.score_d92_lite

    def wrapped_build(support_zid, support_labels, registered_classes, *, config):
        seen["qknn_support"].append(id(support_zid))
        return original_build(support_zid, support_labels, registered_classes, config=config)

    def wrapped_qknn_score(bank, query_zid, *, metric):
        seen["qknn_query"].append(id(query_zid))
        return original_qknn_score(bank, query_zid, metric=metric)

    def wrapped_lite_fit(support_zid, support_labels, classes):
        seen["lite_support"].append(id(support_zid))
        return original_lite_fit(support_zid, support_labels, classes)

    def wrapped_lite_score(state, query_zid, **kwargs):
        seen["lite_query"].append(id(query_zid))
        return original_lite_score(state, query_zid, **kwargs)

    monkeypatch.setattr(joint.qknn, "build_typed_zid_support_bank", wrapped_build)
    monkeypatch.setattr(joint.qknn, "score_zid_student_t_logits", wrapped_qknn_score)
    monkeypatch.setattr(joint.d92_lite, "fit_d92_lite", wrapped_lite_fit)
    monkeypatch.setattr(joint.d92_lite, "score_d92_lite", wrapped_lite_score)

    joint.run_d127_joint_four_arm(**kwargs)

    assert seen["qknn_support"] == [id(base_support), id(adapted_support)]
    assert seen["lite_support"] == [id(base_support), id(adapted_support)]
    assert seen["qknn_query"] == [id(base_query), id(adapted_query)]
    assert seen["lite_query"] == [id(base_query), id(adapted_query)]


def test_class_order_is_equivariant_for_all_four_arms() -> None:
    kwargs, classes = _row(5, seed=621)
    baseline = joint.run_d127_joint_four_arm(**kwargs)
    permutation = np.asarray([2, 0, 1])
    permuted_classes = tuple(classes[index] for index in permutation)
    permuted = joint.run_d127_joint_four_arm(
        **{**kwargs, "registered_classes": permuted_classes}
    )

    for base_arm, permuted_arm in zip(baseline.arms, permuted.arms, strict=True):
        np.testing.assert_array_equal(permuted_arm.logits, base_arm.logits[:, permutation])
        assert permuted_arm.predictions == base_arm.predictions


def test_exact_top_logit_tie_fails_closed_before_and_after_class_reorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, classes = _row(5, seed=687)

    def tied_qknn_score(bank, query_zid, *, metric):
        del metric
        logits = np.zeros((len(query_zid), len(bank.classes)), dtype=np.float32)
        logits[:, :2] = np.float32(1.0)
        logits.setflags(write=False)
        return logits

    monkeypatch.setattr(joint.qknn, "score_zid_student_t_logits", tied_qknn_score)
    with pytest.raises(joint.D127JointScreenError, match="exact top-logit tie"):
        joint.run_d127_joint_four_arm(**kwargs)

    permuted_classes = (classes[2], classes[0], classes[1])
    with pytest.raises(joint.D127JointScreenError, match="exact top-logit tie"):
        joint.run_d127_joint_four_arm(
            **{**kwargs, "registered_classes": permuted_classes}
        )


def test_all_four_arms_are_consistent_per_row_chunk_and_query_reorder() -> None:
    kwargs, _classes = _row(5, seed=733)
    base_query = kwargs["base_query_zid"]
    adapted_query = kwargs["adapted_query_zid"]
    opaque_ids = kwargs["opaque_query_ids"]
    assert isinstance(base_query, np.ndarray)
    assert isinstance(adapted_query, np.ndarray)
    assert isinstance(opaque_ids, tuple)
    batch = joint.run_d127_joint_four_arm(**kwargs)

    def run_indices(indices: np.ndarray) -> joint.D127JointFourArmResult:
        return joint.run_d127_joint_four_arm(
            **{
                **kwargs,
                "base_query_zid": base_query[indices],
                "adapted_query_zid": adapted_query[indices],
                "opaque_query_ids": tuple(opaque_ids[int(index)] for index in indices),
            }
        )

    per_row_results = [
        run_indices(np.asarray([index], dtype=np.int64))
        for index in range(len(base_query))
    ]
    chunk_indices = (
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2, 3, 4], dtype=np.int64),
        np.asarray([5], dtype=np.int64),
    )
    chunk_results = [run_indices(indices) for indices in chunk_indices]
    permutation = np.asarray([5, 1, 4, 0, 3, 2], dtype=np.int64)
    reordered = run_indices(permutation)

    for arm_index in range(4):
        batch_arm = batch.arms[arm_index]
        per_row_logits = np.vstack(
            [result.arms[arm_index].logits for result in per_row_results]
        )
        chunk_logits = np.vstack(
            [result.arms[arm_index].logits for result in chunk_results]
        )
        np.testing.assert_array_equal(per_row_logits, batch_arm.logits)
        np.testing.assert_array_equal(chunk_logits, batch_arm.logits)
        np.testing.assert_array_equal(
            reordered.arms[arm_index].logits, batch_arm.logits[permutation]
        )
        assert tuple(
            result.arms[arm_index].predictions[0] for result in per_row_results
        ) == batch_arm.predictions
        assert tuple(
            prediction
            for result in chunk_results
            for prediction in result.arms[arm_index].predictions
        ) == batch_arm.predictions
        assert reordered.arms[arm_index].predictions == tuple(
            batch_arm.predictions[int(index)] for index in permutation
        )


def test_rejects_cross_row_shape_and_k_lock_drift() -> None:
    kwargs, _classes = _row(5)
    bad_shape = dict(kwargs)
    bad_shape["adapted_query_zid"] = bad_shape["adapted_query_zid"][:-1]
    with pytest.raises(joint.D127JointScreenError, match="same-row shapes"):
        joint.run_d127_joint_four_arm(**bad_shape)

    bad_lock = dict(kwargs)
    bad_lock["qknn_lock"] = _lock(1)
    with pytest.raises(joint.D127JointScreenError, match="K-shot"):
        joint.run_d127_joint_four_arm(**bad_lock)


def test_public_surface_has_no_truth_role_quota_or_global_assignment_inputs() -> None:
    parameters = set(inspect.signature(joint.run_d127_joint_four_arm).parameters)
    forbidden = {
        "truth",
        "query_truth",
        "role",
        "quota",
        "global_assignment",
        "global_reassignment",
        "source_rows",
        "clean_rows",
        "adapter",
        "adaptation_function",
    }
    assert not parameters & forbidden
    assert parameters == {
        "base_support_zid",
        "adapted_support_zid",
        "base_query_zid",
        "adapted_query_zid",
        "support_labels",
        "registered_classes",
        "opaque_query_ids",
        "qknn_lock",
    }


def test_results_are_immutable_and_same_row_bound() -> None:
    kwargs, _classes = _row(5, seed=712)
    result = joint.run_d127_joint_four_arm(**kwargs)

    with pytest.raises(ValueError):
        result.m0.logits[0, 0] = np.float32(0.0)
    with pytest.raises(TypeError):
        result.receipt["query_state_updates"] = 1
    with pytest.raises(TypeError):
        result.m_l92.receipt["fit_mode"] = "tampered"
    row_sha = result.receipt["row_input_sha256"]
    assert all(arm.receipt["row_input_sha256"] == row_sha for arm in result.arms)
    assert result.receipt["query_rows_used_for_fit"] == 0
    assert result.receipt["query_state_updates"] == 0
    assert result.receipt["query_selection_count"] == 0
    assert result.receipt["query_batch_dependency"] is False
