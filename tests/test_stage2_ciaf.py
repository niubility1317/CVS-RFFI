from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_ciaf import (
    CiafError,
    FEATURE_DIM,
    Int8DomainClassComponent,
    fit_old_ciaf,
    predict_one,
    register_new_classes,
    score_one,
)


OLD = ("old-a", "old-b", "old-c")


def _direction(index: int) -> np.ndarray:
    value = np.zeros(FEATURE_DIM, dtype=np.float32)
    value[index] = 1.0
    return value


def _component(domain_order: tuple[int, ...] = (0, 1, 2)) -> Int8DomainClassComponent:
    q = np.zeros((3, 3, FEATURE_DIM), dtype=np.int8)
    scale = np.ones((3, 3), dtype=np.float16) / 127.0
    mask = np.ones((3, 3), dtype=np.uint8)
    for domain in range(3):
        for class_index in range(3):
            q[domain, class_index, class_index] = 127
            q[domain, class_index, 10 + domain] = 3 * (domain + 1)
    order = np.asarray(domain_order)
    return Int8DomainClassComponent(q[order], scale[order], mask[order], OLD)


def _old_support(k: int = 5, seed: int = 7) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for class_index, label in enumerate(OLD):
        for _ in range(k):
            row = _direction(class_index) + rng.normal(0.0, 0.015, FEATURE_DIM)
            rows.append(row.astype(np.float32))
            labels.append(label)
    return np.stack(rows), labels


def _new_support(k: int = 5) -> tuple[np.ndarray, list[str]]:
    rows = []
    labels = []
    for class_index, label in enumerate(("new-x", "new-y")):
        base = _direction(20 + class_index)
        for rank in range(k):
            row = base.copy()
            row[30 + rank] = 0.01
            rows.append(row)
            labels.append(label)
    return np.stack(rows), labels


def test_int8_component_is_copied_readonly_and_dequantized_transiently() -> None:
    component = _component()
    assert component.domain_class_q.dtype == np.int8
    assert component.domain_class_scale.dtype == np.float16
    assert not component.domain_class_q.flags.writeable
    assert not component.domain_class_scale.flags.writeable
    with pytest.raises(ValueError):
        component.domain_class_q[0, 0, 0] = 0
    anchors = component.dequantized_class_anchors(0)
    assert anchors.shape == (3, FEATURE_DIM)
    np.testing.assert_allclose(np.linalg.norm(anchors, axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("k_shot", (1, 5, 10))
def test_fit_and_registration_are_closed_form_and_resource_bounded(k_shot: int) -> None:
    old_z, old_y = _old_support(k_shot)
    before = fit_old_ciaf(_component(), old_z, old_y)
    new_z, new_y = _new_support(k_shot)
    after = register_new_classes(before, new_z, new_y)
    assert before.k_shot == k_shot
    assert before.old_class_count == 3
    assert after.classes == OLD + ("new-x", "new-y")
    assert np.all((before.old_support_alpha >= 0.25) & (before.old_support_alpha <= 0.95))
    if k_shot == 1:
        np.testing.assert_array_equal(before.old_support_robustness, 0.0)
        np.testing.assert_allclose(before.old_support_alpha, 0.25)
    resource = after.resource_audit()
    assert resource["trainable_parameters"] == 0
    assert resource["adaptation_epochs"] == 0
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["persistent_state_limit_pass"] is True
    assert resource["int8_component_update_access"] is False
    assert resource["persistent_full_precision_ground_anchor_count"] == 0
    assert not hasattr(after, "old_anchor_prototypes")


def test_registration_keeps_every_old_score_column_bitwise_unchanged() -> None:
    old_z, old_y = _old_support()
    before = fit_old_ciaf(_component(), old_z, old_y)
    probe = _direction(0) + 0.1 * _direction(20)
    old_scores = score_one(before, probe)
    new_z, new_y = _new_support()
    after = register_new_classes(before, new_z, new_y)
    np.testing.assert_array_equal(after.prototypes[:3], before.prototypes)
    np.testing.assert_array_equal(after.score_bias[:3], before.score_bias)
    np.testing.assert_array_equal(score_one(after, probe)[:3], old_scores)


def test_registration_honors_exact_registered_class_order() -> None:
    old_z, old_y = _old_support()
    before = fit_old_ciaf(_component(), old_z, old_y)
    new_z, new_y = _new_support()
    after = register_new_classes(
        before,
        new_z,
        new_y,
        registered_classes=("new-y", "new-x"),
    )
    assert after.classes == OLD + ("new-y", "new-x")


def test_collision_bias_is_support_only_and_larger_for_old_like_new_class() -> None:
    old_z, old_y = _old_support()
    before = fit_old_ciaf(_component(), old_z, old_y)
    k = before.k_shot
    collision_rows = np.stack([_direction(0)] * k + [_direction(40)] * k)
    labels = ["new-collision"] * k + ["new-far"] * k
    after = register_new_classes(before, collision_rows, labels)
    bias = dict(zip(after.classes, after.score_bias.tolist()))
    assert bias["new-collision"] > bias["new-far"]
    assert bias["new-far"] == pytest.approx(0.0)
    assert after.support_audit["collision_bias_source"] == "old_registered_support_prototypes_only"


def test_class_and_domain_permutations_do_not_change_matched_scores() -> None:
    old_z, old_y = _old_support(seed=19)
    reference = fit_old_ciaf(_component(), old_z, old_y)
    permutation = np.random.default_rng(5).permutation(len(old_y))
    shuffled = fit_old_ciaf(
        _component((2, 0, 1)), old_z[permutation], np.asarray(old_y)[permutation]
    )
    probe = _direction(1) + 0.03 * _direction(12)
    np.testing.assert_allclose(score_one(reference, probe), score_one(shuffled, probe), atol=1e-7)
    np.testing.assert_allclose(
        np.sort(reference.old_domain_weights, axis=1),
        np.sort(shuffled.old_domain_weights, axis=1),
        atol=1e-7,
    )


def test_predict_one_scores_all_registered_and_rejects_batches() -> None:
    old_z, old_y = _old_support()
    before = fit_old_ciaf(_component(), old_z, old_y)
    new_z, new_y = _new_support()
    state = register_new_classes(before, new_z, new_y)
    label, scores = predict_one(state, _direction(20))
    assert label == "new-x"
    assert scores.shape == (len(state.classes),)
    assert not scores.flags.writeable
    with pytest.raises(CiafError, match="exactly one"):
        score_one(state, np.stack([_direction(0), _direction(1)]))


def test_public_fit_and_predict_surfaces_have_no_forbidden_oracle_inputs() -> None:
    forbidden = ("query", "truth", "role", "quota", "assignment", "source", "clean")
    for function in (fit_old_ciaf, register_new_classes, score_one, predict_one):
        parameters = inspect.signature(function).parameters
        assert not any(token in name.lower() for name in parameters for token in forbidden)


def test_invalid_full_precision_or_incomplete_component_fails_closed() -> None:
    q = np.zeros((2, 3, FEATURE_DIM), dtype=np.float32)
    scale = np.ones((2, 3), dtype=np.float16)
    mask = np.ones((2, 3), dtype=np.uint8)
    with pytest.raises(CiafError, match="int8"):
        Int8DomainClassComponent(q, scale, mask, OLD)
    q8 = q.astype(np.int8)
    mask[:, 0] = 0
    with pytest.raises(CiafError, match="component drift"):
        Int8DomainClassComponent(q8, scale, mask, OLD)


def test_registration_requires_disjoint_labels_and_same_k() -> None:
    old_z, old_y = _old_support(k=5)
    state = fit_old_ciaf(_component(), old_z, old_y)
    with pytest.raises(CiafError, match="overlap"):
        register_new_classes(state, np.stack([_direction(0)] * 5), ["old-a"] * 5)
    with pytest.raises(CiafError, match="K-shot"):
        register_new_classes(state, np.stack([_direction(50)] * 4), ["new-z"] * 4)
