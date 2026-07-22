from __future__ import annotations

import inspect

import numpy as np
import pytest

import cvsrffi.stage2_local_contrastive_boundary as d7


def _support(
    *,
    classes: tuple[str, ...] = ("a", "b", "c"),
    k: int = 6,
    dim: int = 12,
    seed: int = 1,
):
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    ids = []
    for class_index, label in enumerate(classes):
        center = np.zeros(dim, dtype=np.float32)
        center[class_index * 2 : class_index * 2 + 2] = 2.0
        for sample_index in range(k):
            rows.append(
                center
                + rng.normal(scale=0.15, size=dim).astype(np.float32)
            )
            labels.append(label)
            ids.append(f"{label}-physical-{sample_index}")
    return np.stack(rows), np.asarray(labels), tuple(ids)


def test_nearest_rival_comes_only_from_support_prototype_gram():
    rows, labels, ids = _support()
    head = d7.fit_local_contrastive_boundary(
        rows,
        labels,
        physical_sample_ids=ids,
        beta_candidates=(0.0,),
    )
    gram = head.prototypes @ head.prototypes.T
    np.fill_diagonal(gram, -np.inf)
    expected = np.argmax(gram, axis=1)
    np.testing.assert_array_equal(head.rival_indices[:, 0], expected)
    assert head.support_audit["query_rows_used"] == 0


def test_local_margin_score_matches_formula():
    prototypes = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
    )
    classes = np.asarray(["a", "b", "c"])
    rivals = np.asarray([[1], [0], [1]], dtype=np.int64)
    beta = np.asarray([0.2, 0.1, 0.0], dtype=np.float32)
    head = d7.LocalBoundaryHead(
        schema="test",
        classes=classes,
        prototypes=prototypes,
        rival_indices=rivals,
        beta=beta,
        old_class_count=3,
        support_audit={},
    )
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    base = query @ prototypes.T
    expected = base.copy()
    for index in range(3):
        rival = base[0, rivals[index, 0]]
        expected[0, index] += beta[index] * (
            base[0, index] - rival
        )
    np.testing.assert_allclose(
        d7.score_local_contrastive_boundary(query, head),
        expected,
        rtol=0.0,
        atol=1.0e-7,
    )


def test_beta_selection_uses_leave_two_and_never_degrades_class():
    rows, labels, ids = _support(k=6)
    head = d7.fit_local_contrastive_boundary(
        rows,
        labels,
        physical_sample_ids=ids,
        beta_candidates=(0.0, 0.05, 0.10),
    )
    evidence = head.support_audit["beta_selection"]
    assert {
        row["protocol"] for row in evidence["candidate_evidence"]
    } == {"support_leave_two_physical_samples_out"}
    selected = next(
        row
        for row in evidence["candidate_evidence"]
        if row["beta"] == evidence["selected_beta"]
    )
    assert selected["eligible"]
    assert selected["per_class_non_degradation_pass"]


def test_after_registration_bitwise_locks_old_state():
    old_x, old_y, old_ids = _support(classes=("a", "b", "c"))
    parent = d7.fit_local_contrastive_boundary(
        old_x,
        old_y,
        physical_sample_ids=old_ids,
        beta_candidates=(0.0, 0.05),
    )
    new_x, new_y, new_ids = _support(
        classes=("n1", "n2"),
        dim=old_x.shape[1],
        seed=22,
    )
    mixed_x = np.concatenate([old_x, new_x], axis=0)
    mixed_y = np.concatenate([old_y, new_y])
    mixed_ids = old_ids + new_ids
    child = d7.extend_local_contrastive_boundary(
        parent,
        mixed_x,
        mixed_y,
        physical_sample_ids=mixed_ids,
        beta_candidates=(0.0, 0.05),
    )
    old_count = len(parent.classes)
    np.testing.assert_array_equal(
        child.prototypes[:old_count], parent.prototypes
    )
    np.testing.assert_array_equal(
        child.rival_indices[:old_count], parent.rival_indices
    )
    np.testing.assert_array_equal(child.beta[:old_count], parent.beta)
    assert np.all(child.rival_indices[:old_count] < old_count)
    assert child.support_audit["old_state_bitwise_locked"]
    extension = child.support_audit["beta_selection_for_new_classes"]
    assert extension["old_state_during_selection"] == "bitwise_locked"
    assert {
        row["protocol"] for row in extension["candidate_evidence"]
    } == {"new_support_leave_two_out_old_state_locked"}
    selected = next(
        row
        for row in extension["candidate_evidence"]
        if row["beta"] == extension["selected_beta"]
    )
    assert selected["per_class_non_degradation_pass"]
    assert selected["old_floor_non_degradation_pass"]
    assert selected["new_floor_non_degradation_pass"]
    assert child.classes.tolist() == ["a", "b", "c", "n1", "n2"]


def test_query_is_batch_local_all_registered_and_oracle_free():
    rows, labels, ids = _support()
    head = d7.fit_local_contrastive_boundary(
        rows,
        labels,
        physical_sample_ids=ids,
        beta_candidates=(0.0, 0.05),
    )
    query = rows[:2]
    first = d7.score_local_contrastive_boundary(query, head)
    extended = d7.score_local_contrastive_boundary(
        np.concatenate(
            [query, np.full((1, rows.shape[1]), 4.0, dtype=np.float32)]
        ),
        head,
    )
    np.testing.assert_array_equal(first, extended[:2])
    assert first.shape == (2, len(head.classes))
    assert d7.public_interface_is_query_oracle_free()
    signature = str(
        inspect.signature(d7.score_local_contrastive_boundary)
    ).lower()
    assert "role" not in signature
    assert "quota" not in signature
    assert "label" not in signature


def test_resource_is_extremely_small_and_no_dense_graph():
    rows, labels, ids = _support(dim=288)
    head = d7.fit_local_contrastive_boundary(
        rows,
        labels,
        physical_sample_ids=ids,
        beta_candidates=(0.0,),
    )
    audit = head.resource_audit()
    assert audit["trainable_parameters"] == 0
    assert audit["adaptation_epochs"] == 0
    assert audit["persistent_state_limit_pass"]
    assert audit["dense_query_graph_bytes"] == 0
    assert audit["rival_count_per_class"] == 1


def test_duplicate_physical_support_and_invalid_beta_fail_closed():
    rows, labels, ids = _support()
    duplicate = ids[:-1] + (ids[0],)
    with pytest.raises(
        d7.LocalContrastiveBoundaryError, match="unique physical IDs"
    ):
        d7.fit_local_contrastive_boundary(
            rows,
            labels,
            physical_sample_ids=duplicate,
        )
    with pytest.raises(
        d7.LocalContrastiveBoundaryError, match="begin at zero"
    ):
        d7.fit_local_contrastive_boundary(
            rows,
            labels,
            physical_sample_ids=ids,
            beta_candidates=(0.05, 0.10),
        )
