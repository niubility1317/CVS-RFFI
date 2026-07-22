from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import cvsrffi.stage2_collision_aware_multiproto as d5


def _ids(count: int, prefix: str = "p") -> list[str]:
    return [f"{prefix}-{index:03d}" for index in range(count)]


def test_bimodal_class_can_select_multiple_prototypes_from_physical_loo() -> None:
    support = np.asarray(
        [
            [1.0, 0.12, 0.0],
            [0.98, -0.10, 0.0],
            [-1.0, 0.10, 0.0],
            [-0.98, -0.12, 0.0],
            [0.12, 1.0, 0.0],
            [-0.10, 0.98, 0.0],
            [0.10, -1.0, 0.0],
            [-0.12, -0.98, 0.0],
        ],
        dtype=np.float32,
    )
    labels = ["a"] * 4 + ["b"] * 4
    head = d5.fit_collision_aware_multiproto(
        support,
        labels,
        physical_sample_ids=_ids(len(support)),
        config=d5.D5Config(
            residual_shrinkage=1.0,
            collision_penalty_weight=0.0,
            loo_gain_required=0.0,
            margin_gain_required=0.0,
            compactness_gain_required=0.0,
            max_collision_worsening=2.0,
            complexity_penalty=0.0,
        ),
    )
    assert head.prototype_count_by_class.max() >= 2
    assert all(
        item["physical_loo_available"] == 1.0
        for value in head.support_audit["per_class"].values()
        for item in value["candidate_diagnostics"]
    )
    assert head.support_audit["computation_views_used_as_physical_loo"] is False


def test_compact_unimodal_classes_conservatively_fall_back() -> None:
    rng = np.random.default_rng(11)
    support = np.concatenate(
        (
            np.asarray([1.0, 0.0, 0.0])[None, :]
            + 0.005 * rng.normal(size=(10, 3)),
            np.asarray([0.0, 1.0, 0.0])[None, :]
            + 0.005 * rng.normal(size=(10, 3)),
        ),
        axis=0,
    ).astype(np.float32)
    head = d5.fit_collision_aware_multiproto(
        support,
        ["a"] * 10 + ["b"] * 10,
        physical_sample_ids=_ids(20),
    )
    assert head.prototype_count_by_class.tolist() == [1, 1]


def test_query_scoring_is_batch_composition_invariant_and_label_blind() -> None:
    support = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    head = d5.fit_collision_aware_multiproto(
        support,
        ["a", "a", "b", "b"],
        physical_sample_ids=_ids(4),
    )
    query = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    together = d5.score_collision_aware_multiproto(query, head)
    separate = np.concatenate(
        [
            d5.score_collision_aware_multiproto(row[None, :], head)
            for row in query
        ],
        axis=0,
    )
    np.testing.assert_array_equal(together, separate)
    parameters = inspect.signature(
        d5.score_collision_aware_multiproto
    ).parameters
    assert set(parameters) == {"query_features", "head"}


def test_after_locks_old_state_and_uses_bounded_support_only_deconfusion() -> None:
    before_support = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.98, 0.0],
        ],
        dtype=np.float32,
    )
    parent = d5.fit_collision_aware_multiproto(
        before_support,
        ["a", "a", "b", "b"],
        physical_sample_ids=_ids(4, "before"),
    )
    after_support = np.concatenate(
        (
            before_support,
            np.asarray(
                [
                    [0.88, 0.20, 0.0],
                    [0.86, 0.23, 0.0],
                    [-0.1, -0.95, 0.0],
                    [-0.05, -0.98, 0.0],
                ],
                dtype=np.float32,
            ),
        ),
        axis=0,
    )
    child = d5.extend_collision_aware_multiproto(
        parent,
        after_support,
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        physical_sample_ids=_ids(8, "after"),
    )
    old_count = parent.class_count
    np.testing.assert_array_equal(
        child.prototypes[:old_count], parent.prototypes
    )
    np.testing.assert_array_equal(
        child.prototype_mask[:old_count], parent.prototype_mask
    )
    np.testing.assert_array_equal(
        child.centroids[:old_count], parent.centroids
    )
    np.testing.assert_array_equal(
        child.class_penalty[:old_count], parent.class_penalty
    )
    np.testing.assert_array_equal(child.residual_scale, parent.residual_scale)
    audit = child.support_audit
    assert audit["old_head_bitwise_locked"] is True
    assert audit["old_head_update_count"] == 0
    assert audit["bounded_deconfusion"]["query_rows_used"] == 0
    assert (
        audit["bounded_deconfusion"]["selected"][
            "old_before_correct_after_intruded"
        ]
        == 0
    )


def test_resource_audit_stays_inside_deployment_caps() -> None:
    rng = np.random.default_rng(7)
    support = rng.normal(size=(26 * 10, 288)).astype(np.float32)
    labels = np.repeat(
        np.asarray([f"tx-{index:02d}" for index in range(26)]), 10
    )
    head = d5.fit_collision_aware_multiproto(
        support,
        labels,
        physical_sample_ids=_ids(len(support)),
    )
    resource = head.resource_audit()
    assert resource["trainable_parameters"] == 0
    assert resource["persistent_state_limit_pass"] is True
    assert resource["persistent_state_bytes_fp16"] < 256 * 1024
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["estimated_extra_macs_per_query"] > 0
    assert np.all(head.prototype_count_by_class <= 3)


def test_duplicate_physical_support_is_rejected() -> None:
    support = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    with pytest.raises(
        d5.CollisionAwareMultiPrototypeError,
        match="unique physical sample",
    ):
        d5.fit_collision_aware_multiproto(
            support,
            ["a", "a", "b", "b"],
            physical_sample_ids=["same", "same", "p2", "p3"],
        )


def test_d6c_margin_is_selected_from_support_only_and_committed(
    tmp_path,
) -> None:
    parent_rows = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.04, 0.0],
            [0.96, -0.03, 0.0],
            [0.0, 1.0, 0.0],
            [0.04, 0.98, 0.0],
            [-0.03, 0.96, 0.0],
        ],
        dtype=np.float32,
    )
    parent = d5.fit_collision_aware_multiproto(
        parent_rows,
        ["a"] * 3 + ["b"] * 3,
        physical_sample_ids=_ids(6, "parent"),
    )
    after_rows = np.concatenate(
        (
            parent_rows,
            np.asarray(
                [
                    [0.83, 0.22, 0.0],
                    [0.81, 0.24, 0.0],
                    [0.79, 0.20, 0.0],
                    [0.82, 0.18, 0.0],
                ],
                dtype=np.float32,
            ),
        )
    )
    selection = d5.select_support_only_margin(
        parent,
        after_rows,
        ["a"] * 3 + ["b"] * 3 + ["c"] * 4,
        physical_sample_ids=_ids(10, "after"),
        margin_candidates=(0.0, 0.01),
    )
    assert selection.selected_margin in {0.0, 0.01}
    evidence = selection.selection_evidence
    assert evidence["query_package_opened"] is False
    assert evidence["query_rows_used"] == 0
    assert evidence["query_truth_used"] is False
    for candidate in evidence["candidate_evidence"]:
        assert (
            candidate["new_physical_leave_one_two_out"]["per_class"]["c"][
                "leave_one_rows"
            ]
            == 4
        )
        assert (
            candidate["new_physical_leave_one_two_out"]["per_class"]["c"][
                "leave_two_rows"
            ]
            == 4
        )
    commit = d5.write_support_only_margin_commit(
        selection, tmp_path / "margin_lock"
    )
    assert commit["status"] == "SUPPORT_ONLY_MARGIN_LOCKED_BEFORE_QUERY_OPEN"
    decoded = json.loads(
        (tmp_path / "margin_lock" / "COMMIT.json").read_text("utf-8")
    )
    assert decoded["query_package_opened"] is False
    assert decoded["selected_margin"] == selection.selected_margin
