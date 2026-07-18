from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d34_collision_local_registration import (
    D34CollisionLocalConfig,
    D34CollisionLocalRegistrationError,
    fit_d34_collision_local_registration,
    score_d34_collision_local_registration,
)


DIM = 288
OLD_COUNT = 6


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return np.asarray(rows / np.linalg.norm(rows, axis=1, keepdims=True), dtype=np.float32)


def _fixture(new_count: int, k: int, *, seed: int = 34):
    rng = np.random.default_rng(seed + 100 * new_count + k)
    old_classes = tuple(f"old_{i}" for i in range(OLD_COUNT))
    new_classes = tuple(f"new_{i}" for i in range(new_count))
    old_centers = _normalize(rng.normal(size=(OLD_COUNT, DIM)))
    new_centers = []
    for index in range(new_count):
        residual = _normalize(rng.normal(size=(1, DIM)))[0]
        new_centers.append(
            _normalize(
                (np.float32(0.82) * old_centers[index % OLD_COUNT]
                 + np.float32(0.18) * residual)[None, :]
            )[0]
        )
    new_centers = np.stack(new_centers).astype(np.float32)
    old_rows = []
    old_labels = []
    for index, name in enumerate(old_classes):
        rows = _normalize(
            old_centers[index][None, :]
            + np.float32(0.025) * rng.normal(size=(k, DIM)).astype(np.float32)
        )
        old_rows.append(rows)
        old_labels.extend([name] * k)
    new_rows = []
    new_labels = []
    for index, name in enumerate(new_classes):
        rows = _normalize(
            new_centers[index][None, :]
            + np.float32(0.03) * rng.normal(size=(k, DIM)).astype(np.float32)
        )
        new_rows.append(rows)
        new_labels.extend([name] * k)
    old_x = np.concatenate(old_rows).astype(np.float32)
    new_x = np.concatenate(new_rows).astype(np.float32)
    old_prefix = np.asarray(np.float32(18.0) * (old_x @ old_centers.T), dtype=np.float32)
    new_old_prefix = np.asarray(
        np.float32(18.0) * (new_x @ old_centers.T), dtype=np.float32
    )
    return (
        old_x,
        old_labels,
        old_classes,
        old_prefix,
        new_x,
        new_labels,
        new_classes,
        new_old_prefix,
    )


def _fit(new_count: int, k: int, arm: str):
    args = _fixture(new_count, k)
    return (
        fit_d34_collision_local_registration(
            *args,
            config=D34CollisionLocalConfig(arm=arm),
        ),
        args,
    )


@pytest.mark.parametrize("new_count", [2, 5, 10, 20])
@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_all_scales_are_int8_sparse_finite_and_old_prefix_safe(
    new_count: int, arm: str
) -> None:
    result, args = _fit(new_count, 3, arm)
    old_x, _, old_classes, old_prefix, new_x, _, new_classes, new_prefix = args
    state = result.state
    assert state.classes == old_classes + new_classes
    assert state.new_prototypes_qint8.shape == (new_count, DIM)
    assert state.new_prototypes_qint8.dtype == np.int8
    assert state.new_prototype_scales.dtype == np.float32
    assert state.new_prototype_inverse_norms.dtype == np.float32
    assert not state.new_prototypes_qint8.flags.writeable
    degrees_by_new = np.sum(state.collision_edge_mask, axis=1)
    if arm == "A":
        np.testing.assert_array_equal(degrees_by_new, np.ones(new_count))
    elif arm == "B":
        np.testing.assert_array_equal(degrees_by_new, np.full(new_count, 2))
    else:
        assert bool(np.all((degrees_by_new >= 1) & (degrees_by_new <= 3)))

    old_scores = score_d34_collision_local_registration(state, old_x, old_prefix)
    np.testing.assert_array_equal(old_scores[:, :OLD_COUNT], old_prefix)
    before_correct = np.argmax(old_prefix, axis=1) == np.repeat(
        np.arange(OLD_COUNT), 3
    )
    after_correct = np.argmax(old_scores, axis=1) == np.repeat(
        np.arange(OLD_COUNT), 3
    )
    assert not bool(np.any(before_correct & ~after_correct))
    new_scores = score_d34_collision_local_registration(state, new_x, new_prefix)
    assert new_scores.shape == (new_count * 3, OLD_COUNT + new_count)
    assert np.isfinite(new_scores).all()

    old_anchor_degrees = np.sum(state.collision_edge_mask, axis=0)
    resource = result.resource_audit
    assert resource["estimated_macs_per_query"] == int(
        np.max(old_anchor_degrees) * (DIM + 2)
    )
    assert resource["estimated_macs_per_query_average_degree"] == pytest.approx(
        float(np.mean(old_anchor_degrees) * (DIM + 2))
    )
    assert resource["active_parameters"] < 50_000
    assert resource["resident_fp32_new_prototype_count"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["query_role_oracle_access"] is False
    assert resource["query_true_batch_class_count_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["query_batch_global_assignment"] is False
    assert resource["clean_sample_access"] is False
    assert resource["source_sample_access"] is False


def test_k1_has_no_pseudo_loso_and_uses_fixed_uncertainty() -> None:
    result, _ = _fit(2, 1, "C")
    geometry = result.geometry_audit
    assert geometry["old_loso_trace"] == [
        {
            "mode": "k1_no_pseudo_loso",
            "intrusion_evaluated": False,
            "method_lock_uncertainty": 0.05,
        }
    ]
    assert all(row["mode"] == "k1_no_pseudo_loso" for row in geometry["new_loso_trace"])
    assert all(
        row["mode"] == "k1_method_lock"
        for row in geometry["old_uncertainty_trace"]
    )
    assert result.state.optimizer_steps == 0
    assert geometry["old_loso_evaluated"] is False
    assert geometry["old_loso_status"] == "NOT_EVALUATED_K1"
    assert geometry["old_loso_zero_intrusion_pass"] is False


def test_new_scores_use_same_temperature_as_frozen_old_prefix() -> None:
    result, args = _fit(5, 3, "A")
    _, _, _, _, new_x, _, _, new_prefix = args
    state = result.state
    scores = score_d34_collision_local_registration(state, new_x[:1], new_prefix[:1])
    winner = int(np.argmax(new_prefix[0]))
    adjacent = np.flatnonzero(state.collision_edge_mask[:, winner])
    assert len(adjacent) >= 1
    new_index = int(adjacent[0])
    expected = (
        np.float32(18.0)
        * np.float32(new_x[0] @ state.new_prototypes_qint8[new_index].astype(np.float32))
        * state.new_prototype_inverse_norms[new_index]
        + state.old_anchor_offsets[winner]
    )
    np.testing.assert_allclose(scores[0, OLD_COUNT + new_index], expected, atol=2.0e-6)
    nonadjacent = np.flatnonzero(~state.collision_edge_mask[:, winner])
    if len(nonadjacent):
        np.testing.assert_array_equal(
            scores[0, OLD_COUNT + nonadjacent],
            np.full(
                len(nonadjacent),
                new_prefix[0, winner] - np.float32(2.0),
                dtype=np.float32,
            ),
        )


def test_real_old_and_new_loso_traces_are_complete() -> None:
    result, _ = _fit(5, 3, "B")
    geometry = result.geometry_audit
    assert len(geometry["old_loso_trace"]) == OLD_COUNT * 3
    assert all(
        row["mode"] == "leave_one_old_out_rebuild_offset"
        for row in geometry["old_loso_trace"]
    )
    assert len(geometry["new_loso_trace"]) == 5
    assert all(
        row["mode"] == "physical_support_leave_one_out"
        for row in geometry["new_loso_trace"]
    )
    assert geometry["old_loso_zero_intrusion_pass"] == (
        geometry["old_loso_intrusion_count"] == 0
    )


def test_rejects_non_unit_rows_k_mismatch_and_unapproved_arm() -> None:
    args = list(_fixture(2, 2))
    bad_old = np.asarray(args[0]).copy()
    bad_old[0] *= np.float32(2.0)
    args[0] = bad_old
    with pytest.raises(D34CollisionLocalRegistrationError, match="unit rows"):
        fit_d34_collision_local_registration(*args)

    args = list(_fixture(2, 2))
    args[4] = args[4][:-1]
    args[5] = args[5][:-1]
    args[7] = args[7][:-1]
    with pytest.raises(D34CollisionLocalRegistrationError, match="K-shot"):
        fit_d34_collision_local_registration(*args)

    with pytest.raises(D34CollisionLocalRegistrationError, match="arm"):
        D34CollisionLocalConfig(arm="D")
