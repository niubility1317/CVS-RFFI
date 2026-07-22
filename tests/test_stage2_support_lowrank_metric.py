from __future__ import annotations

import inspect

import numpy as np
import pytest

import cvsrffi.stage2_support_lowrank_metric as d5


def _support(
    *,
    classes: tuple[str, ...] = ("tx-a", "tx-b", "tx-c"),
    k: int = 5,
    dim: int = 18,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, label in enumerate(classes):
        center = np.zeros(dim, dtype=np.float32)
        center[index : index + 3] = 3.0
        rows.extend(
            center + rng.normal(scale=0.12, size=dim).astype(np.float32)
            for _ in range(k)
        )
        labels.extend([label] * k)
    return np.stack(rows), np.asarray(labels)


def _scenario_support(
    *,
    classes: tuple[str, ...] = ("tx-a", "tx-b", "tx-c"),
    k: int = 5,
    dim: int = 18,
) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]]:
    output = {}
    for scenario_index, scenario in enumerate(d5.FORMAL_LEO_WEAK_SCENARIOS):
        rows, labels = _support(
            classes=classes,
            k=k,
            dim=dim,
            seed=17 + scenario_index,
        )
        iq = np.zeros((len(rows), 2, 16), dtype=np.float32)
        iq[:, 0, 0] = scenario_index + 1
        iq[:, 1, 0] = np.arange(len(rows), dtype=np.float32) + 0.25
        ids = tuple(
            f"{scenario}-physical-{index}" for index in range(len(rows))
        )
        output[scenario] = (
            rows,
            labels,
            ids,
            d5.received_iq_sha256(iq),
        )
    return output


def test_closed_form_support_selection_has_no_query_or_role_interface():
    rows, labels = _support(k=5)
    state = d5.fit_support_lowrank_metric(
        rows,
        labels,
        scenario="leo_clear_weak",
        rank_candidates=(4, 8),
        shrinkage_candidates=(0.2, 0.8),
    )
    signature = inspect.signature(d5.fit_support_lowrank_metric)
    assert "query" not in " ".join(signature.parameters).lower()
    assert "role" not in " ".join(signature.parameters).lower()
    assert state.rank in {4, 8}
    assert state.shrinkage in {0.2, 0.8}
    assert len(state.selection_trace) == 4
    assert {
        row["validation_protocol"] for row in state.selection_trace
    } == {"support_leave_two_out_per_class"}
    resource = state.resource_audit()
    assert resource["adaptation_epochs"] == 0
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["role_oracle_access"] is False
    assert resource["class_quota_access"] is False
    assert resource["dense_query_graph_bytes"] == 0


def test_before_projection_and_old_head_are_bitwise_locked_after_registration():
    old_x, old_y = _support(k=5)
    before = d5.fit_support_lowrank_metric(
        old_x,
        old_y,
        scenario="leo_low_elev_weak",
        rank_candidates=(8,),
        shrinkage_candidates=(0.6,),
    )
    old_projection = before.projection.copy()
    old_prototypes = before.prototypes.copy()
    new_x, new_y = _support(
        classes=("tx-new-1", "tx-new-2"),
        k=5,
        dim=old_x.shape[1],
        seed=51,
    )
    mixed_x = np.concatenate([old_x[:3], new_x], axis=0)
    mixed_y = np.concatenate([old_y[:3], new_y], axis=0)
    after = d5.register_absent_classes(before, mixed_x, mixed_y)
    np.testing.assert_array_equal(after.projection, old_projection)
    np.testing.assert_array_equal(after.center, before.center)
    np.testing.assert_array_equal(
        after.prototypes[: len(before.classes)], old_prototypes
    )
    assert after.classes[: len(before.classes)] == before.classes
    assert set(after.classes[len(before.classes) :]) == {
        "tx-new-1",
        "tx-new-2",
    }
    assert after.resource_audit()["old_projection_reused_after_registration"]
    assert after.resource_audit()["old_prototypes_reused_after_registration"]


def test_prediction_is_per_sample_all_registered_and_batch_local():
    rows, labels = _support(k=5)
    before = d5.fit_support_lowrank_metric(
        rows,
        labels,
        scenario="leo_rain_weak",
        rank_candidates=(8,),
        shrinkage_candidates=(0.6,),
    )
    new_x, new_y = _support(
        classes=("tx-new",),
        k=5,
        dim=rows.shape[1],
        seed=73,
    )
    after = d5.register_absent_classes(before, new_x, new_y)
    query = np.stack([rows[0], new_x[0]])
    first = d5.predict_all_registered(after, query)
    extra = np.full((1, rows.shape[1]), 9.0, dtype=np.float32)
    extended = d5.predict_all_registered(
        after, np.concatenate([query, extra], axis=0)
    )
    assert first.scores.shape == (2, len(after.classes))
    np.testing.assert_array_equal(first.scores, extended.scores[:2])
    assert first.labels == extended.labels[:2]
    assert set(first.labels).issubset(set(after.classes))


def test_scenario_atomic_arm_is_unified_but_fit_is_not_concatenated():
    support = _scenario_support(k=5, dim=18)
    state = d5.fit_scenario_atomic_lowrank(
        support,
        rank_candidates=(4, 8),
        shrinkage_candidates=(0.2, 0.8),
    )
    assert len(state.selection_trace) == 4
    assert {item.rank for item in state.states} == {state.rank}
    assert {item.shrinkage for item in state.states} == {state.shrinkage}
    assert {
        item.support_rows_used_for_fit for item in state.states
    } == {15}
    resource = state.resource_audit()
    assert resource["cross_scenario_support_concat"] is False
    assert resource["trainable_parameter_limit_pass"] is True
    assert resource["persistent_state_limit_pass"] is True
    assert resource["trainable_parameters"] == 18 * state.rank * 3


def test_scenario_atomic_rejects_cross_scenario_physical_or_iq_reuse():
    support = _scenario_support(k=5)
    first = d5.FORMAL_LEO_WEAK_SCENARIOS[0]
    second = d5.FORMAL_LEO_WEAK_SCENARIOS[1]
    rows, labels, ids, hashes = support[second]
    bad_ids = (support[first][2][0],) + ids[1:]
    bad = dict(support)
    bad[second] = rows, labels, bad_ids, hashes
    with pytest.raises(
        d5.SupportLowRankMetricError, match="physical support reuse"
    ):
        d5.fit_scenario_atomic_lowrank(
            bad, rank_candidates=(8,), shrinkage_candidates=(0.6,)
        )

    bad_hashes = (support[first][3][0],) + hashes[1:]
    bad = dict(support)
    bad[second] = rows, labels, ids, bad_hashes
    with pytest.raises(
        d5.SupportLowRankMetricError, match="received-IQ reuse"
    ):
        d5.fit_scenario_atomic_lowrank(
            bad, rank_candidates=(8,), shrinkage_candidates=(0.6,)
        )


def test_288_by_rank_by_three_resource_caps_and_locked_k1_path():
    six_classes = tuple(f"tx-{index}" for index in range(6))
    support = _scenario_support(classes=six_classes, k=5, dim=288)
    state = d5.fit_scenario_atomic_lowrank(
        support,
        rank_candidates=(8,),
        shrinkage_candidates=(0.6,),
    )
    assert state.trainable_parameters == 288 * 8 * 3
    assert state.trainable_parameters < d5.MAX_TRAINABLE_PARAMETERS
    assert state.persistent_state_bytes < d5.MAX_PERSISTENT_STATE_BYTES

    k1_x, k1_y = _support(classes=six_classes, k=1, dim=288)
    locked = d5.fit_support_lowrank_metric(
        k1_x,
        k1_y,
        scenario="leo_clear_weak",
        locked_rank=8,
        locked_shrinkage=0.6,
    )
    assert locked.selection_trace == ()
    assert (
        locked.selection_protocol
        == "locked_from_support_only_development_selection"
    )


def test_invalid_k1_selection_and_parameter_overflow_fail_closed():
    rows, labels = _support(k=1, dim=18)
    with pytest.raises(
        d5.SupportLowRankMetricError, match="at least two samples per class"
    ):
        d5.fit_support_lowrank_metric(
            rows,
            labels,
            scenario="leo_clear_weak",
            rank_candidates=(4,),
            shrinkage_candidates=(0.6,),
        )

    large_x, large_y = _support(k=5, dim=3000)
    with pytest.raises(
        d5.SupportLowRankMetricError, match="trainable parameter cap"
    ):
        d5.fit_support_lowrank_metric(
            large_x,
            large_y,
            scenario="leo_clear_weak",
            locked_rank=32,
            locked_shrinkage=0.6,
        )
