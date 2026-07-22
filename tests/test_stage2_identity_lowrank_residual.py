from __future__ import annotations

import inspect

import numpy as np
import pytest

import cvsrffi.stage2_identity_lowrank_residual as d6b
from cvsrffi.stage2_support_lowrank_metric import (
    FORMAL_LEO_WEAK_SCENARIOS,
    received_iq_sha256,
)


def _support(classes=("a", "b", "c"), k=5, dim=24, seed=10):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for index, label in enumerate(classes):
        center = np.zeros(dim, dtype=np.float32)
        center[index * 2 : index * 2 + 2] = 3.0
        rows.extend(
            center + rng.normal(0.0, 0.15, dim).astype(np.float32)
            for _ in range(k)
        )
        labels.extend([label] * k)
    return np.stack(rows), np.asarray(labels)


def _scenario(classes=("a", "b", "c"), k=5, dim=24):
    output = {}
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        rows, labels = _support(
            classes=classes, k=k, dim=dim, seed=20 + scenario_index
        )
        iq = np.zeros((len(rows), 2, 8), dtype=np.float32)
        iq[:, 0, 0] = scenario_index + 1
        iq[:, 1, 0] = np.arange(len(rows), dtype=np.float32)
        output[scenario] = (
            rows,
            labels,
            tuple(f"{scenario}-{i}" for i in range(len(rows))),
            received_iq_sha256(iq),
        )
    return output


def test_selection_is_support_only_and_identity_fallback_is_present():
    state = d6b.fit_scenario_atomic_identity_residual(
        _scenario(),
        rank_candidates=(4, 8),
        shrinkage_candidates=(0.3,),
        residual_weight_candidates=(0.0, 0.1, 0.2),
    )
    signature = inspect.signature(
        d6b.fit_scenario_atomic_identity_residual
    )
    assert "query" not in " ".join(signature.parameters).lower()
    assert "role" not in " ".join(signature.parameters).lower()
    assert state.residual_weight in {0.0, 0.1, 0.2}
    assert any(row["residual_weight"] == 0.0 for row in state.selection_trace)
    chosen = [
        row
        for row in state.selection_trace
        if row["rank"] == state.rank
        and row["shrinkage"] == state.shrinkage
        and row["residual_weight"] == state.residual_weight
    ][0]
    assert chosen["eligible"] is True
    for row in chosen["per_scenario"].values():
        assert row["floor_non_degradation_pass"] is True
        assert row["overall_tolerance_pass"] is True


def test_zero_residual_is_exact_identity_cosine():
    state = d6b.fit_scenario_atomic_identity_residual(
        _scenario(),
        rank_candidates=(4,),
        shrinkage_candidates=(0.3,),
        residual_weight_candidates=(0.0,),
    ).state_for("leo_clear_weak")
    query = np.stack([state.identity_prototypes[0], state.identity_prototypes[1]])
    result = d6b.predict_all_registered(state, query)
    np.testing.assert_array_equal(result.scores, result.identity_scores)
    assert result.labels == state.classes[:2]


def test_after_registration_locks_before_identity_and_lowrank_state():
    before = d6b.fit_scenario_atomic_identity_residual(
        _scenario(),
        rank_candidates=(4,),
        shrinkage_candidates=(0.3,),
        residual_weight_candidates=(0.0, 0.1),
    )
    new_support = _scenario(classes=("new-1", "new-2"), k=5)
    after = d6b.register_scenario_atomic_absent_classes(
        before, new_support
    )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        old = before.state_for(scenario)
        new = after.state_for(scenario)
        np.testing.assert_array_equal(
            new.identity_prototypes[: len(old.classes)],
            old.identity_prototypes,
        )
        np.testing.assert_array_equal(
            new.lowrank_state.projection,
            old.lowrank_state.projection,
        )
        np.testing.assert_array_equal(
            new.lowrank_state.prototypes[: len(old.classes)],
            old.lowrank_state.prototypes,
        )
        assert new.residual_weight == old.residual_weight


def test_query_prediction_is_batch_local_all_registered_and_no_update():
    state = d6b.fit_scenario_atomic_identity_residual(
        _scenario(),
        rank_candidates=(4,),
        shrinkage_candidates=(0.3,),
        residual_weight_candidates=(0.0, 0.1),
    ).state_for("leo_rain_weak")
    query = np.stack(
        [state.identity_prototypes[0], state.identity_prototypes[-1]]
    )
    first = d6b.predict_all_registered(state, query)
    extended = d6b.predict_all_registered(
        state,
        np.concatenate(
            [query, np.full((1, state.feature_dim), 5.0, dtype=np.float32)]
        ),
    )
    np.testing.assert_array_equal(first.scores, extended.scores[:2])
    assert first.labels == extended.labels[:2]
    resource = state.resource_audit()
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["query_updates"] == 0
    assert resource["role_oracle_access"] is False
    assert resource["class_quota_access"] is False


def test_resource_caps_and_no_cross_scenario_concat():
    classes = tuple(f"tx-{i}" for i in range(6))
    state = d6b.fit_scenario_atomic_identity_residual(
        _scenario(classes=classes, dim=288),
        rank_candidates=(8,),
        shrinkage_candidates=(0.6,),
        residual_weight_candidates=(0.0, 0.1),
    )
    resource = state.resource_audit()
    assert state.trainable_parameters == 288 * 8 * 3
    assert resource["cross_scenario_support_concat"] is False
    assert resource["trainable_parameter_limit_pass"] is True
    assert resource["persistent_state_limit_pass"] is True
    assert resource["identity_primary"] is True
    assert resource["pure_lowrank_replacement"] is False


def test_invalid_grid_and_cross_scenario_reuse_fail_closed():
    with pytest.raises(
        d6b.IdentityLowRankResidualError, match="must include identity fallback"
    ):
        d6b.fit_scenario_atomic_identity_residual(
            _scenario(),
            rank_candidates=(4,),
            shrinkage_candidates=(0.3,),
            residual_weight_candidates=(0.1,),
        )
    support = _scenario()
    first, second = FORMAL_LEO_WEAK_SCENARIOS[:2]
    rows, labels, ids, hashes = support[second]
    bad = dict(support)
    bad[second] = rows, labels, (support[first][2][0],) + ids[1:], hashes
    with pytest.raises(
        d6b.IdentityLowRankResidualError, match="cross-scenario support reuse"
    ):
        d6b.fit_scenario_atomic_identity_residual(
            bad,
            rank_candidates=(4,),
            shrinkage_candidates=(0.3,),
            residual_weight_candidates=(0.0,),
        )
