from __future__ import annotations

import inspect

import numpy as np
import pytest

import cvsrffi.stage2_class_conditional_iq_head as d7a
import cvsrffi.stage2_floor_sparse_operator_fusion as d9
from cvsrffi.stage2_support_lowrank_metric import received_iq_sha256


def _iq_labels(
    classes=("20-19", "1-18", "14-10"),
    k=6,
    length=64,
    seed=91,
):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    time = np.arange(length, dtype=np.float32)
    for class_index, label in enumerate(classes):
        for sample_index in range(k):
            phase = 0.19 * sample_index + rng.uniform(-0.1, 0.1)
            tone = np.exp(
                1j
                * (
                    (0.031 + 0.017 * class_index) * time
                    + phase
                )
            )
            tone += (0.15 * class_index) + 1j * (-0.08 * class_index)
            tone *= 0.8 + 0.15 * ((sample_index + class_index) % 3)
            tone += 0.045 * (
                rng.normal(size=length)
                + 1j * rng.normal(size=length)
            )
            rows.append(np.stack([tone.real, tone.imag]))
            labels.append(label)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels)


def _extractor(rows):
    complex_rows = rows[:, 0] + 1j * rows[:, 1]
    spectrum = np.abs(np.fft.fft(complex_rows, axis=1))[:, :12]
    stats = np.stack(
        [
            rows[:, 0].mean(axis=1),
            rows[:, 1].mean(axis=1),
            rows.std(axis=(1, 2)),
            np.mean(np.abs(complex_rows), axis=1),
        ],
        axis=1,
    )
    return np.concatenate([spectrum, stats], axis=1).astype(np.float32)


def _inputs(
    classes=("20-19", "1-18", "14-10"), k=10, seed=91
):
    iq, labels = _iq_labels(classes=classes, k=k, seed=seed)
    features = d7a.extract_operator_features(
        iq, feature_extractor=_extractor
    )
    ids = tuple(f"physical-{seed}-{index}" for index in range(len(iq)))
    hashes = received_iq_sha256(iq)
    provenance = d9.build_operator_feature_provenance(
        hashes, view_seed=seed
    )
    return iq, labels, features, provenance, ids, hashes


def _fit(classes=("20-19", "1-18", "14-10"), k=10, seed=91):
    iq, labels, features, provenance, ids, hashes = _inputs(
        classes=classes, k=k, seed=seed
    )
    state = d9.fit_floor_sparse_operator_fusion(
        features,
        provenance,
        labels,
        physical_sample_ids=ids,
        parent_received_iq_sha256=hashes,
        base_resource_audit={
            "persistent_state_bytes": 4096,
            "estimated_head_macs_per_query": 1024,
        },
    )
    return iq, labels, features, provenance, ids, hashes, state


def test_fixed_candidate_registry_is_sparse_convex_and_small():
    assert len(d9.FIXED_CANDIDATES) == 12
    assert d9.BASE_CANDIDATE.candidate_id == "single_base"
    for candidate in d9.FIXED_CANDIDATES:
        active = [
            weight for weight in candidate.weights if weight > 0.0
        ]
        assert 1 <= len(active) <= 2
        assert sum(active) == pytest.approx(1.0)
        assert set(active).issubset({0.25, 0.5, 0.75, 1.0})
        assert all(
            -1 <= index < len(d7a.OPERATORS)
            for index in candidate.operator_indices
        )


def test_support_only_selection_marks_floor_classes_and_enforces_gates():
    _iq, _labels, _features, _provenance, _ids, _hashes, state = _fit()
    selection = state.support_audit["selection"]
    assert selection["query_rows_used"] == 0
    assert selection["query_labels_used"] is False
    assert selection["query_roles_used"] is False
    assert selection["query_quota_used"] is False
    assert (
        selection["combined_final"]["overall_accuracy"]
        >= selection["baseline"]["overall_accuracy"]
    )
    assert (
        selection["combined_final"]["min_class_accuracy"]
        >= selection["baseline"]["min_class_accuracy"]
    )
    for label in state.classes:
        assert (
            selection["combined_final"]["per_class_accuracy"][label]
            >= selection["baseline"]["per_class_accuracy"][label]
        )
    rows = {
        row["class_handle"]: row
        for row in selection["per_class_selection"]
    }
    assert rows["20-19"]["floor_priority"] is True
    assert rows["1-18"]["floor_priority"] is True
    for row in rows.values():
        chosen = next(
            evidence
            for evidence in row["candidate_evidence"]
            if evidence["candidate_id"]
            == row["selected_candidate_id"]
        )
        assert chosen["eligible"] is True
        assert chosen["active_operator_count"] <= 2


def test_query_formula_one_forward_per_used_operator_and_batch_local():
    iq, _labels, _features, _provenance, _ids, _hashes, state = _fit()
    calls = []

    def counted(rows):
        calls.append(len(rows))
        return _extractor(rows)

    sealed_counted = d9.seal_samplewise_feature_extractor(
        counted,
        extractor_id="synthetic-fft-stats-v1",
        validation_rows=iq[:3],
    )
    sealed_plain = d9.seal_samplewise_feature_extractor(
        _extractor,
        extractor_id="synthetic-fft-stats-v1",
        validation_rows=iq[:3],
    )
    calls.clear()
    first = d9.predict_floor_sparse_operator_fusion(
        state, iq[:2], feature_extractor=sealed_counted
    )
    extended = d9.predict_floor_sparse_operator_fusion(
        state, iq[:3], feature_extractor=sealed_plain
    )
    assert len(calls) == len(state.used_operators)
    assert len(calls) <= 3
    np.testing.assert_allclose(first.scores, extended.scores[:2], atol=1e-6)

    query_features = d7a.extract_operator_features(
        iq[:2],
        feature_extractor=_extractor,
        operator_ids=state.used_operators,
    )
    normalized = {
        operator: rows
        / np.maximum(
            np.linalg.norm(rows, axis=1, keepdims=True), 1e-8
        )
        for operator, rows in query_features.items()
    }
    manual = np.zeros_like(first.scores)
    for class_index in range(state.class_count):
        for slot in range(2):
            weight = float(state.weights[class_index, slot])
            operator_index = int(
                state.operator_indices[class_index, slot]
            )
            if weight <= 0.0:
                continue
            calibration = state.calibration_for(operator_index)
            component = (
                normalized[d7a.OPERATORS[operator_index]]
                @ state.prototypes[class_index, slot]
                - calibration.center
            ) / calibration.scale
            manual[:, class_index] += weight * component
    np.testing.assert_allclose(first.scores, manual, atol=1e-6)
    with pytest.raises(
        d9.FloorSparseOperatorFusionError, match="samplewise sealed"
    ):
        d9.predict_floor_sparse_operator_fusion(
            state, iq[:2], feature_extractor=_extractor
        )
    def batch_dependent(rows):
        features = _extractor(rows)
        return features - features.mean(axis=0, keepdims=True)

    with pytest.raises(
        d9.FloorSparseOperatorFusionError, match="batch-dependent"
    ):
        d9.seal_samplewise_feature_extractor(
            batch_dependent,
            extractor_id="forbidden-batch-dependent",
            validation_rows=iq[:3],
        )


def test_after_registration_only_appends_and_locks_old_state_bitwise():
    (
        _old_iq,
        old_labels,
        old_features,
        old_provenance,
        old_ids,
        old_hashes,
        before,
    ) = _fit(classes=("20-19", "14-10", "14-7"), seed=101)
    (
        _new_iq,
        new_labels,
        new_features,
        new_provenance,
        new_ids,
        new_hashes,
    ) = _inputs(classes=("1-18", "18-10"), seed=103)
    all_features = {
        operator: np.concatenate(
            [old_features[operator], new_features[operator]], axis=0
        )
        for operator in d7a.OPERATORS
    }
    all_provenance = {
        operator: old_provenance[operator] + new_provenance[operator]
        for operator in d7a.OPERATORS
    }
    after = d9.extend_floor_sparse_operator_fusion(
        before,
        all_features,
        all_provenance,
        np.concatenate([old_labels, new_labels]),
        physical_sample_ids=old_ids + new_ids,
        parent_received_iq_sha256=old_hashes + new_hashes,
    )
    old_count = before.class_count
    assert after.classes[:old_count] == before.classes
    np.testing.assert_array_equal(
        after.operator_indices[:old_count], before.operator_indices
    )
    np.testing.assert_array_equal(
        after.weights[:old_count], before.weights
    )
    np.testing.assert_array_equal(
        after.prototypes[:old_count], before.prototypes
    )
    assert after.calibrations == before.calibrations
    assert after.support_audit["old_state_bitwise_locked"] is True
    selection = after.support_audit["selection"]
    assert selection["query_rows_used"] == 0
    assert selection[
        "global_overall_floor_and_old_class_non_degradation_pass"
    ] in {True, False}
    final_old = selection["combined_final_old"]
    baseline_old = selection["baseline_old"]
    assert final_old["overall_accuracy"] >= baseline_old["overall_accuracy"]
    for label in before.classes:
        assert (
            final_old["per_class_accuracy"][label]
            >= baseline_old["per_class_accuracy"][label]
        )


def test_resources_and_query_api_obey_deployment_contract():
    _iq, _labels, _features, _provenance, _ids, _hashes, state = _fit()
    resource = state.resource_audit()
    assert resource["adaptation_epochs"] == 0
    assert resource["optimizer_steps"] == 0
    assert resource["trainable_parameters"] == 0
    assert resource["trainable_parameter_limit"] == 50_000
    assert resource["persistent_state_limit_pass"] is True
    assert resource["base_persistent_state_bytes"] == 4096
    assert resource["combined_persistent_state_bytes"] == (
        resource["base_persistent_state_bytes"]
        + resource["d9_incremental_state_bytes"]
    )
    assert resource["base_head_macs_per_query"] == 1024
    assert resource["combined_head_macs_per_query"] == (
        resource["base_head_macs_per_query"]
        + resource["d9_incremental_head_macs_per_query"]
    )
    assert resource["backbone_forwards_per_query"] <= 3
    assert resource["views_count_as_additional_k"] is False
    assert resource["additional_leo_channel_states_generated"] == 0
    assert resource["clean_sample_access"] is False
    assert resource["clean_derived_signal_access"] is False
    assert resource["source_sample_access"] is False
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["query_role_oracle_access"] is False
    assert resource["query_true_batch_class_count_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["query_batch_global_assignment"] is False
    assert d9.public_query_interface_is_oracle_free()
    parameters = " ".join(
        inspect.signature(
            d9.predict_floor_sparse_operator_fusion
        ).parameters
    ).lower()
    assert all(
        token not in parameters
        for token in ("label", "truth", "role", "quota", "graph")
    )


def test_k10_lock_rebuilds_only_prototypes_for_nested_k1_and_k5():
    (
        _iq,
        labels,
        features,
        provenance,
        ids,
        hashes,
        locked,
    ) = _fit()
    for nested_k in (1, 5):
        indices = np.asarray(
            [
                index
                for label in locked.classes
                for index in np.flatnonzero(labels == label).tolist()[
                    :nested_k
                ]
            ],
            dtype=np.int64,
        )
        nested_features = {
            operator: rows[indices]
            for operator, rows in features.items()
        }
        nested_provenance = {
            operator: tuple(
                provenance[operator][index]
                for index in indices.tolist()
            )
            for operator in d7a.OPERATORS
        }
        nested = d9.rebuild_locked_floor_sparse_prototypes(
            locked,
            nested_features,
            nested_provenance,
            labels[indices],
            physical_sample_ids=tuple(ids[index] for index in indices),
            parent_received_iq_sha256=tuple(
                hashes[index] for index in indices
            ),
        )
        assert nested.current_k == nested_k
        assert nested.selection_lock_k == 10
        assert (
            nested.selection_lock_sha256
            == locked.selection_lock_sha256
        )
        assert nested.operator_indices is locked.operator_indices
        assert nested.weights is locked.weights
        assert nested.calibrations == locked.calibrations
        assert nested.support_audit[
            "selection_reused_without_reselection"
        ] is True
        assert nested.support_audit["only_prototypes_rebuilt"] is True


def test_nested_rebuild_accepts_append_only_after_registry_order():
    (
        _old_iq,
        old_labels,
        old_features,
        old_provenance,
        old_ids,
        old_hashes,
        before,
    ) = _fit(classes=("z-old", "m-old"), seed=131)
    (
        _new_iq,
        new_labels,
        new_features,
        new_provenance,
        new_ids,
        new_hashes,
    ) = _inputs(classes=("a-new", "b-new"), seed=133)
    all_labels = np.concatenate([old_labels, new_labels])
    all_ids = old_ids + new_ids
    all_hashes = old_hashes + new_hashes
    all_features = {
        operator: np.concatenate(
            [old_features[operator], new_features[operator]], axis=0
        )
        for operator in d7a.OPERATORS
    }
    all_provenance = {
        operator: old_provenance[operator] + new_provenance[operator]
        for operator in d7a.OPERATORS
    }
    after = d9.extend_floor_sparse_operator_fusion(
        before,
        all_features,
        all_provenance,
        all_labels,
        physical_sample_ids=all_ids,
        parent_received_iq_sha256=all_hashes,
    )
    assert after.classes == (
        "m-old",
        "z-old",
        "a-new",
        "b-new",
    )
    indices = np.asarray(
        [
            index
            for label in after.classes
            for index in np.flatnonzero(all_labels == label).tolist()[:1]
        ],
        dtype=np.int64,
    )
    nested = d9.rebuild_locked_floor_sparse_prototypes(
        after,
        {
            operator: rows[indices]
            for operator, rows in all_features.items()
        },
        {
            operator: tuple(
                all_provenance[operator][index]
                for index in indices.tolist()
            )
            for operator in d7a.OPERATORS
        },
        all_labels[indices],
        physical_sample_ids=tuple(all_ids[index] for index in indices),
        parent_received_iq_sha256=tuple(
            all_hashes[index] for index in indices
        ),
    )
    assert nested.classes == after.classes
    assert nested.current_k == 1


def test_operator_provenance_base_resource_and_old_lineage_fail_closed():
    (
        _iq,
        labels,
        features,
        provenance,
        ids,
        hashes,
        state,
    ) = _fit()
    bad_provenance = {
        operator: list(rows) for operator, rows in provenance.items()
    }
    bad_provenance[d7a.BASE][0] = {
        **bad_provenance[d7a.BASE][0],
        "operator_id": d7a.DC_RMS,
    }
    with pytest.raises(
        d9.FloorSparseOperatorFusionError, match="provenance"
    ):
        d9.fit_floor_sparse_operator_fusion(
            features,
            bad_provenance,
            labels,
            physical_sample_ids=ids,
            parent_received_iq_sha256=hashes,
            base_resource_audit={
                "persistent_state_bytes": 4096,
                "estimated_head_macs_per_query": 1024,
            },
        )
    with pytest.raises(
        d9.FloorSparseOperatorFusionError, match="state cap"
    ):
        d9.fit_floor_sparse_operator_fusion(
            features,
            provenance,
            labels,
            physical_sample_ids=ids,
            parent_received_iq_sha256=hashes,
            base_resource_audit={
                "persistent_state_bytes": 256 * 1024,
                "estimated_head_macs_per_query": 1024,
            },
        )

    (
        _new_iq,
        new_labels,
        new_features,
        new_provenance,
        new_ids,
        new_hashes,
    ) = _inputs(classes=("new-a", "new-b"), seed=121)
    all_features = {
        operator: np.concatenate(
            [features[operator], new_features[operator]], axis=0
        )
        for operator in d7a.OPERATORS
    }
    all_provenance = {
        operator: provenance[operator] + new_provenance[operator]
        for operator in d7a.OPERATORS
    }
    drifted_old_ids = list(ids + new_ids)
    first_class_indices = np.flatnonzero(
        np.concatenate([labels, new_labels]) == state.classes[0]
    )
    left, right = first_class_indices[:2]
    drifted_old_ids[left], drifted_old_ids[right] = (
        drifted_old_ids[right],
        drifted_old_ids[left],
    )
    with pytest.raises(
        d9.FloorSparseOperatorFusionError, match="old support lineage"
    ):
        d9.extend_floor_sparse_operator_fusion(
            state,
            all_features,
            all_provenance,
            np.concatenate([labels, new_labels]),
            physical_sample_ids=tuple(drifted_old_ids),
            parent_received_iq_sha256=hashes + new_hashes,
        )


def test_lineage_operator_and_registration_class_drift_fail_closed():
    _iq, labels, features, provenance, ids, hashes, state = _fit()
    with pytest.raises(
        d9.FloorSparseOperatorFusionError, match="lineage"
    ):
        d9.fit_floor_sparse_operator_fusion(
            features,
            provenance,
            labels,
            physical_sample_ids=(ids[0],) + ids[1:-1] + (ids[0],),
            parent_received_iq_sha256=hashes,
            base_resource_audit={
                "persistent_state_bytes": 4096,
                "estimated_head_macs_per_query": 1024,
            },
        )
    bad = dict(features)
    bad["forbidden"] = bad.pop(d7a.DC_RMS_SPEC15)
    with pytest.raises(
        d9.FloorSparseOperatorFusionError, match="operator set"
    ):
        d9.fit_floor_sparse_operator_fusion(
            bad,
            provenance,
            labels,
            physical_sample_ids=ids,
            parent_received_iq_sha256=hashes,
            base_resource_audit={
                "persistent_state_bytes": 4096,
                "estimated_head_macs_per_query": 1024,
            },
        )
    (
        _new_iq,
        new_labels,
        new_features,
        new_provenance,
        new_ids,
        new_hashes,
    ) = _inputs(
        classes=("new-a", "new-b"), seed=111
    )
    with pytest.raises(
        d9.FloorSparseOperatorFusionError, match="retain all old"
    ):
        d9.extend_floor_sparse_operator_fusion(
            state,
            new_features,
            new_provenance,
            new_labels,
            physical_sample_ids=new_ids,
            parent_received_iq_sha256=new_hashes,
        )
