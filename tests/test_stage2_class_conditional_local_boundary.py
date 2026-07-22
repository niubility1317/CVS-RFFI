from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

import cvsrffi.stage2_class_conditional_iq_head as d7a
import cvsrffi.stage2_class_conditional_local_boundary as d7c
from cvsrffi.stage2_support_lowrank_metric import received_iq_sha256


def _iq_labels(classes=("a", "b", "c"), k=6, length=64, seed=31):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    time = np.arange(length, dtype=np.float32)
    for class_index, label in enumerate(classes):
        for sample_index in range(k):
            phase = 0.25 * sample_index + rng.uniform(-0.08, 0.08)
            signal = np.exp(
                1j
                * (
                    (0.035 + 0.018 * class_index) * time
                    + phase
                )
            )
            signal += (0.12 * class_index) + 1j * (-0.07 * class_index)
            signal += 0.04 * (
                rng.normal(size=length) + 1j * rng.normal(size=length)
            )
            rows.append(np.stack([signal.real, signal.imag]))
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


def _inputs(classes=("a", "b", "c"), k=6, seed=31):
    iq, labels = _iq_labels(classes=classes, k=k, seed=seed)
    ids = tuple(f"physical-{seed}-{index}" for index in range(len(iq)))
    hashes = received_iq_sha256(iq)
    artifact = d7a.build_validated_operator_feature_artifact(
        iq,
        feature_extractor=_extractor,
        physical_sample_ids=ids,
        parent_received_iq_sha256=hashes,
    )
    return iq, labels, artifact, ids, hashes


def _fit_before(classes=("a", "b", "c"), k=6, seed=31):
    iq, labels, artifact, ids, hashes = _inputs(
        classes=classes, k=k, seed=seed
    )
    base = d7a.fit_class_conditional_head(
        artifact,
        labels,
    )
    combined = d7c.fit_class_conditional_local_boundary(
        base,
        artifact,
        labels,
    )
    return iq, labels, artifact, ids, hashes, base, combined


def test_d7c_reuses_d7a_state_and_matches_manual_score_formula():
    iq, _labels, _features, _ids, _hashes, base, combined = _fit_before()
    assert combined.base_state is base
    prediction = d7c.predict_class_conditional_local_boundary(
        combined,
        iq[:4],
        feature_extractor=_extractor,
        physical_sample_ids=_ids[:4],
        parent_received_iq_sha256=_hashes[:4],
    )
    manual = prediction.base_scores + combined.beta[None, :] * (
        prediction.base_scores
        - prediction.base_scores[:, combined.rival_indices]
    )
    np.testing.assert_allclose(prediction.scores, manual, atol=1e-6)
    assert prediction.operators_computed == base.used_operators
    assert set(prediction.labels).issubset(set(base.classes))


def test_query_uses_one_backbone_forward_per_deduplicated_operator():
    iq, _labels, _features, _ids, _hashes, base, combined = _fit_before()
    calls = []

    def counted_extractor(rows):
        calls.append(len(rows))
        return _extractor(rows)

    first = d7c.predict_class_conditional_local_boundary(
        combined,
        iq[:2],
        feature_extractor=counted_extractor,
        physical_sample_ids=_ids[:2],
        parent_received_iq_sha256=_hashes[:2],
    )
    extended = d7c.predict_class_conditional_local_boundary(
        combined,
        iq[:3],
        feature_extractor=_extractor,
        physical_sample_ids=_ids[:3],
        parent_received_iq_sha256=_hashes[:3],
    )
    assert len(calls) == 2 * len(base.used_operators)
    assert set(calls) == {1}
    np.testing.assert_allclose(first.scores, extended.scores[:2], atol=1e-6)
    resource = combined.resource_audit()
    assert (
        resource["backbone_forwards_per_query"]
        == resource["used_operator_count"]
        == len(base.used_operators)
    )
    assert resource["views_count_as_additional_k"] is False
    assert resource["additional_leo_channel_states_generated"] == 0


def test_batch_coupled_callback_cannot_observe_other_query_rows():
    iq, _labels, _artifact, ids, hashes, _base, combined = _fit_before()

    def batch_coupled_extractor(rows):
        value = _extractor(rows)
        return value - value.mean(axis=0, keepdims=True)

    first = d7c.predict_class_conditional_local_boundary(
        combined,
        iq[:1],
        feature_extractor=batch_coupled_extractor,
        physical_sample_ids=ids[:1],
        parent_received_iq_sha256=hashes[:1],
    )
    extended = d7c.predict_class_conditional_local_boundary(
        combined,
        iq[:3],
        feature_extractor=batch_coupled_extractor,
        physical_sample_ids=ids[:3],
        parent_received_iq_sha256=hashes[:3],
    )
    np.testing.assert_array_equal(first.scores, extended.scores[:1])


def test_support_selection_is_physical_deletion_only_and_non_degrading():
    _iq, _labels, _features, _ids, _hashes, _base, combined = _fit_before()
    selection = combined.support_audit["selection"]
    assert selection["query_rows_used"] == 0
    assert selection["query_labels_used"] is False
    assert selection["query_roles_used"] is False
    assert selection["query_quota_used"] is False
    assert (
        selection["rival_source"]
        == "heldout_d7a_calibrated_class_confusion"
    )
    final = selection["combined_final"]
    baseline = selection["baseline"]
    assert final["overall_accuracy"] >= baseline["overall_accuracy"]
    assert final["min_class_accuracy"] >= baseline["min_class_accuracy"]
    for row in selection["per_class_selection"]:
        selected = next(
            evidence
            for evidence in row["candidate_evidence"]
            if evidence["beta"] == row["selected_beta"]
        )
        assert selected["eligible"] is True


def test_after_registration_bitwise_locks_old_d7a_and_d7c_state():
    (
        old_iq,
        old_labels,
        old_artifact,
        old_ids,
        old_hashes,
        old_base,
        old_combined,
    ) = _fit_before(classes=("old-a", "old-b", "old-c"), seed=41)
    (
        new_iq,
        new_labels,
        new_artifact,
        new_ids,
        new_hashes,
    ) = _inputs(classes=("new-a", "new-b"), seed=43)
    extended_base = d7a.register_absent_classes(
        old_base,
        new_artifact,
        new_labels,
    )
    all_iq = np.concatenate([old_iq, new_iq], axis=0)
    all_ids = old_ids + new_ids
    all_hashes = old_hashes + new_hashes
    all_artifact = d7a.build_validated_operator_feature_artifact(
        all_iq,
        feature_extractor=_extractor,
        physical_sample_ids=all_ids,
        parent_received_iq_sha256=all_hashes,
    )
    extended = d7c.extend_class_conditional_local_boundary(
        old_combined,
        extended_base,
        all_artifact,
        np.concatenate([old_labels, new_labels]),
    )
    old_count = len(old_base.classes)
    assert extended.base_state.classes[:old_count] == old_base.classes
    assert (
        extended.base_state.class_operators[:old_count]
        == old_base.class_operators
    )
    np.testing.assert_array_equal(
        extended.base_state.prototypes[:old_count], old_base.prototypes
    )
    assert extended.base_state.calibrations == old_base.calibrations
    np.testing.assert_array_equal(
        extended.rival_indices[:old_count],
        old_combined.rival_indices,
    )
    np.testing.assert_array_equal(
        extended.beta[:old_count], old_combined.beta
    )
    assert np.all(extended.rival_indices[:old_count] < old_count)
    assert extended.support_audit["old_state_bitwise_locked"] is True
    assert (
        extended.support_audit["selection"]["query_rows_used"] == 0
    )
    assert len(old_iq) + len(new_iq) == len(old_labels) + len(new_labels)


def test_resources_and_public_query_api_obey_deployment_contract():
    _iq, _labels, _features, _ids, _hashes, _base, combined = _fit_before()
    resource = combined.resource_audit()
    assert resource["adaptation_epochs"] == 0
    assert resource["trainable_parameters"] == 0
    assert resource["trainable_parameter_limit"] == 50_000
    assert resource["persistent_state_bytes"] == combined.persistent_state_bytes
    assert resource["end_to_end_macs_per_query"] is None
    assert resource["head_macs_scope"].endswith("local_margin_only")
    assert resource["persistent_state_limit_pass"] is True
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["query_decision_policy"] == (
        "per_sample_all_registered_classes"
    )
    assert resource["query_role_oracle_access"] is False
    assert resource["query_true_batch_class_count_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["query_batch_global_assignment"] is False
    assert resource["query_feature_extractor_batch_size"] == 1
    assert resource["query_query_feature_interaction_possible"] is False
    assert d7c.public_query_interface_is_oracle_free()
    query_signature = " ".join(
        inspect.signature(
            d7c.predict_class_conditional_local_boundary
        ).parameters
    ).lower()
    assert all(
        token not in query_signature
        for token in ("label", "truth", "role", "quota", "graph")
    )


def test_inherited_resource_drift_fails_closed():
    _iq, labels, artifact, _ids, _hashes, base, _combined = _fit_before()
    for mutated in (
        replace(base, trainable_parameters=60_000),
        replace(base, query_rows_used_for_fit=1),
        replace(base, query_updates=1),
        replace(base, persistent_state_bytes=1),
    ):
        with pytest.raises(
            d7c.ClassConditionalLocalBoundaryError,
            match="resource or query-fit",
        ):
            d7c.fit_class_conditional_local_boundary(
                mutated, artifact, labels
            )


def test_joint_beta_gate_never_degrades_any_registered_class():
    rng = np.random.default_rng(20260717)
    classes = ("a", "b", "c")
    truth = np.repeat(np.arange(3), 10)
    rivals = np.asarray([1, 2, 0], dtype=np.int64)
    for _ in range(100):
        scores = rng.normal(size=(30, 3)).astype(np.float32)
        scores[np.arange(30), truth] += rng.uniform(
            -0.2, 0.8, size=30
        )
        _selected, audit = d7c._select_betas_before(
            ({"scores": scores, "truth": truth},),
            rivals,
            classes,
            (0.0, 0.5),
        )
        baseline = audit["baseline"]["per_class_accuracy"]
        final = audit["combined_final"]["per_class_accuracy"]
        assert all(final[label] >= baseline[label] for label in classes)


def test_k10_lock_allows_k1_k5_prototype_only_rebuild():
    (
        _iq10,
        _labels10,
        _artifact10,
        _ids10,
        _hashes10,
        _base10,
        state10,
    ) = _fit_before(k=10, seed=81)
    locked = d7c.lock_k10_class_conditional_local_boundary_strategy(
        state10
    )
    for k, seed in ((1, 82), (5, 83)):
        iq, labels, artifact, ids, hashes = _inputs(
            classes=locked.classes, k=k, seed=seed
        )
        rebuilt = d7c.rebuild_from_locked_k10_strategy(
            locked, artifact, labels, expected_k=k
        )
        assert (
            rebuilt.base_state.class_operators
            == locked.base_state.class_operators
        )
        assert rebuilt.base_state.calibrations == locked.base_state.calibrations
        np.testing.assert_array_equal(
            rebuilt.rival_indices, locked.rival_indices
        )
        np.testing.assert_array_equal(rebuilt.beta, locked.beta)
        assert rebuilt.support_audit["prototype_rebuild_k"] == k
        assert rebuilt.support_audit["operator_reselected"] is False
        assert rebuilt.support_audit["rival_reselected"] is False
        assert rebuilt.support_audit["beta_reselected"] is False
        prediction = d7c.predict_class_conditional_local_boundary(
            rebuilt,
            iq[:1],
            feature_extractor=_extractor,
            physical_sample_ids=ids[:1],
            parent_received_iq_sha256=hashes[:1],
        )
        assert prediction.scores.shape == (1, len(locked.classes))


def test_duplicate_lineage_operator_drift_and_old_mutation_fail_closed():
    iq, labels, artifact, ids, hashes, base, combined = _fit_before()
    with pytest.raises(
        d7c.ClassConditionalLocalBoundaryError, match="artifact"
    ):
        d7c.fit_class_conditional_local_boundary(
            base,
            artifact.feature_map(),
            labels,
        )

    new_iq, new_labels, new_artifact, new_ids, new_hashes = _inputs(
        classes=("new-a", "new-b"), seed=51
    )
    extended_base = d7a.register_absent_classes(
        base,
        new_artifact,
        new_labels,
    )
    mutated = d7a.ClassConditionalIQHeadState(
        **{
            **extended_base.__dict__,
            "class_operators": (
                d7a.DC_RMS
                if extended_base.class_operators[0] != d7a.DC_RMS
                else d7a.BASE,
            )
            + extended_base.class_operators[1:],
        }
    )
    all_artifact = d7a.build_validated_operator_feature_artifact(
        np.concatenate([iq, new_iq], axis=0),
        feature_extractor=_extractor,
        physical_sample_ids=ids + new_ids,
        parent_received_iq_sha256=hashes + new_hashes,
    )
    with pytest.raises(
        d7c.ClassConditionalLocalBoundaryError, match="bitwise locked"
    ):
        d7c.extend_class_conditional_local_boundary(
            combined,
            mutated,
            all_artifact,
            np.concatenate([labels, new_labels]),
        )


def test_after_rejects_changed_parent_old_support_lineage():
    iq, labels, _artifact, ids, hashes, base, combined = _fit_before(seed=61)
    new_iq, new_labels, new_artifact, new_ids, new_hashes = _inputs(
        classes=("new-a", "new-b"), seed=62
    )
    extended_base = d7a.register_absent_classes(
        base, new_artifact, new_labels
    )
    changed_old = iq.copy()
    changed_old[0, 0, 0] += 0.25
    changed_hashes = received_iq_sha256(changed_old)
    changed_artifact = d7a.build_validated_operator_feature_artifact(
        np.concatenate([changed_old, new_iq], axis=0),
        feature_extractor=_extractor,
        physical_sample_ids=ids + new_ids,
        parent_received_iq_sha256=changed_hashes + new_hashes,
    )
    with pytest.raises(
        d7c.ClassConditionalLocalBoundaryError,
        match="old support lineage",
    ):
        d7c.extend_class_conditional_local_boundary(
            combined,
            extended_base,
            changed_artifact,
            np.concatenate([labels, new_labels]),
        )
    shifted_artifact = d7a.build_validated_operator_feature_artifact(
        np.concatenate([iq, new_iq], axis=0),
        feature_extractor=lambda rows: _extractor(rows) + 0.125,
        physical_sample_ids=ids + new_ids,
        parent_received_iq_sha256=hashes + new_hashes,
    )
    with pytest.raises(
        d7c.ClassConditionalLocalBoundaryError,
        match="feature binding drift",
    ):
        d7c.extend_class_conditional_local_boundary(
            combined,
            extended_base,
            shifted_artifact,
            np.concatenate([labels, new_labels]),
        )
