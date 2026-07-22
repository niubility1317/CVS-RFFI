from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

import cvsrffi.stage2_class_conditional_iq_head as d7a
from cvsrffi.stage2_support_lowrank_metric import received_iq_sha256


def _iq_labels(classes=("a", "b", "c"), k=6, length=64, seed=4):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    time = np.arange(length, dtype=np.float32)
    for index, label in enumerate(classes):
        for _ in range(k):
            phase = rng.uniform(-0.2, 0.2)
            signal = np.exp(1j * (0.05 * (index + 1) * time + phase))
            signal += 0.05 * (
                rng.normal(size=length) + 1j * rng.normal(size=length)
            )
            signal += (0.2 * index) + 1j * (-0.1 * index)
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
        ],
        axis=1,
    )
    return np.concatenate([spectrum, stats], axis=1).astype(np.float32)


def _inputs(classes=("a", "b", "c"), k=6, seed=4):
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


def test_operators_are_fixed_received_iq_only_and_no_cfo():
    iq, _labels, _features, _ids, _hashes = _inputs()
    assert d7a.OPERATORS == ("base", "dc_rms", "dc_rms_spec15")
    assert all("cfo" not in value for value in d7a.OPERATORS)
    base = d7a.apply_received_iq_operator(iq, d7a.BASE)
    dc = d7a.apply_received_iq_operator(iq, d7a.DC_RMS)
    spec = d7a.apply_received_iq_operator(iq, d7a.DC_RMS_SPEC15)
    np.testing.assert_array_equal(base, iq)
    assert np.max(np.abs(dc.mean(axis=2))) < 1e-5
    assert dc.shape == spec.shape == iq.shape
    assert np.isfinite(spec).all()


def test_per_class_leave_two_selection_and_global_non_degradation():
    _iq, labels, artifact, _ids, _hashes = _inputs()
    state = d7a.fit_class_conditional_head(
        artifact,
        labels,
    )
    assert len(state.class_operators) == len(state.classes)
    assert set(state.class_operators).issubset(set(d7a.OPERATORS))
    assert state.selection_trace[-1]["global_non_degradation_pass"] in {
        True,
        False,
    }
    for row in state.selection_trace[:-1]:
        selected = row["operators"][row["selected_operator"]]
        assert selected["class_non_degradation_pass"] is True
        assert selected["overall_tolerance_pass"] is True


def test_query_computes_only_registry_used_operators_and_is_batch_local():
    iq, labels, artifact, ids, hashes = _inputs()
    state = d7a.fit_class_conditional_head(
        artifact,
        labels,
    )
    calls = []

    def extractor(rows):
        calls.append(len(rows))
        return _extractor(rows)

    first = d7a.predict_all_registered(
        state,
        iq[:2],
        feature_extractor=extractor,
        physical_sample_ids=ids[:2],
        parent_received_iq_sha256=hashes[:2],
    )
    assert len(calls) == 2 * len(state.used_operators)
    assert set(calls) == {1}
    extended = d7a.predict_all_registered(
        state,
        iq[:3],
        feature_extractor=_extractor,
        physical_sample_ids=ids[:3],
        parent_received_iq_sha256=hashes[:3],
    )
    np.testing.assert_allclose(first.scores, extended.scores[:2], atol=1e-6)
    assert first.operators_computed == state.used_operators
    assert set(first.labels).issubset(set(state.classes))


def test_after_locks_old_operator_prototype_and_calibration():
    _iq, labels, artifact, _ids, _hashes = _inputs()
    before = d7a.fit_class_conditional_head(
        artifact,
        labels,
    )
    _new_iq, new_labels, new_artifact, _new_ids, _new_hashes = _inputs(
        classes=("new-1", "new-2"), seed=9
    )
    after = d7a.register_absent_classes(
        before,
        new_artifact,
        new_labels,
    )
    assert after.class_operators[: len(before.classes)] == before.class_operators
    np.testing.assert_array_equal(
        after.prototypes[: len(before.classes)], before.prototypes
    )
    assert after.calibrations == before.calibrations
    assert after.classes[: len(before.classes)] == before.classes
    assert after.resource_audit()["old_state_locked_after_registration"]


def test_api_has_no_query_label_role_or_quota_fit_and_resources_are_tiny():
    signature = inspect.signature(d7a.fit_class_conditional_head)
    text = " ".join(signature.parameters).lower()
    assert "query" not in text
    assert "role" not in text
    assert "quota" not in text
    _iq, labels, artifact, _ids, _hashes = _inputs()
    state = d7a.fit_class_conditional_head(
        artifact,
        labels,
    )
    resource = state.resource_audit()
    assert resource["trainable_parameters"] == 0
    assert resource["persistent_state_limit_pass"] is True
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["dense_query_graph_bytes"] == 0


def test_duplicate_lineage_and_operator_drift_fail_closed():
    iq, labels, artifact, ids, hashes = _inputs()
    with pytest.raises(
        d7a.ClassConditionalIQHeadError, match="lineage"
    ):
        d7a.build_validated_operator_feature_artifact(
            iq,
            feature_extractor=_extractor,
            physical_sample_ids=(ids[0],) + ids[1:-1] + (ids[0],),
            parent_received_iq_sha256=hashes,
        )
    with pytest.raises(
        d7a.ClassConditionalIQHeadError, match="lineage"
    ):
        d7a.build_validated_operator_feature_artifact(
            iq,
            feature_extractor=_extractor,
            physical_sample_ids=ids,
            parent_received_iq_sha256=("0" * 64,) + hashes[1:],
        )
    with pytest.raises(
        (d7a.ClassConditionalIQHeadError, AttributeError, TypeError)
    ):
        d7a.fit_class_conditional_head(artifact.feature_map(), labels)


def test_validated_artifact_records_every_sample_operator_binding():
    _iq, _labels, artifact, ids, hashes = _inputs()
    assert len(artifact.bindings) == len(ids) * len(d7a.OPERATORS)
    assert {
        binding.operator_id for binding in artifact.bindings
    } == set(d7a.OPERATORS)
    assert {binding.view_seed for binding in artifact.bindings} == {0}
    assert {
        binding.parent_received_iq_sha256 for binding in artifact.bindings
    } == set(hashes)


def test_tampered_random_feature_payload_fails_artifact_seal():
    _iq, labels, artifact, _ids, _hashes = _inputs()
    rng = np.random.default_rng(20260717)
    rows = []
    for operator, value in artifact._operator_features:
        random_value = rng.normal(size=value.shape).astype(np.float32)
        random_value.setflags(write=False)
        rows.append((operator, random_value))
    tampered = replace(artifact, _operator_features=tuple(rows))
    with pytest.raises(
        d7a.ClassConditionalIQHeadError,
        match="binding drift|seal drift",
    ):
        d7a.fit_class_conditional_head(tampered, labels)
