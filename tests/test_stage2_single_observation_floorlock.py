from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

import cvsrffi.stage2_single_observation_floorlock as floorlock


def _iq_and_hashes(row_count: int, *, seed: int = 17):
    rng = np.random.default_rng(seed)
    iq = rng.normal(size=(row_count, 2, 64)).astype(np.float32)
    return iq, floorlock.received_iq_sha256(iq)


def _separable_support(*, k: int, dim: int = 12):
    rng = np.random.default_rng(23 + k)
    centers = np.eye(3, dim, dtype=np.float32)
    features = np.vstack(
        [
            centers[class_index]
            + 0.01 * rng.normal(size=(k, dim)).astype(np.float32)
            for class_index in range(3)
        ]
    ).astype(np.float32)
    labels = np.repeat(["old-a", "old-b", "old-c"], k)
    sample_ids = tuple(f"physical-{index}" for index in range(len(features)))
    iq, hashes = _iq_and_hashes(len(features), seed=101 + k)
    return iq, hashes, features, labels, sample_ids


def test_received_iq_hash_and_lineage_views_share_one_physical_observation():
    iq, hashes, features, _labels, sample_ids = _separable_support(k=2)
    assert hashes == floorlock.received_iq_sha256(iq.copy())
    assert hashes[0] == hashlib.sha256(
        np.ascontiguousarray(iq[0], dtype="<f4").tobytes(order="C")
    ).hexdigest()

    views = floorlock.derive_post_reception_views(
        features,
        parent_received_iq_sha256=hashes,
        physical_sample_ids=sample_ids,
        operator_id="fixed-rx-feature-test",
        view_seed=91,
    )
    assert views.features.shape == (len(features), 3, features.shape[1])
    assert views.physical_sample_count == len(features)
    assert views.representation_view_count_per_sample == 3
    assert views.k_increment == 0
    assert views.features.flags.writeable is False
    for row_index, lineage_row in enumerate(views.lineages):
        assert [item.view_name for item in lineage_row] == [
            "base",
            "plus",
            "minus",
        ]
        for item in lineage_row:
            assert item.physical_sample_id == sample_ids[row_index]
            assert item.parent_received_iq_sha256 == hashes[row_index]
            assert item.operator_id == "fixed-rx-feature-test"
            assert item.view_seed == 91
            assert item.counts_as_additional_physical_sample is False
            assert item.additional_leo_channel_state_generation is False

    with pytest.raises(
        floorlock.SingleObservationFloorLockError, match=r"float32 \[N,2,L\]"
    ):
        floorlock.received_iq_sha256(iq.astype(np.float64))


def test_support_only_equalizer_audits_parameters_and_loo_floor():
    _iq, hashes, features, labels, sample_ids = _separable_support(k=3)
    fit = floorlock.fit_support_equalizer(
        features,
        labels,
        parent_received_iq_sha256=hashes,
        physical_sample_ids=sample_ids,
        operator_id="d4a-test",
        view_seed=713101,
    )

    assert all(
        "query" not in name
        for name in inspect.signature(
            floorlock.fit_support_equalizer
        ).parameters
    )
    assert fit.state.trainable_parameters == features.shape[1]
    assert (
        fit.state.trainable_parameters
        <= floorlock.MAX_TRAINABLE_PARAMETERS
    )
    assert fit.state.query_rows_used_for_fit == 0
    assert fit.state.query_updates == 0
    assert fit.state.log_scale.flags.writeable is False
    audit = fit.state.resource_audit()
    assert audit["trainable_parameter_limit_pass"] is True
    assert audit["persistent_state_limit_pass"] is True
    assert audit["additional_physical_samples_from_views"] == 0
    assert audit["additional_leo_channel_states_generated"] == 0

    stats = fit.loo_floor_statistics
    assert stats["query_rows_used"] == 0
    assert stats["physical_support_sample_count"] == len(features)
    assert stats["k_increment_from_views"] == 0
    assert stats["overall_loo_accuracy"] == 1.0
    assert stats["min_class_loo_accuracy"] == 1.0
    assert stats["worst_class_margin"] > 0.0
    for value in stats["per_class"].values():
        assert value["loo_mode"] == "leave_one_physical_sample_out"
        assert value["unique_physical_sample_count"] == 3
        assert value["evaluation_rows"] == 3
        assert value["accuracy"] == 1.0


def test_k1_floor_uses_leave_one_view_out_without_increasing_k():
    _iq, hashes, features, labels, sample_ids = _separable_support(k=1)
    fit = floorlock.fit_support_equalizer(
        features,
        labels,
        parent_received_iq_sha256=hashes,
        physical_sample_ids=sample_ids,
    )
    stats = fit.loo_floor_statistics
    assert stats["physical_support_sample_count"] == 3
    assert fit.support_views.physical_sample_count == 3
    assert fit.support_views.representation_view_count_per_sample == 3
    assert fit.support_views.k_increment == 0
    for value in stats["per_class"].values():
        assert value["loo_mode"] == "leave_one_view_out_k1"
        assert value["unique_physical_sample_count"] == 1
        assert value["evaluation_rows"] == 3


def test_query_transform_is_inference_only_immutable_and_batch_local():
    _iq, hashes, support, labels, sample_ids = _separable_support(k=2)
    fit = floorlock.fit_support_equalizer(
        support,
        labels,
        parent_received_iq_sha256=hashes,
        physical_sample_ids=sample_ids,
    )
    assert floorlock.query_interface_is_inference_only() is True
    query_parameters = inspect.signature(
        floorlock.transform_query_features
    ).parameters
    assert "query_labels" not in query_parameters
    assert "labels" not in query_parameters

    rng = np.random.default_rng(77)
    query = rng.normal(size=(2, support.shape[1])).astype(np.float32)
    _query_iq, query_hashes = _iq_and_hashes(2, seed=303)
    query_ids = ("query-0", "query-1")
    before = hashlib.sha256(fit.state.log_scale.tobytes()).hexdigest()
    first = floorlock.transform_query_features(
        fit.state,
        query,
        parent_received_iq_sha256=query_hashes,
        physical_sample_ids=query_ids,
    )
    after = hashlib.sha256(fit.state.log_scale.tobytes()).hexdigest()
    assert before == after
    assert fit.state.query_updates == 0
    assert first.features.shape == (2, 1, support.shape[1])
    assert first.lineages[0][0].view_name == "base"

    extra = rng.normal(size=(1, support.shape[1])).astype(np.float32)
    _extra_iq, extra_hash = _iq_and_hashes(1, seed=404)
    extended = floorlock.transform_query_features(
        fit.state,
        np.vstack((query, extra)),
        parent_received_iq_sha256=query_hashes + extra_hash,
        physical_sample_ids=query_ids + ("query-extra",),
    )
    assert np.array_equal(first.features, extended.features[:2])
    with pytest.raises(ValueError):
        fit.state.log_scale[0] = 0.0


def test_duplicate_physical_sample_and_parameter_overflow_fail_closed():
    _iq, hashes, features, labels, sample_ids = _separable_support(k=1)
    with pytest.raises(
        floorlock.SingleObservationFloorLockError,
        match="unique physical sample",
    ):
        floorlock.fit_support_equalizer(
            features,
            labels,
            parent_received_iq_sha256=hashes,
            physical_sample_ids=(sample_ids[0], sample_ids[0], sample_ids[2]),
        )

    oversized = np.zeros((2, 80_001), dtype=np.float32)
    with pytest.raises(
        floorlock.SingleObservationFloorLockError,
        match="exceeds 80k",
    ):
        floorlock.fit_support_equalizer(
            oversized,
            ("old-a", "old-b"),
            parent_received_iq_sha256=("a" * 64, "b" * 64),
            physical_sample_ids=("physical-a", "physical-b"),
        )
