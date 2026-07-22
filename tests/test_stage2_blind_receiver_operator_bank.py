from __future__ import annotations

import inspect

import numpy as np
import pytest

import cvsrffi.stage2_blind_receiver_operator_bank as d10
from cvsrffi.stage2_support_lowrank_metric import received_iq_sha256


def _iq_labels(
    classes=("old-a", "old-b", "old-c"),
    *,
    k=10,
    length=64,
    seed=401,
):
    rng = np.random.default_rng(seed)
    time = np.arange(length, dtype=np.float64)
    rows, labels = [], []
    for class_index, label in enumerate(classes):
        for sample_index in range(k):
            phase = 0.11 * sample_index + rng.uniform(-0.08, 0.08)
            clean = np.exp(
                1j
                * (
                    2.0
                    * np.pi
                    * (0.055 + 0.018 * class_index)
                    * time
                    + phase
                )
            )
            image = (
                0.18
                * np.exp(1j * (0.4 + 0.2 * class_index))
                * np.conjugate(clean)
            )
            envelope = 1.0 + 0.13 * np.cos(
                2.0 * np.pi * time / length
            )
            row = envelope * (clean + image)
            row += 0.03 * (
                rng.normal(size=length)
                + 1j * rng.normal(size=length)
            )
            rows.append(np.stack([row.real, row.imag]))
            labels.append(label)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels)


def _extractor(rows):
    complex_rows = rows[:, 0] + 1j * rows[:, 1]
    spectrum = np.abs(np.fft.fft(complex_rows, axis=1))[:, :16]
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


def _inputs(classes=("old-a", "old-b", "old-c"), seed=401):
    iq, labels = _iq_labels(classes, seed=seed)
    hashes = received_iq_sha256(iq)
    ids = tuple(
        f"sid_{seed:032x}{index:032x}" for index in range(len(iq))
    )
    features = d10.extract_operator_features(
        iq, feature_extractor=_extractor
    )
    provenance = d10.build_operator_feature_provenance(
        hashes, view_seed=0
    )
    return iq, labels, hashes, ids, features, provenance


def _fit(classes=("old-a", "old-b", "old-c"), seed=401):
    iq, labels, hashes, ids, features, provenance = _inputs(
        classes, seed
    )
    state = d10.fit_blind_receiver_operator_bank(
        features,
        provenance,
        labels,
        physical_sample_ids=ids,
        parent_received_iq_sha256=hashes,
        base_resource_audit={
            "persistent_state_bytes": 4096,
            "estimated_head_macs_per_query": 1024,
        },
        received_iq_length=iq.shape[-1],
    )
    return iq, labels, hashes, ids, features, provenance, state


def _circularity(row):
    centered = row - np.mean(row)
    return abs(np.mean(centered**2)) / max(
        np.mean(abs(centered) ** 2), 1e-8
    )


def test_operator_bank_is_three_fixed_samplewise_views():
    assert d10.OPERATORS == (
        "base",
        "wl_iq_circularize",
        "fft_envelope_eq",
    )
    iq, _labels = _iq_labels(k=1)
    for operator in d10.OPERATORS:
        batch = d10.apply_received_iq_operator(iq, operator)
        singles = np.concatenate(
            [
                d10.apply_received_iq_operator(
                    iq[index : index + 1], operator
                )
                for index in range(len(iq))
            ],
            axis=0,
        )
        assert batch.dtype == np.float32
        assert np.isfinite(batch).all()
        np.testing.assert_array_equal(batch, singles)
    np.testing.assert_array_equal(
        d10.apply_received_iq_operator(iq, d10.BASE), iq
    )


def test_widely_linear_circularization_is_bounded_and_stable():
    length = 128
    time = np.arange(length)
    source = np.exp(1j * 2.0 * np.pi * 0.09375 * time)
    imbalanced = source + 0.42j * np.conjugate(source)
    iq = np.asarray(
        [[imbalanced.real, imbalanced.imag]], dtype=np.float32
    )
    output = d10.apply_received_iq_operator(
        iq, d10.WL_IQ_CIRCULARIZE
    )
    before = iq[0, 0] + 1j * iq[0, 1]
    after = output[0, 0] + 1j * output[0, 1]
    assert _circularity(after) < _circularity(before)
    before_rms = np.sqrt(np.mean(abs(before - before.mean()) ** 2))
    after_rms = np.sqrt(np.mean(abs(after - after.mean()) ** 2))
    assert after_rms / before_rms <= d10.WL_OUTPUT_GAIN_CAP + 1e-5
    zero = np.zeros((2, 2, 16), dtype=np.float32)
    np.testing.assert_array_equal(
        d10.apply_received_iq_operator(
            zero, d10.WL_IQ_CIRCULARIZE
        ),
        zero,
    )


def test_fft_equalizer_preserves_bin_phase_peak_and_residual_shape():
    length = 128
    time = np.arange(length)
    row = (
        0.8 * np.exp(1j * 2.0 * np.pi * 13 * time / length)
        + 0.2 * np.exp(1j * 2.0 * np.pi * 29 * time / length)
        + 0.05
    )
    iq = np.asarray([[row.real, row.imag]], dtype=np.float32)
    output = d10.apply_received_iq_operator(
        iq, d10.FFT_ENVELOPE_EQ
    )
    original_fft = np.fft.fft(row)
    output_row = output[0, 0] + 1j * output[0, 1]
    output_fft = np.fft.fft(output_row)
    mask = np.abs(original_fft) > 1e-4
    phase_error = np.angle(output_fft[mask] / original_fft[mask])
    assert np.max(np.abs(phase_error)) < 1e-5
    assert np.argmax(abs(output_fft)) == np.argmax(abs(original_fft))
    assert np.std(abs(output_fft)) > 0.05
    gain = abs(output_fft[mask]) / abs(original_fft[mask])
    assert np.min(gain) > 0.75
    assert np.max(gain) < 1.35


def test_invalid_operator_iq_and_real_provenance_fail_closed():
    iq, labels, hashes, ids, features, provenance = _inputs()
    with pytest.raises(
        d10.BlindReceiverOperatorBankError, match="finite float32"
    ):
        d10.apply_received_iq_operator(iq.astype(np.float64), d10.BASE)
    with pytest.raises(
        d10.BlindReceiverOperatorBankError, match="unsupported"
    ):
        d10.apply_received_iq_operator(iq, "cfo_derotate")
    bad = {key: list(value) for key, value in provenance.items()}
    bad[d10.WL_IQ_CIRCULARIZE][0] = {
        **bad[d10.WL_IQ_CIRCULARIZE][0],
        "operator_id": d10.FFT_ENVELOPE_EQ,
    }
    with pytest.raises(
        d10.BlindReceiverOperatorBankError, match="provenance"
    ):
        d10.fit_blind_receiver_operator_bank(
            features,
            bad,
            labels,
            physical_sample_ids=ids,
            parent_received_iq_sha256=hashes,
            base_resource_audit={
                "persistent_state_bytes": 0,
                "estimated_head_macs_per_query": 0,
            },
            received_iq_length=iq.shape[-1],
        )


def test_equal_accuracy_bank_strictly_falls_back_to_base():
    iq, labels, hashes, ids, features, provenance = _inputs()
    identical = {
        operator: features[d10.BASE].copy()
        for operator in d10.OPERATORS
    }
    state = d10.fit_blind_receiver_operator_bank(
        identical,
        provenance,
        labels,
        physical_sample_ids=ids,
        parent_received_iq_sha256=hashes,
        base_resource_audit={
            "persistent_state_bytes": 0,
            "estimated_head_macs_per_query": 0,
        },
        received_iq_length=iq.shape[-1],
    )
    assert state.strict_accuracy_gate["pass"] is False
    assert (
        state.strict_accuracy_gate["fallback_reason"]
        == "no_accuracy_improvement_over_all_base"
    )
    assert state.used_operators == (d10.BASE,)
    assert state.support_audit["selection"][
        "d10_fallback_to_all_base"
    ] is True
    assert np.all(state.operator_indices[:, 0] == 0)
    assert np.all(state.weights[:, 0] == 1.0)


def test_d9_selection_resources_and_query_contract_are_reused():
    iq, _labels, _hashes, _ids, _features, _provenance, state = _fit()
    selection = state.support_audit["selection"]
    baseline = selection["baseline"]
    final = selection["combined_final"]
    assert final["overall_accuracy"] >= baseline["overall_accuracy"]
    assert final["min_class_accuracy"] >= baseline["min_class_accuracy"]
    for label in state.classes:
        assert (
            final["per_class_accuracy"][label]
            >= baseline["per_class_accuracy"][label]
        )
    resource = state.resource_audit()
    assert resource["trainable_parameters"] == 0
    assert resource["adaptation_epochs"] == 0
    assert resource["persistent_state_limit_pass"] is True
    assert resource["used_operator_count"] <= 3
    assert resource["backbone_forwards_per_query"] <= 3
    assert resource["additional_leo_channel_states_generated"] == 0
    assert resource["views_count_as_additional_k"] is False
    assert resource["cfo_estimation"] is False
    assert resource["cfo_derotation"] is False
    assert resource["dense_query_graph_bytes"] == 0
    assert d10.public_query_interface_is_oracle_free()
    parameters = " ".join(
        inspect.signature(
            d10.predict_blind_receiver_operator_bank
        ).parameters
    ).lower()
    assert all(
        token not in parameters
        for token in ("label", "truth", "role", "quota", "graph")
    )

    sealed = d10.seal_samplewise_feature_extractor(
        _extractor,
        extractor_id="d10-synthetic-v1",
        validation_rows=iq[:3],
    )
    first = d10.predict_blind_receiver_operator_bank(
        state, iq[:2], feature_extractor=sealed
    )
    extended = d10.predict_blind_receiver_operator_bank(
        state, iq[:3], feature_extractor=sealed
    )
    np.testing.assert_allclose(first.scores, extended.scores[:2], atol=1e-6)
    with pytest.raises(
        d10.BlindReceiverOperatorBankError, match="samplewise sealed"
    ):
        d10.predict_blind_receiver_operator_bank(
            state, iq[:2], feature_extractor=_extractor
        )


def test_registration_locks_old_state_and_k1_k5_only_rebuild_prototypes():
    (
        _old_iq,
        old_labels,
        old_hashes,
        old_ids,
        old_features,
        old_provenance,
        before,
    ) = _fit(classes=("old-a", "old-b"), seed=421)
    (
        new_iq,
        new_labels,
        new_hashes,
        new_ids,
        new_features,
        new_provenance,
    ) = _inputs(classes=("new-a", "new-b"), seed=431)
    all_labels = np.concatenate([old_labels, new_labels])
    all_hashes = old_hashes + new_hashes
    all_ids = old_ids + new_ids
    all_features = {
        operator: np.concatenate(
            [old_features[operator], new_features[operator]], axis=0
        )
        for operator in d10.OPERATORS
    }
    all_provenance = {
        operator: old_provenance[operator] + new_provenance[operator]
        for operator in d10.OPERATORS
    }
    after = d10.extend_blind_receiver_operator_bank(
        before,
        all_features,
        all_provenance,
        all_labels,
        physical_sample_ids=all_ids,
        parent_received_iq_sha256=all_hashes,
    )
    old_count = before.class_count
    np.testing.assert_array_equal(
        after.operator_indices[:old_count], before.operator_indices
    )
    np.testing.assert_array_equal(
        after.weights[:old_count], before.weights
    )
    np.testing.assert_array_equal(
        after.prototypes[:old_count], before.prototypes
    )
    assert after.inner.calibrations == before.inner.calibrations

    for nested_k in (1, 5):
        indices = np.asarray(
            [
                index
                for label in after.classes
                for index in np.flatnonzero(all_labels == label).tolist()[
                    :nested_k
                ]
            ],
            dtype=np.int64,
        )
        nested = d10.rebuild_locked_blind_receiver_prototypes(
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
                for operator in d10.OPERATORS
            },
            all_labels[indices],
            physical_sample_ids=tuple(
                all_ids[index] for index in indices
            ),
            parent_received_iq_sha256=tuple(
                all_hashes[index] for index in indices
            ),
        )
        assert nested.current_k == nested_k
        assert nested.operator_indices is after.operator_indices
        assert nested.weights is after.weights
        assert nested.inner.calibrations == after.inner.calibrations
        assert nested.selection_lock_sha256 == after.selection_lock_sha256
        assert nested.support_audit["only_prototypes_rebuilt"] is True


def test_non_k10_and_combined_state_cap_fail_closed():
    iq, labels = _iq_labels(k=5)
    hashes = received_iq_sha256(iq)
    ids = tuple(f"sid_{index:064x}" for index in range(len(iq)))
    features = d10.extract_operator_features(
        iq, feature_extractor=_extractor
    )
    provenance = d10.build_operator_feature_provenance(
        hashes, view_seed=0
    )
    with pytest.raises(ValueError, match="K10"):
        d10.fit_blind_receiver_operator_bank(
            features,
            provenance,
            labels,
            physical_sample_ids=ids,
            parent_received_iq_sha256=hashes,
            base_resource_audit={
                "persistent_state_bytes": 0,
                "estimated_head_macs_per_query": 0,
            },
            received_iq_length=iq.shape[-1],
        )
    iq, labels, hashes, ids, features, provenance = _inputs()
    with pytest.raises(ValueError, match="state cap"):
        d10.fit_blind_receiver_operator_bank(
            features,
            provenance,
            labels,
            physical_sample_ids=ids,
            parent_received_iq_sha256=hashes,
            base_resource_audit={
                "persistent_state_bytes": 256 * 1024,
                "estimated_head_macs_per_query": 0,
            },
            received_iq_length=iq.shape[-1],
        )
