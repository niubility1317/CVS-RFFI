"""Focused D108 SMME tests and compact in-file traceability.

| ID | Requirement | Verification |
|---|---|---|
| D108-01 | Support-only D92-LDA margin equation | formula and D92-logit tests |
| D108-02 | K1 active and class-permutation equivariant | K1 and permutation tests |
| D108-03 | Typed immutable state, canonical wire, receipt | state/wire/tamper tests |
| D108-04 | Singleton query bias and no query update | query-independence test |
| D108-05 | Resource and numerical totalization closure | resource/extreme-logit tests |
| D108-06 | Invalid inputs fail closed | negative-input tests |
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
import math

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
import cvsrffi.stage2_d108_smme as smme


CLASSES = ("tx_a", "tx_b", "tx_c", "tx_d")


def _support_logits(k_shot: int = 2) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_name in enumerate(CLASSES):
        for shot in range(k_shot):
            row = np.asarray(
                (
                    0.4 + 0.23 * class_index + 0.07 * shot,
                    -0.8 + 0.19 * class_index - 0.03 * shot,
                    0.6 - 0.17 * class_index + 0.05 * shot,
                    -0.2 + 0.11 * class_index + 0.02 * shot,
                ),
                dtype=np.float32,
            )
            row[class_index] += np.float32(2.1 + 0.09 * shot)
            rows.append(row)
            labels.append(class_name)
    order = list(reversed(range(len(rows))))
    return (
        np.stack([rows[index] for index in order]).astype(np.float32),
        [labels[index] for index in order],
    )


def _manual_logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(math.fsum(sorted(float(np.exp(value - maximum)) for value in values)))


def _manual_state(logits: np.ndarray, labels: list[str], classes: tuple[str, ...]):
    margins: list[float] = []
    for class_index, class_name in enumerate(classes):
        class_rows = [index for index, label in enumerate(labels) if label == class_name]
        values = [
            float(logits[row, class_index])
            - _manual_logsumexp(
                np.concatenate((logits[row, :class_index], logits[row, class_index + 1 :]))
            )
            for row in class_rows
        ]
        margins.append(math.fsum(sorted(values)) / len(values))
    margins_array = np.asarray(margins, dtype=np.float64)
    mean_margin = math.fsum(sorted(float(value) for value in margins_array)) / len(
        margins_array
    )
    return margins_array, np.ascontiguousarray(mean_margin - margins_array)


def _d92_equal_prior_logits(k_shot: int) -> tuple[np.ndarray, list[str]]:
    """Produce real D42/D92 equal-prior LDA support logits without query rows."""

    rng = np.random.default_rng(108)
    class_count = len(CLASSES)
    targets = np.repeat(np.arange(class_count), k_shot).astype(np.int64)
    means = rng.normal(size=(class_count, d42.FEATURE_DIM))
    support = means[targets] + 0.07 * rng.normal(
        size=(class_count * k_shot, d42.FEATURE_DIM)
    )
    if k_shot == 1:
        support = means[targets].copy()
    coefficients, intercept, _audit = d42._fit_equal_prior_lda(
        support.astype(np.float32), targets, class_count, k_shot
    )
    logits = np.ascontiguousarray(
        support.astype(np.float32) @ coefficients.T + intercept[None, :],
        dtype=np.float32,
    )
    return logits, [CLASSES[index] for index in targets.tolist()]


def test_d92_equal_prior_logits_follow_the_frozen_smme_equation() -> None:
    logits, labels = _d92_equal_prior_logits(k_shot=3)
    state = smme.build_smme_state(logits, labels, CLASSES)
    expected_margins, expected_delta = _manual_state(logits, labels, CLASSES)
    np.testing.assert_allclose(state.class_margins_fp64, expected_margins, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(state.delta_fp64, expected_delta, rtol=0.0, atol=1e-12)
    assert state.k_shot == 3
    assert not state.class_margins_fp64.flags.writeable
    assert not state.delta_fp64.flags.writeable


def test_k1_is_active_not_a_d92_identity_fallback() -> None:
    logits, labels = _support_logits(k_shot=1)
    state = smme.build_smme_state(logits, labels, CLASSES)
    query = np.asarray([[0.2, -0.4, 0.8, -0.1]], dtype=np.float32)
    result = smme.apply_smme_query(state, query)
    assert state.k_shot == 1
    assert np.any(state.delta_fp64 != 0.0)
    assert not np.array_equal(result, query)
    np.testing.assert_array_equal(
        result,
        (query.astype(np.float64) + state.delta_fp64[None, :]).astype(np.float32),
    )


def test_registered_class_permutation_only_permutes_state_and_query_columns() -> None:
    logits, labels = _support_logits(k_shot=3)
    query = np.asarray(
        [[0.7, -0.3, 0.1, 0.5], [0.0, 0.6, -0.2, 0.8]], dtype=np.float32
    )
    original = smme.build_smme_state(logits, labels, CLASSES)
    permutation = ("tx_c", "tx_a", "tx_d", "tx_b")
    source_columns = [CLASSES.index(name) for name in permutation]
    permuted = smme.build_smme_state(logits[:, source_columns], labels, permutation)
    original_scores = smme.apply_smme_query(original, query)
    permuted_scores = smme.apply_smme_query(permuted, query[:, source_columns])
    restore_columns = [permutation.index(name) for name in CLASSES]
    np.testing.assert_allclose(
        original.class_margins_fp64,
        permuted.class_margins_fp64[restore_columns],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original.delta_fp64,
        permuted.delta_fp64[restore_columns],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(original_scores, permuted_scores[:, restore_columns])


def test_query_batch_is_singleton_equivalent_and_does_not_update_state() -> None:
    logits, labels = _support_logits(k_shot=2)
    state = smme.build_smme_state(logits, labels, CLASSES)
    query = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [0.5, -0.2, 0.7, -0.8]], dtype=np.float32
    )
    receipt_before = state.state_receipt_sha256
    batch = smme.apply_smme_query(state, query)
    singleton = np.concatenate(
        [smme.apply_smme_query(state, query[index : index + 1]) for index in range(len(query))],
        axis=0,
    )
    np.testing.assert_array_equal(batch, singleton)
    assert state.state_receipt_sha256 == receipt_before


def test_canonical_wire_roundtrip_is_immutable_and_tamper_evident() -> None:
    logits, labels = _support_logits(k_shot=2)
    state = smme.build_smme_state(logits, labels, CLASSES)
    wire = smme.serialize_smme_state(state)
    restored = smme.deserialize_smme_state(
        wire, expected_wire_sha256=hashlib.sha256(wire).hexdigest()
    )
    assert smme.serialize_smme_state(restored) == wire
    assert not restored.class_margins_fp64.flags.writeable
    assert not restored.delta_fp64.flags.writeable
    np.testing.assert_array_equal(restored.delta_fp64, state.delta_fp64)
    tampered = bytearray(wire)
    tampered[-1] ^= 1
    with pytest.raises(smme.SMMEError):
        smme.deserialize_smme_state(bytes(tampered))
    with pytest.raises(smme.SMMEError):
        smme.deserialize_smme_state(wire + b"x")
    with pytest.raises(smme.SMMEError):
        smme.deserialize_smme_state(wire[:-1])
    with pytest.raises(smme.SMMEError):
        smme.deserialize_smme_state(wire, expected_wire_sha256="0" * 64)


def test_resource_receipt_and_extreme_logit_totalization_close() -> None:
    classes = tuple(f"tx_{index:02d}" for index in range(26))
    k_shot = 10
    logits = np.full((len(classes) * k_shot, len(classes)), -1.0e30, dtype=np.float32)
    labels: list[str] = []
    for class_index, class_name in enumerate(classes):
        for shot in range(k_shot):
            row = class_index * k_shot + shot
            logits[row, class_index] = np.float32(1.0e30 - 1.0e27 * class_index)
            logits[row, (class_index + 1) % len(classes)] = np.float32(
                7.0e29 - 1.0e26 * shot
            )
            labels.append(class_name)
    state = smme.build_smme_state(logits, labels, classes)
    resource = smme.smme_resource_receipt(state)
    assert abs(float.fromhex(resource["delta_totalization_hex"])) <= float.fromhex(
        resource["delta_totalization_tolerance_hex"]
    )
    assert resource["registered_class_count"] == 26
    assert resource["k_shot"] == 10
    assert resource["trainable_parameters"] == 0
    assert resource["support_only"] is True
    assert resource["query_fit_rows"] == 0
    assert resource["query_state_updates"] == 0
    assert resource["query_bias_additions_per_row"] == 26
    assert resource["canonical_wire_bytes"] <= smme.MAX_CANONICAL_WIRE_BYTES
    assert resource["numeric_state_bytes"] == 26 * 2 * np.dtype("<f8").itemsize
    assert len(resource["resource_receipt_sha256"]) == 64


def test_public_surface_has_no_forbidden_query_information_or_state() -> None:
    build_parameters = set(inspect.signature(smme.build_smme_state).parameters)
    apply_parameters = set(inspect.signature(smme.apply_smme_query).parameters)
    forbidden = {
        "truth",
        "query_truth",
        "role",
        "old_role",
        "new_role",
        "quota",
        "batch_class_count",
        "query_labels",
        "fit",
        "update",
    }
    assert not build_parameters.intersection(forbidden)
    assert apply_parameters == {"state", "query_logits"}
    assert not any(
        token in field.name
        for field in fields(smme.SMMEState)
        for token in ("truth", "role", "quota", "query")
    )


def test_invalid_inputs_and_tampered_state_fail_closed() -> None:
    logits, labels = _support_logits(k_shot=2)
    state = smme.build_smme_state(logits, labels, CLASSES)
    with pytest.raises(smme.SMMEError):
        smme.build_smme_state(logits.astype(np.float64), labels, CLASSES)
    with pytest.raises(smme.SMMEError):
        smme.build_smme_state(logits, labels[:-1], CLASSES)
    with pytest.raises(smme.SMMEError):
        smme.build_smme_state(logits, labels[:-1] + ["not_registered"], CLASSES)
    bad = logits.copy()
    bad[0, 0] = np.nan
    with pytest.raises(smme.SMMEError):
        smme.build_smme_state(bad, labels, CLASSES)
    with pytest.raises(smme.SMMEError):
        smme.apply_smme_query(state, logits[:1].astype(np.float64))
    with pytest.raises(smme.SMMEError):
        replace(state, delta_fp64=np.zeros_like(state.delta_fp64))
