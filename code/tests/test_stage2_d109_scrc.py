"""Focused D109 SCRC tests and compact in-file traceability.

| ID | Requirement | Verification |
|---|---|---|
| D109-01 | Frozen Q/R/T/rho equations and orientation | formula and row-stochastic checks |
| D109-02 | K={1,5,10}, balanced support, and K1 activity | K and negative-input tests |
| D109-03 | Class permutation and identity degeneration | permutation and exact-Q checks |
| D109-04 | Singleton query-only inference | batch/singleton and no-update checks |
| D109-05 | Typed state, receipt, wire, and resource closure | state/wire/resource tests |
| D109-06 | No forbidden query-information surface | signature and field checks |
| D109-07 | Extreme nonidentity pT semantics remain exact in log domain | float32-limit permutation test |
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect

import numpy as np
import pytest

import cvsrffi.stage2_d109_scrc as scrc


CLASSES = ("tx_a", "tx_b", "tx_c", "tx_d")


def _support_logits(k_shot: int = 5) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_name in enumerate(CLASSES):
        for shot in range(k_shot):
            row = np.asarray(
                (
                    -0.85 + 0.16 * class_index + 0.04 * shot,
                    -0.35 - 0.12 * class_index + 0.02 * shot,
                    0.25 + 0.09 * class_index - 0.03 * shot,
                    0.55 - 0.14 * class_index + 0.01 * shot,
                ),
                dtype=np.float32,
            )
            row[class_index] += np.float32(1.55 + 0.06 * shot)
            rows.append(row)
            labels.append(class_name)
    order = list(reversed(range(len(rows))))
    return (
        np.stack([rows[index] for index in order]).astype(np.float32),
        [labels[index] for index in order],
    )


def _manual_softmax(row: np.ndarray) -> np.ndarray:
    values = row.astype(np.float64)
    shifted = values - np.max(values)
    values = np.exp(shifted)
    return values / np.sum(values)


def _manual_q(logits: np.ndarray, labels: list[str]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for class_name in CLASSES:
        members = [
            _manual_softmax(logits[index])
            for index, label in enumerate(labels)
            if label == class_name
        ]
        rows.append(np.mean(np.stack(members), axis=0))
    return np.asarray(rows, dtype=np.float32)


def test_frozen_q_r_t_rho_formula_has_the_specified_orientation() -> None:
    logits, labels = _support_logits(5)
    state = scrc.build_scrc_state(logits, labels, CLASSES)
    expected_q = _manual_q(logits, labels)
    np.testing.assert_allclose(
        state.support_confusion_fp32, expected_q, rtol=0.0, atol=2.0e-7
    )
    q = state.support_confusion_fp32.astype(np.float64)
    expected_r = np.empty_like(q)
    for observed_index in range(len(CLASSES)):
        expected_r[observed_index, :] = (
            q[:, observed_index] / np.sum(q[:, observed_index])
        )
    expected_rho = np.float32(1.0 - np.trace(q) / len(CLASSES))
    expected_t = (
        (1.0 - float(expected_rho)) * np.eye(len(CLASSES))
        + float(expected_rho) * expected_r
    ).astype(np.float32)
    assert state.rho_fp32.tobytes() == expected_rho.tobytes()
    np.testing.assert_allclose(state.transition_fp32, expected_t, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(expected_r.sum(axis=1), np.ones(len(CLASSES)), atol=1e-12)
    np.testing.assert_allclose(
        state.transition_fp32.astype(np.float64).sum(axis=1),
        np.ones(len(CLASSES)),
        atol=2.0e-6,
    )
    query = np.asarray([[0.2, -0.8, 0.4, 0.7]], dtype=np.float32)
    p = _manual_softmax(query[0])
    expected_h = query.astype(np.float64) + np.log(p @ expected_t) - np.log(p)
    actual_h = scrc.apply_scrc_query(state, query).astype(np.float64)
    row_delta = actual_h - expected_h
    np.testing.assert_allclose(
        row_delta, np.full_like(row_delta, row_delta[0, 0]), atol=2.0e-6
    )
    actual_p = np.exp(actual_h[0] - np.max(actual_h[0]))
    actual_p /= actual_p.sum()
    expected_p = p @ expected_t
    np.testing.assert_allclose(actual_p, expected_p, rtol=0.0, atol=2.0e-7)


@pytest.mark.parametrize("k_shot", (1, 5, 10))
def test_all_and_only_frozen_k_values_build_balanced_states(k_shot: int) -> None:
    logits, labels = _support_logits(k_shot)
    state = scrc.build_scrc_state(logits, labels, CLASSES)
    assert state.k_shot == k_shot
    assert state.support_confusion_fp32.shape == (len(CLASSES), len(CLASSES))
    assert not state.support_confusion_fp32.flags.writeable
    assert not state.transition_fp32.flags.writeable


def test_k1_is_active_not_a_hidden_identity_fallback() -> None:
    logits, labels = _support_logits(1)
    state = scrc.build_scrc_state(logits, labels, CLASSES)
    query = np.asarray([[0.4, -0.3, 0.1, 0.7]], dtype=np.float32)
    result = scrc.apply_scrc_query(state, query)
    assert state.k_shot == 1
    assert state.rho_fp32 > 0.0
    assert not np.array_equal(state.transition_fp32, np.eye(len(CLASSES), dtype=np.float32))
    assert not np.array_equal(result, query)


def test_identity_q_has_only_its_mathematical_identity_degeneration() -> None:
    identity_q = np.eye(3, dtype=np.float32)
    rho, reciprocal, transition = scrc._transition_from_confusion(identity_q)
    assert rho == np.float32(0.0)
    np.testing.assert_array_equal(reciprocal, np.eye(3, dtype=np.float64))
    np.testing.assert_array_equal(transition, identity_q)
    state = scrc._make_state(
        registered_classes=("a", "b", "c"),
        k_shot=1,
        support_confusion_fp32=identity_q,
        transition_fp32=transition,
        rho_fp32=rho,
    )
    query = np.asarray(
        [[np.finfo(np.float32).max, -np.finfo(np.float32).max, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(scrc.apply_scrc_query(state, query), query)


def test_registered_class_permutation_only_permutes_q_t_and_scores() -> None:
    logits, labels = _support_logits(5)
    query = np.asarray(
        [[0.3, -0.7, 0.1, 0.8], [-0.2, 0.5, 0.9, -0.3]], dtype=np.float32
    )
    original = scrc.build_scrc_state(logits, labels, CLASSES)
    permutation = ("tx_c", "tx_a", "tx_d", "tx_b")
    source_columns = [CLASSES.index(name) for name in permutation]
    permuted = scrc.build_scrc_state(logits[:, source_columns], labels, permutation)
    restore_columns = [permutation.index(name) for name in CLASSES]
    np.testing.assert_allclose(
        original.support_confusion_fp32,
        permuted.support_confusion_fp32[np.ix_(restore_columns, restore_columns)],
        rtol=0.0,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        original.transition_fp32,
        permuted.transition_fp32[np.ix_(restore_columns, restore_columns)],
        rtol=0.0,
        atol=2.0e-7,
    )
    original_scores = scrc.apply_scrc_query(original, query)
    permuted_scores = scrc.apply_scrc_query(permuted, query[:, source_columns])
    np.testing.assert_allclose(
        original_scores,
        permuted_scores[:, restore_columns],
        rtol=0.0,
        atol=2.0e-6,
    )


def test_query_batch_is_singleton_equivalent_and_does_not_update_state() -> None:
    logits, labels = _support_logits(5)
    state = scrc.build_scrc_state(logits, labels, CLASSES)
    query = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [0.5, -0.2, 0.7, -0.8]], dtype=np.float32
    )
    receipt_before = state.state_receipt_sha256
    batch = scrc.apply_scrc_query(state, query)
    singleton = np.concatenate(
        [
            scrc.apply_scrc_query(state, query[index : index + 1])
            for index in range(len(query))
        ],
        axis=0,
    )
    np.testing.assert_array_equal(batch, singleton)
    assert state.state_receipt_sha256 == receipt_before


def test_extreme_float32_logits_keep_the_query_transition_probability_positive() -> None:
    classes = ("a", "b", "c")
    magnitude = np.finfo(np.float32).max
    support = np.asarray(
        [
            [magnitude, -magnitude, -magnitude],
            [-magnitude, magnitude, -magnitude],
            [-magnitude, -magnitude, magnitude],
        ],
        dtype=np.float32,
    )
    state = scrc.build_scrc_state(support, list(classes), classes)
    query = np.asarray([[magnitude, -magnitude, -magnitude]], dtype=np.float32)
    probability = scrc._stable_softmax_row(query[0], "extreme query")
    transformed = probability @ state.transition_fp32.astype(np.float64)
    assert np.isfinite(probability).all() and np.all(probability > 0.0)
    assert np.isfinite(transformed).all() and np.all(transformed > 0.0)
    assert np.isfinite(scrc.apply_scrc_query(state, query)).all()


def test_extreme_nonidentity_transition_preserves_log_domain_p_t_semantics() -> None:
    classes = ("a", "b", "c")
    confusion = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    rho, _reciprocal, transition = scrc._transition_from_confusion(confusion)
    assert rho == np.float32(1.0)
    assert np.count_nonzero(transition == 0.0) == 6
    state = scrc._make_state(
        registered_classes=classes,
        k_shot=1,
        support_confusion_fp32=confusion,
        transition_fp32=transition,
        rho_fp32=rho,
    )
    magnitude = np.finfo(np.float32).max
    query = np.asarray([[magnitude, -magnitude, -magnitude]], dtype=np.float32)
    result = scrc.apply_scrc_query(state, query)
    assert np.isfinite(result).all()
    assert int(np.argmax(result[0])) == 2
    log_p = scrc._true_log_softmax_row(query[0], "extreme expected query")
    expected_log = scrc._log_transition_probabilities(
        log_p, transition.astype(np.float64)
    )
    expected = np.exp(expected_log - np.max(expected_log))
    actual = np.exp(result[0].astype(np.float64) - np.max(result[0]))
    expected /= expected.sum()
    actual /= actual.sum()
    np.testing.assert_allclose(
        actual, expected, rtol=0.0, atol=scrc._FLOAT32_EXPONENT_FLOOR
    )


def test_extreme_dense_transition_does_not_lose_p_t_differences_to_row_offset() -> None:
    classes = ("a", "b", "c")
    confusion = np.full((3, 3), np.float32(1.0 / 3.0), dtype=np.float32)
    rho, _reciprocal, transition = scrc._transition_from_confusion(confusion)
    np.testing.assert_allclose(
        transition[0], np.asarray([5.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0]),
        rtol=0.0, atol=1.0e-7,
    )
    state = scrc._make_state(
        registered_classes=classes,
        k_shot=1,
        support_confusion_fp32=confusion,
        transition_fp32=transition,
        rho_fp32=rho,
    )
    magnitude = np.finfo(np.float32).max
    query = np.asarray([[magnitude, -magnitude, -magnitude]], dtype=np.float32)
    result = scrc.apply_scrc_query(state, query)
    assert np.isfinite(result).all()
    actual = np.exp(result[0].astype(np.float64) - np.max(result[0]))
    actual /= actual.sum()
    np.testing.assert_allclose(actual, transition[0], rtol=0.0, atol=2.0e-7)


def test_canonical_wire_receipt_and_resource_are_tamper_evident() -> None:
    logits, labels = _support_logits(10)
    state = scrc.build_scrc_state(logits, labels, CLASSES)
    wire = scrc.serialize_scrc_state(state)
    restored = scrc.deserialize_scrc_state(
        wire, expected_wire_sha256=hashlib.sha256(wire).hexdigest()
    )
    assert scrc.serialize_scrc_state(restored) == wire
    assert not restored.support_confusion_fp32.flags.writeable
    assert not restored.transition_fp32.flags.writeable
    resource = scrc.scrc_resource_receipt(state)
    assert resource["numeric_state_bytes"] == (
        2 * len(CLASSES) * len(CLASSES) * np.dtype("<f4").itemsize
        + np.dtype("<f4").itemsize
    )
    assert resource["reciprocal_response_orientation"] == "R[b,a]=Q[a,b]/sum_l Q[l,b]"
    assert resource["transition_orientation"] == "T[b,a]=(1-rho)I[b,a]+rhoR[b,a]"
    assert resource["query_transform_orientation"] == "p_tilde[a]=sum_b p[b]T[b,a]"
    assert resource["query_fit_rows"] == 0
    assert resource["query_state_updates"] == 0
    assert resource["canonical_wire_bytes"] <= scrc.MAX_CANONICAL_WIRE_BYTES
    assert len(resource["resource_receipt_sha256"]) == 64
    tampered = bytearray(wire)
    tampered[-1] ^= 1
    with pytest.raises(scrc.SCRCError):
        scrc.deserialize_scrc_state(bytes(tampered))
    with pytest.raises(scrc.SCRCError):
        scrc.deserialize_scrc_state(wire + b"x")
    with pytest.raises(scrc.SCRCError):
        scrc.deserialize_scrc_state(wire[:-1])
    with pytest.raises(scrc.SCRCError):
        scrc.deserialize_scrc_state(wire, expected_wire_sha256="0" * 64)
    with pytest.raises(scrc.SCRCError):
        replace(state, transition_fp32=np.eye(len(CLASSES), dtype=np.float32))


def test_public_surface_has_no_truth_role_quota_or_query_fit_api() -> None:
    build_parameters = set(inspect.signature(scrc.build_scrc_state).parameters)
    apply_parameters = set(inspect.signature(scrc.apply_scrc_query).parameters)
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
        for field in fields(scrc.SCRCState)
        for token in ("truth", "role", "quota", "query")
    )


def test_invalid_nonfinite_and_unbalanced_inputs_fail_closed() -> None:
    logits, labels = _support_logits(5)
    with pytest.raises(scrc.SCRCError):
        scrc.build_scrc_state(logits.astype(np.float64), labels, CLASSES)
    with pytest.raises(scrc.SCRCError):
        scrc.build_scrc_state(logits, labels[:-1], CLASSES)
    with pytest.raises(scrc.SCRCError):
        scrc.build_scrc_state(logits, labels[:-1] + ["not_registered"], CLASSES)
    unbalanced_labels = labels[:-1] + [CLASSES[1]]
    with pytest.raises(scrc.SCRCError):
        scrc.build_scrc_state(logits, unbalanced_labels, CLASSES)
    k2_logits, k2_labels = _support_logits(2)
    with pytest.raises(scrc.SCRCError):
        scrc.build_scrc_state(k2_logits, k2_labels, CLASSES)
    for bad_value in (np.nan, np.inf, -np.inf):
        bad_logits = logits.copy()
        bad_logits[0, 0] = bad_value
        with pytest.raises(scrc.SCRCError):
            scrc.build_scrc_state(bad_logits, labels, CLASSES)
    state = scrc.build_scrc_state(logits, labels, CLASSES)
    with pytest.raises(scrc.SCRCError):
        scrc.apply_scrc_query(state, logits[:1].astype(np.float64))
