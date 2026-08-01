from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect

import numpy as np
import pytest

import cvsrffi.stage2_d108_cbrrc as cbrrc


CLASSES = ("tx_a", "tx_b", "tx_c", "tx_d", "tx_e", "tx_f")


def _support(k: int = 5) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_name in enumerate(CLASSES):
        for shot in range(k):
            row = np.zeros(cbrrc.FEATURE_DIM, dtype=np.float32)
            row[class_index] = np.float32(1.1 + 0.07 * shot)
            row[20 + class_index] = np.float32(0.34 + 0.02 * class_index)
            row[60 + shot] = np.float32(0.05 * (class_index + 1))
            row[100 + (class_index + shot) % 30] = np.float32(0.025 * (shot + 1))
            row[160:256] = np.linspace(
                -0.3 + 0.01 * class_index,
                0.4 + 0.01 * shot,
                96,
                dtype=np.float32,
            )
            row[256:] = np.linspace(
                0.2 - 0.01 * shot,
                -0.15 + 0.01 * class_index,
                32,
                dtype=np.float32,
            )
            rows.append(row)
            labels.append(class_name)
    order = list(reversed(range(len(rows))))
    return (
        np.stack([rows[index] for index in order]).astype(np.float32),
        [labels[index] for index in order],
    )


def _state(k: int = 5) -> tuple[cbrrc.CBRRCState, np.ndarray, list[str]]:
    support, labels = _support(k)
    return (
        cbrrc.build_cbrrc_state(support, labels, CLASSES),
        support,
        labels,
    )


def _reference(state: cbrrc.CBRRCState, rows: np.ndarray) -> np.ndarray:
    result = rows.copy()
    relu = rows[:, : cbrrc.RELU_DIM].astype(np.float64)
    norms = np.linalg.norm(relu, axis=1)
    unit = np.zeros_like(relu)
    active = norms > cbrrc.MACHINE_EPSILON
    unit[active] = relu[active] / norms[active, None]
    energy = state.energy_fp16.astype(np.float64)
    base = energy + state.energy_mean + cbrrc.MACHINE_EPSILON
    gain = np.sqrt(base[None, :] / (unit * unit + base[None, :]))
    transformed = unit * (1.0 + gain)
    transformed_norm = np.linalg.norm(transformed, axis=1)
    transformed_active = transformed_norm > cbrrc.MACHINE_EPSILON
    transformed[transformed_active] /= transformed_norm[transformed_active, None]
    transformed[transformed_active] *= norms[transformed_active, None]
    transformed[~transformed_active] = 0.0
    result[:, : cbrrc.RELU_DIM] = transformed.astype(np.float32)
    return result


@pytest.mark.parametrize("k", cbrrc.ALLOWED_K)
def test_frozen_formula_matches_independent_reference_and_keeps_auxiliary_blocks(
    k: int,
) -> None:
    state, support, _ = _state(k)
    query = support[:4].copy()
    query[:, 130] += np.asarray([0.01, 0.03, 0.05, 0.07], dtype=np.float32)
    actual = cbrrc.transform_cbrrc_features(state, query)
    expected = _reference(state, query)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 160:], query[:, 160:])
    np.testing.assert_allclose(
        np.linalg.norm(actual[:, :160], axis=1),
        np.linalg.norm(query[:, :160], axis=1),
        rtol=2e-7,
        atol=2e-7,
    )
    assert actual.dtype == np.float32
    assert not actual.flags.writeable


def test_k1_builds_active_nonfallback_state() -> None:
    state, support, _ = _state(1)
    raw = support[:1].copy()
    transformed = cbrrc.transform_cbrrc_features(state, raw)
    assert state.k_shot == 1
    assert state.energy_fp16.shape == (160,)
    assert np.count_nonzero(state.energy_fp16) > 0
    assert not np.allclose(
        transformed[:, :160], raw[:, :160], rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        np.linalg.norm(transformed[:, :160], axis=1),
        np.linalg.norm(raw[:, :160], axis=1),
        rtol=2e-7,
        atol=2e-7,
    )


def test_after_rows_reuse_before_state_without_mutation() -> None:
    state, support, _ = _state(5)
    receipt = state.state_receipt_sha256
    after = np.concatenate([support[:6], support[6:12] * np.float32(0.8)], axis=0)
    first = cbrrc.transform_cbrrc_features(state, after)
    second = cbrrc.transform_cbrrc_features(state, after)
    np.testing.assert_array_equal(first, second)
    assert state.state_receipt_sha256 == receipt
    assert state.k_shot == 5


def test_class_and_support_order_permutations_preserve_energy_and_transform() -> None:
    support, labels = _support(5)
    original = cbrrc.build_cbrrc_state(support, labels, CLASSES)
    class_permutation = ("tx_d", "tx_a", "tx_f", "tx_c", "tx_b", "tx_e")
    permuted_registry = cbrrc.build_cbrrc_state(
        support, labels, class_permutation
    )
    order = np.arange(len(support))[::-1]
    permuted_rows = cbrrc.build_cbrrc_state(
        support[order], [labels[index] for index in order], CLASSES
    )
    np.testing.assert_array_equal(original.energy_fp16, permuted_registry.energy_fp16)
    np.testing.assert_array_equal(original.energy_fp16, permuted_rows.energy_fp16)
    query = support[:3]
    np.testing.assert_array_equal(
        cbrrc.transform_cbrrc_features(original, query),
        cbrrc.transform_cbrrc_features(permuted_registry, query),
    )
    assert original.state_receipt_sha256 != permuted_registry.state_receipt_sha256


def test_positive_relu_row_scaling_is_equivariant_and_state_is_invariant() -> None:
    support, labels = _support(5)
    scaled = support.copy()
    scaled[:, :160] *= np.float32(4.0)
    original = cbrrc.build_cbrrc_state(support, labels, CLASSES)
    transformed_state = cbrrc.build_cbrrc_state(scaled, labels, CLASSES)
    np.testing.assert_array_equal(original.energy_fp16, transformed_state.energy_fp16)
    query = support[:2].copy()
    query_scaled = query.copy()
    query_scaled[:, :160] *= np.float32(8.0)
    np.testing.assert_array_equal(
        cbrrc.transform_cbrrc_features(original, query)[:, :160],
        cbrrc.transform_cbrrc_features(transformed_state, query_scaled)[:, :160]
        / np.float32(8.0),
    )


def test_transform_is_sample_dependent_and_not_a_shared_diagonal_noop() -> None:
    state, support, _ = _state(5)
    queries = support[:2].copy()
    queries[0, 120] = np.float32(0.01)
    queries[1, 120] = np.float32(0.8)
    unit = queries[:, :160].astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    energy = state.energy_fp16.astype(np.float64)
    base = energy + state.energy_mean + cbrrc.MACHINE_EPSILON
    gains = 1.0 + np.sqrt(base[None, :] / (unit * unit + base[None, :]))
    assert gains[0, 120] != gains[1, 120]
    assert np.all(gains > 1.0)
    assert np.all(gains <= 2.0)


@pytest.mark.parametrize("k", cbrrc.ALLOWED_K)
def test_query_batch_equals_independent_single_rows(k: int) -> None:
    state, support, _ = _state(k)
    queries = support[: min(5, len(support))].copy()
    batch = cbrrc.transform_cbrrc_features(state, queries)
    singles = np.concatenate(
        [
            cbrrc.transform_cbrrc_features(state, queries[index : index + 1])
            for index in range(len(queries))
        ],
        axis=0,
    )
    np.testing.assert_array_equal(batch, singles)


def test_public_api_and_state_have_no_truth_role_quota_or_query_fit_surface() -> None:
    build_parameters = set(inspect.signature(cbrrc.build_cbrrc_state).parameters)
    transform_parameters = set(
        inspect.signature(cbrrc.transform_cbrrc_features).parameters
    )
    forbidden = ("truth", "role", "quota", "query_label", "query_fit", "old", "new")
    assert not any(
        token in parameter
        for parameter in build_parameters | transform_parameters
        for token in forbidden
    )
    assert transform_parameters == {"state", "features"}
    state_fields = {field.name for field in fields(cbrrc.CBRRCState)}
    assert not any(
        token in name for name in state_fields for token in ("truth", "role", "quota")
    )


def test_zero_relu_rows_are_totalized_without_signed_replacement() -> None:
    support, labels = _support(1)
    support[:, :160] = 0.0
    state = cbrrc.build_cbrrc_state(support, labels, CLASSES)
    assert np.count_nonzero(state.energy_fp16) == 0
    transformed = cbrrc.transform_cbrrc_features(state, support[:2])
    assert np.count_nonzero(transformed[:, :160]) == 0
    np.testing.assert_array_equal(transformed[:, 160:], support[:2, 160:])


def test_wire_roundtrip_is_canonical_immutable_and_tamper_evident() -> None:
    state, support, _ = _state(10)
    wire = cbrrc.serialize_cbrrc_state(state)
    restored = cbrrc.deserialize_cbrrc_state(
        wire, expected_wire_sha256=hashlib.sha256(wire).hexdigest()
    )
    assert cbrrc.serialize_cbrrc_state(restored) == wire
    assert not restored.energy_fp16.flags.writeable
    np.testing.assert_array_equal(
        cbrrc.transform_cbrrc_features(state, support[:3]),
        cbrrc.transform_cbrrc_features(restored, support[:3]),
    )
    tampered = bytearray(wire)
    tampered[-1] ^= 1
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.deserialize_cbrrc_state(bytes(tampered))
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.deserialize_cbrrc_state(wire[:-1])
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.deserialize_cbrrc_state(wire + b"x")
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.deserialize_cbrrc_state(wire, expected_wire_sha256="0" * 64)


def test_state_dataclass_tamper_fails_closed() -> None:
    state, _, _ = _state(5)
    with pytest.raises(cbrrc.CBRRCError):
        replace(state, energy_mean=state.energy_mean + 0.1)
    writable = state.energy_fp16.copy()
    with pytest.raises(cbrrc.CBRRCError):
        replace(state, energy_fp16=writable)
    with pytest.raises(cbrrc.CBRRCError):
        replace(state, state_receipt_sha256="0" * 64)


@pytest.mark.parametrize("fault", ("nan", "negative", "wrong_dtype", "wrong_shape"))
def test_invalid_feature_inputs_fail_closed(fault: str) -> None:
    support, labels = _support(5)
    if fault == "nan":
        value = support.copy()
        value[0, 0] = np.nan
    elif fault == "negative":
        value = support.copy()
        value[0, 0] = -1.0
    elif fault == "wrong_dtype":
        value = support.astype(np.float64)
    else:
        value = support[:, :-1].copy()
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.build_cbrrc_state(value, labels, CLASSES)
    state, _, _ = _state(5)
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.transform_cbrrc_features(state, value[:1])


def test_registry_balance_and_k_contract_fail_closed() -> None:
    support, labels = _support(5)
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.build_cbrrc_state(support[:-1], labels[:-1], CLASSES)
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.build_cbrrc_state(support, labels, CLASSES + ("tx_g",))
    k2_support, k2_labels = _support(2)
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.build_cbrrc_state(k2_support, k2_labels, CLASSES)
    bad_labels = list(labels)
    bad_labels[0] = "not_registered"
    with pytest.raises(cbrrc.CBRRCError):
        cbrrc.build_cbrrc_state(support, bad_labels, CLASSES)


@pytest.mark.parametrize("k", cbrrc.ALLOWED_K)
def test_resource_receipt_is_small_and_closed(k: int) -> None:
    state, _, _ = _state(k)
    receipt = cbrrc.cbrrc_resource_receipt(state)
    assert receipt["numeric_state_bytes"] == 160 * 2
    assert receipt["canonical_wire_bytes"] <= cbrrc.MAX_WIRE_BYTES
    assert receipt["support_rows"] == 6 * k
    assert receipt["query_extra_mac_upper_bound"] == 160 * 5
    assert receipt["output_feature_dimension"] == 288
