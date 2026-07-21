from __future__ import annotations

from dataclasses import replace
import inspect
import json
import math
import struct

import numpy as np
import pytest

from cvsrffi.stage2_zid_student_t_qknn import (
    ALLOWED_K,
    MAX_METRIC_RANK,
    WIRE_MAGIC,
    Z_DIM,
    Phase1ZIDStudentTLock,
    TypedMetricProvenanceReceipt,
    ZIDStudentTQKNNError,
    audit_int8_margin,
    audit_runtime_state,
    build_typed_shared_psd_metric,
    build_typed_zid_support_bank,
    decode_zid_support_bank,
    deserialize_typed_zid_runtime_state,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
    serialize_typed_zid_runtime_state,
    softmax_probabilities,
)


CLASSES = ("cls_a", "cls_b", "cls_c")


def _lock(k_shot: int = 5) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _metric_provenance(
    fit_scope: str = "phase1_lodo",
) -> TypedMetricProvenanceReceipt:
    return TypedMetricProvenanceReceipt(
        fit_scope=fit_scope,
        source_receipt_sha256="3" * 64,
        query_rows_used_for_fit=0,
    )


def _rows(k_shot: int = 5, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = normalize_zid_rows(
        rng.normal(size=(len(CLASSES), Z_DIM)).astype(np.float32)
    )
    rows = []
    labels = []
    for class_index, class_name in enumerate(CLASSES):
        local = centers[class_index] + 0.04 * rng.normal(size=(k_shot, Z_DIM))
        rows.append(local.astype(np.float32))
        labels.extend([class_name] * k_shot)
    return np.concatenate(rows), np.asarray(labels, dtype=str)


def _bank(k_shot: int = 5, seed: int = 11):
    rows, labels = _rows(k_shot, seed)
    return (
        build_typed_zid_support_bank(rows, labels, CLASSES, config=_lock(k_shot)),
        rows,
        labels,
    )


def _manual_identity_logits(bank, query: np.ndarray) -> np.ndarray:
    support = decode_zid_support_bank(bank).astype(np.float64)
    query64 = normalize_zid_rows(query).astype(np.float64)
    distance = np.maximum(2.0 * (1.0 - query64 @ support.T), 0.0)
    columns = []
    cfg = bank.config
    for class_index, count in enumerate(bank.support_counts):
        local = distance[:, bank.class_indices_int16 == class_index]
        assert local.shape[1] == count
        h = float(bank.class_scales_fp16[class_index])
        kernel = (
            -cfg.kernel_volume_gamma * cfg.kernel_effective_dim * math.log(h)
            - 0.5
            * (cfg.student_nu + cfg.kernel_effective_dim)
            * np.log1p(local / (cfg.student_nu * h * h))
        )
        maximum = np.max(kernel, axis=1, keepdims=True)
        columns.append(
            maximum[:, 0]
            + np.log(np.sum(np.exp(kernel - maximum), axis=1))
            - math.log(count)
        )
    return np.stack(columns, axis=1).astype(np.float32)


def _manual_fp32_identity_logits(
    support: np.ndarray,
    labels: np.ndarray,
    query: np.ndarray,
    config: Phase1ZIDStudentTLock,
    class_scales: np.ndarray,
) -> np.ndarray:
    support64 = normalize_zid_rows(support).astype(np.float64)
    query64 = normalize_zid_rows(query).astype(np.float64)
    indices = np.asarray([CLASSES.index(str(label)) for label in labels])
    distance = np.maximum(2.0 * (1.0 - query64 @ support64.T), 0.0)
    columns = []
    for class_index in range(len(CLASSES)):
        local = distance[:, indices == class_index]
        h = float(class_scales[class_index])
        kernel = (
            -config.kernel_volume_gamma * config.kernel_effective_dim * math.log(h)
            - 0.5
            * (config.student_nu + config.kernel_effective_dim)
            * np.log1p(local / (config.student_nu * h * h))
        )
        maximum = np.max(kernel, axis=1, keepdims=True)
        columns.append(
            maximum[:, 0]
            + np.log(np.sum(np.exp(kernel - maximum), axis=1))
            - math.log(local.shape[1])
        )
    return np.stack(columns, axis=1).astype(np.float32)


def test_identity_metric_matches_plain_zid_cosine_student_t_reference() -> None:
    bank, _, _ = _bank()
    query, _ = _rows(k_shot=1, seed=19)
    metric = identity_shared_psd_metric(config=bank.config)

    actual = score_zid_student_t_logits(bank, query, metric=metric)
    expected = _manual_identity_logits(bank, query)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-6)
    assert metric.effective_rank == 0
    assert metric.exact_identity
    assert metric.minimum_eigenvalue == 1.0
    assert metric.condition_number == 1.0


def test_class_and_support_permutations_are_equivariant() -> None:
    bank, rows, labels = _bank()
    query, _ = _rows(k_shot=1, seed=23)
    identity = identity_shared_psd_metric(config=bank.config)
    baseline = score_zid_student_t_logits(bank, query, metric=identity)

    row_order = np.asarray([8, 0, 12, 4, 9, 1, 13, 5, 10, 2, 14, 6, 11, 3, 7])
    reordered = build_typed_zid_support_bank(
        rows[row_order], labels[row_order], CLASSES, config=bank.config
    )
    np.testing.assert_array_equal(reordered.codes_qint8, bank.codes_qint8)
    np.testing.assert_array_equal(reordered.scales_fp16, bank.scales_fp16)
    np.testing.assert_array_equal(reordered.class_indices_int16, bank.class_indices_int16)
    np.testing.assert_allclose(
        score_zid_student_t_logits(reordered, query, metric=identity),
        baseline,
        rtol=0.0,
        atol=0.0,
    )

    permuted_classes = ("cls_c", "cls_a", "cls_b")
    permuted = build_typed_zid_support_bank(
        rows, labels, permuted_classes, config=bank.config
    )
    permuted_metric = identity_shared_psd_metric(config=permuted.config)
    permuted_logits = score_zid_student_t_logits(
        permuted, query, metric=permuted_metric
    )
    expected_columns = [CLASSES.index(name) for name in permuted_classes]
    np.testing.assert_allclose(
        permuted_logits,
        baseline[:, expected_columns],
        rtol=0.0,
        atol=2.0e-6,
    )


def test_query_batch_chunk_and_row_permutation_are_equivalent_and_stateless() -> None:
    bank, _, _ = _bank()
    query, _ = _rows(k_shot=1, seed=29)
    metric = identity_shared_psd_metric(config=bank.config)
    before_bank = serialize_typed_zid_runtime_state(bank, metric)

    together = score_zid_student_t_logits(bank, query, metric=metric)
    chunked = np.concatenate(
        [
            score_zid_student_t_logits(bank, query[:1], metric=metric),
            score_zid_student_t_logits(bank, query[1:], metric=metric),
        ]
    )
    order = np.asarray([2, 0, 1])
    permuted = score_zid_student_t_logits(bank, query[order], metric=metric)

    np.testing.assert_allclose(chunked, together, rtol=0.0, atol=2.0e-6)
    np.testing.assert_allclose(permuted, together[order], rtol=0.0, atol=2.0e-6)
    assert serialize_typed_zid_runtime_state(bank, metric) == before_bank


def test_low_rank_metric_is_class_shared_psd_and_changes_geometry() -> None:
    bank, _, _ = _bank()
    query, _ = _rows(k_shot=1, seed=31)
    basis = np.zeros((1, Z_DIM), dtype=np.float32)
    basis[0, 0] = 1.0
    metric = build_typed_shared_psd_metric(
        basis,
        np.asarray([0.25], dtype=np.float32),
        config=bank.config,
        source="unit_test_closed_form",
        provenance=_metric_provenance(),
    )

    identity_logits = score_zid_student_t_logits(
        bank, query, metric=identity_shared_psd_metric(config=bank.config)
    )
    adapted_logits = score_zid_student_t_logits(bank, query, metric=metric)

    assert metric.effective_rank == 1
    assert not metric.exact_identity
    assert metric.class_shared
    assert metric.minimum_eigenvalue > 0.0
    assert metric.condition_number >= 1.0
    assert not np.allclose(adapted_logits, identity_logits, rtol=0.0, atol=1.0e-7)


def test_metric_provenance_is_typed_and_identity_is_explicit_builder_no_fit() -> None:
    config = _lock()
    identity = identity_shared_psd_metric(config=config)
    assert identity.builder_no_fit is True
    assert identity.provenance is None
    assert identity.source == "identity_rank0"

    basis = np.eye(1, Z_DIM, dtype=np.float32)
    with pytest.raises(ZIDStudentTQKNNError, match="provenance"):
        build_typed_shared_psd_metric(
            basis,
            np.asarray([0.2], dtype=np.float32),
            config=config,
            source="missing_provenance",
            provenance=None,
        )
    with pytest.raises(ZIDStudentTQKNNError, match="fit scope"):
        TypedMetricProvenanceReceipt(
            fit_scope="query_fit",
            source_receipt_sha256="3" * 64,
            query_rows_used_for_fit=0,
        )
    with pytest.raises(ZIDStudentTQKNNError, match="query rows"):
        TypedMetricProvenanceReceipt(
            fit_scope="phase1_lodo",
            source_receipt_sha256="3" * 64,
            query_rows_used_for_fit=1,
        )
    tampered = _metric_provenance()
    object.__setattr__(tampered, "query_rows_used_for_fit", 1)
    with pytest.raises(ZIDStudentTQKNNError, match="query rows"):
        build_typed_shared_psd_metric(
            basis,
            np.asarray([0.2], dtype=np.float32),
            config=config,
            source="tampered_provenance",
            provenance=tampered,
        )


def test_int8_roundtrip_margin_and_top1_audit() -> None:
    bank, rows, labels = _bank()
    validation, _ = _rows(k_shot=1, seed=37)
    metric = identity_shared_psd_metric(config=bank.config)
    decoded = decode_zid_support_bank(bank)

    assert bank.codes_qint8.dtype == np.int8
    assert bank.scales_fp16.dtype == np.float16
    assert bank.class_indices_int16.dtype == np.int16
    np.testing.assert_allclose(np.linalg.norm(decoded, axis=1), 1.0, atol=2.0e-6)
    audit = audit_int8_margin(
        bank,
        rows,
        labels,
        validation,
        metric=metric,
    )
    assert audit["top1_agreement"] >= 0.0
    assert audit["top1_agreement"] <= 1.0
    assert audit["margin_sign_flip_count"] >= 0
    assert audit["validation_row_count"] == len(validation)
    assert audit["query_rows_used_for_fit"] == 0


def test_fp32_teacher_recomputes_exact_bandwidth_without_bank_fp16_reuse() -> None:
    config = replace(_lock(1), shared_h0=0.7)
    rows, labels = _rows(k_shot=1, seed=47)
    bank = build_typed_zid_support_bank(rows, labels, CLASSES, config=config)
    validation, _ = _rows(k_shot=1, seed=53)
    metric = identity_shared_psd_metric(config=config)

    audit = audit_int8_margin(
        bank,
        rows,
        labels,
        validation,
        metric=metric,
    )

    assert float(bank.class_scales_fp16[0]) == 0.7001953125
    assert audit["fp32_teacher_class_scales"] == [0.7, 0.7, 0.7]
    assert audit["int8_bank_class_scales"] == [0.7001953125] * len(CLASSES)
    assert audit["teacher_bank_bandwidth_abs_delta_max"] == pytest.approx(
        0.0001953125000000444,
        rel=0.0,
        abs=1.0e-15,
    )
    assert audit["fp32_teacher_bandwidth_source"] == (
        "full_precision_support_same_class_symmetric_formula"
    )
    int8_logits = score_zid_student_t_logits(bank, validation, metric=metric)
    exact_teacher_logits = _manual_fp32_identity_logits(
        rows,
        labels,
        validation,
        config,
        np.full(len(CLASSES), 0.7, dtype=np.float64),
    )
    reused_bank_logits = _manual_fp32_identity_logits(
        rows,
        labels,
        validation,
        config,
        bank.class_scales_fp16.astype(np.float64),
    )
    expected_error = float(
        np.mean(
            np.abs(
                exact_teacher_logits.astype(np.float64)
                - int8_logits.astype(np.float64)
            )
        )
    )
    reused_error = float(
        np.mean(
            np.abs(
                reused_bank_logits.astype(np.float64)
                - int8_logits.astype(np.float64)
            )
        )
    )
    assert audit["logit_abs_error_mean"] == expected_error
    assert abs(expected_error - reused_error) > 1.0e-7


def test_serialization_and_resource_audit_use_actual_arrays() -> None:
    bank, _, _ = _bank(k_shot=10)
    metric = identity_shared_psd_metric(config=bank.config)
    wire = serialize_typed_zid_runtime_state(bank, metric)
    resource = audit_runtime_state(bank, metric)
    expected_numeric = (
        bank.codes_qint8.nbytes
        + bank.scales_fp16.nbytes
        + bank.class_indices_int16.nbytes
        + bank.class_scales_fp16.nbytes
        + metric.basis_codes_qint8.nbytes
        + metric.basis_scales_fp16.nbytes
        + metric.attenuation_fp16.nbytes
    )

    assert wire == serialize_typed_zid_runtime_state(bank, metric)
    restored_bank, restored_metric = deserialize_typed_zid_runtime_state(wire)
    assert serialize_typed_zid_runtime_state(restored_bank, restored_metric) == wire
    assert restored_metric.builder_no_fit is True
    assert restored_metric.provenance is None
    assert resource["numeric_array_state_bytes"] == expected_numeric
    assert resource["actual_serialized_state_bytes"] == len(wire)
    assert resource["optimizer_steps"] == 0
    assert resource["epochs"] == 0
    assert resource["query_state_updates"] == 0
    assert resource["query_batch_dependency"] is False
    assert resource["query_file_io"] is False


def _wire_record_layout(wire: bytes):
    position = len(WIRE_MAGIC)
    header_length = struct.unpack_from("<Q", wire, position)[0]
    position += 8 + header_length
    count = struct.unpack_from("<H", wire, position)[0]
    position += 2
    records_start = position
    records = []
    for _ in range(count):
        start = position
        name_length = struct.unpack_from("<H", wire, position)[0]
        position += 2 + name_length
        dtype_length = struct.unpack_from("<H", wire, position)[0]
        position += 2
        dtype_start = position
        position += dtype_length
        ndim = struct.unpack_from("<H", wire, position)[0]
        position += 2
        shape_start = position
        position += 8 * ndim
        payload_length = struct.unpack_from("<Q", wire, position)[0]
        position += 8
        payload_start = position
        position += payload_length
        records.append(
            {
                "start": start,
                "end": position,
                "dtype_start": dtype_start,
                "dtype_length": dtype_length,
                "shape_start": shape_start,
                "ndim": ndim,
                "payload_start": payload_start,
            }
        )
    assert position == len(wire)
    return records_start, records


def test_wire_deserialize_roundtrip_is_byte_exact() -> None:
    bank, _, _ = _bank(k_shot=10)
    basis = np.eye(2, Z_DIM, dtype=np.float32)
    metric = build_typed_shared_psd_metric(
        basis,
        np.asarray([0.2, 0.3], dtype=np.float32),
        config=bank.config,
        source="wire_roundtrip",
        provenance=_metric_provenance("target_support_only"),
    )
    wire = serialize_typed_zid_runtime_state(bank, metric)

    decoded_bank, decoded_metric = deserialize_typed_zid_runtime_state(wire)

    assert serialize_typed_zid_runtime_state(decoded_bank, decoded_metric) == wire
    np.testing.assert_array_equal(decoded_bank.codes_qint8, bank.codes_qint8)
    np.testing.assert_array_equal(
        decoded_metric.basis_codes_qint8, metric.basis_codes_qint8
    )
    assert decoded_metric.provenance == metric.provenance
    assert decoded_metric.builder_no_fit is False


@pytest.mark.parametrize("cut", [0, 1, len(WIRE_MAGIC) - 1, len(WIRE_MAGIC) + 4])
def test_wire_deserialize_rejects_prefix_truncation(cut: int) -> None:
    bank, _, _ = _bank()
    metric = identity_shared_psd_metric(config=bank.config)
    wire = serialize_typed_zid_runtime_state(bank, metric)
    with pytest.raises(ZIDStudentTQKNNError, match="wire|truncated|magic"):
        deserialize_typed_zid_runtime_state(wire[:cut])
    with pytest.raises(ZIDStudentTQKNNError, match="truncated"):
        deserialize_typed_zid_runtime_state(wire[:-1])


def test_wire_deserialize_rejects_dtype_shape_order_and_payload_tampering() -> None:
    bank, _, _ = _bank()
    metric = identity_shared_psd_metric(config=bank.config)
    wire = serialize_typed_zid_runtime_state(bank, metric)
    records_start, records = _wire_record_layout(wire)

    dtype_drift = bytearray(wire)
    first = records[0]
    assert first["dtype_length"] == 3
    dtype_drift[first["dtype_start"] : first["dtype_start"] + 3] = b"|u1"
    with pytest.raises(ZIDStudentTQKNNError, match="dtype"):
        deserialize_typed_zid_runtime_state(bytes(dtype_drift))

    shape_drift = bytearray(wire)
    struct.pack_into("<Q", shape_drift, first["shape_start"] + 8, Z_DIM - 1)
    with pytest.raises(ZIDStudentTQKNNError, match="shape"):
        deserialize_typed_zid_runtime_state(bytes(shape_drift))

    reordered = b"".join(
        [
            wire[:records_start],
            wire[records[1]["start"] : records[1]["end"]],
            wire[records[0]["start"] : records[0]["end"]],
            wire[records[2]["start"] :],
        ]
    )
    with pytest.raises(ZIDStudentTQKNNError, match="order"):
        deserialize_typed_zid_runtime_state(reordered)

    payload_drift = bytearray(wire)
    payload_drift[first["payload_start"]] ^= 1
    with pytest.raises(ZIDStudentTQKNNError, match="receipt"):
        deserialize_typed_zid_runtime_state(bytes(payload_drift))

    with pytest.raises(ZIDStudentTQKNNError, match="trailing"):
        deserialize_typed_zid_runtime_state(wire + b"x")

    header_length = struct.unpack_from("<Q", wire, len(WIRE_MAGIC))[0]
    header_start = len(WIRE_MAGIC) + 8
    header_end = header_start + header_length
    header = json.loads(wire[header_start:header_end].decode("utf-8"))
    reversed_header = json.dumps(
        dict(reversed(tuple(header.items()))),
        ensure_ascii=True,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert len(reversed_header) == header_length
    noncanonical = b"".join(
        [
            wire[: len(WIRE_MAGIC)],
            struct.pack("<Q", len(reversed_header)),
            reversed_header,
            wire[header_end:],
        ]
    )
    with pytest.raises(ZIDStudentTQKNNError, match="field order"):
        deserialize_typed_zid_runtime_state(noncanonical)


def test_low_rank_resource_counts_batch1_support_and_query_projection() -> None:
    bank, _, _ = _bank(k_shot=20)
    basis = np.zeros((2, Z_DIM), dtype=np.float32)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    metric = build_typed_shared_psd_metric(
        basis,
        np.asarray([0.2, 0.3], dtype=np.float32),
        config=bank.config,
        source="resource_test",
        provenance=_metric_provenance("target_support_only"),
    )
    resource = audit_runtime_state(bank, metric)
    assert resource["score_call_fixed_matmul_mac"] == (
        bank.support_row_count * metric.effective_rank * Z_DIM
    )
    assert resource["score_query_variable_matmul_mac_per_query"] == (
        bank.support_row_count * Z_DIM
        + metric.effective_rank * Z_DIM
        + bank.support_row_count * metric.effective_rank
    )
    assert resource["matmul_mac_formula"] == "S*r*d + Q*(S*d + r*d + S*r)"
    assert resource["non_matmul_work_included_in_mac"] is False
    assert resource["persistent_decoded_cache_bytes"] == 0
    assert not any("upper_bound" in key for key in resource)


def test_phase1_lock_fixed_point_and_active_k_binding() -> None:
    lock = _lock(10)
    same = _lock(10)
    changed = replace(lock, temperature=0.9)
    assert lock.lock_digest == same.lock_digest
    assert changed.lock_digest != lock.lock_digest
    assert lock.active_k == 10
    assert tuple(ALLOWED_K) == (1, 5, 10, 20)

    rows, labels = _rows(5)
    with pytest.raises(ZIDStudentTQKNNError, match="active K"):
        build_typed_zid_support_bank(rows, labels, CLASSES, config=lock)
    with pytest.raises(ZIDStudentTQKNNError, match="active K"):
        replace(lock, active_k=True)


def test_dimension_validation() -> None:
    rows, labels = _rows()
    with pytest.raises(ZIDStudentTQKNNError, match="160"):
        build_typed_zid_support_bank(rows[:, :159], labels, CLASSES, config=_lock())


def test_invalid_support_and_registry_are_rejected() -> None:
    rows, labels = _rows()
    nonfinite = rows.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ZIDStudentTQKNNError, match="finite"):
        build_typed_zid_support_bank(nonfinite, labels, CLASSES, config=_lock())
    with pytest.raises(ZIDStudentTQKNNError, match="registry"):
        build_typed_zid_support_bank(rows, labels, ("cls_a", "cls_a"), config=_lock())
    with pytest.raises(ZIDStudentTQKNNError, match="registered class"):
        build_typed_zid_support_bank(rows, labels, CLASSES + ("empty",), config=_lock())
    with pytest.raises(ZIDStudentTQKNNError, match="balanced"):
        build_typed_zid_support_bank(rows[:-1], labels[:-1], CLASSES, config=_lock())


def test_out_of_range_class_index_and_receipt_tampering_are_rejected() -> None:
    bank, _, _ = _bank()
    bad_indices = np.asarray(bank.class_indices_int16).copy()
    bad_indices[0] = len(bank.classes)
    with pytest.raises(ZIDStudentTQKNNError, match="class index"):
        replace(bank, class_indices_int16=bad_indices)
    repeated_indices = np.asarray(bank.class_indices_int16).copy()
    repeated_indices[0] = 1
    with pytest.raises(ZIDStudentTQKNNError, match="balanced"):
        replace(bank, class_indices_int16=repeated_indices)
    invalid_code = np.asarray(bank.codes_qint8).copy()
    invalid_code[0, 0] = -128
    with pytest.raises(ZIDStudentTQKNNError, match="code range"):
        replace(bank, codes_qint8=invalid_code)
    with pytest.raises(ZIDStudentTQKNNError, match="receipt"):
        replace(bank, bank_receipt_sha256="0" * 64)
    with pytest.raises(ZIDStudentTQKNNError, match="lock"):
        replace(bank, config=replace(bank.config, shared_h0=0.4))


def test_score_rechecks_in_memory_metric_rank_and_numeric_audit() -> None:
    bank, _, _ = _bank()
    query, _ = _rows(k_shot=1, seed=43)
    basis = np.zeros((1, Z_DIM), dtype=np.float32)
    basis[0, 0] = 1.0
    metric = build_typed_shared_psd_metric(
        basis,
        np.asarray([0.25], dtype=np.float32),
        config=bank.config,
        source="tamper_test",
        provenance=_metric_provenance(),
    )
    object.__setattr__(metric, "effective_rank", 0)
    with pytest.raises(ZIDStudentTQKNNError, match="rank"):
        score_zid_student_t_logits(bank, query, metric=metric)

    metric = build_typed_shared_psd_metric(
        basis,
        np.asarray([0.25], dtype=np.float32),
        config=bank.config,
        source="tamper_test",
        provenance=_metric_provenance(),
    )
    object.__setattr__(metric, "minimum_eigenvalue", 0.123)
    with pytest.raises(ZIDStudentTQKNNError, match="numeric"):
        score_zid_student_t_logits(bank, query, metric=metric)


@pytest.mark.parametrize("attenuation", [-0.1, 0.0, 1.0, 1.1, np.nan])
def test_invalid_attenuation_is_rejected(attenuation: float) -> None:
    basis = np.zeros((1, Z_DIM), dtype=np.float32)
    basis[0, 0] = 1.0
    with pytest.raises(ZIDStudentTQKNNError, match="attenuation"):
        build_typed_shared_psd_metric(
            basis,
            np.asarray([attenuation], dtype=np.float32),
            config=_lock(),
            source="invalid",
            provenance=_metric_provenance(),
        )


def test_metric_rank_is_bounded_as_low_rank() -> None:
    basis = np.eye(MAX_METRIC_RANK + 1, Z_DIM, dtype=np.float32)
    attenuation = np.full(MAX_METRIC_RANK + 1, 0.1, dtype=np.float32)
    with pytest.raises(ZIDStudentTQKNNError, match="low-rank"):
        build_typed_shared_psd_metric(
            basis,
            attenuation,
            config=_lock(),
            source="too_wide",
            provenance=_metric_provenance(),
        )


def test_non_psd_quantized_basis_is_rejected() -> None:
    basis = np.zeros((2, Z_DIM), dtype=np.float32)
    basis[:, 0] = 1.0
    with pytest.raises(ZIDStudentTQKNNError, match="positive definite"):
        build_typed_shared_psd_metric(
            basis,
            np.asarray([0.75, 0.75], dtype=np.float32),
            config=_lock(),
            source="non_psd",
            provenance=_metric_provenance(),
        )


def test_query_and_probability_validation_reject_drift() -> None:
    bank, _, _ = _bank()
    metric = identity_shared_psd_metric(config=bank.config)
    with pytest.raises(ZIDStudentTQKNNError, match="finite"):
        score_zid_student_t_logits(
            bank, np.full((1, Z_DIM), np.nan, dtype=np.float32), metric=metric
        )
    logits = np.zeros((2, len(CLASSES)), dtype=np.float32)
    probabilities = softmax_probabilities(logits, config=bank.config)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0, atol=2.0e-6)
    with pytest.raises(ZIDStudentTQKNNError, match="Phase1 lock"):
        softmax_probabilities(logits, config=object())


def test_public_head_surface_has_no_ground_role_receiver_or_free_temperature() -> None:
    forbidden = {"ground", "receiver", "tx", "old", "new", "role", "temperature"}
    for function in (
        build_typed_zid_support_bank,
        score_zid_student_t_logits,
        audit_int8_margin,
    ):
        names = set(inspect.signature(function).parameters)
        assert not (names & forbidden)
    assert "config" in inspect.signature(softmax_probabilities).parameters
    assert "temperature" not in inspect.signature(softmax_probabilities).parameters


def test_arrays_are_read_only() -> None:
    bank, _, _ = _bank()
    metric = identity_shared_psd_metric(config=bank.config)
    for array in (
        bank.codes_qint8,
        bank.scales_fp16,
        bank.class_indices_int16,
        bank.class_scales_fp16,
        metric.basis_codes_qint8,
        metric.basis_scales_fp16,
        metric.attenuation_fp16,
    ):
        assert array.flags.writeable is False
