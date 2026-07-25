from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cvsrffi.stage2_d104_angq_qknn import (
    ANGQ_MAC_PER_SUPPORT,
    ANGQ_SCHEMA,
    ANGQ_VECTOR_ELEMENTWISE_OPS_PER_SUPPORT,
    FACTORS,
    D104ANGQError,
    _decode_candidate,
    audit_d104_angq_resource_delta,
    build_d104_angq_support_bank,
    build_matched_legacy_and_d104_banks,
    quantize_d104_angq_normalized_rows,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Z_DIM,
    Phase1ZIDStudentTLock,
    _quantize_rows,
    decode_zid_support_bank,
    deserialize_typed_zid_runtime_state,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
    serialize_typed_zid_runtime_state,
)


CLASSES = ("opaque_a", "opaque_b", "opaque_c")


def _lock(k_shot: int) -> Phase1ZIDStudentTLock:
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


def _support(
    k_shot: int,
    seed: int = 104713,
) -> tuple[np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    centers = normalize_zid_rows(
        rng.normal(size=(len(CLASSES), Z_DIM)).astype(np.float32)
    )
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_id in enumerate(CLASSES):
        rows.append(
            (
                centers[class_index]
                + 0.03 * rng.normal(size=(k_shot, Z_DIM))
            ).astype(np.float32)
        )
        labels.extend([class_id] * k_shot)
    return np.concatenate(rows), tuple(labels)


def _class_payload(bank, class_id: str) -> tuple[list[bytes], float]:
    class_index = bank.classes.index(class_id)
    local = bank.class_indices_int16 == class_index
    rows = [
        b"".join(
            (
                bank.codes_qint8[index].tobytes(order="C"),
                bank.scales_fp16[index : index + 1].tobytes(order="C"),
            )
        )
        for index in np.flatnonzero(local)
    ]
    return sorted(rows), float(bank.class_scales_fp16[class_index])


def test_d104_grid_and_c1_are_exactly_frozen() -> None:
    assert FACTORS.dtype == np.float64
    assert len(FACTORS) == 101
    assert FACTORS[0] == 0.75
    assert FACTORS[50] == 1.0
    assert FACTORS[-1] == 1.25
    np.testing.assert_array_equal(
        np.diff(FACTORS),
        np.diff(
            np.asarray(
                [0.75 + 0.005 * index for index in range(101)],
                dtype=np.float64,
            )
        ),
    )


def test_c1_scale_code_and_decoded_match_legacy_bitwise() -> None:
    support, _ = _support(5)
    normalized = normalize_zid_rows(support)
    legacy_codes, legacy_scales, legacy_decoded = _quantize_rows(normalized)
    c1_scales = np.empty(len(normalized), dtype=np.float16)
    c1_codes = np.empty_like(legacy_codes)
    c1_decoded = np.empty_like(legacy_decoded)
    for index, row in enumerate(normalized):
        scale, code, decoded, _ = _decode_candidate(row, 1.0)
        c1_scales[index] = scale
        c1_codes[index] = code
        c1_decoded[index] = decoded
    np.testing.assert_array_equal(c1_scales, legacy_scales)
    np.testing.assert_array_equal(c1_codes, legacy_codes)
    np.testing.assert_array_equal(c1_decoded, legacy_decoded)


def test_angq_never_regresses_c1_and_uses_smaller_factor_on_tie() -> None:
    support, _ = _support(5)
    normalized = normalize_zid_rows(support)
    _, _, legacy = _quantize_rows(normalized)
    _, _, decoded, factors, cosines = quantize_d104_angq_normalized_rows(
        normalized
    )
    legacy_cosines = np.sum(
        normalized.astype(np.float64) * legacy.astype(np.float64),
        axis=1,
    )
    assert np.all(cosines >= legacy_cosines - 1.0e-15)
    np.testing.assert_allclose(
        cosines,
        np.sum(
            normalized.astype(np.float64) * decoded.astype(np.float64),
            axis=1,
        ),
        rtol=0.0,
        atol=3.0e-16,
    )
    assert np.all((factors >= 0.75) & (factors <= 1.25))

    one_hot = np.zeros((1, Z_DIM), dtype=np.float32)
    one_hot[0, 0] = 1.0
    _, _, one_hot_decoded, one_hot_factor, one_hot_cosine = (
        quantize_d104_angq_normalized_rows(one_hot)
    )
    assert one_hot_factor[0] == 0.75
    assert one_hot_cosine[0] == 1.0
    np.testing.assert_array_equal(one_hot_decoded, one_hot)


@pytest.mark.parametrize("k_shot", [1, 5, 10])
def test_d104_bank_keeps_typed_abi_bandwidth_and_wire_roundtrip(
    k_shot: int,
) -> None:
    support, labels = _support(k_shot)
    config = _lock(k_shot)
    bank = build_d104_angq_support_bank(
        support,
        labels,
        CLASSES,
        config=config,
    )
    audit = dict(bank.quantization_audit)
    assert audit["schema"] == ANGQ_SCHEMA
    assert audit["input_normalization_count"] == 1
    assert audit["candidate_input_renormalization_count"] == 0
    assert audit["query_features_used_for_scale"] == 0
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["query_truth_read"] is False
    assert audit["query_state_updates"] == 0
    assert bank.codes_qint8.dtype == np.int8
    assert bank.scales_fp16.dtype == np.float16
    assert bank.class_indices_int16.dtype == np.int16
    assert bank.class_scales_fp16.dtype == np.float16
    assert bank.codes_qint8.shape == (len(CLASSES) * k_shot, Z_DIM)
    if k_shot == 1:
        np.testing.assert_array_equal(
            bank.class_scales_fp16,
            np.full(len(CLASSES), config.shared_h0, dtype=np.float16),
        )
        assert audit["class_scale_source"] == "phase1_locked_shared_h0"
    else:
        assert (
            audit["class_scale_source"]
            == "angq_decoded_support_only_uniform_class_formula"
        )
    metric = identity_shared_psd_metric(config=config)
    wire = serialize_typed_zid_runtime_state(bank, metric)
    restored_bank, restored_metric = deserialize_typed_zid_runtime_state(wire)
    np.testing.assert_array_equal(restored_bank.codes_qint8, bank.codes_qint8)
    np.testing.assert_array_equal(restored_bank.scales_fp16, bank.scales_fp16)
    np.testing.assert_array_equal(
        restored_bank.class_scales_fp16,
        bank.class_scales_fp16,
    )
    assert restored_bank.bank_receipt_sha256 == bank.bank_receipt_sha256
    assert restored_metric.metric_receipt_sha256 == metric.metric_receipt_sha256


def test_resource_receipt_is_parameterized_and_query_neutral() -> None:
    support, labels = _support(5)
    legacy, angq, receipt = build_matched_legacy_and_d104_banks(
        support,
        labels,
        CLASSES,
        config=_lock(5),
    )
    assert receipt["registered_class_count"] == 3
    assert receipt["active_k"] == 5
    assert receipt["support_count"] == 15
    assert receipt["adaptation_mac_per_support"] == ANGQ_MAC_PER_SUPPORT
    assert receipt["adaptation_mac_total"] == 32320 * 3 * 5
    assert (
        receipt["adaptation_vector_elementwise_ops_per_support"]
        == ANGQ_VECTOR_ELEMENTWISE_OPS_PER_SUPPORT
    )
    assert (
        receipt["adaptation_vector_elementwise_ops_total"]
        == 64640 * 3 * 5
    )
    assert receipt["numeric_bank_array_bytes_delta"] == 0
    assert receipt["query_mac_delta"] == 0
    assert receipt["query_features_used_for_scale"] == 0
    assert receipt["query_truth_read"] is False
    assert receipt["passes_d104_resource_gate"] is True
    assert len(receipt["receipt_sha256"]) == 64
    assert (
        dict(audit_d104_angq_resource_delta(legacy, angq))
        == dict(receipt)
    )


def test_builder_has_no_query_or_role_surface_and_rejects_extra_keywords() -> None:
    signature = inspect.signature(build_d104_angq_support_bank)
    assert tuple(signature.parameters) == (
        "support_zid",
        "support_labels",
        "registered_classes",
        "config",
    )
    assert not any(
        token in name.lower()
        for name in signature.parameters
        for token in ("query", "truth", "old", "new", "receiver", "scene")
    )
    support, labels = _support(1)
    with pytest.raises(TypeError):
        build_d104_angq_support_bank(
            support,
            labels,
            CLASSES,
            config=_lock(1),
            query_zid=support,
        )
    with pytest.raises(TypeError):
        build_d104_angq_support_bank(
            support,
            labels,
            CLASSES,
            config=_lock(1),
            old_new_roles=("old", "new"),
        )


def test_class_registry_permutation_is_equivariant_and_ties_are_stable_first() -> None:
    support, labels = _support(1)
    config = _lock(1)
    first = build_d104_angq_support_bank(
        support,
        labels,
        CLASSES,
        config=config,
    )
    permutation = ("opaque_c", "opaque_a", "opaque_b")
    second = build_d104_angq_support_bank(
        support,
        labels,
        permutation,
        config=config,
    )
    for class_id in CLASSES:
        rows_first, scale_first = _class_payload(first, class_id)
        rows_second, scale_second = _class_payload(second, class_id)
        assert rows_first == rows_second
        assert scale_first == scale_second

    identical = np.zeros((len(CLASSES), Z_DIM), dtype=np.float32)
    identical[:, 0] = 1.0
    tie_bank = build_d104_angq_support_bank(
        identical,
        CLASSES,
        CLASSES,
        config=config,
    )
    logits = score_zid_student_t_logits(
        tie_bank,
        identical[:1],
        metric=identity_shared_psd_metric(config=config),
    )
    np.testing.assert_array_equal(logits[0], np.full(len(CLASSES), logits[0, 0]))
    assert int(np.argmax(logits[0])) == 0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: rows.astype(np.float64), "float32"),
        (
            lambda rows: np.pad(rows[:, :-1], ((0, 0), (0, 0))),
            "160",
        ),
        (
            lambda rows: np.where(
                np.arange(rows.size).reshape(rows.shape) == 0,
                np.float32(np.nan),
                rows,
            ),
            "finite",
        ),
        (lambda rows: np.zeros_like(rows), "zero-norm"),
    ],
)
def test_invalid_support_fails_closed(mutator, message: str) -> None:
    support, labels = _support(1)
    with pytest.raises((D104ANGQError, ValueError), match=message):
        build_d104_angq_support_bank(
            mutator(support),
            labels,
            CLASSES,
            config=_lock(1),
        )


def test_unbalanced_or_unregistered_support_fails_closed() -> None:
    support, labels = _support(5)
    with pytest.raises(D104ANGQError, match="balanced"):
        build_d104_angq_support_bank(
            support[:-1],
            labels[:-1],
            CLASSES,
            config=_lock(5),
        )
    bad_labels = list(labels)
    bad_labels[0] = "not_registered"
    with pytest.raises(D104ANGQError, match="opaque registry"):
        build_d104_angq_support_bank(
            support,
            bad_labels,
            CLASSES,
            config=_lock(5),
        )


def test_decoded_bank_contains_only_finite_unit_rows() -> None:
    support, labels = _support(10)
    bank = build_d104_angq_support_bank(
        support,
        labels,
        CLASSES,
        config=_lock(10),
    )
    decoded = decode_zid_support_bank(bank)
    assert np.isfinite(decoded).all()
    np.testing.assert_allclose(
        np.linalg.norm(decoded.astype(np.float64), axis=1),
        np.ones(len(decoded)),
        rtol=0.0,
        atol=2.0e-7,
    )


def test_full_tap_c1_verifier_has_only_immutable_tap_and_output_cli() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "verify_d104_angq_c1_full_tap.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--tap-archive" in result.stdout
    assert "--output-json" in result.stdout
    assert "query" not in result.stdout.lower()
    assert "truth" not in result.stdout.lower()


def test_full_tap_c1_verifier_binds_exact_archive_and_row_count() -> None:
    script_root = Path(__file__).resolve().parents[1] / "code" / "scripts"
    sys.path.insert(0, str(script_root))
    try:
        import verify_d104_angq_c1_full_tap as verifier

        assert verifier.EXPECTED_TAP_ROWS == 8400
        assert (
            verifier.EXPECTED_TAP_SHA256
            == "c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1"
        )
        assert verifier.HISTORICAL_DIAGNOSTIC_EXPOSED_ROWS == 2478
    finally:
        sys.path.remove(str(script_root))
