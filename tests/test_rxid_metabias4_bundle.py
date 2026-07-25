from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_bundle import (
    AMAX_FP16,
    LAMBDA0_FP16,
    PAYLOAD_MEMBERS,
    PRECISION_BOUNDS,
    RADIUS_FP16,
    RXIDMetaBias4BundleError,
    SIGMA_BOUNDS,
    TEMPERATURE_FP16,
    build_rxid_metabias4_bundle,
    deserialize_rxid_metabias4_bundle,
    quantize_rowwise_symmetric_int8,
    serialize_rxid_metabias4_bundle,
)


HASHES = tuple(f"{index:x}" * 64 for index in range(1, 9))


def _values() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(103713)
    u = rng.normal(0.0, 0.2, (32, 160)).astype(np.float32)
    b = rng.normal(0.0, 0.1, (160, 4)).astype(np.float32)
    g = rng.normal(0.0, 0.3, (5, 32)).astype(np.float32)
    t = rng.normal(0.0, 0.2, (5, 4)).astype(np.float32)
    precision = np.geomspace(0.01, 30.0, 20).reshape(5, 4).astype(np.float32)
    sigma = np.asarray([0.01, 0.05, 0.4, 2.0, 3.0], dtype=np.float32)
    return u, b, g, t, precision, sigma


def _bundle():
    u, b, g, t, precision, sigma = _values()
    return build_rxid_metabias4_bundle(
        u,
        b,
        g,
        t,
        precision,
        sigma,
        cell_min_physical_count=np.full(5, 2, dtype=np.int16),
        cell_class_count=np.full(5, 3, dtype=np.int16),
        checkpoint_sha256=HASHES[0],
        runtime_sha256=HASHES[1],
        method_lock_sha256=HASHES[2],
        training_receipt_sha256=HASHES[3],
        nested_receipt_sha256=HASHES[4],
        tx_probe_receipt_sha256=HASHES[5],
        aggregation_receipt_sha256=HASHES[6],
        quantization_receipt_sha256=HASHES[7],
        tx_probe_mean_balanced_accuracy=0.20,
        tx_probe_max_balanced_accuracy=0.24,
    )


def test_rowwise_int8_uses_rne_forbids_minus128_and_closes_zero_rows() -> None:
    rows = np.zeros((2, 160), dtype=np.float32)
    rows[0, :7] = np.asarray(
        [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype=np.float32
    )
    codes, scales = quantize_rowwise_symmetric_int8(rows, name="golden")
    np.testing.assert_array_equal(
        codes[0, :7], np.asarray([127, 0, 2, 2, 0, -2, -2], dtype=np.int8)
    )
    assert scales[0] == np.float16(1.0)
    assert np.all(codes[1] == 0)
    assert scales[1] == np.float16(1.0)
    assert not np.any(codes == np.int8(-128))


def test_bundle_is_deterministic_int8_only_and_uses_frozen_binary16_scalars() -> None:
    first = _bundle()
    second = _bundle()
    assert first.content_root_sha256 == second.content_root_sha256
    assert serialize_rxid_metabias4_bundle(first) == serialize_rxid_metabias4_bundle(
        second
    )
    assert first.b_codes_qint8.shape == (160, 4)
    assert first.b_scales_fp16.shape == (160,)
    assert first.temperature_fp16.tobytes() == np.asarray(TEMPERATURE_FP16).tobytes()
    assert first.radius_fp16.tobytes() == np.asarray(RADIUS_FP16).tobytes()
    np.testing.assert_array_equal(first.lambda0_fp16, LAMBDA0_FP16)
    np.testing.assert_array_equal(first.amax_fp16, AMAX_FP16)
    learned_code_names = (
        "u_codes_qint8",
        "b_codes_qint8",
        "bank_g_codes_qint8",
        "bank_t_codes_qint8",
        "bank_precision_codes_qint8",
        "bank_sigma_codes_qint8",
    )
    assert all(getattr(first, name).dtype == np.int8 for name in learned_code_names)
    assert all(
        np.asarray(getattr(first, name)).dtype != np.float32
        for name in PAYLOAD_MEMBERS
    )
    assert first.numeric_state_bytes < 80 * 1024


def test_precision_and_sigma_are_clipped_then_log_affine_quantized() -> None:
    bundle = _bundle()
    *_, precision, sigma = _values()
    decoded_precision = bundle.decode_bank_precision()
    decoded_sigma = bundle.decode_bank_sigma()
    expected_precision = np.clip(precision, *PRECISION_BOUNDS)
    expected_sigma = np.clip(sigma, *SIGMA_BOUNDS)
    np.testing.assert_allclose(decoded_precision, expected_precision, rtol=0.035, atol=1e-3)
    np.testing.assert_allclose(decoded_sigma, expected_sigma, rtol=0.02, atol=1e-3)
    assert decoded_precision.min() >= PRECISION_BOUNDS[0]
    assert decoded_precision.max() <= PRECISION_BOUNDS[1]
    assert decoded_sigma.min() >= SIGMA_BOUNDS[0]
    assert decoded_sigma.max() <= SIGMA_BOUNDS[1]


def test_bundle_rejects_forbidden_code_tamper_but_keeps_failed_tx_diagnostic() -> None:
    bundle = _bundle()
    tampered = bundle.u_codes_qint8.copy()
    tampered[0, 0] = np.int8(-128)
    with pytest.raises(RXIDMetaBias4BundleError, match="forbidden"):
        replace(bundle, u_codes_qint8=tampered)
    u, b, g, t, precision, sigma = _values()
    diagnostic = build_rxid_metabias4_bundle(
        u,
        b,
        g,
        t,
        precision,
        sigma,
        cell_min_physical_count=np.full(5, 2, dtype=np.int16),
        cell_class_count=np.full(5, 3, dtype=np.int16),
        checkpoint_sha256=HASHES[0],
        runtime_sha256=HASHES[1],
        method_lock_sha256=HASHES[2],
        training_receipt_sha256=HASHES[3],
        nested_receipt_sha256=HASHES[4],
        tx_probe_receipt_sha256=HASHES[5],
        aggregation_receipt_sha256=HASHES[6],
        quantization_receipt_sha256=HASHES[7],
        tx_probe_mean_balanced_accuracy=0.20,
        tx_probe_max_balanced_accuracy=0.2501,
    )
    assert diagnostic.tx_probe_gate_pass is False
    with pytest.raises(RXIDMetaBias4BundleError, match="diagnostic-only"):
        serialize_rxid_metabias4_bundle(diagnostic)


def test_bundle_wire_roundtrip_is_exact_and_corruption_is_fail_closed() -> None:
    bundle = _bundle()
    wire = serialize_rxid_metabias4_bundle(bundle)
    restored = deserialize_rxid_metabias4_bundle(wire)
    assert type(restored) is type(bundle)
    assert restored.content_root_sha256 == bundle.content_root_sha256
    assert serialize_rxid_metabias4_bundle(restored) == wire
    for broken in (
        bytes([wire[0] ^ 1]) + wire[1:],
        wire[:-1],
        wire + b"\x00",
    ):
        with pytest.raises(RXIDMetaBias4BundleError):
            deserialize_rxid_metabias4_bundle(broken)
    flipped = bytearray(wire)
    flipped[-5] ^= 1
    with pytest.raises(RXIDMetaBias4BundleError):
        deserialize_rxid_metabias4_bundle(bytes(flipped))
