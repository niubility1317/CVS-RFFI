from __future__ import annotations

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_bidirectional_newguard as newguard


def _d42_roundtrip(
    coefficient: np.ndarray, intercept: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Use the deployed D42 two-level int8 plus FP16-intercept codec."""

    _, _, _, _, decoded = d42._quantize_coefficients(
        np.asarray(coefficient, dtype=np.float32)
    )
    return decoded, np.asarray(intercept, dtype=np.float16).astype(np.float32)


def _fixture(*, shots: int = 5):
    """A support-only state with a nontrivial protected new-class row span."""

    old_count, new_count, dimension = 6, 2, 288
    classes = old_count + new_count
    labels = np.repeat(np.arange(classes), shots).astype(np.int64)
    rows = np.zeros((classes * shots, dimension), dtype=np.float32)
    for class_index in range(old_count):
        start = class_index * shots
        rows[start : start + shots, class_index] = 1.0
        rows[start : start + shots, (class_index + 1) % old_count] = np.linspace(
            0.00, 0.16, shots, dtype=np.float32
        )
        rows[start : start + shots, 32 + class_index] = np.linspace(
            -0.10, 0.10, shots, dtype=np.float32
        )
    for class_index in range(old_count, classes):
        start = class_index * shots
        rows[start : start + shots, class_index] = 1.0
        rows[start : start + shots, 64 + class_index] = np.linspace(
            -0.06, 0.06, shots, dtype=np.float32
        )
    coefficient = np.zeros((classes, dimension), dtype=np.float32)
    coefficient[np.arange(classes), np.arange(classes)] = 1.0
    intercept = np.zeros(classes, dtype=np.float32)
    return rows, labels, coefficient, intercept, classes, shots


def _build(
    *,
    roundtrip=_d42_roundtrip,
    shots: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rows, labels, coefficient, intercept, classes, actual_shots = _fixture(
        shots=shots
    )
    return newguard.build_bidirectional_newguard_affine_state(
        full_rows=rows,
        full_labels=labels,
        full_coefficient=coefficient,
        full_intercept=intercept,
        class_count=classes,
        k_shot=actual_shots,
        quantize_decode=roundtrip,
    )


def test_newguard_uses_compact_nullspace_and_preserves_protected_support():
    """Would fail if NewGuard allocated a dense 289x289 projector or moved X_new."""

    coefficient, intercept, audit = _build()

    assert coefficient.shape == (8, 288)
    assert intercept.shape == (8,)
    assert audit["d92_newguard_active"] is True
    assert audit["d92_newguard_nullspace_operator"] == "compact_rowspace_svd"
    assert audit["d92_newguard_explicit_projector_bytes"] == 0
    assert audit["d92_newguard_nullspace_rank"] > 0
    assert audit["d92_newguard_rank_threshold"] > 0.0
    assert audit["d92_newguard_max_abs_Xnew_internal_residual"] <= audit[
        "d92_newguard_closure_tolerance"
    ]
    assert audit["d92_newguard_old_group_zero_sum_residual_max_abs"] <= audit[
        "d92_newguard_closure_tolerance"
    ]
    assert audit[
        "d92_newguard_deployment_max_abs_Xnew_internal_residual"
    ] <= audit["d92_newguard_closure_tolerance"]
    assert audit[
        "d92_newguard_deployment_old_group_zero_sum_residual_max_abs"
    ] <= audit["d92_newguard_closure_tolerance"]
    assert audit[
        "d92_newguard_deployment_new_support_old_envelope_change_max_abs_error"
    ] <= audit["d92_newguard_closure_tolerance"]


def test_newguard_protects_new_rows_tau_and_all_fixed_old_tails():
    """Would fail if an accepted state lowered a frozen old tail or lifted old envelope."""

    rows, labels, baseline_coefficient, baseline_intercept, classes, shots = _fixture()
    coefficient, intercept, audit = newguard.build_bidirectional_newguard_affine_state(
        full_rows=rows,
        full_labels=labels,
        full_coefficient=baseline_coefficient,
        full_intercept=baseline_intercept,
        class_count=classes,
        k_shot=shots,
        quantize_decode=_d42_roundtrip,
    )

    assert coefficient[6:].tobytes() == baseline_coefficient[6:].tobytes()
    assert intercept[6:].tobytes() == baseline_intercept[6:].tobytes()
    assert audit["d92_newguard_new_rows_byte_exact"] is True
    assert audit["d92_newguard_deployment_new_rows_byte_exact"] is True
    assert audit["d92_newguard_deployment_strength_scale"] == 1.0
    assert audit["d92_newguard_deployment_candidate_count"] == 1
    assert audit["d92_newguard_deployment_full_head_byte_exact"] is False
    assert audit["d92_newguard_deployment_codec_roundtrip_count"] == 2
    assert audit["d92_newguard_deployment_codec_macs_upper_bound"] == (
        2 * 8 * classes * 288
    )
    assert audit["d92_newguard_tau_old_envelope_shift"] <= 0.0
    assert audit["d92_newguard_new_support_min_margin_change"] >= -audit[
        "d92_newguard_protection_tolerance"
    ]
    assert all(
        value >= -audit["d92_newguard_protection_tolerance"]
        for value in audit["d92_newguard_tail_margin_change_by_old_class"]
    )
    assert all(
        value >= -audit["d92_newguard_protection_tolerance"]
        for value in audit[
            "d92_newguard_deployment_tail_margin_change_by_old_class"
        ]
    )
    assert audit[
        "d92_newguard_deployment_new_support_min_margin_change"
    ] >= -audit["d92_newguard_protection_tolerance"]
    assert audit[
        "d92_newguard_deployment_new_support_old_envelope_change_max"
    ] <= audit["d92_newguard_protection_tolerance"]
    assert audit["d92_newguard_deployment_protection_pass"] is True


def test_newguard_is_equivariant_under_independent_old_and_new_label_permutations():
    """Would fail if a direction or constraint privileged a TX/class identifier."""

    rows, labels, coefficient, intercept, classes, shots = _fixture()
    result_coefficient, result_intercept, result_audit = (
        newguard.build_bidirectional_newguard_affine_state(
            full_rows=rows,
            full_labels=labels,
            full_coefficient=coefficient,
            full_intercept=intercept,
            class_count=classes,
            k_shot=shots,
            quantize_decode=_d42_roundtrip,
        )
    )
    mapping = np.asarray([2, 0, 5, 1, 4, 3, 7, 6], dtype=np.int64)
    inverse = np.argsort(mapping)
    row_order = np.asarray(
        [
            31,
            1,
            20,
            7,
            36,
            4,
            12,
            0,
            28,
            9,
            16,
            3,
            11,
            21,
            6,
            8,
            13,
            14,
            15,
            17,
            19,
            22,
            23,
            24,
            25,
            26,
            27,
            29,
            30,
            32,
            33,
            34,
            35,
            37,
            38,
            39,
            2,
            5,
            10,
            18,
        ],
        dtype=np.int64,
    )
    p_coefficient, p_intercept, p_audit = (
        newguard.build_bidirectional_newguard_affine_state(
            full_rows=rows[row_order],
            full_labels=mapping[labels[row_order]],
            full_coefficient=coefficient[inverse],
            full_intercept=intercept[inverse],
            class_count=classes,
            k_shot=shots,
            quantize_decode=_d42_roundtrip,
        )
    )

    tolerance = max(
        result_audit["d92_newguard_protection_tolerance"],
        p_audit["d92_newguard_protection_tolerance"],
    )
    np.testing.assert_allclose(
        p_coefficient[mapping], result_coefficient, rtol=0.0, atol=tolerance
    )
    np.testing.assert_allclose(
        p_intercept[mapping], result_intercept, rtol=0.0, atol=tolerance
    )
    assert p_audit["d92_newguard_new_rows_byte_exact"] is True


def test_newguard_rechecks_deployed_d42_head_and_falls_back_byte_exactly():
    """Would fail if a floating-point-only guard published an unsafe deployed head."""

    rows, labels, baseline_coefficient, baseline_intercept, classes, shots = _fixture()
    full_delta = None
    candidate_call_count = 0

    def deployment_drift(
        coefficient: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal candidate_call_count, full_delta
        decoded_coefficient, decoded_intercept = _d42_roundtrip(
            coefficient, intercept
        )
        if (
            coefficient.tobytes() != baseline_coefficient.tobytes()
            or intercept.tobytes() != baseline_intercept.tobytes()
        ):
            candidate_call_count += 1
            delta = max(
                float(np.max(np.abs(coefficient - baseline_coefficient))),
                float(np.max(np.abs(intercept - baseline_intercept))),
            )
            if full_delta is None:
                full_delta = delta
            scale = delta / full_delta
            decoded_coefficient = decoded_coefficient.copy()
            decoded_coefficient[0, 0] -= np.float32(0.1577)
        return decoded_coefficient, decoded_intercept

    coefficient, intercept, audit = newguard.build_bidirectional_newguard_affine_state(
        full_rows=rows,
        full_labels=labels,
        full_coefficient=baseline_coefficient,
        full_intercept=baseline_intercept,
        class_count=classes,
        k_shot=shots,
        quantize_decode=deployment_drift,
    )

    assert candidate_call_count == 1
    assert coefficient.tobytes() == baseline_coefficient.tobytes()
    assert intercept.tobytes() == baseline_intercept.tobytes()
    assert audit["d92_newguard_active"] is False
    assert audit["d92_newguard_fallback_active"] is True
    assert audit["d92_newguard_fallback_reason"] == "deployment_protection_failed"
    assert audit["d92_newguard_full_head_byte_exact"] is True
    assert audit["d92_newguard_deployment_strength_scale"] is None
    assert audit["d92_newguard_deployment_candidate_count"] == 1
    assert audit["d92_newguard_deployment_full_head_byte_exact"] is True
    assert audit["d92_newguard_deployment_codec_roundtrip_count"] == 2
    assert audit["d92_newguard_deployment_codec_macs_upper_bound"] == (
        2 * 8 * classes * 288
    )
    assert audit["d92_newguard_support_optimization_macs_upper_bound"] > audit[
        "d92_newguard_deployment_codec_macs_upper_bound"
    ]


def test_newguard_returns_e0_after_single_candidate_deployed_tail_flip():
    """One unsafe deployed candidate must return exact E0 without a scale search."""

    rows, labels, baseline_coefficient, baseline_intercept, classes, shots = _fixture()
    full_delta = None
    attempted_scales = []

    def d42_tail_flip_above_half_scale(
        coefficient: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal full_delta
        deployed_coefficient, deployed_intercept = _d42_roundtrip(
            coefficient, intercept
        )
        delta = max(
            float(np.max(np.abs(coefficient - baseline_coefficient))),
            float(np.max(np.abs(intercept - baseline_intercept))),
        )
        if delta > 0.0:
            if full_delta is None:
                full_delta = delta / 128.0
            scale = delta / full_delta
            attempted_scales.append(scale)
        else:
            scale = 0.0
        if scale > 64.0 + 1.0e-6:
            deployed_coefficient = deployed_coefficient.copy()
            deployed_coefficient[0, 0] -= np.float32(0.1577)
        return deployed_coefficient, deployed_intercept

    coefficient, intercept, audit = newguard.build_bidirectional_newguard_affine_state(
        full_rows=rows,
        full_labels=labels,
        full_coefficient=baseline_coefficient,
        full_intercept=baseline_intercept,
        class_count=classes,
        k_shot=shots,
        quantize_decode=d42_tail_flip_above_half_scale,
    )

    np.testing.assert_allclose(attempted_scales, [128.0], rtol=0.0, atol=1.0e-4)
    assert coefficient.tobytes() == baseline_coefficient.tobytes()
    assert intercept.tobytes() == baseline_intercept.tobytes()
    assert audit["d92_newguard_active"] is False
    assert audit["d92_newguard_fallback_active"] is True
    assert audit["d92_newguard_fallback_reason"] == "deployment_protection_failed"
    assert audit["d92_newguard_deployment_strength_scale"] is None
    assert audit["d92_newguard_deployment_candidate_count"] == 1
    assert audit["d92_newguard_deployment_full_head_byte_exact"] is True
    assert audit["d92_newguard_deployment_protection_pass"] is False


def test_newguard_protection_failure_does_not_scan_a_second_strength():
    """The frozen method has one support-side candidate; any failed guard returns E0."""

    rows, labels, baseline_coefficient, baseline_intercept, classes, shots = _fixture()
    full_delta = None
    attempted_scales = []

    def large_scale_raw_tail_flip(
        coefficient: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal full_delta
        delta = max(
            float(np.max(np.abs(coefficient - baseline_coefficient))),
            float(np.max(np.abs(intercept - baseline_intercept))),
        )
        if delta > 0.0:
            if full_delta is None:
                full_delta = delta / 128.0
            attempted_scales.append(delta / full_delta)
        return _d42_roundtrip(coefficient, intercept)

    original_receipt = newguard._protection_receipt
    receipt_calls = 0

    def reject_first_raw_candidate(**kwargs):
        nonlocal receipt_calls
        receipt_calls += 1
        if receipt_calls == 1:
            raise newguard.D92NewGuardNumericalError("raw_protection_failed")
        return original_receipt(**kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(newguard, "_protection_receipt", reject_first_raw_candidate)
        coefficient, intercept, audit = newguard.build_bidirectional_newguard_affine_state(
            full_rows=rows,
            full_labels=labels,
            full_coefficient=baseline_coefficient,
            full_intercept=baseline_intercept,
            class_count=classes,
            k_shot=shots,
            quantize_decode=large_scale_raw_tail_flip,
        )

    assert attempted_scales == []
    assert receipt_calls == 1
    assert coefficient.tobytes() == baseline_coefficient.tobytes()
    assert intercept.tobytes() == baseline_intercept.tobytes()
    assert audit["d92_newguard_active"] is False
    assert audit["d92_newguard_fallback_active"] is True
    assert audit["d92_newguard_fallback_reason"] == "raw_protection_failed"
    assert audit["d92_newguard_deployment_strength_scale"] is None
    assert audit["d92_newguard_deployment_candidate_count"] == 0


def test_newguard_deployed_xnew_closure_failure_cannot_publish_active_head():
    """A small codec drift can pass margin guards but must fail the exact Xnew closure."""

    rows, labels, baseline_coefficient, baseline_intercept, classes, shots = _fixture()
    candidate_calls = 0

    def xnew_closure_drift(
        coefficient: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal candidate_calls
        deployed_coefficient, deployed_intercept = _d42_roundtrip(
            coefficient, intercept
        )
        if (
            coefficient.tobytes() != baseline_coefficient.tobytes()
            or intercept.tobytes() != baseline_intercept.tobytes()
        ):
            candidate_calls += 1
            deployed_coefficient = deployed_coefficient.copy()
            deployed_coefficient[0, 6] += np.float32(1.0e-4)
            deployed_coefficient[1, 6] -= np.float32(1.0e-4)
        return deployed_coefficient, deployed_intercept

    coefficient, intercept, audit = newguard.build_bidirectional_newguard_affine_state(
        full_rows=rows,
        full_labels=labels,
        full_coefficient=baseline_coefficient,
        full_intercept=baseline_intercept,
        class_count=classes,
        k_shot=shots,
        quantize_decode=xnew_closure_drift,
    )

    assert candidate_calls == 1
    assert coefficient.tobytes() == baseline_coefficient.tobytes()
    assert intercept.tobytes() == baseline_intercept.tobytes()
    assert audit["d92_newguard_active"] is False
    assert audit["d92_newguard_fallback_active"] is True
    assert audit["d92_newguard_fallback_reason"] == "deployment_protection_failed"
    assert audit["d92_newguard_deployment_protection_pass"] is False
    assert audit[
        "d92_newguard_deployment_max_abs_Xnew_internal_residual"
    ] > audit["d92_newguard_closure_tolerance"]


def test_newguard_rejects_registry_drift_instead_of_hiding_it_as_numeric_fallback():
    """Would fail if incomplete support labels were silently relabelled as E0 fallback."""

    rows, labels, coefficient, intercept, classes, shots = _fixture()
    labels[-1] = 6
    with pytest.raises(newguard.D92NewGuardError, match="registry"):
        newguard.build_bidirectional_newguard_affine_state(
            full_rows=rows,
            full_labels=labels,
            full_coefficient=coefficient,
            full_intercept=intercept,
            class_count=classes,
            k_shot=shots,
            quantize_decode=_d42_roundtrip,
        )
