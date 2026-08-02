from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest

from cvsrffi import stage2_d122_rdce_ground_head as d122
from cvsrffi.stage2_d112_seam_bundle import FEATURE_DIM, build_d112_source_held_g1_bundle
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


OLD = tuple(f"old-{index}" for index in range(6))
CLASSES = OLD + ("new-0",)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ground = np.zeros((6, FEATURE_DIM), dtype=np.float64)
    for index in range(6):
        ground[index, 10 + index] = 1.0
    q0 = np.sum(ground, axis=0)
    q0 /= np.linalg.norm(q0)
    U = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    U[np.arange(3), np.arange(3)] = 1.0
    return ground, q0, U


def _bundle():
    ground, q0, U = _geometry()
    return build_d112_source_held_g1_bundle(
        class_registry=OLD,
        g=ground,
        q0=q0,
        U=U,
        sigma0_r=np.linspace(0.0020, 0.0025, 6),
        sigma0_amb=np.linspace(0.0020, 0.0025, 6),
        v_g_r=np.linspace(0.0010, 0.0015, 6),
        v_g_amb=np.linspace(0.0010, 0.0015, 6),
        tau_h_r=0.004,
        checkpoint_sha256="1" * 64,
        source_aggregate_sha256="2" * 64,
        phase1_seal_sha256="3" * 64,
        source_held_split_sha256="4" * 64,
    )


def _lock(k: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.2,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256="5" * 64,
        quantization_margin_audit_sha256="6" * 64,
    )


def _raw_support(k: int) -> tuple[np.ndarray, tuple[str, ...]]:
    ground, _q0, _U = _geometry()
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_id in enumerate(CLASSES):
        base = ground[class_index] if class_index < 6 else np.eye(1, FEATURE_DIM, 40)[0]
        for sample_index in range(k):
            row = base.copy()
            row[0] = 0.015 * (sample_index + 1)
            row[2] = -0.007 * (class_index + 1) * (sample_index + 1)
            row /= np.linalg.norm(row)
            rows.append(row)
            labels.append(class_id)
    return np.asarray(rows, dtype=np.float32), tuple(labels)


def _basis() -> np.ndarray:
    # Deliberately not orthogonal after the notional INT8 decode.
    basis = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    basis[0, 0] = 0.98
    basis[0, 1] = 0.14
    basis[1, 1] = 0.91
    basis[1, 2] = -0.19
    basis[2, 0] = -0.11
    basis[2, 2] = 0.87
    basis[2, 3] = 0.22
    return basis


def _rdce_state(raw_support: np.ndarray, k: int) -> dict[str, object]:
    normalized = np.asarray(raw_support, dtype=np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    attenuation = np.asarray([0.300048828125, 0.349853515625, 0.39990234375], dtype=np.float64)
    payload = {
        "scope": "SOURCE_HELD_NON_TARGET_NO_P2_AUTHORITY",
        "asset_receipt_sha256": "7" * 64,
        "K": k,
        "attenuation_fp16": [float(value) for value in attenuation],
        "support_root_sha256": hashlib.sha256(
            np.ascontiguousarray(normalized, dtype=np.float64).tobytes()
        ).hexdigest(),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
    }
    return {
        "basis": _basis(),
        "attenuation": attenuation,
        "payload": payload,
        "receipt": _canonical_sha(payload),
    }


def _joint_inputs(k: int, *, basis: np.ndarray | None = None):
    raw_support, labels = _raw_support(k)
    rdce = _rdce_state(raw_support, k)
    if basis is not None:
        rdce["basis"] = basis
    transformed, _norm = d122._d106_like_transform(
        raw_support,
        np.asarray(rdce["basis"], dtype=np.float64),
        np.asarray(rdce["attenuation"], dtype=np.float64),
    )
    bank = build_typed_zid_support_bank(transformed, labels, CLASSES, config=_lock(k))
    return raw_support, labels, rdce, bank


def test_low_rank_jacobian_matches_dense_without_assuming_orthogonality() -> None:
    basis = _basis()
    assert not np.allclose(basis @ basis.T, np.eye(3), atol=1.0e-6, rtol=0.0)
    attenuation = np.asarray([0.300048828125, 0.349853515625, 0.39990234375])
    raw = np.zeros(FEATURE_DIM, dtype=np.float64)
    raw[[0, 4, 31]] = [0.4, 0.7, -0.2]
    raw /= np.linalg.norm(raw)
    transformed, squared = d122._d106_like_transform(raw[None, :], basis, attenuation)
    actual = d122._low_rank_jacobian_multiplier(
        basis=basis,
        attenuation=attenuation,
        transformed=transformed[0],
        ax_squared_norm=float(squared[0]),
    )
    expected = d122.d122_dense_jacobian_multiplier(
        basis=basis,
        attenuation=attenuation,
        raw_unit_point=raw,
    )
    assert np.isclose(actual, expected, atol=2.0e-8, rtol=0.0)


@pytest.mark.parametrize("k", [1, 5, 10])
def test_joint_keeps_all_new_logits_bit_exactly_at_m_da(k: int) -> None:
    raw_support, labels, rdce, bank = _joint_inputs(k)
    state = d122.fit_d122_rdce_ground_head_source_held_g1_state(
        _bundle(), bank, raw_support, labels, rdce
    )
    query = np.vstack((raw_support[:3], raw_support[-3:])).astype(np.float32)
    transformed_query, _ = d122._d106_like_transform(
        query,
        np.asarray(rdce["basis"], dtype=np.float64),
        np.asarray(rdce["attenuation"], dtype=np.float64),
    )
    baseline = score_zid_student_t_logits(
        bank, transformed_query, metric=identity_shared_psd_metric(config=bank.config)
    )
    actual = d122.score_d122_rdce_ground_head_source_held_g1_logits(
        state, bank, transformed_query
    )
    new_index = CLASSES.index("new-0")
    assert state.global_component_valid
    assert np.all(state.information_valid[np.asarray(state.old_class_indices)])
    assert np.array_equal(actual[:, new_index], baseline[:, new_index])
    assert np.max(np.abs(actual[:, :6] - baseline[:, :6])) > 0.0
    audit = d122.audit_d122_rdce_ground_head_state(state)
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["query_state_updates"] == 0
    assert audit["query_selection_count"] == 0
    assert audit["resource_receipt"]["query_dependent_state_bytes"] == 0


def test_global_rdce_receipt_failure_is_exact_m_da_for_every_column() -> None:
    raw_support, labels, rdce, bank = _joint_inputs(1)
    rdce["receipt"] = "0" * 64
    state = d122.fit_d122_rdce_ground_head_source_held_g1_state(
        _bundle(), bank, raw_support, labels, rdce
    )
    query = raw_support[:2]
    transformed_query, _ = d122._d106_like_transform(
        query,
        _basis(),
        np.asarray(rdce["attenuation"], dtype=np.float64),
    )
    baseline = score_zid_student_t_logits(
        bank, transformed_query, metric=identity_shared_psd_metric(config=bank.config)
    )
    actual = d122.score_d122_rdce_ground_head_source_held_g1_logits(
        state, bank, transformed_query
    )
    assert not state.global_component_valid
    assert state.global_failure_reason == "RDCE_STATE_RECEIPT_OR_BINDING_INVALID"
    assert np.array_equal(actual, baseline)


def test_one_old_class_geometry_failure_is_only_that_class_exact_m_da() -> None:
    attenuation = np.asarray([0.300048828125, 0.349853515625, 0.39990234375])
    coefficient = 1.0 - np.sqrt(1.0 - attenuation[0])
    singular_for_old_zero = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    # A=I-B.T diag(c)B maps the immutable old-0 ground e10 to zero, while
    # every raw support row retains its deterministic e0/e2 perturbation.
    singular_for_old_zero[0, 10] = 1.0 / np.sqrt(coefficient)
    raw_support, labels, rdce, bank = _joint_inputs(5, basis=singular_for_old_zero)
    state = d122.fit_d122_rdce_ground_head_source_held_g1_state(
        _bundle(), bank, raw_support, labels, rdce
    )
    transformed_query, _ = d122._d106_like_transform(
        raw_support[:2], singular_for_old_zero, attenuation
    )
    baseline = score_zid_student_t_logits(
        bank, transformed_query, metric=identity_shared_psd_metric(config=bank.config)
    )
    actual = d122.score_d122_rdce_ground_head_source_held_g1_logits(
        state, bank, transformed_query
    )
    old_zero = CLASSES.index("old-0")
    assert state.fallback_to_m_da[old_zero]
    assert not state.information_valid[old_zero]
    assert np.array_equal(actual[:, old_zero], baseline[:, old_zero])
    assert np.count_nonzero(state.information_valid[np.asarray(state.old_class_indices)]) >= 1


@pytest.mark.parametrize("mismatch", ["labels", "row_order"])
def test_same_count_wrong_raw_to_rdce_bank_binding_is_rejected(mismatch: str) -> None:
    raw_support, labels, rdce, bank = _joint_inputs(5)
    if mismatch == "labels":
        # Counts remain exactly K for every registered class, but the raw
        # support-to-label pairing is not the pairing that produced the bank.
        supplied_support = raw_support
        supplied_labels = labels[5:] + labels[:5]
        supplied_rdce = rdce
    else:
        # Typed qKNN canonicalization makes a benign input reordering legal;
        # this is instead an order/pairing mismatch with labels held fixed.
        supplied_support = np.ascontiguousarray(raw_support[::-1])
        supplied_labels = labels
        supplied_rdce = _rdce_state(supplied_support, 5)
    assert all(supplied_labels.count(class_id) == 5 for class_id in CLASSES)
    with pytest.raises(d122.D122RDCEGroundHeadError, match="support-bank binding"):
        d122.fit_d122_rdce_ground_head_source_held_g1_state(
            _bundle(), bank, supplied_support, supplied_labels, supplied_rdce
        )


def test_fit_surface_has_no_query_truth_role_quota_or_selection_argument() -> None:
    parameters = set(
        inspect.signature(d122.fit_d122_rdce_ground_head_source_held_g1_state).parameters
    )
    assert parameters == {
        "bundle",
        "bank",
        "raw_support_zid",
        "support_labels",
        "rdce_state",
    }
    assert not parameters & {"query", "truth", "role", "quota", "selection"}


def test_cross_class_bit_exact_tie_fails_closed() -> None:
    with pytest.raises(d122.D122RDCEGroundHeadError, match="CLASS_SCORE_TIE_UNRESOLVED"):
        d122.unique_d122_argmax(
            np.asarray([[1.0, 1.0, 0.5]], dtype=np.float32),
            ("a", "b", "c"),
        )
