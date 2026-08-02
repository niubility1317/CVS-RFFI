from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json

import numpy as np
import pytest

from cvsrffi import stage2_d122_rdce_ground_head as d122
from cvsrffi import stage2_d123_loo_cres as d123
from cvsrffi.stage2_d112_seam_bundle import FEATURE_DIM, build_d112_source_held_g1_bundle
from cvsrffi.stage2_d112_seam_qknn import (
    fit_d112_ground_head_source_held_g1_state,
    score_d112_seam_source_held_g1_logits,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
)


OLD = tuple(f"old-{index}" for index in range(6))
CLASSES = OLD + ("new-0",)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ground = np.zeros((6, FEATURE_DIM), dtype=np.float64)
    for index in range(6):
        ground[index, 10 + index] = 1.0
    q0 = np.sum(ground, axis=0)
    q0 /= np.linalg.norm(q0)
    basis = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    basis[np.arange(3), np.arange(3)] = 1.0
    return ground, q0, basis


def _bundle():
    ground, q0, basis = _geometry()
    return build_d112_source_held_g1_bundle(
        class_registry=OLD,
        g=ground,
        q0=q0,
        U=basis,
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
    ground, _q0, _basis = _geometry()
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_id in enumerate(CLASSES):
        base = ground[class_index].copy() if class_index < 6 else np.eye(1, FEATURE_DIM, 40)[0]
        if class_index < 6:
            base[80] = 5.0
        for sample_index in range(k):
            row = base.copy()
            row[0] += 0.01 * (sample_index + 1)
            row[2] -= 0.004 * (class_index + 1) * (sample_index + 1)
            row /= np.linalg.norm(row)
            rows.append(row)
            labels.append(class_id)
    return np.asarray(rows, dtype=np.float32), tuple(labels)


def _rdce_state(raw_support: np.ndarray, k: int) -> dict[str, object]:
    normalized = np.asarray(raw_support, dtype=np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    basis = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    basis[0, [0, 1]] = [0.98, 0.14]
    basis[1, [1, 2]] = [0.91, -0.19]
    basis[2, [0, 2, 3]] = [-0.11, 0.87, 0.22]
    attenuation = np.asarray([0.300048828125, 0.349853515625, 0.39990234375])
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
    return {"basis": basis, "attenuation": attenuation, "payload": payload, "receipt": _sha(payload)}


def _inputs(k: int = 5):
    raw, labels = _raw_support(k)
    identity_bank = build_typed_zid_support_bank(raw, labels, CLASSES, config=_lock(k))
    rdce = _rdce_state(raw, k)
    transformed, _ = d122._d106_like_transform(raw, rdce["basis"], rdce["attenuation"])
    rdce_bank = build_typed_zid_support_bank(transformed, labels, CLASSES, config=_lock(k))
    return raw, labels, identity_bank, rdce, rdce_bank, transformed


def test_fit_apis_have_no_truth_or_role_surface() -> None:
    identity = set(inspect.signature(d123.fit_d123_loo_cres_ground_head_source_held_g1_state).parameters)
    rdce = set(inspect.signature(d123.fit_d123_loo_cres_rdce_ground_head_source_held_g1_state).parameters)
    assert identity == {"bundle", "bank"}
    assert rdce == {"bundle", "bank", "raw_support_zid", "support_labels", "rdce_state"}
    assert not (identity | rdce) & {"held_class", "query", "truth", "role", "quota", "selection"}


def test_identity_and_rdce_shrink_only_old_and_preserve_reference_dtype() -> None:
    raw, labels, identity_bank, rdce, rdce_bank, transformed = _inputs()
    identity = d123.fit_d123_loo_cres_ground_head_source_held_g1_state(_bundle(), identity_bank)
    joint = d123.fit_d123_loo_cres_rdce_ground_head_source_held_g1_state(
        _bundle(), rdce_bank, raw, labels, rdce
    )
    assert identity.rho.dtype == identity.reference_state.rho.dtype == np.float32
    assert joint.rho.dtype == joint.reference_state.rho.dtype == np.float64
    assert np.all(identity.rho <= identity.reference_state.rho)
    assert np.all(joint.rho <= joint.reference_state.rho)
    assert np.any(identity.delta[np.asarray(identity.old_class_indices)] > 0.0)
    new = CLASSES.index("new-0")
    identity_query = raw[:4]
    rdce_query = transformed[:4]
    identity_reference = score_d112_seam_source_held_g1_logits(
        identity.reference_state, identity_bank, identity_query
    )
    joint_reference = d122.score_d122_rdce_ground_head_source_held_g1_logits(
        joint.reference_state, rdce_bank, rdce_query
    )
    assert np.array_equal(
        d123.score_d123_loo_cres_ground_head_source_held_g1_logits(
            identity, identity_bank, identity_query
        )[:, new],
        identity_reference[:, new],
    )
    assert np.array_equal(
        d123.score_d123_loo_cres_rdce_ground_head_source_held_g1_logits(
            joint, rdce_bank, rdce_query
        )[:, new],
        joint_reference[:, new],
    )
    for state in (identity, joint):
        audit = d123.audit_d123_loo_cres_state(state)
        assert [audit[name] for name in (
            "query_rows_used_for_fit", "query_state_updates", "query_selection_count", "truth_role_quota_inputs"
        )] == [0, 0, 0, 0]


def test_delta_is_frozen_leave_one_out_median_excess_and_permutation_equivariant() -> None:
    reference = np.asarray([0.4] * 6, dtype=np.float32)
    valid = np.ones(6, dtype=np.bool_)
    v_s = np.asarray([0.2] * 6)
    v_g = np.asarray([0.1] * 6)
    discrepancy = np.asarray([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    actual = d123._cres_arrays(
        reference_rho=reference,
        information_valid=valid,
        old_class_indices=tuple(range(6)),
        v_s=v_s,
        v_g=v_g,
        discrepancy=discrepancy,
    )
    expected_zero = max(0.0, float(np.median(discrepancy[1:] - v_s[1:] - v_g[1:])))
    assert actual[1][0] == expected_zero
    permutation = np.asarray([3, 1, 5, 0, 4, 2])
    permuted = d123._cres_arrays(
        reference_rho=reference[permutation],
        information_valid=valid[permutation],
        old_class_indices=tuple(range(6)),
        v_s=v_s[permutation],
        v_g=v_g[permutation],
        discrepancy=discrepancy[permutation],
    )
    assert np.array_equal(permuted[0], actual[0][permutation])
    assert np.array_equal(permuted[1], actual[1][permutation])


def test_fewer_than_three_donors_reuses_reference_scores_bit_exactly() -> None:
    raw, _labels, bank, _rdce, _rdce_bank, _transformed = _inputs()
    reference = fit_d112_ground_head_source_held_g1_state(_bundle(), bank)
    valid = np.zeros(len(CLASSES), dtype=np.bool_)
    valid[np.asarray(reference.old_class_indices[:3])] = True
    v_g = np.zeros(len(CLASSES), dtype=np.float64)
    for position, index in enumerate(reference.old_class_indices):
        v_g[index] = _bundle().v_g_amb[position]
    arrays = d123._cres_arrays(
        reference_rho=reference.rho,
        information_valid=valid,
        old_class_indices=tuple(reference.old_class_indices),
        v_s=np.asarray(reference.v_s_amb, dtype=np.float64),
        v_g=v_g,
        discrepancy=np.asarray(reference.discrepancy_amb, dtype=np.float64),
    )
    state = d123._make_state(d123.D123LOOCRESGroundHeadState, reference, arrays)
    expected = score_d112_seam_source_held_g1_logits(reference, bank, raw[:5])
    actual = d123.score_d123_loo_cres_ground_head_source_held_g1_logits(state, bank, raw[:5])
    assert not np.any(state.cres_applied)
    assert np.array_equal(actual, expected)


def test_placeholder_receipt_is_not_scoreable_and_rdce_tie_fails_closed(monkeypatch) -> None:
    raw, labels, _identity_bank, rdce, bank, transformed = _inputs()
    state = d123.fit_d123_loo_cres_rdce_ground_head_source_held_g1_state(
        _bundle(), bank, raw, labels, rdce
    )
    placeholder = replace(state, state_receipt_sha256="0" * 64)
    with pytest.raises(d123.D123LOOCRESError, match="placeholder"):
        d123.audit_d123_loo_cres_state(placeholder)
    monkeypatch.setattr(
        d123,
        "score_d123_loo_cres_rdce_ground_head_source_held_g1_logits",
        lambda *_args: np.asarray([[1.0, 1.0] + [0.0] * (len(CLASSES) - 2)], dtype=np.float32),
    )
    with pytest.raises(d122.D122RDCEGroundHeadError, match="CLASS_SCORE_TIE_UNRESOLVED"):
        d123.predict_d123_loo_cres_source_held_g1(state, bank, transformed[:1])
