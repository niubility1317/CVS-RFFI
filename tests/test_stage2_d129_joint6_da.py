from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from cvsrffi import stage2_d129_joint6_da as da


_CHECKPOINT_SHA = "a" * 64
_PHASE1_SEAL_SHA = "b" * 64


def _cspar_asset() -> da.CSPAR2Asset:
    basis = np.zeros((da.Z_DIM, da.RANK), dtype=np.int8)
    basis[0, 0] = 1
    basis[1, 1] = 1
    return da.CSPAR2Asset(
        checkpoint_sha256=_CHECKPOINT_SHA,
        phase1_seal_sha256=_PHASE1_SEAL_SHA,
        basis_qint8=basis,
        basis_scale_fp16=np.ones(da.RANK, dtype=np.float16),
        alpha0_fp16=np.asarray([0.25, 0.125], dtype=np.float16),
        alpha_max_fp16=np.asarray([0.5], dtype=np.float16),
        eps_fp16=np.asarray([0.01], dtype=np.float16),
    )


def _srdh_asset() -> da.SRDH2Asset:
    p = np.zeros((da.Z_DIM, da.RANK), dtype=np.int8)
    q = np.zeros((da.Z_DIM, da.RANK), dtype=np.int8)
    p[0, 0] = 1
    p[1, 1] = 1
    q[2, 0] = 1
    q[3, 1] = 1
    return da.SRDH2Asset(
        checkpoint_sha256=_CHECKPOINT_SHA,
        phase1_seal_sha256=_PHASE1_SEAL_SHA,
        p_qint8=p,
        p_scale_fp16=np.ones(da.RANK, dtype=np.float16),
        q_qint8=q,
        q_scale_fp16=np.ones(da.RANK, dtype=np.float16),
        mean_fp16=np.zeros(da.RANK, dtype=np.float16),
        std_fp16=np.ones(da.RANK, dtype=np.float16),
        a_max_fp16=np.asarray([0.25], dtype=np.float16),
    )


def _support_k5() -> np.ndarray:
    rng = np.random.default_rng(20260803)
    support = rng.normal(0.0, 0.03, size=(3, 5, da.Z_DIM)).astype(np.float32)
    for class_index in range(support.shape[0]):
        for shot_index in range(support.shape[1]):
            support[class_index, shot_index, 0] += 1.0 + 0.09 * class_index
            support[class_index, shot_index, 1] += 0.35 + 0.04 * shot_index
            support[class_index, shot_index, 2] += 0.20 * (shot_index + 1)
            support[class_index, shot_index, 3] += 0.15 * (class_index + 1)
            support[class_index, shot_index, 10 + class_index] += 0.12
    return np.ascontiguousarray(support)


def _phase1_aggregate() -> np.ndarray:
    rng = np.random.default_rng(1329)
    aggregate = rng.normal(0.0, 0.03, size=(4, 5, 4, da.Z_DIM))
    for receiver in range(aggregate.shape[0]):
        for class_index in range(aggregate.shape[1]):
            aggregate[receiver, class_index, :, 0] += 0.55 + 0.08 * receiver
            aggregate[receiver, class_index, :, 1] += 0.20 * (receiver - 1.5)
            aggregate[receiver, class_index, :, 2] += 0.16 * (class_index - 2)
            aggregate[receiver, class_index, :, 3] += 0.06 * receiver * (class_index + 1)
    return np.ascontiguousarray(aggregate, dtype=np.float32)


def _normalised_support(support: np.ndarray) -> np.ndarray:
    return support.astype(np.float64) / np.linalg.norm(
        support.astype(np.float64), axis=2, keepdims=True
    )


def test_phase1_builder_is_deterministic_quantized_and_has_no_target_surface() -> None:
    aggregate = _phase1_aggregate()
    first = da.build_d129_phase1_assets(
        aggregate,
        checkpoint_sha256=_CHECKPOINT_SHA,
        phase1_seal_sha256=_PHASE1_SEAL_SHA,
    )
    second = da.build_d129_phase1_assets(
        aggregate.copy(order="C"),
        checkpoint_sha256=_CHECKPOINT_SHA,
        phase1_seal_sha256=_PHASE1_SEAL_SHA,
    )
    assert [da.serialize_d129_joint6_asset(asset) for asset in first] == [
        da.serialize_d129_joint6_asset(asset) for asset in second
    ]
    assert da.d129_phase1_aggregate_sha256(aggregate) == da.d129_phase1_aggregate_sha256(
        aggregate.copy(order="C")
    )
    c1, c2 = first
    assert c1.candidate_id == da.CSPAR2_CANDIDATE_ID
    assert c2.candidate_id == da.SRDH2_CANDIDATE_ID
    assert c1.basis_qint8.dtype == np.dtype("i1")
    assert c2.p_qint8.dtype == np.dtype("i1")
    assert c2.q_qint8.dtype == np.dtype("i1")
    assert all(array.dtype == np.dtype("<f2") for array in (
        c1.basis_scale_fp16,
        c1.alpha0_fp16,
        c2.p_scale_fp16,
        c2.q_scale_fp16,
        c2.mean_fp16,
        c2.std_fp16,
        c2.a_max_fp16,
    ))
    assert np.all(c1.alpha0_fp16 > 0.0)
    assert float(c2.a_max_fp16[0]) > 0.0
    signature = inspect.signature(da.build_d129_phase1_assets)
    assert set(signature.parameters) == {
        "phase1_receiver_class_z",
        "checkpoint_sha256",
        "phase1_seal_sha256",
    }
    with pytest.raises(da.D129Joint6DAError, match="rank-deficient"):
        da.build_d129_phase1_assets(
            np.ones((2, 2, 2, da.Z_DIM), dtype=np.float32),
            checkpoint_sha256=_CHECKPOINT_SHA,
            phase1_seal_sha256=_PHASE1_SEAL_SHA,
        )


@pytest.mark.parametrize("asset_factory", [_cspar_asset, _srdh_asset])
def test_wire_round_trip_is_int8_fp16_only_and_pinned(asset_factory) -> None:
    asset = asset_factory()
    payload = da.serialize_d129_joint6_asset(asset)
    assert b'"dtype":"<f4"' not in payload
    rebuilt = da.deserialize_d129_joint6_asset(
        payload,
        expected_sha256=da.d129_joint6_asset_sha256(asset),
        expected_checkpoint_sha256=_CHECKPOINT_SHA,
        expected_phase1_seal_sha256=_PHASE1_SEAL_SHA,
    )
    assert type(rebuilt) is type(asset)
    assert da.serialize_d129_joint6_asset(rebuilt) == payload
    with pytest.raises(da.D129Joint6DAError, match="checkpoint pin"):
        da.deserialize_d129_joint6_asset(
            payload, expected_checkpoint_sha256="c" * 64
        )
    with pytest.raises(da.D129Joint6DAError, match="trailing bytes"):
        da.deserialize_d129_joint6_asset(payload + b"x")


def test_cspar2_k1_k5_formula_spd_query_readonly_and_permutation() -> None:
    asset = _cspar_asset()
    support = _support_k5()
    state_k1 = da.fit_cspar2_support(asset, support[:, :1])
    state_k5 = da.fit_cspar2_support(asset, support)
    np.testing.assert_array_equal(state_k1.alpha_fp16, asset.alpha0_fp16)
    normalized = _normalised_support(support)
    residual = normalized - normalized.mean(axis=1, keepdims=True)
    denominator = support.shape[0] * (support.shape[1] - 1)
    basis = da.decode_cspar2_basis(asset)
    trace = np.square(residual).sum() / denominator
    axial = np.square(residual @ basis).sum(axis=(0, 1)) / denominator
    v_perp = max((trace - axial.sum()) / (da.Z_DIM - da.RANK), 0.0)
    alpha = np.clip(
        1.0 - (v_perp + float(asset.eps_fp16[0])) / (axial + float(asset.eps_fp16[0])),
        0.0,
        float(asset.alpha_max_fp16[0]),
    )
    expected = np.minimum(
        np.asarray(alpha, dtype=np.float16), asset.alpha_max_fp16[0]
    )
    np.testing.assert_array_equal(state_k5.alpha_fp16, expected)
    query = support[0, :2].copy()
    query_before = query.copy()
    transformed = da.transform_cspar2(asset, state_k1, query)
    assert transformed.dtype == np.float32 and not transformed.flags.writeable
    np.testing.assert_array_equal(query, query_before)
    assert np.allclose(np.linalg.norm(transformed, axis=1), 1.0, atol=2.0e-6)
    assert not np.array_equal(transformed, query)
    metric = da.cspar2_metric_matrix(asset, state_k5)
    assert np.all(np.linalg.eigvalsh(metric) > 0.0)
    audit = da.audit_d129_query_read_only(asset, state_k5, query)
    assert audit["protocol_closed"] is True
    assert all(audit[key] == 0 for key in (
        "query_rows_used_for_fit",
        "query_state_updates",
        "query_selection_count",
        "query_gradient_calls",
        "truth_role_quota_inputs",
        "global_reassignment_calls",
    ))
    receipt = da.d129_label_permutation_receipt(
        asset, support, np.asarray([2, 0, 1], dtype=np.int64), query
    )
    assert receipt["coefficient_bitwise_equal"] is True
    assert receipt["query_map_bitwise_equal"] is True


def test_cspar_quantized_near_orthonormal_seal_uses_one_polar_basis_everywhere() -> None:
    basis = np.zeros((da.Z_DIM, da.RANK), dtype=np.int8)
    basis[0, 0] = 1
    basis[1, 1] = 1
    asset = da.CSPAR2Asset(
        checkpoint_sha256=_CHECKPOINT_SHA,
        phase1_seal_sha256=_PHASE1_SEAL_SHA,
        basis_qint8=basis,
        basis_scale_fp16=np.asarray([1.024, 1.024], dtype=np.float16),
        alpha0_fp16=np.asarray([0.2, 0.2], dtype=np.float16),
        alpha_max_fp16=np.asarray([0.5], dtype=np.float16),
        eps_fp16=np.asarray([0.01], dtype=np.float16),
    )
    decoded = da.decode_cspar2_basis(asset)
    np.testing.assert_allclose(decoded.T @ decoded, np.eye(2), rtol=0.0, atol=1e-12)
    support = np.zeros((3, 5, da.Z_DIM), dtype=np.float32)
    coefficients = np.asarray([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float32)
    support[:, :, 2] = 1.0
    support[:, :, 0] = coefficients[None, :]
    state = da.fit_cspar2_support(asset, support)
    transformed = da.transform_cspar2(asset, state, support.reshape(-1, da.Z_DIM))
    assert state.active_k == 5
    assert np.isfinite(transformed).all()


def test_srdh2_k1_k5_nonlinear_query_readonly_and_permutation() -> None:
    asset = _srdh_asset()
    support = _support_k5()
    state_k1 = da.fit_srdh2_support(asset, support[:, :1])
    state_k5 = da.fit_srdh2_support(asset, support)
    normalized = _normalised_support(support)
    _p, q = da.decode_srdh2_dictionary(asset)
    summary = np.mean(np.mean(np.tanh(normalized @ q), axis=1), axis=0)
    response = float(asset.a_max_fp16[0]) * np.tanh(
        (summary - asset.mean_fp16.astype(np.float64)) / asset.std_fp16.astype(np.float64)
    )
    expected_response = np.clip(
        np.asarray(response, dtype=np.float16),
        -asset.a_max_fp16[0],
        asset.a_max_fp16[0],
    )
    np.testing.assert_array_equal(state_k5.response_fp16, expected_response)
    assert state_k1.receipt.active_k == 1
    assert state_k5.receipt.active_k == 5
    assert np.all(np.abs(state_k5.response_fp16) <= asset.a_max_fp16[0])
    query = support[1, :2].copy()
    transformed = da.transform_srdh2(asset, state_k5, query)
    assert transformed.dtype == np.float32 and not transformed.flags.writeable
    assert np.allclose(np.linalg.norm(transformed, axis=1), 1.0, atol=2.0e-6)
    assert not np.array_equal(transformed, query)
    audit = da.audit_d129_query_read_only(asset, state_k5, query)
    assert audit["protocol_closed"] is True
    receipt = da.d129_label_permutation_receipt(
        asset, support, np.asarray([1, 2, 0], dtype=np.int64), query
    )
    assert receipt["coefficient_bitwise_equal"] is True
    assert receipt["query_map_bitwise_equal"] is True


def _loco_records() -> list[da.D129LOCORecord]:
    return [
        da.D129LOCORecord(f"receiver-{receiver}", f"class-{class_index}", f"physical-{receiver}-{class_index}-{row}")
        for receiver in range(7)
        for class_index in range(6)
        for row in range(14)
    ]


def test_loco_plan_preserves_phase1_exclusion_and_full_registration_support() -> None:
    records = _loco_records()
    plan = da.build_d129_loco_plan(records)
    reverse = da.build_d129_loco_plan(list(reversed(records)))
    receipt = plan.coverage_receipt()
    assert len(plan.folds) == 42
    assert receipt["fold_count"] == 42
    assert receipt["k1_is_k5_prefix_all_folds"] is True
    assert receipt["physical_ids_persisted_in_plan"] == 0
    assert all(fold.phase1_fit_count == 420 for fold in plan.folds)
    assert all(fold.support_k1_count == 6 for fold in plan.folds)
    assert all(fold.support_k5_count == 30 for fold in plan.folds)
    assert all(fold.outer_query_count == 54 for fold in plan.folds)
    assert all(fold.k1_is_k5_prefix for fold in plan.folds)
    assert receipt == reverse.coverage_receipt()
    assert "physical-0-0-0" not in json.dumps(receipt, sort_keys=True)


def test_loco_plan_fails_closed_for_missing_or_reused_physical_ids() -> None:
    records = _loco_records()
    with pytest.raises(da.D129Joint6DAError, match="14 physical IDs"):
        da.build_d129_loco_plan(records[:-1])
    duplicate = records + [
        da.D129LOCORecord("receiver-1", "class-1", "physical-0-0-0")
    ]
    with pytest.raises(da.D129Joint6DAError, match="reuses a physical ID"):
        da.build_d129_loco_plan(duplicate)


def test_rejects_unsupported_k_malformed_assets_and_forbidden_dependency_chain() -> None:
    support = _support_k5()
    with pytest.raises(da.D129Joint6DAError, match="K in"):
        da.fit_cspar2_support(_cspar_asset(), support[:, :2])
    with pytest.raises(da.D129Joint6DAError, match=r"exact \|i1"):
        da.CSPAR2Asset(
            checkpoint_sha256=_CHECKPOINT_SHA,
            phase1_seal_sha256=_PHASE1_SEAL_SHA,
            basis_qint8=np.ones((da.Z_DIM, da.RANK), dtype=np.float32),
            basis_scale_fp16=np.ones(da.RANK, dtype=np.float16),
            alpha0_fp16=np.asarray([0.1, 0.1], dtype=np.float16),
            alpha_max_fp16=np.asarray([0.5], dtype=np.float16),
            eps_fp16=np.asarray([0.01], dtype=np.float16),
        )
    source = inspect.getsource(da)
    assert "import torch" not in source
    assert "stage2_d127" not in source
    assert "stage2_d128" not in source
