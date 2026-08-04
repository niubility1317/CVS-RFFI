from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cvsrffi import stage2_next_r4_fa_rdce3 as fa


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _unit(index: int) -> np.ndarray:
    value = np.zeros((1, fa.Z_DIM), dtype=np.float32)
    value[0, index] = np.float32(1.0)
    return value


def _asset(
    *,
    old_classes: tuple[str, ...] = ("c0", "c1"),
    centers: np.ndarray | None = None,
    rho: float = 1.0,
    kappa: tuple[float, float, float] = (0.25, 0.0, 0.0),
    basis_codes: np.ndarray | None = None,
) -> fa.FARDCE3Phase1Asset:
    if centers is None:
        centers = np.zeros((len(old_classes), fa.RANK), dtype=np.int8)
    if basis_codes is None:
        basis_codes = np.zeros((fa.RANK, fa.Z_DIM), dtype=np.int8)
        basis_codes[0, 0] = 1
        basis_codes[1, 1] = 1
        basis_codes[2, 2] = 1
    return fa.FARDCE3Phase1Asset(
        old_classes=old_classes,
        aggregate_samples_per_class=tuple(2 for _ in old_classes),
        centers_codes_qint8=np.asarray(centers, dtype=np.int8),
        centers_scales_fp16=np.ones(len(old_classes), dtype=np.float16),
        fisher_codes_qint8=np.ones(fa.RANK, dtype=np.int8),
        fisher_scales_fp16=np.full(fa.RANK, np.float16(0.5), dtype=np.float16),
        residual_variance_codes_qint8=np.ones(fa.RANK, dtype=np.int8),
        residual_variance_scales_fp16=np.ones(fa.RANK, dtype=np.float16),
        rho_codes_qint8=np.asarray([1], dtype=np.int8),
        rho_scales_fp16=np.asarray([rho], dtype=np.float16),
        kappa_codes_qint8=np.asarray([1 if item > 0.0 else 0 for item in kappa], dtype=np.int8),
        kappa_scales_fp16=np.asarray(
            [item if item > 0.0 else 1.0 for item in kappa], dtype=np.float16
        ),
        basis_codes_qint8=np.asarray(basis_codes, dtype=np.int8),
        basis_scales_fp16=np.ones(fa.RANK, dtype=np.float16),
        checkpoint_sha256=_sha("checkpoint"),
        phase1_bundle_sha256=_sha("phase1-bundle"),
        phase1_aggregate_receipt_sha256=_sha("phase1-aggregates"),
        method_lock_sha256=_sha("method-lock"),
    )


def _binding(asset: fa.FARDCE3Phase1Asset, *, active_k: int = 1) -> fa.FARDCE3RuntimeBinding:
    return fa.FARDCE3RuntimeBinding(
        checkpoint_sha256=asset.checkpoint_sha256,
        capsule_id=_sha("capsule"),
        split_id=_sha("split"),
        row_id=f"row-k{active_k}",
        seed=9,
        active_k=active_k,
        old_classes=asset.old_classes,
        support_physical_root_sha256=_sha(f"support-root-k{active_k}"),
        support_authority_sha256=_sha(f"support-authority-k{active_k}"),
    )


def _support(asset: fa.FARDCE3Phase1Asset, *, active_k: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for index, class_handle in enumerate(asset.old_classes):
        result[class_handle] = np.repeat(_unit(index % fa.RANK), active_k, axis=0)
    return result


def test_aggregate_only_phase1_wire_is_immutable_and_roundtrips() -> None:
    old_classes = ("c0", "c1")
    asset = fa.build_fa_rdce3_phase1_asset(
        old_classes=old_classes,
        aggregate_samples_per_class=(3, 4),
        class_centers_3d=np.asarray([[0.2, -0.1, 0.0], [-0.3, 0.4, 0.1]], dtype=np.float32),
        fisher_precision_3d=np.asarray([0.5, 0.6, 0.7], dtype=np.float32),
        residual_variance_3d=np.asarray([0.8, 0.9, 1.0], dtype=np.float32),
        fisher_radius=np.asarray([1.25], dtype=np.float32),
        rdce_kappa_3d=np.asarray([0.2, 0.1, 0.0], dtype=np.float32),
        basis_3x160=np.pad(np.eye(fa.RANK, dtype=np.float32), ((0, 0), (0, fa.Z_DIM - fa.RANK))),
        checkpoint_sha256=_sha("checkpoint"),
        phase1_bundle_sha256=_sha("phase1-bundle"),
        phase1_aggregate_receipt_sha256=_sha("phase1-aggregates"),
        method_lock_sha256=_sha("method-lock"),
    )
    assert not asset.centers_codes_qint8.flags.writeable
    assert asset.wire_mapping["aggregate_only"] is True
    assert asset.wire_mapping["phase1_source_rows_retained"] is False
    assert asset.wire_mapping["phase1_per_row_features_retained"] is False
    assert asset.wire_mapping["phase1_loo_required"] is False
    assert asset.numeric_payload_bytes > 0
    wire = fa.serialize_fa_rdce3_phase1_asset(asset)
    recovered = fa.roundtrip_fa_rdce3_phase1_asset(asset)
    assert fa.serialize_fa_rdce3_phase1_asset(recovered) == wire
    assert recovered.asset_sha256 == asset.asset_sha256


def test_reg0_closed_form_is_radius_projected_and_reports_six_byte_state() -> None:
    asset = _asset(rho=0.25)
    state = fa.fit_fa_rdce3_reg0(asset, _support(asset, active_k=1), binding=_binding(asset))
    fisher = fa.decode_fa_rdce3_fisher_precision(asset)
    fisher_norm = float(np.sqrt(np.sum(fisher * np.square(state.a))))
    assert state.fit_mode == fa.FIT_MODE_FISHER_CLOSED_FORM
    assert fisher_norm == pytest.approx(fa.decode_fa_rdce3_radius(asset), abs=2.0e-4)
    assert state.dynamic_numeric_bytes == 6
    assert state.a_fp16.dtype == np.dtype("<f2")
    assert not state.a_fp16.flags.writeable
    receipt = fa.fa_rdce3_resource_receipt(state)
    assert receipt["dynamic_numeric_bytes"] == 6
    assert receipt["fit_mac_formula"] == "C*K*3*160"
    assert receipt["fit_mac"] == 2 * 1 * 3 * 160
    assert receipt["fixed_rdce_query_mac"] == 2 * 3 * 160
    assert receipt["query_rows_used_for_fit"] == 0
    assert receipt["query_truth_access"] is False


def test_k1_zero_posterior_keeps_fixed_rdce_and_reg1_reuses_exact_state() -> None:
    asset = _asset(centers=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int8))
    state = fa.fit_fa_rdce3_reg0(asset, _support(asset, active_k=1), binding=_binding(asset))
    assert state.fit_mode == fa.FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE
    assert np.array_equal(state.a_fp16, np.zeros(fa.RANK, dtype=np.float16))
    reused = fa.reuse_fa_rdce3_state_for_reg1(
        state, registered_classes=("new0", "c1", "c0")
    )
    assert reused is state
    reuse_receipt = fa.fa_rdce3_reg1_reuse_receipt(
        state, registered_classes=("new0", "c1", "c0")
    )
    assert reuse_receipt["bitwise_state_reuse"] is True
    assert reuse_receipt["reg1_fit_calls"] == 0
    assert reuse_receipt["new_class_support_rows_used_for_da"] == 0
    wire = fa.serialize_fa_rdce3_runtime_state(state)
    recovered = fa.roundtrip_fa_rdce3_runtime_state(state)
    assert fa.serialize_fa_rdce3_runtime_state(recovered) == wire
    assert recovered.a_fp16.tobytes() == state.a_fp16.tobytes()


def test_r1_remains_signed_unit_without_relu_or_second_normalisation() -> None:
    asset = _asset(
        centers=np.asarray([[-1, 0, 0], [-1, 0, 0]], dtype=np.int8),
        rho=2.0,
        kappa=(0.0, 0.0, 0.0),
    )
    support = {class_handle: _unit(0) for class_handle in asset.old_classes}
    state = fa.fit_fa_rdce3_reg0(asset, support, binding=_binding(asset))
    result = fa.transform_fa_rdce3_r1(state, _unit(0))
    assert state.a[0] > 1.0
    assert result.dtype == np.float32
    assert not result.flags.writeable
    assert result[0, 0] < 0.0
    assert float(np.linalg.norm(result[0])) == pytest.approx(1.0, abs=2.0e-6)


def test_class_permutation_and_common_coordinate_transform_are_equivariant() -> None:
    asset = _asset(kappa=(0.2, 0.1, 0.0))
    support = _support(asset, active_k=1)
    state = fa.fit_fa_rdce3_reg0(asset, support, binding=_binding(asset))
    query = np.concatenate((_unit(0), _unit(1)), axis=0)
    baseline = fa.transform_fa_rdce3_r1(state, query)

    permuted_asset = _asset(
        old_classes=("c1", "c0"),
        centers=np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int8),
        kappa=(0.2, 0.1, 0.0),
    )
    permuted_support = {"c1": support["c1"], "c0": support["c0"]}
    permuted_state = fa.fit_fa_rdce3_reg0(
        permuted_asset,
        permuted_support,
        binding=_binding(permuted_asset),
    )
    assert np.allclose(permuted_state.a, state.a, rtol=0.0, atol=1.0e-6)
    assert np.allclose(
        fa.transform_fa_rdce3_r1(permuted_state, query), baseline, rtol=0.0, atol=1.0e-6
    )

    coordinate_permutation = np.eye(fa.Z_DIM, dtype=np.float32)
    coordinate_permutation[[0, 1]] = coordinate_permutation[[1, 0]]
    rotated_basis = asset.basis_codes_qint8.astype(np.float32) @ coordinate_permutation
    rotated_asset = _asset(
        kappa=(0.2, 0.1, 0.0),
        basis_codes=rotated_basis.astype(np.int8),
    )
    rotated_support = {
        class_handle: (rows @ coordinate_permutation).astype(np.float32)
        for class_handle, rows in support.items()
    }
    rotated_state = fa.fit_fa_rdce3_reg0(
        rotated_asset,
        rotated_support,
        binding=_binding(rotated_asset),
    )
    rotated = fa.transform_fa_rdce3_r1(
        rotated_state, (query @ coordinate_permutation).astype(np.float32)
    )
    assert np.allclose(rotated, baseline @ coordinate_permutation, rtol=0.0, atol=2.0e-6)


def test_reg0_only_and_nonfinite_negative_paths_fail_closed() -> None:
    asset = _asset()
    binding = _binding(asset)
    support = _support(asset, active_k=1)
    support["c0"][0, 0] = np.float32(np.nan)
    with pytest.raises(fa.NextR4FARDCE3Error, match="finite"):
        fa.fit_fa_rdce3_reg0(asset, support, binding=binding)

    valid = fa.fit_fa_rdce3_reg0(asset, _support(asset, active_k=1), binding=binding)
    invalid_r0 = _unit(0)
    invalid_r0[0, 1] = np.float32(-0.1)
    with pytest.raises(fa.NextR4FARDCE3Error, match="non-negative"):
        fa.transform_fa_rdce3_r1(valid, invalid_r0)
    with pytest.raises(fa.NextR4FARDCE3Error, match="old-class"):
        fa.fit_fa_rdce3_reg0(
            asset,
            {**_support(asset, active_k=1), "new0": _unit(2)},
            binding=binding,
        )
    with pytest.raises(fa.NextR4FARDCE3Error, match="finite"):
        fa.build_fa_rdce3_phase1_asset(
            old_classes=("c0", "c1"),
            aggregate_samples_per_class=(2, 2),
            class_centers_3d=np.zeros((2, 3), dtype=np.float32),
            fisher_precision_3d=np.asarray([np.nan, 1.0, 1.0], dtype=np.float32),
            residual_variance_3d=np.ones(3, dtype=np.float32),
            fisher_radius=np.ones(1, dtype=np.float32),
            rdce_kappa_3d=np.zeros(3, dtype=np.float32),
            basis_3x160=np.pad(np.eye(3, dtype=np.float32), ((0, 0), (0, 157))),
            checkpoint_sha256=_sha("checkpoint"),
            phase1_bundle_sha256=_sha("phase1-bundle"),
            phase1_aggregate_receipt_sha256=_sha("phase1-aggregates"),
            method_lock_sha256=_sha("method-lock"),
        )
