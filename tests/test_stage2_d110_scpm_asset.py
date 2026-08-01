"""Traceability checks for the frozen D110 SCPM Phase1 aggregate.

The tests cover: D106 closed-basis reuse; all 168 TX/receiver/day cells with
equal weighting; the d-3 perpendicular normalization; the 4xINT8+4xFP16
state; no retained source rows or IDs; lineage binding; and fail-closed zeros
or non-finite values.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

import cvsrffi.stage2_d106_rdce_asset as d106
import cvsrffi.stage2_d110_scpm_asset as scpm
from cvsrffi.stage2_d106_phase1_tap import (
    D106Phase1TapRows,
    TAP_ARCHIVE_NAME,
    TAP_MEMBERS,
    TAP_RECEIPT_SCHEMA,
)


def _sha(character: str) -> str:
    return character * 64


def _d106_lock(*, method: str = "c") -> d106.D106RDCEBuildLock:
    return d106.D106RDCEBuildLock(
        method_lock_sha256=_sha(method),
        construction_code_sha256=_sha("e"),
    )


def _d110_lock() -> scpm.D110SCPMBuildLock:
    return scpm.D110SCPMBuildLock(
        method_lock_sha256=_sha("4"),
        construction_code_sha256=_sha("5"),
    )


def _make_tap(*, zero_first_cell: bool = False) -> D106Phase1TapRows:
    rng = np.random.default_rng(11020260802)
    rows: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    tx: list[str] = []
    receiver: list[str] = []
    day: list[str] = []
    for receiver_index in range(7):
        for day_index in range(4):
            for tx_index in range(6):
                count = 4 if (receiver_index + day_index + tx_index) % 2 == 0 else 3
                cell_anchor = (
                    0.15
                    + 0.016 * receiver_index
                    + 0.011 * day_index
                    + 0.021 * tx_index
                )
                base = rng.uniform(0.05, 0.22, size=d106.Z_DIM).astype(np.float32)
                base[tx_index] += np.float32(0.9)
                for sample_index in range(count):
                    if zero_first_cell and (receiver_index, day_index, tx_index) == (0, 0, 0):
                        row = base.copy()
                    else:
                        scale = np.float32(0.006 + 0.001 * (receiver_index + day_index))
                        row = base + np.float32(cell_anchor) + rng.normal(
                            0.0,
                            float(scale),
                            size=d106.Z_DIM,
                        ).astype(np.float32)
                    rows.append(np.maximum(row, np.float32(1.0e-4)))
                    domains.append(rng.normal(size=d106.Z_DIM).astype(np.float32))
                    tx.append(f"tx_{tx_index}")
                    receiver.append(f"rx_{receiver_index}")
                    day.append(f"day_{day_index}")
    pre_relu = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    arrays = {
        "pre_relu": pre_relu,
        "z_dom": np.ascontiguousarray(np.stack(domains), dtype=np.float32),
        "tx_labels": np.asarray(tx, dtype="<U16"),
        "receiver_ids": np.asarray(receiver, dtype="<U16"),
        "day_ids": np.asarray(day, dtype="<U16"),
        "physical_ids": np.asarray(
            [f"physical_{index:04d}" for index in range(len(pre_relu))],
            dtype="<U20",
        ),
        "scenario_names": np.full(len(pre_relu), "leo_clear_weak", dtype="<U20"),
        "observation_ids": np.asarray(
            [f"obs_{index:04d}" for index in range(len(pre_relu))],
            dtype="<U20",
        ),
    }
    receipt: dict[str, object] = {
        "schema": TAP_RECEIPT_SCHEMA,
        "candidate_id": d106.CANDIDATE_ID,
        "split_id": d106.D104_SPLIT_ID,
        "protocol_schema": "p2_min_v1",
        "selected_iq_archive_sha256": _sha("6"),
        "selected_iq_receipt_sha256": _sha("1"),
        "storage_validator_receipt_sha256": _sha("7"),
        "storage_validation_binding": {
            "schema": d106.LS_IQ_VALIDATOR_SCHEMA,
            "storage_validation_root_sha256": _sha("8"),
            "selected_content_root_sha256": _sha("9"),
            "all_8400x3_storage_semantics_verified": True,
        },
        "extraction_binding": {
            "schema": d106.LS_IQ_RECEIPT_SCHEMA,
            "row_count": len(pre_relu),
            "selection_salt_sha256": _sha("0"),
            "selected_content_root_sha256": _sha("9"),
            "input_ls_archive_sha256": _sha("a"),
            "execution_root_sha256": _sha("b"),
        },
        "input_ls_archive_sha256": _sha("a"),
        "checkpoint_sha256": _sha("a"),
        "runtime_sha256": _sha("b"),
        "tap_archive_name": TAP_ARCHIVE_NAME,
        "tap_archive_sha256": _sha("d"),
        "tap_archive_members": list(TAP_MEMBERS),
        "array_sha256": {
            name: d106._tap_array_sha256(value) for name, value in arrays.items()
        },
        "row_count": len(pre_relu),
        "physical_id_root_sha256": d106._tap_ordered_id_root(
            arrays["physical_ids"]
        ),
    }
    return D106Phase1TapRows(
        **arrays,
        z_id=np.maximum(arrays["pre_relu"], np.float32(0.0)).astype(
            np.float32,
            copy=False,
        ),
        receipt=receipt,
    )


def _math_d106_asset(tap: D106Phase1TapRows, *, method: str = "c") -> d106.D106RDCEAsset:
    result = d106._try_build_d106_rdce_asset_math(tap, build_lock=_d106_lock(method=method))
    assert isinstance(result, d106.D106RDCEAsset)
    return result


def _formal_d106_asset(
    tap: D106Phase1TapRows,
) -> tuple[d106.D106RDCEAsset, d106._D106RDCETapAuthority]:
    authority = d106._D106RDCETapAuthority(
        archive_sha256=_sha("d"),
        receipt_sha256=_sha("e"),
    )
    result = d106._try_build_d106_rdce_asset_from_loaded_tap(
        tap,
        build_lock=_d106_lock(),
        tap_authority=authority,
    )
    assert isinstance(result, d106.D106RDCEAsset)
    assert result.is_formal_deployable
    return result, authority


def _expected_equal_cell_variances(
    tap: D106Phase1TapRows,
    closed_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = tap.z_id.astype(np.float64)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    cells: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for index, row in enumerate(rows):
        key = (
            str(tap.tx_labels[index]),
            str(tap.receiver_ids[index]),
            str(tap.day_ids[index]),
        )
        cells.setdefault(key, []).append(row)
    per_cell: list[np.ndarray] = []
    numerators = np.zeros(scpm.SCPM_GROUP_COUNT, dtype=np.float64)
    denominators = np.zeros(scpm.SCPM_GROUP_COUNT, dtype=np.float64)
    for values in cells.values():
        ordered = np.stack(sorted(values, key=lambda value: value.tobytes()), axis=0)
        residual = ordered - np.mean(ordered, axis=0, dtype=np.float64)
        projected = residual @ closed_u.T
        directional_numerator = np.sum(np.square(projected), axis=0)
        perpendicular = residual - projected @ closed_u
        perp_numerator = np.sum(np.square(perpendicular))
        denominator = float(len(ordered) - 1)
        per_cell.append(
            np.concatenate(
                (
                    directional_numerator / denominator,
                    np.asarray([perp_numerator / (denominator * scpm.SCPM_PERP_DIM)]),
                )
            )
        )
        numerators[: scpm.SCPM_RANK] += directional_numerator
        numerators[-1] += perp_numerator
        denominators[: scpm.SCPM_RANK] += denominator
        denominators[-1] += denominator * scpm.SCPM_PERP_DIM
    equal = np.mean(
        np.stack(sorted(per_cell, key=lambda value: value.tobytes()), axis=0),
        axis=0,
        dtype=np.float64,
    )
    return equal, numerators / denominators


def test_scpm_asset_uses_closed_d106_basis_and_exact_equal_cell_formula() -> None:
    tap = _make_tap()
    d106_asset = _math_d106_asset(tap)
    asset = scpm._try_build_d110_scpm_asset_math(
        tap, d106_asset, build_lock=_d110_lock()
    )

    closed_u, decoded = scpm.decode_d110_scpm_inputs(asset, d106_asset)
    equal_cell, pooled = _expected_equal_cell_variances(tap, closed_u)

    assert asset.d106_lineage == d106_asset.lineage
    assert asset.d106_asset_binding_sha256 == d106_asset.binding_sha256
    assert closed_u.shape == (3, 160)
    assert scpm.SCPM_GROUP_NAMES == ("u1", "u2", "u3", "perp")
    assert np.allclose(closed_u @ closed_u.T, np.eye(3), rtol=0.0, atol=2.0e-10)
    assert decoded.shape == (4,)
    assert np.all(decoded > 0.0)
    assert np.allclose(
        decoded,
        equal_cell,
        rtol=asset.quantization_max_relative_error + 1.0e-9,
        atol=1.0e-12,
    )
    # Different 3/4-sample cell weights and intentionally heterogeneous scatter
    # make the required equal-cell estimator distinct from pooled variance.
    assert not np.allclose(equal_cell, pooled, rtol=1.0e-6, atol=1.0e-12)
    assert asset.quantization_max_relative_error < 2.0e-2


def test_scpm_asset_retains_only_the_twelve_byte_quantized_state() -> None:
    tap = _make_tap()
    asset = scpm._try_build_d110_scpm_asset_math(
        tap, _math_d106_asset(tap), build_lock=_d110_lock()
    )

    assert asset.variance_codes_qint8.dtype == np.int8
    assert asset.variance_codes_qint8.shape == (4,)
    assert asset.variance_scales_fp16.dtype == np.dtype("<f2")
    assert asset.variance_scales_fp16.shape == (4,)
    assert asset.variance_codes_qint8.nbytes + asset.variance_scales_fp16.nbytes == 12
    assert not asset.variance_codes_qint8.flags.writeable
    assert not asset.variance_scales_fp16.flags.writeable
    assert asset.source_rows_retained is False
    assert asset.source_ids_retained is False
    field_names = {item.name for item in fields(scpm.D110SCPMAsset)}
    assert "physical_ids" not in field_names
    assert "source_rows" not in field_names
    assert "z_id" not in field_names
    assert "tx_labels" not in field_names


def test_scpm_asset_rejects_d106_binding_drift_and_zero_or_nonfinite_values() -> None:
    tap = _make_tap()
    d106_asset = _math_d106_asset(tap)
    asset = scpm._try_build_d110_scpm_asset_math(
        tap, d106_asset, build_lock=_d110_lock()
    )
    other_d106_asset = _math_d106_asset(tap, method="f")

    with pytest.raises(scpm.D110SCPMAssetError, match="binding mismatch"):
        scpm.decode_d110_scpm_inputs(asset, other_d106_asset)

    zero_tap = _make_tap(zero_first_cell=True)
    zero_d106_asset = _math_d106_asset(zero_tap)
    zero_cell_asset = scpm._try_build_d110_scpm_asset_math(
        zero_tap,
        zero_d106_asset,
        build_lock=_d110_lock(),
    )
    assert np.all(scpm.decode_d110_scpm_prior_variances(zero_cell_asset) > 0.0)

    with pytest.raises(scpm.D110SCPMAssetError, match="aggregate variance"):
        scpm._cell_conditional_variances(
            np.ones((scpm.SCPM_SOURCE_ROW_COUNT, scpm.Z_DIM), dtype=np.float32),
            d106._typed_tokens(zero_tap.tx_labels, "tx", scpm.SCPM_SOURCE_ROW_COUNT),
            d106._typed_tokens(
                zero_tap.receiver_ids, "receiver", scpm.SCPM_SOURCE_ROW_COUNT
            ),
            d106._typed_tokens(zero_tap.day_ids, "day", scpm.SCPM_SOURCE_ROW_COUNT),
            d106.decode_d106_rdce_basis(zero_d106_asset),
        )

    with pytest.raises(scpm.D110SCPMAssetError, match="finite and positive"):
        scpm._quantize_positive_variances(
            np.asarray([1.0, np.nan, 2.0, 3.0], dtype=np.float64)
        )
    with pytest.raises(scpm.D110SCPMAssetError, match="scale range"):
        replace(
            asset,
            variance_scales_fp16=np.asarray([0.0, 1.0, 1.0, 1.0], dtype="<f2"),
        )
    with pytest.raises(scpm.D110SCPMAssetError, match="loader-origin"):
        replace(asset, deployment_status=scpm.FORMAL_DEPLOYMENT_STATUS)


def test_formal_scpm_builder_requires_loader_origin_and_matching_formal_d106(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tap = _make_tap()
    formal_d106, authority = _formal_d106_asset(tap)
    monkeypatch.setattr(
        d106,
        "_load_formal_d106_tap",
        lambda *args, **kwargs: (tap, authority),
    )

    asset = scpm.build_d110_scpm_asset(
        "sealed_tap.npz",
        "sealed_tap_receipt.json",
        expected_tap_archive_sha256=authority.archive_sha256,
        expected_tap_receipt_sha256=authority.receipt_sha256,
        d106_asset=formal_d106,
        build_lock=_d110_lock(),
    )

    assert asset.is_formal_deployable
    assert asset.checkpoint_sha256 == formal_d106.checkpoint_sha256
    basis, prior = scpm.decode_d110_scpm_inputs(asset, formal_d106)
    assert basis.shape == (3, 160)
    assert prior.shape == (4,)

    support_z = np.zeros((3, 160), dtype=np.float64)
    support_z[0, 0] = 1.0
    support_z[1, 1] = 1.0
    support_z[2, 2] = 1.0
    support_labels = np.asarray(["old_a", "old_b", "new_c"])
    state = scpm.fit_d110_scpm_from_assets(
        support_z,
        support_labels,
        asset=asset,
        d106_asset=formal_d106,
    )
    assert state.active_k == 1
    assert np.array_equal(state.prior_variances, prior)

    math_asset = scpm._try_build_d110_scpm_asset_math(
        tap,
        _math_d106_asset(tap),
        build_lock=_d110_lock(),
    )
    with pytest.raises(scpm.D110SCPMAssetError, match="deployable"):
        scpm.fit_d110_scpm_from_assets(
            support_z,
            support_labels,
            asset=math_asset,
            d106_asset=_math_d106_asset(tap),
        )
    with pytest.raises(scpm.D110SCPMAssetError, match="matching formal D106"):
        scpm.build_d110_scpm_asset(
            "sealed_tap.npz",
            "sealed_tap_receipt.json",
            expected_tap_archive_sha256=authority.archive_sha256,
            expected_tap_receipt_sha256=authority.receipt_sha256,
            d106_asset=_math_d106_asset(tap),
            build_lock=_d110_lock(),
        )
