"""D110 SCPM Phase1 conditional-variance asset.

This builder deliberately consumes the sealed D106 tap only while constructing
the aggregate.  Its result retains exactly four positive quantized variances,
the D106 asset binding, and build hashes.  It never retains source features,
labels, receiver/day names, or physical IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import stage2_d106_rdce_asset as _d106
from .stage2_d106_phase1_tap import D106Phase1TapRows


Z_DIM = _d106.Z_DIM
SCPM_RANK = _d106.RDCE_RANK
SCPM_GROUP_COUNT = SCPM_RANK + 1
SCPM_PERP_DIM = Z_DIM - SCPM_RANK
SCPM_GROUP_NAMES = ("u1", "u2", "u3", "perp")
SCPM_CELL_COUNT = (
    _d106.D104_SOURCE_CLASS_COUNT
    * _d106.D104_RECEIVER_COUNT
    * _d106.D104_DAY_COUNT
)
SCPM_SOURCE_ROW_COUNT = _d106.D104_SOURCE_ROW_COUNT
SCPM_CELL_MIN_SAMPLES = _d106.D104_CELL_MIN_SAMPLES
SCPM_CELL_MAX_SAMPLES = _d106.D104_CELL_MAX_SAMPLES
INT8_MAX = 127
EPSILON = 1.0e-12

SCHEMA = "cvs.phase1.d110.scpm_asset.v1"
BUILD_LOCK_SCHEMA = "cvs.phase1.d110.scpm_build_lock.v1"
CANDIDATE_ID = "D110-SCPM/r3-CONDITIONAL-VAR01"
FORMAL_DEPLOYMENT_STATUS = _d106.FORMAL_DEPLOYMENT_STATUS
NON_DEPLOYABLE_MATH_STATUS = _d106.NON_DEPLOYABLE_MATH_STATUS
_FORMAL_LOADER_TOKEN = object()


class D110SCPMAssetError(ValueError):
    """Raised when the SCPM aggregate or its D106 binding drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str:
        raise D110SCPMAssetError(f"{name} must be an exact string SHA256")
    if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise D110SCPMAssetError(f"{name} must be a lowercase SHA256")
    return value


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


@dataclass(frozen=True, slots=True)
class D110SCPMBuildLock:
    """Hashes for the D110 method lock and this exact construction code."""

    method_lock_sha256: str
    construction_code_sha256: str
    schema: str = BUILD_LOCK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BUILD_LOCK_SCHEMA:
            raise D110SCPMAssetError("D110 SCPM build-lock schema drift")
        object.__setattr__(
            self,
            "method_lock_sha256",
            _require_sha256(self.method_lock_sha256, "method_lock_sha256"),
        )
        object.__setattr__(
            self,
            "construction_code_sha256",
            _require_sha256(
                self.construction_code_sha256,
                "construction_code_sha256",
            ),
        )


def _asset_payload(
    *,
    d106_lineage: _d106.D106RDCEAssetLineage,
    d106_asset_binding_sha256: str,
    method_lock_sha256: str,
    construction_code_sha256: str,
    deployment_status: str,
    variance_codes_qint8: np.ndarray,
    variance_scales_fp16: np.ndarray,
    quantization_max_relative_error: float,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "d106_lineage": d106_lineage.as_dict(),
        "d106_asset_binding_sha256": d106_asset_binding_sha256,
        "method_lock_sha256": method_lock_sha256,
        "construction_code_sha256": construction_code_sha256,
        "deployment_status": deployment_status,
        "source_row_count": SCPM_SOURCE_ROW_COUNT,
        "tx_receiver_day_cell_count": SCPM_CELL_COUNT,
        "rank": SCPM_RANK,
        "perp_dimensions": SCPM_PERP_DIM,
        "group_order": list(SCPM_GROUP_NAMES),
        "cell_equal_weighted": True,
        "source_rows_retained": False,
        "source_ids_retained": False,
        "source_names_retained": False,
        "variance_codes_qint8": _array_receipt(variance_codes_qint8),
        "variance_scales_fp16": _array_receipt(variance_scales_fp16),
        "quantization_max_relative_error": quantization_max_relative_error,
    }


def _decode_positive_variances(
    codes: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    values = codes.astype(np.float64) * scales.astype(np.float64)
    if (
        values.shape != (SCPM_GROUP_COUNT,)
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
    ):
        raise D110SCPMAssetError("SCPM decoded variances must be finite and positive")
    return np.ascontiguousarray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class D110SCPMAsset:
    """Immutable four-scalar SCPM prior, jointly pinned to the D106 asset."""

    d106_lineage: _d106.D106RDCEAssetLineage
    d106_asset_binding_sha256: str
    method_lock_sha256: str
    construction_code_sha256: str
    variance_codes_qint8: np.ndarray
    variance_scales_fp16: np.ndarray
    quantization_max_relative_error: float
    asset_receipt_sha256: str
    deployment_status: str = NON_DEPLOYABLE_MATH_STATUS
    _tap_authority: _d106._D106RDCETapAuthority | None = None
    _authority_token: object | None = None
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise D110SCPMAssetError("D110 SCPM asset schema drift")
        if type(self.d106_lineage) is not _d106.D106RDCEAssetLineage:
            raise D110SCPMAssetError("SCPM requires an exact D106 lineage")
        if self.deployment_status not in {
            FORMAL_DEPLOYMENT_STATUS,
            NON_DEPLOYABLE_MATH_STATUS,
        }:
            raise D110SCPMAssetError("SCPM deployment status drift")
        formal = self.deployment_status == FORMAL_DEPLOYMENT_STATUS
        if formal:
            if (
                type(self._tap_authority) is not _d106._D106RDCETapAuthority
                or self._authority_token is not _FORMAL_LOADER_TOKEN
                or not self.d106_lineage.has_external_tap_authority
                or self.d106_lineage.tap_sha256 != self._tap_authority.archive_sha256
                or self.d106_lineage.tap_receipt_sha256
                != self._tap_authority.receipt_sha256
                or self.d106_lineage.tap_authority_sha256
                != self._tap_authority.authority_sha256
            ):
                raise D110SCPMAssetError(
                    "formal SCPM asset requires loader-origin D106 tap authority"
                )
        elif self._tap_authority is not None or self._authority_token is not None:
            raise D110SCPMAssetError(
                "non-deployable SCPM math asset may not carry formal authority"
            )
        d106_binding = _require_sha256(
            self.d106_asset_binding_sha256,
            "d106_asset_binding_sha256",
        )
        method_lock = _require_sha256(self.method_lock_sha256, "method_lock_sha256")
        construction = _require_sha256(
            self.construction_code_sha256,
            "construction_code_sha256",
        )
        receipt = _require_sha256(self.asset_receipt_sha256, "asset_receipt_sha256")
        error = float(self.quantization_max_relative_error)
        if not math.isfinite(error) or error < 0.0:
            raise D110SCPMAssetError("SCPM quantization error must be finite and nonnegative")

        arrays = {
            "variance_codes_qint8": (
                self.variance_codes_qint8,
                np.dtype(np.int8),
            ),
            "variance_scales_fp16": (
                self.variance_scales_fp16,
                np.dtype("<f2"),
            ),
        }
        normalized: dict[str, np.ndarray] = {}
        for name, (value, dtype) in arrays.items():
            array = np.asarray(value)
            if (
                array.dtype != dtype
                or array.shape != (SCPM_GROUP_COUNT,)
                or not np.isfinite(array).all()
            ):
                raise D110SCPMAssetError(f"{name} dtype/shape/finite drift")
            if name == "variance_codes_qint8" and (
                np.any(array <= 0) or np.any(array == np.int8(-128))
            ):
                raise D110SCPMAssetError("SCPM variance code range drift")
            if name == "variance_scales_fp16" and np.any(array <= 0.0):
                raise D110SCPMAssetError("SCPM variance scale range drift")
            normalized[name] = np.ascontiguousarray(array)
        _decode_positive_variances(
            normalized["variance_codes_qint8"],
            normalized["variance_scales_fp16"],
        )
        payload = _asset_payload(
            d106_lineage=self.d106_lineage,
            d106_asset_binding_sha256=d106_binding,
            method_lock_sha256=method_lock,
            construction_code_sha256=construction,
            deployment_status=self.deployment_status,
            quantization_max_relative_error=error,
            **normalized,
        )
        if _sha256_bytes(_canonical_bytes(payload)) != receipt:
            raise D110SCPMAssetError("SCPM asset receipt drift")
        object.__setattr__(self, "d106_asset_binding_sha256", d106_binding)
        object.__setattr__(self, "method_lock_sha256", method_lock)
        object.__setattr__(self, "construction_code_sha256", construction)
        object.__setattr__(self, "quantization_max_relative_error", error)
        object.__setattr__(self, "asset_receipt_sha256", receipt)
        for name, value in normalized.items():
            object.__setattr__(self, name, _readonly(value, value.dtype))

    @property
    def binding_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_bytes(
                {
                    "candidate_id": CANDIDATE_ID,
                    "d106_lineage": self.d106_lineage.as_dict(),
                    "d106_asset_binding_sha256": self.d106_asset_binding_sha256,
                    "method_lock_sha256": self.method_lock_sha256,
                    "construction_code_sha256": self.construction_code_sha256,
                    "asset_receipt_sha256": self.asset_receipt_sha256,
                }
            )
        )

    @property
    def checkpoint_sha256(self) -> str:
        """The exact D106 checkpoint identity jointly binding this prior."""

        return self.d106_lineage.checkpoint_sha256

    @property
    def is_formal_deployable(self) -> bool:
        return self.deployment_status == FORMAL_DEPLOYMENT_STATUS

    @property
    def source_rows_retained(self) -> bool:
        return False

    @property
    def source_ids_retained(self) -> bool:
        return False


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (SCPM_SOURCE_ROW_COUNT, Z_DIM) or not np.isfinite(values).all():
        raise D110SCPMAssetError("SCPM tap rows must be finite [588,160]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= EPSILON):
        raise D110SCPMAssetError("SCPM tap contains a zero-norm row")
    return np.ascontiguousarray(values / norms, dtype=np.float64)


def _canonical_rows(rows: np.ndarray) -> np.ndarray:
    order = sorted(range(len(rows)), key=lambda index: rows[index].tobytes(order="C"))
    return np.ascontiguousarray(rows[np.asarray(order, dtype=np.int64)], dtype=np.float64)


def _verify_d106_binding(
    tap: D106Phase1TapRows,
    d106_asset: _d106.D106RDCEAsset,
    *,
    tap_authority: _d106._D106RDCETapAuthority | None,
) -> tuple[np.ndarray, tuple[bytes, ...], tuple[bytes, ...], tuple[bytes, ...]]:
    """Reuse D106's exact tap receipt boundary before consuming any source rows."""

    if type(tap) is not D106Phase1TapRows:
        raise D110SCPMAssetError("SCPM requires exact D106Phase1TapRows")
    if type(d106_asset) is not _d106.D106RDCEAsset:
        raise D110SCPMAssetError("SCPM requires an exact D106 asset")
    if tap_authority is None:
        if d106_asset.is_formal_deployable:
            raise D110SCPMAssetError(
                "bare-tap SCPM math construction requires a non-deployable D106 asset"
            )
    elif (
        type(tap_authority) is not _d106._D106RDCETapAuthority
        or not d106_asset.is_formal_deployable
        or d106_asset._tap_authority != tap_authority
        or d106_asset._authority_token is not _d106._TAP_LOADER_TOKEN
    ):
        raise D110SCPMAssetError(
            "formal SCPM construction requires the matching formal D106 asset"
        )
    d106_lock = _d106.D106RDCEBuildLock(
        method_lock_sha256=d106_asset.method_lock_sha256,
        construction_code_sha256=d106_asset.construction_code_sha256,
    )
    try:
        z_id, tx, receiver, day, lineage = _d106._verified_tap(
            tap,
            d106_lock,
            tap_authority=tap_authority,
        )
    except _d106.D106RDCEAssetError as error:
        raise D110SCPMAssetError("SCPM rejected the D106 tap binding") from error
    if lineage != d106_asset.lineage:
        raise D110SCPMAssetError("SCPM D106 lineage/checkpoint binding mismatch")
    return z_id, tx, receiver, day


def _cell_conditional_variances(
    z_id: np.ndarray,
    tx: tuple[bytes, ...],
    receiver: tuple[bytes, ...],
    day: tuple[bytes, ...],
    closed_u: np.ndarray,
) -> np.ndarray:
    """Return the equally weighted 168-cell variance vector before quantization."""

    if (
        closed_u.shape != (SCPM_RANK, Z_DIM)
        or not np.isfinite(closed_u).all()
        or not np.allclose(
            closed_u @ closed_u.T,
            np.eye(SCPM_RANK),
            rtol=0.0,
            atol=2.0e-10,
        )
    ):
        raise D110SCPMAssetError("SCPM requires a closed row-orthonormal D106 basis")
    rows = _normalize_rows(z_id)
    cells: dict[tuple[bytes, bytes, bytes], list[np.ndarray]] = {}
    for index, row in enumerate(rows):
        cells.setdefault((tx[index], receiver[index], day[index]), []).append(row)
    tx_set, receiver_set, day_set = frozenset(tx), frozenset(receiver), frozenset(day)
    if (
        len(tx_set) != _d106.D104_SOURCE_CLASS_COUNT
        or len(receiver_set) != _d106.D104_RECEIVER_COUNT
        or len(day_set) != _d106.D104_DAY_COUNT
    ):
        raise D110SCPMAssetError("SCPM TX/receiver/day cardinality drift")
    expected = {
        (tx_token, receiver_token, day_token)
        for tx_token in tx_set
        for receiver_token in receiver_set
        for day_token in day_set
    }
    if set(cells) != expected or len(cells) != SCPM_CELL_COUNT:
        raise D110SCPMAssetError("SCPM requires the complete 6x7x4 cell grid")
    cell_values: list[np.ndarray] = []
    for values in cells.values():
        if not SCPM_CELL_MIN_SAMPLES <= len(values) <= SCPM_CELL_MAX_SAMPLES:
            raise D110SCPMAssetError("SCPM cell sample count must be two through four")
        ordered = _canonical_rows(np.stack(values, axis=0))
        residual = ordered - np.mean(ordered, axis=0, dtype=np.float64)
        projected = residual @ closed_u.T
        directional = np.sum(np.square(projected), axis=0) / float(len(ordered) - 1)
        perpendicular = residual - projected @ closed_u
        perp = np.sum(np.square(perpendicular)) / float(
            (len(ordered) - 1) * SCPM_PERP_DIM
        )
        values_by_group = np.concatenate((directional, np.asarray([perp])))
        if not np.isfinite(values_by_group).all() or np.any(values_by_group < 0.0):
            raise D110SCPMAssetError(
                "SCPM conditional cell variance is negative or non-finite"
            )
        cell_values.append(values_by_group)
    # Sorting by the computed values makes the equal-weighted aggregate invariant
    # to source row order and to arbitrary names assigned to TX/receiver/day cells.
    ordered_cells = np.stack(sorted(cell_values, key=lambda value: value.tobytes()), axis=0)
    variances = np.mean(ordered_cells, axis=0, dtype=np.float64)
    if not np.isfinite(variances).all() or np.any(variances <= 0.0):
        raise D110SCPMAssetError("SCPM aggregate variance is zero or non-finite")
    return np.ascontiguousarray(variances, dtype=np.float64)


def _quantize_positive_variances(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    raw = np.asarray(values, dtype=np.float64)
    if (
        raw.shape != (SCPM_GROUP_COUNT,)
        or not np.isfinite(raw).all()
        or np.any(raw <= 0.0)
    ):
        raise D110SCPMAssetError("SCPM variances must be finite and positive")
    # The design freezes an independent FP16 scale for every scalar.  Use the
    # largest positive code whose scale remains representable; this preserves
    # the normal 127-code path without allowing a small valid variance to
    # underflow its FP16 scale to zero.
    minimum_scale = float(np.nextafter(np.float16(0.0), np.float16(1.0)))
    codes = np.empty(SCPM_GROUP_COUNT, dtype=np.int8)
    scales = np.empty(SCPM_GROUP_COUNT, dtype=np.float16)
    for index, value in enumerate(raw):
        maximum_code = min(INT8_MAX, max(1, int(math.floor(value / minimum_scale))))
        scale = np.float16(value / float(maximum_code))
        if not math.isfinite(float(scale)) or scale <= 0.0:
            raise D110SCPMAssetError("SCPM variance scale underflow or non-finite")
        codes[index] = np.int8(maximum_code)
        scales[index] = scale
    decoded = _decode_positive_variances(codes, scales)
    error = np.abs(decoded - raw) / raw
    if not np.isfinite(error).all():
        raise D110SCPMAssetError("SCPM quantization error is non-finite")
    return codes, scales, float(np.max(error))


def _build_d110_scpm_asset_core(
    tap: D106Phase1TapRows,
    d106_asset: _d106.D106RDCEAsset,
    *,
    build_lock: D110SCPMBuildLock,
    tap_authority: _d106._D106RDCETapAuthority | None,
    formal_loader_token: object | None,
) -> D110SCPMAsset:
    """Shared core for loader-origin formal and bare-tap math construction."""

    if type(build_lock) is not D110SCPMBuildLock:
        raise D110SCPMAssetError("SCPM requires an exact typed build lock")
    if tap_authority is not None and formal_loader_token is not _FORMAL_LOADER_TOKEN:
        raise D110SCPMAssetError("formal SCPM core requires the loader-only token")
    if tap_authority is None and formal_loader_token is not None:
        raise D110SCPMAssetError("SCPM math core may not carry a formal token")
    z_id, tx, receiver, day = _verify_d106_binding(
        tap,
        d106_asset,
        tap_authority=tap_authority,
    )
    closed_u = _d106.decode_d106_rdce_basis(d106_asset)
    variances = _cell_conditional_variances(z_id, tx, receiver, day, closed_u)
    codes, scales, error = _quantize_positive_variances(variances)
    deployment_status = (
        FORMAL_DEPLOYMENT_STATUS
        if tap_authority is not None
        else NON_DEPLOYABLE_MATH_STATUS
    )
    payload = _asset_payload(
        d106_lineage=d106_asset.lineage,
        d106_asset_binding_sha256=d106_asset.binding_sha256,
        method_lock_sha256=build_lock.method_lock_sha256,
        construction_code_sha256=build_lock.construction_code_sha256,
        deployment_status=deployment_status,
        variance_codes_qint8=codes,
        variance_scales_fp16=scales,
        quantization_max_relative_error=error,
    )
    return D110SCPMAsset(
        d106_lineage=d106_asset.lineage,
        d106_asset_binding_sha256=d106_asset.binding_sha256,
        method_lock_sha256=build_lock.method_lock_sha256,
        construction_code_sha256=build_lock.construction_code_sha256,
        variance_codes_qint8=codes,
        variance_scales_fp16=scales,
        quantization_max_relative_error=error,
        asset_receipt_sha256=_sha256_bytes(_canonical_bytes(payload)),
        deployment_status=deployment_status,
        _tap_authority=tap_authority,
        _authority_token=formal_loader_token,
    )


def _try_build_d110_scpm_asset_math(
    tap: D106Phase1TapRows,
    d106_asset: _d106.D106RDCEAsset,
    *,
    build_lock: D110SCPMBuildLock,
) -> D110SCPMAsset:
    """Private pure-math path; bare taps can only produce NON_DEPLOYABLE state."""

    return _build_d110_scpm_asset_core(
        tap,
        d106_asset,
        build_lock=build_lock,
        tap_authority=None,
        formal_loader_token=None,
    )


def build_d110_scpm_asset(
    tap_archive_path: str | Path,
    tap_receipt_path: str | Path,
    *,
    expected_tap_archive_sha256: str,
    expected_tap_receipt_sha256: str,
    d106_asset: _d106.D106RDCEAsset,
    build_lock: D110SCPMBuildLock,
) -> D110SCPMAsset:
    """Formal loader-only builder; bare D106Phase1TapRows are never accepted."""

    try:
        tap, authority = _d106._load_formal_d106_tap(
            tap_archive_path,
            tap_receipt_path,
            expected_tap_archive_sha256=expected_tap_archive_sha256,
            expected_tap_receipt_sha256=expected_tap_receipt_sha256,
        )
    except _d106.D106RDCEAssetError as error:
        raise D110SCPMAssetError("SCPM formal tap loader rejected the source") from error
    return _build_d110_scpm_asset_core(
        tap,
        d106_asset,
        build_lock=build_lock,
        tap_authority=authority,
        formal_loader_token=_FORMAL_LOADER_TOKEN,
    )


def decode_d110_scpm_prior_variances(asset: D110SCPMAsset) -> np.ndarray:
    """Decode the positive [u1,u2,u3,perp] prior variances for runtime use."""

    if type(asset) is not D110SCPMAsset:
        raise D110SCPMAssetError("SCPM variance decode requires an exact asset")
    values = _decode_positive_variances(
        asset.variance_codes_qint8,
        asset.variance_scales_fp16,
    )
    values.setflags(write=False)
    return values


def decode_d110_scpm_inputs(
    asset: D110SCPMAsset,
    d106_asset: _d106.D106RDCEAsset,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the matching closed D106 basis and the four SCPM variances."""

    if type(asset) is not D110SCPMAsset or type(d106_asset) is not _d106.D106RDCEAsset:
        raise D110SCPMAssetError("SCPM decode requires exact D110 and D106 assets")
    if (
        asset.d106_lineage != d106_asset.lineage
        or asset.d106_asset_binding_sha256 != d106_asset.binding_sha256
    ):
        raise D110SCPMAssetError("SCPM runtime D106 asset binding mismatch")
    expected_status = (
        FORMAL_DEPLOYMENT_STATUS
        if d106_asset.is_formal_deployable
        else NON_DEPLOYABLE_MATH_STATUS
    )
    if asset.deployment_status != expected_status:
        raise D110SCPMAssetError("SCPM deployment status is not inherited from D106")
    basis = _d106.decode_d106_rdce_basis(d106_asset)
    basis.setflags(write=False)
    return basis, decode_d110_scpm_prior_variances(asset)


def fit_d110_scpm_from_assets(
    support_z: np.ndarray,
    support_labels: np.ndarray,
    *,
    asset: D110SCPMAsset,
    d106_asset: _d106.D106RDCEAsset,
) -> Any:
    """Formal fit entry binding the runtime arrays to both sealed assets.

    Capsule/split and physical-ID authority remain the responsibility of the
    typed Target5 materializer.  This boundary prevents the deployable runtime
    from accepting a caller-selected basis or prior variance vector.
    """

    if (
        type(asset) is not D110SCPMAsset
        or not asset.is_formal_deployable
        or type(d106_asset) is not _d106.D106RDCEAsset
        or not d106_asset.is_formal_deployable
    ):
        raise D110SCPMAssetError(
            "formal SCPM fit requires matching deployable D110 and D106 assets"
        )
    basis, prior = decode_d110_scpm_inputs(asset, d106_asset)
    from .stage2_d110_scpm_runtime import fit_d110_scpm_runtime

    return fit_d110_scpm_runtime(support_z, support_labels, basis, prior)


__all__ = [
    "BUILD_LOCK_SCHEMA",
    "CANDIDATE_ID",
    "D110SCPMAsset",
    "D110SCPMAssetError",
    "D110SCPMBuildLock",
    "EPSILON",
    "SCPM_CELL_COUNT",
    "SCPM_GROUP_COUNT",
    "SCPM_GROUP_NAMES",
    "SCPM_PERP_DIM",
    "SCPM_RANK",
    "SCPM_SOURCE_ROW_COUNT",
    "SCHEMA",
    "Z_DIM",
    "build_d110_scpm_asset",
    "decode_d110_scpm_inputs",
    "decode_d110_scpm_prior_variances",
    "fit_d110_scpm_from_assets",
]
