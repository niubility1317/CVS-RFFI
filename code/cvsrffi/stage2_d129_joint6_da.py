"""Pure-NumPy D129 joint6 domain-adaptation core.

Traceability (kept in the owned implementation surface):

* J6-DA-01: C1=CSPAR-2 uses only an INT8/FP16 Phase1 seal and current
  all-class K-shot support.  K1 is the sealed ``alpha0`` boundary; K5 uses
  the frozen class-balanced scatter equation.
* J6-DA-02: C2=SRDH-2 uses only an INT8/FP16 response dictionary and a
  permutation-invariant all-class support summary.  It is nonlinear and has
  no encoder parameter replacement or Phase2 gradient path.
* J6-DA-03: query application is a stateless map.  State/receipt counters
  explicitly prove zero query fit, update, selection, truth, role, quota,
  source, clean, optimizer, and backward access.
* J6-DA-04: the Phase1 receiver-held x class-LOCO audit plan is exactly
  seven receivers x six classes, with deterministic K1-as-K5-prefix support
  and physical-ID roots only in its coverage receipt.
* J6-DA-05: a deterministic Phase1-only builder constructs and quantizes
  CSPAR ``B`` and SRDH ``P/Q/m/d/a_max`` from receiver x class aggregates;
  it has no target, outer-score, query, truth, or parameter-search input.

The module intentionally does not import Torch, checkpoints, D127/D128, a
dataset loader, a scorer, or a head.  The caller supplies sealed aggregate
arrays and already received, frozen ``z_id160`` rows.  No raw/source sample,
query label, role, quota, or batch statistic has an API surface here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


Z_DIM = 160
RANK = 2
ALLOWED_K = (1, 5)

CSPAR2_CANDIDATE_ID = "CSPAR-2"
SRDH2_CANDIDATE_ID = "SRDH-2"

ASSET_SCHEMA = "cvs.phase1.d129.joint6.da_asset.v1"
STATE_SCHEMA = "cvs.phase2.d129.joint6.da_state.v1"
RESOURCE_SCHEMA = "cvs.phase2.d129.joint6.da_resource.v1"
LOCO_PLAN_SCHEMA = "cvs.phase1.d129.joint6.receiver_class_loco.v1"
PHASE1_BUILD_SCHEMA = "cvs.phase1.d129.joint6.aggregate_builder.v1"
LOCO_SALT = "d129-joint6-loco-v1"
ASSET_WIRE_MAGIC = b"CVSD129J6DA\x00\x01"

_F16 = np.dtype("<f2")
_I8 = np.dtype("i1")
_SQRT_EPS = 1.0e-12


class D129Joint6DAError(ValueError):
    """Raised when a frozen D129 joint6 invariant is violated."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise D129Joint6DAError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise D129Joint6DAError(f"{name} must be a lowercase SHA256") from error
    if value.lower() != value:
        raise D129Joint6DAError(f"{name} must be a lowercase SHA256")
    return value


def _readonly(value: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _require_exact_array(
    value: object, name: str, dtype: np.dtype, shape: Tuple[int, ...]
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        raise D129Joint6DAError(
            f"{name} must be an exact {dtype.str} NumPy array with shape {list(shape)}"
        )
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise D129Joint6DAError(f"{name} must contain finite values")
    if not value.flags.c_contiguous:
        raise D129Joint6DAError(f"{name} must be C-contiguous")
    return value


def _require_real_array(value: object, name: str, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != ndim or value.dtype.kind != "f":
        raise D129Joint6DAError(f"{name} must be a finite {ndim}D floating NumPy array")
    if not np.isfinite(value).all():
        raise D129Joint6DAError(f"{name} must contain finite values")
    return np.ascontiguousarray(value, dtype=np.float64)


def _array_receipt(value: np.ndarray) -> Dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _require_int8_codes(value: object, name: str, shape: Tuple[int, ...]) -> np.ndarray:
    codes = _require_exact_array(value, name, _I8, shape)
    if np.any(codes == np.int8(-128)):
        raise D129Joint6DAError(f"{name} must use symmetric INT8 range [-127,127]")
    if not np.any(codes):
        raise D129Joint6DAError(f"{name} may not be all zero")
    return codes


def _require_positive_f16(value: object, name: str, shape: Tuple[int, ...]) -> np.ndarray:
    result = _require_exact_array(value, name, _F16, shape)
    if np.any(result <= np.float16(0.0)):
        raise D129Joint6DAError(f"{name} must be finite positive FP16")
    return result


def _decode_columns(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        codes.astype(np.float64) * scales.astype(np.float64)[None, :], dtype=np.float64
    )


def _orthonormalize_rank2(value: np.ndarray, name: str) -> np.ndarray:
    """Return the deterministic polar factor of one full-rank 160x2 seal.

    INT8/FP16 quantization preserves the nuisance subspace but not exact unit
    column norms.  CSPAR equations require an orthonormal basis, so every
    builder/fit/transform path consumes the same closest polar factor.
    """

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (Z_DIM, RANK) or not np.isfinite(matrix).all():
        raise D129Joint6DAError(f"{name} must be a finite [160,2] matrix")
    gram = 0.5 * (matrix.T @ matrix + (matrix.T @ matrix).T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= _SQRT_EPS):
        raise D129Joint6DAError(f"{name} lost rank before polar normalization")
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    result = np.ascontiguousarray(matrix @ inverse_sqrt, dtype=np.float64)
    if not np.allclose(
        result.T @ result, np.eye(RANK, dtype=np.float64), rtol=0.0, atol=1.0e-12
    ):
        raise D129Joint6DAError(f"{name} polar normalization drift")
    return result


def _normalise_rows(value: object, name: str) -> Tuple[np.ndarray, bool]:
    if not isinstance(value, np.ndarray) or value.dtype.kind != "f" or value.ndim not in (1, 2):
        raise D129Joint6DAError(f"{name} must be a finite [160] or [N,160] NumPy array")
    was_vector = value.ndim == 1
    rows = value.reshape(1, -1) if was_vector else value
    if rows.shape[0] < 1 or rows.shape[1] != Z_DIM or not np.isfinite(rows).all():
        raise D129Joint6DAError(f"{name} must be finite with shape [N,{Z_DIM}]")
    result = np.ascontiguousarray(rows, dtype=np.float64)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise D129Joint6DAError(f"{name} contains a zero-norm row")
    return np.ascontiguousarray(result / norms, dtype=np.float64), was_vector


def _canonicalise_support(value: object) -> np.ndarray:
    """Normalize and canonicalize class/support ordering without using labels.

    The state formulas are exchangeable over registered classes.  Sorting by
    content digest makes that property bitwise reproducible under an arbitrary
    class-label permutation and under per-class support row reordering.
    """

    if not isinstance(value, np.ndarray) or value.dtype.kind != "f" or value.ndim != 3:
        raise D129Joint6DAError(
            f"support_z must be a finite floating [C,K,{Z_DIM}] NumPy array"
        )
    classes, k_shot, dimension = value.shape
    if classes < 2 or k_shot not in ALLOWED_K or dimension != Z_DIM:
        raise D129Joint6DAError(
            f"support_z must have C>=2, K in {ALLOWED_K}, and dimension {Z_DIM}"
        )
    if not np.isfinite(value).all():
        raise D129Joint6DAError("support_z must contain only finite values")
    rows = np.ascontiguousarray(value, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=2, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise D129Joint6DAError("support_z contains a zero-norm row")
    normalized = np.ascontiguousarray(rows / norms, dtype=np.float64)

    class_entries: List[Tuple[str, np.ndarray]] = []
    for class_index in range(classes):
        class_rows = normalized[class_index]
        row_order = sorted(
            range(k_shot),
            key=lambda index: _sha256_bytes(
                np.ascontiguousarray(class_rows[index]).tobytes(order="C")
            ),
        )
        ordered_rows = np.ascontiguousarray(class_rows[row_order], dtype=np.float64)
        row_digests = b"".join(
            _sha256_bytes(np.ascontiguousarray(row).tobytes(order="C")).encode("ascii")
            for row in ordered_rows
        )
        class_entries.append((_sha256_bytes(row_digests), ordered_rows))
    class_entries.sort(key=lambda item: item[0])
    return np.ascontiguousarray(
        np.stack([rows_for_class for _digest, rows_for_class in class_entries], axis=0),
        dtype=np.float64,
    )


def _support_root(support: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(support).tobytes(order="C"))


def _normalise_output(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise D129Joint6DAError("adapted representation contains a zero-norm row")
    return np.ascontiguousarray(rows / norms, dtype=np.float64)


@dataclass(frozen=True)
class CSPAR2Asset:
    """C1 Phase1 seal: a rank-two nuisance basis and frozen scalar policy."""

    checkpoint_sha256: str
    phase1_seal_sha256: str
    basis_qint8: np.ndarray
    basis_scale_fp16: np.ndarray
    alpha0_fp16: np.ndarray
    alpha_max_fp16: np.ndarray
    eps_fp16: np.ndarray
    schema: str = ASSET_SCHEMA
    candidate_id: str = CSPAR2_CANDIDATE_ID

    def __post_init__(self) -> None:
        if self.schema != ASSET_SCHEMA or self.candidate_id != CSPAR2_CANDIDATE_ID:
            raise D129Joint6DAError("CSPAR-2 asset schema/candidate drift")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_sha256(self.phase1_seal_sha256, "phase1_seal_sha256")
        basis = _require_int8_codes(self.basis_qint8, "basis_qint8", (Z_DIM, RANK))
        scale = _require_positive_f16(
            self.basis_scale_fp16, "basis_scale_fp16", (RANK,)
        )
        alpha0 = _require_exact_array(self.alpha0_fp16, "alpha0_fp16", _F16, (RANK,))
        alpha_max = _require_positive_f16(
            self.alpha_max_fp16, "alpha_max_fp16", (1,)
        )
        eps = _require_positive_f16(self.eps_fp16, "eps_fp16", (1,))
        maximum = float(alpha_max[0])
        if maximum > 0.5 + 1.0e-6:
            raise D129Joint6DAError("CSPAR-2 alpha_max must not exceed the frozen 0.50")
        if np.any(alpha0 < np.float16(0.0)) or np.any(alpha0 > alpha_max[0]):
            raise D129Joint6DAError("CSPAR-2 alpha0 must lie in [0,alpha_max]")
        if not np.any(alpha0 > np.float16(0.0)):
            raise D129Joint6DAError("CSPAR-2 K1 alpha0 may not be the identity")
        decoded = _decode_columns(basis, scale)
        gram = decoded.T @ decoded
        if np.linalg.matrix_rank(decoded) != RANK or not np.allclose(
            gram, np.eye(RANK, dtype=np.float64), rtol=0.0, atol=5.0e-2
        ):
            raise D129Joint6DAError(
                "CSPAR-2 decoded INT8 basis must remain a rank-two near-orthonormal seal"
            )
        object.__setattr__(self, "basis_qint8", _readonly(basis, _I8))
        object.__setattr__(self, "basis_scale_fp16", _readonly(scale, _F16))
        object.__setattr__(self, "alpha0_fp16", _readonly(alpha0, _F16))
        object.__setattr__(self, "alpha_max_fp16", _readonly(alpha_max, _F16))
        object.__setattr__(self, "eps_fp16", _readonly(eps, _F16))

    @property
    def numeric_payload_bytes(self) -> int:
        return int(
            self.basis_qint8.nbytes
            + self.basis_scale_fp16.nbytes
            + self.alpha0_fp16.nbytes
            + self.alpha_max_fp16.nbytes
            + self.eps_fp16.nbytes
        )


@dataclass(frozen=True)
class SRDH2Asset:
    """C2 Phase1 seal: a nonlinear rank-two response dictionary."""

    checkpoint_sha256: str
    phase1_seal_sha256: str
    p_qint8: np.ndarray
    p_scale_fp16: np.ndarray
    q_qint8: np.ndarray
    q_scale_fp16: np.ndarray
    mean_fp16: np.ndarray
    std_fp16: np.ndarray
    a_max_fp16: np.ndarray
    schema: str = ASSET_SCHEMA
    candidate_id: str = SRDH2_CANDIDATE_ID

    def __post_init__(self) -> None:
        if self.schema != ASSET_SCHEMA or self.candidate_id != SRDH2_CANDIDATE_ID:
            raise D129Joint6DAError("SRDH-2 asset schema/candidate drift")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_sha256(self.phase1_seal_sha256, "phase1_seal_sha256")
        p = _require_int8_codes(self.p_qint8, "p_qint8", (Z_DIM, RANK))
        p_scale = _require_positive_f16(self.p_scale_fp16, "p_scale_fp16", (RANK,))
        q = _require_int8_codes(self.q_qint8, "q_qint8", (Z_DIM, RANK))
        q_scale = _require_positive_f16(self.q_scale_fp16, "q_scale_fp16", (RANK,))
        mean = _require_exact_array(self.mean_fp16, "mean_fp16", _F16, (RANK,))
        std = _require_positive_f16(self.std_fp16, "std_fp16", (RANK,))
        a_max = _require_positive_f16(self.a_max_fp16, "a_max_fp16", (1,))
        if float(a_max[0]) > 1.0:
            raise D129Joint6DAError("SRDH-2 sealed a_max must be in (0,1]")
        if np.linalg.matrix_rank(_decode_columns(p, p_scale)) != RANK:
            raise D129Joint6DAError("SRDH-2 decoded P must retain rank two")
        if np.linalg.matrix_rank(_decode_columns(q, q_scale)) != RANK:
            raise D129Joint6DAError("SRDH-2 decoded Q must retain rank two")
        object.__setattr__(self, "p_qint8", _readonly(p, _I8))
        object.__setattr__(self, "p_scale_fp16", _readonly(p_scale, _F16))
        object.__setattr__(self, "q_qint8", _readonly(q, _I8))
        object.__setattr__(self, "q_scale_fp16", _readonly(q_scale, _F16))
        object.__setattr__(self, "mean_fp16", _readonly(mean, _F16))
        object.__setattr__(self, "std_fp16", _readonly(std, _F16))
        object.__setattr__(self, "a_max_fp16", _readonly(a_max, _F16))

    @property
    def numeric_payload_bytes(self) -> int:
        return int(
            self.p_qint8.nbytes
            + self.p_scale_fp16.nbytes
            + self.q_qint8.nbytes
            + self.q_scale_fp16.nbytes
            + self.mean_fp16.nbytes
            + self.std_fp16.nbytes
            + self.a_max_fp16.nbytes
        )


D129Joint6Asset = Union[CSPAR2Asset, SRDH2Asset]


def _normalise_phase1_receiver_class_rows(value: object) -> np.ndarray:
    """Validate the only permitted builder input: Phase1 receiver/class rows."""

    if (
        not isinstance(value, np.ndarray)
        or value.dtype.kind != "f"
        or value.ndim != 4
        or value.shape[0] < 2
        or value.shape[1] < 2
        or value.shape[2] < 2
        or value.shape[3] != Z_DIM
        or not np.isfinite(value).all()
    ):
        raise D129Joint6DAError(
            "Phase1 builder requires finite [receiver>=2,class>=2,row>=2,160] aggregates"
        )
    rows = np.ascontiguousarray(value, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=3, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise D129Joint6DAError("Phase1 aggregate contains a zero-norm feature row")
    return np.ascontiguousarray(rows / norms, dtype=np.float64)


def d129_phase1_aggregate_sha256(phase1_receiver_class_z: np.ndarray) -> str:
    """Return the deterministic normalized aggregate root used by the builder.

    This is an audit helper only.  The caller still supplies the separately
    jointly-sealed ``phase1_seal_sha256`` pin when constructing an asset.
    """

    aggregate = _normalise_phase1_receiver_class_rows(phase1_receiver_class_z)
    return _sha256_bytes(aggregate.tobytes(order="C"))


def _canonicalise_axis_signs(axes: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(axes, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _top_rank2_axes(rows: np.ndarray, name: str) -> np.ndarray:
    """Deterministically extract a rank-two Phase1-only covariance basis."""

    if rows.ndim != 2 or rows.shape[1] != Z_DIM or rows.shape[0] < RANK:
        raise D129Joint6DAError(f"{name} Phase1 aggregate has no rank-two surface")
    centered = rows - np.mean(rows, axis=0, keepdims=True, dtype=np.float64)
    covariance = (centered.T @ centered) / float(rows.shape[0])
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if (
        not np.isfinite(eigenvalues).all()
        or float(eigenvalues[-RANK]) <= 1.0e-12
    ):
        raise D129Joint6DAError(f"{name} Phase1 aggregate is rank-deficient")
    axes = eigenvectors[:, -RANK:][:, ::-1]
    axes = _canonicalise_axis_signs(axes)
    if np.linalg.matrix_rank(axes) != RANK or not np.allclose(
        axes.T @ axes, np.eye(RANK), atol=1.0e-10, rtol=0.0
    ):
        raise D129Joint6DAError(f"{name} Phase1 rank-two axes are invalid")
    return np.ascontiguousarray(axes, dtype=np.float64)


def _quantize_rank2_columns(matrix: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetrically quantize rank-two columns into the only deployable wire."""

    if matrix.shape != (Z_DIM, RANK) or not np.isfinite(matrix).all():
        raise D129Joint6DAError(f"{name} must be a finite [160,2] matrix")
    peak = np.max(np.abs(matrix), axis=0)
    if np.any(peak <= 0.0):
        raise D129Joint6DAError(f"{name} has a zero rank-two column")
    scale = np.asarray(peak / 127.0, dtype=_F16)
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise D129Joint6DAError(f"{name} FP16 quantization scale drift")
    codes = np.clip(
        np.rint(matrix / scale.astype(np.float64)[None, :]), -127, 127
    ).astype(_I8)
    if np.linalg.matrix_rank(_decode_columns(codes, scale)) != RANK:
        raise D129Joint6DAError(f"{name} INT8 quantization lost rank two")
    return np.ascontiguousarray(codes), np.ascontiguousarray(scale)


def build_d129_phase1_assets(
    phase1_receiver_class_z: np.ndarray,
    *,
    checkpoint_sha256: str,
    phase1_seal_sha256: str,
) -> Tuple[CSPAR2Asset, SRDH2Asset]:
    """Build both frozen assets from allowed Phase1 receiver/class aggregates.

    The entry deliberately has no target, query, truth, score, outer-fold, or
    hyperparameter-selection input.  It uses only normalized Phase1 feature
    rows arranged as ``[receiver,class,physical_row,z160]`` and fixed formulas:
    receiver effects provide CSPAR-2's nuisance axes and SRDH-2's response
    probes, while cell means provide SRDH-2's residual dictionary.  Returned
    assets retain only INT8/FP16 arrays plus the supplied checkpoint/Phase1
    seal pins.
    """

    checkpoint_sha256 = _require_sha256(checkpoint_sha256, "checkpoint_sha256")
    phase1_seal_sha256 = _require_sha256(phase1_seal_sha256, "phase1_seal_sha256")
    rows = _normalise_phase1_receiver_class_rows(phase1_receiver_class_z)
    receiver_count, class_count, _rows_per_cell, _dimension = rows.shape
    cell_mean = np.mean(rows, axis=2, dtype=np.float64)

    # Remove each class's source-wide centre before estimating common receiver
    # drift.  The resulting axes contain no class-ID-specific state.
    receiver_effect = cell_mean - np.mean(cell_mean, axis=0, keepdims=True, dtype=np.float64)
    receiver_axes = _top_rank2_axes(
        receiver_effect.reshape(receiver_count * class_count, Z_DIM), "receiver-effect"
    )
    b_codes, b_scale = _quantize_rank2_columns(receiver_axes, "CSPAR-2 B")
    decoded_b = _orthonormalize_rank2(
        _decode_columns(b_codes, b_scale), "CSPAR-2 quantized B"
    )
    within = rows - cell_mean[:, :, None, :]
    trace = float(np.mean(np.sum(np.square(within), axis=3), dtype=np.float64))
    axial = np.mean(
        np.square(within.reshape(-1, Z_DIM) @ decoded_b), axis=0, dtype=np.float64
    )
    v_perp = max((trace - float(np.sum(axial))) / float(Z_DIM - RANK), 0.0)
    eps_value = max(1.0e-3, 0.02 * max(trace, 1.0e-3))
    raw_alpha0 = 1.0 - (v_perp + eps_value) / (axial + eps_value)
    # A non-zero Phase1 K1 policy is frozen before any held receiver exists;
    # the fixed lower boundary is not tuned from target/outer performance.
    alpha0 = np.clip(np.maximum(raw_alpha0, 0.05), 0.05, 0.50)
    cspar_asset = CSPAR2Asset(
        checkpoint_sha256=checkpoint_sha256,
        phase1_seal_sha256=phase1_seal_sha256,
        basis_qint8=b_codes,
        basis_scale_fp16=b_scale,
        alpha0_fp16=np.ascontiguousarray(np.asarray(alpha0, dtype=_F16)),
        alpha_max_fp16=np.asarray([0.50], dtype=_F16),
        eps_fp16=np.asarray([eps_value], dtype=_F16),
    )

    global_cell_centre = np.mean(cell_mean, axis=(0, 1), dtype=np.float64)
    dictionary_rows = cell_mean.reshape(-1, Z_DIM) - global_cell_centre[None, :]
    p_axes = _top_rank2_axes(dictionary_rows, "cell-mean")
    p_codes, p_scale = _quantize_rank2_columns(p_axes, "SRDH-2 P")
    q_codes, q_scale = _quantize_rank2_columns(receiver_axes, "SRDH-2 Q")
    decoded_q = _decode_columns(q_codes, q_scale)
    nonlinear_response = np.tanh(rows.reshape(-1, Z_DIM) @ decoded_q)
    mean = np.mean(nonlinear_response, axis=0, dtype=np.float64)
    std = np.maximum(np.std(nonlinear_response, axis=0, dtype=np.float64), 1.0e-3)
    a_max_value = float(
        np.clip(0.05 + 0.45 * np.mean(np.abs(nonlinear_response)), 0.05, 0.50)
    )
    srdh_asset = SRDH2Asset(
        checkpoint_sha256=checkpoint_sha256,
        phase1_seal_sha256=phase1_seal_sha256,
        p_qint8=p_codes,
        p_scale_fp16=p_scale,
        q_qint8=q_codes,
        q_scale_fp16=q_scale,
        mean_fp16=np.ascontiguousarray(np.asarray(mean, dtype=_F16)),
        std_fp16=np.ascontiguousarray(np.asarray(std, dtype=_F16)),
        a_max_fp16=np.asarray([a_max_value], dtype=_F16),
    )
    return cspar_asset, srdh_asset


def decode_cspar2_basis(asset: CSPAR2Asset) -> np.ndarray:
    if not isinstance(asset, CSPAR2Asset):
        raise D129Joint6DAError("CSPAR-2 basis requires an exact CSPAR2Asset")
    return _readonly(
        _orthonormalize_rank2(
            _decode_columns(asset.basis_qint8, asset.basis_scale_fp16),
            "CSPAR-2 decoded B",
        ),
        np.float64,
    )


def decode_srdh2_dictionary(asset: SRDH2Asset) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(asset, SRDH2Asset):
        raise D129Joint6DAError("SRDH-2 dictionary requires an exact SRDH2Asset")
    return (
        _readonly(_decode_columns(asset.p_qint8, asset.p_scale_fp16), np.float64),
        _readonly(_decode_columns(asset.q_qint8, asset.q_scale_fp16), np.float64),
    )


def _asset_arrays(asset: D129Joint6Asset) -> Tuple[Tuple[str, np.ndarray], ...]:
    if isinstance(asset, CSPAR2Asset):
        return (
            ("basis_qint8", asset.basis_qint8),
            ("basis_scale_fp16", asset.basis_scale_fp16),
            ("alpha0_fp16", asset.alpha0_fp16),
            ("alpha_max_fp16", asset.alpha_max_fp16),
            ("eps_fp16", asset.eps_fp16),
        )
    if isinstance(asset, SRDH2Asset):
        return (
            ("p_qint8", asset.p_qint8),
            ("p_scale_fp16", asset.p_scale_fp16),
            ("q_qint8", asset.q_qint8),
            ("q_scale_fp16", asset.q_scale_fp16),
            ("mean_fp16", asset.mean_fp16),
            ("std_fp16", asset.std_fp16),
            ("a_max_fp16", asset.a_max_fp16),
        )
    raise D129Joint6DAError("D129 joint6 serialization requires a recognized asset")


def serialize_d129_joint6_asset(asset: D129Joint6Asset) -> bytes:
    """Serialize only the sealed INT8/FP16 numeric payload and provenance pins."""

    arrays = _asset_arrays(asset)
    header = {
        "schema": ASSET_SCHEMA,
        "candidate_id": asset.candidate_id,
        "checkpoint_sha256": asset.checkpoint_sha256,
        "phase1_seal_sha256": asset.phase1_seal_sha256,
        "arrays": [
            {"name": name, **_array_receipt(array)} for name, array in arrays
        ],
    }
    encoded = _canonical_bytes(header)
    body = b"".join(np.ascontiguousarray(array).tobytes(order="C") for _name, array in arrays)
    return ASSET_WIRE_MAGIC + struct.pack(">I", len(encoded)) + encoded + body


def d129_joint6_asset_sha256(asset: D129Joint6Asset) -> str:
    return _sha256_bytes(serialize_d129_joint6_asset(asset))


def _wire_array_specs(candidate_id: str) -> Tuple[Tuple[str, np.dtype, Tuple[int, ...]], ...]:
    if candidate_id == CSPAR2_CANDIDATE_ID:
        return (
            ("basis_qint8", _I8, (Z_DIM, RANK)),
            ("basis_scale_fp16", _F16, (RANK,)),
            ("alpha0_fp16", _F16, (RANK,)),
            ("alpha_max_fp16", _F16, (1,)),
            ("eps_fp16", _F16, (1,)),
        )
    if candidate_id == SRDH2_CANDIDATE_ID:
        return (
            ("p_qint8", _I8, (Z_DIM, RANK)),
            ("p_scale_fp16", _F16, (RANK,)),
            ("q_qint8", _I8, (Z_DIM, RANK)),
            ("q_scale_fp16", _F16, (RANK,)),
            ("mean_fp16", _F16, (RANK,)),
            ("std_fp16", _F16, (RANK,)),
            ("a_max_fp16", _F16, (1,)),
        )
    raise D129Joint6DAError("D129 joint6 asset candidate is not recognized")


def deserialize_d129_joint6_asset(
    payload: bytes,
    *,
    expected_sha256: Optional[str] = None,
    expected_checkpoint_sha256: Optional[str] = None,
    expected_phase1_seal_sha256: Optional[str] = None,
) -> D129Joint6Asset:
    """Parse a pinned D129 seal without accepting hidden FP32 sidecars."""

    if not isinstance(payload, bytes) or not payload.startswith(ASSET_WIRE_MAGIC):
        raise D129Joint6DAError("D129 joint6 asset wire magic drift")
    if expected_sha256 is not None and _sha256_bytes(payload) != _require_sha256(
        expected_sha256, "expected_sha256"
    ):
        raise D129Joint6DAError("D129 joint6 asset wire SHA256 mismatch")
    offset = len(ASSET_WIRE_MAGIC)
    if len(payload) < offset + 4:
        raise D129Joint6DAError("D129 joint6 asset wire is truncated")
    header_size = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4
    if header_size <= 0 or offset + header_size > len(payload):
        raise D129Joint6DAError("D129 joint6 asset header length drift")
    raw_header = payload[offset : offset + header_size]
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D129Joint6DAError("D129 joint6 asset header must be canonical JSON") from error
    if not isinstance(header, dict) or raw_header != _canonical_bytes(header):
        raise D129Joint6DAError("D129 joint6 asset header is not canonical")
    required_header = {
        "schema",
        "candidate_id",
        "checkpoint_sha256",
        "phase1_seal_sha256",
        "arrays",
    }
    if set(header) != required_header or header["schema"] != ASSET_SCHEMA:
        raise D129Joint6DAError("D129 joint6 asset header schema drift")
    candidate_id = header["candidate_id"]
    if candidate_id not in {CSPAR2_CANDIDATE_ID, SRDH2_CANDIDATE_ID}:
        raise D129Joint6DAError("D129 joint6 asset candidate drift")
    checkpoint = _require_sha256(header["checkpoint_sha256"], "checkpoint_sha256")
    phase1_seal = _require_sha256(header["phase1_seal_sha256"], "phase1_seal_sha256")
    if expected_checkpoint_sha256 is not None and checkpoint != _require_sha256(
        expected_checkpoint_sha256, "expected_checkpoint_sha256"
    ):
        raise D129Joint6DAError("D129 joint6 checkpoint pin mismatch")
    if expected_phase1_seal_sha256 is not None and phase1_seal != _require_sha256(
        expected_phase1_seal_sha256, "expected_phase1_seal_sha256"
    ):
        raise D129Joint6DAError("D129 joint6 Phase1 seal pin mismatch")
    specs = _wire_array_specs(candidate_id)
    entries = header["arrays"]
    if not isinstance(entries, list) or len(entries) != len(specs):
        raise D129Joint6DAError("D129 joint6 asset array receipt count drift")
    offset += header_size
    arrays: Dict[str, np.ndarray] = {}
    for entry, (name, dtype, shape) in zip(entries, specs):
        if not isinstance(entry, dict) or set(entry) != {"name", "dtype", "shape", "sha256"}:
            raise D129Joint6DAError("D129 joint6 asset array receipt schema drift")
        if entry["name"] != name or entry["dtype"] != dtype.str or entry["shape"] != list(shape):
            raise D129Joint6DAError("D129 joint6 asset array type/shape drift")
        count = int(np.prod(shape, dtype=np.int64))
        size = count * dtype.itemsize
        if offset + size > len(payload):
            raise D129Joint6DAError("D129 joint6 asset array payload is truncated")
        array = np.frombuffer(payload[offset : offset + size], dtype=dtype).copy().reshape(shape)
        offset += size
        if _array_receipt(array) != {key: entry[key] for key in ("dtype", "shape", "sha256")}:
            raise D129Joint6DAError("D129 joint6 asset array receipt mismatch")
        arrays[name] = np.ascontiguousarray(array)
    if offset != len(payload):
        raise D129Joint6DAError("D129 joint6 asset wire has trailing bytes")
    common = {
        "checkpoint_sha256": checkpoint,
        "phase1_seal_sha256": phase1_seal,
    }
    if candidate_id == CSPAR2_CANDIDATE_ID:
        return CSPAR2Asset(**common, **arrays)
    return SRDH2Asset(**common, **arrays)


@dataclass(frozen=True)
class D129ResourceReceipt:
    """Auditable resource and protocol counters for one immutable DA state."""

    candidate_id: str
    active_k: int
    registered_class_count: int
    asset_numeric_payload_bytes: int
    dynamic_numeric_bytes: int
    support_fit_mac_equivalent: int
    query_transform_mac_equivalent: int
    additional_backbone_forwards: int = 0
    query_rows_used_for_fit: int = 0
    query_state_updates: int = 0
    query_selection_count: int = 0
    query_gradient_calls: int = 0
    phase2_backward_calls: int = 0
    phase2_optimizer_steps: int = 0
    source_rows_used_at_phase2: int = 0
    clean_rows_used_at_phase2: int = 0
    truth_role_quota_inputs: int = 0
    global_reassignment_calls: int = 0
    class_specific_parameter_count: int = 0
    schema: str = RESOURCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESOURCE_SCHEMA or self.candidate_id not in {
            CSPAR2_CANDIDATE_ID,
            SRDH2_CANDIDATE_ID,
        }:
            raise D129Joint6DAError("D129 resource receipt schema/candidate drift")
        if self.active_k not in ALLOWED_K or self.registered_class_count < 2:
            raise D129Joint6DAError("D129 resource receipt K/class count drift")
        numeric_values = (
            self.asset_numeric_payload_bytes,
            self.dynamic_numeric_bytes,
            self.support_fit_mac_equivalent,
            self.query_transform_mac_equivalent,
            self.additional_backbone_forwards,
            self.query_rows_used_for_fit,
            self.query_state_updates,
            self.query_selection_count,
            self.query_gradient_calls,
            self.phase2_backward_calls,
            self.phase2_optimizer_steps,
            self.source_rows_used_at_phase2,
            self.clean_rows_used_at_phase2,
            self.truth_role_quota_inputs,
            self.global_reassignment_calls,
            self.class_specific_parameter_count,
        )
        if any(type(item) is not int or item < 0 for item in numeric_values):
            raise D129Joint6DAError("D129 resource receipt counters must be nonnegative ints")

    @property
    def protocol_closed(self) -> bool:
        return (
            self.additional_backbone_forwards == 0
            and self.query_rows_used_for_fit == 0
            and self.query_state_updates == 0
            and self.query_selection_count == 0
            and self.query_gradient_calls == 0
            and self.phase2_backward_calls == 0
            and self.phase2_optimizer_steps == 0
            and self.source_rows_used_at_phase2 == 0
            and self.clean_rows_used_at_phase2 == 0
            and self.truth_role_quota_inputs == 0
            and self.global_reassignment_calls == 0
            and self.class_specific_parameter_count == 0
        )

    def as_dict(self) -> Dict[str, Any]:
        values = {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "active_k": self.active_k,
            "registered_class_count": self.registered_class_count,
            "asset_numeric_payload_bytes": self.asset_numeric_payload_bytes,
            "dynamic_numeric_bytes": self.dynamic_numeric_bytes,
            "support_fit_mac_equivalent": self.support_fit_mac_equivalent,
            "query_transform_mac_equivalent": self.query_transform_mac_equivalent,
            "additional_backbone_forwards": self.additional_backbone_forwards,
            "query_rows_used_for_fit": self.query_rows_used_for_fit,
            "query_state_updates": self.query_state_updates,
            "query_selection_count": self.query_selection_count,
            "query_gradient_calls": self.query_gradient_calls,
            "phase2_backward_calls": self.phase2_backward_calls,
            "phase2_optimizer_steps": self.phase2_optimizer_steps,
            "source_rows_used_at_phase2": self.source_rows_used_at_phase2,
            "clean_rows_used_at_phase2": self.clean_rows_used_at_phase2,
            "truth_role_quota_inputs": self.truth_role_quota_inputs,
            "global_reassignment_calls": self.global_reassignment_calls,
            "class_specific_parameter_count": self.class_specific_parameter_count,
            "protocol_closed": self.protocol_closed,
        }
        values["resource_receipt_sha256"] = _sha256_bytes(_canonical_bytes(values))
        return values


@dataclass(frozen=True)
class CSPAR2State:
    """C1 support-only runtime state; it retains no support rows or labels."""

    asset_sha256: str
    active_k: int
    registered_class_count: int
    alpha_fp16: np.ndarray
    support_root_sha256: str
    receipt: D129ResourceReceipt
    schema: str = STATE_SCHEMA
    candidate_id: str = CSPAR2_CANDIDATE_ID

    def __post_init__(self) -> None:
        if self.schema != STATE_SCHEMA or self.candidate_id != CSPAR2_CANDIDATE_ID:
            raise D129Joint6DAError("CSPAR-2 state schema/candidate drift")
        _require_sha256(self.asset_sha256, "asset_sha256")
        _require_sha256(self.support_root_sha256, "support_root_sha256")
        alpha = _require_exact_array(self.alpha_fp16, "alpha_fp16", _F16, (RANK,))
        if self.active_k not in ALLOWED_K or self.registered_class_count < 2:
            raise D129Joint6DAError("CSPAR-2 state K/class count drift")
        if not isinstance(self.receipt, D129ResourceReceipt) or (
            self.receipt.candidate_id != CSPAR2_CANDIDATE_ID
            or self.receipt.active_k != self.active_k
            or self.receipt.registered_class_count != self.registered_class_count
        ):
            raise D129Joint6DAError("CSPAR-2 state resource receipt drift")
        object.__setattr__(self, "alpha_fp16", _readonly(alpha, _F16))

    @property
    def is_nonidentity(self) -> bool:
        return bool(np.any(self.alpha_fp16 > np.float16(0.0)))


@dataclass(frozen=True)
class SRDH2State:
    """C2 support-only runtime state; it retains no support rows or labels."""

    asset_sha256: str
    active_k: int
    registered_class_count: int
    response_fp16: np.ndarray
    summary_fp16: np.ndarray
    support_root_sha256: str
    receipt: D129ResourceReceipt
    schema: str = STATE_SCHEMA
    candidate_id: str = SRDH2_CANDIDATE_ID

    def __post_init__(self) -> None:
        if self.schema != STATE_SCHEMA or self.candidate_id != SRDH2_CANDIDATE_ID:
            raise D129Joint6DAError("SRDH-2 state schema/candidate drift")
        _require_sha256(self.asset_sha256, "asset_sha256")
        _require_sha256(self.support_root_sha256, "support_root_sha256")
        response = _require_exact_array(self.response_fp16, "response_fp16", _F16, (RANK,))
        summary = _require_exact_array(self.summary_fp16, "summary_fp16", _F16, (RANK,))
        if self.active_k not in ALLOWED_K or self.registered_class_count < 2:
            raise D129Joint6DAError("SRDH-2 state K/class count drift")
        if not isinstance(self.receipt, D129ResourceReceipt) or (
            self.receipt.candidate_id != SRDH2_CANDIDATE_ID
            or self.receipt.active_k != self.active_k
            or self.receipt.registered_class_count != self.registered_class_count
        ):
            raise D129Joint6DAError("SRDH-2 state resource receipt drift")
        object.__setattr__(self, "response_fp16", _readonly(response, _F16))
        object.__setattr__(self, "summary_fp16", _readonly(summary, _F16))

    @property
    def is_nonidentity(self) -> bool:
        return bool(np.any(self.response_fp16 != np.float16(0.0)))


def _checked_cspar_pair(asset: object, state: object) -> Tuple[CSPAR2Asset, CSPAR2State]:
    if not isinstance(asset, CSPAR2Asset) or not isinstance(state, CSPAR2State):
        raise D129Joint6DAError("CSPAR-2 transform requires exact asset/state types")
    if d129_joint6_asset_sha256(asset) != state.asset_sha256:
        raise D129Joint6DAError("CSPAR-2 asset/state binding mismatch")
    alpha = state.alpha_fp16.astype(np.float64)
    if np.any(alpha < 0.0) or np.any(alpha > asset.alpha_max_fp16.astype(np.float64)):
        raise D129Joint6DAError("CSPAR-2 state attenuation exceeds its sealed bound")
    return asset, state


def _checked_srdh_pair(asset: object, state: object) -> Tuple[SRDH2Asset, SRDH2State]:
    if not isinstance(asset, SRDH2Asset) or not isinstance(state, SRDH2State):
        raise D129Joint6DAError("SRDH-2 transform requires exact asset/state types")
    if d129_joint6_asset_sha256(asset) != state.asset_sha256:
        raise D129Joint6DAError("SRDH-2 asset/state binding mismatch")
    response = state.response_fp16.astype(np.float64)
    if np.any(np.abs(response) > float(asset.a_max_fp16[0]) + 1.0e-12):
        raise D129Joint6DAError("SRDH-2 state response exceeds its sealed bound")
    return asset, state


def fit_cspar2_support(asset: CSPAR2Asset, support_z: np.ndarray) -> CSPAR2State:
    """Fit C1 from all registered classes of K=1 or K=5 support only."""

    if not isinstance(asset, CSPAR2Asset):
        raise D129Joint6DAError("CSPAR-2 fit requires an exact CSPAR2Asset")
    support = _canonicalise_support(support_z)
    class_count, k_shot, _dimension = support.shape
    if k_shot == 1:
        alpha = asset.alpha0_fp16.astype(np.float64)
        support_fit_mac = 0
    else:
        basis = decode_cspar2_basis(asset)
        residual = support - np.mean(support, axis=1, keepdims=True, dtype=np.float64)
        denominator = float(class_count * (k_shot - 1))
        trace = float(np.sum(np.square(residual), dtype=np.float64) / denominator)
        projected = residual @ basis
        v = np.sum(np.square(projected), axis=(0, 1), dtype=np.float64) / denominator
        v_perp_raw = (trace - float(np.sum(v))) / float(Z_DIM - RANK)
        if not math.isfinite(v_perp_raw) or v_perp_raw < -1.0e-10:
            raise D129Joint6DAError("CSPAR-2 support scatter left the orthogonal complement")
        v_perp = max(v_perp_raw, 0.0)
        eps = float(asset.eps_fp16[0])
        raw_alpha = 1.0 - (v_perp + eps) / (v + eps)
        if not np.isfinite(raw_alpha).all():
            raise D129Joint6DAError("CSPAR-2 scatter attenuation became non-finite")
        alpha = np.clip(raw_alpha, 0.0, float(asset.alpha_max_fp16[0]))
        support_fit_mac = int(class_count * k_shot * 2 * Z_DIM)
    alpha_fp16 = np.minimum(
        np.asarray(alpha, dtype=np.float16), asset.alpha_max_fp16[0]
    ).astype(_F16, copy=False)
    receipt = D129ResourceReceipt(
        candidate_id=CSPAR2_CANDIDATE_ID,
        active_k=k_shot,
        registered_class_count=class_count,
        asset_numeric_payload_bytes=asset.numeric_payload_bytes,
        dynamic_numeric_bytes=alpha_fp16.nbytes,
        support_fit_mac_equivalent=support_fit_mac,
        query_transform_mac_equivalent=4 * Z_DIM,
    )
    return CSPAR2State(
        asset_sha256=d129_joint6_asset_sha256(asset),
        active_k=k_shot,
        registered_class_count=class_count,
        alpha_fp16=np.ascontiguousarray(alpha_fp16),
        support_root_sha256=_support_root(support),
        receipt=receipt,
    )


def fit_srdh2_support(asset: SRDH2Asset, support_z: np.ndarray) -> SRDH2State:
    """Fit C2's all-class shared response with no optimizer or gradient path."""

    if not isinstance(asset, SRDH2Asset):
        raise D129Joint6DAError("SRDH-2 fit requires an exact SRDH2Asset")
    support = _canonicalise_support(support_z)
    class_count, k_shot, _dimension = support.shape
    _p, q = decode_srdh2_dictionary(asset)
    class_response = np.mean(np.tanh(support @ q), axis=1, dtype=np.float64)
    summary = np.mean(class_response, axis=0, dtype=np.float64)
    standardized = (summary - asset.mean_fp16.astype(np.float64)) / asset.std_fp16.astype(
        np.float64
    )
    response = float(asset.a_max_fp16[0]) * np.tanh(standardized)
    if not np.isfinite(response).all():
        raise D129Joint6DAError("SRDH-2 support response became non-finite")
    # Float16 rounding may otherwise move a mathematically bounded response a
    # single representable value beyond the sealed FP16 bound.  Clamp in the
    # same FP16 domain that is persisted and consumed by the query map.
    response_fp16 = np.clip(
        np.asarray(response, dtype=np.float16),
        -asset.a_max_fp16[0],
        asset.a_max_fp16[0],
    ).astype(_F16, copy=False)
    summary_fp16 = np.asarray(summary, dtype=np.float16)
    receipt = D129ResourceReceipt(
        candidate_id=SRDH2_CANDIDATE_ID,
        active_k=k_shot,
        registered_class_count=class_count,
        asset_numeric_payload_bytes=asset.numeric_payload_bytes,
        dynamic_numeric_bytes=int(response_fp16.nbytes + summary_fp16.nbytes),
        support_fit_mac_equivalent=int(class_count * k_shot * 2 * Z_DIM),
        query_transform_mac_equivalent=4 * Z_DIM,
    )
    return SRDH2State(
        asset_sha256=d129_joint6_asset_sha256(asset),
        active_k=k_shot,
        registered_class_count=class_count,
        response_fp16=np.ascontiguousarray(response_fp16),
        summary_fp16=np.ascontiguousarray(summary_fp16),
        support_root_sha256=_support_root(support),
        receipt=receipt,
    )


def _cspar_sqrt_apply(rows: np.ndarray, basis: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Apply ``(I-B diag(alpha) B.T)^(1/2)`` with only rank-two operations."""

    if np.all(alpha == 0.0):
        return np.array(rows, dtype=np.float64, copy=True, order="C")
    loading = basis * np.sqrt(alpha)[None, :]
    gram = 0.5 * (loading.T @ loading + (loading.T @ loading).T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues >= 1.0 - _SQRT_EPS):
        raise D129Joint6DAError("CSPAR-2 low-rank metric is not strictly SPD")
    coefficients = np.empty(RANK, dtype=np.float64)
    for index, value in enumerate(eigenvalues):
        coefficients[index] = -0.5 if value <= _SQRT_EPS else (
            math.sqrt(1.0 - float(value)) - 1.0
        ) / float(value)
    correction = eigenvectors @ np.diag(coefficients) @ eigenvectors.T
    return np.ascontiguousarray(rows + ((rows @ loading) @ correction) @ loading.T, dtype=np.float64)


def transform_cspar2(asset: CSPAR2Asset, state: CSPAR2State, z_id: np.ndarray) -> np.ndarray:
    """Apply C1 independently to each query/support row without state mutation."""

    asset, state = _checked_cspar_pair(asset, state)
    rows, was_vector = _normalise_rows(z_id, "z_id")
    transformed = _normalise_output(
        _cspar_sqrt_apply(
            rows, decode_cspar2_basis(asset), state.alpha_fp16.astype(np.float64)
        )
    )
    output = transformed[0] if was_vector else transformed
    return _readonly(output, np.float32)


def transform_srdh2(asset: SRDH2Asset, state: SRDH2State, z_id: np.ndarray) -> np.ndarray:
    """Apply C2 independently to each row with its frozen nonlinear dictionary."""

    asset, state = _checked_srdh_pair(asset, state)
    rows, was_vector = _normalise_rows(z_id, "z_id")
    p, q = decode_srdh2_dictionary(asset)
    response = state.response_fp16.astype(np.float64)
    delta = (np.tanh(rows @ q) * response[None, :]) @ p.T
    transformed = _normalise_output(rows + delta)
    output = transformed[0] if was_vector else transformed
    return _readonly(output, np.float32)


def cspar2_metric_matrix(asset: CSPAR2Asset, state: CSPAR2State) -> np.ndarray:
    """Dense audit-only C1 metric; deployment transforms use the rank-two path."""

    asset, state = _checked_cspar_pair(asset, state)
    basis = decode_cspar2_basis(asset)
    metric = np.eye(Z_DIM, dtype=np.float64) - (
        basis * state.alpha_fp16.astype(np.float64)[None, :]
    ) @ basis.T
    metric = 0.5 * (metric + metric.T)
    if float(np.min(np.linalg.eigvalsh(metric))) <= 0.0:
        raise D129Joint6DAError("CSPAR-2 audit metric is not SPD")
    return _readonly(metric, np.float64)


def audit_d129_query_read_only(
    asset: D129Joint6Asset,
    state: Union[CSPAR2State, SRDH2State],
    query_z: np.ndarray,
) -> Dict[str, Any]:
    """Transform query rows and prove the state fingerprint/counters do not change."""

    if isinstance(asset, CSPAR2Asset) and isinstance(state, CSPAR2State):
        before = (state.alpha_fp16.tobytes(), state.support_root_sha256, state.receipt.as_dict())
        output = transform_cspar2(asset, state, query_z)
        after = (state.alpha_fp16.tobytes(), state.support_root_sha256, state.receipt.as_dict())
    elif isinstance(asset, SRDH2Asset) and isinstance(state, SRDH2State):
        before = (
            state.response_fp16.tobytes(),
            state.summary_fp16.tobytes(),
            state.support_root_sha256,
            state.receipt.as_dict(),
        )
        output = transform_srdh2(asset, state, query_z)
        after = (
            state.response_fp16.tobytes(),
            state.summary_fp16.tobytes(),
            state.support_root_sha256,
            state.receipt.as_dict(),
        )
    else:
        raise D129Joint6DAError("query audit requires a matched D129 asset/state pair")
    if before != after or not state.receipt.protocol_closed:
        raise D129Joint6DAError("D129 query path mutated support-only state")
    return {
        "schema": STATE_SCHEMA,
        "candidate_id": state.candidate_id,
        "query_output_shape": list(output.shape),
        "query_rows_used_for_fit": state.receipt.query_rows_used_for_fit,
        "query_state_updates": state.receipt.query_state_updates,
        "query_selection_count": state.receipt.query_selection_count,
        "query_gradient_calls": state.receipt.query_gradient_calls,
        "truth_role_quota_inputs": state.receipt.truth_role_quota_inputs,
        "global_reassignment_calls": state.receipt.global_reassignment_calls,
        "protocol_closed": state.receipt.protocol_closed,
    }


def d129_label_permutation_receipt(
    asset: D129Joint6Asset,
    support_z: np.ndarray,
    permutation: np.ndarray,
    probe_z: np.ndarray,
) -> Dict[str, Any]:
    """Check support-class permutation invariance and query-map equality.

    Labels never enter a state formula.  The caller supplies a class-axis
    permutation solely for this audit; it is not stored or used at inference.
    """

    support = _canonicalise_support(support_z)
    if not isinstance(permutation, np.ndarray) or permutation.dtype.kind not in {"i", "u"}:
        raise D129Joint6DAError("permutation must be an integer NumPy vector")
    if permutation.shape != (support.shape[0],) or set(permutation.tolist()) != set(
        range(support.shape[0])
    ):
        raise D129Joint6DAError("permutation must bijectively cover support classes")
    permuted_input = np.asarray(support_z)[permutation]
    if isinstance(asset, CSPAR2Asset):
        first = fit_cspar2_support(asset, support_z)
        second = fit_cspar2_support(asset, permuted_input)
        np.testing.assert_array_equal(first.alpha_fp16, second.alpha_fp16)
        first_output = transform_cspar2(asset, first, probe_z)
        second_output = transform_cspar2(asset, second, probe_z)
        coefficient_equal = bool(np.array_equal(first.alpha_fp16, second.alpha_fp16))
    elif isinstance(asset, SRDH2Asset):
        first = fit_srdh2_support(asset, support_z)
        second = fit_srdh2_support(asset, permuted_input)
        np.testing.assert_array_equal(first.response_fp16, second.response_fp16)
        np.testing.assert_array_equal(first.summary_fp16, second.summary_fp16)
        first_output = transform_srdh2(asset, first, probe_z)
        second_output = transform_srdh2(asset, second, probe_z)
        coefficient_equal = bool(
            np.array_equal(first.response_fp16, second.response_fp16)
            and np.array_equal(first.summary_fp16, second.summary_fp16)
        )
    else:
        raise D129Joint6DAError("permutation audit requires a D129 joint6 asset")
    if not np.array_equal(first_output, second_output):
        raise D129Joint6DAError("D129 class permutation changed the query map")
    return {
        "schema": STATE_SCHEMA,
        "candidate_id": asset.candidate_id,
        "class_count": int(support.shape[0]),
        "coefficient_bitwise_equal": coefficient_equal,
        "query_map_bitwise_equal": True,
    }


@dataclass(frozen=True)
class D129LOCORecord:
    """One source-audit physical identifier; never stored in a deployment asset."""

    receiver: str
    class_token: str
    physical_id: str

    def __post_init__(self) -> None:
        if not all(isinstance(item, str) and item for item in (
            self.receiver,
            self.class_token,
            self.physical_id,
        )):
            raise D129Joint6DAError("LOCO receiver/class/physical identifiers must be nonempty strings")


@dataclass(frozen=True)
class D129LOCOFold:
    """One 7x6 Phase1 outer fold, retaining only content roots and counts."""

    held_receiver: str
    held_class: str
    phase1_fit_count: int
    phase1_fit_physical_root_sha256: str
    support_k1_count: int
    support_k1_physical_root_sha256: str
    support_k5_count: int
    support_k5_physical_root_sha256: str
    outer_query_count: int
    outer_query_physical_root_sha256: str
    k1_is_k5_prefix: bool

    def __post_init__(self) -> None:
        if not self.held_receiver or not self.held_class:
            raise D129Joint6DAError("LOCO fold requires held receiver and class")
        if (
            self.phase1_fit_count <= 0
            or self.support_k1_count <= 0
            or self.support_k5_count <= 0
            or self.outer_query_count <= 0
            or self.support_k1_count > self.support_k5_count
            or not self.k1_is_k5_prefix
        ):
            raise D129Joint6DAError("LOCO fold count/prefix invariant drift")
        for value, name in (
            (self.phase1_fit_physical_root_sha256, "phase1_fit_physical_root_sha256"),
            (self.support_k1_physical_root_sha256, "support_k1_physical_root_sha256"),
            (self.support_k5_physical_root_sha256, "support_k5_physical_root_sha256"),
            (self.outer_query_physical_root_sha256, "outer_query_physical_root_sha256"),
        ):
            _require_sha256(value, name)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "held_receiver": self.held_receiver,
            "held_class": self.held_class,
            "phase1_fit_count": self.phase1_fit_count,
            "phase1_fit_physical_root_sha256": self.phase1_fit_physical_root_sha256,
            "support_k1_count": self.support_k1_count,
            "support_k1_physical_root_sha256": self.support_k1_physical_root_sha256,
            "support_k5_count": self.support_k5_count,
            "support_k5_physical_root_sha256": self.support_k5_physical_root_sha256,
            "outer_query_count": self.outer_query_count,
            "outer_query_physical_root_sha256": self.outer_query_physical_root_sha256,
            "k1_is_k5_prefix": self.k1_is_k5_prefix,
        }


@dataclass(frozen=True)
class D129LOCOPlan:
    """Complete receiver-held x class-LOCO coverage plan with no raw IDs."""

    receivers: Tuple[str, ...]
    classes: Tuple[str, ...]
    folds: Tuple[D129LOCOFold, ...]
    salt: str = LOCO_SALT
    schema: str = LOCO_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LOCO_PLAN_SCHEMA or self.salt != LOCO_SALT:
            raise D129Joint6DAError("D129 LOCO plan schema/salt drift")
        if len(self.receivers) != 7 or len(self.classes) != 6:
            raise D129Joint6DAError("D129 LOCO plan requires exactly 7 receivers and 6 classes")
        if len(set(self.receivers)) != 7 or len(set(self.classes)) != 6:
            raise D129Joint6DAError("D129 LOCO plan receiver/class duplication")
        if len(self.folds) != 42:
            raise D129Joint6DAError("D129 LOCO plan requires exactly 42 folds")
        expected = {(receiver, token) for receiver in self.receivers for token in self.classes}
        actual = {(fold.held_receiver, fold.held_class) for fold in self.folds}
        if actual != expected:
            raise D129Joint6DAError("D129 LOCO fold coverage is incomplete or duplicated")

    def coverage_receipt(self) -> Dict[str, Any]:
        folds = [fold.as_dict() for fold in self.folds]
        receipt: Dict[str, Any] = {
            "schema": self.schema,
            "salt": self.salt,
            "receiver_count": len(self.receivers),
            "class_count": len(self.classes),
            "fold_count": len(folds),
            "k1_is_k5_prefix_all_folds": all(fold.k1_is_k5_prefix for fold in self.folds),
            "physical_ids_persisted_in_plan": 0,
            "folds": folds,
        }
        receipt["coverage_receipt_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
        return receipt


def _root_from_ids(ids: Sequence[str]) -> str:
    return _sha256_bytes("\n".join(ids).encode("utf-8"))


def _ordered_cell_ids(receiver: str, class_token: str, ids: Sequence[str]) -> Tuple[str, ...]:
    return tuple(
        sorted(
            ids,
            key=lambda physical_id: _sha256_bytes(
                f"{LOCO_SALT}|{receiver}|{class_token}|{physical_id}".encode("utf-8")
            ),
        )
    )


def build_d129_loco_plan(records: Iterable[D129LOCORecord]) -> D129LOCOPlan:
    """Build the frozen 7 receiver x 6 class LOCO coverage receipt.

    Each cell must provide exactly fourteen physical samples.  For fold ``(r,c)``
    the Phase1 asset-build side excludes both ``r`` and ``c``.  The held
    receiver still contributes legitimate K-shot registration support for all
    six classes: K1 is one physical sample per class and K5 is the five-sample
    per-class extension.  The disjoint nine-sample suffix of every held-
    receiver cell is outer query, so old/new and floor metrics are available
    from the same fold.  Only roots/counts survive in the returned plan,
    never physical identifiers themselves.
    """

    values = tuple(records)
    if not values or not all(isinstance(value, D129LOCORecord) for value in values):
        raise D129Joint6DAError("LOCO plan requires nonempty D129LOCORecord values")
    receivers = tuple(sorted({value.receiver for value in values}))
    classes = tuple(sorted({value.class_token for value in values}))
    if len(receivers) != 7 or len(classes) != 6:
        raise D129Joint6DAError("LOCO input must cover exactly 7 receivers and 6 classes")
    cells: Dict[Tuple[str, str], List[str]] = {
        (receiver, token): [] for receiver in receivers for token in classes
    }
    seen = set()
    seen_physical_ids = set()
    for value in values:
        key = (value.receiver, value.class_token, value.physical_id)
        if key in seen:
            raise D129Joint6DAError("LOCO input contains duplicate physical IDs in one cell")
        if value.physical_id in seen_physical_ids:
            raise D129Joint6DAError("LOCO input reuses a physical ID across receiver/class cells")
        seen.add(key)
        seen_physical_ids.add(value.physical_id)
        cells[(value.receiver, value.class_token)].append(value.physical_id)
    ordered_cells: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    for key, ids in cells.items():
        if len(ids) != 14:
            raise D129Joint6DAError("every D129 LOCO receiver/class cell requires 14 physical IDs")
        ordered_cells[key] = _ordered_cell_ids(key[0], key[1], ids)

    folds: List[D129LOCOFold] = []
    for held_receiver in receivers:
        for held_class in classes:
            # Keep the five retained seen classes first and the held seen-class
            # proxy last.  This is a task-group ordering only; it does not
            # re-label the Phase1-seen held TX as a registered-new class.
            registry = tuple(token for token in classes if token != held_class) + (
                held_class,
            )
            phase1_fit = tuple(
                physical_id
                for receiver in receivers
                if receiver != held_receiver
                for token in classes
                if token != held_class
                for physical_id in ordered_cells[(receiver, token)]
            )
            support_k5 = tuple(
                physical_id
                for token in registry
                for physical_id in ordered_cells[(held_receiver, token)][:5]
            )
            support_k1 = tuple(
                ordered_cells[(held_receiver, token)][0]
                for token in registry
            )
            outer_query = tuple(
                physical_id
                for token in registry
                for physical_id in ordered_cells[(held_receiver, token)][5:]
            )
            if set(phase1_fit) & set(support_k5) or set(phase1_fit) & set(outer_query) or set(support_k5) & set(outer_query):
                raise D129Joint6DAError("LOCO physical-ID isolation drift")
            expected_k1 = tuple(
                ordered_cells[(held_receiver, token)][0]
                for token in registry
            )
            per_class_prefix = all(
                support_k1[index] == support_k5[index * 5]
                for index in range(len(registry))
            )
            folds.append(
                D129LOCOFold(
                    held_receiver=held_receiver,
                    held_class=held_class,
                    phase1_fit_count=len(phase1_fit),
                    phase1_fit_physical_root_sha256=_root_from_ids(phase1_fit),
                    support_k1_count=len(support_k1),
                    support_k1_physical_root_sha256=_root_from_ids(support_k1),
                    support_k5_count=len(support_k5),
                    support_k5_physical_root_sha256=_root_from_ids(support_k5),
                    outer_query_count=len(outer_query),
                    outer_query_physical_root_sha256=_root_from_ids(outer_query),
                    k1_is_k5_prefix=(support_k1 == expected_k1 and per_class_prefix),
                )
            )
    return D129LOCOPlan(receivers=receivers, classes=classes, folds=tuple(folds))


# Concise aliases for the integration surface; all retain the strict typed APIs.
fit_c1_cspar2_support = fit_cspar2_support
fit_c2_srdh2_support = fit_srdh2_support
apply_c1_cspar2 = transform_cspar2
apply_c2_srdh2 = transform_srdh2


__all__ = [
    "ALLOWED_K",
    "ASSET_SCHEMA",
    "ASSET_WIRE_MAGIC",
    "CSPAR2_CANDIDATE_ID",
    "CSPAR2Asset",
    "CSPAR2State",
    "D129Joint6Asset",
    "D129Joint6DAError",
    "D129LOCOFold",
    "D129LOCOPlan",
    "D129LOCORecord",
    "D129ResourceReceipt",
    "LOCO_PLAN_SCHEMA",
    "LOCO_SALT",
    "PHASE1_BUILD_SCHEMA",
    "RANK",
    "RESOURCE_SCHEMA",
    "SRDH2_CANDIDATE_ID",
    "SRDH2Asset",
    "SRDH2State",
    "STATE_SCHEMA",
    "Z_DIM",
    "apply_c1_cspar2",
    "apply_c2_srdh2",
    "audit_d129_query_read_only",
    "build_d129_phase1_assets",
    "build_d129_loco_plan",
    "cspar2_metric_matrix",
    "d129_joint6_asset_sha256",
    "d129_label_permutation_receipt",
    "d129_phase1_aggregate_sha256",
    "decode_cspar2_basis",
    "decode_srdh2_dictionary",
    "deserialize_d129_joint6_asset",
    "fit_c1_cspar2_support",
    "fit_c2_srdh2_support",
    "fit_cspar2_support",
    "fit_srdh2_support",
    "serialize_d129_joint6_asset",
    "transform_cspar2",
    "transform_srdh2",
]
