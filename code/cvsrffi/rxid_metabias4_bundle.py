"""D103-R1 RXID-DUALSPLIT-MB4 immutable deployment bundle.

Only aggregate learned arrays are accepted.  U, B, g and t use row-wise
symmetric INT8.  Precision and sigma use per-tensor log-affine INT8.  The
only floating-point payload values are binary16 quantization parameters and
the frozen binary16 scalar controls.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any

import numpy as np


SCHEMA = "cvs.phase1.rxid_dualsplit_metabias4.bundle.v1"
WIRE_MAGIC = b"CVSRXIDMB4\x00\x01"
Z_DIM = 160
DOMAIN_DIM = 32
CODE_DIM = 4
INT8_MAX = 127.0
PRECISION_BOUNDS = (0.05, 20.0)
SIGMA_BOUNDS = (0.05, 2.0)

TEMPERATURE_FP16 = np.float16(0.25)
LAMBDA0_FP16 = np.ones(CODE_DIM, dtype=np.float16)
AMAX_FP16 = np.full(CODE_DIM, np.float16(0.25), dtype=np.float16)
RADIUS_FP16 = np.float16(0.35009765625)

PAYLOAD_MEMBERS = (
    "u_codes_qint8",
    "u_scales_fp16",
    "b_codes_qint8",
    "b_scales_fp16",
    "bank_g_codes_qint8",
    "bank_g_scales_fp16",
    "bank_t_codes_qint8",
    "bank_t_scales_fp16",
    "bank_precision_codes_qint8",
    "bank_precision_log_offset_fp16",
    "bank_precision_log_scale_fp16",
    "bank_sigma_codes_qint8",
    "bank_sigma_log_offset_fp16",
    "bank_sigma_log_scale_fp16",
    "temperature_fp16",
    "lambda0_fp16",
    "amax_fp16",
    "radius_fp16",
    "cell_min_physical_count_int16",
    "cell_class_count_int16",
)


class RXIDMetaBias4BundleError(ValueError):
    """Raised when the frozen D103-R1 bundle ABI is violated."""


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


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise RXIDMetaBias4BundleError(f"{name} must be a lowercase SHA256")
    return text


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _array_receipt(value: Any) -> dict[str, Any]:
    source = np.asarray(value)
    array = (
        np.array(source, dtype=source.dtype, copy=True).reshape(())
        if source.ndim == 0
        else np.ascontiguousarray(source)
    )
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def quantize_rowwise_symmetric_int8(
    value: np.ndarray, *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize finite rows with RNE, [-127,127], and the frozen zero-row rule."""

    rows = np.asarray(value)
    if rows.dtype != np.float32 or rows.ndim != 2 or not np.isfinite(rows).all():
        raise RXIDMetaBias4BundleError(f"{name} must be finite float32 rows")
    maximum = np.max(np.abs(rows.astype(np.float64)), axis=1)
    scales64 = np.where(maximum > 0.0, maximum / INT8_MAX, 1.0)
    codes = np.rint(rows.astype(np.float64) / scales64[:, None])
    codes = np.clip(codes, -127.0, 127.0).astype(np.int8)
    scales = scales64.astype(np.float16)
    zero = maximum == 0.0
    codes[zero] = np.int8(0)
    scales[zero] = np.float16(1.0)
    if (
        np.any(codes == np.int8(-128))
        or np.any(scales <= 0.0)
        or not np.isfinite(scales).all()
    ):
        raise RXIDMetaBias4BundleError(f"{name} rowwise INT8 closure failed")
    return _readonly(codes, np.int8), _readonly(scales, np.float16)


def decode_rowwise_symmetric_int8(
    codes: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    code_array = np.asarray(codes)
    scale_array = np.asarray(scales)
    if (
        code_array.dtype != np.int8
        or code_array.ndim != 2
        or scale_array.dtype != np.float16
        or scale_array.shape != (len(code_array),)
        or np.any(code_array == np.int8(-128))
        or np.any(scale_array <= 0.0)
        or not np.isfinite(scale_array).all()
    ):
        raise RXIDMetaBias4BundleError("rowwise INT8 decode contract drift")
    return np.ascontiguousarray(
        code_array.astype(np.float32) * scale_array.astype(np.float32)[:, None],
        dtype=np.float32,
    )


def quantize_log_affine_int8(
    value: np.ndarray,
    *,
    bounds: tuple[float, float],
    name: str,
) -> tuple[np.ndarray, np.float16, np.float16]:
    """Clip, log, and per-tensor affine-quantize using signed INT8 and RNE."""

    array = np.asarray(value)
    if array.dtype != np.float32 or array.size < 1 or not np.isfinite(array).all():
        raise RXIDMetaBias4BundleError(f"{name} must be finite non-empty float32")
    low, high = (float(bounds[0]), float(bounds[1]))
    if not 0.0 < low < high:
        raise RXIDMetaBias4BundleError(f"{name} log-affine bounds drift")
    logged = np.log(np.clip(array.astype(np.float64), low, high))
    minimum = float(np.min(logged))
    maximum = float(np.max(logged))
    if maximum == minimum:
        offset64 = minimum
        scale64 = 1.0
        codes = np.zeros(array.shape, dtype=np.int8)
    else:
        offset64 = 0.5 * (minimum + maximum)
        scale64 = (maximum - minimum) / 254.0
        codes = np.clip(
            np.rint((logged - offset64) / scale64), -127.0, 127.0
        ).astype(np.int8)
    offset = np.float16(offset64)
    scale = np.float16(scale64)
    if (
        not np.isfinite(offset)
        or not np.isfinite(scale)
        or scale <= 0.0
        or np.any(codes == np.int8(-128))
    ):
        raise RXIDMetaBias4BundleError(f"{name} log-affine INT8 closure failed")
    return _readonly(codes, np.int8), offset, scale


def decode_log_affine_int8(
    codes: np.ndarray,
    offset: np.float16,
    scale: np.float16,
    *,
    bounds: tuple[float, float],
) -> np.ndarray:
    code_array = np.asarray(codes)
    offset16 = np.asarray(offset)
    scale16 = np.asarray(scale)
    if (
        code_array.dtype != np.int8
        or np.any(code_array == np.int8(-128))
        or offset16.dtype != np.float16
        or offset16.shape != ()
        or scale16.dtype != np.float16
        or scale16.shape != ()
        or not np.isfinite(offset16)
        or not np.isfinite(scale16)
        or scale16 <= 0.0
    ):
        raise RXIDMetaBias4BundleError("log-affine INT8 decode contract drift")
    decoded = np.exp(
        code_array.astype(np.float32) * np.float32(scale16)
        + np.float32(offset16)
    )
    return np.ascontiguousarray(
        np.clip(decoded, float(bounds[0]), float(bounds[1])), dtype=np.float32
    )


@dataclass(frozen=True, slots=True)
class RXIDMetaBias4Bundle:
    u_codes_qint8: np.ndarray
    u_scales_fp16: np.ndarray
    b_codes_qint8: np.ndarray
    b_scales_fp16: np.ndarray
    bank_g_codes_qint8: np.ndarray
    bank_g_scales_fp16: np.ndarray
    bank_t_codes_qint8: np.ndarray
    bank_t_scales_fp16: np.ndarray
    bank_precision_codes_qint8: np.ndarray
    bank_precision_log_offset_fp16: np.float16
    bank_precision_log_scale_fp16: np.float16
    bank_sigma_codes_qint8: np.ndarray
    bank_sigma_log_offset_fp16: np.float16
    bank_sigma_log_scale_fp16: np.float16
    temperature_fp16: np.float16
    lambda0_fp16: np.ndarray
    amax_fp16: np.ndarray
    radius_fp16: np.float16
    cell_min_physical_count_int16: np.ndarray
    cell_class_count_int16: np.ndarray
    checkpoint_sha256: str
    runtime_sha256: str
    method_lock_sha256: str
    training_receipt_sha256: str
    nested_receipt_sha256: str
    tx_probe_receipt_sha256: str
    aggregation_receipt_sha256: str
    quantization_receipt_sha256: str
    tx_probe_mean_balanced_accuracy: float
    tx_probe_max_balanced_accuracy: float
    content_root_sha256: str = ""
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        arrays = {name: np.asarray(getattr(self, name)) for name in PAYLOAD_MEMBERS}
        cells = int(arrays["bank_g_codes_qint8"].shape[0])
        expected = {
            "u_codes_qint8": (np.dtype(np.int8), (DOMAIN_DIM, Z_DIM)),
            "u_scales_fp16": (np.dtype(np.float16), (DOMAIN_DIM,)),
            "b_codes_qint8": (np.dtype(np.int8), (Z_DIM, CODE_DIM)),
            "b_scales_fp16": (np.dtype(np.float16), (Z_DIM,)),
            "bank_g_codes_qint8": (np.dtype(np.int8), (cells, DOMAIN_DIM)),
            "bank_g_scales_fp16": (np.dtype(np.float16), (cells,)),
            "bank_t_codes_qint8": (np.dtype(np.int8), (cells, CODE_DIM)),
            "bank_t_scales_fp16": (np.dtype(np.float16), (cells,)),
            "bank_precision_codes_qint8": (
                np.dtype(np.int8),
                (cells, CODE_DIM),
            ),
            "bank_precision_log_offset_fp16": (np.dtype(np.float16), ()),
            "bank_precision_log_scale_fp16": (np.dtype(np.float16), ()),
            "bank_sigma_codes_qint8": (np.dtype(np.int8), (cells,)),
            "bank_sigma_log_offset_fp16": (np.dtype(np.float16), ()),
            "bank_sigma_log_scale_fp16": (np.dtype(np.float16), ()),
            "temperature_fp16": (np.dtype(np.float16), ()),
            "lambda0_fp16": (np.dtype(np.float16), (CODE_DIM,)),
            "amax_fp16": (np.dtype(np.float16), (CODE_DIM,)),
            "radius_fp16": (np.dtype(np.float16), ()),
            "cell_min_physical_count_int16": (np.dtype(np.int16), (cells,)),
            "cell_class_count_int16": (np.dtype(np.int16), (cells,)),
        }
        if self.schema != SCHEMA or cells < 2:
            raise RXIDMetaBias4BundleError("bundle schema/bank count drift")
        for name, (dtype, shape) in expected.items():
            array = arrays[name]
            if array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
                raise RXIDMetaBias4BundleError(f"{name} dtype/shape/finite drift")
        int8_names = (
            "u_codes_qint8",
            "b_codes_qint8",
            "bank_g_codes_qint8",
            "bank_t_codes_qint8",
            "bank_precision_codes_qint8",
            "bank_sigma_codes_qint8",
        )
        if any(np.any(arrays[name] == np.int8(-128)) for name in int8_names):
            raise RXIDMetaBias4BundleError("bundle contains forbidden INT8 -128")
        for code_name, scale_name in (
            ("u_codes_qint8", "u_scales_fp16"),
            ("b_codes_qint8", "b_scales_fp16"),
            ("bank_g_codes_qint8", "bank_g_scales_fp16"),
            ("bank_t_codes_qint8", "bank_t_scales_fp16"),
        ):
            codes = arrays[code_name]
            scales = arrays[scale_name]
            if np.any(scales <= 0.0):
                raise RXIDMetaBias4BundleError(f"{scale_name} must be positive")
            zero_rows = np.all(codes == 0, axis=1)
            if np.any(scales[zero_rows] != np.float16(1.0)):
                raise RXIDMetaBias4BundleError(
                    f"{code_name} zero rows require binary16 scale 1"
                )
        if (
            arrays["bank_precision_log_scale_fp16"] <= 0.0
            or arrays["bank_sigma_log_scale_fp16"] <= 0.0
            or not np.array_equal(arrays["lambda0_fp16"], LAMBDA0_FP16)
            or not np.array_equal(arrays["amax_fp16"], AMAX_FP16)
            or arrays["temperature_fp16"].tobytes()
            != np.asarray(TEMPERATURE_FP16).tobytes()
            or arrays["radius_fp16"].tobytes()
            != np.asarray(RADIUS_FP16).tobytes()
            or np.any(arrays["cell_min_physical_count_int16"] < 2)
            or np.any(arrays["cell_class_count_int16"] < 2)
        ):
            raise RXIDMetaBias4BundleError("fixed scalar/aggregate closure drift")
        for field in (
            "checkpoint_sha256",
            "runtime_sha256",
            "method_lock_sha256",
            "training_receipt_sha256",
            "nested_receipt_sha256",
            "tx_probe_receipt_sha256",
            "aggregation_receipt_sha256",
            "quantization_receipt_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        tx_mean = float(self.tx_probe_mean_balanced_accuracy)
        tx_max = float(self.tx_probe_max_balanced_accuracy)
        if (
            not np.isfinite(tx_mean)
            or not np.isfinite(tx_max)
            or not 0.0 <= tx_mean <= 1.0
            or not 0.0 <= tx_max <= 1.0
            or tx_mean > tx_max
        ):
            raise RXIDMetaBias4BundleError("dual TX probe values must be finite [0,1]")
        for name, (dtype, shape) in expected.items():
            if shape == ():
                object.__setattr__(self, name, np.float16(arrays[name]))
            else:
                object.__setattr__(self, name, _readonly(arrays[name], dtype))
        expected_root = self._content_root()
        if self.content_root_sha256 and self.content_root_sha256 != expected_root:
            raise RXIDMetaBias4BundleError("bundle content root drift")
        object.__setattr__(self, "content_root_sha256", expected_root)

    def _content_root(self) -> str:
        return _sha256_bytes(
            _canonical_bytes(
                {
                    "schema": self.schema,
                    "arrays": {
                        name: _array_receipt(getattr(self, name))
                        for name in PAYLOAD_MEMBERS
                    },
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "runtime_sha256": self.runtime_sha256,
                    "method_lock_sha256": self.method_lock_sha256,
                    "training_receipt_sha256": self.training_receipt_sha256,
                    "nested_receipt_sha256": self.nested_receipt_sha256,
                    "tx_probe_receipt_sha256": self.tx_probe_receipt_sha256,
                    "aggregation_receipt_sha256": self.aggregation_receipt_sha256,
                    "quantization_receipt_sha256": self.quantization_receipt_sha256,
                    "tx_probe_mean_balanced_accuracy": float(
                        self.tx_probe_mean_balanced_accuracy
                    ),
                    "tx_probe_max_balanced_accuracy": float(
                        self.tx_probe_max_balanced_accuracy
                    ),
                    "persistent_fp16_or_fp32_learned_sidecar": False,
                }
            )
        )

    @property
    def bank_count(self) -> int:
        return int(len(self.bank_g_codes_qint8))

    @property
    def tx_probe_gate_pass(self) -> bool:
        return bool(
            float(self.tx_probe_mean_balanced_accuracy) <= 0.25
            and float(self.tx_probe_max_balanced_accuracy) <= 0.25
        )

    @property
    def numeric_state_bytes(self) -> int:
        return int(sum(np.asarray(getattr(self, name)).nbytes for name in PAYLOAD_MEMBERS))

    def decode_u(self) -> np.ndarray:
        return decode_rowwise_symmetric_int8(
            self.u_codes_qint8, self.u_scales_fp16
        )

    def decode_b(self) -> np.ndarray:
        return decode_rowwise_symmetric_int8(
            self.b_codes_qint8, self.b_scales_fp16
        )

    def decode_bank_g(self) -> np.ndarray:
        return decode_rowwise_symmetric_int8(
            self.bank_g_codes_qint8, self.bank_g_scales_fp16
        )

    def decode_bank_t(self) -> np.ndarray:
        return decode_rowwise_symmetric_int8(
            self.bank_t_codes_qint8, self.bank_t_scales_fp16
        )

    def decode_bank_precision(self) -> np.ndarray:
        return decode_log_affine_int8(
            self.bank_precision_codes_qint8,
            self.bank_precision_log_offset_fp16,
            self.bank_precision_log_scale_fp16,
            bounds=PRECISION_BOUNDS,
        )

    def decode_bank_sigma(self) -> np.ndarray:
        return decode_log_affine_int8(
            self.bank_sigma_codes_qint8,
            self.bank_sigma_log_offset_fp16,
            self.bank_sigma_log_scale_fp16,
            bounds=SIGMA_BOUNDS,
        )


def build_rxid_metabias4_bundle(
    u: np.ndarray,
    b: np.ndarray,
    bank_g: np.ndarray,
    bank_t: np.ndarray,
    bank_precision: np.ndarray,
    bank_sigma: np.ndarray,
    *,
    cell_min_physical_count: np.ndarray,
    cell_class_count: np.ndarray,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    training_receipt_sha256: str,
    nested_receipt_sha256: str,
    tx_probe_receipt_sha256: str,
    aggregation_receipt_sha256: str,
    quantization_receipt_sha256: str,
    tx_probe_mean_balanced_accuracy: float,
    tx_probe_max_balanced_accuracy: float,
) -> RXIDMetaBias4Bundle:
    """Compile already-trained aggregate values into the frozen D103-R1 ABI."""

    arrays = tuple(np.asarray(value) for value in (u, b, bank_g, bank_t))
    expected = (
        (DOMAIN_DIM, Z_DIM),
        (Z_DIM, CODE_DIM),
        (len(arrays[2]), DOMAIN_DIM),
        (len(arrays[2]), CODE_DIM),
    )
    if any(array.dtype != np.float32 or array.shape != shape for array, shape in zip(arrays, expected)):
        raise RXIDMetaBias4BundleError("U/B/g/t float32 shape drift")
    cells = len(arrays[2])
    precision = np.asarray(bank_precision)
    sigma = np.asarray(bank_sigma)
    if (
        precision.dtype != np.float32
        or precision.shape != (cells, CODE_DIM)
        or sigma.dtype != np.float32
        or sigma.shape != (cells,)
    ):
        raise RXIDMetaBias4BundleError("precision/sigma float32 shape drift")
    u_codes, u_scales = quantize_rowwise_symmetric_int8(arrays[0], name="U")
    b_codes, b_scales = quantize_rowwise_symmetric_int8(arrays[1], name="B")
    g_codes, g_scales = quantize_rowwise_symmetric_int8(arrays[2], name="g")
    t_codes, t_scales = quantize_rowwise_symmetric_int8(arrays[3], name="t")
    p_codes, p_offset, p_scale = quantize_log_affine_int8(
        precision, bounds=PRECISION_BOUNDS, name="precision"
    )
    s_codes, s_offset, s_scale = quantize_log_affine_int8(
        sigma, bounds=SIGMA_BOUNDS, name="sigma"
    )
    return RXIDMetaBias4Bundle(
        u_codes_qint8=u_codes,
        u_scales_fp16=u_scales,
        b_codes_qint8=b_codes,
        b_scales_fp16=b_scales,
        bank_g_codes_qint8=g_codes,
        bank_g_scales_fp16=g_scales,
        bank_t_codes_qint8=t_codes,
        bank_t_scales_fp16=t_scales,
        bank_precision_codes_qint8=p_codes,
        bank_precision_log_offset_fp16=p_offset,
        bank_precision_log_scale_fp16=p_scale,
        bank_sigma_codes_qint8=s_codes,
        bank_sigma_log_offset_fp16=s_offset,
        bank_sigma_log_scale_fp16=s_scale,
        temperature_fp16=TEMPERATURE_FP16,
        lambda0_fp16=LAMBDA0_FP16,
        amax_fp16=AMAX_FP16,
        radius_fp16=RADIUS_FP16,
        cell_min_physical_count_int16=np.asarray(cell_min_physical_count),
        cell_class_count_int16=np.asarray(cell_class_count),
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        training_receipt_sha256=training_receipt_sha256,
        nested_receipt_sha256=nested_receipt_sha256,
        tx_probe_receipt_sha256=tx_probe_receipt_sha256,
        aggregation_receipt_sha256=aggregation_receipt_sha256,
        quantization_receipt_sha256=quantization_receipt_sha256,
        tx_probe_mean_balanced_accuracy=tx_probe_mean_balanced_accuracy,
        tx_probe_max_balanced_accuracy=tx_probe_max_balanced_accuracy,
    )


def _wire_array(name: str, value: Any) -> bytes:
    source = np.asarray(value)
    array = (
        np.array(source, dtype=source.dtype, copy=True).reshape(())
        if source.ndim == 0
        else np.ascontiguousarray(source)
    )
    encoded_name = name.encode("ascii")
    encoded_dtype = array.dtype.str.encode("ascii")
    shape = b"".join(struct.pack("<Q", int(axis)) for axis in array.shape)
    payload = array.tobytes(order="C")
    return b"".join(
        (
            struct.pack("<H", len(encoded_name)),
            encoded_name,
            struct.pack("<H", len(encoded_dtype)),
            encoded_dtype,
            struct.pack("<H", array.ndim),
            shape,
            struct.pack("<Q", len(payload)),
            payload,
        )
    )


def serialize_rxid_metabias4_bundle(bundle: RXIDMetaBias4Bundle) -> bytes:
    if type(bundle) is not RXIDMetaBias4Bundle:
        raise RXIDMetaBias4BundleError("serialization requires an exact D103 bundle")
    bundle.__post_init__()
    if not bundle.tx_probe_gate_pass:
        raise RXIDMetaBias4BundleError(
            "failed dual TX probe bundle is diagnostic-only and cannot be serialized"
        )
    header = _canonical_bytes(
        {
            "schema": bundle.schema,
            "content_root_sha256": bundle.content_root_sha256,
            "payload_members": list(PAYLOAD_MEMBERS),
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "runtime_sha256": bundle.runtime_sha256,
            "method_lock_sha256": bundle.method_lock_sha256,
            "training_receipt_sha256": bundle.training_receipt_sha256,
            "nested_receipt_sha256": bundle.nested_receipt_sha256,
            "tx_probe_receipt_sha256": bundle.tx_probe_receipt_sha256,
            "aggregation_receipt_sha256": bundle.aggregation_receipt_sha256,
            "quantization_receipt_sha256": bundle.quantization_receipt_sha256,
            "tx_probe_mean_balanced_accuracy": float(
                bundle.tx_probe_mean_balanced_accuracy
            ),
            "tx_probe_max_balanced_accuracy": float(
                bundle.tx_probe_max_balanced_accuracy
            ),
        }
    )
    return b"".join(
        (
            WIRE_MAGIC,
            struct.pack("<Q", len(header)),
            header,
            struct.pack("<H", len(PAYLOAD_MEMBERS)),
            *[
                _wire_array(name, getattr(bundle, name))
                for name in PAYLOAD_MEMBERS
            ],
        )
    )


def deserialize_rxid_metabias4_bundle(wire: bytes) -> RXIDMetaBias4Bundle:
    """Fail-closed reconstruction of one exact D103-R1 deployment bundle."""

    if type(wire) is not bytes:
        raise RXIDMetaBias4BundleError("bundle wire must be exact bytes")
    cursor = 0

    def take(length: int, name: str) -> bytes:
        nonlocal cursor
        if (
            type(length) is not int
            or length < 0
            or cursor + length > len(wire)
        ):
            raise RXIDMetaBias4BundleError(f"truncated bundle wire at {name}")
        result = wire[cursor : cursor + length]
        cursor += length
        return result

    def unpack(fmt: str, name: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        try:
            return struct.unpack(fmt, take(size, name))
        except struct.error as error:
            raise RXIDMetaBias4BundleError(
                f"invalid bundle wire integer at {name}"
            ) from error

    if take(len(WIRE_MAGIC), "magic") != WIRE_MAGIC:
        raise RXIDMetaBias4BundleError("bundle wire magic drift")
    (header_length,) = unpack("<Q", "header length")
    if header_length < 2 or header_length > 1_048_576:
        raise RXIDMetaBias4BundleError("bundle wire header length drift")
    header_bytes = take(int(header_length), "header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RXIDMetaBias4BundleError("bundle wire header JSON drift") from error
    header_keys = {
        "schema",
        "content_root_sha256",
        "payload_members",
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "training_receipt_sha256",
        "nested_receipt_sha256",
        "tx_probe_receipt_sha256",
        "aggregation_receipt_sha256",
        "quantization_receipt_sha256",
        "tx_probe_mean_balanced_accuracy",
        "tx_probe_max_balanced_accuracy",
    }
    if (
        type(header) is not dict
        or set(header) != header_keys
        or header_bytes != _canonical_bytes(header)
        or header.get("schema") != SCHEMA
        or header.get("payload_members") != list(PAYLOAD_MEMBERS)
    ):
        raise RXIDMetaBias4BundleError("bundle wire canonical header/schema drift")
    _require_sha256(
        header.get("content_root_sha256"), "wire content_root_sha256"
    )
    (member_count,) = unpack("<H", "member count")
    if member_count != len(PAYLOAD_MEMBERS):
        raise RXIDMetaBias4BundleError("bundle wire member count drift")

    arrays: dict[str, np.ndarray] = {}
    for expected_name in PAYLOAD_MEMBERS:
        (name_length,) = unpack("<H", f"{expected_name} name length")
        if name_length < 1 or name_length > 128:
            raise RXIDMetaBias4BundleError("bundle wire member name length drift")
        try:
            name = take(int(name_length), f"{expected_name} name").decode("ascii")
        except UnicodeDecodeError as error:
            raise RXIDMetaBias4BundleError(
                "bundle wire member name encoding drift"
            ) from error
        if name != expected_name:
            raise RXIDMetaBias4BundleError(
                f"bundle wire member order drift: expected {expected_name}"
            )
        (dtype_length,) = unpack("<H", f"{name} dtype length")
        if dtype_length < 1 or dtype_length > 32:
            raise RXIDMetaBias4BundleError("bundle wire dtype length drift")
        try:
            dtype_text = take(int(dtype_length), f"{name} dtype").decode("ascii")
            dtype = np.dtype(dtype_text)
        except (UnicodeDecodeError, TypeError) as error:
            raise RXIDMetaBias4BundleError(
                f"bundle wire dtype drift: {name}"
            ) from error
        if dtype.hasobject:
            raise RXIDMetaBias4BundleError("bundle wire object dtype forbidden")
        (ndim,) = unpack("<H", f"{name} ndim")
        if ndim > 4:
            raise RXIDMetaBias4BundleError(f"bundle wire ndim drift: {name}")
        shape = tuple(
            int(unpack("<Q", f"{name} shape")[0]) for _ in range(int(ndim))
        )
        if any(axis > 1_000_000 for axis in shape):
            raise RXIDMetaBias4BundleError(f"bundle wire shape bound drift: {name}")
        (payload_length,) = unpack("<Q", f"{name} payload length")
        element_count = math.prod(shape)
        expected_length = int(element_count) * int(dtype.itemsize)
        if payload_length != expected_length:
            raise RXIDMetaBias4BundleError(
                f"bundle wire payload length drift: {name}"
            )
        payload = take(int(payload_length), f"{name} payload")
        try:
            arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        except ValueError as error:
            raise RXIDMetaBias4BundleError(
                f"bundle wire array reconstruction drift: {name}"
            ) from error
    if cursor != len(wire):
        raise RXIDMetaBias4BundleError("bundle wire contains trailing bytes")
    try:
        bundle = RXIDMetaBias4Bundle(
            **arrays,
            checkpoint_sha256=header["checkpoint_sha256"],
            runtime_sha256=header["runtime_sha256"],
            method_lock_sha256=header["method_lock_sha256"],
            training_receipt_sha256=header["training_receipt_sha256"],
            nested_receipt_sha256=header["nested_receipt_sha256"],
            tx_probe_receipt_sha256=header["tx_probe_receipt_sha256"],
            aggregation_receipt_sha256=header["aggregation_receipt_sha256"],
            quantization_receipt_sha256=header["quantization_receipt_sha256"],
            tx_probe_mean_balanced_accuracy=header[
                "tx_probe_mean_balanced_accuracy"
            ],
            tx_probe_max_balanced_accuracy=header[
                "tx_probe_max_balanced_accuracy"
            ],
            content_root_sha256=header["content_root_sha256"],
            schema=header["schema"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, RXIDMetaBias4BundleError):
            raise
        raise RXIDMetaBias4BundleError(
            "bundle wire semantic reconstruction failed"
        ) from error
    if serialize_rxid_metabias4_bundle(bundle) != wire:
        raise RXIDMetaBias4BundleError("bundle wire exact roundtrip verification failed")
    return bundle


__all__ = [
    "AMAX_FP16",
    "CODE_DIM",
    "DOMAIN_DIM",
    "LAMBDA0_FP16",
    "PAYLOAD_MEMBERS",
    "PRECISION_BOUNDS",
    "RADIUS_FP16",
    "RXIDMetaBias4Bundle",
    "RXIDMetaBias4BundleError",
    "SCHEMA",
    "SIGMA_BOUNDS",
    "TEMPERATURE_FP16",
    "Z_DIM",
    "build_rxid_metabias4_bundle",
    "deserialize_rxid_metabias4_bundle",
    "decode_log_affine_int8",
    "decode_rowwise_symmetric_int8",
    "quantize_log_affine_int8",
    "quantize_rowwise_symmetric_int8",
    "serialize_rxid_metabias4_bundle",
]
