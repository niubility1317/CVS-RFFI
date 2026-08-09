"""Immutable Phase1 single-readout local4 technical control bundle.

This module intentionally implements only the frozen F1C control vertical
slice.  It is not a candidate selector, a performance reporter, or a
Phase3 multi-node method.  All fitting happens from caller-provided source
L/U/V physical records before :func:`build_bundle` writes a new package.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn

from cvsrffi.checkpoint_loading import (
    build_exact_ssdg_model_from_checkpoint,
    infer_num_domains_from_state,
    strip_module_prefix,
)
from cvsrffi.identity_only_forward import identity_only_feature_forward
from cvsrffi.phase3_care_poe import (
    FusionConfig,
    fuse_event,
    seal_local_evidence,
    validate_local_evidence,
)


SCHEMA = "cvs.phase1.single_control_bundle.v1"
FORMULA_ID = "single_control_local4_stress_tail_v1"
BUNDLE_STATUS = "TECHNICAL_LOCAL4_CONTROL_BUNDLE"
FIXTURE_STATUS = "FIXTURE_TECHNICAL_LOCAL4_CONTROL_BUNDLE"
F1C_CHECKPOINT_SHA256 = "0b1e1d24621f5c044b0a77f30915ec1f67342e6132fba8df28f21b43ad6b2ab8"
F1C_RUN_ID = "phase1_cp_sfce12_20260809_v2"
F1C_CANDIDATE_ID = "F1C_CP_SFCE12"
F1C_CHECKPOINT_LEAF = "final_ssdg.pth"
LOCAL4_HANDLES = ("20-15", "20-19", "6-15", "8-20")
KNOWN_VALIDATION_TX = "14-7"
PROXY_UNKNOWN_TX = "14-10"
EXPECTED_DATASET_SHA256 = "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
EXPECTED_TRAINING_COMPLETION_SHA256 = "c31edd31f1ec322615b4d0647cfcb9ece4e8ef5c3940d54aaa89c85c60f4431c"
EXPECTED_TERMINAL_STATUS_SHA256 = "0575ed6ee778e5b7b94e1e5b842e9ff24bf32496b05d36f82f658117a791c3a2"
EXPECTED_CP_TERMINAL_SHA256 = "5a9677d6eab883f221ceb5c544f8e0bf6bcdb26479bba326766494bb7ce482e0"
EXPECTED_SCENARIO_CONFIG_SHA256 = {
    "leo_clear_weak": "c046cdfbb48d8a0a6b011418374939e86f2a4ff450ab40a3f3ed4a333a53f159",
    "leo_low_elev_weak": "323aa6613292049605e04eb6be63c9754acb0655176a52c5d11501e2a1ae7e87",
    "leo_rain_weak": "66e72208dc21c4dea80130435eec50afd03d24fc3f014d0f8a73d720a14ead2b",
}
EXPECTED_TRAIN_EVAL_CONFIG_SHA256 = [
    EXPECTED_SCENARIO_CONFIG_SHA256["leo_low_elev_weak"],
    EXPECTED_SCENARIO_CONFIG_SHA256["leo_rain_weak"],
    EXPECTED_SCENARIO_CONFIG_SHA256["leo_clear_weak"],
]
EXPECTED_SCENARIO_REGISTRY_SHA256 = "d38c3bcc85699c97c9bca53a84a5268b51db140a6767c28fae06cf65cc5db215"
EXPECTED_CODE_SHA_PATHS = frozenset(
    {
        "code/cvsrffi/checkpoint_loading.py",
        "code/post_stage_common.py",
        "code/SSDG/train_ssdg.py",
        "code/model_dual_cvsincnet.py",
        "code/training_controls.py",
        "code/cvsrffi/eval.py",
        "code/sat_channel.py",
        "code/cvsrffi/tensors.py",
    }
)

INPUT_LEN = 256
ALPHA = 0.01
TAIL_LEVELS = np.asarray(
    [1.0 - 10.0 ** (-float(k) / 32.0) for k in range(129)], dtype=np.float64
)
TAIL_Q99_INDEX = 64
TAIL_FLOOR_MIN = 1.0e-4
MIN_CALIBRATION_PHYSICALS = 199
MIN_LABELED_PHYSICALS_PER_CLASS = 32
EPS = 1.0e-12
NORM_EPS = 1.0e-8
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
MAX_EVIDENCE_BYTES = 64 * 1024
MAX_CPU_RSS_DELTA_BYTES = 512 * 1024 * 1024
MAX_CUDA_VRAM_BYTES = 256 * 1024 * 1024
CPU_WARMUPS = 20
CPU_TRIALS = 100
CPU_P99_LIMIT_MS = 250.0
RESOURCE_INPUT_SEED = 7281105
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
VIEW_ORDER = ("clean", *SCENARIOS)
SEED_RULE_ID = "SCB1-SOURCE-VIEW-SEED-v2"
PREPROCESS_OPERATOR_ID = "wisig_center256_rms_iq_v1"

PAYLOAD_MEMBERS = (
    "runtime/local_evidence.ts",
    "state/class_geometry.npz",
    "state/domain_descriptor_stats.npz",
    "state/rank_tail_summary.npz",
    "locks/checkpoint_binding.json",
    "locks/class_binding.json",
    "locks/source_partition_receipt.json",
    "locks/runtime_parity_receipt.json",
    "locks/resource_receipt.json",
)
MANIFEST_NAME = "manifest.json"
ALL_BUNDLE_MEMBERS = (*PAYLOAD_MEMBERS, MANIFEST_NAME)

ALLOWED_REASON_CODES = {
    "SCB_REGISTERED",
    "SCB_TECHNICAL_UNKNOWN",
    "SCB_CONTEXT_DEFER",
}
FORBIDDEN_PREDICTOR_KEYS = {
    "role",
    "truth",
    "true_label",
    "query_truth",
    "registration_authorized",
    "credential",
    "scorer",
}


class SingleControlBundleError(ValueError):
    """Raised when the frozen local4 bundle contract is violated."""


@dataclass(frozen=True)
class ClassGeometry:
    class_handles: tuple[str, ...]
    centers: np.ndarray
    radii: np.ndarray
    class_counts: np.ndarray


@dataclass(frozen=True)
class DescriptorStats:
    median: np.ndarray
    scale: np.ndarray
    descriptor_count: int


@dataclass(frozen=True)
class TailSummary:
    levels: np.ndarray
    distance_values: np.ndarray
    energy_values: np.ndarray
    domain_values: np.ndarray
    n_calibration: int
    calibration_set_sha256: str


@dataclass(frozen=True)
class BundleState:
    geometry: ClassGeometry
    descriptor: DescriptorStats
    tail: TailSummary


@dataclass(frozen=True)
class LocalFields:
    z_id: np.ndarray
    z_dom: np.ndarray
    q: float
    d_class: np.ndarray
    e_unknown: float
    p_local: np.ndarray
    local_decision: str
    local_label: str | None
    reason_code: str
    known_consistency: float
    energy: float
    distance_score: float
    domain_score: float


def _error(message: str) -> SingleControlBundleError:
    return SingleControlBundleError(message)


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise _error(f"{field} must be a 64-character SHA256")
    lowered = value.lower()
    if any(char not in "0123456789abcdef" for char in lowered):
        raise _error(f"{field} must be lowercase hexadecimal SHA256")
    return lowered


def _require_nfc_string(value: Any, *, field: str, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise _error(f"{field} must be a string without implicit conversion")
    if unicodedata.normalize("NFC", value) != value:
        raise _error(f"{field} must be Unicode NFC")
    if nonempty and not value:
        raise _error(f"{field} must be non-empty")
    return value


def _require_nonbool_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(f"{field} must be a non-bool integer")
    integer = int(value)
    if minimum is not None and integer < int(minimum):
        raise _error(f"{field} must be >= {minimum}")
    return integer


def _require_finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _error(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _error(f"{field} must be finite")
    return result


def _canonical_value(value: Any) -> Any:
    """Return the exact restricted representation used for binding hashes."""

    if isinstance(value, Mapping):
        if any(not isinstance(raw_key, str) for raw_key in value):
            raise _error("canonical mappings require string keys")
        result: dict[str, Any] = {}
        for raw_key in sorted(value.keys()):
            key = _require_nfc_string(raw_key, field="canonical mapping key", nonempty=False)
            if key in result:
                raise _error("canonical mapping has duplicate normalized keys")
            result[key] = _canonical_value(value[raw_key])
        return result
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, str):
        return _require_nfc_string(value, field="canonical string", nonempty=False)
    if isinstance(value, bool):
        return value
    if type(value) is int:
        return int(value)
    if type(value) is float:
        number = float(value)
        if not math.isfinite(number):
            raise _error("canonical float must be finite")
        if number == 0.0:
            return "f64:0x0.0p+0"
        return "f64:" + number.hex()
    raise _error(f"canonical value type is forbidden: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_key(
    tx_label: Any, rx_label: Any, day_label: Any, sig_i: Any
) -> tuple[str, str, str, int]:
    """Return the frozen physical key; ``eq_label`` is deliberately excluded."""

    return (
        _require_nfc_string(tx_label, field="tx_label"),
        _require_nfc_string(rx_label, field="rx_label"),
        _require_nfc_string(day_label, field="day_label"),
        _require_nonbool_int(sig_i, field="sig_i", minimum=0),
    )


def physical_token(key: Sequence[Any]) -> str:
    if len(key) != 4:
        raise _error("physical key must have exactly four fields")
    verified = physical_key(key[0], key[1], key[2], key[3])
    return canonical_json_bytes(list(verified)).decode("utf-8")


def physical_set_sha256(keys: Iterable[Sequence[Any]]) -> str:
    tokens = sorted(physical_token(key) for key in keys)
    if len(tokens) != len(set(tokens)):
        raise _error("physical set contains duplicate canonical keys")
    return canonical_sha256(tokens)


def opaque_source_sample_hash(opaque_sample_index: Any) -> str:
    """Return the label-blind identity used by descriptor aggregation only."""

    index = _require_nonbool_int(opaque_sample_index, field="opaque sample index", minimum=0)
    return _sha256_bytes(b"SCB1-OPAQUE-SAMPLE\x00" + str(index).encode("ascii"))


def derive_source_view_seed(
    *, split_seed: Any, opaque_sample_index: Any, scenario: Any
) -> int:
    """Derive a per-source-sample view seed without any class/TX material."""

    seed = _require_nonbool_int(split_seed, field="split_seed", minimum=0)
    if seed != 7281105:
        raise _error("split_seed must equal the frozen F1C seed")
    if not isinstance(scenario, str) or scenario not in SCENARIOS or not scenario.isascii():
        raise _error("scenario must be one frozen ASCII LEO scenario")
    index = _require_nonbool_int(opaque_sample_index, field="opaque sample index", minimum=0)
    material = (
        b"SCB1-SOURCE-VIEW-SEED\x00"
        + str(seed).encode("ascii")
        + b"\x00"
        + str(index).encode("ascii")
        + b"\x00"
        + scenario.encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF


def _as_numpy_vector(value: Any, *, field: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or (size is not None and array.size != int(size)):
        raise _error(f"{field} must be a rank-1 vector of the required size")
    if not np.isfinite(array).all():
        raise _error(f"{field} must be finite")
    return array


def _normalize_vector(value: Any, *, field: str) -> np.ndarray:
    array = _as_numpy_vector(value, field=field)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= NORM_EPS:
        raise _error(f"{field} has zero or degenerate norm")
    return array / norm


def _quantile_higher(values: np.ndarray, q: float) -> float:
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise _error("quantile input must be non-empty finite rank-1")
    try:
        result = np.quantile(values, q, method="higher")
    except TypeError:  # pragma: no cover - retained for older local NumPy only.
        result = np.quantile(values, q, interpolation="higher")
    value = float(result)
    if not math.isfinite(value):
        raise _error("quantile result must be finite")
    return value


def _softmax(logits: Any) -> np.ndarray:
    values = _as_numpy_vector(logits, field="tx_logits", size=len(LOCAL4_HANDLES))
    shifted = values - float(np.max(values))
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if not math.isfinite(total) or total <= 0.0:
        raise _error("softmax normalization failed")
    return exp / total


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    total = float(np.sum(np.exp(values - maximum)))
    if not math.isfinite(total) or total <= 0.0:
        raise _error("logsumexp normalization failed")
    return maximum + math.log(total)


def _ensure_model_input(rows: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(rows):
        raise _error("model input must be a torch Tensor")
    if rows.dtype != torch.float32 or rows.ndim != 3 or tuple(rows.shape[1:]) != (2, INPUT_LEN):
        raise _error("model input must be contiguous float32 [B,2,256]")
    if rows.shape[0] < 1 or not bool(torch.isfinite(rows).all()) or not rows.is_contiguous():
        raise _error("model input must be non-empty, finite, and contiguous")
    return rows


def preprocess_iq(raw: Any) -> torch.Tensor:
    """Center crop/pad and RMS-normalize one raw IQ record to ``[2,256]``."""

    tensor = torch.as_tensor(raw)
    if tensor.ndim != 2:
        raise _error("raw IQ must have shape [T,2] or [2,T]")
    if tensor.shape[0] == 2:
        rows = tensor
    elif tensor.shape[1] == 2:
        rows = tensor.transpose(0, 1)
    else:
        raise _error("raw IQ must expose exactly I/Q channels")
    if not bool(torch.isfinite(rows).all()):
        raise _error("raw IQ must be finite")
    rows = rows.to(dtype=torch.float32, device="cpu")
    length = int(rows.shape[1])
    if length <= 0:
        raise _error("raw IQ must have non-empty time dimension")
    if length > INPUT_LEN:
        start = (length - INPUT_LEN) // 2
        rows = rows[:, start : start + INPUT_LEN]
    elif length < INPUT_LEN:
        left = (INPUT_LEN - length) // 2
        right = INPUT_LEN - length - left
        rows = torch.nn.functional.pad(rows, (left, right))
    scale = torch.sqrt(torch.mean(rows[0].square() + rows[1].square()) + EPS)
    if not bool(torch.isfinite(scale)) or float(scale.item()) <= 0.0:
        raise _error("RMS preprocessing scale is invalid")
    result = (rows / scale).contiguous()
    if result.dtype != torch.float32 or tuple(result.shape) != (2, INPUT_LEN):
        raise _error("preprocessing did not produce [2,256] float32")
    return result


def tensor_sha256(rows: torch.Tensor) -> str:
    checked = _ensure_model_input(rows)
    cpu = checked.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest()


def domain_descriptor(rows: torch.Tensor) -> np.ndarray:
    """Compute the frozen five-dimensional non-learned IQ descriptor."""

    if rows.ndim == 3:
        if rows.shape[0] != 1:
            raise _error("domain_descriptor accepts exactly one [2,256] record")
        rows = rows[0]
    if not torch.is_tensor(rows) or rows.dtype != torch.float32 or tuple(rows.shape) != (2, INPUT_LEN):
        raise _error("domain_descriptor requires float32 [2,256] model input")
    if not rows.is_contiguous() or not bool(torch.isfinite(rows).all()):
        raise _error("domain_descriptor input must be contiguous and finite")
    values = rows.detach().to(device="cpu", dtype=torch.float64).numpy()
    i_values = values[0]
    q_values = values[1]
    complex_values = i_values + 1j * q_values
    amplitudes = np.abs(complex_values)
    eps = float(EPS)
    mean_i2 = float(np.mean(i_values * i_values))
    mean_q2 = float(np.mean(q_values * q_values))
    stat1 = (mean_i2 - mean_q2) / (mean_i2 + mean_q2 + eps)
    stat2 = float(np.max(amplitudes)) / (math.sqrt(float(np.mean(amplitudes * amplitudes))) + eps)
    stat3 = float(np.mean(i_values * q_values)) / (math.sqrt(mean_i2 * mean_q2) + eps)
    rms = math.sqrt(float(np.mean(amplitudes * amplitudes)))
    valid = (amplitudes[1:] >= 0.1 * rms) & (amplitudes[:-1] >= 0.1 * rms)
    if int(np.sum(valid)) < 16:
        raise _error("domain descriptor has fewer than 16 valid phase increments")
    products = complex_values[1:] * np.conj(complex_values[:-1])
    circular = np.mean(np.exp(1j * np.angle(products[valid])))
    stat4 = 1.0 - float(abs(circular))
    length = int(complex_values.size)
    if length != INPUT_LEN:
        raise _error("domain descriptor FFT length drift")
    positions = np.arange(length, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * math.pi * positions / float(length))
    power = np.abs(np.fft.fft(hann * complex_values, n=length)) ** 2
    stat5 = math.exp(float(np.mean(np.log(power + eps)))) / (float(np.mean(power)) + eps)
    result = np.asarray((stat1, stat2, stat3, stat4, stat5), dtype=np.float64)
    if result.shape != (5,) or not np.isfinite(result).all():
        raise _error("domain descriptor is non-finite")
    return result


def fit_class_geometry(
    labeled_physical_views: Sequence[Mapping[str, Any]], *, class_handles: Sequence[str] = LOCAL4_HANDLES
) -> ClassGeometry:
    """Fit class centers/radii only from labelled L physicals and four views."""

    handles = tuple(_require_nfc_string(item, field="class handle") for item in class_handles)
    if handles != LOCAL4_HANDLES:
        raise _error("class handles must exactly match frozen local4 order")
    by_class: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {item: [] for item in handles}
    seen_tokens: set[str] = set()
    for row in labeled_physical_views:
        if not isinstance(row, Mapping) or set(row) != {"physical_token", "label", "z_views"}:
            raise _error("labeled physical row schema mismatch")
        token = _require_nfc_string(row["physical_token"], field="labeled physical token")
        label = _require_nfc_string(row["label"], field="labeled physical label")
        if label not in by_class or token in seen_tokens:
            raise _error("labeled physical token/label binding drift")
        seen_tokens.add(token)
        views = np.asarray(row["z_views"], dtype=np.float64)
        if views.ndim != 2 or views.shape[0] != len(VIEW_ORDER) or not np.isfinite(views).all():
            raise _error("labeled physical z_views must be finite four-view matrix")
        normalized_views = np.vstack([_normalize_vector(item, field="L z_id view") for item in views])
        physical_direction = _normalize_vector(np.sum(normalized_views, axis=0, dtype=np.float64), field="L physical direction")
        by_class[label].append((token, normalized_views, physical_direction))
    centers: list[np.ndarray] = []
    radii: list[float] = []
    counts: list[int] = []
    for handle in handles:
        rows = sorted(by_class[handle], key=lambda item: item[0])
        if len(rows) < MIN_LABELED_PHYSICALS_PER_CLASS:
            raise _error("each class requires at least 32 labelled physicals")
        center = _normalize_vector(
            np.sum([item[2] for item in rows], axis=0, dtype=np.float64), field="class center"
        )
        angles = np.asarray(
            [
                max(
                    math.acos(float(np.clip(np.dot(view, center), -1.0, 1.0)))
                    for view in item[1]
                )
                for item in rows
            ],
            dtype=np.float64,
        )
        radius = _quantile_higher(angles, 0.95)
        if radius <= 1.0e-6:
            raise _error("class radius is degenerate")
        centers.append(center)
        radii.append(radius)
        counts.append(len(rows))
    return ClassGeometry(
        class_handles=handles,
        centers=np.asarray(centers, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        class_counts=np.asarray(counts, dtype=np.int64),
    )


def score_class_geometry(z_id: Any, geometry: ClassGeometry) -> tuple[np.ndarray, float]:
    if tuple(geometry.class_handles) != LOCAL4_HANDLES:
        raise _error("geometry class handles drift")
    centers = np.asarray(geometry.centers, dtype=np.float64)
    radii = np.asarray(geometry.radii, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] != len(LOCAL4_HANDLES) or radii.shape != (len(LOCAL4_HANDLES),):
        raise _error("geometry shape drift")
    if not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii <= 1.0e-6):
        raise _error("geometry state is invalid")
    vector = _normalize_vector(z_id, field="runtime z_id")
    if centers.shape[1] != vector.size:
        raise _error("runtime z_id dimension differs from class geometry")
    center_norms = np.linalg.norm(centers, axis=1)
    if np.any(~np.isfinite(center_norms)) or np.any(center_norms <= NORM_EPS):
        raise _error("class center norm drift")
    normalized_centers = centers / center_norms[:, None]
    angles = np.arccos(np.clip(normalized_centers @ vector, -1.0, 1.0))
    distances = angles / radii
    if not np.isfinite(distances).all() or np.any(distances < 0.0):
        raise _error("class distance computation failed")
    return distances.astype(np.float64), float(np.min(distances))


def fit_descriptor_stats(opaque_descriptor_rows: Iterable[Mapping[str, Any]]) -> DescriptorStats:
    """Fit exact robust descriptor statistics from label-free streamed rows.

    The builder releases every IQ/view before the next row.  This accumulator
    retains only all five-float descriptors in a compact ``array('d')`` so the
    frozen full-population ``median`` and ``1.4826*MAD`` remain exact without
    materializing the 39,200-by-4 source IQ tensors or sample identities.
    """

    values = array("d")
    descriptor_count = 0
    for row in opaque_descriptor_rows:
        if not isinstance(row, Mapping) or set(row) != {"opaque_hash", "iq_views"}:
            raise _error("descriptor rows accept only opaque_hash and iq_views")
        _require_sha256(row["opaque_hash"], field="opaque descriptor hash")
        views = row["iq_views"]
        if not isinstance(views, Sequence) or isinstance(views, (str, bytes)) or not views:
            raise _error("descriptor row must have non-empty IQ views")
        for view_index, view in enumerate(views):
            tensor = torch.as_tensor(view)
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != (2, INPUT_LEN) or not tensor.is_contiguous():
                raise _error("descriptor IQ views must be exact model-input tensors")
            descriptor = domain_descriptor(tensor)
            descriptor_count += 1
            values.extend(float(value) for value in descriptor)
    if descriptor_count < 1 or not values:
        raise _error("descriptor fitting requires at least one opaque physical")
    if values.itemsize != np.dtype(np.float64).itemsize or len(values) != descriptor_count * 5:
        raise _error("descriptor accumulator layout drift")
    descriptors = np.frombuffer(values, dtype=np.float64).reshape(descriptor_count, 5)
    median = np.median(descriptors, axis=0)
    mad = np.median(np.abs(descriptors - median[None, :]), axis=0)
    scale = 1.4826 * mad
    if median.shape != (5,) or scale.shape != (5,) or not np.isfinite(median).all() or not np.isfinite(scale).all():
        raise _error("descriptor aggregation is non-finite")
    if np.any(scale <= NORM_EPS):
        raise _error("descriptor MAD scale is degenerate")
    return DescriptorStats(median=median.astype(np.float64), scale=scale.astype(np.float64), descriptor_count=descriptor_count)


def normalize_descriptor(raw_descriptor: Any, stats: DescriptorStats) -> tuple[np.ndarray, float]:
    raw = _as_numpy_vector(raw_descriptor, field="raw domain descriptor", size=5)
    median = _as_numpy_vector(stats.median, field="descriptor median", size=5)
    scale = _as_numpy_vector(stats.scale, field="descriptor scale", size=5)
    if np.any(scale <= NORM_EPS):
        raise _error("descriptor scale is degenerate")
    z_dom = (raw - median) / scale
    if not np.isfinite(z_dom).all():
        raise _error("normalized domain descriptor is non-finite")
    score = float(np.linalg.norm(z_dom) / math.sqrt(5.0))
    if not math.isfinite(score) or score < 0.0:
        raise _error("domain descriptor score is invalid")
    return z_dom.astype(np.float32), score


def _tail_values(scores: Sequence[float], *, field: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < MIN_CALIBRATION_PHYSICALS or not np.isfinite(values).all():
        raise _error(f"{field} requires at least {MIN_CALIBRATION_PHYSICALS} finite physical atoms")
    return values


def fit_tail_summary(
    *,
    distance_scores: Sequence[float],
    energy_scores: Sequence[float],
    domain_scores: Sequence[float],
    calibration_set_sha256: str,
) -> TailSummary:
    distance = _tail_values(distance_scores, field="distance tail")
    energy = _tail_values(energy_scores, field="energy tail")
    domain = _tail_values(domain_scores, field="domain tail")
    if not (distance.size == energy.size == domain.size):
        raise _error("tail atom arrays must have equal physical count")
    calibration_sha = _require_sha256(calibration_set_sha256, field="calibration_set_sha256")
    levels = TAIL_LEVELS.copy()
    values = []
    for name, score in (("distance", distance), ("energy", energy), ("domain", domain)):
        aggregate = np.asarray([_quantile_higher(score, float(level)) for level in levels], dtype=np.float64)
        if aggregate.shape != levels.shape or not np.isfinite(aggregate).all() or np.any(np.diff(aggregate) < 0.0):
            raise _error(f"{name} tail aggregate is invalid")
        values.append(aggregate)
    return TailSummary(
        levels=levels,
        distance_values=values[0],
        energy_values=values[1],
        domain_values=values[2],
        n_calibration=int(distance.size),
        calibration_set_sha256=calibration_sha,
    )


def _rank_from_quantile_aggregate(score: Any, *, levels: Any, values: Any, n_calibration: Any) -> float:
    current = _require_finite_float(score, field="tail score")
    q = _as_numpy_vector(levels, field="tail levels", size=129)
    v = _as_numpy_vector(values, field="tail values", size=129)
    count = _require_nonbool_int(n_calibration, field="n_calibration", minimum=MIN_CALIBRATION_PHYSICALS)
    if not np.array_equal(q, TAIL_LEVELS) or np.any(np.diff(q) <= 0.0) or np.any(np.diff(v) < 0.0):
        raise _error("tail level/value monotonicity drift")
    floor = max(TAIL_FLOOR_MIN, 1.0 / float(count + 1))
    if current <= float(v[0]):
        cdf_lt = 0.0
    elif current > float(v[-1]):
        return float(floor)
    else:
        index = int(np.searchsorted(v, current, side="left"))
        if index < 0 or index >= v.size:
            raise _error("tail search index drift")
        if float(v[index]) == current:
            cdf_lt = float(q[index - 1]) if index > 0 else 0.0
        else:
            if index == 0 or not float(v[index]) > float(v[index - 1]):
                raise _error("tail interpolation enters plateau/non-monotonic branch")
            cdf_lt = float(q[index - 1]) + (
                float(q[index] - q[index - 1])
                * (current - float(v[index - 1]))
                / float(v[index] - v[index - 1])
            )
    result = (1.0 + float(count) * (1.0 - cdf_lt)) / float(count + 1)
    result = float(np.clip(max(floor, result), floor, 1.0))
    if not math.isfinite(result):
        raise _error("tail rank is non-finite")
    return result


def tail_rank(summary: TailSummary, kind: str, score: Any) -> float:
    mapping = {
        "distance": summary.distance_values,
        "energy": summary.energy_values,
        "domain": summary.domain_values,
    }
    if kind not in mapping:
        raise _error("unknown tail kind")
    return _rank_from_quantile_aggregate(
        score,
        levels=summary.levels,
        values=mapping[kind],
        n_calibration=summary.n_calibration,
    )


def evaluate_local_fields(
    *,
    z_id: Any,
    tx_logits: Any,
    model_input: torch.Tensor,
    state: BundleState,
) -> LocalFields:
    """Evaluate the six frozen local fields for one already-preprocessed IQ row."""

    vector = _as_numpy_vector(z_id, field="z_id")
    logits = _as_numpy_vector(tx_logits, field="tx_logits", size=len(LOCAL4_HANDLES))
    if vector.size < 2:
        raise _error("z_id dimension must be at least two")
    if model_input.ndim == 3:
        if model_input.shape[0] != 1:
            raise _error("single local-field evaluation requires one IQ row")
        model_input = model_input[0]
    if not torch.is_tensor(model_input) or model_input.dtype != torch.float32 or tuple(model_input.shape) != (2, INPUT_LEN):
        raise _error("local-field evaluation requires exact [2,256] float32 input")
    if not model_input.is_contiguous() or not bool(torch.isfinite(model_input).all()):
        raise _error("local-field IQ must be finite and contiguous")
    d_class, distance_score = score_class_geometry(vector, state.geometry)
    p_reg = _softmax(logits)
    energy = -_logsumexp(logits)
    descriptor = domain_descriptor(model_input)
    z_dom, domain_score = normalize_descriptor(descriptor, state.descriptor)
    r_distance = tail_rank(state.tail, "distance", distance_score)
    r_energy = tail_rank(state.tail, "energy", energy)
    r_domain = tail_rank(state.tail, "domain", domain_score)
    known_consistency = max(r_distance, r_energy)
    e_unknown = 1.0 - known_consistency
    u = float(np.clip((ALPHA - known_consistency) / ALPHA, 0.0, 1.0))
    p_local = np.concatenate(((1.0 - u) * p_reg, np.asarray([u], dtype=np.float64)))
    if p_local.shape != (len(LOCAL4_HANDLES) + 1,) or not np.isfinite(p_local).all() or np.any(p_local < 0.0):
        raise _error("local probability construction failed")
    total = float(np.sum(p_local))
    if abs(total - 1.0) > 1.0e-12:
        raise _error("local probability must be a strict simplex before CARE sealing")
    maximum_registered = float(np.max(p_reg))
    if u >= (1.0 - u) * maximum_registered:
        decision = "unknown"
        label = None
        reason = "SCB_TECHNICAL_UNKNOWN"
    else:
        decision = "registered"
        label = LOCAL4_HANDLES[int(np.argmax(p_reg))]
        reason = "SCB_REGISTERED"
    entropy = -float(np.sum(np.where(p_reg > 0.0, p_reg * np.log(p_reg), 0.0)))
    entropy_confidence = float(np.clip(1.0 - entropy / math.log(len(LOCAL4_HANDLES)), 0.0, 1.0))
    quality = math.sqrt(float(np.clip(r_domain, 0.0, 1.0)) * entropy_confidence)
    if not all(math.isfinite(item) for item in (quality, energy, distance_score, domain_score, known_consistency, e_unknown)):
        raise _error("local field evaluation yielded non-finite scalar")
    return LocalFields(
        z_id=vector.astype(np.float32),
        z_dom=z_dom.astype(np.float32),
        q=float(quality),
        d_class=d_class.astype(np.float32),
        e_unknown=float(e_unknown),
        p_local=p_local.astype(np.float64),
        local_decision=decision,
        local_label=label,
        reason_code=reason,
        known_consistency=float(known_consistency),
        energy=float(energy),
        distance_score=float(distance_score),
        domain_score=float(domain_score),
    )


class SingleControlIdentityRuntime(nn.Module):
    """TorchScript payload: only F1C identity branch and local4 logits."""

    def __init__(self, model: nn.Module, *, runtime_batch_size: int = 1) -> None:
        super().__init__()
        if int(runtime_batch_size) < 1:
            raise ValueError("runtime_batch_size must be positive")
        self.model = model
        self.register_buffer("runtime_capacity_token", torch.tensor([int(runtime_batch_size)], dtype=torch.int64))

    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = rows.size(0)
        capacity = int(self.runtime_capacity_token.item())
        if count != capacity:
            raise RuntimeError("single-control TorchScript runtime has fixed batch-1 capacity")
        padded = rows.new_zeros((capacity, rows.size(1), rows.size(2)))
        padded[:count].copy_(rows)
        result = identity_only_feature_forward(self.model, padded, "z_id")
        if result is None:
            raise RuntimeError("checkpoint has no identity-only z_id export")
        z_id, logits = result
        return z_id[:count], logits[:count]


class DirectIdentityReference(nn.Module):
    """Direct raw-model identity path used solely for parity evidence."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = identity_only_feature_forward(self.model, rows, "z_id")
        if result is None:
            raise RuntimeError("checkpoint has no identity-only z_id export")
        return result


def _runtime_outputs(runtime: Any, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = runtime(rows)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise _error("identity runtime must return (z_id, tx_logits)")
    z_id, logits = result
    if not torch.is_tensor(z_id) or not torch.is_tensor(logits):
        raise _error("identity runtime outputs must be tensors")
    if z_id.ndim != 2 or logits.ndim != 2 or z_id.shape[0] != rows.shape[0] or tuple(logits.shape) != (rows.shape[0], len(LOCAL4_HANDLES)):
        raise _error("identity runtime output shape drift")
    if not bool(torch.isfinite(z_id).all()) or not bool(torch.isfinite(logits).all()):
        raise _error("identity runtime outputs must be finite")
    return z_id.float(), logits.float()


def build_torchscript_runtime(
    *,
    checkpoint: Mapping[str, Any],
    device: torch.device,
    runtime_path: str | Path,
    runtime_batch_size: int = 1,
) -> tuple[torch.jit.ScriptModule, nn.Module, dict[str, Any]]:
    """Strictly rebuild F1C and write a TorchScript ``.ts`` identity runtime."""

    target = Path(runtime_path)
    if target.exists():
        raise FileExistsError("refusing to overwrite TorchScript runtime")
    model, audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=INPUT_LEN, device=device
    )
    model.to(device).eval()
    if int(runtime_batch_size) != 1:
        raise _error("single-control deployment runtime capacity is frozen at batch1")
    reference = DirectIdentityReference(model).to(device).eval()
    wrapper = SingleControlIdentityRuntime(model, runtime_batch_size=runtime_batch_size).to(device).eval()
    example = torch.zeros((1, 2, INPUT_LEN), dtype=torch.float32, device=device)
    with torch.no_grad():
        direct_z, direct_logits = _runtime_outputs(reference, example)
        _, logits = _runtime_outputs(wrapper, example)
    if logits.shape[1] != len(LOCAL4_HANDLES):
        raise _error("strict checkpoint runtime does not expose local4 logits")
    if not torch.equal(direct_z, _runtime_outputs(wrapper, example)[0]) or not torch.equal(direct_logits, logits):
        raise _error("raw-model identity path differs from batch1 runtime adapter")
    target.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
    torch.jit.save(traced, str(target))
    runtime = torch.jit.load(str(target), map_location=device).eval()
    with torch.no_grad():
        traced_z, traced_logits = _runtime_outputs(runtime, example)
    if not torch.equal(direct_z, traced_z) or not torch.equal(direct_logits, traced_logits):
        raise _error("raw-model identity path differs from exported TorchScript batch1 runtime")
    audit = dict(audit)
    audit.update({"runtime_batch_capacity": 1, "runtime_internal_padding": False, "raw_model_direct_identity_parity": True})
    return runtime, reference, audit


def runtime_state_schema(runtime: torch.jit.ScriptModule) -> dict[str, Any]:
    """Inspect a loadable TorchScript payload without retaining sample data."""

    rows: list[dict[str, Any]] = []
    state_bytes = 0
    for kind, entries in (("parameter", runtime.named_parameters()), ("buffer", runtime.named_buffers())):
        for name, tensor in entries:
            if not torch.is_tensor(tensor):
                raise _error("TorchScript state entry is not a tensor")
            token = _require_nfc_string(str(name), field="runtime state name")
            size = int(tensor.numel()) * int(tensor.element_size())
            state_bytes += size
            rows.append(
                {
                    "kind": kind,
                    "name": token,
                    "dtype": str(tensor.dtype),
                    "shape": [int(item) for item in tensor.shape],
                    "size_bytes": size,
                }
            )
    rows = sorted(rows, key=lambda item: (item["kind"], item["name"]))
    return {
        "runtime_state_schema_sha256": canonical_sha256(rows),
        "runtime_state_bytes": int(state_bytes),
        "runtime_state_tensor_count": int(len(rows)),
    }


def _npz_bytes(arrays: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    prepared: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise _error("NPZ state key must be a non-empty string")
        array = np.asarray(value)
        if array.dtype == object:
            raise _error("NPZ object arrays are forbidden")
        prepared[name] = array
    np.savez(stream, **prepared)
    return stream.getvalue()


def _load_npz_bytes(raw: bytes, *, expected_keys: set[str], context: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != expected_keys:
                raise _error(f"{context} NPZ key allowlist drift")
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except SingleControlBundleError:
        raise
    except Exception as exc:
        raise _error(f"{context} NPZ is unreadable") from exc
    if any(array.dtype == object for array in arrays.values()):
        raise _error(f"{context} NPZ object array is forbidden")
    return arrays


def geometry_npz_arrays(geometry: ClassGeometry) -> dict[str, np.ndarray]:
    return {
        "class_handles": np.asarray(geometry.class_handles, dtype="<U32"),
        "centers": np.asarray(geometry.centers, dtype=np.float64),
        "radii": np.asarray(geometry.radii, dtype=np.float64),
        "class_counts": np.asarray(geometry.class_counts, dtype=np.int64),
    }


def descriptor_npz_arrays(descriptor: DescriptorStats) -> dict[str, np.ndarray]:
    return {
        "median": np.asarray(descriptor.median, dtype=np.float64),
        "scale": np.asarray(descriptor.scale, dtype=np.float64),
        "descriptor_count": np.asarray([descriptor.descriptor_count], dtype=np.int64),
    }


def tail_npz_arrays(tail: TailSummary) -> dict[str, np.ndarray]:
    return {
        "levels": np.asarray(tail.levels, dtype=np.float64),
        "distance_values": np.asarray(tail.distance_values, dtype=np.float64),
        "energy_values": np.asarray(tail.energy_values, dtype=np.float64),
        "domain_values": np.asarray(tail.domain_values, dtype=np.float64),
        "n_calibration": np.asarray([tail.n_calibration], dtype=np.int64),
        "calibration_set_sha256": np.asarray([tail.calibration_set_sha256], dtype="<U64"),
    }


def state_from_npz_bytes(
    geometry_raw: bytes, descriptor_raw: bytes, tail_raw: bytes
) -> BundleState:
    geometry_arrays = _load_npz_bytes(
        geometry_raw,
        expected_keys={"class_handles", "centers", "radii", "class_counts"},
        context="class geometry",
    )
    handles = tuple(str(value) for value in geometry_arrays["class_handles"].tolist())
    geometry = ClassGeometry(
        class_handles=handles,
        centers=np.asarray(geometry_arrays["centers"], dtype=np.float64),
        radii=np.asarray(geometry_arrays["radii"], dtype=np.float64),
        class_counts=np.asarray(geometry_arrays["class_counts"], dtype=np.int64),
    )
    if handles != LOCAL4_HANDLES or geometry.centers.ndim != 2 or geometry.centers.shape[0] != 4 or geometry.radii.shape != (4,) or geometry.class_counts.shape != (4,):
        raise _error("class geometry state shape/handles drift")
    if not np.isfinite(geometry.centers).all() or not np.isfinite(geometry.radii).all() or np.any(geometry.radii <= 1.0e-6) or np.any(geometry.class_counts < MIN_LABELED_PHYSICALS_PER_CLASS):
        raise _error("class geometry state is invalid")
    descriptor_arrays = _load_npz_bytes(
        descriptor_raw,
        expected_keys={"median", "scale", "descriptor_count"},
        context="descriptor stats",
    )
    count_arr = np.asarray(descriptor_arrays["descriptor_count"], dtype=np.int64)
    if count_arr.shape != (1,):
        raise _error("descriptor count shape drift")
    descriptor = DescriptorStats(
        median=np.asarray(descriptor_arrays["median"], dtype=np.float64),
        scale=np.asarray(descriptor_arrays["scale"], dtype=np.float64),
        descriptor_count=int(count_arr[0]),
    )
    if descriptor.median.shape != (5,) or descriptor.scale.shape != (5,) or descriptor.descriptor_count < 1 or not np.isfinite(descriptor.median).all() or not np.isfinite(descriptor.scale).all() or np.any(descriptor.scale <= NORM_EPS):
        raise _error("descriptor state is invalid")
    tail_arrays = _load_npz_bytes(
        tail_raw,
        expected_keys={"levels", "distance_values", "energy_values", "domain_values", "n_calibration", "calibration_set_sha256"},
        context="tail summary",
    )
    n_arr = np.asarray(tail_arrays["n_calibration"], dtype=np.int64)
    sha_arr = np.asarray(tail_arrays["calibration_set_sha256"])
    if n_arr.shape != (1,) or sha_arr.shape != (1,):
        raise _error("tail count/SHA shape drift")
    tail = TailSummary(
        levels=np.asarray(tail_arrays["levels"], dtype=np.float64),
        distance_values=np.asarray(tail_arrays["distance_values"], dtype=np.float64),
        energy_values=np.asarray(tail_arrays["energy_values"], dtype=np.float64),
        domain_values=np.asarray(tail_arrays["domain_values"], dtype=np.float64),
        n_calibration=int(n_arr[0]),
        calibration_set_sha256=_require_sha256(str(sha_arr[0]), field="tail calibration SHA"),
    )
    for kind, values in (("distance", tail.distance_values), ("energy", tail.energy_values), ("domain", tail.domain_values)):
        _rank_from_quantile_aggregate(0.0, levels=tail.levels, values=values, n_calibration=tail.n_calibration)
    return BundleState(geometry=geometry, descriptor=descriptor, tail=tail)


MODEL_CONFIG_SPEC: dict[str, tuple[type, Any]] = {
    "model_size": (str, "M"),
    "dataset": (str, "wisig"),
    "sample_rate_hz": (float, 25000000.0),
    "id_feature_key": (str, "feat_joint"),
    "dom_feature_key": (str, "feat_imp"),
    "model_variant": (str, "lite_d"),
    "branch_ablation": (str, "no_dac"),
    "use_mixstyle": (bool, True),
    "mixstyle_p": (float, 0.18),
    "mixstyle_alpha": (float, 0.10),
    "mixstyle_eps": (float, 1.0e-6),
    "mixstyle_layers": (str, "time_down,t1"),
    "mixstyle_use_domain_label": (bool, True),
    "mixstyle_mix": (str, "same_tx_crossdomain"),
    "mixstyle_strength": (float, 0.70),
    "mixstyle_fallback": (str, "skip"),
    "domain_branch_ablation": (str, "no_stats"),
    "domain_enhancer": (str, "rcn_stats"),
    "domain_enhancer_strength": (float, 0.35),
    "id_time_stability_mode": (str, "off"),
    "id_freq_stability_mode": (str, "off"),
    "domain_time_stability_mode": (str, "off"),
    "domain_freq_stability_mode": (str, "off"),
    "time_stability_channels": (int, 8),
    "freq_stability_channels": (int, 4),
    "fast_infer_when_no_aux": (bool, True),
    "arch_family": (str, "cvsincnet"),
    "representation_mode": (str, "dual"),
}


# ``build_baseline_model`` reads these nine attributes through ``getattr`` but
# the frozen F1C training parser does not declare them.  They are therefore
# permitted *only* when both the checkpoint args and the final reconstructed
# namespace prove the attribute was absent.  This is deliberately narrower
# than MODEL_CONFIG_SPEC: parser-backed keys must remain present in the final
# namespace, and explicit ``None`` remains a configuration error.
MODEL_NAMESPACE_ABSENT_DEFAULTS: dict[str, Any] = {
    "dom_feature_key": "feat_imp",
    "id_time_stability_mode": "off",
    "id_freq_stability_mode": "off",
    "domain_time_stability_mode": "off",
    "domain_freq_stability_mode": "off",
    "time_stability_channels": 8,
    "freq_stability_channels": 4,
    "fast_infer_when_no_aux": True,
    "arch_family": "cvsincnet",
}
_MISSING = object()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_regular_json(path: str | Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise _error(f"{context} must be a regular JSON file")
    try:
        raw = target.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(f"{context} is unreadable JSON") from exc
    if not isinstance(value, Mapping):
        raise _error(f"{context} JSON root must be a mapping")
    return dict(value), raw


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise _error(f"{field} mapping keys must be strings")
    return dict(value)


def _require_exact_bool(value: Any, *, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise _error(f"{field} must be a boolean")
    if expected is not None and value is not expected:
        raise _error(f"{field} must be {expected}")
    return value


def _projection(source: Mapping[str, Any], fields: Sequence[str], *, context: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        if field not in source:
            raise _error(f"{context} is missing required field {field}")
        result[field] = source[field]
    return result


SOURCE_SPLIT_FIELDS = (
    "schema",
    "seed",
    "split_mode",
    "source_days",
    "target_days",
    "source_receivers",
    "target_receivers",
    "source_target_receiver_overlap_count",
    "labeled_indices_sha256",
    "unlabeled_indices_sha256",
    "source_validation_indices_sha256",
    "split_manifest_sha256",
    "labeled_size",
    "unlabeled_size",
    "source_validation_size",
    "source_pool_size",
    "requested_labeled_ratio",
    "requested_unlabeled_ratio",
    "requested_source_val_ratio",
    "requested_rho_label",
    "realized_rho_label",
    "realized_rho_tolerance",
    "realized_rho_within_tolerance",
    "realized_source_val_fraction",
    "realized_source_val_tolerance",
    "realized_source_val_within_tolerance",
)
COMPLETION_FIELDS = (
    "run_id",
    "phase1_training_complete",
    "terminal_status",
    "exit_code",
    "technical_only",
    "formal_performance_claim",
    "selected_checkpoint_sha256",
)
TERMINAL_FIELDS = (
    "run_id",
    "candidate_id",
    "status",
    "exit_code",
    "selection_source",
    "selected_checkpoint",
    "selected_checkpoint_exists",
    "selected_checkpoint_sha256",
    "technical_only",
    "promotion_ready",
    "performance_result_available",
)
CP_FIELDS = (
    "enabled",
    "checkpoint_role",
    "source_train_tx",
    "source_known_validation_tx",
    "source_proxy_unknown_tx",
    "local_tx_class_order",
    "checkpoint_train_tx_class_order",
    "local_to_head_class_ids",
    "checkpoint_head_class_count",
    "live_head_class_count",
    "class_order_binding_sha256",
    "class_order_contract",
    "selected_checkpoint",
    "selected_checkpoint_sha256",
    "terminal_status",
    "terminal_exit_code",
    "technical_only",
    "promotion_ready",
    "performance_result_available",
)
SATELLITE_PROTOCOL_FIELDS = (
    "schema",
    "train_scenarios",
    "eval_scenarios",
    "train_families",
    "eval_families",
    "scenario_config_sha256",
    "train_config_sha256",
    "eval_config_sha256",
    "channel_implementation",
    "registry_version",
    "registry_sha256",
    "disjoint",
    "require_disjoint",
    "evaluation_claim",
)


def _strict_tx_list(value: Any, *, field: str, expected: Sequence[str]) -> list[str]:
    if not isinstance(value, list) or [str(item) for item in value] != list(expected):
        raise _error(f"{field} must exactly match frozen TX order")
    if any(not isinstance(item, str) for item in value):
        raise _error(f"{field} must contain literal string TX identifiers")
    return list(value)


def _strict_singleton_tx_receipt(value: Any, *, field: str, expected: str) -> str:
    """Parse the byte-locked F1C receipt's one-element TX array exactly.

    The frozen CP receipt itself serializes these two roles as literal arrays;
    accepting a scalar would reject its locked byte SHA.  The returned binding
    is the sole scalar identity, but no alternative collection form is allowed.
    """

    if not isinstance(value, list) or len(value) != 1:
        raise _error(f"{field} must be the frozen one-element TX array")
    literal = _require_nfc_string(value[0], field=field)
    if literal != expected:
        raise _error(f"{field} must exactly match frozen TX identifier")
    return literal


def _strict_string_list(value: Any, *, field: str, expected: Sequence[str]) -> list[str]:
    if not isinstance(value, list) or value != list(expected):
        raise _error(f"{field} must exactly match frozen string order")
    if any(not isinstance(item, str) or unicodedata.normalize("NFC", item) != item or not item for item in value):
        raise _error(f"{field} must contain literal NFC strings")
    return list(value)


def _recompute_satellite_protocol_projection() -> dict[str, Any]:
    """Recompute the live scenario/registry closure from the owned protocol code."""

    from training_controls import satellite_protocol_manifest

    live = _require_mapping(
        satellite_protocol_manifest(list(SCENARIOS), list(SCENARIOS), require_disjoint=False),
        field="live satellite protocol manifest",
    )
    return _projection(live, SATELLITE_PROTOCOL_FIELDS, context="live satellite protocol")


def _validate_source_split(value: Any, *, fixture_mode: bool) -> dict[str, Any]:
    receipt = _require_mapping(value, field="source_split_receipt")
    projected = _projection(receipt, SOURCE_SPLIT_FIELDS, context="source split receipt")
    if projected["schema"] != "cvs.phase1.source_split_receipt.v1":
        raise _error("source split receipt schema drift")
    if _require_nonbool_int(projected["seed"], field="source split seed", minimum=0) != 7281105:
        raise _error("source split seed drift")
    if projected["split_mode"] != "tx_rx_day_1_6_3":
        raise _error("source split mode drift")
    if _require_nonbool_int(projected["source_target_receiver_overlap_count"], field="source target receiver overlap", minimum=0) != 0:
        raise _error("source/target receivers must be disjoint")
    for field in (
        "labeled_indices_sha256",
        "unlabeled_indices_sha256",
        "source_validation_indices_sha256",
        "split_manifest_sha256",
    ):
        projected[field] = _require_sha256(projected[field], field=field)
    for field in ("labeled_size", "unlabeled_size", "source_validation_size", "source_pool_size"):
        projected[field] = _require_nonbool_int(projected[field], field=field, minimum=1)
    if projected["labeled_size"] + projected["unlabeled_size"] + projected["source_validation_size"] != projected["source_pool_size"]:
        raise _error("source split sizes do not close source pool")
    for field in (
        "requested_labeled_ratio",
        "requested_unlabeled_ratio",
        "requested_source_val_ratio",
        "requested_rho_label",
        "realized_rho_label",
        "realized_rho_tolerance",
        "realized_source_val_fraction",
        "realized_source_val_tolerance",
    ):
        projected[field] = _require_finite_float(projected[field], field=field)
    _require_exact_bool(projected["realized_rho_within_tolerance"], field="realized_rho_within_tolerance", expected=True)
    _require_exact_bool(projected["realized_source_val_within_tolerance"], field="realized_source_val_within_tolerance", expected=True)
    if not fixture_mode:
        _strict_string_list(projected["source_days"], field="source_days", expected=("0", "1"))
        _strict_string_list(projected["target_days"], field="target_days", expected=("2", "3"))
        _strict_string_list(
            projected["source_receivers"],
            field="source_receivers",
            expected=("0", "1", "2", "3", "4", "5", "6"),
        )
        _strict_string_list(
            projected["target_receivers"],
            field="target_receivers",
            expected=("10", "11", "7", "8", "9"),
        )
        expected_sizes = {
            "labeled_size": 3920,
            "unlabeled_size": 35280,
            "source_validation_size": 16800,
            "source_pool_size": 56000,
        }
        for field, expected in expected_sizes.items():
            if projected[field] != expected:
                raise _error(f"real F1C source split {field} drift")
        expected_ratios = {
            "requested_labeled_ratio": 0.07,
            "requested_unlabeled_ratio": 0.63,
            "requested_source_val_ratio": 0.30,
            "requested_rho_label": 0.10,
        }
        for field, expected in expected_ratios.items():
            if not math.isclose(float(projected[field]), expected, rel_tol=0.0, abs_tol=1.0e-15):
                raise _error(f"real F1C source split {field} drift")
    return projected


def validate_f1c_receipts(
    *,
    completion_path: str | Path,
    terminal_path: str | Path,
    cp_terminal_path: str | Path,
    checkpoint_sha256: str,
    dataset_sha256: str,
    fixture_mode: bool = False,
) -> dict[str, Any]:
    """Validate the three frozen receipts and return their allowed projection.

    Fixture mode is deliberately segregated: it validates schemas and internal
    closure but never accepts a fixture as the real F1C control.
    """

    completion, completion_raw = _load_regular_json(completion_path, context="training completion receipt")
    terminal, terminal_raw = _load_regular_json(terminal_path, context="terminal status receipt")
    cp, cp_raw = _load_regular_json(cp_terminal_path, context="CP terminal receipt")
    checkpoint_sha = _require_sha256(checkpoint_sha256, field="checkpoint_sha256")
    dataset_sha = _require_sha256(dataset_sha256, field="dataset_sha256")
    if not fixture_mode:
        if _sha256_bytes(completion_raw) != EXPECTED_TRAINING_COMPLETION_SHA256:
            raise _error("training completion receipt byte SHA drift")
        if _sha256_bytes(terminal_raw) != EXPECTED_TERMINAL_STATUS_SHA256:
            raise _error("terminal status receipt byte SHA drift")
        if _sha256_bytes(cp_raw) != EXPECTED_CP_TERMINAL_SHA256:
            raise _error("CP terminal receipt byte SHA drift")
        if checkpoint_sha != F1C_CHECKPOINT_SHA256 or dataset_sha != EXPECTED_DATASET_SHA256:
            raise _error("real F1C checkpoint/dataset SHA drift")
    if completion.get("schema") != "cvs.phase1.training_completion_receipt.v1":
        raise _error("training completion schema drift")
    if terminal.get("schema") != "phase1_terminal_status_v2":
        raise _error("terminal status schema drift")
    if cp.get("schema") != "cvs.phase1.cp_sfce_receipt.v2":
        raise _error("CP terminal schema drift")
    completion_projection = _projection(completion, COMPLETION_FIELDS, context="training completion")
    completion_projection["source_split_receipt"] = _validate_source_split(
        completion.get("source_split_receipt"), fixture_mode=fixture_mode
    )
    terminal_projection = _projection(terminal, TERMINAL_FIELDS, context="terminal status")
    terminal_split = _validate_source_split(terminal.get("source_split_receipt"), fixture_mode=fixture_mode)
    if terminal_split != completion_projection["source_split_receipt"]:
        raise _error("terminal source split projection differs from completion")
    satellite = _projection(
        _require_mapping(terminal.get("satellite_protocol"), field="satellite_protocol"),
        SATELLITE_PROTOCOL_FIELDS,
        context="satellite protocol",
    )
    if satellite["schema"] != "phase1_satellite_train_eval_protocol_v1":
        raise _error("satellite protocol schema drift")
    if _strict_string_list(satellite["train_scenarios"], field="train_scenarios", expected=SCENARIOS) != list(SCENARIOS) or _strict_string_list(
        satellite["eval_scenarios"], field="eval_scenarios", expected=SCENARIOS
    ) != list(SCENARIOS):
        raise _error("satellite scenario order drift")
    _strict_string_list(
        satellite["train_families"],
        field="train_families",
        expected=("simplified_leo_residual_weak_v1",),
    )
    _strict_string_list(
        satellite["eval_families"],
        field="eval_families",
        expected=("simplified_leo_residual_weak_v1",),
    )
    scenario_sha = _require_mapping(satellite["scenario_config_sha256"], field="scenario_config_sha256")
    if set(scenario_sha) != set(SCENARIOS):
        raise _error("scenario config SHA mapping keys drift")
    for scenario in SCENARIOS:
        scenario_sha[scenario] = _require_sha256(scenario_sha[scenario], field=f"scenario config SHA {scenario}")
        if not fixture_mode and scenario_sha[scenario] != EXPECTED_SCENARIO_CONFIG_SHA256[scenario]:
            raise _error(f"scenario config SHA drift for {scenario}")
    satellite["scenario_config_sha256"] = {scenario: scenario_sha[scenario] for scenario in SCENARIOS}
    for field in ("train_config_sha256", "eval_config_sha256"):
        aggregate = _strict_string_list(
            satellite[field], field=field, expected=EXPECTED_TRAIN_EVAL_CONFIG_SHA256
        )
        for digest in aggregate:
            _require_sha256(digest, field=field)
        satellite[field] = aggregate
    channel_implementation = _require_mapping(satellite["channel_implementation"], field="channel_implementation")
    if channel_implementation != {scenario: "leo_residual" for scenario in SCENARIOS}:
        raise _error("satellite channel implementation drift")
    satellite["channel_implementation"] = dict(channel_implementation)
    if satellite["registry_version"] != "phase1_satellite_protocol_registry_v1":
        raise _error("satellite registry version drift")
    satellite["registry_sha256"] = _require_sha256(satellite["registry_sha256"], field="registry_sha256")
    if not fixture_mode and satellite["registry_sha256"] != EXPECTED_SCENARIO_REGISTRY_SHA256:
        raise _error("satellite registry SHA drift")
    if _require_exact_bool(satellite["disjoint"], field="satellite disjoint", expected=False) is not False or _require_exact_bool(
        satellite["require_disjoint"], field="satellite require_disjoint", expected=False
    ) is not False:
        raise _error("satellite protocol disjointness drift")
    if satellite["evaluation_claim"] != "SOURCE_SYNTHETIC_HELDOUT_CHANNEL_STRESS_NOT_REAL_SATELLITE":
        raise _error("satellite evaluation claim drift")
    if not fixture_mode and satellite != _recompute_satellite_protocol_projection():
        raise _error("live satellite protocol manifest differs from the frozen receipt projection")
    cp_projection = _projection(cp, CP_FIELDS, context="CP terminal")
    checkpoint_fields = (
        completion_projection["selected_checkpoint_sha256"],
        terminal_projection["selected_checkpoint_sha256"],
        cp_projection["selected_checkpoint_sha256"],
    )
    if any(_require_sha256(value, field="receipt checkpoint SHA") != checkpoint_sha for value in checkpoint_fields):
        raise _error("receipt/checkpoint SHA closure failed")
    if not fixture_mode:
        if completion_projection["run_id"] != F1C_RUN_ID or terminal_projection["run_id"] != F1C_RUN_ID:
            raise _error("F1C run ID drift")
        if terminal_projection["candidate_id"] != F1C_CANDIDATE_ID:
            raise _error("F1C candidate ID drift")
        selected = str(terminal_projection["selected_checkpoint"])
        if not selected.endswith(f"/{F1C_CANDIDATE_ID}/{F1C_CHECKPOINT_LEAF}"):
            raise _error("F1C selected checkpoint path drift")
    if terminal_projection["status"] != "NON_PROMOTABLE_P0_DISABLED" or cp_projection["terminal_status"] != "NON_PROMOTABLE_P0_DISABLED":
        raise _error("F1C must remain NON_PROMOTABLE_P0_DISABLED")
    if completion_projection["terminal_status"] != "NON_PROMOTABLE_P0_DISABLED":
        raise _error("F1C completion terminal status drift")
    if _require_nonbool_int(completion_projection["exit_code"], field="completion exit code") != 8:
        raise _error("F1C completion exit code drift")
    _require_exact_bool(completion_projection["phase1_training_complete"], field="phase1_training_complete", expected=False)
    _require_exact_bool(completion_projection["technical_only"], field="completion technical_only", expected=False)
    _require_exact_bool(completion_projection["formal_performance_claim"], field="formal_performance_claim", expected=False)
    _require_exact_bool(terminal_projection["selected_checkpoint_exists"], field="selected_checkpoint_exists", expected=True)
    _require_exact_bool(terminal_projection["technical_only"], field="terminal technical_only", expected=False)
    _require_exact_bool(terminal_projection["promotion_ready"], field="terminal promotion_ready", expected=False)
    _require_exact_bool(terminal_projection["performance_result_available"], field="terminal performance_result_available", expected=False)
    if _require_nonbool_int(terminal_projection["exit_code"], field="terminal exit code") != 8 or _require_nonbool_int(cp_projection["terminal_exit_code"], field="CP terminal exit code") != 8:
        raise _error("F1C terminal exit code drift")
    if _require_exact_bool(cp_projection["enabled"], field="CP enabled", expected=False) is not False:
        raise _error("F1C CP control must be disabled")
    _require_exact_bool(cp_projection["technical_only"], field="CP technical_only", expected=False)
    _require_exact_bool(cp_projection["promotion_ready"], field="CP promotion_ready", expected=False)
    _require_exact_bool(cp_projection["performance_result_available"], field="CP performance_result_available", expected=False)
    if cp_projection["checkpoint_role"] != "training_final_only":
        raise _error("F1C checkpoint role drift")
    _strict_tx_list(cp_projection["source_train_tx"], field="source_train_tx", expected=LOCAL4_HANDLES)
    _strict_tx_list(cp_projection["local_tx_class_order"], field="local_tx_class_order", expected=LOCAL4_HANDLES)
    _strict_tx_list(cp_projection["checkpoint_train_tx_class_order"], field="checkpoint_train_tx_class_order", expected=LOCAL4_HANDLES)
    cp_projection["source_known_validation_tx"] = _strict_singleton_tx_receipt(
        cp_projection["source_known_validation_tx"],
        field="source_known_validation_tx",
        expected=KNOWN_VALIDATION_TX,
    )
    cp_projection["source_proxy_unknown_tx"] = _strict_singleton_tx_receipt(
        cp_projection["source_proxy_unknown_tx"],
        field="source_proxy_unknown_tx",
        expected=PROXY_UNKNOWN_TX,
    )
    if list(cp_projection["local_to_head_class_ids"]) != [0, 1, 2, 3]:
        raise _error("local-to-head binding drift")
    if _require_nonbool_int(cp_projection["checkpoint_head_class_count"], field="checkpoint_head_class_count") != 4 or _require_nonbool_int(cp_projection["live_head_class_count"], field="live_head_class_count") != 4:
        raise _error("F1C head must have exactly four rows")
    _require_sha256(cp_projection["class_order_binding_sha256"], field="class_order_binding_sha256")
    if cp_projection["class_order_contract"] != "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER":
        raise _error("F1C class order contract drift")
    for label, receipt in (("completion", completion_projection), ("terminal", terminal_projection), ("CP", cp_projection)):
        for field in ("promotion_ready", "performance_result_available"):
            if field in receipt and _require_exact_bool(receipt[field], field=f"{label}.{field}", expected=False) is not False:
                raise _error("F1C control receipt claims promotion/performance")
    return {
        "schema": "cvs.phase1.single_control_input_receipts.v1",
        "fixture_mode": bool(fixture_mode),
        "completion_receipt_sha256": _sha256_bytes(completion_raw),
        "terminal_receipt_sha256": _sha256_bytes(terminal_raw),
        "cp_terminal_receipt_sha256": _sha256_bytes(cp_raw),
        "completion": completion_projection,
        "terminal": terminal_projection,
        "cp_terminal": cp_projection,
        "satellite_protocol": satellite,
        "dataset_sha256": dataset_sha,
        "checkpoint_sha256": checkpoint_sha,
    }


def _namespace_value(namespace: Any, key: str) -> Any:
    """Return a presence sentinel instead of conflating absent and ``None``."""

    if isinstance(namespace, Mapping):
        return namespace[key] if key in namespace else _MISSING
    return getattr(namespace, key, _MISSING)


def _typed_config_value(value: Any, *, expected_type: type, field: str) -> Any:
    if expected_type is bool:
        if not isinstance(value, bool):
            raise _error(f"model config {field} must be bool")
        return value
    if expected_type is int:
        return _require_nonbool_int(value, field=f"model config {field}")
    if expected_type is float:
        return _require_finite_float(value, field=f"model config {field}")
    if expected_type is str:
        return _require_nfc_string(value, field=f"model config {field}")
    raise AssertionError("unsupported frozen config type")


def _normalized_sample_rate(value: Any, *, field: str) -> float:
    """Implement the one intentional checkpoint-argument normalization."""

    if value is None:
        return 25000000.0
    rate = _typed_config_value(value, expected_type=float, field=field)
    return 25000000.0 if rate <= 0.0 else rate


def state_tensor_schema_sha256(state: Mapping[str, Any]) -> str:
    rows: list[list[Any]] = []
    for name, tensor in sorted(state.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise _error("checkpoint state must map string names to tensors")
        rows.append([name, str(tensor.dtype), [int(value) for value in tensor.shape]])
    if not rows:
        raise _error("checkpoint state tensor schema is empty")
    return canonical_sha256(rows)


def resolve_model_config_projection(
    *,
    checkpoint: Mapping[str, Any],
    resolved_namespace: Any,
) -> dict[str, Any]:
    """Freeze only the architecture keys named by the design card."""

    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("args"), Mapping) or not isinstance(checkpoint.get("model"), Mapping):
        raise _error("checkpoint must contain args and model mappings")
    checkpoint_args = dict(checkpoint["args"])
    state = strip_module_prefix(checkpoint["model"])
    num_domains = infer_num_domains_from_state(state)
    config: dict[str, Any] = {
        "num_classes": _require_nonbool_int(checkpoint_args.get("num_classes"), field="checkpoint num_classes"),
        "num_domains": int(num_domains),
        "input_len": INPUT_LEN,
    }
    if config["num_classes"] != len(LOCAL4_HANDLES):
        raise _error("checkpoint args num_classes must equal local4")
    for key, (value_type, default) in MODEL_CONFIG_SPEC.items():
        raw = checkpoint_args[key] if key in checkpoint_args else default
        if key == "sample_rate_hz":
            config[key] = _normalized_sample_rate(raw, field=key)
        else:
            config[key] = _typed_config_value(raw, expected_type=value_type, field=key)
    if config["dataset"] != "wisig":
        raise _error("frozen checkpoint dataset must be wisig")
    if config["arch_family"] != "cvsincnet":
        raise _error("frozen checkpoint arch_family must be cvsincnet")
    for key, expected in config.items():
        actual = _namespace_value(resolved_namespace, key)
        if actual is _MISSING:
            if key not in MODEL_NAMESPACE_ABSENT_DEFAULTS:
                raise _error(f"resolved model config {key} is absent")
            if key in checkpoint_args:
                raise _error(f"resolved model config {key} is absent despite checkpoint args")
            actual = _typed_config_value(
                MODEL_NAMESPACE_ABSENT_DEFAULTS[key],
                expected_type=MODEL_CONFIG_SPEC[key][0],
                field=f"frozen model fallback {key}",
            )
            if actual != expected:
                raise _error(f"frozen model fallback drift for {key}")
        elif key in ("num_domains", "input_len"):
            actual = _require_nonbool_int(actual, field=f"resolved {key}")
        elif key in MODEL_CONFIG_SPEC:
            if key == "sample_rate_hz":
                actual = _normalized_sample_rate(actual, field=f"resolved {key}")
            else:
                actual = _typed_config_value(actual, expected_type=MODEL_CONFIG_SPEC[key][0], field=f"resolved {key}")
        else:
            actual = _require_nonbool_int(actual, field=f"resolved {key}")
        if actual != expected:
            raise _error(f"resolved model config drift for {key}")
    config["strict_state_tensor_schema_sha256"] = state_tensor_schema_sha256(state)
    return config


def resolved_config_sha256(
    *,
    receipt_projection: Mapping[str, Any],
    checkpoint_sha256: str,
    dataset_sha256: str,
    model_config: Mapping[str, Any],
    code_sha256: Mapping[str, str],
    preprocessing_code_sha256: str,
    scenario_code_sha256: str,
) -> str:
    """Hash the complete frozen configuration projection, never vague args."""

    code_map = _require_mapping(code_sha256, field="code_sha256")
    if set(code_map) != EXPECTED_CODE_SHA_PATHS:
        raise _error("code SHA projection must exactly match the frozen model/scenario allowlist")
    for name, digest in code_map.items():
        _require_nfc_string(name, field="code SHA path")
        code_map[name] = _require_sha256(digest, field=f"code SHA {name}")
    payload = {
        "receipt_projection": _require_mapping(receipt_projection, field="receipt projection"),
        "checkpoint_sha256": _require_sha256(checkpoint_sha256, field="checkpoint SHA"),
        "dataset_sha256": _require_sha256(dataset_sha256, field="dataset SHA"),
        "model_config": _require_mapping(model_config, field="model config"),
        "input_len": INPUT_LEN,
        "equalized": 1,
        "local4_class_order": list(LOCAL4_HANDLES),
        "preprocess_operator_id": PREPROCESS_OPERATOR_ID,
        "preprocessing_code_sha256": _require_sha256(preprocessing_code_sha256, field="preprocessing code SHA"),
        "scenario_code_sha256": _require_sha256(scenario_code_sha256, field="scenario code SHA"),
        "seed_rule_id": SEED_RULE_ID,
        "scenarios": list(SCENARIOS),
        "formula_id": FORMULA_ID,
        "alpha": ALPHA,
        "tail_levels": TAIL_LEVELS.tolist(),
        "resource_gate": {
            "bundle_bytes": MAX_BUNDLE_BYTES,
            "evidence_bytes": MAX_EVIDENCE_BYTES,
            "cpu_rss_delta_bytes": MAX_CPU_RSS_DELTA_BYTES,
            "cuda_vram_bytes": MAX_CUDA_VRAM_BYTES,
            "cpu_warmups": CPU_WARMUPS,
            "cpu_trials": CPU_TRIALS,
            "cpu_p99_limit_ms": CPU_P99_LIMIT_MS,
        },
        "code_sha256": code_map,
    }
    return canonical_sha256(payload)


def build_source_partition_receipt(
    *,
    dataset_sha256: str,
    source_split_projection: Mapping[str, Any],
    tx_partition_receipt: Mapping[str, Any],
    labeled_keys: Iterable[Sequence[Any]],
    unlabeled_keys: Iterable[Sequence[Any]],
    calibration_keys: Iterable[Sequence[Any]],
    excluded_role_keys: Mapping[str, Iterable[Sequence[Any]]],
) -> dict[str, Any]:
    """Prove L/U/V physical exclusivity before dropping all sample identifiers.

    Raw canonical tokens live only in this function's local variables.  The
    returned receipt contains counts and one-way hashes, never a sample key.
    """

    dataset_sha = _require_sha256(dataset_sha256, field="dataset SHA")
    source_split = _require_mapping(source_split_projection, field="source split projection")
    tx_partition = _require_mapping(tx_partition_receipt, field="TX partition receipt")
    groups: dict[str, set[str]] = {}
    for name, keys in (
        ("labeled", labeled_keys),
        ("unlabeled", unlabeled_keys),
        ("calibration", calibration_keys),
    ):
        tokens = [physical_token(key) for key in keys]
        if len(tokens) != len(set(tokens)):
            raise _error(f"{name} physical partition contains duplicate canonical keys")
        groups[name] = set(tokens)
    if any(not values for values in groups.values()):
        raise _error("L/U/V partitions must all contain physical records")
    pairs = (("labeled", "unlabeled"), ("labeled", "calibration"), ("unlabeled", "calibration"))
    if any(groups[left].intersection(groups[right]) for left, right in pairs):
        raise _error("L/U/V canonical physical sets overlap")
    labels = {json.loads(token)[0] for token in groups["labeled"]}
    calibration_labels = {json.loads(token)[0] for token in groups["calibration"]}
    if labels != set(LOCAL4_HANDLES) or calibration_labels != set(LOCAL4_HANDLES):
        raise _error("L and V TX sets must exactly equal the frozen local4 registry")
    excluded_summary: dict[str, dict[str, Any]] = {}
    all_source = set().union(*groups.values())
    for role, keys in excluded_role_keys.items():
        role_name = _require_nfc_string(role, field="excluded role name")
        values_list = [physical_token(key) for key in keys]
        if len(values_list) != len(set(values_list)):
            raise _error(f"excluded {role_name} partition contains duplicate canonical keys")
        values = set(values_list)
        if all_source.intersection(values):
            raise _error(f"source L/U/V intersects excluded {role_name} physical keys")
        excluded_summary[role_name] = {
            "physical_count": len(values),
            "physical_set_sha256": canonical_sha256(sorted(values)),
        }
    if set(excluded_summary) != {"proxy", "held", "target"}:
        raise _error("excluded-role receipt must prove proxy/held/target zero intersection")
    split_hashes = {
        name: _require_sha256(source_split.get(name), field=f"source split {name}")
        for name in (
            "labeled_indices_sha256",
            "unlabeled_indices_sha256",
            "source_validation_indices_sha256",
            "split_manifest_sha256",
        )
    }
    split_sizes = {
        name: _require_nonbool_int(source_split.get(name), field=f"source split {name}", minimum=1)
        for name in ("labeled_size", "unlabeled_size", "source_validation_size", "source_pool_size")
    }
    return {
        "schema": "cvs.phase1.single_control_source_partition.v1",
        "dataset_sha256": dataset_sha,
        "equalized": 1,
        "source_local4_tx": list(LOCAL4_HANDLES),
        "source_split_projection_sha256": canonical_sha256(source_split),
        "source_split_hashes": split_hashes,
        "source_split_sizes": split_sizes,
        "tx_partition_projection_sha256": canonical_sha256(tx_partition),
        "labeled_physical_count": len(groups["labeled"]),
        "unlabeled_physical_count": len(groups["unlabeled"]),
        "calibration_physical_count": len(groups["calibration"]),
        "labeled_physical_set_sha256": canonical_sha256(sorted(groups["labeled"])),
        "unlabeled_physical_set_sha256": canonical_sha256(sorted(groups["unlabeled"])),
        "calibration_physical_set_sha256": canonical_sha256(sorted(groups["calibration"])),
        "pairwise_disjoint": True,
        "excluded_roles": excluded_summary,
        "sample_tokens_retained": False,
        "calibration_optimizer_updates": False,
    }


_RESOURCE_RECEIPT_FIELDS = {
    "schema", "input_shape", "input_dtype", "input_sha256", "input_seed", "torch_num_threads",
    "cpu_rss_baseline_bytes", "cpu_rss_peak_bytes", "cpu_rss_delta_bytes", "cpu_warmups", "cpu_trials",
    "cpu_latency_p99_ms", "cpu_latency_quantile_method", "cuda_available", "cuda_peak_bytes",
    "cuda_latency_recorded", "cuda_latency_p99_ms", "measurement_scope", "evidence_bytes",
    "measurement_process", "baseline_before_payload_load", "state_payload_reloaded",
    "runtime_state_before_sha256", "runtime_state_after_sha256",
}


def resource_probe_model_input() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RESOURCE_INPUT_SEED)
    return torch.randn((1, 2, INPUT_LEN), generator=generator, dtype=torch.float32).contiguous()


def validate_resource_receipt(value: Any) -> dict[str, Any]:
    receipt = _require_mapping(value, field="resource receipt")
    if set(receipt) != _RESOURCE_RECEIPT_FIELDS:
        raise _error("resource receipt key allowlist drift")
    if receipt["schema"] != "cvs.phase1.single_control_resource_receipt.v1":
        raise _error("resource receipt schema drift")
    if receipt["input_shape"] != [1, 2, INPUT_LEN] or receipt["input_dtype"] != "torch.float32":
        raise _error("resource receipt input contract drift")
    if _require_sha256(receipt["input_sha256"], field="resource input SHA") != tensor_sha256(resource_probe_model_input()):
        raise _error("resource receipt input SHA does not match the frozen deterministic probe")
    if _require_nonbool_int(receipt["input_seed"], field="resource input seed", minimum=0) != RESOURCE_INPUT_SEED:
        raise _error("resource receipt input seed drift")
    if _require_nonbool_int(receipt["torch_num_threads"], field="resource torch_num_threads", minimum=1) != 1:
        raise _error("resource receipt must freeze torch.set_num_threads(1)")
    if _require_nonbool_int(receipt["cpu_warmups"], field="resource warmups", minimum=0) != CPU_WARMUPS:
        raise _error("resource receipt warmup count drift")
    if _require_nonbool_int(receipt["cpu_trials"], field="resource trials", minimum=1) != CPU_TRIALS:
        raise _error("resource receipt trial count drift")
    if receipt["cpu_latency_quantile_method"] != "higher_q99_100":
        raise _error("resource receipt CPU p99 method drift")
    if receipt["measurement_scope"] != "fresh_process_bundle_load_warmup_full_local_evidence":
        raise _error("resource receipt measurement scope drift")
    if receipt["measurement_process"] != "fresh_python_subprocess_v1":
        raise _error("resource receipt must use a fresh Python measurement process")
    _require_exact_bool(receipt["baseline_before_payload_load"], field="resource baseline_before_payload_load", expected=True)
    _require_exact_bool(receipt["state_payload_reloaded"], field="resource state_payload_reloaded", expected=True)
    before = _require_sha256(receipt["runtime_state_before_sha256"], field="resource runtime state before SHA")
    after = _require_sha256(receipt["runtime_state_after_sha256"], field="resource runtime state after SHA")
    if before != after:
        raise _error("resource receipt records runtime/state mutation during measurement")
    for field in ("cpu_rss_baseline_bytes", "cpu_rss_peak_bytes", "cpu_rss_delta_bytes", "evidence_bytes"):
        receipt[field] = _require_nonbool_int(receipt[field], field=field, minimum=0)
    if receipt["cpu_rss_peak_bytes"] < receipt["cpu_rss_baseline_bytes"] or receipt["cpu_rss_delta_bytes"] != receipt["cpu_rss_peak_bytes"] - receipt["cpu_rss_baseline_bytes"]:
        raise _error("resource receipt CPU RSS closure failed")
    if receipt["cpu_rss_delta_bytes"] > MAX_CPU_RSS_DELTA_BYTES:
        raise _error("resource receipt CPU RSS gate failed")
    if receipt["evidence_bytes"] < 1 or receipt["evidence_bytes"] > MAX_EVIDENCE_BYTES:
        raise _error("resource receipt evidence size gate failed")
    receipt["cpu_latency_p99_ms"] = _require_finite_float(receipt["cpu_latency_p99_ms"], field="cpu_latency_p99_ms")
    if receipt["cpu_latency_p99_ms"] < 0.0 or receipt["cpu_latency_p99_ms"] > CPU_P99_LIMIT_MS:
        raise _error("resource receipt CPU latency gate failed")
    _require_exact_bool(receipt["cuda_available"], field="cuda_available")
    _require_exact_bool(receipt["cuda_latency_recorded"], field="cuda_latency_recorded")
    receipt["cuda_peak_bytes"] = _require_nonbool_int(receipt["cuda_peak_bytes"], field="cuda_peak_bytes", minimum=0)
    receipt["cuda_latency_p99_ms"] = _require_finite_float(receipt["cuda_latency_p99_ms"], field="cuda_latency_p99_ms")
    if receipt["cuda_latency_p99_ms"] < 0.0:
        raise _error("resource receipt CUDA latency must be non-negative")
    if receipt["cuda_available"]:
        if receipt["cuda_peak_bytes"] > MAX_CUDA_VRAM_BYTES or receipt["cuda_latency_recorded"] is not True:
            raise _error("resource receipt CUDA gate/latency recording failed")
    elif receipt["cuda_peak_bytes"] != 0 or receipt["cuda_latency_recorded"] is not False or receipt["cuda_latency_p99_ms"] != 0.0:
        raise _error("resource receipt CUDA absence fields drift")
    return receipt


@dataclass(frozen=True)
class LoadedBundle:
    """A bundle that has passed external-root and payload validation."""

    root: Path
    content_root: str
    manifest: Mapping[str, Any]
    state: BundleState
    runtime: torch.jit.ScriptModule


def _plain_json_bytes(value: Any) -> bytes:
    """Stable payload JSON; root binding always uses :func:`canonical_json_bytes`."""

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error("payload JSON is not serializable") from exc


def _read_regular_file(path: Path, *, context: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise _error(f"{context} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise _error(f"{context} is unreadable") from exc


def _payload_must_not_reference_root(raw: bytes, *, member: str) -> None:
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Binary NPZ/TorchScript payloads do not expose JSON member names.
        return

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"content_root", "manifest_sha256"}:
                    raise _error(f"payload member {member} must not reference a content root or manifest hash")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(decoded)


def _member_descriptors(payloads: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    if set(payloads) != set(PAYLOAD_MEMBERS):
        raise _error("payload member allowlist drift")
    result: dict[str, dict[str, Any]] = {}
    for name in PAYLOAD_MEMBERS:
        raw = payloads[name]
        if not isinstance(raw, bytes) or not raw:
            raise _error(f"payload member {name} must be non-empty bytes")
        _payload_must_not_reference_root(raw, member=name)
        result[name] = {"sha256": _sha256_bytes(raw), "size_bytes": len(raw)}
    return result


def _require_binding_mapping(value: Any, *, context: str, checkpoint_sha256: str, resolved_config_digest: str) -> dict[str, Any]:
    mapping = _require_mapping(value, field=context)
    if mapping.get("checkpoint_sha256") != checkpoint_sha256:
        raise _error(f"{context} checkpoint SHA closure failed")
    if mapping.get("resolved_config_sha256") != resolved_config_digest:
        raise _error(f"{context} resolved-config SHA closure failed")
    return mapping


_CHECKPOINT_BINDING_FIELDS = {
    "checkpoint_sha256", "resolved_config_sha256", "checkpoint_role", "strict_state_tensor_schema_sha256", "strict_load_audit",
}
_STRICT_LOAD_AUDIT_FIELDS = {"strict", "missing_keys", "unexpected_keys"}
_CLASS_BINDING_FIELDS = {
    "class_handles", "local_to_head_class_ids", "class_order_binding_sha256", "checkpoint_head_class_count",
    "live_head_class_count", "checkpoint_train_tx_class_order",
}
_SOURCE_PARTITION_FIELDS = {
    "schema", "dataset_sha256", "equalized", "source_local4_tx", "source_split_projection_sha256", "source_split_hashes",
    "source_split_sizes", "tx_partition_projection_sha256", "labeled_physical_count", "unlabeled_physical_count",
    "calibration_physical_count", "labeled_physical_set_sha256", "unlabeled_physical_set_sha256",
    "calibration_physical_set_sha256", "pairwise_disjoint", "excluded_roles", "sample_tokens_retained",
    "calibration_optimizer_updates",
}
_EXCLUDED_ROLE_SUMMARY_FIELDS = {"physical_count", "physical_set_sha256"}


def _validate_build_inputs(
    *,
    state: BundleState,
    checkpoint_binding: Mapping[str, Any],
    class_binding: Mapping[str, Any],
    source_partition_receipt: Mapping[str, Any],
    runtime_parity_receipt: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
    checkpoint_sha256: str,
    resolved_config_digest: str,
    dataset_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reject ambiguous state/receipt combinations before any output is created."""

    if tuple(state.geometry.class_handles) != LOCAL4_HANDLES:
        raise _error("state geometry must bind exactly the frozen local4 order")
    if state.tail.n_calibration < MIN_CALIBRATION_PHYSICALS:
        raise _error("tail calibration must contain at least 199 physical atoms")
    if 1.0 / float(state.tail.n_calibration + 1) > ALPHA / 2.0:
        raise _error("tail calibration resolution does not satisfy the frozen alpha gate")
    checkpoint = _require_binding_mapping(
        checkpoint_binding,
        context="checkpoint binding",
        checkpoint_sha256=checkpoint_sha256,
        resolved_config_digest=resolved_config_digest,
    )
    if set(checkpoint) != _CHECKPOINT_BINDING_FIELDS:
        raise _error("checkpoint binding key allowlist drift")
    if checkpoint.get("checkpoint_role") != "training_final_only":
        raise _error("checkpoint binding must identify a training-final-only control")
    _require_sha256(checkpoint.get("strict_state_tensor_schema_sha256"), field="checkpoint strict state tensor schema SHA")
    strict_audit = _require_mapping(checkpoint.get("strict_load_audit"), field="checkpoint strict_load_audit")
    if set(strict_audit) != _STRICT_LOAD_AUDIT_FIELDS:
        raise _error("checkpoint strict-load audit key allowlist drift")
    if strict_audit.get("strict") is not True or strict_audit.get("missing_keys") != [] or strict_audit.get("unexpected_keys") != []:
        raise _error("checkpoint binding lacks a closed strict-load audit")
    classes = _require_mapping(class_binding, field="class binding")
    if set(classes) != _CLASS_BINDING_FIELDS:
        raise _error("class binding key allowlist drift")
    if classes.get("class_handles") != list(LOCAL4_HANDLES):
        raise _error("class binding must freeze the local4 class order")
    if classes.get("local_to_head_class_ids") != [0, 1, 2, 3]:
        raise _error("class binding must freeze the identity local-to-head mapping")
    _require_sha256(classes.get("class_order_binding_sha256"), field="class binding order SHA")
    if _require_nonbool_int(classes.get("checkpoint_head_class_count"), field="checkpoint head rows") != 4 or _require_nonbool_int(
        classes.get("live_head_class_count"), field="live head rows"
    ) != 4:
        raise _error("class binding must close both checkpoint and live local4 head rows")
    if classes.get("checkpoint_train_tx_class_order") != list(LOCAL4_HANDLES):
        raise _error("class binding checkpoint class order drift")
    source = _require_mapping(source_partition_receipt, field="source partition receipt")
    if set(source) != _SOURCE_PARTITION_FIELDS:
        raise _error("source partition receipt key allowlist drift")
    if source.get("schema") != "cvs.phase1.single_control_source_partition.v1":
        raise _error("source partition receipt schema drift")
    if source.get("dataset_sha256") != dataset_sha256:
        raise _error("source partition dataset SHA closure failed")
    if source.get("equalized") != 1 or source.get("source_local4_tx") != list(LOCAL4_HANDLES):
        raise _error("source partition equalized/local4 binding drift")
    if source.get("pairwise_disjoint") is not True or source.get("sample_tokens_retained") is not False or source.get("calibration_optimizer_updates") is not False:
        raise _error("source partition physical/optimizer boundary drift")
    _require_sha256(source.get("source_split_projection_sha256"), field="source split projection SHA")
    _require_sha256(source.get("tx_partition_projection_sha256"), field="TX partition projection SHA")
    split_hashes = _require_mapping(source.get("source_split_hashes"), field="source split hashes")
    if set(split_hashes) != {
        "labeled_indices_sha256", "unlabeled_indices_sha256", "source_validation_indices_sha256", "split_manifest_sha256"
    }:
        raise _error("source partition split-hash allowlist drift")
    for name, digest in split_hashes.items():
        _require_sha256(digest, field=f"source partition {name}")
    split_sizes = _require_mapping(source.get("source_split_sizes"), field="source split sizes")
    if set(split_sizes) != {"labeled_size", "unlabeled_size", "source_validation_size", "source_pool_size"}:
        raise _error("source partition split-size allowlist drift")
    for name, value in split_sizes.items():
        _require_nonbool_int(value, field=f"source partition {name}", minimum=1)
    excluded = _require_mapping(source.get("excluded_roles"), field="source partition excluded roles")
    if set(excluded) != {"proxy", "held", "target"}:
        raise _error("source partition excluded role allowlist drift")
    for role, summary in excluded.items():
        summary_map = _require_mapping(summary, field=f"source partition excluded {role}")
        if set(summary_map) != _EXCLUDED_ROLE_SUMMARY_FIELDS:
            raise _error("source partition excluded-role summary key allowlist drift")
        _require_nonbool_int(summary_map.get("physical_count"), field=f"excluded {role} count", minimum=0)
        _require_sha256(summary_map.get("physical_set_sha256"), field=f"excluded {role} physical SHA")
    for name in ("labeled_physical_set_sha256", "unlabeled_physical_set_sha256", "calibration_physical_set_sha256"):
        source[name] = _require_sha256(source.get(name), field=f"source partition {name}")
    for name in ("labeled_physical_count", "unlabeled_physical_count", "calibration_physical_count"):
        source[name] = _require_nonbool_int(source.get(name), field=f"source partition {name}", minimum=1)
    if len({source["labeled_physical_set_sha256"], source["unlabeled_physical_set_sha256"], source["calibration_physical_set_sha256"]}) != 3:
        raise _error("L/U/V physical-set receipt must be pairwise distinct")
    if int(np.asarray(state.geometry.class_counts, dtype=np.int64).sum()) != source["labeled_physical_count"]:
        raise _error("class-geometry count does not close the labeled physical partition")
    expected_descriptor_count = len(VIEW_ORDER) * (source["labeled_physical_count"] + source["unlabeled_physical_count"])
    if int(state.descriptor.descriptor_count) != expected_descriptor_count:
        raise _error("descriptor count does not close L/U physical views")
    if int(state.tail.n_calibration) != source["calibration_physical_count"]:
        raise _error("tail calibration count does not close V physical partition")
    if state.tail.calibration_set_sha256 != source["calibration_physical_set_sha256"]:
        raise _error("tail calibration set SHA does not bind V physical partition")
    parity = _require_binding_mapping(
        runtime_parity_receipt,
        context="runtime parity receipt",
        checkpoint_sha256=checkpoint_sha256,
        resolved_config_digest=resolved_config_digest,
    )
    resource = validate_resource_receipt(resource_receipt)
    return checkpoint, classes, source, parity, resource


_RUNTIME_PARITY_FIELDS = {
    "schema", "checkpoint_sha256", "resolved_config_sha256", "runtime_sha256", "runtime_state_schema_sha256",
    "input_sha256", "atol", "state_unchanged", "strict_load_complete", "full_path_fields", "max_abs",
    "decision_equal", "label_equal", "reason_equal", "bundle_id_equal", "class_handles_equal",
    "eager_state_before_sha256", "eager_state_after_sha256", "loaded_state_before_sha256", "loaded_state_after_sha256",
    "runtime_batch_capacity", "runtime_internal_padding", "raw_model_direct_identity_parity",
}


def _validate_runtime_parity_binding(parity: Mapping[str, Any], *, runtime_bytes: bytes, state: BundleState) -> dict[str, Any]:
    receipt = _require_mapping(parity, field="runtime parity receipt")
    if set(receipt) != _RUNTIME_PARITY_FIELDS or receipt.get("schema") != "cvs.phase1.single_control_runtime_parity.v1":
        raise _error("runtime parity receipt schema/key allowlist drift")
    if _require_sha256(receipt.get("runtime_sha256"), field="runtime parity runtime SHA") != _sha256_bytes(runtime_bytes):
        raise _error("runtime parity receipt does not bind the actual TorchScript bytes")
    _require_sha256(receipt.get("input_sha256"), field="runtime parity input SHA")
    if _require_finite_float(receipt.get("atol"), field="runtime parity atol") < 0.0:
        raise _error("runtime parity tolerance is invalid")
    if receipt.get("state_unchanged") is not True or receipt.get("strict_load_complete") is not True:
        raise _error("runtime parity receipt lacks strict/full-path state closure")
    if _require_nonbool_int(receipt.get("runtime_batch_capacity"), field="runtime batch capacity", minimum=1) != 1:
        raise _error("runtime parity receipt capacity must be fixed batch1")
    _require_exact_bool(receipt.get("runtime_internal_padding"), field="runtime internal padding", expected=False)
    _require_exact_bool(receipt.get("raw_model_direct_identity_parity"), field="raw model direct identity parity", expected=True)
    expected_fields = ["z_id", "z_dom", "q", "d_class", "e_unknown", "p_local", "local_decision", "local_label", "reason_code"]
    if receipt.get("full_path_fields") != expected_fields:
        raise _error("runtime parity receipt does not cover all six local fields and decision metadata")
    max_abs = _require_mapping(receipt.get("max_abs"), field="runtime parity max_abs")
    if set(max_abs) != {"z_id", "z_dom", "q", "d_class", "e_unknown", "p_local"}:
        raise _error("runtime parity max_abs field set drift")
    tolerance = _require_finite_float(receipt.get("atol"), field="runtime parity atol")
    for field, value in max_abs.items():
        error = _require_finite_float(value, field=f"runtime parity {field} max_abs")
        if error < 0.0 or error > tolerance:
            raise _error("runtime parity numerical error exceeds frozen tolerance")
    for field in ("decision_equal", "label_equal", "reason_equal", "bundle_id_equal", "class_handles_equal"):
        _require_exact_bool(receipt.get(field), field=f"runtime parity {field}", expected=True)
    for field in (
        "eager_state_before_sha256", "eager_state_after_sha256", "loaded_state_before_sha256", "loaded_state_after_sha256"
    ):
        _require_sha256(receipt.get(field), field=f"runtime parity {field}")
    if receipt["eager_state_before_sha256"] != receipt["eager_state_after_sha256"] or receipt["loaded_state_before_sha256"] != receipt["loaded_state_after_sha256"]:
        raise _error("runtime parity receipt records state mutation")
    try:
        runtime = torch.jit.load(io.BytesIO(runtime_bytes), map_location="cpu").eval()
        actual_schema = runtime_state_schema(runtime)["runtime_state_schema_sha256"]
    except Exception as exc:
        raise _error("runtime parity cannot strict-load TorchScript bytes") from exc
    if _require_sha256(receipt.get("runtime_state_schema_sha256"), field="runtime parity state schema SHA") != actual_schema:
        raise _error("runtime parity state schema does not bind actual TorchScript")
    actual_loaded_state = _runtime_state_digest(runtime, state)
    if receipt["loaded_state_before_sha256"] != actual_loaded_state or receipt["loaded_state_after_sha256"] != actual_loaded_state:
        raise _error("runtime parity receipt loaded-state digest does not bind actual runtime/state")
    # State participates in the full-path audit; force early shape validation
    # here, before writing any payload byte.
    if tuple(state.geometry.class_handles) != LOCAL4_HANDLES:
        raise _error("runtime parity state geometry is not local4")
    return receipt


def _assert_new_bundle_output_target(output_dir: str | Path) -> Path:
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite an existing bundle root")
    if target.name in {"", ".", ".."}:
        raise _error("bundle output root must have a concrete directory name")
    if target.parent.is_dir() and any(target.parent.glob(f".{target.name}.single-control-staging-*")):
        raise FileExistsError("refusing to reuse a single-control staging root")
    return target


def build_bundle(
    *,
    output_dir: str | Path,
    runtime_source: bytes | str | Path,
    state: BundleState,
    checkpoint_binding: Mapping[str, Any],
    class_binding: Mapping[str, Any],
    source_partition_receipt: Mapping[str, Any],
    runtime_parity_receipt: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
    checkpoint_sha256: str,
    resolved_config_digest: str,
    dataset_sha256: str,
    preprocessing_code_sha256: str,
    scenario_registry_sha256: str,
    bundle_status: str = BUNDLE_STATUS,
) -> dict[str, Any]:
    """Write exactly one immutable 10-member single-control bundle.

    The caller must have already built the runtime and source-only state.  This
    function performs no fitting and refuses any pre-existing output root.
    """

    target = _assert_new_bundle_output_target(output_dir)
    checkpoint_sha = _require_sha256(checkpoint_sha256, field="checkpoint SHA")
    config_sha = _require_sha256(resolved_config_digest, field="resolved config SHA")
    dataset_sha = _require_sha256(dataset_sha256, field="dataset SHA")
    preprocessing_sha = _require_sha256(preprocessing_code_sha256, field="preprocessing code SHA")
    scenario_sha = _require_sha256(scenario_registry_sha256, field="scenario registry SHA")
    if bundle_status not in {BUNDLE_STATUS, FIXTURE_STATUS}:
        raise _error("bundle status is not one frozen technical-control status")
    checkpoint, classes, source, parity, resource = _validate_build_inputs(
        state=state,
        checkpoint_binding=checkpoint_binding,
        class_binding=class_binding,
        source_partition_receipt=source_partition_receipt,
        runtime_parity_receipt=runtime_parity_receipt,
        resource_receipt=resource_receipt,
        checkpoint_sha256=checkpoint_sha,
        resolved_config_digest=config_sha,
        dataset_sha256=dataset_sha,
    )
    if isinstance(runtime_source, bytes):
        runtime_bytes = runtime_source
    else:
        runtime_bytes = _read_regular_file(Path(runtime_source), context="TorchScript runtime")
    if not runtime_bytes:
        raise _error("TorchScript runtime is empty")
    parity = _validate_runtime_parity_binding(parity, runtime_bytes=runtime_bytes, state=state)
    payloads: dict[str, bytes] = {
        "runtime/local_evidence.ts": runtime_bytes,
        "state/class_geometry.npz": _npz_bytes(geometry_npz_arrays(state.geometry)),
        "state/domain_descriptor_stats.npz": _npz_bytes(descriptor_npz_arrays(state.descriptor)),
        "state/rank_tail_summary.npz": _npz_bytes(tail_npz_arrays(state.tail)),
        "locks/checkpoint_binding.json": _plain_json_bytes(checkpoint),
        "locks/class_binding.json": _plain_json_bytes(classes),
        "locks/source_partition_receipt.json": _plain_json_bytes(source),
        "locks/runtime_parity_receipt.json": _plain_json_bytes(parity),
        "locks/resource_receipt.json": _plain_json_bytes(resource),
    }
    members = _member_descriptors(payloads)
    manifest_without_root: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_status": bundle_status,
        "members": members,
        "checkpoint_sha256": checkpoint_sha,
        "resolved_config_sha256": config_sha,
        "dataset_sha256": dataset_sha,
        "preprocessing_code_sha256": preprocessing_sha,
        "scenario_registry_sha256": scenario_sha,
        "class_handles": list(LOCAL4_HANDLES),
        "formula_id": FORMULA_ID,
        "alpha": ALPHA,
        "calibration_set_sha256": state.tail.calibration_set_sha256,
        "raw_iq": False,
        "source_checkpoint_container": False,
        "runtime_embeds_frozen_weights": True,
        "sample_feature_cache": False,
        "physical_ids": False,
        "role_or_truth": False,
        "performance_promoted": False,
        "finite_sample_exact_conformal": False,
        "source_exchangeable_calibration": False,
        "z_dom_provenance": "fixed_iq_statistical_domain_descriptor_v1",
        "learned_domain_representation": False,
        "q_semantics": "model_reliability_not_physical_quality",
        "runtime_engine": "torchscript_identity_logits_plus_fixed_state_v1",
        "runtime_batch_capacity": 1,
        "runtime_internal_padding": False,
    }
    content_root = canonical_sha256(manifest_without_root)
    manifest = dict(manifest_without_root)
    manifest["content_root"] = content_root
    manifest_bytes = _plain_json_bytes(manifest)
    staging = target.parent / f".{target.name}.single-control-staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("single-control staging path already exists")
    try:
        staging.mkdir(parents=True, exist_ok=False)
        for name, raw in payloads.items():
            path = staging / name
            if staging not in path.parents:
                raise _error("payload path escapes single-control staging root")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.is_symlink():
                raise FileExistsError("refusing to overwrite staging payload member")
            path.write_bytes(raw)
        (staging / MANIFEST_NAME).write_bytes(manifest_bytes)
        os.replace(staging, target)
    except Exception:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)
        raise
    # Independently re-open via the immutable loader before reporting success.
    loaded = load_bundle(
        target,
        expected_content_root=content_root,
        device=torch.device("cpu"),
        expected_bundle_status=bundle_status,
    )
    return {
        "schema": "cvs.phase1.single_control_build_result.v1",
        "bundle_dir": str(target),
        "content_root": loaded.content_root,
        "manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "member_count": len(ALL_BUNDLE_MEMBERS),
        "bundle_bytes": int(sum((target / name).stat().st_size for name in ALL_BUNDLE_MEMBERS)),
    }


_MANIFEST_REQUIRED_KEYS = {
    "schema", "bundle_status", "members", "checkpoint_sha256", "resolved_config_sha256", "dataset_sha256",
    "preprocessing_code_sha256", "scenario_registry_sha256", "class_handles", "formula_id", "alpha",
    "calibration_set_sha256", "raw_iq", "source_checkpoint_container", "runtime_embeds_frozen_weights",
    "sample_feature_cache", "physical_ids", "role_or_truth", "performance_promoted",
    "finite_sample_exact_conformal", "source_exchangeable_calibration", "z_dom_provenance",
    "learned_domain_representation", "q_semantics", "runtime_engine", "runtime_batch_capacity",
    "runtime_internal_padding", "content_root",
}


def _bundle_file_names(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise _error("bundle root must be a real directory")
    names: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise _error("bundle symlinks are forbidden")
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
    return names


def load_bundle(
    bundle_dir: str | Path,
    *,
    expected_content_root: str,
    device: torch.device | str = "cpu",
    expected_bundle_status: str = BUNDLE_STATUS,
) -> LoadedBundle:
    """Validate all bytes against an externally supplied immutable root."""

    root = Path(bundle_dir)
    expected = _require_sha256(expected_content_root, field="external expected_content_root")
    if _bundle_file_names(root) != set(ALL_BUNDLE_MEMBERS):
        raise _error("bundle must contain exactly the frozen nine payloads plus manifest")
    manifest_raw = _read_regular_file(root / MANIFEST_NAME, context="bundle manifest")
    try:
        manifest = _require_mapping(json.loads(manifest_raw.decode("utf-8")), field="bundle manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("bundle manifest is unreadable JSON") from exc
    if set(manifest) != _MANIFEST_REQUIRED_KEYS or manifest.get("schema") != SCHEMA:
        raise _error("bundle manifest schema/key allowlist drift")
    if manifest.get("bundle_status") not in {BUNDLE_STATUS, FIXTURE_STATUS}:
        raise _error("bundle manifest status drift")
    if manifest.get("bundle_status") != expected_bundle_status:
        raise _error("bundle manifest status differs from the caller's expected status")
    root_in_manifest = _require_sha256(manifest.get("content_root"), field="manifest content_root")
    unsigned = dict(manifest)
    unsigned.pop("content_root")
    recomputed = canonical_sha256(unsigned)
    if root_in_manifest != recomputed or recomputed != expected:
        raise _error("bundle content-root external anchor closure failed")
    for field in (
        "checkpoint_sha256", "resolved_config_sha256", "dataset_sha256", "preprocessing_code_sha256",
        "scenario_registry_sha256", "calibration_set_sha256",
    ):
        _require_sha256(manifest.get(field), field=f"manifest {field}")
    if manifest.get("class_handles") != list(LOCAL4_HANDLES) or manifest.get("formula_id") != FORMULA_ID or manifest.get("alpha") != ALPHA:
        raise _error("bundle manifest semantic binding drift")
    expected_flags = {
        "raw_iq": False, "source_checkpoint_container": False, "runtime_embeds_frozen_weights": True,
        "sample_feature_cache": False, "physical_ids": False, "role_or_truth": False,
        "performance_promoted": False, "finite_sample_exact_conformal": False,
        "source_exchangeable_calibration": False, "learned_domain_representation": False,
    }
    for field, value in expected_flags.items():
        if manifest.get(field) is not value:
            raise _error(f"bundle manifest {field} semantic flag drift")
    if manifest.get("z_dom_provenance") != "fixed_iq_statistical_domain_descriptor_v1":
        raise _error("bundle manifest z_dom provenance drift")
    if manifest.get("q_semantics") != "model_reliability_not_physical_quality":
        raise _error("bundle manifest q semantics drift")
    if manifest.get("runtime_engine") != "torchscript_identity_logits_plus_fixed_state_v1" or manifest.get("runtime_batch_capacity") != 1 or manifest.get("runtime_internal_padding") is not False:
        raise _error("bundle manifest runtime engine/capacity drift")
    members = _require_mapping(manifest.get("members"), field="manifest members")
    if set(members) != set(PAYLOAD_MEMBERS):
        raise _error("bundle manifest payload member allowlist drift")
    total_bytes = len(manifest_raw)
    raw_members: dict[str, bytes] = {}
    for name in PAYLOAD_MEMBERS:
        descriptor = _require_mapping(members[name], field=f"member descriptor {name}")
        if set(descriptor) != {"sha256", "size_bytes"}:
            raise _error("member descriptor key allowlist drift")
        raw = _read_regular_file(root / name, context=f"payload {name}")
        if _sha256_bytes(raw) != _require_sha256(descriptor["sha256"], field=f"member SHA {name}"):
            raise _error(f"payload SHA mismatch for {name}")
        if _require_nonbool_int(descriptor["size_bytes"], field=f"member size {name}", minimum=1) != len(raw):
            raise _error(f"payload size mismatch for {name}")
        _payload_must_not_reference_root(raw, member=name)
        raw_members[name] = raw
        total_bytes += len(raw)
    if total_bytes > MAX_BUNDLE_BYTES:
        raise _error("bundle exceeds the frozen 32 MiB resource gate")
    state = state_from_npz_bytes(
        geometry_raw=raw_members["state/class_geometry.npz"],
        descriptor_raw=raw_members["state/domain_descriptor_stats.npz"],
        tail_raw=raw_members["state/rank_tail_summary.npz"],
    )
    if state.tail.calibration_set_sha256 != manifest["calibration_set_sha256"]:
        raise _error("tail calibration SHA does not close manifest")
    lock_values: dict[str, dict[str, Any]] = {}
    for name in (
        "locks/checkpoint_binding.json",
        "locks/class_binding.json",
        "locks/source_partition_receipt.json",
        "locks/runtime_parity_receipt.json",
        "locks/resource_receipt.json",
    ):
        try:
            lock_values[name] = _require_mapping(json.loads(raw_members[name].decode("utf-8")), field=name)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _error(f"bundle lock {name} is unreadable JSON") from exc
    _validate_build_inputs(
        state=state,
        checkpoint_binding=lock_values["locks/checkpoint_binding.json"],
        class_binding=lock_values["locks/class_binding.json"],
        source_partition_receipt=lock_values["locks/source_partition_receipt.json"],
        runtime_parity_receipt=lock_values["locks/runtime_parity_receipt.json"],
        resource_receipt=lock_values["locks/resource_receipt.json"],
        checkpoint_sha256=manifest["checkpoint_sha256"],
        resolved_config_digest=manifest["resolved_config_sha256"],
        dataset_sha256=manifest["dataset_sha256"],
    )
    _validate_runtime_parity_binding(
        lock_values["locks/runtime_parity_receipt.json"],
        runtime_bytes=raw_members["runtime/local_evidence.ts"],
        state=state,
    )
    try:
        runtime = torch.jit.load(io.BytesIO(raw_members["runtime/local_evidence.ts"]), map_location=device).eval()
        _runtime_outputs(runtime, torch.zeros((1, 2, INPUT_LEN), dtype=torch.float32, device=device))
    except SingleControlBundleError:
        raise
    except Exception as exc:
        raise _error("TorchScript runtime cannot be strictly loaded") from exc
    if manifest["bundle_status"] == BUNDLE_STATUS:
        try:
            capacity_token = getattr(runtime, "runtime_capacity_token")
            capacity = int(capacity_token.detach().cpu().item())
        except Exception as exc:
            raise _error("real TorchScript runtime does not expose its frozen batch capacity") from exc
        if capacity != 1:
            raise _error("real TorchScript runtime batch capacity drift")
    return LoadedBundle(root=root, content_root=expected, manifest=manifest, state=state, runtime=runtime)


def _runtime_state_digest(runtime: Any, state: BundleState) -> str:
    schema = runtime_state_schema(runtime)
    runtime_values: list[dict[str, Any]] = []
    for kind, entries in (("parameter", runtime.named_parameters()), ("buffer", runtime.named_buffers())):
        for name, tensor in entries:
            if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
                raise _error("runtime state digest requires dense tensor state")
            cpu = tensor.detach().to(device="cpu").contiguous()
            try:
                value_bytes = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            except Exception as exc:
                raise _error("runtime state digest cannot serialize tensor values") from exc
            runtime_values.append(
                {
                    "kind": kind,
                    "name": _require_nfc_string(str(name), field="runtime state name"),
                    "dtype": str(cpu.dtype),
                    "shape": [int(value) for value in cpu.shape],
                    "value_sha256": _sha256_bytes(value_bytes),
                }
            )
    runtime_values.sort(key=lambda item: (item["kind"], item["name"]))

    def state_values_sha256(arrays: Mapping[str, Any]) -> str:
        rows: list[dict[str, Any]] = []
        for name in sorted(arrays):
            array = np.ascontiguousarray(np.asarray(arrays[name]))
            if array.dtype == object:
                raise _error("state digest cannot serialize object arrays")
            rows.append(
                {
                    "name": _require_nfc_string(str(name), field="state array name"),
                    "dtype": str(array.dtype),
                    "shape": [int(value) for value in array.shape],
                    "value_sha256": _sha256_bytes(array.tobytes(order="C")),
                }
            )
        return canonical_sha256(rows)

    return canonical_sha256(
        {
            "runtime_state_schema_sha256": schema["runtime_state_schema_sha256"],
            "runtime_state_values": runtime_values,
            "geometry_state_values_sha256": state_values_sha256(geometry_npz_arrays(state.geometry)),
            "descriptor_state_values_sha256": state_values_sha256(descriptor_npz_arrays(state.descriptor)),
            "tail_state_values_sha256": state_values_sha256(tail_npz_arrays(state.tail)),
        }
    )


def _local_fields_from_runtime(runtime: Any, state: BundleState, model_input: torch.Tensor) -> LocalFields:
    checked = _ensure_model_input(model_input)
    if checked.shape[0] != 1:
        raise _error("local evidence runtime accepts exactly one model-input row")
    devices: set[torch.device] = set()
    for entries in (runtime.named_parameters(), runtime.named_buffers()):
        for _, tensor in entries:
            if not torch.is_tensor(tensor):
                raise _error("runtime parameter/buffer device binding is invalid")
            devices.add(tensor.device)
    if len(devices) > 1:
        raise _error("runtime parameters/buffers span multiple devices")
    runtime_device = next(iter(devices), torch.device("cpu"))
    if runtime_device.type == "meta":
        raise _error("runtime cannot execute on a meta device")
    runtime_input = checked.to(device=runtime_device).contiguous()
    with torch.no_grad():
        z_id, logits = _runtime_outputs(runtime, runtime_input)
    return evaluate_local_fields(
        z_id=z_id[0].detach().cpu().numpy(),
        tx_logits=logits[0].detach().cpu().numpy(),
        model_input=checked[0].detach().cpu().contiguous(),
        state=state,
    )


def _safe_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        raise _error("CARE context must be a truth-free mapping")
    forbidden = {str(key) for key in context}.intersection(FORBIDDEN_PREDICTOR_KEYS)
    if forbidden:
        raise _error(f"CARE context contains forbidden predictor keys: {sorted(forbidden)}")
    protected = {
        "schema_version", "bundle_id", "class_handles", "z_id", "z_dom", "q", "d_class", "e_unknown",
        "p_local", "local_decision", "local_label", "reason_code", "evidence_hash", "js_disagreement",
    }
    if protected.intersection(context):
        raise _error("CARE context must not override bundle-generated local evidence fields")
    return dict(context)


def _evidence_payload(*, content_root: str, fields: LocalFields, context: Mapping[str, Any]) -> dict[str, Any]:
    payload = _safe_context(context)
    payload.update(
        {
            "schema_version": "cvs.phase3.local_evidence.v3",
            "bundle_id": content_root,
            "class_handles": list(LOCAL4_HANDLES),
            "z_id": [float(value) for value in fields.z_id.tolist()],
            "z_dom": [float(value) for value in fields.z_dom.tolist()],
            "q": float(fields.q),
            "d_class": [float(value) for value in fields.d_class.tolist()],
            "e_unknown": float(fields.e_unknown),
            "p_local": [float(value) for value in fields.p_local.tolist()],
            "local_decision": fields.local_decision,
            "local_label": fields.local_label,
            "reason_code": fields.reason_code,
        }
    )
    # First seal the complete, non-timeout evidence.  Missing/conflicting or
    # malformed context therefore has no locally emitted fallback; the caller
    # above this bundle may map that failure to a transport-level defer.
    try:
        sealed = seal_local_evidence(payload)
    except Exception as exc:
        raise _error("CARE context is incomplete, conflicting, or cannot be sealed") from exc
    delay = _require_finite_float(sealed["delay_ms"], field="CARE delay_ms")
    deadline = _require_finite_float(sealed["deadline_ms"], field="CARE deadline_ms")
    if delay > deadline:
        timeout_payload = dict(sealed)
        timeout_payload.update(
            {"local_decision": "defer", "local_label": None, "reason_code": "SCB_CONTEXT_DEFER"}
        )
        timeout_payload.pop("evidence_hash", None)
        try:
            sealed = seal_local_evidence(timeout_payload)
        except Exception as exc:
            raise _error("complete CARE timeout context cannot be sealed") from exc
    try:
        validated = validate_local_evidence(sealed)
    except Exception as exc:
        raise _error("sealed CARE evidence failed local validation") from exc
    if validated["bundle_id"] != content_root or validated["reason_code"] not in ALLOWED_REASON_CODES:
        raise _error("sealed CARE evidence violates the single-control bridge")
    if len(_plain_json_bytes(validated)) > MAX_EVIDENCE_BYTES:
        raise _error("sealed CARE evidence exceeds the frozen single-row byte gate")
    return validated


def _local_evidence_from_model_input(
    *,
    runtime: Any,
    state: BundleState,
    content_root: str,
    model_input: torch.Tensor,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Shared loaded-runtime evidence path used by deployment and measurement."""

    root = _require_sha256(content_root, field="content_root")
    fields = _local_fields_from_runtime(runtime, state, _ensure_model_input(model_input))
    return _evidence_payload(content_root=root, fields=fields, context=context)


def local_evidence_from_bundle(
    bundle: LoadedBundle,
    *,
    raw_iq: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the complete six-field local readout and seal a CARE-v3 evidence row."""

    model_input = preprocess_iq(raw_iq).unsqueeze(0).contiguous()
    return _local_evidence_from_model_input(
        runtime=bundle.runtime,
        state=bundle.state,
        content_root=bundle.content_root,
        model_input=model_input,
        context=context,
    )


def local_evidence_from_components(
    *,
    runtime: Any,
    state: BundleState,
    content_root: str,
    raw_iq: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Eager/reference full-path implementation used only for strict parity tests."""

    root = _require_sha256(content_root, field="content_root")
    model_input = preprocess_iq(raw_iq).unsqueeze(0).contiguous()
    return _local_evidence_from_model_input(
        runtime=runtime,
        state=state,
        content_root=root,
        model_input=model_input,
        context=context,
    )


def assert_full_path_parity(
    *,
    eager_runtime: Any,
    loaded_bundle: LoadedBundle,
    raw_iq: Any,
    context: Mapping[str, Any],
    atol: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare all six fields and decision metadata, not only TorchScript logits."""

    tolerance = _require_finite_float(atol, field="parity atol")
    if tolerance < 0.0:
        raise _error("parity atol must be non-negative")
    before_eager = _runtime_state_digest(eager_runtime, loaded_bundle.state)
    before_loaded = _runtime_state_digest(loaded_bundle.runtime, loaded_bundle.state)
    eager = local_evidence_from_components(
        runtime=eager_runtime,
        state=loaded_bundle.state,
        content_root=loaded_bundle.content_root,
        raw_iq=raw_iq,
        context=context,
    )
    loaded = local_evidence_from_bundle(loaded_bundle, raw_iq=raw_iq, context=context)
    after_eager = _runtime_state_digest(eager_runtime, loaded_bundle.state)
    after_loaded = _runtime_state_digest(loaded_bundle.runtime, loaded_bundle.state)
    if before_eager != after_eager or before_loaded != after_loaded:
        raise _error("runtime state changed during full-path parity check")
    vector_fields = ("z_id", "z_dom", "d_class", "p_local")
    scalar_fields = ("q", "e_unknown")
    for field in vector_fields:
        left = np.asarray(eager[field], dtype=np.float64)
        right = np.asarray(loaded[field], dtype=np.float64)
        if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=tolerance):
            raise _error(f"full-path parity failed for {field}")
    for field in scalar_fields:
        if not math.isclose(float(eager[field]), float(loaded[field]), rel_tol=0.0, abs_tol=tolerance):
            raise _error(f"full-path parity failed for {field}")
    for field in ("local_decision", "local_label", "reason_code", "bundle_id", "class_handles"):
        if eager[field] != loaded[field]:
            raise _error(f"full-path parity failed for {field}")
    return {
        "schema": "cvs.phase1.single_control_full_path_parity.v1",
        "checkpoint_sha256": loaded_bundle.manifest["checkpoint_sha256"],
        "resolved_config_sha256": loaded_bundle.manifest["resolved_config_sha256"],
        "atol": tolerance,
        "state_unchanged": True,
        "fields": ["z_id", "z_dom", "q", "d_class", "e_unknown", "p_local", "local_decision", "local_label", "reason_code"],
        "input_sha256": tensor_sha256(preprocess_iq(raw_iq).unsqueeze(0).contiguous()),
    }


def care_n1_parity(evidence: Mapping[str, Any], *, config: FusionConfig | None = None) -> dict[str, Any]:
    """Prove the CARE N=1 branch is byte/decision compatible with local evidence."""

    validated = validate_local_evidence(evidence)
    result = fuse_event([validated], config or FusionConfig())
    expected = {
        "decision": validated["local_decision"],
        "label": validated.get("local_label"),
        "reason_code": validated["reason_code"],
        "p_fused": validated["p_local"],
    }
    for name, value in expected.items():
        if result.get(name) != value:
            raise _error(f"CARE N=1 parity failed for {name}")
    return {
        "schema": "cvs.phase1.single_control_care_n1_parity.v1",
        "content_root": validated["bundle_id"],
        "evidence_hash": validated["evidence_hash"],
        "decision": result["decision"],
        "label": result["label"],
        "reason_code": result["reason_code"],
        "p_fused": result["p_fused"],
    }


def _project_root_code_hashes(project_root: str | Path) -> tuple[dict[str, str], str, str]:
    root = Path(project_root)
    code_map = {name: sha256_file(root / name) for name in sorted(EXPECTED_CODE_SHA_PATHS)}
    preprocessing = sha256_file(root / "code/dataset_wisig.py")
    scenario = canonical_sha256(
        {
            name: code_map[name]
            for name in (
                "code/training_controls.py",
                "code/cvsrffi/eval.py",
                "code/sat_channel.py",
                "code/cvsrffi/tensors.py",
            )
        }
    )
    return code_map, preprocessing, scenario


def _resolved_ssdg_namespace(checkpoint: Mapping[str, Any], *, device: torch.device) -> Any:
    """Reuse the checkpoint loader's exact merge path to expose its final args."""

    from SSDG import train_ssdg as ssdg

    parser = ssdg.build_arg_parser()
    parsed = parser.parse_args(["--output_dir", ".tmp_scb1_namespace"])
    checkpoint_args = _require_mapping(checkpoint.get("args"), field="checkpoint args")
    for key, value in checkpoint_args.items():
        setattr(parsed, key, value)
    parsed.device = str(device)
    state = strip_module_prefix(_require_mapping(checkpoint.get("model"), field="checkpoint model"))
    merged = ssdg.merge_checkpoint_args(
        checkpoint,
        parsed,
        input_len=INPUT_LEN,
        num_domains=infer_num_domains_from_state(state),
    )
    return ssdg._apply_model_cli_args(merged, parsed)


def _load_manysig_no_default_equalized(pkl_path: str | Path) -> dict[str, Any]:
    """Refuse the dataset loader's historical missing-equalized default."""

    path = Path(pkl_path)
    if path.is_symlink() or not path.is_file():
        raise _error("ManySig PKL must be a regular local file")
    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    except Exception as exc:
        raise _error("ManySig PKL is unreadable") from exc
    if not isinstance(raw, Mapping) or "equalized_list" not in raw:
        raise _error("ManySig PKL must explicitly carry equalized_list; loader defaults are forbidden")
    equalized_values = raw.get("equalized_list")
    if not isinstance(equalized_values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in equalized_values):
        raise _error("ManySig equalized_list must be an explicit integer list")
    if 1 not in equalized_values:
        raise _error("ManySig equalized_list must explicitly contain equalized=1")
    from dataset_wisig import load_wisig_compact_pkl

    loaded = load_wisig_compact_pkl(str(path))
    if list(loaded.get("equalized_list", [])) != list(equalized_values):
        raise _error("ManySig equalized_list changed across strict load")
    return dict(loaded)


def _label_index(values: Sequence[Any], wanted: Sequence[str], *, axis: str) -> list[int]:
    labels = [_require_nfc_string(value, field=f"{axis} physical label") for value in values]
    if len(labels) != len(set(labels)):
        raise _error(f"ManySig {axis} labels are not unique")
    output: list[int] = []
    for label in wanted:
        if label not in labels:
            raise _error(f"frozen {axis} label is absent from ManySig: {label}")
        output.append(labels.index(label))
    return output


def _source_dataset_and_indices(
    *,
    pkl_path: str | Path,
    source_split: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, list[int], list[int], list[int], dict[str, Any]]:
    """Build F1C-local source index in the required TX -> day/RX -> eq=1 order."""

    from dataset_wisig import WiSigCompactDataset
    from SSDG.train_ssdg import _build_source_split_receipt, _phase1_tx_partition_view, split_tx_rx_day_1_6_3

    original = _load_manysig_no_default_equalized(pkl_path)
    source_view, tx_partition = _phase1_tx_partition_view(
        original,
        train_spec=",".join(LOCAL4_HANDLES),
        known_validation_spec=KNOWN_VALIDATION_TX,
        proxy_unknown_spec=PROXY_UNKNOWN_TX,
    )
    if list(source_view.get("tx_list", [])) != list(LOCAL4_HANDLES):
        raise _error("F1C local4 TX partition did not preserve frozen order")
    if source_view.get("equalized_list") != original.get("equalized_list"):
        raise _error("F1C partition drifted equalized labels")
    days = _strict_string_list(source_split["source_days"], field="source source_days", expected=("0", "1"))
    receivers = _strict_string_list(
        source_split["source_receivers"],
        field="source source_receivers",
        expected=("0", "1", "2", "3", "4", "5", "6"),
    )
    day_keep = _label_index(source_view.get("capture_date_list", []), days, axis="day")
    rx_keep = _label_index(source_view.get("rx_list", []), receivers, axis="receiver")
    # Explicit equalized=1 is the only accepted constructor path; neither
    # ``both`` nor a default list is reachable.
    dataset = WiSigCompactDataset(
        source_view,
        out_len=INPUT_LEN,
        crop_mode="center",
        normalize=False,
        center=False,
        equalized=1,
        tx_keep=list(range(len(LOCAL4_HANDLES))),
        rx_keep=rx_keep,
        day_keep=day_keep,
        domain="day",
        build_index=True,
    )
    labeled, unlabeled, calibration = split_tx_rx_day_1_6_3(
        dataset, labeled_ratio=0.07, unlabeled_ratio=0.63, source_val_ratio=0.30
    )
    dataset_sha = sha256_file(pkl_path)
    rebuilt = _build_source_split_receipt(
        seed=7281105,
        split_mode="tx_rx_day_1_6_3",
        source_days=days,
        target_days=source_split["target_days"],
        source_receivers=receivers,
        target_receivers=source_split["target_receivers"],
        labeled_indices=labeled,
        unlabeled_indices=unlabeled,
        source_validation_indices=calibration,
        # The frozen v2 completion receipt predates the external ManySig root
        # and intentionally contains an empty legacy field.  Its byte hash is
        # independently locked; the real dataset SHA is bound outside it.
        wisig_pkl_sha256="",
        requested_labeled_ratio=0.07,
        requested_unlabeled_ratio=0.63,
        requested_source_val_ratio=0.30,
    )
    for field in (
        "labeled_indices_sha256", "unlabeled_indices_sha256", "source_validation_indices_sha256", "split_manifest_sha256"
    ):
        if rebuilt.get(field) != source_split.get(field):
            raise _error(f"reconstructed F1C source split hash drift: {field}")
    for field in ("labeled_size", "unlabeled_size", "source_validation_size", "source_pool_size"):
        if rebuilt.get(field) != source_split.get(field):
            raise _error(f"reconstructed F1C source split size drift: {field}")
    return source_view, dataset, labeled, unlabeled, calibration, tx_partition


def _validate_source_view_indices(
    labeled_indices: Sequence[Any],
    unlabeled_indices: Sequence[Any],
    calibration_indices: Sequence[Any],
) -> tuple[list[int], list[int], list[int]]:
    """Freeze disjoint label-blind source sample identities for all view RNG."""

    groups: list[list[int]] = []
    for name, values in (("L", labeled_indices), ("U", unlabeled_indices), ("V", calibration_indices)):
        parsed = [_require_nonbool_int(value, field=f"{name} opaque source index", minimum=0) for value in values]
        if len(parsed) != len(set(parsed)):
            raise _error(f"{name} source view indices contain a duplicate opaque sample identity")
        if not parsed:
            raise _error(f"{name} source view indices are empty")
        groups.append(parsed)
    if set(groups[0]).intersection(groups[1]) or set(groups[0]).intersection(groups[2]) or set(groups[1]).intersection(groups[2]):
        raise _error("L/U/V source view indices must be pairwise disjoint")
    return groups[0], groups[1], groups[2]


def _dataset_item_raw(
    *,
    source_iq_data: Any,
    equalized_values: Any,
    dataset: Any,
    index: int,
) -> np.ndarray:
    """Read one source IQ without touching a class/TX-label table.

    This deliberately accepts only the raw nested data object, the explicit
    equalized registry, and an opaque dataset index.  The U descriptor path
    calls this helper directly; its view seed, opaque hash, and numerical
    aggregation consequently have no label-bearing input.
    """

    item = dataset.index[_require_nonbool_int(index, field="dataset index", minimum=0)]
    if not isinstance(equalized_values, list) or int(item.eq_i) >= len(equalized_values) or equalized_values[int(item.eq_i)] != 1:
        raise _error("source index must explicitly select equalized=1")
    try:
        raw = np.asarray(source_iq_data[int(item.tx_i)][int(item.rx_i)][int(item.day_i)][int(item.eq_i)][int(item.sig_i)])
    except (IndexError, KeyError, TypeError) as exc:
        raise _error("source index cannot resolve raw ManySig IQ") from exc
    return raw


def _dataset_item_physical_key(source_view: Mapping[str, Any], dataset: Any, index: int) -> tuple[str, str, str, int]:
    """Read the physical key only for the separate partition-audit path."""

    item = dataset.index[_require_nonbool_int(index, field="dataset index", minimum=0)]
    try:
        return physical_key(
            source_view["tx_list"][int(item.tx_i)],
            source_view["rx_list"][int(item.rx_i)],
            source_view["capture_date_list"][int(item.day_i)],
            item.sig_i,
        )
    except (IndexError, KeyError, TypeError) as exc:
        raise _error("source index cannot resolve physical key") from exc


def _dataset_item_key_and_raw(source_view: Mapping[str, Any], dataset: Any, index: int) -> tuple[tuple[str, str, str, int], np.ndarray]:
    """Compatibility helper for labelled L/V and partition-audit paths."""

    return (
        _dataset_item_physical_key(source_view, dataset, index),
        _dataset_item_raw(
            source_iq_data=source_view.get("data"),
            equalized_values=source_view.get("equalized_list"),
            dataset=dataset,
            index=index,
        ),
    )


def _scb_views_for_source_sample(
    *,
    raw_iq: Any,
    opaque_sample_index: Any,
    args: Any,
) -> list[torch.Tensor]:
    """Build clean plus deterministic label-blind source views with no TTA."""

    from cvsrffi.eval import apply_sat_channel_for_scenario
    from training_controls import sat_channel_config_for_scenario

    # Frozen SCB1 channel generation is always batch-1 CPU.  Moving the
    # resulting tensors to a runtime device happens only after view bytes are
    # fixed, so CUDA RNG state cannot influence a calibration view.
    clean_cpu = preprocess_iq(raw_iq).unsqueeze(0).contiguous()
    views = [clean_cpu]
    for scenario in SCENARIOS:
        config = sat_channel_config_for_scenario(scenario)
        if config.get("channel_model") != "leo_residual":
            raise _error(f"frozen satellite scenario implementation drift: {scenario}")
        if float(getattr(args, "sat_fs_hz", 25000000.0)) != 25000000.0 or float(
            getattr(args, "sat_fc_hz", 2462000000.0)
        ) != 2462000000.0:
            raise _error(f"frozen satellite scenario frequency drift: {scenario}")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            derive_source_view_seed(
                split_seed=7281105,
                opaque_sample_index=opaque_sample_index,
                scenario=scenario,
            )
        )
        satellite, _ = apply_sat_channel_for_scenario(clean_cpu, scenario, args, gen=generator, return_meta=False)
        views.append(_ensure_model_input(satellite.detach().cpu().contiguous()))
    if len(views) != len(VIEW_ORDER):
        raise _error("SCB view count drift")
    return views


def _resource_context() -> dict[str, Any]:
    return {
        "linkage_mode": "proxy_unverified",
        "proxy_group_id": "scb-resource-probe",
        "satellite_reception_id": "scb-resource-reception",
        "node_id": "scb-resource-node",
        "base_manifest_id": "scb-resource-base",
        "correlation_group_id": "scb-resource-group",
        "delay_ms": 0.0,
        "deadline_ms": 1.0,
        "sealed_at_ms": 0.0,
    }


def _rss_bytes() -> int:
    """Read this process RSS without making ``psutil`` a deployment dependency."""

    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except Exception:
            pass
    if os.name == "nt":  # pragma: no cover - exercised on Windows runner.
        try:
            import ctypes
            import ctypes.wintypes as wintypes

            class _ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
            get_memory_info = psapi.GetProcessMemoryInfo
            get_memory_info.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCountersEx),
                wintypes.DWORD,
            )
            get_memory_info.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            if get_memory_info(process, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:
            pass
    raise _error("resource measurement cannot obtain process RSS")


_RESOURCE_RUNTIME_NAME = "local_evidence.ts"
_RESOURCE_GEOMETRY_NAME = "class_geometry.npz"
_RESOURCE_DESCRIPTOR_NAME = "domain_descriptor_stats.npz"
_RESOURCE_TAIL_NAME = "rank_tail_summary.npz"
_RESOURCE_RESULT_NAME = "resource_receipt.json"


def _resource_worker_paths(payload_dir: str | Path) -> tuple[Path, Path, Path, Path, Path]:
    root = Path(payload_dir)
    if root.is_symlink() or not root.is_dir():
        raise _error("resource worker payload root must be a real directory")
    paths = tuple(root / name for name in (
        _RESOURCE_RUNTIME_NAME, _RESOURCE_GEOMETRY_NAME, _RESOURCE_DESCRIPTOR_NAME, _RESOURCE_TAIL_NAME, _RESOURCE_RESULT_NAME
    ))
    for path in paths[:-1]:
        if path.is_symlink() or not path.is_file():
            raise _error("resource worker payload is missing a regular member")
    return paths  # type: ignore[return-value]


def _resource_probe_subprocess(payload_dir: str | Path) -> None:
    """Fresh-process resource probe used only by :func:`make_resource_receipt`."""

    runtime_path, geometry_path, descriptor_path, tail_path, result_path = _resource_worker_paths(payload_dir)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Imports have completed before this point; no runtime or NPZ payload has
        # been read when the baseline is taken.
        baseline = _rss_bytes()
        probe = resource_probe_model_input()
        runtime_bytes = _read_regular_file(runtime_path, context="resource runtime payload")
        geometry_raw = _read_regular_file(geometry_path, context="resource geometry payload")
        descriptor_raw = _read_regular_file(descriptor_path, context="resource descriptor payload")
        tail_raw = _read_regular_file(tail_path, context="resource tail payload")
        state = state_from_npz_bytes(geometry_raw, descriptor_raw, tail_raw)
        try:
            cpu_runtime = torch.jit.load(io.BytesIO(runtime_bytes), map_location="cpu").eval()
        except Exception as exc:
            raise _error("resource probe cannot CPU-load TorchScript runtime") from exc
        cpu_before = _runtime_state_digest(cpu_runtime, state)
        cpu_peak = max(baseline, _rss_bytes())
        evidence: dict[str, Any] | None = None
        for _ in range(CPU_WARMUPS):
            evidence = _local_evidence_from_model_input(
                runtime=cpu_runtime, state=state, content_root="0" * 64, model_input=probe, context=_resource_context()
            )
            cpu_peak = max(cpu_peak, _rss_bytes())
        timings: list[float] = []
        for _ in range(CPU_TRIALS):
            started = time.perf_counter_ns()
            evidence = _local_evidence_from_model_input(
                runtime=cpu_runtime, state=state, content_root="0" * 64, model_input=probe, context=_resource_context()
            )
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
            cpu_peak = max(cpu_peak, _rss_bytes())
        if evidence is None:
            raise _error("resource probe produced no local evidence")
        cpu_after = _runtime_state_digest(cpu_runtime, state)
        if cpu_before != cpu_after:
            raise _error("resource CPU runtime/state changed during measurement")

        cuda_available = bool(torch.cuda.is_available())
        cuda_peak = 0
        cuda_p99 = 0.0
        if cuda_available:
            cuda_device = torch.device("cuda")
            # Torch 2.10 accepts no positional device argument here.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cuda_device)
            try:
                cuda_runtime = torch.jit.load(io.BytesIO(runtime_bytes), map_location=cuda_device).eval()
            except Exception as exc:
                raise _error("resource probe cannot CUDA-load TorchScript runtime") from exc
            cuda_before = _runtime_state_digest(cuda_runtime, state)
            for _ in range(CPU_WARMUPS):
                _local_evidence_from_model_input(
                    runtime=cuda_runtime, state=state, content_root="0" * 64, model_input=probe, context=_resource_context()
                )
            cuda_timings: list[float] = []
            for _ in range(CPU_TRIALS):
                torch.cuda.synchronize(cuda_device)
                started = time.perf_counter_ns()
                _local_evidence_from_model_input(
                    runtime=cuda_runtime, state=state, content_root="0" * 64, model_input=probe, context=_resource_context()
                )
                torch.cuda.synchronize(cuda_device)
                cuda_timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
            if cuda_before != _runtime_state_digest(cuda_runtime, state):
                raise _error("resource CUDA runtime/state changed during measurement")
            cuda_peak = int(torch.cuda.max_memory_allocated(cuda_device))
            cuda_p99 = _quantile_higher(np.asarray(cuda_timings, dtype=np.float64), 0.99)
        receipt = {
            "schema": "cvs.phase1.single_control_resource_receipt.v1",
            "input_shape": [1, 2, INPUT_LEN],
            "input_dtype": "torch.float32",
            "input_sha256": tensor_sha256(resource_probe_model_input()),
            "input_seed": RESOURCE_INPUT_SEED,
            "torch_num_threads": 1,
            "cpu_rss_baseline_bytes": int(baseline),
            "cpu_rss_peak_bytes": int(cpu_peak),
            "cpu_rss_delta_bytes": int(max(0, cpu_peak - baseline)),
            "cpu_warmups": CPU_WARMUPS,
            "cpu_trials": CPU_TRIALS,
            "cpu_latency_p99_ms": float(_quantile_higher(np.asarray(timings, dtype=np.float64), 0.99)),
            "cpu_latency_quantile_method": "higher_q99_100",
            "cuda_available": cuda_available,
            "cuda_peak_bytes": int(cuda_peak),
            "cuda_latency_recorded": cuda_available,
            "cuda_latency_p99_ms": float(cuda_p99),
            "measurement_scope": "fresh_process_bundle_load_warmup_full_local_evidence",
            "evidence_bytes": len(_plain_json_bytes(evidence)),
            "measurement_process": "fresh_python_subprocess_v1",
            "baseline_before_payload_load": True,
            "state_payload_reloaded": True,
            "runtime_state_before_sha256": cpu_before,
            "runtime_state_after_sha256": cpu_after,
        }
        validated = validate_resource_receipt(receipt)
        result_path.write_bytes(_plain_json_bytes(validated))
    finally:
        torch.set_num_threads(original_threads)


def make_resource_receipt(
    *,
    runtime_bytes: bytes,
    state: BundleState,
    device: torch.device,
) -> dict[str, Any]:
    """Run the resource gate in a clean Python process with only package bytes."""

    del device  # Availability is discovered in the fresh measurement process.
    if not isinstance(runtime_bytes, bytes) or not runtime_bytes:
        raise _error("resource measurement requires non-empty TorchScript bytes")
    with tempfile.TemporaryDirectory(prefix="scb1-resource-") as temporary:
        root = Path(temporary)
        (root / _RESOURCE_RUNTIME_NAME).write_bytes(runtime_bytes)
        (root / _RESOURCE_GEOMETRY_NAME).write_bytes(_npz_bytes(geometry_npz_arrays(state.geometry)))
        (root / _RESOURCE_DESCRIPTOR_NAME).write_bytes(_npz_bytes(descriptor_npz_arrays(state.descriptor)))
        (root / _RESOURCE_TAIL_NAME).write_bytes(_npz_bytes(tail_npz_arrays(state.tail)))
        code_root = str(Path(__file__).resolve().parents[1])
        environment = dict(os.environ)
        environment["PYTHONPATH"] = code_root + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, "-m", "cvsrffi.phase1_single_control_bundle_v1", "--resource-probe", str(root)],
            cwd=code_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            check=False,
        )
        if completed.returncode != 0:
            raise _error("fresh resource subprocess failed")
        result_path = root / _RESOURCE_RESULT_NAME
        try:
            receipt = _require_mapping(json.loads(_read_regular_file(result_path, context="resource worker result").decode("utf-8")), field="resource worker result")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _error("fresh resource subprocess returned unreadable result") from exc
        return validate_resource_receipt(receipt)


def make_runtime_parity_receipt(
    *,
    eager_runtime: Any,
    runtime: Any,
    state: BundleState,
    checkpoint_sha256: str,
    resolved_config_digest: str,
    runtime_bytes: bytes,
    raw_iq: Any | None = None,
    atol: float = 1.0e-6,
) -> dict[str, Any]:
    """Generate an auditable six-field parity receipt on one real source IQ."""

    tolerance = _require_finite_float(atol, field="parity atol")
    raw = resource_probe_model_input()[0] if raw_iq is None else raw_iq
    model_input = preprocess_iq(raw).unsqueeze(0).contiguous()
    root = "0" * 64
    eager_before = _runtime_state_digest(eager_runtime, state)
    loaded_before = _runtime_state_digest(runtime, state)
    eager = local_evidence_from_components(
        runtime=eager_runtime, state=state, content_root=root, raw_iq=raw, context=_resource_context()
    )
    loaded = local_evidence_from_components(
        runtime=runtime, state=state, content_root=root, raw_iq=raw, context=_resource_context()
    )
    eager_after = _runtime_state_digest(eager_runtime, state)
    loaded_after = _runtime_state_digest(runtime, state)
    max_abs: dict[str, float] = {}
    for field in ("z_id", "z_dom", "d_class", "p_local"):
        left = np.asarray(eager[field], dtype=np.float64)
        right = np.asarray(loaded[field], dtype=np.float64)
        max_abs[field] = float(np.max(np.abs(left - right)))
    for field in ("q", "e_unknown"):
        max_abs[field] = abs(float(eager[field]) - float(loaded[field]))
    if any(not math.isfinite(value) or value > tolerance for value in max_abs.values()):
        raise _error("fixed-IQ full-path numerical parity failed")
    decision_equal = eager["local_decision"] == loaded["local_decision"]
    label_equal = eager["local_label"] == loaded["local_label"]
    reason_equal = eager["reason_code"] == loaded["reason_code"]
    bundle_equal = eager["bundle_id"] == loaded["bundle_id"]
    handles_equal = eager["class_handles"] == loaded["class_handles"]
    if not all((decision_equal, label_equal, reason_equal, bundle_equal, handles_equal)):
        raise _error("fixed-IQ full-path categorical parity failed")
    if eager_before != eager_after or loaded_before != loaded_after:
        raise _error("fixed-IQ full-path parity observed state mutation")
    return {
        "schema": "cvs.phase1.single_control_runtime_parity.v1",
        "checkpoint_sha256": _require_sha256(checkpoint_sha256, field="checkpoint SHA"),
        "resolved_config_sha256": _require_sha256(resolved_config_digest, field="resolved config SHA"),
        "runtime_sha256": _sha256_bytes(runtime_bytes),
        "runtime_state_schema_sha256": runtime_state_schema(runtime)["runtime_state_schema_sha256"],
        "input_sha256": tensor_sha256(model_input),
        "atol": tolerance,
        "state_unchanged": True,
        "strict_load_complete": True,
        "full_path_fields": ["z_id", "z_dom", "q", "d_class", "e_unknown", "p_local", "local_decision", "local_label", "reason_code"],
        "max_abs": max_abs,
        "decision_equal": True,
        "label_equal": True,
        "reason_equal": True,
        "bundle_id_equal": True,
        "class_handles_equal": True,
        "eager_state_before_sha256": eager_before,
        "eager_state_after_sha256": eager_after,
        "loaded_state_before_sha256": loaded_before,
        "loaded_state_after_sha256": loaded_after,
        "runtime_batch_capacity": 1,
        "runtime_internal_padding": False,
        "raw_model_direct_identity_parity": True,
    }


def _identity_features_for_views(runtime: Any, views: Sequence[torch.Tensor], *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    if len(views) != len(VIEW_ORDER):
        raise _error("identity feature extraction must receive clean plus three LEO views")
    z_rows: list[np.ndarray] = []
    logit_rows: list[np.ndarray] = []
    # The exported deployment runtime has batch-1 capacity by contract.  This
    # source-only builder deliberately invokes each frozen view independently.
    for view in views:
        batch = _ensure_model_input(view).to(device=device).contiguous()
        with torch.no_grad():
            z_id, logits = _runtime_outputs(runtime, batch)
        z_rows.append(z_id[0].detach().cpu().numpy().astype(np.float64))
        logit_rows.append(logits[0].detach().cpu().numpy().astype(np.float64))
    return np.vstack(z_rows), np.vstack(logit_rows)


def _enumerate_role_physical_keys(
    dataset: Mapping[str, Any],
    *,
    tx_labels: Sequence[str],
    rx_labels: Sequence[str],
    day_labels: Sequence[str],
) -> list[tuple[str, str, str, int]]:
    tx_indices = _label_index(dataset.get("tx_list", []), tx_labels, axis="TX")
    rx_indices = _label_index(dataset.get("rx_list", []), rx_labels, axis="receiver")
    day_indices = _label_index(dataset.get("capture_date_list", []), day_labels, axis="day")
    eq_values = dataset.get("equalized_list")
    if not isinstance(eq_values, list) or 1 not in eq_values:
        raise _error("role enumeration requires explicit equalized=1")
    eq_i = eq_values.index(1)
    keys: list[tuple[str, str, str, int]] = []
    for tx_i in tx_indices:
        for rx_i in rx_indices:
            for day_i in day_indices:
                values = dataset["data"][tx_i][rx_i][day_i][eq_i]
                if values is None:
                    continue
                for sig_i in range(int(np.asarray(values).shape[0])):
                    keys.append(
                        physical_key(
                            dataset["tx_list"][tx_i],
                            dataset["rx_list"][rx_i],
                            dataset["capture_date_list"][day_i],
                            sig_i,
                        )
                    )
    if len(keys) != len({physical_token(key) for key in keys}):
        raise _error("role enumeration discovered duplicate physical keys")
    return keys


def build_real_bundle_from_paths(
    *,
    project_root: str | Path,
    checkpoint_path: str | Path,
    wisig_pkl_path: str | Path,
    completion_receipt_path: str | Path,
    terminal_receipt_path: str | Path,
    cp_terminal_receipt_path: str | Path,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """The future real-input builder; missing or drifting evidence fails closed.

    It deliberately never loads proxy/held/target IQ into a model.  Those roles
    are enumerated only as transient physical-key sets for zero-intersection
    proof, and no individual key survives in the bundle.
    """

    # This must precede every checkpoint/dataset hash, receipt read, or model
    # reconstruction so an accidental output collision never consumes inputs.
    _assert_new_bundle_output_target(output_dir)
    runtime_device = torch.device(device)
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_symlink() or not checkpoint_file.is_file():
        raise _error("real F1C checkpoint must be a regular local file")
    checkpoint_sha = sha256_file(checkpoint_file)
    dataset_sha = sha256_file(wisig_pkl_path)
    receipts = validate_f1c_receipts(
        completion_path=completion_receipt_path,
        terminal_path=terminal_receipt_path,
        cp_terminal_path=cp_terminal_receipt_path,
        checkpoint_sha256=checkpoint_sha,
        dataset_sha256=dataset_sha,
        fixture_mode=False,
    )
    try:
        checkpoint = torch.load(checkpoint_file, map_location=runtime_device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility for earlier torch.
        checkpoint = torch.load(checkpoint_file, map_location=runtime_device)
    checkpoint_map = _require_mapping(checkpoint, field="F1C checkpoint")
    namespace = _resolved_ssdg_namespace(checkpoint_map, device=runtime_device)
    model_config = resolve_model_config_projection(checkpoint=checkpoint_map, resolved_namespace=namespace)
    code_hashes, preprocessing_sha, scenario_code_sha = _project_root_code_hashes(project_root)
    config_sha = resolved_config_sha256(
        receipt_projection=receipts,
        checkpoint_sha256=checkpoint_sha,
        dataset_sha256=dataset_sha,
        model_config=model_config,
        code_sha256=code_hashes,
        preprocessing_code_sha256=preprocessing_sha,
        scenario_code_sha256=scenario_code_sha,
    )
    source_split = receipts["completion"]["source_split_receipt"]
    source_view, dataset, labeled_indices, unlabeled_indices, calibration_indices, tx_partition = _source_dataset_and_indices(
        pkl_path=wisig_pkl_path, source_split=source_split
    )
    with tempfile.TemporaryDirectory(prefix="scb1-runtime-") as temporary:
        runtime_path = Path(temporary) / "local_evidence.ts"
        runtime, eager_runtime, strict_audit = build_torchscript_runtime(
            checkpoint=checkpoint_map,
            device=runtime_device,
            runtime_path=runtime_path,
        )
        runtime_bytes = runtime_path.read_bytes()
        labeled_rows: list[dict[str, Any]] = []
        labeled_keys: list[tuple[str, str, str, int]] = []
        unlabeled_keys: list[tuple[str, str, str, int]] = []
        calibration_keys: list[tuple[str, str, str, int]] = []
        first_raw: np.ndarray | None = None
        labeled_indices, unlabeled_indices, calibration_indices = _validate_source_view_indices(
            labeled_indices, unlabeled_indices, calibration_indices
        )
        source_iq_data = source_view.get("data")
        source_equalized_values = source_view.get("equalized_list")

        def descriptor_row_stream() -> Iterable[Mapping[str, Any]]:
            for descriptor_index in labeled_indices:
                descriptor_raw = _dataset_item_raw(
                    source_iq_data=source_iq_data,
                    equalized_values=source_equalized_values,
                    dataset=dataset,
                    index=descriptor_index,
                )
                descriptor_views = _scb_views_for_source_sample(
                    raw_iq=descriptor_raw, opaque_sample_index=descriptor_index, args=namespace
                )
                yield {
                    "opaque_hash": opaque_source_sample_hash(descriptor_index),
                    "iq_views": [view[0].cpu().contiguous() for view in descriptor_views],
                }
            for descriptor_index in unlabeled_indices:
                descriptor_raw = _dataset_item_raw(
                    source_iq_data=source_iq_data,
                    equalized_values=source_equalized_values,
                    dataset=dataset,
                    index=descriptor_index,
                )
                descriptor_views = _scb_views_for_source_sample(
                    raw_iq=descriptor_raw, opaque_sample_index=descriptor_index, args=namespace
                )
                yield {
                    "opaque_hash": opaque_source_sample_hash(descriptor_index),
                    "iq_views": [view[0].cpu().contiguous() for view in descriptor_views],
                }

        for index in labeled_indices:
            key, raw = _dataset_item_key_and_raw(source_view, dataset, index)
            views = _scb_views_for_source_sample(raw_iq=raw, opaque_sample_index=index, args=namespace)
            z_views, _ = _identity_features_for_views(runtime, views, device=runtime_device)
            token = physical_token(key)
            labeled_rows.append({"physical_token": token, "label": key[0], "z_views": z_views})
            labeled_keys.append(key)
            if first_raw is None:
                first_raw = raw
        geometry = fit_class_geometry(labeled_rows)
        for index in unlabeled_indices:
            key = _dataset_item_physical_key(source_view, dataset, index)
            unlabeled_keys.append(key)
        descriptor = fit_descriptor_stats(descriptor_row_stream())
        distance_atoms: list[float] = []
        energy_atoms: list[float] = []
        domain_atoms: list[float] = []
        for index in calibration_indices:
            key, raw = _dataset_item_key_and_raw(source_view, dataset, index)
            views = _scb_views_for_source_sample(raw_iq=raw, opaque_sample_index=index, args=namespace)
            z_views, logits = _identity_features_for_views(runtime, views, device=runtime_device)
            per_view_distance: list[float] = []
            per_view_energy: list[float] = []
            per_view_domain: list[float] = []
            for view_index, view in enumerate(views):
                _, distance = score_class_geometry(z_views[view_index], geometry)
                per_view_distance.append(distance)
                per_view_energy.append(-_logsumexp(logits[view_index]))
                _, domain = normalize_descriptor(domain_descriptor(view[0].cpu().contiguous()), descriptor)
                per_view_domain.append(domain)
            distance_atoms.append(max(per_view_distance))
            energy_atoms.append(max(per_view_energy))
            domain_atoms.append(max(per_view_domain))
            calibration_keys.append(key)
        tail = fit_tail_summary(
            distance_scores=distance_atoms,
            energy_scores=energy_atoms,
            domain_scores=domain_atoms,
            calibration_set_sha256=physical_set_sha256(calibration_keys),
        )
        full_dataset = _load_manysig_no_default_equalized(wisig_pkl_path)
        source_days = tuple(source_split["source_days"])
        source_receivers = tuple(source_split["source_receivers"])
        excluded = {
            "proxy": _enumerate_role_physical_keys(
                full_dataset, tx_labels=(PROXY_UNKNOWN_TX,), rx_labels=source_receivers, day_labels=source_days
            ),
            "held": _enumerate_role_physical_keys(
                full_dataset, tx_labels=(KNOWN_VALIDATION_TX,), rx_labels=source_receivers, day_labels=source_days
            ),
            "target": _enumerate_role_physical_keys(
                full_dataset,
                tx_labels=LOCAL4_HANDLES,
                rx_labels=tuple(source_split["target_receivers"]),
                day_labels=tuple(source_split["target_days"]),
            ),
        }
        partition = build_source_partition_receipt(
            dataset_sha256=dataset_sha,
            source_split_projection=source_split,
            tx_partition_receipt=tx_partition,
            labeled_keys=labeled_keys,
            unlabeled_keys=unlabeled_keys,
            calibration_keys=calibration_keys,
            excluded_role_keys=excluded,
        )
        state = BundleState(geometry=geometry, descriptor=descriptor, tail=tail)
        checkpoint_binding = {
            "checkpoint_sha256": checkpoint_sha,
            "resolved_config_sha256": config_sha,
            "checkpoint_role": "training_final_only",
            "strict_state_tensor_schema_sha256": model_config["strict_state_tensor_schema_sha256"],
            "strict_load_audit": {
                "strict": bool(strict_audit["checkpoint_load_strict"]),
                "missing_keys": [],
                "unexpected_keys": [],
            },
        }
        cp = receipts["cp_terminal"]
        class_binding = {
            "class_handles": list(LOCAL4_HANDLES),
            "local_to_head_class_ids": [0, 1, 2, 3],
            "class_order_binding_sha256": cp["class_order_binding_sha256"],
            "checkpoint_head_class_count": cp["checkpoint_head_class_count"],
            "live_head_class_count": cp["live_head_class_count"],
            "checkpoint_train_tx_class_order": cp["checkpoint_train_tx_class_order"],
        }
        parity = make_runtime_parity_receipt(
            eager_runtime=eager_runtime,
            runtime=runtime,
            state=state,
            checkpoint_sha256=checkpoint_sha,
            resolved_config_digest=config_sha,
            runtime_bytes=runtime_bytes,
            raw_iq=first_raw,
        )
        resource = make_resource_receipt(runtime_bytes=runtime_bytes, state=state, device=runtime_device)
        result = build_bundle(
            output_dir=output_dir,
            runtime_source=runtime_bytes,
            state=state,
            checkpoint_binding=checkpoint_binding,
            class_binding=class_binding,
            source_partition_receipt=partition,
            runtime_parity_receipt=parity,
            resource_receipt=resource,
            checkpoint_sha256=checkpoint_sha,
            resolved_config_digest=config_sha,
            dataset_sha256=dataset_sha,
            preprocessing_code_sha256=preprocessing_sha,
            scenario_registry_sha256=receipts["satellite_protocol"]["registry_sha256"],
            bundle_status=BUNDLE_STATUS,
        )
        if first_raw is None:
            raise _error("real source partition has no labelled raw IQ for final smoke")
        loaded = load_bundle(
            output_dir,
            expected_content_root=result["content_root"],
            device=runtime_device,
            expected_bundle_status=BUNDLE_STATUS,
        )
        smoke_context = dict(_resource_context())
        smoke_context["satellite_reception_id"] = "scb-real-smoke-reception"
        assert_full_path_parity(
            eager_runtime=eager_runtime,
            loaded_bundle=loaded,
            raw_iq=first_raw,
            context=smoke_context,
        )
        return result


def _module_main(argv: Sequence[str]) -> int:
    if len(argv) == 2 and argv[0] == "--resource-probe":
        _resource_probe_subprocess(argv[1])
        return 0
    raise SystemExit("single-control core accepts only internal --resource-probe")


if __name__ == "__main__":  # pragma: no cover - invoked by the fresh worker.
    raise SystemExit(_module_main(sys.argv[1:]))
