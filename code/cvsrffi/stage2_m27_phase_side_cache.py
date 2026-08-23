"""Truth-free Phase32 side cache for the ERBT-IDR M2.7 veto arm."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS


PHASE_SIDE_CACHE_SCHEMA = "cvs.erbt_idr.m27.phase_side_cache.v1"
PHASE_SIDE_MANIFEST_SCHEMA = "cvs.erbt_idr.m27.phase_side_cache_manifest.v1"
PHASE_DIM = 32

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLASS_HANDLE = re.compile(r"^cls_[0-9a-f]{32,64}$")
_QUERY_TOKEN = re.compile(r"^qid_[0-9a-f]{32,64}$")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_FIELDS = (
    "old_support_phase32",
    "old_support_labels",
    "new_support_phase32",
    "new_support_labels",
    "query_phase32",
    "query_tokens",
)
_EPS = 1.0e-12


class M27PhaseSideCacheError(ValueError):
    """Raised when a Phase32 view or side cache fails closed."""


def _hash(value: str, *, name: str) -> str:
    result = str(value).strip().lower()
    if _SHA256.fullmatch(result) is None:
        raise M27PhaseSideCacheError(f"{name} must be SHA256")
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _phase_matrix(value: Any, *, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[0] <= 0
        or rows.shape[1] != PHASE_DIM
        or not np.isfinite(rows).all()
    ):
        raise M27PhaseSideCacheError(f"{name} must be finite N x {PHASE_DIM}")
    return rows


def _strings(value: Any, *, name: str, pattern: re.Pattern[str]) -> np.ndarray:
    rows = np.asarray(value)
    if rows.ndim != 1 or rows.dtype.kind not in {"U", "S", "O"}:
        raise M27PhaseSideCacheError(f"{name} must be a string vector")
    result = rows.astype(str)
    if any(pattern.fullmatch(item) is None for item in result.tolist()):
        raise M27PhaseSideCacheError(f"{name} value drift")
    return result


def _unit_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norm <= _EPS):
        raise M27PhaseSideCacheError("phase descriptor is degenerate")
    return value / norm


def _band_circular_mean(angle: np.ndarray, weight: np.ndarray) -> np.ndarray:
    result = np.empty((len(angle), 8), dtype=np.float64)
    for index, members in enumerate(np.array_split(np.arange(angle.shape[1]), 8)):
        if len(members) == 0:
            result[:, index] = 0.0
            continue
        selected_weight = weight[:, members]
        sine = np.sum(selected_weight * np.sin(angle[:, members]), axis=1)
        cosine = np.sum(selected_weight * np.cos(angle[:, members]), axis=1)
        result[:, index] = np.arctan2(sine, cosine) / np.pi
    return result


def _band_mean(value: np.ndarray) -> np.ndarray:
    result = np.empty((len(value), 8), dtype=np.float64)
    for index, members in enumerate(np.array_split(np.arange(value.shape[1]), 8)):
        result[:, index] = np.mean(value[:, members], axis=1) if len(members) else 0.0
    return result


def _normalise_group(value: np.ndarray) -> np.ndarray:
    centred = value - np.mean(value, axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1, keepdims=True)
    return centred / np.maximum(norm, _EPS)


def phase_coherence32(iq: Any) -> np.ndarray:
    """Create a gain/global-phase invariant 32D phase/cepstral descriptor."""

    rows = np.asarray(iq)
    if rows.ndim == 2 and np.iscomplexobj(rows):
        complex_rows = rows.astype(np.complex128)
    elif rows.ndim == 3 and rows.shape[1] == 2:
        complex_rows = rows[:, 0].astype(np.float64) + 1j * rows[:, 1].astype(np.float64)
    elif rows.ndim == 3 and rows.shape[2] == 2:
        complex_rows = rows[:, :, 0].astype(np.float64) + 1j * rows[:, :, 1].astype(np.float64)
    else:
        raise M27PhaseSideCacheError("IQ must be complex N x T or real N x 2 x T")
    if (
        complex_rows.shape[0] <= 0
        or complex_rows.shape[1] < 64
        or not np.isfinite(complex_rows.real).all()
        or not np.isfinite(complex_rows.imag).all()
    ):
        raise M27PhaseSideCacheError("IQ rows must be finite with at least 64 samples")
    complex_rows = complex_rows - np.mean(complex_rows, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.abs(complex_rows) ** 2, axis=1, keepdims=True))
    if np.any(rms <= _EPS):
        raise M27PhaseSideCacheError("IQ row energy is degenerate")
    complex_rows = complex_rows / rms
    window = np.hanning(complex_rows.shape[1])[None, :]
    spectrum = np.fft.fftshift(np.fft.fft(complex_rows * window, axis=1), axes=1)
    magnitude = np.abs(spectrum)

    adjacent = spectrum[:, 1:] * np.conj(spectrum[:, :-1])
    adjacent_weight = np.sqrt(magnitude[:, 1:] * magnitude[:, :-1])
    adjacent_phase = _band_circular_mean(np.angle(adjacent), adjacent_weight)

    unit_adjacent = adjacent / np.maximum(np.abs(adjacent), _EPS)
    curvature = unit_adjacent[:, 1:] * np.conj(unit_adjacent[:, :-1])
    curvature_weight = np.minimum(adjacent_weight[:, 1:], adjacent_weight[:, :-1])
    curvature_phase = _band_circular_mean(np.angle(curvature), curvature_weight)

    centre = spectrum.shape[1] // 2
    offsets = np.arange(1, centre, dtype=np.int64)
    positive = centre + offsets
    negative = centre - offsets
    valid = positive < spectrum.shape[1]
    positive = positive[valid]
    negative = negative[valid]
    mirror = spectrum[:, positive] * np.conj(spectrum[:, negative])
    mirror_weight = np.sqrt(magnitude[:, positive] * magnitude[:, negative])
    mirror_phase = _band_circular_mean(np.angle(mirror), mirror_weight)

    log_magnitude = np.log(np.maximum(magnitude, _EPS))
    log_magnitude -= np.mean(log_magnitude, axis=1, keepdims=True)
    cepstrum = np.fft.ifft(np.fft.ifftshift(log_magnitude, axes=1), axis=1).real
    upper = min(65, cepstrum.shape[1] // 2)
    cepstral_derivative = np.diff(cepstrum[:, 1:upper], axis=1)
    cepstral = _band_mean(cepstral_derivative)

    descriptor = np.concatenate(
        [
            _normalise_group(adjacent_phase),
            _normalise_group(curvature_phase),
            _normalise_group(mirror_phase),
            _normalise_group(cepstral),
        ],
        axis=1,
    )
    return _unit_rows(descriptor).astype(np.float32)


def _write_exclusive_readonly(path: Path, value: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing Phase32 side cache")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def _payload_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    value = buffer.getvalue()
    try:
        with zipfile.ZipFile(io.BytesIO(value), "r") as archive:
            names = [item.filename for item in archive.infolist()]
            expected = [f"{name}.npy" for name in arrays]
            if len(names) != len(set(names)) or set(names) != set(expected) or archive.testzip() is not None:
                raise M27PhaseSideCacheError("Phase32 NPZ member drift")
    except zipfile.BadZipFile as exc:
        raise M27PhaseSideCacheError("Phase32 NPZ is invalid") from exc
    return value


def _balanced(labels: np.ndarray, classes: tuple[str, ...], k_shot: int, *, name: str) -> None:
    if set(labels.tolist()) != set(classes) or any(
        int(np.sum(labels == item)) != int(k_shot) for item in classes
    ):
        raise M27PhaseSideCacheError(f"{name} must be class-symmetric K-shot")


def publish_phase_side_cache(
    payload_path: str | Path,
    manifest_path: str | Path,
    *,
    base_manifest_sha256: str,
    capsule_id: str,
    split_id: str,
    receiver: str,
    method_seed: int,
    k_shot: int,
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    scenario_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(scenario_payloads) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise M27PhaseSideCacheError("Phase32 cache requires exactly three formal scenarios")
    if not str(capsule_id).strip() or not str(split_id).strip() or int(k_shot) not in {1, 2, 5, 10}:
        raise M27PhaseSideCacheError("Phase32 cache protocol handles drift")
    old_registry = tuple(str(item) for item in old_classes)
    new_registry = tuple(str(item) for item in new_classes)
    if (
        not old_registry
        or not new_registry
        or any(_CLASS_HANDLE.fullmatch(item) is None for item in old_registry + new_registry)
        or len(set(old_registry + new_registry)) != len(old_registry + new_registry)
    ):
        raise M27PhaseSideCacheError("Phase32 class registry drift")

    arrays: dict[str, np.ndarray] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        payload = scenario_payloads[scenario]
        if set(payload) != set(_FIELDS):
            raise M27PhaseSideCacheError(f"{scenario} Phase32 payload allowlist drift")
        old_x = _phase_matrix(payload["old_support_phase32"], name=f"{scenario}.old_support_phase32")
        old_y = _strings(payload["old_support_labels"], name=f"{scenario}.old_support_labels", pattern=_CLASS_HANDLE)
        new_x = _phase_matrix(payload["new_support_phase32"], name=f"{scenario}.new_support_phase32")
        new_y = _strings(payload["new_support_labels"], name=f"{scenario}.new_support_labels", pattern=_CLASS_HANDLE)
        query_x = _phase_matrix(payload["query_phase32"], name=f"{scenario}.query_phase32")
        query_tokens = _strings(payload["query_tokens"], name=f"{scenario}.query_tokens", pattern=_QUERY_TOKEN)
        if len(old_x) != len(old_y) or len(new_x) != len(new_y) or len(query_x) != len(query_tokens):
            raise M27PhaseSideCacheError(f"{scenario} Phase32 row count drift")
        _balanced(old_y, old_registry, int(k_shot), name=f"{scenario}.old_support")
        _balanced(new_y, new_registry, int(k_shot), name=f"{scenario}.new_support")
        if len(set(query_tokens.tolist())) != len(query_tokens):
            raise M27PhaseSideCacheError(f"{scenario} duplicate query token")
        for field, value in (
            ("old_support_phase32", old_x),
            ("old_support_labels", old_y),
            ("new_support_phase32", new_x),
            ("new_support_labels", new_y),
            ("query_phase32", query_x),
            ("query_tokens", query_tokens),
        ):
            arrays[f"{scenario}__{field}"] = value
    payload_destination = Path(payload_path).absolute()
    manifest_destination = Path(manifest_path).absolute()
    if payload_destination.parent != manifest_destination.parent:
        raise M27PhaseSideCacheError("Phase32 payload and manifest must share a directory")
    payload_destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        payload_destination.exists()
        or manifest_destination.exists()
        or payload_destination.is_symlink()
        or manifest_destination.is_symlink()
    ):
        raise FileExistsError("refusing to overwrite Phase32 side-cache evidence")
    payload_bytes = _payload_bytes(arrays)
    payload_sha256 = _sha256_bytes(payload_bytes)
    descriptors = {
        name: {"dtype": value.dtype.str, "shape": list(value.shape)}
        for name, value in sorted(arrays.items())
    }
    manifest = {
        "schema": PHASE_SIDE_MANIFEST_SCHEMA,
        "phase_side_cache_schema": PHASE_SIDE_CACHE_SCHEMA,
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "base_feature_cache_manifest_sha256": _hash(base_manifest_sha256, name="base_manifest_sha256"),
        "capsule_id": str(capsule_id),
        "split_id": str(split_id),
        "receiver": str(receiver),
        "method_seed": int(method_seed),
        "k_shot": int(k_shot),
        "old_classes": list(old_registry),
        "new_classes": list(new_registry),
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "payload_file": payload_destination.name,
        "payload_sha256": payload_sha256,
        "payload_size_bytes": len(payload_bytes),
        "array_names": sorted(arrays),
        "array_descriptors": descriptors,
        "query_truth_present": False,
        "query_role_present": False,
        "clean_source_samples_present": False,
        "query_state_update": False,
        **PHASE2_FULL_CONTRACT,
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    _write_exclusive_readonly(payload_destination, payload_bytes)
    _write_exclusive_readonly(manifest_destination, manifest_bytes + b"\n")
    return {
        "payload_path": str(payload_destination),
        "payload_sha256": payload_sha256,
        "manifest_path": str(manifest_destination),
        "manifest_sha256": manifest_sha256,
        "immutable": True,
    }


def load_phase_side_cache(
    payload_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_payload_sha256: str,
    expected_manifest_sha256: str,
    expected_base_manifest_sha256: str,
    expected_capsule_id: str,
    expected_split_id: str,
    expected_query_tokens_by_scenario: Mapping[str, Any],
) -> dict[str, Any]:
    payload_file = Path(payload_path)
    manifest_file = Path(manifest_path)
    for path, label in ((payload_file, "payload"), (manifest_file, "manifest")):
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & _WRITE_BITS:
            raise M27PhaseSideCacheError(f"Phase32 {label} is not sealed read-only")
    payload_bytes = payload_file.read_bytes()
    manifest_bytes = manifest_file.read_bytes()
    if manifest_bytes.endswith(b"\n"):
        manifest_bytes = manifest_bytes[:-1]
    if _sha256_bytes(payload_bytes) != _hash(expected_payload_sha256, name="expected_payload_sha256"):
        raise M27PhaseSideCacheError("Phase32 payload SHA256 mismatch")
    if _sha256_bytes(manifest_bytes) != _hash(expected_manifest_sha256, name="expected_manifest_sha256"):
        raise M27PhaseSideCacheError("Phase32 manifest SHA256 mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        manifest.get("schema") != PHASE_SIDE_MANIFEST_SCHEMA
        or manifest.get("phase_side_cache_schema") != PHASE_SIDE_CACHE_SCHEMA
        or manifest.get("payload_file") != payload_file.name
        or manifest.get("payload_sha256") != expected_payload_sha256
        or manifest.get("protocol_schema") != "p2_min_v1"
        or manifest.get("phase2_data_status") != "VALIDATED_ONCE"
        or manifest.get("query_truth_present") is not False
        or manifest.get("query_role_present") is not False
        or manifest.get("clean_source_samples_present") is not False
        or manifest.get("query_state_update") is not False
        or tuple(manifest.get("scenarios", ())) != tuple(FORMAL_LEO_WEAK_SCENARIOS)
        or any(manifest.get(key) != value for key, value in PHASE2_FULL_CONTRACT.items())
    ):
        raise M27PhaseSideCacheError("Phase32 manifest contract drift")
    if manifest.get("base_feature_cache_manifest_sha256") != _hash(
        expected_base_manifest_sha256, name="expected_base_manifest_sha256"
    ):
        raise M27PhaseSideCacheError("base feature-cache binding mismatch")
    if manifest.get("capsule_id") != str(expected_capsule_id) or manifest.get("split_id") != str(expected_split_id):
        raise M27PhaseSideCacheError("Phase32 capsule/split binding mismatch")
    try:
        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as archive:
            if set(archive.files) != set(manifest["array_names"]):
                raise M27PhaseSideCacheError("Phase32 array allowlist drift")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise M27PhaseSideCacheError("Phase32 payload cannot be loaded") from exc
    descriptors = {
        name: {"dtype": value.dtype.str, "shape": list(value.shape)}
        for name, value in sorted(arrays.items())
    }
    if descriptors != manifest["array_descriptors"]:
        raise M27PhaseSideCacheError("Phase32 array descriptor drift")
    expected_names = {
        f"{scenario}__{field}"
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
        for field in _FIELDS
    }
    if set(arrays) != expected_names or set(expected_query_tokens_by_scenario) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise M27PhaseSideCacheError("Phase32 scenario allowlist drift")
    scenario_payloads = {
        scenario: {
            field: arrays[f"{scenario}__{field}"] for field in _FIELDS
        }
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        expected_tokens = np.asarray(expected_query_tokens_by_scenario[scenario]).astype(str)
        actual_tokens = scenario_payloads[scenario]["query_tokens"].astype(str)
        if not np.array_equal(actual_tokens, expected_tokens):
            raise M27PhaseSideCacheError(f"{scenario} query-token binding mismatch")
    return {
        "manifest": manifest,
        "old_classes": tuple(manifest["old_classes"]),
        "new_classes": tuple(manifest["new_classes"]),
        "scenario_payloads": scenario_payloads,
    }


__all__ = [
    "M27PhaseSideCacheError",
    "PHASE_DIM",
    "PHASE_SIDE_CACHE_SCHEMA",
    "PHASE_SIDE_MANIFEST_SCHEMA",
    "load_phase_side_cache",
    "phase_coherence32",
    "publish_phase_side_cache",
]
