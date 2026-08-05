#!/usr/bin/env python3
"""Build one Target125 FA-RDCE3 asset from a six-class source-only aggregate.

The CLI accepts a sealed aggregate document, never Target support/query data.
It writes one reusable aggregate-only FA wire and a small manifest without
source row features, physical IDs, raw IQ, or Target information.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from cvsrffi import stage2_next_r5_fa_target125_core as core  # noqa: E402
from cvsrffi import stage2_d106_rdce_asset as d106  # noqa: E402
from cvsrffi import stage2_next_r3_tsl160 as d106_r0  # noqa: E402
from cvsrffi import stage2_next_r4_fa_rdce3 as r4  # noqa: E402


SOURCE_AGGREGATE_SCHEMA = (
    "cvs.phase1.next_r5.fa_rdce3.target125.source_only_aggregate.v2"
)
ASSET_MANIFEST_SCHEMA = "cvs.phase1.next_r5.fa_rdce3.target125.asset_manifest.v2"
METHOD_LOCK_SCHEMA = "cvs.stage2.next_r5.fa_rdce3_qknn.target125.method_lock.v2"
BUILD_STATUS = "NEXT_R5_TARGET125_FA_RDCE3_SOURCE_ONLY_ASSET_COMPLETE"
STRICT_TAP_MEMBERS = d106.TAP_MEMBERS
EXPECTED_ROWS = d106.D104_SOURCE_ROW_COUNT
EXPECTED_RECEIVERS = d106.D104_RECEIVER_COUNT
EXPECTED_CLASSES = d106.D104_SOURCE_CLASS_COUNT
EXPECTED_DAYS = d106.D104_DAY_COUNT
EXPECTED_PER_RECEIVER_CLASS = d106.D104_RECEIVER_TX_FOUR_DAY_COUNT

_AGGREGATE_FIELDS = frozenset(
    {
        "schema",
        "phase1_source_only",
        "source_rows_retained",
        "source_per_row_features_retained",
        "target_support_rows_used",
        "target_query_rows_used",
        "query_rows_used_for_fit",
        "old_classes",
        "source_class_indices",
        "source_old_class_order_sha256",
        "aggregate_samples_per_class",
        "class_centers_3d",
        "fisher_precision_3d",
        "residual_variance_3d",
        "fisher_radius",
        "rdce_kappa_3d",
        "basis_3x160",
        "checkpoint_sha256",
        "phase1_bundle_sha256",
        "phase1_aggregate_receipt_sha256",
        "method_lock_sha256",
    }
)


class NextR5Target125AssetBuildError(ValueError):
    """Raised when the source-only aggregate-to-asset boundary drifts."""


@dataclass(frozen=True, slots=True)
class PreparedTarget125FAAsset:
    asset: core.Target125FARDCE3Asset
    wire: bytes
    source_aggregate_sha256: str


@dataclass(frozen=True, slots=True)
class _VerifiedTarget125MethodLock:
    method_lock_sha256: str
    source_class_indices: tuple[int, ...]
    source_old_class_order_sha256: str


@dataclass(frozen=True, slots=True)
class _StrictTap:
    """D106 Phase1-only strict tap retained only during local aggregation."""

    pre_relu: np.ndarray
    receiver_ids: tuple[str, ...]
    day_ids: tuple[str, ...]
    tx_labels: tuple[str, ...]
    physical_ids: tuple[str, ...]
    archive_sha256: str


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NextR5Target125AssetBuildError("canonical asset payload is invalid") from error


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise NextR5Target125AssetBuildError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR5Target125AssetBuildError(f"{name} must be a lowercase SHA256") from error
    return value


def _read_regular_file(path: Path, expected_sha256: str, *, name: str) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise NextR5Target125AssetBuildError(
            f"{name} must be an absolute regular non-symlink file"
        )
    payload = path.read_bytes()
    if _sha(payload) != _sha256(expected_sha256, f"{name} SHA256"):
        raise NextR5Target125AssetBuildError(f"{name} SHA256 mismatch")
    return payload


def _strict_strings(
    value: np.ndarray,
    *,
    name: str,
    rows: int,
    unique: bool = False,
) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.dtype.kind not in {"U", "S"} or array.shape != (rows,):
        raise NextR5Target125AssetBuildError(f"strict tap {name} dtype/shape drift")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result) or (unique and len(set(result)) != len(result)):
        raise NextR5Target125AssetBuildError(f"strict tap {name} blank/duplicate drift")
    return result


def _load_strict_tap(path: Path, expected_sha256: str) -> _StrictTap:
    payload = _read_regular_file(path, expected_sha256, name="strict tap")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != STRICT_TAP_MEMBERS:
                raise NextR5Target125AssetBuildError("strict tap exact member closure drift")
            arrays = {name: np.asarray(archive[name]).copy() for name in STRICT_TAP_MEMBERS}
    except NextR5Target125AssetBuildError:
        raise
    except Exception as error:
        raise NextR5Target125AssetBuildError("strict tap is not a no-pickle NPZ") from error
    pre_relu = np.asarray(arrays["pre_relu"])
    z_dom = np.asarray(arrays["z_dom"])
    if (
        pre_relu.dtype != np.float32
        or z_dom.dtype != np.float32
        or pre_relu.shape != (EXPECTED_ROWS, core.Z_DIM)
        or z_dom.shape != (EXPECTED_ROWS, core.Z_DIM)
        or not np.isfinite(pre_relu).all()
        or not np.isfinite(z_dom).all()
    ):
        raise NextR5Target125AssetBuildError("strict tap feature dtype/shape/finite drift")
    receiver_ids = _strict_strings(
        arrays["receiver_ids"],
        name="receiver_ids",
        rows=EXPECTED_ROWS,
    )
    day_ids = _strict_strings(arrays["day_ids"], name="day_ids", rows=EXPECTED_ROWS)
    tx_labels = _strict_strings(arrays["tx_labels"], name="tx_labels", rows=EXPECTED_ROWS)
    physical_ids = _strict_strings(
        arrays["physical_ids"],
        name="physical_ids",
        rows=EXPECTED_ROWS,
        unique=True,
    )
    _strict_strings(arrays["scenario_names"], name="scenario_names", rows=EXPECTED_ROWS)
    _strict_strings(
        arrays["observation_ids"],
        name="observation_ids",
        rows=EXPECTED_ROWS,
        unique=True,
    )
    return _StrictTap(
        pre_relu=np.ascontiguousarray(pre_relu, dtype=np.float32),
        receiver_ids=receiver_ids,
        day_ids=day_ids,
        tx_labels=tx_labels,
        physical_ids=physical_ids,
        archive_sha256=_sha(payload),
    )


def _validate_d106_grid(tap: _StrictTap) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    receivers = tuple(sorted(set(tap.receiver_ids)))
    classes = tuple(sorted(set(tap.tx_labels)))
    days = tuple(sorted(set(tap.day_ids)))
    if (
        len(receivers) != EXPECTED_RECEIVERS
        or len(classes) != EXPECTED_CLASSES
        or len(days) != EXPECTED_DAYS
        or len(classes) != core.OLD_CLASS_COUNT
    ):
        raise NextR5Target125AssetBuildError("D106 strict-tap registry drift")
    counts: dict[tuple[str, str, str], int] = {}
    for receiver, day, class_handle in zip(
        tap.receiver_ids,
        tap.day_ids,
        tap.tx_labels,
        strict=True,
    ):
        key = (receiver, day, class_handle)
        counts[key] = counts.get(key, 0) + 1
    if len(counts) != EXPECTED_RECEIVERS * EXPECTED_DAYS * EXPECTED_CLASSES:
        raise NextR5Target125AssetBuildError("D106 strict-tap receiver-day-class grid is incomplete")
    if any(
        not d106.D104_CELL_MIN_SAMPLES
        <= counts[(receiver, day, class_handle)]
        <= d106.D104_CELL_MAX_SAMPLES
        for receiver in receivers
        for day in days
        for class_handle in classes
    ):
        raise NextR5Target125AssetBuildError("D106 strict-tap cell count drift")
    receiver_class_counts: dict[tuple[str, str], int] = {}
    for receiver, class_handle in zip(tap.receiver_ids, tap.tx_labels, strict=True):
        key = (receiver, class_handle)
        receiver_class_counts[key] = receiver_class_counts.get(key, 0) + 1
    if any(
        receiver_class_counts.get((receiver, class_handle)) != EXPECTED_PER_RECEIVER_CLASS
        for receiver in receivers
        for class_handle in classes
    ):
        raise NextR5Target125AssetBuildError("D106 strict-tap receiver-class count drift")
    return receivers, days, classes


def _physical_root(values: Sequence[str]) -> str:
    if (
        not values
        or len(set(values)) != len(values)
        or any(type(item) is not str or not item for item in values)
    ):
        raise NextR5Target125AssetBuildError("source physical-ID root input drift")
    return _sha("\n".join(values).encode("utf-8"))


def _array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, list):
        raise NextR5Target125AssetBuildError(f"{name} must be a JSON numeric array")
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise NextR5Target125AssetBuildError(f"{name} must be numeric") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise NextR5Target125AssetBuildError(f"{name} float32 shape/finite drift")
    return np.ascontiguousarray(array, dtype=np.float32)


def _classes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise NextR5Target125AssetBuildError("old_classes must be a JSON string list")
    result = tuple(value)
    if (
        len(result) != core.OLD_CLASS_COUNT
        or any(type(item) is not str or not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise NextR5Target125AssetBuildError("old_classes must close six unique old classes")
    return result


def _source_class_indices(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise NextR5Target125AssetBuildError("source_class_indices must be a JSON integer list")
    result = tuple(value)
    if (
        len(result) != core.OLD_CLASS_COUNT
        or any(type(item) is not int for item in result)
        or result != tuple(range(core.OLD_CLASS_COUNT))
    ):
        raise NextR5Target125AssetBuildError("source_class_indices must close source slots 0..5")
    return result


def _source_old_class_order_sha256(classes: Sequence[str]) -> str:
    return _sha(_canonical(list(_classes(list(classes)))))


def _load_verified_method_lock(
    path: Path,
    expected_sha256: str,
) -> _VerifiedTarget125MethodLock:
    payload = _read_regular_file(path, expected_sha256, name="method lock")
    try:
        lock = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR5Target125AssetBuildError("method lock must be UTF-8 JSON") from error
    if (
        type(lock) is not dict
        or lock.get("schema") != METHOD_LOCK_SCHEMA
        or lock.get("candidate_id") != core.CANDIDATE_ID
        or lock.get("protocol_schema") != core.PROTOCOL_SCHEMA
    ):
        raise NextR5Target125AssetBuildError("method lock identity/schema drift")
    bridge = lock.get("class_identity_bridge")
    required_bridge = {
        "source_class_indices",
        "source_asset_old_class_order_sha256",
        "sealed_package_class_index_to_row_local_handle",
        "row_local_handle_scope",
        "cross_row_handle_reuse",
    }
    if type(bridge) is not dict or set(bridge) != required_bridge:
        raise NextR5Target125AssetBuildError("method lock class-identity bridge field closure drift")
    indices = _source_class_indices(bridge["source_class_indices"])
    root = _sha256(
        bridge["source_asset_old_class_order_sha256"],
        "method lock source_asset_old_class_order_sha256",
    )
    if (
        bridge["sealed_package_class_index_to_row_local_handle"] is not True
        or bridge["row_local_handle_scope"] != "per_package_row"
        or bridge["cross_row_handle_reuse"] is not False
    ):
        raise NextR5Target125AssetBuildError("method lock class-identity bridge policy drift")
    return _VerifiedTarget125MethodLock(
        method_lock_sha256=_sha(payload),
        source_class_indices=indices,
        source_old_class_order_sha256=root,
    )


def _verify_source_aggregate_method_lock(
    document: Mapping[str, Any],
    verified_method_lock: _VerifiedTarget125MethodLock,
) -> None:
    classes = _classes(document["old_classes"])
    if (
        _sha256(document["method_lock_sha256"], "method_lock_sha256")
        != verified_method_lock.method_lock_sha256
        or _source_class_indices(document["source_class_indices"])
        != verified_method_lock.source_class_indices
        or _sha256(
            document["source_old_class_order_sha256"],
            "source_old_class_order_sha256",
        )
        != verified_method_lock.source_old_class_order_sha256
        or _source_old_class_order_sha256(classes)
        != verified_method_lock.source_old_class_order_sha256
    ):
        raise NextR5Target125AssetBuildError("source aggregate / method-lock class-identity drift")


def _counts(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise NextR5Target125AssetBuildError("aggregate_samples_per_class must be a JSON integer list")
    result = tuple(value)
    if (
        len(result) != core.OLD_CLASS_COUNT
        or any(type(item) is not int or item < 2 for item in result)
    ):
        raise NextR5Target125AssetBuildError(
            "aggregate_samples_per_class must prove multi-sample aggregation"
        )
    return result


def _validate_source_only_document(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != _AGGREGATE_FIELDS:
        raise NextR5Target125AssetBuildError("source-only aggregate field closure drift")
    if (
        value["schema"] != SOURCE_AGGREGATE_SCHEMA
        or value["phase1_source_only"] is not True
        or value["source_rows_retained"] is not False
        or value["source_per_row_features_retained"] is not False
        or value["target_support_rows_used"] != 0
        or value["target_query_rows_used"] != 0
        or value["query_rows_used_for_fit"] != 0
    ):
        raise NextR5Target125AssetBuildError("source-only aggregate access contract drift")
    classes = _classes(value["old_classes"])
    _source_class_indices(value["source_class_indices"])
    if _sha256(value["source_old_class_order_sha256"], "source_old_class_order_sha256") != _source_old_class_order_sha256(classes):
        raise NextR5Target125AssetBuildError("source old-class order root drift")
    _counts(value["aggregate_samples_per_class"])
    _array(value["class_centers_3d"], "class_centers_3d", (core.OLD_CLASS_COUNT, core.FA_RANK))
    _array(value["fisher_precision_3d"], "fisher_precision_3d", (core.FA_RANK,))
    _array(value["residual_variance_3d"], "residual_variance_3d", (core.FA_RANK,))
    _array(value["fisher_radius"], "fisher_radius", (1,))
    _array(value["rdce_kappa_3d"], "rdce_kappa_3d", (core.FA_RANK,))
    _array(value["basis_3x160"], "basis_3x160", (core.FA_RANK, core.Z_DIM))
    for name in (
        "checkpoint_sha256",
        "phase1_bundle_sha256",
        "phase1_aggregate_receipt_sha256",
        "method_lock_sha256",
    ):
        _sha256(value[name], name)
    return value


def _prepare_target125_fa_asset_from_source_only_aggregate(
    aggregate: Mapping[str, Any],
    *,
    verified_method_lock: _VerifiedTarget125MethodLock,
) -> PreparedTarget125FAAsset:
    """Compile one Target125 asset from a validated six-class aggregate object."""

    document = _validate_source_only_document(aggregate)
    _verify_source_aggregate_method_lock(document, verified_method_lock)
    try:
        asset = core.build_target_fa_asset(
            old_classes=_classes(document["old_classes"]),
            source_class_indices=_source_class_indices(document["source_class_indices"]),
            source_old_class_order_sha256=_sha256(
                document["source_old_class_order_sha256"],
                "source_old_class_order_sha256",
            ),
            aggregate_samples_per_class=_counts(document["aggregate_samples_per_class"]),
            class_centers_3d=_array(
                document["class_centers_3d"],
                "class_centers_3d",
                (core.OLD_CLASS_COUNT, core.FA_RANK),
            ),
            fisher_precision_3d=_array(
                document["fisher_precision_3d"],
                "fisher_precision_3d",
                (core.FA_RANK,),
            ),
            residual_variance_3d=_array(
                document["residual_variance_3d"],
                "residual_variance_3d",
                (core.FA_RANK,),
            ),
            fisher_radius=_array(document["fisher_radius"], "fisher_radius", (1,)),
            rdce_kappa_3d=_array(
                document["rdce_kappa_3d"],
                "rdce_kappa_3d",
                (core.FA_RANK,),
            ),
            basis_3x160=_array(
                document["basis_3x160"],
                "basis_3x160",
                (core.FA_RANK, core.Z_DIM),
            ),
            checkpoint_sha256=_sha256(document["checkpoint_sha256"], "checkpoint_sha256"),
            phase1_bundle_sha256=_sha256(
                document["phase1_bundle_sha256"],
                "phase1_bundle_sha256",
            ),
            phase1_aggregate_receipt_sha256=_sha256(
                document["phase1_aggregate_receipt_sha256"],
                "phase1_aggregate_receipt_sha256",
            ),
            method_lock_sha256=_sha256(document["method_lock_sha256"], "method_lock_sha256"),
        )
        wire = core.serialize_target_fa_asset(asset)
        recovered = core.deserialize_target_fa_asset(wire)
    except core.NextR5FATarget125CoreError as error:
        raise NextR5Target125AssetBuildError("Target125 FA aggregate wire construction failed") from error
    if recovered.asset_sha256 != asset.asset_sha256:
        raise NextR5Target125AssetBuildError("Target125 FA wire roundtrip receipt drift")
    return PreparedTarget125FAAsset(
        asset=asset,
        wire=wire,
        source_aggregate_sha256=_sha(_canonical(document)),
    )


def prepare_target125_fa_asset_from_source_only_aggregate(
    aggregate: Mapping[str, Any],
    *,
    method_lock_path: Path,
    method_lock_sha256: str,
) -> PreparedTarget125FAAsset:
    """Compile an aggregate only when its external method lock is verified."""

    return _prepare_target125_fa_asset_from_source_only_aggregate(
        aggregate,
        verified_method_lock=_load_verified_method_lock(
            method_lock_path,
            method_lock_sha256,
        ),
    )


def _aggregate_receipt(
    *,
    strict_tap_sha256: str,
    checkpoint_sha256: str,
    method_lock_sha256: str,
    old_classes: tuple[str, ...],
    source_class_indices: tuple[int, ...],
    source_old_class_order_sha256: str,
    aggregate_samples_per_class: tuple[int, ...],
    source_physical_root_sha256: str,
    values: Mapping[str, np.ndarray],
) -> str:
    return _sha(
        _canonical(
            {
                "schema": "cvs.phase1.next_r5.fa_rdce3.target125.aggregate_receipt.v2",
                "candidate_id": core.CANDIDATE_ID,
                "strict_tap_sha256": strict_tap_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "method_lock_sha256": method_lock_sha256,
                "old_classes": list(old_classes),
                "source_class_indices": list(source_class_indices),
                "source_old_class_order_sha256": source_old_class_order_sha256,
                "aggregate_samples_per_class": list(aggregate_samples_per_class),
                "source_physical_root_sha256": source_physical_root_sha256,
                "basis_formula": "d106_receiver_day_class_balanced_scatter_canonical_rank3",
                "residual_variance_formula": "D_v=tau_class_balanced_projected_within_class_variance_closed_int8_basis",
                "fisher_precision_formula": "D_F=1/spectrum_receiver_day_common_shift_scatter_closed_int8_basis",
                "fisher_radius_formula": "sqrt(rank)=sqrt(3)",
                "kappa_formula": "spectrum/(spectrum+tau)",
                "aggregate_arrays": {
                    name: {
                        "dtype": np.asarray(array).dtype.str,
                        "shape": list(np.asarray(array).shape),
                        "sha256": _sha(np.ascontiguousarray(array).tobytes(order="C")),
                    }
                    for name, array in values.items()
                },
            }
        )
    )


def source_only_aggregate_from_d106_strict_tap(
    *,
    strict_tap: Path,
    strict_tap_sha256: str,
    checkpoint_sha256: str,
    method_lock_path: Path,
    method_lock_sha256: str,
) -> Mapping[str, Any]:
    """Derive the one six-old-class aggregate from the sealed D106 strict tap.

    All raw source rows and physical IDs are scoped to this Phase1-only
    function.  The returned document retains only fixed-size aggregates and
    opaque SHA256 lineage values.
    """

    verified_method_lock = _load_verified_method_lock(
        method_lock_path,
        method_lock_sha256,
    )
    tap = _load_strict_tap(
        strict_tap,
        _sha256(strict_tap_sha256, "strict tap SHA256"),
    )
    _receivers, _days, classes = _validate_d106_grid(tap)
    checkpoint = _sha256(checkpoint_sha256, "checkpoint_sha256")
    source_class_indices = tuple(range(core.OLD_CLASS_COUNT))
    source_old_class_order = _source_old_class_order_sha256(classes)
    if (
        source_class_indices != verified_method_lock.source_class_indices
        or source_old_class_order != verified_method_lock.source_old_class_order_sha256
    ):
        raise NextR5Target125AssetBuildError("strict-tap classes / method-lock root drift")
    method_lock = verified_method_lock.method_lock_sha256
    try:
        r0 = d106_r0.canonical_d106_relu_zid160(tap.pre_relu)
    except Exception as error:
        raise NextR5Target125AssetBuildError(
            "D106 strict-tap canonical R0 conversion failed"
        ) from error
    if (
        r0.dtype != np.float32
        or r0.shape != (EXPECTED_ROWS, core.Z_DIM)
        or np.any(r0 < 0.0)
        or not np.allclose(
            np.linalg.norm(r0.astype(np.float64), axis=1),
            1.0,
            rtol=0.0,
            atol=2.0e-6,
        )
    ):
        raise NextR5Target125AssetBuildError("D106 strict-tap canonical R0 closure drift")
    class_rows: dict[str, list[np.ndarray]] = {name: [] for name in classes}
    cell_class_rows: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for index, (receiver, day, class_handle) in enumerate(
        zip(tap.receiver_ids, tap.day_ids, tap.tx_labels, strict=True)
    ):
        if class_handle not in class_rows:
            raise NextR5Target125AssetBuildError("strict-tap class registry drift")
        row = r0[index]
        class_rows[class_handle].append(row)
        cell_class_rows.setdefault((receiver, day, class_handle), []).append(row)
    if any(
        len(class_rows[class_handle])
        != EXPECTED_RECEIVERS * EXPECTED_PER_RECEIVER_CLASS
        for class_handle in classes
    ):
        raise NextR5Target125AssetBuildError("strict-tap per-class aggregate count drift")
    cell_keys = tuple(sorted({(receiver, day) for receiver, day, _ in cell_class_rows}))
    if len(cell_keys) != EXPECTED_RECEIVERS * EXPECTED_DAYS:
        raise NextR5Target125AssetBuildError("strict-tap receiver-day cell coverage drift")
    if any(
        (receiver, day, class_handle) not in cell_class_rows
        or not d106.D104_CELL_MIN_SAMPLES
        <= len(cell_class_rows[(receiver, day, class_handle)])
        <= d106.D104_CELL_MAX_SAMPLES
        for receiver, day in cell_keys
        for class_handle in classes
    ):
        raise NextR5Target125AssetBuildError("strict-tap class-balanced cell coverage drift")
    cell_means = {
        key: d106._mean_canonical(rows)  # noqa: SLF001 - frozen D106 geometry.
        for key, rows in cell_class_rows.items()
    }
    class_cell_means: dict[str, list[tuple[tuple[str, str], np.ndarray]]] = {
        class_handle: [] for class_handle in classes
    }
    cell_values: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {
        key: [] for key in cell_keys
    }
    for (receiver, day, class_handle), value in cell_means.items():
        cell = (receiver, day)
        class_cell_means[class_handle].append((cell, value))
        cell_values[cell].append((class_handle, value))
    class_reference: dict[str, np.ndarray] = {}
    for class_handle in classes:
        entries = class_cell_means[class_handle]
        if len(entries) != len(cell_keys):
            raise NextR5Target125AssetBuildError("strict-tap class cell closure drift")
        ordered = [
            value
            for _cell, value in sorted(
                entries,
                key=lambda item: d106._content_key([item[1]]),  # noqa: SLF001
            )
        ]
        class_reference[class_handle] = d106._mean_canonical(ordered)  # noqa: SLF001
    common_shift_rows: list[tuple[bytes, np.ndarray]] = []
    for cell in cell_keys:
        entries = cell_values[cell]
        if len(entries) != len(classes):
            raise NextR5Target125AssetBuildError("strict-tap class-equal cell mean drift")
        deltas = [value - class_reference[class_handle] for class_handle, value in entries]
        common = d106._mean_canonical(  # noqa: SLF001
            sorted(deltas, key=lambda value: d106._content_key([value]))  # noqa: SLF001
        )
        common_shift_rows.append(
            (
                d106._content_key([value for _class_handle, value in entries]),  # noqa: SLF001
                common,
            )
        )
    ordered_common = [value for _key, value in sorted(common_shift_rows, key=lambda item: item[0])]
    if len(ordered_common) != len(cell_keys):
        raise NextR5Target125AssetBuildError("strict-tap common-shift count drift")
    centered_common = np.stack(ordered_common, axis=0)
    centered_common = centered_common - d106._mean_canonical(ordered_common)  # noqa: SLF001
    scatter = (centered_common.T @ centered_common) / float(len(cell_keys))
    try:
        raw_basis = d106._canonical_top_eigensystem(scatter)  # noqa: SLF001
        basis_f32 = np.ascontiguousarray(raw_basis, dtype=np.float32)
        basis_codes, basis_scales = r4._quantize_signed_rows(  # noqa: SLF001
            basis_f32,
            name="Target125 FA-RDCE3 basis",
        )
        closed_basis = r4._decoded_basis(basis_codes, basis_scales)  # noqa: SLF001
        tau = d106._balanced_tau_for_closed_basis(  # noqa: SLF001
            [class_rows[class_handle] for class_handle in classes],
            closed_basis,
        )
    except Exception as error:
        raise NextR5Target125AssetBuildError(
            "strict-tap closed-basis aggregate construction failed"
        ) from error
    spectrum = np.diag(closed_basis @ scatter @ closed_basis.T)
    if (
        not np.isfinite(tau).all()
        or not np.isfinite(spectrum).all()
        or np.any(tau <= d106.EPSILON)
        or np.any(spectrum <= d106.MIN_SPECTRUM)
    ):
        raise NextR5Target125AssetBuildError("strict-tap tau/spectrum is nonpositive")
    fisher_precision = 1.0 / spectrum
    kappa = spectrum / (spectrum + tau)
    if (
        not np.isfinite(fisher_precision).all()
        or not np.isfinite(kappa).all()
        or np.any(fisher_precision <= 0.0)
        or np.any(kappa < 0.0)
        or np.any(kappa >= 1.0)
    ):
        raise NextR5Target125AssetBuildError("strict-tap Fisher/kappa aggregate drift")
    values = {
        "class_centers_3d": np.asarray(
            [
                d106._mean_canonical(class_rows[class_handle]) @ closed_basis.T  # noqa: SLF001
                for class_handle in classes
            ],
            dtype=np.float32,
        ),
        "fisher_precision_3d": np.asarray(fisher_precision, dtype=np.float32),
        "residual_variance_3d": np.asarray(tau, dtype=np.float32),
        "fisher_radius": np.asarray([math.sqrt(float(core.FA_RANK))], dtype=np.float32),
        "rdce_kappa_3d": np.asarray(kappa, dtype=np.float32),
        "basis_3x160": basis_f32,
    }
    if any(not np.isfinite(value).all() for value in values.values()):
        raise NextR5Target125AssetBuildError("strict-tap float32 aggregate conversion failed")
    aggregate_count = EXPECTED_RECEIVERS * EXPECTED_PER_RECEIVER_CLASS
    aggregate_receipt = _aggregate_receipt(
        strict_tap_sha256=tap.archive_sha256,
        checkpoint_sha256=checkpoint,
        method_lock_sha256=method_lock,
        old_classes=classes,
        source_class_indices=source_class_indices,
        source_old_class_order_sha256=source_old_class_order,
        aggregate_samples_per_class=(aggregate_count,) * core.OLD_CLASS_COUNT,
        source_physical_root_sha256=_physical_root(tap.physical_ids),
        values=values,
    )
    return {
        "schema": SOURCE_AGGREGATE_SCHEMA,
        "phase1_source_only": True,
        "source_rows_retained": False,
        "source_per_row_features_retained": False,
        "target_support_rows_used": 0,
        "target_query_rows_used": 0,
        "query_rows_used_for_fit": 0,
        "old_classes": list(classes),
        "source_class_indices": list(source_class_indices),
        "source_old_class_order_sha256": source_old_class_order,
        "aggregate_samples_per_class": [aggregate_count] * core.OLD_CLASS_COUNT,
        "class_centers_3d": values["class_centers_3d"].tolist(),
        "fisher_precision_3d": values["fisher_precision_3d"].tolist(),
        "residual_variance_3d": values["residual_variance_3d"].tolist(),
        "fisher_radius": values["fisher_radius"].tolist(),
        "rdce_kappa_3d": values["rdce_kappa_3d"].tolist(),
        "basis_3x160": values["basis_3x160"].tolist(),
        "checkpoint_sha256": checkpoint,
        "phase1_bundle_sha256": tap.archive_sha256,
        "phase1_aggregate_receipt_sha256": aggregate_receipt,
        "method_lock_sha256": method_lock,
    }


def prepare_target125_fa_asset_from_d106_strict_tap(
    *,
    strict_tap: Path,
    strict_tap_sha256: str,
    checkpoint_sha256: str,
    method_lock_path: Path,
    method_lock_sha256: str,
) -> PreparedTarget125FAAsset:
    aggregate = source_only_aggregate_from_d106_strict_tap(
        strict_tap=strict_tap,
        strict_tap_sha256=strict_tap_sha256,
        checkpoint_sha256=checkpoint_sha256,
        method_lock_path=method_lock_path,
        method_lock_sha256=method_lock_sha256,
    )
    return prepare_target125_fa_asset_from_source_only_aggregate(
        aggregate,
        method_lock_path=method_lock_path,
        method_lock_sha256=method_lock_sha256,
    )


def _new_output_dir(value: Path) -> Path:
    resolved = value.resolve(strict=False)
    if (
        not value.is_absolute()
        or value.exists()
        or value.is_symlink()
        or not resolved.parent.is_dir()
    ):
        raise NextR5Target125AssetBuildError(
            "output directory must be a new absolute child of an existing directory"
        )
    return resolved


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise NextR5Target125AssetBuildError(f"output overwrite/path drift: {path}")
    with path.open("xb") as handle:
        handle.write(payload)


def _read_source_only_aggregate(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise NextR5Target125AssetBuildError("source-only aggregate must be an absolute regular file")
    payload = path.read_bytes()
    if _sha(payload) != _sha256(expected_sha256, "source-only aggregate SHA256"):
        raise NextR5Target125AssetBuildError("source-only aggregate SHA256 mismatch")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR5Target125AssetBuildError("source-only aggregate is not UTF-8 JSON") from error
    _validate_source_only_document(document)
    return document


def _materialize_prepared_target125_fa_asset(
    prepared: PreparedTarget125FAAsset,
    *,
    output_dir: Path,
    strict_tap_sha256: str | None,
) -> Mapping[str, Any]:
    root = _new_output_dir(output_dir)
    root.mkdir()
    wire_path = root / "fa_rdce3_target125.wire"
    _write_new(wire_path, prepared.wire)
    manifest = {
        "schema": ASSET_MANIFEST_SCHEMA,
        "candidate_id": core.CANDIDATE_ID,
        "build_status": BUILD_STATUS,
        "asset_path": str(wire_path.resolve()),
        "asset_sha256": prepared.asset.asset_sha256,
        "asset_wire_sha256": _sha(prepared.wire),
        "source_aggregate_sha256": prepared.source_aggregate_sha256,
        "strict_tap_sha256": strict_tap_sha256,
        "old_classes": list(prepared.asset.old_classes),
        "source_class_indices": list(prepared.asset.source_class_indices),
        "source_old_class_order_sha256": prepared.asset.source_old_class_order_sha256,
        "checkpoint_sha256": prepared.asset.checkpoint_sha256,
        "phase1_bundle_sha256": prepared.asset.phase1_bundle_sha256,
        "phase1_aggregate_receipt_sha256": prepared.asset.phase1_aggregate_receipt_sha256,
        "method_lock_sha256": prepared.asset.method_lock_sha256,
        "phase1_source_only": True,
        "phase1_source_rows_retained": False,
        "phase1_per_row_features_retained": False,
        "target_support_rows_used": 0,
        "target_query_rows_used": 0,
        "query_rows_used_for_fit": 0,
    }
    manifest_path = root / "fa_target125_asset_manifest.json"
    _write_new(manifest_path, _canonical(manifest))
    return {
        "status": BUILD_STATUS,
        "output_dir": str(root),
        "asset": str(wire_path),
        "manifest": str(manifest_path),
        "asset_sha256": prepared.asset.asset_sha256,
        "manifest_sha256": _sha(manifest_path.read_bytes()),
        "old_class_count": len(prepared.asset.old_classes),
        "source_class_indices": list(prepared.asset.source_class_indices),
        "source_old_class_order_sha256": prepared.asset.source_old_class_order_sha256,
        "phase1_source_rows_retained": False,
        "phase1_per_row_features_retained": False,
        "target_support_rows_used": 0,
        "target_query_rows_used": 0,
    }


def build_target125_fa_asset_from_source_only_aggregate(
    *,
    source_only_aggregate_json: Path,
    source_only_aggregate_sha256: str,
    method_lock_path: Path,
    method_lock_sha256: str,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Non-release helper for an already prepared source-only aggregate."""

    aggregate = _read_source_only_aggregate(
        source_only_aggregate_json,
        source_only_aggregate_sha256,
    )
    prepared = prepare_target125_fa_asset_from_source_only_aggregate(
        aggregate,
        method_lock_path=method_lock_path,
        method_lock_sha256=method_lock_sha256,
    )
    return _materialize_prepared_target125_fa_asset(
        prepared,
        output_dir=output_dir,
        strict_tap_sha256=None,
    )


def build_target125_fa_asset(
    *,
    strict_tap: Path,
    strict_tap_sha256: str,
    checkpoint_sha256: str,
    method_lock_path: Path,
    method_lock_sha256: str,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Release entry: build one six-class asset directly from sealed D106 tap."""

    prepared = prepare_target125_fa_asset_from_d106_strict_tap(
        strict_tap=strict_tap,
        strict_tap_sha256=strict_tap_sha256,
        checkpoint_sha256=checkpoint_sha256,
        method_lock_path=method_lock_path,
        method_lock_sha256=method_lock_sha256,
    )
    return _materialize_prepared_target125_fa_asset(
        prepared,
        output_dir=output_dir,
        strict_tap_sha256=_sha256(strict_tap_sha256, "strict tap SHA256"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Target125 FA-RDCE3 asset directly from sealed D106 strict tap"
    )
    parser.add_argument("--strict-tap", required=True, type=Path)
    parser.add_argument("--strict-tap-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--method-lock", required=True, type=Path)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_target125_fa_asset(
        strict_tap=args.strict_tap,
        strict_tap_sha256=args.strict_tap_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        method_lock_path=args.method_lock,
        method_lock_sha256=args.method_lock_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
