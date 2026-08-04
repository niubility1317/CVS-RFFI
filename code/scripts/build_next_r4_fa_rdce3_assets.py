#!/usr/bin/env python3
"""Build the 12 aggregate-only NEXT-R4 FA-RDCE3 Phase1 assets.

The input is the already sealed D106 ``L_s`` strict tap.  This builder is a
Phase1-only boundary: it derives the canonical ``R0`` representation in
memory, builds one receiver-held/class-LOCO aggregate asset per outer cell,
and writes only aggregate INT8/FP16 wires plus an ID-free manifest.  It never
writes a source row, feature, class-member list, or physical ID to an output.

The statistical construction is frozen by the NEXT-R4 design:

* D106's canonical receiver-day, class-balanced rank-three nuisance basis;
* ``D_v=tau`` from the five equally weighted retained-class projected
  within-class variances on the *dequantized closed* basis;
* ``D_F=1/spectrum`` from the same closed-basis receiver-day common-shift
  scatter;
* ``rho=sqrt(3)`` and ``kappa=spectrum/(spectrum+tau)``.

There are no target/query arguments, no tunable numerical constants, and no
performance calculation in this entry point.
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

from cvsrffi import stage2_d106_rdce_asset as d106  # noqa: E402
from cvsrffi import stage2_next_r3_tsl160 as d106_r0  # noqa: E402
from cvsrffi import stage2_next_r4_fa_rdce3 as fa  # noqa: E402
from cvsrffi import stage2_next_r4_matrix as matrix  # noqa: E402


ASSET_MANIFEST_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.asset_manifest.v1"
AGGREGATE_RECEIPT_SCHEMA = "cvs.stage2.next_r4.fa_rdce3.phase1_aggregate_receipt.v1"
BUILD_STATUS = "NEXT_R4_FA_RDCE3_PHASE1_ASSETS_COMPLETE"
EXPECTED_ROWS = d106.D104_SOURCE_ROW_COUNT
EXPECTED_RECEIVERS = d106.D104_RECEIVER_COUNT
EXPECTED_CLASSES = d106.D104_SOURCE_CLASS_COUNT
EXPECTED_DAYS = d106.D104_DAY_COUNT
EXPECTED_PER_RECEIVER_CLASS = d106.D104_RECEIVER_TX_FOUR_DAY_COUNT
EXPECTED_OUTER_FIT_ROWS = (EXPECTED_RECEIVERS - 1) * (EXPECTED_CLASSES - 1) * EXPECTED_PER_RECEIVER_CLASS
EXPECTED_OUTER_PER_CLASS = (EXPECTED_RECEIVERS - 1) * EXPECTED_PER_RECEIVER_CLASS
STRICT_TAP_MEMBERS = d106.TAP_MEMBERS


class NextR4FAAssetBuildError(ValueError):
    """Raised when the frozen strict-tap-to-aggregate route drifts."""


@dataclass(frozen=True, slots=True)
class _StrictTap:
    """Validated D106 strict-tap arrays kept only during Phase1 construction."""

    pre_relu: np.ndarray
    receiver_ids: tuple[str, ...]
    day_ids: tuple[str, ...]
    tx_labels: tuple[str, ...]
    physical_ids: tuple[str, ...]
    archive_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedFARDCE3Asset:
    """One in-memory aggregate wire; deliberately has no member-ID field."""

    held_receiver: str
    held_class: str
    asset: fa.FARDCE3Phase1Asset
    wire: bytes
    phase1_fit_count: int
    phase1_fit_physical_root_sha256: str
    aggregate_receipt_sha256: str

    @property
    def outer_key(self) -> str:
        return f"{self.held_receiver}|{self.held_class}"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise NextR4FAAssetBuildError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR4FAAssetBuildError(f"{name} must be a lowercase SHA256") from error
    return value


def _read_regular_file(path: Path, expected_sha256: str, *, name: str) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise NextR4FAAssetBuildError(f"{name} must be an absolute regular non-symlink file")
    payload = path.read_bytes()
    if _sha(payload) != _require_sha256(expected_sha256, name=f"{name} SHA256"):
        raise NextR4FAAssetBuildError(f"{name} SHA256 mismatch")
    return payload


def _strict_strings(value: np.ndarray, *, name: str, rows: int, unique: bool = False) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.dtype.kind not in {"U", "S"} or array.shape != (rows,):
        raise NextR4FAAssetBuildError(f"strict tap {name} dtype/shape drift")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result) or (unique and len(set(result)) != len(result)):
        raise NextR4FAAssetBuildError(f"strict tap {name} blank/duplicate drift")
    return result


def _load_strict_tap(path: Path, expected_sha256: str) -> _StrictTap:
    payload = _read_regular_file(path, expected_sha256, name="strict tap")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != STRICT_TAP_MEMBERS:
                raise NextR4FAAssetBuildError("strict tap exact member closure drift")
            arrays = {name: np.asarray(archive[name]).copy() for name in STRICT_TAP_MEMBERS}
    except NextR4FAAssetBuildError:
        raise
    except Exception as error:
        raise NextR4FAAssetBuildError("strict tap is not a no-pickle NPZ") from error

    pre_relu = np.asarray(arrays["pre_relu"])
    z_dom = np.asarray(arrays["z_dom"])
    expected_shape = (EXPECTED_ROWS, fa.Z_DIM)
    if (
        pre_relu.dtype != np.float32
        or z_dom.dtype != np.float32
        or pre_relu.shape != expected_shape
        or z_dom.shape != expected_shape
        or not np.isfinite(pre_relu).all()
        or not np.isfinite(z_dom).all()
    ):
        raise NextR4FAAssetBuildError("strict tap feature dtype/shape/finite drift")
    receiver_ids = _strict_strings(arrays["receiver_ids"], name="receiver_ids", rows=EXPECTED_ROWS)
    day_ids = _strict_strings(arrays["day_ids"], name="day_ids", rows=EXPECTED_ROWS)
    tx_labels = _strict_strings(arrays["tx_labels"], name="tx_labels", rows=EXPECTED_ROWS)
    physical_ids = _strict_strings(
        arrays["physical_ids"], name="physical_ids", rows=EXPECTED_ROWS, unique=True
    )
    _strict_strings(arrays["scenario_names"], name="scenario_names", rows=EXPECTED_ROWS)
    _strict_strings(
        arrays["observation_ids"], name="observation_ids", rows=EXPECTED_ROWS, unique=True
    )
    return _StrictTap(
        pre_relu=np.ascontiguousarray(pre_relu, dtype=np.float32),
        receiver_ids=receiver_ids,
        day_ids=day_ids,
        tx_labels=tx_labels,
        physical_ids=physical_ids,
        archive_sha256=_sha(payload),
    )


def _validate_d106_grid(tap: _StrictTap) -> tuple[tuple[str, ...], tuple[str, ...]]:
    receivers = tuple(sorted(set(tap.receiver_ids)))
    classes = tuple(sorted(set(tap.tx_labels)))
    days = tuple(sorted(set(tap.day_ids)))
    if (
        len(receivers) != EXPECTED_RECEIVERS
        or len(classes) != EXPECTED_CLASSES
        or len(days) != EXPECTED_DAYS
        or tuple(matrix.HELD_RECEIVERS) != ("1-1", "18-2")
        or any(receiver not in receivers for receiver in matrix.HELD_RECEIVERS)
    ):
        raise NextR4FAAssetBuildError("D106 receiver/class/day registry drift")
    counts: dict[tuple[str, str, str], int] = {}
    for receiver, day, class_handle in zip(
        tap.receiver_ids, tap.day_ids, tap.tx_labels, strict=True
    ):
        key = (receiver, day, class_handle)
        counts[key] = counts.get(key, 0) + 1
    if len(counts) != EXPECTED_RECEIVERS * EXPECTED_DAYS * EXPECTED_CLASSES:
        raise NextR4FAAssetBuildError("D106 receiver-day-class grid is incomplete")
    if any(
        not d106.D104_CELL_MIN_SAMPLES <= counts[(receiver, day, class_handle)] <= d106.D104_CELL_MAX_SAMPLES
        for receiver in receivers
        for day in days
        for class_handle in classes
    ):
        raise NextR4FAAssetBuildError("D106 strict-tap cell count drift")
    receiver_class_counts: dict[tuple[str, str], int] = {}
    for receiver, class_handle in zip(tap.receiver_ids, tap.tx_labels, strict=True):
        key = (receiver, class_handle)
        receiver_class_counts[key] = receiver_class_counts.get(key, 0) + 1
    if any(
        receiver_class_counts.get((receiver, class_handle)) != EXPECTED_PER_RECEIVER_CLASS
        for receiver in receivers
        for class_handle in classes
    ):
        raise NextR4FAAssetBuildError("D106 receiver-class physical count drift")
    return receivers, classes


def _physical_root(values: Sequence[str]) -> str:
    """Match the NEXT-R4 metadata/matrix newline-order root exactly."""

    if not values or len(set(values)) != len(values) or any(not isinstance(item, str) or not item for item in values):
        raise NextR4FAAssetBuildError("Phase1 physical-ID root input drift")
    return _sha("\n".join(values).encode("utf-8"))


def _aggregate_receipt(
    *,
    held_receiver: str,
    held_class: str,
    old_classes: Sequence[str],
    phase1_fit_count: int,
    phase1_fit_physical_root_sha256: str,
    strict_tap_sha256: str,
    checkpoint_sha256: str,
    method_lock_sha256: str,
    class_centers_3d: np.ndarray,
    fisher_precision_3d: np.ndarray,
    residual_variance_3d: np.ndarray,
    fisher_radius: np.ndarray,
    rdce_kappa_3d: np.ndarray,
    basis_3x160: np.ndarray,
) -> str:
    arrays = {
        "class_centers_3d": class_centers_3d,
        "fisher_precision_3d": fisher_precision_3d,
        "residual_variance_3d": residual_variance_3d,
        "fisher_radius": fisher_radius,
        "rdce_kappa_3d": rdce_kappa_3d,
        "basis_3x160": basis_3x160,
    }
    payload = {
        "schema": AGGREGATE_RECEIPT_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "representation_rule": fa.R0_REPRESENTATION_RULE,
        "held_receiver": held_receiver,
        "held_class": held_class,
        "old_classes": list(old_classes),
        "phase1_fit_count": phase1_fit_count,
        "phase1_fit_physical_root_sha256": phase1_fit_physical_root_sha256,
        "strict_tap_sha256": strict_tap_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "method_lock_sha256": method_lock_sha256,
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
            for name, array in arrays.items()
        },
    }
    return _sha(_canonical(payload))


def _geometry_for_outer(
    *,
    tap: _StrictTap,
    r0: np.ndarray,
    held_receiver: str,
    held_class: str,
    all_classes: tuple[str, ...],
    checkpoint_sha256: str,
    method_lock_sha256: str,
) -> PreparedFARDCE3Asset:
    mask = np.asarray(
        [
            receiver != held_receiver and class_handle != held_class
            for receiver, class_handle in zip(tap.receiver_ids, tap.tx_labels, strict=True)
        ],
        dtype=bool,
    )
    if int(np.sum(mask)) != EXPECTED_OUTER_FIT_ROWS:
        raise NextR4FAAssetBuildError("outer-cell Phase1 fit count must be exactly 420")
    old_classes = tuple(class_handle for class_handle in all_classes if class_handle != held_class)
    if len(old_classes) != EXPECTED_CLASSES - 1:
        raise NextR4FAAssetBuildError("outer-cell old class registry drift")
    selected = np.flatnonzero(mask)
    fit_ids = tuple(tap.physical_ids[index] for index in selected.tolist())
    fit_root = _physical_root(fit_ids)
    class_rows: dict[str, list[np.ndarray]] = {class_handle: [] for class_handle in old_classes}
    cell_class_rows: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for index in selected.tolist():
        receiver = tap.receiver_ids[index]
        day = tap.day_ids[index]
        class_handle = tap.tx_labels[index]
        if receiver == held_receiver or class_handle not in class_rows:
            raise NextR4FAAssetBuildError("outer-cell filtering drift")
        row = r0[index]
        class_rows[class_handle].append(row)
        cell_class_rows.setdefault((receiver, day, class_handle), []).append(row)
    if any(len(class_rows[class_handle]) != EXPECTED_OUTER_PER_CLASS for class_handle in old_classes):
        raise NextR4FAAssetBuildError("outer-cell class balancing drift")
    cell_keys = tuple(sorted({(receiver, day) for receiver, day, _ in cell_class_rows}))
    if len(cell_keys) != (EXPECTED_RECEIVERS - 1) * EXPECTED_DAYS:
        raise NextR4FAAssetBuildError("outer-cell receiver-day grid drift")
    if any(
        (receiver, day, class_handle) not in cell_class_rows
        or not d106.D104_CELL_MIN_SAMPLES <= len(cell_class_rows[(receiver, day, class_handle)]) <= d106.D104_CELL_MAX_SAMPLES
        for receiver, day in cell_keys
        for class_handle in old_classes
    ):
        raise NextR4FAAssetBuildError("outer-cell receiver-day-class coverage drift")

    # This exactly follows D106's Phase1 geometry.  The only changed facts are
    # the already frozen held receiver and held class, so it operates on 24
    # receiver-day cells and five balanced classes rather than 28/6.
    cell_means = {
        key: d106._mean_canonical(rows)  # noqa: SLF001 - exact frozen D106 formula.
        for key, rows in cell_class_rows.items()
    }
    class_cell_means: dict[str, list[tuple[tuple[str, str], np.ndarray]]] = {
        class_handle: [] for class_handle in old_classes
    }
    cell_values: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {
        key: [] for key in cell_keys
    }
    for (receiver, day, class_handle), value in cell_means.items():
        cell = (receiver, day)
        class_cell_means[class_handle].append((cell, value))
        cell_values[cell].append((class_handle, value))
    class_reference: dict[str, np.ndarray] = {}
    for class_handle in old_classes:
        entries = class_cell_means[class_handle]
        if len(entries) != len(cell_keys):
            raise NextR4FAAssetBuildError("outer-cell class receiver-day coverage drift")
        ordered = [
            value
            for _cell, value in sorted(
                entries,
                key=lambda item: d106._content_key([item[1]]),  # noqa: SLF001
            )
        ]
        class_reference[class_handle] = d106._mean_canonical(ordered)  # noqa: SLF001
    g_rows: list[tuple[bytes, np.ndarray]] = []
    for cell in cell_keys:
        entries = cell_values[cell]
        if len(entries) != len(old_classes):
            raise NextR4FAAssetBuildError("outer-cell class-equal cell mean drift")
        deltas = [value - class_reference[class_handle] for class_handle, value in entries]
        g = d106._mean_canonical(  # noqa: SLF001
            sorted(deltas, key=lambda value: d106._content_key([value]))  # noqa: SLF001
        )
        g_rows.append((d106._content_key([value for _class_handle, value in entries]), g))  # noqa: SLF001
    ordered_g = [value for _key, value in sorted(g_rows, key=lambda item: item[0])]
    if len(ordered_g) != len(cell_keys):
        raise NextR4FAAssetBuildError("outer-cell common-shift count drift")
    centered_g = np.stack(ordered_g, axis=0)
    centered_g = centered_g - d106._mean_canonical(ordered_g)  # noqa: SLF001
    scatter = (centered_g.T @ centered_g) / float(len(cell_keys))
    try:
        raw_basis = d106._canonical_top_eigensystem(scatter)  # noqa: SLF001
    except Exception as error:
        raise NextR4FAAssetBuildError("outer-cell canonical rank-three basis is unavailable") from error

    # FA-RDCE3 deploys the actual INT8 basis followed by its deterministic
    # orthogonal closure.  Every aggregate below uses that deployed coordinate,
    # never the pre-quantization eigenvectors.
    basis_f32 = np.ascontiguousarray(raw_basis, dtype=np.float32)
    try:
        basis_codes, basis_scales = fa._quantize_signed_rows(  # noqa: SLF001
            basis_f32, name="FA-RDCE3 Phase1 RDCE basis"
        )
        closed_basis = fa._decoded_basis(basis_codes, basis_scales)  # noqa: SLF001
        tau = d106._balanced_tau_for_closed_basis(  # noqa: SLF001
            [class_rows[class_handle] for class_handle in old_classes], closed_basis
        )
    except Exception as error:
        raise NextR4FAAssetBuildError("outer-cell closed-basis residual variance is invalid") from error
    spectrum = np.diag(closed_basis @ scatter @ closed_basis.T)
    if (
        not np.isfinite(tau).all()
        or not np.isfinite(spectrum).all()
        or np.any(tau <= d106.EPSILON)
        or np.any(spectrum <= d106.MIN_SPECTRUM)
    ):
        raise NextR4FAAssetBuildError("outer-cell tau/spectrum is nonpositive")
    fisher_precision = 1.0 / spectrum
    kappa = spectrum / (spectrum + tau)
    if (
        not np.isfinite(fisher_precision).all()
        or not np.isfinite(kappa).all()
        or np.any(fisher_precision <= 0.0)
        or np.any(kappa < 0.0)
        or np.any(kappa >= 1.0)
    ):
        raise NextR4FAAssetBuildError("outer-cell Fisher/kappa aggregate is invalid")
    centers = np.stack(
        [
            d106._mean_canonical(class_rows[class_handle]) @ closed_basis.T  # noqa: SLF001
            for class_handle in old_classes
        ],
        axis=0,
    )
    radius = np.asarray([math.sqrt(float(fa.RANK))], dtype=np.float32)
    values = {
        "class_centers_3d": np.asarray(centers, dtype=np.float32),
        "fisher_precision_3d": np.asarray(fisher_precision, dtype=np.float32),
        "residual_variance_3d": np.asarray(tau, dtype=np.float32),
        "fisher_radius": radius,
        "rdce_kappa_3d": np.asarray(kappa, dtype=np.float32),
        "basis_3x160": basis_f32,
    }
    if any(not np.isfinite(array).all() for array in values.values()):
        raise NextR4FAAssetBuildError("outer-cell float32 aggregate conversion is non-finite")
    aggregate_sha = _aggregate_receipt(
        held_receiver=held_receiver,
        held_class=held_class,
        old_classes=old_classes,
        phase1_fit_count=len(fit_ids),
        phase1_fit_physical_root_sha256=fit_root,
        strict_tap_sha256=tap.archive_sha256,
        checkpoint_sha256=checkpoint_sha256,
        method_lock_sha256=method_lock_sha256,
        **values,
    )
    try:
        asset = fa.build_fa_rdce3_phase1_asset(
            old_classes=old_classes,
            aggregate_samples_per_class=tuple(len(class_rows[class_handle]) for class_handle in old_classes),
            checkpoint_sha256=checkpoint_sha256,
            # The sealed strict tap is the Phase1-only source bundle consumed
            # by this builder.  No Phase1 path is retained in the resulting wire.
            phase1_bundle_sha256=tap.archive_sha256,
            phase1_aggregate_receipt_sha256=aggregate_sha,
            method_lock_sha256=method_lock_sha256,
            **values,
        )
        wire = fa.serialize_fa_rdce3_phase1_asset(asset)
        recovered = fa.deserialize_fa_rdce3_phase1_asset(wire)
    except Exception as error:
        raise NextR4FAAssetBuildError("FA-RDCE3 aggregate wire failed closed construction") from error
    if recovered.asset_sha256 != asset.asset_sha256 or _sha(wire) != asset.asset_sha256:
        raise NextR4FAAssetBuildError("FA-RDCE3 aggregate wire hash/roundtrip drift")
    return PreparedFARDCE3Asset(
        held_receiver=held_receiver,
        held_class=held_class,
        asset=asset,
        wire=wire,
        phase1_fit_count=len(fit_ids),
        phase1_fit_physical_root_sha256=fit_root,
        aggregate_receipt_sha256=aggregate_sha,
    )


def prepare_next_r4_fa_rdce3_assets(
    *,
    strict_tap: Path,
    strict_tap_sha256: str,
    checkpoint_sha256: str,
    method_lock_sha256: str,
) -> tuple[PreparedFARDCE3Asset, ...]:
    """Read one strict tap and prepare the 12 assets without writing output."""

    expected_tap_sha = _require_sha256(strict_tap_sha256, name="strict tap SHA256")
    checkpoint_sha = _require_sha256(checkpoint_sha256, name="checkpoint SHA256")
    method_lock_sha = _require_sha256(method_lock_sha256, name="method lock SHA256")
    tap = _load_strict_tap(strict_tap, expected_tap_sha)
    _receivers, classes = _validate_d106_grid(tap)
    try:
        r0 = d106_r0.canonical_d106_relu_zid160(tap.pre_relu)
    except Exception as error:
        raise NextR4FAAssetBuildError("D105 bridge-compatible canonical R0 conversion failed") from error
    if r0.dtype != np.float32 or r0.shape != (EXPECTED_ROWS, fa.Z_DIM):
        raise NextR4FAAssetBuildError("canonical R0 dtype/shape drift")
    norms = np.linalg.norm(r0.astype(np.float64), axis=1)
    if np.any(r0 < 0.0) or not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise NextR4FAAssetBuildError("canonical R0 nonnegative/unit closure drift")
    assets = tuple(
        _geometry_for_outer(
            tap=tap,
            r0=r0,
            held_receiver=held_receiver,
            held_class=held_class,
            all_classes=classes,
            checkpoint_sha256=checkpoint_sha,
            method_lock_sha256=method_lock_sha,
        )
        for held_receiver in matrix.HELD_RECEIVERS
        for held_class in classes
    )
    if len(assets) != len(matrix.HELD_RECEIVERS) * len(classes):
        raise NextR4FAAssetBuildError("FA-RDCE3 outer asset cardinality drift")
    if len({item.outer_key for item in assets}) != len(assets):
        raise NextR4FAAssetBuildError("FA-RDCE3 outer asset key collision")
    if any(
        item.phase1_fit_count != EXPECTED_OUTER_FIT_ROWS
        or item.asset.aggregate_samples_per_class != (EXPECTED_OUTER_PER_CLASS,) * (EXPECTED_CLASSES - 1)
        for item in assets
    ):
        raise NextR4FAAssetBuildError("FA-RDCE3 aggregate count closure drift")
    return assets


def _new_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve(strict=False)
    if (
        not output_dir.is_absolute()
        or output_dir.exists()
        or output_dir.is_symlink()
        or not resolved.parent.is_dir()
    ):
        raise NextR4FAAssetBuildError("output directory must be a new absolute child of an existing directory")
    return resolved


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise NextR4FAAssetBuildError(f"output overwrite/path drift: {path}")
    with path.open("xb") as handle:
        handle.write(payload)


def build_next_r4_fa_rdce3_assets(
    *,
    strict_tap: Path,
    strict_tap_sha256: str,
    checkpoint_sha256: str,
    method_lock_sha256: str,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Materialize an immutable directory of 12 asset wires and one manifest."""

    assets = prepare_next_r4_fa_rdce3_assets(
        strict_tap=strict_tap,
        strict_tap_sha256=strict_tap_sha256,
        checkpoint_sha256=checkpoint_sha256,
        method_lock_sha256=method_lock_sha256,
    )
    root = _new_output_dir(output_dir)
    manifest_entries: dict[str, dict[str, str]] = {}
    root.mkdir()
    for prepared in assets:
        filename = f"fa_rdce3_{_sha(prepared.outer_key.encode('utf-8'))[:20]}.wire"
        destination = root / filename
        _write_new(destination, prepared.wire)
        manifest_entries[prepared.outer_key] = {
            "asset_path": str(destination.resolve()),
            "asset_sha256": _sha(prepared.wire),
            "checkpoint_sha256": prepared.asset.checkpoint_sha256,
            "phase1_fit_physical_root_sha256": prepared.phase1_fit_physical_root_sha256,
        }
    manifest = {"schema": ASSET_MANIFEST_SCHEMA, "entries": manifest_entries}
    manifest_path = root / "fa_asset_manifest.json"
    _write_new(manifest_path, _canonical(manifest))
    manifest_sha = _sha(manifest_path.read_bytes())
    return {
        "status": BUILD_STATUS,
        "output_dir": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "asset_count": len(assets),
        "outer_phase1_fit_count": EXPECTED_OUTER_FIT_ROWS,
        "old_classes_per_asset": EXPECTED_CLASSES - 1,
        "phase1_member_ids_written": False,
        "phase1_per_row_features_written": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build 12 aggregate-only NEXT-R4 FA-RDCE3 assets from one D106 strict tap"
    )
    parser.add_argument("--strict-tap", required=True, type=Path)
    parser.add_argument("--strict-tap-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_next_r4_fa_rdce3_assets(
        strict_tap=args.strict_tap,
        strict_tap_sha256=args.strict_tap_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        method_lock_sha256=args.method_lock_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry.
    raise SystemExit(main())
