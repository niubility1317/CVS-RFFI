"""Seal one immutable LEO-weak received-IQ cache for held source-V rows.

The source-V cache is deliberately distinct from the existing source-L cache:
it contains the 16,800-row held validation role only, never fits geometry or
thresholds, and is shared byte-for-byte by the C/G arms of one frozen fold.
It is a post-target completion audit artifact and cannot feed candidate
selection, tuning, retry, revival or promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import build_phase1_clic_source_leo_iq as _source_l


FORMAL_LEO_WEAK_SCENARIOS = _source_l.FORMAL_LEO_WEAK_SCENARIOS
SOURCE_V_LEO_CACHE_SCHEMA = "cvs.phase1.clic_source_v_leo_received_iq.v1"
SOURCE_V_ROLE = "source_validation_known_leo_weak"
EXPECTED_TRAINING_RUN_ID = "phase1_clic12_20260812_v5"
EXPECTED_CLEAN_RUN_ID = "phase1_clic_postfreeze_20260812_v4"
EXPECTED_CACHE_RUN_ID = "phase1_clic_source_metrics_20260816_v4"
FROZEN_SOURCE_V_ROW_COUNT = 16_800
FROZEN_SOURCE_CLASS_COUNT = 4
FROZEN_SOURCE_RECEIVER_COUNT = 7
FROZEN_SOURCE_V_DAY_IDS = ("2021_03_01", "2021_03_08")
FROZEN_SOURCE_DAY_COUNT = 2
FROZEN_SOURCE_V_ROWS_PER_TX_RX_CELL = 600
FROZEN_SOURCE_V_ROWS_PER_TX_RX_DAY = 300
FROZEN_WISIG_SHA256 = _source_l.FROZEN_WISIG_SHA256
SOURCE_LEO_SEED_OFFSET = _source_l.SOURCE_LEO_SEED_OFFSET
SOURCE_LEO_SCENE_SEED_STRIDE = _source_l.SOURCE_LEO_SCENE_SEED_STRIDE


class CLICSourceVLeoCacheError(RuntimeError):
    """Raised when the held source-V cache cannot be sealed safely."""


def _tensor_to_numpy_float32(value: torch.Tensor) -> np.ndarray:
    """Use the already-audited source-L safe Torch-to-array bridge."""

    try:
        return _source_l._tensor_to_numpy_float32(value)
    except _source_l.CLICSourceLeoCacheError as exc:
        raise CLICSourceVLeoCacheError("source-V tensor conversion failed") from exc


def _numpy_float32_to_tensor(value: np.ndarray, *, device: torch.device) -> torch.Tensor:
    """Use the already-audited source-L safe array-to-Torch bridge."""

    try:
        return _source_l._numpy_float32_to_tensor(value, device=device)
    except _source_l.CLICSourceLeoCacheError as exc:
        raise CLICSourceVLeoCacheError("source-V array conversion failed") from exc


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CLICSourceVLeoCacheError("cannot canonicalize source-V cache state") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_string_rows(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise CLICSourceVLeoCacheError(f"{label} rows are invalid") from exc
    if any(not isinstance(value, str) or not value for value in rows):
        raise CLICSourceVLeoCacheError(
            f"{label} rows contain an empty or non-string ID"
        )
    return rows


def _parse_csv(value: str, *, label: str, expected: int) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if len(parsed) != expected or len(set(parsed)) != expected:
        raise CLICSourceVLeoCacheError(
            f"{label} must contain exactly {expected} unique IDs"
        )
    return parsed


def _physical_key(
    *, tx_id: str, rx_id: str, day_id: str, eq_id: str, sig_id: str
) -> str:
    return "\x1f".join((tx_id, rx_id, day_id, eq_id, sig_id))


def _physical_sample_id(
    *,
    dataset_sha256: str,
    tx_id: str,
    rx_id: str,
    day_id: str,
    eq_id: str,
    sig_id: str,
) -> str:
    payload = "\x1f".join(
        (
            SOURCE_V_LEO_CACHE_SCHEMA,
            dataset_sha256,
            tx_id,
            rx_id,
            day_id,
            eq_id,
            sig_id,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_frozen_source_v_axes_and_cell_day_coverage(
    *,
    tx_ids: Sequence[str],
    rx_ids: Sequence[str],
    day_ids: Sequence[str],
    label: str,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, int]]:
    """Freeze the held-V physical day labels and every TX/RX/day cell count."""

    tx_rows = _strict_string_rows(tx_ids, label="TX")
    rx_rows = _strict_string_rows(rx_ids, label="RX")
    day_rows = _strict_string_rows(day_ids, label="day")
    if not (len(tx_rows) == len(rx_rows) == len(day_rows)):
        raise CLICSourceVLeoCacheError(f"{label} metadata row lengths drifted")
    classes = tuple(sorted(set(tx_rows)))
    receivers = tuple(sorted(set(rx_rows)))
    days = tuple(sorted(set(day_rows)))
    if len(classes) != FROZEN_SOURCE_CLASS_COUNT:
        raise CLICSourceVLeoCacheError(
            f"{label} must contain exactly four TX classes"
        )
    if len(receivers) != FROZEN_SOURCE_RECEIVER_COUNT:
        raise CLICSourceVLeoCacheError(
            f"{label} must contain exactly seven RX cells"
        )
    if days != FROZEN_SOURCE_V_DAY_IDS:
        raise CLICSourceVLeoCacheError(f"{label} frozen day axis drifted")

    counts = {
        (tx_id, rx_id, day_id): 0
        for tx_id in classes
        for rx_id in receivers
        for day_id in FROZEN_SOURCE_V_DAY_IDS
    }
    for tx_id, rx_id, day_id in zip(tx_rows, rx_rows, day_rows, strict=True):
        counts[(tx_id, rx_id, day_id)] += 1
    for (tx_id, rx_id, day_id), observed in counts.items():
        if observed != FROZEN_SOURCE_V_ROWS_PER_TX_RX_DAY:
            raise CLICSourceVLeoCacheError(
                f"{label} TX/RX/day coverage drifted: "
                f"tx={tx_id!r} rx={rx_id!r} day={day_id!r} "
                f"observed={observed} expected={FROZEN_SOURCE_V_ROWS_PER_TX_RX_DAY}"
            )
    return (
        classes,
        receivers,
        {
            f"{tx_id}|{rx_id}|{day_id}": observed
            for (tx_id, rx_id, day_id), observed in counts.items()
        },
    )


def assign_source_v_scenarios(
    tx_ids: Sequence[str],
    rx_ids: Sequence[str],
    day_ids: Sequence[str],
    physical_sample_ids: Sequence[str],
) -> dict[str, str]:
    """Assign stable single-scene LEO observations to every held-V physical row.

    The assignment key is the opaque physical ID, not the input row position.
    It therefore survives loader/input permutations and never offers a second
    weak-channel draw for the same physical sample.
    """

    tx_rows = _strict_string_rows(tx_ids, label="TX")
    rx_rows = _strict_string_rows(rx_ids, label="RX")
    day_rows = _strict_string_rows(day_ids, label="day")
    physical_rows = _strict_string_rows(physical_sample_ids, label="physical")
    if not (
        len(tx_rows)
        == len(rx_rows)
        == len(day_rows)
        == len(physical_rows)
        == FROZEN_SOURCE_V_ROW_COUNT
    ):
        raise CLICSourceVLeoCacheError(
            "source-V must contain exactly 16800 aligned physical rows"
        )
    if len(set(physical_rows)) != len(physical_rows):
        raise CLICSourceVLeoCacheError(
            "source-V physical sample IDs must be globally unique"
        )
    classes, receivers, _ = _validate_frozen_source_v_axes_and_cell_day_coverage(
        tx_ids=tx_rows,
        rx_ids=rx_rows,
        day_ids=day_rows,
        label="source-V",
    )

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tx_id, rx_id, physical_id in zip(tx_rows, rx_rows, physical_rows, strict=True):
        grouped[(tx_id, rx_id)].append(physical_id)
    assignment: dict[str, str] = {}
    for tx_id in classes:
        for rx_id in receivers:
            cell = sorted(grouped.get((tx_id, rx_id), ()))
            if len(cell) != FROZEN_SOURCE_V_ROWS_PER_TX_RX_CELL:
                raise CLICSourceVLeoCacheError(
                    "source-V TX/RX cell requires exactly 600 frozen physical rows: "
                    f"tx={tx_id!r} rx={rx_id!r} observed={len(cell)}"
                )
            for rank, physical_id in enumerate(cell):
                assignment[physical_id] = FORMAL_LEO_WEAK_SCENARIOS[
                    rank % len(FORMAL_LEO_WEAK_SCENARIOS)
                ]
    if set(assignment) != set(physical_rows):
        raise CLICSourceVLeoCacheError(
            "source-V scene assignment did not cover every physical row"
        )

    scene_values = [assignment[physical_id] for physical_id in physical_rows]
    coverage = _coverage_counts(tx_rows, rx_rows, day_rows, scene_values)
    for label in (
        "scenario_coverage",
        "scenario_class_coverage",
        "scenario_rx_coverage",
        "scenario_day_coverage",
    ):
        counts = coverage[label]
        if not counts or min(counts.values()) <= 0:
            raise CLICSourceVLeoCacheError(
                f"source-V {label} contains a zero denominator"
            )
    return assignment


def _coverage_counts(
    tx_ids: Sequence[str],
    rx_ids: Sequence[str],
    day_ids: Sequence[str],
    scenes: Sequence[str],
) -> dict[str, dict[str, int]]:
    if not (len(tx_ids) == len(rx_ids) == len(day_ids) == len(scenes)):
        raise CLICSourceVLeoCacheError("source-V coverage row lengths drifted")
    classes = tuple(sorted(set(tx_ids)))
    receivers = tuple(sorted(set(rx_ids)))
    days = tuple(sorted(set(day_ids)))
    by_scene = {scene: 0 for scene in FORMAL_LEO_WEAK_SCENARIOS}
    by_class = {
        f"{scene}|{tx_id}": 0
        for scene in FORMAL_LEO_WEAK_SCENARIOS
        for tx_id in classes
    }
    by_rx = {
        f"{scene}|{rx_id}": 0
        for scene in FORMAL_LEO_WEAK_SCENARIOS
        for rx_id in receivers
    }
    by_day = {
        f"{scene}|{day_id}": 0
        for scene in FORMAL_LEO_WEAK_SCENARIOS
        for day_id in days
    }
    by_cell: dict[str, int] = {}
    for tx_id, rx_id, day_id, scene in zip(tx_ids, rx_ids, day_ids, scenes, strict=True):
        if scene not in FORMAL_LEO_WEAK_SCENARIOS:
            raise CLICSourceVLeoCacheError("source-V coverage contains an unapproved scene")
        by_scene[scene] += 1
        by_class[f"{scene}|{tx_id}"] += 1
        by_rx[f"{scene}|{rx_id}"] += 1
        by_day[f"{scene}|{day_id}"] += 1
        key = f"{scene}|{tx_id}|{rx_id}|{day_id}"
        by_cell[key] = by_cell.get(key, 0) + 1
    return {
        "scenario_coverage": by_scene,
        "scenario_class_coverage": by_class,
        "scenario_rx_coverage": by_rx,
        "scenario_day_coverage": by_day,
        "scenario_tx_rx_day_coverage": by_cell,
    }


@dataclass(frozen=True)
class _ImmutablePublication:
    """Identity of a file published through the no-replace link operation."""

    path: Path
    device: int
    inode: int
    sha256: str | None


def _regular_file_identity(path: Path, *, label: str) -> tuple[int, int]:
    """Return a platform-neutral regular-file identity without following links."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CLICSourceVLeoCacheError(f"{label} cannot be statted: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CLICSourceVLeoCacheError(f"{label} must remain a regular file: {path}")
    return int(metadata.st_dev), int(metadata.st_ino)


def _unlink_if_owned(
    publication: _ImmutablePublication, *, require_sha: bool = True
) -> bool:
    """Remove only a path still owned by this builder's exact publication.

    A successfully exclusive-created temporary has a trustworthy pathname and
    inode before its payload can be sealed, so its write-failure cleanup uses
    identity-only ownership.  Published final artifacts additionally require
    the pre-publish SHA to match, preserving any in-place external mutation.
    """

    path = publication.path
    try:
        identity = _regular_file_identity(path, label="immutable publication")
    except CLICSourceVLeoCacheError:
        return False
    if identity != (publication.device, publication.inode):
        return False
    if require_sha:
        if not isinstance(publication.sha256, str) or len(publication.sha256) != 64:
            return False
        try:
            if _sha256_file(path) != publication.sha256:
                return False
        except OSError:
            return False
    try:
        path.unlink()
    except OSError as exc:
        raise CLICSourceVLeoCacheError(
            f"unable to clean builder-owned immutable publication: {path}"
        ) from exc
    return True


def _assert_publication_current(
    publication: _ImmutablePublication, *, expected_sha256: str, label: str
) -> None:
    """Reject a post-publish replacement or in-place mutation before sealing."""

    if not isinstance(publication.sha256, str) or len(publication.sha256) != 64:
        raise CLICSourceVLeoCacheError(
            f"immutable {label} has no pre-publish SHA seal: {publication.path}"
        )
    if expected_sha256 != publication.sha256:
        raise CLICSourceVLeoCacheError(
            f"immutable {label} requested SHA differs from its pre-publish seal: {publication.path}"
        )
    expected_identity = (publication.device, publication.inode)
    observed_identity = _regular_file_identity(publication.path, label=label)
    if observed_identity != expected_identity:
        raise CLICSourceVLeoCacheError(
            f"immutable {label} publication identity changed after publish: {publication.path}"
        )
    observed_sha256 = _sha256_file(publication.path)
    if observed_sha256 != publication.sha256:
        raise CLICSourceVLeoCacheError(
            f"immutable {label} publication bytes changed after publish: {publication.path}"
        )


def _publish_immutable_temporary(
    temporary: Path,
    path: Path,
    *,
    label: str,
    temporary_publication: _ImmutablePublication,
) -> _ImmutablePublication:
    """Publish one same-directory temporary file without ever replacing a target.

    ``os.link`` is an exclusive destination creation on both NTFS and POSIX
    filesystems: it either links our already-fsynced temporary bytes to a new
    destination name, or reports that another owner won the name.  Unlike
    ``Path.replace``, it cannot overwrite a concurrent immutable artifact.
    """

    try:
        _assert_publication_current(
            temporary_publication,
            expected_sha256=temporary_publication.sha256,
            label=f"temporary {label}",
        )
        temporary_identity = (temporary_publication.device, temporary_publication.inode)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CLICSourceVLeoCacheError(
                f"refusing to overwrite immutable {label} during concurrent publish: {path}"
            ) from exc
        except OSError as exc:
            raise CLICSourceVLeoCacheError(
                f"exclusive immutable {label} publish failed: {path}"
            ) from exc
        try:
            destination_identity = _regular_file_identity(path, label=f"published {label}")
            if destination_identity != temporary_identity:
                raise CLICSourceVLeoCacheError(
                    f"immutable {label} destination changed during exclusive publish: {path}"
                )
            return _ImmutablePublication(
                path=path,
                device=destination_identity[0],
                inode=destination_identity[1],
                sha256=temporary_publication.sha256,
            )
        except Exception:
            _unlink_if_owned(
                _ImmutablePublication(
                    path=path,
                    device=temporary_identity[0],
                    inode=temporary_identity[1],
                    sha256=temporary_publication.sha256,
                )
            )
            raise
    finally:
        _unlink_if_owned(temporary_publication)


def _atomic_save_npz(path: Path, payload: Mapping[str, Any]) -> _ImmutablePublication:
    if path.exists():
        raise CLICSourceVLeoCacheError(
            f"refusing to overwrite immutable source-V received-IQ cache: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CLICSourceVLeoCacheError(
            f"refusing to overwrite temporary source-V cache: {temporary}"
        )
    temporary_publication: _ImmutablePublication | None = None
    try:
        with temporary.open("xb") as handle:
            temporary_identity = _regular_file_identity(
                temporary, label="temporary source-V received-IQ cache"
            )
            temporary_publication = _ImmutablePublication(
                path=temporary,
                device=temporary_identity[0],
                inode=temporary_identity[1],
                sha256="",
            )
            np.savez(handle, **dict(payload))
            handle.flush()
            os.fsync(handle.fileno())
        temporary_publication = _ImmutablePublication(
            path=temporary,
            device=temporary_publication.device,
            inode=temporary_publication.inode,
            sha256=_sha256_file(temporary),
        )
        return _publish_immutable_temporary(
            temporary,
            path,
            label="source-V received-IQ cache",
            temporary_publication=temporary_publication,
        )
    except Exception:
        if temporary_publication is not None:
            _unlink_if_owned(temporary_publication, require_sha=False)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> _ImmutablePublication:
    if path.exists():
        raise CLICSourceVLeoCacheError(
            f"refusing to overwrite immutable source-V receipt: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CLICSourceVLeoCacheError(
            f"refusing to overwrite temporary source-V receipt: {temporary}"
        )
    temporary_publication: _ImmutablePublication | None = None
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            temporary_identity = _regular_file_identity(
                temporary, label="temporary source-V receipt"
            )
            temporary_publication = _ImmutablePublication(
                path=temporary,
                device=temporary_identity[0],
                inode=temporary_identity[1],
                sha256="",
            )
            handle.write(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_publication = _ImmutablePublication(
            path=temporary,
            device=temporary_publication.device,
            inode=temporary_publication.inode,
            sha256=_sha256_file(temporary),
        )
        return _publish_immutable_temporary(
            temporary,
            path,
            label="source-V receipt",
            temporary_publication=temporary_publication,
        )
    except Exception:
        if temporary_publication is not None:
            _unlink_if_owned(temporary_publication, require_sha=False)
        raise


def _data_config_projection(args: Mapping[str, Any]) -> dict[str, Any]:
    return _source_l._data_config_projection(args)


def _load_validated_arm(
    *,
    checkpoint_path: Path,
    terminal_path: Path,
    source_tx_ids: tuple[str, ...],
    known_validation_tx_ids: tuple[str, ...],
    proxy_unknown_tx_ids: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    try:
        return _source_l._load_validated_arm(
            checkpoint_path=checkpoint_path,
            terminal_path=terminal_path,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known_validation_tx_ids,
            proxy_unknown_tx_ids=proxy_unknown_tx_ids,
        )
    except _source_l.CLICSourceLeoCacheError as exc:
        raise CLICSourceVLeoCacheError(
            "source-V checkpoint/terminal reopening failed"
        ) from exc


def _read_clean_manifest(value: Any, *, label: str) -> dict[str, Any]:
    rendered = np.asarray(value)
    if rendered.size != 1:
        raise CLICSourceVLeoCacheError(f"{label} manifest must contain exactly one value")
    try:
        manifest = json.loads(str(rendered.reshape(-1)[0]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CLICSourceVLeoCacheError(f"{label} manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise CLICSourceVLeoCacheError(f"{label} manifest must be an object")
    return manifest


def _string_array(archive: Any, field: str, *, row_count: int, label: str) -> list[str]:
    if field not in archive.files:
        raise CLICSourceVLeoCacheError(f"{label} clean archive is missing {field}")
    values = np.asarray(archive[field], dtype=str).reshape(-1)
    if values.size != row_count or any(not str(value) for value in values.tolist()):
        raise CLICSourceVLeoCacheError(
            f"{label} clean archive {field} is empty or misaligned"
        )
    return [str(value) for value in values.tolist()]


def _read_clean_validation_binding(
    *,
    path: Path,
    arm: str,
    fold_index: int,
    source_tx_ids: tuple[str, ...],
    checkpoint_sha256: str,
    terminal_sha256: str,
) -> dict[str, Any]:
    candidate = f"F{fold_index}{arm}_CLIC12"
    if (
        path.name != "source_clean_proxy.npz"
        or path.parent.name != candidate
        or path.parent.parent.name != EXPECTED_CLEAN_RUN_ID
    ):
        raise CLICSourceVLeoCacheError(
            "source-V clean input has a noncanonical run/candidate/output binding"
        )
    if not path.is_file():
        raise CLICSourceVLeoCacheError(f"{arm} clean-v4 input is missing")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "dataset_role",
                "tx_ids",
                "rx_ids",
                "day_ids",
                "eq_ids",
                "sig_ids",
                "manifest_json",
            }
            if not required.issubset(set(archive.files)):
                raise CLICSourceVLeoCacheError(
                    f"{arm} clean-v4 metadata contract is incomplete"
                )
            roles = np.asarray(archive["dataset_role"], dtype=str).reshape(-1)
            if roles.size != 21_120:
                raise CLICSourceVLeoCacheError(
                    f"{arm} clean-v4 total row count drifted"
                )
            tx_rows = _string_array(archive, "tx_ids", row_count=roles.size, label=arm)
            rx_rows = _string_array(archive, "rx_ids", row_count=roles.size, label=arm)
            day_rows = _string_array(archive, "day_ids", row_count=roles.size, label=arm)
            eq_rows = _string_array(archive, "eq_ids", row_count=roles.size, label=arm)
            sig_rows = _string_array(archive, "sig_ids", row_count=roles.size, label=arm)
            manifest = _read_clean_manifest(archive["manifest_json"], label=arm)
    except (OSError, ValueError) as exc:
        raise CLICSourceVLeoCacheError(f"{arm} clean-v4 archive cannot be opened") from exc
    role_rows = [str(value) for value in roles.tolist()]
    allowed_roles = {"labeled_fit", "source_validation_known", "proxy_unknown"}
    if set(role_rows) != allowed_roles:
        raise CLICSourceVLeoCacheError(f"{arm} clean-v4 role set drifted")
    masks = {role: [value == role for value in role_rows] for role in allowed_roles}
    expected_counts = {
        "labeled_fit": 3920,
        "source_validation_known": FROZEN_SOURCE_V_ROW_COUNT,
        "proxy_unknown": 400,
    }
    for role, expected_count in expected_counts.items():
        if sum(masks[role]) != expected_count:
            raise CLICSourceVLeoCacheError(
                f"{arm} clean-v4 {role} count drifted"
            )
    all_keys = [
        _physical_key(
            tx_id=tx_id,
            rx_id=rx_id,
            day_id=day_id,
            eq_id=eq_id,
            sig_id=sig_id,
        )
        for tx_id, rx_id, day_id, eq_id, sig_id in zip(
            tx_rows, rx_rows, day_rows, eq_rows, sig_rows, strict=True
        )
    ]
    if len(all_keys) != len(set(all_keys)):
        raise CLICSourceVLeoCacheError(
            f"{arm} clean-v4 L/V/proxy physical rows overlap or repeat"
        )
    validation_positions = [
        index for index, role in enumerate(role_rows) if role == "source_validation_known"
    ]
    validation_keys = tuple(all_keys[index] for index in validation_positions)
    validation_tx = tuple(tx_rows[index] for index in validation_positions)
    validation_rx = tuple(rx_rows[index] for index in validation_positions)
    validation_day = tuple(day_rows[index] for index in validation_positions)
    validation_eq = tuple(eq_rows[index] for index in validation_positions)
    validation_sig = tuple(sig_rows[index] for index in validation_positions)
    if tuple(sorted(set(validation_tx))) != tuple(sorted(source_tx_ids)):
        raise CLICSourceVLeoCacheError(f"{arm} clean-v4 V TX class order/set drifted")
    _, _, validation_tx_rx_day_coverage = (
        _validate_frozen_source_v_axes_and_cell_day_coverage(
            tx_ids=validation_tx,
            rx_ids=validation_rx,
            day_ids=validation_day,
            label=f"{arm} clean-v4 V",
        )
    )
    expected_manifest = {
        "schema": "cvs.phase1.clic_lv_export.v1",
        "method": "P1_CLIC",
        "source_only": True,
        "candidate_id": candidate,
        "run_id": EXPECTED_TRAINING_RUN_ID,
        "source_tx_ids": list(source_tx_ids),
        "labeled_validation_physical_disjoint": True,
        "labeled_validation_proxy_physical_disjoint": True,
        "labeled_row_count": 3920,
        "source_validation_row_count": FROZEN_SOURCE_V_ROW_COUNT,
        "proxy_row_count": 400,
        "source_checkpoint_sha256": checkpoint_sha256,
        "terminal_receipt_sha256": terminal_sha256,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise CLICSourceVLeoCacheError(
                f"{arm} clean-v4 manifest {field} drifted"
            )
    if manifest.get("source_receiver_ids") != sorted(set(validation_rx)):
        raise CLICSourceVLeoCacheError(
            f"{arm} clean-v4 manifest source receiver axis drifted"
        )
    if manifest.get("source_day_ids") != list(FROZEN_SOURCE_V_DAY_IDS):
        raise CLICSourceVLeoCacheError(
            f"{arm} clean-v4 manifest source day axis drifted"
        )
    validation_order_sha = _canonical_sha256(list(validation_keys))
    manifest_order_sha = manifest.get("source_validation_physical_order_sha256")
    if not isinstance(manifest_order_sha, str) or manifest_order_sha != validation_order_sha:
        raise CLICSourceVLeoCacheError(
            f"{arm} clean-v4 V physical metadata/order hash drifted"
        )
    manifest_index_sha = manifest.get("source_validation_indices_sha256")
    if not isinstance(manifest_index_sha, str) or len(manifest_index_sha) != 64:
        raise CLICSourceVLeoCacheError(
            f"{arm} clean-v4 V index hash is missing"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "validation_keys": validation_keys,
        "validation_tx_ids": validation_tx,
        "validation_rx_ids": validation_rx,
        "validation_day_ids": validation_day,
        "validation_tx_rx_day_coverage": validation_tx_rx_day_coverage,
        "validation_eq_ids": validation_eq,
        "validation_sig_ids": validation_sig,
        "validation_metadata_order_sha256": validation_order_sha,
        "validation_indices_sha256": manifest_index_sha,
        "manifest": manifest,
    }


def _collect_source_v_rows(
    dataset: Any,
    *,
    source_tx_ids: tuple[str, ...],
    dataset_sha256: str,
    batch_size: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    from export_spaceborne_features import _meta_to_list

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    clean_rows: list[np.ndarray] = []
    tx_ids: list[str] = []
    rx_ids: list[str] = []
    day_ids: list[str] = []
    eq_ids: list[str] = []
    sig_ids: list[str] = []
    physical_ids: list[str] = []
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) != 4:
            raise CLICSourceVLeoCacheError(
                "source-V cache builder expects (x,y,domain,metadata) batches"
            )
        x, y, _domain, meta = batch
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise CLICSourceVLeoCacheError("source-V cache batch tensors are malformed")
        clean = _tensor_to_numpy_float32(x)
        if clean.ndim != 3 or clean.shape[1] != 2 or not np.isfinite(clean).all():
            raise CLICSourceVLeoCacheError(
                "source-V clean IQ rows are non-finite or malformed"
            )
        count = int(clean.shape[0])
        meta_tx = [str(value) for value in _meta_to_list(meta, "tx", count)]
        meta_rx = [str(value) for value in _meta_to_list(meta, "rx", count)]
        meta_day = [str(value) for value in _meta_to_list(meta, "day", count)]
        meta_eq = [str(value) for value in _meta_to_list(meta, "equalized", count)]
        meta_sig = [str(value) for value in _meta_to_list(meta, "sig_i", count)]
        labels = [int(value) for value in y.detach().cpu().reshape(-1).tolist()]
        if len(labels) != count:
            raise CLICSourceVLeoCacheError("source-V label row count drifted")
        for index in range(count):
            label = labels[index]
            if label not in range(len(source_tx_ids)) or source_tx_ids[label] != meta_tx[index]:
                raise CLICSourceVLeoCacheError("source-V local label/TX binding drifted")
            physical_ids.append(
                _physical_sample_id(
                    dataset_sha256=dataset_sha256,
                    tx_id=meta_tx[index],
                    rx_id=meta_rx[index],
                    day_id=meta_day[index],
                    eq_id=meta_eq[index],
                    sig_id=meta_sig[index],
                )
            )
        clean_rows.append(clean)
        tx_ids.extend(meta_tx)
        rx_ids.extend(meta_rx)
        day_ids.extend(meta_day)
        eq_ids.extend(meta_eq)
        sig_ids.extend(meta_sig)
    if not clean_rows:
        raise CLICSourceVLeoCacheError("source-V cache builder observed no validation rows")
    return {
        "clean_iq": np.concatenate(clean_rows, axis=0).astype(np.float32),
        "tx_ids": tx_ids,
        "rx_ids": rx_ids,
        "day_ids": day_ids,
        "eq_ids": eq_ids,
        "sig_ids": sig_ids,
        "physical_sample_ids": physical_ids,
    }


def _assert_output_binding(
    *, fold_index: int, cache_run_root: Path, output_path: Path, receipt_path: Path
) -> None:
    expected_output = (
        cache_run_root / f"F{fold_index}_SHARED" / "source_validation_known_leo_weak.npz"
    )
    expected_receipt = (
        cache_run_root
        / f"F{fold_index}_SHARED"
        / "source_validation_known_leo_weak.receipt.json"
    )
    if cache_run_root.name != EXPECTED_CACHE_RUN_ID:
        raise CLICSourceVLeoCacheError("source-V cache run binding drifted")
    if output_path != expected_output or receipt_path != expected_receipt:
        raise CLICSourceVLeoCacheError(
            "source-V cache output binding must be canonical F*_SHARED paths"
        )


def _input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in paths.items()}


def _validate_materialized_cache(path: Path, *, expected_row_count: int) -> None:
    expected_members = {
        "received_iq",
        "tx_ids",
        "rx_ids",
        "day_ids",
        "physical_sample_id",
        "sat_scenarios",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_members:
                raise CLICSourceVLeoCacheError("source-V cache member set drifted")
            received = np.asarray(archive["received_iq"], dtype=np.float32)
            row_count = int(received.shape[0]) if received.ndim == 3 else -1
            if (
                row_count != expected_row_count
                or received.shape[1] != 2
                or not np.isfinite(received).all()
            ):
                raise CLICSourceVLeoCacheError("source-V cache received IQ is malformed")
            physical = np.asarray(archive["physical_sample_id"], dtype=str).reshape(-1)
            scenes = np.asarray(archive["sat_scenarios"], dtype=str).reshape(-1)
            if physical.size != expected_row_count or scenes.size != expected_row_count:
                raise CLICSourceVLeoCacheError("source-V cache metadata row count drifted")
            if len(set(physical.tolist())) != expected_row_count:
                raise CLICSourceVLeoCacheError("source-V cache physical IDs repeat")
            if set(scenes.tolist()) != set(FORMAL_LEO_WEAK_SCENARIOS):
                raise CLICSourceVLeoCacheError("source-V cache formal scene coverage drifted")
            for field in ("tx_ids", "rx_ids", "day_ids"):
                if np.asarray(archive[field], dtype=str).reshape(-1).size != expected_row_count:
                    raise CLICSourceVLeoCacheError(
                        f"source-V cache {field} row count drifted"
                    )
    except (OSError, ValueError) as exc:
        raise CLICSourceVLeoCacheError("source-V cache cannot be reopened") from exc


def build_source_v_received_iq(args: argparse.Namespace) -> dict[str, Any]:
    """Build the one immutable held-V received-IQ cache for one C/G fold."""

    import export_phase1_clic_features as clean_export
    from cvsrffi.eval import apply_sat_channel_for_scenario
    from cvsrffi.tensors import make_torch_generator
    from dataset_wisig import WiSigSubsetDataset, load_wisig_compact_pkl
    from training_controls import sat_channel_config_for_scenario

    fold_index = int(args.fold_index)
    if fold_index not in range(1, 7):
        raise CLICSourceVLeoCacheError("source-V cache fold_index must be F1..F6")
    source_tx_ids = _parse_csv(args.source_tx_ids, label="source TX", expected=4)
    known_validation_tx_ids = _parse_csv(
        args.known_validation_tx_ids, label="known validation TX", expected=1
    )
    proxy_unknown_tx_ids = _parse_csv(
        args.proxy_unknown_tx_ids, label="proxy unknown TX", expected=1
    )
    if (
        set(source_tx_ids)
        & (set(known_validation_tx_ids) | set(proxy_unknown_tx_ids))
    ) or set(known_validation_tx_ids) & set(proxy_unknown_tx_ids):
        raise CLICSourceVLeoCacheError("source, held and proxy TX roles overlap")

    c_checkpoint_path = Path(args.c_ckpt).resolve()
    c_terminal_path = Path(args.c_terminal_receipt_json).resolve()
    c_clean_path = Path(args.c_clean_npz).resolve()
    g_checkpoint_path = Path(args.g_ckpt).resolve()
    g_terminal_path = Path(args.g_terminal_receipt_json).resolve()
    g_clean_path = Path(args.g_clean_npz).resolve()
    dataset_path = Path(args.wisig_pkl).resolve()
    cache_run_root = Path(args.cache_run_root).resolve()
    output_path = Path(args.out_npz).resolve()
    receipt_path = Path(args.receipt_json).resolve()
    _assert_output_binding(
        fold_index=fold_index,
        cache_run_root=cache_run_root,
        output_path=output_path,
        receipt_path=receipt_path,
    )
    for path, label in (
        (c_checkpoint_path, "C checkpoint"),
        (c_terminal_path, "C terminal"),
        (c_clean_path, "C clean-v4"),
        (g_checkpoint_path, "G checkpoint"),
        (g_terminal_path, "G terminal"),
        (g_clean_path, "G clean-v4"),
        (dataset_path, "WiSig dataset"),
    ):
        if not path.is_file():
            raise CLICSourceVLeoCacheError(f"source-V cache {label} is missing")
    if output_path.exists() or receipt_path.exists():
        raise CLICSourceVLeoCacheError("refusing to overwrite source-V cache output")
    expected_candidates = {
        "C": f"F{fold_index}C_CLIC12",
        "G": f"F{fold_index}G_CLIC12",
    }
    for arm, checkpoint_path, terminal_path in (
        ("C", c_checkpoint_path, c_terminal_path),
        ("G", g_checkpoint_path, g_terminal_path),
    ):
        if (
            checkpoint_path.name != "final_ssdg.pth"
            or checkpoint_path.parent.name != expected_candidates[arm]
            or checkpoint_path.parent.parent.name != EXPECTED_TRAINING_RUN_ID
            or terminal_path.parent != checkpoint_path.parent
        ):
            raise CLICSourceVLeoCacheError(
                f"source-V cache {arm} checkpoint/terminal binding drifted"
            )
    input_paths = {
        "c_checkpoint": c_checkpoint_path,
        "c_terminal": c_terminal_path,
        "c_clean": c_clean_path,
        "g_checkpoint": g_checkpoint_path,
        "g_terminal": g_terminal_path,
        "g_clean": g_clean_path,
        "wisig": dataset_path,
    }
    input_hashes_before = _input_hashes(input_paths)
    c_payload, c_args, c_terminal, c_arm = _load_validated_arm(
        checkpoint_path=c_checkpoint_path,
        terminal_path=c_terminal_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    g_payload, g_args, g_terminal, g_arm = _load_validated_arm(
        checkpoint_path=g_checkpoint_path,
        terminal_path=g_terminal_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    if (c_arm, g_arm) != ("C", "G"):
        raise CLICSourceVLeoCacheError("source-V cache requires one C and one G arm")
    if _data_config_projection(c_args) != _data_config_projection(g_args):
        raise CLICSourceVLeoCacheError(
            "source-V cache C/G data or channel configuration drifted"
        )
    for arm_args in (c_args, g_args):
        if Path(str(arm_args.get("wisig_pkl", ""))).resolve() != dataset_path:
            raise CLICSourceVLeoCacheError(
                "source-V cache dataset path differs from checkpoint"
            )
    expected_wisig_sha = str(args.expected_wisig_sha256).lower()
    dataset_sha = input_hashes_before["wisig"]
    if expected_wisig_sha != FROZEN_WISIG_SHA256 or dataset_sha != FROZEN_WISIG_SHA256:
        raise CLICSourceVLeoCacheError("source-V cache WiSig bytes drifted")

    c_clean = _read_clean_validation_binding(
        path=c_clean_path,
        arm="C",
        fold_index=fold_index,
        source_tx_ids=source_tx_ids,
        checkpoint_sha256=input_hashes_before["c_checkpoint"],
        terminal_sha256=input_hashes_before["c_terminal"],
    )
    g_clean = _read_clean_validation_binding(
        path=g_clean_path,
        arm="G",
        fold_index=fold_index,
        source_tx_ids=source_tx_ids,
        checkpoint_sha256=input_hashes_before["g_checkpoint"],
        terminal_sha256=input_hashes_before["g_terminal"],
    )
    for field in (
        "validation_keys",
        "validation_tx_ids",
        "validation_rx_ids",
        "validation_day_ids",
        "validation_tx_rx_day_coverage",
        "validation_eq_ids",
        "validation_sig_ids",
        "validation_metadata_order_sha256",
        "validation_indices_sha256",
    ):
        if c_clean[field] != g_clean[field]:
            raise CLICSourceVLeoCacheError(
                f"source-V cache C/G clean-v4 {field} does not share exact V binding"
            )

    raw_dataset = load_wisig_compact_pkl(str(dataset_path))
    if _sha256_file(dataset_path) != dataset_sha:
        raise CLICSourceVLeoCacheError("WiSig bytes changed while loading source-V rows")
    reconstructed = clean_export._reconstruct_source_l_v(
        raw_dataset=raw_dataset,
        checkpoint_args=c_args,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
        wisig_sha256=dataset_sha,
    )
    for checkpoint, terminal in ((c_payload, c_terminal), (g_payload, g_terminal)):
        clean_export._assert_current_source_split(
            checkpoint=checkpoint,
            receipt=terminal,
            reconstructed=reconstructed,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known_validation_tx_ids,
            proxy_unknown_tx_ids=proxy_unknown_tx_ids,
        )
    labeled_indices = tuple(int(value) for value in reconstructed["labeled_indices"])
    validation_indices = tuple(int(value) for value in reconstructed["validation_indices"])
    if (
        len(validation_indices) != FROZEN_SOURCE_V_ROW_COUNT
        or len(set(validation_indices)) != FROZEN_SOURCE_V_ROW_COUNT
        or set(labeled_indices) & set(validation_indices)
    ):
        raise CLICSourceVLeoCacheError(
            "source-V reconstructed validation indices do not close cleanly"
        )
    validation_index_sha = _canonical_sha256(list(validation_indices))
    if validation_index_sha != c_clean["validation_indices_sha256"]:
        raise CLICSourceVLeoCacheError(
            "source-V reconstructed validation index hash does not equal clean-v4"
        )
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise CLICSourceVLeoCacheError("source-V cache batch size must be positive")
    validation_dataset = WiSigSubsetDataset(
        reconstructed["source_base"],
        validation_indices,
        split_source="clic_source_validation_known_leo_weak",
    )
    rows = _collect_source_v_rows(
        validation_dataset,
        source_tx_ids=source_tx_ids,
        dataset_sha256=dataset_sha,
        batch_size=batch_size,
    )
    required_row_fields = (
        "clean_iq",
        "tx_ids",
        "rx_ids",
        "day_ids",
        "eq_ids",
        "sig_ids",
        "physical_sample_ids",
    )
    if any(field not in rows for field in required_row_fields):
        raise CLICSourceVLeoCacheError("source-V row collection contract is incomplete")
    row_count = len(rows["physical_sample_ids"])
    if row_count != FROZEN_SOURCE_V_ROW_COUNT:
        raise CLICSourceVLeoCacheError("source-V row collection count drifted")
    row_keys = tuple(
        _physical_key(
            tx_id=str(tx_id),
            rx_id=str(rx_id),
            day_id=str(day_id),
            eq_id=str(eq_id),
            sig_id=str(sig_id),
        )
        for tx_id, rx_id, day_id, eq_id, sig_id in zip(
            rows["tx_ids"],
            rows["rx_ids"],
            rows["day_ids"],
            rows["eq_ids"],
            rows["sig_ids"],
            strict=True,
        )
    )
    if row_keys != c_clean["validation_keys"]:
        raise CLICSourceVLeoCacheError(
            "source-V reconstructed metadata/order does not equal clean-v4 V"
        )
    if _canonical_sha256(list(row_keys)) != c_clean["validation_metadata_order_sha256"]:
        raise CLICSourceVLeoCacheError("source-V reconstructed V order hash drifted")
    _, _, source_validation_tx_rx_day_coverage = (
        _validate_frozen_source_v_axes_and_cell_day_coverage(
            tx_ids=rows["tx_ids"],
            rx_ids=rows["rx_ids"],
            day_ids=rows["day_ids"],
            label="source-V reconstructed V",
        )
    )
    if source_validation_tx_rx_day_coverage != c_clean[
        "validation_tx_rx_day_coverage"
    ]:
        raise CLICSourceVLeoCacheError(
            "source-V reconstructed TX/RX/day coverage does not equal clean-v4 V"
        )
    assignment = assign_source_v_scenarios(
        rows["tx_ids"],
        rows["rx_ids"],
        rows["day_ids"],
        rows["physical_sample_ids"],
    )
    scenes = [assignment[str(physical_id)] for physical_id in rows["physical_sample_ids"]]
    coverage = _coverage_counts(rows["tx_ids"], rows["rx_ids"], rows["day_ids"], scenes)

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    clean_iq = np.asarray(rows["clean_iq"], dtype=np.float32)
    if (
        clean_iq.ndim != 3
        or clean_iq.shape[0] != FROZEN_SOURCE_V_ROW_COUNT
        or clean_iq.shape[1] != 2
        or not np.isfinite(clean_iq).all()
    ):
        raise CLICSourceVLeoCacheError("source-V clean IQ collection is malformed")
    received_iq = np.empty_like(clean_iq)
    base_seed = int(c_args["seed"]) + SOURCE_LEO_SEED_OFFSET
    scene_seeds = {
        scene: base_seed + index * SOURCE_LEO_SCENE_SEED_STRIDE
        for index, scene in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
    }
    channel_args = argparse.Namespace(
        sat_fs_hz=float(c_args.get("sat_fs_hz", 25e6)),
        sat_fc_hz=float(c_args.get("sat_fc_hz", 2.462e9)),
    )
    channel_config_sha256: dict[str, str] = {}
    scene_array = np.asarray(scenes, dtype=str)
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        channel_config = dict(sat_channel_config_for_scenario(scene))
        channel_config.update({"fs_hz": channel_args.sat_fs_hz, "fc_hz": channel_args.sat_fc_hz})
        channel_config_sha256[scene] = _canonical_sha256(channel_config)
        positions = np.flatnonzero(scene_array == scene)
        if positions.size <= 0:
            raise CLICSourceVLeoCacheError("source-V scene assignment is empty")
        generator = make_torch_generator(device, scene_seeds[scene])
        for start in range(0, int(positions.size), batch_size):
            current = positions[start : start + batch_size]
            source = _numpy_float32_to_tensor(clean_iq[current], device=device)
            with torch.no_grad():
                observed_torch, metadata = apply_sat_channel_for_scenario(
                    source,
                    scene,
                    channel_args,
                    gen=generator,
                    return_meta=True,
                )
            if not isinstance(metadata, Mapping) or metadata.get("channel_model") != "leo_residual":
                raise CLICSourceVLeoCacheError("source-V channel metadata drifted")
            observed = _tensor_to_numpy_float32(observed_torch)
            if observed.shape != clean_iq[current].shape or not np.isfinite(observed).all():
                raise CLICSourceVLeoCacheError(
                    "source-V received IQ rows are malformed or non-finite"
                )
            received_iq[current] = observed

    input_hashes_after_materialization = _input_hashes(input_paths)
    if input_hashes_after_materialization != input_hashes_before:
        raise CLICSourceVLeoCacheError(
            "source-V cache input bytes changed during materialization"
        )
    payload = {
        "received_iq": received_iq,
        "tx_ids": np.asarray(rows["tx_ids"], dtype=str),
        "rx_ids": np.asarray(rows["rx_ids"], dtype=str),
        "day_ids": np.asarray(rows["day_ids"], dtype=str),
        "physical_sample_id": np.asarray(rows["physical_sample_ids"], dtype=str),
        "sat_scenarios": scene_array,
    }
    output_publication: _ImmutablePublication | None = None
    receipt_publication: _ImmutablePublication | None = None
    try:
        output_publication = _atomic_save_npz(output_path, payload)
        output_sha256 = output_publication.sha256
        if not isinstance(output_sha256, str):
            raise CLICSourceVLeoCacheError("source-V received-IQ publication lacks a pre-publish SHA seal")
        _assert_publication_current(
            output_publication,
            expected_sha256=output_sha256,
            label="source-V received-IQ cache",
        )
        _validate_materialized_cache(
            output_path, expected_row_count=FROZEN_SOURCE_V_ROW_COUNT
        )
        _assert_publication_current(
            output_publication,
            expected_sha256=output_sha256,
            label="source-V received-IQ cache after validation",
        )
        input_hashes_after_write = _input_hashes(input_paths)
        if input_hashes_after_write != input_hashes_before:
            raise CLICSourceVLeoCacheError(
                "source-V cache input bytes changed before receipt sealing"
            )
        _assert_publication_current(
            output_publication,
            expected_sha256=output_sha256,
            label="source-V received-IQ cache before receipt sealing",
        )
        receipt = {
            "schema": SOURCE_V_LEO_CACHE_SCHEMA,
            "method": "P1_CLIC",
            "role": SOURCE_V_ROLE,
            "source_v_only": True,
            "post_target_completion_audit_non_selection": True,
            "fold_index": fold_index,
            "training_run_id": EXPECTED_TRAINING_RUN_ID,
            "clean_evidence_run_id": EXPECTED_CLEAN_RUN_ID,
            "source_tx_ids": list(source_tx_ids),
            "source_rx_ids": sorted(set(str(value) for value in rows["rx_ids"])),
            "source_day_ids": list(FROZEN_SOURCE_V_DAY_IDS),
            "source_validation_row_count": FROZEN_SOURCE_V_ROW_COUNT,
            "source_validation_tx_rx_day_coverage": source_validation_tx_rx_day_coverage,
            "source_validation_indices_sha256": validation_index_sha,
            "source_validation_physical_order_sha256": c_clean[
                "validation_metadata_order_sha256"
            ],
            "physical_order_sha256": _canonical_sha256(
                [str(value) for value in rows["physical_sample_ids"]]
            ),
            "scenario_assignment_sha256": _canonical_sha256(
                list(zip(rows["physical_sample_ids"], scenes, strict=True))
            ),
            "formal_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
            "scene_seeds": scene_seeds,
            "channel_config_sha256": channel_config_sha256,
            **coverage,
            "minimum_scene_class_count": min(
                coverage["scenario_class_coverage"].values()
            ),
            "minimum_scene_rx_count": min(coverage["scenario_rx_coverage"].values()),
            "minimum_scene_day_count": min(coverage["scenario_day_coverage"].values()),
            "checkpoint_sha256_by_arm": {
                "C": input_hashes_before["c_checkpoint"],
                "G": input_hashes_before["g_checkpoint"],
            },
            "terminal_receipt_sha256_by_arm": {
                "C": input_hashes_before["c_terminal"],
                "G": input_hashes_before["g_terminal"],
            },
            "clean_npz_sha256_by_arm": {
                "C": input_hashes_before["c_clean"],
                "G": input_hashes_before["g_clean"],
            },
            "clean_validation_metadata_order_sha256_by_arm": {
                "C": c_clean["validation_metadata_order_sha256"],
                "G": g_clean["validation_metadata_order_sha256"],
            },
            "clean_validation_indices_sha256_by_arm": {
                "C": c_clean["validation_indices_sha256"],
                "G": g_clean["validation_indices_sha256"],
            },
            "wisig_pkl_sha256": dataset_sha,
            "received_iq_npz_path": str(output_path),
            "received_iq_npz_sha256": output_sha256,
            "same_received_iq_bytes_for_c_and_g": True,
            "single_leo_observation_per_physical_sample": True,
            "cross_scene_physical_sample_reuse": False,
            "clean_source_runtime_access": False,
            "target_access": False,
            "query_access": False,
            "fit_rows": 0,
            "threshold_fit_rows": 0,
            "proxy_forward_rows": 0,
            "source_l_forward_rows": 0,
            "source_v_forward_rows": 0,
            "selection_access": False,
            "retry_access": False,
        }
        _assert_publication_current(
            output_publication,
            expected_sha256=output_sha256,
            label="source-V received-IQ cache at receipt publish",
        )
        receipt_publication = _atomic_write_json(receipt_path, receipt)
        receipt_sha256 = receipt_publication.sha256
        if not isinstance(receipt_sha256, str):
            raise CLICSourceVLeoCacheError("source-V receipt publication lacks a pre-publish SHA seal")
        _assert_publication_current(
            receipt_publication,
            expected_sha256=receipt_sha256,
            label="source-V receipt",
        )
        _assert_publication_current(
            output_publication,
            expected_sha256=output_sha256,
            label="source-V received-IQ cache after receipt publish",
        )
        _assert_publication_current(
            receipt_publication,
            expected_sha256=receipt_sha256,
            label="source-V receipt before return",
        )
        _assert_publication_current(
            output_publication,
            expected_sha256=output_sha256,
            label="source-V received-IQ cache before return",
        )
    except Exception:
        for publication in (receipt_publication, output_publication):
            if publication is not None:
                _unlink_if_owned(publication)
        raise
    return {
        "out_npz": str(output_path),
        "receipt_json": str(receipt_path),
        "received_iq_npz_sha256": receipt["received_iq_npz_sha256"],
        "source_validation_row_count": receipt["source_validation_row_count"],
        "minimum_scene_class_count": receipt["minimum_scene_class_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--c-ckpt", required=True)
    parser.add_argument("--c-terminal-receipt-json", required=True)
    parser.add_argument("--c-clean-npz", required=True)
    parser.add_argument("--g-ckpt", required=True)
    parser.add_argument("--g-terminal-receipt-json", required=True)
    parser.add_argument("--g-clean-npz", required=True)
    parser.add_argument("--wisig-pkl", required=True)
    parser.add_argument("--expected-wisig-sha256", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--known-validation-tx-ids", required=True)
    parser.add_argument("--proxy-unknown-tx-ids", required=True)
    parser.add_argument("--cache-run-root", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--receipt-json", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = build_source_v_received_iq(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "CLICSourceVLeoCacheError",
    "EXPECTED_CACHE_RUN_ID",
    "FORMAL_LEO_WEAK_SCENARIOS",
    "FROZEN_SOURCE_V_ROW_COUNT",
    "SOURCE_V_LEO_CACHE_SCHEMA",
    "SOURCE_V_ROLE",
    "assign_source_v_scenarios",
    "build_source_v_received_iq",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
