"""Deterministic D104 source-held split with historical-query exclusion.

The metadata-only row selector is intentionally separate from feature archive
publication.  It receives no target data and derives the historical exclusion
set from the frozen D103-R2 development packaging rule.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from .rxid_metabias4_held_execution import (
    build_receiver_package_indices,
    canonical_sha256,
)
from .rxid_metabias4_phase1_trainer import (
    CANDIDATE_ID as D103_PHASE1_CANDIDATE_ID,
)
from .rxid_metabias4_source_archive import partition_source_pool
from .rxid_metabias4_source_archive import (
    ARCHIVE_MANIFEST_SCHEMA,
    LABELED_MEMBERS,
    SCORER_MANIFEST_SCHEMA,
    SCORER_MEMBERS,
    UNLABELED_MEMBERS,
    _array_sha256,
    _canonical_bytes,
    _require_sha256,
    _sha256_file,
    _write_new,
)


CANDIDATE_ID = "D104-R1-ANGQ-RXID-MB4"
SPLIT_ID = "d104_source_seed104713_v2"
SPLIT_SALT = "D104-R1-ANGQ-RXID-MB4|source-split|104713|v1"
HISTORICAL_QUERY_COUNT = 2478
HISTORICAL_QUERY_CANONICAL_ROOT_SHA256 = (
    "7870604d8ddba8268ba127065d4eaf1142931660d95411c9633c2ffa59d6b558"
)
HELD_PER_CELL = 15
EXPECTED_COUNTS = {"L_s": 588, "U_s": 5292, "source_val": 2520}
EXCLUSION_MANIFEST_FILE_SHA256 = (
    "3fd07b7afcb53b12a08df1643efae80c52917c893cc7453104e68932dc1f5b26"
)
EXCLUSION_MANIFEST_CONTENT_ROOT_SHA256 = (
    "89c91bc8bc11d74e6b12bd2df2c2eeac53ca75d8f2d3a983a8e823da52765b27"
)
EXCLUSION_BUILDER_SHA256 = (
    "dd9a13e908bdb4f607c1c257e5d8a36f5530219d7157dfa87343629b391a9b7d"
)


class D104SourceSplitError(ValueError):
    """Raised when the D104 split or its exclusion commitment drifts."""


def load_d104_exclusion_manifest(path: str | Path) -> dict[str, Any]:
    """Load and fully validate the frozen r3 historical exclusion manifest."""

    manifest_path = Path(path).resolve(strict=True)
    if _sha256_file(manifest_path) != EXCLUSION_MANIFEST_FILE_SHA256:
        raise D104SourceSplitError("D104 exclusion manifest file SHA drift")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise D104SourceSplitError("D104 exclusion manifest must be an object")
    content_root = value.get("manifest_content_root_sha256")
    body = dict(value)
    body.pop("manifest_content_root_sha256", None)
    query_ids = value.get("query_physical_ids_sorted")
    support_ids = value.get("support_physical_ids_sorted")
    code = value.get("derivation_code")
    packages = value.get("packages")
    if (
        value.get("schema")
        != "cvs.d104_r1.historical_query_exclusion_manifest.v2"
        or value.get("candidate_id") != CANDIDATE_ID
        or value.get("split_id") != SPLIT_ID
        or value.get("status") != "ACTIVE_REPRODUCIBLE_EXCLUSION_CONTROL"
        or content_root != EXCLUSION_MANIFEST_CONTENT_ROOT_SHA256
        or canonical_sha256(body) != content_root
        or value.get("active_query_physical_id_root_sha256")
        != HISTORICAL_QUERY_CANONICAL_ROOT_SHA256
        or value.get("query_physical_id_count") != HISTORICAL_QUERY_COUNT
        or not isinstance(query_ids, list)
        or len(query_ids) != HISTORICAL_QUERY_COUNT
        or query_ids != sorted(query_ids)
        or len(set(query_ids)) != HISTORICAL_QUERY_COUNT
        or canonical_sha256(query_ids) != HISTORICAL_QUERY_CANONICAL_ROOT_SHA256
        or value.get("support_physical_id_count") != 42
        or not isinstance(support_ids, list)
        or len(support_ids) != 42
        or support_ids != sorted(support_ids)
        or len(set(support_ids)) != 42
        or set(query_ids).intersection(support_ids)
        or value.get("support_query_intersection_count") != 0
        or not isinstance(code, dict)
        or code.get("builder_script", {}).get("sha256")
        != EXCLUSION_BUILDER_SHA256
        or value.get("source_val_labels_used_for_package_reconstruction") is not True
        or value.get("query_truth_passed_to_predictor") is not False
        or value.get("query_truth_used_for_scoring") is not False
        or value.get("performance_computed") is not False
        or value.get("target_access") is not False
        or not isinstance(packages, list)
        or len(packages) != 7
        or any(row.get("K") != 1 for row in packages)
    ):
        raise D104SourceSplitError("D104 exclusion manifest semantic closure drift")
    return value


def _rank(prefix: str, receiver: str, tx: str, day: str, physical_id: str) -> str:
    value = (
        f"{SPLIT_SALT}|{prefix}|{receiver}|{tx}|{day}|{physical_id}"
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _metadata(
    labels: Sequence[str] | np.ndarray,
    receiver_ids: Sequence[str] | np.ndarray,
    day_ids: Sequence[str] | np.ndarray,
    physical_ids: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(
        np.asarray(member).astype(str)
        for member in (labels, receiver_ids, day_ids, physical_ids)
    )
    sizes = {len(member) for member in values}
    if sizes != {8400} or any(member.ndim != 1 for member in values):
        raise D104SourceSplitError("D104 source metadata must be four [8400] arrays")
    labels_array, receiver_array, day_array, physical_array = values
    if (
        np.unique(labels_array).size != 6
        or np.unique(receiver_array).size != 7
        or np.unique(day_array).size != 4
        or np.unique(physical_array).size != 8400
        or any(not value for value in physical_array.tolist())
    ):
        raise D104SourceSplitError("D104 source metadata registry/identity drift")
    cells = set(
        zip(
            labels_array.tolist(),
            receiver_array.tolist(),
            day_array.tolist(),
        )
    )
    if len(cells) != 168:
        raise D104SourceSplitError("D104 source pool must cover 168 TX/receiver/day cells")
    return labels_array, receiver_array, day_array, physical_array


def reconstruct_historical_query_ids(
    source_pool: Mapping[str, np.ndarray],
) -> tuple[str, ...]:
    """Rebuild the exposed D103-R2 K1 query union without reading truth scores."""

    _labeled, _unlabeled, scorer, receipt = partition_source_pool(source_pool)
    if receipt.get("counts") != {"L_s": 588, "U_s": 5292, "source_val": 2520}:
        raise D104SourceSplitError("legacy source split count drift")
    classes = tuple(sorted(set(scorer["labels"].astype(str).tolist())))
    receivers = tuple(sorted(set(scorer["receiver_ids"].astype(str).tolist())))
    query_ids: set[str] = set()
    support_ids: set[str] = set()
    for receiver in receivers:
        support, query = build_receiver_package_indices(
            scorer["receiver_ids"],
            scorer["labels"],
            scorer["physical_ids"],
            held_receiver=receiver,
            registered_classes=classes,
            k_shot=1,
        )
        support_ids.update(scorer["physical_ids"][support].astype(str).tolist())
        query_ids.update(scorer["physical_ids"][query].astype(str).tolist())
    ordered = tuple(sorted(query_ids))
    if (
        len(ordered) != HISTORICAL_QUERY_COUNT
        or len(support_ids) != 42
        or query_ids.intersection(support_ids)
    ):
        raise D104SourceSplitError("historical D103 query/support reconstruction drift")
    root = canonical_sha256(list(ordered))
    if root != HISTORICAL_QUERY_CANONICAL_ROOT_SHA256:
        raise D104SourceSplitError("historical query canonical root drift")
    return ordered


def partition_d104_source_rows(
    labels: Sequence[str] | np.ndarray,
    receiver_ids: Sequence[str] | np.ndarray,
    day_ids: Sequence[str] | np.ndarray,
    physical_ids: Sequence[str] | np.ndarray,
    *,
    historical_query_ids: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select exact D104 L/U/source-val row indices from source metadata."""

    labels_array, receiver_array, day_array, physical_array = _metadata(
        labels, receiver_ids, day_ids, physical_ids
    )
    historical = tuple(sorted(str(value) for value in historical_query_ids))
    if (
        len(historical) != HISTORICAL_QUERY_COUNT
        or len(set(historical)) != HISTORICAL_QUERY_COUNT
        or canonical_sha256(list(historical))
        != HISTORICAL_QUERY_CANONICAL_ROOT_SHA256
        or not set(historical).issubset(set(physical_array.tolist()))
    ):
        raise D104SourceSplitError("historical query exclusion commitment drift")

    historical_set = set(historical)
    cells = sorted(
        set(
            zip(
                receiver_array.tolist(),
                labels_array.tolist(),
                day_array.tolist(),
            )
        )
    )
    candidate_by_cell: dict[tuple[str, str, str], list[int]] = {}
    held_by_cell: dict[tuple[str, str, str], list[int]] = {}
    remaining_by_cell: dict[tuple[str, str, str], list[int]] = {}
    for receiver, tx, day in cells:
        local = np.flatnonzero(
            (receiver_array == receiver)
            & (labels_array == tx)
            & (day_array == day)
        ).astype(int)
        candidates = [
            row for row in local.tolist() if physical_array[row] not in historical_set
        ]
        ranked_held = sorted(
            candidates,
            key=lambda row: (
                _rank("held", receiver, tx, day, physical_array[row]),
                physical_array[row],
            ),
        )
        if len(ranked_held) < HELD_PER_CELL + 4:
            raise D104SourceSplitError(
                f"D104 cell lacks held/train capacity: {receiver}/{tx}/{day}"
            )
        held = ranked_held[:HELD_PER_CELL]
        held_set = set(held)
        remaining = [row for row in local.tolist() if row not in held_set]
        candidate_by_cell[(receiver, tx, day)] = candidates
        held_by_cell[(receiver, tx, day)] = held
        remaining_by_cell[(receiver, tx, day)] = remaining

    held_rows = sorted(row for rows in held_by_cell.values() for row in rows)
    if len(held_rows) != EXPECTED_COUNTS["source_val"]:
        raise D104SourceSplitError("D104 held count must be 2520")

    labeled_counts = {cell: 2 for cell in cells}
    groups = sorted(set((receiver, tx) for receiver, tx, _day in cells))
    for receiver, tx in groups:
        local_cells = sorted(
            cell for cell in cells if cell[0] == receiver and cell[1] == tx
        )
        if len(local_cells) != 4:
            raise D104SourceSplitError("D104 receiver/TX day closure drift")
        while sum(labeled_counts[cell] for cell in local_cells) < 14:
            eligible = [
                cell for cell in local_cells if labeled_counts[cell] < 4
            ]
            if not eligible:
                raise D104SourceSplitError("D104 L allocation exhausted day cap")
            chosen = sorted(
                eligible,
                key=lambda cell: (
                    -(
                        0.07 * len(remaining_by_cell[cell])
                        - labeled_counts[cell]
                    ),
                    cell[2],
                ),
            )[0]
            labeled_counts[chosen] += 1
        counts = [labeled_counts[cell] for cell in local_cells]
        leave_day = [14 - value for value in counts]
        if (
            sum(counts) != 14
            or min(counts) < 2
            or max(counts) > 4
            or min(leave_day) < 10
            or max(leave_day) > 12
        ):
            raise D104SourceSplitError("D104 L leave-day closure failed")

    labeled_rows: list[int] = []
    cell_receipts: list[dict[str, Any]] = []
    for receiver, tx, day in cells:
        cell = (receiver, tx, day)
        remaining = remaining_by_cell[cell]
        ranked_l = sorted(
            remaining,
            key=lambda row: (
                _rank("L", receiver, tx, day, physical_array[row]),
                physical_array[row],
            ),
        )
        selected = ranked_l[: labeled_counts[cell]]
        labeled_rows.extend(selected)
        cell_receipts.append(
            {
                "receiver": receiver,
                "tx": tx,
                "day": day,
                "pool": int(np.sum(
                    (receiver_array == receiver)
                    & (labels_array == tx)
                    & (day_array == day)
                )),
                "historical_query_excluded": int(
                    np.sum(
                        (receiver_array == receiver)
                        & (labels_array == tx)
                        & (day_array == day)
                    )
                )
                - len(candidate_by_cell[cell]),
                "eligible_after_historical_exclusion": len(candidate_by_cell[cell]),
                "source_val": len(held_by_cell[cell]),
                "train_after_held": len(remaining),
                "L_s": labeled_counts[cell],
                "U_s": len(remaining) - labeled_counts[cell],
            }
        )

    labeled_rows = sorted(labeled_rows)
    labeled_set = set(labeled_rows)
    held_set = set(held_rows)
    unlabeled_rows = sorted(
        set(range(8400)) - held_set - labeled_set
    )
    if (
        len(labeled_rows) != EXPECTED_COUNTS["L_s"]
        or len(unlabeled_rows) != EXPECTED_COUNTS["U_s"]
        or len(held_rows) != EXPECTED_COUNTS["source_val"]
    ):
        raise D104SourceSplitError("D104 split exact count closure failed")
    historical_rows = {
        int(row)
        for row in np.flatnonzero(
            np.isin(physical_array, np.asarray(historical, dtype=str))
        ).tolist()
    }
    combined = held_set | labeled_set | set(unlabeled_rows)
    if (
        len(combined) != 8400
        or historical_rows & held_set
        or held_set & labeled_set
        or held_set & set(unlabeled_rows)
        or labeled_set & set(unlabeled_rows)
    ):
        raise D104SourceSplitError("D104 split overlap/union closure failed")

    split_rows = {
        "L_s": np.asarray(labeled_rows, dtype=np.int64),
        "U_s": np.asarray(unlabeled_rows, dtype=np.int64),
        "source_val": np.asarray(held_rows, dtype=np.int64),
    }
    roots = {
        name: canonical_sha256(
            physical_array[rows].astype(str).tolist()
        )
        for name, rows in split_rows.items()
    }
    receipt = {
        "schema": "cvs.d104_r1.source_split.rows.v1",
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "salt": SPLIT_SALT,
        "historical_query_count": HISTORICAL_QUERY_COUNT,
        "historical_query_canonical_root_sha256": (
            HISTORICAL_QUERY_CANONICAL_ROOT_SHA256
        ),
        "historical_query_derivation": (
            "union_of_d103_r2_source_val_k1_queries_all_7_receivers"
        ),
        "counts": {name: int(len(rows)) for name, rows in split_rows.items()},
        "physical_id_roots": roots,
        "cell_count": len(cell_receipts),
        "receiver_tx_group_count": len(groups),
        "held_per_cell": HELD_PER_CELL,
        "labeled_per_receiver_tx": 14,
        "four_day_labeled_range": [2, 4],
        "leave_day_labeled_range": [10, 12],
        "overlap_count": 0,
        "union_complete": True,
        "historical_queries_excluded_from_new_source_val_only": True,
        "source_labels_used_for_stratified_split": True,
        "query_truth_used_for_method_selection": False,
        "query_truth_used_for_performance_selection": False,
        "source_val_performance_computed": False,
        "target_access": False,
        "cells": cell_receipts,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return split_rows, receipt


def partition_d104_source_pool(
    source_pool: Mapping[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Return D104 feature views after deriving the frozen exclusion list."""

    historical = reconstruct_historical_query_ids(source_pool)
    rows, receipt = partition_d104_source_rows(
        source_pool["labels"],
        source_pool["receiver_ids"],
        source_pool["day_ids"],
        source_pool["physical_ids"],
        historical_query_ids=historical,
    )

    def take(names: tuple[str, ...], selected: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(np.asarray(source_pool[name])[selected])
            for name in names
            if name != "class_ids"
        } | (
            {"class_ids": np.array(source_pool["class_ids"], copy=True)}
            if "class_ids" in names
            else {}
        )

    labeled = take(
        ("z_dom", "pre_relu", "receiver_ids", "day_ids", "labels", "physical_ids"),
        rows["L_s"],
    )
    labeled["tx_labels"] = labeled.pop("labels")
    labeled = {
        name: labeled[name]
        for name in (
            "z_dom",
            "pre_relu",
            "receiver_ids",
            "day_ids",
            "tx_labels",
            "physical_ids",
        )
    }
    unlabeled = take(
        ("z_dom", "receiver_ids", "day_ids", "physical_ids"),
        rows["U_s"],
    )
    scorer = take(
        (
            "z_id",
            "z_dom",
            "pre_relu",
            "labels",
            "receiver_ids",
            "day_ids",
            "physical_ids",
            "scenario_names",
            "observation_ids",
            "class_ids",
        ),
        rows["source_val"],
    )
    return labeled, unlabeled, scorer, receipt


def publish_d104_source_split_archives(
    source_pool: Mapping[str, np.ndarray],
    *,
    exclusion_manifest_path: str | Path,
    output_dir: str | Path,
    checkpoint_sha256: str,
    runtime_sha256: str,
    cache_set_sha256: str,
    selection_salt_receipt_sha256: str,
    upstream_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish immutable D104 L/U/scorer views and a split-root manifest."""

    exclusion = load_d104_exclusion_manifest(exclusion_manifest_path)
    historical = reconstruct_historical_query_ids(source_pool)
    if list(historical) != exclusion["query_physical_ids_sorted"]:
        raise D104SourceSplitError("D104 reconstructed exclusion IDs drift from manifest")
    checkpoint_sha = _require_sha256(checkpoint_sha256, "checkpoint")
    runtime_sha = _require_sha256(runtime_sha256, "runtime")
    cache_sha = _require_sha256(cache_set_sha256, "cache set")
    salt_sha = _require_sha256(
        selection_salt_receipt_sha256, "selection-salt receipt"
    )
    labeled, unlabeled, scorer, partition = partition_d104_source_pool(source_pool)
    root = Path(output_dir).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing to overwrite D104 source split root: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        role_arrays = {"L_s": labeled, "U_s": unlabeled}
        role_specs = {"L_s": (0.07, "visible"), "U_s": (0.63, "hidden")}
        role_artifacts: dict[str, Any] = {}
        for role in ("L_s", "U_s"):
            role_dir = staging / role
            role_dir.mkdir()
            archive_path = role_dir / "features.npz"
            with archive_path.open("xb") as stream:
                np.savez(stream, **role_arrays[role])
                stream.flush()
                os.fsync(stream.fileno())
            fraction, tx_visibility = role_specs[role]
            manifest = {
                "schema": ARCHIVE_MANIFEST_SCHEMA,
                "candidate_id": D103_PHASE1_CANDIDATE_ID,
                "role": role,
                "fraction": fraction,
                "tx_visibility": tx_visibility,
                "archive_sha256": _sha256_file(archive_path),
                "target_access": False,
                "formal_query_access": False,
                "source_validation_gradient_access": False,
                "physical_id_unique": True,
                "checkpoint_sha256": checkpoint_sha,
                "runtime_sha256": runtime_sha,
            }
            manifest_path = role_dir / "manifest.json"
            _write_new(manifest_path, _canonical_bytes(manifest))
            role_artifacts[role] = {
                "archive": str(Path(role) / archive_path.name),
                "archive_sha256": manifest["archive_sha256"],
                "manifest": str(Path(role) / manifest_path.name),
                "manifest_sha256": _sha256_file(manifest_path),
                "row_count": len(role_arrays[role]["physical_ids"]),
            }

        scorer_dir = staging / "scorer_only" / "source_val"
        scorer_dir.mkdir(parents=True)
        scorer_archive = scorer_dir / "features.npz"
        with scorer_archive.open("xb") as stream:
            np.savez(stream, **scorer)
            stream.flush()
            os.fsync(stream.fileno())
        scorer_roots = {name: _array_sha256(value) for name, value in scorer.items()}
        content_sha = hashlib.sha256(_canonical_bytes(scorer_roots)).hexdigest()
        seal = {"row_count": 2520, "content_sha256": content_sha}
        seal_path = staging / "source_val.seal.json"
        _write_new(seal_path, _canonical_bytes(seal))
        fit_manifest = {
            "schema": ARCHIVE_MANIFEST_SCHEMA,
            "candidate_id": D103_PHASE1_CANDIDATE_ID,
            "role": "source_val",
            "fraction": 0.30,
            "tx_visibility": "scorer_only",
            "archive_sha256": None,
            "target_access": False,
            "formal_query_access": False,
            "source_validation_gradient_access": False,
            "physical_id_unique": True,
            "checkpoint_sha256": checkpoint_sha,
            "runtime_sha256": runtime_sha,
        }
        fit_manifest_path = staging / "source_val.manifest.json"
        _write_new(fit_manifest_path, _canonical_bytes(fit_manifest))
        scorer_manifest = {
            "schema": SCORER_MANIFEST_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "split_id": SPLIT_ID,
            "role": "source_val_scorer_only",
            "fraction": 0.30,
            "archive": {
                "path": str(Path("scorer_only") / "source_val" / "features.npz"),
                "sha256": _sha256_file(scorer_archive),
            },
            "exact_member_allowlist": list(SCORER_MEMBERS),
            "array_sha256": scorer_roots,
            "content_sha256": content_sha,
            "row_count": 2520,
            "asset_access": False,
            "gradient_access": False,
            "selection_access": False,
            "target_access": False,
            "formal_query_access": False,
            "checkpoint_sha256": checkpoint_sha,
            "runtime_sha256": runtime_sha,
        }
        scorer_manifest_path = scorer_dir / "manifest.json"
        _write_new(scorer_manifest_path, _canonical_bytes(scorer_manifest))
        root_manifest = {
            "schema": "cvs.d104_r1.source_split.archive.v2",
            "candidate_id": CANDIDATE_ID,
            "split_id": SPLIT_ID,
            "status": "FORMAL_PHASE1_SOURCE_SPLIT_COMPLETE",
            "artifact_stage": "phase1_offline_before_new_source_held_truth_open",
            "protocol_schema": "p2_min_v1",
            "target25_authorized": False,
            "target_access": False,
            "formal_query_access": False,
            "historical_exclusion_manifest": {
                "sha256": EXCLUSION_MANIFEST_FILE_SHA256,
                "content_root_sha256": EXCLUSION_MANIFEST_CONTENT_ROOT_SHA256,
                "query_count": HISTORICAL_QUERY_COUNT,
            },
            "phase1_training_candidate_id": D103_PHASE1_CANDIDATE_ID,
            "phase1_training_manifests_use_inherited_d103_schema": True,
            "inputs": {
                "checkpoint_sha256": checkpoint_sha,
                "runtime_sha256": runtime_sha,
                "source_train_cache_set_sha256": cache_sha,
                "selection_salt_receipt_sha256": salt_sha,
            },
            "upstream_audit": dict(upstream_audit or {}),
            "partition": partition,
            "roles": role_artifacts,
            "source_val": {
                "fit_manifest": fit_manifest_path.name,
                "fit_manifest_sha256": _sha256_file(fit_manifest_path),
                "seal": seal_path.name,
                "seal_sha256": _sha256_file(seal_path),
                "scorer_archive": scorer_manifest["archive"],
                "scorer_manifest": str(
                    Path("scorer_only") / "source_val" / "manifest.json"
                ),
                "scorer_manifest_sha256": _sha256_file(scorer_manifest_path),
            },
        }
        root_path = staging / "source_split_manifest.json"
        _write_new(root_path, _canonical_bytes(root_manifest))
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": root_manifest["status"],
        "output_dir": str(root),
        "manifest": str(root / "source_split_manifest.json"),
        "manifest_sha256": _sha256_file(root / "source_split_manifest.json"),
        "counts": partition["counts"],
        "split_receipt_sha256": partition["receipt_sha256"],
    }


__all__ = [
    "CANDIDATE_ID",
    "D104SourceSplitError",
    "EXPECTED_COUNTS",
    "HISTORICAL_QUERY_CANONICAL_ROOT_SHA256",
    "HISTORICAL_QUERY_COUNT",
    "SPLIT_ID",
    "SPLIT_SALT",
    "partition_d104_source_pool",
    "partition_d104_source_rows",
    "load_d104_exclusion_manifest",
    "publish_d104_source_split_archives",
    "reconstruct_historical_query_ids",
]
