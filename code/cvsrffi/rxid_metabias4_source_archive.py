"""Immutable D103-R1 source-pool partition and archive publication.

The builder is the only component allowed to see the complete source-pool
metadata.  It applies the project's deterministic grouped Meta-SSL partition
and publishes three structurally separated views:

* ``L_s`` keeps z_dom, pre_relu, receiver/day and TX labels;
* ``U_s`` omits both TX labels and pre_relu;
* source validation is stored under a scorer-only directory while the Phase1
  fit receives only a row-count/content seal.

No target, query, clean-IQ or received-IQ array is accepted or persisted.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import numpy as np

from .rxid_metabias4_phase1_trainer import CANDIDATE_ID, FEATURE_DIM, SEED


SCHEMA = "cvs.d103_r2.rxid_dualsplit.source_split.v1"
ARCHIVE_MANIFEST_SCHEMA = "cvs.d103_r2.rxid_dualsplit.source_feature_archive.v1"
SCORER_MANIFEST_SCHEMA = "cvs.d103_r2.rxid_dualsplit.source_val_scorer.v1"

POOL_MEMBERS = (
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
)
LABELED_MEMBERS = (
    "z_dom",
    "pre_relu",
    "receiver_ids",
    "day_ids",
    "tx_labels",
    "physical_ids",
)
UNLABELED_MEMBERS = (
    "z_dom",
    "receiver_ids",
    "day_ids",
    "physical_ids",
)
SCORER_MEMBERS = POOL_MEMBERS


class D103R1SourceArchiveError(ValueError):
    """Raised when the source split or archive closure drifts."""


def _canonical_bytes(value: Any) -> bytes:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(member) for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(member) for member in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return item

    return (
        json.dumps(
            convert(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
    normalized = str(value)
    if (
        len(normalized) != 64
        or normalized != normalized.lower()
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise D103R1SourceArchiveError(f"{name} must be lowercase SHA256")
    return normalized


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype == object:
        raise D103R1SourceArchiveError("object arrays are forbidden")
    if array.dtype.kind in {"U", "S"}:
        descriptor = {"dtype": "utf8-string", "shape": list(array.shape)}
        body = _canonical_bytes(array.astype(str).tolist())
    else:
        canonical = np.ascontiguousarray(array)
        descriptor = {"dtype": canonical.dtype.str, "shape": list(canonical.shape)}
        body = canonical.tobytes(order="C")
    return hashlib.sha256(
        _canonical_bytes(descriptor).rstrip(b"\n") + b"\0" + body
    ).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_pool(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if tuple(arrays) != POOL_MEMBERS:
        raise D103R1SourceArchiveError("source-pool member order/closure drift")
    copied = {
        name: np.array(value, copy=True, order="C") for name, value in arrays.items()
    }
    row_count = len(copied["physical_ids"])
    if row_count != 8400:
        raise D103R1SourceArchiveError(
            f"formal source pool must contain exactly 8400 rows, got {row_count}"
        )
    for name in ("z_id", "z_dom", "pre_relu"):
        value = copied[name]
        if (
            value.dtype != np.float32
            or value.shape != (row_count, FEATURE_DIM)
            or not np.isfinite(value).all()
        ):
            raise D103R1SourceArchiveError(f"{name} must be finite float32 [N,160]")
    if not np.array_equal(
        copied["z_id"], np.maximum(copied["pre_relu"], np.float32(0.0))
    ):
        raise D103R1SourceArchiveError("z_id/pre_relu exact ReLU binding failed")
    for name in (
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "observation_ids",
    ):
        value = copied[name]
        if (
            value.ndim != 1
            or len(value) != row_count
            or value.dtype.kind not in {"U", "S"}
            or any(not item for item in value.astype(str).tolist())
        ):
            raise D103R1SourceArchiveError(f"{name} must be nonempty text [N]")
        copied[name] = np.asarray(value.astype(str), dtype=np.str_)
    class_ids = copied["class_ids"]
    classes = class_ids.astype(str).tolist()
    if (
        class_ids.ndim != 1
        or len(classes) != 6
        or len(set(classes)) != 6
        or set(classes) != set(copied["labels"].astype(str).tolist())
    ):
        raise D103R1SourceArchiveError("source-pool class registry drift")
    copied["class_ids"] = np.asarray(classes, dtype=np.str_)
    physical = copied["physical_ids"].astype(str)
    observations = copied["observation_ids"].astype(str)
    if np.unique(physical).size != row_count or np.unique(observations).size != row_count:
        raise D103R1SourceArchiveError("source-pool physical/observation IDs must be unique")
    cells = set(
        zip(
            copied["labels"].astype(str).tolist(),
            copied["receiver_ids"].astype(str).tolist(),
            copied["day_ids"].astype(str).tolist(),
        )
    )
    if (
        len(set(copied["receiver_ids"].astype(str).tolist())) != 7
        or len(set(copied["day_ids"].astype(str).tolist())) != 4
        or len(cells) != 6 * 7 * 4
    ):
        raise D103R1SourceArchiveError(
            "source pool must cover every 6 TX x 7 receiver x 4 day cell"
        )
    return copied


def partition_source_pool(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Apply the fixed 0.07/0.63/0.30 grouped partition once."""

    pool = _validate_pool(arrays)
    labels = pool["labels"].astype(str)
    receivers = pool["receiver_ids"].astype(str)
    days = pool["day_ids"].astype(str)
    physical = pool["physical_ids"].astype(str)
    rng = np.random.default_rng(SEED)
    cells = sorted(set(zip(labels.tolist(), receivers.tolist(), days.tolist())))
    cell_rows: dict[tuple[str, str, str], list[int]] = {}
    for cell in cells:
        label, receiver, day = cell
        local = np.flatnonzero(
            (labels == label) & (receivers == receiver) & (days == day)
        ).astype(int)
        if local.size < 4:
            raise D103R1SourceArchiveError(
                f"source cell lacks L/U/source-val reserve capacity: "
                f"{label}/{receiver}/{day}"
            )
        ordered = sorted(local.tolist(), key=lambda index: physical[index])
        permutation = rng.permutation(len(ordered)).astype(int).tolist()
        cell_rows[cell] = [ordered[index] for index in permutation]

    labeled_counts = {cell: 2 for cell in cells}
    receiver_tx_groups = sorted(
        set((receiver, label) for label, receiver, _day in cells)
    )
    for receiver, label in receiver_tx_groups:
        local_cells = sorted(
            cell
            for cell in cells
            if cell[0] == label and cell[1] == receiver
        )
        if len(local_cells) != 4:
            raise D103R1SourceArchiveError(
                f"receiver/TX group must contain 4 days: {receiver}/{label}"
            )
        while sum(labeled_counts[cell] for cell in local_cells) < 14:
            eligible = [
                cell
                for cell in local_cells
                if labeled_counts[cell] < 4
                and labeled_counts[cell] + 2 < len(cell_rows[cell])
            ]
            if not eligible:
                raise D103R1SourceArchiveError(
                    f"receiver/TX group cannot allocate 14 labeled rows: "
                    f"{receiver}/{label}"
                )
            chosen = sorted(
                eligible,
                key=lambda cell: (
                    -(
                        0.07 * len(cell_rows[cell])
                        - labeled_counts[cell]
                    ),
                    cell,
                ),
            )[0]
            labeled_counts[chosen] += 1
        day_counts = [labeled_counts[cell] for cell in local_cells]
        if (
            sum(day_counts) != 14
            or min(day_counts) < 2
            or max(day_counts) > 4
            or any(14 - count < 10 for count in day_counts)
        ):
            raise D103R1SourceArchiveError(
                f"leave-one-day K10 allocation failed: {receiver}/{label}"
            )

    if sum(labeled_counts.values()) != 588:
        raise D103R1SourceArchiveError("global labeled count must equal 588")
    unlabeled_counts = {cell: 1 for cell in cells}
    while sum(unlabeled_counts.values()) < 5292:
        eligible = [
            cell
            for cell in cells
            if (
                labeled_counts[cell]
                + unlabeled_counts[cell]
                + 1
                < len(cell_rows[cell])
            )
        ]
        if not eligible:
            raise D103R1SourceArchiveError(
                "source pool cannot allocate exact U_s count while reserving source-val"
            )
        chosen = sorted(
            eligible,
            key=lambda cell: (
                -(0.63 * len(cell_rows[cell]) - unlabeled_counts[cell]),
                cell,
            ),
        )[0]
        unlabeled_counts[chosen] += 1

    labeled_rows: list[int] = []
    unlabeled_rows: list[int] = []
    source_val_rows: list[int] = []
    cell_receipts: list[dict[str, Any]] = []
    for cell in cells:
        label, receiver, day = cell
        ordered = cell_rows[cell]
        n_labeled = labeled_counts[cell]
        n_unlabeled = unlabeled_counts[cell]
        n_source_val = len(ordered) - n_labeled - n_unlabeled
        if n_labeled < 2 or n_unlabeled < 1 or n_source_val < 1:
            raise D103R1SourceArchiveError(
                f"split cell minimum failed: {label}/{receiver}/{day}"
            )
        labeled_rows.extend(ordered[:n_labeled])
        unlabeled_rows.extend(ordered[n_labeled : n_labeled + n_unlabeled])
        source_val_rows.extend(ordered[n_labeled + n_unlabeled :])
        cell_receipts.append(
            {
                "tx": label,
                "receiver": receiver,
                "day": day,
                "pool": len(ordered),
                "L_s": n_labeled,
                "U_s": n_unlabeled,
                "source_val": n_source_val,
            }
        )
    split_rows = {
        "L_s": np.asarray(sorted(labeled_rows), dtype=np.int64),
        "U_s": np.asarray(sorted(unlabeled_rows), dtype=np.int64),
        "source_val": np.asarray(sorted(source_val_rows), dtype=np.int64),
    }
    combined = np.concatenate(tuple(split_rows.values()))
    if (
        combined.size != len(physical)
        or np.unique(combined).size != len(physical)
        or int(combined.min()) != 0
        or int(combined.max()) != len(physical) - 1
    ):
        raise D103R1SourceArchiveError("source split overlap/coverage failed")
    if tuple(int(split_rows[name].size) for name in ("L_s", "U_s", "source_val")) != (
        588,
        5292,
        2520,
    ):
        raise D103R1SourceArchiveError("global 0.07/0.63/0.30 count closure failed")

    def take(names: tuple[str, ...], rows: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(pool[name][rows])
            for name in names
            if name != "class_ids"
        } | (
            {"class_ids": np.array(pool["class_ids"], copy=True)}
            if "class_ids" in names
            else {}
        )

    labeled = take(
        ("z_dom", "pre_relu", "receiver_ids", "day_ids", "labels", "physical_ids"),
        split_rows["L_s"],
    )
    labeled["tx_labels"] = labeled.pop("labels")
    labeled = {name: labeled[name] for name in LABELED_MEMBERS}
    unlabeled = take(UNLABELED_MEMBERS, split_rows["U_s"])
    scorer = take(SCORER_MEMBERS, split_rows["source_val"])
    counts = {name: int(rows.size) for name, rows in split_rows.items()}
    receipt = {
        "seed": SEED,
        "grouping": ["tx", "receiver", "day"],
        "partition_implementation": (
            "grouped_exact_global_largest_deficit_rx_tx_l14_day2to4_v1"
        ),
        "nominal_fractions": {"L_s": 0.07, "U_s": 0.63, "source_val": 0.30},
        "pool_row_count": len(physical),
        "counts": counts,
        "effective_fractions": {
            name: count / len(physical) for name, count in counts.items()
        },
        "cell_count": len(cell_receipts),
        "receiver_tx_group_count": len(receiver_tx_groups),
        "leave_one_day_k10_reachable": True,
        "cells": cell_receipts,
        "physical_id_roots": {
            name: hashlib.sha256(
                _canonical_bytes(physical[rows].tolist())
            ).hexdigest()
            for name, rows in split_rows.items()
        },
        "overlap_count": 0,
        "union_complete": True,
    }
    return labeled, unlabeled, scorer, receipt


def publish_source_split_archives(
    arrays: Mapping[str, np.ndarray],
    *,
    output_dir: str | Path,
    checkpoint_sha256: str,
    runtime_sha256: str,
    cache_set_sha256: str,
    selection_salt_receipt_sha256: str,
    upstream_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a new immutable split root from an authorized source pool."""

    checkpoint_sha = _require_sha256(checkpoint_sha256, "checkpoint")
    runtime_sha = _require_sha256(runtime_sha256, "runtime")
    cache_sha = _require_sha256(cache_set_sha256, "cache set")
    salt_sha = _require_sha256(
        selection_salt_receipt_sha256, "selection-salt receipt"
    )
    labeled, unlabeled, scorer, partition = partition_source_pool(arrays)
    root = Path(output_dir).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing to overwrite source split root: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        role_arrays = {"L_s": labeled, "U_s": unlabeled}
        role_specs = {
            "L_s": (0.07, "visible"),
            "U_s": (0.63, "hidden"),
        }
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
                "candidate_id": CANDIDATE_ID,
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
        scorer_array_roots = {
            name: _array_sha256(value) for name, value in scorer.items()
        }
        content_sha = hashlib.sha256(
            _canonical_bytes(scorer_array_roots)
        ).hexdigest()
        seal = {"row_count": len(scorer["physical_ids"]), "content_sha256": content_sha}
        seal_path = staging / "source_val.seal.json"
        _write_new(seal_path, _canonical_bytes(seal))
        fit_manifest = {
            "schema": ARCHIVE_MANIFEST_SCHEMA,
            "candidate_id": CANDIDATE_ID,
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
            "role": "source_val_scorer_only",
            "fraction": 0.30,
            "archive": {
                "path": str(Path("scorer_only") / "source_val" / "features.npz"),
                "sha256": _sha256_file(scorer_archive),
            },
            "exact_member_allowlist": list(SCORER_MEMBERS),
            "array_sha256": scorer_array_roots,
            "content_sha256": content_sha,
            "row_count": len(scorer["physical_ids"]),
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
            "schema": SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "status": "FORMAL_PHASE1_SOURCE_SPLIT_COMPLETE",
            "artifact_stage": "phase1_offline_before_target_access",
            "protocol_schema": "p2_min_v1",
            "target25_authorized": False,
            "target_access": False,
            "formal_query_access": False,
            "clean_iq_persisted": False,
            "received_iq_persisted": False,
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
        root_manifest_path = staging / "source_split_manifest.json"
        _write_new(root_manifest_path, _canonical_bytes(root_manifest))
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
    }


__all__ = [
    "ARCHIVE_MANIFEST_SCHEMA",
    "D103R1SourceArchiveError",
    "LABELED_MEMBERS",
    "POOL_MEMBERS",
    "SCHEMA",
    "SCORER_MANIFEST_SCHEMA",
    "SCORER_MEMBERS",
    "UNLABELED_MEMBERS",
    "partition_source_pool",
    "publish_source_split_archives",
]
