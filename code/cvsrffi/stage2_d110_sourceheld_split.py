"""Build the unopened D110 Phase1 source-held split.

The selector is intentionally independent of the frozen D104 split builder.  It
uses source metadata only for stratification, excludes every physical ID from
the three committed historical surfaces, and publishes no performance result.
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

from .rxid_metabias4_held_execution import canonical_sha256
from .rxid_metabias4_phase1_trainer import FEATURE_DIM
from .rxid_metabias4_source_archive import (
    SCORER_MEMBERS,
    _array_sha256,
    _canonical_bytes,
    _sha256_file,
    _write_new,
)


CANDIDATE_ID = "D110-SCPM-USQKNN"
SPLIT_ID = "d110_source_seed110813_v1"
SPLIT_SALT = "D110-SCPM-USQKNN|source-held|110813|v1"
POOL_ROWS = 8400
HISTORICAL_QUERY_ROWS = 2478
D104_HELD_ROWS = 2520
D110_LS_ROWS = 588
HELD_PER_CELL = 7
CELL_COUNT = 168
GROUP_COUNT = 42
EXPECTED_HELD_ROWS = CELL_COUNT * HELD_PER_CELL
SCORER_SCHEMA = "cvs.d110.scpm_usqknn.source_val_scorer.v1"
SELECTION_SCHEMA = "cvs.d110.scpm_usqknn.sourceheld_selection.v1"
DUAL_MEMBERS = (
    "z_id",
    "z_dom",
    "tx_logits",
    "labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "class_ids",
    "observation_ids",
)
STRICT_TAP_MEMBERS = (
    "pre_relu",
    "z_dom",
    "labels",
    "receiver_ids",
    "physical_ids",
)


class D110SourceHeldSplitError(ValueError):
    """Raised when the D110 source-held data closure drifts."""


def _string_rows(value: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or len(array) != rows
        or array.dtype.kind not in {"U", "S"}
    ):
        raise D110SourceHeldSplitError(f"{name} must be a string [{rows}] array")
    result = array.astype(str)
    if any(not item for item in result.tolist()):
        raise D110SourceHeldSplitError(f"{name} contains an empty value")
    return result


def validate_source_feature_pool(
    dual_archive: Mapping[str, np.ndarray],
    strict_tap: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bind the D103 full archive to the same-row D105 strict pre-ReLU tap."""

    if tuple(dual_archive) != DUAL_MEMBERS:
        raise D110SourceHeldSplitError("D103 dual archive member closure drift")
    if tuple(strict_tap) != STRICT_TAP_MEMBERS:
        raise D110SourceHeldSplitError("D105 strict tap member closure drift")

    rows = len(np.asarray(dual_archive["physical_ids"]))
    if rows != POOL_ROWS:
        raise D110SourceHeldSplitError("D110 source pool must contain 8400 rows")
    dual_text = {
        name: _string_rows(np.asarray(dual_archive[name]), name=name, rows=rows)
        for name in (
            "labels",
            "receiver_ids",
            "day_ids",
            "physical_ids",
            "scenario_names",
            "observation_ids",
        )
    }
    strict_text = {
        name: _string_rows(np.asarray(strict_tap[name]), name=f"strict_{name}", rows=rows)
        for name in ("labels", "receiver_ids", "physical_ids")
    }
    class_ids = np.asarray(dual_archive["class_ids"])
    if class_ids.ndim != 1 or class_ids.dtype.kind not in {"U", "S"}:
        raise D110SourceHeldSplitError("class_ids must be a string registry")
    class_ids = class_ids.astype(str)
    if (
        len(set(dual_text["physical_ids"].tolist())) != rows
        or np.unique(dual_text["labels"]).size != 6
        or np.unique(dual_text["receiver_ids"]).size != 7
        or np.unique(dual_text["day_ids"]).size != 4
        or set(dual_text["labels"].tolist()) != set(class_ids.tolist())
    ):
        raise D110SourceHeldSplitError("D110 source registry/identity drift")
    cells = set(
        zip(
            dual_text["receiver_ids"].tolist(),
            dual_text["labels"].tolist(),
            dual_text["day_ids"].tolist(),
        )
    )
    if len(cells) != CELL_COUNT:
        raise D110SourceHeldSplitError("D110 source pool must cover 168 cells")
    for name in ("labels", "receiver_ids", "physical_ids"):
        if not np.array_equal(dual_text[name], strict_text[name]):
            raise D110SourceHeldSplitError(f"D103/D105 row binding drift: {name}")

    dual_zid = np.asarray(dual_archive["z_id"])
    dual_zdom = np.asarray(dual_archive["z_dom"])
    dual_logits = np.asarray(dual_archive["tx_logits"])
    strict_pre = np.asarray(strict_tap["pre_relu"])
    strict_zdom = np.asarray(strict_tap["z_dom"])
    for name, value in (
        ("z_id", dual_zid),
        ("z_dom", dual_zdom),
        ("pre_relu", strict_pre),
        ("strict_z_dom", strict_zdom),
    ):
        if (
            value.dtype != np.float32
            or value.shape != (rows, FEATURE_DIM)
            or not np.isfinite(value).all()
        ):
            raise D110SourceHeldSplitError(
                f"{name} must be finite float32 [8400,{FEATURE_DIM}]"
            )
    if (
        dual_logits.dtype != np.float32
        or dual_logits.shape != (rows, 6)
        or not np.isfinite(dual_logits).all()
    ):
        raise D110SourceHeldSplitError("tx_logits must be finite float32 [8400,6]")
    if not np.array_equal(np.maximum(strict_pre, np.float32(0.0)), dual_zid):
        raise D110SourceHeldSplitError("D105 pre_relu/D103 z_id exact parity failed")
    if not np.array_equal(strict_zdom, dual_zdom):
        raise D110SourceHeldSplitError("D105/D103 z_dom exact parity failed")

    pool = {
        "z_id": np.ascontiguousarray(dual_zid),
        "z_dom": np.ascontiguousarray(dual_zdom),
        "pre_relu": np.ascontiguousarray(strict_pre),
        "labels": dual_text["labels"],
        "receiver_ids": dual_text["receiver_ids"],
        "day_ids": dual_text["day_ids"],
        "physical_ids": dual_text["physical_ids"],
        "scenario_names": dual_text["scenario_names"],
        "observation_ids": dual_text["observation_ids"],
        "class_ids": class_ids,
    }
    pool_arrays = {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "sha256": _array_sha256(np.asarray(value)),
        }
        for name, value in pool.items()
    }
    return pool, {
        "row_count": rows,
        "dual_array_receipts": {
            name: {
                "shape": list(np.asarray(dual_archive[name]).shape),
                "dtype": str(np.asarray(dual_archive[name]).dtype),
                "sha256": _array_sha256(np.asarray(dual_archive[name])),
            }
            for name in DUAL_MEMBERS
        },
        "strict_tap_array_receipts": {
            name: {
                "shape": list(np.asarray(strict_tap[name]).shape),
                "dtype": str(np.asarray(strict_tap[name]).dtype),
                "sha256": _array_sha256(np.asarray(strict_tap[name])),
            }
            for name in STRICT_TAP_MEMBERS
        },
        "source_pool_array_receipts": pool_arrays,
        "physical_ids_same_order": True,
        "labels_same_order": True,
        "receiver_ids_same_order": True,
        "relu_pre_relu_equals_z_id_exact": True,
        "z_dom_equal_exact": True,
    }


def _normalize_exclusion(
    values: Sequence[str], *, name: str, expected_rows: int, pool_ids: set[str]
) -> tuple[str, ...]:
    ordered = tuple(sorted(str(value) for value in values))
    if (
        len(ordered) != expected_rows
        or len(set(ordered)) != expected_rows
        or any(not value for value in ordered)
        or not set(ordered).issubset(pool_ids)
    ):
        raise D110SourceHeldSplitError(f"{name} ID commitment drift")
    return ordered


def _rank(receiver: str, tx: str, day: str, physical_id: str) -> str:
    payload = f"{SPLIT_SALT}|held|{receiver}|{tx}|{day}|{physical_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def partition_d110_sourceheld_rows(
    labels: Sequence[str] | np.ndarray,
    receiver_ids: Sequence[str] | np.ndarray,
    day_ids: Sequence[str] | np.ndarray,
    physical_ids: Sequence[str] | np.ndarray,
    *,
    historical_query_ids: Sequence[str],
    d104_held_ids: Sequence[str],
    d110_ls_ids: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose exactly seven never-opened rows from every receiver/TX/day cell."""

    arrays = tuple(
        _string_rows(np.asarray(value), name=name, rows=POOL_ROWS)
        for name, value in zip(
            ("labels", "receiver_ids", "day_ids", "physical_ids"),
            (labels, receiver_ids, day_ids, physical_ids),
            strict=True,
        )
    )
    label_array, receiver_array, day_array, physical_array = arrays
    if (
        np.unique(label_array).size != 6
        or np.unique(receiver_array).size != 7
        or np.unique(day_array).size != 4
        or np.unique(physical_array).size != POOL_ROWS
    ):
        raise D110SourceHeldSplitError("D110 split metadata closure drift")
    cells = sorted(
        set(zip(receiver_array.tolist(), label_array.tolist(), day_array.tolist()))
    )
    if len(cells) != CELL_COUNT:
        raise D110SourceHeldSplitError("D110 split requires 168 cells")

    pool_ids = set(physical_array.tolist())
    exclusions = {
        "d103_historical_query": _normalize_exclusion(
            historical_query_ids,
            name="D103 historical query",
            expected_rows=HISTORICAL_QUERY_ROWS,
            pool_ids=pool_ids,
        ),
        "d104_source_val": _normalize_exclusion(
            d104_held_ids,
            name="D104 source held",
            expected_rows=D104_HELD_ROWS,
            pool_ids=pool_ids,
        ),
        "d110_ls": _normalize_exclusion(
            d110_ls_ids,
            name="D110 L_s",
            expected_rows=D110_LS_ROWS,
            pool_ids=pool_ids,
        ),
    }
    exclusion_sets = {name: set(values) for name, values in exclusions.items()}
    if exclusion_sets["d104_source_val"] & exclusion_sets["d103_historical_query"]:
        raise D110SourceHeldSplitError("D104 held overlaps historical query exclusion")
    if exclusion_sets["d104_source_val"] & exclusion_sets["d110_ls"]:
        raise D110SourceHeldSplitError("D104 held overlaps D110 L_s exclusion")
    excluded_union = set().union(*exclusion_sets.values())

    selected: list[int] = []
    available_counts: list[int] = []
    for receiver, tx, day in cells:
        local = np.flatnonzero(
            (receiver_array == receiver)
            & (label_array == tx)
            & (day_array == day)
        ).astype(int)
        available = [
            row for row in local.tolist() if physical_array[row] not in excluded_union
        ]
        if len(available) < HELD_PER_CELL:
            raise D110SourceHeldSplitError(
                f"D110 cell lacks seven unopened rows: {receiver}/{tx}/{day}"
            )
        ranked = sorted(
            available,
            key=lambda row: (
                _rank(receiver, tx, day, physical_array[row]),
                physical_array[row],
            ),
        )
        selected.extend(ranked[:HELD_PER_CELL])
        available_counts.append(len(available))

    selected_rows = np.asarray(sorted(selected), dtype=np.int64)
    selected_ids = physical_array[selected_rows].astype(str).tolist()
    selected_set = set(selected_ids)
    groups = set(zip(receiver_array[selected_rows], label_array[selected_rows]))
    group_counts = sorted(
        int(np.sum((receiver_array[selected_rows] == receiver) & (label_array[selected_rows] == tx)))
        for receiver, tx in groups
    )
    cell_counts = sorted(
        int(
            np.sum(
                (receiver_array[selected_rows] == receiver)
                & (label_array[selected_rows] == tx)
                & (day_array[selected_rows] == day)
            )
        )
        for receiver, tx, day in cells
    )
    intersections = {
        name: len(selected_set & values) for name, values in exclusion_sets.items()
    }
    if (
        len(selected_rows) != EXPECTED_HELD_ROWS
        or len(selected_set) != EXPECTED_HELD_ROWS
        or len(groups) != GROUP_COUNT
        or group_counts != [28] * GROUP_COUNT
        or cell_counts != [HELD_PER_CELL] * CELL_COUNT
        or any(intersections.values())
    ):
        raise D110SourceHeldSplitError("D110 selected-row balance/exclusion closure failed")

    receipt: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "salt": SPLIT_SALT,
        "source_pool_rows": POOL_ROWS,
        "source_val_rows": EXPECTED_HELD_ROWS,
        "cell_count": CELL_COUNT,
        "held_per_cell": HELD_PER_CELL,
        "receiver_tx_group_count": GROUP_COUNT,
        "held_per_receiver_tx": 28,
        "available_after_exclusion_min": min(available_counts),
        "available_after_exclusion_max": max(available_counts),
        "selected_physical_id_root_sha256": canonical_sha256(selected_ids),
        "exclusions": {
            name: {
                "row_count": len(values),
                "physical_id_root_sha256": canonical_sha256(list(values)),
                "selected_intersection_count": intersections[name],
            }
            for name, values in exclusions.items()
        },
        "exclusion_union_row_count": len(excluded_union),
        "source_labels_used_for_stratified_split": True,
        "truth_values_persisted_in_selection_receipt": False,
        "performance_computed": False,
        "target_access": False,
        "formal_query_access": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return selected_rows, receipt


def publish_d110_sourceheld_split(
    source_pool: Mapping[str, np.ndarray],
    *,
    historical_query_ids: Sequence[str],
    d104_held_ids: Sequence[str],
    d110_ls_ids: Sequence[str],
    validation_receipt: Mapping[str, Any],
    input_files: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish an immutable scorer-only archive and truth-free selection receipt."""

    if tuple(source_pool) != SCORER_MEMBERS:
        raise D110SourceHeldSplitError("source pool member order/closure drift")
    rows, selection = partition_d110_sourceheld_rows(
        source_pool["labels"],
        source_pool["receiver_ids"],
        source_pool["day_ids"],
        source_pool["physical_ids"],
        historical_query_ids=historical_query_ids,
        d104_held_ids=d104_held_ids,
        d110_ls_ids=d110_ls_ids,
    )
    scorer = {
        name: (
            np.array(source_pool[name], copy=True)
            if name == "class_ids"
            else np.ascontiguousarray(np.asarray(source_pool[name])[rows])
        )
        for name in SCORER_MEMBERS
    }
    root = Path(output_dir).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing to overwrite D110 source-held root: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        scorer_dir = staging / "scorer_only" / "source_val"
        scorer_dir.mkdir(parents=True)
        archive_path = scorer_dir / "features.npz"
        with archive_path.open("xb") as stream:
            np.savez(stream, **scorer)
            stream.flush()
            os.fsync(stream.fileno())
        array_roots = {name: _array_sha256(value) for name, value in scorer.items()}
        content_sha = hashlib.sha256(_canonical_bytes(array_roots)).hexdigest()
        manifest = {
            "schema": SCORER_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "split_id": SPLIT_ID,
            "role": "source_val_scorer_only",
            "archive": {
                "path": str(Path("scorer_only") / "source_val" / "features.npz"),
                "sha256": _sha256_file(archive_path),
            },
            "exact_member_allowlist": list(SCORER_MEMBERS),
            "array_sha256": array_roots,
            "content_sha256": content_sha,
            "row_count": EXPECTED_HELD_ROWS,
            "asset_access": False,
            "gradient_access": False,
            "selection_access": False,
            "target_access": False,
            "formal_query_access": False,
            "performance_computed": False,
            "d106_prepare_member_compatible": True,
        }
        manifest_path = scorer_dir / "manifest.json"
        _write_new(manifest_path, _canonical_bytes(manifest))

        receipt = {
            **selection,
            "source_feature_validation": dict(validation_receipt),
            "input_files": {name: dict(value) for name, value in input_files.items()},
            "scorer_archive_sha256": manifest["archive"]["sha256"],
            "scorer_manifest_sha256": _sha256_file(manifest_path),
        }
        receipt["receipt_sha256"] = canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        receipt_path = staging / "selection_receipt.json"
        _write_new(receipt_path, _canonical_bytes(receipt))
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "D110_SOURCE_HELD_SPLIT_COMPLETE",
        "output_dir": str(root),
        "row_count": EXPECTED_HELD_ROWS,
        "scorer_archive": str(root / "scorer_only" / "source_val" / "features.npz"),
        "scorer_manifest": str(root / "scorer_only" / "source_val" / "manifest.json"),
        "selection_receipt": str(root / "selection_receipt.json"),
    }


def load_historical_query_ids(path: str | Path) -> tuple[str, ...]:
    """Read only the ID list from the frozen D103-query exclusion manifest."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = value.get("query_physical_ids_sorted") if isinstance(value, dict) else None
    if not isinstance(ids, list) or value.get("query_physical_id_count") != len(ids):
        raise D110SourceHeldSplitError("historical-query manifest ID closure drift")
    return tuple(str(item) for item in ids)


def load_d104_held_ids(
    manifest_path: str | Path, *, package_root: str | Path | None = None
) -> tuple[str, ...]:
    """Reconstruct D104 held IDs from seven K10 no-truth predictor packages."""

    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("split_id") != "d104_source_seed104713_v2"
        or manifest.get("query_truth_present") is not False
        or manifest.get("package_count") != 21
        or not isinstance(manifest.get("packages"), list)
    ):
        raise D110SourceHeldSplitError("D104 no-truth package manifest drift")
    root = Path(package_root).resolve() if package_root is not None else path.parent
    selected = [row for row in manifest["packages"] if row.get("K") == 10]
    if len(selected) != 7 or any(row.get("query_truth_present") is not False for row in selected):
        raise D110SourceHeldSplitError("D104 K10 no-truth package closure drift")
    ids: set[str] = set()
    allowed = {
        "support_pre_relu",
        "support_zdom",
        "support_labels",
        "support_physical_ids",
        "query_pre_relu",
        "query_physical_ids",
        "registered_classes",
    }
    for row in selected:
        package = (root / str(row["path"])).resolve(strict=True)
        if _sha256_file(package) != row.get("sha256"):
            raise D110SourceHeldSplitError("D104 K10 package SHA drift")
        with np.load(package, allow_pickle=False) as archive:
            if set(archive.files) != allowed or any("truth" in name.lower() for name in archive.files):
                raise D110SourceHeldSplitError("D104 predictor package truth/member drift")
            support = archive["support_physical_ids"].astype(str).tolist()
            query = archive["query_physical_ids"].astype(str).tolist()
        if canonical_sha256(support) != row.get("support_physical_id_root_sha256"):
            raise D110SourceHeldSplitError("D104 support ID root drift")
        if canonical_sha256(query) != row.get("query_physical_id_root_sha256"):
            raise D110SourceHeldSplitError("D104 query ID root drift")
        if set(support) & set(query):
            raise D110SourceHeldSplitError("D104 package support/query ID overlap")
        ids.update(support)
        ids.update(query)
    if len(ids) != D104_HELD_ROWS:
        raise D110SourceHeldSplitError("D104 held ID union must contain 2520 rows")
    return tuple(sorted(ids))


def load_physical_ids_only(path: str | Path) -> tuple[str, ...]:
    """Read only ``physical_ids`` from an NPZ source; no other member is opened."""

    with np.load(Path(path).resolve(strict=True), allow_pickle=False) as archive:
        if "physical_ids" not in archive.files:
            raise D110SourceHeldSplitError("physical-ID source lacks physical_ids")
        ids = archive["physical_ids"].astype(str).tolist()
    return tuple(ids)
