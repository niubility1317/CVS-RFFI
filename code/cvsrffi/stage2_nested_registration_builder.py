"""Build truth-isolated nested new-class slices from one validated new20 row."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


NEW_CLASS_COUNTS = (1, 2, 3, 5, 10, 15, 20)
_SUPPORT_REQUIRED = frozenset(
    {
        "support_pool_leo_weak_iq",
        "support_pool_class_indices",
        "support_pool_rank_within_class",
        "support_pool_tokens",
    }
)
_QUERY_REQUIRED = frozenset({"query_leo_weak_iq", "query_tokens"})


class NestedRegistrationBuildError(ValueError):
    """Raised when a nested Phase2-C slice cannot be built safely."""


def _load_required(path: str | Path, required: frozenset[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise NestedRegistrationBuildError("validated parent payload schema drift")
            return {name: np.asarray(archive[name]).copy() for name in required}
    except NestedRegistrationBuildError:
        raise
    except (OSError, ValueError) as exc:
        raise NestedRegistrationBuildError("cannot load validated parent payload") from exc


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, **arrays)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_nested_registration_slices(
    *,
    max_support_path: str | Path,
    max_query_path: str | Path,
    truth_sidecar_path: str | Path,
    scenario: str,
    output_root: str | Path,
    capsule_id: str,
    da_split_id: str,
    base_checkpoint_path: str,
    new_class_counts: Iterable[int] = NEW_CLASS_COUNTS,
) -> dict[str, Any]:
    """Create nested predictor/scorer slices without exposing truth to prediction."""

    counts = tuple(int(value) for value in new_class_counts)
    if counts != NEW_CLASS_COUNTS:
        raise NestedRegistrationBuildError("new-class matrix is not the frozen nested set")
    destination = Path(output_root)
    if destination.exists() or destination.is_symlink():
        raise NestedRegistrationBuildError("nested output root already exists")
    if not capsule_id.strip() or not da_split_id.strip() or not scenario.startswith("leo_"):
        raise NestedRegistrationBuildError("nested data identity drift")

    support = _load_required(max_support_path, _SUPPORT_REQUIRED)
    query = _load_required(max_query_path, _QUERY_REQUIRED)
    support_iq = np.asarray(support["support_pool_leo_weak_iq"], dtype=np.float32)
    support_labels = np.asarray(support["support_pool_class_indices"], dtype=np.int64)
    support_ranks = np.asarray(support["support_pool_rank_within_class"], dtype=np.int64)
    support_ids = np.asarray(support["support_pool_tokens"]).astype(str)
    query_iq = np.asarray(query["query_leo_weak_iq"], dtype=np.float32)
    query_ids = np.asarray(query["query_tokens"]).astype(str)
    if (
        support_iq.ndim != 3
        or support_iq.shape[1:] != (2, 256)
        or query_iq.ndim != 3
        or query_iq.shape[1:] != (2, 256)
        or support_labels.shape != support_ranks.shape
        or support_labels.shape != support_ids.shape
        or support_labels.shape[0] != support_iq.shape[0]
        or query_ids.shape != (query_iq.shape[0],)
        or len(set(support_ids.tolist())) != len(support_ids)
        or len(set(query_ids.tolist())) != len(query_ids)
        or set(support_ids.tolist()) & set(query_ids.tolist())
    ):
        raise NestedRegistrationBuildError("validated parent geometry drift")

    try:
        sidecar = json.loads(Path(truth_sidecar_path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NestedRegistrationBuildError("cannot load detached parent truth") from exc
    if sidecar.get("schema") != "cvs.phase2.query_truth_sidecar.v2" or not isinstance(
        sidecar.get("rows"), list
    ):
        raise NestedRegistrationBuildError("detached parent truth schema drift")
    truth_by_token: dict[str, int] = {}
    for row in sidecar["rows"]:
        if not isinstance(row, dict) or row.get("scenario") != scenario:
            continue
        token = str(row.get("query_token", ""))
        label = row.get("true_class_index")
        if token in truth_by_token or not isinstance(label, int):
            raise NestedRegistrationBuildError("detached parent truth row drift")
        truth_by_token[token] = int(label)
    if set(truth_by_token) != set(query_ids.tolist()):
        raise NestedRegistrationBuildError("query/truth token join drift")
    aligned_truth = np.asarray([truth_by_token[token] for token in query_ids], dtype=np.int64)
    if set(support_labels.tolist()) != set(range(26)) or set(aligned_truth.tolist()) != set(
        range(26)
    ):
        raise NestedRegistrationBuildError("new20 class coverage drift")

    support_order = np.lexsort((support_ranks, support_labels))
    old_support_indices = support_order[
        (support_labels[support_order] < 6) & (support_ranks[support_order] < 10)
    ]
    if np.bincount(support_labels[old_support_indices], minlength=6).tolist() != [10] * 6:
        raise NestedRegistrationBuildError("nested old6 K-shot balance drift")
    _write_npz(
        destination / "old_support.npz",
        received_iq=support_iq[old_support_indices],
        support_labels=support_labels[old_support_indices],
        support_physical_ids=support_ids[old_support_indices],
    )

    rows: list[dict[str, Any]] = []
    for new_count in counts:
        class_count = 6 + new_count
        support_indices = support_order[
            (support_labels[support_order] < class_count)
            & (support_ranks[support_order] < 10)
        ]
        query_mask = aligned_truth < class_count
        selected_support_labels = support_labels[support_indices]
        selected_query_labels = aligned_truth[query_mask]
        if (
            np.bincount(selected_support_labels, minlength=class_count).tolist()
            != [10] * class_count
            or len(set(np.bincount(selected_query_labels, minlength=class_count).tolist()))
            != 1
        ):
            raise NestedRegistrationBuildError("nested K/query balance drift")
        split_id = f"{da_split_id}-new{new_count}"
        root = destination / f"new{new_count}"
        predictor = root / "predictor"
        scorer = root / "scorer"
        _write_npz(
            predictor / f"support_{scenario}.npz",
            received_iq=support_iq[support_indices],
            support_labels=selected_support_labels,
            support_physical_ids=support_ids[support_indices],
        )
        _write_npz(
            predictor / f"query_{scenario}.npz",
            received_iq=query_iq[query_mask],
            query_ids=query_ids[query_mask],
        )
        _write_npz(
            scorer / f"truth_{scenario}.npz",
            query_ids=query_ids[query_mask],
            query_labels=selected_query_labels,
        )
        handle = {
            "schema": "cvs.sf_erbt_four_state.handle.v1",
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": capsule_id,
            "split_id": split_id,
            "da_split_id": da_split_id,
            "scenario": scenario,
            "k_shot": 10,
            "old_class_count": 6,
            "new_class_count": new_count,
            "old_support_rows": 60,
            "registered_support_rows": class_count * 10,
            "query_rows": int(np.sum(query_mask)),
            "base_checkpoint_path": base_checkpoint_path,
        }
        _write_json(root / "data_handle.json", handle)
        rows.append(
            {
                "new_class_count": new_count,
                "registered_support_rows": class_count * 10,
                "query_rows": int(np.sum(query_mask)),
                "query_rows_per_class": int(np.sum(query_mask) // class_count),
                "split_id": split_id,
            }
        )
    audit = {
        "schema": "cvs.phase2.nested_registration_build.v1",
        "status": "VALIDATED_ONCE",
        "protocol_schema": "p2_min_v1",
        "capsule_id": capsule_id,
        "da_split_id": da_split_id,
        "scenario": scenario,
        "new_class_counts": list(counts),
        "parent_new20_support_query_disjoint_preserved_by_subsetting": True,
        "predictor_truth_isolated": True,
        "old_support_rows": 60,
        "rows": rows,
    }
    _write_json(destination / "build_audit.json", audit)
    return audit


__all__ = ["NEW_CLASS_COUNTS", "build_nested_registration_slices"]
