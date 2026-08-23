"""Export minimal ground-prepared target rows for Stage2 adaptation.

The exporter only reshapes already prepared target support/query NPZ files. It
does not load a checkpoint, source/clean data, query truth, or query roles.
Support-only mode is suitable for a real no-query smoke; formal mode adds the
fixed query IQ and opaque query IDs after all inputs have passed their exact
field allowlists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_SUPPORT_INPUT_REQUIRED = frozenset(
    {
        "support_pool_leo_weak_iq",
        "support_pool_class_indices",
        "support_pool_rank_within_class",
        "support_pool_tokens",
    }
)
_SUPPORT_VALIDATED_ONCE_SCHEMA = _SUPPORT_INPUT_REQUIRED | frozenset(
    {
        "support_pool_overlay_tokens",
        "support_pool_satellite_seeds",
        "support_pool_post_channel_iq_sha256",
        "manifest_json",
    }
)
_SUPPORT_INPUT_SCHEMAS = (
    _SUPPORT_INPUT_REQUIRED,
    _SUPPORT_VALIDATED_ONCE_SCHEMA,
)

_QUERY_INPUT_REQUIRED = frozenset({"query_leo_weak_iq", "query_tokens"})
_QUERY_VALIDATED_ONCE_SCHEMA = _QUERY_INPUT_REQUIRED | frozenset(
    {
        "query_overlay_tokens",
        "query_satellite_seeds",
        "query_post_channel_iq_sha256",
        "manifest_json",
    }
)
_QUERY_INPUT_SCHEMAS = (
    _QUERY_INPUT_REQUIRED,
    _QUERY_VALIDATED_ONCE_SCHEMA,
)


class TargetRowExportError(ValueError):
    """Raised when the minimal target-row export contract is violated."""


def _validate_exact_schema(
    actual: frozenset[str],
    allowed_schemas: Sequence[frozenset[str]],
    *,
    label: str,
) -> None:
    if any(actual == allowed for allowed in allowed_schemas):
        return
    expected = [sorted(allowed) for allowed in allowed_schemas]
    raise TargetRowExportError(
        f"{label} payload allowlist mismatch: "
        f"actual={sorted(actual)}, expected_one_of={expected}"
    )


def _load_exact_npz(
    path: str | Path,
    *,
    allowed_schemas: Sequence[frozenset[str]],
    required: frozenset[str],
    label: str,
) -> dict[str, np.ndarray]:
    resolved = Path(path)
    if not resolved.is_file() or resolved.suffix.lower() != ".npz":
        raise TargetRowExportError(f"{label} NPZ is missing or invalid: {resolved}")
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            # Validate names before indexing any member.  Thus a forbidden
            # query-truth/role member is never deserialized or read.
            names = frozenset(str(name) for name in archive.files)
            _validate_exact_schema(names, allowed_schemas, label=label)
            # Builder metadata in a recognized VALIDATED_ONCE bundle is not
            # copied into the Phase2 payload and is never materialized here.
            return {name: np.asarray(archive[name]).copy() for name in required}
    except TargetRowExportError:
        raise
    except (OSError, ValueError) as exc:
        raise TargetRowExportError(f"cannot load {label} NPZ: {resolved}") from exc


def _validate_iq(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] != 2
        or array.shape[2] < 1
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise TargetRowExportError(
            f"{label} IQ must be a finite nonempty [N,2,L] array"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def _validate_integer_vector(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.shape[0] < 1
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise TargetRowExportError(f"{label} must be a nonempty integer vector")
    return np.ascontiguousarray(array, dtype=np.int64)


def _validate_ids(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] < 1 or array.dtype.kind == "O":
        raise TargetRowExportError(
            f"{label} must be a nonempty non-object ID vector"
        )
    ids = array.astype(str)
    if any(not item for item in ids.tolist()):
        raise TargetRowExportError(f"{label} contains an empty ID")
    if len(set(ids.tolist())) != len(ids):
        raise TargetRowExportError(f"{label} must contain unique IDs")
    return ids


def _prepare_support(
    payload: Mapping[str, np.ndarray],
    *,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int], dict[str, int]]:
    iq = _validate_iq(payload["support_pool_leo_weak_iq"], label="support")
    labels = _validate_integer_vector(
        payload["support_pool_class_indices"],
        label="support_pool_class_indices",
    )
    ranks = _validate_integer_vector(
        payload["support_pool_rank_within_class"],
        label="support_pool_rank_within_class",
    )
    tokens = _validate_ids(payload["support_pool_tokens"], label="support_pool_tokens")
    row_count = int(iq.shape[0])
    if not (labels.shape[0] == ranks.shape[0] == tokens.shape[0] == row_count):
        raise TargetRowExportError("support IQ, labels, ranks, and IDs must align")
    if np.any(ranks < 0):
        raise TargetRowExportError("support ranks must be nonnegative")

    selected_mask = np.zeros(row_count, dtype=bool)
    per_class_counts: dict[str, int] = {}
    class_ids = np.unique(labels)
    for class_id in class_ids.tolist():
        positions = np.flatnonzero(labels == int(class_id))
        class_ranks = ranks[positions]
        if np.unique(class_ranks).shape[0] != class_ranks.shape[0]:
            raise TargetRowExportError(
                f"support rank prefix contains duplicate rank for class {class_id}"
            )
        prefix_positions = positions[class_ranks < int(k_shot)]
        observed_prefix = sorted(int(value) for value in ranks[prefix_positions].tolist())
        expected_prefix = list(range(int(k_shot)))
        if observed_prefix != expected_prefix:
            raise TargetRowExportError(
                f"support rank prefix is incomplete for class {class_id}: "
                f"observed={observed_prefix}, expected={expected_prefix}"
            )
        selected_mask[prefix_positions] = True
        per_class_counts[str(int(class_id))] = int(prefix_positions.shape[0])

    selected_indices = np.flatnonzero(selected_mask)
    selected_tokens = tokens[selected_indices].tolist()
    return (
        iq[selected_indices].copy(),
        labels[selected_indices].copy(),
        selected_tokens,
        [int(value) for value in class_ids.tolist()],
        per_class_counts,
    )


def _prepare_query(
    payload: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    iq = _validate_iq(payload["query_leo_weak_iq"], label="query")
    query_ids = _validate_ids(payload["query_tokens"], label="query_tokens")
    if query_ids.shape[0] != iq.shape[0]:
        raise TargetRowExportError("query IQ and IDs must align")
    return iq.copy(), query_ids.copy()


def _preflight_outputs(paths: Sequence[Path]) -> None:
    if len(set(paths)) != len(paths):
        raise TargetRowExportError("output paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise TargetRowExportError(f"output already exists: {existing}")


def _write_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, **payload)


def export_target_row(
    *,
    support_input: str | Path,
    support_output: str | Path,
    audit_output: str | Path,
    k_shot: int,
    query_input: str | Path | None = None,
    query_output: str | Path | None = None,
) -> dict[str, Any]:
    """Export fixed rank-prefix support and, optionally, a truth-free query."""

    requested_k = int(k_shot)
    if requested_k < 1:
        raise TargetRowExportError("k_shot must be positive")
    if (query_input is None) != (query_output is None):
        raise TargetRowExportError(
            "query_input and query_output must both be supplied for formal mode"
        )

    support_destination = Path(support_output)
    audit_destination = Path(audit_output)
    query_destination = Path(query_output) if query_output is not None else None
    output_paths = [support_destination, audit_destination]
    if query_destination is not None:
        output_paths.append(query_destination)
    _preflight_outputs(output_paths)

    support_payload = _load_exact_npz(
        support_input,
        allowed_schemas=_SUPPORT_INPUT_SCHEMAS,
        required=_SUPPORT_INPUT_REQUIRED,
        label="support",
    )
    (
        support_iq,
        support_labels,
        selected_support_ids,
        support_class_ids,
        support_per_class_counts,
    ) = _prepare_support(support_payload, k_shot=requested_k)

    query_iq: np.ndarray | None = None
    query_ids: np.ndarray | None = None
    if query_input is not None:
        query_payload = _load_exact_npz(
            query_input,
            allowed_schemas=_QUERY_INPUT_SCHEMAS,
            required=_QUERY_INPUT_REQUIRED,
            label="query",
        )
        query_iq, query_ids = _prepare_query(query_payload)

    audit: dict[str, Any] = {
        "schema": "cvs.stage2.target_row_export.v1",
        "mode": (
            "formal_support_and_query"
            if query_iq is not None
            else "support_only_no_query_smoke"
        ),
        "k_shot": requested_k,
        "support_input_rows": int(
            np.asarray(support_payload["support_pool_leo_weak_iq"]).shape[0]
        ),
        "support_output_rows": int(support_iq.shape[0]),
        "support_class_count": len(support_class_ids),
        "support_class_ids": support_class_ids,
        "support_per_class_counts": support_per_class_counts,
        "support_selected_ids": selected_support_ids,
        "support_ids_preserved": True,
        "query_input_opened": query_iq is not None,
        "query_input_rows": int(query_iq.shape[0]) if query_iq is not None else 0,
        "query_output_rows": int(query_iq.shape[0]) if query_iq is not None else 0,
        "query_ids": query_ids.tolist() if query_ids is not None else [],
        "query_ids_preserved": query_ids is not None,
        "query_truth_opened": False,
        "query_role_opened": False,
    }

    # No output is created until every requested input has passed validation.
    _write_npz(
        support_destination,
        {
            "received_iq": support_iq,
            "support_labels": support_labels,
        },
    )
    if query_destination is not None:
        assert query_iq is not None and query_ids is not None
        _write_npz(
            query_destination,
            {"received_iq": query_iq, "query_ids": query_ids},
        )
    audit_destination.parent.mkdir(parents=True, exist_ok=True)
    with audit_destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-input", type=Path, required=True)
    parser.add_argument("--support-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--k-shot", type=int, required=True)
    parser.add_argument("--query-input", type=Path)
    parser.add_argument("--query-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = export_target_row(
        support_input=args.support_input,
        support_output=args.support_output,
        audit_output=args.audit_output,
        k_shot=args.k_shot,
        query_input=args.query_input,
        query_output=args.query_output,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TargetRowExportError", "export_target_row", "main"]
