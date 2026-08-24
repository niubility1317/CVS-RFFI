"""Truth-blind sequential launcher for a frozen meta-adapter matrix.

The launcher validates the complete Target5/Target25 cartesian product before
creating its output root.  It delegates every row to the existing single-row
runner and never accepts or opens a truth artifact.  Completed row outputs are
preserved when a later row fails.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .stage2_meta_adapter_runner import (
    _validate_config,
    run_meta_adapter_stage2_row,
)


_MATRIX_SCHEMA = "cvs.stage2.meta_adapter.matrix.v1"
_TOP_LEVEL_KEYS = frozenset({"schema", "target", "rows"})
_ROW_KEYS = frozenset({"row_id", "config"})
_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
_OPERATING_POINTS = {
    "K10/new5": 10,
    "K10/new10": 10,
    "K10/new20": 10,
    "K5/new20": 5,
    "K1/new20": 1,
}
_TARGET_RECEIVERS = {
    "Target5": ("20-1",),
    "Target25": ("20-1", "3-19", "7-14", "7-7", "8-8"),
}
_ROW_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")


class MetaAdapterMatrixError(ValueError):
    """Raised when a matrix launch violates its frozen contract."""


def _exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise MetaAdapterMatrixError(
            f"{label} allowlist mismatch: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _validate_matrix(config: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(config, Mapping):
        raise MetaAdapterMatrixError("matrix config must be a mapping")
    _exact_keys(config, _TOP_LEVEL_KEYS, label="matrix config")
    if config["schema"] != _MATRIX_SCHEMA:
        raise MetaAdapterMatrixError(f"matrix schema must be {_MATRIX_SCHEMA}")
    target = str(config["target"])
    if target not in _TARGET_RECEIVERS:
        raise MetaAdapterMatrixError("matrix target must be Target5 or Target25")
    raw_rows = config["rows"]
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise MetaAdapterMatrixError("matrix rows must be a sequence")
    expected_count = len(_TARGET_RECEIVERS[target]) * len(_OPERATING_POINTS) * len(
        _SCENARIOS
    )
    if len(raw_rows) != expected_count:
        raise MetaAdapterMatrixError(
            f"{target} requires exactly {expected_count} rows"
        )

    rows: list[dict[str, Any]] = []
    row_ids: set[str] = set()
    identities: set[tuple[str, str, str]] = set()
    candidate_ids: set[str] = set()
    bundle_ids: set[str] = set()
    seeds: set[int] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise MetaAdapterMatrixError(f"matrix row {index} must be a mapping")
        _exact_keys(raw, _ROW_KEYS, label=f"matrix row {index}")
        row_id = str(raw["row_id"])
        if not _ROW_ID_PATTERN.fullmatch(row_id):
            raise MetaAdapterMatrixError(f"matrix row {index} has unsafe row_id")
        if row_id in row_ids:
            raise MetaAdapterMatrixError(f"duplicate matrix row_id: {row_id}")
        row_ids.add(row_id)
        try:
            resolved = _validate_config(raw["config"], require_query=True)
        except (TypeError, ValueError) as exc:
            raise MetaAdapterMatrixError(str(exc)) from exc
        identity = (
            str(resolved["receiver"]),
            str(resolved["operating_point"]),
            str(resolved["scenario"]),
        )
        identities.add(identity)
        candidate_ids.add(str(resolved["candidate_id"]))
        bundle_ids.add(str(resolved["bundle_id"]))
        seeds.add(int(resolved["seed"]))
        rows.append({"row_id": row_id, "config": resolved})

    expected_identities = {
        (receiver, operating_point, scenario)
        for receiver in _TARGET_RECEIVERS[target]
        for operating_point in _OPERATING_POINTS
        for scenario in _SCENARIOS
    }
    if identities != expected_identities or len(identities) != len(rows):
        raise MetaAdapterMatrixError(
            f"{target} rows must equal the frozen receiver/operating-point/scenario "
            "cartesian product"
        )
    for row in rows:
        row_config = row["config"]
        expected_k = _OPERATING_POINTS[str(row_config["operating_point"])]
        if int(row_config["k_shot"]) != expected_k:
            raise MetaAdapterMatrixError(
                "matrix k_shot must match its frozen operating point"
            )
    if len(candidate_ids) != 1 or len(bundle_ids) != 1 or len(seeds) != 1:
        raise MetaAdapterMatrixError(
            "matrix rows must share one candidate, bundle, and seed"
        )
    return target, rows


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"matrix artifact already exists: {path}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"matrix artifact appeared during write: {path}")
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def run_meta_adapter_matrix(
    config: Mapping[str, Any],
    output_dir: str | Path,
    device: str | torch.device,
) -> Mapping[str, Any]:
    """Run a validated Target5/Target25 matrix without opening scorer truth."""

    target, rows = _validate_matrix(config)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"matrix output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=False, exist_ok=False)

    completed: list[dict[str, Any]] = []
    active_row_id = ""
    try:
        for row in rows:
            active_row_id = str(row["row_id"])
            receipt = dict(
                run_meta_adapter_stage2_row(
                    row["config"], destination / active_row_id, device
                )
            )
            if receipt.get("status") != "PREDICTIONS_COMPLETE":
                raise MetaAdapterMatrixError(
                    f"row {active_row_id} did not close predictions"
                )
            if any(
                bool(receipt.get(key))
                for key in ("query_truth_opened", "query_role_opened", "source_opened")
            ):
                raise MetaAdapterMatrixError(
                    f"row {active_row_id} reported a forbidden input access"
                )
            if int(receipt.get("query_state_update_count", -1)) != 0:
                raise MetaAdapterMatrixError(
                    f"row {active_row_id} reported a query state update"
                )
            if receipt.get("states_same_row") is not True:
                raise MetaAdapterMatrixError(
                    f"row {active_row_id} did not preserve the same-row comparison"
                )
            if int(receipt.get("backward_count", -1)) != 3:
                raise MetaAdapterMatrixError(
                    f"row {active_row_id} did not perform exactly three updates"
                )
            trainable_fraction = float(receipt.get("trainable_fraction", float("inf")))
            if not (0.0 < trainable_fraction <= 0.01):
                raise MetaAdapterMatrixError(
                    f"row {active_row_id} exceeded the trainable parameter budget"
                )
            completed.append(
                {
                    "row_id": active_row_id,
                    "receiver": receipt["receiver"],
                    "scenario": receipt["scenario"],
                    "operating_point": receipt["operating_point"],
                    "seed": int(receipt["seed"]),
                    "k_shot": int(receipt["k_shot"]),
                    "status": str(receipt["status"]),
                }
            )
    except Exception as exc:
        failure = {
            "schema": "cvs.stage2.meta_adapter.matrix_failure.v1",
            "status": "FAILED",
            "target": target,
            "completed_row_count": len(completed),
            "completed_rows": completed,
            "failed_row_id": active_row_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "truth_opened": False,
            "source_opened": False,
        }
        _write_json_exclusive(destination / "matrix_failure.json", failure)
        raise

    receipt = {
        "schema": "cvs.stage2.meta_adapter.matrix_receipt.v1",
        "status": "PREDICTIONS_COMPLETE",
        "target": target,
        "completed_row_count": len(completed),
        "completed_rows": completed,
        "truth_opened": False,
        "source_opened": False,
    }
    _write_json_exclusive(destination / "matrix_receipt.json", receipt)
    return receipt


__all__ = [
    "MetaAdapterMatrixError",
    "run_meta_adapter_matrix",
]
