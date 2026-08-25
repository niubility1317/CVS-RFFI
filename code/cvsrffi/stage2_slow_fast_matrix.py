"""Truth-blind nine-row diagnostic launcher for slow/fast adapters."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .stage2_meta_adapter_matrix import _write_json_exclusive
from .stage2_slow_fast_runner import _validate_config, run_slow_fast_stage2_row


_SCHEMA = "cvs.stage2.slow_fast.diag9.v1"
_CANDIDATES = ("COMMON_SHIFT_R4", "FAST_FILM_R8", "FAST_LOWRANK_R8")
_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
_TOP_KEYS = frozenset({"schema", "rows"})
_ROW_KEYS = frozenset({"row_id", "config"})
_ROW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")


def _exact(payload: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise ValueError(
            f"{label} allowlist mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _validate_matrix(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("diagnostic matrix must be a mapping")
    _exact(payload, _TOP_KEYS, "diagnostic matrix")
    if payload["schema"] != _SCHEMA:
        raise ValueError(f"diagnostic matrix schema must be {_SCHEMA}")
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("diagnostic rows must be a sequence")
    if len(raw_rows) != 9:
        raise ValueError("slow-fast diagnostic requires exactly 9 rows")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    row_ids: set[str] = set()
    shared: set[tuple[str, str, str, int, str, str]] = set()
    scene_inputs: dict[str, set[tuple[str, str]]] = {}
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"diagnostic row {index} must be a mapping")
        _exact(raw, _ROW_KEYS, f"diagnostic row {index}")
        row_id = str(raw["row_id"])
        if not _ROW_ID.fullmatch(row_id) or row_id in row_ids:
            raise ValueError(f"diagnostic row {index} has unsafe or duplicate row_id")
        row_ids.add(row_id)
        config = _validate_config(raw["config"])
        if config["receiver"] != "20-1" or config["operating_point"] != "K10/new10":
            raise ValueError("diagnostic rows require receiver20-1 and K10/new10")
        if config["k_shot"] != 10 or config["steps"] != 3:
            raise ValueError("diagnostic rows require K=10 and exactly three fast updates")
        identities.add((config["candidate_id"], config["scenario"]))
        scene_inputs.setdefault(config["scenario"], set()).add(
            (config["support_path"], config["query_path"])
        )
        shared.add(
            (
                config["capsule_id"], config["split_id"], config["receiver"],
                config["seed"], config["prototype_path"], config["base_checkpoint_path"],
            )
        )
        rows.append({"row_id": row_id, "config": config})
    expected = {(candidate, scenario) for candidate in _CANDIDATES for scenario in _SCENARIOS}
    if identities != expected:
        raise ValueError("diagnostic rows must equal the three-candidate/three-scene product")
    if len(shared) != 1:
        raise ValueError(
            "diagnostic rows must share capsule, split, receiver, seed, prototypes and base checkpoint"
        )
    if any(len(paths) != 1 for paths in scene_inputs.values()):
        raise ValueError("all candidates in one scene must share support/query paths")
    return rows


def run_slow_fast_matrix(
    payload: Mapping[str, Any],
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    rows = _validate_matrix(payload)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"matrix output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    completed: list[dict[str, Any]] = []
    active = ""
    try:
        for row in rows:
            active = row["row_id"]
            receipt = run_slow_fast_stage2_row(
                row["config"], destination / active, device=device
            )
            if receipt.get("status") != "PREDICTIONS_COMPLETE":
                raise ValueError(f"row {active} did not close predictions")
            if any(
                bool(receipt.get(key))
                for key in ("query_truth_opened", "query_role_opened", "source_opened")
            ):
                raise ValueError(f"row {active} reported forbidden input access")
            if receipt.get("states_same_row") is not True or int(
                receipt.get("query_state_update_count", -1)
            ) != 0:
                raise ValueError(f"row {active} violated same-row/query-read-only semantics")
            completed.append(
                {
                    "row_id": active,
                    "candidate_id": receipt["candidate_id"],
                    "scenario": receipt["scenario"],
                    "selected_lambda": float(receipt["selected_lambda"]),
                    "status": receipt["status"],
                }
            )
    except Exception as exc:
        _write_json_exclusive(
            destination / "matrix_failure.json",
            {
                "schema": "cvs.stage2.slow_fast.matrix_failure.v1",
                "status": "FAILED",
                "failed_row_id": active,
                "completed_row_count": len(completed),
                "completed_rows": completed,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "truth_opened": False,
                "source_opened": False,
            },
        )
        raise
    receipt = {
        "schema": "cvs.stage2.slow_fast.matrix_receipt.v1",
        "status": "PREDICTIONS_COMPLETE",
        "completed_row_count": 9,
        "completed_rows": completed,
        "truth_opened": False,
        "source_opened": False,
    }
    _write_json_exclusive(destination / "matrix_receipt.json", receipt)
    return receipt


__all__ = ["run_slow_fast_matrix"]
