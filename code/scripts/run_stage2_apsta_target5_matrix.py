"""Execute the preregistered APSTA-P1 Target5 prediction/score matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "cvs.stage2b.apsta_p1_target5_matrix.v1"
CANDIDATE = "APSTA_P1_TIME_FUSION_ROBUST"
FORMAL_CHECKPOINTS = (0, 10, 30, 100, 300)
CONFIG_KEYS = frozenset({
    "schema", "run_id", "checkpoint", "candidate_id", "checkpoints",
    "learning_rate", "anchor_strength", "head_ce_weight", "loo_mean_weight",
    "tail_weight", "tail_temperature", "topology_weight", "l2sp_weight",
    "margin_epsilon", "row_source_config",
})
ROW_KEYS = frozenset({
    "row_id", "scenario", "work_dir", "query_package", "package_manifest",
    "validated_row_manifest", "row_binding", "truth", "baseline_prediction",
})


class MatrixConfigError(ValueError):
    """Raised when the frozen APSTA Target5 matrix drifts."""


@dataclass(frozen=True)
class MatrixTask:
    row_id: str
    scenario: str
    prediction_output: Path
    score_output: Path
    prediction_command: tuple[str, ...]
    score_command: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise MatrixConfigError(f"JSON object required: {path}")
    return value


def _remote_absolute(value: Any, *, field: str) -> str:
    text = str(value)
    if not text.startswith("/") or ".." in Path(text).parts:
        raise MatrixConfigError(f"{field} must be an absolute remote path")
    return text


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    value = _read_json(config_path)
    if frozenset(value) != CONFIG_KEYS:
        raise MatrixConfigError("matrix config schema is not exhaustive")
    locked = (
        value["schema"] == SCHEMA
        and str(value["run_id"]).strip()
        and value["candidate_id"] == CANDIDATE
        and tuple(int(item) for item in value["checkpoints"]) == FORMAL_CHECKPOINTS
        and float(value["learning_rate"]) == 2.0e-4
        and float(value["anchor_strength"]) == 3.0
        and float(value["head_ce_weight"]) == 0.25
        and float(value["loo_mean_weight"]) == 1.0
        and float(value["tail_weight"]) == 0.5
        and float(value["tail_temperature"]) == 0.5
        and float(value["topology_weight"]) == 0.25
        and float(value["l2sp_weight"]) == 1.0e-3
        and float(value["margin_epsilon"]) == 0.0
    )
    if not locked:
        raise MatrixConfigError("matrix method lock drift")
    _remote_absolute(value["checkpoint"], field="checkpoint")
    source_path = Path(str(value["row_source_config"]))
    if not source_path.is_absolute():
        source_path = config_path.parent / source_path
    rows = _read_json(source_path).get("rows")
    if not isinstance(rows, list) or not rows:
        raise MatrixConfigError("row source is missing the frozen Target5 rows")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or frozenset(row) != ROW_KEYS:
            raise MatrixConfigError("matrix row schema drift")
        row_id = str(row["row_id"])
        if (
            not row_id or row_id in seen or "_k5_new20_" not in row_id
            or str(row["scenario"]) not in (
                "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"
            )
        ):
            raise MatrixConfigError("Target5 row identity drift")
        seen.add(row_id)
        for field in ROW_KEYS - {"row_id", "scenario"}:
            _remote_absolute(row[field], field=field)
    return {**value, "rows": rows, "row_source_config_resolved": str(source_path)}


def plan_tasks(
    config: dict[str, Any],
    *,
    release_root: Path,
    output_root: Path,
    device: str,
) -> tuple[MatrixTask, ...]:
    predictor = release_root / "code" / "scripts" / "run_stage2_apsta_p1.py"
    scorer = release_root / "code" / "scripts" / "score_stage2_structured_late_block_pair.py"
    tasks: list[MatrixTask] = []
    for row in config["rows"]:
        row_id = str(row["row_id"])
        work_dir = Path(str(row["work_dir"]))
        prediction = output_root / "predictions" / f"{row_id}_da1_reg0.json"
        score = output_root / "scores" / f"{row_id}_pair.json"
        command = [
            str(predictor), "run-row", "--checkpoint", str(config["checkpoint"]),
            "--support-only", str(work_dir / "support_only.npz"),
            "--frozen-prototypes", str(work_dir / "frozen_class_prototypes.npz"),
            "--context", str(work_dir / "context.json"),
            "--output", str(prediction), "--device", str(device),
            "--checkpoints", *[str(item) for item in config["checkpoints"]],
            "--learning-rate", str(config["learning_rate"]),
            "--anchor-strength", str(config["anchor_strength"]),
            "--head-ce-weight", str(config["head_ce_weight"]),
            "--loo-mean-weight", str(config["loo_mean_weight"]),
            "--tail-weight", str(config["tail_weight"]),
            "--tail-temperature", str(config["tail_temperature"]),
            "--topology-weight", str(config["topology_weight"]),
            "--l2sp-weight", str(config["l2sp_weight"]),
            "--margin-epsilon", str(config["margin_epsilon"]),
            "--query-package", str(row["query_package"]),
            "--package-manifest", str(row["package_manifest"]),
            "--validated-row-manifest", str(row["validated_row_manifest"]),
            "--row-binding", str(row["row_binding"]),
        ]
        score_command = (
            str(scorer), "--da0", str(row["baseline_prediction"]),
            "--da1", str(prediction), "--truth", str(row["truth"]),
            "--scenario", str(row["scenario"]), "--output", str(score),
        )
        tasks.append(MatrixTask(
            row_id=row_id, scenario=str(row["scenario"]),
            prediction_output=prediction, score_output=score,
            prediction_command=tuple(command), score_command=score_command,
        ))
    return tuple(tasks)


def _append_event(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def run_matrix(
    config: dict[str, Any],
    *,
    release_root: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse matrix output root: {output_root}")
    output_root.mkdir(parents=True)
    event_path = output_root / "matrix_events.jsonl"
    tasks = plan_tasks(config, release_root=release_root, output_root=output_root, device=device)
    for index, task in enumerate(tasks):
        _append_event(event_path, {"event": "TASK_START", "index": index, "row_id": task.row_id})
        for phase, command in (("prediction", task.prediction_command), ("score", task.score_command)):
            try:
                completed = subprocess.run(
                    [sys.executable, *command], check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
            except subprocess.CalledProcessError as exc:
                _append_event(event_path, {
                    "event": "TASK_FAILED", "index": index, "phase": phase,
                    "row_id": task.row_id, "returncode": exc.returncode,
                    "output": exc.stdout,
                })
                raise
            _append_event(event_path, {
                "event": "PHASE_COMPLETE", "index": index, "phase": phase,
                "row_id": task.row_id, "output": completed.stdout,
            })
        if not task.prediction_output.is_file() or not task.score_output.is_file():
            raise RuntimeError("matrix subprocess returned without required artifacts")
    summary = {
        "schema": SCHEMA, "status": "ARTIFACTS_COMPLETE",
        "run_id": str(config["run_id"]), "row_count": len(config["rows"]),
        "candidate_count": 1, "prediction_count": len(tasks),
        "score_count": len(tasks), "failed_count": 0, "device": str(device),
    }
    with (output_root / "matrix_summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_matrix_config(args.config)
    result = run_matrix(
        config, release_root=Path(args.release_root),
        output_root=Path(args.output_root), device=str(args.device),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

