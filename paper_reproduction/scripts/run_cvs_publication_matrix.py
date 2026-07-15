from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (713101, 713102, 713103, 713104, 713105)
DEFAULT_K = (1, 2, 5, 10, 20)
DEFAULT_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
PHASE_METHODS = {
    "stage2b": ("cvs_opgac", "protonet_cda", "mrior_sda", "dadda_sda"),
    "stage2c": ("cvs_qknnv42", "csil", "mopc_hr", "orthogonal_incremental"),
}
MODULES = {
    "stage2b": "paper_reproduction.cvs_aligned.supervised_da_runner",
    "stage2c": "paper_reproduction.cvs_aligned.class_incremental",
}
EXPECTED_QUERY_TX = {"stage2b": 6, "stage2c": 8}


@dataclass(frozen=True)
class MatrixRow:
    index: int
    phase: str
    method: str
    receiver: str
    k_shot: int
    seed: int
    split_seed: int
    experiment_id: str
    run_dir: str
    log_path: str


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer grids must contain positive values")
    return values


def _assert_cvs_config_uses_independent_query_decisions(path: Path | None) -> None:
    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("launchable") is False:
        raise ValueError(f"CVS config is not launchable: {payload.get('protocol_status', path)}")
    if str(payload.get("qknnv42_decision_mode", "per_sample_argmax")) != "per_sample_argmax":
        raise ValueError("publication matrix prohibits role Oracle and class-quota decisions")
    adapter = str(payload.get("qknnv42_feature_adapter_mode", ""))
    if adapter.startswith("support_role_"):
        raise ValueError("publication matrix prohibits query-role partition adapters")


def _parse_strings(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    if not values:
        raise ValueError("string grids must not be empty")
    return values


def _matrix_manifest_path(
    output_root: Path,
    *,
    phase: str,
    methods: tuple[str, ...],
    receivers: tuple[str, ...],
    k_grid: tuple[int, ...],
    seeds: tuple[int, ...],
) -> Path:
    canonical = (
        methods == PHASE_METHODS[phase]
        and receivers == DEFAULT_RECEIVERS
        and k_grid == DEFAULT_K
        and seeds == DEFAULT_SEEDS
    )
    if canonical:
        return output_root / "matrix_manifest.json"
    payload = json.dumps(
        {
            "phase": phase,
            "methods": methods,
            "receivers": receivers,
            "k_grid": k_grid,
            "seeds": seeds,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return output_root / f"matrix_manifest_subset_{digest}.json"


def build_rows(
    *,
    phase: str,
    methods: tuple[str, ...],
    receivers: tuple[str, ...],
    k_grid: tuple[int, ...],
    seeds: tuple[int, ...],
    output_root: Path,
    log_root: Path,
) -> list[MatrixRow]:
    allowed = set(PHASE_METHODS[phase])
    if not methods or not set(methods) <= allowed:
        raise ValueError(f"methods for {phase} must be a non-empty subset of {sorted(allowed)}")
    rows: list[MatrixRow] = []
    for receiver in receivers:
        for seed in seeds:
            for k_shot in k_grid:
                for method in methods:
                    experiment_id = f"{method}_{phase}_rx{receiver}_k{k_shot}_seed{seed}"
                    run_dir = output_root / f"rx_{receiver}" / f"seed_{seed}" / f"k_{k_shot}" / method
                    log_path = log_root / f"rx_{receiver}" / f"seed_{seed}" / f"k_{k_shot}" / f"{method}.log"
                    rows.append(
                        MatrixRow(
                            index=len(rows),
                            phase=phase,
                            method=method,
                            receiver=receiver,
                            k_shot=k_shot,
                            seed=seed,
                            split_seed=seed,
                            experiment_id=experiment_id,
                            run_dir=str(run_dir),
                            log_path=str(log_path),
                        )
                    )
    return rows


def _artifact_status(row: MatrixRow, *, scenarios: int = 3, query_per_tx: int = 20) -> dict[str, Any]:
    run_dir = Path(row.run_dir)
    expected = (
        "metrics.json",
        "split_manifest.json",
        "resolved_config.json",
        "score_table.csv",
        "detailed_metrics.json",
        "detailed_metrics.csv",
        "loss_trace.json",
        "loss_trace.csv",
    )
    missing = [name for name in expected if not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0]
    if missing:
        return {"complete": False, "reason": "missing_artifacts", "missing": missing}
    try:
        manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        with (run_dir / "score_table.csv").open("r", encoding="utf-8", newline="") as handle:
            score_count = sum(1 for _ in csv.DictReader(handle))
        with (run_dir / "detailed_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            detailed = list(csv.DictReader(handle))
        with (run_dir / "loss_trace.csv").open("r", encoding="utf-8", newline="") as handle:
            trace = list(csv.DictReader(handle))
    except Exception as exc:
        return {"complete": False, "reason": "artifact_parse_error", "error": repr(exc)}
    expected_scores = EXPECTED_QUERY_TX[row.phase] * int(query_per_tx) * int(scenarios)
    group_types = {entry.get("group_type", "") for entry in detailed}
    required_groups = {
        "per_receiver",
        "per_transmitter",
        "per_receiver_transmitter",
        "per_receiver_transmitter_day",
    }
    errors: list[str] = []
    if bool(manifest.get("support_query_overlap", True)):
        errors.append("support_query_overlap")
    if manifest.get("all_tests_satellite_augmented") is not True:
        errors.append("tests_not_all_satellite")
    if row.method == "cvs_qknnv42":
        if manifest.get("qknnv42_decision_mode") != "per_sample_argmax":
            errors.append("non_independent_query_decision")
        if manifest.get("non_deployment_oracle_diagnostic") is not False:
            errors.append("oracle_diagnostic_status_missing_or_true")
        if manifest.get("query_used_for_joint_decision") is not False:
            errors.append("query_used_for_joint_decision_missing_or_true")
        adapter = resolved.get("qknnv42_feature_adapter_mode")
        if adapter is None:
            errors.append("feature_adapter_mode_missing")
        elif str(adapter).startswith("support_role_"):
            errors.append("query_role_partition_adapter")
        scenario_metrics = metrics.get("metrics_by_scenario")
        if not isinstance(scenario_metrics, dict) or len(scenario_metrics) != int(scenarios):
            errors.append("scenario_decision_metadata_missing_or_incomplete")
        else:
            required_false = ("role_oracle_used", "equal_class_quota_used")
            for field in required_false:
                if any(item.get(field) is not False for item in scenario_metrics.values()):
                    errors.append(f"{field}_missing_or_true")
    if score_count != expected_scores:
        errors.append(f"score_count={score_count},expected={expected_scores}")
    if not required_groups <= group_types:
        errors.append("missing_detail_group_type")
    if not trace:
        errors.append("empty_loss_trace")
    for trace_row in trace:
        try:
            value = float(trace_row["loss"])
        except Exception:
            errors.append("invalid_loss_trace")
            break
        if value != value or value in {float("inf"), float("-inf")}:
            errors.append("nonfinite_loss_trace")
            break
    return {
        "complete": not errors,
        "reason": "pass" if not errors else "artifact_contract_failure",
        "errors": errors,
        "score_count": score_count,
        "detailed_count": len(detailed),
        "loss_trace_count": len(trace),
    }


def _command(
    row: MatrixRow,
    *,
    python: str,
    config: Path,
    cvs_config: Path | None = None,
    module_override: str | None = None,
) -> list[str]:
    module = module_override or (
        "paper_reproduction.cvs_aligned.cvs_method_runner"
        if row.method in {"cvs_opgac", "cvs_qknnv42"}
        else MODULES[row.phase]
    )
    selected_config = cvs_config if row.method in {"cvs_opgac", "cvs_qknnv42"} else config
    if selected_config is None:
        raise ValueError("--cvs-config is required when the matrix includes a proposed CVS method")
    command = [
        python,
        "-u",
        "-m",
        module,
        "--config",
        str(selected_config),
        "--run-dir",
        row.run_dir,
        "--device",
        "cuda:0",
        "--experiment-id",
        row.experiment_id,
        "--method",
        row.method,
        "--target-receiver",
        row.receiver,
        "--seed",
        str(row.seed),
        "--split-seed",
        str(row.split_seed),
        "--k-shot",
        str(row.k_shot),
    ]
    return command


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume-safe CVS Stage2 publication matrix worker")
    parser.add_argument("--phase", choices=sorted(PHASE_METHODS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cvs-config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--receivers", default=",".join(DEFAULT_RECEIVERS))
    parser.add_argument("--k-grid", default=",".join(str(value) for value in DEFAULT_K))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--module-override",
        default=None,
        help="Run every selected row through this module; used for shared-backbone controlled extensions.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    _assert_cvs_config_uses_independent_query_decisions(args.cvs_config)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0,shard_count)")
    methods = _parse_strings(args.methods) if args.methods else PHASE_METHODS[args.phase]
    receivers = _parse_strings(args.receivers)
    k_grid = _parse_ints(args.k_grid)
    seeds = _parse_ints(args.seeds)
    rows = build_rows(
        phase=args.phase,
        methods=methods,
        receivers=receivers,
        k_grid=k_grid,
        seeds=seeds,
        output_root=args.output_root,
        log_root=args.log_root,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = _matrix_manifest_path(
        args.output_root,
        phase=args.phase,
        methods=methods,
        receivers=receivers,
        k_grid=k_grid,
        seeds=seeds,
    )
    if args.shard_index == 0:
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": "cvs_publication_stage2_matrix_v1",
                    "phase": args.phase,
                    "methods": list(methods),
                    "row_count": len(rows),
                    "rows": [asdict(row) for row in rows],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    shard_rows = [row for row in rows if row.index % args.shard_count == args.shard_index]
    if args.max_rows > 0:
        shard_rows = shard_rows[: args.max_rows]
    if not args.execute:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "total_rows": len(rows),
                    "shard_rows": len(shard_rows),
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "manifest": str(manifest_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    event_path = args.log_root / f"worker_{args.shard_index}_events.jsonl"
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    for row in shard_rows:
        status = _artifact_status(row)
        if status["complete"]:
            counts["skipped"] += 1
            _append_event(event_path, {"event": "skip_complete", "row": asdict(row), "status": status})
            continue
        Path(row.run_dir).mkdir(parents=True, exist_ok=True)
        Path(row.log_path).parent.mkdir(parents=True, exist_ok=True)
        command = _command(
            row,
            python=args.python,
            config=args.config,
            cvs_config=args.cvs_config,
            module_override=args.module_override,
        )
        started = time.time()
        _append_event(event_path, {"event": "start", "row": asdict(row), "command": command, "time": started})
        with Path(row.log_path).open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        final_status = _artifact_status(row)
        if completed.returncode == 0 and final_status["complete"]:
            counts["completed"] += 1
            event = "complete"
        else:
            counts["failed"] += 1
            event = "failed"
        _append_event(
            event_path,
            {
                "event": event,
                "row": asdict(row),
                "returncode": completed.returncode,
                "elapsed_sec": time.time() - started,
                "status": final_status,
            },
        )
        if event == "failed":
            break
    summary = {
        "schema": "cvs_publication_stage2_worker_summary_v1",
        "phase": args.phase,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_rows": len(shard_rows),
        **counts,
    }
    (args.log_root / f"worker_{args.shard_index}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
