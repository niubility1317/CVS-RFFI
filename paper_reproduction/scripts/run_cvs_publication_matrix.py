from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import re
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
    if str(payload.get("qknnv42_labelprop_mode", "disabled")) == "dense_transductive":
        raise ValueError("publication matrix prohibits dense query-query graph inference")
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
        "prediction_manifest.json",
        "prediction_artifact.npz",
        "scoring_audit.json",
        "runtime_isolation_evidence.json",
        "filesystem_access_audit.json",
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
    required_protocol = {
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "target_channel_view": "leo_weak_only",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
    }
    for field, expected in required_protocol.items():
        if manifest.get(field) != expected:
            errors.append(f"protocol_guard:{field}")
    if manifest.get("overlay_applied_before_phase2") is not True:
        errors.append("overlay_not_proven_before_phase2")
    for field in (
        "predictor_query_truth_access", "predictor_query_role_access",
        "predictor_query_true_batch_class_count_access", "predictor_query_class_quota_access",
    ):
        if manifest.get(field) is not False:
            errors.append(f"predictor_access_guard:{field}")
    if manifest.get("prediction_scoring_process_isolated") is not True:
        errors.append("predict_score_process_not_isolated")
    runtime_evidence = manifest.get("phase2_runtime_isolation_evidence", {})
    if runtime_evidence.get("os_isolation_mode") != "equivalent_verified_isolation":
        errors.append("os_isolation_evidence_missing")
    if runtime_evidence.get("filesystem_access_audit_status") != "PASS":
        errors.append("actual_filesystem_access_audit_failed_or_missing")
    forbidden_config = {
        "manysig_pkl", "manytx_pkl", "dataset_path", "source_dataset",
        "target_dataset", "source_train_channel_view", "train_channel_view",
        "adv3b02_checkpoint",
    }
    leaked = sorted(forbidden_config & set(resolved))
    if leaked:
        errors.append(f"raw_or_clean_config_keys={leaked}")
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
        if manifest.get("qknnv42_labelprop_mode") == "dense_transductive":
            errors.append("dense_query_graph_inference")
        if manifest.get("query_used_for_transductive_inference") is not False:
            errors.append("query_used_for_transductive_inference_missing_or_true")
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
            for field in ("query_query_graph_used", "query_batch_state_required"):
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
    device: str = "cuda:0",
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
        str(device),
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


def _safe_receiver(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finalize_runtime_evidence(
    row: MatrixRow, *, trace_prefix: Path, base_evidence_path: Path
) -> Path:
    trace_files = sorted(trace_prefix.parent.glob(trace_prefix.name + "*"))
    if not trace_files:
        raise RuntimeError("strace did not emit a filesystem access ledger")
    forbidden_tokens = (".pkl", "truth_sidecar", "scoring_manifest", "manysig", "manytx")
    forbidden_hits: list[str] = []
    trace_entries: list[dict[str, Any]] = []
    for path in trace_files:
        content = path.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            lowered = line.lower()
            if any(token in lowered for token in forbidden_tokens):
                forbidden_hits.append(line[:500])
        trace_entries.append({"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size})
    audit = {
        "schema": "cvs_phase2_actual_filesystem_access_audit_v1",
        "status": "PASS" if not forbidden_hits else "FAIL",
        "landlock_enforced": True,
        "trace_files": trace_entries,
        "forbidden_access_hits": forbidden_hits,
    }
    audit_path = Path(row.run_dir) / "filesystem_access_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if forbidden_hits:
        raise RuntimeError(f"predictor attempted forbidden filesystem access: {forbidden_hits[:3]}")
    evidence = json.loads(base_evidence_path.read_text(encoding="utf-8-sig"))
    evidence.update({
        "filesystem_access_audit_sha256": _sha256(audit_path),
        "filesystem_access_audit_status": "PASS",
        "prediction_artifact_sha256": _sha256(
            Path(row.run_dir) / "prediction_artifact.npz"
        ),
        "prediction_seal_sha256": _sha256(
            Path(row.run_dir) / "prediction_manifest.json"
        ),
    })
    evidence_path = Path(row.run_dir) / "runtime_isolation_evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = Path(row.run_dir) / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["phase2_runtime_isolation_evidence"] = evidence
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence_path


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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--module-override",
        default=None,
        help="Run every selected row through this module; used for shared-backbone controlled extensions.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--post-prediction-scorer", type=Path, default=None)
    parser.add_argument("--scoring-root", type=Path, default=None)
    parser.add_argument("--isolation-launcher", type=Path, default=None)
    parser.add_argument("--runtime-allowlist", type=Path, default=None)
    parser.add_argument("--runtime-evidence-root", type=Path, default=None)
    parser.add_argument("--isolation-runtime-read-dir", type=Path, action="append", default=[])
    parser.add_argument("--strace", default="strace")
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
            device=args.device,
        )
        isolated = all(
            value is not None for value in (
                args.post_prediction_scorer, args.scoring_root, args.isolation_launcher,
                args.runtime_allowlist, args.runtime_evidence_root,
            )
        )
        if any(value is not None for value in (
            args.post_prediction_scorer, args.scoring_root, args.isolation_launcher,
            args.runtime_allowlist, args.runtime_evidence_root,
        )) and not isolated:
            raise ValueError("isolated predictor/scorer arguments must be provided together")
        trace_prefix = Path(row.run_dir) / "fs_trace"
        predictor_command = command
        if isolated:
            predictor_command = [
                args.strace, "-ff", "-e", "trace=%file", "-o", str(trace_prefix),
                args.python, str(args.isolation_launcher),
                "--allowlist", str(args.runtime_allowlist),
                "--write-dir", row.run_dir, "--", *command,
            ]
            insert_at = predictor_command.index("--")
            for runtime_dir in args.isolation_runtime_read_dir:
                predictor_command[insert_at:insert_at] = [
                    "--runtime-read-dir", str(runtime_dir)
                ]
                insert_at += 2
        started = time.time()
        _append_event(event_path, {"event": "predictor_start", "row": asdict(row), "command": predictor_command, "time": started})
        with Path(row.log_path).open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(predictor_command, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
            scorer_returncode = None
            if completed.returncode == 0 and isolated:
                base_evidence_path = (
                    args.runtime_evidence_root / f"rx_{_safe_receiver(row.receiver)}" /
                    f"seed_{row.seed}" / "runtime_isolation_evidence.json"
                )
                evidence_path = _finalize_runtime_evidence(
                    row, trace_prefix=trace_prefix, base_evidence_path=base_evidence_path
                )
                scoring_manifest = (
                    args.scoring_root / f"rx_{_safe_receiver(row.receiver)}" /
                    f"seed_{row.seed}" / "scoring_manifest.json"
                )
                scorer_command = [
                    args.python, "-u", str(args.post_prediction_scorer),
                    "--run-dir", row.run_dir,
                    "--scoring-manifest", str(scoring_manifest),
                    "--runtime-evidence", str(evidence_path),
                ]
                _append_event(event_path, {
                    "event": "scorer_start", "row": asdict(row), "command": scorer_command,
                    "prediction_artifact_sha256": _sha256(Path(row.run_dir) / "prediction_artifact.npz"),
                })
                scorer = subprocess.run(
                    scorer_command, stdout=log_handle, stderr=subprocess.STDOUT, check=False
                )
                scorer_returncode = scorer.returncode
        final_status = _artifact_status(row)
        if completed.returncode == 0 and scorer_returncode in (None, 0) and final_status["complete"]:
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
                "scorer_returncode": scorer_returncode,
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
