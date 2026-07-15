#!/usr/bin/env python3
"""Run the 875-task qKNN support-only adaptation matrix in resume-safe shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


EPOCHS = (2, 5, 10, 20, 30, 60)
K_GRID = (1, 2, 5, 10, 20)
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713101, 713102, 713103, 713104, 713105)
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
EXPECTED_EVAL_FILES = (
    "metrics.json",
    "split_manifest.json",
    "resolved_config.json",
    "score_table.csv",
    "detailed_metrics.json",
    "detailed_metrics.csv",
    "loss_trace.json",
    "loss_trace.csv",
)


@dataclass(frozen=True)
class Task:
    index: int
    arm: str
    receiver: str
    seed: int
    k_shot: int
    epochs: int
    task_id: str
    adapter_run_dir: str
    eval_run_dir: str
    log_path: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_base_config(config: dict[str, Any], *, new_count: int) -> None:
    if config.get("method") != "cvs_qknnv42" or config.get("stage") != "Stage2-C":
        raise ValueError("base config must be a Stage2-C cvs_qknnv42 config")
    if len(config.get("target_new_tx_labels", [])) != int(new_count):
        raise ValueError("new_count must exactly match target_new_tx_labels")
    required = {
        "qknnv42_decision_mode": "per_sample_argmax",
        "qknnv42_labelprop_mode": "disabled",
        "non_deployment_oracle_diagnostic": False,
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
        "qknnv42_aux_feature_key": "fft_logmag_features",
        "qknnv42_aux_feature_dim": 96,
        "qknnv42_expected_tta_view_count": 1,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"invalid base config field {key}={config.get(key)!r}; expected {expected!r}")
    mapping = config.get("feature_npz_by_scenario", {})
    if tuple(mapping) != SCENARIOS:
        raise ValueError(f"feature_npz_by_scenario must be ordered as {SCENARIOS}")
    if int(config.get("support_pool_max_k", -1)) != 20:
        raise ValueError("875-task nested K protocol requires support_pool_max_k=20")


def build_tasks(*, adapter_root: Path, output_root: Path, log_root: Path) -> list[Task]:
    tasks: list[Task] = []
    for epochs in (0,) + EPOCHS:
        arm = "singlehead_fft96" if epochs == 0 else f"E{epochs}"
        for receiver in RECEIVERS:
            for seed in SEEDS:
                for k_shot in K_GRID:
                    task_id = f"{arm}_rx_{receiver}_seed_{seed}_k_{k_shot}"
                    adapter_name = (
                        ""
                        if epochs == 0
                        else f"micro_iq_rx_{receiver}_new_2_seed_{seed}_k_{k_shot}_e_{epochs}"
                    )
                    tasks.append(
                        Task(
                            index=len(tasks),
                            arm=arm,
                            receiver=receiver,
                            seed=seed,
                            k_shot=k_shot,
                            epochs=epochs,
                            task_id=task_id,
                            adapter_run_dir=str(adapter_root / adapter_name) if adapter_name else "",
                            eval_run_dir=str(
                                output_root
                                / arm
                                / f"rx_{receiver}"
                                / f"seed_{seed}"
                                / f"k_{k_shot}"
                                / "cvs_qknnv42"
                            ),
                            log_path=str(log_root / arm / f"{task_id}.log"),
                        )
                    )
    if len(tasks) != 875:
        raise AssertionError(f"expected 875 tasks, built {len(tasks)}")
    return tasks


def _validate_training(task: Task) -> Path:
    run_dir = Path(task.adapter_run_dir)
    manifest_path = run_dir / "training_manifest.json"
    resolved_path = run_dir / "resolved_qknn_config.json"
    trace_path = run_dir / "loss_trace.json"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (manifest_path, resolved_path, trace_path)):
        raise ValueError(f"incomplete support-only adapter artifacts: {run_dir}")
    manifest = _read_json(manifest_path)
    trace = _read_json(trace_path)
    contract = manifest.get("optimizer_sample_contract", {})
    required = {
        "support_only": True,
        "query_update_forbidden": True,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "query_view_count": 1,
        "epochs": task.epochs,
        "receiver": task.receiver,
        "seed": task.seed,
        "k_shot": task.k_shot,
        "new_count": 2,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"invalid adapter manifest {task.task_id}: {key}")
    for key in ("clean_samples_used", "source_samples_used", "proxy_samples_used", "query_samples_used"):
        if contract.get(key) is not False:
            raise ValueError(f"forbidden optimizer samples in {task.task_id}: {key}")
    if contract.get("roles") != ["target_old_support", "target_new_support"]:
        raise ValueError(f"invalid support roles in {task.task_id}")
    if len(trace) != task.epochs or any(int(row.get("epoch", -1)) != i for i, row in enumerate(trace, 1)):
        raise ValueError(f"incomplete full loss trace in {task.task_id}")
    return resolved_path


def _validate_evaluation(task: Task) -> bool:
    run_dir = Path(task.eval_run_dir)
    if not all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in EXPECTED_EVAL_FILES):
        return False
    manifest = _read_json(run_dir / "split_manifest.json")
    required = {
        "support_query_overlap": False,
        "qknnv42_decision_mode": "per_sample_argmax",
        "qknnv42_labelprop_mode": "disabled",
        "non_deployment_oracle_diagnostic": False,
        "query_used_for_joint_decision": False,
        "query_used_for_transductive_inference": False,
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
        "qknnv42_aux_feature_dim": 96,
        "satellite_tta_view_count": 1,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"invalid evaluation manifest {task.task_id}: {key}")
    for scenario in SCENARIOS:
        detail = _read_json(run_dir / "metrics.json")["metrics_by_scenario"][scenario]
        for key in (
            "feature_adapter_uses_query",
            "query_labels_used_for_adaptation",
            "query_query_graph_used",
            "query_batch_state_required",
            "role_oracle_used",
            "equal_class_quota_used",
            "decision_batch_state_required",
        ):
            if detail.get(key) is not False:
                raise ValueError(f"forbidden query/Oracle state {task.task_id}:{scenario}:{key}")
    if task.epochs:
        _validate_training(task)
    return True


def _run(command: Sequence[str], *, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("[COMMAND] " + json.dumps(list(command), ensure_ascii=False) + "\n")
        handle.flush()
        subprocess.run(
            list(command),
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


def execute_task(
    task: Task,
    *,
    base_config: Path,
    checkpoint: Path,
    adapter_root: Path,
    new_count: int,
    device: str,
) -> str:
    if _validate_evaluation(task):
        return "skipped_complete"
    eval_dir = Path(task.eval_run_dir)
    if eval_dir.exists() and any(eval_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite incomplete evaluation: {eval_dir}")
    log_path = Path(task.log_path)
    env = dict(os.environ)
    resolved_config = base_config
    if task.epochs:
        adapter_dir = Path(task.adapter_run_dir)
        if adapter_dir.exists() and any(adapter_dir.iterdir()):
            resolved_config = _validate_training(task)
        else:
            _run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "paper_reproduction.scripts.train_export_cvs_micro_iq_adapter",
                    "--config",
                    str(base_config),
                    "--ckpt",
                    str(checkpoint),
                    "--out_root",
                    str(adapter_root),
                    "--receiver",
                    task.receiver,
                    "--new_count",
                    str(new_count),
                    "--seed",
                    str(task.seed),
                    "--k_shot",
                    str(task.k_shot),
                    "--epochs",
                    str(task.epochs),
                    "--device",
                    device,
                ],
                log_path=log_path,
                env=env,
            )
            resolved_config = _validate_training(task)
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "paper_reproduction.cvs_aligned.cvs_method_runner",
            "--config",
            str(resolved_config),
            "--run-dir",
            task.eval_run_dir,
            "--method",
            "cvs_qknnv42",
            "--target-receiver",
            task.receiver,
            "--seed",
            str(task.seed),
            "--split-seed",
            str(task.seed),
            "--k-shot",
            str(task.k_shot),
            "--experiment-id",
            task.task_id,
            "--device",
            "cpu",
        ],
        log_path=log_path,
        env=env,
    )
    if not _validate_evaluation(task):
        raise RuntimeError(f"task completed without full evaluation artifacts: {task.task_id}")
    return "completed"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--new-count", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    config = _read_json(args.config)
    validate_base_config(config, new_count=int(args.new_count))
    tasks = build_tasks(
        adapter_root=args.adapter_root,
        output_root=args.output_root,
        log_root=args.log_root,
    )
    manifest_payload = {
        "protocol": "task_specific_support_only_micro_iq_adapter_qknn_fft96_v1",
        "task_count": len(tasks),
        "baseline_tasks": sum(task.epochs == 0 for task in tasks),
        "adaptation_tasks": sum(task.epochs > 0 for task in tasks),
        "receivers": list(RECEIVERS),
        "seeds": list(SEEDS),
        "k_grid": list(K_GRID),
        "epoch_grid": list(EPOCHS),
        "old_class_count": len(config["target_old_tx_labels"]),
        "new_class_count": len(config["target_new_tx_labels"]),
        "optimizer_samples": "task-specific target receiver LEO support only",
        "clean_samples_used": False,
        "source_samples_used": False,
        "proxy_samples_used": False,
        "query_samples_used_for_fit_or_selection": False,
        "dense_query_used": False,
        "role_or_class_quota_oracle_used": False,
        "tasks": [asdict(task) for task in tasks],
    }
    if args.manifest.exists():
        existing = _read_json(args.manifest)
        if existing != manifest_payload:
            raise ValueError(f"existing matrix manifest differs: {args.manifest}")
    else:
        _write_json(args.manifest, manifest_payload)
    if args.prepare_only and not args.execute:
        print(json.dumps({"manifest": str(args.manifest), "task_count": len(tasks)}, sort_keys=True))
        return 0
    if not args.execute:
        raise ValueError("select --prepare-only or --execute")
    if int(args.shard_count) <= 0 or not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid shard index/count")

    shard_tasks = [task for task in tasks if task.index % int(args.shard_count) == int(args.shard_index)]
    event_path = args.log_root / f"shard_{args.shard_index:02d}_events.jsonl"
    completed = skipped = 0
    for task in shard_tasks:
        started = time.time()
        try:
            status = execute_task(
                task,
                base_config=args.config,
                checkpoint=args.ckpt,
                adapter_root=args.adapter_root,
                new_count=int(args.new_count),
                device=str(args.device),
            )
            completed += status == "completed"
            skipped += status == "skipped_complete"
            event = {
                "task_id": task.task_id,
                "status": status,
                "elapsed_seconds": time.time() - started,
                "timestamp": time.time(),
            }
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print("[TASK-DONE] " + json.dumps(event, sort_keys=True), flush=True)
        except Exception as exc:
            event = {
                "task_id": task.task_id,
                "status": "failed",
                "error": repr(exc),
                "elapsed_seconds": time.time() - started,
                "timestamp": time.time(),
            }
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print("[TASK-FAILED] " + json.dumps(event, sort_keys=True), flush=True)
            raise
    print(
        json.dumps(
            {
                "shard_index": int(args.shard_index),
                "shard_count": int(args.shard_count),
                "task_count": len(shard_tasks),
                "completed": completed,
                "skipped": skipped,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
