#!/usr/bin/env python3
"""Run the 1000-task single-view baseline plus 289,685-param TTA5 matrix."""

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


EPOCHS = (1, 2, 5, 10, 20, 30, 60)
K_GRID = (1, 2, 5, 10, 20)
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713101, 713102, 713103, 713104, 713105)
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
PHASE2_VIEW_POLICY = "leo_weak_only_no_clean_access"
ADAPTER_METHOD = "support_only_id_norm_late_feature_tta5_v1"
EXPECTED_EVAL_FILES = (
    "metrics.json", "split_manifest.json", "resolved_config.json", "score_table.csv",
    "detailed_metrics.json", "detailed_metrics.csv", "loss_trace.json", "loss_trace.csv",
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_base_config(config: dict[str, Any]) -> None:
    required = {
        "method": "cvs_qknnv42",
        "stage": "Stage2-C",
        "qknnv42_decision_mode": "per_sample_argmax",
        "qknnv42_labelprop_mode": "disabled",
        "non_deployment_oracle_diagnostic": False,
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
        "qknnv42_aux_feature_key": "fft_logmag_features",
        "qknnv42_aux_feature_dim": 96,
        "qknnv42_expected_tta_view_count": 1,
        "phase2_sample_view_policy": PHASE2_VIEW_POLICY,
        "clean_sample_access": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"invalid base config {key}={config.get(key)!r}; expected {expected!r}")
    if tuple(config.get("feature_npz_by_scenario", {})) != SCENARIOS:
        raise ValueError(f"feature_npz_by_scenario must be ordered as {SCENARIOS}")
    if len(config.get("target_old_tx_labels", [])) != 6 or len(config.get("target_new_tx_labels", [])) != 2:
        raise ValueError("matrix requires exactly six old and two registered new classes")
    if int(config.get("support_pool_max_k", -1)) != 20:
        raise ValueError("matrix requires support_pool_max_k=20")


def build_tasks(*, adapter_root: Path, output_root: Path, log_root: Path) -> list[Task]:
    tasks: list[Task] = []
    for epochs in (0,) + EPOCHS:
        arm = "singlehead_fft96" if epochs == 0 else f"E{epochs}_idnorm_tta5"
        for receiver in RECEIVERS:
            for seed in SEEDS:
                for k_shot in K_GRID:
                    task_id = f"{arm}_rx_{receiver}_seed_{seed}_k_{k_shot}"
                    adapter_name = (
                        "" if epochs == 0 else
                        f"idnorm_tta5_rx_{receiver}_new_2_seed_{seed}_k_{k_shot}_e_{epochs}"
                    )
                    tasks.append(
                        Task(
                            index=len(tasks), arm=arm, receiver=receiver, seed=seed,
                            k_shot=k_shot, epochs=epochs, task_id=task_id,
                            adapter_run_dir=str(adapter_root / adapter_name) if adapter_name else "",
                            eval_run_dir=str(output_root / arm / f"rx_{receiver}" / f"seed_{seed}" / f"k_{k_shot}" / "cvs_qknnv42"),
                            log_path=str(log_root / arm / f"{task_id}.log"),
                        )
                    )
    if len(tasks) != 1000:
        raise AssertionError(f"expected 1000 tasks, built {len(tasks)}")
    return tasks


def _validate_training(task: Task) -> Path:
    run_dir = Path(task.adapter_run_dir)
    manifest_path = run_dir / "training_manifest.json"
    resolved_path = run_dir / "resolved_qknn_config.json"
    trace_path = run_dir / "loss_trace.json"
    delta_path = run_dir / "adapter_delta_fp16.pt"
    required_paths = (manifest_path, resolved_path, trace_path, delta_path)
    if not all(path.is_file() and path.stat().st_size > 0 for path in required_paths):
        raise ValueError(f"incomplete id_norm adapter artifacts: {run_dir}")
    manifest = _read_json(manifest_path)
    resolved = _read_json(resolved_path)
    trace = _read_json(trace_path)
    contract = manifest.get("optimizer_sample_contract", {})
    required = {
        "method": ADAPTER_METHOD,
        "adapter_selection_mode": "id_norm_late_feature",
        "support_only": True,
        "query_update_forbidden": True,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "query_view_count": 5,
        "epochs": task.epochs,
        "receiver": task.receiver,
        "seed": task.seed,
        "k_shot": task.k_shot,
        "new_count": 2,
        "phase2_sample_view_policy": PHASE2_VIEW_POLICY,
        "clean_sample_access": False,
        "resource_tier": "non_extreme_light_large_adapter_diagnostic",
        "diagnostic_only": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"invalid adapter manifest {task.task_id}: {key}")
    for key in ("clean_samples_used", "source_samples_used", "proxy_samples_used", "query_samples_used"):
        if contract.get(key) is not False:
            raise ValueError(f"forbidden optimizer sample in {task.task_id}: {key}")
    if contract.get("roles") != ["target_old_support", "target_new_support"]:
        raise ValueError(f"invalid optimizer roles in {task.task_id}")
    resources = manifest.get("resources", {})
    if int(resources.get("trainable_parameters", -1)) != 289_685:
        raise ValueError(f"adapter is not exactly 289,685 parameters: {task.task_id}")
    if int(resources.get("original_checkpoint_trainable_parameters", -1)) != 289_685:
        raise ValueError(f"checkpoint update set is not exactly 289,685: {task.task_id}")
    if int(resources.get("adapter_state_bytes_fp16", -1)) != 579_370:
        raise ValueError(f"invalid FP16 delta byte count: {task.task_id}")
    if int(resources.get("backbone_forward_count_per_query", -1)) != 5:
        raise ValueError(f"adapted query is not five-view: {task.task_id}")
    audit = manifest.get("checkpoint_load_audit", {})
    for singular, plural, fallback in (
        ("missing_key_count", "missing_keys", "missing_keys"),
        ("unexpected_key_count", "unexpected_keys", "unexpected_keys"),
        ("shape_mismatch_count", "shape_mismatches", "skipped_mismatch"),
    ):
        value = audit.get(singular, audit.get(plural, audit.get(fallback, 0)))
        count = len(value) if isinstance(value, (list, tuple, dict)) else int(value)
        if count != 0:
            raise ValueError(f"non-strict ADV3B02 load in {task.task_id}: {plural}={count}")
    if len(trace) != task.epochs or any(int(row.get("epoch", -1)) != index for index, row in enumerate(trace, 1)):
        raise ValueError(f"incomplete full loss trace in {task.task_id}")
    if resolved.get("qknnv42_expected_tta_view_count") != 5:
        raise ValueError(f"resolved config is not TTA5: {task.task_id}")
    return resolved_path


def _validate_evaluation(task: Task) -> bool:
    run_dir = Path(task.eval_run_dir)
    if not all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in EXPECTED_EVAL_FILES):
        return False
    manifest = _read_json(run_dir / "split_manifest.json")
    expected_views = 1 if task.epochs == 0 else 5
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
        "satellite_tta_view_count": expected_views,
        "phase2_sample_view_policy": PHASE2_VIEW_POLICY,
        "clean_sample_access": False,
        "resource_diagnostic_only": task.epochs > 0,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"invalid evaluation manifest {task.task_id}: {key}")
    metrics = _read_json(run_dir / "metrics.json")
    for scenario in SCENARIOS:
        detail = metrics["metrics_by_scenario"][scenario]
        for key in (
            "feature_adapter_uses_query", "query_labels_used_for_adaptation", "query_query_graph_used",
            "query_batch_state_required", "role_oracle_used", "equal_class_quota_used",
            "decision_batch_state_required",
        ):
            if detail.get(key) is not False:
                raise ValueError(f"forbidden query/Oracle state {task.task_id}:{scenario}:{key}")
        if task.epochs and int(detail.get("post_feature_adapter_parameter_count", -1)) != 289_685:
            raise ValueError(f"evaluation lost exact adapter provenance: {task.task_id}:{scenario}")
    if task.epochs:
        _validate_training(task)
    return True


def _run(command: Sequence[str], *, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("[COMMAND] " + json.dumps(list(command), ensure_ascii=False) + "\n")
        handle.flush()
        subprocess.run(list(command), check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)


def execute_task(
    task: Task, *, base_config: Path, checkpoint: Path, adapter_root: Path,
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
                    sys.executable, "-u", "-m",
                    "paper_reproduction.scripts.train_export_cvs_idnorm_tta5_adapter",
                    "--config", str(base_config), "--ckpt", str(checkpoint),
                    "--out_root", str(adapter_root), "--receiver", task.receiver,
                    "--new_count", "2", "--seed", str(task.seed),
                    "--k_shot", str(task.k_shot), "--epochs", str(task.epochs),
                    "--device", device,
                ],
                log_path=log_path, env=env,
            )
            resolved_config = _validate_training(task)
    _run(
        [
            sys.executable, "-u", "-m", "paper_reproduction.cvs_aligned.cvs_method_runner",
            "--config", str(resolved_config), "--run-dir", task.eval_run_dir,
            "--method", "cvs_qknnv42", "--target-receiver", task.receiver,
            "--seed", str(task.seed), "--split-seed", str(task.seed),
            "--k-shot", str(task.k_shot), "--experiment-id", task.task_id,
            "--device", "cpu",
        ],
        log_path=log_path, env=env,
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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = _read_json(args.config)
    validate_base_config(config)
    tasks = build_tasks(adapter_root=args.adapter_root, output_root=args.output_root, log_root=args.log_root)
    manifest_payload = {
        "protocol": "task_specific_support_only_289685_idnorm_tta5_qknn_fft96_v1",
        "task_count": len(tasks),
        "baseline_tasks": sum(task.epochs == 0 for task in tasks),
        "adaptation_tasks": sum(task.epochs > 0 for task in tasks),
        "receivers": list(RECEIVERS), "seeds": list(SEEDS), "k_grid": list(K_GRID),
        "epoch_grid": list(EPOCHS), "old_class_count": 6, "new_class_count": 2,
        "exact_trainable_parameters_per_adapter": 289_685,
        "optimizer_samples": "task-specific target receiver LEO support only",
        "phase2_sample_view_policy": PHASE2_VIEW_POLICY,
        "clean_samples_used": False, "source_samples_used": False,
        "proxy_samples_used": False, "query_samples_used_for_fit_or_selection": False,
        "dense_query_used": False, "role_or_class_quota_oracle_used": False,
        "baseline_query_views": 1, "adapted_query_views": 5,
        "resource_claim": "non_extreme_light_large_adapter_diagnostic",
        "tasks": [asdict(task) for task in tasks],
    }
    if args.manifest.exists():
        if _read_json(args.manifest) != manifest_payload:
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
                task, base_config=args.config, checkpoint=args.ckpt,
                adapter_root=args.adapter_root, device=str(args.device),
            )
            completed += status == "completed"
            skipped += status == "skipped_complete"
            event = {"task_id": task.task_id, "status": status, "elapsed_seconds": time.time() - started, "timestamp": time.time()}
        except Exception as exc:
            event = {"task_id": task.task_id, "status": "failed", "error": repr(exc), "elapsed_seconds": time.time() - started, "timestamp": time.time()}
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print("[TASK-FAILED] " + json.dumps(event, sort_keys=True), flush=True)
            raise
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        print("[TASK-DONE] " + json.dumps(event, sort_keys=True), flush=True)
    print(json.dumps({"shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "task_count": len(shard_tasks), "completed": completed, "skipped": skipped}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
