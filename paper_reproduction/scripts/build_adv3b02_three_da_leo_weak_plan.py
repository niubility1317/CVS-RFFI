"""Build the sealed-cache and 375-row ADV3B02 Stage2-B DA execution plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
METHODS = ("protonet_cda", "mrior_sda", "dadda_sda")
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713101, 713102, 713103, 713104, 713105)
K_VALUES = (1, 2, 5, 10, 20)
POLICY = "leo_weak_only_no_clean_access"
BUILD_SPEC_SCHEMA = "cvs_leo_weak_iq_cache_build_spec_v1"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _cache_spec(
    *, cache_set_id: str, cache_scope: str, role_specs: list[dict[str, Any]],
    dataset_seed: int, satellite_seeds: dict[str, int], out_root: str,
) -> dict[str, Any]:
    root = PurePosixPath(out_root)
    return {
        "schema": BUILD_SPEC_SCHEMA,
        "cache_set_id": cache_set_id,
        "cache_scope": cache_scope,
        "phase2_sample_view_policy": POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "star_ground_channel_impl": "simplified_leo_residual",
        "role_specs": role_specs,
        "dataset_seed": int(dataset_seed),
        "satellite_seed_by_scenario": satellite_seeds,
        "out_npz_by_scenario": {
            scenario: str(root / f"{scenario}.npz") for scenario in SCENARIOS
        },
        "out_manifest": str(root / "cache_set.json"),
        "batch_size": 256,
        "wisig_out_len": 256,
        "wisig_equalized": "1",
        "wisig_domain": "rx_day",
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    base = json.loads(args.base_config.read_text(encoding="utf-8-sig"))
    old_labels = [str(value) for value in base["target_old_tx_labels"]]
    source_receivers = [str(value) for value in base["source_receiver_labels"]]
    if tuple(base["publication_target_receiver_grid"]) != RECEIVERS:
        raise ValueError("base config receiver grid drift")
    run_root = PurePosixPath(args.runtime_run_root)
    plan_root = PurePosixPath(args.runtime_plan_dir)
    source_root = run_root / "phase1_caches/source"
    target_root = run_root / "phase1_caches/target"
    predictor_root = run_root / "phase2_predictor_packages"
    scorer_root = run_root / "phase2_scoring_sidecars"
    predictor_seal_root = run_root / "phase2_predictor_seals"
    seal_root = run_root / "runtime_seal"
    phase2_config = dict(base)
    phase2_config.pop("adv3b02_checkpoint", None)
    phase2_config.update({
        "source_leo_weak_cache_set_manifest": str(source_root / "cache_set.json"),
        "target_predictor_bundle_root": str(predictor_root),
        "target_predictor_seal_root": str(predictor_seal_root),
        "phase2_runtime_isolation_evidence_root": str(seal_root),
    })
    forbidden = {
        "manysig_pkl", "manytx_pkl", "source_dataset", "target_dataset",
        "dataset_path", "source_train_channel_view", "train_channel_view",
    }
    leaked = sorted(forbidden & set(phase2_config))
    if leaked:
        raise ValueError(f"Phase2 config exposes raw/clean inputs: {leaked}")
    phase2_config_rel = "phase2_config.json"
    _write_json(output_dir / phase2_config_rel, phase2_config)

    package_artifacts = {
        "candidate_lock.json": {
            "schema": "adv3b02_three_da_stage2b_candidate_lock_v1",
            "experiment_id": str(base["experiment_id"]),
            "stage": "Stage2-B",
            "methods": list(METHODS),
            "receiver_grid": list(RECEIVERS),
            "seed_grid": list(SEEDS),
            "k_grid": list(K_VALUES),
            "phase2_sample_view_policy": POLICY,
            "query_decision_policy": "per_sample_all_registered_classes",
        },
        "adapter.json": {
            "schema": "adv3b02_online_da_runtime_adapter_v1",
            "pretrained_adapter_loaded": False,
            "support_only_online_optimization": True,
            "methods": list(METHODS),
        },
        "head.json": {
            "schema": "adv3b02_native_or_support_prototype_head_v1",
            "query_truth_access": False,
            "query_role_access": False,
            "query_batch_quota_access": False,
        },
        "tta_policy.json": {"mode": "single_view", "views": 1},
    }
    package_artifact_rel: dict[str, str] = {}
    for filename, payload in package_artifacts.items():
        rel = f"package_artifacts/{filename}"
        _write_json(output_dir / rel, payload)
        package_artifact_rel[filename] = rel

    cache_specs: list[dict[str, Any]] = []
    cache_commands: list[list[str]] = []
    source_spec = _cache_spec(
        cache_set_id=f"{base['experiment_id']}_source",
        cache_scope="source_train",
        role_specs=[{
            "role": "source",
            "pkl": args.source_dataset,
            "tx_ids": ",".join(old_labels),
            "rxs": ",".join(source_receivers),
            "days": "0,1",
            "max_samples_per_combo": 100,
        }],
        dataset_seed=713101,
        satellite_seeds={scenario: 913100 + index for index, scenario in enumerate(SCENARIOS)},
        out_root=str(source_root),
    )
    source_rel = "cache_specs/source.json"
    _write_json(output_dir / source_rel, source_spec)
    cache_specs.append({"scope": "source_train", "spec": source_rel})
    cache_commands.append([
        "python", "code/scripts/build_cvs_leo_weak_iq_cache.py",
        "--spec", str(plan_root / source_rel), "--device", str(args.cache_device),
    ])

    target_contracts: list[dict[str, Any]] = []
    bundle_commands: list[list[str]] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            receiver_root = target_root / f"rx_{_safe(receiver)}" / f"seed_{seed}"
            satellite_seeds = {
                scenario: int(seed * 10 + index) for index, scenario in enumerate(SCENARIOS)
            }
            spec = _cache_spec(
                cache_set_id=f"{base['experiment_id']}_{receiver}_{seed}",
                cache_scope="stage2_target_old",
                role_specs=[{
                    "role": "target_old",
                    "pkl": args.target_dataset,
                    "tx_ids": ",".join(old_labels),
                    "rxs": receiver,
                    "days": "0",
                    "max_samples_per_tx": int(base["support_pool_max_k"])
                    + int(base["query_per_tx"]),
                }],
                dataset_seed=int(seed),
                satellite_seeds=satellite_seeds,
                out_root=str(receiver_root),
            )
            rel = f"cache_specs/target/rx_{_safe(receiver)}/seed_{seed}.json"
            _write_json(output_dir / rel, spec)
            cache_specs.append({"scope": "stage2_target_old", "spec": rel})
            cache_commands.append([
                "python", "code/scripts/build_cvs_leo_weak_iq_cache.py",
                "--spec", str(plan_root / rel), "--device", str(args.cache_device),
            ])
            target_contracts.append({
                "receiver": receiver,
                "seed": seed,
                "cache_set_manifest": str(receiver_root / "cache_set.json"),
                "satellite_seed_by_scenario": satellite_seeds,
            })
            bundle_root = predictor_root / f"rx_{_safe(receiver)}" / f"seed_{seed}"
            scoring_root = scorer_root / f"rx_{_safe(receiver)}" / f"seed_{seed}"
            detached_seal = (
                predictor_seal_root / f"rx_{_safe(receiver)}" / f"seed_{seed}" / "seal.json"
            )
            bundle_commands.append([
                "python", "code/scripts/build_cvs_stage2_predictor_bundle.py",
                "--target-cache-set", str(receiver_root / "cache_set.json"),
                "--expected-cache-scope", "stage2_target_old",
                "--predictor-out-root", str(bundle_root),
                "--scorer-out-root", str(scoring_root),
                "--detached-seal-path", str(detached_seal),
                "--stage", "stage2b",
                "--receiver", receiver,
                "--seed", str(seed),
                "--old-class-labels", ",".join(old_labels),
                "--new-class-count", "0",
                "--support-pool-max-k", str(base["support_pool_max_k"]),
                "--query-per-tx", str(base["query_per_tx"]),
                "--candidate-lock", str(
                    plan_root / package_artifact_rel["candidate_lock.json"]
                ),
                "--checkpoint", str(base["adv3b02_checkpoint"]),
                "--adapter", str(plan_root / package_artifact_rel["adapter.json"]),
                "--head-artifact", str(plan_root / package_artifact_rel["head.json"]),
                "--tta-policy-json", str(
                    plan_root / package_artifact_rel["tta_policy.json"]
                ),
            ])

    project_root = PurePosixPath(args.runtime_project_root)
    runtime_code_files = (
        "code/scripts/run_phase2_landlock_isolated.py",
        "code/cvsrffi/__init__.py",
        "code/cvsrffi/leo_weak_cache.py",
        "code/cvsrffi/phase2_runtime_contract.py",
        "code/cvsrffi/stage2_predictor_bundle.py",
        "code/baseline_origin_sat_view.py",
        "code/model.py",
        "code/model_dual_cvsincnet.py",
        "baselines/__init__.py",
        "baselines/common/__init__.py",
        "baselines/common/resnet1d.py",
        "baselines/cvcnn_ce/__init__.py",
        "baselines/cvcnn_ce/model.py",
        "paper_reproduction/__init__.py",
        "paper_reproduction/common/__init__.py",
        "paper_reproduction/common/config.py",
        "paper_reproduction/cvs_aligned/__init__.py",
        "paper_reproduction/cvs_aligned/adv3b02_supervised_da_runner.py",
        "paper_reproduction/cvs_aligned/supervised_da.py",
        "paper_reproduction/DADDA/__init__.py",
        "paper_reproduction/DADDA/losses.py",
        "paper_reproduction/mitigating_receiver_impact_da/__init__.py",
        "paper_reproduction/mitigating_receiver_impact_da/losses.py",
    )
    runtime_seal_command = [
        "python", "code/scripts/build_phase2_runtime_seal.py",
        "--config", str(plan_root / phase2_config_rel),
        "--out-root", str(seal_root),
    ]
    for relative_path in runtime_code_files:
        runtime_seal_command.extend([
            "--runtime-code-file", str(project_root / relative_path)
        ])

    output_root = run_root / "stage2_runs"
    log_root = run_root / "stage2_logs"
    worker_commands: list[list[str]] = []
    for shard_index in range(int(args.shard_count)):
        device = f"cuda:{shard_index % int(args.gpu_count)}"
        worker_commands.append([
            "python", "-u", "paper_reproduction/scripts/run_cvs_publication_matrix.py",
            "--phase", "stage2b",
            "--config", str(plan_root / phase2_config_rel),
            "--output-root", str(output_root),
            "--log-root", str(log_root),
            "--methods", ",".join(METHODS),
            "--module-override", "paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner",
            "--shard-count", str(args.shard_count),
            "--shard-index", str(shard_index),
            "--device", device,
            "--post-prediction-scorer",
            "paper_reproduction/scripts/score_adv3b02_three_da_predictions.py",
            "--scoring-root", str(scorer_root),
            "--isolation-launcher", "code/scripts/run_phase2_landlock_isolated.py",
            "--runtime-allowlist", str(seal_root / "artifact_member_allowlist.json"),
            "--runtime-evidence-root", str(seal_root),
            "--isolation-runtime-read-dir",
            "/home/szu2070436088/.conda/envs/CVS-RFFI",
            "--execute",
        ])
    manifest = {
        "schema": "adv3b02_three_da_leo_weak_only_plan_v1",
        "experiment_id": base["experiment_id"],
        "phase": "Stage2-B",
        "phase2_sample_view_policy": POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "methods": list(METHODS),
        "receivers": list(RECEIVERS),
        "seeds": list(SEEDS),
        "k_values": list(K_VALUES),
        "formal_method_rows": len(METHODS) * len(RECEIVERS) * len(SEEDS) * len(K_VALUES),
        "rows_per_method": len(RECEIVERS) * len(SEEDS) * len(K_VALUES),
        "phase1_offline_cache_build_count": len(cache_commands),
        "phase1_offline_predictor_bundle_build_count": len(bundle_commands),
        "phase2_runtime_seal_build_count": 1,
        "offline_preparation_task_count": len(cache_commands) + len(bundle_commands) + 1,
        "phase2_config": str(plan_root / phase2_config_rel),
        "phase2_config_sha256": _sha256(output_dir / phase2_config_rel),
        "phase2_config_exposes_dataset_path": False,
        "cache_specs": cache_specs,
        "package_artifacts": package_artifact_rel,
        "target_cache_contracts": target_contracts,
        "commands": {
            "phase1_offline_cache_build": cache_commands,
            "phase1_offline_predictor_bundle_build": bundle_commands,
            "phase2_runtime_seal": runtime_seal_command,
            "phase2_workers": worker_commands,
        },
        "runtime_plan_dir": str(plan_root),
        "runtime_run_root": str(run_root),
    }
    _write_json(output_dir / "plan_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-plan-dir", required=True)
    parser.add_argument("--runtime-run-root", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--cache-device", default="cuda:0")
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument(
        "--runtime-project-root",
        default="/home/szu2070436088/2510044040/CV-SincNet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shard_count <= 0 or args.gpu_count <= 0:
        raise ValueError("shard-count and gpu-count must be positive")
    manifest = build_plan(args)
    print(json.dumps({
        "plan_manifest": str((args.output_dir / "plan_manifest.json").resolve()),
        "formal_method_rows": manifest["formal_method_rows"],
        "phase1_offline_cache_build_count": manifest["phase1_offline_cache_build_count"],
        "phase1_offline_predictor_bundle_build_count": manifest[
            "phase1_offline_predictor_bundle_build_count"
        ],
        "offline_preparation_task_count": manifest["offline_preparation_task_count"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
