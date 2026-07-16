"""Build the strict 25-row JG_R8_LR020 matched-K10 Stage2-B plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713101, 713102, 713103, 713104, 713105)
POLICY = "leo_weak_only_no_clean_access"
METHOD = "jg_r8_lr020"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _cache_spec(
    *,
    cache_set_id: str,
    receiver: str,
    old_labels: list[str],
    seed: int,
    target_dataset: str,
    out_root: str,
    support_pool_max_k: int,
    query_per_tx: int,
) -> dict[str, Any]:
    root = PurePosixPath(out_root)
    return {
        "schema": "cvs_leo_weak_iq_cache_build_spec_v1",
        "cache_set_id": cache_set_id,
        "cache_scope": "stage2_target_old",
        "phase2_sample_view_policy": POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "star_ground_channel_impl": "simplified_leo_residual",
        "role_specs": [
            {
                "role": "target_old",
                "pkl": target_dataset,
                "tx_ids": ",".join(old_labels),
                "rxs": receiver,
                "days": "0",
                "max_samples_per_tx": support_pool_max_k + query_per_tx,
            }
        ],
        "dataset_seed": int(seed),
        "support_pool_max_k": int(support_pool_max_k),
        "query_per_tx": int(query_per_tx),
        "satellite_seed_by_scenario": {
            scenario: int(seed + 1000 + index)
            for index, scenario in enumerate(SCENARIOS)
        },
        "support_satellite_seed_by_scenario": {
            scenario: int(seed + 1000 + index)
            for index, scenario in enumerate(SCENARIOS)
        },
        "query_satellite_seed_by_scenario": {
            scenario: int(seed + 2000 + index)
            for index, scenario in enumerate(SCENARIOS)
        },
        "offline_split_partition_policy": "legacy_seeded_nested_exact",
        "legacy_runner_sha256": "1270dbdb40285393519796a65a4f9bce3a0a89debdfce0e9a3ca1521a930a9db",
        "legacy_runner_git_commit": "d7f2f549ceb4903c1ab8b219b44f581379deacf3",
        "apply_scenario_source_sha256": "0441168c391db173db25501165098e0b7236d475003cfdb31b56f5a1f139a22d",
        "legacy_support_query_call_ast_sha256": "1d6f306184fdee90b1c3333714fc187e3c25a0f6836a88c93bf43aa401ecfdf4",
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
    output = args.output_dir.resolve()
    base = json.loads(args.base_config.read_text(encoding="utf-8-sig"))
    if tuple(base["publication_target_receiver_grid"]) != RECEIVERS:
        raise ValueError("receiver grid drift")
    if int(base["k_shot"]) != 10 or str(base["method_id"]) != METHOD:
        raise ValueError("JG matched plan requires method=jg_r8_lr020 and K=10")
    old_labels = [str(value) for value in base["target_old_tx_labels"]]
    plan_root = PurePosixPath(args.runtime_plan_dir)
    run_root = PurePosixPath(args.runtime_run_root)
    log_root = PurePosixPath(args.runtime_log_root)
    target_root = run_root / "phase1_caches/target"
    predictor_root = run_root / "phase2_predictor_packages"
    scorer_root = run_root / "phase2_scoring_sidecars"
    predictor_seal_root = run_root / "phase2_predictor_seals"
    runtime_seal_root = run_root / "runtime_seal"

    phase2 = dict(base)
    phase2.pop("adv3b02_checkpoint", None)
    phase2.update(
        {
            "target_predictor_bundle_root": str(predictor_root),
            "target_predictor_seal_root": str(predictor_seal_root),
            "phase2_runtime_isolation_evidence_root": str(runtime_seal_root),
        }
    )
    forbidden = {
        "manysig_pkl",
        "manytx_pkl",
        "source_dataset",
        "target_dataset",
        "dataset_path",
        "source_train_channel_view",
        "train_channel_view",
        "source_leo_weak_cache_set_manifest",
    }
    if forbidden & set(phase2):
        raise ValueError("JG Phase2 config exposes a forbidden input route")
    _write_json(output / "phase2_config.json", phase2)

    artifacts = {
        "candidate_lock.json": {
            "schema": "adv3b02_jg020_matched_stage2b_candidate_lock_v1",
            "candidate_id": "JG_R8_LR020",
            "method": METHOD,
            "stage": "Stage2-B",
            "k_shot": 10,
            "support_view_count": 3,
            "query_view_count": 1,
            "scope": "joint_gate",
            "rank": 8,
            "alpha": 8.0,
            "learning_rate": 0.02,
            "epochs": 5,
            "max_optimizer_steps": 50,
            "phase2_sample_view_policy": POLICY,
            "query_decision_policy": "per_sample_all_registered_classes",
            "query_role_oracle_access": False,
            "query_class_quota_access": False,
        },
        "head.json": {
            "schema": "adv3b02_support_prototype_head_v1",
            "metric": "cosine",
            "query_truth_access": False,
            "query_role_access": False,
            "query_batch_quota_access": False,
        },
        "tta_policy.json": {"mode": "single_view", "views": 1},
    }
    artifact_rel: dict[str, str] = {}
    for name, payload in artifacts.items():
        relative = f"package_artifacts/{name}"
        _write_json(output / relative, payload)
        artifact_rel[name] = relative

    cache_specs: list[dict[str, Any]] = []
    cache_commands: list[list[str]] = []
    bundle_commands: list[list[str]] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            relative = f"cache_specs/target/rx_{_safe(receiver)}/seed_{seed}.json"
            receiver_root = target_root / f"rx_{_safe(receiver)}" / f"seed_{seed}"
            spec = _cache_spec(
                cache_set_id=f"{base['experiment_id']}_{receiver}_{seed}",
                receiver=receiver,
                old_labels=old_labels,
                seed=seed,
                target_dataset=args.target_dataset,
                out_root=str(receiver_root),
                support_pool_max_k=int(base["support_pool_max_k"]),
                query_per_tx=int(base["query_per_tx"]),
            )
            _write_json(output / relative, spec)
            cache_specs.append({"scope": "stage2_target_old", "spec": relative})
            cache_commands.append(
                [
                    "python",
                    "paper_reproduction/scripts/build_adv3b02_matched_legacy_leo_weak_cache.py",
                    "--spec",
                    str(plan_root / relative),
                    "--device",
                    str(args.cache_device),
                ]
            )
            bundle_root = predictor_root / f"rx_{_safe(receiver)}" / f"seed_{seed}"
            score_root = scorer_root / f"rx_{_safe(receiver)}" / f"seed_{seed}"
            seal = predictor_seal_root / f"rx_{_safe(receiver)}" / f"seed_{seed}" / "seal.json"
            bundle_commands.append(
                [
                    "python",
                    "code/scripts/build_cvs_stage2_predictor_bundle.py",
                    "--target-cache-set",
                    str(receiver_root / "cache_set.json"),
                    "--expected-cache-scope",
                    "stage2_target_old",
                    "--predictor-out-root",
                    str(bundle_root),
                    "--scorer-out-root",
                    str(score_root),
                    "--detached-seal-path",
                    str(seal),
                    "--stage",
                    "stage2b",
                    "--receiver",
                    receiver,
                    "--seed",
                    str(seed),
                    "--old-class-labels",
                    ",".join(old_labels),
                    "--new-class-count",
                    "0",
                    "--support-pool-max-k",
                    str(base["support_pool_max_k"]),
                    "--query-per-tx",
                    str(base["query_per_tx"]),
                    "--offline-split-partition-policy",
                    "legacy_seeded_nested_exact",
                    "--candidate-lock",
                    str(plan_root / artifact_rel["candidate_lock.json"]),
                    "--checkpoint",
                    str(base["adv3b02_checkpoint"]),
                    "--adapter",
                    str(args.ground_adapter),
                    "--head-artifact",
                    str(plan_root / artifact_rel["head.json"]),
                    "--tta-policy-json",
                    str(plan_root / artifact_rel["tta_policy.json"]),
                ]
            )

    id_audit_command = [
        "python",
        "paper_reproduction/scripts/verify_adv3b02_jg020_matched_ids.py",
        "--plan-manifest",
        str(plan_root / "plan_manifest.json"),
        "--legacy-run-root",
        str(args.legacy_run_root),
        "--out",
        str(plan_root / "matched_id_audit.json"),
    ]
    project_root = PurePosixPath(args.runtime_project_root)
    runtime_files = (
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
        "paper_reproduction/cvs_aligned/jg020_stage2c.py",
        "paper_reproduction/cvs_aligned/jg020_runtime_primitives.py",
        "paper_reproduction/DADDA/__init__.py",
        "paper_reproduction/DADDA/losses.py",
        "paper_reproduction/mitigating_receiver_impact_da/__init__.py",
        "paper_reproduction/mitigating_receiver_impact_da/losses.py",
    )
    seal_command = [
        "python",
        "code/scripts/build_phase2_runtime_seal.py",
        "--config",
        str(plan_root / "phase2_config.json"),
        "--out-root",
        str(runtime_seal_root),
    ]
    for relative in runtime_files:
        seal_command.extend(["--runtime-code-file", str(project_root / relative)])

    worker_commands: list[list[str]] = []
    for shard in range(int(args.shard_count)):
        worker_commands.append(
            [
                "python",
                "-u",
                "paper_reproduction/scripts/run_cvs_publication_matrix.py",
                "--phase",
                "stage2b",
                "--config",
                str(plan_root / "phase2_config.json"),
                "--output-root",
                str(run_root / "stage2_runs"),
                "--log-root",
                str(log_root),
                "--methods",
                METHOD,
                "--receivers",
                ",".join(RECEIVERS),
                "--k-grid",
                "10",
                "--seeds",
                ",".join(str(value) for value in SEEDS),
                "--module-override",
                "paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner",
                "--shard-count",
                str(args.shard_count),
                "--shard-index",
                str(shard),
                "--device",
                f"cuda:{shard % int(args.gpu_count)}",
                "--post-prediction-scorer",
                "paper_reproduction/scripts/score_adv3b02_three_da_predictions.py",
                "--scoring-root",
                str(scorer_root),
                "--isolation-launcher",
                "code/scripts/run_phase2_landlock_isolated.py",
                "--runtime-allowlist",
                str(runtime_seal_root / "artifact_member_allowlist.json"),
                "--runtime-evidence-root",
                str(runtime_seal_root),
                "--isolation-runtime-read-dir",
                "/home/szu2070436088/.conda/envs/CVS-RFFI",
                "--execute",
            ]
        )
    manifest = {
        "schema": "adv3b02_jg020_matched_stage2b_k10_plan_v1",
        "experiment_id": base["experiment_id"],
        "phase": "Stage2-B",
        "method": METHOD,
        "receivers": list(RECEIVERS),
        "seeds": list(SEEDS),
        "k_values": [10],
        "formal_method_rows": 25,
        "support_view_count": 3,
        "query_view_count": 1,
        "phase2_sample_view_policy": POLICY,
        "phase2_config": str(plan_root / "phase2_config.json"),
        "phase2_config_sha256": _sha256(output / "phase2_config.json"),
        "phase2_config_exposes_dataset_path": False,
        "cache_specs": cache_specs,
        "commands": {
            "phase1_offline_cache_build": cache_commands,
            "phase1_offline_predictor_bundle_build": bundle_commands,
            "matched_id_audit": id_audit_command,
            "phase2_runtime_seal": seal_command,
            "phase2_workers": worker_commands,
        },
        "runtime_plan_dir": str(plan_root),
        "runtime_run_root": str(run_root),
        "runtime_log_root": str(log_root),
        "legacy_run_root": str(args.legacy_run_root),
        "ground_adapter": str(args.ground_adapter),
    }
    _write_json(output / "plan_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-plan-dir", required=True)
    parser.add_argument("--runtime-run-root", required=True)
    parser.add_argument("--runtime-log-root", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--ground-adapter", required=True)
    parser.add_argument("--legacy-run-root", required=True)
    parser.add_argument("--cache-device", default="cuda:0")
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument(
        "--runtime-project-root",
        default="/home/szu2070436088/2510044040/CV-SincNet",
    )
    args = parser.parse_args()
    if args.shard_count <= 0 or args.gpu_count <= 0:
        raise ValueError("shard-count and gpu-count must be positive")
    result = build_plan(args)
    print(json.dumps({"plan": str(args.output_dir), "rows": result["formal_method_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
