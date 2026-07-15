#!/usr/bin/env python
"""Generate the locked LEO_weak-only effective8 Stage2-C execution plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713101, 713102, 713103, 713104, 713105)
K_VALUES = (1, 5, 10, 20)
NEW_COUNTS = (5, 10, 20)
POLICY = "leo_weak_only_no_clean_access"
BUILD_SPEC_SCHEMA = "cvs_leo_weak_iq_cache_build_spec_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _runtime_path(root: str, relative: str) -> str:
    return str(PurePosixPath(str(root)) / PurePosixPath(str(relative)))


def _safe_receiver(receiver: str) -> str:
    return str(receiver).replace("-", "_")


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema": "cvs_stage2c_effective8_formal_matrix_plan_v1",
        "phase2_sample_view_policy": POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
    }
    failed = [key for key, expected in required.items() if plan.get(key) != expected]
    if failed:
        raise ValueError(f"formal matrix plan contract failed: {failed}")
    exact = {
        "target_receivers": RECEIVERS,
        "confirmation_seeds": SEEDS,
        "k_values": K_VALUES,
        "new_class_counts": NEW_COUNTS,
    }
    for key, expected in exact.items():
        if tuple(plan.get(key, ())) != expected:
            raise ValueError(f"formal matrix plan {key} drift")
    if int(plan.get("support_pool_max_k", -1)) != 20:
        raise ValueError("formal matrix support_pool_max_k must be 20")
    if int(plan.get("query_per_tx", -1)) != 20:
        raise ValueError("formal matrix query_per_tx must be 20")
    adapter = dict(plan.get("adapter", {}))
    if (
        adapter.get("scope") != "effective_feature"
        or int(adapter.get("rank", -1)) != 16
        or not 1 <= int(adapter.get("epochs", -1)) <= 20
        or int(adapter.get("trainable_parameter_cap", -1)) > 50_000
        or int(adapter.get("persistent_state_cap_bytes", -1)) > 256 * 1024
    ):
        raise ValueError("formal matrix adapter resource contract drift")
    tta = dict(plan.get("adaptive_tta", {}))
    if (
        int(tta.get("default_view_count", -1)) != 1
        or tuple(tta.get("allowed_view_counts", ())) != (1, 3, 5)
        or tta.get("per_sample_decision") is not True
        or tta.get("query_fit_used") is not False
        or tta.get("old_new_role_oracle_used") is not False
        or tta.get("class_quota_used") is not False
    ):
        raise ValueError("formal matrix adaptive TTA contract drift")
    return dict(plan)


def _cache_spec(
    *,
    cache_set_id: str,
    cache_scope: str,
    role_specs: list[dict[str, Any]],
    dataset_seed: int,
    satellite_seed_by_scenario: Mapping[str, int],
    out_root: str,
    input_len: int,
    batch_size: int,
) -> dict[str, Any]:
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
        "satellite_seed_by_scenario": {
            scenario: int(satellite_seed_by_scenario[scenario])
            for scenario in SCENARIOS
        },
        "out_npz_by_scenario": {
            scenario: str(PurePosixPath(out_root) / f"{scenario}.npz")
            for scenario in SCENARIOS
        },
        "out_manifest": str(PurePosixPath(out_root) / "cache_set.json"),
        "batch_size": int(batch_size),
        "wisig_out_len": int(input_len),
        "wisig_equalized": "1",
        "wisig_domain": "rx_day",
        "sat_fs_hz": 25_000_000.0,
        "sat_fc_hz": 2_462_000_000.0,
    }


def generate_plan(
    plan_path: Path,
    *,
    out_dir: Path,
    runtime_project_root: str,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite formal plan directory: {out_dir}")
    plan = validate_plan(json.loads(plan_path.read_text(encoding="utf-8-sig")))
    experiment_id = str(plan["experiment_id"])
    runtime_run_root = _runtime_path(
        runtime_project_root, f"runs/{experiment_id}"
    )
    runtime_plan_dir = str(PurePosixPath(runtime_run_root) / "protocol_plan")
    source_dataset = _runtime_path(
        runtime_project_root, str(plan["datasets"]["source"])
    )
    target_dataset = _runtime_path(
        runtime_project_root, str(plan["datasets"]["target"])
    )
    checkpoint = _runtime_path(runtime_project_root, str(plan["checkpoint"]))
    class_split_runtime = _runtime_path(
        runtime_project_root, str(plan["class_split_manifest"])
    )
    class_split_local = REPO_ROOT / str(plan["class_split_manifest"])
    split = json.loads(class_split_local.read_text(encoding="utf-8-sig"))
    old_labels = [str(value) for value in split["target_old_tx_labels"]]
    nested_new = {
        str(key): [str(value) for value in values]
        for key, values in dict(split["nested_target_new_tx_labels"]).items()
    }
    if (
        nested_new["10"][:5] != nested_new["5"]
        or nested_new["20"][:10] != nested_new["10"]
    ):
        raise ValueError("class split target-new lists are not ordered nested prefixes")
    mapping_local = REPO_ROOT / str(
        split["direct_adv3b02_class_mapping_source"]
    )
    if (
        not mapping_local.is_file()
        or _sha256_file(mapping_local)
        != str(split["direct_adv3b02_class_mapping_sha256"])
    ):
        raise ValueError("strict direct ADV3B02 mapping artifact/hash drift")
    mapping_payload = json.loads(mapping_local.read_text(encoding="utf-8-sig"))
    if [str(value) for value in mapping_payload.get("class_id_to_tx", [])] != old_labels:
        raise ValueError("strict direct ADV3B02 mapping order drift")
    # Keep the exact class-split string in every benchmark row.  Candidate-lock
    # verification compares the value byte-for-byte and resolves this trusted
    # repository-relative artifact from the project working directory.
    mapping_source = str(split["direct_adv3b02_class_mapping_source"])

    source_cache_specs: list[dict[str, Any]] = []
    target_cache_specs: list[dict[str, Any]] = []
    target_cache_contracts: list[dict[str, Any]] = []
    stage2_configs: list[dict[str, Any]] = []
    stage2_config_contracts: list[dict[str, Any]] = []
    commands: dict[str, list[list[str]]] = {
        "source_cache_build": [],
        "train": [],
        "source_validation": [],
        "candidate_lock": [],
        "target_cache_build": [],
        "benchmark": [],
        "collect": [],
        "summarize": [],
    }

    input_len = int(plan["leo_weak_iq_input_len"])
    source_train = dict(plan["source_train"])
    source_train_root = str(PurePosixPath(runtime_run_root) / "phase1_caches/source_train")
    source_train_spec = _cache_spec(
        cache_set_id=f"{experiment_id}_source_train",
        cache_scope="source_train",
        role_specs=[
            {
                "role": "source",
                "pkl": source_dataset,
                "tx_ids": str(source_train["tx_ids"]),
                "rxs": str(source_train["receivers"]),
                "max_samples_per_tx": int(source_train["max_samples_per_tx"]),
            }
        ],
        dataset_seed=int(source_train["dataset_seed"]),
        satellite_seed_by_scenario=source_train["satellite_seed_by_scenario"],
        out_root=source_train_root,
        input_len=input_len,
        batch_size=int(plan["adapter"]["batch_size"]),
    )
    source_train_spec_rel = "cache_specs/source_train.json"
    _write_json(out_dir / source_train_spec_rel, source_train_spec)
    source_cache_specs.append(source_train_spec)
    commands["source_cache_build"].append(
        [
            "python",
            "code/scripts/build_cvs_leo_weak_iq_cache.py",
            "--spec",
            str(PurePosixPath(runtime_plan_dir) / source_train_spec_rel),
            "--device",
            "cuda:0",
        ]
    )

    source_validation = dict(plan["source_validation"])
    source_validation_root = str(
        PurePosixPath(runtime_run_root) / "phase1_caches/source_validation"
    )
    source_validation_spec = _cache_spec(
        cache_set_id=f"{experiment_id}_source_validation",
        cache_scope="source_validation",
        role_specs=[
            {
                "role": "source",
                "pkl": source_dataset,
                "tx_ids": str(source_validation["tx_ids"]),
                "rxs": str(source_validation["cache_receivers"]),
                "max_samples_per_tx": int(
                    source_validation["max_samples_per_tx"]
                ),
            }
        ],
        dataset_seed=int(source_validation["dataset_seed"]),
        satellite_seed_by_scenario=source_validation[
            "satellite_seed_by_scenario"
        ],
        out_root=source_validation_root,
        input_len=input_len,
        batch_size=int(plan["adapter"]["batch_size"]),
    )
    source_validation_spec_rel = "cache_specs/source_validation.json"
    _write_json(out_dir / source_validation_spec_rel, source_validation_spec)
    source_cache_specs.append(source_validation_spec)
    commands["source_cache_build"].append(
        [
            "python",
            "code/scripts/build_cvs_leo_weak_iq_cache.py",
            "--spec",
            str(PurePosixPath(runtime_plan_dir) / source_validation_spec_rel),
            "--device",
            "cuda:0",
        ]
    )

    adapter_state = str(PurePosixPath(runtime_run_root) / "effective8_adapter_fp16.pt")
    training_manifest = str(PurePosixPath(runtime_run_root) / "training_manifest.json")
    source_validation_out = str(PurePosixPath(runtime_run_root) / "source_validation")
    promotion_manifest = str(PurePosixPath(source_validation_out) / "promotion_manifest.json")
    candidate_lock = str(PurePosixPath(runtime_run_root) / "candidate_lock.json")
    adapter = dict(plan["adapter"])
    commands["train"].append(
        [
            "python",
            "code/scripts/train_apply_phase1_iq_preadapter_20260703.py",
            "--ckpt",
            checkpoint,
            "--runs_root",
            runtime_run_root,
            "--source_tx_ids",
            str(source_train["tx_ids"]),
            "--source_rxs",
            str(source_train["receivers"]),
            "--source_leo_weak_cache_set_manifest",
            str(PurePosixPath(source_train_root) / "cache_set.json"),
            "--wisig_out_len",
            str(input_len),
            "--num_old_classes",
            str(len(old_labels)),
            "--feature_name",
            "z_id",
            "--sat_scenarios",
            ",".join(SCENARIOS),
            "--star_ground_channel_impl",
            "simplified_leo_residual",
            "--batch_size",
            str(adapter["batch_size"]),
            "--epochs",
            str(adapter["epochs"]),
            "--no-input_adapter_enabled",
            "--model_adapter_mode",
            "lora_effective_feature",
            "--lora_rank",
            str(adapter["rank"]),
            "--lora_alpha",
            str(adapter["alpha"]),
            "--adapter_state_out",
            adapter_state,
            "--adapter_manifest_out",
            training_manifest,
            "--source_only_ground_lora",
            "--input_repair",
            "raw",
            "--lr",
            str(adapter["learning_rate"]),
            "--weight_decay",
            str(adapter["weight_decay"]),
            "--mse_weight",
            "1",
            "--cos_weight",
            "2",
            "--proto_ce_weight",
            "0.2",
            "--logit_ce_weight",
            "0",
            "--leo_reference_identity_weight",
            "22",
            "--leo_reference_cos_weight",
            "1",
            "--feature_margin_weight",
            "4.5",
            "--leo_reference_margin_weight",
            "7.5",
            "--feature_margin_tolerance",
            "0.01",
            "--teacher_logit_distill_weight",
            "0.16",
            "--multiview_consistency_weight",
            "0.25",
            "--relation_preservation_weight",
            "0.5",
            "--prototype_gram_weight",
            "0.25",
            "--prototype_gram_max_cosine",
            "0.65",
            "--worst_k_risk_weight",
            "0.5",
            "--worst_k_values",
            "1,2,5,10,20",
            "--worst_k_tau",
            "0.2",
            "--worst_k_proto_temperature",
            "0.07",
            "--distill_temperature",
            "2",
            "--residual_weight",
            "0",
            "--proto_temperature",
            "0.07",
            "--grad_clip",
            "1",
            "--log_every",
            "1",
            "--device",
            "cuda:0",
            "--seed",
            str(source_train["dataset_seed"]),
        ]
    )
    commands["source_validation"].append(
        [
            "python",
            "paper_reproduction/scripts/validate_cvs_ground_lora_multiview.py",
            "--ckpt",
            checkpoint,
            "--adapter_state",
            adapter_state,
            "--training_manifest",
            training_manifest,
            "--source_cache_set",
            str(PurePosixPath(source_validation_root) / "cache_set.json"),
            "--out_dir",
            source_validation_out,
            "--source_train_rxs",
            str(source_validation["train_receivers"]),
            "--source_val_rxs",
            str(source_validation["validation_receivers"]),
            "--num_old_classes",
            str(len(old_labels)),
            "--batch_size",
            str(adapter["batch_size"]),
            "--device",
            "cuda:0",
        ]
    )
    commands["candidate_lock"].append(
        [
            "python",
            "paper_reproduction/scripts/build_cvs_stage2c_candidate_lock.py",
            "--candidate_id",
            "effective8-r16-e12-leoweak-v14",
            "--checkpoint",
            checkpoint,
            "--adapter_state",
            adapter_state,
            "--promotion_manifest",
            promotion_manifest,
            "--class_split_manifest",
            class_split_runtime,
            "--execution_plan_manifest",
            str(PurePosixPath(runtime_plan_dir) / "plan_manifest.json"),
            "--out_json",
            candidate_lock,
        ]
    )

    for receiver in RECEIVERS:
        for seed in SEEDS:
            target_root = str(
                PurePosixPath(runtime_run_root)
                / "phase1_caches/target"
                / f"rx_{_safe_receiver(receiver)}"
                / f"seed_{seed}"
            )
            satellite_seeds = {
                scenario: int(seed * 10 + index)
                for index, scenario in enumerate(SCENARIOS)
            }
            target_spec = _cache_spec(
                cache_set_id=f"{experiment_id}_{receiver}_{seed}",
                cache_scope="stage2_registered",
                role_specs=[
                    {
                        "role": "target_old",
                        "pkl": target_dataset,
                        "tx_ids": ",".join(old_labels),
                        "rxs": receiver,
                        "days": str(plan["target_day"]),
                        "max_samples_per_tx": int(plan["target_samples_per_tx"]),
                    },
                    {
                        "role": "target_new",
                        "pkl": target_dataset,
                        "tx_ids": ",".join(nested_new["20"]),
                        "rxs": receiver,
                        "days": str(plan["target_day"]),
                        "max_samples_per_tx": int(plan["target_samples_per_tx"]),
                    },
                ],
                dataset_seed=int(seed),
                satellite_seed_by_scenario=satellite_seeds,
                out_root=target_root,
                input_len=input_len,
                batch_size=int(adapter["batch_size"]),
            )
            target_spec_rel = (
                f"cache_specs/target/rx_{_safe_receiver(receiver)}/"
                f"seed_{seed}.json"
            )
            _write_json(out_dir / target_spec_rel, target_spec)
            target_cache_specs.append(target_spec)
            target_cache_contracts.append(
                {
                    "receiver": receiver,
                    "seed": int(seed),
                    "cache_build_spec": str(
                        PurePosixPath(runtime_plan_dir) / target_spec_rel
                    ),
                    "cache_build_spec_sha256": _sha256_file(
                        out_dir / target_spec_rel
                    ),
                    "cache_build_spec_content_sha256": _canonical_sha256(
                        target_spec
                    ),
                    "cache_set_manifest": str(
                        PurePosixPath(target_root) / "cache_set.json"
                    ),
                    "satellite_seed_by_scenario": satellite_seeds,
                }
            )
            commands["target_cache_build"].append(
                [
                    "python",
                    "code/scripts/build_cvs_leo_weak_iq_cache.py",
                    "--spec",
                    str(PurePosixPath(runtime_plan_dir) / target_spec_rel),
                    "--device",
                    "cuda:0",
                ]
            )
            for new_count in NEW_COUNTS:
                for k_shot in K_VALUES:
                    config_rel = (
                        f"stage2_configs/rx_{_safe_receiver(receiver)}/seed_{seed}/"
                        f"new_{new_count}_k_{k_shot}.json"
                    )
                    config_runtime = str(PurePosixPath(runtime_plan_dir) / config_rel)
                    config = {
                        "schema": "cvs_stage2c_effective8_formal_row_config_v1",
                        "experiment_id": experiment_id,
                        "phase2_sample_view_policy": POLICY,
                        "clean_sample_access": False,
                        "clean_derived_signal_access": False,
                        "target_channel_view": "leo_weak_only",
                        "target_channel_scenarios": list(SCENARIOS),
                        "leo_weak_cache_set_manifest": str(
                            PurePosixPath(target_root) / "cache_set.json"
                        ),
                        "leo_weak_iq_input_len": input_len,
                        "target_receiver_labels": [receiver],
                        "target_old_tx_labels": old_labels,
                        "target_new_tx_labels": nested_new[str(new_count)],
                        "k_shot": int(k_shot),
                        "support_pool_max_k": int(plan["support_pool_max_k"]),
                        "query_per_tx": int(plan["query_per_tx"]),
                        "seed": int(seed),
                        "direct_adv3b02_class_mapping_source": mapping_source,
                        "direct_adv3b02_class_mapping_sha256": str(
                            split["direct_adv3b02_class_mapping_sha256"]
                        ),
                        "direct_adv3b02_class_id_to_tx": old_labels,
                        "old_new_role_oracle_used": False,
                        "class_quota_used": False,
                        "query_fit_used": False,
                        "query_batch_state_required": False,
                        "extreme_light_max_persistent_state_bytes": int(
                            adapter["persistent_state_cap_bytes"]
                        ),
                    }
                    _write_json(out_dir / config_rel, config)
                    stage2_configs.append(config)
                    stage2_config_contracts.append(
                        {
                            "receiver": receiver,
                            "seed": int(seed),
                            "new_class_count": int(new_count),
                            "k_shot": int(k_shot),
                            "config": config_runtime,
                            "config_file_sha256": _sha256_file(out_dir / config_rel),
                            "config_content_sha256": _canonical_sha256(config),
                            "cache_set_manifest": str(
                                PurePosixPath(target_root) / "cache_set.json"
                            ),
                        }
                    )
                    result_root = str(
                        PurePosixPath(runtime_run_root)
                        / "formal_rows"
                        / f"rx_{_safe_receiver(receiver)}"
                        / f"seed_{seed}"
                        / f"new_{new_count}_k_{k_shot}"
                    )
                    commands["benchmark"].append(
                        [
                            "python",
                            "paper_reproduction/scripts/benchmark_cvs_adaptive_rxlight_tta.py",
                            "--config",
                            config_runtime,
                            "--ckpt",
                            checkpoint,
                            "--adapter_state",
                            adapter_state,
                            "--adapter_manifest",
                            promotion_manifest,
                            "--candidate_lock",
                            candidate_lock,
                            "--out_dir",
                            result_root,
                            "--head_mode",
                            "symmetric_locked",
                            "--batch_size",
                            str(adapter["batch_size"]),
                            "--device",
                            "cuda:0",
                        ]
                    )

    evidence_root = str(PurePosixPath(runtime_run_root) / "formal_matrix_evidence")
    commands["collect"].append(
        [
            "python",
            "paper_reproduction/scripts/collect_cvs_stage2c_formal_outputs.py",
            "--plan_manifest",
            str(PurePosixPath(runtime_plan_dir) / "plan_manifest.json"),
            "--out_dir",
            evidence_root,
        ]
    )
    commands["summarize"].append(
        [
            "python",
            "paper_reproduction/scripts/summarize_cvs_stage2c_locked_matrix.py",
            "--row_csv",
            str(PurePosixPath(evidence_root) / "formal_rows.csv"),
            "--prediction_csv",
            str(PurePosixPath(evidence_root) / "formal_predictions.csv"),
            "--out_json",
            str(PurePosixPath(runtime_run_root) / "formal_matrix_summary.json"),
        ]
    )

    manifest = {
        "schema": "cvs_stage2c_effective8_generated_execution_plan_v1",
        "experiment_id": experiment_id,
        "source_plan": str(plan_path),
        "source_plan_sha256": _sha256_file(plan_path),
        "runtime_project_root": str(runtime_project_root),
        "runtime_plan_dir": runtime_plan_dir,
        "phase2_sample_view_policy": POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "formal_matrix_contract": {
            "target_receivers": list(RECEIVERS),
            "confirmation_seeds": list(SEEDS),
            "leo_weak_scenarios": list(SCENARIOS),
            "new_class_counts": list(NEW_COUNTS),
            "k_values": list(K_VALUES),
            "query_per_tx": int(plan["query_per_tx"]),
            "support_pool_max_k": int(plan["support_pool_max_k"]),
        },
        "target_cache_contracts": target_cache_contracts,
        "stage2_config_contracts": stage2_config_contracts,
        "stage_order": [
            "source_cache_build",
            "train",
            "source_validation",
            "candidate_lock",
            "target_cache_build",
            "benchmark",
            "collect",
            "summarize",
        ],
        "fail_closed_dependencies": {
            "train_requires_source_train_cache": True,
            "source_validation_requires_training_complete": True,
            "candidate_lock_requires_source_validation_pass": True,
            "target_cache_build_before_phase2": True,
            "benchmark_requires_candidate_lock": True,
            "collect_requires_all_benchmarks": True,
            "summarize_requires_collection": True,
        },
        "expected_counts": {
            "source_cache_sets": len(source_cache_specs),
            "target_cache_sets": len(target_cache_specs),
            "benchmark_invocations": len(commands["benchmark"]),
            "formal_scenario_rows": len(commands["benchmark"]) * len(SCENARIOS),
            "collection_invocations": len(commands["collect"]),
            "summary_invocations": len(commands["summarize"]),
        },
        "commands": commands,
    }
    _write_json(out_dir / "plan_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--runtime_project_root", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate_plan(
        args.plan,
        out_dir=args.out_dir,
        runtime_project_root=str(args.runtime_project_root),
    )
    print(json.dumps(manifest["expected_counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
