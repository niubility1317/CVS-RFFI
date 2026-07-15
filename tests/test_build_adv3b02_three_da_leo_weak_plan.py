from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_reproduction.scripts.build_adv3b02_three_da_leo_weak_plan import build_plan


def test_three_da_plan_has_375_phase2_rows_and_no_dataset_path(tmp_path: Path) -> None:
    base = {
        "experiment_id": "test",
        "adv3b02_checkpoint": "/remote/checkpoint.pth",
        "target_old_tx_labels": ["a", "b"],
        "source_receiver_labels": ["s0", "s1"],
        "publication_target_receiver_grid": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        "support_pool_max_k": 20,
        "query_per_tx": 20,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    output = tmp_path / "plan"
    manifest = build_plan(argparse.Namespace(
        output_dir=output,
        base_config=base_path,
        runtime_plan_dir="/remote/plan",
        runtime_run_root="/remote/run",
        source_dataset="/datasets/ManySig.pkl",
        target_dataset="/datasets/ManySig.pkl",
        cache_device="cuda:0",
        shard_count=8,
        gpu_count=4,
        runtime_project_root="/remote/project",
    ))
    assert manifest["formal_method_rows"] == 375
    assert manifest["rows_per_method"] == 125
    assert manifest["phase1_offline_cache_build_count"] == 26
    assert manifest["phase1_offline_predictor_bundle_build_count"] == 25
    assert manifest["offline_preparation_task_count"] == 52
    assert len(manifest["commands"]["phase2_workers"]) == 8
    phase2_config = json.loads((output / "phase2_config.json").read_text(encoding="utf-8"))
    assert "source_dataset" not in phase2_config
    assert "target_dataset" not in phase2_config
    assert "manysig_pkl" not in phase2_config
    assert "adv3b02_checkpoint" not in phase2_config
    assert phase2_config["source_leo_weak_cache_set_manifest"].endswith(
        "/phase1_caches/source/cache_set.json"
    )
    assert phase2_config["target_predictor_bundle_root"].endswith(
        "/phase2_predictor_packages"
    )
    assert phase2_config["target_predictor_seal_root"].endswith(
        "/phase2_predictor_seals"
    )
    assert "target_leo_weak_cache_root" not in phase2_config
    worker = manifest["commands"]["phase2_workers"][0]
    assert "--post-prediction-scorer" in worker
    assert "--runtime-allowlist" in worker
    assert worker[worker.index("--scoring-root") + 1].endswith(
        "/phase2_scoring_sidecars"
    )
    bundle = manifest["commands"]["phase1_offline_predictor_bundle_build"][0]
    assert "--expected-cache-scope" in bundle
    assert "--predictor-out-root" in bundle
    assert "--scorer-out-root" in bundle
    assert "--detached-seal-path" in bundle
    assert "--old-class-labels" in bundle
    assert "--out-root" not in bundle
    assert (output / "package_artifacts/candidate_lock.json").is_file()
    assert (output / "package_artifacts/tta_policy.json").is_file()
    seal_command = manifest["commands"]["phase2_runtime_seal"]
    roots = [
        seal_command[index + 1]
        for index, value in enumerate(seal_command)
        if value == "--runtime-code-root"
    ]
    assert roots == [
        "/remote/project/code", "/remote/project/paper_reproduction",
        "/remote/project/baselines",
    ]
    target_spec = json.loads(
        (output / "cache_specs/target/rx_20_1/seed_713101.json").read_text(encoding="utf-8")
    )
    assert target_spec["cache_scope"] == "stage2_target_old"
    assert target_spec["role_specs"][0]["max_samples_per_tx"] == 40
