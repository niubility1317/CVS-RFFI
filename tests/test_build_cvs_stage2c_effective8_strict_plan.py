from __future__ import annotations

from pathlib import Path

from paper_reproduction.scripts.build_cvs_stage2c_effective8_strict_plan import (
    generate_strict_plan,
    validate_strict_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "paper_reproduction/configs/cvs_stage2c_effective8_formal_matrix_20260715.json"


def test_strict_plan_covers_75_packages_300_cells_and_900_rows(tmp_path: Path) -> None:
    manifest = generate_strict_plan(
        PLAN,
        out_dir=tmp_path / "strict",
        runtime_project_root="/srv/CV-SincNet",
        runtime_artifact_root="/srv/CV-SincNet/runs/v14/runtime_artifacts",
        expected_candidate_capsule_sha256="a" * 64,
    )
    validate_strict_plan(manifest)
    assert len(manifest["cache_steps"]) == 25
    assert len(manifest["package_steps"]) == 75
    assert sum(len(item["cells"]) for item in manifest["package_steps"]) == 300
    assert manifest["expected_counts"]["formal_scenario_rows"] == 900
    assert manifest["launch_authority"] is False
    assert manifest["smoke_authority"] is True
    assert manifest["runtime_artifacts"]["candidate_lock"].endswith(
        "/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/candidate_lock_v2.json"
    )


def test_strict_plan_packages_share_one_sealed_pool_across_all_k(tmp_path: Path) -> None:
    manifest = generate_strict_plan(
        PLAN,
        out_dir=tmp_path / "strict",
        runtime_project_root="/srv/CV-SincNet",
        runtime_artifact_root="/srv/CV-SincNet/runs/v14/runtime_artifacts",
        expected_candidate_capsule_sha256="b" * 64,
    )
    first = manifest["package_steps"][0]
    assert [cell["k_shot"] for cell in first["cells"]] == [1, 5, 10, 20]
    assert len({cell["cell_id"] for cell in first["cells"]}) == 4
    assert first["target_cache_set"].endswith("cache_set.json")
    assert manifest["phase2_query_decision_policy"] == "per_sample_all_registered_classes"
