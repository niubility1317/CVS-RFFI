from __future__ import annotations

from pathlib import Path

from paper_reproduction.scripts.build_cvs_stage2c_effective8_strict_plan import generate_strict_plan
from paper_reproduction.scripts.run_cvs_stage2c_effective8_strict_package import package_build_command


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "paper_reproduction/configs/cvs_stage2c_effective8_formal_matrix_20260715.json"


def test_strict_package_command_binds_dual_runtime_capsule_and_receipt(tmp_path: Path) -> None:
    manifest = generate_strict_plan(
        PLAN,
        out_dir=tmp_path / "strict",
        runtime_project_root="/srv/CV-SincNet",
        runtime_artifact_root="/srv/CV-SincNet/runs/v14/runtime_artifacts",
        expected_candidate_capsule_sha256="c" * 64,
    )
    command = package_build_command(manifest, manifest["package_steps"][0])
    assert "--base-checkpoint" in command
    assert "--candidate-capsule" in command
    assert "--expected-candidate-capsule-sha256" in command
    assert command[command.index("--expected-candidate-capsule-sha256") + 1] == "c" * 64
    assert "--runtime-config-receipt" in command
    assert command[command.index("--checkpoint") + 1].endswith("candidate_runtime.ts")
    assert command[command.index("--base-checkpoint") + 1].endswith("base_runtime.ts")
