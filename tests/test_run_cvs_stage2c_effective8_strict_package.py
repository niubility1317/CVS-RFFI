from __future__ import annotations

from pathlib import Path

import pytest

from paper_reproduction.scripts.build_cvs_stage2c_effective8_strict_plan import generate_strict_plan
from paper_reproduction.scripts.run_cvs_stage2c_effective8_strict_package import (
    _runtime_python_executable,
    package_build_command,
    run_package,
)


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


def test_runtime_python_executable_is_physical_canonical_file(monkeypatch, tmp_path: Path) -> None:
    interpreter = tmp_path / "bin" / "python-real"
    interpreter.parent.mkdir()
    interpreter.write_bytes(b"python")
    monkeypatch.setattr(
        "paper_reproduction.scripts.run_cvs_stage2c_effective8_strict_package.sys.executable",
        str(interpreter.parent / ".." / "bin" / interpreter.name),
    )

    assert _runtime_python_executable() == str(interpreter.resolve(strict=True))


def test_formal_single_cell_fails_before_package_materialization_without_smoke_authority(
    monkeypatch, tmp_path: Path,
) -> None:
    manifest = generate_strict_plan(
        PLAN,
        out_dir=tmp_path / "strict",
        runtime_project_root="/srv/CV-SincNet",
        runtime_artifact_root="/srv/CV-SincNet/runs/v14/runtime_artifacts",
        expected_candidate_capsule_sha256="d" * 64,
    )
    monkeypatch.setattr(
        "paper_reproduction.scripts.run_cvs_stage2c_effective8_strict_package._ensure_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("package materialization must not run before authority")
        ),
    )
    package_id = manifest["smoke_package_id"]
    with pytest.raises(RuntimeError, match="fail-closed"):
        run_package(
            manifest, package_id=package_id, project_root=tmp_path,
            device="cuda:0", execution_mode="formal", k_values=[10],
        )


def test_smoke_mode_rejects_unlocked_k_before_package_materialization(
    monkeypatch, tmp_path: Path,
) -> None:
    manifest = generate_strict_plan(
        PLAN,
        out_dir=tmp_path / "strict",
        runtime_project_root="/srv/CV-SincNet",
        runtime_artifact_root="/srv/CV-SincNet/runs/v14/runtime_artifacts",
        expected_candidate_capsule_sha256="e" * 64,
    )
    monkeypatch.setattr(
        "paper_reproduction.scripts.run_cvs_stage2c_effective8_strict_package._ensure_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("package materialization must not run for an unlocked smoke cell")
        ),
    )
    with pytest.raises(RuntimeError, match="locked pre-authority cell"):
        run_package(
            manifest, package_id=manifest["smoke_package_id"], project_root=tmp_path,
            device="cuda:0", execution_mode="smoke", k_values=[10],
        )
