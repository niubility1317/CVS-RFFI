from __future__ import annotations

import os
import subprocess
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_source_metrics12_v1_20260813.sh"
RUN_ID = "phase1_clic_source_metrics_20260813_v1"
V2_LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_source_metrics12_v2_20260816.sh"
V2_SMOKE_LAUNCHER = CODE_ROOT / "scripts" / "smoke_phase1_clic_source_metrics_f1_v2_20260816.sh"
V2_RUN_ID = "phase1_clic_source_metrics_20260813_v2"
V2_SMOKE_ROOT_NAME = ".smoke_phase1_clic_source_metrics_20260813_v2_F1"


def _git_bash_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/{drive}/{relative}"


def _dry_run() -> list[str]:
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(LAUNCHER), "--dry-run"],
        cwd=CODE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _dry_run_script(script: Path) -> list[str]:
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(script), "--dry-run"],
        cwd=CODE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _project_root_environment(project_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PROJECT_ROOT"] = _git_bash_path(project_root)
    return environment


def test_source_metrics_v2_formal_dry_run_keeps_the_complete_frozen_matrix() -> None:
    """Break caught: v2 changes matrix stages, v1 identity, or source-only controls."""

    lines = _dry_run_script(V2_LAUNCHER)
    assert len(lines) == 25
    joined = "\n".join(lines)
    assert V2_RUN_ID in joined
    assert RUN_ID not in joined
    assert len([line for line in lines if "stage=CLIC_SOURCE_V_CACHE" in line]) == 6
    assert len([line for line in lines if "stage=CLIC_SOURCE_V_FORWARD" in line]) == 12
    assert len([line for line in lines if "stage=CLIC_SOURCE_METRICS_PAIR" in line]) == 6
    assert len([line for line in lines if "stage=CLIC_SOURCE_METRICS_AGGREGATE" in line]) == 1
    assert "--technical-smoke" not in joined
    for forbidden in ("target", "query", "truth", "prediction", "package", "--retry"):
        assert forbidden not in joined.lower()


def test_source_metrics_v2_formal_rejects_immutable_root_collision(tmp_path: Path) -> None:
    """Break caught: a new v2 launcher resumes or overwrites its formal run root."""

    project_root = tmp_path / "project"
    (project_root / "runs" / V2_RUN_ID).mkdir(parents=True)
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(V2_LAUNCHER)],
        cwd=CODE_ROOT.parent,
        env=_project_root_environment(project_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "refusing to overwrite source metrics run/log root" in result.stderr


def test_source_metrics_v2_f1_smoke_dry_run_is_path_strict_and_score_free() -> None:
    """Break caught: smoke mirrors formal inputs, scores performance, or launches more than F1 C/G."""

    lines = _dry_run_script(V2_SMOKE_LAUNCHER)
    assert len(lines) == 3
    joined = "\n".join(lines)
    assert V2_RUN_ID in joined
    assert V2_SMOKE_ROOT_NAME in joined
    assert "SMOKE_INVOCATION=1" in joined
    assert "FORMAL_INVOCATION=0" in joined
    assert len([line for line in lines if "stage=CLIC_SOURCE_V_CACHE" in line]) == 1
    forwards = [line for line in lines if "stage=CLIC_SOURCE_V_FORWARD" in line]
    assert len(forwards) == 2
    assert "fold=1" in joined
    assert "F1C_CLIC12" in joined and "F1G_CLIC12" in joined
    assert "F2C_CLIC12" not in joined and "F2G_CLIC12" not in joined
    assert "--technical-smoke" in joined
    assert "stage=CLIC_SOURCE_METRICS_PAIR" not in joined
    assert "stage=CLIC_SOURCE_METRICS_AGGREGATE" not in joined
    assert "source_validation_known_leo_weak.npz" in joined
    assert "phase1_clic12_20260812_v5/F1C_CLIC12/final_ssdg.pth" in joined
    assert "phase1_clic_postfreeze_20260812_v4/F1C_CLIC12/source_clean_proxy.npz" in joined
    for forbidden in ("target", "query", "truth", "prediction", "package", "--retry"):
        assert forbidden not in joined.lower()


def test_source_metrics_v2_smoke_rejects_preexisting_root_before_input_checks(tmp_path: Path) -> None:
    """Break caught: a smoke can overwrite its cache/feature/log root on a second invocation."""

    project_root = tmp_path / "project"
    (project_root / "runs" / V2_SMOKE_ROOT_NAME).mkdir(parents=True)
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(V2_SMOKE_LAUNCHER)],
        cwd=CODE_ROOT.parent,
        env=_project_root_environment(project_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "refusing to overwrite source metrics v2 smoke run/log root" in result.stderr


def test_source_metrics_launcher_dry_run_has_exact_source_only_release_matrix() -> None:
    """Catches a launcher that omits a fold/arm, leaks target input, or misbinds a shared cache."""

    lines = _dry_run()
    cache_lines = [line for line in lines if "stage=CLIC_SOURCE_V_CACHE" in line]
    forward_lines = [line for line in lines if "stage=CLIC_SOURCE_V_FORWARD" in line]
    score_lines = [line for line in lines if "stage=CLIC_SOURCE_METRICS_PAIR" in line]
    aggregate_lines = [line for line in lines if "stage=CLIC_SOURCE_METRICS_AGGREGATE" in line]

    assert len(cache_lines) == 6
    assert len(forward_lines) == 12
    assert len(score_lines) == 6
    assert len(aggregate_lines) == 1

    joined = "\n".join(lines)
    assert "source_only=1" in joined
    assert "retry=NO" in joined
    for forbidden in ("target", "query", "truth", "prediction", "package", "--retry"):
        assert forbidden not in joined.lower()

    for fold in range(1, 7):
        candidate_c = f"F{fold}C_CLIC12"
        candidate_g = f"F{fold}G_CLIC12"
        cache = next(line for line in cache_lines if f"fold={fold}" in line)
        shared_cache = f"F{fold}_SHARED/source_validation_known_leo_weak.npz"
        shared_receipt = f"F{fold}_SHARED/source_validation_known_leo_weak.receipt.json"
        assert candidate_c in cache
        assert candidate_g in cache
        assert shared_cache in cache
        assert shared_receipt in cache

        fold_forwards = [line for line in forward_lines if f"fold={fold}" in line]
        assert len(fold_forwards) == 2
        physical_gpus = {line.split("physical_gpu=", 1)[1].split()[0] for line in fold_forwards}
        assert len(physical_gpus) == 1
        for arm, candidate in (("C", candidate_c), ("G", candidate_g)):
            forward = next(line for line in fold_forwards if f"arm={arm}" in line)
            assert candidate in forward
            assert shared_cache in forward
            assert shared_receipt in forward
            assert f"{candidate}/source_clean_proxy.npz" in forward
            assert f"{candidate}/source_validation_known_leo_weak_features.npz" in forward
            assert f"{candidate}/source_validation_known_leo_weak.binding.json" in forward
            assert f"F{fold}_C_vs_G_pair.json" in forward

        score = next(line for line in score_lines if f"fold={fold}" in line)
        assert "physical_gpu=CPU" in score
        assert candidate_c in score
        assert candidate_g in score
        assert shared_cache in score
        assert shared_receipt in score
        assert f"F{fold}_PAIR/source_metrics_pair.json" in score

    aggregate = aggregate_lines[0]
    assert "physical_gpu=CPU" in aggregate
    assert "--aggregate-folds" in aggregate
    assert "source_metrics_aggregate.json" in aggregate
    for fold in range(1, 7):
        assert f"F{fold}_PAIR/source_metrics_pair.json" in aggregate


def test_source_metrics_launcher_rejects_preexisting_run_root_before_input_checks(tmp_path: Path) -> None:
    """Catches an accidental resume/overwrite of the immutable formal run root."""

    project_root = tmp_path / "project"
    (project_root / "runs" / RUN_ID).mkdir(parents=True)
    environment = _project_root_environment(project_root)
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(LAUNCHER)],
        cwd=CODE_ROOT.parent,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "refusing to overwrite source metrics run/log root" in result.stderr


def test_source_metrics_launcher_rejects_target_and_retry_controls() -> None:
    """Catches acceptance of controls that could add target input or a rerun path."""

    for forbidden in ("--target-root", "--retry"):
        result = subprocess.run(
            [str(GIT_BASH), _git_bash_path(LAUNCHER), forbidden],
            cwd=CODE_ROOT.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert f"invalid argument: {forbidden}" in result.stderr
