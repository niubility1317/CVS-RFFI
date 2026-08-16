from __future__ import annotations

import os
import subprocess
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_source_metrics12_v1_20260813.sh"
RUN_ID = "phase1_clic_source_metrics_20260813_v1"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def _dry_run() -> list[str]:
    result = subprocess.run(
        ["bash", _wsl_path(LAUNCHER), "--dry-run"],
        cwd=CODE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


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
    environment = dict(os.environ)
    environment["PROJECT_ROOT"] = _wsl_path(project_root)
    if os.name == "nt":
        wsl_entries = [
            item
            for item in environment.get("WSLENV", "").split(":")
            if item and not item.startswith("PROJECT_ROOT/") and item != "PROJECT_ROOT"
        ]
        environment["WSLENV"] = ":".join([*wsl_entries, "PROJECT_ROOT"])
    result = subprocess.run(
        ["bash", _wsl_path(LAUNCHER)],
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
            ["bash", _wsl_path(LAUNCHER), forbidden],
            cwd=CODE_ROOT.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert f"invalid argument: {forbidden}" in result.stderr
