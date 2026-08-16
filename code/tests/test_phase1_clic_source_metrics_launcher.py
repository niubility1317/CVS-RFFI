from __future__ import annotations

import os
import subprocess
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_source_metrics12_v1_20260813.sh"
RUN_ID = "phase1_clic_source_metrics_20260813_v1"
V3_LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_source_metrics12_v3_20260816.sh"
V3_SMOKE_LAUNCHER = CODE_ROOT / "scripts" / "smoke_phase1_clic_source_metrics_f1_v3_20260816.sh"
V3_RUN_ID = "phase1_clic_source_metrics_20260816_v3"
V3_SMOKE_ROOT_NAME = ".smoke_phase1_clic_source_metrics_20260816_v3_F1"
V3_WISIG_SHA256 = "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
FROZEN_CANONICAL_PROJECT_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"


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
    environment.pop("BASH_ENV", None)
    return environment


def _write_v3_launcher_inputs(project_root: Path) -> None:
    """Create only the formal path-shaped files reached before a launcher dispatch."""

    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)
    wisig = project_root / "Dataset_WigSig" / "ManySig.pkl"
    wisig.parent.mkdir(parents=True, exist_ok=True)
    wisig.write_bytes(b"test-only-wisig")
    for fold in range(1, 7):
        for arm in ("C", "G"):
            candidate = f"F{fold}{arm}_CLIC12"
            checkpoint = project_root / "runs" / "phase1_clic12_20260812_v5" / candidate / "final_ssdg.pth"
            terminal = checkpoint.parent / "phase1_clic_terminal_receipt.json"
            clean = project_root / "runs" / "phase1_clic_postfreeze_20260812_v4" / candidate / "source_clean_proxy.npz"
            for path in (checkpoint, terminal, clean):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.name.encode("ascii"))
        pair = project_root / "runs" / "phase1_clic_source_pair_20260812_v3" / f"F{fold}_C_vs_G_pair.json"
        pair.parent.mkdir(parents=True, exist_ok=True)
        pair.write_bytes(b"pair")


def _smoke_launcher_for_project(tmp_path: Path, project_root: Path) -> Path:
    """Use a disposable launcher copy only to exercise a local path-shaped fixture."""

    fixture_code_root = tmp_path / "smoke_launcher_fixture" / "code"
    fixture_launcher = fixture_code_root / "scripts" / V3_SMOKE_LAUNCHER.name
    fixture_launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher_text = V3_SMOKE_LAUNCHER.read_text(encoding="utf-8")
    frozen_line = f'CANONICAL_PROJECT_ROOT="{FROZEN_CANONICAL_PROJECT_ROOT}"'
    fixture_line = f'CANONICAL_PROJECT_ROOT="{_git_bash_path(project_root)}"'
    assert launcher_text.count(frozen_line) == 1
    fixture_launcher.write_text(launcher_text.replace(frozen_line, fixture_line), encoding="utf-8")
    for entry in ("build_phase1_clic_source_v_leo_iq.py", "export_phase1_clic_source_v_leo_features.py"):
        (fixture_code_root / entry).write_text("# local launcher fixture\n", encoding="utf-8")
    return fixture_launcher


def _post_guard_race_environment(
    *,
    tmp_path: Path,
    project_root: Path,
    run_root: Path,
    log_root: Path,
    marker_path: Path,
    marker_text: str,
) -> dict[str, str]:
    """Make root creation install a rival marker only after the old exists guard."""

    hook = tmp_path / "post_guard_race.sh"
    hook.write_text(
        "\n".join(
            (
                "sha256sum() {",
                '  printf "%s  %s\\n" "${RACE_WISIG_SHA256}" "$1"',
                "}",
                "mkdir() {",
                '  command mkdir "$@"',
                "  local status=$?",
                '  if [[ "${status}" == "0" && "$*" == *"${RACE_LOG_ROOT}"* ]]; then',
                '    printf "%s\\n" "${RACE_MARKER_TEXT}" > "${RACE_MARKER_PATH}"',
                "  fi",
                '  return "${status}"',
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )
    environment = _project_root_environment(project_root)
    environment.update(
        {
            "BASH_ENV": _git_bash_path(hook),
            "RACE_RUN_ROOT": _git_bash_path(run_root),
            "RACE_LOG_ROOT": _git_bash_path(log_root),
            "RACE_MARKER_PATH": _git_bash_path(marker_path),
            "RACE_MARKER_TEXT": marker_text,
            "RACE_WISIG_SHA256": V3_WISIG_SHA256,
        }
    )
    return environment


def test_source_metrics_v3_formal_dry_run_keeps_the_complete_frozen_matrix() -> None:
    """Break caught: v3 changes matrix stages, v1 identity, or source-only controls."""

    lines = _dry_run_script(V3_LAUNCHER)
    assert len(lines) == 25
    joined = "\n".join(lines)
    assert V3_RUN_ID in joined
    assert RUN_ID not in joined
    assert len([line for line in lines if "stage=CLIC_SOURCE_V_CACHE" in line]) == 6
    assert len([line for line in lines if "stage=CLIC_SOURCE_V_FORWARD" in line]) == 12
    assert len([line for line in lines if "stage=CLIC_SOURCE_METRICS_PAIR" in line]) == 6
    assert len([line for line in lines if "stage=CLIC_SOURCE_METRICS_AGGREGATE" in line]) == 1
    assert "--technical-smoke" not in joined
    for forbidden in ("target", "query", "truth", "prediction", "package", "--retry"):
        assert forbidden not in joined.lower()


def test_source_metrics_v3_formal_rejects_immutable_root_collision(tmp_path: Path) -> None:
    """Break caught: a new v3 launcher resumes or overwrites its formal run root."""

    project_root = tmp_path / "project"
    (project_root / "runs" / V3_RUN_ID).mkdir(parents=True)
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(V3_LAUNCHER)],
        cwd=CODE_ROOT.parent,
        env=_project_root_environment(project_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "refusing to overwrite source metrics run/log root" in result.stderr


def test_source_metrics_v3_f1_smoke_dry_run_is_path_strict_and_score_free() -> None:
    """Break caught: smoke mirrors formal inputs, scores performance, or launches more than F1 C/G."""

    lines = _dry_run_script(V3_SMOKE_LAUNCHER)
    assert len(lines) == 3
    joined = "\n".join(lines)
    assert V3_RUN_ID in joined
    assert V3_SMOKE_ROOT_NAME in joined
    assert "SMOKE_INVOCATION=1" in joined
    assert "FORMAL_INVOCATION=0" in joined
    assert len([line for line in lines if "stage=CLIC_SOURCE_V_CACHE" in line]) == 1
    forwards = [line for line in lines if "stage=CLIC_SOURCE_V_FORWARD" in line]
    assert len(forwards) == 2
    assert "fold=1" in joined
    assert "F1C_CLIC12" in joined and "F1G_CLIC12" in joined
    assert "F2C_CLIC12" not in joined and "F2G_CLIC12" not in joined
    assert "--technical-smoke" in joined
    assert f"--formal-project-root {FROZEN_CANONICAL_PROJECT_ROOT}" in joined
    assert "stage=CLIC_SOURCE_METRICS_PAIR" not in joined
    assert "stage=CLIC_SOURCE_METRICS_AGGREGATE" not in joined
    assert "source_validation_known_leo_weak.npz" in joined
    assert "phase1_clic12_20260812_v5/F1C_CLIC12/final_ssdg.pth" in joined
    assert "phase1_clic_postfreeze_20260812_v4/F1C_CLIC12/source_clean_proxy.npz" in joined
    for forbidden in ("target", "query", "truth", "prediction", "package", "--retry"):
        assert forbidden not in joined.lower()


def test_source_metrics_v3_smoke_rejects_preexisting_root_before_input_checks(tmp_path: Path) -> None:
    """Break caught: a smoke can overwrite its cache/feature/log root on a second invocation."""

    project_root = tmp_path / "project"
    (project_root / "runs" / V3_SMOKE_ROOT_NAME).mkdir(parents=True)
    fixture_launcher = _smoke_launcher_for_project(tmp_path, project_root)
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(fixture_launcher)],
        cwd=CODE_ROOT.parent,
        env=_project_root_environment(project_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "refusing to overwrite source metrics v3 smoke run/log root" in result.stderr


def test_source_metrics_v3_smoke_rejects_noncanonical_project_root_before_any_output(tmp_path: Path) -> None:
    """Break caught: caller-controlled PROJECT_ROOT cannot redirect a smoke cache before export."""

    project_root = tmp_path / "mirror_only"
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(V3_SMOKE_LAUNCHER)],
        cwd=CODE_ROOT.parent,
        env=_project_root_environment(project_root),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    assert "frozen canonical project root" in result.stderr
    assert not (project_root / "runs").exists()
    assert not (project_root / "logs").exists()


def test_source_metrics_v3_formal_rejects_post_guard_pid_root_race(tmp_path: Path) -> None:
    """Break caught: a rival root created after the guard cannot replace formal PID evidence."""

    project_root = tmp_path / "project"
    _write_v3_launcher_inputs(project_root)
    run_root = project_root / "runs" / V3_RUN_ID
    log_root = project_root / "logs" / V3_RUN_ID
    marker_path = log_root / "pids_source_metrics12.tsv"
    marker_text = "attacker-formal-pids"
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(V3_LAUNCHER)],
        cwd=CODE_ROOT.parent,
        env=_post_guard_race_environment(
            tmp_path=tmp_path,
            project_root=project_root,
            run_root=run_root,
            log_root=log_root,
            marker_path=marker_path,
            marker_text=marker_text,
        ),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    assert marker_path.read_text(encoding="utf-8") == f"{marker_text}\n"
    assert not (run_root / "F1_SHARED").exists()


def test_source_metrics_v3_smoke_rejects_post_guard_log_root_race(tmp_path: Path) -> None:
    """Break caught: a rival smoke log marker cannot be truncated after an exists check."""

    project_root = tmp_path / "project"
    _write_v3_launcher_inputs(project_root)
    run_root = project_root / "runs" / V3_SMOKE_ROOT_NAME
    log_root = project_root / "logs" / V3_SMOKE_ROOT_NAME
    marker_path = log_root / "F1_source_v_cache.out"
    marker_text = "attacker-smoke-log"
    fixture_launcher = _smoke_launcher_for_project(tmp_path, project_root)
    result = subprocess.run(
        [str(GIT_BASH), _git_bash_path(fixture_launcher)],
        cwd=CODE_ROOT.parent,
        env=_post_guard_race_environment(
            tmp_path=tmp_path,
            project_root=project_root,
            run_root=run_root,
            log_root=log_root,
            marker_path=marker_path,
            marker_text=marker_text,
        ),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    assert marker_path.read_text(encoding="utf-8") == f"{marker_text}\n"
    assert not (run_root / V3_RUN_ID / "F1_SHARED" / "source_validation_known_leo_weak.npz").exists()


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
