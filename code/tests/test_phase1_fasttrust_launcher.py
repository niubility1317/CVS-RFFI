from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "code/scripts/launch_phase1_adv3b02_fasttrust16_20260821.sh"
MATRIX = ROOT / "configs/phase1_adv3b02_fasttrust16_s392002_20260821.json"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")


def _bash_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/{drive}/{resolved.as_posix().split(':', 1)[1].lstrip('/')}"


def _dry_run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "ROOT": _bash_path(ROOT),
            "MATRIX": _bash_path(MATRIX),
            "RUNS_ROOT": _bash_path(tmp_path / "runs"),
            "PYTHON": "/opt/conda/envs/cvs/bin/python",
            "CONTROL_PYTHON": os.sys.executable,
        }
    )
    return subprocess.run(
        [GIT_BASH.as_posix(), LAUNCHER.relative_to(ROOT).as_posix(), "--dry-run", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_matrix_has_exactly_two_same_seed_rows_per_gpu():
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert data["seed"] == 392002
    assert data["epochs"] == 200
    assert len(data["rows"]) == 16
    assert Counter(row["gpu"] for row in data["rows"]) == Counter({gpu: 2 for gpu in range(8)})
    assert len({row["candidate"] for row in data["rows"]}) == 16


def test_dry_run_expands_all_16_rows_with_fasttrust_and_four_evaluations(tmp_path):
    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[FASTTRUST-ROW]") == 16
    assert result.stdout.count("[MUSE-TRAIN-CMD]") == 16
    assert result.stdout.count("[MUSE-EVAL-CMD]") == 16
    assert result.stdout.count("[MUSE-EVAL-OUTPUT]") == 64
    for gpu in range(8):
        assert result.stdout.count(f"gpu={gpu} slot=") == 2
    for token in (
        "--seed 392002",
        "--epochs 200",
        "--max_grad_norm 5",
        "--muse_fused_student_forward true",
        "--muse_lr_schedule fasttrust",
        "--lambda_sat_cls 0.68",
        "--lambda_sat_cons 0",
        "--strict_reconstruction",
    ):
        assert token in result.stdout
    assert not (tmp_path / "runs").exists()


def test_filters_select_one_candidate_or_one_gpu_pair(tmp_path):
    one = _dry_run(tmp_path, "--only=R4_FAST_FULL_U256")
    assert one.returncode == 0, one.stderr
    assert one.stdout.count("[FASTTRUST-ROW]") == 1
    assert "candidate=R4_FAST_FULL_U256" in one.stdout

    pair = _dry_run(tmp_path, "--gpu=6")
    assert pair.returncode == 0, pair.stderr
    assert pair.stdout.count("[FASTTRUST-ROW]") == 2
    assert "R4_NUISANCE_DETACHED_U256" in pair.stdout
    assert "R4_NO_NUISANCE_U256" in pair.stdout


def test_release_code_root_is_separate_from_project_data_and_run_root(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "ROOT": "/srv/CV-SincNet",
            "CODE_ROOT": _bash_path(ROOT),
            "MATRIX": _bash_path(MATRIX),
            "RUNS_ROOT": "/srv/CV-SincNet/runs/fasttrust",
            "PYTHON": "/opt/conda/envs/cvs/bin/python",
            "CONTROL_PYTHON": os.sys.executable,
        }
    )
    result = subprocess.run(
        [
            GIT_BASH.as_posix(),
            LAUNCHER.relative_to(ROOT).as_posix(),
            "--dry-run",
            "--only=R4_FAST_FULL_U256",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"{_bash_path(ROOT)}/code/SSDG/train_ssdg.py" in result.stdout
    assert "/srv/CV-SincNet/Dataset_WigSig/ManySig.pkl" in result.stdout
    assert "/srv/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701" in result.stdout


def test_real_dispatch_invokes_both_rows_on_each_selected_gpu_and_preserves_failures(tmp_path):
    fake_worker = tmp_path / "fake_worker.sh"
    fake_worker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\t%s\\t%s\\t%s\\t%s\\n' \"$GPU\" \"$CANDIDATE_ID_OVERRIDE\" \"$INIT_MODE\" \"$MUSE_UNLABELED_BATCH_SIZE\" \"$ABLATION\" >> \"$FAKE_CALLS\"\n"
        "[[ \"$CANDIDATE_ID_OVERRIDE\" != R4_NO_U_SAT_ID_U256 ]]\n",
        encoding="utf-8",
        newline="\n",
    )
    calls = tmp_path / "calls.tsv"
    runs = tmp_path / "runs"
    env = os.environ.copy()
    env.update(
        {
            "ROOT": _bash_path(ROOT),
            "MATRIX": _bash_path(MATRIX),
            "RUNS_ROOT": _bash_path(runs),
            "CONTROL_PYTHON": os.sys.executable,
            "WORKER_LAUNCHER": _bash_path(fake_worker),
            "FAKE_CALLS": _bash_path(calls),
        }
    )
    result = subprocess.run(
        [GIT_BASH.as_posix(), LAUNCHER.relative_to(ROOT).as_posix(), "--gpu=2"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    rows = [line.split("\t") for line in calls.read_text(encoding="utf-8").splitlines()]
    assert {row[1] for row in rows} == {"R4_FAST_FULL_U256", "R4_NO_U_SAT_ID_U256"}
    assert all(row[0] == "2" for row in rows)
    assert (runs / "dispatcher_logs" / "R4_FAST_FULL_U256.log").is_file()
    assert (runs / "dispatcher_logs" / "R4_NO_U_SAT_ID_U256.log").is_file()
