from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "code/scripts/launch_phase1_adv3b02_sat_anchor_ssl8_20260822.sh"
MATRIX = ROOT / "configs/phase1_adv3b02_sat_anchor_ssl8_s392002_20260822.json"
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
            "CODE_ROOT": _bash_path(ROOT),
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


def test_matrix_is_one_u256_row_per_gpu_with_fixed_controls():
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    rows = data["rows"]

    assert data["seed"] == 392002
    assert data["epochs"] == 200
    assert data["unlabeled_batch_size"] == 256
    assert len(rows) == 8
    assert {row["gpu"] for row in rows} == set(range(8))
    assert len({row["candidate"] for row in rows}) == 8
    assert rows[0]["muse_level"] == "M0"
    assert rows[0]["sat_anchor_ssl"] is False
    assert all(row["muse_level"] == "M3" for row in rows[1:])
    assert all(row["sat_anchor_ssl"] is True for row in rows[1:])
    fixed = next(row for row in rows if row["candidate"] == "A3_FIXED_50_FILL")
    adaptive = next(row for row in rows if row["candidate"] == "A3_ADAPTIVE_NO_FILL")
    assert fixed["fill_to_fraction"] == 0.5
    assert adaptive["fill_to_fraction"] == 0.0
    assert all(row["pair_interval"] == 2 for row in rows[5:])


def test_dry_run_expands_eight_rows_and_four_final_evaluations(tmp_path):
    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[SAT-ANCHOR-ROW]") == 8
    assert result.stdout.count("[MUSE-TRAIN-CMD]") == 8
    assert result.stdout.count("[MUSE-EVAL-CMD]") == 8
    assert result.stdout.count("[MUSE-EVAL-OUTPUT]") == 32
    for token in (
        "--sat_anchor_ssl true",
        "--teacher_ckpt",
        "--muse_unlabeled_batch_size 256",
        "--lambda_sat_cls 0.68",
        "--lambda_sat_cons 0",
        "--checkpoint_selection final_only",
        "--strict_reconstruction",
    ):
        assert token in result.stdout
    assert not (tmp_path / "runs").exists()


def test_filter_and_acceleration_controls_are_exact(tmp_path):
    result = _dry_run(tmp_path, "--only=A3_PAIR_INTERVAL2")

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[SAT-ANCHOR-ROW]") == 1
    assert "--sat_anchor_pair_interval 2" in result.stdout
    assert "--sat_anchor_fill_to_fraction 0.0" in result.stdout
    assert "--sat_anchor_adapter false" in result.stdout

    adapter = _dry_run(tmp_path, "--gpu=7")
    assert adapter.returncode == 0, adapter.stderr
    assert "candidate=A5_ADAPTER_TAIL" in adapter.stdout
    assert "--sat_anchor_adapter true" in adapter.stdout
    assert "--sat_anchor_u_gradient_scope adapter_tail" in adapter.stdout


def test_real_dispatch_owns_only_selected_gpu_row(tmp_path):
    fake_worker = tmp_path / "fake_worker.sh"
    fake_worker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\t%s\\t%s\\t%s\\n' \"$GPU\" \"$CANDIDATE_ID_OVERRIDE\" \"$SAT_ANCHOR_PAIR_INTERVAL\" \"$SAT_ANCHOR_GRADIENT_SCOPE\" > \"$FAKE_CALLS\"\n",
        encoding="utf-8",
        newline="\n",
    )
    calls = tmp_path / "calls.tsv"
    runs = tmp_path / "runs"
    env = os.environ.copy()
    env.update(
        {
            "ROOT": _bash_path(ROOT),
            "CODE_ROOT": _bash_path(ROOT),
            "MATRIX": _bash_path(MATRIX),
            "RUNS_ROOT": _bash_path(runs),
            "CONTROL_PYTHON": os.sys.executable,
            "WORKER_LAUNCHER": _bash_path(fake_worker),
            "FAKE_CALLS": _bash_path(calls),
        }
    )
    result = subprocess.run(
        [GIT_BASH.as_posix(), LAUNCHER.relative_to(ROOT).as_posix(), "--gpu=7"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").strip() == "7\tA5_ADAPTER_TAIL\t2\tadapter_tail"
    assert (runs / "dispatcher_logs" / "A5_ADAPTER_TAIL.log").is_file()
