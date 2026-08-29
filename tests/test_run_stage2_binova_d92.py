from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_stage2_binova_d92.py"
PLAN = ROOT / "configs" / "stage2_binova_d92_minimal_20260829.json"


def test_inspect_plan_declares_stage_a_then_conditional_stage_b() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "inspect-plan", "--plan", str(PLAN)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["stage_order"] == ["A", "B_IF_A_GATE_PASS"]


def test_predict_surface_has_no_truth_argument() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "predict", "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "--truth" not in result.stdout
    assert "--stage-a-root" in result.stdout


def test_cli_exposes_automatic_stage_transition() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "run-auto" in result.stdout


def test_adapt_b_stops_before_output_creation_when_stage_a_gate_failed(tmp_path: Path) -> None:
    stage_a = tmp_path / "stage_a"
    stage_a.mkdir()
    (stage_a / "stage_a_selection.json").write_text(
        json.dumps(
            {
                "schema": "cvs.binova_d92.stage_a.selection.v1",
                "continue_stage_b": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "stage_b"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "adapt-b", "--plan", str(PLAN),
            "--stage-a-root", str(stage_a), "--output-root", str(output),
            "--device", "cpu",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "continuation gate not met" in result.stderr
    assert not output.exists()


def test_plan_rejects_extra_truth_path(tmp_path: Path) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["scene"]["truth"] = "forbidden.npz"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "inspect-plan", "--plan", str(bad)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "scene allowlist mismatch" in result.stderr
