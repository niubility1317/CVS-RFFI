from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


WORKTREE = Path(__file__).resolve().parents[3]
SCRIPT = WORKTREE / "code" / "scripts" / "analyze_phase1_pairbicad_matrix.py"
CANDIDATES = ("P0", "P1", "P2", "P3", "P4")
FOLDS = (1, 8)
SEEDS = (392001, 392002, 392003)
UPDATES = 4000
SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
QUALITY = {
    "P0": 0.70,
    "P1": 0.70,
    "P2": 0.60,
    "P3": 0.80,
    "P4": 0.90,
}


def _write_json(path: Path, payload: object, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=allow_nan),
        encoding="utf-8",
        newline="\n",
    )


def _row_id(candidate: str, fold: int, seed: int) -> str:
    return f"{candidate}-F{fold}-S{seed}"


def _scenario_payload(candidate: str, scenario: str) -> dict[str, object]:
    base = QUALITY[candidate]
    offset = {
        "clean": 0.0,
        "leo_clear_weak": -0.05,
        "leo_low_elev_weak": -0.10,
        "leo_rain_weak": -0.15,
    }[scenario]
    accuracy = base + offset
    per_class = {"0": accuracy - 0.02, "1": accuracy + 0.02}
    return {
        "scenario": scenario,
        "checkpoint": "bicad_xr_final.pth",
        "checkpoint_load_strict": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "accuracy": accuracy,
        "floor_accuracy": min(per_class.values()),
        "per_class_accuracy": per_class,
    }


def _write_row(
    run_root: Path,
    candidate: str,
    fold: int,
    seed: int,
) -> Path:
    row_root = run_root / _row_id(candidate, fold, seed)
    row_root.mkdir(parents=True)

    telemetry_rows = [
        {
            "schema": "ssdg_epoch_telemetry_v1",
            "epoch": 1,
            "candidate_id": candidate,
            "seed": seed,
            "optimizer_update": UPDATES - 1,
            "target_access": False,
            "train_loss": 0.4,
        },
        {
            "schema": "ssdg_epoch_telemetry_v1",
            "epoch": 2,
            "candidate_id": candidate,
            "seed": seed,
            "optimizer_update": UPDATES,
            "target_access": False,
            "train_loss": 0.3,
        },
    ]
    (row_root / "metrics_epoch.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in telemetry_rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (row_root / "metrics_epoch.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(telemetry_rows[0]))
        writer.writeheader()
        writer.writerows(telemetry_rows)

    runtime = {
        "phase1_method": "bicad_xr",
        "candidate_id": candidate,
        "fold": fold,
        "seed": seed,
        "optimizer_update": UPDATES,
        "total_updates": UPDATES,
        "source_receivers": [1, 3, 4, 6, 8],
        "train_days": [1, 2, 3],
        "source_only": True,
        "target_access": False,
        "phase2_access": False,
        "support_access": False,
        "query_access": False,
        "truth_access": False,
    }
    reconstruction = {"missing": [], "unexpected": [], "shape_mismatch": []}
    _write_json(
        row_root / "checkpoint_runtime.json",
        {
            "checkpoint_path": "bicad_xr_final.pth",
            "runtime": runtime,
            "reconstruction": reconstruction,
            "strict_reconstruction": True,
            "trainer_runtime_strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
        },
    )
    (row_root / "bicad_xr_final.pth").write_bytes(b"synthetic-checkpoint")
    _write_json(row_root / "diagnostics.json", {"throughput": 12.5, "note": "N/A"})

    evaluations: dict[str, dict[str, object]] = {}
    for scenario in SCENARIOS:
        payload = _scenario_payload(candidate, scenario)
        evaluations[scenario] = payload
        _write_json(row_root / "evaluations" / f"{scenario}.json", payload)
        (row_root / "evaluations" / f"{scenario}.log").write_text(
            f"scenario={scenario} complete\n", encoding="utf-8", newline="\n"
        )
    _write_json(
        row_root / "ARTIFACTS_COMPLETE.json",
        {
            "complete": True,
            "status": "ARTIFACTS_COMPLETE",
            "missing": [],
            "reconstruction": reconstruction,
            "evaluations": evaluations,
            "checkpoint": str(row_root / "bicad_xr_final.pth"),
        },
    )
    return row_root


def _write_valid_run(run_root: Path) -> None:
    for candidate in CANDIDATES:
        for fold in FOLDS:
            for seed in SEEDS:
                _write_row(run_root, candidate, fold, seed)


def _run_cli(run_root: Path, output_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-root",
            str(run_root),
            "--expected-candidates",
            ",".join(CANDIDATES),
            "--expected-folds",
            ",".join(map(str, FOLDS)),
            "--expected-seeds",
            ",".join(map(str, SEEDS)),
            "--expected-updates",
            str(UPDATES),
            "--output-json",
            str(output_dir / "analysis.json"),
            "--output-csv",
            str(output_dir / "analysis.csv"),
            *extra,
        ],
        cwd=WORKTREE,
        capture_output=True,
        text=True,
        check=False,
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stdout}\n{result.stderr}"


def test_valid_30_row_run_writes_same_row_metrics_and_ranked_candidates(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    output_dir = tmp_path / "out"
    _write_valid_run(run_root)

    result = _run_cli(run_root, output_dir)

    assert result.returncode == 0, _combined_output(result)
    payload = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    assert payload["row_count"] == 30
    assert len(payload["rows"]) == 30
    assert payload["ranking"] == ["P4", "P3", "P0", "P1", "P2"]
    assert payload["candidates"][2]["candidate_id"] == "P0"
    assert payload["candidates"][3]["candidate_id"] == "P1"

    first_row = next(row for row in payload["rows"] if row["row_id"] == "P4-F1-S392001")
    assert first_row["leo_mean"] == pytest.approx(0.80)
    assert first_row["leo_scenario_floor"] == pytest.approx(0.75)
    assert first_row["leo_class_floor"] == pytest.approx(0.73)
    assert first_row["source_sat_hmean"] == pytest.approx(2 * 0.90 * 0.75 / 1.65)

    csv_rows = list(csv.DictReader((output_dir / "analysis.csv").open(encoding="utf-8")))
    assert len(csv_rows) == 30
    assert csv_rows[0]["row_id"] == "P0-F1-S392001"


def test_missing_row_fails_with_exact_row_id(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_valid_run(run_root)
    shutil.rmtree(run_root / "P4-F8-S392003")

    result = _run_cli(run_root, tmp_path / "out")

    assert result.returncode != 0
    assert "P4-F8-S392003" in _combined_output(result)
    assert "row" in _combined_output(result).lower()


def test_wrong_final_update_fails_with_row_and_field(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_valid_run(run_root)
    row_root = run_root / "P0-F1-S392001"
    lines = (row_root / "metrics_epoch.jsonl").read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1])
    last["optimizer_update"] = UPDATES - 1
    lines[-1] = json.dumps(last)
    (row_root / "metrics_epoch.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = _run_cli(run_root, tmp_path / "out")

    assert result.returncode != 0
    output = _combined_output(result)
    assert "P0-F1-S392001" in output
    assert "optimizer_update" in output


def test_strict_reconstruction_failure_fails_with_row_and_field(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_valid_run(run_root)
    path = run_root / "P1-F8-S392002" / "checkpoint_runtime.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reconstruction"]["shape_mismatch"] = ["head.weight"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_cli(run_root, tmp_path / "out")

    assert result.returncode != 0
    output = _combined_output(result)
    assert "P1-F8-S392002" in output
    assert "shape_mismatch" in output


def test_nonfinite_metric_fails_with_row_and_field(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_valid_run(run_root)
    path = run_root / "P2-F1-S392003" / "evaluations" / "leo_rain_weak.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["accuracy"] = float("nan")
    _write_json(path, payload, allow_nan=True)

    result = _run_cli(run_root, tmp_path / "out")

    assert result.returncode != 0
    output = _combined_output(result)
    assert "P2-F1-S392003" in output
    assert "accuracy" in output
    assert "finite" in output.lower()


def test_forbidden_access_flag_fails_with_row_and_field(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_valid_run(run_root)
    path = run_root / "P3-F8-S392001" / "checkpoint_runtime.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime"]["query_access"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_cli(run_root, tmp_path / "out")

    assert result.returncode != 0
    output = _combined_output(result)
    assert "P3-F8-S392001" in output
    assert "query_access" in output


def test_per_class_dimension_mismatch_fails_with_row_and_field(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_valid_run(run_root)
    path = run_root / "P4-F1-S392002" / "evaluations" / "leo_clear_weak.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["per_class_accuracy"]["1"]
    _write_json(path, payload)

    result = _run_cli(run_root, tmp_path / "out")

    assert result.returncode != 0
    output = _combined_output(result)
    assert "P4-F1-S392002" in output
    assert "per_class_accuracy" in output
    assert "dimension" in output.lower() or "shape" in output.lower()
