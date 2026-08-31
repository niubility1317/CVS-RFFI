from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cvsrffi.phase1_bicad_xr.config import candidate_config
from cvsrffi.phase1_bicad_xr.metrics import BiCADXRMetricStore
from scripts.analyze_phase1_pairbicad_cv2_matrix import (
    MatrixAnalysisError,
    analyze_cv2_matrix,
)


SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")


def _write_row(
    run_root: Path,
    candidate: str,
    *,
    fold: int = 1,
    seed: int = 392002,
    quality: float,
    receiver_floor: float,
) -> Path:
    updates = candidate_config(candidate).optimizer_updates
    row_root = run_root / f"{candidate}-F{fold}-S{seed}"
    row_root.mkdir(parents=True)
    telemetry = {
        "schema": "ssdg_epoch_telemetry_v1",
        "optimizer_update": updates,
        "candidate_id": candidate,
        "fold": fold,
        "seed": seed,
        "source_only": True,
        "target_access": False,
        "phase2_access": False,
        "support_access": False,
        "query_access": False,
        "truth_access": False,
    }
    (row_root / "metrics_epoch.jsonl").write_text(
        json.dumps(telemetry) + "\n", encoding="utf-8", newline="\n"
    )
    with (row_root / "metrics_epoch.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(telemetry))
        writer.writeheader()
        writer.writerow({key: str(value).lower() if isinstance(value, bool) else value for key, value in telemetry.items()})

    checkpoint = row_root / "final_checkpoint.pt"
    checkpoint.write_bytes(b"cv2-checkpoint")
    runtime = {
        "phase1_method": "bicad_xr",
        "candidate_id": candidate,
        "fold": fold,
        "seed": seed,
        "optimizer_update": updates,
        "total_updates": updates,
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
            "checkpoint_path": checkpoint.name,
            "runtime": runtime,
            "reconstruction": reconstruction,
            "strict_reconstruction": True,
            "trainer_runtime_strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
        },
    )
    diagnostics = BiCADXRMetricStore().snapshot()
    diagnostics.update(
        {
            "receiver_floor": receiver_floor,
            "receiver_std": 0.02,
            "negative_margin_rate": 0.01,
        }
    )
    _write_json(row_root / "diagnostics.json", diagnostics)

    evaluations = {}
    offsets = {"clean": 0.0, "leo_clear_weak": -0.05, "leo_low_elev_weak": -0.10, "leo_rain_weak": -0.15}
    for scenario in SCENARIOS:
        accuracy = quality + offsets[scenario]
        payload = {
            "scenario": scenario,
            "checkpoint": checkpoint.name,
            "checkpoint_load_strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
            "accuracy": accuracy,
            "floor_accuracy": accuracy - 0.02,
            "per_class_accuracy": {"0": accuracy - 0.02, "1": accuracy + 0.02},
        }
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
        },
    )
    return row_root


def test_cv2_analysis_closes_four_scenarios_and_keeps_negative_result_scientific(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    _write_row(run_root, "CV2-D0", quality=0.80, receiver_floor=0.55)
    _write_row(run_root, "CV2-D1", quality=0.79, receiver_floor=0.54)

    result = analyze_cv2_matrix(
        run_root,
        expected_candidates=("CV2-D0", "CV2-D1"),
        expected_folds=(1,),
        expected_seeds=(392002,),
    )

    assert result["row_count"] == 2
    assert set(result["scenarios"]) == set(SCENARIOS)
    d1 = next(row for row in result["rows"] if row["candidate_id"] == "CV2-D1")
    assert d1["technical_failure"] is False
    assert d1["scientific_result"] == "NEGATIVE_SCIENTIFIC_RESULT"
    assert d1["gate"]["same_row"] is True
    assert d1["s_dg"] < next(row for row in result["rows"] if row["candidate_id"] == "CV2-D0")["s_dg"]


def test_cv2_analysis_rejects_missing_formal_scenario(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    row_root = _write_row(run_root, "CV2-D0", quality=0.80, receiver_floor=0.55)
    (row_root / "evaluations" / "leo_rain_weak.json").unlink()

    with pytest.raises(MatrixAnalysisError, match="leo_rain_weak"):
        analyze_cv2_matrix(
            run_root,
            expected_candidates=("CV2-D0",),
            expected_folds=(1,),
            expected_seeds=(392002,),
        )
