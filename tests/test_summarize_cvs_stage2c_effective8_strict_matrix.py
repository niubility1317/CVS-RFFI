from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "paper_reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import summarize_cvs_stage2c_effective8_strict_matrix as summary  # noqa: E402


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(tmp_path: Path, *, bad_row_id: bool = False) -> Path:
    run = tmp_path / "run"
    for index in range(2):
        cell_id = f"rx_20_1__seed_{713101 + index}__new_5__k_1"
        cell = run / "cells" / cell_id
        receipt = {
            "status": "PROTOCOL_VALID",
            "cell_id": cell_id,
            "receiver": "20-1",
            "seed": 713101 + index,
            "new_class_count": 5,
            "k_shot": 1,
        }
        rows = []
        for scenario in sorted(summary.SCENARIOS):
            rows.append(
                {
                    "row_id": "bad" if bad_row_id and index == 0 else cell_id,
                    "scenario": scenario,
                    "receiver_label": "20-1",
                    "k_shot": 1,
                    "old_acc_before_increment": 0.8,
                    "old_acc_after_increment": 0.6,
                    "direct_adv3b02_old_acc": 0.7,
                    "seen_new_acc_after_increment": 0.4,
                    "H_old_new_after_increment": 0.48,
                    "candidate_average_forgetting": 0.2,
                    "identity_old_acc_after_increment": 0.55,
                    "shared_view_count_mean": 2.5,
                    "candidate_old_class_acc_before_increment": {"old-a": 0.8},
                    "candidate_old_class_acc_after_increment": {"old-a": 0.6},
                    "candidate_old_class_forgetting": {"old-a": 0.2},
                }
            )
        resource = {
            "schema": "cvs.phase2.predictor_resource_receipt.v2",
            "trainable_parameters": 44048,
            "adapt_epochs": 12,
            "persistent_state_bytes": 109818,
            "peak_cuda_memory_bytes": 168441856,
            "candidate_query_latency_ms": 0.8,
            "mean_backbone_forwards": 2.7,
            "p95_backbone_forwards": 5,
            "view1_rate": 0.2,
            "view3_rate": 0.75,
            "view5_rate": 0.05,
        }
        _write(cell / "cell_receipt.json", receipt)
        _write(cell / "scoring_output" / "formal_rows.json", {"rows": rows})
        _write(cell / "predictor_output" / "predictor_resource_receipt.json", resource)
    return run


def test_summary_preserves_cell_binding_and_writes_joint_rows(tmp_path: Path) -> None:
    run = _run(tmp_path)
    audit = summary.summarize(run, tmp_path / "summary", expected_cells=2)
    assert audit["status"] == "PASS"
    assert audit["cell_count"] == 2
    assert audit["formal_scenario_row_count"] == 6
    assert audit["by_new_k_count"] == 1
    rows = json.loads((tmp_path / "summary" / "cell_summary.json").read_text())["rows"]
    assert rows[0]["delta_before_vs_direct"] == pytest.approx(0.1)
    assert rows[0]["delta_after_vs_direct"] == pytest.approx(-0.1)
    assert rows[0]["min_old_class_acc_after_global"] == pytest.approx(0.6)


def test_summary_rejects_formal_row_cell_mismatch(tmp_path: Path) -> None:
    run = _run(tmp_path, bad_row_id=True)
    with pytest.raises(ValueError, match="formal row/cell binding drift"):
        summary.summarize(run, tmp_path / "summary", expected_cells=2)
