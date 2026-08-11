from __future__ import annotations

import pytest
import os
import subprocess
import sys
from pathlib import Path

from cvsrffi.stage2_d92_e0_full_only_target125 import SCENES
from cvsrffi.stage2_d92_e0_full_only_target125_analysis import _scenario_pairs


def test_scenario_pairs_do_not_invent_per_scene_old_floor_from_pooled_tx_rows() -> None:
    job = {
        "outer_key": "rx_20_1__seed_713102__k_5__new_20",
        "receiver": "20-1",
        "seed": 713102,
        "k_shot": 5,
        "new_class_count": 20,
    }
    score = {
        "before": {"by_scenario": {scene: {"old_acc": 0.70} for scene in SCENES}},
        "after": {
            "by_scenario": {
                scene: {"h_old_new": 0.60, "old_acc": 0.65, "seen_new_acc": 0.56}
                for scene in SCENES
            },
            # These rows are pooled across all scenes and cannot define a per-scene floor.
            "by_tx": {
                "old_a": {"role": "target_old", "accuracy": 0.01},
                "old_b": {"role": "target_old", "accuracy": 0.99},
            },
        },
    }
    baseline = {
        ("20-1", 713102, 5, 20, scene): {
            "h_old_new": "0.55",
            "c_old_acc": "0.60",
            "c_old_floor": "0.10",
            "seen_new_acc": "0.51",
            "average_forgetting": "0.10",
        }
        for scene in SCENES
    }

    rows = _scenario_pairs(score, job, baseline)

    assert len(rows) == 3
    assert all("candidate_old_floor" not in row for row in rows)
    assert all(row["delta_h_old_new"] == pytest.approx(0.05) for row in rows)


def test_target125_analyzer_cli_exposes_frozen_inputs() -> None:
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "code")

    completed = subprocess.run(
        [sys.executable, str(repo / "code" / "scripts" / "analyze_d92_e0_full_only_target125.py"), "--help"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--matrix-manifest" in completed.stdout
    assert "--baseline-row-metrics" in completed.stdout
