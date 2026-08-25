from __future__ import annotations

from pathlib import Path

import pytest

from cvsrffi import stage2_slow_fast_matrix as subject


CANDIDATES = ("COMMON_SHIFT_R4", "FAST_FILM_R8", "FAST_LOWRANK_R8")
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _matrix() -> dict[str, object]:
    rows = []
    for candidate in CANDIDATES:
        for scenario in SCENARIOS:
            rows.append(
                {
                    "row_id": f"{candidate.lower()}-{scenario}",
                    "config": {
                        "candidate_id": candidate,
                        "bundle_id": f"bundle-{candidate.lower()}",
                        "protocol_schema": "p2_min_v1",
                        "phase2_data_status": "VALIDATED_ONCE",
                        "capsule_id": "capsule-fixed",
                        "split_id": "split-fixed",
                        "base_checkpoint_path": "base.pth",
                        "bundle_path": f"{candidate}.pt",
                        "support_path": f"support-{scenario}.npz",
                        "query_path": f"query-{scenario}.npz",
                        "prototype_path": "prototype.npz",
                        "receiver": "20-1",
                        "scenario": scenario,
                        "operating_point": "K10/new10",
                        "seed": 392002,
                        "k_shot": 10,
                        "steps": 3,
                    },
                }
            )
    return {"schema": "cvs.stage2.slow_fast.diag9.v1", "rows": rows}


def test_diag9_requires_exact_candidate_scene_cartesian_product(tmp_path: Path) -> None:
    payload = _matrix()
    payload["rows"] = payload["rows"][:-1]
    with pytest.raises(ValueError, match="exactly 9"):
        subject.run_slow_fast_matrix(payload, tmp_path / "out", "cpu")
    assert not (tmp_path / "out").exists()


def test_diag9_runs_truth_blind_rows_and_closes_matrix(tmp_path: Path, monkeypatch) -> None:
    def fake_runner(config, output, *, device):
        Path(output).mkdir(parents=True)
        return {
            "status": "PREDICTIONS_COMPLETE",
            "candidate_id": config["candidate_id"],
            "scenario": config["scenario"],
            "query_truth_opened": False,
            "query_role_opened": False,
            "source_opened": False,
            "query_state_update_count": 0,
            "states_same_row": True,
            "selected_lambda": 0.5,
        }

    monkeypatch.setattr(subject, "run_slow_fast_stage2_row", fake_runner)
    receipt = subject.run_slow_fast_matrix(_matrix(), tmp_path / "out", "cpu")

    assert receipt["status"] == "PREDICTIONS_COMPLETE"
    assert receipt["completed_row_count"] == 9
    assert receipt["truth_opened"] is False
    assert len(list((tmp_path / "out").glob("*/receipt.json"))) == 0


def test_diag9_rejects_candidate_specific_query_drift(tmp_path: Path) -> None:
    payload = _matrix()
    payload["rows"][1]["config"]["query_path"] = "different-query.npz"
    with pytest.raises(ValueError, match="share support/query"):
        subject.run_slow_fast_matrix(payload, tmp_path / "out", "cpu")
