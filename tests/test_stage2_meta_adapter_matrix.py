from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
OPERATING_POINTS = {
    "K10/new5": 10,
    "K10/new10": 10,
    "K10/new20": 10,
    "K5/new20": 5,
    "K1/new20": 1,
}


def _row(receiver: str, scenario: str, operating_point: str) -> dict[str, object]:
    slug = f"{receiver}-{scenario}-{operating_point}".replace("/", "-")
    return {
        "row_id": slug,
        "config": {
            "candidate_id": "CVS_META_ADAPTER_TRI_R4_V1",
            "bundle_id": "ADV3B02_CORE90_SOFT_E200_META_TRI_R4_V1",
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": f"capsule-{operating_point}",
            "split_id": f"split-{operating_point}",
            "checkpoint_path": "checkpoint.pt",
            "support_path": f"support-{slug}.npz",
            "query_path": f"query-{slug}.npz",
            "prototype_path": "prototypes.npz",
            "receiver": receiver,
            "scenario": scenario,
            "operating_point": operating_point,
            "seed": 713101,
            "k_shot": OPERATING_POINTS[operating_point],
            "steps": 3,
        },
    }


def _target5_config() -> dict[str, object]:
    return {
        "schema": "cvs.stage2.meta_adapter.matrix.v1",
        "target": "Target5",
        "rows": [
            _row("20-1", scenario, operating_point)
            for operating_point in OPERATING_POINTS
            for scenario in SCENARIOS
        ],
    }


def test_target5_matrix_runs_frozen_cartesian_product_without_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cvsrffi.stage2_meta_adapter_matrix as sut

    events: list[tuple[str, str]] = []

    def fake_run(config, output_dir, device):
        events.append((str(config["operating_point"]), str(config["scenario"])))
        destination = Path(output_dir)
        destination.mkdir(parents=True)
        receipt = {
            "status": "PREDICTIONS_COMPLETE",
            "candidate_id": config["candidate_id"],
            "bundle_id": config["bundle_id"],
            "receiver": config["receiver"],
            "scenario": config["scenario"],
            "operating_point": config["operating_point"],
            "seed": config["seed"],
            "k_shot": config["k_shot"],
            "query_truth_opened": False,
            "query_role_opened": False,
            "source_opened": False,
            "query_state_update_count": 0,
            "states_same_row": True,
            "backward_count": 3,
            "trainable_fraction": 0.008192,
        }
        (destination / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        return receipt

    monkeypatch.setattr(sut, "run_meta_adapter_stage2_row", fake_run)
    output = tmp_path / "target5"
    receipt = sut.run_meta_adapter_matrix(_target5_config(), output, "cuda:0")

    assert len(events) == 15
    assert set(events) == {
        (operating_point, scenario)
        for operating_point in OPERATING_POINTS
        for scenario in SCENARIOS
    }
    assert receipt["status"] == "PREDICTIONS_COMPLETE"
    assert receipt["target"] == "Target5"
    assert receipt["completed_row_count"] == 15
    assert receipt["truth_opened"] is False
    assert receipt["source_opened"] is False
    assert (output / "matrix_receipt.json").is_file()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload["rows"].pop(), "15 rows"),
        (
            lambda payload: payload["rows"][0]["config"].update(
                {"operating_point": "K2/new20", "k_shot": 2}
            ),
            "cartesian product",
        ),
        (
            lambda payload: payload["rows"][0]["config"].update(
                {"query_truth_path": "forbidden.json"}
            ),
            "allowlist",
        ),
    ],
)
def test_target5_matrix_rejects_incomplete_or_non_protocol_rows(
    tmp_path: Path, mutation, match: str
) -> None:
    import cvsrffi.stage2_meta_adapter_matrix as sut

    payload = _target5_config()
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        sut.run_meta_adapter_matrix(payload, tmp_path / "target5", "cpu")
    assert not (tmp_path / "target5").exists()


def test_matrix_preserves_completed_rows_and_stops_on_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cvsrffi.stage2_meta_adapter_matrix as sut

    calls = 0

    def fake_run(config, output_dir, device):
        nonlocal calls
        calls += 1
        destination = Path(output_dir)
        if calls == 2:
            raise RuntimeError("synthetic row failure")
        destination.mkdir(parents=True)
        return {
            "status": "PREDICTIONS_COMPLETE",
            "candidate_id": config["candidate_id"],
            "bundle_id": config["bundle_id"],
            "receiver": config["receiver"],
            "scenario": config["scenario"],
            "operating_point": config["operating_point"],
            "seed": config["seed"],
            "k_shot": config["k_shot"],
            "query_truth_opened": False,
            "query_role_opened": False,
            "source_opened": False,
            "query_state_update_count": 0,
            "states_same_row": True,
            "backward_count": 3,
            "trainable_fraction": 0.008192,
        }

    monkeypatch.setattr(sut, "run_meta_adapter_stage2_row", fake_run)
    output = tmp_path / "target5"
    with pytest.raises(RuntimeError, match="synthetic row failure"):
        sut.run_meta_adapter_matrix(_target5_config(), output, "cuda:0")

    failure = json.loads((output / "matrix_failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED"
    assert failure["completed_row_count"] == 1
    assert failure["failed_row_id"]
    assert calls == 2
    assert len([path for path in output.iterdir() if path.is_dir()]) == 1


def test_matrix_rejects_row_receipt_with_query_state_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cvsrffi.stage2_meta_adapter_matrix as sut

    def fake_run(config, output_dir, device):
        destination = Path(output_dir)
        destination.mkdir(parents=True)
        return {
            "status": "PREDICTIONS_COMPLETE",
            "candidate_id": config["candidate_id"],
            "bundle_id": config["bundle_id"],
            "receiver": config["receiver"],
            "scenario": config["scenario"],
            "operating_point": config["operating_point"],
            "seed": config["seed"],
            "k_shot": config["k_shot"],
            "query_truth_opened": False,
            "query_role_opened": False,
            "source_opened": False,
            "query_state_update_count": 1,
            "states_same_row": True,
            "backward_count": 3,
            "trainable_fraction": 0.008192,
        }

    monkeypatch.setattr(sut, "run_meta_adapter_stage2_row", fake_run)
    output = tmp_path / "target5"
    with pytest.raises(ValueError, match="query state"):
        sut.run_meta_adapter_matrix(_target5_config(), output, "cuda:0")
    failure = json.loads((output / "matrix_failure.json").read_text(encoding="utf-8"))
    assert failure["completed_row_count"] == 0
