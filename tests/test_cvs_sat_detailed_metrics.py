import csv
from types import SimpleNamespace

import torch

from baselines.common import cvs_sat_eval
from baselines.common.cvs_trainer import save_satellite_detailed_csv


def test_satellite_evaluation_records_receiver_transmitter_day_details(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cvs_sat_eval,
        "apply_sat_channel_for_scenario",
        lambda x, scenario, args, gen=None: x,
    )
    batch = {
        "iq": torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0]]),
        "label": torch.tensor([0, 1, 1]),
        "meta": [
            {"tx_i": 0, "tx": "tx-a", "rx_i": 7, "rx": "rx-a", "day_i": 2, "day": "day-c"},
            {"tx_i": 1, "tx": "tx-b", "rx_i": 7, "rx": "rx-a", "day_i": 2, "day": "day-c"},
            {"tx_i": 1, "tx": "tx-b", "rx_i": 8, "rx": "rx-b", "day_i": 3, "day": "day-d"},
        ],
    }
    stats = cvs_sat_eval.evaluate_loader_sat_channel(
        torch.nn.Identity(),
        [batch],
        torch.device("cpu"),
        scenario="leo_clear_weak",
        args=SimpleNamespace(),
    )

    assert stats["tx_total"] == 3
    assert stats["detailed"]["per_receiver"]["rx-a"]["sample_count"] == 2
    assert stats["detailed"]["per_transmitter"]["tx-b"]["correct_count"] == 1
    assert "rx-b|tx-b|day-d" in stats["detailed"]["per_receiver_transmitter_day"]

    extra = {
        "sat_channel": {
            "leo_clear_weak": {
                "named": {"test_unseen_day_unseen_rx": stats},
            }
        }
    }
    path = tmp_path / "details.csv"
    count = save_satellite_detailed_csv(extra, str(path))
    assert count == 10
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["group_type"] for row in rows} == {
        "per_receiver",
        "per_transmitter",
        "per_receiver_transmitter",
        "per_receiver_transmitter_day",
    }
