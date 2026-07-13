from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cvsrffi import eval as sat_eval  # noqa: E402


def test_eval_sat_on_all_records_per_receiver_results_and_explicit_seeds(monkeypatch):
    observed_seeds = {}

    def fake_evaluate_loader_sat_channel(
        model,
        loader,
        device,
        domain_label_map,
        scenario,
        args,
        max_batches=0,
        seed=0,
    ):
        name = str(loader)
        observed_seeds[(scenario, name)] = seed
        score = {
            "test_unseen_day_seen_rx": 80.0,
            "test_seen_day_unseen_rx": 78.0,
            "test_unseen_day_unseen_rx": 70.0,
            "test_rx_9": 76.0,
            "test_rx_11": 74.0,
            "test_unseen_day_rx_9": 69.0,
            "test_unseen_day_rx_11": 67.0,
        }[name]
        return {"tx_acc": score, "tx_correct": int(score), "tx_total": 100}

    monkeypatch.setattr(sat_eval, "evaluate_loader_sat_channel", fake_evaluate_loader_sat_channel)
    named_loaders = {
        name: name
        for name in (
            "test_unseen_day_seen_rx",
            "test_seen_day_unseen_rx",
            "test_unseen_day_unseen_rx",
            "test_rx_9",
            "test_rx_11",
            "test_unseen_day_rx_9",
            "test_unseen_day_rx_11",
        )
    }

    result = sat_eval.evaluate_sat_scenarios(
        object(),
        named_loaders,
        torch.device("cpu"),
        {},
        ["leo_clear_weak", "leo_rain_weak"],
        SimpleNamespace(eval_sat_on="all", sat_seed=500),
    )

    clear = result["leo_clear_weak"]
    rain = result["leo_rain_weak"]
    assert clear["aggregate"]["tx_acc"] == 76.0
    assert clear["receiver_floor"] == 67.0
    assert clear["receiver_seen_day_floor"] == 74.0
    assert clear["receiver_strict_floor"] == 67.0
    assert set(clear["receiver_named"]) == {
        "test_rx_9",
        "test_rx_11",
        "test_unseen_day_rx_9",
        "test_unseen_day_rx_11",
    }
    assert clear["evaluation_seed"]["base"] == 500
    assert clear["evaluation_seed"]["named"]["test_rx_9"] == 500 + 3 * 97
    assert rain["evaluation_seed"]["named"]["test_rx_9"] == 500 + 1009 + 3 * 97
    for scenario, scenario_stats in result.items():
        for name, stats in scenario_stats["named"].items():
            assert stats["sat_seed"] == observed_seeds[(scenario, name)]


def test_main_sat_eval_preserves_aggregate_without_receiver_evidence(monkeypatch):
    def fake_evaluate_loader_sat_channel(*args, **kwargs):
        return {"tx_acc": 75.0, "tx_correct": 75, "tx_total": 100}

    monkeypatch.setattr(sat_eval, "evaluate_loader_sat_channel", fake_evaluate_loader_sat_channel)
    named_loaders = {
        "test_unseen_day_seen_rx": object(),
        "test_seen_day_unseen_rx": object(),
        "test_unseen_day_unseen_rx": object(),
        "test_rx_9": object(),
    }

    result = sat_eval.evaluate_sat_scenarios(
        object(),
        named_loaders,
        torch.device("cpu"),
        {},
        ["leo_clear_weak"],
        SimpleNamespace(eval_sat_on="main", sat_seed=2027),
    )["leo_clear_weak"]

    assert result["aggregate"]["tx_acc"] == 75.0
    assert result["receiver_named"] == {}
    assert result["receiver_floor"] != result["receiver_floor"]
