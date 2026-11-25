from experiments.pamf_lora_main.w24_formal import ROUTES, SEEDS, cases, receipt
from experiments.pamf_lora_main.w24_native_full import TARGET


def test_registered_shards_cover_each_receiver_once_per_route():
    assert SEEDS == (2026092701, 2026092702, 2026092703)
    matrix = cases()
    for route in ROUTES:
        records = [receipt(name) for name, case_route in matrix.items() if route == case_route]
        assert sum((r["receivers"] for r in records), []) == list(TARGET)
        assert all(r["seeds"] == list(SEEDS) and not r["target_truth_read"] for r in records)
        assert all(r["per_receiver"] == 2000 and len(r["views"]) == 7 for r in records)


def test_bootstrap_preserves_same_packet_correlations_across_reader_seeds(monkeypatch):
    import numpy as np
    from experiments.pamf_lora_main import w24_formal_score as scorer

    monkeypatch.setattr(scorer, "DRAWS", 50)
    delta = np.zeros((2, 7, 3, 10), dtype=np.int16)
    delta[:, 1, 0, :5] = 1
    delta[:, 1, 1, :5] = -1
    intervals, rx_intervals = scorer.bootstrap_counts(delta)
    assert not intervals[:, -1].any()
    assert not rx_intervals[:, :, -1].any()
    assert intervals[1, 0, 0] > 0
    assert intervals[1, 1, 1] < 0
