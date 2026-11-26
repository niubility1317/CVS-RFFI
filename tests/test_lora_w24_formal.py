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


def test_ungated_recovery_calls_network_and_classifies_all_rows(monkeypatch):
    import numpy as np
    import torch
    from experiments.pamf_lora_main import w24_formal as formal

    original = torch.zeros(3, 8, 2, 432)
    original[..., 384:] = 7
    calls = []

    class Model:
        def restore(self, *args):
            raise AssertionError('Gated entry point used')

        def propose(self, y, quality, rms, edges):
            calls.append(len(y))
            return (y, y, y + 2), None, None

    def coordinates(tracker, center):
        y = torch.complex(center[..., 0, :], center[..., 1, :])
        scale = torch.ones_like(y[..., :1])
        return y, scale, None, scale.real

    def classify(head, aperture):
        assert len(aperture) == 3
        assert (aperture[..., 384:] == 7).all()
        return aperture[:, 0, 0, 0].numpy().astype(np.int16)

    monkeypatch.setattr(formal, 'received_aperture', lambda packets, device: original)
    monkeypatch.setattr(formal, 'coordinates', coordinates, raising=False)
    monkeypatch.setattr(formal, 'classify_aperture', classify)
    labels, activity = formal.batch_labels([None], Model(), None, 1000., lambda _: (),
                                           np.zeros((3, 1)), 'cpu', ungated=True)
    assert calls == [3]
    assert activity.tolist() == [3]
    assert labels.tolist() == [[[0, 0, 0], [1, 1, 1]]]
    assert (original[..., :384] == 0).all()


def test_ungated_scorer_rejects_incomplete_forward_coverage(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pytest
    from experiments.pamf_lora_main import w24_formal_score as scorer

    name = f'{scorer.FULL}-0'
    record = receipt(name, ungated=True)
    assert record['seeds'] == [2026092702]
    assert record['restoration_input_policy'] == 'all_packets'
    monkeypatch.setattr(scorer, 'cases', lambda _trial=False: {name: scorer.FULL})
    folder = tmp_path / name
    folder.mkdir()
    (folder / 'complete.json').write_text(json.dumps(record), encoding='utf-8')
    activity = {f'{rx}/{view}': [0, 0, 1999] for rx in record['receivers'] for view in record['views']}
    (folder / 'activity.json').write_text(json.dumps(activity), encoding='utf-8')
    output = np.zeros((1, len(record['modes']), 2000), dtype=np.int16)
    np.savez(folder / f"{record['receivers'][0]}.npz", **dict.fromkeys(record['views'], output))
    with pytest.raises(ValueError, match='Incomplete ungated recovery coverage'):
        scorer.load_frozen(tmp_path, ungated=True)
