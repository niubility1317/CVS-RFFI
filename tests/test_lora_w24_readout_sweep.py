import numpy as np
import torch

from experiments.pamf_lora_main import w24_native_residual_source as source


def test_readout_sweep_uses_one_restoration_and_keeps_actual_edges(monkeypatch):
    original = torch.zeros(2, 8, 2, 432)
    original[..., 384:] = 7
    calls = []

    def delta(model, tracker, iq):
        calls.append(iq.shape[0])
        return torch.full_like(iq[..., :384], 4)

    def classify(model, iq):
        assert (iq[..., 384:] == 7).all()
        return iq[:, 0, 0, 0].numpy().astype(np.int16)

    monkeypatch.setattr(source, 'received_aperture', lambda *_: original)
    monkeypatch.setattr(source, 'wide_delta', delta, raising=False)
    monkeypatch.setattr(source, 'classify_aperture', classify, raising=False)
    decisions = source.sweep_labels(None, None, None, np.zeros((2, 1)), 'cpu')
    assert calls == [2]
    assert decisions.tolist() == [[0, 0], [2, 2], [3, 3], [4, 4]]
    assert (original[..., :384] == 0).all()
