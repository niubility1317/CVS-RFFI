import numpy as np
import torch

from experiments.pamf_lora_main import w24_wide_transfer as transfer


def test_wide_batch_uses_registered_aperture_and_keeps_inactive_labels(monkeypatch):
    iq = np.zeros((3, 8, 2, 432), dtype=np.float32)
    iq[..., -48:] = 5
    iq[:, 0, 0, 0] = [1, 2, 3]
    raw = np.zeros((3, 8192), dtype=np.complex64)
    monkeypatch.setattr(transfer, "received_aperture", lambda packets, device: torch.tensor(iq))
    calls = []

    def reader(x):
        assert (x[..., -48:] == 5).all()
        calls.append(len(x))
        return (torch.nn.functional.one_hot(x[:, 0, 0, 0].long(), 10).float(),)

    class Model:
        def restore(self, tracker, x, threshold):
            assert threshold == 2
            changed = x.clone()
            changed[1, 0, 0, 0] = 7
            return changed, torch.tensor([False, True, False])

    prediction, count = transfer.aperture_batch(reader, Model(), None, 2, raw, "cpu")
    assert prediction.tolist() == [[1, 2, 3], [1, 7, 3]]
    assert calls == [3, 1]
    assert count == 1
