import numpy as np
import torch

from experiments.pamf_lora_main.w24_residual_pilot import restore_r69


def test_old_r69_only_updates_active_middle_chirp_interiors():
    class Gate:
        def restore(self, iq, model, projector):
            assert iq.shape == (2, 6, 2, 256)
            restored = iq.clone()
            restored[0, :, 0] += .25
            return restored, torch.tensor([True, False])

    x= np.exp(2j*np.pi*.03*np.arange(8192)).astype(np.complex64)
    packets = np.stack([x, x])
    updated, active = restore_r69(packets, Gate(), None, None, "cpu")
    assert active.tolist() == [True, False]
    assert np.array_equal(packets[1], updated[1])
    a,b = packets[0].reshape(8,1024), updated[0].reshape(8,1024)
    assert np.array_equal(a[(0,7), :], b[(0,7), :])
    assert np.array_equal(a[:, :24], b[:, :24])
    assert np.array_equal(a[:, -24:], b[:, -24:])
    assert np.any(a[1:7,24:-24] != b[1:7,24:-24])
