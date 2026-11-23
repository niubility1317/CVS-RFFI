import numpy as np
import torch

from experiments.pamf_lora_main.tail_band_ablation_source import band_path, shapley_twelfths


def test_frequency_switches_keep_both_signs_and_parent_endpoints():
    low = torch.zeros(256, dtype=torch.complex64)
    spectrum = low.clone()
    spectrum[[1, -8, 9, -32, 33, -64, 65, 128]] = 1 + 1j
    high = torch.fft.ifft(spectrum)
    path = list(band_path(low, high))
    assert len(path) == 16
    assert path[0] is low
    assert path[-1] is high
    for bit, bins in ((1, [1, -8]), (2, [9, -32]), (4, [33, -64]), (8, [65, 128])):
        want = torch.zeros_like(spectrum)
        want[bins] = spectrum[bins]
        torch.testing.assert_close(torch.fft.fft(path[bit]), want)
    torch.testing.assert_close(path[11], path[1] + path[2] + path[8])


def test_factorial_attributes_signed_main_effects_and_splits_joint_effects():
    hits = np.array([
        [bool(mask & 1), not bool(mask & 2), bool(mask & 12 == 12)]
        for mask in range(16)
    ])
    np.testing.assert_array_equal(shapley_twelfths(hits),
                                  [[12, 0, 0], [0, -12, 0], [0, 0, 6], [0, 0, 6]])
