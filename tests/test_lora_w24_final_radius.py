import torch

from experiments.pamf_lora_main.w24_final_radius import final_radius
from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes
from experiments.pamf_lora_main.w24_third_prior import trial


def test_final_radius_uses_packet_power_and_preserves_the_original_floor():
    noise = torch.tensor([[[0.], [0.]], [[0.], [.1]], [[1.], [1.]]])
    rms = torch.tensor([[[1.], [1.]], [[1.], [2.]], [[1.], [1.]]])
    base = torch.tensor(.55)

    radius = final_radius(noise, rms, base)

    assert radius.shape == (3, 1, 1)
    torch.testing.assert_close(radius[:, 0, 0], torch.tensor([.55, (.55**2 + .32)**.5, .95]))


def test_final_radius_remains_full_only_with_matching_half_readouts():
    spec = trial('final-radius')
    assert spec.arms == ('radius_control', 'radius_adaptive')
    assert set(cases('final-radius')) == {f'{FULL}-{i}' for i in range(3)}
    assert evaluation_modes(FULL, True, third_prior='final-radius')[-2:] == spec.source_modes[2:4]
