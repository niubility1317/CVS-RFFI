import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_noise_color import noise_views, shape_noise
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, trial
from experiments.pamf_lora_main.w24_white_corruption import center_power


def test_colored_noise_resolves_bandwidth_and_center_on_raw_clock():
    white = torch.zeros(1, 8192, dtype=torch.complex64)
    white[0, 0] = 1
    colored = shape_noise(white, torch.tensor([[.05]]), torch.tensor([[.07]]))
    power = torch.fft.fft(colored).abs().square()[0]
    f = torch.fft.fftfreq(len(power))
    inside = (f - .07).abs() <= .03
    outside = (f - .07).abs() >= .15
    assert power[inside].mean() > 1000 * power[outside].mean() > 0


def test_matched_views_take_same_generic_draw_and_packet_energy():
    torch.manual_seed(77)
    raw = torch.randn(16, 8192, dtype=torch.complex64)

    def aperture(inputs):
        return inputs.unflatten(-1, (8, 1024))[..., :432]

    def generic(inputs, rng):
        return inputs + .45 * torch.randn(inputs.shape, dtype=inputs.dtype, generator=rng)

    clean = aperture(raw)
    generator_a = torch.Generator().manual_seed(31)
    generator_b = torch.Generator().manual_seed(31)
    a = noise_views(raw, clean, aperture, generic, generator_a, colored=False)
    b = noise_views(raw, clean, aperture, generic, generator_b, colored=True)
    for white, colored in zip(a, b, strict=True):
        torch.testing.assert_close(center_power(white - clean), center_power(colored - clean),
                                   rtol=3e-6, atol=1e-7)
        changed = (white - colored).abs().amax((1, 2)) > 1e-5
        assert changed.any() and (~changed).any()
    assert torch.equal(generator_a.get_state(), generator_b.get_state())


def test_color_trial_fit_carries_only_same_frozen_source_parent():
    spec = trial('noise-color')
    assert spec.steps == 12000 and spec.routes == (FULL,)
    assert set(cases('noise-color')) == {f'{FULL}-{i}' for i in range(3)}
    runtime = dict(source_root='source', run_root='runs/new', unet_parent='frozen U',
                   classifier_root='forbidden F', native_target_root='forbidden target')
    for arm in spec.arms:
        assert fit_inputs(runtime, arm) == dict(source_root='source', run_root=f'runs/new/{arm}',
                                                unet_parent='frozen U')
