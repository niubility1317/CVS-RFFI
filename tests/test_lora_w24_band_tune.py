import torch

from experiments.pamf_lora_main.w24_band_tune import spectral_direction_risk
from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, promotion, trial


def test_signed_complex_direction_response_and_outside_gradient():
    spectrum = torch.zeros((1, 8, 384), dtype=torch.complex64)
    spectrum[..., 9] = 1
    spectrum[..., -32] = 2j
    spectrum[..., 65] = 3
    target = torch.fft.ifft(spectrum, norm='ortho')
    edit = (target * 1j).requires_grad_(True)
    same = spectral_direction_risk(target, target)
    assert same < .01
    bad = spectral_direction_risk(edit, target)
    assert torch.isclose(bad, torch.ones_like(bad), atol=1e-6)
    assert spectral_direction_risk(-target, target) > 1.99
    assert spectral_direction_risk(edit, torch.zeros_like(target)) == 0
    gradient = torch.autograd.grad(bad, edit)[0]
    assert gradient.norm() > 0
    assert torch.fft.fft(gradient, norm='ortho')[..., 65].abs().max() < 1e-6


def test_warmstart_and_full_selection_use_the_frozen_source_parent():
    spec = trial('band-tune')
    assert spec.steps == 1800 and spec.source_modes[-1] == 'outer_parent_half'
    assert set(cases('band-tune')) == {f'{FULL}-{i}' for i in range(3)}
    rt = dict(source_root='/U', run_root='/run', source_parent='/G', native_target_root='/target', classifier_root='/F')
    for arm in spec.arms:
        assert fit_inputs(rt, arm) == dict(source_root='/U', run_root=f'/run/{arm}', source_parent='/G')

    def result(six, low=160):
        return dict(six_correct=six, low_urban_correct=low, by_scene=dict(clean=297))

    rows = {mode: result(1400) for mode in spec.source_modes}
    rows[spec.source_modes[3]] = result(1450)
    rows[spec.source_modes[-1]] = result(1451)
    assert not promotion([rows], 'band-tune')
    rows[spec.source_modes[-1]] = result(1449, 161)
    assert not promotion([rows], 'band-tune')
    rows[spec.source_modes[-1]] = result(1449)
    assert promotion([rows], 'band-tune')
