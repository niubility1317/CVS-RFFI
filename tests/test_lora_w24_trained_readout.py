from types import SimpleNamespace

import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes
from experiments.pamf_lora_main.w24_readout_strength import readout_gain
from experiments.pamf_lora_main.w24_third_prior import restorer, trial


def test_adaptive_readout_uses_native_noise_and_preserves_earlier_stages():
    class Base:
        def propose(self, y, quality, rms, edges):
            return (y + 1, y + 2, y + edges), quality, torch.ones_like(rms), rms.flip(0), quality

    training = SimpleNamespace(DuraW24=Base)
    observed = torch.full((2, 8, 8), 1 + 1j)
    rms = torch.stack((torch.ones(8, 1), torch.zeros(8, 1))).add(1e-4)
    edit = torch.full_like(observed, 2j, requires_grad=True)
    model = restorer(training, 'readout_adaptive')
    (coarse, first, final), _, _, native_noise, _ = model.propose(observed, None, rms, edit)
    gain = readout_gain(native_noise, rms)
    torch.testing.assert_close(coarse, observed + 1)
    torch.testing.assert_close(first, observed + 2)
    torch.testing.assert_close(.5 * (final - observed), gain * edit)
    assert .5 < gain[0].item() < .51
    assert 1.4 < gain[1].item() < 1.5
    final.imag.sum().backward()
    torch.testing.assert_close(edit.grad.imag, (2 * gain).expand_as(edit.real))
    fixed = restorer(training, 'readout_control').propose(observed, None, rms, edit)[0][-1]
    torch.testing.assert_close(.5 * (fixed - observed), .5 * edit)


def test_trained_readout_formal_matrix_uses_only_full_and_the_trained_endpoint():
    spec = trial('readout')
    assert set(cases('readout')) == {f'{FULL}-{i}' for i in range(3)}
    assert spec.source_modes[2:4] == ('readout_control_half', 'readout_adaptive_half')
    assert evaluation_modes(FULL, True, third_prior='readout')[-2:] == spec.source_modes[2:4]
