import pytest
import torch

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.preamble_noise import WhitenedAction, preamble_weights, repeat_dispersion
from experiments.pamf_lora_main.repeated_edit import PREAMBLE
from experiments.pamf_lora_main.shared_gain_dura import SharedGainDura


@pytest.mark.parametrize('length', (256, 8192))
def test_observed_dispersion_downweights_variable_frequency(length):
    start, count, period = PREAMBLE[length]
    time = torch.arange(period)
    signal = torch.exp(2j*torch.pi*time/period)
    disturbance_bin = 4
    contamination = .4 * torch.exp(2j*torch.pi*disturbance_bin*time/period)
    observed = torch.zeros(1,length,dtype=torch.complex64)
    clean = observed.clone()
    section = slice(start,start + count * period)
    clean[:,section] = signal.repeat(count)
    sign = torch.where(torch.arange(count) % 2 == 0, 1., -1.)
    observed[:,section] = (signal + sign[:,None] * contamination).flatten()
    floor = repeat_dispersion(clean)[0]
    torch.testing.assert_close(preamble_weights(clean,floor), torch.ones_like(clean.real))
    weights = preamble_weights(observed,floor)
    assert weights.isfinite().all()
    target_bin = round(disturbance_bin * length / period)
    assert weights[0,target_bin] < weights.median() * .80
    torch.testing.assert_close(weights.square().mean(-1),torch.ones(1))


def test_whitened_action_obeys_real_linear_adjoint_and_consistent_identity():
    torch.manual_seed(3)
    left = torch.randn(2,256,dtype=torch.complex64)
    right = torch.randn_like(left)
    operator = Action(torch.randn(2,12,dtype=torch.complex64), torch.randn_like(left), .2*torch.randn_like(left))
    weights = .5 + torch.rand(2,256)
    adjusted = WhitenedAction(operator,weights)
    forward = (adjusted(left).conj()*right).sum(-1).real
    adjoint = (left.conj()*adjusted.adjoint(right)).sum(-1).real
    torch.testing.assert_close(forward,adjoint)
    stage = SharedGainDura(retain_received=True,clean_noise_floor=torch.zeros(20))
    state = stage.restoration(left)[0][-1]
    torch.testing.assert_close(state,left,atol=3e-6,rtol=3e-6)
