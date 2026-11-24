import torch

from experiments.pamf_lora_main.polar_shape_source import polar_parts


def test_polar_readouts_swap_only_the_requested_component():
    received = torch.tensor([2+2j, -3+.5j])
    proposal = torch.tensor([-4+1j, .25-2j])
    phase, amplitude = polar_parts(received, proposal)
    torch.testing.assert_close(phase.abs(), received.abs())
    torch.testing.assert_close(amplitude.abs(), proposal.abs())
    torch.testing.assert_close(phase / phase.abs(), proposal / proposal.abs())
    torch.testing.assert_close(amplitude / amplitude.abs(), received / received.abs())


def test_polar_readouts_preserve_defined_phase_at_zero():
    received = torch.tensor([0j, -1+1j, 0j])
    proposal = torch.tensor([1-2j, 0j, 0j])
    phase, amplitude = polar_parts(received, proposal)
    torch.testing.assert_close(phase, torch.tensor([0j, -1+1j, 0j]))
    torch.testing.assert_close(amplitude, proposal)
