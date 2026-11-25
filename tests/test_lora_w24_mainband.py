import torch

from experiments.pamf_lora_main.w24_mainband import private_mainband_risk


def test_private_risk_separates_mainband_and_common_mode():
    n = torch.arange(1024)
    tone = torch.exp((2j * torch.pi / 1024) * 22 * n)
    reference = tone[None, None].expand(1, 8, -1).clone()
    offsets = torch.linspace(-.2, .2, 8)[None, :, None]
    inner = offsets * tone
    outer = offsets * torch.exp((2j * torch.pi / 1024) * 320 * n)
    assert private_mainband_risk(reference, reference) == 0
    assert private_mainband_risk(reference + inner, reference) > .01
    assert private_mainband_risk(reference + outer, reference) < 1e-5
    assert private_mainband_risk(reference + .1 * tone, reference) < 1e-5
