import torch
from torch import nn

from experiments.pamf_lora_main.stage_splice import read_stages


class TwoStage(nn.Module):
    def __init__(self, coarse, gain, offset):
        super().__init__()
        self.coarse = coarse
        self.prior = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.prior.weight.fill_(gain)
        self.prior.register_buffer('offset', torch.tensor(offset))

    def forward(self, observed, quality, rotation):
        coarse = observed + self.coarse
        return coarse, coarse + self.prior(coarse) + self.prior.offset


def test_stage_swap_exchanges_both_refinement_weights_and_buffers_only():
    inputs = torch.ones(3, 6, 1)
    a, b = TwoStage(2, 3, 5), TwoStage(4, 7, 11)
    before = [{k: v.clone() for k, v in m.state_dict().items()} for m in (a, b)]
    outputs = read_stages(inputs, inputs, inputs, (a, b))
    expected = (3., 5., 9., 26., 18., 13.)
    for wave, value in zip(outputs, expected, strict=True):
        torch.testing.assert_close(wave, torch.full_like(wave, value))
    for model, old in zip((a, b), before, strict=True):
        for key, actual in model.state_dict().items():
            assert torch.equal(actual, old[key])
