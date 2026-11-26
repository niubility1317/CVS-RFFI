from types import SimpleNamespace

import torch
from torch import nn

from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_spectral_experts import SpectralExperts
from experiments.pamf_lora_main.w24_third_prior import UNET_BASELINE, fit_inputs, promotion, trial


class Expert(nn.Module):
    def __init__(self, strength):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(strength))

    def propose(self, y, *_):
        return (y + self.strength * y.roll(1, dims=-1),),


def test_expert_combination_preserves_dc_and_uses_actual_half_readout():
    torch.manual_seed(32)
    y = torch.randn(2, 8, 384, dtype=torch.complex64)
    model = SpectralExperts(Expert(.2), local=True)
    model.unet = Expert(-.2).requires_grad_(False)
    quality = y.real.new_ones((2, 8, 1))
    edges = y.new_ones((2, 8, 48))
    features, _, proposals = model.sources(y, quality, quality, edges)
    assert features.shape == (2, 8, 4, 12)

    logits = torch.full((2, 8, 4, 5), -100.)
    logits[..., 2] = 100.
    selected = model.combine(proposals, logits)
    torch.testing.assert_close(selected[..., 0], torch.zeros_like(selected[..., 0]))
    torch.testing.assert_close(selected[..., 1:], proposals[..., 2, 1:])

    restored, weights, _, _ = model.estimate(y, quality, quality, edges)
    (stages,), = model.propose(y, quality, quality, edges)
    torch.testing.assert_close(y + .5 * (stages - y), restored)
    (restored.abs().square().mean() + weights.square().mean()).backward()
    assert model.router[-1].weight.grad.abs().sum() > 0
    assert model.dura.strength.grad is None and model.unet.strength.grad is None


def test_expert_source_selection_includes_both_expert_baselines_and_full_only():
    plan = trial('spectral-experts')
    assert plan.steps == 6000 and set(cases('spectral-experts')) == {f'{FULL}-{i}' for i in range(3)}
    inputs = dict(source_root='U', run_root='out', source_parent='DURA', unet_parent='U-Net',
                  classifier_root='F', native_target_root='target')
    for arm in plan.arms:
        assert fit_inputs(inputs, arm) == dict(source_root='U', source_parent='DURA',
                                             unet_parent='U-Net', run_root=f'out/{arm}')

    row = {m: dict(six_correct=1440, low_urban_correct=177, by_scene={'clean': 297}) for m in plan.source_modes}
    candidate = row[plan.source_modes[3]]
    candidate['six_correct'] = 1445
    row[UNET_BASELINE]['low_urban_correct'] = 181
    assert not promotion([row], 'spectral-experts')
    candidate['low_urban_correct'] = 181
    assert promotion([row], 'spectral-experts')
