import pytest
import torch
from torch import nn

from experiments.pamf_lora_main.w24_band_adapter import BandAdapter, load_parent
from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes
from experiments.pamf_lora_main.w24_third_prior import OUTER_BASELINE, fit_inputs, promotion, trial


@pytest.mark.parametrize('frequency', (False, True))
def test_band_adapter_starts_at_zero_and_has_gradients_only_through_allowed_outputs(frequency):
    torch.manual_seed(121)
    adapter = BandAdapter(frequency=frequency)
    fields = torch.randn((2, 2, 384), dtype=torch.complex64)
    noise = torch.ones_like(fields[..., :1].real) * .1
    estimate = adapter(fields, fields, fields, fields, noise)
    assert not estimate.any()
    truth = torch.fft.ifft(torch.fft.fft(fields, norm='ortho') * adapter.band, norm='ortho')
    (estimate - truth).abs().square().mean().backward()
    assert adapter.filter[-1].weight.grad.norm() > 0

    with torch.no_grad():
        adapter.filter[-1].weight.normal_(std=.04)
        spectrum = torch.fft.fft(adapter(fields, fields, fields, fields, noise), norm='ortho')
    assert spectrum[..., adapter.band].abs().max() > 1e-3
    torch.testing.assert_close(spectrum[..., ~adapter.band], torch.zeros_like(spectrum[..., ~adapter.band]),
                               atol=1e-6, rtol=0)


def test_parent_load_requires_exact_base_keys_and_preserves_adapter():
    model = nn.Module()
    model.base = nn.Linear(2, 2)
    model.band_adapter = BandAdapter(frequency=False)
    weights = {f'base.{key}': value + 1 for key, value in model.base.state_dict().items()}
    initial = {key: value.clone() for key, value in model.band_adapter.state_dict().items()}
    load_parent(model, weights)
    torch.testing.assert_close(model.base.weight, weights['base.weight'])
    for key, value in model.band_adapter.state_dict().items():
        torch.testing.assert_close(value, initial[key])
    with pytest.raises(ValueError, match='source parent'):
        load_parent(model, {'base.weight': model.base.weight})


def test_adapter_only_uses_three_full_shards_and_must_beat_the_fixed_source_parent():
    spec = trial('band-adapter')
    assert spec.steps == 1800
    assert set(cases('band-adapter')) == {f'{FULL}-{i}' for i in range(3)}
    assert evaluation_modes(FULL, True, third_prior='band-adapter')[-2:] == spec.source_modes[2:4]
    outputs = fit_inputs(dict(source_root='/source', source_parent='/parent', run_root='/run', target_root='/target'),
                         spec.arms[1])
    assert set(outputs) == {'source_root', 'source_parent', 'run_root'}

    def score(correct):
        return dict(six_correct=correct, low_urban_correct=162, by_scene={'clean': 299})

    direct, mature, control, candidate, *_ = spec.source_modes
    row = {name: score(1400) for name in (direct, mature, control)}
    row.update({OUTER_BASELINE: score(1440), candidate: score(1430)})
    assert not promotion([row], 'band-adapter')
    row[candidate] = score(1441)
    assert promotion([row], 'band-adapter')
