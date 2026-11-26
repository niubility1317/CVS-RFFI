import pytest
import torch

from experiments.pamf_lora_main.w24_band_precision import BandPrecision, precision_adjustment
from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes
from experiments.pamf_lora_main.w24_third_prior import OUTER_BASELINE, fit_inputs, trial


@pytest.mark.parametrize('resolved', (False, True))
def test_precision_starts_at_identity_and_only_changes_the_registered_band(resolved):
    torch.manual_seed(41)
    precision = BandPrecision(resolved=resolved)
    signal = torch.randn(1, 3, 384, dtype=torch.complex64)
    residual = torch.randn_like(signal)
    noise = signal[..., :1].real.new_full((1, 3, 1), .1)
    gains = precision(signal, residual, noise)
    assert torch.equal(gains, torch.ones_like(gains))
    with torch.no_grad():
        precision.net[-1].weight.normal_(std=.1)
    gains = precision(signal, residual, noise)
    band = torch.fft.ifftshift(precision.band)
    assert torch.equal(gains[..., ~band], torch.ones_like(gains[..., ~band]))
    assert gains.min() >= .125
    assert gains.max() <= 8.
    spread = gains[..., band].std(dim=-1)
    if resolved:
        assert torch.all(spread > .01)
    else:
        assert torch.equal(spread, torch.zeros_like(spread))
    gains.sum().backward()
    assert precision.net[-1].weight.grad.norm() > 0


def test_weighted_observations_are_positive_and_self_adjoint_for_complex_inputs():
    torch.manual_seed(242)
    x, y = torch.randn(2, 3, 24, dtype=torch.complex128)
    variance = torch.rand(3, 24, dtype=torch.float64) + .02
    gain = torch.rand_like(variance) * 2 + .125

    def precision(value):
        return value / variance + precision_adjustment(value, variance.rsqrt(), gain)

    torch.testing.assert_close((x.conj() * precision(y)).sum(),
                               (precision(x).conj() * y).sum())
    assert (x.conj() * precision(x)).sum().real > 0
    assert not precision_adjustment(x, variance.rsqrt(), torch.ones_like(gain)).any()


def test_precision_trial_includes_parent_control_and_full_only():
    spec = trial('band-precision')
    assert spec.source_modes[-1] == OUTER_BASELINE
    assert set(cases('band-precision')) == {f'{FULL}-{i}' for i in range(3)}
    assert evaluation_modes(FULL, True, third_prior='band-precision')[-2:] == spec.source_modes[2:4]
    permitted = fit_inputs(dict(source_root='/source', source_parent='/frozen', run_root='/run',
                                target_root='/target', classifier_root='/f'), spec.arms[1])
    assert set(permitted) == {'source_root', 'source_parent', 'run_root'}
