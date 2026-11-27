from pathlib import Path

import pytest
import torch

from experiments.pamf_lora_main.w24_regional_dose import (
    apply_gain, fit_gain, gain_features, optimal_gain, predict_gain,
)
from experiments.pamf_lora_main import w24_regional_dose_source as experiment


def test_optimal_regional_gain_is_real_and_nonnegative():
    edit = torch.ones(3, 8, 432, dtype=torch.complex64)
    wanted = torch.cat((1.3 * edit[..., :384], .8 * edit[..., 384:]), dim=-1)
    wanted[1] *= -1
    wanted[2] *= 1j
    gain = optimal_gain(edit, wanted)
    torch.testing.assert_close(gain, torch.tensor([[1.3, .8], [0., 0.], [0., 0.]]))
    torch.testing.assert_close(apply_gain(edit[:1], gain[:1]), wanted[:1])


def test_features_are_invariant_to_common_packet_scale_and_phase():
    rng = torch.Generator().manual_seed(2026112991)
    raw = torch.randn(3, 8192, dtype=torch.complex64, generator=rng)
    assembled = raw.reshape(3, 8, 1024)[..., :432]
    edit = .15 * assembled + .015j * assembled.roll(1, dims=-1)
    fixed = gain_features(raw, assembled, edit)
    transform = torch.tensor([.5j, -1.2, -2j])[:, None]
    changed = gain_features(raw * transform, assembled * transform[:, :, None], edit * transform[:, :, None])
    assert fixed.shape[:2] == (3, 2)
    torch.testing.assert_close(changed, fixed, atol=2e-6, rtol=2e-6)


def test_adaptive_gain_learns_that_varies_with_observation_and_bounds_output():
    level = torch.linspace(-1, 1, 80)
    features = level[:, None, None].expand(-1, 2, -1).contiguous()
    target = torch.stack((1 + .4 * level, .8 - .3 * level), dim=-1)
    calibration = fit_gain(features, target, torch.ones_like(target))
    predicted = predict_gain(calibration, features)
    fixed = predict_gain(calibration, features, adaptive=False)
    assert (predicted-target).square().mean() < .01 * (fixed-target).square().mean()
    stress = predict_gain(calibration, features * 1e3)
    assert (stress >= 0).all() and (stress <= 1.75).all()


def test_zero_wave_and_edits_yield_finite_noop():
    raw = torch.zeros(2, 8192, dtype=torch.complex64)
    observed = torch.zeros(2, 8, 432, dtype=torch.complex64)
    features = gain_features(raw, observed, observed)
    target = optimal_gain(observed, observed)
    calibration = fit_gain(features, target, torch.zeros_like(target))
    prediction = predict_gain(calibration, features)
    assert torch.isfinite(features).all() and torch.isfinite(prediction).all()
    torch.testing.assert_close(apply_gain(observed, prediction), observed)


@pytest.mark.parametrize('extra', ['target_root', 'classifier_root', 'source_validation'])
def test_calibration_rejects_non_u_phase_handles(extra):
    project = Path('some-project')
    config = dict(project=str(project), run_root=str(project/'runs'/experiment.RUN),
                  practical_cache=str(project/'runs'/experiment.PRACTICAL_CACHE_RUN/'source_practical'),
                  source_root=str(project/'releases'/f'{experiment.DATA_RUN}-release'/'data'/'source'),
                  training_release_root=str(project/'releases'/f'{experiment.TRAIN_RUN}-release'))
    assert experiment.fit_config(config) == project
    with pytest.raises(ValueError, match='only the registered source'):
        experiment.fit_config({**config, extra: 'forbidden'})
