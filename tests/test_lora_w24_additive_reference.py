from types import SimpleNamespace

import torch

from experiments.pamf_lora_main.w24_additive_reference import paired_action_noise
from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_packet_unet import objective_for
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, trial


def test_same_source_action_and_noise_are_replayed_with_the_probe():
    rng = torch.Generator().manual_seed(32)
    raw = torch.randn(8, 8192, dtype=torch.complex64, generator=rng)
    weak = .1j * torch.randn(8, 8, 384, generator=rng)
    tracker = SimpleNamespace(aperture=lambda x: x.unflatten(-1, (8, 1024))[..., :432].clone())
    observed, reference, probe = paired_action_noise(raw, tracker, rng, weak, with_identity=True)
    assert observed.shape == (8, 3, 8, 432)
    assert probe.shape == reference.shape == (8, 2, 8, 432)
    same_noise = observed[:, 2] - probe[:, 1]
    torch.testing.assert_close(same_noise, observed[:, 0] - reference[:, 1], atol=5e-7, rtol=1e-5)
    assert (observed[:-2, 1] - observed[:-2, 0]).abs().max() > .05
    assert (reference[:-2, 1] - reference[:-2, 0]).abs().max() > .05
    torch.testing.assert_close(reference[-2:, 0], reference[-2:, 1])
    torch.testing.assert_close(observed[-2:, 0], reference[-2:, 1])
    torch.testing.assert_close(observed[-2:, 1], reference[-2:, 1])
    torch.testing.assert_close(observed[-2:, 2], probe[-2:, 1])


def test_objectives_use_identical_inputs_and_different_targets():
    def compare(wave, target):
        return (wave - target).abs().square().mean()

    def split(wave):
        common = wave.mean(1, keepdim=True)
        return torch.cat((common, wave - common), dim=1)

    def coordinates(_tracker, x):
        scale = x.abs().square().mean(-1, keepdim=True).sqrt()
        return x / scale, torch.ones_like(x), torch.ones_like(scale), scale

    archive = SimpleNamespace(coordinates=coordinates, real_iq=lambda x: x,
                              weak_difference=lambda x: .05j * x.roll(1, 0), waveform_risk=compare,
                              split=split, transfer_risk=compare)
    tracker = SimpleNamespace(aperture=lambda x: x.unflatten(-1, (8, 1024))[..., :432].clone())
    seen = []

    def identity_model(y, _quality, _rms, edges):
        seen.append((y.clone(), edges.clone()))
        return y, y

    raw = torch.randn(8, 8192, dtype=torch.complex64, generator=torch.Generator().manual_seed(44))
    outputs = [objective_for(archive, posterior_primary=True, additive_reference=post)(
        identity_model, tracker, raw, torch.Generator().manual_seed(55),
        with_identity=True, recipe=None, step=1) for post in (False, True)]
    for left, right in zip(*seen, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert all(parts.shape == (10,) and torch.isfinite(parts).all() for _, parts in outputs)
    torch.testing.assert_close(outputs[0][1][-2:], outputs[1][1][-2:])
    assert outputs[0][1][2] > outputs[1][1][2]
    assert outputs[0][0] != outputs[1][0]


def test_scratch_additive_reference_trial_has_only_registered_u_fit_paths():
    spec = trial('additive-reference')
    assert spec.steps == 48000 and spec.routes == (FULL,)
    assert set(cases('additive-reference')) == {f'{FULL}-{i}' for i in range(3)}
    runtime = dict(source_root='U', run_root='run', source_parent='evaluation only',
                   classifier_root='forbidden W24', native_target_root='forbidden target')
    for arm in spec.arms:
        assert fit_inputs(runtime, arm) == dict(source_root='U', run_root=f'run/{arm}')
