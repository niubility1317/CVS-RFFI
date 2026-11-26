import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes
from experiments.pamf_lora_main.w24_packet_unet import PacketUNet, correction_risk
from experiments.pamf_lora_main.w24_third_prior import OUTER_BASELINE, fit_inputs, trial


def test_matched_packet_attention_can_transport_context_without_replacing_carrier():
    torch.manual_seed(29)
    independent = PacketUNet(across_chirps=False).eval()
    joint = PacketUNet(across_chirps=True).eval()
    joint.load_state_dict(independent.state_dict(), strict=True)
    y = torch.randn(1, 8, 384, dtype=torch.complex64)
    edges = torch.randn(1, 8, 48, dtype=torch.complex64)
    quality = torch.ones(1, 8, 1)
    rms = torch.ones_like(quality)
    changed = y.clone()
    changed[:, 7] *= 2.
    with torch.no_grad():
        for model, depends_on_neighbor in ((independent, False), (joint, True)):
            (time, restored), = model.propose(y, quality, rms, edges)
            _, perturbed = model(changed, quality, rms, edges)
            _, rescaled = model(y, quality, 2. * rms, 2. * edges)
            torch.testing.assert_close(rescaled, restored)
            assert time.shape == restored.shape == y.shape and torch.isfinite(restored).all()
            torch.testing.assert_close(restored.mean(-1), y.mean(-1), rtol=0, atol=2e-7)
            if depends_on_neighbor:
                assert (perturbed[:, 0] - restored[:, 0]).abs().max() > 1e-6
            else:
                torch.testing.assert_close(perturbed[:, 0], restored[:, 0], rtol=0, atol=0)


def test_unet_fit_excludes_every_parent_and_uses_full_only():
    spec = trial('packet-unet')
    assert spec.steps == 3600 and spec.routes == ('full_native',)
    assert spec.source_modes[-1] == OUTER_BASELINE
    assert set(cases('packet-unet')) == {f'{FULL}-{i}' for i in range(3)}
    assert evaluation_modes(FULL, True, third_prior='packet-unet')[-2:] == spec.source_modes[2:4]
    for arm in spec.arms:
        paths = dict(source_root='U', run_root='run', source_parent='G', classifier_root='F', native_target_root='T')
        assert fit_inputs(paths, arm) == dict(source_root='U', run_root=f'run/{arm}')


def test_complex_correction_penalizes_reversed_direction_at_equal_energy():
    torch.manual_seed(32)
    desired = torch.randn(2, 2, 8, 384, dtype=torch.complex64)
    same, same_gain = correction_risk(desired, desired)
    reversed_direction, reversed_gain = correction_risk(-desired, desired)
    estimated = torch.zeros_like(desired, requires_grad=True)
    zero, zero_gain = correction_risk(estimated, desired)
    (zero + zero_gain).backward()
    assert same < zero < reversed_direction and zero_gain > 0
    torch.testing.assert_close(same_gain, reversed_gain, rtol=0, atol=0)
    assert torch.isfinite(estimated.grad).all() and estimated.grad.abs().sum() > 0


def test_packet_fidelity_is_a_matched_long_trial_with_full_target_plan():
    plan = trial('packet-unet-fidelity')
    assert plan.steps == 24000
    assert evaluation_modes(FULL, True, third_prior='packet-unet-fidelity')[-2:] == plan.source_modes[2:4]
    assert set(cases('packet-unet-fidelity')) == {f'{FULL}-{i}' for i in range(3)}
    for arm in plan.arms:
        runtime = dict(source_root='U', run_root='run', source_parent='baseline only', classifier_root='F')
        assert set(fit_inputs(runtime, arm)) == {'source_root', 'run_root'}
