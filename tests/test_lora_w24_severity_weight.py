import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_severity_weight import disturbance, redistribution, waveform_per_chirp
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, trial


def test_weights_redistribute_without_changing_constant_risk():
    severity = torch.tensor([[.001, .01], [.1, 1.]])
    risk = torch.ones_like(severity).requires_grad_()
    difference, upper = redistribution(risk, severity)
    torch.testing.assert_close(difference, torch.tensor(0.))
    torch.testing.assert_close(upper, torch.tensor(1.))
    (risk.mean() + difference).backward()
    assert risk.grad[0, 0] < risk.grad[0, 1] < risk.grad[1, 0] < risk.grad[1, 1]
    torch.testing.assert_close(risk.grad.sum(), torch.tensor(1.))


def test_each_source_observation_uses_the_same_clean_reference():
    source = torch.ones(3, 8, 384, dtype=torch.complex64)
    errors = torch.tensor([[0., .1], [.2, .4], [.8, 1.]]).view(3, 2, 1, 1)
    t = torch.arange(384) / 384
    tone = torch.polar(torch.ones_like(t), 2 * torch.pi * t)
    received = source[:, None] + errors * tone
    torch.testing.assert_close(disturbance(received, source), errors.square()[..., 0, 0])
    risk = waveform_per_chirp(received, source[:, None])
    assert risk.shape == (3, 2, 8)
    assert (risk[0, 1] < risk[2, 1]).all()
    assert torch.isfinite(risk).all()


def test_severity_trial_uses_the_full_only_existing_outer_parent():
    plan = trial('severity-weight')
    assert plan.steps == 1800
    assert plan.source_modes[-1] == 'outer_parent_half'
    assert set(cases('severity-weight')) == {f'{FULL}-{i}' for i in range(3)}
    rt = dict(source_root='/U', source_parent='/G', run_root='/out', classifier_root='/F', native_target_root='/T')
    for arm in plan.arms:
        assert fit_inputs(rt, arm) == dict(source_root='/U', source_parent='/G', run_root=f'/out/{arm}')
