import torch

from experiments.pamf_lora_main.w24_clean_projection import CleanProjection
from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_third_prior import UNET_BASELINE, fit_inputs, trial


def test_projection_is_differentiable_and_preserves_the_receive_carrier():
    torch.manual_seed(60)
    model = CleanProjection()
    y = torch.randn(1, 8, 384, dtype=torch.complex64)
    rms = .2 + torch.rand(1, 8, 1)
    clean = model.project_clean(y, rms)
    first, second = model(y, None, rms, None)
    rescaled = model(y, None, rms * 2, None)
    for decoded in (clean, first, second):
        assert decoded.shape == y.shape and torch.isfinite(decoded).all()
        torch.testing.assert_close(decoded.mean(-1), y.mean(-1), rtol=0, atol=3e-7)
    torch.testing.assert_close(first, rescaled[0])
    torch.testing.assert_close(second, rescaled[1])
    assert (first - second).abs().max() > 1e-7
    actual = torch.lerp(y, second, .5)
    ((actual - y).abs().square().mean() + (clean - y).abs().square().mean()).backward()
    for module in (model.encoder, model.context, model.decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_second_iteration_does_not_apply_observation_penalty_twice(monkeypatch):
    model = CleanProjection()
    monkeypatch.setattr(model, 'project', torch.zeros_like)
    y = torch.randn(1, 8, 384, dtype=torch.complex64)
    first, second = model(y, None, torch.ones(1, 8, 1), None)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_long_projection_fit_uses_only_u_and_its_own_output():
    spec = trial('clean-projection')
    assert spec.steps == 48000 and spec.routes == (FULL,)
    assert UNET_BASELINE in spec.source_modes
    assert set(cases('clean-projection')) == {f'{FULL}-{i}' for i in range(3)}
    runtime = dict(source_root='U', run_root='new', unet_parent='frozen comparator only',
                   classifier_root='unavailable', native_target_root='unavailable')
    for arm in spec.arms:
        assert fit_inputs(runtime, arm) == dict(source_root='U', run_root=f'new/{arm}')
