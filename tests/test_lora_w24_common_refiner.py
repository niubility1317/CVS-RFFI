import torch

from experiments.pamf_lora_main.w24_common_refiner import CommonRefiner, common_band_risk
from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_third_prior import UNET_BASELINE, fit_inputs, trial


def test_refiner_starts_at_exact_backbone_and_updates_only_common_receive_iq():
    torch.manual_seed(51)
    model = CommonRefiner().eval()
    y = torch.randn(2, 8, 384, dtype=torch.complex64)
    edges = torch.randn(2, 8, 48, dtype=torch.complex64)
    quality = torch.rand(2, 8, 1)
    rms = .2 + torch.rand(2, 8, 1)
    initial = model.base(y, quality, rms, edges)
    torch.testing.assert_close(model(y, quality, rms, edges)[-1], initial[-1], rtol=0, atol=0)

    with torch.no_grad():
        model.head[-1].bias.copy_(torch.tensor([.9, -.7]))
    _, corrected = model(y, quality, rms, edges)
    physical_edit = .5 * (corrected - initial[-1]) * rms
    common = physical_edit.mean(-2, keepdim=True)
    torch.testing.assert_close(physical_edit, common.expand_as(physical_edit), rtol=1e-5, atol=4e-7)
    motion = (physical_edit.abs().square().mean((-2, -1)) / rms.square().mean((-2, -1))).sqrt()
    assert (motion <= .22).all() and (motion > .01).all()
    corrected.abs().square().mean().backward()
    assert model.head[-1].weight.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())


def test_common_band_risk_ignores_private_contrasts_and_resolves_each_band():
    target = torch.zeros(3, 8, 384, dtype=torch.complex64)
    rms = torch.arange(1, 9, dtype=torch.float32)[None, :, None].expand(3, -1, -1)
    t = torch.arange(384)
    near = torch.exp(2j * torch.pi * 8 * t / 384)
    far = torch.exp(2j * torch.pi * 47 * t / 384)
    alternating = (-1.) ** torch.arange(8)[:, None]
    detail = alternating * near / rms
    observed = (near + far) / rms
    low_error = common_band_risk(near / rms + detail, target, observed, rms)
    high_error = common_band_risk(far / rms + detail, target, observed, rms)
    assert low_error[0] > .1 and low_error[1] < 1e-8
    assert high_error[1] > .1 and high_error[0] < 1e-8
    torch.testing.assert_close(common_band_risk(detail, target, observed, rms), torch.zeros(2), atol=1e-8, rtol=0)


def test_refiner_fit_is_source_only_and_keeps_frozen_parent_control():
    kind = 'common-refine'
    spec = trial(kind)
    assert spec.steps == 12000 and spec.routes == (FULL,)
    assert UNET_BASELINE in spec.source_modes and set(cases(kind)) == {f'{FULL}-{i}' for i in range(3)}
    paths = dict(source_root='U', unet_parent='verified source model', run_root='runs/new',
                 classifier_root='forbidden F', native_target_root='forbidden target')
    for arm in spec.arms:
        assert fit_inputs(paths, arm) == dict(source_root='U', unet_parent='verified source model',
                                             run_root=f'runs/new/{arm}')
