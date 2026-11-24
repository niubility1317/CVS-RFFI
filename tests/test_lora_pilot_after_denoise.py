import torch

from experiments.pamf_lora_main import pilot_after_denoise as module
from experiments.pamf_lora_main.crossfit_pilot import apply_transfer, estimate_shared
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.pilot_after_denoise_source import RECIPES, smoke_gradients


def inputs():
    rng = torch.Generator().manual_seed(214)
    angle = torch.randn((2, 6, 256), generator=rng)
    template = torch.polar(torch.ones_like(angle), angle)
    taps = torch.zeros(2, 8, dtype=template.dtype)
    taps[:, 4] = .35 + .08j
    observed = apply_transfer(template, taps) * template.conj()
    quality = torch.ones_like(observed.real[..., :1])
    return observed, quality, template, rng


def test_explicit_raw_fit_matches_existing_estimate():
    observed, q, template, _ = inputs()
    variance = q * .02
    old = estimate_shared(observed, template, q, variance)
    same = estimate_shared(observed, template, q, variance, fit_observed=observed)
    assert old[1].min() > 0
    for expected, value in zip(old, same, strict=True):
        assert torch.equal(expected, value)


def test_synthetic_evidence_in_model_fit_cannot_replace_raw_holdout():
    measured, q, template, _ = inputs()
    original = torch.ones_like(measured)
    response, support = estimate_shared(original, template, q, q*.02, fit_observed=measured)
    assert torch.equal(support, torch.zeros_like(support))
    assert torch.equal(response, torch.ones_like(response))


def test_each_coarse_fit_is_independent_of_other_group():
    measured, q, _, _ = inputs()
    model = module.PilotAfterDenoise((4, 8, 8, 8), denoised_fit=True).eval()
    altered = measured.clone()
    altered[:, 1::2] *= .3j
    a = model.pilot_groups(measured, q)
    b = model.pilot_groups(altered, q)
    for original, perturbed in zip(a, b, strict=True):
        torch.testing.assert_close(original[:, ::2], perturbed[:, ::2], rtol=0, atol=0)
    assert not torch.equal(a[0][:, 1::2], b[0][:, 1::2])


def test_two_fit_arms_share_parameter_layout_and_receive_gradients():
    measured, q, template, _ = inputs()
    reference = torch.ones_like(measured)
    arms = [module.PilotAfterDenoise((4, 8, 8, 8), denoised_fit=denoised)
            for denoised in (False, True)]
    arms[1].load_state_dict(arms[0].state_dict(), strict=True)
    for model in arms:
        coarse, fine, _ = model(measured, q, template, diagnostics=True)
        loss = (coarse-reference).abs().square().mean() + (fine-reference).abs().square().mean()
        loss.backward()
        for stage in (model.coarse, model.prior):
            assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in stage.parameters())


def test_smoke_exercises_bounded_refinement_after_first_update():
    tracker = PacketRestorer('consensus').eval()
    rng = torch.Generator().manual_seed(623)
    real = torch.randn((6, 6, 256), generator=rng)
    imag = torch.randn((6, 6, 256), generator=rng)
    clean = real_iq(tracker.reference.chirp * (1+.12*torch.complex(real, imag)))
    model = module.PilotAfterDenoise((4, 8, 8, 8), denoised_fit=True)
    norms = smoke_gradients(model, tracker, clean, rng, RECIPES['denoised_pilot'])
    assert min(norms) > 0
