import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_spectral_posterior import SpectralPosterior, weighted_posterior
from experiments.pamf_lora_main.w24_third_prior import UNET_BASELINE, fit_inputs, trial


def test_excluded_bin_cannot_enter_its_spectral_prior_with_fixed_receive_statistics():
    torch.manual_seed(52)
    blind = SpectralPosterior(exclude_current=True).eval()
    seen = SpectralPosterior(exclude_current=False).eval()
    seen.load_state_dict(blind.state_dict())
    x = torch.randn(2, 8, 384, dtype=torch.complex64)
    changed = x.clone()
    changed[..., 17] += 3. + 4.j
    quality = torch.rand(2, 8, 1)
    edge = torch.rand(2, 8, 1)
    noise = torch.rand(2, 1, 1)
    with torch.no_grad():
        blind_before, blind_after = (blind.prior_spectrum(z, quality, edge, noise) for z in (x, changed))
        seen_before, seen_after = (seen.prior_spectrum(z, quality, edge, noise) for z in (x, changed))
    for a, b in zip(blind_before, blind_after, strict=True):
        torch.testing.assert_close(a[..., 17], b[..., 17], rtol=0, atol=0)
    assert (seen_after[0][..., 17] - seen_before[0][..., 17]).abs().max() > 1e-7
    assert (blind_after[1] - blind_before[1]).abs().max() > 1e-7


def test_primary_readout_is_a_finite_posterior_with_physical_carrier_unchanged():
    torch.manual_seed(53)
    model = SpectralPosterior(exclude_current=True)
    x = torch.randn(2, 8, 384, dtype=torch.complex64)
    edges = torch.randn(2, 8, 48, dtype=torch.complex64)
    quality = torch.rand(2, 8, 1)
    rms = .2 + torch.rand(2, 8, 1)
    stages, = model.propose(x, quality, rms, edges)
    primary = torch.lerp(x, stages[-1], .5)
    torch.testing.assert_close(primary.mean(-1), x.mean(-1), rtol=0, atol=3e-7)
    assert torch.isfinite(primary).all() and (primary - x).abs().sum() > 0
    (primary - x).abs().square().mean().backward()
    for tower in (model.lower, model.upper, model.prior):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tower.parameters())

    observed = torch.tensor([1. + 2.j])
    predicted = torch.tensor([0.])
    torch.testing.assert_close(weighted_posterior(observed, predicted, .5, .0), observed)
    torch.testing.assert_close(weighted_posterior(observed, predicted, .5, .5), .5 * observed)


def test_scratch_trial_does_not_open_classifier_target_or_parent_during_fit():
    kind = 'spectral-posterior'
    spec = trial(kind)
    assert spec.steps == 24000 and spec.routes == (FULL,)
    assert UNET_BASELINE in spec.source_modes and set(cases(kind)) == {f'{FULL}-{i}' for i in range(3)}
    runtime = dict(source_root='U', run_root='runs/new', unet_parent='frozen comparator only',
                   classifier_root='forbidden F', native_target_root='forbidden target')
    for arm in spec.arms:
        assert fit_inputs(runtime, arm) == dict(source_root='U', run_root=f'runs/new/{arm}')
