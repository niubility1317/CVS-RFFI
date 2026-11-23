import torch

from experiments.pamf_lora_main.repeat_covariance import MAX_EDIT, RepeatCovariance, radial_at_same_dose, repeated_noise


def test_repeat_covariance_has_diagonal_match_and_preserves_tracked_carrier():
    rng = torch.Generator().manual_seed(57)
    samples = torch.randn(3, 6, 256, dtype=torch.complex64, generator=rng) + .5
    positive = (samples * .2 + samples[:, :1]).requires_grad_(False)
    full, diag = RepeatCovariance(True), RepeatCovariance(False)
    spectrum = torch.fft.fftshift(torch.fft.fft(positive, norm='ortho'), dim=-1)
    correlated, _ = full.noise_covariance(spectrum)
    diagonal, _ = diag.noise_covariance(spectrum)
    torch.testing.assert_close(correlated.diagonal(dim1=-2, dim2=-1),
                               diagonal.diagonal(dim1=-2, dim2=-1))
    assert (correlated - diagonal).abs().amax() > .1
    assert torch.linalg.eigvalsh(correlated).min() > 0
    quality = torch.full((len(positive), 6, 1), .65)
    for model in (full, diag):
        for result in model(positive, quality):
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result.mean(-1), positive.mean(-1))


def test_matched_training_gets_finite_stage_gradients():
    rng = torch.Generator().manual_seed(25)
    clean = torch.randn(3, 6, 256, dtype=torch.complex64, generator=rng) * .1 + .8
    y = repeated_noise(clean, rng)
    q = torch.rand(3, 6, 1, generator=rng)
    for model in (RepeatCovariance(False), RepeatCovariance(True)):
        waves = model(y, q)
        sum((v - clean).abs().square().mean() for v in waves).backward()
        for head in (model.coarse, model.fine):
            grad = head[-1].weight.grad
            assert torch.isfinite(grad).all() and grad.norm() > 0

def test_protected_edits_stay_in_complex_disks_under_saturated_updates():
    generator = torch.Generator().manual_seed(159)
    samples = torch.randn(2, 6, 256, dtype=torch.complex64, generator=generator)
    quality = torch.full((2, 6, 1), .7)
    model = RepeatCovariance(True, protected=True)
    with torch.no_grad():
        for head in (model.coarse, model.fine):
            head[-1].bias.fill_(80.)
        values = model(samples, quality)
    spectrum = torch.fft.fft(samples, norm='ortho')
    for output in values:
        restored = torch.fft.fft(output, norm='ortho')
        allowance = MAX_EDIT * spectrum.abs()
        assert ((restored - spectrum).abs() <= allowance + 2e-6).all()
        retained = restored[..., 1:].abs().square().sum()
        received = spectrum[..., 1:].abs().square().sum()
        assert (1 - MAX_EDIT) ** 2 * received <= retained
        torch.testing.assert_close(restored[..., 0], spectrum[..., 0])

def test_radial_control_matches_protected_edit_magnitude():
    rng = torch.Generator().manual_seed(251)
    y = torch.randn(3, 6, 256, dtype=torch.complex64, generator=rng)
    q = torch.ones(3, 6, 1)
    restored = RepeatCovariance(True, protected=True)(y, q)[1]
    control = radial_at_same_dose(y, restored)
    transform = torch.fft.fft
    torch.testing.assert_close(transform(restored - y, norm='ortho').abs(),
                               transform(control - y, norm='ortho').abs(),
                               atol=1e-6, rtol=1e-4)
