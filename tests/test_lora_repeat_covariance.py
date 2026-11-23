import torch

from experiments.pamf_lora_main.repeat_covariance import RepeatCovariance, repeated_noise


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
