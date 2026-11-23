import torch

from experiments.pamf_unrolled_restore.denoise import Denoise, complementary_context, noise_loss


def test_companion_statistics_exclude_the_current_chirp():
    torch.manual_seed(47)
    wave = torch.randn(2, 6, 64, dtype=torch.complex64)
    quality = torch.rand(2, 6, 1)
    before = complementary_context(wave, quality)
    wave[:, 2] += 3. - 2j
    after = complementary_context(wave, quality)
    for x, y in zip(before, after, strict=True):
        torch.testing.assert_close(y[:, 2], x[:, 2])
        assert not torch.allclose(y[:, 1], x[:, 1])


def test_denoiser_starts_at_identity_and_both_branches_receive_gradients():
    torch.manual_seed(25)
    model = Denoise()
    sample = torch.randn(2, 3, 64, dtype=torch.complex64)
    noise = torch.randn_like(sample) / 4
    quality = torch.rand(2, 3, 1)
    for packet in (False, True):
        result = model(sample, quality, packet=packet)
        assert torch.count_nonzero(result) == 0
        model.zero_grad()
        primary, _ = noise_loss(result, noise, sample)
        primary.backward()
        assert all(branch.out.weight.grad.norm() > 0 for branch in (model.spectral, model.temporal))
